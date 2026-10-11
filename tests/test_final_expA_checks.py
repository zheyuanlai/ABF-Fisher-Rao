"""Tests of the Experiment A production driver and pre-production checks (docs/final_campaign/PREREGISTRATION.md
sections 1.3 and 1.4): scripts/final_campaign/{make_cells, run_cells, expA_run_common, expA_gate, expA_check_references,
expA_check_pairing, expA_check_smoke, expA_check_old_bitwise}.py.

C1  check 1: analytic grad_V of the shifted fibre (kappa 1) vs central differences (eps 1e-6) at >= 200 points over
    [-1.8, 1.8] x [m - 6 sigma, m + 6 sigma], every lam in {1, 4, 16, 128}: |g - fd| / (1 + |g|) <= 1e-7; the
    kernel-extracted deposit force (one walker per production bin, one step, M_all[1]) equals grad_V's dV/dx to
    <= 1e-12 at every lam, bitwise independent of lam; the one-step y update is the EM formula with lam.
C7  check 7: Langevin MT19937 stream = default_noise_seed(seed, N) advanced by exactly 2N normals per step in BOTH
    arms (so the FR arm's Langevin stream equals the ABF arm's); FR uniforms = PCG64(SeedSequence([seed, N, 1]))
    advanced by n_opp L doubles (FR arm) and untouched (ABF arm); noise seed, FR entropy and initial state
    independent of lam; FR(gamma 0) == ABF bitwise, and with real FR the saves before the first event are equal.
C2  check 2 / cells: the four cell configs are up to date, accepted by analyze_ladder's gateway_family config path,
    carry the expA physics / plan / locations; the four frozen references pass check 2 (bitwise primary arrays).
D   driver: seed-major order, smoke plan, frozen-identity refusals, disk pause, projection gate (incl. the validation /
    check-4b compute it must count), CPU allocation (<= 64 workers, no SMT sibling pairs, workers <= physical cores),
    unplanned seeds refused, tiny end-to-end run (pool, ledgers, provenance, idempotency, STALE refusal), spawn
    pickling of the worker, NONFINITE relabelling, stop on the first FAILED run, refusal on a NONFINITE ledger row.
G   production gate (expA_gate): refuses without every record (G-A1/G-A2 selection PASS at the config h; checks 1/7, 2,
    4b, 8 incl. 8c, 9 and 6 PASS and current); each item's failure modes; --override-gate needs a dated amendment.
C6  check-6 tool on tiny synthetic runs: death rule, 0-compared-save replay (t_first > 0 and t_first = 0),
    permutation-only FR event (event rule + replay) = PASS_EVENT_RULE -> AMENDMENT_REQUIRED (exit 4) unless a dated
    amendment licenses it, tampering and missing arms detected; the input signature covers the frozen inputs.
C9  check-9 tool: plan mode on a tiny driver smoke (pipeline included), tampering of D1 / D2 / finiteness / contract
    detected; RESULT_KEYS equal the engine's result contract; a FAIL verdict keeps exit 1 when a write is refused.
C8  check-8 tool on tiny stored runs: signature reconstruction (default call and explicit-grid fallback), EQUAL,
    MISMATCH, resumable job records, refusal of a different final record and of a different engine; PARTIAL (exit 3)
    until the 8c harness-inertness record exists, PASS once it does, FAIL / PARTIAL when it fails / is stale.
C2r check-2 record: input signature, idempotent, never overwritten without --force.
Every file written goes to pytest's tmp_path.  CPU only.
"""
import csv
import importlib.util
import json
import math
import os
import pickle
import shutil
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
FC = os.path.join(ROOT, "scripts", "final_campaign")
for _p in (os.path.join(ROOT, "src"), FC, os.path.join(ROOT, "scripts", "equal_budget"),
           os.path.join(ROOT, "scripts", "mechanism")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.expanduser("~/.cache/numba_final"))

import gateway_family_numba as gf          # noqa: E402
import expA_run_common as C                # noqa: E402
import expA_check_pairing as CP            # noqa: E402
import expA_check_smoke as CS              # noqa: E402
import expA_check_old_bitwise as C8        # noqa: E402
import expA_check_references as CR         # noqa: E402
import expA_gate as G                      # noqa: E402
import expA_record_tests as RT             # noqa: E402

CELL = dict(beta=16.0, H=0.5, omega_out=1.0, omega_in=32.0, s=0.1)
EASY = dict(beta=1.0, H=0.1, omega_out=1.0, omega_in=4.0, s=0.1)     # frequent FR events (tool tests only)
LAMS = (1.0, 4.0, 16.0, 128.0)
REFS = os.path.join(ROOT, "results", "final_campaign", "references")
HAVE_REFS = all(os.path.exists(os.path.join(REFS, f"expA_shift_lam{int(l)}_reference.npz")) for l in LAMS)
PAIR_KEYS = ["M_all", "C_all", "X_final", "Y_final", "max_abs_bias", "n_force_evals", "hist_inst", "region_frac",
             "events_cum", "lineage_visited", "first_arrival_step", "traces", "traces_y", "traces_anc",
             "traces_rebirth", "dep_age_C", "dep_age_M", "dep_age_M2", "dep_cross_C", "fr_death_pos_hist_cum",
             "fr_birth_pos_hist_cum"]


def shift(lam, **kw):
    return dict(CELL, variant="shift", alpha=None, kappa=1.0, lam=float(lam), h=2.5e-5, **kw)


def env():
    e = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", NUMBA_NUM_THREADS="1", MKL_NUM_THREADS="1",
             OPENBLAS_NUM_THREADS="1")
    e.setdefault("NUMBA_CACHE_DIR", os.path.expanduser("~/.cache/numba_final"))
    return e


def run_script(name, args, timeout=900):
    return subprocess.run([sys.executable, os.path.join(FC, name)] + list(args), env=env(), capture_output=True,
                          text=True, timeout=timeout, cwd=ROOT)


def tamper(path, fn, out=None):
    """Rewrite an npz result with fn(dict of arrays) applied (cfg_json / meta_json kept)."""
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k].copy() for k in z.files}
    fn(d)
    np.savez_compressed(out or path, **d)


def load_rcf(name):
    """scripts/final_campaign/run_cells.py as a fresh module (its main() runs in-process)."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(FC, "run_cells.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh)
    return str(path)


def fake_8c(path, verdict="PASS", current=True):
    """A synthetic check-8c (harness inertness) record; current = its input signature equals the suite's now."""
    return write_json(path, dict(check="check8c", verdict=verdict, counts={"passed": 6},
                                 input_signature=RT.signature(RT.SUITES["check8c"]) if current else "0" * 64))


def prereg_with(tmp_path, body_ref=None, amendment_ref=None):
    """A stand-in preregistration: frozen body (optionally holding body_ref), the amendment marker line, and
    optionally one dated amendment holding amendment_ref."""
    txt = "frozen body " + (body_ref or "") + "\n---\n" + C.AMENDMENT_MARKER + ", appended below)*\n"
    if amendment_ref:
        txt += f"\n### Amendment 1 (2026-10-12): {amendment_ref} -- test amendment\n"
    p = tmp_path / "prereg_test.md"
    p.write_text(txt)
    return str(p)


def fake_gate_records(out_root, smoke_T=4.0):
    """Synthetic records satisfying every production-gate item (the gate reads records only; nothing is run)."""
    P = C.load_config()
    root = C.prod_root(out_root)
    chk = os.path.join(root, "checks")
    write_json(os.path.join(root, "validation", "summary.json"),
               dict(smoke=False, selection=dict(verdict="PASS", h_A=2.5e-5)))
    write_json(os.path.join(chk, "check1_7_tests.json"),
               dict(check="check1_7", verdict="PASS", counts={"passed": 10},
                    input_signature=RT.signature(RT.SUITES["check1_7"])))
    p8c = fake_8c(os.path.join(chk, "check8c_harness_inertness.json"))
    write_json(os.path.join(chk, "check2_references.json"), dict(check="2", verdict="PASS",
                                                                 input_signature=CR.input_signature()))
    write_json(os.path.join(chk, "check4b_frozen_x.json"),
               dict(verdict="PASS", status="complete", signature=dict(h_A=2.5e-5),
                    provenance=dict(engine_sha256=C.EXPECTED_ENGINE_SHA256, wall_s=36.0)))
    write_json(os.path.join(chk, "check8_old_bitwise.json"),
               dict(check="8", verdict="PASS", part_8c=dict(sha256=C.sha256_file(p8c)),
                    engine=dict(expected=C.EXPECTED_ENGINE_SHA256)))
    sroot = C.smoke_root(out_root, smoke_T)
    files = [q for v in CP.discover(sroot, [v["name"] for v in P["variants"]]).values() for q in v.values()]
    o9 = CS.c9_options("smoke", smoke_T, True, True, None)
    o6 = CP.c6_options("smoke", smoke_T, True, None)
    write_json(os.path.join(chk, "check9_smoke.json"),
               dict(check="9", verdict="PASS", root=sroot, n_files_expected=32, signature_options=o9,
                    input_signature=CP.record_signature(files, o9)))
    write_json(os.path.join(chk, "check6_pairing_smoke.json"),
               dict(check="6", verdict="PASS", root=sroot, n_expected_pairs=16, signature_options=o6,
                    input_signature=CP.record_signature(files, o6)))
    return chk


# ================================================================================================ C1
@pytest.mark.parametrize("lam", LAMS)
def test_c1_grad_V_matches_central_differences(lam):
    cfg = shift(lam)
    c = gf.full_cfg(cfg)
    sigma = 1.0 / (math.sqrt(c["beta"]) * c["kappa"])
    rng = np.random.default_rng(8650 + int(lam))
    x = np.concatenate([np.linspace(-1.8, 1.8, 121), rng.uniform(-1.8, 1.8, 100), rng.uniform(-0.4, 0.4, 60),
                        [-0.21, 0.21, 0.0, -1.0, 1.0]])
    m = gf.m_shift(cfg, x)
    y = m + sigma * rng.uniform(-6.0, 6.0, x.size)
    y[:5] = m[:5] - 6.0 * sigma
    y[5:10] = m[5:10] + 6.0 * sigma
    assert x.size >= 200 and np.all(np.abs(x) <= 1.8)
    gx, gy = gf.grad_V(cfg, x, y)
    eps = 1e-6
    fdx = (gf.potential(cfg, x + eps, y) - gf.potential(cfg, x - eps, y)) / (2 * eps)
    fdy = (gf.potential(cfg, x, y + eps) - gf.potential(cfg, x, y - eps)) / (2 * eps)
    ex = np.abs(gx - fdx) / (1.0 + np.abs(gx))
    ey = np.abs(gy - fdy) / (1.0 + np.abs(gy))
    assert ex.max() <= 1e-7, (lam, ex.max(), x[ex.argmax()])
    assert ey.max() <= 1e-7, (lam, ey.max())
    # lam enters neither V nor grad V (Gibbs law, F*, the conditional law and Var(f | x) are lam-independent)
    g1x, g1y = gf.grad_V(shift(1.0), x, y)
    assert np.array_equal(gx, g1x) and np.array_equal(gy, g1y)
    # the shifted-fibre closed forms: dV/dy = kappa^2 (y - m), dV/dx = F*' - kappa^2 (y - m) m'
    assert np.allclose(gy, c["kappa"] ** 2 * (y - m), rtol=1e-13, atol=1e-13)
    mp = math.sqrt(2.0 / c["beta"]) / c["kappa"] * gf.omega_prime_np(cfg, x) / gf.omega_np(cfg, x)
    assert np.allclose(gx, gf.dF_star(cfg, x) - c["kappa"] ** 2 * (y - m) * mp, rtol=1e-12, atol=1e-12)


def _one_step(lam):
    cfg = shift(lam)
    c = gf.full_cfg(cfg)
    nb = 180
    delta = (gf.XMAX - gf.XMIN) / nb
    rng = np.random.default_rng(8651)
    x0 = gf.XMIN + (np.arange(nb) + rng.uniform(0.05, 0.95, nb)) * delta
    sigma = 1.0 / math.sqrt(c["beta"])
    y0 = gf.m_shift(cfg, x0) + sigma * rng.uniform(-6.0, 6.0, nb)
    res = gf.run_arm(cfg, "abf", nb, 8650, 1, [0, 1], x0=x0, y0=y0, trace_steps=[0, 1])
    return cfg, c, x0, y0, res


@pytest.mark.parametrize("lam", LAMS)
def test_c1_kernel_deposit_force_equals_grad_V(lam):
    cfg, c, x0, y0, res = _one_step(lam)
    assert np.array_equal(res["C_all"][1], np.ones(180)) and np.all(res["M_all"][0] == 0)
    fk = res["M_all"][1]
    gx, gy = gf.grad_V(c, x0, y0)
    err = np.abs(fk - gx) / (1.0 + np.abs(gx))
    assert err.max() <= 1e-12, (lam, err.max())
    # bitwise independent of lam (lam enters only the y update)
    _, _, _, _, r1 = _one_step(1.0)
    assert np.array_equal(r1["M_all"], res["M_all"]) and np.array_equal(r1["X_final"], res["X_final"])
    # the y update: y - lam dV/dy h + sqrt(2 lam h / beta) zy with the Langevin stream's normals (N zx, then N zy)
    gf._mt_seed(gf.default_noise_seed(8650, 180))
    Z = gf._mt_normals(360)
    h, beta = c["h"], c["beta"]
    y1 = y0 - lam * gy * h + math.sqrt(2.0 * lam * h / beta) * Z[180:]
    assert np.max(np.abs(res["Y_final"] - y1) / (1.0 + np.abs(y1))) < 1e-13
    if lam != 1.0:
        assert not np.array_equal(r1["Y_final"], res["Y_final"])


# ================================================================================================ C7
@pytest.mark.parametrize("lam", LAMS)
def test_c7_langevin_and_fr_streams(lam, tmp_path):
    """At production cell constants and h: after k steps the MT19937 state of BOTH arms is default_noise_seed(seed,
    N) advanced by exactly 2 N k normals (the FR arm draws nothing else from it: its Langevin stream IS the ABF arm's);
    the PCG64 FR stream is SeedSequence([seed, N, 1]) advanced by n_opp L doubles in the FR arm, untouched in ABF."""
    cfg = shift(lam)
    N, n, k, seed = 16, 4000, 1000, 8390
    kn = gf.derived_knobs(gf.full_cfg(cfg), N)
    assert kn["fr_every"] == 160
    states = {}
    for method in ("abf", "fr"):
        ck = str(tmp_path / f"{method}.ck.npz")
        with pytest.raises(gf.RunInterrupted):
            gf.run_arm(cfg, method, N, seed, n, [n], chunk_steps=k, checkpoint_path=ck, checkpoint_every_s=0.0,
                       stop_after_chunks=1)
        with np.load(ck) as z:
            assert int(z["ictr"][gf.I_STEP]) == k
            states[method] = (int(z["mt_index"]), z["mt_key"].copy(), json.loads(str(z["fr_state_json"])))
        gf._mt_seed(gf.default_noise_seed(seed, N))
        gf._mt_normals(2 * N * k)
        idx, key = gf.mt_get()
        assert states[method][0] == idx and np.array_equal(states[method][1], key)
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, N, 1])))
        if method == "fr":
            rng.random(-(-k // kn["fr_every"]) * kn["fr_block"])
        assert states[method][2] == rng.bit_generator.state
    assert states["abf"][0] == states["fr"][0] and np.array_equal(states["abf"][1], states["fr"][1])
    assert states["abf"][2] != states["fr"][2]


def test_c7_streams_and_initial_state_independent_of_lam():
    sigs = [gf.run_signature(shift(l), m, 512, 8300, 6400000, gf.prereg_save_grid(6400000, 2.5e-5))
            for l in LAMS for m in ("abf", "fr")]
    for f in ("noise_seed", "fr_entropy", "x0_sha", "y0_sha", "save_sha", "trace_sha", "snap_sha", "engine_sha"):
        assert len({json.dumps(s[f]) for s in sigs}) == 1, f
    assert sigs[0]["noise_seed"] == gf.default_noise_seed(8300, 512) == (1000003 * 8300 + 7919 * 512) % (2 ** 31 - 1)
    assert sigs[0]["fr_entropy"] == [8300, 512, 1]
    assert sigs[0]["engine_sha"] == C.EXPECTED_ENGINE_SHA256
    x1, y1 = gf.init_family(8300, 512, shift(1.0))
    for l in LAMS[1:]:
        x, y = gf.init_family(8300, 512, shift(l))
        assert np.array_equal(x, x1) and np.array_equal(y, y1)


@pytest.mark.parametrize("lam", LAMS[1:])
def test_c7_fr_arm_paired_with_abf_before_first_event(lam):
    """Production cell constants, FR-active knobs: FR(gamma 0) == ABF bitwise on every paired key; with FR on, every
    save before the first FR event is bitwise equal and later saves differ (the comparison is sensitive)."""
    N, n = 64, 4000
    cfg = shift(lam, gamma=15.0, ramp_t=0.01)
    sv, _ = gf.budget_save_grid(n)
    abf = gf.run_arm(cfg, "abf", N, 8391, n, sv)
    fr0 = gf.run_arm(dict(cfg, gamma=0.0), "fr", N, 8391, n, sv)
    for k in PAIR_KEYS:
        assert np.array_equal(np.asarray(abf[k]), np.asarray(fr0[k]), equal_nan=True), k
    assert fr0["fr_opp_event_cum"][-1] == 0 and fr0["fr_opp_cum"][-1] == -(-n // 160)
    fr = gf.run_arm(cfg, "fr", N, 8391, n, sv)
    pre = fr["fr_opp_event_cum"] == 0
    assert pre.sum() >= 3 and (~pre).sum() >= 3 and fr["fr_deaths_cum"][-1] > 0
    assert np.array_equal(fr["M_all"][pre], abf["M_all"][pre]) and np.array_equal(fr["C_all"][pre], abf["C_all"][pre])
    assert not np.array_equal(fr["M_all"][~pre], abf["M_all"][~pre])


# ================================================================================================ C2 / cells
def test_cells_up_to_date_and_physics():
    import make_cells as MKC
    assert MKC.main(["--check"]) == 0
    X = json.load(open(os.path.join(ROOT, "configs", "final_campaign", "expA.json")))
    mech = json.load(open(os.path.join(ROOT, "configs", "mechanism", "cells", "matched_free_energy", "shift.json")))
    for v in X["variants"]:
        P = json.load(open(os.path.join(ROOT, "configs", "final_campaign", "cells", "expA", f"{v['name']}.json")))
        assert P["engine_cfg"] == dict(X["engine_cfg"], h=2.5e-5, variant="shift", alpha=None, kappa=1.0,
                                       lam=float(v["lam"]))
        assert P["B"] == 3276800000 and P["N_ladder"] == [512, 128] and P["seeds"] == list(range(8300, 8332))
        assert P["T_N"] == {"512": 160.0, "128": 640.0}
        assert P["results_dir"] == f"results/final_campaign/expA/{v['name']}" and P["out_dir"] == v["name"]
        assert P["analysis_dir"] == f"results/final_campaign/expA/{v['name']}/analysis"
        assert P["fig_dir"] == f"figures/final_campaign/expA/{v['name']}"
        assert P["reference_file"] == f"results/final_campaign/references/expA_{v['name']}_reference.npz"
        assert P["ledger"] == "results/final_campaign/expA/ledger.csv"
        assert P["ledger_filter"] == {"experiment": "expA", "variant": v["name"]}
        assert P["engine_version"] == "gateway_family_numba/1" and P["reuse"] is None
        assert P["thresholds"]["e_F"] == [0.002, 0.0055, 0.02] and P["e_Fp_floor"] == 0.03227
        if v["name"] == "shift_lam1":       # lam = 1 repeats the Exp I shifted-fibre dynamics exactly
            assert P["engine_cfg"] == mech["engine_cfg"] and P["thresholds"] == mech["thresholds"]


def test_cells_accepted_by_analyze_ladder_config_path():
    import make_cells as MKC
    for name in ("shift_lam1", "shift_lam4", "shift_lam16", "shift_lam128"):
        p = os.path.join(ROOT, "configs", "final_campaign", "cells", "expA", f"{name}.json")
        probs = MKC.validate_cell(json.load(open(p)), p, tiny_run=True, need_reference=HAVE_REFS)
        assert probs == [], (name, probs)


@pytest.mark.skipif(not HAVE_REFS, reason="references not built yet (scripts/mechanism/build_references.py)")
def test_check2_references_bitwise():
    import expA_check_references as CR
    rec = CR.check()
    assert rec["verdict"] == "PASS", rec["problems"]
    assert all(all(v["equal_to_equal_budget"].values()) and v["equal_across_cells"] for v in rec["primary"].values())


# ================================================================================================ D driver
def test_driver_seed_major_order_and_plans():
    P = C.load_config()
    J = C.expA_jobs(P)
    assert len(J) == 512
    seeds = list(range(8300, 8332))
    for b, s in enumerate(seeds):
        blk = J[16 * b:16 * (b + 1)]
        assert {j["seed"] for j in blk} == {s}
        assert [(j["variant"], j["N"], j["method"]) for j in blk] == [
            (c, N, m) for c in ("shift_lam1", "shift_lam4", "shift_lam16", "shift_lam128") for N in (512, 128)
            for m in ("abf", "fr")]
    for j in J:
        assert j["experiment"] == "expA" and j["N"] * j["n_steps"] == 3276800000
        assert j["cfg"]["lam"] == float(j["variant"].replace("shift_lam", "")) and j["cfg"]["variant"] == "shift"
        assert j["cfg"]["h"] == 2.5e-5 and j["cfg"]["kappa"] == 1.0
    S = C.expA_jobs(P, smoke=True)
    assert len(S) == 32 and {j["seed"] for j in S[:16]} == {8390} and {j["seed"] for j in S[16:]} == {8391}
    assert {(j["N"], j["n_steps"]) for j in S} == {(512, 640000), (128, 2560000)}
    R = C.rc()
    assert R.out_path(C.prod_root(), J[0]) == os.path.join(ROOT, "results/final_campaign/expA/shift_lam1/N512/"
                                                                 "s8300_abf.npz")
    assert C.smoke_root() == os.path.join(ROOT, "results/final_campaign/expA/smoke/T4")
    with pytest.raises(ValueError):
        C.expA_jobs(P, only_cell=["shift"])


def test_driver_frozen_identities(monkeypatch):
    assert C.check_engine() == C.EXPECTED_ENGINE_SHA256
    assert C.witness_engine_sha() in (None, C.EXPECTED_ENGINE_SHA256)
    with pytest.raises(C.EngineMismatch):
        C.check_engine("0" * 64)
    with pytest.raises(C.ConfigMismatch):
        C.load_config(expected="0" * 64)
    spec = importlib.util.spec_from_file_location("final_run_cells", os.path.join(FC, "run_cells.py"))
    RCF = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(RCF)
    assert RCF.C is C
    monkeypatch.setattr(C, "EXPECTED_ENGINE_SHA256", "0" * 64)
    assert RCF.main(["--dry-run", "--workers", "1"]) == 2
    monkeypatch.undo()
    monkeypatch.setattr(C, "EXPECTED_CONFIG_SHA256", "1" * 64)
    assert RCF.main(["--dry-run", "--workers", "1"]) == 2


def test_wait_for_disk_pause_and_timeout():
    class U:
        def __init__(self, seq):
            self.seq = list(seq)

        def __call__(self, path):
            v = self.seq.pop(0) if len(self.seq) > 1 else self.seq[0]
            return shutil._ntuple_diskusage(0, 0, int(v))
    logs = []
    assert C.wait_for_disk("/", usage=U([5e9]), sleep=lambda s: None, log=logs.append) == 5e9 and not logs
    free = C.wait_for_disk("/", usage=U([1.5e9, 1.9e9, 2.5e9, 3.2e9]), sleep=lambda s: None, log=logs.append)
    assert free == 3.2e9 and "DISK PAUSE" in logs[0] and "DISK RESUME" in logs[-1] and len(logs) == 4
    t = [0.0]

    def clock():
        t[0] += 3600.0
        return t[0]
    with pytest.raises(C.DiskPauseTimeout):
        C.wait_for_disk("/", usage=U([1e9]), sleep=lambda s: None, log=logs.append, clock=clock, max_pause_s=7200)


def test_driver_disk_pause_in_dispatch(tmp_path, monkeypatch, capsys):
    """The driver starts no job while free space < 2 GB, logs the pause to <root>/disk_pause.log, keeps going once
    >= 3 GB are free; a pause beyond the limit stops it (exit 3) without starting anything."""
    spec = importlib.util.spec_from_file_location("final_run_cells2", os.path.join(FC, "run_cells.py"))
    RCF = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(RCF)
    seq = [1.0e9, 2.5e9, 3.5e9]

    def usage(path):
        v = seq.pop(0) if len(seq) > 1 else seq[0]
        return shutil._ntuple_diskusage(0, 0, int(v))
    monkeypatch.setattr(C, "DISK_USAGE", usage)
    monkeypatch.setattr(C, "DISK_POLL_S", 0.01)
    out = str(tmp_path / "expA")
    args = ["--smoke", "--smoke-T", "0.01", "--only-cell", "shift_lam4", "--only-N", "128", "--seeds", "8390",
            "--workers", "1", "--out-root", out, "--compute-ledger", str(tmp_path / "cl.csv")]
    assert RCF.main(args) == 0
    log = open(os.path.join(out, "smoke", "T0.01", "disk_pause.log")).read()
    assert "DISK PAUSE" in log and "disk pause continues" in log and "DISK RESUME" in log
    assert len(list(csv.DictReader(open(os.path.join(out, "smoke", "T0.01", "ledger.csv"))))) == 2
    monkeypatch.setattr(C, "DISK_USAGE", lambda p: shutil._ntuple_diskusage(0, 0, int(1e9)))
    monkeypatch.setattr(C, "DISK_MAX_PAUSE_S", 0.05)
    out2 = str(tmp_path / "expA2")
    args2 = [a if a != out else out2 for a in args]
    assert RCF.main(args2) == 3
    assert "TIMEOUT" in open(os.path.join(out2, "smoke", "T0.01", "disk_pause.log")).read()
    assert not os.path.exists(os.path.join(out2, "smoke", "T0.01", "shift_lam4"))


def test_driver_end_to_end_tiny(tmp_path):
    out, cl = str(tmp_path / "expA"), str(tmp_path / "compute_ledger.csv")
    args = ["--smoke", "--smoke-T", "0.01", "--only-cell", "shift_lam16", "--seeds", "8390", "--workers", "2",
            "--out-root", out, "--compute-ledger", cl]
    p = run_script("run_cells.py", args)
    assert p.returncode == 0, p.stdout + p.stderr
    root = os.path.join(out, "smoke", "T0.01")
    files = sorted(os.path.relpath(os.path.join(d, f), root) for d, _, fs in os.walk(root) for f in fs
                   if f.endswith(".npz"))
    assert files == ["shift_lam16/N128/s8390_abf.npz", "shift_lam16/N128/s8390_fr.npz",
                     "shift_lam16/N512/s8390_abf.npz", "shift_lam16/N512/s8390_fr.npz"]
    led = list(csv.DictReader(open(os.path.join(root, "ledger.csv"))))
    assert len(led) == 4 and all(r["status"] == "complete" and r["experiment"] == "expA" for r in led)
    comp = list(csv.DictReader(open(cl)))
    assert len(comp) == 4 and all(r["stage"] == "expA-smoke" and r["threads"] == "1" for r in comp)
    prov = list(csv.DictReader(open(os.path.join(root, "run_provenance.csv"))))
    assert len(prov) == 4 and all(r["engine_sha256"] == C.EXPECTED_ENGINE_SHA256 and r["precision"] == "float64"
                                  and r["out_sha256"] and r["cpu_set"] for r in prov)
    assert all(r["gate"] == "smoke (no production gate)" and r["config_sha256"] == C.EXPECTED_CONFIG_SHA256
               and r["lam"] == "16.0" for r in prov)
    for f in files:
        meta = gf.load_result(os.path.join(root, f))["meta"]
        assert meta["engine_sha256"] == C.EXPECTED_ENGINE_SHA256 and meta["lam"] == 16.0
    # idempotent
    p2 = run_script("run_cells.py", args)
    assert p2.returncode == 0 and "nothing to run" in p2.stdout, p2.stdout + p2.stderr
    assert len(list(csv.DictReader(open(cl)))) == 4
    # a complete result of a DIFFERENT run at a job path: refused up front, never overwritten
    a, f = os.path.join(root, "shift_lam16/N512/s8390_abf.npz"), os.path.join(root, "shift_lam16/N512/s8390_fr.npz")
    shutil.copyfile(a, f)
    sha = C.sha256_file(f)
    p3 = run_script("run_cells.py", args)
    assert p3.returncode == 2 and "STALE" in p3.stdout, p3.stdout + p3.stderr
    assert C.sha256_file(f) == sha
    # production dry run on the scratch root: 512 jobs planned, nothing written; without the gate records the
    # production gate refuses (prereg 1.3: production only after G-A1/G-A2 and all nine checks pass); the projection
    # gate refuses a tiny cap
    p4 = run_script("run_cells.py", ["--dry-run", "--workers", "1", "--out-root", out, "--compute-ledger", cl])
    assert p4.returncode == 2 and "512 jobs" in p4.stdout, p4.stdout + p4.stderr
    assert "production gate (prereg 1.3): NOT SATISFIED" in p4.stdout and "REFUSED (production gate)" in p4.stdout
    assert not os.path.exists(os.path.join(out, "shift_lam1"))
    p5 = run_script("run_cells.py", ["--dry-run", "--workers", "1", "--out-root", out, "--compute-ledger", cl,
                                     "--stage-cap-core-h", "1"])
    assert p5.returncode == 2 and "projection gate" in p5.stdout
    p6 = run_script("run_cells.py", ["--dry-run", "--workers", "100000", "--out-root", out])
    assert p6.returncode == 2 and "affinity" in p6.stdout


def test_worker_is_importable_under_spawn(tmp_path):
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    assert pickle.loads(pickle.dumps(C.run_timed)) is C.run_timed
    P = C.load_config()
    job = [j for j in C.expA_jobs(P, smoke=True, smoke_T=0.01, only_cell=["shift_lam4"], only_N=[128],
                                  seeds=[8391])][0]
    with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as ex:
        out = ex.submit(C.run_timed, job, str(tmp_path)).result(timeout=600)
    assert out["row"]["status"] == "complete", out["row"]
    assert out["prov"]["engine_sha256"] == C.EXPECTED_ENGINE_SHA256
    assert os.path.exists(C.rc().out_path(str(tmp_path), job))


# ================================================================================================ C6
def _make_pair(root, cell, N, seed, save, ramp=0.05, gamma=15.0, lam=16.0):
    cfg = dict(EASY, variant="shift", kappa=1.0, lam=lam, h=4e-4, gamma=gamma, ramp_t=ramp)
    n = int(save[-1]) if save is not None else 3000
    paths = {}
    for m in ("abf", "fr"):
        p = os.path.join(root, cell, f"N{N}", f"s{seed}_{m}.npz")
        gf.run_job(cfg, m, N, seed, n, p, save_steps=save)
        paths[m] = p
    return paths


def test_c6_rules_replays_and_counts(tmp_path):
    root = str(tmp_path / "r")
    _make_pair(root, "death", 16, 7100, None)                                   # deaths after several saves
    _make_pair(root, "late", 16, 7100, np.array([1500, 3000]))                  # 0 compared saves, t_first > 0
    _make_pair(root, "t0", 16, 7101, np.array([1500, 3000]), ramp=0.0)          # first event at opportunity 0
    _make_pair(root, "perm", 6, 7156, np.arange(1, 601, 10))                    # permutation-only event first
    _make_pair(root, "noev", 16, 7102, None, gamma=0.0)                         # no FR event in the whole run
    rec = CP.check_root(root, verbose=False)
    by = {r["cell"]: r for r in rec["pairs"]}
    # the frozen prereg rule (compare up to the last save before the first realised death) passes 4 pairs; the
    # permutation-only pair passes only the event rule + replay: NOT a pass of check 6 without a dated amendment
    assert rec["verdict"] == "AMENDMENT_REQUIRED" and rec["n_pass"] == 4 and rec["n_pass_event_rule_only"] == 1 \
        and rec["n_fail"] == 0, [(r["cell"], r["status"], r.get("mismatches"), r.get("replay")) for r in rec["pairs"]]
    assert CP.verdict_exit(rec["verdict"]) == 4 and rec["event_rule_amendment"] is None
    assert by["perm"]["status"] == "PASS_EVENT_RULE" and by["perm"]["frozen_rule"] == "FAIL"
    assert by["perm"]["event_rule"] == "PASS" and by["perm"]["death_rule_mismatches"]
    assert all(by[c]["status"] == "PASS" and by[c]["frozen_rule"] == "PASS" for c in ("death", "late", "t0", "noev"))
    assert by["death"]["rule"] == "death" and by["death"]["n_compared_saves"] >= 1 and "replay" not in by["death"]
    assert by["late"]["n_compared_saves"] == 0 and by["late"]["replay"]["passed"]
    rl = by["late"]["replay"]
    assert rl["first_event_opportunity_step"] > 0 and rl["state_at_t_equal"] and rl["accumulators_at_t1_equal"]
    assert rl["fr_state_is_gather_of_abf_state"] and rl["fr_events_at_t1"] == 1
    assert by["t0"]["replay"]["first_event_opportunity_step"] == 0 and by["t0"]["replay"]["passed"]
    assert by["perm"]["rule"] == "event" and by["perm"]["permutation_flag"] and by["perm"]["replay"]["passed"]
    assert by["perm"]["n_saves_before_first_event"] < by["perm"]["n_saves_before_first_death"]
    assert by["perm"]["replay"]["fr_deaths_at_t1"] == 0
    assert by["noev"].get("whole_run_without_fr_event") and by["noev"]["n_compared_saves"] == by["noev"]["n_saves"]
    assert rec["n_replays"] == 3 and rec["n_event_rule"] == 1
    # licensed by a (validated) amendment reference: PASS, the reference recorded; the pair keeps its own status
    lic = CP.check_root(root, verbose=False, event_rule_amendment="AMD-C6-EVENT-RULE")
    assert lic["verdict"] == "PASS" and lic["n_pass"] == 4 and lic["n_pass_event_rule_only"] == 1
    assert lic["event_rule_amendment"] == "AMD-C6-EVENT-RULE" and "AMD-C6-EVENT-RULE" in lic["verdict_note"]
    assert {r["cell"]: r["status"] for r in lic["pairs"]}["perm"] == "PASS_EVENT_RULE"


def test_c6_detects_tampering_and_missing_arm(tmp_path):
    root = str(tmp_path / "r")
    pa = _make_pair(root, "c1", 16, 7100, None)
    pb = _make_pair(root, "c2", 16, 7100, None)
    pc = _make_pair(root, "c3", 16, 7100, np.array([1500, 3000]))

    def bump_m0(d):
        d["M_all"][1, 50] = np.nextafter(d["M_all"][1, 50], np.inf)
    tamper(pa["fr"], bump_m0)

    def bump_trace(d):
        d["traces"][0, 0] += 1e-12
    tamper(pb["abf"], bump_trace)

    def bump_stored_event_save(d):                    # the stored FR save the replay must reproduce
        d["M_all"][0, 10] += 1.0
    tamper(pc["fr"], bump_stored_event_save)
    os.makedirs(os.path.join(root, "c4", "N16"))
    shutil.copyfile(pa["abf"], os.path.join(root, "c4", "N16", "s7100_abf.npz"))
    rec = CP.check_root(root, verbose=False)
    by = {r["cell"]: r for r in rec["pairs"]}
    assert rec["verdict"] == "FAIL" and rec["n_fail"] == 4
    assert by["c1"]["status"] == "FAIL" and "M_all" in by["c1"]["mismatches"]
    assert by["c2"]["status"] == "FAIL" and "traces" in by["c2"]["mismatches"]
    assert by["c3"]["status"] == "FAIL" and not by["c3"]["replay"]["passed"]
    assert by["c4"]["status"] == "INCOMPLETE"


def test_c6_cli_record_idempotent(tmp_path):
    root = str(tmp_path / "r")
    _make_pair(root, "c1", 16, 7100, None)
    out = str(tmp_path / "c6.json")
    p = run_script("expA_check_pairing.py", ["--root", root, "--out", out])
    assert p.returncode == 0 and "written" in p.stdout, p.stdout + p.stderr
    p = run_script("expA_check_pairing.py", ["--root", root, "--out", out])
    assert p.returncode == 0 and "up to date" in p.stdout
    _make_pair(root, "c2", 16, 7101, None)
    p = run_script("expA_check_pairing.py", ["--root", root, "--out", out])
    assert p.returncode == 2 and "REFUSED" in p.stdout
    p = run_script("expA_check_pairing.py", ["--root", root, "--out", out, "--force"])
    assert p.returncode == 0 and json.load(open(out))["n_pass"] == 2


# ================================================================================================ C9
def test_c9_result_keys_are_the_engine_contract(tmp_path):
    p = str(tmp_path / "x.npz")
    gf.run_job(shift(16.0), "fr", 8, 8390, 400, p)
    with np.load(p) as z:
        assert sorted(z.files) == sorted(CS.RESULT_KEYS)
    stored = os.path.join(ROOT, "results/mechanism/matched_free_energy/shift/N128/s8100_abf.npz")
    if os.path.exists(stored):
        with np.load(stored) as z:
            assert sorted(z.files) == sorted(CS.RESULT_KEYS)


@pytest.mark.skipif(not HAVE_REFS, reason="references not built yet")
def test_c9_plan_mode_on_tiny_driver_smoke(tmp_path):
    out = str(tmp_path / "expA")
    p = run_script("run_cells.py", ["--smoke", "--smoke-T", "0.01", "--workers", "4", "--out-root", out,
                                    "--compute-ledger", str(tmp_path / "cl.csv")])
    assert p.returncode == 0, p.stdout + p.stderr
    p = run_script("expA_check_smoke.py", ["--smoke-T", "0.01", "--out-root", out])
    assert p.returncode == 0, p.stdout + p.stderr
    rec = json.load(open(os.path.join(out, "checks", "check9_smoke.json")))
    assert rec["verdict"] == "PASS" and rec["n_files_pass"] == 32 and rec["check6"]["n_pass"] == 16
    assert all(f["pipeline"]["ran"] and f["analysis_contract"] == "ok" for f in rec["files"])
    assert not rec["interpreted"]
    assert json.load(open(os.path.join(out, "checks", "check6_pairing_smoke.json")))["verdict"] == "PASS"
    # the production gate's check-9 / check-6 items accept these records (smoke horizon 0.01 here)
    g = G.production_gate(C.load_config(), out, smoke_T=0.01)
    it = {i["item"]: i for i in g["items"]}
    assert it["check 9 smoke"]["ok"] and it["check 6 pairing (smoke)"]["ok"], G.format_gate(g)
    # a missing planned file fails the plan
    os.remove(os.path.join(out, "smoke", "T0.01", "shift_lam4", "N128", "s8391_fr.npz"))
    p = run_script("expA_check_smoke.py", ["--smoke-T", "0.01", "--out-root", out, "--out",
                                           str(tmp_path / "c9b.json"), "--out6", str(tmp_path / "c6b.json")])
    assert p.returncode == 1, p.stdout + p.stderr
    assert json.load(open(str(tmp_path / "c9b.json")))["verdict"] == "FAIL"
    # the default records (PASS, other inputs) are kept, and the FAIL verdict still exits 1 (not 2)
    p = run_script("expA_check_smoke.py", ["--smoke-T", "0.01", "--out-root", out])
    assert p.returncode == 1 and "DIFFERENT input signature is kept" in p.stdout, p.stdout + p.stderr
    assert json.load(open(os.path.join(out, "checks", "check9_smoke.json")))["verdict"] == "PASS"
    # ... and the gate now sees those PASS records as stale (the smoke files changed)
    g = G.production_gate(C.load_config(), out, smoke_T=0.01)
    it = {i["item"]: i for i in g["items"]}
    assert not it["check 9 smoke"]["ok"] and "stale" in it["check 9 smoke"]["detail"]
    assert not it["check 6 pairing (smoke)"]["ok"] and "stale" in it["check 6 pairing (smoke)"]["detail"]


def test_c9_detects_inconsistent_diagnostics(tmp_path):
    root = str(tmp_path / "r")
    good = _make_pair(root, "ok", 16, 7100, None)
    bads = {}
    for name in ("d2", "d1", "nan", "key", "abf_fr"):
        bads[name] = _make_pair(root, name, 16, 7100, None)

    def d2(d):
        d["dep_age_C"][-1, 0, 90] += 1.0

    def d1(d):
        d["fr_death_pos_hist_cum"][-1, 10] += 1

    def nan(d):
        d["M_all"][5, 3] = np.nan

    def key(d):
        del d["dep_cross_C"]

    def abf_fr(d):
        d["fr_kd_cum"][-1] = 1
    tamper(bads["d2"]["fr"], d2)
    tamper(bads["d1"]["fr"], d1)
    tamper(bads["nan"]["abf"], nan)
    tamper(bads["key"]["fr"], key)
    tamper(bads["abf_fr"]["abf"], abf_fr)
    rec = CS.check_smoke(root, expected=None, pipeline=False, verbose=False)
    st = {(f["cell"], f["method"]): f for f in rec["files"]}
    assert st[("ok", "abf")]["status"] == "PASS" and st[("ok", "fr")]["status"] == "PASS"
    assert st[("ok", "fr")]["fr_deaths_total"] > 0
    assert any("D2" in q for q in st[("d2", "fr")]["problems"])
    assert any("D1" in q for q in st[("d1", "fr")]["problems"])
    assert any("non-finite" in q for q in st[("nan", "abf")]["problems"])
    assert any("contract" in q for q in st[("key", "fr")]["problems"])
    assert any("FR counters" in q for q in st[("abf_fr", "abf")]["problems"])
    assert rec["verdict"] == "FAIL"


# ================================================================================================ C8
def test_c8_reconstruct_rerun_compare(tmp_path):
    p1 = str(tmp_path / "stored" / "s1_fr.npz")
    gf.run_job(shift(16.0, gamma=15.0, ramp_t=0.01), "fr", 16, 8390, 2000, p1)        # default call
    p2 = str(tmp_path / "stored" / "s2_abf.npz")
    gf.run_job(shift(4.0), "abf", 8, 8391, 1200, p2, trace_steps=np.arange(0, 1201, 7),  # explicit grids
               snapshot_steps=[5, 600, 1200])
    out, cl = str(tmp_path / "checks" / "c8.json"), str(tmp_path / "cl.csv")
    rec8c = fake_8c(tmp_path / "c8c.json")
    kw1, r1 = C8.reconstruct(p1)
    kw2, r2 = C8.reconstruct(p2)
    assert kw1 is not None and r1["call"].startswith("run_job default")
    assert kw2 is not None and r2["call"].startswith("explicit")
    p = run_script("expA_check_old_bitwise.py", ["--files", p1, p2, "--out", out, "--compute-ledger", cl,
                                                 "--workers", "2", "--inertness-record", rec8c])
    assert p.returncode == 0, p.stdout + p.stderr
    rec = json.load(open(out))
    assert rec["verdict"] == "PASS" and rec["n_equal"] == 2 and rec["engine"]["stored_equal_expected"]
    assert rec["part_8ab_verdict"] == "PASS" and rec["part_8c"]["verdict"] == "PASS" and rec["part_8c"]["current"]
    assert all(j["n_compared"] >= 45 and j["cfg_json_equal"] and j["run_signature_equal"] for j in rec["jobs"])
    assert len(list(csv.DictReader(open(cl)))) == 2
    # resumable: up-to-date job records are not re-run
    p = run_script("expA_check_old_bitwise.py", ["--files", p1, p2, "--out", out, "--compute-ledger", cl,
                                                 "--inertness-record", rec8c])
    assert p.returncode == 0 and p.stdout.count("up to date") == 2
    assert len(list(csv.DictReader(open(cl)))) == 2
    # a stored array that the HEAD re-run does not reproduce -> MISMATCH (verdict FAIL); a different final record is
    # not overwritten without --force
    p3 = str(tmp_path / "stored" / "s3_fr.npz")

    def bump(d):
        d["C_all"][-1, 0] += 1.0
    tamper(p1, bump, out=p3)
    p = run_script("expA_check_old_bitwise.py", ["--files", p3, "--out", out, "--compute-ledger", cl,
                                                 "--inertness-record", rec8c])
    assert p.returncode == 2 and "REFUSED" in p.stdout
    out3 = str(tmp_path / "checks3" / "c8.json")
    p = run_script("expA_check_old_bitwise.py", ["--files", p3, "--out", out3, "--compute-ledger", cl,
                                                 "--inertness-record", rec8c])
    assert p.returncode == 1, p.stdout + p.stderr
    assert json.load(open(out3))["verdict"] == "FAIL"
    j = json.load(open(out3))["jobs"][0]
    assert j["status"] == "MISMATCH" and j["mismatched"] == ["C_all"]
    # dry run on the four real stored jobs: signatures reproduced at HEAD (nothing run)
    if all(os.path.exists(f) for f in C8.DEFAULT_FILES):
        p = run_script("expA_check_old_bitwise.py", ["--dry-run", "--out", str(tmp_path / "c8real.json")])
        assert p.returncode == 0 and p.stdout.count("signature REPRODUCED") == 4, p.stdout + p.stderr


def test_c8_refuses_other_engine(tmp_path, monkeypatch):
    p1 = str(tmp_path / "s1_abf.npz")
    gf.run_job(shift(1.0), "abf", 8, 8390, 400, p1)
    monkeypatch.setattr(C, "EXPECTED_ENGINE_SHA256", "0" * 64)
    assert C8.main(["--files", p1, "--out", str(tmp_path / "c8.json"), "--compute-ledger",
                    str(tmp_path / "cl.csv")]) == 2
    assert not os.path.exists(str(tmp_path / "c8.json"))


def test_c8_partial_until_harness_record(tmp_path):
    """Check 8 is PASS only with the 8c harness-inertness record PASS and current: PARTIAL (exit 3) while it is missing
    or stale, completed to PASS once it exists (job records reused, no --force), FAIL if it FAILs."""
    p1 = str(tmp_path / "stored" / "s1_abf.npz")
    gf.run_job(shift(4.0), "abf", 8, 8390, 600, p1)
    out, cl, rec8c = str(tmp_path / "c8.json"), str(tmp_path / "cl.csv"), str(tmp_path / "c8c.json")
    args = ["--files", p1, "--out", out, "--compute-ledger", cl, "--inertness-record", rec8c]
    p = run_script("expA_check_old_bitwise.py", args)
    assert p.returncode == 3, p.stdout + p.stderr
    r = json.load(open(out))
    assert r["verdict"] == "PARTIAL" and r["part_8ab_verdict"] == "PASS" and not r["part_8c"]["present"]
    assert "NOT a check-8 pass" in r["verdict_note"]
    fake_8c(rec8c)
    p = run_script("expA_check_old_bitwise.py", args)
    assert p.returncode == 0 and "up to date" in p.stdout, p.stdout + p.stderr
    r = json.load(open(out))
    assert r["verdict"] == "PASS" and r["supersedes"]["verdict"] == "PARTIAL"
    assert r["part_8c"]["sha256"] == C.sha256_file(rec8c) and r["part_8c"]["current"]
    assert len(list(csv.DictReader(open(cl)))) == 1                  # the 8b job ran once
    fake_8c(rec8c, verdict="FAIL")
    p = run_script("expA_check_old_bitwise.py", args)
    assert p.returncode == 1 and json.load(open(out))["verdict"] == "FAIL", p.stdout + p.stderr
    fake_8c(rec8c, current=False)                                       # the tests / harness changed since
    p = run_script("expA_check_old_bitwise.py", args)
    r = json.load(open(out))
    assert p.returncode == 3 and r["verdict"] == "PARTIAL" and not r["part_8c"]["current"]
    # a final record without a signature is a different record: refused without --force
    out2 = write_json(tmp_path / "c8b.json", dict(check="8", verdict="PASS"))
    p = run_script("expA_check_old_bitwise.py", ["--files", p1, "--out", out2, "--compute-ledger", cl,
                                                 "--inertness-record", rec8c])
    assert p.returncode == 2 and "REFUSED" in p.stdout and "input_signature" not in json.load(open(out2))
    assert C8.check8_verdict("PASS", dict(present=True, current=True, verdict="PASS")) == "PASS"
    assert C8.check8_verdict("FAIL", dict(present=True, current=True, verdict="PASS")) == "FAIL"
    assert C8.check8_verdict("PASS", dict(present=False, current=False, verdict=None)) == "PARTIAL"


# ================================================================================================ C6 event rule (CLI)
def test_c6_event_rule_needs_dated_amendment(tmp_path):
    """A pair that passes only the (non-preregistered) event rule makes check 6 AMENDMENT_REQUIRED (exit 4); the
    licensing reference must occur in the prereg's dated-amendment section (refused otherwise, nothing written); a
    licensed run is PASS with the reference in the record and its input signature."""
    root = str(tmp_path / "r")
    _make_pair(root, "perm", 6, 7156, np.arange(1, 601, 10))
    out, out2 = str(tmp_path / "c6.json"), str(tmp_path / "c6_licensed.json")
    p = run_script("expA_check_pairing.py", ["--root", root, "--out", out])
    assert p.returncode == 4 and "AMENDMENT_REQUIRED" in p.stdout, p.stdout + p.stderr
    rec = json.load(open(out))
    assert rec["verdict"] == "AMENDMENT_REQUIRED" and rec["n_pass"] == 0 and rec["n_pass_event_rule_only"] == 1
    assert rec["pairs"][0]["status"] == "PASS_EVENT_RULE" and rec["pairs"][0]["frozen_rule"] == "FAIL"
    ref = "AMD-C6-EVENT-RULE"
    for pr in (prereg_with(tmp_path), prereg_with(tmp_path, body_ref=ref)):    # absent / only in the frozen body
        p = run_script("expA_check_pairing.py", ["--root", root, "--out", out2, "--event-rule-amendment", ref,
                                                 "--prereg", pr])
        assert p.returncode == 2 and "REFUSED" in p.stdout and not os.path.exists(out2), p.stdout + p.stderr
    pr = prereg_with(tmp_path, amendment_ref=ref)
    p = run_script("expA_check_pairing.py", ["--root", root, "--out", out2, "--event-rule-amendment", ref,
                                             "--prereg", pr])
    assert p.returncode == 0, p.stdout + p.stderr
    rec = json.load(open(out2))
    assert rec["verdict"] == "PASS" and rec["event_rule_amendment"] == ref
    assert rec["signature_options"]["event_rule_amendment"] == ref and rec["pairs"][0]["status"] == "PASS_EVENT_RULE"
    # the licensed result does not silently replace the unlicensed record (different input signature): PASS + refused
    p = run_script("expA_check_pairing.py", ["--root", root, "--out", out, "--event-rule-amendment", ref,
                                             "--prereg", pr])
    assert p.returncode == 2 and "REFUSED" in p.stdout and json.load(open(out))["verdict"] == "AMENDMENT_REQUIRED"
    assert CP.record_exit("FAIL", "REFUSED") == 1 and CP.record_exit("AMENDMENT_REQUIRED", "written") == 4
    assert CP.record_exit("PASS", "up to date", None) == 0 and CP.record_exit("PASS", "written", "REFUSED") == 2


def test_c6_c9_signature_covers_frozen_inputs(tmp_path, monkeypatch):
    fi = {os.path.relpath(p, ROOT) for p in CP.frozen_inputs()}
    want = {"configs/final_campaign/expA.json", "src/gateway_family_numba.py", "scripts/equal_budget/eqb_family.py",
            "scripts/equal_budget/eqb_metrics.py"}
    want |= {f"configs/final_campaign/cells/expA/shift_lam{l}.json" for l in (1, 4, 16, 128)}
    want |= {f"results/final_campaign/references/expA_shift_lam{l}_reference.npz" for l in (1, 4, 16, 128)}
    assert want <= fi, want - fi
    f = tmp_path / "frozen.npz"
    f.write_bytes(b"a")
    monkeypatch.setattr(CP, "frozen_inputs", lambda: [str(f)])
    o = CP.c6_options("smoke", 4.0, True, None)
    rec = dict(input_signature=CP.record_signature([], o), signature_options=o)
    assert CP.record_is_current(rec, [])[0]
    f.write_bytes(b"b")                                    # e.g. a reference rebuilt with --force
    assert not CP.record_is_current(rec, [])[0]
    assert CP.record_signature([], CS.c9_options("smoke", 4.0, True, True, "AMD-1")) != \
        CP.record_signature([], CS.c9_options("smoke", 4.0, True, True, None))


# ================================================================================================ C2 record
@pytest.mark.skipif(not HAVE_REFS, reason="references not built yet")
def test_check2_record_signed_and_idempotent(tmp_path):
    out = str(tmp_path / "c2.json")
    p = run_script("expA_check_references.py", ["--out", out])
    assert p.returncode == 0 and "wrote" in p.stdout, p.stdout + p.stderr
    rec = json.load(open(out))
    assert rec["verdict"] == "PASS" and CR.record_is_current(rec) == (True, "current")
    p = run_script("expA_check_references.py", ["--out", out])
    assert p.returncode == 0 and "up to date" in p.stdout
    write_json(out, dict(check="2", verdict="PASS"))                 # an older record without a signature
    p = run_script("expA_check_references.py", ["--out", out])
    assert p.returncode == 2 and "REFUSED" in p.stdout and "input_signature" not in json.load(open(out))
    assert not CR.record_is_current(json.load(open(out)))[0]
    p = run_script("expA_check_references.py", ["--out", out, "--force"])
    assert p.returncode == 0, p.stdout + p.stderr
    rec = json.load(open(out))
    assert rec["supersedes"]["input_signature"] is None and CR.record_is_current(rec)[0]
    # the signature covers the reference files: a rebuilt (changed) reference makes a record stale
    refs2 = tmp_path / "refs"
    shutil.copytree(REFS, refs2)
    s0 = CR.input_signature(refs_dir=str(refs2))
    with open(refs2 / "expA_shift_lam16_reference.npz", "ab") as fh:
        fh.write(b"\0")
    assert CR.input_signature(refs_dir=str(refs2)) != s0


# ================================================================================================ D driver (fixes)
def test_cpu_allocation_rules():
    sib = lambda c: {c, c + 128 if c < 128 else c - 128}            # noqa: E731  (prereg section 3 layout)
    assert C.cpu_allocation_problems(8, cpus=set(range(72, 80)), siblings_of=sib) == []
    assert C.cpu_allocation_problems(64, cpus=set(range(0, 64)), siblings_of=sib) == []
    pr = C.cpu_allocation_problems(9, cpus=set(range(72, 80)), siblings_of=sib)
    assert len(pr) == 1 and "physical cores in this process's affinity set" in pr[0]
    pr = C.cpu_allocation_problems(64, cpus=set(range(64, 96)) | set(range(192, 224)), siblings_of=sib)
    assert any("SMT siblings" in q for q in pr) and any("physical cores" in q for q in pr)
    pr = C.cpu_allocation_problems(16, cpus={72, 73, 200}, siblings_of=sib)
    assert any("SMT siblings" in q and "[72, 200]" in q for q in pr)
    pr = C.cpu_allocation_problems(65, cpus=set(range(0, 100)), siblings_of=sib)
    assert any("> 64" in q for q in pr)
    assert C.cpu_allocation_problems(0, cpus={72}, siblings_of=sib)
    if os.path.exists("/sys/devices/system/cpu/cpu200/topology/thread_siblings_list"):
        assert C.smt_siblings(72) >= {72}


def test_amendment_ref_found(tmp_path):
    ref = "AMD-XYZ-0001"
    assert not C.amendment_ref_found(ref, prereg_with(tmp_path))
    assert not C.amendment_ref_found(ref, prereg_with(tmp_path, body_ref=ref))       # frozen body: not an amendment
    pr = prereg_with(tmp_path, amendment_ref=ref)
    assert C.amendment_ref_found(ref, pr)
    assert not C.amendment_ref_found("AMD", pr) and not C.amendment_ref_found("", pr)  # too short to identify
    assert not C.amendment_ref_found(ref, str(tmp_path / "missing.md"))
    assert not C.amendment_ref_found(ref)               # the frozen preregistration: no such amendment


def test_prior_sessions_accounting():
    meta = dict(sessions=[dict(run_start_epoch=100.0, last_ckpt_epoch=700.0),
                          dict(run_start_epoch=1000.0, last_ckpt_epoch=None), dict(run_start_epoch=2000.0)])
    assert C.prior_sessions_s(meta, 600.0) == pytest.approx(600.0 + 600.0 + 600.0)
    assert C.prior_sessions_s(dict(sessions=[dict(run_start_epoch=1.0)]), 600.0) == 0.0
    assert C.prior_sessions_s({}, 600.0) == 0.0


def test_projection_counts_validation_and_check4b(tmp_path, capsys):
    """Experiment A compute already used = the compute ledger's expA* rows + the G-A1/V5 validation tool's own ledger
    + the check-4b record wall (each unless the compute ledger already carries its stage); the projection gate uses
    the total (prereg section 3: Experiment A <= 110 core-h, validation included)."""
    RCF = load_rcf("final_run_cells_proj")
    root, cl = str(tmp_path / "expA"), str(tmp_path / "cl.csv")
    C.append_csv(cl, C.COMPUTE_FIELDS, [C.compute_row("expA-smoke", "a", 3600.0, 0.0),
                                         C.compute_row("setup", "b", 7200.0, 0.0),
                                         C.compute_row("expB-bench", "c", 3600.0, 0.0)])
    tot, br = RCF.expA_used_core_h(cl, root)
    assert tot == pytest.approx(1.0) and br["validation_ledger"] == 0.0 and br["check4b_records"] == 0.0
    C.append_csv(os.path.join(root, "validation", "ledger.csv"), ["chain", "status", "wall_s"],
                 [dict(chain="em_flat", status="complete", wall_s=7200.0),
                  dict(chain="abf_smoke", status="FAILED: x", wall_s=1800.0)])
    write_json(os.path.join(root, "checks", "check4b_frozen_x.json"), dict(provenance=dict(wall_s=1800.0)))
    tot, br = RCF.expA_used_core_h(cl, root)
    assert tot == pytest.approx(4.0) and br["validation_ledger"] == pytest.approx(2.5)
    assert br["check4b_records"] == pytest.approx(0.5)
    base = ["--dry-run", "--workers", "1", "--out-root", root, "--compute-ledger", cl]
    assert RCF.main(base + ["--stage-cap-core-h", "45"]) == 2                   # 4.0 used + 41.94 projected > 45
    s = capsys.readouterr().out
    assert "REFUSED (projection gate): used 4.00 + projected 41.94" in s
    assert RCF.main(base + ["--stage-cap-core-h", "46"]) == 2                   # passes it; the production gate refuses
    s = capsys.readouterr().out
    assert "projection gate" not in s and "REFUSED (production gate)" in s
    C.append_csv(cl, C.COMPUTE_FIELDS, [C.compute_row("expA-validation", "v", 9000.0, 0.0),
                                         C.compute_row("expA-check4b", "c4b", 1800.0, 0.0)])
    tot, br = RCF.expA_used_core_h(cl, root)
    assert tot == pytest.approx(4.0) and "validation_ledger" not in br and "check4b_records" not in br


def test_driver_refuses_unplanned_seeds(tmp_path, capsys):
    RCF = load_rcf("final_run_cells_seeds")
    out = str(tmp_path / "expA")
    assert RCF.main(["--dry-run", "--workers", "1", "--out-root", out, "--seeds", "8390", "8300"]) == 2
    assert "seed(s) [8390] are not production seeds" in capsys.readouterr().out
    assert RCF.main(["--smoke", "--dry-run", "--workers", "1", "--out-root", out, "--seeds", "8300"]) == 2
    assert "seed(s) [8300] are not smoke seeds" in capsys.readouterr().out
    assert RCF.main(["--smoke", "--dry-run", "--workers", "1", "--out-root", out, "--seeds", "8391"]) == 0
    assert not os.path.exists(out)


def test_driver_nonfinite_and_stop_on_first_failure(tmp_path, capsys):
    P = C.load_config()
    R = C.rc()
    # (a) the worker relabels a result the engine saved 'complete' with a non-finite state
    job = C.expA_jobs(P, smoke=True, smoke_T=0.01, only_cell=["shift_lam4"], only_N=[128], seeds=[8390])[0]
    w = str(tmp_path / "w")
    o = C.run_timed(job, w)
    assert o["row"]["status"] == "complete" and C.nonfinite_keys(R.out_path(w, job)) == []

    def nan_x(d):
        d["X_final"][0] = np.nan
    tamper(R.out_path(w, job), nan_x)
    assert C.nonfinite_keys(R.out_path(w, job)) == ["X_final"]
    o = C.run_timed(job, w)
    assert o["row"]["status"].startswith("NONFINITE: X_final"), o["row"]
    # (b) a root whose ledger records a NONFINITE run: the driver refuses to continue (stopping rule (i))
    RCF = load_rcf("final_run_cells_fail")
    out, cl = str(tmp_path / "expA"), str(tmp_path / "cl.csv")
    C.append_csv(os.path.join(C.smoke_root(out, 0.01), "ledger.csv"), R.LEDGER_FIELDS, [o["row"]])
    args = ["--smoke", "--smoke-T", "0.01", "--only-cell", "shift_lam4", "--seeds", "8390", "--workers", "1",
            "--out-root", out, "--compute-ledger", cl]
    assert RCF.main(args) == 2
    assert "NONFINITE run(s) recorded" in capsys.readouterr().out
    assert RCF.main(args + ["--dry-run", "--continue-on-failure"]) == 0
    capsys.readouterr()
    # (c) the first FAILED run (here: a corrupt checkpoint of the first job) stops dispatching: exit 1, nothing else run
    out2 = str(tmp_path / "expA2")
    sroot = C.smoke_root(out2, 0.01)
    jobs = C.expA_jobs(P, smoke=True, smoke_T=0.01, only_cell=["shift_lam4"], seeds=[8390])
    assert len(jobs) == 4
    first = R.out_path(sroot, jobs[0])
    os.makedirs(os.path.dirname(first))
    with open(first + ".ckpt.npz", "wb") as fh:
        fh.write(b"not an npz")
    args2 = [a if a != out else out2 for a in args]
    assert RCF.main(args2) == 1
    s = capsys.readouterr().out
    assert "STOPPED on the first failed run" in s and "3 job(s) not started" in s, s
    led = list(csv.DictReader(open(os.path.join(sroot, "ledger.csv"))))
    assert len(led) == 1 and led[0]["status"].startswith("FAILED")
    assert not any(f.endswith(".npz") and not f.endswith(".ckpt.npz") for _, _, fs in os.walk(sroot) for f in fs)
    comp = list(csv.DictReader(open(cl)))
    assert len(comp) == 1 and comp[0]["stage"] == "expA-smoke" and comp[0]["note"].startswith("FAILED")
    # --continue-on-failure: the other three run, the corrupt one fails again; exit 1 (a failed run)
    assert RCF.main(args2 + ["--continue-on-failure"]) == 1
    led = list(csv.DictReader(open(os.path.join(sroot, "ledger.csv"))))
    assert len(led) == 5 and sum(r["status"].startswith("FAILED") for r in led) == 2
    assert sum(r["status"] == "complete" for r in led) == 3


# ================================================================================================ G production gate
def test_production_gate_reads_every_record(tmp_path):
    P = C.load_config()
    out = str(tmp_path / "expA")
    g = G.production_gate(P, out)
    assert not g["ok"] and len(g["items"]) == 8 and not any(i["ok"] for i in g["items"])
    assert all("record missing" in i["detail"] for i in g["items"])
    chk = fake_gate_records(out)
    g = G.production_gate(P, out)
    assert g["ok"], G.format_gate(g)
    val = os.path.join(out, "validation", "summary.json")

    def broken(path, mutate, label, n_bad=1, prereg=None):
        orig = open(path).read()
        d = json.loads(orig)
        mutate(d)
        with open(path, "w") as fh:
            json.dump(d, fh)
        try:
            gg = G.production_gate(P, out, prereg=prereg)
        finally:
            with open(path, "w") as fh:
                fh.write(orig)
        bad = [i["item"] for i in gg["items"] if not i["ok"]]
        assert not gg["ok"] and len(bad) == n_bad and any(label in b for b in bad), (label, bad)
        return gg
    sel = lambda **kw: (lambda d: d["selection"].update(kw))              # noqa: E731
    broken(val, sel(verdict="REFINE_REQUIRED (lam 128 fails V3)"), "G-A1")
    broken(val, sel(verdict="PROVISIONAL (harness gates pass; ABF-bias smoke PENDING)"), "G-A1")
    broken(val, sel(h_A=1.25e-5), "G-A1")                                 # refined h: needs amendment + new config
    broken(val, lambda d: d.update(smoke=True), "G-A1")
    c = lambda fn: os.path.join(chk, fn)                                  # noqa: E731
    broken(c("check1_7_tests.json"), lambda d: d.update(verdict="FAIL"), "checks 1 and 7")
    broken(c("check1_7_tests.json"), lambda d: d.update(input_signature="0" * 64), "checks 1 and 7")
    broken(c("check2_references.json"), lambda d: d.update(input_signature="0" * 64), "check 2")
    broken(c("check2_references.json"), lambda d: d.update(verdict="FAIL"), "check 2")
    broken(c("check4b_frozen_x.json"), lambda d: d["signature"].update(h_A=1.25e-5), "check 4b")
    broken(c("check4b_frozen_x.json"), lambda d: d.update(status="running"), "check 4b")
    broken(c("check4b_frozen_x.json"), lambda d: d["provenance"].update(engine_sha256="0" * 64), "check 4b")
    broken(c("check8_old_bitwise.json"), lambda d: d.update(verdict="PARTIAL"), "check 8 old")
    broken(c("check8_old_bitwise.json"), lambda d: d["part_8c"].update(sha256="0" * 64), "check 8 old")
    g8 = broken(c("check8c_harness_inertness.json"), lambda d: d.update(verdict="FAIL"), "check 8c", n_bad=2)
    assert {i["item"] for i in g8["items"] if not i["ok"]} == {"check 8c harness inertness (test record)",
                                                               "check 8 old configurations (8a, 8b, 8c)"}
    broken(c("check9_smoke.json"), lambda d: d.update(verdict="FAIL"), "check 9")
    broken(c("check9_smoke.json"), lambda d: d.update(n_files_expected=30), "check 9")
    broken(c("check9_smoke.json"), lambda d: d["signature_options"].update(pipeline=False), "check 9")
    broken(c("check9_smoke.json"), lambda d: d.update(root=str(tmp_path)), "check 9")
    broken(c("check6_pairing_smoke.json"), lambda d: d.update(verdict="AMENDMENT_REQUIRED"), "check 6")
    broken(c("check6_pairing_smoke.json"), lambda d: d["signature_options"].update(replay=False), "check 6")
    broken(c("check6_pairing_smoke.json"), lambda d: d["signature_options"].update(smoke_T=0.01), "check 6")

    # an event-rule amendment named in a current check-6 record must exist in the prereg's amendment section
    def licensed(d):
        d["signature_options"]["event_rule_amendment"] = "AMD-C6-EVENT-RULE"
        d["input_signature"] = CP.record_signature([], d["signature_options"])
    broken(c("check6_pairing_smoke.json"), licensed, "check 6")                      # the frozen prereg: none
    orig = open(c("check6_pairing_smoke.json")).read()
    d = json.loads(orig)
    licensed(d)
    write_json(c("check6_pairing_smoke.json"), d)
    assert G.production_gate(P, out, prereg=prereg_with(tmp_path, amendment_ref="AMD-C6-EVENT-RULE"))["ok"]
    with open(c("check6_pairing_smoke.json"), "w") as fh:
        fh.write(orig)
    # a smoke result file appearing after the checks makes both smoke records stale
    sroot = C.smoke_root(out, 4.0)
    os.makedirs(os.path.join(sroot, "shift_lam1", "N512"))
    shutil.copyfile(os.path.join(REFS, "references_summary.json"),
                    os.path.join(sroot, "shift_lam1", "N512", "s8390_abf.npz"))
    g = G.production_gate(P, out)
    bad = {i["item"]: i["detail"] for i in g["items"] if not i["ok"]}
    assert set(bad) == {"check 9 smoke", "check 6 pairing (smoke)"} and all("stale" in v for v in bad.values())


def test_driver_production_gate_and_override(tmp_path, capsys):
    RCF = load_rcf("final_run_cells_gate")
    out, cl = str(tmp_path / "expA"), str(tmp_path / "cl.csv")
    base = ["--dry-run", "--workers", "1", "--out-root", out, "--compute-ledger", cl]
    assert RCF.main(base) == 2
    s = capsys.readouterr().out
    assert "REFUSED (production gate)" in s and "[NO] check 9 smoke" in s and "[NO] G-A1/G-A2" in s
    ref = "AMD-GATE-OVERRIDE-1"
    for pr in (prereg_with(tmp_path), prereg_with(tmp_path, body_ref=ref)):
        assert RCF.main(base + ["--override-gate", ref, "--prereg", pr]) == 2
        assert "does not occur in the dated-amendment section" in capsys.readouterr().out
    assert RCF.main(base + ["--override-gate", ref, "--prereg", prereg_with(tmp_path, amendment_ref=ref)]) == 0
    s = capsys.readouterr().out
    assert "WARNING: production gate NOT satisfied; overridden by the amendment" in s
    fake_gate_records(out)
    assert RCF.main(base) == 0
    s = capsys.readouterr().out
    assert "production gate (prereg 1.3): SATISFIED" in s and "dry run" in s
    assert not os.path.exists(os.path.join(out, "shift_lam1"))
