"""Final campaign, Experiment A validation tooling (docs/final_campaign/PREREGISTRATION.md section 1.3; scripts/final_campaign/
expA_validation.py; the passive side_changes output added to src/gateway_family_validation.run_chain).

Inertness test of the prereg (before any G-A1 run):
  (a) short chains of each of the SEVEN OLD dynamics (EM and MALA, flat bias on and off, ACF recorder and traces on)
      through the harness at 189c2408 (git show, imported under another module name) and through the current harness:
      every old output array bitwise identical; the only new output is side_changes;
  (b) the settings, seed, trace-stride and ACF-plan functions of the Experiment A runner return, for every old
      (dynamics, chain, h, group), exactly what scripts/mechanism/run_validation.py (current AND 189c2408) and
      GF.acf_plan (current AND 189c2408) return; the runner's job function reproduces run_validation.run bitwise for an
      old dynamics on all five chains (smoke design);
  (c) the UNMODIFIED analyzer (scripts/mechanism/analyze_validation.py) run on results/mechanism/validation reproduces
      summary.json byte for byte apart from its generation timestamp (written to a temp dir; ~10 s);
  (d) run_validation.py --dry-run lists exactly the jobs the 189c2408 tree lists (subprocess on a reconstructed
      189c2408 tree vs the current tree), and with defaults reports the frozen gate complete.
Plus: side_changes == the label rule applied offline to per-step traces; the Experiment A seed rule == the config
strings, disjoint from every old seed; ACF horizon >= 6 t.u.; --out guards; signature-based resume / refusal; the V1 /
V4 Holm rules and the selection rule on synthetic inputs; the ABF-smoke cell configs and hook plumbing; an end-to-end
smoke run -> analyze -> summary.json / summary.md / A4 / A4b figures (scratch directories only).
"""
import csv
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import time
import types

import numpy as np
import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
import gateway_family_validation as GF  # noqa: E402

FROZEN_COMMIT = "189c2408"
FROZEN_RES = os.path.join(ROOT, "results", "mechanism", "validation")


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


EV = _load("expA_validation_under_test", os.path.join(ROOT, "scripts", "final_campaign", "expA_validation.py"))
RV = EV.RV
AV = EV.AV


@pytest.fixture(autouse=True)
def _ample_disk(monkeypatch):
    """Hermetic disk pause: the real root file system may be below the prereg's 2 GB (it is shared and nearly full), and
    then every in-process write path (drive_pool, collect_smoke, analyze) would legitimately PAUSE.  Tests that exercise
    the pause patch DISK_USAGE themselves (after this fixture, so theirs wins); subprocess tests patch it in a wrapper."""
    monkeypatch.setattr(EV.C, "DISK_USAGE", lambda path: types.SimpleNamespace(free=10 ** 12))


def _git_show(rev_path):
    return subprocess.check_output(["git", "-C", ROOT, "show", f"{FROZEN_COMMIT}:{rev_path}"])


@pytest.fixture(scope="module")
def old_tree(tmp_path_factory):
    """A reconstructed 189c2408 tree with the files the harness / run script need (harness, run script, configs)."""
    t = tmp_path_factory.mktemp("tree189c2408")
    for rel in ("src/gateway_family_validation.py", "scripts/mechanism/run_validation.py",
                "configs/mechanism/matched_free_energy.json", "configs/mechanism/conditional_relaxation.json"):
        p = t / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(_git_show(rel))
    return t


@pytest.fixture(scope="module")
def OLD(old_tree):
    """The harness module at 189c2408, imported under another module name."""
    return _load("gfv_189c2408", str(old_tree / "src" / "gateway_family_validation.py"))


@pytest.fixture(scope="module")
def RV_OLD(old_tree):
    return _load("rv_189c2408", str(old_tree / "scripts" / "mechanism" / "run_validation.py"))


# ------------------------------------------------------------------------------------------------- (a) bitwise
def test_a_old_dynamics_bitwise_identical_to_189c2408(OLD):
    assert GF.run_chain is not OLD.run_chain
    assert OLD.DYNAMICS_ORDER == GF.DYNAMICS_ORDER
    n_cases = 0
    for m in GF.load_dynamics():
        mo = OLD.load_dynamics()[GF.DYNAMICS_ORDER.index(m["name"])]
        assert json.dumps(mo, sort_keys=True) == json.dumps(m, sort_keys=True)
        for sch in (GF.EM, GF.MALA):
            for flat in (0, 1):
                g = np.random.default_rng(5)
                x0 = np.concatenate([[-1.0, -0.21, 0.0, 0.6, -0.6], g.uniform(-1.5, 1.5, 11)])
                y0 = GF.initial_y(x0, g.standard_normal(len(x0)), m)
                h = 2.5e-5 if sch == GF.EM else 1e-4
                plan = GF.acf_plan(m, h)
                kw = dict(trace_every=37, n_trace=5, n_acf=8, acf=plan, acf_rec=50)
                a = OLD.chain(mo, sch, x0, y0, 30000, 700, h, flat, 11, **kw)
                b = GF.chain(m, sch, x0, y0, 30000, 700, h, flat, 11, **kw)
                assert set(b) - set(a) == {"side_changes"} and set(a) <= set(b)
                bad = [k for k in a if not np.array_equal(np.asarray(a[k]), np.asarray(b[k]))]
                assert not bad, (m["name"], sch, flat, bad)
                assert b["side_changes"].dtype == np.int64 and b["side_changes"].shape == x0.shape
                n_cases += 1
    assert n_cases == 28


def test_a_raw_kernel_tuple_appended_at_the_end(OLD):
    m = GF.load_dynamics()[3]
    x0 = np.array([-0.6, 0.0, 0.7])
    y0 = GF.initial_y(x0, np.zeros(3), m)
    args = (GF.EM, x0, y0, 500, 10, 2.5e-5, 16.0, 0.5, 1.0, 32.0, 0.1, *GF.numba_args(m), 1, 1440, 180, 3, 10 ** 9, 1, 0,
            np.ones(2, np.int64), np.zeros(1, np.int64), np.zeros(1, np.int64), np.asarray(GF.ACF_WINDOWS), 0.02, 0)
    new = GF.run_chain(*args)
    old = OLD.run_chain(*args)
    assert len(new) == len(old) + 1
    for u, v in zip(old, new[:-1]):
        assert np.array_equal(np.asarray(u), np.asarray(v))


def _offline_side_changes(x0, traces, x_final, burn):
    """The label rule applied to the post-move positions: trace[k] = state after k moves (k = 0..n-1)."""
    post = np.vstack([traces[1:], x_final[None, :]])
    lab = np.where(x0 > 0.5, 1, np.where(x0 < -0.5, -1, 0))
    cnt = np.zeros(x0.size, np.int64)
    for k, row in enumerate(post):
        new = np.where(row > 0.5, 1, np.where(row < -0.5, -1, 0))
        flip = (new != 0) & (lab != 0) & (new != lab)
        if k >= burn:
            cnt += flip
        lab = np.where(new != 0, new, lab)
    return cnt


@pytest.mark.parametrize("variant,alpha,lam,h,n,burn,scheme", [
    ("alpha", 0.0, 1.0, 1e-3, 50000, 3000, 0), ("alpha", 0.0, 1.0, 1e-3, 50000, 3000, 1),
    ("shift", 1.0, 128.0, 1e-4, 30000, 2000, 0), ("shift", 1.0, 128.0, 1e-4, 30000, 2000, 1)])
def test_side_changes_equal_offline_label_rule(variant, alpha, lam, h, n, burn, scheme):
    m = GF.make_model(variant, alpha, 1.0, lam)
    N = 48
    g = np.random.default_rng(11)
    x0 = g.uniform(-1.5, 1.5, N)
    y0 = GF.initial_y(x0, g.standard_normal(N), m)
    r = GF.chain(m, scheme, x0, y0, n, burn, h, 1, 77, trace_every=1, n_trace=N)
    assert r["traces_x"].shape == (n, N)
    want = _offline_side_changes(x0, r["traces_x"], r["X"], burn)
    assert np.array_equal(r["side_changes"], want)
    if variant == "alpha":
        assert want.sum() > 30, "the test must exercise many side changes"


# ------------------------------------------------------------------------------------------------- (b) design functions
def test_b_settings_seeds_traces_plans_identical_for_old_dynamics(OLD, RV_OLD):
    dyns = GF.load_dynamics()
    n = 0
    for d, m in enumerate(dyns):
        for chain in RV.CHAINS:
            for smoke in (False, True):
                assert EV.settings(chain, m, smoke) == RV.settings(chain, m, smoke) == RV_OLD.settings(chain, m, smoke)
            for h in (*RV.H_CANDIDATES, RV.H_MALA):
                if chain in ("em_flat", "em_unbiased") and h == RV.H_MALA:
                    continue
                for g in range(RV.G):
                    hh = None if chain.startswith("exact") else h
                    assert EV.seed_of(chain, d, hh, g) == RV.seed_of(chain, d, hh, g) == RV_OLD.seed_of(chain, d, hh, g)
                    if chain in ("em_flat", "mala_flat"):
                        want = max(1, int(round(0.05 / (m["lam"] * h)))) if g == 0 else 10 ** 12    # RV.run's expression
                        assert EV.trace_every(chain, m, h, g) == want
                    elif chain == "em_unbiased":
                        assert EV.trace_every(chain, m, h, g) == max(1, int(round(0.05 / h)))
                    n += 1
        for h in (*RV.H_CANDIDATES, RV.H_MALA):
            p, q, r = EV.acf_plan(m, h), GF.acf_plan(m, h), OLD.acf_plan(m, h)
            assert set(p) == set(q) == set(r)
            for k in q:
                assert np.array_equal(np.asarray(p[k]), np.asarray(q[k])) and np.array_equal(np.asarray(q[k]), np.asarray(r[k])), k
            for key in ("lags0", "lags1"):
                assert np.array_equal(EV._lag_grid(int(q[key][-1])), q[key])
    assert n == 7 * (2 * 16 + 3 * 16 + 2 * 3 * 16 + 2 * 16)
    assert RV.jobs(list(GF.DYNAMICS_ORDER), [RV.H_CANDIDATES[0]], list(RV.CHAINS), 16, False) == \
        RV_OLD.jobs(list(GF.DYNAMICS_ORDER), [RV.H_CANDIDATES[0]], list(RV.CHAINS), 16, False)
    # the lam = 1 cell of the runner IS the old 'shift' dynamics with RV's design
    e = EV.BY_NAME["shift"]
    assert json.dumps(e["model"], sort_keys=True) == json.dumps(dyns[3], sort_keys=True)
    assert e["index"] == 3 and not e["new"]
    assert EV.out_path("/o", 3, "em_flat", 2.5e-5, 4) == RV.out_path("/o", "shift", "em_flat", 2.5e-5, 4)


def test_b_new_dynamics_design():
    assert EV.DYNAMICS_ORDER[7:] == ("shift_lam4", "shift_lam16", "shift_lam128")
    assert [EV.BY_NAME[n]["index"] for n in EV.NEW_NAMES] == [7, 8, 9]
    for name in EV.NEW_NAMES:
        m = EV.BY_NAME[name]["model"]
        assert m["variant"] == "shift" and m["kappa"] == 1.0 and m["lam"] > 1 and EV.is_new(m)
        assert all(m[k] == v for k, v in GF.PHYS.items())
        assert EV.settings("em_flat", m, False) == (500.0, 10.0)            # 500 / min(lam, 1), not 500 / lam
        assert EV.settings("em_flat", m, True) == (5.0, 1.0)
        assert EV.settings("mala_flat", m, False) == (500.0, 10.0) and EV.settings("em_unbiased", m, False) == (200.0, 0.0)
        for h in RV.H_CANDIDATES:
            assert EV.trace_every("em_flat", m, h, 0) == int(round(0.05 / h)) and EV.trace_every("em_flat", m, h, 3) == 10 ** 12
            p, q = EV.acf_plan(m, h), GF.acf_plan(m, h)
            assert np.array_equal(p["stride"], q["stride"])                   # same strides, longer horizon
            for lev, key in ((0, "lags0"), (1, "lags1")):
                assert p[key][-1] * p["stride"][lev] * h >= EV.ACF_HORIZON_T
                assert (p[key][-1] - 1) * p["stride"][lev] * h < EV.ACF_HORIZON_T + p["stride"][lev] * h
                assert np.array_equal(p[key][:17], np.arange(17))              # short-lag resolution unchanged
                assert p[key][-1] < 20000                                      # ring stays small
            assert p["lags0"][-1] * p["stride"][0] * h >= max(EV.MC1_CUTS)


def test_b_seed_rule_config_and_disjointness():
    EV._check_seed_rule()
    cs = _load("check_seeds_under_test", os.path.join(ROOT, "scripts", "final_campaign", "check_seeds.py"))
    new = set()
    for d in (7, 8, 9):
        for c in RV.CHAINS:
            for k, h in enumerate(RV.H_CANDIDATES):
                for g in range(16):
                    if c == "em_unbiased" and g:
                        continue
                    new.add(EV.seed_of(c, d, None if c.startswith("exact") else (RV.H_MALA if c == "mala_flat" else h), g))
    old = {RV.seed_of(c, d, None if c.startswith("exact") else (RV.H_MALA if c == "mala_flat" else h), g)
           for d in range(7) for c in RV.CHAINS for h in RV.H_CANDIDATES for g in range(16)}
    assert not new & old
    assert new <= set(cs.NEW_VALIDATION_RAW) and not new & set(cs.USED_VALIDATION_RAW)
    assert EV.seed_of("em_flat", 8, 1.25e-5, 5) == 401105 and EV.seed_of("mala_flat", 9, RV.H_MALA, 2) == 410202
    assert EV.seed_of("em_unbiased", 7, 1.25e-5, 0) == 411010 and EV.seed_of("exact_unbiased", 8, None, 15) == 421115
    with pytest.raises(ValueError):
        EV.seed_of("em_flat", 10, 2.5e-5, 0)


def test_b_runner_reproduces_run_validation_bitwise_for_old_dynamics(tmp_path):
    """EV.run_job == RV.run on the old 'shift' dynamics (index 3), smoke design, all five chains."""
    a, b = tmp_path / "rv", tmp_path / "ev"
    for j in (("em_flat", 3, 2.5e-5, [0]), ("em_unbiased", 3, 2.5e-5, [0]), ("mala_flat", 3, RV.H_MALA, [0]),
              ("exact_flat", 3, None, [0]), ("exact_unbiased", 3, None, [0])):
        RV.run(j, str(a), True)
        EV.run_job(j, str(b), True)
        pa, pb = RV.out_path(str(a), "shift", j[0], j[2], 0), RV.out_path(str(b), "shift", j[0], j[2], 0)
        with np.load(pa) as za, np.load(pb) as zb:
            ka = [k for k in za.files if k != "meta_json"]
            assert set(zb.files) - set(za.files) <= {"side_changes"}
            bad = [k for k in ka if not np.array_equal(za[k], zb[k])]
            assert not bad, (j[0], bad)
            ma, mb = json.loads(str(za["meta_json"])), json.loads(str(zb["meta_json"]))
        for k in ma:
            if k in ("commit", "host", "wall_s", "ns_per_walker_step", "prereg"):
                continue
            assert mb.get(k) == ma[k] or (k == "model" and json.dumps(mb[k], sort_keys=True) == json.dumps(ma[k], sort_keys=True)), k
        assert mb["signature"] == EV.job_signature(j[0], 3, j[2], 0, True)
        assert mb["provenance"]["harness_sha256"] == EV.sha256_file(EV.HARNESS)
        # the replay comparison (used for the lam = 1 G-A2 replay) says equal here, and catches a one-ulp change
        with np.load(pb) as zb:
            arrays = {k: zb[k] for k in zb.files if k != "meta_json"}
        rc = EV.replay_compare(arrays, mb, pa)
        assert rc["bitwise_equal"] and rc["keys_compared"] == ka and not rc["mismatched"]
        k0 = "A1" if "A1" in arrays else ka[0]
        arrays[k0] = np.nextafter(arrays[k0], np.inf)
        assert EV.replay_compare(arrays, mb, pa)["mismatched"] == [k0]


# ------------------------------------------------------------------------------------------------- (c) analyzer
@pytest.mark.skipif(not os.path.exists(os.path.join(FROZEN_RES, "summary.json")), reason="frozen gate record absent")
def test_c_unmodified_analyzer_reproduces_frozen_summary(tmp_path):
    S = AV.analyse_all(FROZEN_RES, os.path.join(FROZEN_RES, "abf_smoke.json"), False)
    out = tmp_path / "summary.json"
    with open(out, "w") as f:
        json.dump(S, f, indent=1, default=AV._json_default)
    norm = lambda s: re.sub(r'"generated_utc": "[^"]*"', '"generated_utc": ""', s)  # noqa: E731
    new, old = out.read_text(), open(os.path.join(FROZEN_RES, "summary.json")).read()
    assert norm(new) == norm(old)
    assert len(re.findall(r'"generated_utc"', new)) == 1


# ------------------------------------------------------------------------------------------------- (d) dry-run
def _run(script, args, cwd):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", NUMBA_NUM_THREADS="1")
    return subprocess.run([sys.executable, str(script)] + args, cwd=str(cwd), env=env, capture_output=True, text=True,
                          timeout=300)


def test_d_run_validation_dry_run_unchanged(old_tree, tmp_path):
    out = tmp_path / "empty_out"
    for hs in (["--h", "2.5e-5"], ["--h", "2.5e-5", "1.25e-5"], []):
        a = _run(old_tree / "scripts" / "mechanism" / "run_validation.py", ["--dry-run", "--out", str(out)] + hs, tmp_path)
        b = _run(os.path.join(ROOT, "scripts", "mechanism", "run_validation.py"), ["--dry-run", "--out", str(out)] + hs,
                 tmp_path)
        assert a.returncode == 0 and b.returncode == 0, (a.stderr[-500:], b.stderr[-500:])
        assert a.stdout == b.stdout
        lines = a.stdout.strip().splitlines()
        nh = max(1, len(hs) - 1)
        assert len(lines) == 1 + 7 * (16 * nh + nh + 16 + 2)                 # em_flat, em_unbiased, mala, 2 exact
        assert not os.path.exists(out)
    c = _run(os.path.join(ROOT, "scripts", "mechanism", "run_validation.py"), ["--dry-run"], tmp_path)
    assert c.returncode == 0 and c.stdout.startswith("0 jobs pending -> " + FROZEN_RES)


# ------------------------------------------------------------------------------------------------- runner guards, jobs
def test_out_guards_and_default_jobs(tmp_path):
    with pytest.raises(SystemExit, match="frozen"):
        EV.main(["run", "--out", FROZEN_RES, "--dry-run"])
    with pytest.raises(SystemExit, match="frozen"):
        EV.main(["run", "--out", os.path.join(FROZEN_RES, "shift"), "--smoke", "--dry-run"])
    with pytest.raises(SystemExit, match="accepted only"):
        EV.main(["run", "--out", str(tmp_path), "--dry-run"])
    with pytest.raises(SystemExit, match="outside"):
        EV.main(["run", "--out", EV.OUT_EXPA, "--smoke", "--dry-run"])
    with pytest.raises(SystemExit, match="16 groups"):
        EV.main(["run", "--groups", "3", "--dry-run"])
    with pytest.raises(SystemExit, match="candidate"):
        EV.main(["run", "--h", "3e-5", "--dry-run"])
    J = EV.jobs(list(EV.ALL_NAMES), [2.5e-5], list(EV.CHAINS), 16, False)
    by = {}
    for c, d, h, gs in J:
        by.setdefault((EV.BY_INDEX[d]["name"], c), []).append((h, tuple(gs)))
    assert len(J) == 16 + 3 * 35
    assert set(k for k in by if k[0] == "shift") == {("shift", "em_flat")}           # lam = 1: the em_flat replay only
    for name in EV.NEW_NAMES:
        assert len(by[(name, "em_flat")]) == 16 and len(by[(name, "mala_flat")]) == 16
        assert by[(name, "em_unbiased")] == [(2.5e-5, (0,))] and by[(name, "exact_flat")] == [(None, tuple(range(16)))]
    J2 = EV.jobs(["shift"], [1.25e-5], list(EV.CHAINS), 16, False)                   # refinement: lam = 1 re-gate
    assert {c for c, *_ in J2} == {"em_flat", "em_unbiased"} and len(J2) == 17
    assert EV.main(["dry-run"]) == 0


def test_resume_and_stale_refusal(tmp_path, capsys):
    out = tmp_path / "smoke_out"
    j = ("em_unbiased", 9, 2.5e-5, [0])
    assert EV.job_states(j, str(out), True) == {0: "missing"}
    EV.run_job(j, str(out), True)
    assert EV.job_states(j, str(out), True) == {0: "complete"}
    assert EV.job_states(j, str(out), False) == {0: "stale"}                          # other design -> refused
    rc = EV.main(["run", "--smoke", "--out", str(out), "--dynamics", "shift_lam128", "--chains", "em_unbiased"])
    assert rc == 0 and "nothing to run" in capsys.readouterr().out
    p = EV.out_path(str(out), 9, "em_unbiased", 2.5e-5, 0)
    with np.load(p) as z:
        arrays = {k: z[k] for k in z.files if k != "meta_json"}
        meta = json.loads(str(z["meta_json"]))
    meta["signature"]["seed"] += 1
    RV._save(p, meta, arrays)
    before = open(p, "rb").read()
    assert EV.job_states(j, str(out), True) == {0: "stale"}
    rc = EV.main(["run", "--smoke", "--out", str(out), "--dynamics", "shift_lam128", "--chains", "em_unbiased"])
    assert rc == 1 and "STALE" in capsys.readouterr().out
    assert open(p, "rb").read() == before


# ------------------------------------------------------------------------------------------------- analysis rules
def test_v1_holm_family_and_gate_rewrite():
    def fam_of(zz):
        return [dict(dynamics=n, index=j, z=z, dof=15) for n, zs in zip(EV.NEW_NAMES, zz) for j, z in enumerate(zs)]
    # |z| = 3.01 fails the old per-clause |z| <= 3 rule but not Holm over the 6 V1 clauses (p 0.0088 > 0.05/6)
    fam = EV.v1_holm(fam_of(([0.3, -1.2], [3.01, 0.5], [2.2, -0.1])))
    assert not any(f["holm_rejected"] for f in fam) and len(fam) == 6
    assert 0.05 / 6 < AV.t_two_sided(3.01, 15) < 0.05 / 5
    # step-down: once 6.5 is rejected, 3.2 (p 0.006 < 0.05/5) is rejected too
    fam = EV.v1_holm(fam_of(([0.3, -1.2], [3.2, 0.5], [6.5, -0.1])))
    assert {(f["dynamics"], f["index"]) for f in fam if f["holm_rejected"]} == {("shift_lam16", 0), ("shift_lam128", 0)}
    assert all(abs(f["p"] - AV.t_two_sided(f["z"], 15)) < 1e-15 for f in fam)
    assert EV.v1_holm(fam_of(([float("nan"), 0.1], [], []))) == [dict(dynamics="shift_lam4", index=1, z=0.1, dof=15,
                                                                    p=AV.t_two_sided(0.1, 15), holm_rejected=False)]
    g = dict(V1="FAIL", V1_components=dict(closed_form=True, central_le_2pct_plus_2se=[True, True], shift_mean_3se=[False, True],
                                           shift_var_region_2pct=True), V1_consistency_3se=[True, True])
    g = EV.apply_v1_holm(g, set(), [3.2, 0.5])
    assert g["V1"] == "PASS" and g["V1_unadjusted_3se"] == "FAIL" and "shift_mean_3se" not in g["V1_components_holm"]
    g2 = dict(V1="PASS", V1_components=dict(closed_form=True, central_le_2pct_plus_2se=[True, False], shift_mean_3se=[True, True],
                                            shift_var_region_2pct=True), V1_consistency_3se=[True, True])
    assert EV.apply_v1_holm(g2, set(), [0.1, 0.2])["V1"] == "FAIL"           # tolerance clauses unchanged
    g3 = dict(V1="FAIL", V1_components=dict(closed_form=True, central_le_2pct_plus_2se=[True, True], shift_mean_3se=[False, True],
                                            shift_var_region_2pct=True), V1_consistency_3se=[True, True])
    assert EV.apply_v1_holm(g3, {0}, [6.5, 0.2])["V1"] == "FAIL"


def _tab(st25, st125=None):
    t = {}
    for n in EV.ALL_NAMES:
        t[n] = {"2.5e-05": dict(core=st25.get(n, "PASS"), full=st25.get(n, "PASS"), core_strict=st25.get(n, "PASS"),
                                full_strict=st25.get(n, "PASS"))}
        if st125 is not None:
            s = st125.get(n, "PASS")
            t[n]["1.25e-05"] = dict(core=s, full=s, core_strict=s, full_strict=s)
    return t


def test_selection_rule():
    v4 = {n: "PASS" for n in EV.ALL_NAMES}
    s = EV.select(_tab({}), v4)
    assert s["verdict"] == "PASS" and s["h_A"] == 2.5e-5 and s["B"] == 2048 * 40 / 2.5e-5 and s["T_N"]["128"] == 640.0
    s = EV.select(_tab({"shift_lam128": "FAIL"}), v4)
    assert s["verdict"].startswith("REFINE_REQUIRED") and s["h_A"] is None and s["full"]["next_h"] == 1.25e-5
    s = EV.select(_tab({"shift_lam16": "FAIL"}, {}), v4)
    assert s["verdict"] == "PASS" and s["h_A"] == 1.25e-5 and s["B"] == 2048 * 40 / 1.25e-5
    s = EV.select(_tab({"shift_lam16": "FAIL"}, {"shift": "FAIL"}), v4)
    assert s["verdict"].startswith("UNRESOLVED") and "Experiment A stops" in s["verdict"] and s["h_A"] is None
    s = EV.select(_tab({}), dict(v4, shift_lam4="FAIL"))
    assert s["verdict"].startswith("HARNESS_FAIL") and s["h_A"] is None
    t = _tab({})
    del t["shift_lam4"]["2.5e-05"]
    assert EV.select(t, v4)["verdict"].startswith("INCOMPLETE")
    t = _tab({})
    t["shift_lam4"]["2.5e-05"]["full"] = "PENDING"
    assert EV.select(t, v4)["verdict"].startswith("PROVISIONAL")


def test_ratio_ci_and_jk_ci():
    r = EV.ratio_ci(dict(est=2.0, se=0.2), dict(est=1.0, se=0.05), 15)
    assert abs(r["est"] - 2.0) < 1e-12 and r["ci95"][0] < 2.0 < r["ci95"][1]
    assert abs(r["se_log"] - math.sqrt(0.1 ** 2 + 0.05 ** 2)) < 1e-12
    assert math.isnan(EV.ratio_ci(dict(est=float("nan"), se=1), dict(est=1, se=1), 15)["est"])
    lo, hi = EV.jk_ci(1.0, 0.1, 15)
    assert abs((hi - lo) / 2 - 2.131449546 * 0.1) < 1e-6


# ------------------------------------------------------------------------------------------------- ABF smoke hook
def test_abf_smoke_cells_and_cfg(tmp_path, monkeypatch):
    J = EV.smoke_jobs(2.5e-5)
    assert len(J) == 12 and {j["variant"] for j in J} == set(EV.NEW_NAMES)
    assert {j["seed"] for j in J} == {8392, 8393} and {j["n_steps"] for j in J} == {160000} and {j["N"] for j in J} == {2048}
    J2 = EV.smoke_jobs(1.25e-5)
    assert len(J2) == 16 and "shift_lam1" in {j["variant"] for j in J2} and {j["n_steps"] for j in J2} == {320000}
    for j in J:
        assert j["cfg"]["lam"] == EV.lam_of(j["variant"]) and j["cfg"]["h"] == 2.5e-5 and j["cfg"]["variant"] == "shift"
    # a cell config that disagrees with expA.json is refused
    d = tmp_path / "cells"
    d.mkdir()
    bad = dict(engine_cfg=dict(json.load(open(EV.CFG_EXPA))["engine_cfg"], gamma=2.0), model=dict(variant="shift", alpha=None,
                                                                                                  kappa=1.0, lam=16.0))
    (d / "shift_lam16.json").write_text(json.dumps(bad))
    monkeypatch.setattr(EV, "CELL_DIR", str(d))
    with pytest.raises(SystemExit, match="gamma"):
        EV.cell_cfg("shift_lam16", 2.5e-5)
    cfg, src = EV.cell_cfg("shift_lam4", 2.5e-5)                     # absent there -> built from expA.json
    assert src == "configs/final_campaign/expA.json" and cfg["lam"] == 4.0


def test_abf_smoke_collect_hook_plumbing(tmp_path, monkeypatch):
    """collect_smoke files a cell only when all 4 runs are COMPLETE results of the expected job (run signature incl.
    engine sha256); a partial set, a stale file of another run or an emptied cell removes the entry (V5 PENDING)."""
    import gateway_family_numba as E
    monkeypatch.setattr(EV, "SMOKE_N0", 8)
    monkeypatch.setattr(EV, "SMOKE_N", 8)
    monkeypatch.setattr(EV, "SMOKE_T", 400 * 2.5e-5)                 # B = 8 x 400 -> n_steps 400 at N 8
    out = tmp_path / "o"
    cfg, _ = EV.cell_cfg("shift_lam4", 2.5e-5)
    assert {j["n_steps"] for j in EV.smoke_jobs(2.5e-5)} == {400}
    root = EV.smoke_root(str(out), 2.5e-5)
    runs = [(8392, "abf"), (8393, "fr"), (8392, "fr"), (8393, "abf")]
    for k, (s, method) in enumerate(runs):
        p = os.path.join(root, "shift_lam4", "N8", f"s{s}_{method}.npz")
        E.run_job(cfg, method, 8, s, 400, p)
        if k == 1:                                                # a partial set is not filed (V5 stays PENDING)
            hook, issues = EV.collect_smoke(str(out), 2.5e-5)
            assert "shift_lam4" not in hook and not issues
    hook, issues = EV.collect_smoke(str(out), 2.5e-5)
    assert not issues
    recs = hook["shift_lam4"]["2.5e-05"]
    assert len(recs) == 4 and {(r["seed"], r["method"]) for r in recs} == set(runs) and all(r["finite"] for r in recs)
    assert all(len(r["sha256"]) == 64 for r in recs)
    back = AV.load_abf_smoke(str(out / "abf_smoke.json"))
    assert list(back) == ["shift_lam4"] and len(back["shift_lam4"]["2.5e-05"]) == 4
    assert AV.abf_smoke_gate(recs) in ("PASS", "FAIL")
    # an unexpected file in a cell directory is ignored and reported; the filed cell stays filed
    extra = os.path.join(root, "shift_lam4", "N8", "s9999_abf.npz")
    os.link(os.path.join(root, "shift_lam4", "N8", "s8392_abf.npz"), extra)
    hook, issues = EV.collect_smoke(str(out), 2.5e-5)
    assert len(hook["shift_lam4"]["2.5e-05"]) == 4 and len(issues) == 1 and "not a preregistered" in issues[0]
    os.remove(extra)
    # a complete result of a DIFFERENT run at a smoke path (here: other n_steps) is STALE: never filed, the earlier
    # entry is REMOVED (V5 reads PENDING, never the other run's record), the file is not touched
    p = os.path.join(root, "shift_lam4", "N8", "s8393_abf.npz")
    os.remove(p)
    E.run_job(cfg, "abf", 8, 8393, 800, p)
    before = open(p, "rb").read()
    hook, issues = EV.collect_smoke(str(out), 2.5e-5)
    assert "shift_lam4" not in hook and len(issues) == 1 and "STALE" in issues[0]
    assert "shift_lam4" not in AV.load_abf_smoke(str(out / "abf_smoke.json"))
    assert open(p, "rb").read() == before
    # ... so is a result of the frozen design run on another engine (the engine sha256 is part of the signature)
    os.remove(p)
    E.run_job(cfg, "abf", 8, 8393, 400, p)
    assert EV.collect_smoke(str(out), 2.5e-5)[0]["shift_lam4"]["2.5e-05"]
    sig = EV.C.rc().job_signature(E, next(j for j in EV.smoke_jobs(2.5e-5) if j["variant"] == "shift_lam4"
                                          and j["seed"] == 8393 and j["method"] == "abf"))
    monkeypatch.setattr(EV.C.rc(), "job_signature", lambda E_, job: dict(sig, engine_sha="0" * 64))
    hook, issues = EV.collect_smoke(str(out), 2.5e-5)
    assert "shift_lam4" not in hook and len(issues) == 4 and all("STALE" in i for i in issues)
    monkeypatch.undo()
    # a cell whose files were all moved away loses its old entry too
    monkeypatch.setattr(EV, "SMOKE_N0", 8)
    monkeypatch.setattr(EV, "SMOKE_N", 8)
    monkeypatch.setattr(EV, "SMOKE_T", 400 * 2.5e-5)
    monkeypatch.setattr(EV.C, "DISK_USAGE", lambda p_: types.SimpleNamespace(free=10 ** 12))
    assert EV.collect_smoke(str(out), 2.5e-5)[0]["shift_lam4"]["2.5e-05"]
    import shutil
    shutil.move(os.path.join(root, "shift_lam4"), str(tmp_path / "moved"))
    hook, issues = EV.collect_smoke(str(out), 2.5e-5)
    assert "shift_lam4" not in hook and not issues


def test_abf_smoke_worker_failure_is_ledgered_finite_and_not_filed(tmp_path, monkeypatch, capsys):
    """A worker that dies (exception out of the pool worker) is a FAILED ledger row with a FINITE wall time (the
    production driver's projection gate sums wall_s; NaN would silently disable it), its cell is not filed (V5 PENDING),
    the other runs complete and are filed, exit 1.  (Thread pool injected; the process-pool path is the CLI test.)"""
    import functools
    from concurrent.futures import ThreadPoolExecutor
    monkeypatch.setattr(EV, "SMOKE_N0", 8)
    monkeypatch.setattr(EV, "SMOKE_N", 8)
    monkeypatch.setattr(EV, "SMOKE_T", 400 * 2.5e-5)
    monkeypatch.setattr(EV, "drive_pool", functools.partial(EV.drive_pool, executor_factory=lambda: ThreadPoolExecutor(2)))
    real = EV.C.run_timed

    def flaky(job, root, expected_engine=None):
        if job["variant"] == "shift_lam16" and job["seed"] == 8393 and job["method"] == "fr":
            raise RuntimeError("worker died")
        return real(job, root, expected_engine)
    monkeypatch.setattr(EV.C, "run_timed", flaky)
    out = tmp_path / "o"
    assert EV.main(["abf-smoke", "--smoke", "--out", str(out), "--workers", "2"]) == 1
    rows = list(csv.DictReader(open(out / "ledger.csv")))
    assert len(rows) == 12 and all(math.isfinite(float(rw["wall_s"])) for rw in rows)
    bad = [rw for rw in rows if rw["status"].startswith("FAILED")]
    assert len(bad) == 1 and bad[0]["dynamics"] == "shift_lam16" and "worker died" in bad[0]["status"]
    hook = json.load(open(out / "abf_smoke.json"))
    assert set(hook) == {"shift_lam4", "shift_lam128"}
    assert "shift_lam16" in capsys.readouterr().out


def test_abf_smoke_refuses_a_non_frozen_engine(tmp_path, monkeypatch, capsys):
    out = tmp_path / "o"
    assert EV.main(["abf-smoke", "--smoke", "--out", str(out), "--dry-run"]) == 0
    assert "12 runs" in capsys.readouterr().out
    monkeypatch.setattr(EV.C, "EXPECTED_ENGINE_SHA256", "0" * 64)
    assert EV.main(["abf-smoke", "--smoke", "--out", str(out), "--dry-run"]) == 2
    assert "REFUSED" in capsys.readouterr().out
    assert EV.main(["abf-smoke", "--smoke", "--out", str(out)]) == 2
    assert not os.path.exists(out / "abf_smoke")


# ------------------------------------------------------------------------------------------------- pool, disk pause
class _FakeDisk:
    """shutil.disk_usage stand-in: free bytes from a list (last value repeated); counts the calls."""
    def __init__(self, frees):
        self.frees, self.n = list(frees), 0

    def __call__(self, path):
        v = self.frees[min(self.n, len(self.frees) - 1)]
        self.n += 1
        return types.SimpleNamespace(free=v)


def test_drive_pool_lazy_submission_and_disk_pause(monkeypatch):
    """At most `workers` jobs in flight; the disk is checked before EVERY job start; below 2 GB nothing starts (the
    finished jobs keep being collected) until >= 3 GB; each poll is logged."""
    import threading
    from concurrent.futures import ThreadPoolExecutor
    lock, inflight, peak, started, log = threading.Lock(), [0], [0], [], []
    disk = _FakeDisk([5e9, 5e9, 1.5e9, 2.5e9, 2.9e9, 3.0e9] + [5e9] * 20)
    monkeypatch.setattr(EV.C, "DISK_USAGE", disk)
    monkeypatch.setattr(EV.C, "DISK_POLL_S", 0.01)

    def work(i):
        with lock:
            inflight[0] += 1
            peak[0] = max(peak[0], inflight[0])
            started.append((i, disk.n))
        time.sleep(0.05)
        with lock:
            inflight[0] -= 1
        return i * i

    done, subs = [], []

    def submit(ex, i):
        subs.append((i, disk.n))                          # submission order + disk checks made before it
        return ex.submit(work, i)
    st = EV.drive_pool(list(range(7)), submit, lambda i, f: done.append((i, f.result())), 2,
                       log=log.append, executor_factory=lambda: ThreadPoolExecutor(4))
    assert st == dict(n_items=7, n_started=7, n_done=7, stopped=None)
    assert sorted(done) == [(i, i * i) for i in range(7)]
    assert peak[0] <= 2                                   # lazy: never more than `workers` in flight (pool has 4)
    assert disk.n >= 7 + 3                                # one check per start + the pause polls
    assert [i for i, _ in subs] == list(range(7))         # submitted in queue order (thread START order may differ)
    assert [n for _, n in subs][:2] == [1, 2]             # a disk check immediately before each start
    assert subs[2][1] == 6                                # job 2 submitted only after the disk came back to >= 3 GB
    assert sorted(i for i, _ in started) == list(range(7))
    assert any("DISK PAUSE" in m for m in log) and any("DISK RESUME" in m for m in log)
    assert sum("disk pause continues" in m for m in log) == 2


def test_drive_pool_stops_on_pause_timeout_and_on_ledger_failure(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    monkeypatch.setattr(EV.C, "DISK_POLL_S", 0.0)
    monkeypatch.setattr(EV.C, "DISK_MAX_PAUSE_S", 0.0)
    monkeypatch.setattr(EV.C, "DISK_USAGE", _FakeDisk([5e9, 1e9]))
    done, log = [], []
    st = EV.drive_pool(list(range(5)), lambda ex, i: ex.submit(time.sleep, 0.02), lambda i, f: done.append(i), 2,
                       log=log.append, executor_factory=lambda: ThreadPoolExecutor(2))
    assert st["n_started"] == 1 and st["n_done"] == 1 and done == [0] and "disk pause" in st["stopped"]
    assert any("DISK PAUSE TIMEOUT" in m for m in log)
    # recording a finished job fails (e.g. ENOSPC on the ledger): nothing further starts, running jobs are collected
    monkeypatch.setattr(EV.C, "DISK_USAGE", _FakeDisk([5e9]))
    seen = []

    def on_done(i, f):
        seen.append(i)
        if i == 0:
            raise OSError(28, "No space left on device")
    st = EV.drive_pool(list(range(6)), lambda ex, i: ex.submit(time.sleep, 0.05 if i == 0 else 0.2), on_done, 2,
                       log=log.append, executor_factory=lambda: ThreadPoolExecutor(2))
    assert "No space left" in st["stopped"] and st["n_started"] == 2 and sorted(seen) == [0, 1]


def _cli_env():
    return dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", NUMBA_NUM_THREADS="1",
                PYTHONPATH=os.path.join(ROOT, "src"))


RUN_WRAPPER = """import os, runpy, sys, types
ROOT = {root!r}
sys.path.insert(0, os.path.join(ROOT, "scripts", "final_campaign"))
import expA_run_common as C  # noqa: E402  (the module the script imports by name: the patch below is what it sees)
_n = [0]


def _usage(path):
    _n[0] += 1
    return types.SimpleNamespace(free={free_first!r} if _n[0] <= 2 else 5.0e9)


C.DISK_USAGE, C.DISK_POLL_S = _usage, 0.05
if __name__ == "__main__":
    # run the REAL script as __main__ (so the pool worker is pickled as __main__.run_job, exactly as on the command line)
    sys.argv = [os.path.join(ROOT, "scripts", "final_campaign", "expA_validation.py")] + sys.argv[1:]
    runpy.run_path(sys.argv[0], run_name="__main__")
"""


def test_cli_run_through_the_default_start_method_pool(tmp_path):
    """`run` through the real script run as __main__ (pool workers = __main__.run_job, the interpreter's DEFAULT start
    method, forkserver on Python 3.14) with the disk pause patched in a wrapper (the real disk may be below 2 GB): a
    pause before the first start is logged, then two tiny smoke jobs complete, are ledgered, and a re-run skips them."""
    import multiprocessing
    out = tmp_path / "o"
    w = tmp_path / "run_wrapper.py"
    w.write_text(RUN_WRAPPER.format(root=ROOT, free_first=1.0e9))
    args = [sys.executable, str(w), "run", "--smoke", "--out", str(out), "--dynamics", "shift_lam128", "--chains",
            "em_unbiased", "exact_unbiased", "--workers", "2"]
    r = subprocess.run(args, cwd=str(tmp_path), env=_cli_env(), capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, (r.stdout[-2000:], r.stderr[-2000:])
    assert multiprocessing.get_start_method() in ("forkserver", "fork", "spawn")
    assert "ModuleNotFoundError" not in r.stderr and "BrokenProcessPool" not in r.stderr
    for c in ("em_unbiased", "exact_unbiased"):
        h = 2.5e-5 if c == "em_unbiased" else None
        assert EV.job_states((c, 9, h, [0]), str(out), True) == {0: "complete"}
    rows = list(csv.DictReader(open(out / "ledger.csv")))
    assert len(rows) == 2 and all(rw["status"] == "complete" for rw in rows)
    assert all(math.isfinite(float(rw["wall_s"])) for rw in rows)
    log = (out / "disk_pause.log").read_text()
    assert "DISK PAUSE" in log and "DISK RESUME" in log
    r2 = subprocess.run(args, cwd=str(tmp_path), env=_cli_env(), capture_output=True, text=True, timeout=300)
    assert r2.returncode == 0 and "nothing to run" in r2.stdout


def test_cli_run_disk_pause_timeout_starts_nothing(tmp_path):
    """Disk below 2 GB for longer than the (patched) maximum pause: no job starts, nothing is written but the pause log,
    exit 3 (prereg section 3 stopping rule (iv))."""
    out = tmp_path / "o"
    w = tmp_path / "run_wrapper.py"
    w.write_text(RUN_WRAPPER.format(root=ROOT, free_first=1.0e9).replace(
        "C.DISK_USAGE, C.DISK_POLL_S = _usage, 0.05",
        "C.DISK_USAGE, C.DISK_POLL_S, C.DISK_MAX_PAUSE_S = (lambda p: types.SimpleNamespace(free=1.0e9)), 0.01, 0.05"))
    r = subprocess.run([sys.executable, str(w), "run", "--smoke", "--out", str(out), "--dynamics", "shift_lam128",
                        "--chains", "em_unbiased", "--workers", "1"], cwd=str(tmp_path), env=_cli_env(),
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 3, (r.stdout[-2000:], r.stderr[-2000:])
    assert "DISK PAUSE TIMEOUT" in (out / "disk_pause.log").read_text()
    assert not os.path.exists(out / "ledger.csv")
    assert EV.job_states(("em_unbiased", 9, 2.5e-5, [0]), str(out), True) == {0: "missing"}


WRAPPER = """import os, sys, types
ROOT = {root!r}
sys.path.insert(0, os.path.join(ROOT, "scripts", "final_campaign"))
import expA_validation as EV  # noqa: E402
import expA_run_common as C  # noqa: E402
EV.SMOKE_N0, EV.SMOKE_N, EV.SMOKE_T = 8, 8, 400 * 2.5e-5          # tiny smoke design: N 8, 400 steps
_n = [0]


def _usage(path):
    _n[0] += 1
    return types.SimpleNamespace(free=1.0e9 if _n[0] <= 2 else 5.0e9)   # a disk pause before the first start


C.DISK_USAGE, C.DISK_POLL_S = _usage, 0.05
if __name__ == "__main__":
    sys.exit(EV.main(sys.argv[1:]))
"""


def test_cli_abf_smoke_through_the_default_start_method_pool(tmp_path):
    """`abf-smoke` through a process pool under the interpreter's DEFAULT start method (forkserver on Python 3.14),
    tiny design (N 8, 400 steps; patched in a wrapper __main__): all 12 runs complete via expA_run_common.run_timed,
    the disk pause before the first start is logged, ledger + provenance rows are written and the hook is filed."""
    w = tmp_path / "wrapper.py"
    w.write_text(WRAPPER.format(root=ROOT))
    out = tmp_path / "o"
    r = subprocess.run([sys.executable, str(w), "abf-smoke", "--smoke", "--out", str(out), "--workers", "2"],
                       cwd=str(tmp_path), env=_cli_env(), capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, (r.stdout[-3000:], r.stderr[-3000:])
    assert "ModuleNotFoundError" not in r.stderr and "BrokenProcessPool" not in r.stderr
    rows = list(csv.DictReader(open(out / "ledger.csv")))
    assert len(rows) == 12 and all(rw["status"] == "complete" and rw["chain"] == "abf_smoke" for rw in rows)
    assert all(int(float(rw["walker_steps"])) == 8 * 400 for rw in rows)
    prov_path = os.path.join(str(out), "abf_smoke", "h2.5e-05", f"T{400 * 2.5e-5:g}", "run_provenance.csv")  # wrapper's T
    prov = list(csv.DictReader(open(prov_path)))
    assert len(prov) == 12 and {p["engine_sha256"] for p in prov} == {EV.C.EXPECTED_ENGINE_SHA256}
    log = (out / "disk_pause.log").read_text()
    assert "DISK PAUSE" in log and "DISK RESUME" in log
    hook = json.load(open(out / "abf_smoke.json"))
    assert set(hook) == set(EV.NEW_NAMES) and all(len(hook[n]["2.5e-05"]) == 4 for n in EV.NEW_NAMES)
    r2 = subprocess.run([sys.executable, str(w), "abf-smoke", "--smoke", "--out", str(out), "--workers", "2"],
                        cwd=str(tmp_path), env=_cli_env(), capture_output=True, text=True, timeout=300)
    assert r2.returncode == 0 and "12 complete, 0 to run" in r2.stdout
    assert len(list(csv.DictReader(open(out / "ledger.csv")))) == 12


# ------------------------------------------------------------------------------------------------- lam = 1 MC inputs
@pytest.mark.skipif(not os.path.exists(os.path.join(FROZEN_RES, "shift", "em_flat_h2.5e-05")), reason="frozen record absent")
def test_lam1_replay_inertness_and_mc2_exclusion(monkeypatch):
    """lam1_inputs / apply_inertness on the frozen G-A2 em_flat records and fabricated replays: PASS only with all 16
    design-conforming, bitwise-equal files; a non-bitwise file -> FAIL -> HARNESS_FAIL; partial / non-conforming sets ->
    INCOMPLETE and MC2 at lam = 1 (and its ratios) excluded; no replay -> NOT_RUN."""
    import copy
    real_load = AV.load
    frozen = real_load(FROZEN_RES, "shift", "em_flat_h2.5e-05")
    assert len(frozen) == 16
    g = np.random.default_rng(1)

    def replay(n=16, bad_group=None, side=True, cmp_=True):
        out = []
        for r in frozen[:n]:
            q = copy.copy(r)
            q["meta"] = copy.deepcopy(r["meta"])
            if side:
                q["side_changes"] = g.poisson(2.0, r["X_final"].shape).astype(np.int32)
            if cmp_:
                q["meta"]["replay"] = dict(bitwise_equal=int(q["meta"]["group"]) != bad_group,
                                           mismatched=["A1"] if int(q["meta"]["group"]) == bad_group else [],
                                           meta_mismatched=[])
            out.append(q)
        return out

    def run(rep):
        monkeypatch.setattr(AV, "load", lambda res, dyn, tag: (rep if res == "RES" else real_load(res, dyn, tag)))
        return EV.lam1_inputs("RES", FROZEN_RES, 2.5e-5, False)

    L = run(replay())
    assert L["inertness_replay"]["status"] == "PASS" and L["mc2_src"] is not None and L["mc1_src"] is not None
    assert L["mc1_source"].endswith(os.path.join("validation", "shift", "em_flat_h2.5e-05")) and not L["design"]
    L = run(replay(bad_group=5))
    assert L["inertness_replay"]["status"] == "FAIL" and L["inertness_replay"]["mismatches"] == {5: (["A1"], [])}
    assert L["mc2_src"] is None
    sel = EV.select(_tab({}), {n: "PASS" for n in EV.ALL_NAMES})
    assert sel["h_A"] == 2.5e-5
    EV.apply_inertness(sel, {"2.5e-05": L["inertness_replay"]})
    assert sel["verdict"].startswith("HARNESS_FAIL (inertness_replay FAIL") and sel["h_A"] is None and "B" not in sel
    assert sel["inertness_replay"] == {"2.5e-05": "FAIL"} and sel["verdict_before_inertness"] == "PASS"
    for rep, why in ((replay(n=15), "groups"), (replay(side=False), "side_changes"), (replay(cmp_=False), None)):
        L = run(rep)
        assert L["inertness_replay"]["status"] == "INCOMPLETE" and L["mc2_src"] is None, why
        assert L["mc1_src"] is not None and any("EXCLUDED" in n for n in L["notes"])
        if why:
            assert any(why in i for i in L["design"]["em_flat_h2.5e-05"])
        sel = EV.select(_tab({}), {n: "PASS" for n in EV.ALL_NAMES})
        EV.apply_inertness(sel, {"2.5e-05": L["inertness_replay"]})
        assert sel["verdict"] == "PASS" and any("INCOMPLETE" in n for n in sel["notes"])
    L = run([])
    assert L["inertness_replay"]["status"] == "NOT_RUN" and L["mc2_src"] is None and L["mc2_trace_only_src"]
    # manipulation_checks: the lam = 1 MC2 rate (and the ratio of a new dynamics to it) only from a PASS replay
    rate = dict(est=0.01, se=0.001, ci95=[0.008, 0.012])
    S = dict(selection={}, dynamics={"shift_lam4": dict(per_h={"2.5e-05": dict(mc1=None, mc2=dict(rate=rate,
                                                                                                 n_groups=16))})})
    for rep, has in ((replay(), True), (replay(n=15), False), (replay(bad_group=0), False)):
        monkeypatch.setattr(AV, "load", lambda res, dyn, tag, rep=rep: (rep if res == "RES" else real_load(res, dyn, tag)))
        mc = EV.manipulation_checks(S, "RES", False, frozen_res=FROZEN_RES)
        row1 = mc["per_h"]["2.5e-05"]["shift"]
        assert (row1["mc2"]["rate"] is not None) == has
        assert ("mc2_rate" in mc["per_h"]["2.5e-05"]["shift_lam4"]["ratio_to_lam1"]) == has
        assert mc["inertness_replay"]["2.5e-05"]["status"] == ("PASS" if has else ("INCOMPLETE" if len(rep) == 15 else "FAIL"))
        assert row1["mc1"]["n_groups"] == 16
        if not has:
            assert "trace_crosscheck_g0" in row1["mc2"] and row1["mc2"]["note_lam1"]
# ------------------------------------------------------------------------------------------------- end to end (smoke)
def test_end_to_end_smoke_analyze_and_figures(tmp_path, monkeypatch):
    """Tiny smoke chains of shift_lam128 (2 groups, 32 walkers) -> analyze (smoke) -> summary.json / .md, A4 / A4b
    figures in scratch directories; lam = 1 frozen data deliberately NOT read (empty frozen dir)."""
    monkeypatch.setattr(EV.RV, "NW", 32)
    out, fig, frozen = tmp_path / "res", tmp_path / "fig", tmp_path / "frozen_empty"
    frozen.mkdir()
    for j in (("em_flat", 9, 2.5e-5, [0]), ("em_flat", 9, 2.5e-5, [1]), ("em_unbiased", 9, 2.5e-5, [0]),
              ("mala_flat", 9, RV.H_MALA, [0]), ("mala_flat", 9, RV.H_MALA, [1]), ("exact_flat", 9, None, [0, 1]),
              ("exact_unbiased", 9, None, [0, 1])):
        EV.run_job(j, str(out), True)
    with np.load(EV.out_path(str(out), 9, "em_flat", 2.5e-5, 0)) as z:
        assert "side_changes" in z.files and z["lags0"][-1] * z["acf_stride"][0] * 2.5e-5 >= 6.0
        meta = json.loads(str(z["meta_json"]))
        assert meta["seed"] == 402000 and meta["T"] == 5.0 and meta["trace_every"] == 2000 and meta["dynamics_index"] == 9
    S = EV.analyse_all(str(out), None, True, frozen_res=str(frozen))
    assert S["selection"]["verdict"].startswith("SMOKE")
    D = S["dynamics"]["shift_lam128"]
    ph = D["per_h"]["2.5e-05"]
    assert ph["em_flat"]["n_groups"] == 2 and "V1_components_holm" in ph["gates"] and ph["gates"]["V4"] in ("PASS", "FAIL", "NO_DATA")
    assert S["multiplicity"]["V1_holm"]["2.5e-05"]["n_tests"] == 2 and S["multiplicity"]["V4_holm"]["n_tests"] == 8
    m1 = ph["mc1"]["windows"]
    assert m1["-1"]["f_reported"] is False and "tau_e_f" not in m1["-1"]
    assert m1["-0.21"]["f_reported"] and np.isfinite(m1["-0.21"]["tau_e_y"]["est"])
    assert m1["-0.21"]["tau_e_y"]["est"] < 0.05                                        # lam 128: fast relaxation
    assert np.isfinite(m1["0"]["int_0.3125_y"]["est"])
    m2 = ph["mc2"]
    assert m2["rate"]["est"] >= 0 and m2["n_groups"] == 2 and m2["T"] == 5.0
    assert S["mc"]["per_h"]["2.5e-05"]["shift_lam128"]["mc2"] is m2 or S["mc"]["per_h"]["2.5e-05"]["shift_lam128"]["mc2"] == m2
    txt = EV.markdown(S)
    assert "SMOKE" in txt and "MC2" in txt and "shift_lam128" in txt
    json.dumps(S, default=EV._json_default)
    rc = EV.main(["analyze", "--smoke", "--res", str(out), "--fig", str(fig), "--frozen-res", str(frozen)])
    assert rc == 0 and (out / "summary.json").exists() and (out / "summary.md").exists()
    for f in ("expA_A4_mc1_conditional_relaxation", "expA_A4b_mc2_gate_transport"):
        assert (fig / f"{f}.png").exists() and (fig / f"{f}.pdf").exists()
