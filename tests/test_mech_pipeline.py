"""The equal-budget analysis pipeline on mechanism cells (system 'gateway_family'; docs/mechanism/SCIENTIFIC_PLAN.md
sections 5-6): eqb_metrics.py is byte-identical to the committed equal-budget scoring code (every committed
equal-budget analysis keeps its code digest); cell configs and frozen references are current; the per-dynamics
secondary mean force; the family scorer is the accepted gateway scorer on the alpha 1 / lam 1 cells (per-run metrics
of the REAL equal-budget files equal the committed equal-budget run cache, plus tau on e_Fp_stat, minus raw e_F' tau
below its floor); engine / model checks (one engine per cell); a cell config only as gateway_family and vice versa;
the guard against writing into the equal-budget trees; the per-experiment ledger layout of run_cells.py; the
bitwise-reuse gate; and analyze_ladder + audit + cross_cell end to end on a small REAL fixture
(scripts/mechanism/make_cell_fixtures.py, built on first use under MECH_FIXTURE_ROOT or <tmp>/mech_fixtures)."""
import copy
import importlib.util
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "equal_budget"))
import eqb_metrics as M  # noqa: E402
import eqb_family as MF  # noqa: E402
import analyze_ladder as AL  # noqa: E402


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MC = _load("mech_make_cell_configs", "scripts/mechanism/make_cell_configs.py")
BR = _load("mech_build_references", "scripts/mechanism/build_references.py")
CELLS = os.path.join(ROOT, "configs", "mechanism", "cells")
EQB_RUNS = os.path.join(ROOT, "results", "equal_budget_v2", "gateway")
EQB_CACHE = os.path.join(EQB_RUNS, "analysis", "run_metrics")


def cell(exp, name):
    return json.load(open(os.path.join(CELLS, exp, f"{name}.json")))


EQB_ANALYSES = ("gateway", "lta_T300", "lta_T150", "confirm_best_gateway", "confirm_best_lta_T300",
                "confirm_best_lta_T150")


def test_equal_budget_scoring_code_is_unchanged():
    """The mechanism extension lives in eqb_family.py: eqb_metrics.py hashes to the sha256 recorded in every committed
    equal-budget summary, and code_provenance() reproduces the committed digests (no committed analysis goes stale)."""
    for d in EQB_ANALYSES:
        sp = os.path.join(ROOT, "results", "equal_budget_v2", d, "analysis", "summary.json")
        if not os.path.exists(sp):
            continue
        S = json.load(open(sp))
        code = S["provenance"]["code"]
        assert code["files"]["scripts/equal_budget/eqb_metrics.py"] == M.sha256_file(
            os.path.join(ROOT, "scripts", "equal_budget", "eqb_metrics.py")), d
        assert MF.code_provenance(S["system"]) == code, d
        assert M.code_provenance(S["system"]) == code, d
    fam = MF.code_provenance("gateway_family")
    assert "scripts/equal_budget/eqb_family.py" in fam["files"] and fam["digest"] != M.code_provenance("gateway")["digest"]


def test_cell_configs_must_be_analysed_as_gateway_family(tmp_path):
    """--system gateway with a cell config (would score with the equal-budget secondary reference) and --system
    gateway_family with a non-cell config (no cell locations, no guard) are both refused."""
    P = cell("matched_free_energy", "alpha0")
    cp = os.path.join(CELLS, "matched_free_energy", "alpha0.json")
    G = json.load(open(MF.default_config_path("gateway")))
    with pytest.raises(M.MetricsError, match="must be analysed with --system gateway_family"):
        MF.make_scorer("gateway", P)
    with pytest.raises(M.MetricsError, match="must be analysed with --system gateway_family"):
        AL.analyze("gateway", config=cp, out=str(tmp_path / "x"), verbose=False)
    with pytest.raises(M.MetricsError, match="needs a mechanism CELL config"):
        AL.analyze("gateway_family", config=MF.default_config_path("gateway"), out=str(tmp_path / "y"), verbose=False)
    with pytest.raises(M.MetricsError, match="needs a mechanism CELL config"):
        MF.make_scorer("gateway_family", G)
    with pytest.raises(M.MetricsError, match="protected"):      # gateway_family never writes into the eqb trees
        MF.guard_output(G, os.path.join(ROOT, "results", "equal_budget_v2", "gateway", "analysis"), "gateway_family")
    assert not os.path.exists(tmp_path / "x") and not os.path.exists(tmp_path / "y")
    AC = _load("eqb_audit_for_mech_inv", "scripts/equal_budget/audit_completeness.py")
    with pytest.raises(M.MetricsError):
        AC.audit_system("gateway", cp, None, None, None)
    PC = _load("eqb_plot_config_for_mech_inv", "scripts/equal_budget/plot_config.py")
    with pytest.raises(M.MetricsError):
        PC.main(["--system", "gateway", "--config", cp, "--fig-root", str(tmp_path / "f")])


def test_cell_configs_and_references_are_current():
    assert MC.main(["--check"]) == 0
    assert BR.main(["--check"]) == 0
    a1, l1, a0 = cell("matched_free_energy", "alpha1"), cell("conditional_relaxation", "lam1"), \
        cell("matched_free_energy", "alpha0")
    assert a1["results_dir"] == l1["results_dir"] == "results/equal_budget_v2/gateway"
    assert a1["engine_version"] == "gateway_ladder_numba/2" and a0["engine_version"] == "gateway_family_numba/1"
    assert a1["B"] == 2048 * 40 / a1["engine_cfg"]["h"] and a1["N_ladder"] == [2048, 512, 128]
    assert l1["N_ladder"] == [2048, 512]
    for P in (a1, l1, a0):
        assert MF.is_cell(P) and P["analysis_dir"].startswith("results/mechanism/")
        assert P["fig_dir"].startswith("figures/mechanism/")
        assert P["thresholds"]["e_Fp_stat"] == [0.004, 0.0136, 0.0946]


def test_family_mean_force():
    import analyze_gateway_replica_ladder as GL
    P = cell("matched_free_energy", "alpha1")
    c = M.gateway_cell(P["engine_cfg"])
    x = np.linspace(-1.8, 1.8, 72001)
    orig = dict(variant="alpha", alpha=1.0, kappa=None, lam=1.0)
    assert np.array_equal(MF.family_mean_force(x, c, orig), GL.em_mean_force(x, c))       # bitwise
    e = np.exp(-x * x / (2 * c["s"] ** 2))
    om = 1.0 + 31.0 * e
    lr = -31.0 * (x / c["s"] ** 2) * e / om
    fstar = 4 * c["H"] * x * (x * x - 1) + lr / c["beta"]
    for m in (dict(variant="alpha", alpha=0.0, kappa=None, lam=1.0), dict(variant="shift", alpha=None, kappa=1.0, lam=1.0)):
        assert np.allclose(MF.family_mean_force(x, c, m), fstar, rtol=0, atol=1e-12)
    for a, lam in ((0.5, 1.0), (1.0, 0.1), (1.0, 0.25)):
        want = (4 * c["H"] * x * (x * x - 1) + (1 - a) / c["beta"] * lr
                + (a / c["beta"]) * lr / (1 - lam * om ** (2 * a) * c["dt"] / 2))
        got = MF.family_mean_force(x, c, dict(variant="alpha", alpha=a, kappa=None, lam=lam))
        assert np.allclose(got, want, rtol=1e-13, atol=1e-13)
    try:                                    # the engine's own secondary (independent implementation), when present
        import gateway_family_numba as G
    except Exception:  # noqa: BLE001
        return
    for exp, name in (("matched_free_energy", "alpha0.5"), ("matched_free_energy", "shift"),
                      ("conditional_relaxation", "lam0.1"), ("matched_free_energy", "alpha1")):
        Pc = cell(exp, name)
        assert np.allclose(MF.family_mean_force(x, c, MF.family_model(Pc["engine_cfg"])),
                           G.em_mean_force(Pc["engine_cfg"], x), rtol=1e-12, atol=1e-12), name


@pytest.mark.parametrize("exp,name", [("matched_free_energy", "alpha1"), ("conditional_relaxation", "lam1")])
def test_reuse_cell_scores_real_files_like_the_equal_budget_analysis(exp, name):
    """alpha 1 / lam 1: same files, same primary scorer, secondary bitwise the equal-budget EM reference ->
    every per-run scalar equals the committed equal-budget run cache; the only additions are tau on e_Fp_stat."""
    P = cell(exp, name)
    sc = MF.make_scorer("gateway_family", P)
    floors = MF.threshold_floors("gateway_family", P, sc)
    assert floors["tau_e_Fp_strict"]["reachable"] is False and floors["tau_e_Fp_mid"]["reachable"] is True
    assert floors["tau_e_Fp_stat_strict"]["reachable"] is True
    assert sc.reference_info["secondary_equals_equal_budget_em"] is True
    assert sc.reference_info["primary_reference"]["identical_bitwise"] is True
    eqb = json.load(open(os.path.join(EQB_RUNS, "analysis", "summary.json")))["reference"]
    for k in ("em_floor_F_rms", "em_bias_Fp_bin_rms", "zero_noise_floor", "n_eval_nodes"):
        assert sc.reference_info[k] == eqb[k], k
    for N, s, m in ((2048, 8100, "fr"), (512, 8117, "abf"), (128, 8131, "fr")):
        if N not in P["N_ladder"]:
            continue
        path = os.path.join(EQB_RUNS, f"N{N}", f"s{s}_{m}.npz")
        res = MF.load_run(path, "gateway_family")
        MF.check_plan(res, P, path, dict(N=N, seed=s, method=m))
        curves, scal = MF.run_metrics(res, "gateway_family", P, sc, path)
        ref = json.load(open(os.path.join(EQB_CACHE, f"N{N}", f"s{s}_{m}.json")))["scalars"]
        extra = sorted(set(scal) - set(ref))
        assert extra == sorted([f"tau_e_Fp_stat_{n}_{f}" for n in M.THR_NAMES for f in ("t", "u", "censored", "eps")]
                               + [f"tau_bgrid_e_Fp_stat_{n}_{f}" for n in M.THR_NAMES for f in ("t", "u", "censored")])
        gone = sorted(set(ref) - set(scal))           # plan section 6: no raw e_F' tau below its 0.03227 floor
        assert gone == sorted([f"tau_{e}_strict_{f}" for e in ("e_Fp", "e_Fp_em") for f in ("t", "u", "censored", "eps")]
                              + [f"tau_bgrid_e_Fp_strict_{f}" for f in ("t", "u", "censored")])
        diff = [k for k in ref if k != "system" and k in scal and ref[k] != scal[k]
                and not (isinstance(ref[k], float) and np.isnan(ref[k]) and np.isnan(scal[k]))]
        assert diff == [], diff
        assert scal["system"] == "gateway_family" and ref["system"] == "gateway"
        with np.load(os.path.join(EQB_CACHE, f"N{N}", f"s{s}_{m}.npz")) as z:
            for k in z.files:
                assert np.array_equal(z[k], curves[k], equal_nan=True), k
        for nm, eps in zip(M.THR_NAMES, P["thresholds"]["e_Fp_stat"]):
            t = M.tau_persistent(curves["e_Fp_stat"], curves["save_t"], curves["save_u"], eps)
            assert scal[f"tau_e_Fp_stat_{nm}_u"] == t["u"]


def test_engine_and_model_checks():
    path = os.path.join(EQB_RUNS, "N2048", "s8100_abf.npz")
    res = MF.load_run(path, "gateway_family")
    MF.check_plan(res, cell("matched_free_energy", "alpha1"), path)                 # reuse cell: accepted
    with pytest.raises(M.MetricsError) as e:                                          # an alpha0 cell never reads it
        MF.check_plan(res, cell("matched_free_energy", "alpha0"), path)
    assert "engine 'gateway_ladder_numba/2' != the cell's engine 'gateway_family_numba/1'" in str(e.value)
    assert "model field alpha: run 1.0 != cell 0.0" in str(e.value)
    fam = copy.deepcopy(res)
    fam["meta"].update(engine="gateway_family_numba/1", system="gateway_family", variant="alpha", alpha=0.5,
                       kappa=None, lam=1.0)
    fam["cfg"].update(system="gateway_family", variant="alpha", alpha=0.5, kappa=None, lam=1.0)
    MF.check_plan(fam, cell("matched_free_energy", "alpha0.5"), path)               # family engine, right model
    with pytest.raises(M.MetricsError, match="never mixes engines"):
        MF.check_plan(fam, cell("matched_free_energy", "alpha1"), path)               # mixed into a reuse cell
    bad = copy.deepcopy(fam)
    del bad["cfg"]["lam"], bad["meta"]["lam"]
    with pytest.raises(M.MetricsError, match="model field lam missing"):
        MF.check_plan(bad, cell("matched_free_energy", "alpha0.5"), path)
    bad = copy.deepcopy(fam)
    bad["meta"]["alpha"] = 0.25
    with pytest.raises(M.MetricsError, match="cfg 0.5 != meta 0.25"):
        MF.check_plan(bad, cell("matched_free_energy", "alpha0.5"), path)
    bad = copy.deepcopy(fam)
    bad["cfg"]["lam"] = 0.5
    with pytest.raises(M.MetricsError):
        MF.check_plan(bad, cell("matched_free_energy", "alpha0.5"), path)
    # the equal-budget gateway config is untouched by the family rules (its cfg has no model fields)
    MF.check_plan(res, json.load(open(MF.default_config_path("gateway"))), path)


def test_guard_and_cell_locations(tmp_path):
    P = cell("matched_free_energy", "alpha1")
    assert MF.cell_results_root(P) == os.path.join(ROOT, "results", "equal_budget_v2")
    assert MF.cell_analysis_dir(P) == os.path.join(ROOT, "results", "mechanism", "matched_free_energy", "alpha1", "analysis")
    assert MF.cell_fig_dir(P) == os.path.join(ROOT, "figures", "mechanism", "matched_free_energy", "alpha1")
    assert MF.cell_fig_dir(P, str(tmp_path)) == os.path.join(str(tmp_path), "alpha1")
    for bad in ("results/equal_budget_v2/gateway/analysis", "figures/equal_budget_v2/gateway/N2048", "results/equal_budget_v2"):
        with pytest.raises(M.MetricsError, match="protected equal-budget tree"):
            MF.guard_output(P, os.path.join(ROOT, bad))
    MF.guard_output(P, os.path.join(ROOT, "results", "mechanism", "x"))
    MF.guard_output(json.load(open(MF.default_config_path("gateway"))), os.path.join(ROOT, "results", "equal_budget_v2"))
    with pytest.raises(M.MetricsError, match="protected"):
        AL.analyze("gateway_family", config=os.path.join(CELLS, "matched_free_energy", "alpha1.json"),
                   out=os.path.join(ROOT, "results", "equal_budget_v2", "gateway", "analysis"), verbose=False)
    with pytest.raises(M.MetricsError, match="no default config"):
        MF.default_config_path("gateway_family")


def test_cell_ledger_layout_and_filter(tmp_path):
    """Production layout: new-engine cells read run_cells.py's ONE ledger per experiment, filtered to their variant
    before keying (variants never overwrite each other); reuse cells read the equal-budget ledger unfiltered."""
    a0, a1 = cell("matched_free_energy", "alpha0"), cell("matched_free_energy", "alpha1")
    rc = _load("mech_run_cells_for_ledger", "scripts/mechanism/run_cells.py")
    X = json.load(open(os.path.join(ROOT, "configs", "mechanism", "matched_free_energy.json")))
    assert MF.cell_ledger(a0) == (os.path.join(ROOT, X["out_root"], "ledger.csv"),
                                  dict(experiment="matched_free_energy", variant="alpha0"))
    assert a0["results_dir"] == f"{X['out_root']}/alpha0"          # run_cells.out_path: <out_root>/<variant>/N<N>/
    assert MF.cell_ledger(a1) == (os.path.join(ROOT, "results", "equal_budget_v2", "gateway", "ledger.csv"), None)
    AC = _load("eqb_audit_for_mech_ledger", "scripts/equal_budget/audit_completeness.py")
    lp = tmp_path / "ledger.csv"
    import csv
    with open(lp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rc.LEDGER_FIELDS)
        w.writeheader()
        for v, st in (("alpha0", "complete"), ("shift", "FAILED: x"), ("alpha0.5", "complete")):
            w.writerow(dict(experiment="matched_free_energy", variant=v, N=2048, seed=8100, method="fr", n_steps=1,
                            status=st, wall_s=1, peak_rss_mb=1, n_force_evals=1, finished_utc="t"))
    got = AC.load_ledger(str(lp), dict(experiment="matched_free_energy", variant="alpha0"))
    assert list(got) == [(2048, 8100, "fr")] and got[(2048, 8100, "fr")]["variant"] == "alpha0"
    assert AC.load_ledger(str(lp), dict(experiment="matched_free_energy", variant="shift"))[(2048, 8100, "fr")]["status"] \
        == "FAILED: x"
    assert AC.load_ledger(str(lp), dict(experiment="conditional_relaxation", variant="alpha0")) == {}
    assert AC.load_ledger(str(lp))[(2048, 8100, "fr")]["variant"] == "alpha0.5"         # unfiltered: last row wins


def test_reuse_gate_record_rules(tmp_path):
    """A reuse cell is licensed only by a PASS gate record that covers >= 4 bitwise-equal jobs, both arms, N 2048 and
    128, written by gateway_family_numba at the original model, whose reused files are still those files."""
    a1 = cell("matched_free_energy", "alpha1")
    assert a1["reuse"]["gate_record"] == "results/mechanism/reuse_gate/reuse_gate.json"
    assert MF.reuse_gate_problems(cell("matched_free_energy", "alpha0")) == []          # not a reuse cell
    rec = tmp_path / "gate.json"
    P = dict(a1, reuse=dict(a1["reuse"], gate_record=str(rec)))
    assert "does not exist" in MF.reuse_gate_problems(P)[0]
    jobs = []
    for N in (2048, 128):
        for m in ("abf", "fr"):
            f = os.path.join(EQB_RUNS, f"N{N}", f"s8100_{m}.npz")
            jobs.append(dict(N=N, seed=8100, method=m, bitwise_equal=True, reused_sha256=M.sha256_file(f)))
    good = dict(schema=MF.REUSE_GATE_SCHEMA, verdict="PASS", engine="gateway_family_numba/1",
                model=dict(MF.FAMILY_MODEL_DEFAULTS), jobs=jobs)
    rec.write_text(json.dumps(good))
    assert MF.reuse_gate_problems(P) == [] and MF.reuse_gate_status(P)["licensed"] is True
    for mut, why in ((dict(verdict="FAIL"), "not PASS"), (dict(jobs=jobs[:3]), "< the required 4"),
                     (dict(jobs=[j for j in jobs if j["N"] == 2048] * 2), "do not cover N [128]"),
                     (dict(jobs=[dict(j, bitwise_equal=(i != 1)) for i, j in enumerate(jobs)]), "not bitwise equal"),
                     (dict(jobs=[dict(j, reused_sha256="0" * 64) for j in jobs]), "changed since"),
                     (dict(model=dict(MF.FAMILY_MODEL_DEFAULTS, lam=0.5)), "not the original dynamics")):
        rec.write_text(json.dumps(dict(good, **mut)))
        assert any(why in p for p in MF.reuse_gate_problems(P)), (mut, MF.reuse_gate_problems(P))



def test_family_scorer_refuses_a_wrong_reference(tmp_path):
    P = cell("matched_free_energy", "alpha0.5")
    with np.load(MF.reference_path("gateway_family", P)) as z:
        ref = {k: z[k] for k in z.files}
    Q = dict(P, reference_file=str(tmp_path / "r.npz"))
    np.savez(tmp_path / "r.npz", **dict(ref, model_json=np.array(json.dumps(dict(P["model"], alpha=0.0)))))
    with pytest.raises(M.MetricsError, match="model"):
        MF.make_scorer("gateway_family", Q)
    F = ref["F_ref"].copy()
    F[10] += 1e-9                                                    # primary no longer bitwise the equal-budget one
    np.savez(tmp_path / "r.npz", **dict(ref, F_ref=F))
    with pytest.raises(M.MetricsError):
        MF.make_scorer("gateway_family", Q)
    Q2 = dict(P, reference_file=str(tmp_path / "missing.npz"))
    with pytest.raises(M.MetricsError, match="does not exist"):
        MF.make_scorer("gateway_family", Q2)


# ------------------------------------------------------------------------------------------------ real fixture
FIX_ROOT = os.environ.get("MECH_FIXTURE_ROOT", os.path.join(tempfile.gettempdir(), "mech_fixtures"))
FIX_CELLS = ["matched_free_energy/alpha0", "matched_free_energy/alpha1"]


def _fixture_complete(c):
    p = os.path.join(FIX_ROOT, "configs", "cells", c + ".json")
    if not os.path.exists(p):
        return False
    P = json.load(open(p))
    if not P.get("ledger"):                       # built before the production ledger layout: rebuild
        return False
    runs = [os.path.join(P["results_dir"], f"N{N}", f"s{s}_{m}.npz") for N in P["N_ladder"] for s in P["seeds"]
            for m in ("abf", "fr")]
    gate = (P.get("reuse") or {}).get("gate_record")
    return (all(os.path.exists(r) for r in runs) and os.path.exists(P["ledger"]) and os.path.exists(P["reference_file"])
            and (not gate or os.path.exists(gate)))



@pytest.fixture(scope="module")
def fixture_root():
    need = [c for c in FIX_CELLS if not _fixture_complete(c)]
    if need:
        pytest.importorskip("gateway_family_numba")
        import subprocess
        r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "mechanism", "make_cell_fixtures.py"),
                            "--root", FIX_ROOT, "--cells", *FIX_CELLS, "--workers", "4"],
                           capture_output=True, text=True, timeout=3600)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return FIX_ROOT


def test_analyze_fixture_cells(fixture_root, tmp_path):
    out = {}
    for c in FIX_CELLS:
        cfg = os.path.join(fixture_root, "configs", "cells", c + ".json")
        out[c] = AL.analyze("gateway_family", config=cfg, out=str(tmp_path / c), n_boot=300, verbose=False)
    S0, S1 = out["matched_free_energy/alpha0"], out["matched_free_energy/alpha1"]
    assert S0["cell"]["engine_version"] == "gateway_family_numba/1" and S1["cell"]["engine_version"] == "gateway_ladder_numba/2"
    for S in (S0, S1):
        assert S["system"] == "gateway_family" and S["kind"] == "gateway"
        for N in ("8", "2"):
            st = S["status"][N]["methods"]
            assert len(st["abf"]["complete"]) == 3 and len(st["fr"]["complete"]) == 3
            blk = S["per_N"][N]["fr"]
            assert "tau_e_Fp_stat_mid_u" in blk and "Ibar_Fp_stat" in blk and blk["fr_deaths"]["median"] > 0
            # plan section 6: tau endpoints e_F, e_F'_stat, TV_half; raw e_F' tau only above its floor (secondary)
            ct = S["contrasts"][N]
            assert list(ct["tau"]) == ([f"tau_e_F_{n}_u" for n in M.THR_NAMES]
                                       + [f"tau_e_Fp_stat_{n}_u" for n in M.THR_NAMES] + ["tau_TV_half_u"])
            assert list(ct["tau_secondary"]) == ([f"tau_e_F_em_{n}_u" for n in M.THR_NAMES]
                                                 + [f"tau_{e}_{n}_u" for e in ("e_Fp", "e_Fp_em") for n in ("mid", "loose")])
            assert not any(k.startswith("tau_e_Fp_strict") or k.startswith("tau_bgrid_e_Fp_strict") for k in blk)
            assert "tau_bgrid_e_Fp_stat_mid_u" in blk
        assert S["threshold_floors"]["tau_e_Fp_stat_strict"]["reachable"] is True
        assert S["reference"]["primary_reference"]["identical_bitwise"] is True
    # same primary reference in both cells; the alpha 0 secondary IS F*' (no EM bias of F')
    assert S0["reference"]["primary_reference"]["digest"] == S1["reference"]["primary_reference"]["digest"]
    assert S0["reference"]["em_bias_Fp_bin_rms"] == 0.0 and S1["reference"]["em_bias_Fp_bin_rms"] > 1e-4
    # the reuse cell's metrics are the gateway system's metrics of the same files
    P1 = json.load(open(os.path.join(fixture_root, "configs", "cells", "matched_free_energy", "alpha1.json")))
    G = json.load(open(MF.default_config_path("gateway")))
    G = dict(G, B=P1["B"], N_ladder=P1["N_ladder"], seeds=P1["seeds"])
    gsc = MF.make_scorer("gateway", G)
    path = os.path.join(P1["results_dir"], "N8", "s8101_fr.npz")
    _, sg = MF.run_metrics(MF.load_run(path, "gateway"), "gateway", G, gsc, path)
    for k, v in sg.items():
        if k == "system":
            continue
        got = S1["per_N"]["8"]["fr"].get(k, {}).get("per_seed", {}).get("8101") if isinstance(v, (int, float)) \
            and not isinstance(v, bool) else None
        if got is not None:
            assert M.json_safe(v) == got, k


def test_audit_fails_a_cell_that_mixes_engines(fixture_root, tmp_path):
    """A gateway_family_numba/1 file inside a reuse cell's tree (or vice versa) is a PLAN_MISMATCH of the audit."""
    AC = _load("eqb_audit_for_mech", "scripts/equal_budget/audit_completeness.py")
    P1 = json.load(open(os.path.join(fixture_root, "configs", "cells", "matched_free_energy", "alpha1.json")))
    P0 = json.load(open(os.path.join(fixture_root, "configs", "cells", "matched_free_energy", "alpha0.json")))
    tree = tmp_path / "eqb_gateway"
    shutil.copytree(P1["results_dir"], tree)
    shutil.copy2(os.path.join(P0["results_dir"], "N8", "s8100_fr.npz"), tree / "N8" / "s8100_fr.npz")
    cfg = dict(P1, results_dir=str(tree), analysis_dir=str(tmp_path / "ana"), fig_dir=str(tmp_path / "fig"))
    cp = tmp_path / "alpha1.json"
    cp.write_text(json.dumps(cfg))
    a = AC.audit_system("gateway_family", str(cp), None, None, None)
    assert a["verdict"] == "FAIL"
    msgs = [f["message"] for f in a["failures"] if f["code"] == "PLAN_MISMATCH" and f["where"] == "N8/s8100_fr.npz"]
    assert msgs and "never mixes engines" in msgs[0] and "model field alpha" in msgs[0]
    assert not [f for f in a["failures"] if f["code"].startswith("REF_")]          # the cell reference itself is fine
    assert a["cell"]["cell"] == "alpha1" and a["analysis_dir"] == str(tmp_path / "ana")


def _fixture_cells_into(fixture_root, tmp_path, n_boot=300):
    """Copies of the fixture cell configs whose analysis_dir is under tmp_path, analysed there."""
    cells = tmp_path / "cells"
    out = {}
    for c in FIX_CELLS:
        P = json.load(open(os.path.join(fixture_root, "configs", "cells", c + ".json")))
        P["analysis_dir"] = str(tmp_path / "ana" / c)
        P["fig_dir"] = str(tmp_path / "fig" / c)
        cp = cells / (c + ".json")
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text(json.dumps(P, indent=1))
        AL.analyze("gateway_family", config=str(cp), n_boot=n_boot, verbose=False)
        out[c] = (str(cp), P)
    return cells, out


def test_fixture_audit_production_ledger_layout_and_gate(fixture_root, tmp_path):
    """The audit of a new-engine cell reads run_cells.py's per-experiment ledger (filtered to the variant) and of a
    reuse cell the reused tree's ledger + the gate record: no LEDGER_* or REUSE_GATE failure on the fixture (whose
    layout is the production one); figures are not made here, so only figure / synthesis failures remain."""
    AC = _load("eqb_audit_for_mech_layout", "scripts/equal_budget/audit_completeness.py")
    cells, made = _fixture_cells_into(fixture_root, tmp_path)
    for c, (cp, P) in made.items():
        a = AC.audit_system("gateway_family", cp, None, None, None)
        codes = sorted({f["code"] for f in a["failures"]})
        assert not [x for x in codes if x.startswith("LEDGER") or x == "REUSE_GATE" or x.startswith("REF_")
                    or (x.startswith("SUMMARY") and x != "SUMMARY_NBOOT") or x.startswith("PLAN")
                    or x.startswith("FILE")], (c, a["failures"][:5])           # n_boot 300 here (fast), 10000 in the plan
        lp, filt = MF.cell_ledger(P)
        assert a["cell"]["ledger"] == lp and a["cell"]["ledger_filter"] == filt
        if P.get("reuse"):
            assert filt is None and a["cell"]["reuse_gate"]["licensed"] is True
        else:
            assert lp == os.path.join(os.path.dirname(P["results_dir"]), "ledger.csv")       # one per experiment
            assert filt == dict(experiment="matched_free_energy", variant=P["cell"])
    # a filter that selects no row of the experiment ledger: every complete file is reported (LEDGER_GAP)
    cp0, P0 = made["matched_free_energy/alpha0"]
    Q = dict(P0, ledger_filter=dict(experiment="matched_free_energy", variant="nonexistent"))
    qp = tmp_path / "q.json"
    qp.write_text(json.dumps(Q))
    a = AC.audit_system("gateway_family", str(qp), None, None, None)
    assert any(w["code"] == "LEDGER_GAP" for w in a["warnings"])


def test_cross_cell_refuses_stale_or_foreign_summaries(fixture_root, tmp_path):
    """cross_cell on real fixture analyses: accepted as analysed; refused when the cell config switches engine /
    run tree (the --rerun-reuse-cells situation), when a run's cache no longer matches, when a run completes after
    the analysis, and when the reuse gate does not license a reuse cell."""
    X = _load("mech_cross_cell_for_stale", "scripts/mechanism/cross_cell.py")
    cells, made = _fixture_cells_into(fixture_root, tmp_path)
    doc = X.analyse(X.load_cells(str(cells)), n_boot=200, production=False, verify_runs=True)
    assert doc["freshness_verified"] is True
    assert doc["cells"]["matched_free_energy/alpha1"]["reuse_gate"]["licensed"] is True
    c1 = [r for r in doc["experiments"]["matched_free_energy"]["metrics"]["Ibar_F"]["contrasts"] if r["id"] == "C1"]
    assert c1 and all(r["n"] == 3 for r in c1)
    cp1, P1 = made["matched_free_energy/alpha1"]
    cp0, P0 = made["matched_free_energy/alpha0"]

    def refused(mutate_cfg, match):
        Q = mutate_cfg(copy.deepcopy(P1))
        open(cp1, "w").write(json.dumps(Q, indent=1))
        try:
            with pytest.raises(X.CrossCellError, match=match):
                X.analyse(X.load_cells(str(cells)), n_boot=200, production=False, verify_runs=True)
        finally:
            open(cp1, "w").write(json.dumps(P1, indent=1))

    # rerun-reuse mode: same analysis_dir / reference_file, other engine and run tree -> the old summary is refused
    refused(lambda Q: dict(Q, engine_version="gateway_family_numba/1", results_dir=P0["results_dir"],
                           out_dir=os.path.basename(P0["results_dir"]), reuse=None), "engine_version")
    refused(lambda Q: dict(Q, model=dict(Q["model"], lam=0.1)), "model")
    # the gate record stops licensing the reuse (same config, same summary) -> refused
    gp = P1["reuse"]["gate_record"]
    keep = open(gp).read()
    try:
        open(gp, "w").write(json.dumps(dict(json.loads(keep), verdict="FAIL")))
        with pytest.raises(X.CrossCellError, match="without a licensing bitwise-reuse gate"):
            X.analyse(X.load_cells(str(cells)), n_boot=200, production=False, verify_runs=True)
    finally:
        open(gp, "w").write(keep)
    # a cached per-run entry whose key no longer matches -> stale
    cache = os.path.join(P0["analysis_dir"], "run_metrics", "N8", "s8101_fr.json")
    blob = json.load(open(cache))
    good = json.dumps(blob)
    blob["key"]["code_digest"] = "0" * 16
    open(cache, "w").write(json.dumps(blob))
    try:
        with pytest.raises(X.CrossCellError, match="STALE"):
            X.analyse(X.load_cells(str(cells)), n_boot=200, production=False, verify_runs=True)
        X.analyse(X.load_cells(str(cells)), n_boot=200, production=False, verify_runs=False)   # explicit opt-out only
    finally:
        open(cache, "w").write(good)
    X.analyse(X.load_cells(str(cells)), n_boot=200, production=False, verify_runs=True)
