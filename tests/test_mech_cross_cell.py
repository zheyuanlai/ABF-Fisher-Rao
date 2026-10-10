"""Known-answer tests of scripts/mechanism/cross_cell.py (the preregistered cross-cell contrasts,
docs/mechanism/SCIENTIFIC_PLAN.md section 6): Delta = median over paired seeds of the log-ratio difference, the seed
bootstrap CI and two-sided p, Holm within the frozen families, the [0.90, 1.11] equivalence rule, seed pairing, the
sign convention (ratio < 1 = a has the larger FR gain) and the input checks.  Synthetic summaries only (fast)."""
import json
import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "mechanism"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "equal_budget"))
import cross_cell as X  # noqa: E402
import eqb_metrics as M  # noqa: E402
import eqb_family as MF  # noqa: E402

SEEDS = list(range(8100, 8132))
NB = 2000                                   # bootstrap resamples in these tests (formulas use it explicitly)


def L_of(vals):
    return {s: math.log(v) for s, v in zip(SEEDS, vals)}


# ------------------------------------------------------------------------------------------------ contrast()
def test_constant_shift_gives_exact_delta_degenerate_ci_and_minimal_p():
    rng = np.random.default_rng(1)
    Lb = {s: float(x) for s, x in zip(SEEDS, rng.normal(-0.4, 0.3, len(SEEDS)))}
    c = math.log(1.3)
    La = {s: v + c for s, v in Lb.items()}
    r = X.contrast(La, Lb, n_boot=NB)
    assert r["n"] == 32
    assert r["Delta"] == pytest.approx(c, abs=1e-12)
    assert r["Delta_ci95"] == pytest.approx([c, c], abs=1e-12)
    assert r["ratio"] == pytest.approx(1.3) and r["ratio_ci95"] == pytest.approx([1.3, 1.3])
    assert r["p"] == pytest.approx(2.0 / (NB + 1))            # every resampled median > 0
    assert r["wins_a"] == 0 and r["losses_a"] == 32 and r["sign_test_p"] == pytest.approx(2 * 0.5 ** 32)
    assert not r["equivalent"]                                 # 1.3 is outside [0.90, 1.11]


def test_identical_cells_are_equivalent_with_p_one():
    Lb = L_of(np.linspace(0.4, 0.9, 32))
    r = X.contrast(dict(Lb), dict(Lb), n_boot=NB)
    assert r["Delta"] == 0.0 and r["ratio_ci95"] == [1.0, 1.0]
    assert r["p"] == 1.0 and r["equivalent"] and r["ties"] == 32 and r["wilcoxon_p"] == 1.0
    assert X.verdict(r, X.holm([r["p"]])[0]) == "equivalent"


def test_equivalence_band_is_inclusive_and_separate_from_differs():
    Lb = L_of(np.full(32, 0.6))
    for ratio, eq in ((1.05, True), (0.92, True), (1.11, True), (0.90, True), (1.12, False), (0.89, False)):
        r = X.contrast({s: v + math.log(ratio) for s, v in Lb.items()}, Lb, n_boot=NB)
        assert r["equivalent"] is eq, ratio
    r = X.contrast({s: v + math.log(1.05) for s, v in Lb.items()}, Lb, n_boot=NB)
    assert X.verdict(r, 0.001).startswith("equivalent (within [0.90, 1.11]) but differs")


def test_median_bootstrap_ci_and_p_match_an_independent_computation():
    rng = np.random.default_rng(7)
    La = {s: float(x) for s, x in zip(SEEDS, rng.normal(-0.5, 0.4, 32))}
    Lb = {s: float(x) for s, x in zip(SEEDS, rng.normal(-0.3, 0.4, 32))}
    del La[8105]                                               # one seed missing in a: excluded from the pairing
    Lb[9999] = 0.0                                             # an extra seed in b: excluded
    r = X.contrast(La, Lb, n_boot=NB, seed=123)
    common = sorted(set(La) & set(Lb))
    assert r["n"] == 31 and r["a_only"] == [] and r["b_only"] == [8105, 9999] and 8105 not in r["seeds"]
    d = np.array([La[s] - Lb[s] for s in common])
    g = np.random.default_rng(123)
    boot = np.median(d[g.integers(0, d.size, size=(NB, d.size))], axis=1)
    assert r["Delta"] == pytest.approx(float(np.median(d)), abs=1e-15)
    assert r["Delta_ci95"] == pytest.approx([np.percentile(boot, 2.5), np.percentile(boot, 97.5)], abs=1e-15)
    lo = (np.sum(boot <= 0) + 1) / (NB + 1)
    hi = (np.sum(boot >= 0) + 1) / (NB + 1)
    assert r["p"] == pytest.approx(min(1.0, 2 * min(lo, hi)), abs=1e-15)
    assert r["per_seed"][8100] == pytest.approx(La[8100] - Lb[8100])


def test_known_median_of_an_odd_sample():
    a = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 9.0}
    b = {1: 3.0, 2: 1.0, 3: 0.0, 4: -2.0, 5: -5.0}           # d = [-3, -1, 0, 2, 5]: median 0; seed 6 unpaired
    r = X.contrast(a, b, n_boot=NB)
    assert r["n"] == 5 and r["Delta"] == 0.0 and r["a_only"] == [6]
    assert r["wins_a"] == 2 and r["losses_a"] == 2 and r["ties"] == 1


def test_sign_convention_ratio_below_one_means_a_has_the_larger_gain():
    # cell a: FR halves the integrated error; cell b: FR changes nothing -> a has the larger FR gain
    abf = {s: 1.0 + 0.01 * i for i, s in enumerate(SEEDS)}
    La, _ = X.log_ratios(abf, {s: 0.5 * v for s, v in abf.items()})
    Lb, _ = X.log_ratios(abf, dict(abf))
    r = X.contrast(La, Lb, n_boot=NB)
    assert r["ratio"] == pytest.approx(0.5) and r["Delta"] < 0
    assert "a has the larger FR gain" in X.verdict(r, X.holm([r["p"]])[0])


def test_small_n_makes_no_claim():
    r = X.contrast({1: 0.0, 2: 0.1, 3: 0.2}, {1: 1.0, 2: 1.1, 3: 1.2}, n_boot=NB)
    assert r["p"] == pytest.approx(2.0 / (NB + 1))            # the bootstrap p of n = 3 is NOT calibrated ...
    assert r["sign_test_p"] == pytest.approx(0.25)             # ... the exact sign test says 0.25
    assert "uncalibrated" in X.verdict(r, 0.001)


def test_log_ratios_exclusions():
    lr, ex = X.log_ratios({1: 2.0, 2: 1.0, 3: 0.0, 4: "inf", 5: 1.0}, {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0})
    assert lr == {1: pytest.approx(math.log(0.5)), 2: 0.0}
    assert set(ex) == {3, 4, 5}


def test_holm_known_values_and_missing_tests():
    adj = X.holm([0.01, 0.04, 0.03, 0.005])
    assert adj == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert X.holm([0.01, float("nan")], m=2) == pytest.approx([0.02, 1.0])
    assert X.holm([0.01], m=12) == pytest.approx([0.12])
    with pytest.raises(X.CrossCellError):
        X.holm([0.1, 0.2, 0.3], m=2)


# ------------------------------------------------------------------------------------------------ analyse()
DIGEST = "d" * 64


def write_cell(root, exp, cell, Ns, ratio_by_N, drop=None, digest=DIGEST, plan_B=None, available=True,
               summary_cell=None, code_digest=None, ref_sha=None):
    """Cell config + a minimal analyze_ladder-shaped summary: ABF Ibar_F = 1 + 0.01 i (seed i), FR = ratio x ABF
    (Ibar_TV_half: FR = sqrt(ratio) x ABF; Ibar_Fp_stat: FR = ABF, i.e. no gain).  The summary's identity (cell block,
    reference path + sha256 of a dummy frozen reference, code and config digests) matches the config unless
    summary_cell / code_digest / ref_sha override it."""
    ref = root / "refs" / f"{exp}_{cell}_reference.npz"
    ref.parent.mkdir(parents=True, exist_ok=True)
    if not ref.exists():
        ref.write_bytes(f"dummy reference {exp}/{cell}".encode())
    P = dict(system="gateway_family", experiment=exp, cell=cell, out_dir=cell, results_dir=str(root / "res" / exp / cell),
             analysis_dir=str(root / "ana" / exp / cell), fig_dir=str(root / "fig" / exp / cell), B=81920 * 40,
             N_ladder=list(Ns), seeds=list(SEEDS), engine_cfg=dict(h=2.5e-5), thresholds=dict(e_F=[1, 2, 3]),
             engine_version="gateway_family_numba/1", model=dict(variant="alpha", alpha=1.0, kappa=None, lam=1.0),
             reference_file=str(ref), reuse=None)
    cp = root / "cells" / exp / f"{cell}.json"
    cp.parent.mkdir(parents=True, exist_ok=True)
    cp.write_text(json.dumps(P))
    if not available:
        return P
    per_N = {}
    for N in Ns:
        abf = {s: 1.0 + 0.01 * i for i, s in enumerate(SEEDS)}
        r = ratio_by_N[N]
        fr = {s: r * v for s, v in abf.items()}
        if drop and N in drop:
            fr.pop(drop[N])
        blk = {}
        for m, vals in (("abf", abf), ("fr", fr)):
            tv = {s: (v if m == "abf" else math.sqrt(r) * abf[s]) for s, v in vals.items()}
            fp = {s: abf[s] for s in vals}
            blk[m] = {"Ibar_F": dict(per_seed={str(s): v for s, v in vals.items()}),
                      "Ibar_TV_half": dict(per_seed={str(s): v for s, v in tv.items()}),
                      "Ibar_Fp_stat": dict(per_seed={str(s): v for s, v in fp.items()})}
        per_N[str(N)] = blk
    cl = {k: P[k] for k in ("experiment", "cell", "engine_version", "results_dir", "reference_file", "model", "reuse")}
    cl.update(summary_cell or {})
    S = dict(system="gateway_family", metrics_version=M.METRICS_VERSION, n_boot=M.N_BOOT, generated_utc="T", cell=cl,
             plan=dict(B=plan_B or P["B"], h=2.5e-5, N_ladder=list(Ns), seeds=list(SEEDS), thresholds=P["thresholds"]),
             reference=dict(path=os.path.relpath(str(ref), ROOT), sha256=ref_sha or M.sha256_file(str(ref)),
                            primary_reference=dict(digest=digest)),
             provenance=dict(code=dict(digest=code_digest or MF.code_provenance("gateway_family")["digest"]),
                             config_digest=MF.config_digest(P)),
             per_N=per_N, warnings=[])
    a = root / "ana" / exp / cell
    a.mkdir(parents=True, exist_ok=True)
    (a / "summary.json").write_text(json.dumps(S))
    return P


@pytest.fixture()
def tree(tmp_path):
    N1, N2 = [2048, 512, 128], [2048, 512]
    e1, e2 = "matched_free_energy", "conditional_relaxation"
    write_cell(tmp_path, e1, "alpha1", N1, {N: 0.6 for N in N1})
    write_cell(tmp_path, e1, "alpha0", N1, {N: 0.6 * 1.1 for N in N1}, drop={512: 8131})   # ratio 1.1: equivalent
    write_cell(tmp_path, e1, "shift", N1, {N: 0.3 for N in N1})                            # ratio 0.5: larger gain
    write_cell(tmp_path, e1, "alpha0.5", N1, {}, available=False)                          # not analysed yet
    write_cell(tmp_path, e2, "lam1", N2, {N: 0.6 for N in N2})
    write_cell(tmp_path, e2, "lam0.1", N2, {2048: 0.9, 512: 0.6})                          # smaller gain at 2048
    write_cell(tmp_path, e2, "lam0.25", N2, {N: 0.6 for N in N2})
    return tmp_path


def test_analyse_known_answers_holm_families_and_missing_cell(tree):
    cells = X.load_cells(str(tree / "cells"))
    doc = X.analyse(cells, n_boot=NB, production=True, verify_runs=False)
    assert doc["freshness_verified"] is False and any("NOT verified" in w for w in doc["warnings"])
    E1 = doc["experiments"]["matched_free_energy"]
    assert E1["family_size"] == 12 and E1["cells_missing"] == ["alpha0.5"]
    blk = E1["metrics"]["Ibar_F"]
    assert not blk["family_complete"] and blk["n_tests"] == 12
    rows = {(r["id"], r["N"]): r for r in blk["contrasts"]}
    pmin = 2.0 / (NB + 1)
    for N in (2048, 512, 128):
        c1, c2, c3, c4 = rows[("C1", N)], rows[("C2", N)], rows[("C3", N)], rows[("C4", N)]
        assert c1["ratio"] == pytest.approx(1.1) and c1["ratio_ci95"] == pytest.approx([1.1, 1.1])
        assert c1["equivalent"] and c1["p"] == pytest.approx(pmin)
        assert c2["ratio"] == pytest.approx(0.5) and not c2["equivalent"]
        assert c3["ratio"] == pytest.approx(0.5 / 1.1)
        assert c4["n"] == 0 and c4["verdict"] == "NOT AVAILABLE" and c4["missing"] == ["alpha0.5"]
        # Holm over the frozen 12: nine computed tests share p = pmin, three missing count as p = 1
        for c in (c1, c2, c3):
            assert c["p_holm"] == pytest.approx(12 * pmin)
        assert c4["p_holm"] == 1.0
        assert "a has the larger FR gain" in c2["verdict"]
        assert c1["verdict"].startswith("equivalent (within [0.90, 1.11]) but differs")
    assert rows[("C1", 512)]["n"] == 31 and rows[("C1", 2048)]["n"] == 32       # seed 8131 lacks FR at 512 in alpha0
    # secondary metrics: TV_half ratio = sqrt(ratio); Fp_stat: no gain anywhere -> ratio 1, equivalent
    tv = {(r["id"], r["N"]): r for r in E1["metrics"]["Ibar_TV_half"]["contrasts"]}
    assert tv[("C2", 2048)]["ratio"] == pytest.approx(math.sqrt(0.5))
    fp = {(r["id"], r["N"]): r for r in E1["metrics"]["Ibar_Fp_stat"]["contrasts"]}
    assert fp[("C2", 128)]["ratio"] == 1.0 and fp[("C2", 128)]["equivalent"]
    # Experiment II: Holm over 2; lam0.1 has the SMALLER gain at 2048 (ratio 1.5), equal at 512; trend rows unadjusted
    E2 = doc["experiments"]["conditional_relaxation"]
    assert E2["family_size"] == 2
    r2 = {(r["id"], r["N"]): r for r in E2["metrics"]["Ibar_F"]["contrasts"]}
    assert r2[("D1", 2048)]["ratio"] == pytest.approx(1.5) and r2[("D1", 2048)]["p_holm"] == pytest.approx(2 * pmin)
    assert "b has the larger FR gain" in r2[("D1", 2048)]["verdict"]
    assert r2[("D1", 512)]["ratio"] == 1.0 and r2[("D1", 512)]["p"] == 1.0 and r2[("D1", 512)]["p_holm"] == 1.0
    assert r2[("T1", 2048)]["kind"] == "trend" and r2[("T1", 2048)]["p_holm"] is None
    assert "differs" not in r2[("T1", 2048)]["verdict"] and "(Holm)" not in r2[("T1", 2048)]["verdict"]
    assert r2[("T2", 512)]["verdict"] == "NOT AVAILABLE"                      # lam0.5 absent
    pc = E2["per_cell"]["lam0.1"]["2048"]["Ibar_F"]
    assert pc["n"] == 32 and pc["median_G"] == pytest.approx(-0.1)


def test_main_writes_json_and_md_and_refuses_the_equal_budget_tree(tree):
    out = tree / "syn"
    doc = X.main(["--cells-root", str(tree / "cells"), "--out-dir", str(out), "--n-boot", str(NB), "--no-verify-runs"])
    S = json.loads((out / "cross_cell.json").read_text())
    assert S["schema"] == X.SCHEMA and S["experiments"]["matched_free_energy"]["family_size"] == 12
    assert "C1 alpha0 vs alpha1" in (out / "cross_cell.md").read_text()
    assert doc["primary_reference_digest"] == DIGEST
    with pytest.raises(X.CrossCellError):
        X.main(["--cells-root", str(tree / "cells"), "--n-boot", "10", "--no-verify-runs",
                "--out-dir", os.path.join(ROOT, "results", "equal_budget_v2", "never_here")])
    assert not os.path.exists(os.path.join(ROOT, "results", "equal_budget_v2", "never_here"))


def test_input_checks(tmp_path):
    N1 = [2048, 512, 128]
    write_cell(tmp_path, "matched_free_energy", "alpha1", N1, {N: 0.6 for N in N1})
    write_cell(tmp_path, "matched_free_energy", "alpha0", N1, {N: 0.6 for N in N1}, digest="e" * 64)
    with pytest.raises(X.CrossCellError, match="primary reference"):
        X.analyse(X.load_cells(str(tmp_path / "cells")), n_boot=NB, verify_runs=False)
    write_cell(tmp_path, "matched_free_energy", "alpha0", N1, {N: 0.6 for N in N1}, plan_B=123)
    with pytest.raises(X.CrossCellError, match="does not belong"):
        X.analyse(X.load_cells(str(tmp_path / "cells")), n_boot=NB, verify_runs=False)
    # a production family with the wrong number of N is refused (the plan freezes 12 / 2 tests)
    write_cell(tmp_path, "matched_free_energy", "alpha0", [2048, 512], {N: 0.6 for N in [2048, 512]})
    write_cell(tmp_path, "matched_free_energy", "alpha1", [2048, 512], {N: 0.6 for N in [2048, 512]})
    with pytest.raises(X.CrossCellError, match="freezes 12"):
        X.analyse(X.load_cells(str(tmp_path / "cells")), n_boot=NB, production=True, verify_runs=False)
    doc = X.analyse(X.load_cells(str(tmp_path / "cells")), n_boot=NB, production=False, verify_runs=False)
    assert doc["experiments"]["matched_free_energy"]["family_size"] == 8


@pytest.mark.parametrize("override,match", [
    (dict(summary_cell=dict(engine_version="gateway_ladder_numba/2")), "engine_version"),
    (dict(summary_cell=dict(model=dict(variant="alpha", alpha=0.5, kappa=None, lam=0.1))), "model"),
    (dict(summary_cell=dict(results_dir="/elsewhere/alpha1")), "results_dir"),
    (dict(code_digest="0" * 16), "scoring code changed"),
    (dict(ref_sha="f" * 64), "reference sha256"),
])
def test_summary_must_belong_to_the_current_cell_config(tmp_path, override, match):
    """A summary made from another engine / model / run tree (e.g. left over from the reuse mode after
    make_cell_configs.py --rerun-reuse-cells), with older scoring code or another reference, is refused."""
    N1 = [2048, 512, 128]
    write_cell(tmp_path, "matched_free_energy", "alpha0", N1, {N: 0.6 for N in N1})
    write_cell(tmp_path, "matched_free_energy", "alpha1", N1, {N: 0.6 for N in N1}, **override)
    with pytest.raises(X.CrossCellError, match=match):
        X.analyse(X.load_cells(str(tmp_path / "cells")), n_boot=NB, verify_runs=False)


def test_trend_verdict_never_claims_a_holm_difference():
    Lb = L_of(np.linspace(0.4, 0.9, 32))
    r = X.contrast({s: v + math.log(1.5) for s, v in Lb.items()}, Lb, n_boot=NB)
    v = X.trend_verdict(r)
    assert "differs" not in v and "(Holm)" not in v and "b has the larger FR gain" in v and "unadjusted p" in v
    r = X.contrast(dict(Lb), dict(Lb), n_boot=NB)
    assert X.trend_verdict(r).startswith("trend (unadjusted, no Holm): equivalent; CI includes 0")

