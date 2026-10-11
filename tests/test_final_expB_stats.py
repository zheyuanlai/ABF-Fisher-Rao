"""Tests of the Experiment B estimators and decision rule (scripts/final_campaign/expB_stats.py), on censored
fixtures, before any production data exist (docs/final_campaign/PREREGISTRATION.md section 2.7).

Covered: inf-aware medians (vector and per-row, vs eqb_metrics.quantile_inf), every ratio convention (finite/inf = 0,
inf/finite = inf, inf/inf undefined, 0/0, x/0, NaN), the bootstrap (fixed stream, unit order independent of dict
order, paired index sharing between ABF 512 and FR 512, independent draws elsewhere, missing seeds), the IUT decision
on constructed cases (FR advantage, ABF advantage, unresolved, censored arms, > 2.5 % undefined resamples, the
mutual exclusivity), rho_best re-selection, the Gehan-Wilcoxon test (Mann-Whitney identity without ties, inf ranked
last), the paired sign test, the budget-artefact label, W* and the effective parallelism.

  CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_final_expB_stats.py -q -p no:cacheprovider
"""
import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "final_campaign"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "equal_budget"))
import expB_stats as S  # noqa: E402
import eqb_metrics as M  # noqa: E402

INF = math.inf
NB = 2000          # fewer resamples where the exact count does not matter (speed)


# ============================================================================================ medians
def test_median_inf_conventions():
    assert S.median_inf([1.0, 2.0, INF]) == 2.0
    assert S.median_inf([1.0, INF]) == INF
    assert S.median_inf([1.0, 2.0, 3.0, INF]) == 2.5
    assert S.median_inf([1.0, 2.0, INF, INF]) == INF
    assert S.median_inf({1: 3.0, 2: 1.0, 3: 2.0}) == 2.0
    assert S.median_inf([INF, INF]) == INF


def test_row_quantile_matches_quantile_inf():
    rng = np.random.default_rng(1)
    A = rng.exponential(size=(300, 9))
    A[rng.random(A.shape) < 0.25] = INF
    A[rng.random(A.shape) < 0.15] = np.nan
    for q in (2.5, 25, 50, 97.5):
        got = S.row_quantile_inf(A, q)
        want = np.array([M.quantile_inf(r, q) if np.any(~np.isnan(r)) else np.nan for r in A])
        assert np.array_equal(np.isnan(got), np.isnan(want))
        ok = ~np.isnan(want)
        assert np.all((got[ok] == want[ok]) | np.isclose(got[ok], want[ok], rtol=1e-12, atol=0))


# ============================================================================================ ratios
def test_ratio_every_convention():
    assert S.ratio(1.0, INF) == 0.0                       # finite / inf
    assert S.ratio(INF, 1.0) == INF                       # inf / finite
    assert math.isnan(S.ratio(INF, INF))                  # inf / inf undefined
    assert S.ratio(2.0, 4.0) == 0.5
    assert math.isnan(S.ratio(0.0, 0.0))
    assert S.ratio(1.0, 0.0) == INF
    assert math.isnan(S.ratio(np.nan, 1.0)) and math.isnan(S.ratio(1.0, np.nan))
    r = S.ratio(np.array([1.0, INF, INF, 3.0]), np.array([INF, 2.0, INF, 1.5]))
    assert r[0] == 0.0 and r[1] == INF and math.isnan(r[2]) and r[3] == 2.0


def test_contrast_counts_undefined_never_drops_silently():
    bf = np.array([1.0, INF, INF, 2.0] * 25)
    ba = np.array([2.0, INF, 1.0, 4.0] * 25)
    c, r = S.contrast(bf, ba, 1.5, 3.0)
    assert c["n_undefined"] == 25 and c["frac_undefined"] == 0.25 and c["unresolved"]
    assert c["point"] == 0.5
    lo, hi = c["ci95"]
    assert lo == 0.5 and hi == INF                         # inf/finite resamples kept as inf


# ============================================================================================ bootstrap
def _vals(rng, n, scale, cens=0.0, seeds=None):
    seeds = seeds or list(range(8400, 8400 + n))
    x = rng.exponential(scale, size=len(seeds)) + scale
    x[rng.random(len(seeds)) < cens] = INF
    return {s: float(v) for s, v in zip(seeds, x)}


def test_bootstrap_paired_index_sharing_and_independence():
    rng = np.random.default_rng(3)
    a512 = _vals(rng, 30, 10.0)
    fr512 = {s: 2.0 * v for s, v in a512.items()}           # exactly 2x per seed
    a16 = dict(a512)                                        # same values, but unpaired
    boot, info = S.bootstrap_medians({"abf512": a512, "fr512": fr512, "abf16": a16},
                                     pairs=[("abf512", "fr512")], n_boot=NB)
    assert np.array_equal(boot["fr512"], 2.0 * boot["abf512"])      # one shared index draw
    assert not np.array_equal(boot["abf16"], boot["abf512"])        # independent draw for the cross-N arm
    assert info["abf512"]["paired"] and info["fr512"]["paired"] and not info["abf16"]["paired"]
    # deterministic, and independent of the dict insertion order
    boot2, _ = S.bootstrap_medians({"abf16": a16, "fr512": fr512, "abf512": a512},
                                   pairs=[("fr512", "abf512")], n_boot=NB)
    for k in boot:
        assert np.array_equal(boot[k], boot2[k])


def test_bootstrap_paired_with_missing_seed():
    a = {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0}
    f = {1: 1.5, 2: 2.5, 3: 3.5}                            # seed 4 missing in FR
    boot, info = S.bootstrap_medians({"abf512": a, "fr512": f}, pairs=[("abf512", "fr512")], n_boot=NB)
    assert info["fr512"]["n_seeds"] == 3 and info["abf512"]["n_seeds"] == 4
    assert np.all(np.isfinite(boot["fr512"]) | np.isnan(boot["fr512"]))
    # a resample that drew only seed 4 has no FR value (NaN), never a fabricated one
    assert np.all(boot["fr512"][~np.isnan(boot["fr512"])] >= 1.5)


def test_boot_ci_median_keeps_inf():
    x = [1.0, 2.0, 3.0, INF, INF]
    med, lo, hi = S.boot_ci_median(x, n_boot=NB)
    assert med == 3.0 and hi == INF and lo <= 2.0


# ============================================================================================ IUT
def _arms(spec, n=40, seed=7):
    rng = np.random.default_rng(seed)
    return {a: _vals(rng, n, sc, cens) for a, (sc, cens) in spec.items()}


def test_iut_fr_advantage():
    v = _arms({"fr512": (5.0, 0), "abf512": (20.0, 0), "abf16": (15.0, 0), "abf64": (18.0, 0)})
    r = S.compare_arms(v, ["fr512"], ["abf16", "abf64", "abf512"], n_boot=NB, gehan_perm=500)
    d = r["decision"]
    assert d["decision"] == "FR advantage" and d["fr_advantage_arms"] == ["fr512"]
    assert all(c["upper_FR_level"] < 1 for c in d["contrasts"].values())
    assert r["tests"]["fr512|abf512"]["kind"] == "paired sign test"
    assert r["tests"]["fr512|abf16"]["kind"].startswith("Gehan")


def test_iut_abf_advantage_needs_one_abf_arm_beating_every_fr_arm():
    v = _arms({"fr512": (30.0, 0), "fr128": (25.0, 0), "abf16": (5.0, 0), "abf512": (60.0, 0)})
    d = S.compare_arms(v, ["fr128", "fr512"], ["abf16", "abf512"], n_boot=NB, gehan_perm=0)["decision"]
    assert d["decision"] == "ABF advantage" and d["abf_advantage_arms"] == ["abf16"]
    assert d["levels"]["abf_side"] == pytest.approx(1 - 0.05 / 2)
    assert d["levels"]["fr_side"] == pytest.approx(1 - 0.05 / 2)


def test_iut_fr_must_beat_every_abf_arm():
    # FR beats abf512 clearly but not abf16 -> no FR advantage; abf16 does not beat FR either -> unresolved
    v = _arms({"fr512": (10.0, 0), "abf512": (40.0, 0), "abf16": (10.0, 0)})
    d = S.compare_arms(v, ["fr512"], ["abf16", "abf512"], n_boot=NB, gehan_perm=0)["decision"]
    assert d["decision"] == "unresolved"
    assert d["contrasts"]["fr512|abf512"]["fr_better"] and not d["contrasts"]["fr512|abf16"]["fr_better"]


def test_iut_censored_arms():
    # FR all censored, ABF finite -> rho = inf in every resample -> ABF advantage
    v = _arms({"fr512": (5.0, 1.0), "abf16": (10.0, 0.0)})
    d = S.compare_arms(v, ["fr512"], ["abf16"], n_boot=NB, gehan_perm=0)["decision"]
    assert d["decision"] == "ABF advantage"
    assert d["contrasts"]["fr512|abf16"]["point"] == INF
    # an ABF arm fully censored and the other ABF arm slower than FR -> FR advantage (finite/inf = 0)
    v = _arms({"fr512": (5.0, 0.0), "abf16": (10.0, 1.0), "abf512": (30.0, 0.0)})
    d = S.compare_arms(v, ["fr512"], ["abf16", "abf512"], n_boot=NB, gehan_perm=0)["decision"]
    assert d["decision"] == "FR advantage"
    assert d["contrasts"]["fr512|abf16"]["point"] == 0.0
    # both censored -> inf/inf undefined in every resample -> unresolved, never an advantage
    v = _arms({"fr512": (5.0, 1.0), "abf16": (10.0, 1.0)})
    d = S.compare_arms(v, ["fr512"], ["abf16"], n_boot=NB, gehan_perm=0)["decision"]
    assert d["decision"] == "unresolved" and d["unresolved_contrasts"] == ["fr512|abf16"]
    assert d["contrasts"]["fr512|abf16"]["frac_undefined"] == 1.0


def test_iut_more_than_2p5_percent_undefined_is_unresolved_even_if_ci_excludes_1():
    # ~half censored in both arms: many resamples have both medians inf (undefined); the defined ones all favour FR
    seeds = list(range(40))
    fr = {s: (1.0 if s % 2 == 0 else INF) for s in seeds}
    ab = {s: (50.0 if s % 2 == 0 else INF) for s in seeds}
    boot, _ = S.bootstrap_medians({"fr512": fr, "abf16": ab}, n_boot=NB)
    med = {"fr512": S.median_inf(fr), "abf16": S.median_inf(ab)}
    d = S.iut_decision(boot, med, ["fr512"], ["abf16"])
    c = d["contrasts"]["fr512|abf16"]
    assert c["frac_undefined"] > 0.025 and c["unresolved"] and d["decision"] == "unresolved"


def test_iut_exclusivity_and_levels_bonferroni():
    rng = np.random.default_rng(11)
    for trial in range(20):
        spec = {f"fr{n}": (rng.uniform(1, 30), rng.uniform(0, 0.3)) for n in (128, 512)}
        spec.update({f"abf{n}": (rng.uniform(1, 30), rng.uniform(0, 0.3)) for n in (2, 64, 512)})
        v = _arms(spec, n=20, seed=trial)
        d = S.compare_arms(v, ["fr128", "fr512"], ["abf2", "abf64", "abf512"], n_boot=500, gehan_perm=0)["decision"]
        assert not (d["fr_advantage_arms"] and d["abf_advantage_arms"])
        for c in d["contrasts"].values():
            if not math.isnan(c["lower_ABF_level"]):
                assert c["lower_ABF_level"] <= c["upper_FR_level"]
        assert d["levels"]["fr_side"] == pytest.approx(0.975) and d["levels"]["abf_side"] == pytest.approx(1 - 0.05 / 3)


def test_missing_arm_blocks_any_advantage():
    v = _arms({"fr512": (5.0, 0), "abf512": (20.0, 0)})
    v["abf16"] = {}                                         # every run of abf16 flagged / missing
    r = S.compare_arms(v, ["fr512"], ["abf16", "abf512"], n_boot=NB, gehan_perm=0)
    assert r["arms_without_data"] == ["abf16"] and r["decision"]["decision"] == "unresolved"
    assert r["decision"]["contrasts"]["fr512|abf512"]["fr_better"]


def test_rho_best_reselects_both_sides():
    v = _arms({"fr512": (10.0, 0), "fr128": (8.0, 0), "abf16": (12.0, 0), "abf512": (40.0, 0)}, n=30)
    r = S.compare_arms(v, ["fr128", "fr512"], ["abf16", "abf512"], n_boot=NB, gehan_perm=0)
    rb = r["rho_best"]
    med = {a: S.median_inf(v[a]) for a in v}
    assert rb["point"] == pytest.approx(min(med["fr512"], med["fr128"]) / min(med["abf16"], med["abf512"]))
    assert sum(rb["best_fr_freq"].values()) == pytest.approx(1.0)
    assert rb["best_abf_freq"]["abf16"] > 0.99
    boot = r["_boot"]
    manual = S.ratio(np.minimum(boot["fr512"], boot["fr128"]), np.minimum(boot["abf16"], boot["abf512"]))
    assert rb["frac_below_1"] == pytest.approx(np.mean(manual < 1))


# ============================================================================================ tests
def test_gehan_equals_mann_whitney_without_ties():
    from scipy.stats import mannwhitneyu
    rng = np.random.default_rng(5)
    x, y = rng.normal(0, 1, 23), rng.normal(0.5, 1, 31)
    g = S.gehan_test(x, y, n_perm=2000)
    U = float(np.sum(x[:, None] > y[None, :]))
    assert g["G"] == pytest.approx(2 * U - x.size * y.size)
    mw = mannwhitneyu(x, y, alternative="two-sided", use_continuity=False, method="asymptotic")
    assert g["p_normal"] == pytest.approx(mw.pvalue, rel=1e-9)
    assert 0 < g["p_perm"] <= 1


def test_gehan_inf_ranked_last():
    g = S.gehan_test([1.0, INF, INF], [2.0, 3.0, INF], n_perm=0)
    assert g["G"] == 1.0                     # x>y: 4 pairs, x<y: 3 pairs, inf-inf: 2 ties
    assert g["prob_x_less"] == pytest.approx((3 + 0.5 * 2) / 9)
    g = S.gehan_test([INF, INF], [INF, INF], n_perm=10)
    assert g["G"] == 0 and g["p_normal"] == 1.0


def test_paired_sign_test_censoring():
    abf = {1: 10.0, 2: INF, 3: 5.0, 4: INF, 5: 7.0}
    fr = {1: 8.0, 2: 9.0, 3: 6.0, 4: INF, 5: 7.0}
    t = S.paired_sign_test(abf, fr)
    assert (t["fr_wins"], t["fr_losses"], t["ties"]) == (2, 1, 2)
    assert t["p"] == pytest.approx(M.sign_test_p(2, 1))


# ============================================================================================ budget, W*, cost
def test_budget_artefact_label():
    # X censored (median inf) and completed at ~30 s while Y needs ~100 s -> not comparable
    b = S.budget_artefact([INF, INF, 20.0], [30.0, 31.0, 29.0], [100.0, 110.0, 90.0], [500.0, 500.0, 500.0])
    assert b["label"] == "not comparable (budget artefact)" and b["x_limited"] and not b["y_limited"]
    # X censored but ran for 300 s, longer than Y's W -> comparable (X genuinely slower)
    b = S.budget_artefact([INF, INF, 200.0], [300.0, 300.0, 300.0], [100.0, 110.0, 90.0], [500.0] * 3)
    assert b["label"] == "comparable"
    # X has one censored run but a finite median: arm not censored -> comparable; partial fraction reported
    b = S.budget_artefact([INF, 20.0, 25.0], [30.0, 30.0, 30.0], [100.0, 110.0, 90.0], [500.0] * 3)
    assert b["label"] == "comparable" and b["frac_runs_budget_limited_x"] == pytest.approx(1 / 3)
    # both censored -> the comparator's W is inf, so the censored arm's completion is shorter: not comparable
    b = S.budget_artefact([INF] * 3, [30.0] * 3, [INF] * 3, [40.0] * 3)
    assert b["label"] == "not comparable (budget artefact)"


def test_w_star_and_effective_parallelism():
    ws, arm, meds = S.w_star({"abf16": [300.0, 310.0, 305.0], "fr512": [40.0, 42.0, 41.0, 43.0]})
    assert ws == 41.5 and arm == "fr512"
    e = S.effective_parallelism(c_serial=1e-7, walker_steps=1e9, wall=10.0, threads=16)
    assert e["speedup"] == pytest.approx(10.0) and e["efficiency"] == pytest.approx(10.0 / 16)


def test_paired_diff_and_fraction():
    d = S.paired_diff_ci({1: 10.0, 2: 20.0, 3: 30.0}, {1: 12.0, 2: 21.0, 3: 35.0, 4: 1.0}, n_boot=NB)
    assert d["n"] == 3 and d["median"] == 2.0
    f, lo, hi = S.boot_ci_fraction([1, 1, 0, 1], n_boot=NB)
    assert f == 0.75 and 0 <= lo <= f <= hi <= 1
