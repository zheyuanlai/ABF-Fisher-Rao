"""Tests of scripts/equal_budget/report_tables.py (appendix tables of the equal-budget ladders).

Tiny synthetic inputs with known answers: the MEAN-based paired bootstrap contrast (ratio of seed means, not the
mean / median of per-seed ratios; a degenerate case with an exact CI; a two-seed case whose bootstrap distribution
is known in closed form; unpaired and non-finite seeds dropped), the MEAN-based best allocation (re-selection,
tie sharing, FR restricted to its candidate N), and the per-run max transient read from a synthetic curve cache
(including the stale-cache guard against the summary).

  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 python -m pytest tests/test_report_tables.py -q
"""
import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "equal_budget"))
import report_tables as RT  # noqa: E402


# ============================================================================================ mean contrast
def test_mean_contrast_constant_ratio_has_exact_ci():
    # FR = ABF / 2 on every seed: every resample has the ratio of means exactly -0.5
    abf = {1: 1.0, 2: 2.0, 3: 3.0, 4: 10.0}
    fr = {s: v / 2 for s, v in abf.items()}
    c = RT.mean_ratio_contrast(abf, fr, n_boot=2000, seed=1)
    assert c["n"] == 4
    assert c["mean_abf"] == pytest.approx(4.0) and c["mean_fr"] == pytest.approx(2.0)
    assert c["G"] == pytest.approx(-0.5)
    assert c["ci95"] == pytest.approx([-0.5, -0.5])
    assert c["frac_boot_neg"] == 1.0 and c["frac_boot_undefined"] == 0.0


def test_mean_contrast_is_ratio_of_means_not_mean_of_ratios():
    # per-seed ratios (FR - ABF)/ABF are +1 and -1/3 (mean +1/3, median +1/3), but the seed means are equal
    abf = {7: 1.0, 8: 3.0}
    fr = {7: 2.0, 8: 2.0}
    c = RT.mean_ratio_contrast(abf, fr, n_boot=100, seed=0)
    assert c["G"] == pytest.approx(0.0, abs=1e-15)


def test_mean_contrast_two_seed_bootstrap_distribution():
    # ABF constant 2; FR in {1, 3}: a resample's FR mean is 1, 2 or 3 with prob 1/4, 1/2, 1/4, so the
    # bootstrap G is -0.5 / 0 / +0.5 with those probabilities: CI [-0.5, +0.5], P(G < 0) = 1/4
    c = RT.mean_ratio_contrast({1: 2.0, 2: 2.0}, {1: 1.0, 2: 3.0}, n_boot=20000, seed=3)
    assert c["G"] == pytest.approx(0.0, abs=1e-15)
    assert c["ci95"] == pytest.approx([-0.5, 0.5])
    assert c["frac_boot_neg"] == pytest.approx(0.25, abs=0.015)
    # deterministic for a fixed seed
    assert RT.mean_ratio_contrast({1: 2.0, 2: 2.0}, {1: 1.0, 2: 3.0}, n_boot=20000, seed=3) == c


def test_mean_contrast_drops_unpaired_and_nonfinite_seeds():
    abf = {1: 1.0, 2: 1.0, 3: 5.0, 4: math.inf}
    fr = {1: 0.5, 2: 0.5, 4: 1.0, 5: 9.0}          # seed 3 lacks FR, 5 lacks ABF, 4 is censored in ABF
    c = RT.mean_ratio_contrast(abf, fr, n_boot=500, seed=0)
    assert (c["n"], c["n_common"]) == (2, 3)
    assert c["G"] == pytest.approx(-0.5)
    empty = RT.mean_ratio_contrast({1: 1.0}, {2: 1.0}, n_boot=10)
    assert empty["n"] == 0 and math.isnan(empty["G"])


# ============================================================================================ mean best allocation
def test_mean_best_allocation_known_minimum_and_restricted_fr():
    seeds = range(6)
    table = {1: {"abf": {s: 1.0 for s in seeds}},                                   # ABF only at N = 1
             2: {"abf": {s: 3.0 for s in seeds}, "fr": {s: 2.0 for s in seeds}},
             4: {"abf": {s: 2.0 + 0.1 * s for s in seeds}, "fr": {s: 4.0 for s in seeds}}}
    b = RT.mean_best_allocation(table, [1, 2, 4], [2, 4], n_boot=1000, seed=5)
    a, f = b["abf"], b["fr"]
    assert (a["best_N"], a["best_value"]) == (1, 1.0)
    assert a["boot_best_N_freq"][1] == 1.0 and a["boot_best_value_ci95"] == pytest.approx([1.0, 1.0])
    assert a["means"][4] == pytest.approx(2.25)
    assert (f["best_N"], f["best_value"]) == (2, 2.0) and f["Ns"] == [2, 4]
    d = b["fr_minus_abf"]
    assert d["point"] == pytest.approx(1.0) and d["point_rel"] == pytest.approx(1.0)
    assert d["ci95"] == pytest.approx([1.0, 1.0]) and d["frac_resamples_fr_better"] == 0.0


def test_mean_best_allocation_ties_share_credit():
    # two N with identical constant values: every resample is a tie, each N gets half the credit
    table = {2: {"abf": {s: 1.0 for s in range(4)}}, 4: {"abf": {s: 1.0 for s in range(4)}},
             8: {"abf": {s: 2.0 for s in range(4)}}}
    a = RT.mean_best_allocation(table, [2, 4, 8], [], n_boot=200, seed=0)["abf"]
    assert a["best_N_tied"] == [2, 4]
    assert a["boot_tie_frac"] == 1.0
    assert a["boot_best_N_freq"] == pytest.approx({2: 0.5, 4: 0.5, 8: 0.0})


def test_mean_best_allocation_reselects_n_in_resamples():
    # point: N = 2 has mean 3.2/3 = 1.067 (an outlier seed), N = 4 has 1.0 -> best N = 4.  In a resample N = 2
    # wins iff its outlier seed is drawn 0 times (mean 0.1; one copy already gives 1.067 > 1): P = (2/3)^3 = 8/27
    table = {2: {"abf": {0: 0.1, 1: 0.1, 2: 3.0}}, 4: {"abf": {0: 1.0, 1: 1.0, 2: 1.0}}}
    a = RT.mean_best_allocation(table, [2, 4], [], n_boot=20000, seed=11)["abf"]
    assert a["best_N"] == 4 and a["best_value"] == pytest.approx(1.0)
    assert a["means"][2] == pytest.approx(3.2 / 3)
    assert a["boot_best_N_freq"][2] == pytest.approx(8 / 27, abs=0.01)
    assert a["boot_best_N_freq"][4] == pytest.approx(19 / 27, abs=0.01)
    assert a["boot_tie_frac"] == 0.0


# ============================================================================================ per-run transient
def _write_run(path, curves, n_uniform=200):
    S = n_uniform + 3                                   # 3 extra early saves before the uniform grid
    u = np.concatenate([[0.001, 0.002, 0.003], np.arange(1, n_uniform + 1) / n_uniform])
    np.savez(path, save_u=u, uniform_index=np.arange(3, S), **{k: np.concatenate([[9.0, 9.0, 9.0], v])
                                                               for k, v in curves.items()})


def test_per_run_transient_known_max_and_stale_guard(tmp_path):
    n = 200
    u = np.arange(1, n + 1) / n
    d = tmp_path / "N4"
    d.mkdir()
    ea = np.ones(n)
    ef = np.ones(n)
    ef[19] = 0.25                                        # rel = 0.75 at u = 0.1 (index 19), seed 1
    _write_run(d / "s1_abf.npz", {"e_F": ea})
    _write_run(d / "s1_fr.npz", {"e_F": ef})
    ef2 = np.full(n, 0.5)                               # rel = 0.5 everywhere, first max at u = 0.005, seed 2
    _write_run(d / "s2_abf.npz", {"e_F": ea})
    _write_run(d / "s2_fr.npz", {"e_F": ef2})
    out = RT.per_run_transient(str(tmp_path), "4", 100.0, [1, 2, 3], ["e_F"])["e_F"]   # seed 3 not cached
    assert out["seeds"] == [1, 2]
    assert out["max_rel"] == pytest.approx([0.75, 0.5])
    assert out["u_at_max"] == pytest.approx([0.1, 0.005]) and out["t_at_max"] == pytest.approx([10.0, 0.5])
    assert out["final_rel"] == pytest.approx([0.0, 0.5])

    def summ(ibar_fr):
        blk = lambda v: {"per_seed": {"1": v}}               # noqa: E731
        return {"per_N": {"4": {"abf": {"Ibar_F": blk(1.0), "final_e_F": blk(1.0)},
                                "fr": {"Ibar_F": blk(ibar_fr), "final_e_F": blk(1.0)}}}}
    good = float(np.mean(ef))
    RT.per_run_transient(str(tmp_path), "4", 100.0, [1], ["e_F"], summ(good))           # consistent: no error
    with pytest.raises(SystemExit, match="stale run_metrics cache"):
        RT.per_run_transient(str(tmp_path), "4", 100.0, [1], ["e_F"], summ(good + 1e-3))
    assert np.allclose(u[19], 0.1)
