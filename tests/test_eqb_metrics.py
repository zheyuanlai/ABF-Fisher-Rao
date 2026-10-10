"""Tests of the equal-budget ladder metrics (scripts/equal_budget/eqb_metrics.py) and the ladder analysis CLI
(scripts/equal_budget/analyze_ladder.py).

Unit tests on synthetic arrays: persistent tau (persistence, censoring, re-crossing, equality, NaN), the TV_inst
finite-N floor (exact enumeration at N <= 4, 17/18 at N = 1, fixed-seed determinism), TV_half windowing (nearest
save to t/2, run start, ties, exact window counts), the paired seed bootstrap, the censoring-aware tau comparison,
the sign test, best allocation with N re-selection, quantiles with censored values, the uniform-u identification,
the LTA reported estimator (vs core_lta) and the frozen-reference guards of both scorers.
Review fixes (2026-10-10): the gateway e_F' zero-noise floor (0.0323, e^2 = stat^2 + floor^2, strict threshold
unreachable by construction), the establishment null calibration (binomial), best allocation with tie sharing and
missing-seed resamples not counted as censored, the exact TV floor.
Integration: analyze_ladder on the REAL fixture tree made by scripts/equal_budget/make_fixtures.py (generated on
first use if absent; location EQB_FIXTURE_ROOT or the session scratchpad default), the content-keyed incremental
cache (same size + mtime but different data is rescored; a reference or engine-knob change is never served from
the cache), tau on the common budget grid, missing-N status, disjoint seeds (no crash), FR-inactive cells (the
production LTA FR rate at N = 2: zero deaths, marked), the gateway engine-/1 fallback of the realised event
fraction and loud failure on a contract violation.

  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 NUMBA_CACHE_DIR=... python -m pytest tests/test_eqb_metrics.py -q
"""
import itertools
import json
import math
import os
import shutil
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "equal_budget"))
sys.path.insert(0, os.path.join(ROOT, "src"))
import eqb_metrics as M  # noqa: E402


# ============================================================================================ tau
def test_tau_persistence_recrossing_and_equality():
    t = np.arange(1, 9) * 0.5
    u = np.arange(1, 9) / 8
    err = np.array([0.5, 0.05, 0.2, 0.1, 0.04, 0.03, 0.1, 0.02])
    r = M.tau_persistent(err, t, u, 0.1)
    # dips below at index 1, re-crosses at 2; from index 3 on every value is <= 0.1 (equality counts as met)
    assert r == dict(eps=0.1, index=3, t=2.0, u=0.5, censored=False)
    r = M.tau_persistent(err, t, u, 0.099)          # index 6 (0.1) now fails -> persistent from 7
    assert (r["index"], r["t"], r["u"], r["censored"]) == (7, 4.0, 1.0, False)
    r = M.tau_persistent(err, t, u, 1.0)            # always met -> first save
    assert (r["index"], r["t"], r["u"]) == (0, 0.5, 0.125)


def test_tau_censoring():
    t = np.arange(1, 6, dtype=float)
    u = t / 5
    for err in (np.array([0.5, 0.4, 0.3, 0.2, 0.15]),          # never meets
                np.array([0.5, 0.01, 0.01, 0.01, 0.2])):        # meets, then re-crosses at the last save
        r = M.tau_persistent(err, t, u, 0.1)
        assert r["censored"] and r["index"] == -1 and math.isinf(r["t"]) and math.isinf(r["u"])
    r = M.tau_persistent(np.array([0.01, np.nan, 0.01]), t[:3], u[:3], 0.1)   # NaN never meets eps
    assert r["index"] == 2
    assert M.persistent_first(np.array([], bool)) is None


def test_establishment_is_persistent_at_least():
    t = np.arange(1, 7, dtype=float)
    frac = np.array([0.0, 0.3, 0.1, 0.2, 0.25, 0.4])
    r = M.persistent_at_least(frac, t, t / 6, 0.5 * 0.361)
    assert r["index"] == 3 and r["t"] == 4.0 and not r["censored"]
    r = M.persistent_at_least(np.array([0.3, 0.3, 0.1]), t[:3], t[:3] / 3, 0.18)
    assert r["censored"] and math.isinf(r["u"])


def test_establishment_null_calibration_matches_binomial():
    from scipy.stats import binom
    u = np.arange(1, 201) / 200
    for N, q, want_kmin in ((8, 0.361, 2), (16, 0.361, 3), (64, 0.361, 12), (32, 0.252, 5), (128, 0.252, 17)):
        e = M.establishment_null(N, u, q)
        assert e["k_min"] == want_kmin
        pf = binom.cdf(want_kmin - 1, N, q)
        assert e["p_fail_per_save"] == pytest.approx(pf, rel=1e-12)
        assert e["p_hold_second_half"] == pytest.approx((1 - pf) ** 100, rel=1e-9)
        # null median: smallest index i with P(no failure at saves i..S-1) >= 1/2
        i = next(i for i in range(200) if (1 - pf) ** (200 - i) >= 0.5)
        assert e["null_median_u"] == u[i]
    assert M.establishment_null(16, u, 0.361)["unreliable"] and not M.establishment_null(128, u, 0.361)["unreliable"]
    assert M.establishment_null(64, u, 0.252)["unreliable"] and not M.establishment_null(128, u, 0.252)["unreliable"]
    assert M.establishment_null(2, u, 0.361)["unreliable"]                       # N <= 2 always flagged
    # p_fail > 1/2 at the last save -> the null median is censored
    assert math.isinf(M.establishment_null(2, u, 0.252)["null_median_u"])
    # Monte Carlo of the exact criterion on independent binomial draws agrees with the closed form
    rng = np.random.default_rng(7)
    X = rng.binomial(32, 0.361, size=(4000, 200)) / 32 >= 0.5 * 0.361
    p_mc = np.mean(np.all(X[:, 100:], axis=1))
    assert abs(p_mc - M.establishment_null(32, u, 0.361)["p_hold_second_half"]) < 0.03


# ============================================================================================ TV
def exact_tv_floor(N, k=18):
    """E_N[TV] by enumerating all k^N sequences (N <= 4)."""
    seqs = np.array(list(itertools.product(range(k), repeat=N)))
    counts = np.zeros((seqs.shape[0], k))
    for j in range(N):
        counts[np.arange(seqs.shape[0]), seqs[:, j]] += 1
    return float(np.mean(0.5 * np.abs(counts / N - 1.0 / k).sum(1)))


def test_tv_floor_exact_small_N_and_determinism():
    m1, se1 = M.tv_floor(1)
    assert abs(m1 - 17 / 18) < 1e-12 and se1 == 0.0
    assert abs(exact_tv_floor(2) - 289 / 324) < 1e-12                 # closed form (1/18)(17/18) + (17/18)(16/18)
    for N in (2, 3, 4):                                                # exact binomial formula = full enumeration
        assert M.tv_floor(N)[0] == pytest.approx(exact_tv_floor(N), abs=1e-13), N
    for N in (8, 64, 2048):                                            # and agrees with the Monte Carlo cross-check
        mc, se = M.tv_floor_mc(N)
        assert abs(M.tv_floor(N)[0] - mc) < 5 * se, (N, mc, se)
    a = M.tv_floor(8)
    M.tv_floor.cache_clear()
    assert M.tv_floor(8) == a
    tab = M.tv_floor_table([8, 1, 2, 8])
    assert list(tab) == [1, 2, 8] and tab[8]["mean"] == a[0]
    # the floor decreases with N and is far below 1 at large N
    assert M.tv_floor(64)[0] < M.tv_floor(8)[0] < M.tv_floor(2)[0]


def test_tv_to_uniform_and_coarse_bins():
    h = np.zeros((3, 180))
    h[0, :] = 1.0                       # uniform -> 0
    h[1, 5] = 4.0                       # one production bin -> one coarse bin -> 17/18
    tv = M.tv_to_uniform(M.coarse_bins(h))
    assert tv[0] == pytest.approx(0.0, abs=1e-15) and tv[1] == pytest.approx(17 / 18) and np.isnan(tv[2])
    c = M.coarse_bins(np.arange(180.0)[None, :])
    assert c.shape == (1, 18) and c[0, 0] == sum(range(10))


def synthetic_C(steps, per_step):
    """Cumulative coarse-resolved visitation: per_step(s) -> (180,) deposits of step s; C at each save = sum over
    steps [0, save) (gateway convention)."""
    C = np.zeros((len(steps), 180))
    acc = np.zeros(180)
    s0 = 0
    for i, s in enumerate(steps):
        for k in range(s0, s):
            acc = acc + per_step(k)
        C[i] = acc
        s0 = s
    return C


def test_tv_half_windowing():
    steps = np.array([10, 20, 40, 80, 100])

    def dep(k):           # steps < 40: two walkers in coarse bin 0; from step 40 on: spread evenly over 18 coarse bins
        d = np.zeros(180)
        if k < 40:
            d[3] = 2.0
        else:
            d[np.arange(18) * 10 + (k % 10)] = 1.0
        return d
    C = synthetic_C(steps, dep)
    r = M.tv_half(steps, C)
    assert r["lo_step"].tolist() == [0, 10, 20, 40, 40]          # nearest to 5, 10, 20, 40, 50
    assert r["lo_index"].tolist() == [-1, 0, 1, 2, 2]
    assert r["n_counts"].tolist() == [20, 20, 40, 18 * 40, 18 * 60]
    assert r["tv"][:3] == pytest.approx([17 / 18] * 3)              # windows inside the bin-0 phase
    assert r["tv"][3] == pytest.approx(0.0, abs=1e-12) and r["tv"][4] == pytest.approx(0.0, abs=1e-12)
    # tie: save at 16, target 8, candidates 0 / 4 / 12 -> 4 and 12 equidistant -> the earlier (longer window)
    steps2 = np.array([4, 12, 16])
    r2 = M.tv_half(steps2, synthetic_C(steps2, dep))
    assert r2["lo_step"].tolist() == [0, 4, 4]
    # a window mixing the phases is not uniform
    steps3 = np.array([30, 60])
    r3 = M.tv_half(steps3, synthetic_C(steps3, dep))
    assert r3["lo_step"].tolist() == [0, 30] and 0 < r3["tv"][1] < 17 / 18
    with pytest.raises(M.MetricsError):
        M.tv_half(np.array([1, 2]), np.array([np.full(180, 2.0), np.full(180, 1.0)]))


# ============================================================================================ statistics
def test_quantile_inf():
    assert M.quantile_inf([3, 1, 2], 50) == 2
    assert math.isinf(M.quantile_inf([1, np.inf], 50))
    assert M.quantile_inf([1, 2, np.inf], 25) == 1.5 and math.isinf(M.quantile_inf([1, 2, np.inf], 75))
    x = np.random.default_rng(0).normal(size=37)
    for q in (2.5, 25, 50, 75, 97.5):
        assert M.quantile_inf(x, q) == pytest.approx(np.percentile(x, q), abs=1e-14)
    assert M.quantile_inf([np.nan, 1.0, 3.0], 50) == 2.0
    assert math.isnan(M.quantile_inf([], 50))


def test_paired_bootstrap():
    seeds = list(range(100, 116))
    rng = np.random.default_rng(1)
    level = np.exp(rng.normal(0, 1.0, len(seeds)))                    # huge between-seed spread
    abf = {s: float(v) for s, v in zip(seeds, level)}
    fr = {s: float(v * 0.8 * np.exp(rng.normal(0, 0.02))) for s, v in zip(seeds, level)}
    c = M.paired_contrast(abf, fr)
    assert c["n"] == 16 and c["wins"] == 16 and c["losses"] == 0
    assert -0.23 < c["G_ci95"][0] <= c["G_median"] <= c["G_ci95"][1] < -0.17   # pairing removes the seed spread
    assert c == M.paired_contrast(abf, fr)                             # fixed bootstrap seed -> reproducible
    # the bootstrap is the textbook one: resample G with replacement, median, 2.5 / 97.5 percentiles
    G = np.array([(fr[s] - abf[s]) / abf[s] for s in seeds])
    b = np.median(G[np.random.default_rng(M.BOOT_SEED).integers(0, 16, size=(M.N_BOOT, 16))], axis=1)
    assert c["G_ci95"] == [pytest.approx(np.percentile(b, 2.5)), pytest.approx(np.percentile(b, 97.5))]
    # only common seeds with both finite enter; absolute medians alongside
    abf2 = {**abf, 999: 1.0}
    fr2 = {**fr, 100: math.inf}
    c2 = M.paired_contrast(abf2, fr2)
    assert c2["n"] == 15 and c2["n_common"] == 16 and 100 not in c2["per_seed"]
    assert c2["abf_median"] == pytest.approx(np.median([abf[s] for s in seeds[1:]]))
    # constant effect -> degenerate CI
    c3 = M.paired_contrast({s: 2.0 for s in seeds}, {s: 1.0 for s in seeds})
    assert c3["G_median"] == -0.5 and c3["G_ci95"] == [-0.5, -0.5]


def test_paired_tau_contrast_with_censoring_and_sign_test():
    inf = math.inf
    abf = {1: 0.5, 2: inf, 3: 0.3, 4: inf}
    fr = {1: 0.4, 2: 0.6, 3: inf, 4: inf}
    c = M.paired_tau_contrast(abf, fr)
    assert c["rank"]["wins"] == 2 and c["rank"]["losses"] == 1 and c["rank"]["ties"] == 1
    assert c["censored_frac_abf"] == 0.5 and c["censored_frac_fr"] == 0.5
    assert c["n_both_finite"] == 1 and c["G_median"] == pytest.approx(-0.2)
    assert c["rank"]["sign_test_p"] == 1.0
    assert math.isinf(c["abf_median"]) and math.isinf(c["fr_median"])     # medians of [0.3, 0.5, inf, inf]
    assert M.sign_test_p(10, 0) == pytest.approx(2 / 1024)
    assert M.sign_test_p(5, 5) == 1.0 and M.sign_test_p(0, 0) == 1.0
    assert M.sign_test_p(8, 2) == pytest.approx(2 * (1 + 10 + 45) / 1024)


def test_best_allocation_reselects_N():
    seeds = list(range(16))
    rng = np.random.default_rng(3)
    table = {}
    for N, ma, mf in ((1, 1.0, None), (2, 0.50, 0.70), (4, 0.300, 0.40), (8, 0.301, 0.200), (16, 0.9, 0.201)):
        table[N] = {"abf": {s: float(ma * np.exp(rng.normal(0, 0.05))) for s in seeds}}
        if mf is not None:
            table[N]["fr"] = {s: float(mf * np.exp(rng.normal(0, 0.05))) for s in seeds}
    b = M.best_allocation(table, [1, 2, 4, 8, 16], [2, 4, 8, 16], n_boot=4000)
    a, f, d = b["abf"], b["fr"], b["fr_minus_abf"]
    assert a["best_N"] in (4, 8) and f["best_N"] in (8, 16)
    # near-ties are re-selected across resamples: both N win a sizeable share
    assert a["boot_best_N_freq"][4] > 0.2 and a["boot_best_N_freq"][8] > 0.2
    assert f["boot_best_N_freq"][8] > 0.2 and f["boot_best_N_freq"][16] > 0.2
    assert abs(sum(a["boot_best_N_freq"].values()) - 1) < 1e-12 and a["boot_best_N_freq"][1] == 0.0
    assert 1 not in f["boot_best_N_freq"]                                # FR only over N >= 2
    assert a["boot_best_value_ci95"][0] < a["best_value"] < a["boot_best_value_ci95"][1]
    assert d["ci95"][1] < 0 and d["frac_resamples_fr_better"] > 0.99 and d["point"] < 0
    # censored values: an arm that never reaches eps cannot be best
    tau = {2: {"abf": {s: 0.5 for s in seeds}, "fr": {s: math.inf for s in seeds}},
           4: {"abf": {s: math.inf for s in seeds}, "fr": {s: math.inf for s in seeds}}}
    bt = M.best_allocation(tau, [2, 4], [2, 4], n_boot=500)
    assert bt["abf"]["best_N"] == 2 and bt["fr"]["best_N"] is None and bt["fr"]["boot_frac_censored"] == 1.0
    assert math.isinf(bt["fr_minus_abf"]["point"]) and bt["fr_minus_abf"]["frac_resamples_fr_better"] == 0.0


def test_best_allocation_ties_share_credit_and_missing_is_not_censored():
    seeds = list(range(16))
    rng = np.random.default_rng(4)
    # identical 3-seed data at every N: resample medians tie often (one of 3 values); by symmetry every N must win
    # 1/3 of the time -- the old argmin gave every tie to the smallest N (reviewer: 0.374 / 0.341 / 0.285)
    x = {0: 1.0, 1: 2.0, 2: 3.0}
    same = {N: {"abf": dict(x), "fr": dict(x)} for N in (1, 2, 4)}
    b = M.best_allocation(same, [1, 2, 4], [2, 4], n_boot=20000)
    assert b["abf"]["boot_tie_frac"] > 0.3 and b["abf"]["best_N_tied"] == [1, 2, 4]
    assert all(abs(v - 1 / 3) < 0.015 for v in b["abf"]["boot_best_N_freq"].values()), b["abf"]["boot_best_N_freq"]
    assert all(abs(v - 1 / 2) < 0.015 for v in b["fr"]["boot_best_N_freq"].values()), b["fr"]["boot_best_N_freq"]
    assert abs(sum(b["abf"]["boot_best_N_freq"].values()) - 1) < 1e-12
    # tau on a 1/200 grid: frequent ties at the minimum are split, the frequencies still sum to 1
    tau = {N: {"abf": {s: float(rng.integers(20, 40)) / 200 for s in seeds}} for N in (1, 2, 4)}
    bt = M.best_allocation(tau, [1, 2, 4], [], n_boot=2000)
    assert bt["abf"]["boot_tie_frac"] > 0 and abs(sum(bt["abf"]["boot_best_N_freq"].values()) - 1) < 1e-12
    # a non-censorable error with missing seeds: an N without data in a resample is skipped, never 'censored'
    part = {2: {"abf": {s: 1.0 for s in (0, 1, 2)}, "fr": {0: 0.5}},
            8: {"abf": {s: 2.0 for s in (0, 1, 2)}, "fr": {0: 0.8, 1: 0.9, 2: 0.7}}}
    bp = M.best_allocation(part, [2, 8], [2, 8], n_boot=4000)
    f = bp["fr"]
    assert f["boot_frac_censored"] == 0.0 and f["boot_frac_no_data"] == 0.0
    assert all(math.isfinite(v) for v in f["boot_best_value_ci95"])
    assert all(math.isfinite(v) for v in bp["fr_minus_abf"]["ci95"])
    assert abs(sum(f["boot_best_N_freq"].values()) - 1) < 1e-12
    # with a single N that can be empty in a resample, those resamples are 'no data' (not censored)
    only = {2: {"abf": {s: 1.0 for s in (0, 1, 2)}, "fr": {0: 0.5}}}
    bo = M.best_allocation(only, [2], [2], n_boot=4000)
    assert bo["fr"]["boot_frac_censored"] == 0.0 and 0.2 < bo["fr"]["boot_frac_no_data"] < 0.4   # (2/3)^3 = 0.30
    assert math.isfinite(bo["fr"]["boot_best_value_ci95"][1])


def test_max_transient_improvement():
    u = np.arange(1, 201) / 200
    Ea = np.tile(1.0 / np.sqrt(u), (5, 1))
    Ef = Ea.copy()
    Ef[:, 49] *= 0.4                     # 60 % better at u = 0.25
    Ef[:, -1] *= 1.5
    r = M.max_transient_improvement(u, Ea, Ef)
    assert r["max_rel"] == pytest.approx(0.6) and r["u_at_max"] == 0.25
    assert r["min_rel"] == pytest.approx(-0.5) and r["u_at_min"] == 1.0 and r["final_rel"] == pytest.approx(-0.5)


def test_uniform_indices_and_json_safe():
    import gateway_ladder_numba as G
    n = 2_000_000
    steps, _ = G.budget_save_grid(n)
    steps = np.unique(np.concatenate([steps, [40, 80000, 1600000]]))    # physical checkpoints mixed in
    iu, exact = M.uniform_indices(steps, n)
    assert exact and iu.size == 200 and np.array_equal(steps[iu] / n, np.arange(1, 201) / 200)
    _, exact2 = M.uniform_indices(np.unique(np.round(np.arange(1, 201) / 200 * 1001).astype(int)), 1001)
    assert not exact2
    with pytest.raises(M.MetricsError):
        M.uniform_indices(steps[1:][steps[1:] != n // 2], n)
    js = M.json_safe({1: [math.inf, -math.inf, math.nan, np.float64(2.5), np.int64(3), np.bool_(True)]})
    assert js == {"1": ["inf", "-inf", None, 2.5, 3, True]}
    json.dumps(js, allow_nan=False)


# ============================================================================================ scorers
def test_lta_estimator_matches_core_lta_and_switches_to_production():
    torch = pytest.importorskip("torch")
    from lta import core_lta
    rng = np.random.default_rng(5)
    C_all = rng.integers(0, 5, size=(4, 180)).astype(float)
    M_all = rng.normal(0, 3, size=(4, 180)) * C_all
    C_prod = C_all.copy()
    C_prod[:2] = 0.0                    # no production counts yet at the first two saves
    M_prod = 0.5 * M_all * (C_prod > 0)
    G, F, use = M.lta_estimator(M_all, C_all, M_prod, C_prod)
    assert use.tolist() == [False, False, True, True]
    for i in range(4):
        fs, cs = (M_prod[i], C_prod[i]) if use[i] else (M_all[i], C_all[i])
        g = core_lta.histogram_mean_force(torch.tensor(fs), torch.tensor(cs))
        f = core_lta.histogram_pmf(g, 2 * math.pi / 180)
        assert np.allclose(G[i], g.numpy(), rtol=0, atol=1e-13) and np.allclose(F[i], f.numpy(), rtol=0, atol=1e-12)


def test_lta_scorer_errors_on_the_reference_itself():
    for T in (300, 150):
        sc = M.LTAScorer(T)
        g = sc.gamma_ref
        C = np.full((1, 180), 7.0)
        res = dict(meta=dict(T_K=float(T)), M_all=np.zeros((1, 180)), C_all=np.zeros((1, 180)),
                   M_prod=g[None, :] * C, C_prod=C)
        e = sc.errors(res)
        assert e["e_Fp"][0] == pytest.approx(0.0, abs=1e-13) and e["e_Fp_proj"][0] == pytest.approx(0.0, abs=1e-13)
        # e_F of the mean-force route against the density-route F_ref = the reference's own internal consistency
        summ = json.load(open(os.path.join(M.REF_DIR, "references_summary.json")))[f"lta_T{T}"]
        assert e["e_F"][0] == pytest.approx(summ["rms_F_mf_minus_F_density"], rel=1e-10)
        # a constant offset of Gamma changes the raw F' error but not the projected one, nor e_F
        res2 = dict(res, M_prod=(g + 0.3)[None, :] * C)
        e2 = sc.errors(res2)
        assert e2["e_Fp"][0] == pytest.approx(0.3, rel=1e-10) and e2["e_Fp_proj"][0] == pytest.approx(0.0, abs=1e-12)
        assert e2["e_F"][0] == pytest.approx(e["e_F"][0], rel=1e-10)
        assert sc.reference_info["gamma_se_rms"] == pytest.approx(summ["gamma_se_rms"])
        fl = sc.reference_info["zero_noise_floor"]
        assert fl["e_Fp"] == 0.0 and fl["e_Fp_proj"] == 0.0
        assert fl["e_F"] == pytest.approx(summ["rms_F_mf_minus_F_density"], rel=1e-10)
        P = json.load(open(M.default_config_path(f"lta{T}")))
        assert all(v["reachable"] for v in M.threshold_floors(f"lta{T}", P, sc).values())
    with pytest.raises(M.MetricsError):
        M.LTAScorer(300).errors(dict(res, meta=dict(T_K=150.0)))


def test_gateway_scorer_is_the_accepted_one_and_guards_the_reference(tmp_path):
    P = json.load(open(M.default_config_path("gateway")))
    sc = M.GatewayScorer(P["engine_cfg"])
    import analyze_gateway_replica_ladder as GL
    rng = np.random.default_rng(2)
    C = rng.integers(1, 50, size=(3, 180)).astype(float)
    Mh = rng.normal(0, 1, size=(3, 180)) * C
    e = sc.errors(dict(M_all=Mh, C_all=C))
    a, b = GL.Scorer(M.gateway_cell(P["engine_cfg"]), "analytic").score(Mh, C)
    assert np.array_equal(e["e_F"], a) and np.array_equal(e["e_Fp"], b)
    assert sc.reference_info["em_floor_F_rms"] == pytest.approx(7.900546534699823e-05, rel=1e-9)
    # the zero-noise floor of the function-space e_F' (within-bin variation of F'_ref) and its exact decomposition
    fl = sc.reference_info["zero_noise_floor"]
    assert fl["e_Fp"] == pytest.approx(0.032267, abs=2e-6) and fl["e_Fp_em"] == pytest.approx(0.032272, abs=2e-6)
    assert fl["e_F"] < 1e-12 and fl["e_F_em"] < 1e-8
    assert np.allclose(e["e_Fp"] ** 2, e["e_Fp_stat"] ** 2 + fl["e_Fp"] ** 2, rtol=1e-10, atol=0)
    assert np.allclose(e["e_Fp_em"] ** 2, e["e_Fp_em_stat"] ** 2 + fl["e_Fp_em"] ** 2, rtol=1e-10, atol=0)
    assert np.all(e["e_Fp"] >= fl["e_Fp"])
    # an independent bin average (256-point midpoint rule) gives the same floor: it is not a quadrature artefact
    import eb_abffr_core as eb
    import torch
    xa = -1.8 + 0.02 * (np.arange(180)[:, None] + (np.arange(256)[None, :] + 0.5) / 256)
    fb = eb.reference_mean_force(torch.as_tensor(xa), *GL.Scorer(M.gateway_cell(P["engine_cfg"]), "analytic").p)
    Cb = np.full((1, 180), 1e15)
    _, f256 = GL.Scorer(M.gateway_cell(P["engine_cfg"]), "analytic").score(fb.numpy().mean(1)[None] * (Cb + 1), Cb)
    assert f256[0] == pytest.approx(fl["e_Fp"], rel=2e-3)
    tf = M.threshold_floors("gateway", P, sc)
    assert not tf["tau_e_Fp_strict"]["reachable"] and not tf["tau_e_Fp_em_strict"]["reachable"]
    assert tf["tau_e_Fp_mid"]["reachable"] and tf["tau_e_Fp_mid"]["stat_budget"] == pytest.approx(0.01356, abs=1e-4)
    assert all(tf[f"tau_e_F_{n}"]["reachable"] for n in M.THR_NAMES)
    with np.load(os.path.join(M.REF_DIR, "gateway_reference.npz")) as z:
        bad = {k: z[k] for k in z.files}
    bad["F_ref"] = bad["F_ref"] + 1e-6
    p = tmp_path / "bad_ref.npz"
    np.savez(p, **bad)
    with pytest.raises(M.MetricsError):
        M.GatewayScorer(P["engine_cfg"], reference_path=str(p))


# ============================================================================================ integration
import tempfile  # noqa: E402
FIXTURE_ROOT = os.environ.get("EQB_FIXTURE_ROOT", os.path.join(tempfile.gettempdir(), "eqb_fixtures"))
FIX = {"gateway": "gateway_fixture.json", "lta300": "lta_300K_fixture.json",
       "lta300_prodrate": "lta_300K_fixture_prodrate.json"}


@pytest.fixture(scope="module")
def fixture_root():
    need = [s for s, f in FIX.items() if not os.path.exists(os.path.join(FIXTURE_ROOT, "configs", f))]
    if need:
        import make_fixtures
        make_fixtures.main(["--root", FIXTURE_ROOT, "--systems", *need])
    return FIXTURE_ROOT


def expected_warnings(S):
    """Warnings a clean fixture analysis may carry: unreachable thresholds (gateway e_F' strict) and FR activity."""
    return [w for w in S["warnings"] if "UNREACHABLE BY CONSTRUCTION" not in w and "FR realised no death" not in w]


def strict_load(path):
    def bad(x):
        raise ValueError(f"non-strict JSON constant {x}")
    with open(path) as fh:
        return json.load(fh, parse_constant=bad)


@pytest.mark.parametrize("system", ["gateway", "lta300"])
def test_analyze_ladder_on_fixture(system, fixture_root, tmp_path):
    import analyze_ladder as A
    cfg = os.path.join(fixture_root, "configs", FIX[system])
    P = json.load(open(cfg))
    res_root = os.path.join(fixture_root, "results", "equal_budget_v2")
    out = str(tmp_path / "analysis")
    A.analyze(system, res_root, cfg, out, verbose=False)
    S = strict_load(os.path.join(out, "summary.json"))
    assert S["schema"] == A.SCHEMA and S["system"] == system and S["metrics_version"] == M.METRICS_VERSION
    n_runs = sum(len(P["seeds"]) * (1 if N == 1 else 2) for N in P["N_ladder"])
    assert S["cache"]["computed"] == n_runs and S["cache"]["hits"] == 0 and not expected_warnings(S)
    unreach = sorted(k for k, v in S["threshold_floors"].items() if not v["reachable"])
    assert unreach == (["tau_e_Fp_em_strict", "tau_e_Fp_strict"] if system == "gateway" else [])
    assert S["provenance"]["reference_sha256"] == M.sha256_file(M.reference_path(system, P))
    assert S["reference"]["sha256"] == S["provenance"]["reference_sha256"]
    for N in P["N_ladder"]:
        st = S["status"][str(N)]
        assert st["n_steps"] == P["B"] // N
        for m in (["abf"] if N == 1 else ["abf", "fr"]):
            assert sorted(st["methods"][m]["complete"]) == sorted(P["seeds"]) and not st["methods"][m]["missing"]
            blk = S["per_N"][str(N)][m]
            assert blk["Ibar_F"]["n"] == len(P["seeds"]) and set(blk["Ibar_F"]["per_seed"]) == {str(s) for s in P["seeds"]}
            assert blk["n_force_evals_ok"]["mean"] == 1.0
            assert blk["TV_inst_floor"]["median"] == pytest.approx(S["tv_floor"]["table"][str(N)]["mean"])
        if N >= 2:
            c = S["contrasts"][str(N)]
            for k in A.PRIMARY:
                assert c["primary"][k]["n"] == len(P["seeds"]) and len(c["primary"][k]["G_ci95"]) == 2
            for k in A.TAU_PRIMARY:
                r = c["tau"][k]["rank"]
                assert r["wins"] + r["losses"] + r["ties"] == len(P["seeds"])
            assert set(S["max_transient"][str(N)]) == set(A.TRANSIENT_KEYS[M.system_kind(system)])
            assert S["per_N"][str(N)]["fr"]["fr_deaths"]["median"] > 0          # FR really fired in the fixture
            assert not S["fr_activity"][str(N)]["any_inactive"]
            for k in ("tau_e_Fp_strict_u",):
                assert c["tau"][k]["unreachable_by_construction"] == (system == "gateway")
            if system == "gateway":                                              # realised events, not candidates
                fr = S["per_N"][str(N)]["fr"]
                assert fr["fr_events"]["per_seed"] == fr["fr_deaths"]["per_seed"]
                assert fr["fr_mean_event_frac_per_opp"]["median"] <= fr["fr_mean_candidate_frac_per_opp"]["median"]
                assert fr["fr_cap_extension"]["mean"] == 0.0
    assert S["tv_floor"]["table"]["1"]["mean"] == pytest.approx(17 / 18)
    for k in A.BEST_PRIMARY + A.BEST_SECONDARY:
        b = S["best_allocation"][k]
        assert set(b["abf"]["Ns"]) == set(P["N_ladder"]) and set(b["fr"]["Ns"]) == {n for n in P["N_ladder"] if n >= 2}
        assert b["n_boot"] == M.N_BOOT
        assert b["candidates"]["abf"]["complete_ladder"] and b["candidates"]["fr"]["complete_ladder"]
    for N in P["N_ladder"]:
        en = S["establishment_null"][str(N)]
        assert en["unreliable"] == (S["per_N"][str(N)]["abf"]["est_unreliable"]["mean"] == 1.0)
        assert en["p_fail_per_save"] == pytest.approx(S["per_N"][str(N)]["abf"]["est_null_p_fail_per_save"]["median"])
    md = open(os.path.join(out, "tables.md")).read()
    assert "## Best allocation" in md and "## Status" in md
    curves = np.load(os.path.join(out, "median_curves.npz"))
    assert curves["u"].shape == (200,) and f"N{P['N_ladder'][0]}_fr_e_F_median" in curves.files

    # independent recomputation of one run's integrated / final errors and the first tau
    N = P["N_ladder"][0]
    s = P["seeds"][0]
    path = os.path.join(res_root, P["out_dir"], f"N{N}", f"s{s}_fr.npz")
    res = M.load_run(path, system)
    u = res["save_u"]
    lin = np.isclose(u * 200, np.round(u * 200)) & (u > 0)             # the accepted ladder script's rule
    assert lin.sum() == 200
    if system == "gateway":
        import analyze_gateway_replica_ladder as GL
        eF, eFp = GL.Scorer(M.gateway_cell(P["engine_cfg"]), "analytic").score(res["M_all"], res["C_all"])
    else:
        ref = np.load(os.path.join(M.REF_DIR, "lta_T300_reference.npz"))
        use = res["C_prod"].sum(1) > 0
        Mm = np.where(use[:, None], res["M_prod"], res["M_all"])
        Cc = np.where(use[:, None], res["C_prod"], res["C_all"])
        G = np.where(Cc > 0, Mm / np.maximum(Cc, 1), 0.0)
        g0 = G - G.mean(1, keepdims=True)
        Fc = np.cumsum(g0 * 2 * np.pi / 180, 1) - 0.5 * g0 * 2 * np.pi / 180
        d = (Fc - Fc.mean(1, keepdims=True)) - ref["F_ref"]
        eF = np.sqrt(np.mean((d - d.mean(1, keepdims=True)) ** 2, 1))
        eFp = np.sqrt(np.mean((G - ref["gamma_ref"]) ** 2, 1))
    sc = S["per_N"][str(N)]["fr"]
    assert sc["Ibar_F"]["per_seed"][str(s)] == pytest.approx(float(np.mean(eF[lin])), rel=1e-12)
    assert sc["Ibar_Fp"]["per_seed"][str(s)] == pytest.approx(float(np.mean(eFp[lin])), rel=1e-12)
    assert sc["final_e_F"]["per_seed"][str(s)] == pytest.approx(float(eF[-1]), rel=1e-12)
    eps = P["thresholds"]["e_F"][2]
    ok = eF <= eps
    want = math.inf if not ok[-1] else float(u[np.flatnonzero(~ok)[-1] + 1] if (~ok).any() else u[0])
    got = sc["tau_e_F_loose_u"]["per_seed"][str(s)]
    assert (got == "inf" and math.isinf(want)) or got == pytest.approx(want)
    # tau on the common budget grid only (u = k/200 and the 24 log-spaced u; no physical checkpoints)
    E = M.engine_module(M.system_kind(system))
    bg = np.isin(res["save_step"], E.budget_save_grid(int(res["meta"]["n_steps"]))[0])
    okb = eF[bg] <= eps
    ub = u[bg]
    wantb = math.inf if not okb[-1] else float(ub[np.flatnonzero(~okb)[-1] + 1] if (~okb).any() else ub[0])
    gotb = sc["tau_bgrid_e_F_loose_u"]["per_seed"][str(s)]
    assert (gotb == "inf" and math.isinf(wantb)) or gotb == pytest.approx(wantb)
    if system == "gateway":                    # floor-free companion: e^2 = stat^2 + floor^2 at every save
        fl = S["reference"]["zero_noise_floor"]["e_Fp"]
        assert sc["final_e_Fp_stat"]["per_seed"][str(s)] == pytest.approx(math.sqrt(eFp[-1] ** 2 - fl ** 2), rel=1e-9)

    # incremental rerun: everything from the cache (keyed by CONTENT, not path / mtime)
    S2 = A.analyze(system, res_root, cfg, out, verbose=False)
    assert S2["cache"]["hits"] == n_runs and S2["cache"]["computed"] == 0
    assert S2["contrasts"] == S["contrasts"] and S2["best_allocation"] == S["best_allocation"]
    tmp_root = tmp_path / "res"
    shutil.copytree(os.path.join(res_root, P["out_dir"]), tmp_root / P["out_dir"])
    f = tmp_root / P["out_dir"] / f"N{N}" / f"s{s}_abf.npz"
    S3 = A.analyze(system, str(tmp_root), cfg, out, verbose=False)      # copied tree, same content -> cache
    assert S3["cache"]["computed"] == 0
    st = os.stat(f)
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 10 ** 9))          # touched only: still the same content
    assert A.analyze(system, str(tmp_root), cfg, out, verbose=False)["cache"]["computed"] == 0
    # same byte size and restored mtime but different data (the reviewer's case) -> rescored
    with np.load(f, allow_pickle=False) as z:
        arr = {k: z[k] for k in z.files}
    np.savez(str(f)[:-4], **arr)                                         # uncompressed, same content
    S5 = A.analyze(system, str(tmp_root), cfg, out, verbose=False)
    assert S5["cache"]["computed"] == 1
    st = os.stat(f)
    for k in ("M_all", "M_prod"):                                       # (LTA reports from M_prod)
        if k in arr:
            arr[k] = arr[k] * 1.37
    np.savez(str(f)[:-4], **arr)
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert os.stat(f).st_size == st.st_size and os.stat(f).st_mtime_ns == st.st_mtime_ns
    S6 = A.analyze(system, str(tmp_root), cfg, out, verbose=False)
    assert S6["cache"]["computed"] == 1 and S6["cache"]["hits"] == n_runs - 1
    assert S6["per_N"][str(N)]["abf"]["final_e_F"]["per_seed"][str(s)] != S["per_N"][str(N)]["abf"]["final_e_F"]["per_seed"][str(s)]


def test_analyze_ladder_missing_N_and_contract_violation(fixture_root, tmp_path):
    import analyze_ladder as A
    cfg = os.path.join(fixture_root, "configs", FIX["lta300"])
    P = json.load(open(cfg))
    res_root = os.path.join(fixture_root, "results", "equal_budget_v2")
    # a planned N with no runs: reported as missing, the rest analysed
    P2 = dict(P, N_ladder=[8] + list(P["N_ladder"]))
    cfg2 = tmp_path / "cfg_with_N8.json"
    json.dump(P2, open(cfg2, "w"))
    S = A.analyze("lta300", res_root, str(cfg2), str(tmp_path / "a1"), n_boot=500, verbose=False)
    st = S["status"]["8"]["methods"]
    assert sorted(st["abf"]["missing"]) == sorted(P["seeds"]) and sorted(st["fr"]["missing"]) == sorted(P["seeds"])
    assert S["per_N"]["8"]["abf"] is None and "8" not in S["contrasts"]
    assert any("N 8 abf: 0/3 complete" in w for w in S["warnings"])
    # a running job (checkpoint present) is reported as running
    tmp_root = tmp_path / "res"
    shutil.copytree(os.path.join(res_root, P["out_dir"]), tmp_root / P["out_dir"])
    N, s = P["N_ladder"][0], P["seeds"][0]
    victim = tmp_root / P["out_dir"] / f"N{N}" / f"s{s}_fr.npz"
    os.rename(victim, str(victim) + ".ckpt.npz")
    S = A.analyze("lta300", str(tmp_root), cfg, str(tmp_path / "a2"), n_boot=500, verbose=False)
    assert S["status"][str(N)]["methods"]["fr"]["running"] == [s]
    assert S["contrasts"][str(N)]["primary"]["Ibar_F"]["n"] == len(P["seeds"]) - 1
    os.rename(str(victim) + ".ckpt.npz", victim)
    # a contract violation fails loudly (default) or is recorded with --skip-invalid
    with np.load(victim, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files if k != "gen_ess"}
    np.savez_compressed(victim, **arrays)
    with pytest.raises(M.MetricsError, match="gen_ess"):
        A.analyze("lta300", str(tmp_root), cfg, str(tmp_path / "a3"), n_boot=500, verbose=False)
    S = A.analyze("lta300", str(tmp_root), cfg, str(tmp_path / "a4"), n_boot=500, skip_invalid=True, verbose=False)
    assert str(s) in S["status"][str(N)]["methods"]["fr"]["invalid"]
    shutil.copy2(os.path.join(res_root, P["out_dir"], f"N{N}", f"s{s}_fr.npz"), victim)
    # a run made with a different config is refused
    with np.load(tmp_root / P["out_dir"] / "N2" / f"s{s}_abf.npz", allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    c = json.loads(str(arrays["cfg_json"]))
    c["fr_rate"] = 0.2
    arrays["cfg_json"] = np.array(json.dumps(c))
    np.savez_compressed(tmp_root / P["out_dir"] / "N2" / f"s{s}_abf.npz", **arrays)
    with pytest.raises(M.MetricsError, match="fr_rate"):
        A.analyze("lta300", str(tmp_root), cfg, str(tmp_path / "a5"), n_boot=500, skip_invalid=False, verbose=False)


def test_cache_is_invalidated_by_config_reference_and_plan(fixture_root, tmp_path, monkeypatch):
    """A warm cache never serves values computed under another config, reference or engine knob."""
    import analyze_ladder as A
    cfg = os.path.join(fixture_root, "configs", FIX["lta300"])
    P = json.load(open(cfg))
    res_root = os.path.join(fixture_root, "results", "equal_budget_v2")
    out = str(tmp_path / "warm")
    S0 = A.analyze("lta300", res_root, cfg, out, n_boot=200, verbose=False)
    n_runs = S0["cache"]["computed"]
    # engine knobs changed in the config: check_plan runs on every cache hit -> refused (it used to be served)
    P2 = json.loads(json.dumps(P))
    P2["engine_cfg"]["kde_bandwidth"] = 0.5
    cfg2 = tmp_path / "knob.json"
    json.dump(P2, open(cfg2, "w"))
    with pytest.raises(M.MetricsError, match="kde_bandwidth"):
        A.analyze("lta300", res_root, str(cfg2), out, n_boot=200, verbose=False)
    # a frozen-reference change (gauge kept) invalidates every entry and changes e_F
    refs = tmp_path / "refs"
    shutil.copytree(M.REF_DIR, refs)
    z = dict(np.load(refs / "lta_T300_reference.npz"))
    z["F_ref"] = z["F_ref"] + 0.5 * np.cos(z["grid_phi"]) - np.mean(0.5 * np.cos(z["grid_phi"]))
    np.savez(refs / "lta_T300_reference", **z)
    monkeypatch.setattr(M, "REF_DIR", str(refs))
    S1 = A.analyze("lta300", res_root, cfg, out, n_boot=200, verbose=False)
    assert S1["cache"]["computed"] == n_runs and S1["cache"]["hits"] == 0
    g = lambda S: S["per_N"]["4"]["abf"]["final_e_F"]["per_seed"]["30000"]  # noqa: E731
    assert g(S1) != g(S0) and S1["provenance"]["reference_sha256"] != S0["provenance"]["reference_sha256"]


def test_analyze_disjoint_seeds_does_not_crash(fixture_root, tmp_path):
    """Both arms complete at an N but no seed in common (a mid-campaign state): contrasts with n = 0, transient n = 0."""
    import analyze_ladder as A
    cfg = os.path.join(fixture_root, "configs", FIX["gateway"])
    P = json.load(open(cfg))
    root = tmp_path / "res"
    shutil.copytree(os.path.join(fixture_root, "results", "equal_budget_v2", P["out_dir"]), root / P["out_dir"],
                    ignore=shutil.ignore_patterns("analysis"))
    d = root / P["out_dir"] / "N8"
    for f in ("s8102_abf.npz", "s8100_fr.npz", "s8101_fr.npz"):
        os.remove(d / f)
    S = A.analyze("gateway", str(root), cfg, str(tmp_path / "a"), n_boot=200, verbose=False)
    assert S["contrasts"]["8"]["primary"]["Ibar_F"]["n"] == 0
    assert all(t["n"] == 0 for t in S["max_transient"]["8"].values())
    assert S["best_allocation"]["Ibar_F"]["fr"]["Ns_short_seeds"] == {"8": 1}


def test_fr_inactive_cells_are_marked(fixture_root, tmp_path):
    """Production LTA FR rate at small N: FR realises no death, the FR arm equals ABF bitwise, G = 0 everywhere --
    reported as FR INACTIVITY (fr_activity, warnings, tables), never as equivalence."""
    import analyze_ladder as A
    cfg = os.path.join(fixture_root, "configs", FIX["lta300_prodrate"])
    P = json.load(open(cfg))
    assert P["engine_cfg"]["fr_rate"] == 0.2
    out = tmp_path / "a"
    S = A.analyze("lta300", os.path.join(fixture_root, "results", "equal_budget_v2"), cfg, str(out), n_boot=500,
                  verbose=False)
    fa = S["fr_activity"]["2"]
    assert fa["inactive"] and fa["n_zero_deaths"] == len(P["seeds"]) == fa["n_fr_seeds"]
    c = S["contrasts"]["2"]["primary"]["Ibar_F"]
    assert c["ties"] == c["n"] == len(P["seeds"]) and c["G_median"] == 0.0
    assert any("FR realised no death in any seed" in w for w in S["warnings"])
    assert "FR INACTIVE" in open(out / "tables.md").read()
    assert S["per_N"]["2"]["fr"]["fr_zero_deaths"]["mean"] == 1.0


def test_gateway_engine1_files_fall_back_to_deaths(fixture_root):
    """Without the engine-/2 realised-death histogram the event fraction comes from fr_deaths_cum and its maximum
    is unknown (NaN), never the candidate count."""
    cfg = os.path.join(fixture_root, "configs", FIX["gateway"])
    P = json.load(open(cfg))
    path = os.path.join(fixture_root, "results", "equal_budget_v2", "gateway", "N8", "s8100_fr.npz")
    res = M.load_run(path, "gateway")
    sc_v2 = M.run_metrics(res, "gateway", P, M.make_scorer("gateway", P))[1]
    res1 = {k: v for k, v in res.items() if k not in ("fr_deaths_hist", "fr_opp_death_cum")}
    sc_v1 = M.run_metrics(res1, "gateway", P, M.make_scorer("gateway", P))[1]
    assert sc_v1["fr_mean_event_frac_per_opp"] == sc_v2["fr_mean_event_frac_per_opp"]
    assert math.isnan(sc_v1["fr_max_event_frac_per_opp"]) and math.isfinite(sc_v2["fr_max_event_frac_per_opp"])
    assert sc_v2["fr_events"] == sc_v2["fr_deaths"] < sc_v2["fr_candidates"]
    k = np.arange(res["fr_deaths_hist"].size)
    assert sc_v2["fr_max_event_frac_per_opp"] == k[res["fr_deaths_hist"] > 0].max() / 8
