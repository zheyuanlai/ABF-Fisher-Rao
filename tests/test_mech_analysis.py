"""Known-answer tests of scripts/mechanism/mech_analysis.py (mechanism-specific analysis of Experiments I and II,
docs/mechanism/SCIENTIFIC_PLAN.md sections 4, 6, 7) on synthetic inputs: the region / eval-bin partition, the D2
statistic (exact signed deviation and RMS bias without noise, unbiased debiasing under seed-level noise, the seed-cluster
variance against a brute-force computation, the bootstrap with duplicated seeds, seeds -- never deposits -- as the
units), paired contrasts, class fractions, D1 normalisation, the standardised conditional y law, the barrier
decomposition, the exact within-bin force variance, and an end-to-end run on fabricated run files (missing files
reported, known D2 answer through the pipeline, every figure legible, absent validation / cross-cell inputs degrade
gracefully)."""
import copy
import json
import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "mechanism"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "equal_budget"))
sys.path.insert(0, os.path.join(ROOT, "src"))
import mech_analysis as MA  # noqa: E402
import gateway_family_validation as GF  # noqa: E402

NB, NA = MA.NB, MA.N_AGE


# ------------------------------------------------------------------------------------------------ geometry
def test_regions_partition_and_eval_bins():
    R = MA.REGIONS
    tot = R["left"].astype(int) + R["gate"].astype(int) + R["right"].astype(int)
    assert np.all(tot == 1)
    assert R["left"].sum() == 65 and R["gate"].sum() == 50 and R["right"].sum() == 65
    assert MA.EVAL.sum() == 150 and np.flatnonzero(MA.EVAL)[[0, -1]].tolist() == [15, 164]
    g = MA.DEV_REGIONS
    assert np.array_equal(g["gate_neg"] | g["gate_pos"], g["gate"]) and not np.any(g["gate_neg"] & g["gate_pos"])
    assert MA.coarse(np.ones(NB)).tolist() == [5.0] * 36


# ------------------------------------------------------------------------------------------------ D2 core
def _bins_ref(seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, NB), np.full(NB, 0.5)


def _noiseless(S=6, seed=1):
    """Every deposit of class c in bin j equals r_j + b_cj exactly, identical in every seed."""
    rng = np.random.default_rng(seed)
    r, etot = _bins_ref(seed)
    b = np.zeros((NA, NB))
    b[0, MA.DEV_REGIONS["gate_neg"]] = 0.3
    b[0, MA.DEV_REGIONS["gate_pos"]] = -0.1
    b[1] = 0.05
    nper = rng.integers(1, 50, size=(NA, NB)).astype(float)
    n = np.repeat(nper[None], S, 0)
    f = r[None, :] + b
    s1 = n * f[None]
    s2 = n * f[None] ** 2
    cat = lambda a: np.concatenate([a, a.sum(1, keepdims=True)], 1)  # noqa: E731
    return cat(n), cat(s1), cat(s2), r, etot, b, nper


def test_d2_noiseless_known_answer():
    n, s1, s2, r, etot, b, nper = _noiseless()
    calc = MA.D2Calc(n, s1, s2, r, etot)
    st = calc.stats(np.ones((1, n.shape[0])))
    ev = MA.EVAL
    for c in range(NA):
        w = nper[c] * ev
        want = math.sqrt(np.sum(w * b[c] ** 2) / np.sum(w))
        assert st["B"][0, c] == pytest.approx(want, abs=1e-10)
        assert st["noise2"][0, c] == pytest.approx(0.0, abs=1e-18)
        for i, R in enumerate(MA.DEV_REGIONS):
            mk = MA.DEV_REGIONS[R] & ev
            assert st["dev"][0, c, i] == pytest.approx(np.sum(nper[c, mk] * b[c, mk]) / np.sum(nper[c, mk]), abs=1e-10)
    # contributions of the classes add up to the all-deposit deviation
    assert np.allclose(st["contrib"][0, :NA].sum(0), st["dev"][0, MA.ALL], atol=1e-12)
    # zero within-bin variance of every class -> variance ratio 0 (the pooled 'all' class mixes class means: > 0)
    assert np.allclose(st["var_ratio"][0, :NA], 0.0, atol=1e-9)
    assert np.nanmax(st["var_ratio"][0, MA.ALL]) > 1e-4
    # the bootstrap of identical seeds is degenerate at the point estimate
    out = MA.d2_summarise(calc, 300)
    assert np.allclose(out["B_lo"], out["B"], atol=1e-9) and np.allclose(out["B_hi"], out["B"], atol=1e-9)


def test_cluster_variance_equals_bruteforce_also_with_duplicated_seeds():
    rng = np.random.default_rng(5)
    S = 7
    r, etot = _bins_ref(2)
    n = rng.integers(0, 30, size=(S, NA + 1, NB)).astype(float)
    n[:, :, :3] = 0.0                                  # empty bins stay NaN
    m = r[None, None, :] + rng.normal(0, 0.2, size=(S, NA + 1, NB))
    s1 = n * m
    s2 = n * (m ** 2 + 0.3)
    calc = MA.D2Calc(n, s1, s2, r, etot)
    K = MA.boot_counts(S, 25, seed=11)
    st = calc.stats(K)
    for b in range(K.shape[0]):
        rows = np.repeat(np.arange(S), K[b].astype(int))           # replicated seeds of this resample
        N_, S_, Q_ = n[rows].sum(0), s1[rows].sum(0), s2[rows].sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mm = S_ / N_
            R2 = ((s1[rows] - mm[None] * n[rows]) ** 2).sum(0)
            G = (n[rows] > 0).sum(0).astype(float)
            Gd = (n[np.unique(rows)] > 0).sum(0)
            V = np.where(Gd >= 2, G / (G - 1) * R2 / N_ ** 2, np.nan)
            SSW = Q_ - S_ ** 2 / N_
        use = MA.EVAL[None, :] & (N_ > 0) & (Gd >= 2)
        delta = mm - r[None, :]
        for c in range(NA + 1):
            u = use[c]
            if not u.any():
                continue
            w = N_[c, u]
            want = np.sum(w * (delta[c, u] ** 2 - V[c, u])) / np.sum(w)
            assert st["B2"][b, c] == pytest.approx(want, rel=1e-9, abs=1e-12)
            mk = MA.REGIONS["gate"] & MA.EVAL & (N_[c] > 0)
            if mk.any():
                e = (N_[c, mk] * etot[mk]).sum()
                assert st["var_ratio"][b, c, 1] == pytest.approx(SSW[c, mk].sum() / e, rel=1e-8)


def _noisy(S, b_gate, sigma, n_scale, seed):
    """Seed-level noise only (deposits within a seed perfectly correlated, the worst case for deposit counting):
    every deposit of seed s, class c, bin j equals r_j + b_cj + sigma * eps_scj."""
    rng = np.random.default_rng(seed)
    r, etot = _bins_ref(3)
    b = np.zeros((NA + 1, NB))
    b[0, MA.REGIONS["gate"]] = b_gate
    n = np.full((S, NA + 1, NB), float(n_scale))
    eps = rng.normal(0, 1, size=(S, NA + 1, NB))
    f = r[None, None, :] + b[None] + sigma * eps
    return n, n * f, n * f ** 2, r, etot, b


def test_debiased_rms_is_unbiased_where_the_raw_rms_is_not():
    vals_B2, vals_raw2, vals_B2_bias = [], [], []
    for k in range(60):
        n, s1, s2, r, etot, b = _noisy(16, 0.0, 0.2, 100, seed=100 + k)
        st = MA.D2Calc(n, s1, s2, r, etot).stats(np.ones((1, 16)))
        vals_B2.append(st["B2"][0, 0])
        vals_raw2.append(st["raw2"][0, 0])
        n, s1, s2, r, etot, b = _noisy(16, 0.15, 0.2, 100, seed=500 + k)
        st = MA.D2Calc(n, s1, s2, r, etot).stats(np.ones((1, 16)))
        vals_B2_bias.append(st["B2"][0, 0])
    v = np.array(vals_B2)
    # E[raw^2] = sigma^2 / S = 0.0025 > 0 ; debiased mean consistent with 0 (3 se)
    assert np.mean(vals_raw2) == pytest.approx(0.2 ** 2 / 16, rel=0.1)
    assert abs(v.mean()) < 3 * v.std(ddof=1) / math.sqrt(v.size)
    # with a true gate bias 0.15 on the gate's share of the eval bins: E[B^2] = 0.15^2 * 50/150
    vb = np.array(vals_B2_bias)
    assert abs(vb.mean() - 0.15 ** 2 * 50 / 150) < 3 * vb.std(ddof=1) / math.sqrt(vb.size)


def test_seeds_not_deposits_are_the_units():
    """Multiplying every seed's deposits by 100 (more correlated copies of the same information) changes neither the
    estimate nor its CI; more SEEDS shrink the CI."""
    n, s1, s2, r, etot, b = _noisy(12, 0.1, 0.3, 50, seed=7)
    a = MA.d2_summarise(MA.D2Calc(n, s1, s2, r, etot), 2000)
    c = MA.d2_summarise(MA.D2Calc(100 * n, 100 * s1, 100 * s2, r, etot), 2000)
    for k in ("B", "B_lo", "B_hi", "dev", "dev_lo", "dev_hi"):
        assert np.allclose(np.asarray(a[k], float), np.asarray(c[k], float), rtol=1e-8, equal_nan=True)
    n2, s12, s22, r2, et2, _ = _noisy(48, 0.1, 0.3, 50, seed=8)
    w = MA.d2_summarise(MA.D2Calc(n2, s12, s22, r2, et2), 2000)
    gate = list(MA.DEV_REGIONS).index("gate")
    assert (w["dev_hi"][0][gate] - w["dev_lo"][0][gate]) < 0.75 * (a["dev_hi"][0][gate] - a["dev_lo"][0][gate])


def test_bootstrap_ci_of_the_signed_deviation_covers_the_truth():
    gate = list(MA.DEV_REGIONS).index("gate")
    hits = 0
    for k in range(40):
        n, s1, s2, r, etot, b = _noisy(20, 0.1, 0.5, 10, seed=900 + k)
        out = MA.d2_summarise(MA.D2Calc(n, s1, s2, r, etot), 1000, seed=k)
        hits += out["dev_lo"][0][gate] <= 0.1 <= out["dev_hi"][0][gate]
    assert hits >= 34                                  # nominal 95 %: 38/40 expected


def test_paired_contrasts_identity_and_known_shift():
    n, s1, s2, r, etot, b = _noisy(10, 0.1, 0.2, 20, seed=3)
    ca = MA.D2Calc(n, s1, s2, r, etot)
    p = MA.d2_paired(ca, 0, ca, 0, 500)
    assert p["B"] == 0.0 and p["B_ci95"] == [0.0, 0.0] and p["B_p"] == 1.0
    # add 0.2 to every class-0 deposit in the gate halves of cell a: the signed gate deviation shifts by exactly 0.2
    s1b = s1.copy()
    s1b[:, 0, MA.REGIONS["gate"]] += 0.2 * n[:, 0, MA.REGIONS["gate"]]
    cb = MA.D2Calc(n, s1b, s2, r, etot)
    q = MA.d2_paired(cb, 0, ca, 0, 500, calc_c=cb, cc=MA.ALL, calc_d=ca, cd=MA.ALL)
    gi = list(MA.DEV_REGIONS).index("gate")
    assert q["dev"][gi] == pytest.approx(0.2, abs=1e-10)
    assert q["dev_ci95"][0][gi] == pytest.approx(0.2, abs=1e-10) and q["dev_ci95"][1][gi] == pytest.approx(0.2, abs=1e-10)
    assert q["dev"][list(MA.DEV_REGIONS).index("left")] == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(MA.MechError):
        MA.d2_paired(ca, 0, MA.D2Calc(n[:5], s1[:5], s2[:5], r, etot), 0, 10)


def _matched_pair(S=8, b0=0.15, g_gate=0.2, sigma=0.0, seed=12):
    """FR arm: class-0 deposits only in the gate, mean r + g_gate + b0; ABF arm: every deposit r + g(x) with a
    position-dependent error g = g_gate in the gate, 0 in the wells (the error every walker has there)."""
    rng = np.random.default_rng(seed)
    r, etot = _bins_ref(4)
    gate = MA.REGIONS["gate"]
    g = np.where(gate, g_gate, 0.0)
    nf = np.zeros((S, NA + 1, NB))
    nf[:, 0, gate] = 20.0
    nf[:, 4, :] = 30.0
    nf[:, MA.ALL] = nf[:, :NA].sum(1)
    mf = np.zeros((S, NA + 1, NB)) + (r + g)[None, None, :]
    mf[:, 0] += b0
    mf += sigma * rng.normal(size=mf.shape)
    s1f = nf * mf
    s1f[:, MA.ALL] = s1f[:, :NA].sum(1)
    na = np.zeros((S, NA + 1, NB))
    na[:, 4, :] = 50.0
    na[:, MA.ALL] = na[:, 4]
    ma = (r + g)[None, None, :] + sigma * rng.normal(size=na.shape)
    s1a = na * ma
    s1a[:, MA.ALL] = s1a[:, 4]
    return (MA.D2Calc(nf, s1f, nf * mf ** 2 + 0.01 * nf, r, etot),
            MA.D2Calc(na, s1a, na * ma ** 2 + 0.01 * na, r, etot))


def test_matched_excess_removes_the_position_effect():
    fr, abf = _matched_pair(b0=0.15, g_gate=0.2)
    p = MA.d2_paired(fr, 0, abf, MA.ALL, 200, matched=True)
    # unmatched: B_0 = 0.35 (gate only) vs ABF over all eval bins 0.2 sqrt(50/150): a large, position-driven gap
    assert p["B"] == pytest.approx(0.35 - 0.2 * math.sqrt(50 / 150), abs=1e-10)
    # matched: ABF at the class-0 positions has error 0.2 -> the clone lineage's own excess is b0
    assert p["Bm_a"] == pytest.approx(0.35, abs=1e-10) and p["Bm_b"] == pytest.approx(0.2, abs=1e-10)
    assert p["B_excess"] == pytest.approx(0.15, abs=1e-10)
    gi, li = list(MA.DEV_REGIONS).index("gate"), list(MA.DEV_REGIONS).index("left")
    assert p["dev_excess"][gi] == pytest.approx(0.15, abs=1e-10) and math.isnan(p["dev_excess"][li])
    assert p["matched_coverage"] == pytest.approx(1.0)
    # no excess when the young clones only sit where everyone is wrong
    fr0, abf0 = _matched_pair(b0=0.0, g_gate=0.2)
    q = MA.d2_paired(fr0, 0, abf0, MA.ALL, 0, matched=True)
    assert q["B_excess"] == pytest.approx(0.0, abs=1e-10)
    assert q["B"] == pytest.approx(0.2 - 0.2 * math.sqrt(50 / 150), abs=1e-10) and q["B"] > 0.08   # unmatched misreads
    assert p["p_calibrated"] is False and MA.d2_paired(fr, 0, abf, MA.ALL, 0)["p_calibrated"] is False


def test_matched_excess_difference_in_differences_and_debiasing():
    a_fr, a_abf = _matched_pair(b0=0.3, g_gate=0.1, sigma=0.0)
    b_fr, b_abf = _matched_pair(b0=0.1, g_gate=0.1, sigma=0.0)
    d = MA.d2_paired(a_fr, [0, MA.ALL], b_fr, [0, MA.ALL], 100, calc_c=a_abf, cc=MA.ALL, calc_d=b_abf, cd=MA.ALL,
                     matched=True)
    assert d[0]["B_excess_a"] == pytest.approx(0.3, abs=1e-10) and d[0]["B_excess_b"] == pytest.approx(0.1, abs=1e-10)
    assert d[0]["B_excess_did"] == pytest.approx(0.2, abs=1e-10)
    assert d[0]["B_excess_did_ci95"][0] == pytest.approx(0.2, abs=1e-10)
    # under pure seed-level noise and no excess, the debiased matched excess is centred on 0 (seeds are the units)
    vals = []
    for k in range(40):
        fr, abf = _matched_pair(S=16, b0=0.0, g_gate=0.0, sigma=0.05, seed=300 + k)
        st_f = fr.stats(np.ones((1, 16)), full=False, perbin=True)
        st_a = abf.stats(np.ones((1, 16)), full=False, perbin=True)
        e = MA._matched(st_f, np.array([0]), st_a, np.array([MA.ALL]), fr.Rm, fr.ev)
        assert e["A2raw"][0, 0] >= e["A2"][0, 0]       # raw plug-in = debiased + noise^2
        vals.append(e["A2"][0, 0])
    v = np.array(vals)
    raw = 0.05 ** 2 / 16                                 # what an undebiased squared RMS would read
    assert abs(v.mean()) < raw / 2


def test_debiased_B_interval_is_pivoted_and_covers_the_truth():
    """B_c^2 is a debiased quadratic: its bootstrap replicates centre on the RAW plug-in (noise^2 above the estimate),
    so a plain percentile interval sits above the estimate (reviewer: 0/40 coverage at B = 0).  The basic (reflected)
    interval must contain its own estimate and cover the truth at B = 0 and B > 0 (seed-level noise only, S = 32)."""
    for b_gate, want in ((0.0, 0.0), (0.15, 0.15 * math.sqrt(50 / 150))):
        cover = contains = lo_pos = 0
        for k in range(40):
            n, s1, s2, r, etot, b = _noisy(32, b_gate, 0.2, 100, seed=2000 + k + int(1000 * b_gate))
            calc = MA.D2Calc(n, s1, s2, r, etot)
            out = MA.d2_summarise(calc, 400, seed=k)
            B, lo, hi = out["B"][0], out["B_lo"][0], out["B_hi"][0]
            cover += lo <= want <= hi
            contains += lo <= B <= hi
            lo_pos += lo > 0
        assert cover >= 36, (b_gate, cover)                 # nominal 38/40; the plain percentile CI gave 0/40 at 0
        assert contains >= 38, (b_gate, contains)
        if b_gate == 0.0:
            assert lo_pos <= 2
    # the reflected draws are centred on the debiased estimate, the raw replicates on the raw plug-in
    n, s1, s2, r, etot, b = _noisy(32, 0.0, 0.2, 100, seed=77)
    calc = MA.D2Calc(n, s1, s2, r, etot)
    pt = calc.stats(np.ones((1, 32)))
    bs = calc.stats(MA.boot_counts(32, 2000, seed=3))
    refl = MA.reflect(pt["B2"], pt["raw2"], bs["B2"])
    noise2 = pt["noise2"][0, 0]
    assert abs(np.nanmean(bs["B2"][:, 0]) - pt["raw2"][0, 0]) < 0.1 * noise2
    assert abs(np.nanmean(refl[:, 0]) - pt["B2"][0, 0]) < 0.1 * noise2
    # raw2 / noise2 are descriptive: no interval
    out = MA.d2_summarise(calc, 50)
    assert np.all(np.isnan(out["raw2_lo"])) and np.all(np.isnan(out["noise2_hi"]))


def _young_vs_abf(S, b_young, sig_young, sig_abf, rng):
    """FR arm: class 0 deposits only in the gate (seed-level noise sig_young per bin mean, true bias b_young), class 4
    everywhere unbiased; ABF arm unbiased everywhere (seed-level noise sig_abf).  True matched excess = b_young."""
    r, etot = np.zeros(NB), np.ones(NB)
    gate = MA.REGIONS["gate"]
    nf = np.zeros((S, NA + 1, NB))
    nf[:, 0, gate] = 20.0
    nf[:, 4, :] = 200.0
    mf = np.zeros((S, NA + 1, NB))
    mf[:, 0, gate] = b_young
    mf[:, 0] += sig_young * rng.normal(size=(S, NB))
    mf[:, 4] += 0.02 * rng.normal(size=(S, NB))
    nf[:, MA.ALL] = nf[:, :NA].sum(1)
    s1f = nf * mf
    s1f[:, MA.ALL] = s1f[:, :NA].sum(1)
    s2f = nf * mf ** 2 + nf
    s2f[:, MA.ALL] = s2f[:, :NA].sum(1)
    na = np.zeros((S, NA + 1, NB))
    na[:, 4, :] = 200.0
    na[:, MA.ALL] = na[:, 4]
    ma = sig_abf * rng.normal(size=(S, NA + 1, NB))
    ma[:, MA.ALL] = ma[:, 4]
    return MA.D2Calc(nf, s1f, s2f, r, etot), MA.D2Calc(na, na * ma, na * ma ** 2 + na, r, etot)


def test_matched_excess_and_its_change_are_calibrated_under_unequal_noise():
    """Known truth, S = 32 seeds, the young class much noisier than ABF (and, for the change from the reference cell,
    noisier in one cell than in the other): no spurious excess / change at zero truth, coverage at a true excess."""
    rng = np.random.default_rng(1234)
    fp = fp_lo = cov = 0
    for k in range(30):
        fr, abf = _young_vs_abf(32, 0.0, 0.3, 0.05, rng)
        p = MA.d2_paired(fr, 0, abf, MA.ALL, 400, seed=k, matched=True)
        fp += p["B_excess_p"] < 0.05
        fp_lo += p["B_excess_ci95"][0] > 0
        fr, abf = _young_vs_abf(32, 0.05, 0.3, 0.05, rng)
        p = MA.d2_paired(fr, 0, abf, MA.ALL, 400, seed=k, matched=True)
        cov += p["B_excess_ci95"][0] <= 0.05 <= p["B_excess_ci95"][1]
        assert p["p_calibrated"] is True
    assert fp <= 3 and fp_lo <= 3                         # the percentile version: 58/60 false positives
    assert cov >= 26                                      # nominal 28.5/30
    rng = np.random.default_rng(99)
    fp = fpd = 0
    for k in range(30):
        fa, aa = _young_vs_abf(32, 0.0, 0.45, 0.05, rng)  # cell a: noisier young class, no excess
        fb, ab = _young_vs_abf(32, 0.0, 0.20, 0.05, rng)  # reference cell: no excess
        d = MA.d2_paired(fa, [0], fb, [0], 400, seed=k, calc_c=aa, cc=MA.ALL, calc_d=ab, cd=MA.ALL, matched=True)[0]
        fp += d["B_excess_did_p"] < 0.05
        fpd += d["B_p"] < 0.05                            # raw B_c(a) - B_c(ref): zero true bias in both
    assert fp <= 3 and fpd <= 3                           # the percentile version: 46/60 false positives


def test_holm_adjust_matches_cross_cell_and_readout_table_flags():
    import cross_cell as XC
    rng = np.random.default_rng(5)
    p = list(rng.uniform(0, 0.2, 9))
    assert np.allclose(MA.holm_adjust(p), XC.holm(p))
    assert np.allclose(MA.holm_adjust(p, m=12), XC.holm(p, m=12))
    assert math.isnan(MA.holm_adjust([0.01, float("nan")])[1]) and MA.holm_adjust([0.01, float("nan")])[0] == 0.02
    exp, N = "conditional_relaxation", 16
    names = MA.AGE_DEFAULT + ["all"]
    end = dict(classes=names, B=np.full(6, 0.1), B_lo=np.full(6, 0.05), B_hi=np.full(6, 0.15), n_seeds=32)
    pv = [0.001, 0.01, 0.02, 0.03, 0.04, 0.5]
    blk = {nm: dict(B_diff=0.02, B_diff_ci95=[0.01, 0.03], B_diff_p=pv[i], p_calibrated=True, B_excess_did=0.01,
                    B_excess_did_ci95=[0.0, 0.02], B_excess_did_p=0.2) for i, nm in enumerate(names)}
    blk.update(n_common_seeds=32, abf_all=dict(B_diff=0.0, B_diff_ci95=[-0.01, 0.01], B_diff_p=0.9, p_calibrated=True))
    fva = {nm: dict(B_excess=0.01, B_excess_ci95=[0.0, 0.02], B_excess_p=0.3, Bm_b=0.09, Bm_b_ci95=[0.05, 0.1],
                    matched_coverage=1.0, p_calibrated=True) for nm in names}
    d2keep = {}
    for c in ("lam1", "lam0.1", "lam0.5"):
        d2keep[(exp, c, N, "fr")] = dict(end=end)
        d2keep[(exp, c, N, "abf")] = dict(end=end)
        d2keep[(exp, c, N, "fr_vs_abf")] = fva
    del d2keep[(exp, "lam0.5", N, "fr_vs_abf")]
    d2c = {exp: {f"lam0.1_vs_lam1_N{N}": blk, f"lam0.5_vs_lam1_N{N}": dict(missing="fewer than 2 common seeds ([])")}}
    t = MA.d2_readout_table(exp, ["lam1", "lam0.5", "lam0.1"], d2keep, d2c, [N])
    rows = t["rows"]
    pre = [rows[f"lam0.1/N{N}"][nm]["preregistered"] for nm in names]
    # family (B_diff_from_ref, N) = 2 non-reference cells x 6 classes = 12 planned tests (lam0.5 missing: p = 1)
    want = XC.holm(pv + [float("nan")] * 6, m=12)[:6]
    assert np.allclose([x["B_diff_from_ref_p_holm"] for x in pre], want)
    assert [x["B_diff_from_ref_p"] for x in pre] == pv                     # unadjusted kept
    assert rows[f"lam0.1/N{N}"]["p_calibrated"] is True and "note" not in rows[f"lam0.1/N{N}"]
    r5 = rows[f"lam0.5/N{N}"]["[0,0.01)"]
    assert r5["preregistered"]["B_diff_from_ref"].startswith("MISSING: fewer than 2 common seeds")
    assert r5["exploratory_analysis_defined"]["B_excess"].startswith("MISSING")
    assert r5["exploratory_analysis_defined"]["change_from_ref"].startswith("MISSING: fewer than 2 common seeds")
    assert rows[f"lam0.5/N{N}"]["abf_B_diff_from_ref"].startswith("MISSING")
    # fewer than SMALL_N seeds: CIs and p NaN, flagged
    small = dict(end, n_seeds=2)
    d2keep[(exp, "lam0.1", N, "fr")] = dict(end=small)
    blk2 = {k: (dict(v, p_calibrated=False, note="2 seeds (< 10): not calibrated") if isinstance(v, dict) else v)
            for k, v in blk.items()}
    t2 = MA.d2_readout_table(exp, ["lam1", "lam0.1"], d2keep, {exp: {f"lam0.1_vs_lam1_N{N}": blk2}}, [N])
    r = t2["rows"][f"lam0.1/N{N}"]
    x = r["[0,0.01)"]["preregistered"]
    assert all(math.isnan(v) for v in x["B_ci95"]) and math.isnan(x["B_diff_from_ref_p"])
    assert math.isnan(x["B_diff_from_ref_p_holm"]) and r["p_calibrated"] is False and "not calibrated" in r["note"]


def test_cross_cell_json_is_used_only_with_the_analysed_summaries(tmp_path):
    import argparse
    tmp = str(tmp_path)
    P = _fake_cell(tmp, "lam0.1")
    _fake_summary(P)
    sp = os.path.join(P["analysis_dir"], "summary.json")
    cells = {("conditional_relaxation", "lam0.1"): P}
    args = argparse.Namespace(analysis_override=None, results_override=None, cells_root=os.path.join(tmp, "cells"),
                              cross_cell=None)
    good = dict(fixture=False, cells={"conditional_relaxation/lam0.1": dict(
        available=True, summary=sp, summary_sha256=MA.M.sha256_file(sp))})
    assert MA.verify_cross_cell(good, cells, args, prod=True) == []
    stale = copy.deepcopy(good)
    stale["cells"]["conditional_relaxation/lam0.1"]["summary_sha256"] = "0" * 64
    assert "changed since" in MA.verify_cross_cell(stale, cells, args, prod=True)[0]
    foreign = copy.deepcopy(good)
    foreign["cells"]["conditional_relaxation/lam0.1"]["summary"] = "/elsewhere/summary.json"
    assert "this run reads" in MA.verify_cross_cell(foreign, cells, args, prod=False)[0]
    assert "not in cross_cell.json" in MA.verify_cross_cell(dict(cells={}), cells, args, prod=False)[0]
    assert "FIXTURE" in MA.verify_cross_cell(dict(good, fixture=True), cells, args, prod=True)[0]
    # the repository file is the default only for the default inputs
    path, why = MA.resolve_cross_cell_path(args)
    assert path is None and "--cells-root" in why
    a2 = argparse.Namespace(**dict(vars(args), cells_root=MA.DEFAULT_CELLS_ROOT))
    assert MA.resolve_cross_cell_path(a2) == (MA.REPO_CROSS_CELL, None)
    a3 = argparse.Namespace(**dict(vars(a2), analysis_override=tmp))
    assert MA.resolve_cross_cell_path(a3)[0] is None
    a4 = argparse.Namespace(**dict(vars(a3), cross_cell="x.json"))
    assert MA.resolve_cross_cell_path(a4) == ("x.json", None)
    keep = MA.s3b_row_filter(cells, [16])
    assert keep("conditional_relaxation", dict(a="lam0.1", b="lam0.1", N=16))
    assert not keep("conditional_relaxation", dict(a="lam0.1", b="lam1", N=16))
    assert not keep("conditional_relaxation", dict(a="lam0.1", b="lam0.1", N=2048))


def test_shared_limits_union_and_fraction_cap():
    e = MA.merge_ext([dict(a=(0.0, 1.0), b=None), dict(a=(-2.0, 0.5), b=(1e-3, 0.4)), None])
    assert e == dict(a=(-2.0, 1.0), b=(1e-3, 0.4))
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    MA.set_log(ax, (0.2, 1.0), cap=1.0)
    lo, hi = ax.get_ylim()
    assert hi == pytest.approx(1.0) and lo <= 0.1 and ax.get_yscale() == "log"
    plt.close(fig)
    fig, ax = plt.subplots()
    MA.set_lin(ax, (0.1, 0.3), zero=True)
    assert ax.get_ylim()[0] < 0 < 0.3 < ax.get_ylim()[1]
    plt.close(fig)


def test_predicted_tau_f_is_undefined_where_f_does_not_depend_on_y():
    for name in ("alpha1", "alpha0.5", "shift", "lam0.1"):
        assert MA.predicted_tau(_model(name), -1.0, "f") is None
        assert MA.predicted_tau(_model(name), -0.21, "f") > 0


def test_thinning_the_save_axis_does_not_change_d1():
    rng = np.random.default_rng(9)
    r = _fake_fr_run(64, rng, n_saves=12)
    full = MA.d1_compute({1: dict(r)}, 64)
    t = dict(r)
    MA.thin_save_axis(t)
    assert t["save_step"].tolist() == sorted(set(r["snap_step"].tolist()) | {int(r["save_step"][-1])})
    thin = MA.d1_compute({1: t}, 64)
    for k in ("death_rate", "birth_rate", "hazard_coarse_EXPLORATORY"):
        assert np.allclose(full["end"][k]["median"], thin["end"][k]["median"], equal_nan=True)
    assert np.allclose(full["interval"]["death_rate_coarse"]["median"], thin["interval"]["death_rate_coarse"]["median"],
                       equal_nan=True)


def test_boot_counts_rows_sum_to_the_seed_count():
    K = MA.boot_counts(9, 400, seed=1)
    assert K.shape == (400, 9) and np.all(K.sum(1) == 9)
    rng = np.random.default_rng(1)
    idx = rng.integers(0, 9, size=(400, 9))
    assert np.array_equal(K[0], np.bincount(idx[0], minlength=9).astype(float))   # cross_cell's draw


# ------------------------------------------------------------------------------------------------ fractions and D1
def test_class_fractions_cumulative_and_interval():
    a = np.zeros((3, 5, NB))
    a[0, 4, 10] = 10.0                                # snapshot 0: 10 'never' deposits in the left well
    a[1] = a[0]
    a[1, 0, 10] = 10.0                                # +10 class-0 in the left well
    a[2] = a[1]
    a[2, 1, 100] = 4.0                                # +4 class-1 in the gate
    cum, inter = MA.class_fractions([{"k": a}], "k")
    assert cum.shape == (1, 3, 5, 3)
    assert cum[0, 1, 0, 0] == 0.5 and cum[0, 1, 4, 0] == 0.5
    assert inter[0, 1, 0, 0] == 1.0 and inter[0, 0, 4, 0] == 1.0
    assert inter[0, 2, 1, 1] == 1.0 and np.isnan(inter[0, 2, 0, 0])    # nothing new in the left well
    assert np.isnan(cum[0, 0, 0, 2])                  # no deposit in the right well


def _fake_fr_run(N, rng, n_saves=10):
    save = np.arange(1, n_saves + 1) * 100
    opp = save // 4
    dh = np.cumsum(rng.integers(0, 3, size=(n_saves, NB)), 0)
    bh = np.cumsum(rng.integers(0, 3, size=(n_saves, NB)), 0)
    snaps = save[[1, 4, 9]]
    return dict(save_step=save, fr_opp_cum=opp, fr_death_pos_hist_cum=dh, fr_birth_pos_hist_cum=bh,
                C_all=np.cumsum(np.full((n_saves, NB), 7.0), 0), snap_step=snaps, snap_u=snaps / save[-1],
                meta=dict(fr_every=4))


def test_d1_rates_are_per_opportunity_per_walker():
    rng = np.random.default_rng(4)
    N = 64
    runs = {1: _fake_fr_run(N, rng), 2: _fake_fr_run(N, rng)}
    D = MA.d1_compute(runs, N)
    r = runs[1]
    want = r["fr_death_pos_hist_cum"][-1] / (r["fr_opp_cum"][-1] * N)
    one = MA.d1_compute({1: r}, N)
    assert np.allclose(one["end"]["death_rate"]["median"], want)
    tot = one["totals"]["deaths_per_opp_per_walker"]["median"]
    assert float(tot) == pytest.approx(r["fr_death_pos_hist_cum"][-1].sum() / (r["fr_opp_cum"][-1] * N))
    s = sum(float(one["region_fraction"]["deaths"][R]["median"]) for R in MA.REGIONS)
    assert s == pytest.approx(1.0)
    # interval rates: increments between snapshots over the opportunity increments
    k1, k2 = 4, 9
    inc = MA.coarse(r["fr_death_pos_hist_cum"][k2] - r["fr_death_pos_hist_cum"][k1]) / \
        ((r["fr_opp_cum"][k2] - r["fr_opp_cum"][k1]) * N)
    assert np.allclose(one["interval"]["death_rate_coarse"]["median"][2], inc)
    hz = MA.coarse(r["fr_death_pos_hist_cum"][-1]) * 4 / MA.coarse(r["C_all"][-1])
    assert np.allclose(one["end"]["hazard_coarse_EXPLORATORY"]["median"], hz)
    assert D["n"] == 2


# ------------------------------------------------------------------------------------------------ model facts
def _model(name):
    return MA.dyn_model(name)


def test_barrier_decomposition_known_values():
    L = math.log(32.0)
    want = {"alpha1": (8.0, L / (8 + L)), "alpha0.5": (8 + L / 2, 0.5 * L / (8 + L)), "alpha0": (8 + L, 0.0),
            "shift": (8 + L, 0.0)}
    for name, (en, frac) in want.items():
        bd = MA.barrier_decomposition(_model(name))
        assert bd["barrier_kT"] == pytest.approx(8.0 + math.log(32.0), abs=1e-6)
        assert bd["energetic_kT"] == pytest.approx(en, abs=2e-4)
        assert bd["entropic_fraction"] == pytest.approx(frac, abs=2e-6)
        assert bd["energetic_kT"] + bd["entropic_kT"] == pytest.approx(bd["barrier_kT"], abs=1e-12)


def _subgrid():
    gl, gw = np.polynomial.legendre.leggauss(8)
    edges = MA.XMIN + MA.DELTA * np.arange(NB)
    sub_x = (edges[:, None] + MA.DELTA * 0.5 * (gl[None, :] + 1.0)).ravel()
    return sub_x, np.tile(0.5 * gw, NB), np.repeat(np.arange(NB), 8)


@pytest.mark.parametrize("name", ["alpha1", "alpha0.5", "alpha0", "shift"])
def test_exact_within_bin_force_variance_matches_monte_carlo(name):
    m = _model(name)
    ex = MA.exact_bin_force_moments(m, *_subgrid())
    rng = np.random.default_rng(1)
    for j in (100, 110, 140):                         # gate flank, steep flank, right well
        lo = MA.XMIN + j * MA.DELTA
        x = rng.uniform(lo, lo + MA.DELTA, 400_000)
        y = GF.m_centre(x, m) + np.sqrt(GF.cond_var_y(x, m)) * rng.normal(size=x.size)
        f = GF.grad_V_np(x, y, m)[0]
        assert f.mean() == pytest.approx(ex["fbar"][j], abs=4 * f.std() / math.sqrt(x.size) + 1e-12)
        assert f.var() == pytest.approx(ex["etot"][j], rel=0.02)


@pytest.mark.parametrize("name", ["alpha1", "alpha0.5", "shift", "lam0.1"])
def test_ylaw_standardisation_is_exact_and_detects_a_wrong_width(name):
    m = _model(name)
    rng = np.random.default_rng(2)
    runs = {}
    for s in range(4):
        x = np.concatenate([rng.uniform(c - 0.05, c + 0.05, 2500) for c in MA.Y_WINDOWS])
        y = GF.m_centre(x, m) + np.sqrt(GF.cond_var_y(x, m)) * rng.normal(size=x.size)
        runs[s] = dict(X_final=x, Y_final=y)
    Y = MA.ylaw_compute({"abf": runs}, m, 2.5e-5, 500)
    for c in MA.Y_WINDOWS:
        e = Y["abf"]["windows"][f"{c:g}"]
        assert e["n"] == 10000 and e["ks_D"] < 0.02
        assert e["var_z"] == pytest.approx(1.0, abs=0.05)
        assert e["var_z_ci95"][0] < e["var_z"] < e["var_z_ci95"][1]
    wide = {s: dict(X_final=r["X_final"], Y_final=GF.m_centre(r["X_final"], m)
                    + 1.2 * (r["Y_final"] - GF.m_centre(r["X_final"], m))) for s, r in runs.items()}
    W = MA.ylaw_compute({"fr": wide}, m, 2.5e-5, 0)
    for c in MA.Y_WINDOWS:
        assert W["fr"]["windows"][f"{c:g}"]["var_z"] == pytest.approx(1.44, abs=0.07)


def test_predicted_tau_is_the_frozen_x_ou_time():
    m = _model("alpha1")
    t = MA.predicted_tau(m, -1.0, "y")
    assert t == pytest.approx(1.0 * (1 - math.exp(-5)), rel=1e-3)        # tau_y = 1/(lam omega^2) = 1 in the well
    assert MA.predicted_tau(_model("alpha0"), -0.21, "f") is None
    assert MA.predicted_tau(_model("lam0.1"), -1.0, "y") == pytest.approx(10 * (1 - math.exp(-5)), rel=1e-3)


def _validation_doc():
    """Minimal validation summary in analyze_validation's layout (dynamics.<d>.per_h.<h>.{acf.windows, em_flat})."""
    blk = dict(acf=dict(windows={"-0.21": dict(tau_int_y_5tau=0.07, tau_int_y_5tau_se=0.001, tau_int_f_5tau=0.031,
                                               tau_int_f_5tau_se=0.001, resolved=True, resolved_y=True,
                                               resolved_f=True)}),
               em_flat=dict(force_var_profile=dict(x=[-0.21, 0.0, 0.21, 0.55], measured=[2.0, 0.0, 2.2, 9.9],
                                                    se=[0.01, 0.0, 0.01, 0.1], exact=[2.06, 0.0, 2.06, 0.0])))
    return dict(dynamics={"lam0.1": dict(per_h={"2.5e-05": blk})}, selection=dict(h_common=2.5e-5))


def test_validation_measurements_are_used_when_present_and_reported_when_absent():
    V = _validation_doc()
    blk, hk = MA.validation_lookup(V, "lam0.1", 2.5e-5)
    assert hk == "2.5e-05" and blk is not None
    assert MA.validation_lookup(None, "lam0.1", 2.5e-5) == (None, "validation summary absent")
    assert MA.validation_lookup(V, "alpha1", 2.5e-5)[0] is None
    tab = MA.s4_table(V, 2.5e-5)
    assert tab["lam0.1"]["windows"]["-0.21"]["tau_f_measured"] == 0.031 and tab["lam0.1"]["validation"] == "present"
    assert tab["alpha1"]["validation"].startswith("dynamics alpha1 not in")
    P = json.load(open(os.path.join(ROOT, "configs", "mechanism", "cells", "conditional_relaxation", "lam0.1.json")))
    S = dict(contrasts={"2048": dict(primary=dict(Ibar_F=dict(G_median=-0.1, G_ci95=[-0.2, 0.0], n=32, per_seed={})))},
             per_N={"2048": dict(abf=dict(tau_TV_half_t=dict(median=12.0, q25=10.0, q75=14.0, n_inf=0, n=32)))})
    cov = MA.covariates({("conditional_relaxation", "lam0.1"): P}, {("conditional_relaxation", "lam0.1"): S}, V, None)
    assert len(cov) == 1
    r = cov[0]
    assert r["tau_f"] == 0.031 and r["tau_f_source"] == "validation (measured)"
    assert r["var_f_gate"] == pytest.approx((2.0 + 0.0 + 2.2) / 3) and r["var_f_gate_source"] == "validation (measured)"
    assert r["tau_TV_half"] == 12.0 and r["tau_TV_half_censored"] is False
    cov0 = MA.covariates({("conditional_relaxation", "lam0.1"): P}, {("conditional_relaxation", "lam0.1"): S}, None, None)
    assert cov0[0]["tau_f_source"].startswith("frozen-x") and "validation absent" in cov0[0]["tau_f_source"]
    assert cov0[0]["var_f_gate_source"] == "exact (validation absent)"


# ------------------------------------------------------------------------------------------------ end to end
SEEDS = [8100, 8101, 8102]
N_E2E, NSTEP, H = 16, 4000, 2.5e-5


def _fake_cell(tmp, name="lam0.1", exp="conditional_relaxation"):
    P = json.load(open(os.path.join(ROOT, "configs", "mechanism", "cells", exp, f"{name}.json")))
    P = copy.deepcopy(P)
    P.update(seeds=SEEDS, N_ladder=[N_E2E], N_coarse=[N_E2E], B=N_E2E * NSTEP, T_N={str(N_E2E): NSTEP * H},
             results_dir=os.path.join(tmp, "runs", exp, name), analysis_dir=os.path.join(tmp, "analysis", exp, name),
             fig_dir=os.path.join(tmp, "unused_figs"))
    os.makedirs(os.path.join(tmp, "cells", exp), exist_ok=True)
    with open(os.path.join(tmp, "cells", exp, f"{name}.json"), "w") as fh:
        json.dump(P, fh)
    return P


def _fake_run(P, seed, method, gate_bias):
    """A synthetic run that satisfies check_plan and the engine's D-identities; class-0 deposits sit in the gate
    with mean force <F*'>_j + gate_bias, every other deposit exactly <F*'>_j (so B_0 = |gate_bias|, others 0)."""
    ref = np.load(os.path.join(ROOT, P["reference_file"]))["Fp_bin_ref"]
    rng = np.random.default_rng(seed + (0 if method == "abf" else 7))
    save = np.unique(np.concatenate([np.arange(1, 41) * 100, [40, 200, 400, 1000, 2000, 3000]]))
    snaps = np.array([40, 200, 400, 1000, 2000, 3000, 4000])
    S, K = save.size, snaps.size
    rate = np.full(NB, 2.0)                             # integral counts, as in the engine
    C_all = save[:, None] * rate[None, :]
    cls = np.zeros((K, 5, NB))
    for i, k in enumerate(snaps):
        tot = k * rate
        if method == "fr":
            cls[i, 0] = np.where(MA.REGIONS["gate"], np.round(0.25 * tot), 0.0)
            cls[i, 2] = np.round(0.1 * tot)
        cls[i, 4] = tot - cls[i].sum(0)
    M1 = cls * ref[None, None, :]
    M1[:, 0] += cls[:, 0] * gate_bias
    M2 = cls * (ref[None, None, :] ** 2 + 0.01)
    cross = np.zeros_like(cls)
    cross[:, 1] = np.round(cls.sum(1) * 0.3)
    cross[:, 4] = cls.sum(1) - cross[:, 1]
    deaths = np.cumsum(rng.integers(0, 2, size=(S, NB)), 0) if method == "fr" else np.zeros((S, NB), np.int64)
    births = np.zeros_like(deaths)
    if method == "fr":                                # births in total equal deaths (each death has one copy)
        tot_d = deaths.sum(1)
        births[:, 90] = tot_d
    tr_t = np.arange(0, 41) * 0.0025
    x = rng.uniform(-1.6, 1.6, N_E2E)
    model = MA.cell_model(P)
    y = GF.m_centre(x, model) + np.sqrt(GF.cond_var_y(x, model)) * rng.normal(size=N_E2E)
    meta = dict(engine=P["engine_version"], system="gateway_family", status="complete", N=N_E2E, seed=seed,
                method=method, n_steps=NSTEP, h=H, fr_every=160, variant="alpha", alpha=1.0, kappa=None,
                lam=P["engine_cfg"]["lam"],
                diag_classes=dict(clone_age_class_names=MA.AGE_DEFAULT, cross_age_class_names=MA.CROSS_DEFAULT))
    arrays = dict(save_step=save, save_t=save * H, save_u=save / NSTEP, C_all=C_all, fr_opp_cum=save // 160,
                  fr_deaths_cum=deaths.sum(1), fr_death_pos_hist_cum=deaths, fr_birth_pos_hist_cum=births,
                  snap_step=snaps, snap_t=snaps * H, snap_u=snaps / NSTEP, dep_age_C=cls, dep_age_M=M1, dep_age_M2=M2,
                  dep_cross_C=cross, traces_t=tr_t, traces=rng.uniform(-1, 1, (41, N_E2E)),
                  traces_y=rng.normal(0, 0.1, (41, N_E2E)), traces_rebirth=np.zeros((41, N_E2E), np.int64),
                  X_final=x, Y_final=y, cfg_json=np.array(json.dumps(P["engine_cfg"])), meta_json=np.array(json.dumps(meta)))
    d = os.path.join(P["results_dir"], f"N{N_E2E}")
    os.makedirs(d, exist_ok=True)
    np.savez_compressed(os.path.join(d, f"s{seed}_{method}.npz"), **arrays)


def _fake_summary(P):
    per = {str(s): -0.2 + 0.01 * i for i, s in enumerate(SEEDS)}
    g = dict(n=3, G_median=-0.19, G_ci95=[-0.2, -0.18], per_seed=per)
    S = dict(contrasts={str(N_E2E): dict(primary=dict(Ibar_F=g), secondary=dict(Ibar_Fp_stat=g),
                                         marginal=dict(Ibar_TV_half=g))},
             per_N={str(N_E2E): dict(abf=dict(tau_TV_half_t=dict(median=0.05, q25=0.04, q75=0.06, n_inf=0, n=3),
                                              est_cum_t=dict(median="inf", q25="inf", q75="inf", n_inf=3, n=3)))})
    os.makedirs(P["analysis_dir"], exist_ok=True)
    with open(os.path.join(P["analysis_dir"], "summary.json"), "w") as fh:
        json.dump(S, fh)


def test_end_to_end_on_fabricated_runs(tmp_path, capsys):
    tmp = str(tmp_path)
    Pa = _fake_cell(tmp, "lam0.1")
    Pb = _fake_cell(tmp, "lam1")
    for P, bias in ((Pa, 0.3), (Pb, 0.1)):
        for s in SEEDS:
            for m in ("abf", "fr"):
                if P is Pa and s == SEEDS[-1] and m == "fr":
                    continue                          # one missing run: must be REPORTED
                _fake_run(P, s, m, bias)
    _fake_summary(Pa)
    out = os.path.join(tmp, "out")
    foreign = os.path.join(tmp, "foreign_cross_cell.json")         # e.g. the production file next to smoke cells
    with open(foreign, "w") as fh:
        json.dump(dict(fixture=False, cells={"conditional_relaxation/lam0.1": dict(
            available=True, summary="/elsewhere/summary.json", summary_sha256="0" * 64)},
            experiments={"conditional_relaxation": dict(metrics=dict(Ibar_F=dict(contrasts=[dict(
                id="D1", a="lam0.1", b="lam1", N=2048, kind="primary", n=32, ratio=1.2, ratio_ci95=[1.1, 1.3])])))}),
            fh)
    syn = MA.main(["--cells-root", os.path.join(tmp, "cells"), "--out", out, "--n-boot", "200",
                   "--validation", os.path.join(tmp, "nope.json"), "--cross-cell", foreign])
    txt = capsys.readouterr().out
    assert "COMPLETENESS" in txt and "MISSING" in txt
    rows = {(r["cell"], r["method"]): r for r in syn["completeness"]["rows"]}
    assert rows[("conditional_relaxation/lam0.1", "fr")]["missing"] == 1
    assert rows[("conditional_relaxation/lam0.1", "fr")]["detail"]["missing"] == [SEEDS[-1]]
    assert rows[("conditional_relaxation/lam1", "abf")]["complete"] == 3
    assert syn["completeness"]["validation_summary"].startswith("absent")
    # a cross_cell.json made from other summaries is rejected (never silently embedded) and reported
    rej = syn["inputs"]["cross_cell"]["rejected_because"]
    assert any("this run reads" in x for x in rej) and any("lam1: not in cross_cell.json" in x for x in rej)
    assert not syn["completeness"]["cross_cell"].startswith("present")
    assert any("does NOT belong" in w for w in syn["warnings"])
    assert not any(r.get("n") == 32 for e in (syn.get("S3b_cross_cell") or {}).values() for rr in e.values()
                   for r in rr)
    assert "conditional_relaxation/lam1" in syn["completeness"]["summaries_missing"]
    # expected cells without configs are listed, not dropped
    assert "matched_free_energy/alpha1" in syn["completeness"]["cells_without_config"]
    # D2 known answer through the pipeline: class 0 carries +0.3 (lam0.1) / +0.1 (lam1) in the gate, nothing else
    d = json.load(open(os.path.join(out, "results", "mechanism", "conditional_relaxation", "lam0.1", "analysis",
                                    "mech_diagnostics.json")))
    D2 = d["per_N"][str(N_E2E)]["D2"]
    B_end = D2["fr"]["B"]["value"][-1]
    assert B_end[0] == pytest.approx(0.3, abs=1e-5)
    assert B_end[2] == pytest.approx(0.0, abs=1e-6) and B_end[4] == pytest.approx(0.0, abs=1e-6)
    dev = D2["fr"]["dev"]["value"][-1]
    gi = D2["regions"].index("gate")
    assert dev[0][gi] == pytest.approx(0.3, abs=1e-6) and dev[0][D2["regions"].index("left")] is None
    assert D2["abf"]["B"]["value"][-1][5] == pytest.approx(0.0, abs=1e-6)
    # paired cross-cell contrast lam0.1 vs lam1 on the two common seeds: B_0 difference 0.2
    cc = syn["D2_cross_cell"]["conditional_relaxation"][f"lam0.1_vs_lam1_N{N_E2E}"]
    assert cc["n_common_seeds"] == 2
    assert cc["[0,0.01)"]["B_diff"] == pytest.approx(0.2, abs=1e-5)
    # the ABF arm has no error: the matched excess of the young class is its whole bias, and its change from lam1 0.2
    assert cc["[0,0.01)"]["B_excess_did"] == pytest.approx(0.2, abs=1e-5)
    fva = d["per_N"][str(N_E2E)]["D2_fr_vs_abf"]["[0,0.01)"]
    assert fva["B_excess"] == pytest.approx(0.3, abs=1e-5) and fva["Bm_b"] == pytest.approx(0.0, abs=1e-6)
    assert fva["p_calibrated"] is False and "not calibrated" in fva["note"]
    rd = syn["S5b_D2_readout"]
    assert set(rd["definition"]) >= {"preregistered", "exploratory_analysis_defined", "holm", "small_n"}
    ro = rd["rows"][f"lam0.1/N{N_E2E}"]["[0,0.01)"]
    pre, ex = ro["preregistered"], ro["exploratory_analysis_defined"]
    assert pre["B"] == pytest.approx(0.3, abs=1e-5) and pre["B_diff_from_ref"] == pytest.approx(0.2, abs=1e-5)
    assert ex["B_excess"] == pytest.approx(0.3, abs=1e-5) and ex["change_from_ref"] == pytest.approx(0.2, abs=1e-5)
    # 2-3 seeds (< SMALL_N): CIs and p are NaN (null) in the readout and the row says so
    nan = lambda v: v is None or (isinstance(v, float) and math.isnan(v))   # noqa: E731  (in memory NaN, JSON null)
    assert all(map(nan, pre["B_ci95"])) and nan(pre["B_diff_from_ref_p"]) and pre["p_calibrated"] is False
    assert nan(ex["B_excess_p"]) and all(map(nan, ex["change_from_ref_ci95"]))
    js = json.load(open(os.path.join(out, "results", "mechanism", "synthesis", "mech_synthesis.json")))
    assert js["S5b_D2_readout"]["rows"][f"lam0.1/N{N_E2E}"]["[0,0.01)"]["preregistered"]["B_ci95"] == [None, None]
    assert rd["rows"][f"lam0.1/N{N_E2E}"]["p_calibrated"] is False and "not calibrated" in \
        rd["rows"][f"lam0.1/N{N_E2E}"]["note"]
    lam1 = rd["rows"][f"lam1/N{N_E2E}"]
    assert lam1["[0,0.01)"]["exploratory_analysis_defined"]["B_excess"] == pytest.approx(0.1, abs=1e-5)
    assert "B_diff_from_ref" not in lam1["[0,0.01)"]["preregistered"]          # the reference cell has no change
    # per-cell D1 / D2 / D3 figures share their y ranges across the cells of the experiment at this N
    d1 = json.load(open(os.path.join(out, "results", "mechanism", "conditional_relaxation", "lam1", "analysis",
                                     "mech_diagnostics.json")))
    assert d["shared_y_ranges"] == d1["shared_y_ranges"]
    assert {f"{k}:conditional_relaxation:N{N_E2E}" for k in ("d1", "d2f", "d2b", "d3")} <= set(d["shared_y_ranges"])
    # D1 present, D3 fractions 0.3 / 0.7
    D3 = d["per_N"][str(N_E2E)]["D3"]
    assert D3["fr_cumulative"]["median"][-1][1][0] == pytest.approx(0.3, abs=1e-6)
    # figures: every one written is legible; the expected per-cell set exists
    names = {f["name"] for f in d["per_N"][str(N_E2E)]["figures"]}
    assert names == {"mech_d1_fr_events", "mech_d2_age_fractions", "mech_d2_mean_force_bias", "mech_d2_force_variance",
                     "mech_d3_crossing", "mech_ylaw", "mech_ytraces"}
    for f in d["per_N"][str(N_E2E)]["figures"] + syn["figures"]:
        assert f["legibility"]["status"] == "pass", (f["name"], f["legibility"]["failures"][:2])
        for p in f["files"]:
            assert os.path.getsize(os.path.join(out, p)) > 0
    # S2 has no per-run metric curves -> skipped WITH a reason; S3b: no cross_cell.json and no full summaries
    skipped = {s["name"]: s["why"] for s in syn["figures_skipped"]}
    assert any(k.startswith("mech_s2_convergence") for k in skipped)
    written = {f["name"] for f in syn["figures"]}
    assert {"mech_s1_models", "mech_s4_force_variance_relaxation", "mech_s5_lambda", "mech_s5b_d2_conditional_relaxation",
            "mech_s6_marginal_vs_free_energy", "mech_m1_gain_vs_covariates"} <= written
    assert "mech_s5c_d2_matched_free_energy" in skipped           # no Experiment I cell in this fixture: reported
    assert syn["S5"]["missing"] == [f"lam1 N{N_E2E}"]           # lam1 has no summary: listed in the figure and JSON
    assert syn["inputs"]["validation_summary"]["available"] is False
    # mechanistic covariates fall back to exact / predicted values and say so
    cov = [r for r in syn["mechanistic_covariates"] if r["cell"] == "lam0.1"][0]
    assert cov["var_f_gate_source"].startswith("exact") and cov["tau_f_source"].startswith("frozen-x")
    assert cov["est_cum_censored"] is True and cov["T_N"] == pytest.approx(NSTEP * H)


def test_file_status_never_reads_an_incomplete_run(tmp_path):
    d = str(tmp_path)
    good, bad, run = (os.path.join(d, f"{k}.npz") for k in ("good", "bad", "run"))
    np.savez(good, meta_json=np.array(json.dumps(dict(status="complete"))))
    np.savez(bad, meta_json=np.array(json.dumps(dict(status="partial"))))
    np.savez(run + ".ckpt.npz", x=np.zeros(1))
    assert MA.file_status(good)[0] == "complete"
    assert MA.file_status(bad)[0] == "invalid"
    assert MA.file_status(run)[0] == "running"
    assert MA.file_status(os.path.join(d, "none.npz"))[0] == "missing"
    with open(os.path.join(d, "trunc.npz"), "wb") as fh:
        fh.write(b"PK\x03\x04 truncated")
    assert MA.file_status(os.path.join(d, "trunc.npz"))[0] == "invalid"
    st = MA.status_counts({1: dict(status="running"), 2: dict(status="missing"), 3: dict(status="invalid", why="x")})
    assert st["running"] == [1] and st["missing"] == [2] and st["invalid"] == {3: "x"} and st["complete"] == 0


def test_refuses_to_write_into_the_equal_budget_tree():
    with pytest.raises(MA.MechError):
        MA.guard(os.path.join(ROOT, "results", "equal_budget_v2", "x.json"))
