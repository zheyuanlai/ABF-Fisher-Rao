"""Shared arithmetic for the convergence / time-to-accuracy analyses of the convergence-FR campaign
(docs/convergence_fr_campaign/PREREGISTRATION.md section 2 defines every quantity; this module is the single
implementation used by the descriptive atlas AND the confirmatory analyses).

Error metrics (per seed, per save t_s; the seed is the unit of inference):
    e_F(t)  = aligned full-circle RMS of F_hat_t - F_ref (both centred), kJ/mol    [= L2 norm / sqrt(2 pi)]
    e_F'(t) = RMS over bins of Gamma_hat_t - Gamma_ref (unvisited bins read Gamma = 0), kJ/mol/rad
    I_F(t)  = int_0^t e_F ds (trapezoid over saves);  Ibar_F(t) = I_F(t) / t
    post-ramp Ibar_F(t) = int_{t_r}^t e_F ds / (t - t_r),  t_r = FR start = end of the bias ramp
Time to accuracy at tolerance eps (save resolution, no interpolation; censored = never within the horizon):
    first     : first save with e_F <= eps
    sustained : first save after which e_F <= eps at EVERY later save up to the horizon (last entry)
    windowed  : first save s with e_F <= eps at every save in [t_s, t_s + 0.1 T] (needs t_s + 0.1 T <= T)
    restricted per-seed speedup S = min(tau_ABF, T) / min(tau_FR, T)  (censored at T; both censored -> S = 1)
"""
import numpy as np

BOOT_SEED = 2026


def cumtrapz(y, t):
    """Cumulative trapezoid along axis 0 of y (n_save, ...), starting at 0."""
    dt = np.diff(t)
    inc = 0.5 * (y[1:] + y[:-1]) * dt.reshape((-1,) + (1,) * (y.ndim - 1))
    return np.concatenate([np.zeros((1,) + y.shape[1:]), np.cumsum(inc, axis=0)], axis=0)


def running_ibar(e, t, t0=0.0):
    """Running time-average of e over [t0, t] (n_save, R); NaN for t <= t0."""
    sel = t >= t0 - 1e-12
    out = np.full(e.shape, np.nan)
    tt = t[sel]
    I = cumtrapz(e[sel], tt)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[sel] = I / (tt - tt[0]).reshape((-1,) + (1,) * (e.ndim - 1))
    out[np.where(sel)[0][0]] = np.nan
    return out


def boot_median(x, n=10000, seed=BOOT_SEED):
    """Median and 95 % percentile bootstrap CI of the median over seeds (1-D)."""
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    m = np.median(x[idx], axis=1)
    return float(np.median(x)), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def boot_median_curve(X, n=2000, seed=BOOT_SEED):
    """X (n_save, R): per-save median over seeds and its 95 % bootstrap CI (seeds resampled jointly over saves)."""
    rng = np.random.default_rng(seed)
    R = X.shape[1]
    idx = rng.integers(0, R, size=(n, R))
    med = np.nanmedian(X, axis=1)
    lo = np.empty(X.shape[0])
    hi = np.empty(X.shape[0])
    for s0 in range(0, X.shape[0], 64):
        blk = X[s0:s0 + 64][:, idx]                       # (b, n, R)
        mb = np.nanmedian(blk, axis=2)
        lo[s0:s0 + 64] = np.nanquantile(mb, 0.025, axis=1)
        hi[s0:s0 + 64] = np.nanquantile(mb, 0.975, axis=1)
    return med, lo, hi


def tau(e, t, eps, kind="sustained", window_frac=0.1):
    """Time to accuracy per seed.  e (n_save, R) -> (R,) times, np.inf where censored."""
    below = e <= eps
    n, R = e.shape
    out = np.full(R, np.inf)
    T = t[-1]
    if kind == "first":
        for r in range(R):
            w = np.where(below[:, r])[0]
            if len(w):
                out[r] = t[w[0]]
    elif kind == "sustained":
        for r in range(R):
            if not below[-1, r]:
                continue
            above = np.where(~below[:, r])[0]
            out[r] = t[0] if len(above) == 0 else t[above[-1] + 1]
    elif kind == "windowed":
        D = window_frac * T
        for r in range(R):
            for s in np.where(below[:, r])[0]:
                if t[s] + D > T + 1e-9:
                    break
                sel = (t >= t[s] - 1e-12) & (t <= t[s] + D + 1e-9)
                if below[sel, r].all():
                    out[r] = t[s]
                    break
    else:
        raise ValueError(kind)
    return out


def speedup_stats(ta, tb, T, n=10000, seed=BOOT_SEED):
    """Paired restricted speedup S = min(ta,T)/min(tb,T) per seed (a = ABF, b = ABF+FR) with censoring bookkeeping."""
    ta = np.asarray(ta, float)
    tb = np.asarray(tb, float)
    ca, cb = ~np.isfinite(ta), ~np.isfinite(tb)
    ra, rb = np.minimum(ta, T), np.minimum(tb, T)
    S = ra / rb
    S[ca & cb] = 1.0
    med, lo, hi = boot_median(np.log(S), n=n, seed=seed)
    faster = int(np.sum((tb < ta) & ~cb))            # FR strictly earlier (a censored ABF counts if FR reached)
    slower = int(np.sum((ta < tb) & ~ca))
    return dict(S_median=float(np.exp(med)), S_ci95=[float(np.exp(lo)), float(np.exp(hi))],
                n_seeds=int(len(ta)), reached_abf=int((~ca).sum()), reached_fr=int((~cb).sum()),
                censored_abf=int(ca.sum()), censored_fr=int(cb.sum()), both_censored=int((ca & cb).sum()),
                fr_faster=faster, abf_faster=slower, ties=int(len(ta) - faster - slower),
                tau_abf_median=float(np.median(ra)) if (~ca).sum() * 2 > len(ta) else None,
                tau_fr_median=float(np.median(rb)) if (~cb).sum() * 2 > len(tb) else None,
                bound=("lower bound (ABF censored in > 25 % of seeds)" if ca.mean() > 0.25 and cb.mean() <= 0.25 else
                       "upper bound (FR censored in > 25 % of seeds)" if cb.mean() > 0.25 and ca.mean() <= 0.25 else
                       "both arms heavily censored" if ca.mean() > 0.25 and cb.mean() > 0.25 else "estimable"),
                per_seed_S=[float(x) for x in S],
                tau_abf=[float(x) for x in ta], tau_fr=[float(x) for x in tb])


def paired_rel(a, b, n=10000, seed=BOOT_SEED):
    """Per-seed relative change b/a - 1 (b = ABF+FR), median, 95 % CI, FR wins (b < a)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = b / a - 1.0
    med, lo, hi = boot_median(d, n=n, seed=seed)
    return dict(median_rel=med, ci95=[lo, hi], wins_fr=int((d < 0).sum()), n=int(len(d)),
                abf_median=float(np.median(a)), fr_median=float(np.median(b)), per_seed=[float(x) for x in d])


def crossing(ea, eb, t, t_min=0.0, n=2000, seed=BOOT_SEED):
    """Where FR (b) becomes permanently worse than ABF (a).
    per seed: first save after which e_b > e_a at every later save (inf if e_b(T) <= e_a(T));
    median curve: the same rule on the per-save median of the paired relative difference, with a bootstrap
    distribution over seeds (fraction of resamples that cross, and the 2.5/50/97.5 % crossing times)."""
    sel = t >= t_min
    d = (eb - ea)[sel]
    tt = t[sel]
    R = d.shape[1]
    per = np.full(R, np.inf)
    for r in range(R):
        if d[-1, r] <= 0:
            continue
        neg = np.where(d[:, r] <= 0)[0]
        per[r] = tt[0] if len(neg) == 0 else tt[neg[-1] + 1]
    rel = (eb / ea - 1.0)[sel]

    def cross_of(m):
        if m[-1] <= 0:
            return np.inf
        neg = np.where(m <= 0)[0]
        return tt[0] if len(neg) == 0 else tt[neg[-1] + 1]
    c0 = cross_of(np.median(rel, axis=1))
    rng = np.random.default_rng(seed)
    cs = np.array([cross_of(np.median(rel[:, rng.integers(0, R, R)], axis=1)) for _ in range(n)])
    fin = cs[np.isfinite(cs)]
    return dict(per_seed=[float(x) for x in per], n_seeds_crossing=int(np.isfinite(per).sum()), n_seeds=R,
                per_seed_median=float(np.median(per)) if np.isfinite(per).sum() * 2 > R else None,
                median_curve_crossing=float(c0), boot_fraction_crossing=float(len(fin) / n),
                boot_quantiles=[float(np.quantile(fin, q)) for q in (0.025, 0.5, 0.975)] if len(fin) else None)


def establishment_time(frac, t, share, level=0.75, smooth_tu=1.0):
    """First save at which the trailing smooth_tu-t.u. mean window occupancy reaches level x the flat-bias
    (uniform-marginal) window share.  frac (n_save, R) -> (R,) times (inf if never)."""
    ds = float(np.median(np.diff(t)))
    w = max(1, int(round(smooth_tu / ds)))
    cs = np.cumsum(np.vstack([np.zeros((1, frac.shape[1])), frac]), axis=0)
    out = np.full(frac.shape[1], np.inf)
    for s in range(frac.shape[0]):
        a = max(0, s - w + 1)
        m = (cs[s + 1] - cs[a]) / (s + 1 - a)
        hit = (m >= level * share) & ~np.isfinite(out)
        out[hit] = t[s]
    return out
