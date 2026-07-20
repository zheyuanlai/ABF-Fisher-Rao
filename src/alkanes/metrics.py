"""Scientific metrics for the alkane runs (per seed), consuming ``core.run_sampler``
/ ``opes.run_opes`` diagnostics and an independent reference.

All profile comparisons are circular and additive-constant-aligned.  Pentane adds
the hidden-conditional diagnostics ``p(phi2|phi1)`` (TV / KL / basin probabilities /
reference-weighted aggregate), which are the crux of the healthy-vs-false improvement
distinction.
"""
from __future__ import annotations

import math

import numpy as np

PI = math.pi
TWO_PI = 2.0 * PI
EPS = 1.0e-12


def _circ_l2(profile, reference, dphi, align=True):
    p = np.asarray(profile, float); r = np.asarray(reference, float)
    if align:
        p = p - np.mean(p - r)
    return math.sqrt(np.sum((p - r) ** 2) * dphi / TWO_PI)


def profile_metrics(pmf_series, mf_series, times, ref_F, ref_Fp, dphi):
    """Final + time-integrated + checkpoint circular L2 of F and F' vs the reference.

    ``pmf_series`` / ``mf_series`` are ``(T, n_grid)`` for ONE seed; ``times`` ``(T,)``.
    """
    T = len(times)
    l2F = np.array([_circ_l2(pmf_series[k], ref_F, dphi, align=True) for k in range(T)])
    l2Fp = np.array([_circ_l2(mf_series[k], ref_Fp, dphi, align=False) for k in range(T)])
    intF = float(np.trapezoid(l2F, times)) if T > 1 else float("nan")
    intFp = float(np.trapezoid(l2Fp, times)) if T > 1 else float("nan")
    early = l2F[min(1, T - 1)]
    mid = l2F[T // 2]
    return {"final_l2_F": float(l2F[-1]), "final_l2_Fp": float(l2Fp[-1]),
            "integrated_l2_F": intF, "integrated_l2_Fp": intFp,
            "early_l2_F": float(early), "mid_l2_F": float(mid),
            "l2_F_series": l2F, "l2_Fp_series": l2Fp}


def marginal_metrics(p_hat, grid, dphi, beta, ref_F):
    """L2/TV/KL of the phi1 marginal vs uniform and vs the Boltzmann marginal."""
    p = np.clip(np.asarray(p_hat, float), 0, None)
    p = p / (p.sum() * dphi + EPS)
    uni = np.ones_like(p) / TWO_PI
    logpb = -beta * (np.asarray(ref_F, float) - np.max(ref_F))
    pb = np.exp(logpb); pb = pb / (pb.sum() * dphi + EPS)
    def l2(a, b): return math.sqrt(np.sum((a - b) ** 2) * dphi / TWO_PI)
    def tv(a, b): return 0.5 * np.sum(np.abs(a - b)) * dphi
    def kl(a, b): return float(np.sum(np.clip(a, EPS, None) * (np.log(np.clip(a, EPS, None)) - np.log(np.clip(b, EPS, None)))) * dphi)
    return {"marginal_l2_uniform": l2(p, uni), "marginal_tv_uniform": tv(p, uni),
            "marginal_l2_boltzmann": l2(p, pb), "marginal_kl_boltzmann": kl(p, pb)}


def fr_event_metrics(total_repl, repl_cumulative_series, steps, N, fr_start, fr_every, n_steps):
    """Mean/max realised event fraction, #applications, deaths/application (per seed)."""
    if n_steps < fr_start:
        n_apps = 0
    else:
        n_apps = (n_steps - fr_start) // max(fr_every, 1) + 1
    total = float(total_repl)
    mean_frac = total / (N * n_apps) if n_apps > 0 else 0.0
    rc = np.asarray(repl_cumulative_series, float)
    steps = np.asarray(steps, int)
    max_frac = 0.0
    for k in range(1, len(steps)):
        lo, hi = int(steps[k - 1]), int(steps[k])
        lo2 = max(lo + 1, fr_start)
        if hi < lo2:
            continue
        rem = (lo2 - fr_start) % max(fr_every, 1)
        first = lo2 + ((max(fr_every, 1) - rem) % max(fr_every, 1))
        apps_k = 0 if first > hi else (hi - first) // max(fr_every, 1) + 1
        if apps_k <= 0:
            continue
        max_frac = max(max_frac, (rc[k] - rc[k - 1]) / (N * apps_k))
    return {"fr_event_fraction": mean_frac, "max_fr_event_fraction": max_frac,
            "n_fr_applications": n_apps, "deaths_per_application": (total / n_apps if n_apps else 0.0)}


# ---------------------------------------------------------------------------
# Pentane hidden-conditional diagnostics
# ---------------------------------------------------------------------------
def _basins_1d(grid, barrier):
    g = np.asarray(grid, float)
    T = np.abs(g) < barrier
    Gp = g >= barrier
    Gm = g <= -barrier
    return T, Gp, Gm


def conditional_from_joint(joint_hist, grid2, dphi2, min_count=1.0):
    """p(phi2 | phi1_bin) from a joint (phi1,phi2) count histogram ``(G1,G2)``.

    Rows with < ``min_count`` total counts are returned as NaN (unsupported phi1).
    Also returns the phi1 weight (normalised row totals).
    """
    J = np.asarray(joint_hist, float)
    row = J.sum(1)
    w1 = row / (row.sum() + EPS)
    cond = np.full_like(J, np.nan)
    ok = row >= min_count
    cond[ok] = J[ok] / (row[ok][:, None] * dphi2 + EPS)
    return cond, w1, ok


def conditional_metrics(joint_hist, grid2, dphi2, ref_cond, ref_joint_weight, barrier):
    """Aggregate hidden-conditional error vs the reference p(phi2|phi1).

    Returns reference-weighted mean conditional TV and KL (over supported phi1 bins),
    plus per-basin conditional basin-probability errors and the fraction of phi1 bins
    with empirical support.  ``ref_cond`` is the reference ``(G1,G2)`` conditional and
    ``ref_joint_weight`` the reference phi1 marginal weight ``(G1,)``.
    """
    cond, w1, ok = conditional_from_joint(joint_hist, grid2, dphi2)
    rc = np.asarray(ref_cond, float)
    wref = np.asarray(ref_joint_weight, float)
    wref = wref / (wref.sum() + EPS)
    T, Gp, Gm = _basins_1d(grid2, barrier)
    tv_bins = np.full(cond.shape[0], np.nan)
    kl_bins = np.full(cond.shape[0], np.nan)
    basin_err = np.full(cond.shape[0], np.nan)
    for i in range(cond.shape[0]):
        if not ok[i]:
            continue
        p = np.clip(cond[i], 0, None); p = p / (p.sum() * dphi2 + EPS)
        r = np.clip(rc[i], 0, None); r = r / (r.sum() * dphi2 + EPS)
        tv_bins[i] = 0.5 * np.sum(np.abs(p - r)) * dphi2
        kl_bins[i] = float(np.sum(np.clip(p, EPS, None) * (np.log(np.clip(p, EPS, None)) - np.log(np.clip(r, EPS, None)))) * dphi2)
        pb = np.array([p[T].sum(), p[Gp].sum(), p[Gm].sum()]) * dphi2
        rb = np.array([r[T].sum(), r[Gp].sum(), r[Gm].sum()]) * dphi2
        basin_err[i] = 0.5 * np.sum(np.abs(pb - rb))
    # reference-weighted aggregate over supported bins (renormalise weights on support)
    m = ok & np.isfinite(tv_bins)
    wsup = wref[m] / (wref[m].sum() + EPS) if m.any() else np.array([])
    agg_tv = float(np.sum(wsup * tv_bins[m])) if m.any() else float("nan")
    agg_kl = float(np.sum(wsup * kl_bins[m])) if m.any() else float("nan")
    agg_basin = float(np.sum(wsup * basin_err[m])) if m.any() else float("nan")
    return {"cond_tv_weighted": agg_tv, "cond_kl_weighted": agg_kl,
            "cond_basin_err_weighted": agg_basin,
            "cond_support_fraction": float(ok.mean()),
            "cond_tv_bins": tv_bins, "cond_basin_err_bins": basin_err}


def joint_basin_visits(joint_hist, grid2, barrier):
    """Fraction of post-burn-in samples in each of the 9 (phi1,phi2) basins + #visited."""
    J = np.asarray(joint_hist, float)
    T, Gp, Gm = _basins_1d(grid2, barrier)
    masks = {"T": T, "G+": Gp, "G-": Gm}
    total = J.sum() + EPS
    out = {}
    n_visited = 0
    for n1, m1 in masks.items():
        for n2, m2 in masks.items():
            frac = J[np.ix_(m1, m2)].sum() / total
            out[f"basin_{n1}_{n2}"] = float(frac)
            if frac > 1e-4:
                n_visited += 1
    out["n_basins_visited"] = n_visited
    return out
