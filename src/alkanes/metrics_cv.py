"""Scientific metrics for the CV-extension runs (distance CV R15/R14 and joint torsion CV).

Distance-CV metrics live on a bounded interval with an optional thermal-region window;
the hidden-conditional diagnostic is the joint ``(phi1,phi2)`` torsion distribution within
R bins.  The 2-D torsion metrics reuse the dihedral-study conditional/basin machinery
(:mod:`alkanes.metrics`) since the 2-D ``joint_hist`` has exactly the ``(G1,G2)`` shape.
"""
from __future__ import annotations

import math

import numpy as np

from . import metrics as M

EPS = 1.0e-12
PI = math.pi
TWO_PI = 2.0 * PI


# ===========================================================================
# Distance CV (interval)
# ===========================================================================
def _interval_l2(profile, reference, dz, mask=None, align=True):
    p = np.asarray(profile, float); r = np.asarray(reference, float)
    if align:
        if mask is None:
            p = p - np.mean(p - r)
        else:
            p = p - np.sum((p - r) * mask) / max(np.sum(mask), EPS)
    if mask is None:
        return math.sqrt(np.sum((p - r) ** 2) * dz / (len(r) * dz))
    w = mask.astype(float)
    width = np.sum(w) * dz
    return math.sqrt(np.sum((p - r) ** 2 * w) * dz / max(width, EPS))


def dist_profile_metrics(pmf_series, mf_series, times, ref_F, ref_Fp, dz, thermal_mask):
    """Final + time-integrated windowed interval L2 of F and F' vs the reference (one seed)."""
    T = len(times)
    l2F = np.array([_interval_l2(pmf_series[k], ref_F, dz, thermal_mask, align=True) for k in range(T)])
    l2Fp = np.array([_interval_l2(mf_series[k], ref_Fp, dz, thermal_mask, align=False) for k in range(T)])
    intF = float(np.trapezoid(l2F, times)) if T > 1 else float("nan")
    intFp = float(np.trapezoid(l2Fp, times)) if T > 1 else float("nan")
    return {"final_l2_F": float(l2F[-1]), "final_l2_Fp": float(l2Fp[-1]),
            "integrated_l2_F": intF, "integrated_l2_Fp": intFp,
            "early_l2_F": float(l2F[min(1, T - 1)]), "mid_l2_F": float(l2F[T // 2]),
            "l2_F_series": l2F}


def dist_support_metrics(final_eff_counts, thermal_mask):
    """Fraction of thermally-relevant R bins with < 25% of median-bin support (starvation)."""
    ec = np.asarray(final_eff_counts, float)
    therm = ec[thermal_mask]
    if therm.size == 0:
        return {"low_support_fraction": float("nan"), "median_support": float("nan")}
    med = np.median(therm[therm > 0]) if np.any(therm > 0) else 0.0
    low = np.mean(therm < 0.25 * med) if med > 0 else 1.0
    return {"low_support_fraction": float(low), "median_support": float(med)}


def dist_conditional_metrics(cond_hist, ref_cond_dens, ref_cond_weight, cond_grid, dphi, barrier):
    """Hidden-conditional error for the distance CV: reference-weighted mean TV/KL of the
    joint ``(phi1,phi2)`` torsion distribution within R bins vs the reference, + 9-basin err.

    ``cond_hist`` ``(n_rbins, g2, g2)`` counts; ``ref_cond_dens`` ``(n_rbins, g2, g2)``
    densities; ``ref_cond_weight`` ``(n_rbins,)`` R-bin weights.
    """
    ch = np.asarray(cond_hist, float)
    rc = np.asarray(ref_cond_dens, float)
    wref = np.asarray(ref_cond_weight, float); wref = wref / (wref.sum() + EPS)
    nb = ch.shape[0]
    gnp = np.asarray(cond_grid, float)
    T = np.abs(gnp) < barrier; Gp = gnp >= barrier; Gm = gnp <= -barrier
    masks = [T, Gp, Gm]
    tv = np.full(nb, np.nan); kl = np.full(nb, np.nan); basin = np.full(nb, np.nan)
    ok = np.zeros(nb, bool)
    for k in range(nb):
        tot = ch[k].sum()
        if tot < 10:
            continue
        ok[k] = True
        p = ch[k] / (tot * dphi * dphi + EPS)
        r = rc[k]
        rs = r.sum() * dphi * dphi
        if rs < EPS:
            ok[k] = False; continue
        r = r / rs
        tv[k] = 0.5 * np.sum(np.abs(p - r)) * dphi * dphi
        kl[k] = float(np.sum(np.clip(p, EPS, None) * (np.log(np.clip(p, EPS, None)) - np.log(np.clip(r, EPS, None)))) * dphi * dphi)
        pb = np.array([[p[np.ix_(m1, m2)].sum() for m2 in masks] for m1 in masks]) * dphi * dphi
        rb = np.array([[r[np.ix_(m1, m2)].sum() for m2 in masks] for m1 in masks]) * dphi * dphi
        basin[k] = 0.5 * np.sum(np.abs(pb - rb))
    m = ok & np.isfinite(tv)
    wsup = wref[m] / (wref[m].sum() + EPS) if m.any() else np.array([])
    return {"dist_cond_tv_weighted": float(np.sum(wsup * tv[m])) if m.any() else float("nan"),
            "dist_cond_kl_weighted": float(np.sum(wsup * kl[m])) if m.any() else float("nan"),
            "dist_cond_basin_err_weighted": float(np.sum(wsup * basin[m])) if m.any() else float("nan"),
            "dist_cond_support_fraction": float(ok.mean())}


# ===========================================================================
# Joint torsion CV (2-D torus)
# ===========================================================================
def l2_2d_np(F, Fref, dz1, dz2, mask=None, align=True):
    F = np.asarray(F, float); Fref = np.asarray(Fref, float)
    if align:
        if mask is None:
            F = F - np.mean(F - Fref)
        else:
            F = F - np.sum((F - Fref) * mask) / max(np.sum(mask), EPS)
    if mask is None:
        return math.sqrt(np.sum((F - Fref) ** 2) * dz1 * dz2 / (TWO_PI * TWO_PI))
    w = mask.astype(float); area = np.sum(w) * dz1 * dz2
    return math.sqrt(np.sum((F - Fref) ** 2 * w) * dz1 * dz2 / max(area, EPS))


def joint_profile_metrics(pmf_series, times, ref_F, dz1, dz2, thermal_mask, eq_weight):
    """Final + integrated thermal-window and equilibrium-weighted 2-D L2(F) (one seed)."""
    T = len(times)
    l2_therm = np.array([l2_2d_np(pmf_series[k], ref_F, dz1, dz2, thermal_mask) for k in range(T)])
    # equilibrium-weighted: sqrt( sum w (F-Fref)^2 ), w = p_ref normalised
    def eqw(F):
        F = F - np.sum((F - ref_F) * eq_weight)      # align by eq-weighted mean
        return math.sqrt(np.sum(eq_weight * (F - ref_F) ** 2))
    l2_eq = np.array([eqw(pmf_series[k]) for k in range(T)])
    intF = float(np.trapezoid(l2_therm, times)) if T > 1 else float("nan")
    return {"final_l2_F": float(l2_therm[-1]), "integrated_l2_F": intF,
            "final_l2_F_eqw": float(l2_eq[-1]), "early_l2_F": float(l2_therm[min(1, T - 1)]),
            "mid_l2_F": float(l2_therm[T // 2]), "l2_F_series": l2_therm}


def reconstructed_fidelity(F_hat, F_ref, grid, beta, barrier):
    """Basin/conditional fidelity of the RECONSTRUCTED 2-D free energy (not the biased
    histogram, which ABF flattens toward uniform by design).

    Compares equilibrium basin populations ``exp(-beta F_hat)`` and the conditional
    ``p(phi2|phi1)`` derived from ``F_hat`` against the reference, weighted by the
    reference phi1 marginal. Returns basin-occupancy TV and reference-weighted conditional
    TV/KL -- all meaningful for a biased run because they are computed from ``F_hat``.
    """
    from . import reference as refmod
    F_hat = np.asarray(F_hat, float); F_ref = np.asarray(F_ref, float)
    g = np.asarray(grid, float); dphi = float(g[1] - g[0])
    cond_hat = refmod.conditional_phi2_given_phi1(F_hat, g, beta)
    cond_ref = refmod.conditional_phi2_given_phi1(F_ref, g, beta)
    w = np.exp(-beta * (F_ref - F_ref.min())).sum(1); w = w / (w.sum() + EPS)   # phi1 weight
    tv_bins = 0.5 * np.sum(np.abs(cond_hat - cond_ref), axis=1) * dphi
    kl_bins = np.sum(np.clip(cond_hat, EPS, None) *
                     (np.log(np.clip(cond_hat, EPS, None)) - np.log(np.clip(cond_ref, EPS, None))), axis=1) * dphi
    pe_hat = np.exp(-beta * (F_hat - F_hat.min())); pe_hat = pe_hat / (pe_hat.sum() + EPS)
    pe_ref = np.exp(-beta * (F_ref - F_ref.min())); pe_ref = pe_ref / (pe_ref.sum() + EPS)
    T = np.abs(g) < barrier; Gp = g >= barrier; Gm = g <= -barrier
    masks = [T, Gp, Gm]
    bh = np.array([pe_hat[np.ix_(m1, m2)].sum() for m1 in masks for m2 in masks])
    br = np.array([pe_ref[np.ix_(m1, m2)].sum() for m1 in masks for m2 in masks])
    return {"basin_occupancy_tv": 0.5 * float(np.sum(np.abs(bh - br))),
            "cond_tv_weighted": float(np.sum(w * tv_bins)),
            "cond_kl_weighted": float(np.sum(w * kl_bins))}


def meanforce_vector_error(F, ref_F, dz1, dz2, mask=None):
    """RMS ||grad F_hat - grad F_ref|| over the torus (spectral gradient)."""
    import torch
    from . import poisson2d as ps
    Ft = torch.as_tensor(np.asarray(F, float))[None]
    Rt = torch.as_tensor(np.asarray(ref_F, float))[None]
    g1, g2 = ps.spectral_gradient(Ft, dz1, dz2)
    r1, r2 = ps.spectral_gradient(Rt, dz1, dz2)
    err = ((g1 - r1) ** 2 + (g2 - r2) ** 2)[0].numpy()
    if mask is None:
        return float(math.sqrt(np.mean(err)))
    return float(math.sqrt(np.sum(err * mask) / max(np.sum(mask), EPS)))
