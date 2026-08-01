"""Category-B diagnostics: FES error, mean-force error, basin, genealogy, cost.

Every error is computed on a support mask derived from the REFERENCE ALONE and therefore common
to both arms -- an arm-dependent mask would structurally flatter whichever method concentrates
population where it is already accurate.

Weightings reported (the sign of any claimed effect must agree across all three):
  * ``equilibrium``  Boltzmann weight of the reference on Omega_eval
  * ``uniform8``     uniform on {F_ref - F_min <= 8 kT}
  * ``uniform10``    uniform on {F_ref - F_min <= 10 kT}

``kernel_matched`` compares the arm's estimate against ``K_h * F_ref`` rather than ``F_ref``.
The ABF estimator is a Nadaraya--Watson ratio, so its converged target is the *smoothed*
reference; scoring against the unsmoothed one charges every arm a fixed bandwidth penalty and,
worse, the penalty depends on occupancy, which is exactly what mFR changes.
"""
from __future__ import annotations

import math

import numpy as np

TWO_PI = 2.0 * math.pi


# --------------------------------------------------------------------------- masks / weights
def build_masks(F_ref, kT):
    """Reference-only masks and weights.  ``F_ref`` may contain non-finite (unvisited) cells."""
    finite = np.isfinite(F_ref)
    F = np.where(finite, F_ref, np.inf)
    F = F - F[finite].min()
    m8 = finite & (F <= 8.0 * kT)
    m10 = finite & (F <= 10.0 * kT)
    w_eq = np.where(m8, np.exp(-np.where(m8, F, 0.0) / kT), 0.0)
    w_eq = w_eq / w_eq.sum()
    w_u8 = m8.astype(float) / max(m8.sum(), 1)
    w_u10 = m10.astype(float) / max(m10.sum(), 1)
    return dict(F=F, finite=finite, mask8=m8, mask10=m10,
                weights=dict(equilibrium=w_eq, uniform8=w_u8, uniform10=w_u10))


def smooth_reference(F_ref, bandwidth_rad, n_grid):
    """``K_h * F_ref`` with the estimator's own wrapped-Gaussian kernel (kernel matching)."""
    import torch
    from alkanes import density2d as d2
    g1, g2, dz1, dz2 = d2.torus_grid(n_grid, n_grid, dtype=torch.float64)
    K1, K2 = d2.kernels(g1, g2, bandwidth_rad, bandwidth_rad)
    finite = np.isfinite(F_ref)
    F = np.where(finite, F_ref, np.nan)
    fill = np.nanmax(F[finite])
    Ft = torch.as_tensor(np.where(finite, F_ref, fill), dtype=torch.float64)[None]
    return d2.smooth2(Ft, K1, K2)[0].numpy()


# --------------------------------------------------------------------------- FES / mean force
def aligned_l2(F_hat, F_ref, w):
    """``min_c || F_hat - F_ref - c ||`` under weight ``w`` (additive constant is arbitrary).

    Restricted to cells with positive weight AND finite reference: the reference carries +inf in
    unvisited bins, and ``inf * 0`` is NaN, so multiplying through by a zero weight is not enough
    to exclude them.
    """
    ok = (w > 0) & np.isfinite(F_ref) & np.isfinite(F_hat)
    if not ok.any():
        return float("nan")
    d = F_hat[ok] - F_ref[ok]
    ww = w[ok]
    c = float((d * ww).sum() / ww.sum())
    return float(math.sqrt(((d - c) ** 2 * ww).sum() / ww.sum()))


def fes_errors(F_hat, ref_pack, F_ref_smoothed=None):
    """All weightings, raw and kernel-matched, for one FES snapshot."""
    out = {}
    F = ref_pack["F"]
    for name, w in ref_pack["weights"].items():
        out[f"eF_{name}"] = aligned_l2(F_hat, F, w)
        if F_ref_smoothed is not None:
            out[f"eF_km_{name}"] = aligned_l2(F_hat, F_ref_smoothed, w)
    return out


def grad_errors(F_hat, F_ref, w, n_grid):
    """``|| grad F_hat - grad F_ref ||`` with the same spectral derivative for both arms."""
    import torch
    from alkanes import poisson2d as ps
    dz = TWO_PI / n_grid
    a1, a2 = ps.spectral_gradient(torch.as_tensor(F_hat, dtype=torch.float64)[None], dz, dz)
    finite = np.isfinite(F_ref)
    Fr = np.where(finite, F_ref, np.nanmax(F_ref[finite]))
    b1, b2 = ps.spectral_gradient(torch.as_tensor(Fr, dtype=torch.float64)[None], dz, dz)
    d = ((a1 - b1) ** 2 + (a2 - b2) ** 2)[0].numpy()
    return float(math.sqrt((d * w).sum() / max(w.sum(), 1e-300)))


def integrated(times, values, t0, t1):
    """Trapezoidal integral of a metric over the declared analysis window."""
    t = np.asarray(times, dtype=float)
    v = np.asarray(values, dtype=float)
    sel = (t >= t0) & (t <= t1)
    if sel.sum() < 2:
        return float("nan")
    return float(np.trapezoid(v[sel], t[sel]))


def time_series(out, ref_pack, F_ref_smoothed, n_grid, window):
    """Per-save-point metrics for one run (all seeds)."""
    times = np.asarray(out["times"], dtype=float)
    pmf = out["pmf"]                                    # (T, R, n, n)
    R = pmf.shape[1]
    rows = []
    for ti, t in enumerate(times):
        for r in range(R):
            F_hat = pmf[ti, r]
            rec = dict(t_ps=float(t), seed=int(out["seeds"][r]), method=str(out["method"]))
            rec.update(fes_errors(F_hat, ref_pack, F_ref_smoothed))
            for nm, w in ref_pack["weights"].items():
                rec[f"egradF_{nm}"] = grad_errors(F_hat, ref_pack["F"], w, n_grid)
            rows.append(rec)
    per_seed = []
    for r in range(R):
        sub = [x for x in rows if x["seed"] == int(out["seeds"][r])]
        tt = [x["t_ps"] for x in sub]
        rec = dict(seed=int(out["seeds"][r]), method=str(out["method"]))
        for key in sub[0]:
            if key.startswith(("eF_", "egradF_")):
                rec[f"int_{key}"] = integrated(tt, [x[key] for x in sub], *window)
                rec[f"final_{key}"] = float(sub[-1][key])
        per_seed.append(rec)
    return rows, per_seed


# --------------------------------------------------------------------------- genealogy / cost
def _rare_key(out, new, old):
    """Read a rare-basin diagnostic, accepting the pre-rename alanine artifacts.

    The keys were ``wmax_c7ax`` / ``ess_age_c7ax`` until the tracked basin became a parameter.
    Artifacts written before that rename are still on disk and are still the accepted alanine
    result, so both spellings are read; only the new one is ever written.
    """
    return out[new] if new in out else out[old]


def genealogy_summary(out, window_ps, fr_start_steps=20000, fr_every=500):
    """Age-aware ESS, max ancestor fraction (global and in the tracked rare basin), events."""
    t = np.asarray(out["times"], dtype=float)
    sel = (t >= window_ps[0]) & (t <= window_ps[1])
    R = out["ess_age"].shape[1]
    rows = []
    for r in range(R):
        ev = float(out["total_events"][r])
        # Per-OPPORTUNITY event fraction is the scale-free quantity and is what
        # max_event_fraction caps.  The cumulative fraction grows with run length (2.6% over the
        # calibration's 21 opportunities becomes ~21% over the pilot's 161 for the same rate),
        # so it cannot be compared against a fixed threshold across different run lengths.
        # Both are reported; the gate uses per-opportunity, and the genealogy criteria
        # (age-aware ESS, max ancestor share) are the substantive turnover guard.
        n_opp = max((int(out["n_steps"]) - fr_start_steps) // fr_every + 1, 1)
        rows.append(dict(
            seed=int(out["seeds"][r]), method=str(out["method"]),
            ess_age_min=float(np.nanmin(out["ess_age"][sel, r])),
            ess_age_mean=float(np.nanmean(out["ess_age"][sel, r])),
            ess_perm_min=float(np.nanmin(out["ess_perm"][sel, r])),
            wmax_max=float(np.nanmax(out["wmax"][sel, r])),
            wmax_rare_max=float(np.nanmax(_rare_key(out, "wmax_rare", "wmax_c7ax")[sel, r])),
            ess_age_rare_min=float(
                np.nanmin(_rare_key(out, "ess_age_rare", "ess_age_c7ax")[sel, r])),
            n_events=ev, n_opportunities=int(n_opp),
            event_fraction=ev / max(out["n_replicas"], 1) / n_opp,
            event_fraction_cumulative=ev / max(out["n_replicas"], 1),
            n_unique_min=float(np.nanmin(out["n_unique"][sel, r]))))
    return rows


def basin_summary(out, basin_names, window_ps, dt):
    """Occupancy, first hit, establishment time, entries and round trips per seed."""
    t = np.asarray(out["times"], dtype=float)
    sel = (t >= window_ps[0]) & (t <= window_ps[1])
    frac = out["basin_frac"]                                # (T, R, n_basins)
    R = frac.shape[1]
    rows = []
    for r in range(R):
        rec = dict(seed=int(out["seeds"][r]), method=str(out["method"]))
        for k, nm in enumerate(basin_names):
            fh = float(out["first_hit"][r, k])
            rec[f"occ_{nm}"] = float(np.nanmean(frac[sel, r, k]))
            rec[f"first_hit_ps_{nm}"] = (fh * dt) if fh >= 0 else float("nan")
            rec[f"entries_{nm}"] = int(out["trans_matrix"][r, :, k].sum())
        rows.append(rec)
    return rows


def cost_summary(out, meta=None):
    """Cost fields; scalars live in the run manifest (``meta``), arrays in the npz."""
    meta = meta or {}
    g = lambda k, d=float("nan"): float(meta.get(k, out[k] if k in out else d))   # noqa: E731
    return dict(method=str(out["method"]), ms_per_step=g("ms_per_step"),
                wall_seconds=g("wall_seconds"),
                force_evaluations=g("force_evaluations"),
                aggregate_simulated_ps=g("aggregate_simulated_ps"),
                peak_cuda_gib=g("peak_cuda_gib"), clip_fraction=g("clip_fraction"),
                n_replicas=int(out["n_replicas"]), n_steps=int(out["n_steps"]))


# --------------------------------------------------------------------------- paired statistics
def paired_bootstrap(a, b, n_boot=10000, seed=20260901, ci=95.0):
    """Paired BCa-style bootstrap of the median relative change ``(b - a) / a``.

    Returns median, CI bounds, win rate.  ``a`` is the baseline (abf), ``b`` the treatment.
    A negative value means the treatment reduced the error.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    rel = (b - a) / np.where(np.abs(a) > 0, a, np.nan)
    rel = rel[np.isfinite(rel)]
    if rel.size == 0:
        return dict(median=float("nan"), lo=float("nan"), hi=float("nan"),
                    win_rate=float("nan"), n=0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, rel.size, size=(n_boot, rel.size))
    meds = np.median(rel[idx], axis=1)
    lo, hi = np.percentile(meds, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    # BCa bias correction
    z0 = 0.0
    prop = float((meds < np.median(rel)).mean())
    if 0.0 < prop < 1.0:
        from scipy.stats import norm
        z0 = norm.ppf(prop)
        al = norm.cdf(2 * z0 + norm.ppf((100 - ci) / 200))
        ah = norm.cdf(2 * z0 + norm.ppf(1 - (100 - ci) / 200))
        lo, hi = np.percentile(meds, [al * 100, ah * 100])
    return dict(median=float(np.median(rel)), lo=float(lo), hi=float(hi),
                win_rate=float((rel < 0).mean()), n=int(rel.size), z0=float(z0))
