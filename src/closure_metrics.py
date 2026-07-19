"""Shared, reviewer-proof metric pipeline for the OPES closure (Part B).

Consumes a per-run result dict / npz (the `execute_opes_run` schema, which OPES,
ABF and mFR all emit) and returns a FLAT dict of comparison metrics in a common
schema, so every method and system is scored the same way. Pure post-hoc: reads
stored profiles + time series only; never re-runs dynamics and never touches the
reference except for L2 evaluation (no oracle leakage into any estimator).

Metric families (all keyed identically across methods):
  accuracy      : l2_f, l2_fp (native + common estimators, labeled distinctly),
                  windowed (compact/transition/stretched)
  anytime       : integrated_l2_f, normalized_anytime_l2_f, tau_abs, tau_rel
  marginal      : marginal_kl, marginal_tv, marginal_l2_ref, marginal_entropy,
                  min_density, covered_fraction
  mixing        : n_round_trips, directional transition counts, first_passage_step,
                  iat_frac_compact, mixing_ess
  opes_health   : neff_frac (final/min), n_kernels, bias_range, sigma_cur, clip proxy
  cost          : runtime_seconds, wall_seconds, budget, force_evaluations, device
"""
from __future__ import annotations
import json
import numpy as np

EPS = 1e-12


def _get(d, k, default=None):
    """npz-and-dict tolerant getter; unwraps 0-d numpy arrays to python scalars."""
    if k not in (d.files if hasattr(d, "files") else d):
        return default
    v = d[k]
    if hasattr(v, "ndim") and v.ndim == 0:
        v = v.item()
    return v


def _arr(d, k):
    v = _get(d, k, None)
    if v is None:
        return np.array([])
    return np.asarray(v, dtype=float)


def _trapz(y, x):
    return float(np.trapezoid(y, x))


def _finite(x, default=float("nan")):
    x = float(x)
    return x if np.isfinite(x) else default


# --------------------------- marginal-quality metrics ------------------------
def marginal_metrics(grid, p_hat, p_ref):
    """KL, TV, L2, entropy of the sampled CV marginal vs the Boltzmann reference,
    plus coverage stats. p_hat, p_ref are densities on `grid` (need not be pre-normed)."""
    out = {}
    if grid.size == 0 or p_hat.size != grid.size:
        return dict(marginal_kl=float("nan"), marginal_tv=float("nan"),
                    marginal_entropy=float("nan"), min_density=float("nan"),
                    covered_fraction=float("nan"))
    ph = np.clip(p_hat, 0, None); pr = np.clip(p_ref, 0, None)
    zh = _trapz(ph, grid); zr = _trapz(pr, grid)
    ph = ph / zh if zh > 0 else ph; pr = pr / zr if zr > 0 else pr
    # KL(p_hat || p_ref) and total variation
    integ = ph * (np.log(ph + EPS) - np.log(pr + EPS))
    out["marginal_kl"] = _finite(_trapz(integ, grid))
    out["marginal_tv"] = _finite(0.5 * _trapz(np.abs(ph - pr), grid))
    out["marginal_entropy"] = _finite(-_trapz(ph * np.log(ph + EPS), grid))
    # coverage: fraction of grid where sampled density exceeds a small floor
    floor = 1e-3 * float(np.max(ph)) if np.max(ph) > 0 else 0.0
    out["min_density"] = _finite(float(np.min(ph)))
    out["covered_fraction"] = _finite(float(np.mean(ph > floor)))
    return out


def anytime_metrics(times, l2_f_t, integrated_l2_f, l2_f_final):
    """Convergence-speed metrics from the L2(t) trajectory.
      integrated_l2_f          : ∫ err dt (already stored; recomputed if absent)
      normalized_anytime_l2_f  : integrated err / total time (time-averaged error)
      tau_abs                  : first time err <= 0.25 (absolute threshold, kT-scale)
      tau_rel                  : first time err <= 2x its final value (self-relative)
    """
    out = dict(integrated_l2_f=_finite(integrated_l2_f),
               normalized_anytime_l2_f=float("nan"),
               tau_abs=float("nan"), tau_rel=float("nan"))
    if times.size < 2 or l2_f_t.size != times.size:
        return out
    T = float(times[-1] - times[0])
    ii = np.trapezoid(l2_f_t, times)
    if not np.isfinite(out["integrated_l2_f"]):
        out["integrated_l2_f"] = _finite(ii)
    out["normalized_anytime_l2_f"] = _finite(ii / T) if T > 0 else float("nan")
    def _first_below(thr):
        below = np.where(l2_f_t <= thr)[0]
        return float(times[below[0]]) if below.size else float("nan")
    out["tau_abs"] = _first_below(0.25)
    if np.isfinite(l2_f_final) and l2_f_final > 0:
        out["tau_rel"] = _first_below(2.0 * l2_f_final)
    return out


# ------------------------------- mixing metrics ------------------------------
def mixing_metrics(d):
    """Transition/mixing diagnostics from stored crossing counts + occupancy series."""
    out = {}
    for k in ("n_round_trips", "n_barrier_crossings",
              "n_compact_to_stretched", "n_stretched_to_compact"):
        out[k] = _finite(_get(d, k, float("nan")))
    fc = _arr(d, "frac_compact"); fs = _arr(d, "frac_stretched"); times = _arr(d, "times")
    # first passage: first time stretched-basin occupancy first exceeds 0.1 from a
    # compact start (proxy for the ensemble first reaching the far basin).
    fp = float("nan")
    if fs.size and times.size == fs.size and fs[0] < 0.1:
        idx = np.where(fs > 0.1)[0]
        fp = float(times[idx[0]]) if idx.size else float("nan")
    out["first_passage_step"] = fp
    # integrated autocorrelation proxy of frac_compact (lag-1 AR estimate => IAT)
    iat = float("nan")
    if fc.size > 4:
        x = fc - fc.mean()
        denom = float(np.dot(x, x))
        if denom > 0:
            r1 = float(np.dot(x[:-1], x[1:])) / denom
            r1 = min(max(r1, -0.999), 0.999)
            iat = (1.0 + r1) / (1.0 - r1)
    out["iat_frac_compact"] = _finite(iat)
    out["mixing_ess"] = _finite(times.size / iat) if np.isfinite(iat) and iat > 0 else float("nan")
    return out


def opes_health(d):
    out = {}
    out["opes_neff_frac_final"] = _finite(_get(d, "opes_neff_frac_final", float("nan")))
    out["opes_neff_frac_min"] = _finite(_get(d, "opes_neff_frac_min", float("nan")))
    out["opes_n_kernels_final"] = _finite(_get(d, "opes_n_kernels_final", float("nan")))
    out["opes_bias_range_final"] = _finite(_get(d, "opes_bias_range_final", float("nan")))
    mb = _arr(d, "opes_log_max_bias"); sc = _arr(d, "opes_log_sigma_cur")
    out["opes_max_bias"] = _finite(float(np.nanmax(mb))) if mb.size else float("nan")
    out["opes_sigma_cur_final"] = _finite(float(sc[-1])) if sc.size else float("nan")
    return out


def cost_metrics(d):
    n_steps = _get(d, "n_steps", float("nan")); n_rep = _get(d, "n_replicas", float("nan"))
    fe = float("nan")
    try:
        fe = float(n_steps) * float(n_rep)
    except Exception:
        pass
    return dict(runtime_seconds=_finite(_get(d, "runtime_seconds", float("nan"))),
                wall_seconds=_finite(_get(d, "wall_seconds", float("nan"))),
                budget=_finite(_get(d, "budget", float("nan"))),
                force_evaluations=_finite(fe),
                device=str(_get(d, "device", "")),
                cuda_visible_devices=str(_get(d, "cuda_visible_devices", "")))


# ------------------------------- main entry ----------------------------------
IDENTITY_KEYS = ("run_id", "study", "stage", "name", "method", "seed",
                 "beta", "h", "w", "n_dim", "a", "beta_h",
                 "opes_barrier", "opes_pace", "opes_sigma", "opes_gamma",
                 "opes_gamma_from_barrier", "n_steps", "n_replicas", "had_nan")


def compute_metrics(d):
    """Return the full flat closure-metric row for one run dict/npz.

    Estimator labeling (NO conflation): accuracy is reported for BOTH
      - common mean-force estimator  -> l2_f_common, l2_fp_common   (== stored l2_f/l2_fp)
      - native OPES reweight         -> l2_f_native, l2_fp_native   (== stored *_reweight)
    """
    row = {}
    for k in IDENTITY_KEYS:
        row[k] = _get(d, k, None)
    # ---- accuracy, both estimators, labeled distinctly ----
    row["l2_f_common"] = _finite(_get(d, "l2_f", float("nan")))
    row["l2_fp_common"] = _finite(_get(d, "l2_fp", float("nan")))
    row["l2_f_native"] = _finite(_get(d, "l2_f_reweight", float("nan")))
    row["l2_fp_native"] = _finite(_get(d, "l2_fp_reweight", float("nan")))
    row["primary_estimator"] = str(_get(d, "opes_estimator", "meanforce"))
    for w in ("compact", "transition", "stretched"):
        row[f"l2_f_{w}"] = _finite(_get(d, f"l2_f_{w}", float("nan")))
        row[f"l2_fp_{w}"] = _finite(_get(d, f"l2_fp_{w}", float("nan")))
    # ---- anytime / convergence ----
    row.update(anytime_metrics(_arr(d, "times"), _arr(d, "l2_f_t"),
                               _get(d, "integrated_l2_f", float("nan")), row["l2_f_common"]))
    # ---- marginal quality ----
    grid = _arr(d, "grid"); p_hat = _arr(d, "final_p_hat"); p_ref = _arr(d, "ref_p_boltzmann")
    row.update(marginal_metrics(grid, p_hat, p_ref))
    row["marginal_l2_ref"] = _finite(_get(d, "marginal_l2_ref", float("nan")))
    row["marginal_l2_uniform"] = _finite(_get(d, "marginal_l2_uniform", float("nan")))
    # ---- mixing, health, cost ----
    row.update(mixing_metrics(d))
    row.update(opes_health(d))
    row.update(cost_metrics(d))
    return row


def metrics_schema():
    """Deterministic column order for the common CSV (identity first)."""
    probe = {k: None for k in IDENTITY_KEYS}
    dummy = {"n_steps": 1, "n_replicas": 1}
    probe.update({k: None for k in (
        "l2_f_common","l2_fp_common","l2_f_native","l2_fp_native","primary_estimator",
        "l2_f_compact","l2_fp_compact","l2_f_transition","l2_fp_transition",
        "l2_f_stretched","l2_fp_stretched","integrated_l2_f","normalized_anytime_l2_f",
        "tau_abs","tau_rel","marginal_kl","marginal_tv","marginal_entropy","min_density",
        "covered_fraction","marginal_l2_ref","marginal_l2_uniform","n_round_trips",
        "n_barrier_crossings","n_compact_to_stretched","n_stretched_to_compact",
        "first_passage_step","iat_frac_compact","mixing_ess","opes_neff_frac_final",
        "opes_neff_frac_min","opes_n_kernels_final","opes_bias_range_final","opes_max_bias",
        "opes_sigma_cur_final","runtime_seconds","wall_seconds","budget","force_evaluations",
        "device","cuda_visible_devices")})
    return list(probe.keys())
