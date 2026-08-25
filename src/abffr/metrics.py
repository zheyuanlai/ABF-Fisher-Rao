"""Error metrics and summary statistics for ABF--FR runs.

Free energy is defined up to an additive constant, so every free-energy profile
is re-centred (``F <- F - mean(F)`` over the evaluation mask) before an L2 error
is computed.  The mean force ``F'`` is compared directly (no centring).

L2 errors are domain-RMS values, ``sqrt( integral_mask (a - b)^2 dx / |mask| )``,
restricted to an interior evaluation window so KDE/ABF edge artefacts near the
reflecting walls do not dominate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class EvalConfig:
    """Evaluation window and reaction-coordinate region definitions."""

    eval_x_min: float = -2.5
    eval_x_max: float = 2.5
    x_barrier: float = 0.0
    barrier_half_width: float = 0.5
    left_basin_min: float = -2.5
    right_basin_max: float = 2.5

    @classmethod
    def from_domain(cls, domain: Dict[str, float], margin: float = 0.5):
        xmn, xmx = domain["x_min"], domain["x_max"]
        return cls(eval_x_min=xmn + margin, eval_x_max=xmx - margin,
                   left_basin_min=xmn + margin, right_basin_max=xmx - margin)

    def eval_mask(self, x_grid: np.ndarray) -> np.ndarray:
        return (x_grid >= self.eval_x_min) & (x_grid <= self.eval_x_max)


def center(profile: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    """Return ``profile - mean(profile)`` (mean taken over ``mask`` if given)."""
    profile = np.asarray(profile, dtype=float)
    if mask is None:
        return profile - np.mean(profile)
    return profile - np.mean(profile[mask])


def _domain_rms(a: np.ndarray, b: np.ndarray, x: np.ndarray, mask: np.ndarray) -> float:
    xa = x[mask]
    d2 = (a[mask] - b[mask]) ** 2
    width = xa[-1] - xa[0]
    if width <= 0:
        return float(np.sqrt(np.mean(d2)))
    return float(np.sqrt(np.trapezoid(d2, xa) / width))


def l2_error_F(F_hat: np.ndarray, F_ref: np.ndarray, x_grid: np.ndarray,
               mask: np.ndarray) -> float:
    """Centred L2 error of the free-energy profile over ``mask``."""
    a = center(F_hat, mask)
    b = center(F_ref, mask)
    return _domain_rms(a, b, x_grid, mask)


def l2_error_Fprime(Fp_hat: np.ndarray, Fp_ref: np.ndarray, x_grid: np.ndarray,
                    mask: np.ndarray) -> float:
    """L2 error of the mean-force profile over ``mask`` (no centring)."""
    return _domain_rms(Fp_hat, Fp_ref, x_grid, mask)


def integrated_l2_over_time(l2_series: List[float], times: List[float]) -> float:
    """Time-integral ``integral_0^T ||.||(t) dt`` by the trapezoidal rule."""
    l2 = np.asarray(l2_series, dtype=float)
    t = np.asarray(times, dtype=float)
    finite = np.isfinite(l2)
    if finite.sum() < 2:
        return float("nan")
    return float(np.trapezoid(l2[finite], t[finite]))


def marginal_l2_to_uniform(p_grid: np.ndarray, x_grid: np.ndarray,
                           mask: np.ndarray) -> float:
    """L2 distance of the (renormalised) x-marginal to the uniform density."""
    xa = x_grid[mask]
    p = np.asarray(p_grid, dtype=float)[mask]
    p = p / max(np.trapezoid(p, xa), 1e-300)
    u = np.full_like(xa, 1.0 / (xa[-1] - xa[0]))
    width = xa[-1] - xa[0]
    return float(np.sqrt(np.trapezoid((p - u) ** 2, xa) / width))


def marginal_l2_to_target(p_grid: np.ndarray, q_grid: np.ndarray,
                          x_grid: np.ndarray, mask: np.ndarray) -> float:
    """L2 distance of the x-marginal to a target density ``q`` over ``mask``."""
    xa = x_grid[mask]
    p = np.asarray(p_grid, dtype=float)[mask]
    q = np.asarray(q_grid, dtype=float)[mask]
    p = p / max(np.trapezoid(p, xa), 1e-300)
    q = q / max(np.trapezoid(q, xa), 1e-300)
    width = xa[-1] - xa[0]
    return float(np.sqrt(np.trapezoid((p - q) ** 2, xa) / width))



def marginal_l2_to_physical_ref(p_grid: np.ndarray, p_ref: np.ndarray,
                                x_grid: np.ndarray, mask: np.ndarray) -> float:
    """Full-domain L2 distance to the independent physical RC reference.

    The physical marginal is a distribution on the complete reaction-coordinate
    domain.  Unlike the legacy target diagnostic, this metric must retain mass
    allocation near the edge strips rather than condition and renormalise on the
    interior ABF evaluation mask.  ``mask`` is accepted for API compatibility.
    """
    del mask
    x = np.asarray(x_grid, dtype=float)
    p = np.asarray(p_grid, dtype=float)
    q = np.asarray(p_ref, dtype=float)
    p = p / max(np.trapezoid(p, x), 1e-300)
    q = q / max(np.trapezoid(q, x), 1e-300)
    width = x[-1] - x[0]
    if width <= 0:
        return float(np.sqrt(np.mean((p - q) ** 2)))
    return float(np.sqrt(np.trapezoid((p - q) ** 2, x) / width))


def free_energy_landmark_errors(
        F_hat: np.ndarray, F_ref: np.ndarray, x_grid: np.ndarray,
        mask: np.ndarray, x_barrier: float = 0.0) -> Dict[str, float]:
    """Evaluation-only unequal-basin and barrier-height errors.

    Reference minima are selected independently on either side of x_barrier;
    the reference maximum between them defines the intervening barrier. These
    landmarks are never exposed to the dynamics or FR target.
    """
    left = np.flatnonzero(mask & (x_grid < x_barrier))
    right = np.flatnonzero(mask & (x_grid > x_barrier))
    if left.size == 0 or right.size == 0:
        return dict(
            deltaF_hat=float("nan"), deltaF_ref=float("nan"),
            deltaF_error=float("nan"),
            barrier_height_error=float("nan"))
    i_left = int(left[np.argmin(F_ref[left])])
    i_right = int(right[np.argmin(F_ref[right])])
    lo, hi = sorted((i_left, i_right))
    between = np.arange(lo, hi + 1)
    i_barrier = int(between[np.argmax(F_ref[between])])

    delta_hat = float(F_hat[i_right] - F_hat[i_left])
    delta_ref = float(F_ref[i_right] - F_ref[i_left])
    barrier_hat = np.array([
        F_hat[i_barrier] - F_hat[i_left],
        F_hat[i_barrier] - F_hat[i_right],
    ], dtype=float)
    barrier_ref = np.array([
        F_ref[i_barrier] - F_ref[i_left],
        F_ref[i_barrier] - F_ref[i_right],
    ], dtype=float)
    return dict(
        deltaF_hat=delta_hat,
        deltaF_ref=delta_ref,
        deltaF_error=float(abs(delta_hat - delta_ref)),
        barrier_height_error=float(np.sqrt(np.mean(
            (barrier_hat - barrier_ref) ** 2))),
    )

def region_fractions(X: np.ndarray, ev: EvalConfig) -> Dict[str, float]:
    """Fraction of particles in the left basin, barrier and right basin."""
    X = np.asarray(X, dtype=float)
    n = len(X)
    lo = ev.x_barrier - ev.barrier_half_width
    hi = ev.x_barrier + ev.barrier_half_width
    left = (X >= ev.left_basin_min) & (X < lo)
    barrier = (X >= lo) & (X <= hi)
    right = (X > hi) & (X <= ev.right_basin_max)
    return dict(
        frac_left=float(np.count_nonzero(left) / n),
        frac_barrier=float(np.count_nonzero(barrier) / n),
        frac_right=float(np.count_nonzero(right) / n),
    )


def time_series_metrics(diag: Dict, x_grid: np.ndarray, F_ref: np.ndarray,
                        Fprime_ref: np.ndarray, ev: EvalConfig,
                        p_ref: np.ndarray = None) -> List[Dict]:
    """Per-snapshot independently referenced metrics in long format."""
    mask = ev.eval_mask(x_grid)
    rows = []
    optional = {
        "ancestor_ess": float,
        "max_clone_multiplicity": int,
        "max_clone_weight": float,
        "cumulative_fr_events": int,
        "cumulative_replacements": int,
    }
    for k, step in enumerate(diag["steps"]):
        F_hat = diag["F_hat"][k]
        Fp_hat = diag["Fprime_hat"][k]
        p_hat = diag["p_hat_grid"][k]
        q = diag["q_target_grid"][k]
        landmarks = free_energy_landmark_errors(
            F_hat, F_ref, x_grid, mask, x_barrier=ev.x_barrier)
        row = dict(
            step=int(step),
            t=float(diag["times"][k]),
            l2_F=l2_error_F(F_hat, F_ref, x_grid, mask),
            l2_Fprime=l2_error_Fprime(Fp_hat, Fprime_ref, x_grid, mask),
            marginal_l2_uniform=marginal_l2_to_uniform(p_hat, x_grid, mask),
            marginal_l2_target=marginal_l2_to_target(p_hat, q, x_grid, mask),
            marginal_l2_physical_ref=(
                marginal_l2_to_physical_ref(p_hat, p_ref, x_grid, mask)
                if p_ref is not None else float("nan")),
            barrier_crossings=int(diag["barrier_crossings"][k]),
            n_unique_ancestors=int(diag["n_unique_ancestors"][k]),
            gamma_eff=float(diag["gamma_eff"][k]),
            fr_applied=bool(diag["fr_applied"][k]),
            fr_event_fraction=float(diag["fr_event_fraction"][k]),
            fr_event_fraction_max=float(diag["fr_event_fraction_max"][k]),
            fr_events_total=int(diag["fr_events_total"][k]),
            score_mean=float(diag["score_mean"][k]),
            score_std=float(diag["score_std"][k]),
            score_min=float(diag["score_min"][k]),
            score_max=float(diag["score_max"][k]),
            **landmarks,
        )
        for key, cast in optional.items():
            if key in diag and k < len(diag[key]):
                row[key] = cast(diag[key][k])
        rows.append(row)
    return rows


def final_summary(diag: Dict, x_grid: np.ndarray, F_ref: np.ndarray,
                  Fprime_ref: np.ndarray, ev: EvalConfig,
                  p_ref: np.ndarray = None) -> Dict:
    """Final-time and full-run integrated metrics for one run."""
    ts = time_series_metrics(
        diag, x_grid, F_ref, Fprime_ref, ev, p_ref=p_ref)
    l2_F_series = [r["l2_F"] for r in ts]
    l2_Fp_series = [r["l2_Fprime"] for r in ts]
    physical_series = [r["marginal_l2_physical_ref"] for r in ts]
    deltaF_series = [r["deltaF_error"] for r in ts]
    barrier_series = [r["barrier_height_error"] for r in ts]
    times = [r["t"] for r in ts]

    final = ts[-1]
    regions = region_fractions(diag["X_snap"][-1], ev)
    fr_fracs = [r["fr_event_fraction"] for r in ts if r["fr_applied"]]
    integrated_physical = integrated_l2_over_time(physical_series, times)
    out = dict(
        final_l2_F=final["l2_F"],
        final_l2_Fprime=final["l2_Fprime"],
        integrated_l2_F=integrated_l2_over_time(l2_F_series, times),
        integrated_l2_Fprime=integrated_l2_over_time(l2_Fp_series, times),
        final_marginal_l2_uniform=final["marginal_l2_uniform"],
        final_marginal_l2_target=final["marginal_l2_target"],
        final_marginal_l2_physical_ref=final["marginal_l2_physical_ref"],
        integrated_marginal_l2_physical_ref=integrated_physical,
        final_deltaF_error=final["deltaF_error"],
        integrated_deltaF_error=integrated_l2_over_time(
            deltaF_series, times),
        final_barrier_height_error=final["barrier_height_error"],
        integrated_barrier_height_error=integrated_l2_over_time(
            barrier_series, times),
        deltaF_ref=final["deltaF_ref"],
        barrier_crossings=int(final["barrier_crossings"]),
        n_unique_ancestors=int(final["n_unique_ancestors"]),
        mean_fr_event_fraction=float(np.mean(fr_fracs)) if fr_fracs else 0.0,
        max_fr_event_fraction=float(np.max(
            [r["fr_event_fraction_max"] for r in ts])) if ts else 0.0,
        any_nan=bool(
            not np.isfinite(final["l2_F"])
            or not np.isfinite(final["l2_Fprime"])
            or (p_ref is not None and (
                not np.isfinite(final["marginal_l2_physical_ref"])
                or not np.isfinite(integrated_physical)))),
    )
    for source, dest, cast in [
        ("ancestor_ess", "final_ancestor_ess", float),
        ("max_clone_multiplicity", "final_max_clone_multiplicity", int),
        ("max_clone_weight", "final_max_clone_weight", float),
        ("cumulative_fr_events", "cumulative_fr_events", int),
        ("cumulative_replacements", "cumulative_replacements", int),
    ]:
        if source in final:
            out[dest] = cast(final[source])
    out.update(regions)
    return out
