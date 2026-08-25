"""Within-fibre conditional-law diagnostics for v4-A.  Diagnostic only.

Frozen protocol: ``docs/V4A_PREREGISTRATION.md``, Phase 0 instrumentation.

Keeping Fisher--Rao masses out of the ABF estimator removes the direct
estimator bias.  It does **not** remove a second route: when the representation
module resamples according to those masses, a path-dependent mass distribution
is converted back into an actual physical population.  Replicas sharing a
reaction-coordinate value but differing in the fibre coordinate can be selected
against one another, so the realized conditional law ``nu(dy | x)`` can move even
though no weight ever touched a force accumulator.

On this toy ``pi(dy | x) propto exp(-beta V(x, y))`` is known exactly, so the
damage and its recovery over the hold-out window can be measured rather than
assumed.  Nothing here defines a gate.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import torch

from . import potentials


def conditional_cdf(x: float, y_grid: np.ndarray, beta: float) -> np.ndarray:
    """Exact ``pi(y | x)`` CDF on ``y_grid``.

    The bias depends on ``xi = x`` alone, so it is constant on the fibre and
    cancels: the biased and physical conditional laws coincide.
    """
    xs = torch.full((len(y_grid),), float(x), dtype=torch.float64)
    ys = torch.as_tensor(y_grid, dtype=torch.float64)
    logw = -float(beta) * potentials.potential_xy_torch(xs, ys)
    w = torch.exp(logw - logw.max()).numpy()
    cdf = np.cumsum(w)
    return cdf / cdf[-1]


def w1_to_conditional(y_samples: np.ndarray, x_centre: float,
                      y_grid: np.ndarray, beta: float) -> float:
    """1-Wasserstein between the empirical fibre sample and ``pi(y | x)``.

    Bandwidth-free on purpose: bins hold only tens of replicas, where a KDE-based
    divergence would mostly report the bandwidth.
    """
    if len(y_samples) < 2:
        return float("nan")
    cdf_ref = conditional_cdf(x_centre, y_grid, beta)
    cdf_emp = np.searchsorted(np.sort(y_samples), y_grid, side="right") / len(y_samples)
    return float(np.trapezoid(np.abs(cdf_emp - cdf_ref), y_grid))


def fibre_report(x: np.ndarray, y: np.ndarray, beta: float,
                 bin_centres: Sequence[float] = (-1.05, 0.0, 1.0),
                 half_width: float = 0.15,
                 y_range: tuple = (-2.5, 3.5), ny: int = 401) -> Dict[str, float]:
    """``D_fibre`` and occupancy for each registered fibre probe.

    Called immediately before a resampling, immediately after, and again after
    the hold-out window, so that damage and recovery are separable.
    """
    y_grid = np.linspace(y_range[0], y_range[1], ny)
    out: Dict[str, float] = {}
    for c in bin_centres:
        m = np.abs(x - c) <= half_width
        tag = f"{c:+.2f}".replace("+", "p").replace("-", "m").replace(".", "")
        out[f"n_{tag}"] = int(m.sum())
        out[f"w1_{tag}"] = w1_to_conditional(y[m], c, y_grid, beta)
    return out
