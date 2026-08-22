"""Shared free-energy estimators.

Every TI-type arm (RC-WFR, W-only, FR-only, fixed-window TI, REUS) and ABF use
the SAME binned mean-force accumulator, so no comparison is contaminated by an
estimator asymmetry:

    S0(z_g) = sum_s K_h(z_g - Z_s)             (smoothed sample count)
    S1(z_g) = sum_s K_h(z_g - Z_s) f_s         (smoothed mean-force deposit)
    F'_hat  = S1 / (S0 + n_min)                (low-count ramp; == ABF's ramp)
    F_hat   = cumtrapz(F'_hat)                 (thermodynamic integration)

`n_min` regularizes empty bins and reproduces the standard ABF ramp function.
The estimator FLOOR (what an infinite-sample oracle would report through this
same pipeline) is measured separately and reported with every campaign.
"""
from __future__ import annotations

import torch

from .grid import (EPS, Grid1D, cumtrapz, gaussian_kernel, scatter_counts,
                   smooth, trapz)


class MeanForceAccumulator:
    """Batched (R, G) accumulator of smoothed mean-force statistics."""

    def __init__(self, rows: int, grid: Grid1D, bw: float, n_min: float,
                 device, dtype, decay: float = 1.0):
        self.grid = grid
        self.bw = bw
        self.n_min = n_min
        self.decay = decay            # 1.0 = plain running average
        self.kernel, self.krad = gaussian_kernel(bw, grid.dx, device, dtype)
        self.S0 = torch.zeros((rows, grid.n), device=device, dtype=dtype)
        self.S1 = torch.zeros((rows, grid.n), device=device, dtype=dtype)

    def deposit(self, X, f, weights=None):
        """Add one batch of (position, mean-force) samples.  X, f: (R, N)."""
        w = torch.ones_like(X) if weights is None else weights
        h0 = scatter_counts(X, self.grid, w)
        h1 = scatter_counts(X, self.grid, w * f)
        if self.decay != 1.0:
            self.S0.mul_(self.decay)
            self.S1.mul_(self.decay)
        self.S0 += smooth(h0, self.kernel, self.krad, self.grid.dx, self.grid.bc)
        self.S1 += smooth(h1, self.kernel, self.krad, self.grid.dx, self.grid.bc)

    def mean_force(self, n_min=None):
        """n_min may be a scalar or an (R,1) column (per-row ABF ramp sweeps)."""
        m = self.n_min if n_min is None else n_min
        return self.S1 / (self.S0 + m)

    def free_energy(self, eval_mask):
        Fp = self.mean_force()
        F = cumtrapz(Fp, self.grid.dx)
        return F - F[:, eval_mask].mean(dim=1, keepdim=True)

    def counts(self):
        return self.S0


def gauge_l2(F_hat, F_ref, eval_mask):
    """RMS error over the eval window after removing the optimal additive constant."""
    d = (F_hat - F_ref)[:, eval_mask]
    d = d - d.mean(dim=1, keepdim=True)
    return torch.sqrt((d * d).mean(dim=1))


def gauge_l2_profile(F_hat, F_ref, eval_mask):
    d = F_hat - F_ref
    return d - d[:, eval_mask].mean(dim=1, keepdim=True)
