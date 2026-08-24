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

from .grid import (EPS, Grid1D, central_diff, cumtrapz, gaussian_kernel, scatter_counts,
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


def smoothing_floor(grid: Grid1D, F_ref, bw, eval_mask=None, rho=None):
    """The L2 error a PERFECT mean-force estimator still makes at this bandwidth.

    The kernel in `MeanForceAccumulator` is a bias as well as a variance
    reduction: even with infinite samples the Nadaraya-Watson ratio returns a
    smoothed mean force, and integrating that gives a smoothed profile.  This
    computes exactly that, by differentiating `F_ref` and putting it back
    through the estimator's own pipeline.

    The comparison is against the UNSMOOTHED reconstruction rather than against
    `F_ref` itself, so the reference's noise and the differentiate/re-integrate
    mismatch both cancel and what is left is only what the kernel did.

    `rho` is the sampling density the ratio actually divides by -- the
    accumulator's own `S0`, which carries the Fixman weight and the window
    placement.  The O(bw^2) bias is `(bw^2/2)[f'' + 2 f' rho'/rho]`, so leaving
    `rho` at None keeps only the first term and assumes the windows are evenly
    spread.  They never are exactly, and for a transported population they are
    not spread the same way as for a placed one.
    """
    dev, dt = F_ref.device, F_ref.dtype
    m = grid.eval_mask(dev, dt) if eval_mask is None else eval_mask
    F = F_ref if F_ref.dim() == 2 else F_ref.unsqueeze(0)
    fr = central_diff(F, grid.dx, grid.bc)
    K, r = gaussian_kernel(bw, grid.dx, dev, dt)
    w = (torch.ones_like(fr) if rho is None
         else rho / rho.mean(dim=-1, keepdim=True).clamp_min(EPS))
    num = smooth(w * fr, K, r, grid.dx, grid.bc)
    den = smooth(w, K, r, grid.dx, grid.bc)

    def gauge(v):
        return v - v[:, m].mean(dim=1, keepdim=True)

    Fs = gauge(cumtrapz(num / den.clamp_min(1e-30), grid.dx))
    F0 = gauge(cumtrapz(fr, grid.dx))
    return gauge_l2(Fs, F0[0], m)
