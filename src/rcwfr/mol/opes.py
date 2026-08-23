"""OPES (On-the-fly Probability Enhanced Sampling), grid form.

The ABP/SHUS baseline already here builds its bias from RAW visit counts.  OPES
(Invernizzi & Parrinello, JPCL 2020) differs in three ways that matter, and all
three are implemented:

1. the density estimate is REWEIGHTED, `w_k = exp(beta V_{n-1}(s_k))`, so it
   estimates the unbiased `P(s)` rather than the biased visit histogram;
2. the bias carries an explicit floor,
   `epsilon = exp(-beta DeltaE / (1 - 1/gamma))`, which caps how deep a barrier
   the method will fill and is what keeps it stable early;
3. the estimate is normalised by `Z_n`, the average of `P_hat` over the region
   explored so far, not over the whole domain.

    V_n(s) = (1 - 1/gamma) beta^-1 log( P_hat_n(s) / Z_n + epsilon )

`gamma -> infinity` targets a uniform marginal, which is what every other arm in
this campaign targets, so that is the default.

Kernel compression is replaced by a fixed grid with the same Gaussian kernel the
rest of the package uses.  That is a faithful approximation for a 1-D CV at this
resolution -- it changes the representation of `P_hat`, not the algorithm -- and
it is stated rather than hidden.
"""
from __future__ import annotations

import math

import torch

from ..grid import (EPS, Grid1D, central_diff, gaussian_kernel, interp1d,
                    scatter_counts, smooth, trapz)


class OpesAccumulator:
    def __init__(self, rows: int, grid: Grid1D, beta: float, sigma: float,
                 device, dtype, gamma: float = float("inf"),
                 barrier: float = 20.0):
        self.grid, self.beta = grid, beta
        self.kernel, self.krad = gaussian_kernel(sigma, grid.dx, device, dtype)
        self.gamma = gamma
        self.pref = 1.0 if not math.isfinite(gamma) else (1.0 - 1.0 / gamma)
        # barrier is quoted in units of 1/beta (k_B T), as OPES does
        self.eps = math.exp(-barrier / self.pref)
        self.S = torch.zeros((rows, grid.n), device=device, dtype=dtype)
        self.sumw = torch.zeros((rows, 1), device=device, dtype=dtype)
        self.V = torch.zeros((rows, grid.n), device=device, dtype=dtype)
        self.Vp = torch.zeros((rows, grid.n), device=device, dtype=dtype)
        self.seen = torch.zeros((rows, grid.n), device=device, dtype=dtype)

    def deposit(self, X):
        """Add one batch of samples, reweighted by the bias they felt."""
        w = torch.exp(self.beta * interp1d(X, self.V, self.grid))
        self.S += smooth(scatter_counts(X, self.grid, w), self.kernel, self.krad,
                         self.grid.dx, self.grid.bc)
        self.sumw += w.sum(1, keepdim=True)
        self.seen = torch.maximum(
            self.seen, (scatter_counts(X, self.grid) > 0).to(self.S.dtype))

    def update(self):
        P = self.S / torch.clamp(self.sumw, min=EPS)
        expl = torch.clamp(self.seen.sum(1, keepdim=True), min=1.0)
        Z = (P * self.seen).sum(1, keepdim=True) / expl
        self.V = self.pref / self.beta * torch.log(
            torch.clamp(P / torch.clamp(Z, min=EPS) + self.eps, min=EPS))
        self.Vp = central_diff(self.V, self.grid.dx, self.grid.bc)

    def bias_force_at(self, X):
        """dV/ds at the walkers; the force on q is -(dV/ds) grad xi."""
        return interp1d(X, self.Vp, self.grid)
