"""Mollified SHUS: the adaptive-biasing-POTENTIAL baseline (ABP / OPES family).

Ported from ABP-Fisher-Rao src/abpfr/shus.py (docs/PROVENANCE.md).  Conventions:
bias V_t = V - F_t(xi); estimator F_t = -beta^-1 log R_t; deposit weight R_n(xi)
frozen over an adaptation block; R renormalized by its max each block (pure gauge).
"""
from __future__ import annotations

import torch

from .grid import (EPS, Grid1D, central_diff, gaussian_kernel, interp1d,
                   scatter_counts, smooth)


class ShusAccumulator:
    def __init__(self, rows: int, grid: Grid1D, beta: float, eps_bw: float,
                 device, dtype, gain: float = 1.0):
        self.grid = grid
        self.beta = beta
        self.kernel, self.krad = gaussian_kernel(eps_bw, grid.dx, device, dtype)
        self.R = torch.ones((rows, grid.n), device=device, dtype=dtype)
        self.buf = torch.zeros((rows, grid.n), device=device, dtype=dtype)
        self.gain = gain if torch.is_tensor(gain) else float(gain)
        if torch.is_tensor(self.gain):
            self.gain = self.gain.reshape(-1, 1).to(device=device, dtype=dtype)
        self._refresh()

    def _refresh(self):
        self.F = -torch.log(torch.clamp(self.R, min=EPS)) / self.beta
        self.Fp = central_diff(self.F, self.grid.dx, self.grid.bc)

    def bias_force_at(self, X):
        return interp1d(X, self.Fp, self.grid)

    def deposit(self, X):
        w = interp1d(X, self.R, self.grid)
        self.buf += scatter_counts(X, self.grid, w)

    def update(self, dt: float, K: int):
        inc = smooth(self.buf, self.kernel, self.krad, self.grid.dx, self.grid.bc) \
            * (dt / K) * self.gain
        self.R = self.R + inc
        self.R = self.R / self.R.max(dim=1, keepdim=True).values
        self.buf.zero_()
        self._refresh()

    def f_estimate(self, eval_mask):
        return self.F - self.F[:, eval_mask].mean(dim=1, keepdim=True)
