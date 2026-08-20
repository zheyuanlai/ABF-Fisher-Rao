"""Periodic 1D reaction-coordinate grid layer + SHUS accumulator (for xi = phi).

The 1D counterpart of grid2d.py for a single angle on the circle: same kernel and
normalization conventions as the validated layers, with circular topology (no
reflecting walls -- -pi and pi are the same point). Used by the alanine xi = phi
study; the non-periodic Grid1D layer of the closed campaigns is untouched.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as tF

from .grid import DEVICE, DTYPE, EPS
from .grid2d import periodic_gaussian_kernel, wrap_periodic  # noqa: F401 (re-export)


@dataclass(frozen=True)
class Grid1P:
    """Periodic axis [xmin, xmin + L), n nodes at xmin + i*dx, dx = L/n."""
    xmin: float
    L: float
    n: int

    @property
    def dx(self) -> float:
        return self.L / self.n

    @property
    def volume(self) -> float:
        return self.L

    def x(self, device=DEVICE, dtype=DTYPE) -> torch.Tensor:
        return self.xmin + self.dx * torch.arange(self.n, device=device, dtype=dtype)


def smooth1p(v, kernel, r):
    """Circular convolution along the last axis.  v: (R, n) -> (R, n)."""
    R, n = v.shape
    x = tF.pad(v.unsqueeze(1), (r, r), mode="circular")
    return tF.conv1d(x, kernel.flip(0).view(1, 1, -1)).squeeze(1)


def integral1p(y, grid: Grid1P):
    return y.sum(dim=-1) * grid.dx


def nearest_bin1p(X, grid: Grid1P):
    return torch.remainder(torch.round((X - grid.xmin) / grid.dx).long(), grid.n)


def binned_density1p(X, kernel, r, grid: Grid1P, weights=None):
    """Mollified empirical density of the walkers.  X: (R, N) -> (R, n).

    weights: None for the equal-weight ensemble, or (R, N) statistical weights (the
    convention of the weighted conditional selection of fisher_rao_cond.py: weights
    are positive and sum to N per row, so weights = 1 reproduces the unweighted call
    BITWISE).  Weighted or not, the return is a normalized density -- what changes is
    which empirical measure it estimates: the ensemble REPRESENTS
    sum_k w_k delta_{X_k} / sum_k w_k, not sum_k delta_{X_k} / N.
    """
    R, N = X.shape
    idx = nearest_bin1p(X, grid)
    hist = torch.zeros((R, grid.n), device=X.device, dtype=X.dtype)
    hist.scatter_add_(1, idx, torch.ones_like(X) if weights is None else weights)
    p = smooth1p(hist, kernel, r)
    p = p / (float(N) if weights is None
             else torch.clamp(weights.sum(dim=1, keepdim=True), min=EPS))
    mass = torch.clamp(integral1p(p, grid), min=EPS).unsqueeze(1)
    return torch.clamp(p / mass, min=EPS)


def interp1p(X, grid_vals, grid: Grid1P):
    """Linear periodic interpolation.  grid_vals: (R, n), X: (R, N) -> (R, N)."""
    pos = (X - grid.xmin) / grid.dx
    i0 = torch.floor(pos).long()
    frac = pos - i0.to(X.dtype)
    i0 = torch.remainder(i0, grid.n)
    i1 = torch.remainder(i0 + 1, grid.n)
    v0 = torch.gather(grid_vals, 1, i0)
    v1 = torch.gather(grid_vals, 1, i1)
    return v0 + frac * (v1 - v0)


def central_diff1p(F, grid: Grid1P):
    return (torch.roll(F, -1, dims=1) - torch.roll(F, 1, dims=1)) / (2.0 * grid.dx)


def kl_to_uniform1p(p, grid: Grid1P):
    u_log = math.log(1.0 / grid.volume)
    return integral1p(p * (torch.log(torch.clamp(p, min=EPS)) - u_log), grid)


def tv_to_uniform1p(p, grid: Grid1P):
    return 0.5 * integral1p((p - 1.0 / grid.volume).abs(), grid)


def uniform_log_ratio1p(X, p_grid, grid: Grid1P):
    u_log = math.log(1.0 / grid.volume)
    return u_log - torch.log(torch.clamp(interp1p(X, p_grid, grid), min=EPS))


class ShusAccumulator1P:
    """Periodic-1D mollified SHUS accumulator; conventions identical to the
    validated 1D/2D cores (block-frozen deposit weight R_n, gain-scaled
    increment, max-gauge renormalization, estimator protection by structure)."""

    def __init__(self, rows: int, grid: Grid1P, beta: torch.Tensor, eps_bw: float,
                 device, dtype, gain=None, R_init=None):
        """R_init: (rows, n) accumulator warm start, gauge-normalized on entry.

        The frozen default R = 1 is a run that learns its bias from nothing, which
        carries an establishment transient into every measurement.  A warm start at
        the analytic fixed point removes that transient by construction, so a run can
        ask about the estimator's VARIANCE around its fixed point rather than about
        the approach to it.  It consults the reference and is therefore an
        experimental CONDITION, applied identically to every arm, never an arm's
        private information."""
        self.grid = grid
        self.beta = beta.reshape(rows, 1).to(device=device, dtype=dtype)
        self.kernel, self.krad = periodic_gaussian_kernel(eps_bw, grid.dx, grid.n,
                                                          device, dtype)
        if R_init is None:
            self.R = torch.ones((rows, grid.n), device=device, dtype=dtype)
        else:
            R0 = R_init.to(device=device, dtype=dtype).reshape(rows, grid.n)
            assert bool((R0 > 0).all()), "accumulator warm start must be positive"
            self.R = R0 / R0.max(dim=1, keepdim=True).values
        self.buf = torch.zeros((rows, grid.n), device=device, dtype=dtype)
        if gain is None:
            self.gain = torch.ones((rows, 1), device=device, dtype=dtype)
        else:
            self.gain = gain.reshape(rows, 1).to(device=device, dtype=dtype)
            assert bool((self.gain > 0).all()), "g_shus must be positive"
        self._refresh_bias()

    def _refresh_bias(self):
        self.F = -torch.log(torch.clamp(self.R, min=EPS)) / self.beta
        self.Fp = central_diff1p(self.F, self.grid)

    def bias_force_at(self, X):
        return interp1p(X, self.Fp, self.grid)

    def deposit(self, X, weights=None):
        """Accumulate this step's deposit.  weights: (R, K) statistical weights,
        mean 1 (None = the equal-weight ensemble, and weights = 1 is bitwise the
        same call).  The SHUS increment estimates R(x) rho_t(x) with rho_t the law
        the ensemble REPRESENTS, so a weighted ensemble must deposit its weights --
        otherwise the accumulator would learn the allocation instead of the
        physics, which is exactly what weighted selection exists to avoid."""
        w = interp1p(X, self.R, self.grid)
        if weights is not None:
            w = w * weights
        idx = nearest_bin1p(X, self.grid)
        self.buf.scatter_add_(1, idx, w)

    def update(self, dt: float, K: int):
        inc = smooth1p(self.buf, self.kernel, self.krad) * (dt / K) * self.gain
        self.R = self.R + inc
        self.R = self.R / self.R.max(dim=1, keepdim=True).values
        self.buf.zero_()
        self._refresh_bias()
        return inc

    def f_estimate(self):
        return self.F - self.F.mean(dim=1, keepdim=True)
