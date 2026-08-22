"""Wasserstein transport of the reaction-coordinate labels toward uniform.

For a uniform target the W-gradient flow of KL(p||u) is heat flow,

    d_t p = kappa * Laplacian(p),

with two exchangeable particle realizations:

  'sde'   Z <- Z + sqrt(2 kappa dtau) eta          (stochastic; unbiased, noisy)
  'flow'  Z <- Z - kappa dtau grad log p_hat(Z)    (deterministic probability flow;
                                                    needs a score estimate)

Boundaries follow the grid: reflecting (a bounded CV) or periodic (a torsion).
The 'flow' variant is the one that connects to the Gaussian-mixture picture:
grad log p is analytic for a GMM, so no KDE differentiation is needed.
"""
from __future__ import annotations

import math

import torch

from .grid import (EPS, Grid1D, binned_density, central_diff, gaussian_kernel,
                   interp1d)


def w_step_sde(X, kappa, dtau: float, grid: Grid1D, gen):
    """kappa may be a scalar or an (R,1) column (per-row parameter sweeps)."""
    noise = torch.randn(X.shape, device=X.device, dtype=X.dtype, generator=gen)
    if torch.is_tensor(kappa):
        amp = torch.sqrt(2.0 * kappa * dtau)
    else:
        amp = math.sqrt(2.0 * kappa * dtau)
    return grid.enforce(X + amp * noise)


def score_kde(X, grid: Grid1D, bw: float):
    """grad log p_hat on the grid from a KDE.  (R,N) -> (R,G)."""
    k, r = gaussian_kernel(bw, grid.dx, X.device, X.dtype)
    p = binned_density(X, k, r, grid)
    logp = torch.log(torch.clamp(p, min=EPS))
    return central_diff(logp, grid.dx, grid.bc)


def w_step_flow(X, kappa: float, dtau: float, grid: Grid1D, bw: float,
                clip: float = None):
    s_grid = score_kde(X, grid, bw)
    s = interp1d(X, s_grid, grid)
    if clip is not None:
        s = torch.clamp(s, -clip, clip)
    return grid.enforce(X - kappa * dtau * s)
