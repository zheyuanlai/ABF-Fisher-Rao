"""Mollified SHUS accumulator on the 2D torus (batched over rows).

Same frozen conventions as the 1D core (shus.py / docs/SPEC_SHUS_FR.md), with the
grid layer swapped for the periodic 2D one:

* F_t = -beta^{-1} log R_t on the (n1, n2) grid; bias V - F_t(xi);
* deposit weight = block-frozen R_n(xi) (bilinear periodic interpolation);
* R_{n+1} = R_n + g_shus * (dt / K) * mollify2(block deposits);  R <- R / max(R);
* the mollifier is the separable periodic Gaussian kernel;
* nothing here is callable at an FR event (estimator protection).
"""
from __future__ import annotations

import torch

from .grid import EPS
from .grid2d import (GridT2, central_diff2, integral2, interp2, nearest_bin2,
                     periodic_gaussian_kernel, smooth2)


class ShusAccumulator2:
    """State: R (rows, n1, n2) accumulator and the current block's deposit buffer."""

    def __init__(self, rows: int, grid: GridT2, beta: torch.Tensor, eps_bw: float,
                 device, dtype, gain=None):
        self.grid = grid
        self.beta = beta.reshape(rows, 1, 1).to(device=device, dtype=dtype)
        self.k1, self.r1 = periodic_gaussian_kernel(eps_bw, grid.dx1, grid.n1,
                                                    device, dtype)
        self.k2, self.r2 = periodic_gaussian_kernel(eps_bw, grid.dx2, grid.n2,
                                                    device, dtype)
        self.R = torch.ones((rows, grid.n1, grid.n2), device=device, dtype=dtype)
        self.buf = torch.zeros((rows, grid.n1 * grid.n2), device=device, dtype=dtype)
        if gain is None:
            self.gain = torch.ones((rows, 1, 1), device=device, dtype=dtype)
        else:
            self.gain = gain.reshape(rows, 1, 1).to(device=device, dtype=dtype)
            assert bool((self.gain > 0).all()), "g_shus must be positive"
        self._refresh_bias()

    # -- bias seen by the dynamics -------------------------------------------------
    def _refresh_bias(self):
        self.F = -torch.log(torch.clamp(self.R, min=EPS)) / self.beta
        self.Fp1, self.Fp2 = central_diff2(self.F, self.grid)

    def bias_force_at(self, X1, X2):
        """(+dF/dx1, +dF/dx2) at walker positions."""
        return (interp2(X1, X2, self.Fp1, self.grid),
                interp2(X1, X2, self.Fp2, self.grid))

    # -- estimator ------------------------------------------------------------------
    def deposit(self, X1, X2):
        """Deposit post-step positions with block-frozen weight R_n(xi)."""
        w = interp2(X1, X2, self.R, self.grid)
        idx = nearest_bin2(X1, X2, self.grid)
        self.buf.scatter_add_(1, idx, w)

    def update(self, dt: float, K: int):
        """Close the block: mollify, gain-scale, accumulate, renormalize."""
        g = self.grid
        inc = smooth2(self.buf.reshape(-1, g.n1, g.n2), self.k1, self.r1,
                      self.k2, self.r2) * (dt / K) * self.gain
        self.R = self.R + inc
        self.R = self.R / self.R.amax(dim=(1, 2), keepdim=True)
        self.buf.zero_()
        self._refresh_bias()
        return inc

    # -- reporting / persistence -----------------------------------------------------
    def f_estimate(self):
        """F_hat centered to zero mean over the (full-torus) eval window."""
        return self.F - self.F.mean(dim=(1, 2), keepdim=True)

    def state_dict(self):
        return {"R": self.R.clone(), "buf": self.buf.clone()}

    def load_state_dict(self, sd):
        self.R = sd["R"].clone().to(self.R)
        self.buf = sd["buf"].clone().to(self.buf)
        self._refresh_bias()


def mollified_fixed_point2(F_ref, beta, eps_bw, grid: GridT2, device="cpu",
                           dtype=torch.float64):
    """Analytic fixed point of mollified SHUS on the torus (2D analog of
    gateway.mollified_fixed_point): R* = K_eps * e^{-beta F_ref}, hence the
    estimator floor e* and the marginal-KL floor KL*.  F_ref: (n1, n2)."""
    import math
    F = F_ref.reshape(1, grid.n1, grid.n2).to(device=device, dtype=dtype)
    F = F - F.mean(dim=(1, 2), keepdim=True)
    rho = torch.exp(-beta * F)
    k1, r1 = periodic_gaussian_kernel(eps_bw, grid.dx1, grid.n1, device, dtype)
    k2, r2 = periodic_gaussian_kernel(eps_bw, grid.dx2, grid.n2, device, dtype)
    rho_m = smooth2(rho, k1, r1, k2, r2)
    F_star = -torch.log(torch.clamp(rho_m, min=EPS)) / beta
    F_star = F_star - F_star.mean(dim=(1, 2), keepdim=True)
    d = F_star - F
    e_star = float(torch.sqrt((d * d).mean()))
    p_star = rho / torch.clamp(rho_m, min=EPS)
    p_star = p_star / integral2(p_star, grid).reshape(1, 1, 1)
    kl_star = float(integral2(
        p_star * (torch.log(torch.clamp(p_star, min=EPS))
                  - math.log(1.0 / grid.volume)), grid))
    return {"F_star": F_star[0], "e_star": e_star, "kl_star": kl_star}
