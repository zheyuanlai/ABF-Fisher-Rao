"""Batched 1-D Gaussian-mixture density model for the reaction-coordinate marginal.

This is "Version A" of the GMM plan: the mixture is used ONLY as a smooth density /
score estimator in place of the KDE.  Both quantities the RC-WFR step needs are then
analytic, with no grid differentiation:

    p(z)          = sum_k w_k N(z; m_k, s_k^2)
    grad log p(z) = sum_k r_k(z) (m_k - z) / s_k^2 ,     r_k = w_k N_k / sum_j w_j N_j

The components are initialized once on a uniform grid over the CV domain and then
WARM-STARTED from the previous outer iteration, so the parameters evolve continuously
and cannot label-switch or collapse between steps (a fresh EM fit per step would make
the induced Wasserstein velocity discontinuous).

A uniform background of weight `eps_bg` is mixed in.  Without it a finite mixture
assigns near-zero density to the parts of the domain it has not reached, and the score
`grad log p` explodes there.

Periodic domains use the nearest-image displacement, which is the standard wrapped-
Gaussian approximation and is accurate whenever s_k is small against the period.
"""
from __future__ import annotations

import math

import torch

from .grid import EPS, Grid1D

LOG2PI = math.log(2.0 * math.pi)


class GMM1D:
    """(R, K) mixtures over a batch of R runs, fitted by warm-started EM."""

    def __init__(self, rows: int, grid: Grid1D, K: int, device, dtype,
                 s_init: float = None, s_floor: float = None, eps_bg: float = 1e-3):
        self.grid, self.K, self.eps_bg = grid, K, eps_bg
        L = grid.volume
        s0 = (L / K) if s_init is None else s_init
        self.s_floor = (0.25 * L / K) if s_floor is None else s_floor
        m = torch.linspace(grid.xmin + 0.5 * L / K, grid.xmax - 0.5 * L / K, K,
                           device=device, dtype=dtype)
        self.m = m.unsqueeze(0).repeat(rows, 1)
        self.s = torch.full((rows, K), s0, device=device, dtype=dtype)
        self.w = torch.full((rows, K), 1.0 / K, device=device, dtype=dtype)

    def _disp(self, Z):
        """z_i - m_k with the nearest periodic image where the grid is periodic."""
        d = Z.unsqueeze(2) - self.m.unsqueeze(1)          # (R, N, K)
        if self.grid.bc == "periodic":
            L = self.grid.volume
            d = d - L * torch.round(d / L)
        return d

    def _log_comp(self, Z):
        d = self._disp(Z)
        s = self.s.unsqueeze(1)
        return -0.5 * (d / s) ** 2 - torch.log(s) - 0.5 * LOG2PI, d

    def responsibilities(self, Z):
        lc, d = self._log_comp(Z)
        la = lc + torch.log(torch.clamp(self.w, min=EPS)).unsqueeze(1)
        r = torch.softmax(la, dim=2)
        return r, d

    def fit(self, Z, n_em: int = 3):
        """A few warm-started EM sweeps on the current labels.  Z: (R, N)."""
        for _ in range(n_em):
            r, d = self.responsibilities(Z)               # (R, N, K)
            nk = r.sum(dim=1)                             # (R, K)
            nk_c = torch.clamp(nk, min=1e-8)
            self.w = torch.clamp(nk / Z.shape[1], min=1e-12)
            self.w = self.w / self.w.sum(dim=1, keepdim=True)
            dm = (r * d).sum(dim=1) / nk_c                # mean displacement
            self.m = self.m + dm
            if self.grid.bc == "periodic":
                from .grid import wrap_into
                self.m = wrap_into(self.m, self.grid.xmin, self.grid.xmax)
            else:
                self.m = torch.clamp(self.m, self.grid.xmin, self.grid.xmax)
            d2 = (r * (self._disp(Z) ** 2)).sum(dim=1) / nk_c
            self.s = torch.clamp(torch.sqrt(torch.clamp(d2, min=0.0)), min=self.s_floor)
        return self

    def log_prob(self, Z):
        lc, _ = self._log_comp(Z)
        la = lc + torch.log(torch.clamp(self.w, min=EPS)).unsqueeze(1)
        lp_mix = torch.logsumexp(la, dim=2)
        lp_bg = math.log(1.0 / self.grid.volume)
        e = self.eps_bg
        return torch.logaddexp(lp_mix + math.log(1.0 - e),
                               torch.full_like(lp_mix, lp_bg + math.log(e)))

    def prob(self, Z):
        return torch.exp(self.log_prob(Z))

    def score(self, Z):
        """grad_z log p(z) at the particles; the uniform background is score-free."""
        r, d = self.responsibilities(Z)
        s2 = (self.s ** 2).unsqueeze(1)
        g_mix = (r * (-d / s2)).sum(dim=2)
        # weight of the mixture part in the background-mixed density
        lp_mix, lp_bg = self._parts(Z)
        wmix = torch.sigmoid(lp_mix - lp_bg)
        return wmix * g_mix

    def _parts(self, Z):
        lc, _ = self._log_comp(Z)
        la = lc + torch.log(torch.clamp(self.w, min=EPS)).unsqueeze(1)
        lp_mix = torch.logsumexp(la, dim=2) + math.log(1.0 - self.eps_bg)
        lp_bg = torch.full_like(lp_mix,
                                math.log(1.0 / self.grid.volume) + math.log(self.eps_bg))
        return lp_mix, lp_bg

    def on_grid(self, xg):
        """Density on the CV grid, for diagnostics.  xg: (G,) -> (R, G)."""
        Z = xg.unsqueeze(0).expand(self.m.shape[0], -1)
        return self.prob(Z)
