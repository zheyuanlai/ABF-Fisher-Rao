"""Batched smooth PME reciprocal space for a rectangular cell -- the methane PME generalised.

Differences from ``methane.pme`` (which this module supersedes for c60, per Amendment 16.5):

* **rectangular orthorhombic cell** ``(Lx, Ly, Lz)`` instead of a cube -- per-axis reciprocal
  vectors and per-axis B-spline moduli;
* **charged-subset spreading**: only TIP4P-Ew's H1/H2/M sites carry charge (3846 of 5248), so
  the mesh spread/gather runs over the charged subset; neutral sites contribute exactly zero
  and are skipped rather than multiplied by it.

The B-spline machinery is imported from ``methane.pme`` unchanged (stateless functions).
Forces come from autograd, as there: the reciprocal energy is smooth in the positions, and the
gradient of the spread is exact.  Validated against OpenMM to 1e-6 like everything else.
"""
from __future__ import annotations

import numpy as np
import torch

from methane.pme import ONE_4PI_EPS0, _bspline_at_integers, bspline_weights


def influence_function_rect(grid, box_nm, alpha, order, device=None, dtype=torch.float64):
    """``kernel(m) = B(m) exp(-pi^2 m^2/alpha^2)/m^2`` on the full fftn grid, rectangular cell."""
    K = tuple(int(k) for k in grid)
    L = tuple(float(x) for x in box_nm)

    Mn = _bspline_at_integers(order, dtype=dtype).to(device)
    b2 = []
    for d, Kd in enumerate(K):
        m_int = torch.arange(Kd, device=device, dtype=dtype)
        k_idx = torch.arange(order - 1, device=device, dtype=dtype)
        phase = 2.0 * np.pi * m_int[:, None] * k_idx[None, :] / Kd
        denom = (Mn[None, :] * torch.exp(1j * phase)).sum(-1)
        bsp_mod = denom.abs() ** 2
        bad = bsp_mod < 1.0e-7
        if bool(bad.any()):
            nb = 0.5 * (torch.roll(bsp_mod, 1) + torch.roll(bsp_mod, -1))
            bsp_mod = torch.where(bad, nb, bsp_mod)
        b2.append(1.0 / bsp_mod)
    B = b2[0][:, None, None] * b2[1][None, :, None] * b2[2][None, None, :]

    def folded(Kd):
        i = torch.arange(Kd, device=device, dtype=dtype)
        return torch.where(i <= Kd // 2, i, i - Kd)

    m1 = folded(K[0]) / L[0]
    m2 = folded(K[1]) / L[1]
    m3 = folded(K[2]) / L[2]
    m2sq = m1[:, None, None] ** 2 + m2[None, :, None] ** 2 + m3[None, None, :] ** 2

    ker = torch.zeros_like(m2sq)
    nz = m2sq > 0
    ker[nz] = torch.exp(-(np.pi ** 2) * m2sq[nz] / (alpha ** 2)) / m2sq[nz]
    return ker * B


class PMEReciprocalRect:
    """Batched reciprocal-space PME energy/force, rectangular cell, charged-subset spreading."""

    def __init__(self, charge, box_nm, grid, alpha_per_nm, order=5,
                 device=None, dtype=torch.float64):
        q = torch.as_tensor(np.asarray(charge), device=device, dtype=dtype)
        self.n_sites = int(q.numel())
        self.q_index = torch.nonzero(q != 0, as_tuple=True)[0]
        self.q = q[self.q_index]
        self.L = torch.as_tensor([float(x) for x in box_nm], device=device, dtype=dtype)
        self.V = float(self.L.prod())
        self.K = tuple(int(k) for k in grid)
        self.Kt = torch.as_tensor(self.K, device=device, dtype=dtype)
        self.alpha = float(alpha_per_nm)
        self.order = int(order)
        self.kernel = influence_function_rect(self.K, [float(x) for x in box_nm],
                                              self.alpha, self.order,
                                              device=device, dtype=dtype)
        self._nK = self.K[0] * self.K[1] * self.K[2]

    def _spread(self, xq):
        """Charge mesh ``(B, K1, K2, K3)`` from charged-site positions ``(B, Nq, 3)``."""
        B, N, _ = xq.shape
        u = (xq / self.L) * self.Kt
        u = u - torch.floor(u / self.Kt) * self.Kt
        base = torch.floor(u)
        w = u - base
        base = base.long()

        wts = bspline_weights(w, self.order)                   # (B, Nq, 3, order)
        n = self.order
        offs = torch.arange(n, device=xq.device)
        idx = [((base[..., d, None] - offs[None, None, :]) % self.K[d]) for d in range(3)]
        wt = (wts[..., 0, :, None, None] * wts[..., 1, None, :, None]
              * wts[..., 2, None, None, :])
        contrib = self.q[None, :, None, None, None] * wt
        flat = (idx[0][..., :, None, None] * (self.K[1] * self.K[2])
                + idx[1][..., None, :, None] * self.K[2]
                + idx[2][..., None, None, :])
        boff = (torch.arange(B, device=xq.device) * self._nK)[:, None, None, None, None]
        mesh = torch.zeros(B * self._nK, device=xq.device, dtype=xq.dtype)
        mesh.index_add_(0, (flat + boff).reshape(-1), contrib.reshape(-1))
        return mesh.view(B, *self.K)

    def energy(self, x):
        """Reciprocal energy ``(B,)`` from full positions ``(B, N, 3)``; differentiable."""
        Q = self._spread(x[:, self.q_index, :])
        Qk = torch.fft.fftn(Q, dim=(1, 2, 3))
        e = (self.kernel[None] * (Qk.real ** 2 + Qk.imag ** 2)).sum(dim=(1, 2, 3))
        return ONE_4PI_EPS0 / (2.0 * np.pi * self.V) * e

    def energy_forces(self, x):
        xr = x.detach().requires_grad_(True)
        e = self.energy(xr)
        g, = torch.autograd.grad(e.sum(), xr)
        return e.detach(), -g

    def self_energy(self):
        return -(self.alpha / np.sqrt(np.pi)) * ONE_4PI_EPS0 * float((self.q ** 2).sum())
