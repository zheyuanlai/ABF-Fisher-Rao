"""Batched smooth PME reciprocal space (Essmann et al., J. Chem. Phys. 103:8577, 1995).

Matches OpenMM's ``NonbondedForce`` PME with the parameters pinned in ``methane.system``:
B-spline order 5, the frozen ``alpha`` and grid, cubic cell.

Batching is the whole point
---------------------------
``torch-pme`` was evaluated and rejected because it takes one system at a time (Amendment 12.4);
everything here carries a leading walker axis ``B``.  The two steps that do not batch for free --
spreading charges onto the mesh and gathering back from it -- are done with a single flat
``index_add_`` over a ``(B*K^3,)`` buffer using per-walker offsets, so the whole population is one
kernel launch rather than ``B``.  The FFT batches natively.

Forces come from autograd
-------------------------
``E_rec`` is a differentiable function of the positions (the spline weights depend smoothly on the
fractional remainder, and ``floor`` correctly contributes zero gradient), so the reciprocal force
is ``-dE/dx`` by ``torch.autograd.grad`` rather than a hand-written gather.  That removes the
most error-prone half of a PME implementation at a cost of roughly 2x the forward pass -- a trade
worth making for a term whose errors are smooth, plausible and silent.  It is validated against
OpenMM to ``1e-6`` like everything else.
"""
from __future__ import annotations

import numpy as np
import torch

ONE_4PI_EPS0 = 138.93545764438198


def bspline_weights(w, order):
    """``M_n(w + i)`` for ``i = 0 .. order-1``; ``w`` is any shape, result gains a trailing axis.

    Cardinal B-spline recursion ``M_k(u) = (u M_{k-1}(u) + (k-u) M_{k-1}(u-1)) / (k-1)``, started
    from ``M_2(w) = w``, ``M_2(w+1) = 1-w``.  Evaluated on the integer-shifted stencil, so
    ``M_{k-1}(w+i-1)`` is just the neighbouring entry.
    """
    m = [w, 1.0 - w]                                        # order 2
    for k in range(3, order + 1):
        prev = m
        new = []
        for i in range(k):
            a = (w + i) * prev[i] if i < len(prev) else torch.zeros_like(w)
            b = (k - w - i) * prev[i - 1] if i > 0 else torch.zeros_like(w)
            new.append((a + b) / (k - 1))
        m = new
    return torch.stack(m, dim=-1)


def _bspline_at_integers(order, dtype=torch.float64):
    """``M_n(1), M_n(2), ..., M_n(n-1)`` -- the coefficients of the ``b`` factor."""
    # bspline_weights(0) = [M_n(0), M_n(1), ..., M_n(n-1)] and M_n(0) = 0, so drop the first.
    w = torch.zeros(1, dtype=dtype)
    return bspline_weights(w, order)[0][1:]


def influence_function(grid, box_nm, alpha, order, device=None, dtype=torch.float64):
    """``kernel(m) = B(m) exp(-pi^2 m^2 / alpha^2) / m^2``, with ``kernel(0) = 0``.

    ``B(m) = |b_1|^2 |b_2|^2 |b_3|^2`` is the B-spline (Euler exponential-spline) correction that
    makes the mesh interpolation unbiased.  Returned on the full ``fftn`` grid.
    """
    K = tuple(int(k) for k in grid)
    L = float(box_nm)

    # M_n at integers 1..n-1, used for every axis (cubic cell, same order)
    Mn = _bspline_at_integers(order, dtype=dtype).to(device)

    b2 = []
    for d, Kd in enumerate(K):
        m_int = torch.arange(Kd, device=device, dtype=dtype)
        k_idx = torch.arange(order - 1, device=device, dtype=dtype)
        # denom(m) = sum_k M_n(k+1) exp(2 pi i m k / K)
        phase = 2.0 * np.pi * m_int[:, None] * k_idx[None, :] / Kd
        denom = (Mn[None, :] * torch.exp(1j * phase)).sum(-1)
        bsp_mod = denom.abs() ** 2

        # Euler exponential-spline failure at odd order.  For odd n the B-spline is symmetric
        # about n/2, so at m = K/2 the alternating sum M(1) - M(2) + M(3) - M(4) ... cancels
        # exactly and |denom|^2 = 0.  OpenMM uses order 5 on an even grid, so this point is
        # always hit.  The standard remedy (AMBER's PME, carried into OpenMM) replaces a
        # vanishing modulus by the mean of its neighbours; without it the influence function is
        # +inf at one frequency and the reciprocal energy is garbage rather than merely wrong.
        bad = bsp_mod < 1.0e-7
        if bool(bad.any()):
            nb = 0.5 * (torch.roll(bsp_mod, 1) + torch.roll(bsp_mod, -1))
            bsp_mod = torch.where(bad, nb, bsp_mod)
        b2.append(1.0 / bsp_mod)                                    # |b(m)|^2
    B = b2[0][:, None, None] * b2[1][None, :, None] * b2[2][None, None, :]

    # physical reciprocal vectors m = m_int / L, folded to [-K/2, K/2)
    def folded(Kd):
        i = torch.arange(Kd, device=device, dtype=dtype)
        return torch.where(i <= Kd // 2, i, i - Kd)
    m1, m2, m3 = (folded(K[0]) / L, folded(K[1]) / L, folded(K[2]) / L)
    m2sq = (m1[:, None, None] ** 2 + m2[None, :, None] ** 2 + m3[None, None, :] ** 2)

    ker = torch.zeros_like(m2sq)
    nz = m2sq > 0
    ker[nz] = torch.exp(-(np.pi ** 2) * m2sq[nz] / (alpha ** 2)) / m2sq[nz]
    return ker * B


class PMEReciprocal:
    """Batched reciprocal-space PME energy and force for a shared cubic cell."""

    def __init__(self, charge, box_nm, grid, alpha_per_nm, order=5,
                 device=None, dtype=torch.float64):
        self.q = torch.as_tensor(charge, device=device, dtype=dtype)
        self.L = float(box_nm)
        self.V = self.L ** 3
        self.K = tuple(int(k) for k in grid)
        self.alpha = float(alpha_per_nm)
        self.order = int(order)
        self.device = device
        self.dtype = dtype
        self.kernel = influence_function(self.K, self.L, self.alpha, self.order,
                                         device=device, dtype=dtype)
        self._nK = self.K[0] * self.K[1] * self.K[2]

    def _spread(self, x):
        """Charge mesh ``Q`` of shape ``(B, K1, K2, K3)`` from positions ``(B, N, 3)``."""
        B, N, _ = x.shape
        Kt = torch.tensor(self.K, device=x.device, dtype=x.dtype)
        u = (x / self.L) * Kt                                  # fractional, in grid units
        u = u - torch.floor(u / Kt) * Kt                       # wrap into [0, K)
        base = torch.floor(u)
        w = u - base                                           # in [0, 1)
        base = base.long()

        wts = bspline_weights(w, self.order)                   # (B, N, 3, order)
        n = self.order
        offs = torch.arange(n, device=x.device)

        # grid index along each axis for each stencil point: (base_d - i) mod K_d
        idx = [((base[..., d, None] - offs[None, None, :]) % self.K[d]) for d in range(3)]

        # outer product over the 3 axes -> (B, N, n, n, n)
        wt = (wts[..., 0, :, None, None] * wts[..., 1, None, :, None]
              * wts[..., 2, None, None, :])
        contrib = self.q[None, :, None, None, None] * wt

        flat = (idx[0][..., :, None, None] * (self.K[1] * self.K[2])
                + idx[1][..., None, :, None] * self.K[2]
                + idx[2][..., None, None, :])                  # (B, N, n, n, n)

        # one flat scatter for the whole population: offset walker b by b * K^3
        boff = (torch.arange(B, device=x.device) * self._nK)[:, None, None, None, None]
        mesh = torch.zeros(B * self._nK, device=x.device, dtype=x.dtype)
        mesh.index_add_(0, (flat + boff).reshape(-1), contrib.reshape(-1))
        return mesh.view(B, *self.K)

    def energy(self, x):
        """Reciprocal energy ``(B,)`` -- differentiable in ``x``."""
        Q = self._spread(x)
        Qk = torch.fft.fftn(Q, dim=(1, 2, 3))
        e = (self.kernel[None] * (Qk.real ** 2 + Qk.imag ** 2)).sum(dim=(1, 2, 3))
        return ONE_4PI_EPS0 / (2.0 * np.pi * self.V) * e

    def energy_forces(self, x):
        """``(E (B,), F (B, N, 3))``; force by autograd, which is exact here."""
        xr = x.detach().requires_grad_(True)
        e = self.energy(xr)
        g, = torch.autograd.grad(e.sum(), xr)
        return e.detach(), -g
