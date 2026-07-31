"""Periodic Poisson projection on the torus ``T^2`` by FFT (2-D ABF bias reconstruction).

The 2-D ABF estimator produces a noisy vector field ``g = (g1, g2)`` on the torus grid
that is **not** exactly conservative (curl != 0).  The scalar bias ``B`` is the
Helmholtz/Hodge projection onto gradients,

    B = argmin_{mean B = 0} int_{T^2} | grad B - g |^2 dz,

whose Euler--Lagrange equation is the periodic Poisson problem ``Delta B = div g``.  In
Fourier space, with angular wavenumbers ``k`` (integer modes; period ``2 pi``),

    -|k|^2 B_hat(k) = i k . g_hat(k)   =>   B_hat(k) = - i (k . g_hat(k)) / |k|^2,  k != 0,
    B_hat(0) = 0   (zero mean; also the only mode where g's mean is unrecoverable).

The projected conservative field ``grad B`` is returned spectrally (``i k B_hat``), and
the residual (solenoidal) part ``g - grad B`` is the divergence-free noise the projection
discards.  Validated (tests): exact recovery of a known gradient field, removal of a pure
curl field, exact periodicity/zero-mode, grid convergence, CPU/GPU parity.
"""
from __future__ import annotations

import math

import torch

EPS = 1.0e-12
TWO_PI = 2.0 * math.pi


def wavenumbers(n, dz, device=None, dtype=torch.float64):
    """Angular wavenumbers ``k_m = 2 pi * fftfreq(n, dz)`` for a period-``n*dz`` grid.

    For the torus ``[-pi, pi)`` (period ``2 pi``, ``dz = 2 pi / n``) these are the integer
    modes ``0, 1, ..., n/2-1, -n/2, ..., -1``.
    """
    return TWO_PI * torch.fft.fftfreq(n, d=dz, device=device).to(dtype)


def _k2_grids(n1, n2, dz1, dz2, device, dtype):
    k1 = wavenumbers(n1, dz1, device, dtype)[:, None]     # (n1,1)
    k2 = wavenumbers(n2, dz2, device, dtype)[None, :]     # (1,n2)
    k2mag = k1 * k1 + k2 * k2                              # (n1,n2)
    return k1, k2, k2mag


def poisson_projection(g1, g2, dz1, dz2):
    """Project the field ``g=(g1,g2)`` onto gradients; return ``(B, gB1, gB2)``.

    ``g1, g2`` are ``(..., n1, n2)`` real fields on the torus (leading batch dims allowed).
    ``B`` has zero spatial mean; ``(gB1, gB2) = grad B`` is the conservative (curl-free)
    part of ``g``.
    """
    n1, n2 = g1.shape[-2], g1.shape[-1]
    dev, dt = g1.device, g1.dtype
    k1, k2, k2mag = _k2_grids(n1, n2, dz1, dz2, dev, dt)
    G1 = torch.fft.fft2(g1.to(torch.complex128) if dt == torch.float64 else g1.to(torch.complex64))
    G2 = torch.fft.fft2(g2.to(torch.complex128) if dt == torch.float64 else g2.to(torch.complex64))
    divhat = 1j * (k1 * G1 + k2 * G2)                     # i k . g_hat  (= Delta B hat)
    inv = torch.where(k2mag > EPS, 1.0 / k2mag.clamp_min(EPS), torch.zeros_like(k2mag))
    Bhat = -divhat * inv                                  # B_hat = -(i k.g_hat)/|k|^2
    Bhat[..., 0, 0] = 0.0                                 # enforce zero mean explicitly
    # For even n the k = n/2 (Nyquist) mode is self-conjugate: taking .real of ifft2 below
    # annihilates it in B but NOT in i k B_hat, so gB would carry content B does not, leaving
    # gB != grad B (measured ~12% relative, curl_norm 1.7 at n=48).  Drop it from both.
    if n1 % 2 == 0:
        Bhat[..., n1 // 2, :] = 0.0
    if n2 % 2 == 0:
        Bhat[..., :, n2 // 2] = 0.0
    gB1hat = 1j * k1 * Bhat
    gB2hat = 1j * k2 * Bhat
    B = torch.fft.ifft2(Bhat).real.to(dt)
    gB1 = torch.fft.ifft2(gB1hat).real.to(dt)
    gB2 = torch.fft.ifft2(gB2hat).real.to(dt)
    return B, gB1, gB2


def spectral_gradient(B, dz1, dz2):
    """Spectral gradient ``(dB/dz1, dB/dz2)`` of a periodic field ``B (...,n1,n2)``."""
    n1, n2 = B.shape[-2], B.shape[-1]
    dev, dt = B.device, B.dtype
    k1, k2, _ = _k2_grids(n1, n2, dz1, dz2, dev, dt)
    Bhat = torch.fft.fft2(B.to(torch.complex128) if dt == torch.float64 else B.to(torch.complex64))
    g1 = torch.fft.ifft2(1j * k1 * Bhat).real.to(dt)
    g2 = torch.fft.ifft2(1j * k2 * Bhat).real.to(dt)
    return g1, g2


def curl_norm(g1, g2, dz1, dz2):
    """RMS of the curl ``dg2/dz1 - dg1/dz2`` (spectral) of the field ``g`` on the torus.

    A diagnostic of how far the raw estimator field is from conservative *before* the
    Poisson projection.  Returns one scalar per leading batch element.
    """
    n1, n2 = g1.shape[-2], g1.shape[-1]
    dev, dt = g1.device, g1.dtype
    k1, k2, _ = _k2_grids(n1, n2, dz1, dz2, dev, dt)
    G1 = torch.fft.fft2(g1.to(torch.complex128) if dt == torch.float64 else g1.to(torch.complex64))
    G2 = torch.fft.fft2(g2.to(torch.complex128) if dt == torch.float64 else g2.to(torch.complex64))
    curl = torch.fft.ifft2(1j * k1 * G2 - 1j * k2 * G1).real.to(dt)
    return torch.sqrt((curl ** 2).mean(dim=(-2, -1)))
