"""Periodic 2-D estimators on the torus ``T^2`` for the joint torsion CV ``(phi1, phi2)``.

Cell-centred uniform torus grid ``n1 x n2`` (reusing :func:`alkanes.periodic.periodic_grid`
per axis).  Separable wrapped-Gaussian smoothing is applied as two 1-D kernel matmuls
``K1 @ field @ K2^T`` -- exact periodic (circular) convolution, identical to an FFT product
but simpler at these grid sizes.  Provides the 2-D ABF mean-force fields, the joint marginal
KDE, bilinear-periodic interpolation to particles, and the centred 2-D Fisher--Rao score.

All ops carry a leading batch (run) dimension ``R``.
"""
from __future__ import annotations

import math

import torch

from . import periodic as per

PI = math.pi
TWO_PI = 2.0 * math.pi
EPS = 1.0e-12


def torus_grid(n1, n2, device=None, dtype=torch.float64):
    g1, dz1 = per.periodic_grid(n1, device=device, dtype=dtype)
    g2, dz2 = per.periodic_grid(n2, device=device, dtype=dtype)
    return g1, g2, dz1, dz2


def kernels(g1, g2, bw1, bw2):
    """Wrapped-Gaussian 1-D kernel matrices for the two axes (``bw`` in radians)."""
    return (per.wrapped_gaussian_kernel_matrix(g1, bw1),
            per.wrapped_gaussian_kernel_matrix(g2, bw2))


def _idx(phi, n, dz):
    return torch.floor((phi + PI) / dz).long() % n


def scatter_counts(phi1, phi2, n1, n2, dz1, dz2):
    """Joint 2-D histogram counts of ``(phi1, phi2)`` (``(R, M)``) -> ``(R, n1, n2)``."""
    R = phi1.shape[0]
    i1 = _idx(phi1, n1, dz1)
    i2 = _idx(phi2, n2, dz2)
    lin = i1 * n2 + i2
    out = phi1.new_zeros(R, n1 * n2)
    out.scatter_add_(1, lin, torch.ones_like(phi1))
    return out.reshape(R, n1, n2)


def scatter_sum(phi1, phi2, values, n1, n2, dz1, dz2):
    """Per-cell sum of ``values`` at ``(phi1, phi2)`` -> ``(R, n1, n2)``."""
    R = phi1.shape[0]
    i1 = _idx(phi1, n1, dz1)
    i2 = _idx(phi2, n2, dz2)
    lin = i1 * n2 + i2
    out = phi1.new_zeros(R, n1 * n2)
    out.scatter_add_(1, lin, values)
    return out.reshape(R, n1, n2)


def smooth2(field, K1, K2):
    """Separable periodic smoothing ``K1 @ field @ K2^T`` -> ``(R, n1, n2)``."""
    return torch.einsum("ab,rbc,dc->rad", K1, field, K2)


def normalize2(p, dz1, dz2):
    p = torch.clamp(p, min=0.0)
    mass = p.sum(dim=(-2, -1), keepdim=True) * dz1 * dz2
    return p / mass.clamp_min(EPS)


def kde2(phi1, phi2, K1, K2, n1, n2, dz1, dz2):
    """Periodic 2-D KDE marginal of samples ``(phi1, phi2)`` on the torus grid."""
    counts = scatter_counts(phi1, phi2, n1, n2, dz1, dz2)
    return normalize2(smooth2(counts, K1, K2), dz1, dz2)


def mean_force_fields(f1_sum, f2_sum, count, K1, K2):
    """Smoothed 2-D conditional mean-force components ``(g1, g2)`` -> each ``(R, n1, n2)``.

    ``g_a = smooth(f_a_sum)/smooth(count)`` with unsupported cells set to 0.
    """
    den = smooth2(count, K1, K2)
    ok = den > EPS
    g1 = torch.where(ok, smooth2(f1_sum, K1, K2) / den.clamp_min(EPS), torch.zeros_like(den))
    g2 = torch.where(ok, smooth2(f2_sum, K1, K2) / den.clamp_min(EPS), torch.zeros_like(den))
    return g1, g2, den


def bilinear_interp2(profile, g1, g2, dz1, dz2, phi1, phi2):
    """Bilinear periodic interpolation of ``profile (R,n1,n2)`` at ``(phi1, phi2) (R,M)``."""
    n1, n2 = profile.shape[-2], profile.shape[-1]
    x1 = (phi1 - g1[0]) / dz1
    x2 = (phi2 - g2[0]) / dz2
    i0 = torch.floor(x1).long(); j0 = torch.floor(x2).long()
    f1 = x1 - i0.to(profile.dtype); f2 = x2 - j0.to(profile.dtype)
    i0m = i0 % n1; i1m = (i0 + 1) % n1
    j0m = j0 % n2; j1m = (j0 + 1) % n2
    R, M = phi1.shape
    flat = profile.reshape(R, n1 * n2)
    def gather(im, jm):
        return torch.gather(flat, 1, im * n2 + jm)
    v00 = gather(i0m, j0m); v01 = gather(i0m, j1m)
    v10 = gather(i1m, j0m); v11 = gather(i1m, j1m)
    return ((1 - f1) * (1 - f2) * v00 + (1 - f1) * f2 * v01
            + f1 * (1 - f2) * v10 + f1 * f2 * v11)


def kl2(p, q, dz1, dz2):
    """KL ``int p log(p/q) dz`` between two torus densities (clamped)."""
    p = torch.clamp(p, min=EPS); q = torch.clamp(q, min=EPS)
    return (p * (torch.log(p) - torch.log(q))).sum(dim=(-2, -1)) * dz1 * dz2


def tv2(p, q, dz1, dz2):
    return 0.5 * (torch.abs(p - q).sum(dim=(-2, -1)) * dz1 * dz2)


def l2_2d(profile, reference, dz1, dz2, mask=None, align=True):
    """RMS L2 between two torus scalar fields (additive-constant aligned; optional mask)."""
    if align:
        if mask is None:
            shift = (profile - reference).mean(dim=(-2, -1), keepdim=True)
        else:
            m = mask.to(profile.dtype)
            shift = ((profile - reference) * m).sum(dim=(-2, -1), keepdim=True) / \
                m.sum(dim=(-2, -1), keepdim=True).clamp_min(EPS)
        profile = profile - shift
    diff2 = (profile - reference) ** 2
    if mask is None:
        area = TWO_PI * TWO_PI
        return torch.sqrt(diff2.sum(dim=(-2, -1)) * dz1 * dz2 / area)
    m = mask.to(profile.dtype)
    area = (m.sum(dim=(-2, -1)) * dz1 * dz2).clamp_min(EPS)
    return torch.sqrt((diff2 * m).sum(dim=(-2, -1)) * dz1 * dz2 / area)


def entropy2(p, dz1, dz2):
    """Differential entropy ``-int p log p dz`` of a torus density (diagnostic)."""
    p = torch.clamp(p, min=EPS)
    return -(p * torch.log(p)).sum(dim=(-2, -1)) * dz1 * dz2


def fr_score_2d(phi1, phi2, p_grid, q_grid, g1, g2, dz1, dz2, clip):
    """Centred+clipped 2-D Fisher--Rao score per particle.

    ``S(z) = log[(p+eps)/(q+eps)] - int p log[(p+eps)/(q+eps)] dz`` interpolated to each
    particle, then recentred over particles and clipped (fixed-population balance).
    Returns ``(score (R,M), kl (R,))``.
    """
    from .core import _recentered_clipped_score
    p_at = bilinear_interp2(p_grid, g1, g2, dz1, dz2, phi1, phi2)
    q_at = bilinear_interp2(q_grid, g1, g2, dz1, dz2, phi1, phi2)
    log_ratio_grid = torch.log(p_grid.clamp_min(EPS)) - torch.log(q_grid.clamp_min(EPS))
    kl = (p_grid * log_ratio_grid).sum(dim=(-2, -1)) * dz1 * dz2
    raw = torch.log(p_at.clamp_min(EPS)) - torch.log(q_at.clamp_min(EPS)) - kl[:, None]
    return _recentered_clipped_score(raw, clip), kl
