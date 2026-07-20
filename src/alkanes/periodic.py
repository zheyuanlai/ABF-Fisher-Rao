"""Periodic (circular) estimators for a dihedral CV on ``phi in [-pi, pi)``.

Everything that touches ``phi`` -- histograms, KDE, smoothing, interpolation,
differentiation, integration, distances -- is periodic.  The grid is uniform and
cell-centred (no duplicated endpoint), spacing ``dphi = 2*pi/n_grid``.

Kernels are wrapped Gaussians on the circular distance (equivalently a truncated
von Mises), precomputed once as ``(n_grid, n_grid)`` matrices so smoothing is a
single matmul; this is exact periodic Nadaraya--Watson / KDE on the grid.

All array ops carry an optional leading batch (run) dimension ``R`` so many seeds
run in one GPU process.
"""
from __future__ import annotations

import math

import torch

PI = math.pi
TWO_PI = 2.0 * math.pi
EPS = 1.0e-12


def periodic_grid(n_grid, device=None, dtype=torch.float64):
    """Cell-centred uniform grid on ``[-pi, pi)`` and spacing ``dphi``."""
    dphi = TWO_PI / n_grid
    grid = -PI + (torch.arange(n_grid, device=device, dtype=dtype) + 0.5) * dphi
    return grid, dphi


def circular_distance(a, b):
    """Signed circular distance ``wrap(a-b)`` in ``[-pi, pi)`` (broadcasts)."""
    d = a - b
    return d - TWO_PI * torch.round(d / TWO_PI)


def wrapped_gaussian_kernel_matrix(grid, bandwidth):
    """``K[i,j] = exp(-0.5 (circdist(grid_i, grid_j)/bw)^2)`` (unnormalised)."""
    d = circular_distance(grid[:, None], grid[None, :])
    return torch.exp(-0.5 * (d / max(float(bandwidth), EPS)) ** 2)


def bin_counts(phi, n_grid):
    """Histogram counts of ``phi`` (``(..., M)``) into ``n_grid`` circular bins -> ``(..., n_grid)``."""
    dphi = TWO_PI / n_grid
    idx = torch.floor((phi + PI) / dphi).long() % n_grid
    out = phi.new_zeros(phi.shape[:-1] + (n_grid,))
    out.scatter_add_(-1, idx, torch.ones_like(phi))
    return out


def bin_sum(phi, values, n_grid):
    """Per-bin sum of ``values`` at locations ``phi`` -> ``(..., n_grid)``."""
    dphi = TWO_PI / n_grid
    idx = torch.floor((phi + PI) / dphi).long() % n_grid
    out = phi.new_zeros(phi.shape[:-1] + (n_grid,))
    out.scatter_add_(-1, idx, values)
    return out


def smooth(y, K):
    """Circular kernel smoothing: ``(... , n_grid) @ K^T`` -> ``(..., n_grid)``."""
    return y @ K.T


def normalize_density(p, dphi):
    """Clamp and normalise a density on the circular grid to integrate to 1."""
    p = torch.clamp(p, min=0.0)
    mass = p.sum(-1, keepdim=True) * dphi
    return p / mass.clamp_min(EPS)


def kde_marginal(phi, K_kde, n_grid, dphi):
    """Periodic KDE marginal of samples ``phi`` (``(..., M)``) on the grid."""
    counts = bin_counts(phi, n_grid)
    return normalize_density(smooth(counts, K_kde), dphi)


def circular_interp(profile, grid, phi):
    """Periodic linear interpolation of ``profile`` (on ``grid``) at ``phi``.

    ``profile`` is ``(..., n_grid)`` and ``phi`` is ``(..., M)`` sharing the leading
    batch dims; returns ``(..., M)``.  Cell-centred uniform grid with circular wrap.
    """
    n_grid = grid.shape[0]
    dphi = TWO_PI / n_grid
    x = (phi - grid[0]) / dphi          # fractional index, grid[0] = -pi + dphi/2
    i0 = torch.floor(x).long()
    frac = x - i0.to(phi.dtype)
    i0m = i0 % n_grid
    i1m = (i0 + 1) % n_grid
    p0 = torch.gather(profile, -1, i0m)
    p1 = torch.gather(profile, -1, i1m)
    return (1.0 - frac) * p0 + frac * p1


def mean_force_profile(force_sum, count, K_abf):
    """Kernel-smoothed conditional mean force ``F'(phi_i)=sum_j K S_j / sum_j K C_j``.

    ``force_sum`` and ``count`` are ``(..., n_grid)`` accumulators.  Bins with no
    kernel-weighted support return 0 (edge handled by the periodicity, not clamping).
    """
    num = smooth(force_sum, K_abf)
    den = smooth(count, K_abf)
    mf = num / den.clamp_min(EPS)
    return torch.where(den > EPS, mf, torch.zeros_like(mf))


def free_energy_from_mean_force(mf, grid, dphi):
    """Periodic PMF from a mean-force profile, enforcing ``integral F' dphi = 0``.

    Subtracts the circular mean of ``F'`` (so the reconstructed F is periodic), then
    circular-cumulative-integrates (trapezoid) and centres F on its circular mean.
    Works with a leading batch dim.
    """
    mf0 = mf - mf.mean(-1, keepdim=True)                 # enforce zero net drift
    # trapezoid increments between consecutive cell centres (periodic)
    incr = 0.5 * (mf0 + torch.roll(mf0, shifts=1, dims=-1)) * dphi
    F = torch.cumsum(incr, dim=-1)
    F = F - F.mean(-1, keepdim=True)
    return F


def effective_counts(count, K_abf):
    """Kernel-weighted per-bin support (proxy for N_eff(phi_j))."""
    return smooth(count, K_abf)


# ---------------------------------------------------------------------------
# Circular error metrics (additive-constant aligned)
# ---------------------------------------------------------------------------
def align_additive_constant(profile, reference):
    """Shift ``profile`` by the constant minimising L2 vs ``reference`` (circular mean)."""
    return profile - (profile - reference).mean(-1, keepdim=True)


def circular_l2(profile, reference, dphi, align=True):
    """RMS L2 distance ``sqrt( (1/2pi) integral (p-r)^2 dphi )`` on the circle."""
    p = align_additive_constant(profile, reference) if align else profile
    diff2 = (p - reference) ** 2
    return torch.sqrt((diff2.sum(-1) * dphi) / TWO_PI)


def marginal_tv(p, q, dphi):
    """Total variation ``0.5 integral |p-q| dphi`` between circular densities."""
    return 0.5 * (torch.abs(p - q).sum(-1) * dphi)


def marginal_kl(p, q, dphi):
    """KL ``integral p log(p/q) dphi`` (clamped)."""
    p = torch.clamp(p, min=EPS)
    q = torch.clamp(q, min=EPS)
    return (p * (torch.log(p) - torch.log(q))).sum(-1) * dphi


def marginal_l2(p, q, dphi):
    """RMS L2 between two circular densities (no additive alignment)."""
    return torch.sqrt(((p - q) ** 2).sum(-1) * dphi / TWO_PI)
