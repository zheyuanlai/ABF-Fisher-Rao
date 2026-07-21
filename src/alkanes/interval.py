"""Non-periodic (bounded-interval) estimators for a distance CV ``R in [lo, hi]``.

Mirrors :mod:`alkanes.periodic` but on a *bounded, non-circular* interval, so all
smoothing/interpolation/integration use Euclidean (not wrapped) distance and the free
energy is integrated from the left edge (no zero-net-drift projection).  Because a
histogram/KDE on a bounded interval is biased at the two walls, the KDE marginal uses a
**reflected** Gaussian kernel (mass reflected across both walls, conserving it inside the
interval); the Nadaraya--Watson mean force uses a plain truncated Gaussian (its ratio
form is far less boundary-sensitive and the soft walls keep support away from the edges).

Grid is uniform and cell-centred (spacing ``dz = (hi-lo)/n_grid``).  All ops carry an
optional leading batch (run) dimension ``R``.
"""
from __future__ import annotations

import torch

EPS = 1.0e-12


def interval_grid(n_grid, lo, hi, device=None, dtype=torch.float64):
    """Cell-centred uniform grid on ``[lo, hi]`` and spacing ``dz``."""
    dz = (hi - lo) / n_grid
    grid = lo + (torch.arange(n_grid, device=device, dtype=dtype) + 0.5) * dz
    return grid, float(dz)


def gaussian_kernel_matrix(grid, bandwidth):
    """``K[i,j] = exp(-0.5 ((grid_i - grid_j)/bw)^2)`` (unnormalised, non-periodic)."""
    d = grid[:, None] - grid[None, :]
    return torch.exp(-0.5 * (d / max(float(bandwidth), EPS)) ** 2)


def reflected_kernel_matrix(grid, bandwidth, lo, hi):
    """KDE kernel with reflecting boundaries at ``lo`` and ``hi`` (mass-conserving).

    ``Kr[i,j] = G(z_i - z_j) + G(z_i - (2lo - z_j)) + G(z_i - (2hi - z_j))`` with the
    unnormalised Gaussian ``G``.  Reflecting each sample across both walls removes the
    leading boundary bias of a bounded-interval KDE.
    """
    bw = max(float(bandwidth), EPS)
    zi = grid[:, None]
    zj = grid[None, :]
    def G(d):
        return torch.exp(-0.5 * (d / bw) ** 2)
    return G(zi - zj) + G(zi - (2.0 * lo - zj)) + G(zi - (2.0 * hi - zj))


def bin_index(x, n_grid, lo, hi):
    dz = (hi - lo) / n_grid
    return torch.floor((x - lo) / dz).long().clamp(0, n_grid - 1)


def bin_counts(x, n_grid, lo, hi):
    """Histogram counts of ``x`` (``(..., M)``) into ``n_grid`` bins on ``[lo,hi]``.

    Out-of-range samples are clamped into the edge bins (soft walls keep this rare).
    """
    idx = bin_index(x, n_grid, lo, hi)
    out = x.new_zeros(x.shape[:-1] + (n_grid,))
    out.scatter_add_(-1, idx, torch.ones_like(x))
    return out


def bin_sum(x, values, n_grid, lo, hi):
    idx = bin_index(x, n_grid, lo, hi)
    out = x.new_zeros(x.shape[:-1] + (n_grid,))
    out.scatter_add_(-1, idx, values)
    return out


def smooth(y, K):
    """Kernel smoothing ``(..., n_grid) @ K^T``."""
    return y @ K.T


def normalize_density(p, dz):
    p = torch.clamp(p, min=0.0)
    mass = p.sum(-1, keepdim=True) * dz
    return p / mass.clamp_min(EPS)


def kde_marginal(x, K_reflect, n_grid, dz, lo, hi):
    """Reflected-boundary Gaussian KDE marginal of samples ``x`` on the interval grid."""
    counts = bin_counts(x, n_grid, lo, hi)
    return normalize_density(smooth(counts, K_reflect), dz)


def interval_interp(profile, grid, x):
    """Linear interpolation of ``profile`` (on ``grid``) at ``x`` with edge clamping.

    ``profile`` ``(..., n_grid)``, ``x`` ``(..., M)`` sharing leading batch dims.
    """
    n_grid = grid.shape[0]
    lo = grid[0]
    dz = grid[1] - grid[0]
    xf = ((x - lo) / dz)
    i0 = torch.floor(xf).long().clamp(0, n_grid - 2)
    frac = (xf - i0.to(x.dtype)).clamp(0.0, 1.0)
    p0 = torch.gather(profile, -1, i0)
    p1 = torch.gather(profile, -1, i0 + 1)
    return (1.0 - frac) * p0 + frac * p1


def mean_force_profile(force_sum, count, K):
    """Kernel-smoothed conditional mean force ``sum_j K S_j / sum_j K C_j`` (0 if unsupported)."""
    num = smooth(force_sum, K)
    den = smooth(count, K)
    mf = num / den.clamp_min(EPS)
    return torch.where(den > EPS, mf, torch.zeros_like(mf))


def free_energy_from_mean_force(mf, grid, dz):
    """Non-periodic PMF ``F(z) = int_lo^z F'(z') dz'`` (cumulative trapezoid), centred.

    No zero-net-drift projection (the interval is not periodic).  Centred on its mean so
    additive-constant-aligned L2 is well posed.  Works with a leading batch dim.
    """
    incr = 0.5 * (mf + torch.roll(mf, shifts=1, dims=-1)) * dz
    incr[..., 0] = 0.0                          # F(grid[0]) := 0 before centring
    F = torch.cumsum(incr, dim=-1)
    return F - F.mean(-1, keepdim=True)


def effective_counts(count, K):
    return smooth(count, K)


# ---------------------------------------------------------------------------
# Interval error metrics (additive-constant aligned; optional thermal-window mask)
# ---------------------------------------------------------------------------
def align_additive_constant(profile, reference, mask=None):
    if mask is None:
        shift = (profile - reference).mean(-1, keepdim=True)
    else:
        m = mask.to(profile.dtype)
        shift = ((profile - reference) * m).sum(-1, keepdim=True) / m.sum(-1, keepdim=True).clamp_min(EPS)
    return profile - shift


def interval_l2(profile, reference, dz, lo, hi, mask=None, align=True):
    """RMS L2 ``sqrt( (1/W) int_window (p-r)^2 dz )`` on the interval (window ``mask``)."""
    p = align_additive_constant(profile, reference, mask) if align else profile
    diff2 = (p - reference) ** 2
    if mask is None:
        width = (hi - lo)
        return torch.sqrt((diff2.sum(-1) * dz) / width)
    m = mask.to(profile.dtype)
    width = (m.sum(-1) * dz).clamp_min(EPS)
    return torch.sqrt((diff2 * m).sum(-1) * dz / width)


def interval_tv(p, q, dz):
    return 0.5 * (torch.abs(p - q).sum(-1) * dz)


def interval_kl(p, q, dz):
    p = torch.clamp(p, min=EPS)
    q = torch.clamp(q, min=EPS)
    return (p * (torch.log(p) - torch.log(q))).sum(-1) * dz
