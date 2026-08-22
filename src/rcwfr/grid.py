"""1D reaction-coordinate grid primitives, batched over a leading run axis R.

Conventions are inherited from the validated ABF-Fisher-Rao / ABP-Fisher-Rao
engines (see docs/PROVENANCE.md):

* Gaussian kernels have radius round(4*bw/dx) and are normalized by (sum * dx);
* smoothing is reflect-pad (or circular-pad) + valid convolution;
* binned_density scatters to the nearest grid point, smooths, then normalizes to
  unit trapezoid mass and clamps at EPS.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as tF

EPS = 1e-30


def choose_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


DEVICE = choose_device()
DTYPE = torch.float64
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False


@dataclass(frozen=True)
class Grid1D:
    """Uniform grid on [xmin, xmax] with n nodes; `bc` fixes the boundary law.

    For bc='periodic' the node xmax is the periodic image of xmin, so the grid
    carries n nodes with the last one duplicated (trapezoid quadrature then
    integrates exactly one period).
    """
    xmin: float
    xmax: float
    n: int
    eval_lo: float
    eval_hi: float
    bc: str = "reflect"          # 'reflect' | 'periodic'

    @property
    def dx(self) -> float:
        return (self.xmax - self.xmin) / (self.n - 1)

    @property
    def volume(self) -> float:
        return self.xmax - self.xmin

    def x(self, device=DEVICE, dtype=DTYPE) -> torch.Tensor:
        return torch.linspace(self.xmin, self.xmax, self.n, device=device, dtype=dtype)

    def eval_mask(self, device=DEVICE, dtype=DTYPE) -> torch.Tensor:
        xg = self.x(device, dtype)
        return (xg >= self.eval_lo) & (xg <= self.eval_hi)

    def enforce(self, q: torch.Tensor) -> torch.Tensor:
        if self.bc == "periodic":
            return wrap_into(q, self.xmin, self.xmax)
        return reflect_into(q, self.xmin, self.xmax)


def gaussian_kernel(bw, dx, device, dtype):
    r = max(1, int(round(4.0 * bw / dx)))
    t = torch.arange(-r, r + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (t * dx / bw) ** 2)
    k = k / (k.sum() * dx)
    return k, r


def smooth(v, kernel, r, dx, bc="reflect"):
    """Pad + valid convolution along the grid axis.  v: (R, G) -> (R, G)."""
    R, G = v.shape
    if bc == "periodic":
        # node G-1 duplicates node 0: convolve the G-1 independent cells circularly
        core = v[:, : G - 1]
        pad = min(r, G - 2)
        vp = torch.cat([core[:, -pad:], core, core[:, :pad]], dim=1).unsqueeze(1)
        k = kernel[r - pad: kernel.numel() - (r - pad)].flip(0).view(1, 1, -1)
        out = tF.conv1d(vp, k).squeeze(1)
        return torch.cat([out, out[:, :1]], dim=1)
    pad = min(r, G - 1)
    vp = tF.pad(v.unsqueeze(1), (pad, pad), mode="reflect")
    k = kernel[r - pad: kernel.numel() - (r - pad)].flip(0).view(1, 1, -1)
    return tF.conv1d(vp, k).squeeze(1)


def cumtrapz(y, dx):
    seg = 0.5 * (y[:, 1:] + y[:, :-1]) * dx
    out = torch.zeros_like(y)
    out[:, 1:] = torch.cumsum(seg, dim=1)
    return out


def trapz(y, dx):
    return torch.sum(0.5 * (y[:, 1:] + y[:, :-1]) * dx, dim=1)


def nearest_bin(X, grid: Grid1D):
    return torch.clamp(torch.round((X - grid.xmin) / grid.dx).long(), 0, grid.n - 1)


def scatter_counts(X, grid: Grid1D, weights=None):
    """Raw nearest-node histogram (counts, or weighted counts).  (R,N) -> (R,G)."""
    R, N = X.shape
    idx = nearest_bin(X, grid)
    hist = torch.zeros((R, grid.n), device=X.device, dtype=X.dtype)
    hist.scatter_add_(1, idx, torch.ones_like(X) if weights is None else weights)
    if grid.bc == "periodic":              # fold the duplicated end node into node 0
        hist[:, 0] = hist[:, 0] + hist[:, -1]
        hist[:, -1] = hist[:, 0]
    return hist


def binned_density(X, kernel, r, grid: Grid1D, weights=None):
    """KDE of particle positions on the grid, normalized to unit mass.  -> (R,G)."""
    hist = scatter_counts(X, grid, weights)
    p = smooth(hist, kernel, r, grid.dx, grid.bc)
    mass = torch.clamp(trapz(p, grid.dx), min=EPS).unsqueeze(1)
    return torch.clamp(p / mass, min=EPS)


def interp1d(X, grid_vals, grid: Grid1D):
    """Linear interpolation of per-row profiles at particle locations."""
    pos = torch.clamp((X - grid.xmin) / grid.dx, 0.0, grid.n - 1.0)
    i0 = torch.clamp(torch.floor(pos).long(), 0, grid.n - 2)
    frac = pos - i0.to(X.dtype)
    v0 = torch.gather(grid_vals, 1, i0)
    v1 = torch.gather(grid_vals, 1, i0 + 1)
    return v0 + frac * (v1 - v0)


def central_diff(F, dx, bc="reflect"):
    Fp = torch.empty_like(F)
    Fp[:, 1:-1] = (F[:, 2:] - F[:, :-2]) / (2.0 * dx)
    if bc == "periodic":
        Fp[:, 0] = (F[:, 1] - F[:, -2]) / (2.0 * dx)
        Fp[:, -1] = Fp[:, 0]
    else:
        Fp[:, 0] = (F[:, 1] - F[:, 0]) / dx
        Fp[:, -1] = (F[:, -1] - F[:, -2]) / dx
    return Fp


def reflect_into(q, lo, hi):
    span = hi - lo
    qm = torch.remainder(q - lo, 2.0 * span)
    return torch.where(qm > span, 2.0 * span - qm, qm) + lo


def wrap_into(q, lo, hi):
    span = hi - lo
    return torch.remainder(q - lo, span) + lo
