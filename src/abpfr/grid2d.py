"""2D periodic (torus) reaction-coordinate grid primitives, batched over rows.

Counterpart of grid.py for xi in T^2 (e.g. (phi, psi) dihedrals). Conventions mirror
the validated 1D layer wherever they transfer, with the boundary treatment replaced
by the torus topology:

* nodes at xmin + i*dx, i = 0..n-1 with dx = L/n — NO duplicated endpoint; -pi and
  pi are the same point;
* Gaussian kernels have radius round(4*bw/dx), normalized by (sum * dx), applied by
  CIRCULAR separable convolution (the periodic mollifier delta_eps);
* integrals are plain Riemann sums (sum * dA), which on a torus is the trapezoid
  rule exactly;
* binned_density2 scatters to the nearest node modulo n, smooths, normalizes to
  unit mass, clamps at EPS;
* interp2/central_diff2 are bilinear/central with periodic index wrap: -pi and pi
  are never treated as distant points.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as tF

from .grid import DEVICE, DTYPE, EPS


@dataclass(frozen=True)
class GridT2:
    """Periodic rectangle [x1min, x1min+L1) x [x2min, x2min+L2), n1 x n2 nodes."""
    x1min: float
    L1: float
    n1: int
    x2min: float
    L2: float
    n2: int

    @property
    def dx1(self) -> float:
        return self.L1 / self.n1

    @property
    def dx2(self) -> float:
        return self.L2 / self.n2

    @property
    def volume(self) -> float:
        return self.L1 * self.L2

    @property
    def dA(self) -> float:
        return self.dx1 * self.dx2

    def x1(self, device=DEVICE, dtype=DTYPE) -> torch.Tensor:
        return self.x1min + self.dx1 * torch.arange(self.n1, device=device,
                                                    dtype=dtype)

    def x2(self, device=DEVICE, dtype=DTYPE) -> torch.Tensor:
        return self.x2min + self.dx2 * torch.arange(self.n2, device=device,
                                                    dtype=dtype)

    def mesh(self, device=DEVICE, dtype=DTYPE):
        """(n1, n2) coordinate grids."""
        return torch.meshgrid(self.x1(device, dtype), self.x2(device, dtype),
                              indexing="ij")


def wrap_periodic(x, lo, L):
    """Map positions into [lo, lo + L)."""
    return lo + torch.remainder(x - lo, L)


def torus_distance(a, b, L):
    """Minimum-image distance on a circle of circumference L."""
    d = torch.remainder(a - b, L)
    return torch.minimum(d, L - d)


def periodic_gaussian_kernel(bw, dx, n, device, dtype):
    """Same truncated kernel as the 1D layer (radius 4*bw/dx, sum*dx normalized).

    The radius must fit the torus (r < n/2) so the circular convolution never
    wraps onto itself.
    """
    r = max(1, int(round(4.0 * bw / dx)))
    assert 2 * r < n, (f"kernel radius {r} does not fit a periodic axis of {n} "
                       f"nodes; reduce bw or refine the grid")
    t = torch.arange(-r, r + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (t * dx / bw) ** 2)
    k = k / (k.sum() * dx)
    return k, r


def smooth2(v, k1, r1, k2, r2):
    """Separable circular convolution.  v: (R, n1, n2) -> (R, n1, n2).

    No dx scaling (kernels are 1/(sum*dx)-normalized), mirroring grid.smooth.
    """
    R, n1, n2 = v.shape
    # axis 1 (length n1): treat n2 as batch
    x = v.permute(0, 2, 1).reshape(R * n2, 1, n1)
    x = tF.pad(x, (r1, r1), mode="circular")
    x = tF.conv1d(x, k1.flip(0).view(1, 1, -1))
    v = x.reshape(R, n2, n1).permute(0, 2, 1)
    # axis 2 (length n2): n1 as batch
    x = v.reshape(R * n1, 1, n2)
    x = tF.pad(x, (r2, r2), mode="circular")
    x = tF.conv1d(x, k2.flip(0).view(1, 1, -1))
    return x.reshape(R, n1, n2)


def integral2(y, grid: GridT2):
    """Torus integral: plain Riemann sum (= trapezoid on a periodic domain).
    y: (R, n1, n2) -> (R,)."""
    return y.sum(dim=(1, 2)) * grid.dA


def nearest_bin2(X1, X2, grid: GridT2):
    """Nearest-node FLAT index modulo the torus.  X1, X2: (R, N) -> (R, N) long."""
    i1 = torch.remainder(torch.round((X1 - grid.x1min) / grid.dx1).long(), grid.n1)
    i2 = torch.remainder(torch.round((X2 - grid.x2min) / grid.dx2).long(), grid.n2)
    return i1 * grid.n2 + i2


def binned_density2(X1, X2, k1, r1, k2, r2, grid: GridT2, weights=None):
    """(Optionally weighted) periodic KDE.  X: (R, N) -> p: (R, n1, n2).

    weights = None is the equal-weight ensemble; (R, N) positive weights estimate the
    density of the measure the ensemble REPRESENTS (sum w_k delta_k / sum w_k).
    Weights of exactly 1 reproduce the unweighted call bitwise.
    """
    R, N = X1.shape
    idx = nearest_bin2(X1, X2, grid)
    hist = torch.zeros((R, grid.n1 * grid.n2), device=X1.device, dtype=X1.dtype)
    hist.scatter_add_(1, idx, torch.ones_like(X1) if weights is None else weights)
    p = smooth2(hist.reshape(R, grid.n1, grid.n2), k1, r1, k2, r2)
    p = p / (float(N) if weights is None
             else torch.clamp(weights.sum(dim=1), min=EPS).reshape(R, 1, 1))
    mass = torch.clamp(integral2(p, grid), min=EPS).reshape(R, 1, 1)
    return torch.clamp(p / mass, min=EPS)


def interp2(X1, X2, grid_vals, grid: GridT2):
    """Bilinear periodic interpolation.  grid_vals: (R, n1, n2), X: (R, N) -> (R, N)."""
    R, N = X1.shape
    pos1 = (X1 - grid.x1min) / grid.dx1
    pos2 = (X2 - grid.x2min) / grid.dx2
    i1 = torch.floor(pos1).long()
    i2 = torch.floor(pos2).long()
    f1 = (pos1 - i1.to(X1.dtype)).unsqueeze(-1)
    f2 = (pos2 - i2.to(X2.dtype)).unsqueeze(-1)
    i1 = torch.remainder(i1, grid.n1)
    i2 = torch.remainder(i2, grid.n2)
    j1 = torch.remainder(i1 + 1, grid.n1)
    j2 = torch.remainder(i2 + 1, grid.n2)
    flat = grid_vals.reshape(R, -1)
    g = lambda a, b: torch.gather(flat, 1, a * grid.n2 + b)
    v00, v01 = g(i1, i2), g(i1, j2)
    v10, v11 = g(j1, i2), g(j1, j2)
    f1, f2 = f1.squeeze(-1), f2.squeeze(-1)
    return (v00 * (1 - f1) * (1 - f2) + v10 * f1 * (1 - f2)
            + v01 * (1 - f1) * f2 + v11 * f1 * f2)


def central_diff2(F, grid: GridT2):
    """Periodic central differences.  F: (R, n1, n2) -> (dF/dx1, dF/dx2)."""
    d1 = (torch.roll(F, -1, dims=1) - torch.roll(F, 1, dims=1)) / (2.0 * grid.dx1)
    d2 = (torch.roll(F, -1, dims=2) - torch.roll(F, 1, dims=2)) / (2.0 * grid.dx2)
    return d1, d2


def kl_to_uniform2(p, grid: GridT2):
    """KL(p || u) on the torus.  p: (R, n1, n2) -> (R,)."""
    import math
    u_log = math.log(1.0 / grid.volume)
    integrand = p * (torch.log(torch.clamp(p, min=EPS)) - u_log)
    return integral2(integrand, grid)


def tv_to_uniform2(p, grid: GridT2):
    return 0.5 * integral2((p - 1.0 / grid.volume).abs(), grid)


def uniform_log_ratio2(X1, X2, p_grid, grid: GridT2):
    """log(u / p_hat) at walker positions (the FR score)."""
    import math
    u_log = math.log(1.0 / grid.volume)
    p_at = torch.clamp(interp2(X1, X2, p_grid, grid), min=EPS)
    return u_log - torch.log(p_at)
