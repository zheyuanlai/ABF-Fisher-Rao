"""Two-dimensional reaction-coordinate machinery.

The 1-D case gets `F` from `cumtrapz` of a binned mean force.  In two dimensions
the mean force is a VECTOR FIELD that the estimator does not force to be a
gradient, so `F` is the least-squares potential of that field:

    minimise  ||grad F - f_hat||^2   <=>   laplacian F = div f_hat,

solved spectrally on the periodic torus in one FFT round trip.  The residual
curl of `f_hat` is exactly the part of the estimate that no `F` can explain, and
it is reported (`curl_frac`) rather than silently projected away -- it is a free
convergence diagnostic that the 1-D case does not have.

Grids are periodic in both coordinates, or reflecting in the first (a CV with an
inaccessible arc, as alanine's `phi` has).  A reflecting axis is handled by even
extension before the transform, which imposes the natural Neumann condition.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as tF

from ..grid import EPS, Grid1D


@dataclass(frozen=True)
class Grid2D:
    gx: Grid1D
    gy: Grid1D

    @property
    def shape(self):
        return (self.gx.n, self.gy.n)

    @property
    def cell(self):
        return self.gx.dx * self.gy.dx

    def axes(self, device, dtype):
        return self.gx.x(device, dtype), self.gy.x(device, dtype)

    def enforce(self, Z):
        """Z: (..., 2)."""
        return torch.stack([self.gx.enforce(Z[..., 0]), self.gy.enforce(Z[..., 1])], -1)

    def mask(self, device, dtype):
        return (self.gx.eval_mask(device, dtype).unsqueeze(1)
                & self.gy.eval_mask(device, dtype).unsqueeze(0))


def _pad_mode(bc):
    return "circular" if bc == "periodic" else "replicate"


def smooth2d(H, kx, rx, ky, ry, gx: Grid1D, gy: Grid1D):
    """Separable smoothing with the 1-D boundary law on each axis.  H: (R,Gx,Gy)."""
    X = H.unsqueeze(1)
    X = tF.pad(X, (0, 0, rx, rx), mode=_pad_mode(gx.bc))
    X = tF.conv2d(X, kx.view(1, 1, -1, 1))
    X = tF.pad(X, (ry, ry, 0, 0), mode=_pad_mode(gy.bc))
    X = tF.conv2d(X, ky.view(1, 1, 1, -1))
    return X.squeeze(1)


def scatter2d(Z, g2: Grid2D, weights=None):
    """Nearest-node histogram of Z (R,N,2) onto (R,Gx,Gy)."""
    R, N, _ = Z.shape
    gx, gy = g2.gx, g2.gy
    ix = torch.clamp(torch.round((Z[..., 0] - gx.xmin) / gx.dx).long(), 0, gx.n - 1)
    iy = torch.clamp(torch.round((Z[..., 1] - gy.xmin) / gy.dx).long(), 0, gy.n - 1)
    w = torch.ones_like(Z[..., 0]) if weights is None else weights
    out = torch.zeros((R, gx.n * gy.n), device=Z.device, dtype=Z.dtype)
    out.scatter_add_(1, ix * gy.n + iy, w)
    return out.reshape(R, gx.n, gy.n)


class MeanForceAccumulator2D:
    """The 2-D analogue of the shared 1-D accumulator, same ramp, same kernel."""

    def __init__(self, rows, g2: Grid2D, bw, n_min, device, dtype):
        from ..grid import gaussian_kernel
        self.g2, self.n_min = g2, n_min
        self.kx, self.rx = gaussian_kernel(bw, g2.gx.dx, device, dtype)
        self.ky, self.ry = gaussian_kernel(bw, g2.gy.dx, device, dtype)
        sh = (rows, g2.gx.n, g2.gy.n)
        self.S0 = torch.zeros(sh, device=device, dtype=dtype)
        self.S1 = torch.zeros(sh + (2,), device=device, dtype=dtype)

    def deposit(self, Z, f, weights=None):
        """Z: (R,N,2); f: (R,N,2)."""
        w = torch.ones_like(Z[..., 0]) if weights is None else weights
        sm = lambda H: smooth2d(H, self.kx, self.rx, self.ky, self.ry,
                                self.g2.gx, self.g2.gy)
        self.S0 += sm(scatter2d(Z, self.g2, w))
        for c in range(2):
            self.S1[..., c] += sm(scatter2d(Z, self.g2, w * f[..., c]))

    def zero_(self):
        self.S0.zero_(); self.S1.zero_()

    def mean_force(self):
        return self.S1 / (self.S0 + self.n_min).unsqueeze(-1)

    def free_energy(self, mask=None):
        F, _ = poisson_potential(self.mean_force(), self.g2)
        if mask is not None:
            R = F.shape[0]
            m = F.reshape(R, -1)[:, mask.reshape(-1)].mean(1)
            F = F - m.view(R, 1, 1)
        return F

    def curl_fraction(self):
        return poisson_potential(self.mean_force(), self.g2)[1]


def poisson_potential(fhat, g2: Grid2D):
    """Least-squares potential of a sampled vector field, plus its curl fraction.

    fhat: (R, Gx, Gy, 2) -> (F: (R,Gx,Gy), curl_frac: (R,))

    Periodic axes transform directly; a reflecting axis is extended evenly first,
    which is the Neumann condition a bounded CV needs.  The curl fraction is
    ||f - grad F|| / ||f||: the share of the estimated mean force that is not a
    gradient of anything, and therefore pure estimation error.
    """
    gx, gy = g2.gx, g2.gy
    px, py = gx.bc == "periodic", gy.bc == "periodic"
    f = fhat
    # drop the duplicated end node on a periodic axis, mirror a reflecting one
    if px:
        f = f[:, :-1]
    else:
        f = torch.cat([f, torch.flip(f, dims=(1,))], dim=1).clone()
        f[:, f.shape[1] // 2:, :, 0] *= -1.0        # x-component is odd under the mirror
    if py:
        f = f[:, :, :-1]
    else:
        f = torch.cat([f, torch.flip(f, dims=(2,))], dim=2).clone()
        f[:, :, f.shape[2] // 2:, 1] *= -1.0
    R, Nx, Ny, _ = f.shape
    Lx, Ly = Nx * gx.dx, Ny * gy.dx
    kx = 2 * math.pi * torch.fft.fftfreq(Nx, d=gx.dx, device=f.device, dtype=f.dtype)
    ky = 2 * math.pi * torch.fft.fftfreq(Ny, d=gy.dx, device=f.device, dtype=f.dtype)
    KX, KY = kx.view(-1, 1), ky.view(1, -1)
    K2 = KX ** 2 + KY ** 2
    Fx = torch.fft.fft2(f[..., 0]); Fy = torch.fft.fft2(f[..., 1])
    num = -1j * (KX * Fx + KY * Fy)
    Fh = torch.where(K2 > 0, num / torch.clamp(K2, min=EPS), torch.zeros_like(num))
    Fs = torch.fft.ifft2(Fh).real
    gxF = torch.fft.ifft2(1j * KX * Fh).real
    gyF = torch.fft.ifft2(1j * KY * Fh).real
    num_c = ((f[..., 0] - gxF) ** 2 + (f[..., 1] - gyF) ** 2).sum((1, 2))
    den = (f[..., 0] ** 2 + f[..., 1] ** 2).sum((1, 2))
    curl = torch.sqrt(num_c / torch.clamp(den, min=EPS))
    # crop back to the original grid, restoring the duplicated end node
    F = Fs[:, :gx.n - 1 if px else gx.n, :gy.n - 1 if py else gy.n]
    if px:
        F = torch.cat([F, F[:, :1]], dim=1)
    if py:
        F = torch.cat([F, F[:, :, :1]], dim=2)
    return F, curl


def gauge_l2_2d(F, F_ref, mask):
    R = F.shape[0]
    d = (F - F_ref).reshape(R, -1)[:, mask.reshape(-1)]
    d = d - d.mean(dim=1, keepdim=True)
    return torch.sqrt((d * d).mean(dim=1))


def kde2d(Z, g2: Grid2D, bw):
    from ..grid import gaussian_kernel
    kx, rx = gaussian_kernel(bw, g2.gx.dx, Z.device, Z.dtype)
    ky, ry = gaussian_kernel(bw, g2.gy.dx, Z.device, Z.dtype)
    p = smooth2d(scatter2d(Z, g2), kx, rx, ky, ry, g2.gx, g2.gy)
    m = torch.clamp(p.sum((1, 2)) * g2.cell, min=EPS).view(-1, 1, 1)
    return torch.clamp(p / m, min=EPS)


def interp2d(Z, vals, g2: Grid2D):
    """Bilinear read of per-row grid profiles at scattered Z.  vals: (R,Gx,Gy)."""
    gx, gy = g2.gx, g2.gy
    px = torch.clamp((Z[..., 0] - gx.xmin) / gx.dx, 0.0, gx.n - 1.0)
    py = torch.clamp((Z[..., 1] - gy.xmin) / gy.dx, 0.0, gy.n - 1.0)
    i0 = torch.clamp(torch.floor(px).long(), 0, gx.n - 2)
    j0 = torch.clamp(torch.floor(py).long(), 0, gy.n - 2)
    a, b = px - i0.to(Z.dtype), py - j0.to(Z.dtype)
    V = vals.reshape(vals.shape[0], -1)
    g = lambda i, j: torch.gather(V, 1, i * gy.n + j)
    return ((1 - a) * (1 - b) * g(i0, j0) + a * (1 - b) * g(i0 + 1, j0)
            + (1 - a) * b * g(i0, j0 + 1) + a * b * g(i0 + 1, j0 + 1))
