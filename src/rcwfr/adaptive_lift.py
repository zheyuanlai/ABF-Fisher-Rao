"""The adiabatic lift, built from the run's own samples instead of an oracle.

docs/MANIFOLD_FORMULATION.md §4.2 argues that the only lift with a statistical
justification is the one solving the fiber continuity equation, and that in one
fiber dimension it equals the monotone map

    y' = CDF^{-1}_{nu(.|z')} ( CDF_{nu(.|z)} (y) ),

whose two ingredients -- the fiber-conditional density and, through it, the
mean-force fluctuation about F' -- a thermodynamic-integration run already
produces.  That argument is only worth anything if the map survives being built
from ESTIMATED conditionals, because the estimate is made from the very ensemble
the lift is steering.  A lift that reinforces its own error would be worse than
useless.

`AdaptiveFiberCDF` is the honest version: a running smoothed (z, y) histogram,
normalized per z to give nu_hat(y|z), with an explicit reliability mask.  Where
the estimate is not yet trustworthy the lift falls back to cartesian, so the
method degrades to the naive lift rather than to nonsense.
"""
from __future__ import annotations

import torch
import torch.nn.functional as tF

from .grid import EPS, Grid1D, gaussian_kernel


class AdaptiveFiberCDF:
    """Running estimate of nu(y | z) and the transport map it induces.

    rows        independent replicate runs, batched
    n_y         fiber-grid resolution (the conditional is 1-D here)
    bw_z, bw_y  smoothing bandwidths, in z and y units
    n_min       per-z effective count below which the lift falls back to cartesian
    decay       1.0 = plain running average; < 1 forgets early, badly-placed samples
    """

    def __init__(self, rows: int, grid: Grid1D, y_max: float, device, dtype,
                 n_y: int = 161, bw_z: float = 0.08, bw_y: float = 0.15,
                 n_min: float = 200.0, decay: float = 1.0):
        self.grid, self.rows = grid, rows
        self.device, self.dtype = device, dtype
        self.y = torch.linspace(-y_max, y_max, n_y, device=device, dtype=dtype)
        self.dy = float(self.y[1] - self.y[0])
        self.n_y, self.n_min, self.decay = n_y, n_min, decay
        self.kz, self.rz = gaussian_kernel(bw_z, grid.dx, device, dtype)
        self.ky, self.ry = gaussian_kernel(bw_y, self.dy, device, dtype)
        self.H = torch.zeros((rows, grid.n, n_y), device=device, dtype=dtype)
        self._cdf = None
        self._ok = None

    # ---- accumulation ------------------------------------------------------
    def deposit(self, Z, Y):
        """Scatter one batch of (z, y) samples to the nearest grid cell."""
        g = self.grid
        iz = torch.clamp(torch.round((Z - g.xmin) / g.dx).long(), 0, g.n - 1)
        iy = torch.clamp(torch.round((Y - self.y[0]) / self.dy).long(), 0, self.n_y - 1)
        flat = iz * self.n_y + iy
        if self.decay != 1.0:
            self.H.mul_(self.decay)
        self.H.view(self.rows, -1).scatter_add_(
            1, flat, torch.ones_like(flat, dtype=self.dtype))
        self._cdf = None

    # ---- conditional -------------------------------------------------------
    def _build(self):
        H = self.H.unsqueeze(1)                                  # (R,1,Gz,Gy)
        mode = "circular" if self.grid.bc == "periodic" else "replicate"
        # y is always non-periodic; pad it by replication in a second pass
        H = tF.pad(H, (0, 0, self.rz, self.rz), mode=mode)
        H = tF.conv2d(H, self.kz.view(1, 1, -1, 1))
        H = tF.pad(H, (self.ry, self.ry, 0, 0), mode="replicate")
        H = tF.conv2d(H, self.ky.view(1, 1, 1, -1)).squeeze(1)   # (R,Gz,Gy)
        H = torch.clamp(H, min=0.0)
        mass = torch.trapezoid(H, dx=self.dy, dim=-1)            # (R,Gz)
        # RAW counts decide reliability, not smoothed mass: smoothing borrows from
        # neighbouring z and would report support where none was sampled.
        self._ok = self.H.sum(-1) >= self.n_min
        pdf = H / torch.clamp(mass, min=EPS).unsqueeze(-1)
        cdf = torch.cumulative_trapezoid(pdf, dx=self.dy, dim=-1)
        cdf = torch.cat([torch.zeros(cdf.shape[:-1] + (1,), device=self.device,
                                     dtype=self.dtype), cdf], -1)
        self._cdf = cdf / torch.clamp(cdf[..., -1:], min=EPS)

    def _rows_at(self, Z):
        """Bilinear-in-z read of the CDF table at scattered z; (R,N,n_y)."""
        g = self.grid
        pos = torch.clamp((Z - g.xmin) / g.dx, 0.0, g.n - 1.0)
        i0 = torch.clamp(torch.floor(pos).long(), 0, g.n - 2)
        f = (pos - i0.to(Z.dtype)).unsqueeze(-1)
        c0 = torch.gather(self._cdf, 1, i0.unsqueeze(-1).expand(-1, -1, self.n_y))
        c1 = torch.gather(self._cdf, 1, (i0 + 1).unsqueeze(-1).expand(-1, -1, self.n_y))
        ok = torch.gather(self._ok, 1, i0) & torch.gather(self._ok, 1, i0 + 1)
        return c0 + f * (c1 - c0), ok

    # ---- the lift ----------------------------------------------------------
    def lift(self, Z, Y, Zn):
        """Transport Y from fiber Z to fiber Zn under the ESTIMATED conditional.

        Falls back to the cartesian lift (Y unchanged) wherever either endpoint's
        conditional is not yet supported by `n_min` samples.
        """
        if self._cdf is None:
            self._build()
        cdf0, ok0 = self._rows_at(Z)
        cdf1, ok1 = self._rows_at(Zn)
        ok = ok0 & ok1
        # u = CDF_hat(z, y)
        pos = torch.clamp((Y - self.y[0]) / self.dy, 0.0, self.n_y - 1.0)
        j0 = torch.clamp(torch.floor(pos).long(), 0, self.n_y - 2)
        fy = (pos - j0.to(Y.dtype)).unsqueeze(-1)
        lo = torch.gather(cdf0, -1, j0.unsqueeze(-1))
        hi = torch.gather(cdf0, -1, (j0 + 1).unsqueeze(-1))
        u = torch.clamp((lo + fy * (hi - lo)).squeeze(-1), 1e-9, 1 - 1e-9)
        # y' = CDF_hat^{-1}(z', u)
        j = torch.clamp(torch.searchsorted(cdf1.contiguous(),
                                           u.unsqueeze(-1).contiguous()),
                        1, self.n_y - 1)
        a = torch.gather(cdf1, -1, j - 1).squeeze(-1)
        b = torch.gather(cdf1, -1, j).squeeze(-1)
        t = (u - a) / torch.clamp(b - a, min=EPS)
        y0 = self.y[(j - 1).squeeze(-1)]
        y1 = self.y[j.squeeze(-1)]
        Yn = y0 + t * (y1 - y0)
        return torch.where(ok, Yn, Y)

    def coverage(self):
        if self._cdf is None:
            self._build()
        return float(self._ok.to(self.dtype).mean())
