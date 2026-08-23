"""Conditional-measure transport of ONE slow fiber coordinate, for molecules.

The manifold phase established that the lift which matters is the one solving
the fiber continuity equation

    d_z nu_z + d_y (nu_z w*) = 0,

and that in one fiber dimension its exact solution is the monotone CDF map

    y' = F^{-1}_{z'} ( F_z(y) ).

For a torsional fiber coordinate y is PERIODIC, so a CDF needs a cut.  The cut
is placed at y = -pi, which for an alkane is the cis barrier -- the least
populated point of every conditional -- so the map is continuous to within the
density there, and the residual is reported (`cut_mass`).

Two sources for F_z:
  `ReferenceFiberCDF`  the reference 2-D histogram: the ORACLE lift, the ceiling
  `AdaptiveFiberCDF`   the run's own (z, y) samples with forgetting: the method
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as tF

from ..grid import EPS, Grid1D, gaussian_kernel


def _smooth2d_periodic(H, kz, rz, ky, ry):
    """Circular smoothing in both axes.  H: (R, Gz, Gy)."""
    X = H.unsqueeze(1)
    X = tF.pad(X, (0, 0, rz, rz), mode="circular")
    X = tF.conv2d(X, kz.view(1, 1, -1, 1))
    X = tF.pad(X, (ry, ry, 0, 0), mode="circular")
    X = tF.conv2d(X, ky.view(1, 1, 1, -1))
    return X.squeeze(1)


class _CDFTable:
    """Shared machinery: a (R, Gz, Gy) table -> per-z conditional CDF -> the map."""

    def __init__(self, rows, gz: Grid1D, gy: Grid1D, device, dtype,
                 bw_z, bw_y, n_min, eps_bg=0.02):
        self.rows, self.gz, self.gy = rows, gz, gy
        self.device, self.dtype = device, dtype
        self.n_y = gy.n
        self.yv = gy.x(device, dtype)
        self.dy = gy.dx
        self.n_min = n_min
        self.kz, self.rz = gaussian_kernel(bw_z, gz.dx, device, dtype)
        self.ky, self.ry = gaussian_kernel(bw_y, gy.dx, device, dtype)
        self.H = torch.zeros((rows, gz.n, gy.n), device=device, dtype=dtype)
        self.eps_bg = eps_bg     # uniform background mixed into nu_hat(y|z).
                                 # It bounds the proposal density away from zero,
                                 # which is what makes the Metropolis-corrected
                                 # move ergodic AND lets the SAME density be used
                                 # for the draw and for the acceptance ratio.
        self._cdf = None
        self._ok = None

    def _build(self):
        Hs = torch.clamp(_smooth2d_periodic(self.H, self.kz, self.rz,
                                            self.ky, self.ry), min=0.0)
        mass = torch.trapezoid(Hs, dx=self.dy, dim=-1)
        self._ok = self.H.sum(-1) >= self.n_min
        pdf = Hs / torch.clamp(mass, min=EPS).unsqueeze(-1)
        span = self.gy.xmax - self.gy.xmin
        pdf = (1.0 - self.eps_bg) * pdf + self.eps_bg / span
        self._logpdf = torch.log(torch.clamp(pdf, min=EPS))
        cdf = torch.cumulative_trapezoid(pdf, dx=self.dy, dim=-1)
        cdf = torch.cat([torch.zeros(cdf.shape[:-1] + (1,), device=self.device,
                                     dtype=self.dtype), cdf], -1)
        self._cdf = cdf / torch.clamp(cdf[..., -1:], min=EPS)
        self._cut_mass = float(pdf[..., 0].mean() * self.dy)

    def _rows_at(self, Z):
        """Linear-in-z read of the CDF table at scattered z; (R, N, n_y)."""
        g = self.gz
        pos = torch.clamp((Z - g.xmin) / g.dx, 0.0, g.n - 1.0)
        i0 = torch.clamp(torch.floor(pos).long(), 0, g.n - 2)
        f = (pos - i0.to(Z.dtype)).unsqueeze(-1)
        c0 = torch.gather(self._cdf, 1, i0.unsqueeze(-1).expand(-1, -1, self.n_y))
        c1 = torch.gather(self._cdf, 1, (i0 + 1).unsqueeze(-1).expand(-1, -1, self.n_y))
        ok = torch.gather(self._ok, 1, i0) & torch.gather(self._ok, 1, i0 + 1)
        return c0 + f * (c1 - c0), ok

    def map(self, Z, Y, Zn):
        """y' = F^{-1}_{z'}(F_z(y)); identity wherever either end is unsupported."""
        if self._cdf is None:
            self._build()
        cdf0, ok0 = self._rows_at(Z)
        cdf1, ok1 = self._rows_at(Zn)
        ok = ok0 & ok1
        pos = torch.clamp((Y - self.gy.xmin) / self.dy, 0.0, self.n_y - 1.0)
        j0 = torch.clamp(torch.floor(pos).long(), 0, self.n_y - 2)
        fy = (pos - j0.to(Y.dtype)).unsqueeze(-1)
        lo = torch.gather(cdf0, -1, j0.unsqueeze(-1))
        hi = torch.gather(cdf0, -1, (j0 + 1).unsqueeze(-1))
        u = torch.clamp((lo + fy * (hi - lo)).squeeze(-1), 1e-9, 1 - 1e-9)
        j = torch.clamp(torch.searchsorted(cdf1.contiguous(),
                                           u.unsqueeze(-1).contiguous()),
                        1, self.n_y - 1)
        a = torch.gather(cdf1, -1, j - 1).squeeze(-1)
        b = torch.gather(cdf1, -1, j).squeeze(-1)
        t = (u - a) / torch.clamp(b - a, min=EPS)
        Yn = self.yv[(j - 1).squeeze(-1)] + t * (self.yv[j.squeeze(-1)]
                                                 - self.yv[(j - 1).squeeze(-1)])
        return torch.where(ok, Yn, Y), ok

    def sample(self, Zn, u):
        """y' ~ nu_hat(. | z') independently: a heat-bath move on the slow mode.

        The CDF MAP is the exact solution of the fiber continuity equation, but
        it is a rearrangement -- it transports a conditional that is already
        right and can never widen one that is wrong.  A cold-started ensemble is
        a delta in y, and the map keeps it a delta.  The refresh both transports
        AND spreads, at the price of discarding the walker's own y.  Which of
        the two is the better lift is exactly the question the arms answer.
        """
        if self._cdf is None:
            self._build()
        cdf1, ok = self._rows_at(Zn)
        j = torch.clamp(torch.searchsorted(cdf1.contiguous(),
                                           u.unsqueeze(-1).contiguous()),
                        1, self.n_y - 1)
        a = torch.gather(cdf1, -1, j - 1).squeeze(-1)
        b = torch.gather(cdf1, -1, j).squeeze(-1)
        t = (u - a) / torch.clamp(b - a, min=EPS)
        y0, y1 = self.yv[(j - 1).squeeze(-1)], self.yv[j.squeeze(-1)]
        return y0 + t * (y1 - y0), ok

    def log_pdf(self, Z, Y):
        """log nu_hat(y | z), linear in z and in y.  (R,N) -> (R,N)."""
        if self._cdf is None:
            self._build()
        g = self.gz
        pos = torch.clamp((Z - g.xmin) / g.dx, 0.0, g.n - 1.0)
        i0 = torch.clamp(torch.floor(pos).long(), 0, g.n - 2)
        f = (pos - i0.to(Z.dtype)).unsqueeze(-1)
        l0 = torch.gather(self._logpdf, 1, i0.unsqueeze(-1).expand(-1, -1, self.n_y))
        l1 = torch.gather(self._logpdf, 1, (i0 + 1).unsqueeze(-1).expand(-1, -1, self.n_y))
        lp = l0 + f * (l1 - l0)
        py = torch.clamp((Y - self.gy.xmin) / self.dy, 0.0, self.n_y - 1.0)
        j0 = torch.clamp(torch.floor(py).long(), 0, self.n_y - 2)
        fy = (py - j0.to(Y.dtype))
        a = torch.gather(lp, -1, j0.unsqueeze(-1)).squeeze(-1)
        b = torch.gather(lp, -1, (j0 + 1).unsqueeze(-1)).squeeze(-1)
        return a + fy * (b - a)

    def coverage(self):
        if self._cdf is None:
            self._build()
        return float(self._ok.to(self.dtype).mean())


class AdaptiveFiberCDF(_CDFTable):
    """nu_hat(y|z) accumulated from the run's own samples, with forgetting.

    The manifold phase found forgetting to be essential, not cosmetic: a plain
    running average deposits the early, self-consistently WRONG conditional
    permanently and ends up worse than no lift at all.
    """

    def __init__(self, rows, gz, gy, device, dtype, bw_z=0.25, bw_y=0.30,
                 n_min=150.0, decay=0.999, eps_bg=0.02):
        super().__init__(rows, gz, gy, device, dtype, bw_z, bw_y, n_min, eps_bg)
        self.decay = decay

    def deposit(self, Z, Y, weight=None):
        iz = torch.clamp(torch.round((Z - self.gz.xmin) / self.gz.dx).long(),
                         0, self.gz.n - 1)
        iy = torch.clamp(torch.round((Y - self.gy.xmin) / self.dy).long(),
                         0, self.n_y - 1)
        flat = iz * self.n_y + iy
        if not (isinstance(self.decay, float) and self.decay == 1.0):
            self.H.mul_(self.decay)
        w = torch.ones_like(flat, dtype=self.dtype) if weight is None else weight
        self.H.view(self.rows, -1).scatter_add_(1, flat, w)
        self._cdf = None


class ReferenceFiberCDF(_CDFTable):
    """The oracle: nu(y|z) from the high-precision unbiased-MD reference."""

    def __init__(self, rows, gz, gy, device, dtype, H2, bw_z=0.05, bw_y=0.05,
                 eps_bg=0.02):
        super().__init__(rows, gz, gy, device, dtype, bw_z, bw_y, 0.0, eps_bg)
        self.H = H2.unsqueeze(0).expand(rows, -1, -1).contiguous().to(dtype)
        self._build()

    def deposit(self, *a, **k):
        pass
