"""Reference conditional  p(y_S | z)  for a promoted SUBSET S of the fiber modes.

The one-dimensional CDF map of the manifold phase does not generalise: in two
or more fiber dimensions there is no canonical monotone rearrangement, and
solving the continuity equation there is a genuine transport problem.  What
does generalise is the conditional REFRESH -- draw y_S from p(y_S | z') and let
constrained dynamics relax everything else.  This class supplies that draw for
|S| = 1 and 2 from the unbiased-MD reference joint histogram.

Marginalising over the un-promoted fiber modes is deliberate and is the thing
under test: promoting phi2 and leaving phi3 alone means drawing phi2 from
p(phi2 | z) with no regard for the phi3 the walker happens to carry.
"""
from __future__ import annotations

import math

import torch

from ..grid import EPS


def _inv_cdf(cdf, u):
    """Inverse of a BIN cdf (cdf[k] = sum_{i<=k} p_i).

    Returns (continuous position in bin units from the domain's left edge,
    integer bin index).  Bin 0 has to be handled explicitly -- clamping the
    searchsorted index up to 1 makes `cdf[j-1]` the wrong lower edge there and
    silently mis-samples the first bin, which is exactly what
    `test_joint_refresh_reproduces_its_table` caught.
    """
    n = cdf.shape[-1]
    j = torch.clamp(torch.searchsorted(cdf.contiguous(), u.unsqueeze(-1).contiguous()),
                    0, n - 1)
    lo = torch.where(j > 0, torch.gather(cdf, -1, torch.clamp(j - 1, min=0)),
                     torch.zeros_like(u.unsqueeze(-1))).squeeze(-1)
    hi = torch.gather(cdf, -1, j).squeeze(-1)
    jj = j.squeeze(-1)
    t = (u - lo) / torch.clamp(hi - lo, min=EPS)
    return jj.to(u.dtype) + torch.clamp(t, 0.0, 1.0), jj


class JointRefresh:
    """Draw y_S ~ p_ref(. | z) from a raw joint count table.

    H : (nz, n1) or (nz, n1, n2), counts on uniform periodic bin grids over
        [-pi, pi).  Bin CENTRES are used as the sampled values, with linear
        interpolation inside a bin so the draw is continuous.
    """

    def __init__(self, H, device, dtype, smooth=1, eps_bg=0.02):
        H = torch.as_tensor(H, device=device, dtype=dtype)
        if smooth > 1:
            H = _circ_box(H, smooth)
        self.eps_bg = eps_bg
        self.k = H.dim() - 1
        self.nz = H.shape[0]
        self.n = H.shape[1:]
        c = lambda n: (torch.arange(n, device=device, dtype=dtype) + 0.5) \
            * (2 * math.pi / n) - math.pi
        self.centres = [c(n) for n in self.n]
        self.dw = [2 * math.pi / n for n in self.n]
        mix = lambda p, n: (1.0 - eps_bg) * p + eps_bg / n
        if self.k == 1:
            p = mix(H / torch.clamp(H.sum(-1, keepdim=True), min=EPS), self.n[0])
            self.p1, self.cdf1 = p, _cum(p)
        else:
            m1 = H.sum(-1)
            p1 = mix(m1 / torch.clamp(m1.sum(-1, keepdim=True), min=EPS), self.n[0])
            self.p1, self.cdf1 = p1, _cum(p1)
            p2 = mix(H / torch.clamp(H.sum(-1, keepdim=True), min=EPS), self.n[1])
            self.p2, self.cdf2 = p2, _cum(p2)

    def _iz(self, Z):
        return torch.clamp(((Z + math.pi) / (2 * math.pi) * self.nz).long(),
                           0, self.nz - 1)

    def _iy(self, Y, d):
        return torch.clamp(((Y + math.pi) / (2 * math.pi) * self.n[d]).long(),
                           0, self.n[d] - 1)

    def log_pdf(self, Z, Y):
        """log p_ref(y_S | z) as a DENSITY in y (bin pmf / bin width).

        Same object the sampler draws from, background included, so it is the
        proposal density the Metropolis ratio needs.
        """
        iz = self._iz(Z)
        j1 = self._iy(Y[..., 0], 0)
        lp = _log_pmf_at(self.p1[iz], j1) - math.log(self.dw[0])
        if self.k == 1:
            return lp
        j2 = self._iy(Y[..., 1], 1)
        return lp + _log_pmf_at(self.p2[iz, j1], j2) - math.log(self.dw[1])

    def sample(self, Z, U):
        """Z: (...); U: (..., k) uniforms -> y: (..., k)."""
        iz = self._iz(Z)
        c1 = self.cdf1[iz]
        pos, j1 = _inv_cdf(c1, U[..., 0])
        y1 = -math.pi + pos * self.dw[0]
        if self.k == 1:
            return y1.unsqueeze(-1)
        c2 = self.cdf2[iz, torch.clamp(j1, 0, self.n[0] - 1)]
        pos2, _ = _inv_cdf(c2, U[..., 1])
        y2 = -math.pi + pos2 * self.dw[1]
        return torch.stack([y1, y2], -1)


def _log_pmf_at(p, idx):
    return torch.log(torch.clamp(torch.gather(p, -1, idx.unsqueeze(-1)).squeeze(-1),
                                 min=EPS))


def _cum(p):
    c = torch.cumsum(p, dim=-1)
    return c / torch.clamp(c[..., -1:], min=EPS)


def _circ_box(H, w):
    """Circular box smoothing on every axis; keeps zero-count corners from
    producing a degenerate inverse CDF."""
    for d in range(H.dim()):
        n = H.shape[d]
        out = torch.zeros_like(H)
        for s in range(-(w // 2), w // 2 + 1):
            out = out + torch.roll(H, s, dims=d)
        H = out / (2 * (w // 2) + 1)
    return H
