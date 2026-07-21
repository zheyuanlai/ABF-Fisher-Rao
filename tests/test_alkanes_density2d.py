"""2-D torus density / smoothing / interpolation / Fisher--Rao score validation.

Run: CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_alkanes_density2d.py -q
"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
torch.set_default_dtype(torch.float64)

from alkanes import density2d as d2  # noqa: E402

PI = math.pi


def test_kde2_recovers_product_von_mises():
    n = 64
    g1, g2, dz1, dz2 = d2.torus_grid(n, n)
    K1, K2 = d2.kernels(g1, g2, 0.15, 0.15)
    kappa, mu1, mu2 = 4.0, 0.5, -0.8
    # sample product von Mises by inverse-CDF per axis
    fine, dfine = d2.per.periodic_grid(4000)
    def sample(mu, M):
        dens = torch.exp(kappa * torch.cos(fine - mu)); dens /= dens.sum() * dfine
        cdf = torch.cumsum(dens * dfine, 0)
        u = torch.rand(M, generator=torch.Generator().manual_seed(int(mu * 100) + 7))
        return fine[torch.searchsorted(cdf, u.clamp(0, cdf[-1] - 1e-9)).clamp(max=fine.numel() - 1)]
    M = 400000
    phi1 = sample(mu1, M)[None, :]; phi2 = sample(mu2, M)[None, :]
    p = d2.kde2(phi1, phi2, K1, K2, n, n, dz1, dz2)[0]
    Z1, Z2 = torch.meshgrid(g1, g2, indexing="ij")
    pt = torch.exp(kappa * torch.cos(Z1 - mu1)) * torch.exp(kappa * torch.cos(Z2 - mu2))
    pt = pt / (pt.sum() * dz1 * dz2)
    assert d2.l2_2d(p[None], pt[None], dz1, dz2, align=False).item() < 0.02
    assert abs((p.sum() * dz1 * dz2).item() - 1.0) < 1e-9


def test_bilinear_interp_of_known_field():
    n1, n2 = 90, 72
    g1, g2, dz1, dz2 = d2.torus_grid(n1, n2)
    Z1, Z2 = torch.meshgrid(g1, g2, indexing="ij")
    prof = (torch.cos(Z1) * torch.cos(Z2))[None]
    gen = torch.Generator().manual_seed(2)
    p1 = (torch.rand(1, 500, generator=gen) * 2 - 1) * (PI - 0.05)
    p2 = (torch.rand(1, 500, generator=gen) * 2 - 1) * (PI - 0.05)
    got = d2.bilinear_interp2(prof, g1, g2, dz1, dz2, p1, p2)[0]
    true = torch.cos(p1[0]) * torch.cos(p2[0])
    assert (got - true).abs().max() < 5e-3


def test_smooth2_separable_matches_double_1d():
    from alkanes import periodic as per
    n = 32
    g1, g2, dz1, dz2 = d2.torus_grid(n, n)
    K1, K2 = d2.kernels(g1, g2, 0.2, 0.3)
    field = torch.randn(3, n, n, generator=torch.Generator().manual_seed(1))
    out = d2.smooth2(field, K1, K2)
    # reference: smooth axis1 then axis2 with plain matmuls
    tmp = torch.einsum("ab,rbc->rac", K1, field)
    ref = torch.einsum("rac,dc->rad", tmp, K2)
    assert (out - ref).abs().max() < 1e-10


def test_fr_score_is_zero_mean_and_bounded():
    n = 48
    g1, g2, dz1, dz2 = d2.torus_grid(n, n)
    K1, K2 = d2.kernels(g1, g2, 0.3, 0.3)
    gen = torch.Generator().manual_seed(4)
    R, M = 3, 2000
    phi1 = (torch.rand(R, M, generator=gen) * 2 - 1) * PI
    phi2 = (torch.rand(R, M, generator=gen) * 2 - 1) * PI
    p = d2.kde2(phi1, phi2, K1, K2, n, n, dz1, dz2)
    q = torch.ones(R, n, n) / (2 * PI) ** 2
    score, kl = d2.fr_score_2d(phi1, phi2, p, q, g1, g2, dz1, dz2, clip=2.0)
    assert score.shape == (R, M)
    assert score.mean(-1).abs().max() < 1e-9         # zero-mean per replica (balance)
    assert score.abs().max() <= 2.0 + 0.5
    assert torch.all(kl >= -1e-9)


def test_mean_force_fields_unsupported_cells_zero():
    n = 24
    g1, g2, dz1, dz2 = d2.torus_grid(n, n)
    K1, K2 = d2.kernels(g1, g2, 0.05, 0.05)   # narrow => empty cells stay empty
    phi1 = torch.tensor([[0.0, 0.0]]); phi2 = torch.tensor([[0.0, 0.1]])
    f1s = d2.scatter_sum(phi1, phi2, torch.ones_like(phi1), n, n, dz1, dz2)
    f2s = d2.scatter_sum(phi1, phi2, torch.ones_like(phi1), n, n, dz1, dz2)
    c = d2.scatter_counts(phi1, phi2, n, n, dz1, dz2)
    a1, a2, den = d2.mean_force_fields(f1s, f2s, c, K1, K2)
    far = (g1 - PI / 2).abs().argmin()               # a cell far from the two samples
    assert a1[0, far, far].abs() < 1e-9              # unsupported => 0


if __name__ == "__main__":
    import subprocess
    raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-q"]))
