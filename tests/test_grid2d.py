"""2D periodic grid layer validation: torus topology, mollifier, KDE, interpolation,
derivatives, and the FR score — the seam (-pi == pi) must never act as a boundary."""
import math

import numpy as np
import torch

from conftest import DEVICE, DTYPE
from abpfr import grid2d as g2
from abpfr.grid2d import GridT2

PI = math.pi
G = GridT2(x1min=-PI, L1=2 * PI, n1=72, x2min=-PI, L2=2 * PI, n2=72)


def kernels(bw=0.25):
    k1, r1 = g2.periodic_gaussian_kernel(bw, G.dx1, G.n1, DEVICE, DTYPE)
    k2, r2 = g2.periodic_gaussian_kernel(bw, G.dx2, G.n2, DEVICE, DTYPE)
    return k1, r1, k2, r2


def test_wrap_and_torus_distance():
    x = torch.tensor([PI + 0.1, -PI - 0.1, 0.5], dtype=DTYPE)
    xw = g2.wrap_periodic(x, -PI, 2 * PI)
    assert torch.allclose(xw, torch.tensor([-PI + 0.1, PI - 0.1, 0.5], dtype=DTYPE))
    d = g2.torus_distance(torch.tensor([PI - 0.05], dtype=DTYPE),
                          torch.tensor([-PI + 0.05], dtype=DTYPE), 2 * PI)
    assert abs(float(d) - 0.1) < 1e-12       # across the seam, NOT ~2 pi


def test_kernel_normalization_and_fit():
    k1, r1, _, _ = kernels()
    assert abs(float(k1.sum()) * G.dx1 - 1.0) < 1e-12
    assert 2 * r1 < G.n1
    import pytest
    with pytest.raises(AssertionError):      # kernel too wide for the axis
        g2.periodic_gaussian_kernel(3.0, G.dx1, G.n1, DEVICE, DTYPE)


def test_smooth2_mass_conservation_and_shift_equivariance():
    k1, r1, k2, r2 = kernels()
    v = torch.zeros((1, G.n1, G.n2), dtype=DTYPE)
    v[0, 0, 0] = 1.0                          # delta at the corner (on the seam)
    s = g2.smooth2(v, k1, r1, k2, r2)
    # kernels are 1/(sum*dx)-normalized, so sum(k) = 1/dx per axis and smoothing a
    # counts field multiplies its total by 1/(dx1*dx2): counts -> counts per area
    assert abs(float(s.sum()) - float(v.sum()) / (G.dx1 * G.dx2)) < 1e-9
    # periodic shift equivariance: smoothing commutes with torus rolls
    v2 = torch.roll(v, shifts=(13, 27), dims=(1, 2))
    s2 = g2.smooth2(v2, k1, r1, k2, r2)
    assert torch.allclose(torch.roll(s, (13, 27), dims=(1, 2)), s2, atol=1e-12)


def test_binned_density2_normalized_and_seamless():
    k1, r1, k2, r2 = kernels()
    gen = torch.Generator().manual_seed(0)
    X1 = g2.wrap_periodic(0.3 * torch.randn((2, 4000), generator=gen, dtype=DTYPE),
                          -PI, 2 * PI)
    X2 = g2.wrap_periodic(0.3 * torch.randn((2, 4000), generator=gen, dtype=DTYPE),
                          -PI, 2 * PI)
    p = g2.binned_density2(X1, X2, k1, r1, k2, r2, G)
    assert torch.allclose(g2.integral2(p, G), torch.ones(2, dtype=DTYPE), atol=1e-9)
    # a population at +pi and one at -pi give the SAME density (same torus point)
    Xa = torch.full((1, 500), PI - 1e-9, dtype=DTYPE)
    Xb = torch.full((1, 500), -PI, dtype=DTYPE)
    Y = torch.zeros((1, 500), dtype=DTYPE)
    pa = g2.binned_density2(Xa, Y, k1, r1, k2, r2, G)
    pb = g2.binned_density2(Xb, Y, k1, r1, k2, r2, G)
    assert torch.allclose(pa, pb, atol=1e-9)


def test_interp2_accuracy_and_seam_continuity():
    P1, P2 = G.mesh(DEVICE, DTYPE)
    vals = (torch.cos(P1) * torch.cos(P2) + 0.3 * torch.sin(P2)).unsqueeze(0)
    gen = torch.Generator().manual_seed(1)
    X1 = g2.wrap_periodic((2 * PI) * torch.rand((1, 3000), generator=gen,
                                                dtype=DTYPE) - PI, -PI, 2 * PI)
    X2 = g2.wrap_periodic((2 * PI) * torch.rand((1, 3000), generator=gen,
                                                dtype=DTYPE) - PI, -PI, 2 * PI)
    est = g2.interp2(X1, X2, vals, G)
    ref = torch.cos(X1) * torch.cos(X2) + 0.3 * torch.sin(X2)
    assert float((est - ref).abs().max()) < 5e-3      # O(dx^2) bilinear error
    # continuity across the seam
    eps = 1e-7
    Y = torch.linspace(-2.0, 2.0, 20, dtype=DTYPE).unsqueeze(0)
    left = g2.interp2(torch.full_like(Y, PI - eps), Y, vals, G)
    right = g2.interp2(torch.full_like(Y, -PI + eps), Y, vals, G)
    assert float((left - right).abs().max()) < 1e-5


def test_central_diff2_periodic():
    P1, P2 = G.mesh(DEVICE, DTYPE)
    F = (torch.cos(P1) + torch.sin(P2)).unsqueeze(0)
    d1, d2 = g2.central_diff2(F, G)
    assert float((d1 - (-torch.sin(P1))).abs().max()) < 2e-3
    assert float((d2 - torch.cos(P2)).abs().max()) < 2e-3


def test_kl_tv_zero_for_uniform():
    p = torch.full((1, G.n1, G.n2), 1.0 / G.volume, dtype=DTYPE)
    assert abs(float(g2.kl_to_uniform2(p, G))) < 1e-12
    assert abs(float(g2.tv_to_uniform2(p, G))) < 1e-12


def test_uniform_log_ratio2_sign():
    k1, r1, k2, r2 = kernels()
    gen = torch.Generator().manual_seed(2)
    # crowded near (0,0): score must be negative there, positive in empty regions
    X1 = g2.wrap_periodic(0.2 * torch.randn((1, 5000), generator=gen, dtype=DTYPE),
                          -PI, 2 * PI)
    X2 = g2.wrap_periodic(0.2 * torch.randn((1, 5000), generator=gen, dtype=DTYPE),
                          -PI, 2 * PI)
    p = g2.binned_density2(X1, X2, k1, r1, k2, r2, G)
    at_origin = g2.uniform_log_ratio2(torch.zeros((1, 1), dtype=DTYPE),
                                      torch.zeros((1, 1), dtype=DTYPE), p, G)
    far = g2.uniform_log_ratio2(torch.full((1, 1), PI, dtype=DTYPE),
                                torch.full((1, 1), PI, dtype=DTYPE), p, G)
    assert float(at_origin) < 0 < float(far)
