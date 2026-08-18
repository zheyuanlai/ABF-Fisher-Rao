import numpy as np
import torch

from conftest import DEVICE, DTYPE
from abpfr.grid import (Grid1D, binned_density, central_diff, cumtrapz,
                        gaussian_kernel, interp1d, reflect_into, smooth, trapz)

G = Grid1D(xmin=-1.8, xmax=1.8, n=181, eval_lo=-1.5, eval_hi=1.5)


def test_kernel_normalization():
    k, r = gaussian_kernel(0.1, G.dx, DEVICE, DTYPE)
    assert abs(float(k.sum()) * G.dx - 1.0) < 1e-12
    assert k.numel() == 2 * r + 1


def test_smooth_of_constant_is_uniform_density():
    # kernel is 1/(sum*dx)-normalized, so smoothing a constant c gives c/dx everywhere
    k, r = gaussian_kernel(0.1, G.dx, DEVICE, DTYPE)
    v = torch.full((1, G.n), 3.0, device=DEVICE, dtype=DTYPE)
    out = smooth(v, k, r, G.dx)
    assert torch.allclose(out, torch.full_like(out, 3.0 / G.dx), rtol=1e-10)


def test_binned_density_normalized_and_positive():
    gen = torch.Generator().manual_seed(0)
    X = -1.0 + 0.1 * torch.randn((3, 5000), generator=gen, dtype=DTYPE)
    X = reflect_into(X, G.xmin, G.xmax)
    k, r = gaussian_kernel(0.1, G.dx, DEVICE, DTYPE)
    p = binned_density(X, k, r, G)
    mass = trapz(p, G.dx)
    assert torch.allclose(mass, torch.ones_like(mass), atol=1e-8)
    assert bool((p > 0).all())


def test_interp1d_matches_numpy():
    xg = G.x(DEVICE, DTYPE)
    vals = torch.sin(xg).unsqueeze(0)
    gen = torch.Generator().manual_seed(1)
    X = (G.xmin + (G.xmax - G.xmin) * torch.rand((1, 1000), generator=gen, dtype=DTYPE))
    mine = interp1d(X, vals, G)
    ref = np.interp(X.numpy()[0], xg.numpy(), vals.numpy()[0])
    assert np.allclose(mine.numpy()[0], ref, atol=1e-12)


def test_central_diff_exact_for_quadratic():
    xg = G.x(DEVICE, DTYPE)
    F = (2.0 * xg * xg + xg).unsqueeze(0)
    Fp = central_diff(F, G.dx)
    expect = (4.0 * xg + 1.0).unsqueeze(0)
    assert torch.allclose(Fp[:, 1:-1], expect[:, 1:-1], atol=1e-10)


def test_cumtrapz_and_trapz():
    xg = G.x(DEVICE, DTYPE)
    y = xg.unsqueeze(0)
    ct = cumtrapz(y, G.dx)
    expect = 0.5 * (xg ** 2 - G.xmin ** 2)
    assert torch.allclose(ct[0], expect, atol=1e-10)
    tot = trapz(y, G.dx)
    assert abs(float(tot)) < 1e-10  # odd function on symmetric domain


def test_reflect_into_bounds_and_identity():
    q = torch.tensor([[0.3, 1.9, -2.0, 5.0, -1.8]], dtype=DTYPE)
    out = reflect_into(q, G.xmin, G.xmax)
    assert bool((out >= G.xmin).all()) and bool((out <= G.xmax).all())
    assert abs(float(out[0, 0]) - 0.3) < 1e-12          # interior point untouched
    assert abs(float(out[0, 1]) - 1.7) < 1e-12          # 1.9 reflects to 1.7


def test_cpu_gpu_component_equivalence():
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("no CUDA device")
    xg = G.x(DEVICE, DTYPE)
    v = torch.sin(3 * xg).unsqueeze(0).abs() + 0.1
    k, r = gaussian_kernel(0.1, G.dx, DEVICE, DTYPE)
    out_cpu = smooth(v, k, r, G.dx)
    out_gpu = smooth(v.cuda(), k.cuda(), r, G.dx).cpu()
    assert torch.allclose(out_cpu, out_gpu, atol=1e-12)
    X = torch.linspace(-1.7, 1.7, 500, dtype=DTYPE).unsqueeze(0)
    a = interp1d(X, v, G)
    b = interp1d(X.cuda(), v.cuda(), G).cpu()
    assert torch.allclose(a, b, atol=1e-12)
