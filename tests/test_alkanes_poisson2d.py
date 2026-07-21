"""FFT Poisson projection on the torus: exact-gradient recovery, curl removal, zero mode,
grid convergence, CPU/GPU parity.

Run: CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_alkanes_poisson2d.py -q
"""
import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
torch.set_default_dtype(torch.float64)

from alkanes import density2d as d2  # noqa: E402
from alkanes import poisson2d as ps  # noqa: E402

PI = math.pi


def _grids(n1, n2, device="cpu"):
    g1, g2, dz1, dz2 = d2.torus_grid(n1, n2, device=device)
    Z1, Z2 = torch.meshgrid(g1, g2, indexing="ij")
    return g1, g2, dz1, dz2, Z1, Z2


def test_recovers_known_gradient_field():
    _, _, dz1, dz2, Z1, Z2 = _grids(48, 48)
    B0 = torch.cos(Z1) + 0.5 * torch.sin(2 * Z2) + 0.3 * torch.cos(Z1 + Z2)
    B0 = B0 - B0.mean()
    g1, g2 = ps.spectral_gradient(B0[None], dz1, dz2)     # exact conservative field
    B, gB1, gB2 = ps.poisson_projection(g1, g2, dz1, dz2)
    assert (B[0] - B0).abs().max() < 1e-9
    assert (gB1[0] - g1[0]).abs().max() < 1e-9
    assert (gB2[0] - g2[0]).abs().max() < 1e-9
    assert B[0].mean().abs() < 1e-12                      # zero mode controlled


def test_removes_divergence_free_noise():
    # pure curl field g = (d psi/d z2, -d psi/d z1) is divergence-free => B ~ 0
    _, _, dz1, dz2, Z1, Z2 = _grids(48, 48)
    psi = torch.sin(Z1) * torch.cos(2 * Z2) + 0.4 * torch.cos(3 * Z1 - Z2)
    p1, p2 = ps.spectral_gradient(psi[None], dz1, dz2)
    g1, g2 = p2, -p1                                      # rot(psi): divergence-free
    B, gB1, gB2 = ps.poisson_projection(g1, g2, dz1, dz2)
    assert B.abs().max() < 1e-9
    assert gB1.abs().max() < 1e-9 and gB2.abs().max() < 1e-9


def test_hodge_split_general_field():
    # g = grad B0 + curl part; projection returns grad B0, residual is the curl part.
    _, _, dz1, dz2, Z1, Z2 = _grids(64, 64)
    B0 = torch.sin(Z1 + 0.5) + 0.6 * torch.cos(2 * Z2)
    B0 = B0 - B0.mean()
    c1, c2 = ps.spectral_gradient(B0[None], dz1, dz2)
    psi = torch.cos(Z1) * torch.sin(Z2)
    p1, p2 = ps.spectral_gradient(psi[None], dz1, dz2)
    g1, g2 = c1 + p2, c2 - p1
    B, gB1, gB2 = ps.poisson_projection(g1, g2, dz1, dz2)
    assert (B[0] - B0).abs().max() < 1e-9
    resid1, resid2 = g1 - gB1, g2 - gB2
    assert (resid1 - p2).abs().max() < 1e-9 and (resid2 + p1).abs().max() < 1e-9
    assert ps.curl_norm(gB1, gB2, dz1, dz2).item() < 1e-9   # projected part is curl-free


def test_grid_convergence():
    # a smooth field is recovered ever more accurately as the grid refines
    errs = {}
    for n in (24, 48, 96):
        _, _, dz1, dz2, Z1, Z2 = _grids(n, n)
        B0 = torch.exp(torch.cos(Z1)) * torch.cos(Z2)
        B0 = B0 - B0.mean()
        g1, g2 = ps.spectral_gradient(B0[None], dz1, dz2)
        B, _, _ = ps.poisson_projection(g1, g2, dz1, dz2)
        errs[n] = (B[0] - B0).abs().max().item()
    assert errs[96] <= errs[48] <= errs[24] + 1e-12


def test_batched_and_cpu_gpu_parity():
    _, _, dz1, dz2, Z1, Z2 = _grids(32, 40)      # non-square grid
    B0 = torch.stack([torch.cos(Z1) + torch.sin(Z2),
                      torch.cos(2 * Z1 - Z2)], dim=0)
    B0 = B0 - B0.mean(dim=(-2, -1), keepdim=True)
    g1, g2 = ps.spectral_gradient(B0, dz1, dz2)
    B, _, _ = ps.poisson_projection(g1, g2, dz1, dz2)
    assert (B - B0).abs().max() < 1e-9
    if torch.cuda.is_available():
        Bg, _, _ = ps.poisson_projection(g1.cuda(), g2.cuda(), dz1, dz2)
        assert (B - Bg.cpu()).abs().max() < 1e-9


if __name__ == "__main__":
    import subprocess
    raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-q"]))
