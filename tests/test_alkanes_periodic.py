"""Periodic KDE / estimator / integration validation.

Run: CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_alkanes_periodic.py -q
"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
torch.set_default_dtype(torch.float64)

from alkanes import periodic as per  # noqa: E402
from alkanes import potentials as pot  # noqa: E402

PI = math.pi


def test_meanforce_to_freeenergy_recovers_V4():
    p = pot.AlkaneParams(n_atoms=4)
    grid, dphi = per.periodic_grid(360)
    mf = pot.V4_prime(grid, p)
    F = per.free_energy_from_mean_force(mf, grid, dphi)
    V = pot.V4(grid, p)
    err = per.circular_l2(F, V, dphi).item()
    assert err < 1e-2, err


def test_kde_recovers_von_mises():
    grid, dphi = per.periodic_grid(200)
    kappa, mu = 4.0, 0.6
    # sample from von Mises via rejection-ish using torch: use built-in? emulate by
    # inverse-CDF on a fine grid.
    fine, dfine = per.periodic_grid(4000)
    dens = torch.exp(kappa * torch.cos(fine - mu))
    dens = dens / (dens.sum() * dfine)
    cdf = torch.cumsum(dens * dfine, 0)
    u = torch.rand(200000, generator=torch.Generator().manual_seed(0))
    idx = torch.searchsorted(cdf, u.clamp(0, cdf[-1] - 1e-9))
    phi = fine[idx.clamp(max=fine.numel() - 1)]
    K = per.wrapped_gaussian_kernel_matrix(grid, 0.15)
    p_hat = per.kde_marginal(phi[None, :], K, 200, dphi)[0]
    p_true = torch.exp(kappa * torch.cos(grid - mu))
    p_true = p_true / (p_true.sum() * dphi)
    assert per.marginal_l2(p_hat, p_true, dphi).item() < 0.02


def test_estimator_recovers_conditional_mean():
    # Nadaraya-Watson has O(bw^2) smoothing bias on F' (peaks attenuated); it must
    # (a) shrink with bandwidth and (b) integrate to a small error in the PMF F,
    # which is what the B0 free-energy gate actually needs.
    p = pot.AlkaneParams(n_atoms=4, beta=1.0)
    n_grid = 180
    grid, dphi = per.periodic_grid(n_grid)
    g = torch.Generator().manual_seed(1)
    phi = (torch.rand(2_000_000, generator=g) * 2 - 1) * PI
    fval = pot.V4_prime(phi, p)
    S = per.bin_sum(phi[None, :], fval[None, :], n_grid)
    C = per.bin_counts(phi[None, :], n_grid)
    errs = {}
    for bw in (0.15, 0.08, 0.05):
        K = per.wrapped_gaussian_kernel_matrix(grid, bw)
        mf = per.mean_force_profile(S, C, K)[0]
        errs[bw] = per.circular_l2(mf, pot.V4_prime(grid, p), dphi, align=False).item()
    assert errs[0.05] < errs[0.08] < errs[0.15]          # converges with bandwidth
    # reconstructed free energy is accurate at the production bandwidth
    K = per.wrapped_gaussian_kernel_matrix(grid, 0.05)
    mf = per.mean_force_profile(S, C, K)[0]
    F = per.free_energy_from_mean_force(mf, grid, dphi)
    # <0.3% of the 7.6 kT barrier: the residual is pure kernel-smoothing floor
    assert per.circular_l2(F, pot.V4(grid, p), dphi).item() < 0.03


def test_circular_interp_periodic():
    grid, dphi = per.periodic_grid(64)
    prof = torch.sin(grid)[None, :]
    q = torch.tensor([[-PI + 0.01, 0.0, PI - 0.01, 3.0]])
    got = per.circular_interp(prof, grid, q)[0]
    assert torch.allclose(got, torch.sin(q[0]), atol=2e-2)


def test_batched_consistency():
    grid, dphi = per.periodic_grid(90)
    K = per.wrapped_gaussian_kernel_matrix(grid, 0.12)
    phi = (torch.rand(4, 10000, generator=torch.Generator().manual_seed(2)) * 2 - 1) * PI
    S = per.bin_sum(phi, torch.cos(phi), 90)
    C = per.bin_counts(phi, 90)
    mf = per.mean_force_profile(S, C, K)
    assert mf.shape == (4, 90)
    # each run independently ~ E[cos|phi]=cos(phi)
    assert per.circular_l2(mf[0], torch.cos(grid), dphi, align=False).item() < 0.1


if __name__ == "__main__":
    import subprocess
    raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-q"]))
