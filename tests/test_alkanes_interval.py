"""Non-periodic interval estimator (distance CV): reflected KDE, mean-force->PMF,
interpolation, windowed L2.

Run: CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_alkanes_interval.py -q
"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
torch.set_default_dtype(torch.float64)

from alkanes import interval as iv  # noqa: E402


def test_reflected_kde_recovers_truncated_gaussian():
    lo, hi, n = 1.0, 4.0, 300
    grid, dz = iv.interval_grid(n, lo, hi)
    Kr = iv.reflected_kernel_matrix(grid, 0.08, lo, hi)
    mu, sig = 2.3, 0.5
    # sample from N(mu,sig) truncated to [lo,hi] by rejection
    gen = torch.Generator().manual_seed(0)
    xs = []
    while sum(x.numel() for x in xs) < 300000:
        z = mu + sig * torch.randn(200000, generator=gen)
        xs.append(z[(z > lo) & (z < hi)])
    x = torch.cat(xs)[None, :300000]
    p = iv.kde_marginal(x, Kr, n, dz, lo, hi)[0]
    pt = torch.exp(-0.5 * ((grid - mu) / sig) ** 2)
    pt = pt / (pt.sum() * dz)
    # interior L2 small (reflection controls boundary bias)
    err = math.sqrt(float(((p - pt) ** 2).sum() * dz / (hi - lo)))
    assert err < 0.03, err
    assert abs(float(p.sum() * dz) - 1.0) < 1e-9


def test_mean_force_reconstructs_harmonic_potential():
    # z ~ exp(-U), U=0.5 k (z-z0)^2 truncated to [lo,hi]; F'=U'(z)=k(z-z0);
    # Nadaraya-Watson mean force integrated back recovers U in the bulk window.
    lo, hi, n = 1.0, 4.0, 256
    grid, dz = iv.interval_grid(n, lo, hi)
    k, z0 = 8.0, 2.4
    gen = torch.Generator().manual_seed(1)
    xs = []
    while sum(x.numel() for x in xs) < 3_000_000:
        z = z0 + (1.0 / math.sqrt(k)) * torch.randn(500000, generator=gen)
        xs.append(z[(z > lo) & (z < hi)])
    x = torch.cat(xs)[None, :3_000_000]
    fval = k * (x - z0)                    # U'(z)
    S = iv.bin_sum(x, fval, n, lo, hi)
    C = iv.bin_counts(x, n, lo, hi)
    K = iv.gaussian_kernel_matrix(grid, 0.05)
    mf = iv.mean_force_profile(S, C, K)
    F = iv.free_energy_from_mean_force(mf, grid, dz)
    U = 0.5 * k * (grid - z0) ** 2
    # thermally relevant window: within ~3 kT of the min
    mask = ((U - U.min()) <= 3.0)[None]
    err = iv.interval_l2(F[None], (U - U.mean())[None], dz, lo, hi, mask=mask).item()
    assert err < 0.05, err


def test_interp_edge_clamped_linear():
    lo, hi, n = 0.0, 2.0, 64
    grid, dz = iv.interval_grid(n, lo, hi)
    prof = (grid ** 2)[None]
    x = torch.tensor([[lo, 1.0, hi, hi + 5.0, lo - 5.0]])
    got = iv.interval_interp(prof, grid, x)[0]
    # interior ~ x^2; out-of-range clamps to edge values
    assert abs(got[1].item() - 1.0) < 2e-2
    assert got[3] == prof[0, -1] or abs(got[3] - prof[0, -1]) < 1e-9   # clamp hi
    assert abs(got[4] - prof[0, 0]) < 0.2                              # clamp lo edge


def test_windowed_l2_masks_correctly():
    lo, hi, n = 0.0, 1.0, 100
    grid, dz = iv.interval_grid(n, lo, hi)
    a = torch.zeros(1, n); b = torch.zeros(1, n)
    b[0, :50] = 1.0                     # discrepancy only on the left half
    mask_right = (grid > 0.5)[None]
    err_right = iv.interval_l2(a, b, dz, lo, hi, mask=mask_right, align=False).item()
    assert err_right < 1e-9             # masked-out region contributes nothing


if __name__ == "__main__":
    import subprocess
    raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-q"]))
