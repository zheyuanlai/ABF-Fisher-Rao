"""Stage-0 FR validation: the finite step realizes p^+ ~ p^{1-theta} u^theta,
limiting cases theta=0 and theta->1, and the ESS backoff."""
import math

import numpy as np
import torch

from conftest import DEVICE, DTYPE
from abpfr.fisher_rao import (fr_weights, kl_to_uniform, theta_backoff,
                              tv_to_uniform, uniform_log_ratio)
from abpfr.grid import Grid1D, binned_density, gaussian_kernel, trapz
from abpfr.resampling import systematic_resample

G = Grid1D(xmin=-1.8, xmax=1.8, n=181, eval_lo=-1.5, eval_hi=1.5)


def sample_from_grid_density(dens, n, seed):
    """Inverse-CDF sampling of a grid density (with uniform jitter inside cells)."""
    d = dens / dens.sum()
    cdf = torch.cumsum(d, 0)
    gen = torch.Generator().manual_seed(seed)
    u = torch.rand(n, generator=gen, dtype=DTYPE)
    idx = torch.searchsorted(cdf, u).clamp(max=G.n - 1)
    xg = G.x(DEVICE, DTYPE)
    jit = (torch.rand(n, generator=gen, dtype=DTYPE) - 0.5) * G.dx
    return (xg[idx] + jit).clamp(G.xmin, G.xmax).unsqueeze(0)


def bimodal_density():
    xg = G.x(DEVICE, DTYPE)
    return (torch.exp(-0.5 * ((xg + 1.0) / 0.25) ** 2)
            + 0.3 * torch.exp(-0.5 * ((xg - 1.0) / 0.25) ** 2))


def geometric_mean_target(p_grid, theta):
    """Analytic p^{1-theta} u^theta, trapz-normalized on the grid."""
    u = 1.0 / G.volume
    q = p_grid.clamp(min=1e-300) ** (1.0 - theta) * (u ** theta)
    return q / trapz(q.unsqueeze(0), G.dx)[0]


def exact_log_ratio(X, p_grid):
    """log(u/p) using the EXACT density, isolating the FR law from KDE error."""
    from abpfr.grid import interp1d
    u = 1.0 / G.volume
    p_at = interp1d(X, p_grid.unsqueeze(0), G).clamp(min=1e-300)
    return math.log(u) - torch.log(p_at)


def weighted_moments(X, w, powers=(1, 2)):
    return [float((w * X ** k).sum()) for k in powers]


def target_moments(q_grid, powers=(1, 2)):
    xg = G.x(DEVICE, DTYPE)
    return [float(trapz((q_grid * xg ** k).unsqueeze(0), G.dx)) for k in powers]


def test_finite_step_matches_power_interpolation():
    p_grid = bimodal_density()
    p_norm = p_grid / trapz(p_grid.unsqueeze(0), G.dx)[0]
    X = sample_from_grid_density(p_grid, 400_000, seed=11)
    for theta in (0.1, 0.3, 0.6):
        logr = exact_log_ratio(X, p_norm)
        w, _ = fr_weights(logr, torch.tensor([theta], dtype=DTYPE))
        q = geometric_mean_target(p_norm, theta)
        mw = weighted_moments(X, w)
        mq = target_moments(q)
        assert abs(mw[0] - mq[0]) < 0.01, f"theta={theta}: mean {mw[0]} vs {mq[0]}"
        assert abs(mw[1] - mq[1]) < 0.01, f"theta={theta}: m2 {mw[1]} vs {mq[1]}"


def test_resampled_population_matches_target():
    p_grid = bimodal_density()
    p_norm = p_grid / trapz(p_grid.unsqueeze(0), G.dx)[0]
    theta = 0.5
    X = sample_from_grid_density(p_grid, 200_000, seed=12)
    logr = exact_log_ratio(X, p_norm)
    w, _ = fr_weights(logr, torch.tensor([theta], dtype=DTYPE))
    gen = torch.Generator().manual_seed(13)
    sel = systematic_resample(w, gen)
    Xr = torch.gather(X, 1, sel)
    assert Xr.shape == X.shape                      # exact population conservation
    k, r = gaussian_kernel(0.10, G.dx, DEVICE, DTYPE)
    p_after = binned_density(Xr, k, r, G)[0]
    q = geometric_mean_target(p_norm, theta)
    tv = 0.5 * float(trapz((p_after - q).abs().unsqueeze(0), G.dx))
    assert tv < 0.05, f"TV(resampled, p^(1-theta) u^theta) = {tv:.3f}"


def test_theta_zero_is_identity():
    X = sample_from_grid_density(bimodal_density(), 5000, seed=14)
    logr = exact_log_ratio(X, bimodal_density())
    w, _ = fr_weights(logr, torch.tensor([0.0], dtype=DTYPE))
    assert torch.allclose(w, torch.full_like(w, 1.0 / w.shape[1]))
    gen = torch.Generator().manual_seed(15)
    sel = systematic_resample(w, gen)
    assert torch.equal(sel[0], torch.arange(w.shape[1]))


def test_theta_near_one_flattens():
    p_grid = bimodal_density()
    p_norm = p_grid / trapz(p_grid.unsqueeze(0), G.dx)[0]
    X = sample_from_grid_density(p_grid, 200_000, seed=16)
    logr = exact_log_ratio(X, p_norm)
    w, _ = fr_weights(logr, torch.tensor([0.999], dtype=DTYPE))
    gen = torch.Generator().manual_seed(17)
    Xr = torch.gather(X, 1, systematic_resample(w, gen))
    k, r = gaussian_kernel(0.10, G.dx, DEVICE, DTYPE)
    p_after = binned_density(Xr, k, r, G)
    assert float(tv_to_uniform(p_after, G)) < 0.05


def test_kl_and_tv_to_uniform():
    u = torch.full((1, G.n), 1.0 / G.volume, dtype=DTYPE)
    assert abs(float(kl_to_uniform(u, G))) < 1e-12
    assert abs(float(tv_to_uniform(u, G))) < 1e-12
    p = bimodal_density().unsqueeze(0)
    p = p / trapz(p, G.dx).unsqueeze(1)
    assert float(kl_to_uniform(p, G)) > 0.1


def test_theta_backoff_enforces_ess_floor():
    # one walker sits alone in a huge under-represented hole -> degenerate weights
    N = 1000
    logr = torch.zeros((1, N), dtype=DTYPE)
    logr[0, 0] = 40.0
    theta0 = torch.tensor([0.8], dtype=DTYPE)
    alpha = torch.tensor([0.5], dtype=DTYPE)
    w, theta, essf = theta_backoff(logr, theta0, alpha)
    assert float(essf) >= 0.5 or float(theta) == 0.0
    assert float(theta) < 0.8                        # it had to back off
    assert abs(float(w.sum()) - 1.0) < 1e-12


def test_kde_score_pipeline_runs_on_particles():
    X = sample_from_grid_density(bimodal_density(), 2048, seed=18)
    k, r = gaussian_kernel(0.10, G.dx, DEVICE, DTYPE)
    p_hat = binned_density(X, k, r, G)
    logr = uniform_log_ratio(X, p_hat, G)
    assert bool(torch.isfinite(logr).all())
    # over-represented walkers (left peak) must carry log-ratio < 0
    left = X < -0.5
    assert float(logr[left].mean()) < 0.0
