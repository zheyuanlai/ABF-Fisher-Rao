"""Discovery / establishment diagnostics and the FR eligibility gate (Stage 1A).

The gate is preregistered BEFORE any FR run exists:

    T_hit : first persistent time every preregistered region holds >= 1 walker,
    T_est : first persistent time D_t = KL(p_hat_t || u) <= D_tol,
    D_tol : 1.5 x the 95th percentile of the finite-K KDE noise floor, estimated by
            drawing K i.i.d. uniform samples and running the IDENTICAL KDE.

Eligible for SHUS+FR when discovery is early and establishment is slow
(T_hit/T small, T_est/T large); the exact cutoffs are frozen in the preregistration
document, not here.
"""
from __future__ import annotations

import numpy as np
import torch

from .fisher_rao import kl_to_uniform
from .grid import Grid1D, binned_density, gaussian_kernel


def first_persistent(cond, times, hold_frac=0.05):
    """First time at which cond holds over a whole trailing window (ported convention:
    one walker brushing a region for a single save is not a discovery)."""
    n = len(times)
    hold = max(1, int(hold_frac * n))
    c = np.asarray(cond, dtype=bool)
    for i in range(n - hold + 1):
        if c[i:i + hold].all():
            return float(times[i])
    return float("nan")


def kde_noise_floor(K, eta_bw, grid: Grid1D, n_rep=256, seed=777, device="cpu",
                    dtype=torch.float64):
    """Finite-K KL noise floor: KL(KDE of K uniform samples || u), n_rep replicates.

    Returns the sorted array of D values; callers take quantiles (the preregistration
    uses the 95th).  Uses the same kernel/normalization as the production marginal.
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    k, r = gaussian_kernel(eta_bw, grid.dx, device, dtype)
    X = grid.xmin + (grid.xmax - grid.xmin) * torch.rand(
        (n_rep, K), device=device, dtype=dtype, generator=gen)
    p = binned_density(X, k, r, grid)
    D = kl_to_uniform(p, grid)
    return np.sort(D.detach().cpu().numpy())


def hit_time(region_occupancy, times, hold_frac=0.05):
    """T_hit for one region: first persistent time its walker fraction is > 0."""
    return first_persistent(np.asarray(region_occupancy) > 0.0, times, hold_frac)


def establishment_time(D_t, times, D_tol, hold_frac=0.1):
    """T_est: first persistent time the marginal KL sits at the noise floor."""
    return first_persistent(np.asarray(D_t) <= D_tol, times, hold_frac)
