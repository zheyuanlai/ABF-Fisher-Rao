"""Fisher-Rao (birth-death) population reallocation toward the uniform target.

The pure FR flow for a frozen target u has the exact finite-time solution

    p^+  propto  p^{1-theta} u^theta,     theta = 1 - exp(-lambda * dtau),

realized on particles by unnormalized weights a_i = [u(Z_i)/p_hat(Z_i)]^theta
followed by exact-N systematic resampling.

Three selection rules share the interface so they are exchangeable controls:

  'fr'     smooth Fisher-Rao        a_i = (u / p_kde(Z_i))^theta
  'count'  count balancing          a_i = (u / p_hist(Z_i))^theta  (raw bin counts)
  'sham'   matched-turnover churn   same number of kills/clones, random direction

Selection depends on Z ONLY.  Because the RC-WFR free-energy estimator is a
CONDITIONAL average at fixed z, a Z-measurable selection cannot bias it: given Z,
the fiber configuration Q is untouched.  (This is the structural difference from
the ABF/ABP+FR campaign, where FR reweighted the marginal the estimator read.)
"""
from __future__ import annotations

import math

import torch

from .grid import (EPS, Grid1D, binned_density, gaussian_kernel, interp1d,
                   nearest_bin, scatter_counts, trapz)
from .resampling import matched_turnover_indices, systematic_resample, turnover_counts


def kl_to_uniform(p, grid: Grid1D):
    u = 1.0 / grid.volume
    integrand = p * (torch.log(torch.clamp(p, min=EPS)) - math.log(u))
    return trapz(integrand, grid.dx)


def tv_to_uniform(p, grid: Grid1D):
    u = 1.0 / grid.volume
    return 0.5 * trapz((p - u).abs(), grid.dx)


def kde_marginal(X, grid: Grid1D, bw: float):
    k, r = gaussian_kernel(bw, grid.dx, X.device, X.dtype)
    return binned_density(X, k, r, grid)


def log_ratio_kde(X, grid: Grid1D, bw: float):
    """log(u / p_kde) at walker positions.  (R,N) -> (R,N)."""
    p = kde_marginal(X, grid, bw)
    p_at = torch.clamp(interp1d(X, p, grid), min=EPS)
    return math.log(1.0 / grid.volume) - torch.log(p_at)


def log_ratio_counts(X, grid: Grid1D, n_bins: int):
    """log(u / p_hist) at walker positions using RAW bin counts (no kernel).

    p_hist(bin) = n_bin / (N * width).  Empty bins cannot host a walker, so the
    clamp only guards numerically.
    """
    R, N = X.shape
    width = grid.volume / n_bins
    pos = (X - grid.xmin) / width
    idx = torch.clamp(torch.floor(pos).long(), 0, n_bins - 1)
    counts = torch.zeros((R, n_bins), device=X.device, dtype=X.dtype)
    counts.scatter_add_(1, idx, torch.ones_like(X))
    p_bin = counts / (float(N) * width)
    p_at = torch.clamp(torch.gather(p_bin, 1, idx), min=EPS)
    return math.log(1.0 / grid.volume) - torch.log(p_at)


def fr_weights(log_ratio, theta):
    a = theta.unsqueeze(1) * log_ratio
    a = a - a.max(dim=1, keepdim=True).values
    w = torch.exp(a)
    w = w / w.sum(dim=1, keepdim=True)
    ess_frac = 1.0 / (w.pow(2).sum(dim=1) * w.shape[1])
    return w, ess_frac


def theta_backoff(log_ratio, theta0, alpha_ess, max_halvings=30):
    """Halve theta per row until ESS_FR >= alpha_ess * N; theta -> 0 makes it a no-op."""
    theta = theta0.clone()
    for _ in range(max_halvings):
        w, essf = fr_weights(log_ratio, theta)
        bad = (essf < alpha_ess) & (theta > 0)
        if not bool(bad.any()):
            return w, theta, essf
        theta = torch.where(bad, theta * 0.5, theta)
    theta = torch.where(fr_weights(log_ratio, theta)[1] < alpha_ess,
                        torch.zeros_like(theta), theta)
    w, essf = fr_weights(log_ratio, theta)
    return w, theta, essf


def selection_indices(X, grid: Grid1D, rule: str, theta0, gen, *, bw=None,
                      n_bins=None, alpha_ess=0.5, sham_turnover=None):
    """Return (sel, info) where new_state = old_state[r, sel[r]].

    rule: 'fr' | 'count' | 'sham' | 'none'.
    """
    R, N = X.shape
    if rule == "none":
        ar = torch.arange(N, device=X.device).unsqueeze(0).expand(R, N)
        return ar, {"theta": torch.zeros(R, device=X.device, dtype=X.dtype),
                    "ess_frac": torch.ones(R, device=X.device, dtype=X.dtype),
                    "turnover": torch.zeros(R, device=X.device, dtype=torch.long)}
    if rule == "sham":
        assert sham_turnover is not None, "sham needs a partner turnover series"
        sel = matched_turnover_indices(sham_turnover, N, gen, X.device, X.dtype)
        return sel, {"theta": torch.zeros(R, device=X.device, dtype=X.dtype),
                     "ess_frac": torch.ones(R, device=X.device, dtype=X.dtype),
                     "turnover": sham_turnover}
    if rule == "fr":
        lr = log_ratio_kde(X, grid, bw)
    elif rule == "count":
        lr = log_ratio_counts(X, grid, n_bins)
    else:
        raise ValueError(f"unknown selection rule {rule}")
    w, theta, essf = theta_backoff(lr, theta0, alpha_ess)
    sel = systematic_resample(w, gen)
    return sel, {"theta": theta, "ess_frac": essf, "turnover": turnover_counts(sel, N)}
