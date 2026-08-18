"""Finite Fisher-Rao step toward the uniform SHUS target (batched).

For a frozen target u, the pure Fisher-Rao flow has the exact finite-time solution

    p^+  propto  p^{1-theta} u^theta,        theta = 1 - exp(-gamma * tau_FR),

realized on particles by the unnormalized resampling weights

    a_k = [ u(xi_k) / p_hat(xi_k) ]^theta,

followed by exact-K systematic resampling (resampling.py).  The target here is ALWAYS
the uniform density on the reaction-coordinate domain (frozen project decision: the
"bias-aware" target is only a lagged difference of two estimates of the same F and
carries no independent information).

Degeneracy control: theta is halved per row until ESS_FR >= alpha_ess * K.  theta = 0
gives exactly uniform weights, under which systematic resampling is the identity, so a
row that cannot satisfy the floor skips its event rather than firing a degenerate one.
"""
from __future__ import annotations

import math

import torch

from .grid import EPS, Grid1D, interp1d, trapz


def kl_to_uniform(p, grid: Grid1D):
    """D_t = KL(p_hat || u) via trapezoid quadrature.  p: (R, G) -> (R,)."""
    u = 1.0 / grid.volume
    integrand = p * (torch.log(torch.clamp(p, min=EPS)) - math.log(u))
    return trapz(integrand, grid.dx)


def tv_to_uniform(p, grid: Grid1D):
    """TV(p_hat, u) = 0.5 * integral |p - u|.  p: (R, G) -> (R,)."""
    u = 1.0 / grid.volume
    return 0.5 * trapz((p - u).abs(), grid.dx)


def uniform_log_ratio(X, p_grid, grid: Grid1D):
    """log(u / p_hat) at walker positions.  X: (R, N), p_grid: (R, G) -> (R, N)."""
    u_log = math.log(1.0 / grid.volume)
    p_at = torch.clamp(interp1d(X, p_grid, grid), min=EPS)
    return u_log - torch.log(p_at)


def fr_weights(log_ratio, theta):
    """Normalized weights w propto exp(theta * log_ratio) and their ESS fraction.

    log_ratio: (R, N); theta: (R,) -> w: (R, N), ess_frac: (R,).
    """
    a = theta.unsqueeze(1) * log_ratio
    a = a - a.max(dim=1, keepdim=True).values
    w = torch.exp(a)
    w = w / w.sum(dim=1, keepdim=True)
    ess_frac = 1.0 / (w.pow(2).sum(dim=1) * w.shape[1])
    return w, ess_frac


def theta_backoff(log_ratio, theta0, alpha_ess, max_halvings=30):
    """Reduce theta per row until ESS_FR >= alpha_ess * K; give up at theta -> 0.

    Returns (w, theta_used, ess_frac).  Rows whose theta reaches ~0 without meeting the
    floor get theta_used = 0 exactly (uniform weights: the event is a no-op).
    """
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
