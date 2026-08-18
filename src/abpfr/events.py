"""One FR/sham/count-balancing event on a batched reaction-coordinate population.

Shared by every system (gateway, WCA): the event sees ONLY reaction-coordinate
values (R, K) plus row masks, and returns a gather index over the K replicas of
each row. What a "replica" is (a 2D point, a 100-particle box) is the caller's
business — which is exactly the marginal-FR structure the theory branch studies.
"""
from __future__ import annotations

import math

import torch

from .fisher_rao import theta_backoff, uniform_log_ratio
from .grid import EPS, Grid1D, binned_density
from .resampling import (matched_turnover_indices, systematic_resample,
                         turnover_counts)


def fr_event(z, fr_act, sham_act, is_coarse_row, coarse_nb, partner, theta0,
             alpha_ess, k_eta, r_eta, grid: Grid1D, gen):
    """Returns (sel, turn, theta_used, essf).

    z: (R, K) reaction-coordinate values; fr_act/sham_act: (R,) bool masks of rows
    firing this event; sel: (R, K) gather index (identity for inactive rows).
    theta_used/essf are nan/0 outside fr_act rows.
    """
    R, K = z.shape
    device, dtype = z.device, z.dtype
    sel = torch.arange(K, device=device).unsqueeze(0).expand(R, K).clone()
    turn = torch.zeros(R, device=device, dtype=torch.long)
    theta_used = torch.zeros(R, device=device, dtype=dtype)
    essf = torch.full((R,), float("nan"), device=device, dtype=dtype)
    if bool(fr_act.any()):
        p_hat = binned_density(z, k_eta, r_eta, grid)
        logr = uniform_log_ratio(z, p_hat, grid)
        if coarse_nb > 0 and bool((fr_act & is_coarse_row).any()):
            # count-balancing control: piecewise-constant histogram density over
            # coarse_nb equal bins replaces the fine KDE
            bw = (grid.xmax - grid.xmin) / coarse_nb
            bidx = torch.clamp(((z - grid.xmin) / bw).long(), 0, coarse_nb - 1)
            cnt = torch.zeros((R, coarse_nb), device=device, dtype=dtype)
            cnt.scatter_add_(1, bidx, torch.ones_like(z))
            p_coarse = torch.clamp(cnt / (K * bw), min=EPS)
            logr_c = (-math.log(grid.volume)
                      - torch.log(torch.gather(p_coarse, 1, bidx)))
            logr = torch.where(is_coarse_row.unsqueeze(1), logr_c, logr)
        w, th, ef = theta_backoff(logr, theta0, alpha_ess)
        sel_fr = systematic_resample(w, gen)
        turn_fr = turnover_counts(sel_fr, K)
        sel = torch.where(fr_act.unsqueeze(1), sel_fr, sel)
        turn = torch.where(fr_act, turn_fr, turn)
        theta_used = torch.where(fr_act, th, theta_used)
        essf = torch.where(fr_act, ef, essf)
    if bool(sham_act.any()):
        m_sham = turn[partner]
        sel_sham = matched_turnover_indices(m_sham, K, gen, device, dtype)
        sel = torch.where(sham_act.unsqueeze(1), sel_sham, sel)
        turn = torch.where(sham_act, m_sham, turn)
    return sel, turn, theta_used, essf
