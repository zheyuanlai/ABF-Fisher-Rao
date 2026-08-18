"""One FR/sham/count-balancing event on a batched 2D reaction-coordinate population.

2D analog of events.py: the event sees ONLY reaction-coordinate pairs (R, K, 2) plus
row masks and returns a gather index over the K replicas of each row. The weight
machinery (theta backoff, systematic resampling, matched-turnover sham) is shared
with 1D — it is dimension-agnostic.

Count balancing generalizes to an nb x nb histogram over equal periodic cells: the
piecewise-constant density replaces the fine periodic KDE. In 2D this control
becomes statistically sparse (K / nb^2 walkers per cell), which is exactly the
regime the geometry question (Q3) probes.
"""
from __future__ import annotations

import math

import torch

from .grid import EPS
from .grid2d import GridT2, binned_density2, uniform_log_ratio2
from .fisher_rao import theta_backoff
from .resampling import (matched_turnover_indices, systematic_resample,
                         turnover_counts)


def fr_event2(z1, z2, fr_act, sham_act, is_coarse_row, coarse_nb, partner, theta0,
              alpha_ess, k1, r1, k2, r2, grid: GridT2, gen):
    """Returns (sel, turn, theta_used, essf).  z1, z2: (R, K) torus coordinates.

    coarse_nb: an int (uniform resolution for the masked rows, the 1D-style
    convention) or an (R,) long tensor of per-row resolutions (0 = fine KDE) —
    the resolution study runs 6x6 / 9x9 / 12x12 arms in ONE paired batch.
    """
    R, K = z1.shape
    device, dtype = z1.device, z1.dtype
    if torch.is_tensor(coarse_nb):
        nb_row = coarse_nb.to(device=device, dtype=torch.long)
    else:
        nb_row = torch.where(is_coarse_row,
                             torch.full((R,), int(coarse_nb), device=device,
                                        dtype=torch.long),
                             torch.zeros(R, device=device, dtype=torch.long))
    sel = torch.arange(K, device=device).unsqueeze(0).expand(R, K).clone()
    turn = torch.zeros(R, device=device, dtype=torch.long)
    theta_used = torch.zeros(R, device=device, dtype=dtype)
    essf = torch.full((R,), float("nan"), device=device, dtype=dtype)
    if bool(fr_act.any()):
        p_hat = binned_density2(z1, z2, k1, r1, k2, r2, grid)
        logr = uniform_log_ratio2(z1, z2, p_hat, grid)
        for nb in sorted(set(nb_row[nb_row > 0].tolist())):
            rows = (nb_row == nb) & fr_act
            if not bool(rows.any()):
                continue
            bw1 = grid.L1 / nb
            bw2 = grid.L2 / nb
            b1 = torch.remainder(((z1 - grid.x1min) / bw1).long(), nb)
            b2 = torch.remainder(((z2 - grid.x2min) / bw2).long(), nb)
            bidx = b1 * nb + b2
            cnt = torch.zeros((R, nb * nb), device=device, dtype=dtype)
            cnt.scatter_add_(1, bidx, torch.ones_like(z1))
            p_coarse = torch.clamp(cnt / (K * bw1 * bw2), min=EPS)
            logr_c = (-math.log(grid.volume)
                      - torch.log(torch.gather(p_coarse, 1, bidx)))
            logr = torch.where(rows.unsqueeze(1), logr_c, logr)
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
