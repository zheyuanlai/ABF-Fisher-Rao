"""One FR/sham/count-balancing event on a periodic-1D reaction coordinate.

Mirror of events2d.py for xi on a circle (alanine xi = phi): the event sees only
(R, K) angles plus row masks and returns a gather index; the weight machinery is
the shared dimension-agnostic layer. Count balancing uses nb equal periodic arcs.
"""
from __future__ import annotations

import math

import torch

from .grid import EPS
from .grid1p import Grid1P, binned_density1p, uniform_log_ratio1p
from .fisher_rao import theta_backoff
from .resampling import (matched_turnover_indices, systematic_resample,
                         turnover_counts)


def fr_event1p(z, fr_act, sham_act, is_coarse_row, coarse_nb, partner, theta0,
               alpha_ess, kernel, krad, grid: Grid1P, gen):
    """Returns (sel, turn, theta_used, essf).  z: (R, K) periodic coordinates.

    coarse_nb: int (uniform for masked rows) or (R,) long tensor (0 = fine KDE).
    """
    R, K = z.shape
    device, dtype = z.device, z.dtype
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
        p_hat = binned_density1p(z, kernel, krad, grid)
        logr = uniform_log_ratio1p(z, p_hat, grid)
        for nb in sorted(set(nb_row[nb_row > 0].tolist())):
            rows = (nb_row == nb) & fr_act
            if not bool(rows.any()):
                continue
            bw = grid.L / nb
            bidx = torch.remainder(((z - grid.xmin) / bw).long(), nb)
            cnt = torch.zeros((R, nb), device=device, dtype=dtype)
            cnt.scatter_add_(1, bidx, torch.ones_like(z))
            p_coarse = torch.clamp(cnt / (K * bw), min=EPS)
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
