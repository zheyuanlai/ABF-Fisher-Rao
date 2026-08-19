"""One fiber-wise (conditional) reallocation event on a batched population.

Mirror of events2d.py / events1p.py for the CONDITIONAL geometry: the event sees
(xi, z) pairs, reallocates walkers WITHIN xi-strata toward a flatter p(z | xi), and
returns a gather index over the K replicas of each row.  Estimator protection is
unchanged -- the caller gathers walker arrays only, never accumulator state -- and
here it is structurally stronger: the xi-histogram is invariant at stratum
resolution, so the SHUS deposit signal is untouched by construction, not by
convention.

Three arms share this one entry point so they stay noise-paired in a single batch:

* conditional FR    -- p_hat(z | xi) from the smooth joint KDE;
* conditional count -- p_hat(z | xi) from an nb1 x nb2 histogram (stratified count
                       balancing: the discrete control that isolates the density
                       ESTIMATOR, which is the only thing FR and count have ever
                       differed by);
* stratified sham   -- matched turnover, fiber direction destroyed.

The stratification is a property of the EXPERIMENT, not of an arm: every
reallocation arm in a batch uses the same n_strata, so all of them preserve the
same xi-marginal and the only contrast left is the direction each one moves in.
"""
from __future__ import annotations

import torch

from .fisher_rao_cond import (conditional_log_ratio, conditional_log_ratio_binned,
                              stratified_sham_indices,
                              stratified_systematic_resample, stratum_of,
                              theta_backoff_cond)
from .grid2d import GridT2, binned_density2
from .resampling import turnover_counts


def fr_event_cond(z1, z2, fr_act, sham_act, cond_nb1, cond_nb2, n_strata, partner,
                  theta0, alpha_ess, k1, r1, k2, r2, grid: GridT2, gen):
    """Returns (sel, turn, theta_used, essf).  z1 = xi (biased), z2 = hidden.

    cond_nb1 / cond_nb2: (R,) long per-row histogram resolutions; 0 = the fine
    joint KDE (the FR arm).  Rows with cond_nb1 > 0 are the stratified-count
    control and share everything else with the FR rows.
    """
    R, K = z1.shape
    device, dtype = z1.device, z1.dtype
    sel = torch.arange(K, device=device).unsqueeze(0).expand(R, K).clone()
    turn = torch.zeros(R, device=device, dtype=torch.long)
    theta_used = torch.zeros(R, device=device, dtype=dtype)
    essf = torch.full((R,), float("nan"), device=device, dtype=dtype)

    strata = stratum_of(z1, grid, n_strata)
    if bool(fr_act.any()):
        p2 = binned_density2(z1, z2, k1, r1, k2, r2, grid)
        logr = conditional_log_ratio(z1, z2, p2, grid)
        nb1 = cond_nb1.to(device=device, dtype=torch.long)
        nb2 = cond_nb2.to(device=device, dtype=torch.long)
        for res in sorted({(int(a), int(b)) for a, b in
                           zip(nb1.tolist(), nb2.tolist()) if a > 0 and b > 0}):
            rows = (nb1 == res[0]) & (nb2 == res[1]) & fr_act
            if not bool(rows.any()):
                continue
            logr_c = conditional_log_ratio_binned(z1, z2, res[0], res[1], grid)
            logr = torch.where(rows.unsqueeze(1), logr_c, logr)
        theta_in = torch.where(fr_act, theta0, torch.zeros_like(theta0))
        w, cnt, th, ef = theta_backoff_cond(logr, strata, n_strata, theta_in,
                                            alpha_ess)
        sel_fr = stratified_systematic_resample(w, strata, cnt, n_strata, gen)
        turn_fr = turnover_counts(sel_fr, K)
        sel = torch.where(fr_act.unsqueeze(1), sel_fr, sel)
        turn = torch.where(fr_act, turn_fr, turn)
        theta_used = torch.where(fr_act, th, theta_used)
        essf = torch.where(fr_act, ef, essf)

    if bool(sham_act.any()):
        cnt_all = torch.zeros((R, n_strata), device=device, dtype=dtype)
        cnt_all.scatter_add_(1, strata, torch.ones_like(z1))
        m_sham = turn[partner]
        sel_sham = stratified_sham_indices(m_sham, strata, cnt_all, n_strata, gen, K)
        sel = torch.where(sham_act.unsqueeze(1), sel_sham, sel)
        turn = torch.where(sham_act, turnover_counts(sel_sham, K), turn)
    return sel, turn, theta_used, essf
