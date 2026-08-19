"""Fiber-wise (conditional) Fisher-Rao step: birth-death INSIDE xi-slices.

Why this module exists (docs/PREREGISTRATION_APPLICATION_MAP.md, Phase F).  The
marginal step of fisher_rao.py drives p_hat(xi) toward the uniform density -- which
is what the ABP's own bias already does, and what count balancing does with a
coarser estimate of the same log-density.  Four head-to-head replications found FR
and count indistinguishable; that is the correct answer, not bad luck: both realize
the SAME continuous flow d_t p = -p (log p/u - KL) and differ only in the estimator
of log p(xi).

The one thing an ABP biasing xi structurally cannot flatten is the CONDITIONAL
p(z | xi) of a coordinate z it does not bias.  This module puts the Fisher-Rao flow
there instead:

    p^+(z | xi)  propto  p(z | xi)^{1-theta} u(z)^theta,      p(xi) LEFT ALONE,

realized on particles by the weights a_k = [u(z_k) / p_hat(z_k | xi_k)]^theta
followed by systematic resampling *within* xi-strata, so every stratum keeps exactly
the walker count it had.

Two consequences the marginal step does not have:

* the xi-histogram the SHUS accumulator learns from is invariant at stratum
  resolution, so the estimator-resampling feedback this project has fought since
  Stage 0 cannot enter through the deposit signal;
* the adaptation gain g_SHUS -- the arm that matched or beat marginal FR on every
  system in this campaign -- rescales the xi-bias and by construction does nothing
  to a barrier in z, so for the first time it is not a competing explanation for
  whatever the population step buys.

Target convention (frozen, transferred from the marginal step): u is UNIFORM on the
z-domain.  It is NOT the stationary conditional (nothing biases z), so the step is
only ever run inside a temporary window, and the realized fiber populations are
recorded at every save so overshoot past the Boltzmann conditional is visible rather
than inferred.

Degeneracy control mirrors fisher_rao.py: theta is halved per row until the ESS of
the induced selection law clears alpha_ess * K; theta = 0 makes the event a no-op
(equal within-stratum weights give the identity selection, tested).
"""
from __future__ import annotations

import math

import torch

from .grid import EPS
from .grid1p import Grid1P, interp1p
from .grid2d import GridT2, interp2


# -----------------------------------------------------------------------------
# the conditional score
# -----------------------------------------------------------------------------
def axis1_grid(grid: GridT2) -> Grid1P:
    """The xi-axis of a torus grid as a periodic 1D grid (for marginal interp)."""
    return Grid1P(xmin=grid.x1min, L=grid.L1, n=grid.n1)


def conditional_log_ratio(z1, z2, p2_hat, grid: GridT2):
    """log[ u(z) / p_hat(z | xi) ] at walker positions.  -> (R, K).

    p_hat(z | xi) = p_hat(xi, z) / p_hat(xi), BOTH read off the same smooth joint
    KDE: the conditional inherits the smooth density estimate that is the only
    place FR can differ from a histogram control, and the two factors share their
    kernel so the ratio is exact on the grid (no independent normalization drift).
    """
    p1 = p2_hat.sum(dim=2) * grid.dx2                     # (R, n1)
    at2 = torch.clamp(interp2(z1, z2, p2_hat, grid), min=EPS)
    at1 = torch.clamp(interp1p(z1, p1, axis1_grid(grid)), min=EPS)
    return -math.log(grid.L2) - torch.log(at2) + torch.log(at1)


def conditional_log_ratio_binned(z1, z2, nb1, nb2, grid: GridT2):
    """Histogram control: the same score from an nb1 x nb2 piecewise-constant joint.

    The discrete analog of the conditional step -- stratified count balancing.  It
    is the control that isolates the DENSITY ESTIMATOR (the only thing FR and count
    have ever differed by) inside the conditional geometry.
    """
    R, K = z1.shape
    bw1, bw2 = grid.L1 / nb1, grid.L2 / nb2
    b1 = torch.remainder(((z1 - grid.x1min) / bw1).long(), nb1)
    b2 = torch.remainder(((z2 - grid.x2min) / bw2).long(), nb2)
    flat = b1 * nb2 + b2
    cell = torch.zeros((R, nb1 * nb2), device=z1.device, dtype=z1.dtype)
    cell.scatter_add_(1, flat, torch.ones_like(z1))
    col = cell.reshape(R, nb1, nb2).sum(dim=2)            # walkers per xi-bin
    n_at = torch.clamp(torch.gather(cell, 1, flat), min=1.0)
    d_at = torch.clamp(torch.gather(col, 1, b1), min=1.0)
    # p_hat(z|xi) = (cell / col) / bw2
    return -math.log(grid.L2) - torch.log(n_at / d_at / bw2)


# -----------------------------------------------------------------------------
# stratification and the within-stratum weight law
# -----------------------------------------------------------------------------
def stratum_of(z1, grid: GridT2, n_strata: int):
    """Equal-width periodic xi-strata.  z1: (R, K) -> (R, K) long in [0, n_strata)."""
    bw = grid.L1 / n_strata
    return torch.remainder(((z1 - grid.x1min) / bw).long(), n_strata)


def stratified_weights(log_ratio, strata, n_strata, theta):
    """Within-stratum normalized weights and the ESS of the induced selection law.

    Returns (w, cnt, ess_frac): w (R, K) sums to 1 inside each stratum, cnt (R, S)
    is the stratum occupancy, and ess_frac is 1 / (K * sum_k q_k^2) where
    q_k = w_k * cnt_{j(k)} / K is the probability that a uniformly chosen output
    slot descends from walker k -- the exact analog of the marginal ESS, so the
    same alpha_ess floor means the same thing in both geometries.
    """
    R, K = log_ratio.shape
    dev, dt_ = log_ratio.device, log_ratio.dtype
    a = theta.unsqueeze(1) * log_ratio
    smax = torch.full((R, n_strata), -float("inf"), device=dev, dtype=dt_)
    smax = smax.scatter_reduce(1, strata, a, reduce="amax", include_self=True)
    a = a - torch.gather(smax, 1, strata)
    e = torch.exp(a)
    ssum = torch.zeros((R, n_strata), device=dev, dtype=dt_)
    ssum.scatter_add_(1, strata, e)
    w = e / torch.clamp(torch.gather(ssum, 1, strata), min=EPS)
    cnt = torch.zeros((R, n_strata), device=dev, dtype=dt_)
    cnt.scatter_add_(1, strata, torch.ones_like(e))
    q = w * torch.gather(cnt, 1, strata) / K
    ess_frac = 1.0 / torch.clamp(q.pow(2).sum(dim=1) * K, min=EPS)
    return w, cnt, ess_frac


def theta_backoff_cond(log_ratio, strata, n_strata, theta0, alpha_ess,
                       max_halvings=30):
    """Halve theta per row until ESS_FR >= alpha_ess * K; give up at theta -> 0."""
    theta = theta0.clone()
    for _ in range(max_halvings):
        w, cnt, essf = stratified_weights(log_ratio, strata, n_strata, theta)
        bad = (essf < alpha_ess) & (theta > 0)
        if not bool(bad.any()):
            return w, cnt, theta, essf
        theta = torch.where(bad, theta * 0.5, theta)
    w, cnt, essf = stratified_weights(log_ratio, strata, n_strata, theta)
    theta = torch.where(essf < alpha_ess, torch.zeros_like(theta), theta)
    w, cnt, essf = stratified_weights(log_ratio, strata, n_strata, theta)
    return w, cnt, theta, essf


# -----------------------------------------------------------------------------
# exact-count-per-stratum systematic resampling
# -----------------------------------------------------------------------------
def stratified_systematic_resample(w, strata, cnt, n_strata, gen):
    """Systematic resampling inside each xi-stratum.  -> sel: (R, K) long.

    Every stratum keeps EXACTLY its own walker count, so the xi-histogram at
    stratum resolution -- and with it the occupancy signal SHUS deposits from -- is
    invariant under the event.  (Mass can still move within a stratum; the design
    keeps the stratum width at or below the SHUS mollifier bandwidth so that
    residual is under the estimator's own resolution.)

    Implementation: sort walkers by stratum, form a globally monotone key
    C = j + (within-stratum cumulative weight) so stratum j owns the interval
    (j, j+1], and query it at j + (l + U_j)/cnt_j for local index l with one U per
    (row, stratum).  A single searchsorted then does every stratum at once.
    Equal within-stratum weights reproduce the identity selection exactly, so a
    theta = 0 event is a no-op -- the same guarantee the marginal step has.
    """
    R, K = w.shape
    dev, dt_ = w.device, w.dtype
    order = torch.argsort(strata, dim=1, stable=True)
    s_sorted = torch.gather(strata, 1, order)
    w_sorted = torch.gather(w, 1, order)

    c = torch.cumsum(w_sorted, dim=1)
    off_l = (torch.cumsum(cnt, dim=1) - cnt).long()            # (R, S) exclusive
    gathered = torch.gather(c, 1, torch.clamp(off_l - 1, min=0))
    c_start = torch.where(off_l > 0, gathered, torch.zeros_like(gathered))
    c_start_w = torch.gather(c_start, 1, s_sorted)
    key = s_sorted.to(dt_) + torch.clamp(c - c_start_w, min=0.0, max=1.0)

    local = torch.arange(K, device=dev).unsqueeze(0) - torch.gather(off_l, 1, s_sorted)
    u = torch.rand((R, n_strata), device=dev, dtype=dt_, generator=gen)
    cnt_w = torch.clamp(torch.gather(cnt, 1, s_sorted), min=1.0)
    query = s_sorted.to(dt_) + (local.to(dt_) + torch.gather(u, 1, s_sorted)) / cnt_w

    pos = torch.searchsorted(key.contiguous(), query.contiguous())
    lo = torch.gather(off_l, 1, s_sorted)
    hi = lo + cnt_w.long() - 1
    pos = torch.maximum(torch.minimum(pos, hi), lo)
    parent = torch.gather(order, 1, pos)
    sel = torch.empty_like(parent)
    sel.scatter_(1, order, parent)
    return sel


# -----------------------------------------------------------------------------
# stratified sham: matched turnover with the fiber direction destroyed
# -----------------------------------------------------------------------------
def stratified_sham_indices(m, strata, cnt, n_strata, gen, K):
    """Kill m walkers at random; each dead slot is refilled from a SURVIVOR OF ITS
    OWN STRATUM, chosen uniformly.  -> sel: (R, K) long.

    The null must differ from the conditional step in exactly one thing: the
    direction.  A marginal sham would also destroy the stratum-count invariance and
    confound two effects, so the sham is stratified too.  Strata whose members all
    died in the kill draw keep their slots (no parent available); realized turnover
    is therefore reported, never assumed equal to m.
    """
    R = m.shape[0]
    dev, dt_ = strata.device, cnt.dtype
    rk = torch.rand((R, K), device=dev, dtype=dt_, generator=gen)
    rank = rk.argsort(dim=1).argsort(dim=1)
    die = rank < m.unsqueeze(1)

    # group as [stratum 0 survivors (random order), stratum 0 dead, stratum 1 ...]
    shuffle = torch.rand((R, K), device=dev, dtype=dt_, generator=gen).argsort(dim=1)
    s_sh = torch.gather(strata, 1, shuffle)
    d_sh = torch.gather(die, 1, shuffle)
    key = s_sh * 2 + d_sh.long()
    grp = torch.argsort(key, dim=1, stable=True)
    order = torch.gather(shuffle, 1, grp)                  # original indices, grouped
    s_ord = torch.gather(s_sh, 1, grp)

    n_surv = torch.zeros((R, n_strata), device=dev, dtype=dt_)
    n_surv.scatter_add_(1, strata, (~die).to(dt_))
    off_l = (torch.cumsum(cnt, dim=1) - cnt).long()
    surv_lo = torch.gather(off_l, 1, strata)               # survivors start at off_j
    ns = torch.gather(n_surv, 1, strata)                   # ... and there are n_surv_j

    u = torch.rand((R, K), device=dev, dtype=dt_, generator=gen)
    pick = torch.clamp(surv_lo + (u * ns).long(), max=K - 1)
    parent = torch.gather(order, 1, pick)
    ar = torch.arange(K, device=dev).unsqueeze(0).expand(R, K)
    keep = (~die) | (ns < 1.0)                             # no survivor in the fiber
    return torch.where(keep, ar, parent)
