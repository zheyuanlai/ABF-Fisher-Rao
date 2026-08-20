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


def conditional_log_ratio(z1, z2, p2_hat, grid: GridT2, log_q=None):
    """log[ q(z | xi) / p_hat(z | xi) ] at walker positions.  -> (R, K).

    log_q: None for the frozen UNIFORM target, or an (R, n1, n2) grid of
    log q(z | xi).  "Uniform" is not a canonical notion for a general descriptor --
    it moves under reparametrization of z -- so the target is a first-class argument
    and Phase F4 varies it deliberately.

    p_hat(z | xi) = p_hat(xi, z) / p_hat(xi), BOTH read off the same smooth joint
    KDE: the conditional inherits the smooth density estimate that is the only
    place FR can differ from a histogram control, and the two factors share their
    kernel so the ratio is exact on the grid (no independent normalization drift).
    """
    p1 = p2_hat.sum(dim=2) * grid.dx2                     # (R, n1)
    at2 = torch.clamp(interp2(z1, z2, p2_hat, grid), min=EPS)
    at1 = torch.clamp(interp1p(z1, p1, axis1_grid(grid)), min=EPS)
    log_at_q = (-math.log(grid.L2) if log_q is None
                else interp2(z1, z2, log_q, grid))
    return log_at_q - torch.log(at2) + torch.log(at1)


def conditional_log_ratio_binned(z1, z2, nb1, nb2, grid: GridT2, log_q=None):
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
    log_at_q = (-math.log(grid.L2) if log_q is None
                else interp2(z1, z2, log_q, grid))
    # p_hat(z|xi) = (cell / col) / bw2
    return log_at_q - torch.log(n_at / d_at / bw2)


def conditional_log_ratio_state(state, strata, n_states, n_strata, log_q=None):
    """Score on a DISCRETE hidden descriptor: log[ q(s) / p_hat(s | stratum) ].

    No kernel, no bins, no density estimate at all -- just the frequency of each
    hidden STATE inside each xi-stratum.  This is the classical stratified-allocation
    rule, and it is the baseline the KDE and histogram scores have to beat once the
    step is measure-preserving: with within-stratum weights a_s = (q_s/p_s)^theta the
    realized allocation is

        n_s  proportional to  p_s^{1-theta} q_s^theta,

    so with a uniform q over states, theta = 0 is PROPORTIONAL allocation (a no-op),
    theta = 1/2 is the square-root compromise allocation, and theta = 1 is EQUAL
    COUNT PER STATE.  Neyman's optimal allocation n_s ~ p_s sigma_s is this family
    only when sigma_s is constant; the per-state deposit spread is recorded as a
    diagnostic rather than fed back, because using it would make the allocation
    depend on the estimand it is being scored on.
    """
    R, K = state.shape
    dev, dt_ = strata.device, torch.float64 if state.dtype == torch.long else state.dtype
    flat = strata * n_states + state
    ones = torch.ones((R, K), device=dev, dtype=dt_)
    cell = torch.zeros((R, n_strata * n_states), device=dev, dtype=dt_)
    cell.scatter_add_(1, flat, ones)
    col = cell.reshape(R, n_strata, n_states).sum(dim=2)
    n_at = torch.clamp(torch.gather(cell, 1, flat), min=1.0)
    d_at = torch.clamp(torch.gather(col, 1, strata), min=1.0)
    log_at_q = (-math.log(n_states) if log_q is None
                else torch.gather(log_q, 1, state))
    return log_at_q - torch.log(n_at / d_at)


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
# weighted selection: allocation decoupled from represented probability
# -----------------------------------------------------------------------------
def stratum_sum(v, strata, n_strata):
    """Per-stratum sums of a per-walker quantity.  v: (R, K) -> (R, S)."""
    out = torch.zeros((v.shape[0], n_strata), device=v.device, dtype=v.dtype)
    out.scatter_add_(1, strata, v)
    return out


def child_weights(W, sel, strata, n_strata, w=None, cnt=None):
    """Statistical weights of the resampled slots.  W: (R, K) -> (R, K).

    WHY (Phase I).  In the equal-weight step of F2/F4 the selection IS the answer:
    reallocating toward a target q makes the ensemble represent something closer to
    q, so a wrong q is a wrong physical distribution -- which is exactly what F4
    measured when one arbitrary reparametrization of the descriptor turned a -15%
    gain into a +5% loss.  Weighted selection separates the two jobs: the score
    decides WHERE COMPUTATIONAL EFFORT GOES, and the weights keep WHAT THE ENSEMBLE
    REPRESENTS fixed.  The selection index `sel` is unchanged -- the weighted arm
    fires the identical event as its equal-weight partner, dose-matched by
    construction -- and only the weights it carries afterwards differ.

    Two rules, one for each kind of selection in this codebase:

    * `w`/`cnt` given (SCORE-DRIVEN arms).  The slots of stratum j are drawn with
      within-stratum probabilities w_k, so the importance weight of a child of k is
      W_k / (cnt_j w_k): summed over the cnt_j slots this returns sum_k W_k in
      expectation, for ANY score.  Note the child weight is inversely proportional
      to the desirability a_k -- allocate more copies, carry less weight each.
    * `w` None (the SHAM).  Its kill is uniform and position-independent, so there
      is no proposal density to divide by; each parent's weight is split equally
      among its realized children, which conserves it exactly (the split rule
      sum_{children(i)} W_j = W_i).

    Both are then renormalized so each stratum's total weight is EXACTLY what it was.
    That is the weighted form of the invariant the whole conditional design rests on
    (F2: stratum COUNTS are preserved, so the SHUS deposit signal cannot be
    perturbed).  With weights, counts no longer carry the marginal -- weight does --
    so it is the stratum weight that must be held fixed, and the renormalization
    also removes the O(1/cnt_j) resampling fluctuation of the importance rule.
    """
    parent_div = (torch.zeros_like(W).scatter_add_(1, sel, torch.ones_like(W))
                  if w is None else w * torch.gather(cnt, 1, strata))
    Wc = (torch.gather(W, 1, sel)
          / torch.clamp(torch.gather(parent_div, 1, sel), min=EPS))
    # children of stratum j land in slots of stratum j, so strata labels are unmoved
    scale = (stratum_sum(W, strata, n_strata)
             / torch.clamp(stratum_sum(Wc, strata, n_strata), min=EPS))
    return Wc * torch.gather(scale, 1, strata)


def weight_ess(W):
    """ESS of the statistical weights as a fraction of K: (sum W)^2 / (K sum W^2).

    The price of weighted selection, and the reason it is reported at every save
    rather than guarded: allocating away from the represented law necessarily
    spreads the weights, and a weight-ESS floor would silently change the dose and
    destroy the pairing with the equal-weight arm.  Equal weights give exactly 1.
    """
    K = W.shape[1]
    return W.sum(dim=1).pow(2) / torch.clamp(W.pow(2).sum(dim=1) * K, min=EPS)


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
