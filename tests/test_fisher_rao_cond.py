"""Validation of the fiber-wise (conditional) Fisher-Rao step.

The claims this file has to defend, because the whole Phase-F design rests on them:

1. the xi-marginal is invariant at stratum resolution (so the SHUS deposit signal
   cannot be perturbed by the reallocation);
2. theta = 0 is exactly the identity (an event that cannot meet the ESS floor is a
   no-op, same guarantee as the marginal step);
3. the selection law is unbiased within a stratum: E[children of k] = cnt_j w_k, and
   systematic resampling realizes it to within one child;
4. the conditional step ENRICHES an under-populated fiber, and the marginal step is
   BLIND to it -- this is the structural asymmetry the whole phase is built on.
"""
import math

import pytest
import torch

from abpfr.events1p import fr_event1p
from abpfr.events_cond import fr_event_cond
from abpfr.fisher_rao_cond import (conditional_log_ratio,
                                   conditional_log_ratio_binned,
                                   stratified_sham_indices,
                                   stratified_systematic_resample, stratum_of,
                                   stratified_weights, theta_backoff_cond)
from abpfr.grid1p import Grid1P
from abpfr.grid2d import GridT2, binned_density2, periodic_gaussian_kernel
from abpfr.resampling import turnover_counts

PI = math.pi
DT = torch.float64
G2 = GridT2(x1min=-PI, L1=2 * PI, n1=96, x2min=-PI, L2=2 * PI, n2=96)
G1 = Grid1P(xmin=-PI, L=2 * PI, n=96)
S = 16


def _pop(R=3, K=1024, p_rare=0.03, seed=0):
    """xi uniform on the circle; a fraction p_rare of walkers sit in the far fiber.

    The rare fiber is INDEPENDENT of xi, which is the worst case for a marginal
    step: no xi-signal whatsoever distinguishes the two channels.
    """
    g = torch.Generator().manual_seed(seed)
    z1 = (torch.rand(R, K, generator=g, dtype=DT) * 2 - 1) * PI
    rare = torch.rand(R, K, generator=g) < p_rare
    z2 = torch.where(rare, torch.full((R, K), PI, dtype=DT),
                     torch.zeros(R, K, dtype=DT))
    z2 = z2 + 0.15 * torch.randn(R, K, generator=g, dtype=DT)
    z2 = torch.remainder(z2 + PI, 2 * PI) - PI
    return z1, z2


def _counts(strata, sel, R, K):
    idx = torch.gather(strata, 1, sel)
    c = torch.zeros(R, S, dtype=DT)
    c.scatter_add_(1, idx, torch.ones(R, K, dtype=DT))
    return c


def _rare_frac(z2, sel=None):
    z = z2 if sel is None else torch.gather(z2, 1, sel)
    return (z.abs() > PI / 2).to(DT).mean(dim=1)


def _kernels(bw=0.25):
    k1, r1 = periodic_gaussian_kernel(bw, G2.dx1, G2.n1, "cpu", DT)
    k2, r2 = periodic_gaussian_kernel(bw, G2.dx2, G2.n2, "cpu", DT)
    return k1, r1, k2, r2


def test_theta_zero_is_the_identity():
    z1, z2 = _pop()
    R, K = z1.shape
    strata = stratum_of(z1, G2, S)
    lr = torch.randn(R, K, dtype=DT)
    w, cnt, essf = stratified_weights(lr, strata, S, torch.zeros(R, dtype=DT))
    sel = stratified_systematic_resample(w, strata, cnt, S,
                                         torch.Generator().manual_seed(1))
    assert torch.equal(sel, torch.arange(K).unsqueeze(0).expand(R, K))
    assert torch.allclose(essf, torch.ones(R, dtype=DT))


@pytest.mark.parametrize("theta", [0.05, 0.3, 1.0])
def test_marginal_is_invariant_at_stratum_resolution(theta):
    z1, z2 = _pop()
    R, K = z1.shape
    strata = stratum_of(z1, G2, S)
    k1, r1, k2, r2 = _kernels()
    p2 = binned_density2(z1, z2, k1, r1, k2, r2, G2)
    lr = conditional_log_ratio(z1, z2, p2, G2)
    w, cnt, _, _ = theta_backoff_cond(lr, strata, S, torch.full((R,), theta,
                                                                dtype=DT), 0.0)
    sel = stratified_systematic_resample(w, strata, cnt, S,
                                         torch.Generator().manual_seed(2))
    assert torch.equal(_counts(strata, sel, R, K), cnt)


def test_selection_is_unbiased_within_a_stratum():
    z1, z2 = _pop()
    R, K = z1.shape
    strata = stratum_of(z1, G2, S)
    lr = torch.randn(R, K, dtype=DT) * 2.0
    w, cnt, _ = stratified_weights(lr, strata, S, torch.full((R,), 0.5, dtype=DT))
    sel = stratified_systematic_resample(w, strata, cnt, S,
                                         torch.Generator().manual_seed(3))
    children = torch.zeros(R, K, dtype=DT).scatter_add_(
        1, sel, torch.ones(R, K, dtype=DT))
    expected = w * torch.gather(cnt, 1, strata)
    assert float((children - expected).abs().max()) < 1.0     # systematic: floor/ceil
    assert torch.allclose(children.sum(dim=1), torch.full((R,), float(K), dtype=DT))


def test_conditional_enriches_the_rare_fiber_and_marginal_is_blind():
    """The structural asymmetry Phase F is built on.

    Two walkers at the same xi in different channels get the SAME marginal score, so
    the marginal step cannot prefer the rare one; the fiber-wise step can.
    """
    z1, z2 = _pop(p_rare=0.03)
    R, K = z1.shape
    k1, r1, k2, r2 = _kernels()
    act = torch.ones(R, dtype=torch.bool)
    off = torch.zeros(R, dtype=torch.bool)
    partner = torch.arange(R)
    theta = torch.full((R,), 0.5, dtype=DT)
    alpha = torch.full((R,), 0.0, dtype=DT)
    zero_nb = torch.zeros(R, dtype=torch.long)

    before = _rare_frac(z2)
    sel_m, _, _, _ = fr_event1p(z1, act, off, off, zero_nb, partner, theta, alpha,
                                k1, r1, G1, torch.Generator().manual_seed(4))
    sel_c, _, _, _ = fr_event_cond(z1, z2, act, off, zero_nb, zero_nb, S, partner,
                                   theta, alpha, k1, r1, k2, r2, G2,
                                   torch.Generator().manual_seed(4))
    after_m = _rare_frac(z2, sel_m)
    after_c = _rare_frac(z2, sel_c)
    # marginal: no systematic movement of the fiber population
    assert float((after_m - before).abs().max()) < 0.01
    # conditional: every row enriches the rare fiber substantially
    assert bool((after_c > 1.5 * before).all()), (before.tolist(), after_c.tolist())


def test_stratified_count_control_shares_the_geometry():
    """The count control differs from FR ONLY in the density estimator."""
    z1, z2 = _pop()
    R, K = z1.shape
    strata = stratum_of(z1, G2, S)
    lr_c = conditional_log_ratio_binned(z1, z2, 9, 9, G2)
    w, cnt, _, _ = theta_backoff_cond(lr_c, strata, S,
                                      torch.full((R,), 0.3, dtype=DT), 0.0)
    sel = stratified_systematic_resample(w, strata, cnt, S,
                                         torch.Generator().manual_seed(5))
    assert torch.equal(_counts(strata, sel, R, K), cnt)
    assert bool((_rare_frac(z2, sel) > _rare_frac(z2)).all())


def test_theta_backoff_respects_the_ess_floor():
    z1, z2 = _pop(p_rare=0.002)
    R, K = z1.shape
    strata = stratum_of(z1, G2, S)
    k1, r1, k2, r2 = _kernels()
    p2 = binned_density2(z1, z2, k1, r1, k2, r2, G2)
    lr = conditional_log_ratio(z1, z2, p2, G2)
    w, cnt, th, essf = theta_backoff_cond(lr, strata, S,
                                          torch.full((R,), 4.0, dtype=DT), 0.5)
    assert bool((essf >= 0.5).all()), essf.tolist()
    assert bool((th <= 4.0).all()) and bool((th > 0).any())


def test_stratified_sham_matches_turnover_without_moving_the_fiber():
    z1, z2 = _pop()
    R, K = z1.shape
    strata = stratum_of(z1, G2, S)
    cnt = torch.zeros(R, S, dtype=DT).scatter_add_(1, strata,
                                                   torch.ones(R, K, dtype=DT))
    m = torch.tensor([0, 64, 200])
    sel = stratified_sham_indices(m, strata, cnt, S,
                                  torch.Generator().manual_seed(6), K)
    assert torch.equal(_counts(strata, sel, R, K), cnt)
    assert torch.equal(turnover_counts(sel, K), m)
    assert float((_rare_frac(z2, sel) - _rare_frac(z2)).abs().max()) < 0.02
