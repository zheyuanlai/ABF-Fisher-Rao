"""Validation of WEIGHTED fiber-wise selection (Phase I).

F4 showed that the equal-weight conditional step is only as good as its target: one
arbitrary reparametrization of the hidden descriptor turned a -15% gain into a +5%
loss, because in that step the selection IS the represented distribution.  Weighted
selection splits the two jobs -- the score decides where computational effort goes,
the weights keep what the ensemble represents fixed -- so the claims this file has to
defend are:

1. the selection index is UNCHANGED (a weighted arm is the dose-matched twin of its
   equal-weight partner, not a different event);
2. statistical weight is conserved exactly, stratum by stratum (the weighted form of
   the count invariance the conditional design rests on);
3. particles migrate toward the target while the REPRESENTED law does not -- the
   decoupling itself, measured on the same population;
4. equal-weight arms are bitwise unaffected by the machinery being present.
"""
import math

import numpy as np
import torch

from abpfr.events_cond import fr_event_cond
from abpfr.fisher_rao_cond import (child_weights, conditional_log_ratio,
                                   stratified_systematic_resample, stratum_of,
                                   stratified_weights, theta_backoff_cond,
                                   weight_ess)
from abpfr.grid1p import Grid1P, ShusAccumulator1P, binned_density1p
from abpfr.grid2d import GridT2, binned_density2, periodic_gaussian_kernel
from abpfr.systems import bichannel as bc
from abpfr.systems.gateway import Method

PI = math.pi
DT = torch.float64
G2 = GridT2(x1min=-PI, L1=2 * PI, n1=96, x2min=-PI, L2=2 * PI, n2=96)
G1 = Grid1P(xmin=-PI, L=2 * PI, n=96)
S = 16


def _pop(R=3, K=1024, p_rare=0.03, seed=0):
    g = torch.Generator().manual_seed(seed)
    z1 = (torch.rand(R, K, generator=g, dtype=DT) * 2 - 1) * PI
    rare = torch.rand(R, K, generator=g) < p_rare
    z2 = torch.where(rare, torch.full((R, K), PI, dtype=DT),
                     torch.zeros(R, K, dtype=DT))
    z2 = z2 + 0.15 * torch.randn(R, K, generator=g, dtype=DT)
    return z1, torch.remainder(z2 + PI, 2 * PI) - PI


def _kernels(bw=0.25):
    k1, r1 = periodic_gaussian_kernel(bw, G2.dx1, G2.n1, "cpu", DT)
    k2, r2 = periodic_gaussian_kernel(bw, G2.dx2, G2.n2, "cpu", DT)
    return k1, r1, k2, r2


def _rare(z2, W=None):
    m = (z2.abs() > PI / 2).to(DT)
    return m.mean(dim=1) if W is None else (W * m).sum(dim=1) / W.sum(dim=1)


def _score(z1, z2, theta, alpha=0.0):
    strata = stratum_of(z1, G2, S)
    k1, r1, k2, r2 = _kernels()
    p2 = binned_density2(z1, z2, k1, r1, k2, r2, G2)
    lr = conditional_log_ratio(z1, z2, p2, G2)
    R = z1.shape[0]
    w, cnt, _, _ = theta_backoff_cond(lr, strata, S,
                                      torch.full((R,), theta, dtype=DT), alpha)
    return strata, w, cnt


def test_weight_is_conserved_stratum_by_stratum():
    z1, z2 = _pop()
    R, K = z1.shape
    strata, w, cnt = _score(z1, z2, 0.5)
    sel = stratified_systematic_resample(w, strata, cnt, S,
                                         torch.Generator().manual_seed(1))
    W = torch.ones(R, K, dtype=DT)
    Wn = child_weights(W, sel, strata, S, w, cnt)
    before = torch.zeros(R, S, dtype=DT).scatter_add_(1, strata, W)
    after = torch.zeros(R, S, dtype=DT).scatter_add_(1, strata, Wn)
    assert float((after - before).abs().max()) < 1e-10
    assert float((Wn.sum(dim=1) - float(K)).abs().max()) < 1e-9
    assert bool((Wn > 0).all())
    # more copies, less weight each: the allocation is paid for in weight
    c = torch.zeros(R, K, dtype=DT).scatter_add_(1, sel, torch.ones(R, K, dtype=DT))
    heavy = torch.gather(Wn, 1, sel)[c.gather(1, sel) > 1]
    assert float(heavy.max()) < float(Wn.max())


def test_theta_zero_leaves_weights_exactly_untouched():
    z1, z2 = _pop()
    R, K = z1.shape
    strata = stratum_of(z1, G2, S)
    w, cnt, _ = stratified_weights(torch.randn(R, K, dtype=DT), strata, S,
                                   torch.zeros(R, dtype=DT))
    sel = stratified_systematic_resample(w, strata, cnt, S,
                                         torch.Generator().manual_seed(2))
    W = 1.0 + 0.5 * torch.rand(R, K, dtype=DT)
    assert torch.equal(child_weights(W, sel, strata, S, w, cnt), W)


def test_particles_migrate_but_the_represented_law_does_not():
    """The decoupling, on one event: the same selection, read two ways."""
    z1, z2 = _pop(p_rare=0.02)
    R, K = z1.shape
    strata, w, cnt = _score(z1, z2, 0.5)
    sel = stratified_systematic_resample(w, strata, cnt, S,
                                         torch.Generator().manual_seed(3))
    W = torch.ones(R, K, dtype=DT)
    Wn = child_weights(W, sel, strata, S, w, cnt)
    z2n = torch.gather(z2, 1, sel)
    before, part, repr_ = _rare(z2), _rare(z2n), _rare(z2n, Wn)
    assert bool((part > 2.0 * before).all()), (before.tolist(), part.tolist())
    moved_p = (part - before).abs()
    moved_w = (repr_ - before).abs()
    assert bool((moved_w < 0.15 * moved_p).all()), (moved_w.tolist(),
                                                    moved_p.tolist())


def test_a_long_run_of_events_holds_the_represented_law_and_costs_weight_ess():
    """Twenty events on a frozen population: particles flow to the target, the
    represented fiber population does not, and the bill arrives as weight ESS."""
    z1, z2 = _pop(p_rare=0.02, seed=4)
    R, K = z1.shape
    W = torch.ones(R, K, dtype=DT)
    before = _rare(z2)
    gen = torch.Generator().manual_seed(5)
    for _ in range(20):
        strata, w, cnt = _score(z1, z2, 0.5)
        sel = stratified_systematic_resample(w, strata, cnt, S, gen)
        W = child_weights(W, sel, strata, S, w, cnt)
        z1, z2 = torch.gather(z1, 1, sel), torch.gather(z2, 1, sel)
    assert bool((_rare(z2) > 5.0 * before).all())
    assert bool(((_rare(z2, W) - before).abs() < 0.05).all()), _rare(z2, W).tolist()
    assert bool((weight_ess(W) < 0.9).all())
    assert float((W.sum(dim=1) - float(K)).abs().max()) < 1e-8


def test_weighted_sham_conserves_weight_without_direction():
    from abpfr.fisher_rao_cond import stratified_sham_indices
    z1, z2 = _pop(seed=6)
    R, K = z1.shape
    strata = stratum_of(z1, G2, S)
    cnt = torch.zeros(R, S, dtype=DT).scatter_add_(1, strata,
                                                   torch.ones(R, K, dtype=DT))
    sel = stratified_sham_indices(torch.tensor([0, 64, 200]), strata, cnt, S,
                                  torch.Generator().manual_seed(7), K)
    W = torch.ones(R, K, dtype=DT)
    Wn = child_weights(W, sel, strata, S)
    assert float((Wn.sum(dim=1) - float(K)).abs().max()) < 1e-9
    assert float((_rare(torch.gather(z2, 1, sel), Wn) - _rare(z2)).abs().max()) < 0.02


def test_weighted_and_equal_weight_arms_fire_the_same_event():
    """Dose matching by construction: weighting changes the weights, not the draw."""
    z1, z2 = _pop(R=2, seed=8)
    R, K = z1.shape
    k1, r1, k2, r2 = _kernels()
    act = torch.ones(R, dtype=torch.bool)
    off = torch.zeros(R, dtype=torch.bool)
    zero = torch.zeros(R, dtype=torch.long)
    args = (z1, z2, act, off, zero, zero, S, torch.arange(R),
            torch.full((R,), 0.3, dtype=DT), torch.zeros(R, dtype=DT),
            k1, r1, k2, r2, G2)
    sel_a, turn_a, _, _, W_a = fr_event_cond(*args,
                                             torch.Generator().manual_seed(9))
    sel_b, turn_b, _, _, W_b = fr_event_cond(*args,
                                             torch.Generator().manual_seed(9),
                                             None, torch.ones(R, K, dtype=DT), act)
    assert torch.equal(sel_a, sel_b) and torch.equal(turn_a, turn_b)
    assert torch.equal(W_a, torch.ones(R, K, dtype=DT))     # unweighted rows: 1
    assert not torch.equal(W_b, torch.ones(R, K, dtype=DT))


def test_weighted_layers_are_bitwise_the_unweighted_ones_at_weight_one():
    X = (torch.rand(3, 512, dtype=DT) * 2 - 1) * PI
    k, r = periodic_gaussian_kernel(0.25, G1.dx, G1.n, "cpu", DT)
    ones = torch.ones_like(X)
    assert torch.equal(binned_density1p(X, k, r, G1),
                       binned_density1p(X, k, r, G1, ones))
    k2, r2 = periodic_gaussian_kernel(0.25, G2.dx2, G2.n2, "cpu", DT)
    Y = (torch.rand(3, 512, dtype=DT) * 2 - 1) * PI
    assert torch.equal(binned_density2(X, Y, k, r, k2, r2, G2),
                       binned_density2(X, Y, k, r, k2, r2, G2, ones))
    beta = torch.full((3,), 4.0, dtype=DT)
    a = ShusAccumulator1P(3, G1, beta, 0.06, "cpu", DT)
    b = ShusAccumulator1P(3, G1, beta, 0.06, "cpu", DT)
    a.deposit(X)
    b.deposit(X, ones)
    assert torch.equal(a.buf, b.buf)
    c = ShusAccumulator1P(3, G1, beta, 0.06, "cpu", DT)
    c.deposit(X, 2.0 * ones)
    assert torch.allclose(c.buf, 2.0 * a.buf)               # linear in the weights


# -----------------------------------------------------------------------------
# engine level
# -----------------------------------------------------------------------------
def _cfg(**kw):
    # a LOW hidden barrier on purpose: the engine tests need channel B to be reached
    # inside a few seconds of CPU, which the production Type-C cells (Hperp >= 2) are
    # specifically designed to prevent
    base = dict(K=128, dt=1e-3, n_steps=8_000, block=20, n_saves=20, profile_every=4,
                joint_every=8, ess_window_steps=1000, n_strata=8, beta=4.0,
                Hperp=0.5, Delta=0.0)
    base.update(kw)
    return bc.BiChannelConfig(**base)


def _arms():
    win = dict(theta=0.05, t_on_frac=0.05, t_off_frac=0.95, fr_every_blocks=3)
    return [Method("shus"),
            Method("fr_cond", use_fr=True, cond_fr=True, **win),
            Method("wfr_cond", use_fr=True, cond_fr=True, cond_weighted=True, **win),
            Method("wsham_cond", sham=True, shadows="wfr_cond", cond_weighted=True,
                   **win)]


def test_engine_decouples_allocation_from_the_represented_channel_population():
    recs = bc.simulate_batch([_cfg()], [0], _arms(), batch_seed=13,
                             device=torch.device("cpu"), dtype=DT)
    by = {r["method"]["name"]: r for r in recs}
    for n in ("fr_cond", "wfr_cond"):
        assert by[n]["total_turnover"] > 0.0, n
    assert abs(by["wsham_cond"]["total_turnover"]
               - by["wfr_cond"]["total_turnover"]) < 1e-9
    # equal-weight arms: the two readings of the population coincide exactly
    for n in ("shus", "fr_cond"):
        assert np.array_equal(by[n]["P_regions"], by[n]["P_regions_n"])
        assert np.allclose(by[n]["ess_w_t"], 1.0)
    w = by["wfr_cond"]
    assert np.allclose(w["w_sum_t"], 1.0, atol=1e-9)         # weight conserved
    # the weighted arm allocates particles to the rare channel ...
    assert w["final_p_B_n"] > 1.5 * by["shus"]["final_p_B"]
    # ... without moving the represented population as far as the equal-weight arm
    d_w = abs(w["final_p_B"] - by["shus"]["final_p_B"])
    d_e = abs(by["fr_cond"]["final_p_B"] - by["shus"]["final_p_B"])
    assert d_w < d_e, (w["final_p_B"], by["fr_cond"]["final_p_B"],
                       by["shus"]["final_p_B"])
    assert w["min_ess_w"] < 1.0                              # weights did spread


def test_a_weighted_arm_in_the_batch_does_not_disturb_the_plain_baseline():
    """Estimator protection at batch level: nothing a weighted arm does reaches
    another row's accumulator or physics.  (Only the reallocation-free baseline is
    comparable ACROSS batches -- the shared event generator's draw depends on the
    row count, which is why every arm of an experiment lives in one batch.)"""
    cfg = _cfg()
    a = bc.simulate_batch([cfg], [0], _arms()[:2], batch_seed=14,
                          device=torch.device("cpu"), dtype=DT)
    b = bc.simulate_batch([cfg], [0], _arms(), batch_seed=14,
                          device=torch.device("cpu"), dtype=DT)
    ra = [r for r in a if r["method"]["name"] == "shus"][0]
    rb = [r for r in b if r["method"]["name"] == "shus"][0]
    assert np.array_equal(ra["l2_f_t"], rb["l2_f_t"])
    assert np.array_equal(ra["pmf_t"], rb["pmf_t"])


def test_weighted_marginal_reallocation_is_refused():
    import pytest
    with pytest.raises(AssertionError, match="FIBER-WISE"):
        bc.simulate_batch([_cfg()], [0],
                          [Method("shus"),
                           Method("bad", use_fr=True, cond_weighted=True, theta=0.05,
                                  t_on_frac=0.1, t_off_frac=0.9)],
                          batch_seed=15, device=torch.device("cpu"), dtype=DT)


# -----------------------------------------------------------------------------
# Phase J: the variance-limited protocol and the discrete-state allocation
# -----------------------------------------------------------------------------
def test_stationary_init_reproduces_the_law_it_claims():
    """The initial condition IS the reference: if it is off, the run measures a bias
    it created itself, which is the whole thing Phase J exists to avoid."""
    cfg = bc.BiChannelConfig(beta=4.0, Hperp=1.5, Delta=1.0)
    ref = bc.reference_objects(cfg.beta, cfg.Hperp, cfg.Delta, cfg.Ha, cfg.Hb,
                               "cpu", DT)
    for biased, key in ((True, "p_B_ref_biased"), (False, "p_B_ref")):
        z1, z2 = bc.stationary_init(cfg, 200_000, 0, biased, "cpu", DT)
        pB = float((torch.cos(z2) < 0).to(DT).mean())
        assert abs(pB - ref[key]) < 0.01, (biased, pB, ref[key])
    z1, _ = bc.stationary_init(cfg, 200_000, 0, True, "cpu", DT)
    h = torch.histc(z1, bins=16, min=-PI, max=PI)
    assert float(h.min() / h.max()) > 0.9          # the biased law is uniform in phi
    z1b, _ = bc.stationary_init(cfg, 200_000, 0, False, "cpu", DT)
    hb = torch.histc(z1b, bins=16, min=-PI, max=PI)
    assert float(hb.min() / hb.max()) < 0.5        # the Boltzmann law is not


def test_warm_start_begins_at_the_analytic_fixed_point():
    cfg = bc.BiChannelConfig(beta=4.0, Hperp=1.5, Delta=1.0, K=64, n_steps=200,
                             n_saves=4, profile_every=1, joint_every=1,
                             init="stationary", warm_start=True)
    e_star = bc.analytic_floors(cfg, "cpu", DT)["e_star"]
    recs = bc.simulate_batch([cfg], [0], [Method("shus")], batch_seed=17,
                             device=torch.device("cpu"), dtype=DT)
    assert recs[0]["l2_f_t"][0] < 1.5 * e_star, (recs[0]["l2_f_t"][0], e_star)
    cold = bc.BiChannelConfig(**{**cfg.__dict__, "warm_start": False})
    rc = bc.simulate_batch([cold], [0], [Method("shus")], batch_seed=17,
                           device=torch.device("cpu"), dtype=DT)
    assert rc[0]["l2_f_t"][0] > 10 * recs[0]["l2_f_t"][0]   # a cold start is nowhere


def test_state_allocation_equalizes_counts_per_state_at_theta_one():
    """theta = 1 with a uniform state target IS equal-count-per-state allocation."""
    from abpfr.fisher_rao_cond import conditional_log_ratio_state
    z1, z2 = _pop(R=2, K=1024, p_rare=0.1, seed=21)
    R, K = z1.shape
    state = (z2.abs() > PI / 2).long()
    strata = stratum_of(z1, G2, S)
    lr = conditional_log_ratio_state(state, strata, 2, S)
    w, cnt, _, _ = theta_backoff_cond(lr, strata, S, torch.ones(R, dtype=DT), 0.0)
    sel = stratified_systematic_resample(w, strata, cnt, S,
                                         torch.Generator().manual_seed(22))
    def share(lab):
        c = torch.zeros(R, S * 2, dtype=DT).scatter_add_(
            1, strata * 2 + lab, torch.ones(R, K, dtype=DT)).reshape(R, S, 2)
        return c, c[..., 1] / torch.clamp(c.sum(dim=2), min=1.0)
    c0, f0 = share(state)
    c1, f1 = share(torch.gather(state, 1, sel))
    both = c0.min(dim=2).values > 0                        # both states present before
    assert bool(both.any())
    # systematic resampling equalizes each walker's count to within one, so a STATE's
    # share lands near 1/2 but not exactly (its members are interleaved in the draw)
    assert float((f1[both] - 0.5).abs().max()) < 0.12, f1[both]
    assert float((f0[both] - 0.5).abs().max()) > 0.25      # it was far from 1/2


def test_state_allocation_needs_no_kernel_and_no_bins():
    """The point of the control: it consumes only the state label and the stratum."""
    from abpfr.fisher_rao_cond import conditional_log_ratio_state
    z1, z2 = _pop(R=2, K=512, p_rare=0.25, seed=23)
    strata = stratum_of(z1, G2, S)
    state = (z2.abs() > PI / 2).long()
    lr = conditional_log_ratio_state(state, strata, 2, S)
    # rare-state walkers score positive (they are the ones to allocate to)
    assert float(lr[state == 1].mean()) > float(lr[state == 0].mean())
    balanced = torch.zeros_like(state)
    balanced[:, ::2] = 1
    lrb = conditional_log_ratio_state(balanced, strata, 2, S)
    assert float(lrb.abs().max()) < 0.7      # already balanced -> nothing to do
