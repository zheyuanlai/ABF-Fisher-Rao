"""Phase II engineering gates for the v3 FR operators.

Frozen protocol: docs/V3_PREREGISTRATION.md (v3.1) with Amendments 1-3.
These are the cloud-level gates; the whole-run gates (no accumulator mutation,
no FR outside the window, zero-strength run identity) come with the engine
wiring and live with the campaign runner.
"""
import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from abffr import fr_v3


def _gen(seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _cloud(K=256, seed=0, spread=1.0):
    rng = np.random.default_rng(seed)
    z = torch.tensor(rng.normal(0.0, 1.0, K), dtype=torch.float64)
    log_p = -0.5 * z ** 2
    log_q = -0.5 * ((z - spread) / 1.3) ** 2
    return fr_v3.FRScore(log_p=log_p, log_q=log_q)


# ---------------------------------------------------------------- gate 1
def test_score_is_exactly_centered():
    s = _cloud()
    assert float(s.S.mean().abs()) < 1e-13
    np.testing.assert_allclose(s.a.numpy(), (-s.r).numpy(), rtol=0, atol=0)


# ---------------------------------------------------------------- gate 2
def test_p_equals_q_is_inert_for_both_operators():
    """No reallocation when the ensemble already has the target distribution."""
    K = 128
    log_p = torch.linspace(-2.0, 2.0, K, dtype=torch.float64)
    s = fr_v3.FRScore(log_p=log_p, log_q=log_p.clone())     # p == q exactly
    assert float(s.S.abs().max()) < 1e-13

    src_bd, n = fr_v3.bd_standard(s, dtau=5.0, generator=_gen(1))
    assert n == 0
    np.testing.assert_array_equal(src_bd.numpy(), np.arange(K))

    src_ft, theta, ess = fr_v3.ft_step(s, rho=0.85, generator=_gen(2))
    assert theta == pytest.approx(1.0)                      # governor unconstrained
    assert ess == pytest.approx(K)
    assert fr_v3.replacement_count(src_ft, K) == 0          # exact, not statistical
    np.testing.assert_array_equal(src_ft.numpy(), np.arange(K))


# ---------------------------------------------------------------- gate 3
def test_bd_sign_convention_overrepresented_dies_underrepresented_clones():
    """One strongly over- and one strongly under-represented particle."""
    K = 64
    log_p = torch.zeros(K, dtype=torch.float64)
    log_q = torch.zeros(K, dtype=torch.float64)
    log_p[0] = 12.0      # p >> q at particle 0 -> S > 0 -> must die
    log_q[1] = 12.0      # q >> p at particle 1 -> S < 0 -> must clone
    s = fr_v3.FRScore(log_p=log_p, log_q=log_q)
    assert float(s.S[0]) > 0 and float(s.S[1]) < 0

    survived_0, cloned_1 = 0, 0
    trials = 200
    for t in range(trials):
        src, _ = fr_v3.bd_standard(s, dtau=1.0, generator=_gen(100 + t))
        counts = fr_v3.offspring_counts(src, K)
        survived_0 += int(counts[0] > 0)
        cloned_1 += int(counts[1] > 1)
    assert survived_0 < 0.1 * trials     # particle 0 is nearly always killed
    assert cloned_1 > 0.5 * trials       # particle 1 is usually duplicated


# ---------------------------------------------------------------- gate 4
def test_bd_weak_generator_matches_the_continuum_drift():
    """E[d phi]/dtau must equal the FR drift, testing sign, partner choice and rate.

    The exact one-step expectation of the implemented scheme is derived as
    follows.  To first order in dtau particle i fires with probability
    |S_i| dtau; if S_i > 0 slot i takes a uniform partner's content, and if
    S_i < 0 a uniform partner takes slot i's content, so

        E[d sum(phi)] = dtau * sum_i S_i * (mean_{-i}(phi) - phi_i)
                      = dtau * K/(K-1) * sum_i S_i * (mean(phi) - phi_i)
                      = -dtau * K/(K-1) * sum_i S_i phi_i        (since sum_i S_i = 0)

    hence E[d mean(phi)]/dtau = -K/(K-1) * mean(phi S).  That is the continuum
    drift -E_p[phi S] times an explicit finite-K factor, and it is what the
    scheme must reproduce.

    phi = S is used as the observable because it maximizes the signal-to-noise
    of the estimator; with a weakly correlated phi the Monte-Carlo error swamps
    a 50% rate error, which is how the first version of this gate passed on its
    absolute tolerance while measuring almost nothing.
    """
    K = 96
    s = _cloud(K=K, seed=3, spread=0.8)
    phi = s.S.clone()
    exact = -K / (K - 1) * float((phi * s.S).mean())
    continuum = -float((phi * s.S).mean())
    assert exact == pytest.approx(continuum, rel=0.02)     # finite-K factor is small

    dtau, trials = 5e-3, 40000
    total = 0.0
    for t in range(trials):
        src, _ = fr_v3.bd_standard(s, dtau, generator=_gen(7000 + t))
        total += float(phi[src].mean() - phi.mean())
    measured = total / trials / dtau
    assert measured == pytest.approx(exact, rel=0.05)

    # Directional power: with p and q exchanged the drift must reverse.
    flipped = fr_v3.FRScore(log_p=s.log_q, log_q=s.log_p)
    total = 0.0
    for t in range(4000):
        src, _ = fr_v3.bd_standard(flipped, dtau, generator=_gen(90000 + t))
        total += float(phi[src].mean() - phi.mean())
    assert (total / 4000 / dtau) > 0.5 * abs(exact)


# ---------------------------------------------------------------- gate 5
def test_ft_weights_are_exactly_the_geometric_interpolation():
    s = _cloud(K=64, seed=7)
    theta = 0.4
    logw = theta * s.a
    logw = logw - logw.max()
    w = torch.exp(logw)
    w = w / w.sum()
    expected = (torch.exp(s.log_q - s.log_p)) ** theta
    expected = expected / expected.sum()
    np.testing.assert_allclose(w.numpy(), expected.numpy(), rtol=1e-12, atol=1e-15)


# ---------------------------------------------------------------- gate 6
def test_ft_semigroup_law_on_densities():
    """T_t2(T_t1[p]) == T_{t1+t2-t1t2}[p]: catches a reversed exponent."""
    x = np.linspace(-4, 4, 1201)
    p = np.exp(-(x + 1.0) ** 2); p /= np.trapezoid(p, x)
    q = np.exp(-0.5 * ((x - 0.8) / 1.2) ** 2); q /= np.trapezoid(q, x)

    def T(pp, th):
        z = (1 - th) * np.log(pp + 1e-300) + th * np.log(q + 1e-300)
        r = np.exp(z - z.max())
        return r / np.trapezoid(r, x)

    t1, t2 = 0.35, 0.55
    lhs = T(T(p, t1), t2)
    np.testing.assert_allclose(lhs, T(p, t1 + t2 - t1 * t2), rtol=1e-10, atol=1e-12)
    assert np.abs(lhs - T(p, min(t1 + t2, 1.0))).max() > 1e-3      # naive law is wrong


def test_theta_and_fr_time_are_inverses():
    for dtau in (0.05, 0.2, 0.5, 1.0, 3.0):
        assert fr_v3.dtau_from_theta(fr_v3.theta_from_dtau(dtau)) == pytest.approx(dtau)
    assert fr_v3.theta_from_dtau(0.2) == pytest.approx(1 - math.exp(-0.2))
    assert fr_v3.theta_from_dtau(0.2) != pytest.approx(0.2, abs=1e-3)   # not the naive map


# ---------------------------------------------------------------- gate 7
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_ess_is_non_increasing_in_theta(seed):
    """Amendment 3's theorem, checked numerically on varied clouds."""
    s = _cloud(K=200, seed=seed, spread=1.0 + 0.4 * seed)
    thetas = np.linspace(0.0, 1.0, 51)
    ess = [fr_v3.ess_of_theta(s.a, float(t)) for t in thetas]
    assert all(ess[i + 1] <= ess[i] + 1e-9 for i in range(len(ess) - 1))
    assert ess[0] == pytest.approx(200)


# ---------------------------------------------------------------- gate 8
@pytest.mark.parametrize("rho", [0.70, 0.85])
def test_ess_governor_returns_the_maximal_theta(rho):
    K = 256
    s = _cloud(K=K, seed=11, spread=2.0)
    theta = fr_v3.ess_governor(s.a, rho, K)
    assert 0.0 < theta < 1.0
    assert fr_v3.ess_of_theta(s.a, theta) >= rho * K - 1e-6
    assert fr_v3.ess_of_theta(s.a, min(theta + 1e-3, 1.0)) < rho * K   # maximal


def test_governor_refuses_to_move_when_rho_is_unsatisfiable():
    s = _cloud(K=64, seed=13)
    assert fr_v3.ess_governor(s.a, rho=1.0, K=64) == 0.0


# ---------------------------------------------------------------- gate 9
def test_log_space_survives_a_nine_hundred_nat_score():
    """Where v2 clipped, v3 must simply compute."""
    K = 128
    z = torch.linspace(-3, 3, K, dtype=torch.float64)
    s = fr_v3.FRScore(log_p=torch.zeros(K, dtype=torch.float64),
                      log_q=-100.0 * z ** 2)      # spans ~900 nats
    assert float(s.S.abs().max()) > 100.0
    src, theta, ess = fr_v3.ft_step(s, rho=0.85, generator=_gen(17))
    assert torch.isfinite(torch.tensor(theta)) and math.isfinite(ess)
    assert int(src.min()) >= 0 and int(src.max()) < K
    dtau = fr_v3.bd_timestep(s, p_max=0.05)
    src_bd, n = fr_v3.bd_standard(s, dtau, _gen(19))
    assert math.isfinite(dtau) and n >= 0


# ---------------------------------------------------------------- gate 10
def test_permutation_equivariance():
    """Relabelling the cloud relabels the outcome; no hidden index bias."""
    K = 128
    s = _cloud(K=K, seed=23, spread=1.5)
    perm = torch.randperm(K, generator=_gen(29))
    s_perm = fr_v3.FRScore(log_p=s.log_p[perm], log_q=s.log_q[perm])

    theta = fr_v3.ess_governor(s.a, 0.85, K)
    theta_perm = fr_v3.ess_governor(s_perm.a, 0.85, K)
    assert theta == pytest.approx(theta_perm, rel=1e-9)
    assert fr_v3.ess_of_theta(s.a, theta) == pytest.approx(
        fr_v3.ess_of_theta(s_perm.a, theta_perm), rel=1e-9)


# ---------------------------------------------------------------- gates 11-12
def test_genealogy_and_replacement_bookkeeping():
    K = 64
    src = torch.arange(K)
    src[5] = 3        # slot 5 now holds a copy of 3
    src[9] = 3        # slot 9 too
    counts = fr_v3.offspring_counts(src, K)
    assert int(counts[3]) == 3
    assert int(counts[5]) == 0 and int(counts[9]) == 0
    # two parents eliminated == two excess children
    assert fr_v3.replacement_count(src, K) == 2
    assert int((counts - 1).clamp_min(0).sum()) == 2

    ancestors = torch.arange(K)
    ancestors = ancestors[src]
    w = torch.bincount(ancestors, minlength=K).double() / K
    ess_anc = float(1.0 / (w * w).sum())
    assert ess_anc < K and float(w.max()) == pytest.approx(3.0 / K)


# ---------------------------------------------------------------- gate 13
def test_holdout_marks_only_the_extra_children_and_refreshes_on_recloning():
    K = 8
    src = torch.tensor([0, 1, 1, 1, 4, 4, 6, 7])
    is_clone = fr_v3.clone_mask(src)
    # parent 1 keeps slot 1 as its continuation; slots 2,3 are new clones
    np.testing.assert_array_equal(
        is_clone.numpy(), np.array([False, False, True, True, False, True, False, False]))
    assert int(is_clone.sum()) == fr_v3.replacement_count(src, K)

    hold = torch.zeros(K, dtype=torch.long)
    hold[1] = 120                       # parent 1 was itself still held out
    new_hold = fr_v3.apply_holdout(hold, src, is_clone, hold_steps=500)
    assert int(new_hold[1]) == 120      # continuation inherits the remainder
    assert int(new_hold[2]) == 500      # a clone of a held-out replica restarts
    assert int(new_hold[3]) == 500
    assert int(new_hold[0]) == 0


def test_bd_and_ft_share_the_same_src_contract():
    """Both operators must be applicable by the identical indexing rule."""
    K = 64
    s = _cloud(K=K, seed=31, spread=1.2)
    X = torch.arange(K, dtype=torch.float64)
    for src in (fr_v3.bd_standard(s, 0.3, _gen(37))[0],
                fr_v3.ft_step(s, 0.85, _gen(41))[0]):
        assert src.shape == (K,)
        assert int(src.min()) >= 0 and int(src.max()) < K
        np.testing.assert_allclose(X[src].numpy(), src.numpy().astype(float))
