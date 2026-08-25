"""Cloud-level gates for the v4-A mass sidecar (docs/V4A_PREREGISTRATION.md).

These cover Gate 0C (equal-weight reduction), Gate 0D (log-space invariance) and
the mass/count ancestry distinction.  Gates 0A, 0B, 0E and 0F need the engine and
live with the sidecar wiring.
"""
import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from abffr.persistent_mass import PersistentMass


def _cloud(K=128, seed=0):
    rng = np.random.default_rng(seed)
    return torch.tensor(rng.normal(0.0, 1.0, K), dtype=torch.float64)


def test_the_sidecar_cannot_reach_the_abf_estimator():
    """Architecture is the guarantee: it has no attribute referring to one."""
    m = PersistentMass(16)
    forbidden = ("accumulator", "S_acc", "C_acc", "Fprime", "F_hat", "abf", "bias")
    for name in dir(m):
        assert not any(f.lower() in name.lower() for f in forbidden), name


# ------------------------------------------------------------------ Gate 0C
def test_equal_weights_reduce_exactly_to_the_unweighted_kde():
    z, eta = _cloud(), 0.10
    m = PersistentMass(z.numel())
    log_p = m.log_density_at(z, eta)

    d = (z.unsqueeze(1) - z.unsqueeze(0)) / eta
    unweighted = torch.logsumexp(-0.5 * d * d, dim=1) - math.log(z.numel())
    # equal up to the additive constant the FR update absorbs
    diff = (log_p - unweighted)
    assert float(diff.max() - diff.min()) < 1e-12


# ------------------------------------------------------------------ Gate 0D
@pytest.mark.parametrize("shift", [1.0, 50.0, 1000.0, -1000.0])
def test_log_space_invariance_to_an_additive_constant(shift):
    """Far beyond the measured 8e6 weight ratio (~16 nats); tested to 1000 nats."""
    z, eta = _cloud(seed=3), 0.10
    a = PersistentMass(z.numel())
    a.log_w = a.log_w + 0.7 * z          # some non-uniform state
    a._normalize()
    b = PersistentMass(z.numel())
    b.log_w = a.log_w + shift            # deliberately unnormalized
    b._normalize()

    torch.testing.assert_close(a.weights, b.weights, rtol=1e-12, atol=1e-14)
    assert a.ess() == pytest.approx(b.ess(), rel=1e-12)
    torch.testing.assert_close(a.log_density_at(z, eta), b.log_density_at(z, eta),
                               rtol=1e-12, atol=1e-12)


def test_extreme_weight_spread_does_not_underflow():
    """Hundreds of nats of spread is ordinary here; normalized weights would die."""
    z = _cloud(seed=5)
    m = PersistentMass(z.numel())
    m.log_w = m.log_w + 400.0 * z        # ~thousands of nats across the cloud
    m._normalize()
    assert torch.isfinite(m.log_w).all()
    assert float(m.log_w.max()) <= 1e-12
    assert 1.0 <= m.ess() <= m.K
    assert 0.0 < m.w_max() <= 1.0
    assert torch.isfinite(m.log_density_at(z, 0.10)).all()


def test_non_finite_log_weight_fails_closed():
    m = PersistentMass(8)
    m.log_w[3] = float("nan")
    with pytest.raises(FloatingPointError):
        m._normalize()


# ---------------------------------------------------- the FR mass update
def test_fr_update_moves_mass_toward_the_target_and_leaves_positions_alone():
    z = _cloud(seed=7)
    m = PersistentMass(z.numel())
    log_q = -0.5 * ((z - 1.5) / 0.8) ** 2         # target concentrated right
    before = float((m.weights * z).sum())
    m.fr_update(log_q, m.log_density_at(z, 0.10), theta=1.0)
    after = float((m.weights * z).sum())
    assert after > before + 0.2                   # mass moved toward the target
    assert m.ess() < m.K                          # and it cost weight uniformity


def test_theta_zero_is_exactly_inert():
    z = _cloud(seed=11)
    m = PersistentMass(z.numel())
    before = m.log_w.clone()
    m.fr_update(-0.5 * z ** 2, m.log_density_at(z, 0.10), theta=0.0)
    torch.testing.assert_close(m.log_w, before, rtol=0, atol=0)


def test_target_equal_to_current_density_is_a_fixed_point():
    z = _cloud(seed=13)
    m = PersistentMass(z.numel())
    log_p = m.log_density_at(z, 0.10)
    m.fr_update(log_p, log_p, theta=1.0)          # q == p_w
    assert m.ess() == pytest.approx(m.K, rel=1e-10)


# ------------------------------------------- count vs mass ancestry
def test_count_and_mass_ancestry_are_different_objects():
    """The v4-A reporting rule: an arm that never resamples has count ESS = K
    by construction while its mass may sit on one ancestor."""
    K = 64
    m = PersistentMass(K)
    ancestors = torch.arange(K)                   # nothing has been resampled
    counts = torch.bincount(ancestors, minlength=K).double() / K
    ess_count = float(1.0 / (counts ** 2).sum())
    assert ess_count == pytest.approx(K)          # perfect by construction

    m.log_w = torch.full((K,), -30.0, dtype=torch.float64)
    m.log_w[0] = 0.0                              # one ancestor holds ~all mass
    m._normalize()
    ess_mass, m_max = m.mass_ancestry(ancestors)
    assert m_max > 0.99
    assert ess_mass < 1.1                         # ...while mass ESS is ~1
    assert ess_count / ess_mass > 50              # the two disagree wildly


def test_mass_ancestry_aggregates_within_a_family():
    K = 8
    m = PersistentMass(K)
    ancestors = torch.tensor([0, 0, 0, 0, 1, 1, 2, 3])
    ess_mass, m_max = m.mass_ancestry(ancestors)
    assert m_max == pytest.approx(4 / 8)          # ancestor 0 holds four eighths
    expected = 1.0 / ((0.5 ** 2) + (0.25 ** 2) + (0.125 ** 2) + (0.125 ** 2))
    assert ess_mass == pytest.approx(expected)


# ------------------------------------------------------- trigger + reindex
def test_trigger_fires_only_below_the_frozen_threshold():
    z = _cloud(seed=17)
    m = PersistentMass(z.numel())
    assert not m.needs_resample(0.50)             # uniform: ESS = K
    m.log_w = m.log_w + 3.0 * z
    m._normalize()
    assert m.ess() / m.K < 0.5
    assert m.needs_resample(0.50)
    assert not m.needs_resample(0.01)             # threshold is load-bearing


def test_reset_uniform_and_reindex():
    z = _cloud(seed=19)
    m = PersistentMass(z.numel())
    m.log_w = m.log_w + 2.0 * z
    m._normalize()
    src = torch.zeros(z.numel(), dtype=torch.long)     # everything from parent 0
    m.take_indices(src)
    assert m.ess() == pytest.approx(m.K)          # identical masses after reindex
    m.log_w = m.log_w + 2.0 * z
    m._normalize()
    m.reset_uniform()
    assert m.ess() == pytest.approx(m.K)
    assert m.w_max() == pytest.approx(1.0 / m.K)
