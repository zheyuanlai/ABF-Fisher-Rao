"""Engineering gates for the v3 family law (docs/V3_PREREGISTRATION.md, Amd. 2).

These are the gate-2 (algebraic family law) and gate-1B (oracle-carrier
stationarity) checks.  Gate 1A needs a long simulation and lives with the
campaign runner; its algebra is exercised here on a frozen carrier.
"""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from abffr import family as fam

BETA = 4.0


def _grid(n=401, lo=-3.0, hi=3.0):
    x = torch.linspace(lo, hi, n, dtype=torch.float64)
    return x, float(x[1] - x[0])


def _toy_carrier(x):
    """A two-well carrier at benchmark scale: depth ~22 kT at beta=4.

    Scale matters here.  A much steeper carrier makes the sigmoid in the capped
    family turn over inside a single grid cell, and then a finite-difference
    check of the chain rule fails for reasons that have nothing to do with the
    code under test.
    """
    A = 0.2 * x ** 4 - 1.5 * x ** 2 - 0.1 * x
    return A.unsqueeze(0)


ALL = [
    fam.Family("flat"),
    fam.Family("physical"),
    fam.Family("tempered", gamma_wt=8.0),
    fam.Family("capped", c_cut=8.0),
    fam.Family("capped", c_cut=12.0),
]


@pytest.mark.parametrize("f", ALL, ids=lambda f: f.kind + str(f.c_cut or f.gamma_wt or ""))
def test_family_law_force_is_the_derivative_of_the_bias(f):
    """Gate 2: -dB/dz must equal A' * (1 - g'(A)) for every member.

    This is the check the v3.0 draft could not have passed: it used one estimate
    of F in the force and another in the target.
    """
    # Fine grid: the identity is exact in the continuum, so the tolerance below
    # is a statement about the finite-difference stencil, not about the family.
    x, dx = _grid(n=8001)
    A = _toy_carrier(x)
    A_prime = torch.gradient(A[0], spacing=dx)[0].unsqueeze(0)

    beta_B = f.beta_bias_potential(A, BETA)
    minus_dB = -torch.gradient(beta_B[0] / BETA, spacing=dx)[0]
    predicted = (A_prime * f.bias_force_multiplier(A, BETA))[0]

    interior = slice(2, -2)   # one-sided gradient stencils at the ends
    np.testing.assert_allclose(minus_dB[interior].numpy(),
                               predicted[interior].numpy(), rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("f", ALL, ids=lambda f: f.kind + str(f.c_cut or f.gamma_wt or ""))
def test_gate_1b_oracle_carrier_makes_stationary_marginal_equal_the_target(f):
    """Gate 1B: with A = F_ref, p* proportional to exp(-beta(F_ref+B)) equals q.

    The gate frozen in v3.1 asserted this for *any* carrier, which is false; see
    the companion test below.
    """
    x, dx = _grid()
    F_ref = _toy_carrier(x)
    q = f.target(F_ref, BETA, dx)
    p_star = fam.stationary_marginal(F_ref, f.beta_bias_potential(F_ref, BETA), BETA, dx)
    np.testing.assert_allclose(p_star.numpy(), q.numpy(), rtol=1e-10, atol=1e-12)


def test_imperfect_carrier_breaks_stationarity_by_exactly_the_estimator_error():
    """Amendment 2's identity: p* proportional to q * exp(-beta(F - A)).

    A correct implementation *must* fail the v3.1 gate for an imperfect carrier.
    """
    x, dx = _grid()
    F_ref = _toy_carrier(x)
    A = F_ref + (0.30 * torch.sin(2.1 * x) + 0.15 * torch.cos(3.7 * x)).unsqueeze(0)
    f = fam.Family("capped", c_cut=12.0)

    q = f.target(A, BETA, dx)
    p_star = fam.stationary_marginal(F_ref, f.beta_bias_potential(A, BETA), BETA, dx)
    predicted = fam._normalize_log_density(
        torch.log(q.clamp_min(1e-300)) - BETA * (F_ref - A), dx)

    assert float((p_star - q).abs().max()) > 1e-3          # the gate would fail
    np.testing.assert_allclose(p_star.numpy(), predicted.numpy(),
                               rtol=1e-10, atol=1e-12)     # for this exact reason


@pytest.mark.parametrize("f", ALL, ids=lambda f: f.kind + str(f.c_cut or f.gamma_wt or ""))
def test_target_normalizes_and_is_gauge_invariant(f):
    """Adding a constant to the carrier changes neither target nor force."""
    x, dx = _grid()
    A = _toy_carrier(x)
    q = f.target(A, BETA, dx)
    assert float(torch.trapezoid(q[0], dx=dx)) == pytest.approx(1.0, abs=1e-12)

    shifted = A + 7.25
    np.testing.assert_allclose(f.target(shifted, BETA, dx).numpy(), q.numpy(),
                               rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(
        f.bias_force_multiplier(shifted, BETA).numpy(),
        f.bias_force_multiplier(A, BETA).numpy(), rtol=1e-11, atol=1e-13)


def test_log_space_survives_a_hundred_nat_target():
    """Gate 4: no overflow/underflow where v2 would have clipped."""
    x, dx = _grid()
    A = (25.0 * x ** 2).unsqueeze(0)          # beta*A spans ~900 nats
    q = fam.Family("physical").target(A, BETA, dx)
    assert torch.isfinite(q).all()
    assert float(torch.trapezoid(q[0], dx=dx)) == pytest.approx(1.0, abs=1e-10)
    assert float(q.max()) > 0.0


def test_capped_flattens_the_core_and_suppresses_only_beyond_the_cap():
    """The declared meaning of c_cut: full flattening within c_cut kT."""
    x, dx = _grid()
    A = _toy_carrier(x)
    depth = BETA * (A - A.min())
    f = fam.Family("capped", c_cut=12.0, sharpness=2.0)
    m = f.bias_force_multiplier(A, BETA)

    deep_core = depth < 6.0        # well inside the cap
    far_tail = depth > 20.0        # well beyond it
    assert float(m[deep_core].min()) > 0.99    # ~full ABF flattening
    assert float(m[far_tail].max()) < 0.01     # bias switched off, physics suppresses


def test_track_p_is_the_only_inconsistent_scheme():
    assert fam.track_p().consistent is False
    assert fam.capped(12.0).consistent is True
    assert fam.tempered(8.0).consistent is True
    assert fam.consistent_physical().consistent is True
    assert fam.plain_abf().consistent is True


def test_plain_abf_is_exactly_the_g_zero_member():
    """No carrier confound between the baseline and Track C (v3.1 decision)."""
    x, dx = _grid()
    A = _toy_carrier(x)
    m = fam.Family("flat").bias_force_multiplier(A, BETA)
    np.testing.assert_allclose(m.numpy(), np.ones_like(m.numpy()), rtol=0, atol=0)
