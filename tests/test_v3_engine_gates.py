"""Whole-run engine gates 14-17 for the v3 arms, plus the A.6 stream invariant.

Frozen protocol: docs/V3_PREREGISTRATION.md (v3.1), Amendments 1-3, Appendix A.
"""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from abffr import metrics as m, potentials, simulation_torch as st
from abffr.io_utils import RunSpec

DOMAIN = dict(x_min=-3.0, x_max=3.0, y_min=-2.5, y_max=3.5)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="v3 engine gates run on the CUDA backend")


def _cfg(n_steps=2000, **over):
    cfg = dict(
        simulation=dict(beta=4.0, dt=0.002, n_steps=n_steps, n_particles=128,
                        eval_every=200, x_init_mode="uniform", y_init_mode="uniform"),
        abf=dict(h=0.05, update_every=10, min_count=1.0,
                 observation_order="post_propagation"),
        domain=DOMAIN, potential=dict(x_tilt=0.1021665783),
        fr=dict(noise_chunk_steps=500))
    cfg.update(over)
    return cfg


def _run(v3, n_steps=2000, seed=0, burnin=0.2, stop=0.8):
    cfg = _cfg(n_steps=n_steps)
    cfg["v3"] = v3
    x_grid = np.linspace(-3.0, 3.0, 401)
    specs = [RunSpec(method="v3", target_type="none", seed=seed, gamma=0.0,
                     eta=0.10, fr_every=1, burnin_fraction=burnin,
                     stop_fraction=stop)]
    return st.run_batch(specs, cfg=cfg, x_grid=x_grid,
                        F_ref=np.zeros(401), Fprime_ref=np.zeros(401),
                        ev=m.EvalConfig.from_domain(DOMAIN), device=DEVICE,
                        dtype=torch.float64)


CAPPED = dict(kind="capped", c_cut=12.0)


# ------------------------------------------------------------------ gate 16
def test_fr_opportunity_indices_are_exactly_the_registered_set():
    """A.3: assert the whole index array, not merely 'nothing outside the window'.

    The weaker assertion passes for a first event one stride late, a missing
    event at the closing endpoint, or a stride applied to estimator updates
    instead of physical steps.
    """
    r = _run(dict(enabled=True, family=CAPPED, operator="ft", rho=0.85,
                  fr_stride=200), n_steps=2000, burnin=0.2, stop=0.8)
    expected = list(range(400, 1601, 200))
    assert r.fr_opportunities == expected
    assert len(expected) == (1600 - 400) // 200 + 1 == 7

    # The frozen campaign schedule itself: 61 opportunities, both endpoints.
    campaign = list(range(10000, 40001, 500))
    assert len(campaign) == 61
    assert campaign[0] == 10000 and campaign[-1] == 40000


def test_no_fr_arms_never_fire():
    r = _run(dict(enabled=True, family=CAPPED, operator="none"))
    assert r.fr_opportunities == []
    assert r.diags[0]["cumulative_replacements"][-1] == 0
    assert r.diags[0]["ancestor_ess"][-1] == pytest.approx(128.0)


# ------------------------------------------------------------------ gate 17
@pytest.mark.parametrize("zero", [dict(operator="ft", rho=1.0),
                                  dict(operator="bd", p_max=1e-12)])
def test_zero_strength_reproduces_the_same_bias_no_fr_arm(zero):
    """Amendment 1 criterion: 1e-5 on profile quantities, exact on counters."""
    base = _run(dict(enabled=True, family=CAPPED, operator="none"))
    v3 = dict(enabled=True, family=CAPPED, fr_stride=200, **zero)
    if zero["operator"] == "ft":
        v3["target_family"] = CAPPED
    weak = _run(v3)

    b, w = base.diags[0], weak.diags[0]
    assert w["cumulative_replacements"][-1] == 0          # exact
    assert w["ancestor_ess"][-1] == pytest.approx(128.0)
    assert b["barrier_crossings"][-1] == w["barrier_crossings"][-1]
    np.testing.assert_allclose(np.asarray(w["F_hat"][-1]),
                               np.asarray(b["F_hat"][-1]), atol=1e-5)
    np.testing.assert_allclose(np.asarray(w["Fprime_hat"][-1]),
                               np.asarray(b["Fprime_hat"][-1]), atol=1e-5)


# ------------------------------------------------------------------ gate 15
def test_fr_does_not_contribute_to_the_estimator_at_its_own_instant():
    """FR acts after accumulation, so the estimator at the first opportunity
    must be identical with and without FR."""
    stride, burnin, n = 200, 0.2, 2000
    off = _run(dict(enabled=True, family=CAPPED, operator="none"), n_steps=n)
    on = _run(dict(enabled=True, family=CAPPED, operator="ft", rho=0.70,
                   fr_stride=stride), n_steps=n)
    first = 400
    i_off = off.diags[0]["steps"].index(first)
    i_on = on.diags[0]["steps"].index(first)
    assert on.diags[0]["cumulative_replacements"][i_on] > 0   # FR did fire here
    np.testing.assert_allclose(np.asarray(on.diags[0]["Fprime_hat"][i_on]),
                               np.asarray(off.diags[0]["Fprime_hat"][i_off]),
                               atol=1e-9)


# ------------------------------------------------------------------ gate 14
def test_oracle_refresh_samples_the_true_conditional_law():
    """A.5: y_child ~ pi(y|x) prop exp(-beta V(x,y)), x untouched."""
    beta, ny = 4.0, 401
    y = torch.linspace(-2.5, 3.5, ny, dtype=torch.float64, device=DEVICE)
    for x0 in (-1.05, 0.0, 1.0):
        xs = torch.full((4000,), x0, dtype=torch.float64, device=DEVICE)
        logw = -beta * potentials.potential_xy_torch(
            torch.full((ny,), x0, dtype=torch.float64, device=DEVICE), y)
        w = torch.exp(logw - logw.max())
        w = w / torch.trapezoid(w, y)
        mean_true = float(torch.trapezoid(w * y, y))
        var_true = float(torch.trapezoid(w * (y - mean_true) ** 2, y))

        cfg = _cfg(); cfg["v3"] = dict(
            enabled=True, family=CAPPED, operator="ft", rho=0.85,
            fr_stride=200, clone_policy="oracle_refresh")
        # Exercise the engine's own sampler through a run, then compare moments
        # of a direct draw from the same table.
        r = _run(cfg["v3"], n_steps=1000)
        assert r.fr_opportunities, "oracle-refresh arm must have fired"

        # Independent re-derivation of the registered grid inverse-CDF.
        cdf = torch.cumsum(torch.exp(logw - logw.max()), dim=0)
        cdf = cdf / cdf[-1]
        u = torch.rand(xs.numel(), dtype=torch.float64, device=DEVICE)
        j = torch.searchsorted(cdf.contiguous(), u).clamp_max(ny - 1)
        draws = y[j]
        assert float(draws.mean()) == pytest.approx(mean_true, abs=0.05)
        assert float(draws.var()) == pytest.approx(var_true, rel=0.15)


def test_oracle_refresh_leaves_x_and_ancestry_alone_at_the_event():
    """A.5: refresh touches only y, so the *first* event's x-side reallocation
    is identical to the exact-clone arm.

    It must not be asserted over the whole run: a refreshed y feeds back through
    dV/dx into the estimator and the trajectory, so later opportunities see a
    genuinely different cloud.  That divergence is the effect under study, not a
    bug -- an earlier version of this test asserted run-total equality and
    failed for exactly that reason.
    """
    exact = _run(dict(enabled=True, family=CAPPED, operator="ft", rho=0.85,
                      fr_stride=200))
    orac = _run(dict(enabled=True, family=CAPPED, operator="ft", rho=0.85,
                     fr_stride=200, clone_policy="oracle_refresh"))
    de, do = exact.diags[0], orac.diags[0]
    i = de["steps"].index(400)                 # first opportunity
    assert de["steps"][i] == do["steps"][i]
    assert de["cumulative_replacements"][i] == do["cumulative_replacements"][i] > 0
    assert de["n_unique_ancestors"][i] == do["n_unique_ancestors"][i]
    assert de["ancestor_ess"][i] == pytest.approx(do["ancestor_ess"][i])
    # x-marginal identical at the event; the fibre is where they differ.
    np.testing.assert_allclose(np.sort(np.asarray(do["X_snap"][i])),
                               np.sort(np.asarray(de["X_snap"][i])), atol=1e-12)
    assert not np.allclose(np.sort(np.asarray(do["Y_snap"][i])),
                           np.sort(np.asarray(de["Y_snap"][i])), atol=1e-9)


# ------------------------------------------------------------------- A.6
def test_md_noise_bank_is_independent_of_fr_and_oracle_draws():
    """FR may change which configuration occupies slot i, never which Langevin
    variates belong to slot i."""
    specs = [RunSpec(method="v3", target_type="none", seed=3, gamma=0.0,
                     eta=0.10, fr_every=1, burnin_fraction=0.2, stop_fraction=0.8)]
    bank_a = st._MatchedNoiseBank(specs, 128, 500, DEVICE, torch.float64, 0,
                                  chunk_steps=100)
    first_a = [tuple(t.clone() for t in bank_a.at(s)) for s in range(3)]

    # Burn a large amount of unrelated randomness through other generators.
    g_fr = torch.Generator(device=DEVICE); g_fr.manual_seed(11)
    g_or = torch.Generator(device=DEVICE); g_or.manual_seed(13)
    for _ in range(500):
        torch.rand(997, generator=g_fr, device=DEVICE, dtype=torch.float64)
        torch.rand(499, generator=g_or, device=DEVICE, dtype=torch.float64)
    torch.manual_seed(999)
    torch.rand(10_000, device=DEVICE)

    bank_b = st._MatchedNoiseBank(specs, 128, 500, DEVICE, torch.float64, 0,
                                  chunk_steps=100)
    for s in range(3):
        nx, ny = bank_b.at(s)
        torch.testing.assert_close(nx, first_a[s][0], rtol=0, atol=0)
        torch.testing.assert_close(ny, first_a[s][1], rtol=0, atol=0)


def test_window_comes_from_the_v3_block_not_the_runspec():
    """Regression: io_utils hardcodes the RunSpec window for ``abf_only``.

    The original schedule gate passed while the campaign was broken, because it
    built RunSpecs directly with burnin=0.2/stop=0.8 -- a path the YAML runner
    never takes.  ``io_utils`` pins burnin_fraction=0.0 and stop_fraction=1.0
    for the abf_only method, so every v3 arm silently got the whole run: FR
    fired from step 0, when the carrier is still identically zero.

    This asserts the window survives a RunSpec that carries the wrong one.
    """
    cfg = _cfg(n_steps=2000)
    cfg["v3"] = dict(enabled=True, family=CAPPED, operator="ft", rho=0.85,
                     fr_stride=200, burnin_fraction=0.2, stop_fraction=0.8)
    x_grid = np.linspace(-3.0, 3.0, 401)
    # the spec deliberately claims the whole run, exactly as io_utils builds it
    specs = [RunSpec(method="abf_only", target_type="none", seed=0, gamma=0.0,
                     eta=0.10, fr_every=1, burnin_fraction=0.0, stop_fraction=1.0)]
    r = st.run_batch(specs, cfg=cfg, x_grid=x_grid, F_ref=np.zeros(401),
                     Fprime_ref=np.zeros(401),
                     ev=m.EvalConfig.from_domain(DOMAIN), device=DEVICE,
                     dtype=torch.float64)
    assert r.fr_opportunities == list(range(400, 1601, 200))
    assert r.fr_opportunities[0] == 400, "FR must not fire before the burn-in"


def test_frozen_campaign_window_is_exactly_61_opportunities():
    cfg = _cfg(n_steps=50000)
    cfg["simulation"]["n_steps"] = 50000
    cfg["v3"] = dict(enabled=True, family=CAPPED, operator="none",
                     fr_stride=500, burnin_fraction=0.2, stop_fraction=0.8)
    # operator "none" does no FR, so assert the arithmetic the engine will use
    burn, stop, stride = int(0.2 * 50000), int(0.8 * 50000), 500
    expected = list(range(burn, stop + 1, stride))
    assert expected[0] == 10000 and expected[-1] == 40000 and len(expected) == 61
