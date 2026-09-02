"""Horizontal-transport arm: the invariants the transport campaign's conclusions rest on.

Prereg: configs/transport_campaign/gateway_horizontal_transport_prereg.json.  The accepted
engine (commit abcfaaf, which produced the gateway and WCA corrected-baseline confirmations)
is pinned by a fixture generated BEFORE the transport code existed; every legacy path must
reproduce it bit for bit, and the transport arm must be exactly what the prereg says it is:
x only, rank-matched, no RNG, no reference, no population change, no same-step deposit.
"""
from __future__ import annotations

import inspect
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import gateway_core as gw  # noqa: E402

CPU = torch.device("cpu")
FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures",
                   "gateway_pre_transport_fixture.npz")
KEYS = ("l2_f_t", "l2_fp_t", "F_hat", "Fp_hat", "ess_t", "P_regions", "Sf_t", "C_t",
        "kl_uniform_t")


def _cfg(**kw):
    base = dict(beta=16.0, H=0.5, s=0.10, r=32.0, N=256, n_steps=2500, save_every=250,
                dt=4e-4, gamma=1.5)
    base.update(kw)
    return gw.GatewayConfig(**base)


def _run(methods, cfg=None, seed=0, batch_seed=12345, **kw):
    spec = gw.BatchSpec(configs=[cfg or _cfg()], seeds=[seed], methods=methods,
                        batch_seed=batch_seed)
    return {r["method"]: r for r in gw.simulate_batch(spec, device=CPU, **kw)}


def _assert_fixture(out, tag, method, fx):
    for k in KEYS:
        assert np.array_equal(out[method][k], fx[f"{tag}/{method}/{k}"]), (tag, method, k)
    counts = np.array([out[method]["n_die"], out[method]["n_clone"], out[method]["n_fr_apply"]])
    assert np.array_equal(counts, fx[f"{tag}/{method}/counts"]), (tag, method)


# ------------------------------------------------ 1, 2: legacy paths are untouched
def test_legacy_abf_and_fr_uniform_reproduce_the_accepted_engine_bit_for_bit():
    fx = np.load(FIX)
    out = _run([gw.ABF, gw.FR_UNIFORM], store_profiles=True, store_accumulators=True)
    for m in ("abf", "fr_uniform"):
        _assert_fixture(out, "two", m, fx)


def test_five_arm_confirmatory_batch_reproduces_the_accepted_engine_bit_for_bit():
    fx = np.load(FIX)
    out = _run([gw.ABF, gw.FR_ORACLE, gw.FR_ESTIMATED, gw.SHAM_ORACLE, gw.SHAM_PRACTICAL],
               store_profiles=True, store_accumulators=True)
    for m in ("abf", "fr_oracle", "fr_estimated", "sham_oracle", "sham_practical"):
        _assert_fixture(out, "five", m, fx)


def test_report_only_flags_are_bit_inert():
    fx = np.load(FIX)
    out = _run([gw.ABF, gw.FR_UNIFORM], store_profiles=True, store_accumulators=True,
               store_conditional=True, store_final_state=True)
    for m in ("abf", "fr_uniform"):
        _assert_fixture(out, "two", m, fx)
    assert np.isfinite(out["abf"]["dcond_t"]).all() and out["abf"]["X_final"].shape == (256,)


# ------------------------------------------------ 3: alpha = 0 is ABF
def test_alpha_zero_transport_is_abf_bit_for_bit():
    out = _run([gw.ABF, gw.horizontal_ot(0.0), gw.horizontal_ot(0.3)],
               store_profiles=True, store_accumulators=True)
    for k in KEYS:
        assert np.array_equal(out["abf"][k], out["ot_0"][k]), k
    assert out["ot_0"]["n_ot_apply"] == 250          # it fired at every opportunity, moving nothing
    # non-vacuous: a real strength does change the run
    assert not np.array_equal(out["abf"]["l2_f_t"], out["ot_0.3"]["l2_f_t"])


# ------------------------------------------------ 9 (batch level): ABF rows do not see OT arms
def test_adding_transport_arms_leaves_the_abf_row_bit_identical():
    """The Langevin noise is drawn per (config, seed) and shared by every arm of that row, so an
    ABF row is the same trajectory whatever else sits in the batch -- which is what pairs the
    transport arms to the same baseline the FR arm was measured against."""
    fx = np.load(FIX)
    out = _run([gw.ABF, gw.horizontal_ot(1.0), gw.horizontal_ot(0.05)],
               store_profiles=True, store_accumulators=True)
    _assert_fixture(out, "two", "abf", fx)


# ------------------------------------------------ 4, 5, 6, 7: one event, engine level
@pytest.mark.parametrize("alpha", [1.0, 0.3])
def test_one_event_moves_x_to_the_interpolated_quantiles_and_leaves_y_untouched(alpha):
    # n_steps = 1: ramp = int(0.1) = 0 -> ramp factor 1, so the step-0 opportunity acts at
    # full alpha.  The ABF row IS the pre-event state (same noise, no transport).
    cfg = _cfg(n_steps=1, save_every=1)
    out = _run([gw.ABF, gw.horizontal_ot(alpha)], cfg=cfg, store_final_state=True)
    Xa, Ya = out["abf"]["X_final"], out["abf"]["Y_final"]
    Xo, Yo = out[f"ot_{alpha:g}"]["X_final"], out[f"ot_{alpha:g}"]["Y_final"]
    assert np.array_equal(Ya, Yo)                                    # Y+ == Y-, exactly
    assert Xo.shape == Xa.shape == (256,)                            # N+ == N-
    u = gw.uniform_quantiles(256, CPU, torch.float64).numpy()[0]
    expect = (1.0 - alpha) * np.sort(Xa) + alpha * u
    if alpha == 1.0:
        assert np.array_equal(np.sort(Xo), u)                        # exact finite-N uniform
    np.testing.assert_allclose(np.sort(Xo), expect, rtol=0, atol=1e-15)
    # every walker keeps its rank, i.e. its identity (and its y)
    assert np.array_equal(np.argsort(Xo, kind="stable"), np.argsort(Xa, kind="stable"))
    assert not np.array_equal(Xo, Xa)


# ------------------------------------------------ 10: no same-step deposit
def test_transported_positions_enter_the_accumulators_only_on_the_next_step():
    out1 = _run([gw.ABF, gw.horizontal_ot(1.0)], cfg=_cfg(n_steps=1, save_every=1),
                store_accumulators=True, store_final_state=True)
    assert np.array_equal(out1["abf"]["C_t"], out1["ot_1"]["C_t"])
    assert np.array_equal(out1["abf"]["Sf_t"], out1["ot_1"]["Sf_t"])   # step-0 deposit predates the event
    assert not np.array_equal(out1["abf"]["X_final"], out1["ot_1"]["X_final"])   # yet x moved
    out2 = _run([gw.ABF, gw.horizontal_ot(1.0)], cfg=_cfg(n_steps=2, save_every=1),
                store_accumulators=True)
    assert np.array_equal(out2["abf"]["C_t"][0], out2["ot_1"]["C_t"][0])
    assert not np.array_equal(out2["abf"]["C_t"][1], out2["ot_1"]["C_t"][1])   # next step sees them


# ------------------------------------------------ 8, 9: the operator is pure
def test_transport_map_is_pure_no_reference_no_rng_no_population_change():
    assert list(inspect.signature(gw.horizontal_ot_map).parameters) == ["X", "alpha_t", "u"]
    src = inspect.getsource(gw.simulate_batch)
    end = src.index("if do_fr and any_refresh:") if "if do_fr and any_refresh:" in src else src.index("X, Y = Xp, Yp")
    block = src[src.index("if do_fr and any_ot:"): end]          # the transport block only
    for forbidden in ("F_ref", "Fp_ref", "F_target", "Bbias", "gen_", "rand", "Yp", "Sf",
                      "C.", "interp1d", "binned_density"):
        assert forbidden not in block, forbidden
    torch.manual_seed(3)
    X = torch.rand((4, 50), dtype=torch.float64) * 3.6 - 1.8
    s0 = torch.get_rng_state().clone()
    u = gw.uniform_quantiles(50, CPU, torch.float64)
    Xn = gw.horizontal_ot_map(X, 0.4, u)
    assert torch.equal(torch.get_rng_state(), s0)                    # no random number drawn
    assert Xn.shape == X.shape
    Xs, order = torch.sort(X, dim=1, stable=True)
    assert torch.equal(torch.gather(Xn, 1, order), 0.6 * Xs + 0.4 * u)
    assert torch.equal(gw.horizontal_ot_map(X, 1.0, u).sort(dim=1).values, u.expand(4, 50))
    assert torch.equal(gw.horizontal_ot_map(X, 0.0, u), X)
    assert torch.equal(torch.argsort(Xn, dim=1, stable=True), order)  # ranks preserved
    # the quantiles are the exact finite-N uniform: mean XMIN+XMAX over 2, spacing L/N
    assert abs(float(u.mean())) < 1e-14 and abs(float(u[0, 1] - u[0, 0]) - 3.6 / 50) < 1e-14


# ------------------------------------------------ schedule, ramp, no genealogy
def test_transport_fires_on_the_fr_schedule_with_the_fr_ramp():
    out = _run([gw.ABF, gw.FR_UNIFORM, gw.horizontal_ot(0.5)])
    ot = out["ot_0.5"]
    assert ot["n_ot_apply"] == out["fr_uniform"]["n_fr_apply"] == 250
    steps = np.array([250 * k for k in range(10)] + [2499])
    last_event = (steps // 10) * 10
    expect = 0.5 * (1.0 - np.exp(-last_event / int(0.1 * 2500)))
    np.testing.assert_allclose(ot["alpha_t"], expect, rtol=0, atol=1e-12)
    assert ot["alpha_t"][0] == 0.0          # the step-0 opportunity moves nothing, like the FR rate
    assert ot["n_die"] == 0 == ot["n_clone"] and ot["min_ess_frac"] == 1.0 and ot["max_wmax"] == 1 / 256
    assert (ot["ot_absdx_t"][1:] > 0).all() and (ot["dmove_max_t"][1:] > 0).all()
    assert out["abf"]["n_ot_apply"] == 0 and np.all(out["abf"]["ot_absdx_t"] == 0)


# ------------------------------------------------ diagnostics
def test_d_move_is_the_gaussian_kl_and_vanishes_for_a_null_move():
    rng = np.random.default_rng(0)
    om_old, om_new = torch.tensor(rng.uniform(1, 32, 100)), torch.tensor(rng.uniform(1, 32, 100))
    beta = 16.0
    s_old, s_new = 1 / (np.sqrt(beta) * om_old), 1 / (np.sqrt(beta) * om_new)
    kl = torch.log(s_new / s_old) + s_old ** 2 / (2 * s_new ** 2) - 0.5
    np.testing.assert_allclose(gw.d_move(om_old, om_new).numpy(), kl.numpy(), rtol=1e-9, atol=1e-14)
    assert torch.all(gw.d_move(om_old, om_old) == 0) and torch.all(gw.d_move(om_old, om_new) >= 0)


def test_conditional_moment_kl_is_small_at_equilibrium_and_large_when_y_is_stale():
    rng = np.random.default_rng(1)
    N = 20000
    X = torch.tensor(rng.uniform(-1.8, 1.8, (1, N)))
    beta, oout, oin, sw = (torch.tensor([[v]], dtype=torch.float64) for v in (16.0, 1.0, 32.0, 0.1))
    om = gw.omega_of(X, oout, oin, sw)
    Z = torch.tensor(rng.normal(size=(1, N)))
    d_eq, nb = gw.conditional_moment_kl(X, Z / (torch.sqrt(beta) * om), beta, oout, oin, sw)
    assert int(nb) == gw.COND_BINS and float(d_eq) < 0.01
    # y carried from omega = 1 everywhere: 32x too wide inside the gate
    d_stale, _ = gw.conditional_moment_kl(X, Z / torch.sqrt(beta), beta, oout, oin, sw)
    assert float(d_stale) > 1.0
    # too few walkers per bin -> NaN, not a silently reassuring zero
    d_none, nb0 = gw.conditional_moment_kl(X[:, :100], Z[:, :100], beta, oout, oin, sw, min_count=1000)
    assert int(nb0) == 0 and np.isnan(float(d_none))


def test_transport_arm_refuses_to_also_resample_or_carry_a_target():
    with pytest.raises(AssertionError):
        gw.assert_no_oracle_leakage([gw.Method("bad", True, "none", transport="horizontal_ot", alpha=0.5)])
    with pytest.raises(AssertionError):
        gw.assert_no_oracle_leakage([gw.Method("bad", False, "uniform", transport="horizontal_ot", alpha=0.5)])
    with pytest.raises(AssertionError):
        gw.horizontal_ot(1.5)
    gw.assert_no_oracle_leakage([gw.ABF, gw.FR_UNIFORM, gw.horizontal_ot(0.02), gw.horizontal_ot(1.0)])
