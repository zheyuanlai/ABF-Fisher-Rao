"""Oracle fibre refresh: the invariants the transport-refresh campaign rests on.

Prereg: configs/transport_campaign/gateway_transport_refresh_prereg.json.  The refresh is a
causal intervention -- after any FR gather and any transport, every walker's y is redrawn from
the EXACT conditional at its current x -- and it must (1) leave every non-refresh arm bit for
bit untouched, (2) draw exactly the conditional, from its own stream, shared across the arms of a
row, (3) act last, on y only, and (4) be refused on shams and on arms that hide it in their name.
"""
from __future__ import annotations

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


ABF_R = gw.with_refresh(gw.ABF)
FR_R = gw.with_refresh(gw.FR_UNIFORM)


def test_refresh_arms_leave_the_legacy_arms_bit_identical():
    """The refresh stream is its own generator: abf and fr_uniform must not see it."""
    fx = np.load(FIX)
    out = _run([gw.ABF, gw.FR_UNIFORM, ABF_R, FR_R, gw.with_refresh(gw.horizontal_ot(0.3))],
               store_profiles=True, store_accumulators=True)
    # abf never depends on batch composition; fr_uniform in a 5-arm batch is a different FR
    # realisation (its draws depend on R), so pin it against a 5-arm batch WITHOUT refresh.
    _assert_fixture(out, "two", "abf", fx)
    ref = _run([gw.ABF, gw.FR_UNIFORM, gw.horizontal_ot(0.01), gw.horizontal_ot(0.02), gw.horizontal_ot(0.3)],
               store_profiles=True, store_accumulators=True)
    for k in KEYS:
        assert np.array_equal(out["fr_uniform"][k], ref["fr_uniform"][k]), k
    assert out["abf_refresh"]["n_refresh_apply"] == 250 and out["abf"]["n_refresh_apply"] == 0
    assert not np.array_equal(out["abf"]["l2_f_t"], out["abf_refresh"]["l2_f_t"])   # it acts


def test_refresh_draws_exactly_the_conditional_from_its_own_stream():
    """Replay the generator: after one opportunity, y == z / (sqrt(beta) omega(x)) exactly."""
    cfg = _cfg(n_steps=1, save_every=1)
    out = _run([gw.ABF, ABF_R, gw.with_refresh(gw.horizontal_ot(1.0))], cfg=cfg,
               store_final_state=True, batch_seed=77)
    gen = torch.Generator(device=CPU); gen.manual_seed(4000 + 77)
    z = torch.randn((1, 256), dtype=torch.float64, generator=gen)[0].numpy()
    for name in ("abf_refresh", "ot_1_refresh"):
        X, Y = out[name]["X_final"], out[name]["Y_final"]
        om = 1.0 + 31.0 * np.exp(-X ** 2 / (2 * 0.1 ** 2))
        np.testing.assert_allclose(Y, z / (np.sqrt(16.0) * om), rtol=1e-13, atol=0)   # torch vs numpy exp: 1 ulp
    # x is untouched by the refresh: abf_refresh has abf's x, ot_1_refresh has the quantiles
    assert np.array_equal(out["abf_refresh"]["X_final"], out["abf"]["X_final"])
    u = gw.uniform_quantiles(256, CPU, torch.float64).numpy()[0]
    assert np.array_equal(np.sort(out["ot_1_refresh"]["X_final"]), u)
    # the same z across the arms of the row: refresh arms are paired with each other
    np.testing.assert_allclose(
        out["abf_refresh"]["Y_final"] * (1.0 + 31.0 * np.exp(-out["abf_refresh"]["X_final"] ** 2 / 0.02)),
        out["ot_1_refresh"]["Y_final"] * (1.0 + 31.0 * np.exp(-out["ot_1_refresh"]["X_final"] ** 2 / 0.02)), rtol=1e-12)


def test_refresh_acts_after_fr_and_after_transport_on_y_only():
    """Ordering: FR gather -> transport -> refresh.  Under refresh, FR's clones get INDEPENDENT y
    (fibre rejuvenation) while their x and ancestry are FR's; alpha = 0 transport + refresh is
    ABF + refresh bit for bit (the transport block did nothing, the refresh block did the same)."""
    out = _run([gw.ABF, gw.FR_UNIFORM, FR_R, ABF_R, gw.with_refresh(gw.horizontal_ot(0.0), name="ot0_refresh")],
               store_profiles=True, store_accumulators=True, store_final_state=True)
    # state and accumulators bit for bit; the smoothed profiles to 1e-12 only, because CPU conv1d
    # rounds the reflect-padded edge differently by ROW POSITION (1 ulp, identical inputs)
    for k in ("X_final", "Y_final", "Sf_t", "C_t", "P_regions", "ess_t"):
        assert np.array_equal(out["abf_refresh"][k], out["ot0_refresh"][k]), k
    for k in ("Fp_hat", "F_hat", "l2_f_t", "l2_fp_t", "kl_uniform_t"):
        np.testing.assert_allclose(out["abf_refresh"][k], out["ot0_refresh"][k], rtol=0, atol=1e-12, err_msg=k)
    assert out["fr_uniform_refresh"]["n_die"] > 0 and out["fr_uniform_refresh"]["n_refresh_apply"] == 250
    assert out["fr_uniform_refresh"]["final_ess"] < 256           # ancestry is still FR's
    assert not np.array_equal(out["fr_uniform_refresh"]["l2_f_t"], out["fr_uniform"]["l2_f_t"])


def test_refresh_keeps_the_conditional_at_its_floor():
    """D_cond of abf_refresh must not exceed abf's (it can only equilibrate the fibre)."""
    out = _run([gw.ABF, ABF_R], cfg=_cfg(N=2048), store_conditional=True)
    a, r = np.nanmean(out["abf"]["dcond_t"][1:]), np.nanmean(out["abf_refresh"]["dcond_t"][1:])
    assert r <= a * 1.05, (a, r)


def test_refresh_is_refused_where_it_would_hide():
    with pytest.raises(AssertionError):
        gw.assert_no_oracle_leakage([gw.Method("abf_oracle_y", False, "none", refresh="oracle")])
    with pytest.raises(AssertionError):
        gw.assert_no_oracle_leakage([gw.FR_ORACLE, gw.with_refresh(gw.SHAM_ORACLE)])
    with pytest.raises(AssertionError):
        gw.assert_no_oracle_leakage([gw.Method("x_refresh", False, "none", refresh="gibbs")])
    gw.assert_no_oracle_leakage([gw.ABF, ABF_R, FR_R, gw.with_refresh(gw.horizontal_ot(1.0))])
    assert gw.with_refresh(gw.horizontal_ot(0.0035, name="ot_exact")).name == "ot_exact_refresh"
