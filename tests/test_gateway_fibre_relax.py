"""Finite fibre relaxation (exact constrained OU propagation): the invariants of the relax campaign.

Prereg: configs/transport_campaign/gateway_fibre_relax_prereg.json.  ``refresh == 'ou'`` must be
(1) the identity at c = 0, bit for bit, (2) the oracle refresh in the limit c -> inf, (3) the exact
OU propagator at one opportunity (generator replay), (4) obey the variance-contraction law
e^{-2c} (the preregistered dose-response), (5) leave every legacy and oracle arm bit-identical,
(6) record the notional constrained-MD cost and the displacement-weighted relaxation time.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import gateway_core as gw  # noqa: E402

CPU = torch.device("cpu")
FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "gateway_pre_transport_fixture.npz")
STATE = ("X_final", "Y_final", "Sf_t", "C_t", "P_regions", "ess_t")
PROF = ("Fp_hat", "F_hat", "l2_f_t", "l2_fp_t", "kl_uniform_t")


def _cfg(**kw):
    base = dict(beta=16.0, H=0.5, s=0.10, r=32.0, N=256, n_steps=2500, save_every=250, dt=4e-4, gamma=1.5)
    base.update(kw)
    return gw.GatewayConfig(**base)


def _run(methods, cfg=None, seed=0, batch_seed=12345, **kw):
    spec = gw.BatchSpec(configs=[cfg or _cfg()], seeds=[seed], methods=methods, batch_seed=batch_seed)
    return {r["method"]: r for r in gw.simulate_batch(spec, device=CPU, **kw)}


OT = gw.horizontal_ot(0.00325, name="ot_exact")


def test_c_zero_is_the_identity_bit_for_bit():
    out = _run([gw.ABF, gw.with_relax(gw.ABF, 0.0), OT, gw.with_relax(OT, 0.0)],
               store_profiles=True, store_accumulators=True, store_final_state=True)
    for a, b in (("abf", "abf_relax0"), ("ot_exact", "ot_exact_relax0")):
        for k in STATE:
            assert np.array_equal(out[a][k], out[b][k]), (a, b, k)
        for k in PROF:   # CPU conv1d rounds the padded edge by row position (1 ulp)
            np.testing.assert_allclose(out[a][k], out[b][k], rtol=0, atol=1e-12, err_msg=k)
    assert out["abf_relax0"]["n_relax_apply"] == 250 and out["abf_relax0"]["fibre_cost_ratio"] == 0.0


def test_large_c_is_the_oracle_refresh_and_inf_maps_to_it():
    assert gw.with_relax(gw.ABF, float("inf")).refresh == "oracle"
    assert gw.with_relax(gw.ABF, float("inf")).name == "abf_refresh"
    out = _run([gw.with_refresh(gw.ABF), gw.with_relax(gw.ABF, 50.0)], cfg=_cfg(n_steps=1, save_every=1),
               store_final_state=True)
    np.testing.assert_allclose(out["abf_relax50"]["Y_final"], out["abf_refresh"]["Y_final"], rtol=1e-12)
    assert np.array_equal(out["abf_relax50"]["X_final"], out["abf_refresh"]["X_final"])


def test_ou_update_is_the_exact_propagator_at_one_opportunity():
    """Replay the shared draw: Y' = e^{-c} Y + sqrt((1 - e^{-2c}) / (beta omega^2)) z, with Y the
    ABF row's y (same Langevin step, no update) and omega at the walker's current x."""
    c = 1.0
    out = _run([gw.ABF, gw.with_relax(gw.ABF, c)], cfg=_cfg(n_steps=1, save_every=1), batch_seed=77,
               store_final_state=True)
    gen = torch.Generator(device=CPU); gen.manual_seed(4000 + 77)
    z = torch.randn((1, 256), dtype=torch.float64, generator=gen)[0].numpy()
    X, Y0 = out["abf"]["X_final"], out["abf"]["Y_final"]
    om = 1.0 + 31.0 * np.exp(-X ** 2 / (2 * 0.1 ** 2))
    expect = math.exp(-c) * Y0 + np.sqrt((1 - math.exp(-2 * c)) / (16.0 * om ** 2)) * z
    np.testing.assert_allclose(out["abf_relax1"]["Y_final"], expect, rtol=1e-12)
    assert np.array_equal(out["abf_relax1"]["X_final"], X)          # x untouched


@pytest.mark.parametrize("c", [0.5, 1.0, 2.0])
def test_variance_mismatch_contracts_by_exp_minus_2c(c):
    """Eq. (2) of the prereg: Var(Y') - sigma_x^2 = e^{-2c} (Var(Y) - sigma_x^2), on the pure function."""
    rng = np.random.default_rng(0)
    beta, om = 16.0, 20.0
    sig2 = 1.0 / (beta * om ** 2)
    Y = torch.tensor(rng.normal(0, 1 / np.sqrt(beta), (1, 400_000)))       # stale: from omega = 1
    z = torch.tensor(rng.normal(size=(1, 400_000)))
    Yn = gw.ou_relax(Y, torch.tensor(om), torch.tensor(beta), torch.tensor(c), z)
    excess = (float(Yn.var()) - sig2) / sig2
    predicted = math.exp(-2 * c) * ((1 / beta) - sig2) / sig2
    assert abs(excess / predicted - 1) < 0.03, (c, excess, predicted)


def test_relax_arms_leave_legacy_and_oracle_arms_bit_identical():
    fx = np.load(FIX)
    with_relax = _run([gw.ABF, gw.FR_UNIFORM, gw.with_refresh(gw.ABF), gw.with_relax(gw.ABF, 1.0), gw.with_relax(OT, 0.5)],
                      store_profiles=True, store_accumulators=True)
    without = _run([gw.ABF, gw.FR_UNIFORM, gw.with_refresh(gw.ABF), gw.horizontal_ot(0.01), gw.horizontal_ot(0.02)],
                   store_profiles=True, store_accumulators=True)
    for m in ("abf", "fr_uniform", "abf_refresh"):        # same R, so even the FR draws coincide
        for k in ("l2_f_t", "F_hat", "Fp_hat", "Sf_t", "C_t"):
            assert np.array_equal(with_relax[m][k], without[m][k]), (m, k)
    for k in ("l2_f_t", "F_hat", "Sf_t", "C_t"):
        assert np.array_equal(with_relax["abf"][k], fx[f"two/abf/{k}"]), k
    assert not np.array_equal(with_relax["abf_relax1"]["l2_f_t"], with_relax["abf"]["l2_f_t"])   # it acts


def test_cost_and_tau_move_are_recorded():
    out = _run([gw.ABF, gw.with_relax(gw.ABF, 1.0), gw.with_relax(OT, 1.0)])
    N, dt, n_ev = 256, 4e-4, 250
    for m in ("abf_relax1", "ot_exact_relax1"):
        tot = out[m]["fibre_steps_total"]
        assert n_ev * N / (32.0 ** 2 * dt) <= tot <= n_ev * N / dt          # between all-gate and all-basin
        assert abs(out[m]["fibre_cost_ratio"] - tot / (N * 2500)) < 1e-9
        assert np.isclose(out[m]["fibre_steps_t"].sum(), tot)
    assert "fibre_cost_ratio" not in out["abf"] or out["abf"]["fibre_cost_ratio"] == 0.0
    tm = out["ot_exact_relax1"]["ot_tau_move_t"][1:]
    assert np.all(tm > 1 / 1024 - 1e-12) and np.all(tm <= 1.0 + 1e-12)      # between tau_y(0) and tau_y(basin)
    assert np.all(out["abf_relax1"]["ot_tau_move_t"] == 0.0)


def test_refusals():
    with pytest.raises(AssertionError):
        gw.assert_no_oracle_leakage([gw.Method("abf_ou", False, "none", refresh="ou", relax_c=1.0)])
    with pytest.raises(AssertionError):
        gw.assert_no_oracle_leakage([gw.Method("abf_relax_bad", False, "none", refresh="ou", relax_c=-1.0)])
    with pytest.raises(AssertionError):
        gw.assert_no_oracle_leakage([gw.FR_ORACLE, gw.with_relax(gw.SHAM_ORACLE, 1.0)])
    gw.assert_no_oracle_leakage([gw.ABF, gw.with_relax(gw.ABF, 0.5), gw.with_relax(gw.FR_UNIFORM, 2.0),
                                 gw.with_relax(OT, 5.0), gw.with_refresh(OT)])
