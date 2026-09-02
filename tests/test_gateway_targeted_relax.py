"""Targeted, budgeted fibre relaxation: the invariants of the targeted-relax campaign.

Prereg: configs/transport_campaign/gateway_targeted_relax_prereg.json.  (1) The second-moment
accumulator is bit-inert for every other arm; (2) the online sensitivity profile is what the
offline formula gives and the analytic reference is Var(f | x); (3) the water-filling allocation
spends exactly the budget and satisfies the KKT conditions; (4) a targeted arm spends rho times
the outer cost, a zero budget is the identity bit for bit, and other arms are untouched.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import gateway_core as gw  # noqa: E402
from analyze_gateway_bandwidth_audit import eb_smooth  # noqa: E402

CPU = torch.device("cpu")
FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "gateway_pre_transport_fixture.npz")
OT = gw.horizontal_ot(0.00325, name="ot_exact")


def _cfg(**kw):
    base = dict(beta=16.0, H=0.5, s=0.10, r=32.0, N=256, n_steps=2500, save_every=250, dt=4e-4, gamma=1.5)
    base.update(kw)
    return gw.GatewayConfig(**base)


def _run(methods, cfg=None, seed=0, batch_seed=12345, **kw):
    spec = gw.BatchSpec(configs=[cfg or _cfg()], seeds=[seed], methods=methods, batch_seed=batch_seed)
    return {r["method"]: r for r in gw.simulate_batch(spec, device=CPU, **kw)}


def test_second_moment_accumulator_is_bit_inert_and_consistent():
    fx = np.load(FIX)
    out = _run([gw.ABF, gw.FR_UNIFORM], store_profiles=True, store_accumulators=True)
    for m in ("abf", "fr_uniform"):
        for k in ("l2_f_t", "F_hat", "Fp_hat", "Sf_t", "C_t", "kl_uniform_t"):
            assert np.array_equal(out[m][k], fx[f"two/{m}/{k}"]), (m, k)
    Sf2, Sf, C = out["abf"]["Sf2_t"][-1], out["abf"]["Sf_t"][-1], out["abf"]["C_t"][-1]
    ok = C > 0
    assert np.all(Sf2[ok] * C[ok] >= Sf[ok] ** 2 * (1 - 1e-12))      # Cauchy-Schwarz: E[f^2] >= E[f]^2
    assert Sf2.sum() > 0


def test_online_vhat_matches_the_offline_formula_and_the_analytic_reference_shape():
    out = _run([gw.ABF], cfg=_cfg(N=2048, n_steps=5000, save_every=1000), store_accumulators=True)
    r = out["abf"]
    x = r["x_grid"]; dx = float(x[1] - x[0])
    kt, rt = gw.gaussian_kernel(0.07, dx, CPU, torch.float64)
    v_on = gw.vhat_from(torch.tensor(r["Sf2_t"][-1:]), torch.tensor(r["Sf_t"][-1:]), torch.tensor(r["C_t"][-1:]), kt, rt, dx, 1.0).numpy()[0]
    den = eb_smooth(r["C_t"][-1:], 0.07, dx) + 1.0 + gw.EPS
    v_off = np.clip(eb_smooth(r["Sf2_t"][-1:], 0.07, dx) / den - (eb_smooth(r["Sf_t"][-1:], 0.07, dx) / den) ** 2, 0, None)[0]
    np.testing.assert_allclose(v_on, v_off, rtol=1e-10, atol=1e-14)
    assert v_on.min() >= 0
    # the online field peaks where the analytic sensitivity peaks (the flank), not in the basin
    v_ref = gw.sensitivity_ref(torch.tensor(x), torch.tensor(16.0), torch.tensor(1.0), torch.tensor(32.0), torch.tensor(0.1)).numpy()
    sampled = r["C_t"][-1] > 50
    xs = x[sampled]
    assert abs(xs[np.argmax(v_on[sampled])]) < 0.35 and abs(x[np.argmax(v_ref)]) < 0.35


def test_sensitivity_ref_is_the_sampled_conditional_variance():
    rng = np.random.default_rng(0)
    beta, oout, oin, sw = 16.0, 1.0, 32.0, 0.1
    for x in (0.1, 0.2, 0.3):
        om = oout + (oin - oout) * np.exp(-x ** 2 / (2 * sw ** 2)); dom = -(oin - oout) * (x / sw ** 2) * np.exp(-x ** 2 / (2 * sw ** 2))
        y = rng.normal(0, 1 / np.sqrt(beta * om ** 2), 1_000_000)
        f = 4 * 0.5 * x * (x ** 2 - 1) + om * dom * y ** 2
        ref = float(gw.sensitivity_ref(torch.tensor(x), torch.tensor(beta), torch.tensor(oout), torch.tensor(oin), torch.tensor(sw)))
        assert abs(f.var() / ref - 1) < 0.03, (x, f.var(), ref)


def test_budgeted_relaxation_spends_the_budget_and_satisfies_kkt():
    rng = np.random.default_rng(1)
    a = torch.tensor(rng.uniform(0, 1, (3, 500))); a[0, :100] = 0.0; a[2, :] = 0.0       # row 2: nothing to do
    tau = torch.tensor(rng.uniform(1e-3, 1.0, (3, 500)))
    B = torch.tensor([[2.0], [0.5], [1.0]])
    c = gw.budgeted_relaxation(a, tau, B)
    assert torch.all(c >= 0)
    spend = (c * tau).sum(dim=1)
    assert abs(float(spend[0]) - 2.0) < 1e-6 and abs(float(spend[1]) - 0.5) < 1e-6 and float(spend[2]) == 0.0
    assert torch.all(c[0, :100] == 0)                                        # zero importance -> zero relaxation
    for row in (0, 1):
        act = c[row] > 0
        lam = (2 * a[row][act] * torch.exp(-2 * c[row][act]) / tau[row][act])   # KKT: common multiplier on the active set
        assert float(lam.max() / lam.min() - 1) < 1e-6
        r_inactive = (2 * a[row][~act] / tau[row][~act])
        assert float(r_inactive.max()) <= float(lam.mean()) * (1 + 1e-6)     # inactive walkers are below it
    # a zero budget relaxes nothing; a larger budget relaxes more, everywhere
    assert torch.all(gw.budgeted_relaxation(a, tau, torch.zeros((3, 1), dtype=torch.float64)) == 0)
    c2 = gw.budgeted_relaxation(a, tau, 2 * B)
    assert torch.all(c2 >= c - 1e-12)


def test_targeted_arm_spends_rho_and_leaves_other_arms_bit_identical():
    with_t = _run([gw.ABF, gw.FR_UNIFORM, gw.with_targeted(gw.ABF, 1.0), gw.with_targeted(OT, 0.5),
                   gw.with_targeted(OT, 0.5, "v_dx")], store_profiles=True, store_accumulators=True)
    without = _run([gw.ABF, gw.FR_UNIFORM, gw.horizontal_ot(0.01), gw.horizontal_ot(0.02), gw.horizontal_ot(0.3)],
                   store_profiles=True, store_accumulators=True)
    for m in ("abf", "fr_uniform"):
        for k in ("l2_f_t", "F_hat", "Sf_t", "C_t"):
            assert np.array_equal(with_t[m][k], without[m][k]), (m, k)
    for name, rho in (("abf_targ1", 1.0), ("ot_exact_targ0.5", 0.5), ("ot_exact_targ0.5move", 0.5)):
        r = with_t[name]
        assert abs(r["fibre_cost_ratio"] - rho) < 0.02 * rho, (name, r["fibre_cost_ratio"])   # the budget is spent
        assert r["n_targ_apply"] == 250
        assert np.all((r["targ_flank_frac_t"] >= 0) & (r["targ_flank_frac_t"] <= 1))
        assert np.all(r["targ_active_frac_t"][1:] > 0) and np.all(r["targ_active_frac_t"] <= 1)
        assert not np.array_equal(r["l2_f_t"], with_t["abf" if name.startswith("abf") else "abf"]["l2_f_t"])
    assert np.all(with_t["ot_exact_targ0.5"]["ot_flank_dx_frac_t"][1:] >= 0)


def test_zero_budget_is_the_identity_bit_for_bit():
    out = _run([gw.ABF, gw.with_targeted(gw.ABF, 0.0), OT, gw.with_targeted(OT, 0.0)],
               store_accumulators=True, store_final_state=True)
    for a, b in (("abf", "abf_targ0"), ("ot_exact", "ot_exact_targ0")):
        for k in ("X_final", "Y_final", "Sf_t", "C_t"):
            assert np.array_equal(out[a][k], out[b][k]), (a, b, k)
    assert out["abf_targ0"]["fibre_cost_ratio"] == 0.0


def test_refusals():
    with pytest.raises(AssertionError):
        gw.assert_no_oracle_leakage([gw.with_targeted(gw.ABF, 1.0, "v_dx")])          # needs transport
    with pytest.raises(AssertionError):
        gw.assert_no_oracle_leakage([gw.Method("abf_budget", False, "none", refresh="targeted", budget_rho=1.0)])
    with pytest.raises(AssertionError):
        gw.assert_no_oracle_leakage([gw.with_targeted(gw.ABF, -1.0)])
    gw.assert_no_oracle_leakage([gw.ABF, gw.with_targeted(gw.ABF, 1.0), gw.with_targeted(gw.FR_UNIFORM, 0.25),
                                 gw.with_targeted(OT, 2.0), gw.with_targeted(OT, 2.0, "v_dx"), gw.with_relax(OT, 0.5)])


def test_binvar_estimator_removes_the_kernel_gradient_floor_and_is_opt_in():
    out = _run([gw.ABF], cfg=_cfg(N=2048, n_steps=5000, save_every=1000), store_accumulators=True)
    r = out["abf"]; x = r["x_grid"]; dx = float(x[1] - x[0])
    kt, rt = gw.gaussian_kernel(0.07, dx, CPU, torch.float64)
    args = (torch.tensor(r["Sf2_t"][-1:]), torch.tensor(r["Sf_t"][-1:]), torch.tensor(r["C_t"][-1:]), kt, rt, dx, 1.0)
    v1 = gw.vhat_from(*args).numpy()[0]; v2 = gw.vhat_from(*args, mode="binvar").numpy()[0]
    sampled = r["C_t"][-1] > 50
    basin = sampled & (abs(x) > 0.8)
    if basin.any():
        assert v2[basin].max() < 0.1 * max(v1[basin].max(), 1e-12) + 1e-6      # the (F'' h)^2 floor is gone
    assert np.array_equal(gw.vhat_from(*args).numpy(), gw.vhat_from(*args, mode="moments").numpy())   # default unchanged
    # opt-in arm naming and refusal
    assert gw.with_targeted(OT, 1.0, sensitivity="binvar").name == "ot_exact_targ1v2"
    with pytest.raises(AssertionError):
        gw.assert_no_oracle_leakage([gw.Method("abf_targ", False, "none", refresh="targeted", budget_rho=1.0, sensitivity="v3")])
