"""Targeted constrained solvent relaxation on the WCA dimer: the invariants of the campaign.

Prereg: configs/targeted_relax_campaign/wca_fr_targeted_relax_prereg.json.  The accepted engine
(commit d3fc93e) is pinned by tests/fixtures/wca_pre_relax_fixture.npz, generated BEFORE this
code existed.  Tiny CPU float64 dimer (tests/test_wca_sham.py's), so every comparison is exact.
"""
from __future__ import annotations

import inspect
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import wca_abffr_core as wca  # noqa: E402

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "wca_pre_relax_fixture.npz")
KEYS = ("mean_force", "pmf", "p_hat", "eff_counts", "ancestor_ess", "raw_fsum", "raw_csum", "birth_hist",
        "death_hist", "fr_event_counts")


def _tiny():
    params = wca.DimerWCAParams(n_dim=4, beta=1.0, h=1.0, w=2.0, a=1.5)
    sim = wca.SimConfig(n_replicas=64, n_steps=3000, save_every=1000, dt=2e-3, n_grid=48, fr_start_steps=500,
                        fr_every=25, abf_warmup_steps=200, estimator_burn_in_steps=200, fr_rate=0.5,
                        max_event_fraction=0.1, seed=7)
    engine = wca.WCADimerEngine(params, device=torch.device("cpu"), dtype=torch.float64)
    return params, sim, engine


def _relax(rho, sim, target="sensitivity", tau=0.05):
    return wca.RelaxConfig(rho=rho, tau_grid=tuple([tau] * sim.n_grid), target=target)


def _run(method, **kw):
    params, sim, engine = _tiny()
    return wca.run_sampler_gpu(method, params, sim, engine, collect_diagnostics=True, verbose=False,
                               readout_bandwidths=(0.025, 0.0), **kw)


def _same_as_fixture(d, m, fx):
    for k in KEYS:
        assert np.array_equal(np.asarray(d[k]), fx[f"{m}/{k}"], equal_nan=True), (m, k)
    assert np.array_equal(np.asarray(d["readout_mean_force"][0.025]), fx[f"{m}/readout_0.025"])
    assert d["total_replacement_events"] == int(fx[f"{m}/total_replacement_events"])


# 1, 2: legacy paths untouched
def test_legacy_abf_and_fr_uniform_are_bit_identical_to_the_accepted_engine():
    fx = np.load(FIX)
    _same_as_fixture(_run("abf"), "abf", fx)
    _same_as_fixture(_run("fr_uniform"), "fr_uniform", fx)


# 4: instrumentation inert
def test_sensitivity_instrumentation_is_inert():
    fx = np.load(FIX)
    d = _run("abf", sensitivity_record=True)
    _same_as_fixture(d, "abf", fx)
    assert d["vhat"].shape == (4, 48) and np.all(d["vhat"] >= 0) and d["final_q"].shape == (64, 16, 2)
    # the accumulator deposits once per outer step, all steps: C sums to N x (n_steps + 1)
    assert float(d["sens_C"][-1].sum()) == 64 * 3001
    # per-bin variance then smoothing: E[f^2] >= E[f]^2 holds bin-wise
    C, Sf, Sf2 = d["sens_C"][-1], d["sens_Sf"][-1], d["sens_Sf2"][-1]
    ok = C > 1
    assert np.all(Sf2[ok] * C[ok] >= Sf[ok] ** 2 * (1 - 1e-12))


# 3: rho = 0 is the identity
def test_zero_budget_is_the_identity_bit_for_bit():
    fx = np.load(FIX)
    _, sim, _ = _tiny()
    da = _run("abf", relax=_relax(0.0, sim))
    _same_as_fixture(da, "abf", fx)
    assert da["relax_steps_total"] == 0 and da["relax_cost_ratio"] == 0.0
    df = _run("fr_uniform", relax=_relax(0.0, sim))
    _same_as_fixture(df, "fr_uniform", fx)


# 5: no oracle / reference reachable from the targeting code
def test_targeting_code_has_no_reference_access():
    src = inspect.getsource(wca.run_sampler_gpu)
    block = src[src.index("if relax is not None:\n            # Targeted"): src.index('diag["runtime_seconds"]')]
    for forbidden in ("oracle", "reference", "ref[", "F_target_ema", "current_bias_profile", "ancestors", "anc_win",
                      "bias_estimator", "production_estimator", "readout.", "fr_event_counts"):
        assert forbidden not in block, forbidden
    assert set(wca.RelaxConfig.__dataclass_fields__) == {"rho", "tau_grid", "target", "inner_seed_offset", "min_count", "scheme"}
    for f in (wca.SensitivityAccumulator.update, wca.SensitivityAccumulator.profile, wca.water_filling_durations,
              wca.frozen_dimer_relax):
        assert "oracle" not in inspect.getsource(f) and "reference" not in inspect.getsource(f)


# 6, 7, 8: the inner relaxation moves only the solvent, deposits nothing, touches no ancestry
def test_frozen_dimer_relax_keeps_the_dimer_and_z_fixed_and_counts_steps_exactly():
    params, sim, engine = _tiny()
    q = wca.lattice_initial_conditions(params, 16, engine.device, engine.dtype, seed=3)
    gen = torch.Generator(device="cpu"); gen.manual_seed(1)
    m = torch.tensor([0, 1, 2, 5, 5, 5, 8, 0, 3, 3, 3, 1, 0, 0, 7, 2])
    q_new, done = wca.frozen_dimer_relax(engine, params, sim, q, m, gen)
    assert done == int(m.sum())                                                    # replica-steps counted exactly
    assert torch.equal(q_new[:, :2, :], q[:, :2, :])                                # dimer frozen bit for bit
    assert torch.equal(wca.reaction_coordinate(q_new, params), wca.reaction_coordinate(q, params))
    moved = (q_new[:, 2:, :] != q[:, 2:, :]).any(dim=(1, 2))
    assert torch.equal(moved, m > 0)                                                # exactly the active replicas moved


def test_inner_relaxation_deposits_nothing_and_leaves_ancestry_to_fr():
    _, sim, _ = _tiny()
    d0 = _run("abf", sensitivity_record=True)
    d1 = _run("abf", relax=_relax(1.0, sim), sensitivity_record=True)
    assert d1["relax_steps_total"] > 0 and d1["relax_n_opportunities"] > 0
    # one deposit per OUTER step in both runs, whatever the inner relaxation did
    assert float(d1["sens_C"][-1].sum()) == float(d0["sens_C"][-1].sum()) == 64 * 3001
    assert float(d1["raw_csum"][-1].sum()) == float(d0["raw_csum"][-1].sum())
    # the trajectories differ (relaxation acted) ...
    assert not np.array_equal(d1["mean_force"], d0["mean_force"])
    # ... and an FR arm's ancestry is only ever changed by birth-death
    df = _run("fr_uniform", relax=_relax(1.0, sim))
    assert np.isfinite(df["ancestor_ess"]).all() and df["total_replacement_events"] > 0


# 9, 10, 11, 12: the allocation
def test_water_filling_budget_monotonicity_zero_sensitivity_and_matched_random_control():
    rng = np.random.default_rng(0)
    a = torch.tensor(rng.uniform(0, 1, 500)); a[:100] = 0.0
    tau = torch.tensor(rng.uniform(0.01, 1.0, 500))
    dt = 2e-3
    for B in (5.0, 20.0):
        t = wca.water_filling_durations(a, tau, B)
        assert torch.all(t >= 0) and torch.all(t[:100] == 0)
        assert abs(float(t.sum()) - B) < 1e-6 * B                                   # spends exactly the budget
        m = wca.integer_steps_largest_remainder(t, dt, int(round(B / dt)))
        assert int(m.sum()) == int(round(B / dt)) and torch.all(m >= 0)              # integer budget met, never exceeded
        assert int(wca.integer_steps_largest_remainder(t, dt, 100).sum()) <= 100
    t1, t2 = wca.water_filling_durations(a, tau, 5.0), wca.water_filling_durations(a, tau, 10.0)
    assert torch.all(t2 >= t1 - 1e-12)                                              # more budget never relaxes anyone less
    assert torch.all(wca.water_filling_durations(torch.zeros(500, dtype=torch.float64), tau, 5.0) == 0)
    assert torch.all(wca.water_filling_durations(a, tau, 0.0) == 0)
    # the random control: same multiset of durations, permuted
    m = wca.integer_steps_largest_remainder(t1, dt, int(round(5.0 / dt)))
    gen = torch.Generator(); gen.manual_seed(5)
    mr = m[torch.randperm(500, generator=gen)]
    assert int(mr.sum()) == int(m.sum()) and torch.equal(torch.sort(mr).values, torch.sort(m).values)


def test_random_control_spends_the_same_budget_in_the_engine():
    _, sim, _ = _tiny()
    dt_ = _run("abf", relax=_relax(0.5, sim, target="sensitivity"))
    dr = _run("abf", relax=_relax(0.5, sim, target="random"))
    assert dt_["relax_budget_steps_per_opportunity"] == dr["relax_budget_steps_per_opportunity"] == int(round(0.5 * 64 * 25))
    assert dt_["relax_n_opportunities"] == dr["relax_n_opportunities"]
    # the budget is spent exactly whenever the sensitivity field is positive somewhere
    assert dt_["relax_steps_total"] == dr["relax_steps_total"] == dt_["relax_n_opportunities"] * dt_["relax_budget_steps_per_opportunity"]
    assert abs(dt_["relax_cost_ratio"] - 0.5 * dt_["relax_n_opportunities"] * 25 / 3000) < 1e-9
    assert not np.array_equal(dt_["relax_budget_hist"], dr["relax_budget_hist"])         # the association is destroyed


# 13: the bank still reproduces the production estimator with relaxation present
def test_readout_bank_reproduces_the_production_estimator_with_relaxation():
    _, sim, _ = _tiny()
    d = _run("fr_uniform", relax=_relax(0.5, sim))
    assert np.allclose(np.asarray(d["readout_mean_force"][0.025]), np.asarray(d["mean_force"]), rtol=0, atol=1e-12)


# the W0-B instrument
def test_constrained_force_series_keeps_z_fixed_and_autocorrelation_time_is_recovered():
    params, sim, engine = _tiny()
    q0 = wca.lattice_initial_conditions(params, 8, engine.device, engine.dtype, seed=11)
    gen = torch.Generator(); gen.manual_seed(2)
    f = wca.constrained_force_series(engine, params, sim, q0, 0.3, n_eq=50, n_prod=200, gen=gen, record_every=1)
    assert f.shape == (8, 200) and np.isfinite(f).all()
    # tau on a synthetic AR(1) with known correlation time
    rng = np.random.default_rng(0); T, phi = 200_000, np.exp(-1 / 50.0)
    x = np.zeros((4, T)); e = rng.normal(size=(4, T))
    for t in range(1, T):
        x[:, t] = phi * x[:, t - 1] + e[:, t]
    tau, rho = wca.autocorrelation_time(x, dt=1.0, max_lag=2000)
    assert abs(tau / 50.0 - 1) < 0.15, tau                                          # integral of exp(-k/50) ~ 50


# amendment A2: the projected (reference-scheme) inner step keeps z fixed and is a distinct operator
def test_projected_scheme_keeps_z_fixed_and_differs_from_frozen():
    params, sim, engine = _tiny()
    q = wca.lattice_initial_conditions(params, 12, engine.device, engine.dtype, seed=5)
    m = torch.tensor([0, 300, 300, 5, 0, 2, 7, 1, 4, 0, 300, 2])      # long enough for solvent-dimer contacts to form
    z0 = wca.reaction_coordinate(q, params)
    gen = torch.Generator(); gen.manual_seed(3)
    qp, done_p = wca.frozen_dimer_relax(engine, params, sim, q, m, gen, scheme="projected")
    gen = torch.Generator(); gen.manual_seed(3)
    qf, done_f = wca.frozen_dimer_relax(engine, params, sim, q, m, gen, scheme="frozen")
    assert done_p == done_f == int(m.sum())
    np.testing.assert_allclose(wca.reaction_coordinate(qp, params).numpy(), z0.numpy(), rtol=0, atol=1e-10)   # z re-projected exactly
    moved = (qp[:, :2, :] != q[:, :2, :]).any(dim=(1, 2))
    assert torch.equal(moved, m > 0)                                    # under 'projected' the dimer atoms DO move (only z is held)
    assert torch.equal(qf[:, :2, :], q[:, :2, :])                       # under 'frozen' they do not
    long = m >= 300
    assert not torch.equal(qp[long][:, 2:, :], qf[long][:, 2:, :])      # the two operators are different once contacts form
    _, sim2, _ = _tiny()
    d = _run("abf", relax=wca.RelaxConfig(rho=0.5, tau_grid=tuple([0.05] * sim2.n_grid), scheme="projected"))
    assert d["relax_scheme"] == "projected" and d["relax_steps_total"] > 0
