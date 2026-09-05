"""Stage A0 invariants of the WCA OT lift + repair (docs/WCA_OT_REPAIR_MECHANISM.md)."""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import wca_abffr_core as core                          # noqa: E402
from wca_ot_repair import OTConfig, ot_displacement, uniform_quantiles   # noqa: E402


def _params():
    return core.DimerWCAParams(beta=1.0, h=2.0, w=2.0, n_dim=10, a=1.5, sigma=1.0, epsilon=1.0)


def test_displacement_identity_ranks_cap():
    g = torch.Generator().manual_seed(0)
    z = torch.rand(257, generator=g, dtype=torch.float64) * 1.4 - 0.2
    u = uniform_quantiles(257, -0.2, 1.2, z.device, z.dtype)
    assert torch.equal(ot_displacement(z, 0.0, 0.05, u), z)                       # alpha = 0 is the identity
    z1 = ot_displacement(z, 1.0, 10.0, u)
    assert torch.allclose(torch.sort(z1).values, u)                               # alpha = 1, no cap: exactly the quantiles
    assert torch.equal(torch.argsort(z1, stable=True), torch.argsort(z, stable=True))   # ranks preserved
    zc = ot_displacement(z, 1.0, 0.01, u)
    assert float((zc - z).abs().max()) <= 0.01 + 1e-12                             # cap respected
    assert torch.equal(torch.argsort(zc, stable=True), torch.argsort(z, stable=True))
    assert z1.shape == z.shape == zc.shape                                         # population unchanged


def test_lift_lands_on_the_fibre():
    p = _params()
    q = core.lattice_initial_conditions(p, 64, core.DEVICE, core.DTYPE, seed=3)
    z = core.reaction_coordinate(q, p)
    u = uniform_quantiles(64, -0.2, 1.2, q.device, q.dtype)
    z_new = ot_displacement(z, 0.3, 0.05, u)
    q_new = core.project_dimer_to_z(q, z_new, p)
    # the engine is float32: the projection is exact up to float32 rounding (~1e-7 in z)
    assert float((core.reaction_coordinate(q_new, p) - z_new).abs().max()) < 1e-5
    assert torch.equal(q_new[:, 2:, :], q[:, 2:, :])                               # bath untouched
    mid_old = q[:, 1, :] + 0.5 * core.minimum_image(q[:, 0, :] - q[:, 1, :], p.box_length)
    mid_new = q_new[:, 1, :] + 0.5 * core.minimum_image(q_new[:, 0, :] - q_new[:, 1, :], p.box_length)
    assert float(core.minimum_image(mid_new - mid_old, p.box_length).abs().max()) < 1e-5   # midpoint kept


def _tiny_sim(**kw):
    base = dict(n_replicas=16, n_steps=80, save_every=40, seed=5, abf_warmup_steps=10, estimator_burn_in_steps=10,
                fr_start_steps=20, fr_every=5, n_grid=40)
    base.update(kw)
    return core.SimConfig(**base)


def test_sampler_alpha_zero_matches_abf_and_repair_is_accounted():
    p = _params()
    engine = core.WCADimerEngine(p, core.DEVICE, core.DTYPE)
    sim = _tiny_sim()
    tau = tuple([0.02] * sim.n_grid)                                               # 10 steps at dt 2e-3
    ref = core.run_sampler_gpu("abf", p, sim, engine, verbose=False)
    ot0 = core.run_sampler_gpu("abf", p, sim, engine, verbose=False,
                               ot=OTConfig(alpha=0.0, dz_max=0.0176, c_repair=0.0, tau_grid=tau))
    # alpha = 0 lifts to the walker's own z: the dynamics are unchanged up to the projection's rounding
    assert np.allclose(ref["mean_force"][-1], ot0["mean_force"][-1], atol=1e-6)
    assert int(ot0["relax_steps_total"]) == 0 and int(ot0["ot_n_opportunities"]) == 13
    tr = core.run_sampler_gpu("abf", p, sim, engine, verbose=False,
                              ot=OTConfig(alpha=0.5, dz_max=0.0176, c_repair=1.0, tau_grid=tau))
    assert int(tr["ot_n_opportunities"]) == 13
    # every moved walker gets ceil(c tau / dt) = 10 inner steps; all are charged
    assert int(tr["relax_steps_total"]) == 10 * int(round(float(tr["ot_moved_frac"]) * 16 * 13))
    assert abs(float(tr["relax_cost_ratio"]) - tr["relax_steps_total"] / (16 * 80)) < 1e-12
    assert np.isfinite(tr["mean_force"][-1]).all()
    assert float(tr["ot_absdz_max"]) <= 0.0176 + 1e-12
    # the pre/post deposit-free accumulators only see moved walkers
    assert float(np.sum(tr["ot_C_pre"])) == pytest.approx(float(tr["ot_moved_frac"]) * 16 * 13, abs=1e-6)


def test_repair_all_and_dz_table():
    """M3: repair_all charges every walker (R arm with alpha 0); the (z, |dz|) table sums to C_post."""
    p = _params()
    engine = core.WCADimerEngine(p, core.DEVICE, core.DTYPE)
    sim = _tiny_sim()
    tau = tuple([0.02] * sim.n_grid)                                               # the frozen 10-dt map: c 0.5 -> 5 steps
    R = core.run_sampler_gpu("abf", p, sim, engine, verbose=False,
                             ot=OTConfig(alpha=0.0, dz_max=0.0176, c_repair=0.5, tau_grid=tau, repair_all=True))
    assert int(R["ot_n_opportunities"]) == 13 and float(R["ot_moved_frac"]) == 0.0
    assert int(R["relax_steps_total"]) == 5 * 16 * 13                             # every walker, every event
    assert float(np.sum(R["ot_C_post"])) == pytest.approx(16 * 13, abs=1e-6)       # pending = repaired walkers
    T = core.run_sampler_gpu("abf", p, sim, engine, verbose=False,
                             ot=OTConfig(alpha=0.5, dz_max=0.0176, c_repair=0.0, tau_grid=tau))
    assert T["ot_C2_post"].shape == (sim.n_grid, len(T["ot_absdz_edges"]) - 1)
    assert float(np.sum(T["ot_C2_post"])) == pytest.approx(float(np.sum(T["ot_C_post"])), abs=1e-6)
    assert float(np.sum(T["ot_Sf2_post"])) == pytest.approx(float(np.sum(T["ot_Sf_post"])), rel=1e-9, abs=1e-6)
    assert len(T["ot_absdz_t"]) == len(T["ot_steps"]) and np.nanmax(T["ot_absdz_t"]) <= 0.0176 + 1e-12
    F = core.run_sampler_gpu("fr_uniform", p, sim, engine, verbose=False,
                             ot=OTConfig(alpha=0.0, dz_max=0.0176, c_repair=0.5, tau_grid=tau, repair_all=True))
    assert int(F["relax_steps_total"]) == 5 * 16 * 13                             # FR + the same repair (F+R arm)
