"""Invariants of the distance-CV OT lift + projected repair (docs/PENTANE_R15_OT_REPAIR.md).

Run: CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_pentane_ot_repair.py -q
"""
import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
torch.set_default_dtype(torch.float64)

from alkanes import potentials as pot, core_dist as cd, geometry as geom  # noqa: E402
from alkanes.distance_cv import DistanceCV  # noqa: E402
from alkanes.ot_repair_dist import (DistOTConfig, lj_forbidden_radius, lift_to_R,  # noqa: E402
                                    ot_displacement_batched, uniform_quantiles, projected_relax)


def _params():
    return pot.AlkaneParams(n_atoms=5, beta=2.0, sigma=2.3, epsilon=1.0, decouple=False, force_clip=200.0)


def _sim(**kw):
    base = dict(dt=5e-4, n_steps=300, n_replicas=48, save_every=100, rng_seed=11,
                R_lo=1.4, R_hi=3.7, wall_lo=1.45, wall_hi=3.65, n_grid=64,
                abf_warmup_steps=50, estimator_burn_in_steps=50, fr_start_steps=100, fr_every=5,
                fr_rate=0.02, max_event_fraction=0.01, n_grid2=12, n_rbins=6)
    base.update(kw)
    return cd.DistSimConfig(**base)


def _run(method, ot=None, **kw):
    p = _params(); cv = DistanceCV(0, 4)
    return cd.run_sampler_dist(method, p, _sim(**kw), [0, 1], cv, "cpu", initial_dihedrals=[0.0, 0.0],
                               verbose=False, ot=ot)


def test_lj_forbidden_radius():
    p = _params()
    R = lj_forbidden_radius(p, 10.0)
    x = (p.sigma / R) ** 6
    assert abs(4.0 * p.epsilon * (x * x - x) - 10.0) < 1e-9
    assert 2.0 < R < 2.05                                   # ~2.023 for sigma 2.3
    assert lj_forbidden_radius(p, 20.0) < R                 # a higher allowance reaches further in


def test_lift_lands_on_R_and_keeps_com():
    g = torch.Generator().manual_seed(0)
    q = geom.place_chain(torch.rand(64, 2, generator=g) * 2 * math.pi - math.pi, 5)
    q = q + 0.05 * torch.randn(q.shape, generator=g)
    cv = DistanceCV(0, 4)
    R0 = cv.value(q)
    R1 = R0 + (torch.rand(64, generator=g) - 0.5) * 0.2
    q1 = lift_to_R(q, R1, 0, 4)
    assert float((cv.value(q1) - R1).abs().max()) < 1e-12
    assert float((q1.mean(1) - q.mean(1)).abs().max()) < 1e-12          # COM preserved
    assert torch.equal(q1[:, 1:4, :], q[:, 1:4, :])                    # inner atoms untouched
    assert torch.equal(lift_to_R(q, R0, 0, 4), q) or float((lift_to_R(q, R0, 0, 4) - q).abs().max()) < 1e-14


def test_displacement_identity_ranks_cap_batched():
    g = torch.Generator().manual_seed(1)
    R = torch.rand(3, 257, generator=g) * 2.3 + 1.4
    u = uniform_quantiles(257, 2.0, 3.65, R.device, R.dtype)
    assert torch.equal(ot_displacement_batched(R, 0.0, 0.05, u), R)
    R1 = ot_displacement_batched(R, 1.0, 10.0, u)
    for b in range(3):
        assert torch.allclose(torch.sort(R1[b]).values, u)
        assert torch.equal(torch.argsort(R1[b], stable=True), torch.argsort(R[b], stable=True))
    Rc = ot_displacement_batched(R, 1.0, 0.018, u)
    assert float((Rc - R).abs().max()) <= 0.018 + 1e-12
    for b in range(3):
        assert torch.equal(torch.argsort(Rc[b], stable=True), torch.argsort(R[b], stable=True))


def test_projected_relax_preserves_R():
    p = _params(); cv = DistanceCV(0, 4)
    g = torch.Generator().manual_seed(2)
    q = geom.place_chain(torch.rand(32, 2, generator=g) * 2 * math.pi - math.pi, 5)
    Rf = cv.value(q) + 0.03
    q = lift_to_R(q, Rf, 0, 4)
    q2, f_first = projected_relax(q, Rf, 12, p, 5e-4, p.beta, g, pot.forces, cv, record_first=True)
    assert float((cv.value(q2) - Rf).abs().max()) < 1e-11
    assert float(q2.mean(1).abs().max()) < 1e-12
    assert f_first.shape == (32,) and torch.isfinite(f_first).all()
    assert float((q2 - q).abs().max()) > 1e-3                          # the fibre actually moved


def test_sampler_default_path_is_byte_identical():
    ref = _run("abf")
    ot0 = _run("abf", ot=DistOTConfig(alpha=0.0, m_repair=0))
    assert np.array_equal(ref["pmf"][-1], ot0["pmf"][-1])
    assert np.array_equal(ref["cond_hist"], ot0["cond_hist"])
    ff = cd.run_sampler_dist("abf", _params(), _sim(), [0, 1], DistanceCV(0, 4), "cpu",
                             initial_dihedrals=[0.0, 0.0], verbose=False, force_fn=pot.forces)
    assert np.array_equal(ref["pmf"][-1], ff["pmf"][-1])
    assert int(ot0["ot_n_opportunities"]) == 41 and int(ot0["inner_steps_total"]) == 0


def test_repair_arm_accounting_and_diagnostics():
    N = 48
    r_arm = _run("abf", ot=DistOTConfig(alpha=0.0, m_repair=3))
    assert int(r_arm["ot_n_opportunities"]) == 41
    assert int(r_arm["inner_steps_total"]) == 3 * N * 41
    assert np.isfinite(r_arm["mean_force"][-1]).all()
    assert float(r_arm["ot_C_pre"].sum()) == 0.0 and float(r_arm["ot_C_post"].sum()) == 0.0
    assert len(r_arm["series_inner_steps"]) == len(r_arm["steps"])
    assert r_arm["series_inner_steps"][-1] == 3 * N * 41
    assert r_arm["series_cond_hist"].shape == (len(r_arm["steps"]), 2, 6, 12, 12)
    t_arm = _run("abf", ot=DistOTConfig(alpha=0.3, dR_max=0.018, m_repair=2, domain=(2.0, 3.65)))
    moved = float(t_arm["ot_moved_frac"].sum()) * 41 * N
    assert moved > 0
    assert abs(float(t_arm["ot_C_pre"].sum()) - moved) < 1e-6
    assert abs(float(t_arm["ot_cond_pre"].sum()) - moved) < 1e-6
    # the post accumulators see every moved walker except those of the very last opportunity (no outer step follows)
    assert float(t_arm["ot_C_post"].sum()) <= moved and float(t_arm["ot_C_post"].sum()) >= moved - 2 * N
    assert abs(float(t_arm["ot_cond_post"].sum()) - float(t_arm["ot_C_post"].sum())) < 1e-6
    assert float(t_arm["ot_absdR_max"].max()) <= 0.018 + 1e-12
    assert int(t_arm["inner_steps_total"]) == 2 * N * 41
    assert np.array_equal(t_arm["ot_domain"], np.array([2.0, 3.65]))
    # the T arm (no repair) still records post deposits and no pre deposits
    t0 = _run("abf", ot=DistOTConfig(alpha=0.3, dR_max=0.018, m_repair=0, domain=(2.0, 3.65)))
    assert float(t0["ot_C_pre"].sum()) == 0.0 and float(t0["ot_C_post"].sum()) > 0
    assert int(t0["inner_steps_total"]) == 0


def test_fr_uniform_domain_target():
    grid, dz = __import__("alkanes.interval", fromlist=["x"]).interval_grid(64, 1.4, 3.7)
    B = torch.zeros(2, 64)
    q_full = cd._fr_target("fr_uniform", grid, dz, 1.4, 3.7, None, B, None, 2.0)
    q_dom = cd._fr_target("fr_uniform", grid, dz, 1.4, 3.7, None, B, None, 2.0, fr_domain=(2.0, 3.65))
    assert torch.allclose(q_full.sum(-1) * dz, torch.ones(2)) and torch.allclose(q_dom.sum(-1) * dz, torch.ones(2))
    assert float(q_dom[0, grid < 2.0].max()) == 0.0 and float(q_dom[0, (grid > 2.05) & (grid < 3.6)].min()) > 0
    out = _run("fr_uniform", fr_domain=(2.0, 3.65), fr_rate=0.5, max_event_fraction=0.1)
    assert np.isfinite(out["pmf"][-1]).all()
    assert float(out["q_target"][-1][0][grid.numpy() < 2.0].max()) == 0.0
