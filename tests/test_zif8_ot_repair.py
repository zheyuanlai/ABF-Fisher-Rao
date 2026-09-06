"""Invariants of the ZIF-8 OT lift + constrained BAOAB repair (docs/ZIF8_OT_REPAIR.md), on the
synthetic framework of tests/test_zif8.py (CPU)."""
import importlib.util
import os
import sys

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
spec = importlib.util.spec_from_file_location("tz", os.path.join(os.path.dirname(__file__), "test_zif8.py"))
tz = importlib.util.module_from_spec(spec); spec.loader.exec_module(tz)
from zif8.core_zif8 import ZIF8SimConfig   # noqa: E402
from zif8.ot_repair_zif8 import (ConstrainedBAOAB, lift_guest, project_guest_velocity,   # noqa: E402
                                 local_mean_force_xi, gate_pdf, tv, integrated_autocorr)


def test_lift_is_exact_and_touches_only_the_guest(tmp_path):
    s = tz.make_system(tmp_path)
    q = tz.rand_config(s, B=4)
    xi0 = s.xi_value(q)
    xi1 = xi0 + torch.tensor([0.3, -0.3, 1.2, 0.0], dtype=torch.float64)
    q1 = lift_guest(s, q, xi1)
    assert float((s.xi_value(q1) - xi1).abs().max()) < 1e-12
    assert torch.equal(q1[:, :s.n_frame], q[:, :s.n_frame])
    d = q1[:, s.n_frame:] - q[:, s.n_frame:]
    assert float((d[:, 0] - d[:, 1]).abs().max()) < 1e-12                      # rigid translation
    assert float((d - (d * s.normal).sum(-1, keepdim=True) * s.normal).abs().max()) < 1e-12   # along n only
    ag0, _ = s.gate_observables(q); ag1, _ = s.gate_observables(q1)
    assert torch.equal(ag0, ag1)                                                # framework observables unchanged


def test_velocity_projection_removes_only_com_normal_component(tmp_path):
    s = tz.make_system(tmp_path)
    g = torch.Generator().manual_seed(3)
    v = torch.randn(4, s.n_atoms, 3, generator=g, dtype=torch.float64)
    v1 = project_guest_velocity(s, v)
    vg = v1[:, s.n_frame:]
    vcom_n = ((vg * s.mass_w[None, :, None]).sum(1) * s.normal).sum(-1)
    assert float(vcom_n.abs().max()) < 1e-12
    assert torch.equal(v1[:, :s.n_frame], v[:, :s.n_frame])
    rel = (v[:, s.n_frame + 1] - v[:, s.n_frame]) - (v1[:, s.n_frame + 1] - v1[:, s.n_frame])
    assert float(rel.abs().max()) < 1e-12                                       # internal motion untouched


def test_constrained_baoab_holds_xi_and_moves_the_frame(tmp_path):
    s = tz.make_system(tmp_path)
    sim = ZIF8SimConfig(dt=0.0005, gamma=1.0)
    g = torch.Generator().manual_seed(11)
    q = tz.rand_config(s, B=3, jitter=0.02)
    v = s.pin_frame_com(s.maxwell_velocities((3,), g))
    xi_fixed = s.xi_value(q) + torch.tensor([0.1, 0.0, -0.1], dtype=torch.float64)
    q = lift_guest(s, q, xi_fixed)
    rec = []
    dyn = ConstrainedBAOAB(s, sim, g)
    q2, v2, F2 = dyn.run(q, v, xi_fixed, 40, record=lambda k, qq, vv, FF: rec.append(float(local_mean_force_xi(s, qq, FF).mean())))
    assert float((s.xi_value(q2) - xi_fixed).abs().max()) < 1e-10
    assert float((q2[:, :s.n_frame] - q[:, :s.n_frame]).abs().max()) > 1e-4     # framework moved
    assert float((q2[:, s.n_frame:] - q[:, s.n_frame:]).abs().max()) > 1e-5      # guest moved laterally / internally
    assert len(rec) == 40 and np.isfinite(rec).all()
    assert torch.isfinite(F2).all()
    # pulling schedule reaches its target exactly
    tgt = xi_fixed + 0.5
    sched = lambda k: xi_fixed + 0.5 * (k + 1) / 20                             # noqa: E731
    q3, _, _ = dyn.run(q2, v2, tgt, 20, F=F2, xi_schedule=sched)
    assert float((s.xi_value(q3) - tgt).abs().max()) < 1e-10


def test_gate_pdf_tv_and_autocorr():
    edges = np.linspace(0, 1, 11)
    p, drop = gate_pdf(np.array([0.05, 0.15, 0.15, 1.5]), edges)
    assert drop == 1 and abs(p.sum() - 1) < 1e-12 and p[1] == 2 / 3
    assert tv(p, p) == 0.0 and abs(tv(p, np.roll(p, 5)) - 1.0) < 1e-12
    rng = np.random.default_rng(0)
    x = np.zeros((4000, 8)); x[0] = rng.normal(size=8)
    for t in range(1, 4000):                                                     # AR(1), tau = -1/ln(0.9) ~ 9.5
        x[t] = 0.9 * x[t - 1] + rng.normal(size=8) * np.sqrt(1 - 0.81)
    tau, _ = integrated_autocorr(x, 1.0)
    assert 6.0 < tau < 14.0
