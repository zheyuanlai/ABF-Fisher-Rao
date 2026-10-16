"""Gate 0 of the ERI campaign: force field, geometry and reaction coordinate (docs/eri_campaign/VALIDATION_REPORT.md).

CPU only.  Run:  CUDA_VISIBLE_DEVICES="" /home/zheyuanlai/miniconda3/envs/abffr/bin/python -m pytest -q tests/test_eri_model.py
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from eri import framework as fwm  # noqa: E402
from eri.model import (KB, PARAMS, Model, brute_energy, cv_phi, local_mean_force, lj_shift)  # noqa: E402

VARIANTS = ("schlenker", "gard_tait")


@pytest.fixture(scope="module", params=VARIANTS)
def model(request):
    return Model(request.param)


def _ethane(com, u, r=1.54):
    u = np.asarray(u, float)
    u = u / np.linalg.norm(u)
    return np.array([com + 0.5 * r * u, com - 0.5 * r * u])


def _configs(m, rng):
    """Representative single-molecule configurations: cage centres, window centres (aligned with the window
    normal and random), close contact with a framework O, a molecule straddling the box faces, far-unwrapped."""
    f = m.fw
    out = []
    for c in f.cages[:3]:
        out.append(("cage", _ethane(c + 0.3 * rng.standard_normal(3), rng.standard_normal(3))))
    for w in f.windows[:3]:
        out.append(("window", _ethane(w + 0.1 * rng.standard_normal(3), rng.standard_normal(3))))
    # window crossing along the window normal (horizontal, perpendicular to the ring plane)
    w = f.windows[1]
    out.append(("window-normal", _ethane(w, [math.cos(math.pi / 6), -math.sin(math.pi / 6), 0.0])))
    # close contact: a bead 2.9 A from a framework O near a cage
    c = f.cages[0]
    O = f.pos[f.types == 0]
    d = O - c
    d -= f.box * np.round(d / f.box)
    j = int(np.argmin((d * d).sum(1)))
    o = c + d[j]
    n = (c - o) / np.linalg.norm(c - o)
    b0 = o + 2.9 * n
    out.append(("contact", np.array([b0, b0 + 1.55 * n])))
    # straddling the box faces and far unwrapped
    corner = np.array([f.box[0] - 0.4, f.box[1] - 0.3, 0.2])
    out.append(("straddle", _ethane(corner, [1, 1, -1])))
    out.append(("unwrapped", _ethane(f.cages[5] + np.array([3, -2, 5]) * f.box, rng.standard_normal(3))))
    return out


def test_framework_counts_and_geometry():
    for v in VARIANTS:
        fw = fwm.build(v)
        ok, dmin, (lo, hi) = fwm.coordination_check(fw)
        assert ok and 1.59 < lo and hi < 1.68 and dmin > 1.59
        assert (fw["types"] == 1).sum() == 1296 and (fw["types"] == 0).sum() == 2592
        s = fwm.special_sites(fw)
        assert len(s["cage"]) == 72 and len(s["window"]) == 216
        # every window is equidistant from exactly two cage centres, a/sqrt(3) apart in xy, c/2 in z
        box = fw["box"]
        for w in s["window"][:20]:
            d = s["cage"] - w
            d -= box * np.round(d / box)
            r = np.sqrt((d * d).sum(1))
            idx = np.argsort(r)[:3]
            assert abs(r[idx[0]] - r[idx[1]]) < 1e-9 and r[idx[2]] - r[idx[1]] > 1.0
            dd = d[idx[0]] - d[idx[1]]
            assert abs(math.hypot(dd[0], dd[1]) - fw["a"] / math.sqrt(3)) < 1e-9
            assert abs(abs(dd[2]) - fw["c"] / 2) < 1e-9


def test_cif_parser_refuses_non_symop():
    with pytest.raises(ValueError):
        fwm.parse_symop("__import__('os'),y,z")
    R, t = fwm.parse_symop("x-y,x,z+1/2")
    assert (R == np.array([[1, -1, 0], [1, 0, 0], [0, 0, 1]])).all() and np.allclose(t, [0, 0, 0.5])


def test_energy_equals_independent_bruteforce(model):
    rng = np.random.default_rng(11)
    f = model.fw
    for name, q in _configs(model, rng):
        U, F = model.energy_force(q[None])
        Ub = brute_energy(q[None], f.pos, f.types, f.box, model.p)
        assert abs(U[0] - Ub) < 1e-9 * max(1.0, abs(Ub)), name


def test_forces_equal_finite_differences(model):
    rng = np.random.default_rng(12)
    h = 1e-5
    for name, q in _configs(model, rng):
        q = q[None].copy()
        U, F = model.energy_force(q)
        num = np.zeros_like(q)
        for b in range(2):
            for k in range(3):
                qp = q.copy(); qp[0, b, k] += h
                qm = q.copy(); qm[0, b, k] -= h
                num[0, b, k] = -(model.energy_force(qp)[0][0] - model.energy_force(qm)[0][0]) / (2 * h)
        scale = max(1.0, np.abs(F).max())
        assert np.abs(num - F).max() < 2e-6 * scale, (name, np.abs(num - F).max(), scale)


def test_loaded_energy_forces_and_min_image(model):
    """Guest-guest CH3-CH3: four molecules incl. a close pair (3.3 A) and a pair interacting across the box face."""
    rng = np.random.default_rng(13)
    f = model.fw
    c = f.cages[0]
    # a cage centred on the X = 0 box face: two molecules 3.4 A apart THROUGH the face (minimum image needed)
    cf = f.cages[np.argmin(np.minimum(f.cages[:, 0], f.box[0] - f.cages[:, 0]))]
    assert min(cf[0], f.box[0] - cf[0]) < 1e-6
    qa = _ethane(cf + [-1.7, 0.0, 0.3], [0, 0, 1])
    qa = qa % f.box                                             # wrapped to the far face
    q = np.stack([_ethane(c + [0, 0, -1.9], [0, 0, 1]), _ethane(c + [0.0, 0.0, 1.9], [0, 0.3, 1]),
                  qa, _ethane(cf + [1.7, 0.0, -0.3], [0, 0, 1])])
    U, F = model.energy_force(q)
    Ub = brute_energy(q, f.pos, f.types, f.box, model.p)
    assert abs(U[0] - Ub) < 1e-9 * max(1, abs(Ub))
    assert U[3] != 0.0                                          # guest-guest term active
    # the across-face pair (molecules 2 and 3) is seen by minimum image
    q2 = q.copy(); q2[3] += np.array([f.box[0], 0, 0])          # unwrap by a box vector: identical energy
    assert abs(model.energy_force(q2)[0][0] - U[0]) < 1e-9 * max(1, abs(U[0]))
    h = 1e-5
    num = np.zeros_like(q)
    for m_ in range(4):
        for b in range(2):
            for k in range(3):
                qp = q.copy(); qp[m_, b, k] += h
                qm = q.copy(); qm[m_, b, k] -= h
                num[m_, b, k] = -(model.energy_force(qp)[0][0] - model.energy_force(qm)[0][0]) / (2 * h)
    assert np.abs(num - F).max() < 2e-6 * max(1.0, np.abs(F).max())
    # Newton's third law: the total framework-free (guest-guest + bond) force sums to zero
    Fgg = F.copy()
    for m_ in range(4):
        Fgg[m_] -= model.energy_force(q[m_:m_ + 1])[1][0]          # remove framework + bond forces
    assert np.abs(Fgg.sum((0, 1))).max() < 1e-9 * max(1.0, np.abs(Fgg).max())


def test_bond_force():
    m = Model("schlenker")
    c = m.fw.cages[0]
    for r in (1.40, 1.54, 1.70):
        q = _ethane(c, [0.3, -0.2, 0.9], r)[None]
        U, F = m.energy_force(q)
        q0 = _ethane(c, [0.3, -0.2, 0.9], 1.54)[None]
        # bond energy component exactly (k/2)(r - r0)^2
        assert abs(U[2] - 0.5 * PARAMS["k_bond_K"] * KB * (r - 1.54) ** 2) < 1e-10
    assert abs(PARAMS["k_bond_K"] * KB - 802.3456) < 1e-3


def test_cutoff_energy_continuity():
    """Energy-shifted LJ: V(r) - V(rc) is continuous at rc (the force is truncated there: the convention)."""
    for e, s in ((142.0, 3.17), (82.0, 2.12), (104.0, 3.78)):
        eps = e * KB
        v = lambda r: 4 * eps * ((s / r) ** 12 - (s / r) ** 6) - lj_shift(eps, s, 12.0)
        assert abs(v(12.0 - 1e-9)) < 1e-9
        # the force jump at rc is tiny compared with kT at 600 K over 1 A
        fjump = abs(24 * eps * (2 * (s / 12.0) ** 12 - (s / 12.0) ** 6) / 12.0)
        assert fjump < 0.01 * KB * 600.0


def test_lattice_translation_invariance(model):
    rng = np.random.default_rng(14)
    f = model.fw
    q = _ethane(f.cages[3] + 0.2 * rng.standard_normal(3), rng.standard_normal(3))[None]
    U0 = model.energy_force(q)[0][0]
    for t in ([f.a, 0, 0], [0.5 * f.a, 0.5 * math.sqrt(3) * f.a, 0], [0, 0, f.c], [f.a, math.sqrt(3) * f.a, -f.c]):
        assert abs(model.energy_force(q + np.array(t))[0][0] - U0) < 1e-9 * max(1, abs(U0))


def test_cv_periodicity_gradient_and_mean_force(model):
    rng = np.random.default_rng(15)
    f = model.fw
    P, xw = f.P, f.x_window
    # windows of the between-row family sit at phi = 0, cage rows at phi = +-pi
    for w in f.windows:
        ph = cv_phi(_ethane(w, [0, 0, 1])[None], P, xw)
        assert min(abs(ph), abs(abs(ph) - math.pi)) < 1e-9
    n0 = sum(abs(cv_phi(_ethane(w, [0, 0, 1])[None], P, xw)) < 1e-9 for w in f.windows)
    assert n0 == 144                                           # 2 of the 3 window directions per cage are between rows
    for c in f.cages:
        assert abs(abs(cv_phi(_ethane(c, [0, 0, 1])[None], P, xw)) - math.pi) < 1e-9
    for name, q in _configs(model, rng):
        q = q[None]
        ph = cv_phi(q, P, xw)
        assert abs(cv_phi(q + np.array([P, 0, 0]), P, xw) - ph) < 1e-9 or abs(abs(ph) - math.pi) < 1e-9
        # gradient: d phi / d q_{b,x} = pi / P, zero in y, z
        h = 1e-6
        for b in range(2):
            for k in range(3):
                qp = q.copy(); qp[0, b, k] += h
                g = (cv_phi(qp, P, xw) - ph) / h
                if abs(abs(ph) - math.pi) > 1e-4:
                    assert abs(g - (math.pi / P if k == 0 else 0.0)) < 1e-5
        # local mean force = derivative of U along a rigid translation that moves phi at unit rate
        U, F = model.energy_force(q)
        f_loc = local_mean_force(F, P)
        d = 1e-6
        sh = np.array([P / (2 * math.pi) * d, 0, 0])
        dU = (model.energy_force(q + sh)[0][0] - model.energy_force(q - sh)[0][0]) / (2 * d)
        assert abs(f_loc - dU) < 1e-5 * max(1.0, abs(dU)), name


def test_abf_bias_force_direction():
    """+A'(phi) grad phi is the force of the potential -A(phi(q)): checked by finite differences for A = sin."""
    m = Model("schlenker")
    P, xw = m.fw.P, m.fw.x_window
    q = _ethane(m.fw.windows[1] + [0.4, 0.1, 0], [1, 0.2, 0.1])[None]
    ph = cv_phi(q, P, xw)
    bias_force_x = math.cos(ph) * (math.pi / P)                  # per bead, x component
    h = 1e-6
    for b in range(2):
        qp = q.copy(); qp[0, b, 0] += h
        qm = q.copy(); qm[0, b, 0] -= h
        num = -(-math.sin(cv_phi(qp, P, xw)) + math.sin(cv_phi(qm, P, xw))) / (2 * h)
        assert abs(num - bias_force_x) < 1e-6


def test_batch_equals_single_and_threads():
    import numba
    m = Model("schlenker")
    rng = np.random.default_rng(16)
    Q = np.stack([_ethane(m.fw.cages[i % 72] + 0.3 * rng.standard_normal(3), rng.standard_normal(3))
                  for i in range(64)])[:, None]
    U1, F1 = m.batch(Q)
    for i in range(0, 64, 7):
        U, F = m.energy_force(Q[i])
        assert U[0] == U1[i, 0] and (F == F1[i]).all()           # bitwise: replicas are independent
    nt = numba.get_num_threads()
    numba.set_num_threads(1)
    U2, F2 = m.batch(Q)
    numba.set_num_threads(nt)
    assert (U2 == U1).all() and (F2 == F1).all()
