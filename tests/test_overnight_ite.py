"""Gate 0 for candidate A' (ethane in rigid Si-ITE / ITQ-3, Dubbeldam-2004 UA force field, 300 K).  CPU only.

    NUMBA_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" /home/zheyuanlai/miniconda3/envs/abffr/bin/python -m pytest -q tests/test_overnight_ite.py
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from eri import framework as fwm  # noqa: E402
from eri.model import KB, Model, brute_energy, cv_phi, local_mean_force  # noqa: E402

FF = dict(T_K=300.0, eps_O_K=93.0, sig_O=3.48, eps_Si_K=0.0, sig_Si=1.0, eps_CC_K=108.0, sig_CC=3.76, rc=12.0,
          shift=True, k_bond_K=96500.0, r0=1.54)


@pytest.fixture(scope="module")
def m():
    return Model("ite_itq3", params=FF)


def _eth(com, u, r=1.54):
    u = np.asarray(u, float)
    u = u / np.linalg.norm(u)
    return np.array([com + 0.5 * r * u, com - 0.5 * r * u])


def _configs(m, rng):
    f = m.fw
    out = []
    for c in f.cages[:4]:
        out.append(("cage", _eth(c + 0.3 * rng.standard_normal(3), rng.standard_normal(3))))
    for w in f.windows[:4]:
        out.append(("window-axial", _eth(w + 0.05 * rng.standard_normal(3), [1, 0.05, 0.05])))
    out.append(("window-tilted", _eth(f.windows[0], [1, 1, 0])))       # strongly repulsive orientation
    out.append(("straddle", _eth(np.array([f.box[0] - 0.3, f.cages[0][1], f.box[2] - 0.2]), [1, 0, 0.2])))
    out.append(("unwrapped", _eth(f.cages[3] + np.array([2, -1, 3]) * f.box, rng.standard_normal(3))))
    return out


def test_framework_and_parameters(m):
    fw = fwm.load("ite_itq3")
    assert (fw["types"] == 1).sum() == 768 and (fw["types"] == 0).sum() == 1536
    assert abs(m.fw.P - 9.7242) < 1e-12
    assert m.fw.lists[1][0][-1] == 0                       # O-only: no Si candidates at all
    assert abs(m.pp[0, 0] - 4 * 93.0 * KB) < 1e-12 and abs(m.pp[0, 1] - 3.48 ** 2) < 1e-12
    assert m.pp[1, 0] == 0.0 and abs(m.kb - 96500.0 * KB) < 1e-12 and m.p["rc"] == 12.0
    assert min(m.fw.box) > 2 * 12.0


def test_energy_bruteforce_and_fd_forces(m):
    rng = np.random.default_rng(7)
    f = m.fw
    h = 1e-5
    for name, q in _configs(m, rng):
        q = q[None]
        U, F = m.energy_force(q)
        Ub = brute_energy(q, f.pos, f.types, f.box, m.p)
        assert abs(U[0] - Ub) < 1e-9 * max(1.0, abs(Ub)), name
        num = np.zeros_like(q)
        for b in range(2):
            for k in range(3):
                qp = q.copy(); qp[0, b, k] += h
                qm = q.copy(); qm[0, b, k] -= h
                num[0, b, k] = -(m.energy_force(qp)[0][0] - m.energy_force(qm)[0][0]) / (2 * h)
        assert np.abs(num - F).max() < 2e-6 * max(1.0, np.abs(F).max()), name


def test_guest_guest_dubbeldam(m):
    c = m.fw.cages[0]
    q = np.stack([_eth(c + [-1.9, 0, 0], [1, 0, 0]), _eth(c + [1.9, 0, 0], [1, 0, 0.1])])
    U, F = m.energy_force(q)
    assert abs(U[0] - brute_energy(q, m.fw.pos, m.fw.types, m.fw.box, m.p)) < 1e-9 * max(1, abs(U[0]))
    assert U[3] != 0.0


def test_cv_windows_and_mean_force(m):
    f = m.fw
    for w in f.windows:
        assert abs(cv_phi(_eth(w, [1, 0, 0])[None], f.P, f.x_window)) < 1e-9
    for c in f.cages:
        assert abs(abs(cv_phi(_eth(c, [1, 0, 0])[None], f.P, f.x_window)) - math.pi) < 1e-9
    rng = np.random.default_rng(9)
    for name, q in _configs(m, rng):
        q = q[None]
        U, F = m.energy_force(q)
        d = 1e-6
        sh = np.array([f.P / (2 * math.pi) * d, 0, 0])
        dU = (m.energy_force(q + sh)[0][0] - m.energy_force(q - sh)[0][0]) / (2 * d)
        assert abs(local_mean_force(F, f.P) - dU) < 1e-5 * max(1.0, abs(dU)), name


def test_channels_are_closed_to_ethane_sideways(m):
    """Moving a rigid ethane from the channel axis sideways (Y or Z) by half the channel spacing costs > 50 kJ/mol at
    every orientation tried: the 1-D channel family is a physical invariant set (no walker can change channel)."""
    f = m.fw
    c = f.cages[0]
    rng = np.random.default_rng(11)
    for shift in ([0, f.cell[1] / 2, 0], [0, 0, f.cell[2] / 2], [0, f.cell[1] / 4, f.cell[2] / 4]):
        best = np.inf
        for x in np.linspace(0, f.P, 25):
            for _ in range(40):
                q = _eth(c + np.array([x, 0, 0]) + 0.5 * np.array(shift), rng.standard_normal(3))[None]
                best = min(best, m.energy_force(q)[0][0])
        assert best > 50.0, (shift, best)
