"""LTA independent model (src/lta_validation.py) vs the production engine (src/lta/core_lta.py).

L1 forces and energies equal core_lta (float64 CPU torch) to 1e-10 relative at 300 K and 150 K;
L2 analytic force = central differences of the energy;
L3 phi and the local mean force equal core_lta's cv_value / cv_local_mean_force;
L4 binned WHAM recovers a known density from synthetic umbrella histograms.
"""
import os
import sys

import numpy as np
import pytest
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import lta_validation as LV  # noqa: E402
from lta.core_lta import LTAParams, LTASystem  # noqa: E402


@pytest.mark.parametrize("T", [300.0, 150.0])
def test_L1_L3_equal_production(T):
    O, a, L = LV.framework()
    pp = LV.params(T)
    sysm = LTASystem(LTAParams(temperature=T), torch.device("cpu"), torch.float64, root=ROOT)
    q = np.concatenate([LV.initial_states(32, 1, a, L), LV.window_states(32, 2, a, L, 0.0),
                        LV.window_states(32, 3, a, L, 1.0)])
    Ft = sysm.forces(torch.as_tensor(q)).numpy()
    Ut = sysm.potential_energy(torch.as_tensor(q)).numpy()
    fl, ph, _ = sysm.cv_local_mean_force(torch.as_tensor(q), torch.as_tensor(Ft))
    for m in range(len(q)):
        F = np.zeros((2, 3))
        U = LV.energy_force(q[m], O, pp, F)
        assert np.abs(F - Ft[m]).max() < 1e-10 * max(1.0, np.abs(Ft[m]).max())
        assert abs(U - Ut[m]) < 1e-10 * max(1.0, abs(Ut[m]))
        assert abs(LV.phi_of(q[m], a) - float(ph[m])) < 1e-12
        assert abs(-(a / (2 * np.pi)) * (F[0, 0] + F[1, 0]) - float(fl[m])) < 1e-9 * max(1.0, abs(float(fl[m])))


def test_L2_gradient():
    O, a, L = LV.framework()
    pp = LV.params(300.0)
    q = LV.window_states(8, 5, a, L, 0.3)
    worst = 0.0
    for m in range(len(q)):
        F = np.zeros((2, 3))
        LV.energy_force(q[m], O, pp, F)
        h = 1e-5
        for b in range(2):
            for k in range(3):
                qp = q[m].copy(); qp[b, k] += h
                qm = q[m].copy(); qm[b, k] -= h
                fd = -(LV.energy_force(qp, O, pp, np.zeros((2, 3))) - LV.energy_force(qm, O, pp, np.zeros((2, 3)))) / (2 * h)
                worst = max(worst, abs(fd - F[b, k]) / max(1.0, abs(F[b, k])))
    assert worst < 1e-6, worst


def test_L4_wham_recovers_known_density():
    rng = np.random.default_rng(0)
    nb = 1800
    edges = np.linspace(-np.pi, np.pi, nb + 1)
    mid = 0.5 * (edges[1:] + edges[:-1])
    kT, kappa = 2.5, 300.0
    F = 10 * np.cos(mid) + 3 * np.sin(2 * mid)
    p = np.exp(-F / kT); p /= p.sum()
    cen = np.linspace(-np.pi, np.pi, 40, endpoint=False)
    H = []
    for c in cen:
        d = mid - c
        d = d - 2 * np.pi * np.floor((d + np.pi) / (2 * np.pi))
        pb = p * np.exp(-0.5 * kappa * d ** 2 / kT); pb /= pb.sum()
        H.append(rng.multinomial(2_000_000, pb))
    pw, it = LV.wham(np.array(H), cen, kappa, kT)
    Fw = -kT * np.log(np.maximum(pw, 1e-300))
    ok = p > 1e-8
    d = (Fw - F)[ok]
    assert np.sqrt(np.mean((d - d.mean()) ** 2)) < 0.02, np.sqrt(np.mean((d - d.mean()) ** 2))
