"""Distance CV (R15/R14) geometry + generalized-force validation.

Run: CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_alkanes_distance.py -q
"""
import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
torch.set_default_dtype(torch.float64)

from alkanes import geometry as geom  # noqa: E402
from alkanes import potentials as pot  # noqa: E402
from alkanes.distance_cv import DistanceCV  # noqa: E402

PI = math.pi


def _rand_pentane(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    dih = (torch.rand(n, 2, generator=g) * 2 - 1) * (PI - 1e-3)
    q = geom.place_chain(dih, n_atoms=5)
    return q + 0.05 * torch.randn(q.shape, generator=g)   # off-equilibrium jitter


def test_value_positive_and_invariances():
    q = _rand_pentane(64, 1)
    cv = DistanceCV(0, 4)
    R = cv.value(q)
    assert torch.all(R > 0.5)
    # translation invariance
    RT = cv.value(q + torch.tensor([2.0, -3.0, 1.0]))
    assert (RT - R).abs().max() < 1e-12
    # rotation invariance
    A = torch.linalg.qr(torch.randn(3, 3, generator=torch.Generator().manual_seed(3)))[0]
    if torch.det(A) < 0:
        A[:, 0] = -A[:, 0]
    RR = cv.value(q @ A.T)
    assert (RR - R).abs().max() < 1e-12


def test_grad_norm_is_sqrt2_and_analytic_matches_autodiff():
    q = _rand_pentane(48, 2)
    cv = DistanceCV(0, 4)
    R, g, div = cv.geometry(q)
    Ra, ga, diva = cv.geometry_autodiff(q)
    assert (R - Ra).abs().max() < 1e-12
    assert (g - ga).abs().max() < 1e-9
    assert (div - diva).abs().max() < 1e-7
    gg = (g * g).sum(dim=(-2, -1))
    assert (gg - 2.0).abs().max() < 1e-10          # |grad R|^2 = 2
    assert (div - 2.0 / R).abs().max() < 1e-10     # div v = 2/R


def test_grad_vs_finite_difference():
    cv = DistanceCV(0, 4)
    q = _rand_pentane(12, 5)
    _, g, _ = cv.geometry(q)
    eps = 1e-6
    gfd = torch.zeros_like(q)
    for a in range(5):
        for c in range(3):
            qp = q.clone(); qp[:, a, c] += eps
            qm = q.clone(); qm[:, a, c] -= eps
            gfd[:, a, c] = (cv.value(qp) - cv.value(qm)) / (2 * eps)
    assert (g - gfd).abs().max() < 1e-6


def test_div_vs_finite_difference_of_v():
    cv = DistanceCV(0, 4)
    q = _rand_pentane(12, 7)
    _, _, div = cv.geometry(q)

    def vfield(qq):
        _, gg, _ = cv.geometry(qq)
        n2 = (gg * gg).sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
        return gg / n2
    eps = 1e-6
    div_fd = torch.zeros(q.shape[0])
    for a in range(5):
        for c in range(3):
            qp = q.clone(); qp[:, a, c] += eps
            qm = q.clone(); qm[:, a, c] -= eps
            div_fd += (vfield(qp)[:, a, c] - vfield(qm)[:, a, c]) / (2 * eps)
    assert (div - div_fd).abs().max() < 1e-4


def test_local_mean_force_sign_and_formula():
    # f_R = 1/2 e.(gradV_j - gradV_i) - 2/(beta R); check vs the explicit formula.
    p = pot.AlkaneParams(n_atoms=5, beta=1.0, sigma=2.3)
    cv = DistanceCV(0, 4)
    q = _rand_pentane(32, 11)
    F = pot.forces(q, p)                      # -grad V
    f_loc, R, gfull = cv.local_mean_force(q, F, p.beta)
    r = q[:, 4, :] - q[:, 0, :]
    e = r / torch.linalg.norm(r, dim=-1, keepdim=True)
    gradV_j = -F[:, 4, :]; gradV_i = -F[:, 0, :]
    expected = 0.5 * (e * (gradV_j - gradV_i)).sum(-1) - 2.0 / (p.beta * R)
    assert (f_loc - expected).abs().max() < 1e-9


def test_cpu_gpu_parity():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    p = pot.AlkaneParams(n_atoms=5, beta=1.0, sigma=2.3)
    cv = DistanceCV(0, 4)
    q = _rand_pentane(64, 4)
    f_cpu, _, _ = cv.local_mean_force(q, pot.forces(q, p), p.beta)
    qg = q.cuda()
    f_gpu, _, _ = cv.local_mean_force(qg, pot.forces(qg, p), p.beta)
    assert (f_cpu - f_gpu.cpu()).abs().max() < 1e-8


if __name__ == "__main__":
    import subprocess
    raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-q"]))
