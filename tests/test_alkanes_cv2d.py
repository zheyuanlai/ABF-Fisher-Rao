"""2-D joint dihedral CV: Gram, dual biorthogonality, analytic divergence, decoupled
reduction, CPU/GPU parity.

Run: CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_alkanes_cv2d.py -q
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
from alkanes.cv2d import JointDihedralCV2D, abf_bias_force_2d  # noqa: E402

PI = math.pi


def _rand_pentane(n, seed=0, jitter=0.05):
    g = torch.Generator().manual_seed(seed)
    dih = (torch.rand(n, 2, generator=g) * 2 - 1) * (PI - 0.2)
    q = geom.place_chain(dih, n_atoms=5)
    return q + jitter * torch.randn(q.shape, generator=g)


def test_dual_biorthogonality():
    cv = JointDihedralCV2D()
    q = _rand_pentane(48, 1)
    w, geo = cv.dual_fields(q)                      # (B,2,A,3)
    g = geo["g"]
    # w_a . g_c = delta_ac
    M = torch.einsum("paAc,pbAc->pab", w, g)        # (B,2,2)
    eye = torch.eye(2)[None]
    assert (M - eye).abs().max() < 1e-9


def test_gram_symmetric_positive_definite():
    cv = JointDihedralCV2D()
    q = _rand_pentane(64, 2)
    geo = cv.geometry(q)
    G = geo["G"]
    assert (G - G.transpose(-1, -2)).abs().max() < 1e-12
    assert torch.all(geo["lam_min"] > 0)           # PD (grad phi1, grad phi2 independent)
    assert torch.all(geo["cond"] < 1e4)            # well-conditioned at generic geometries


def test_divergence_analytic_vs_fd_of_dual():
    cv = JointDihedralCV2D()
    q = _rand_pentane(10, 7)
    geo = cv.geometry(q)
    div_fd = cv.divergence_autodiff(q, eps=1e-5)
    assert (geo["div_v"] - div_fd).abs().max() < 1e-4


def test_decoupled_projection_reduces_to_V4prime():
    # LJ off + rigid equilibrium bonds/angles => grad V . w_a == V4'(phi_a) exactly.
    p = pot.AlkaneParams(n_atoms=5, decouple=True)
    cv = JointDihedralCV2D()
    g = torch.Generator().manual_seed(13)
    dih = (torch.rand(48, 2, generator=g) * 2 - 1) * (PI - 0.2)
    q = geom.place_chain(dih, n_atoms=5)           # exact equilibrium bonds/angles
    F = pot.forces(q, p)
    geo = cv.geometry(q)
    Ginv = geo["Ginv"]; gg = geo["g"]
    Fdotg = (F[:, None] * gg).sum(dim=(-2, -1))    # (B,2)
    gradV_dot_w = -torch.einsum("pab,pb->pa", Ginv, Fdotg)
    phi1, phi2 = cv.values(q)
    assert (gradV_dot_w[:, 0] - pot.V4_prime(phi1, p)).abs().max() < 1e-6
    assert (gradV_dot_w[:, 1] - pot.V4_prime(phi2, p)).abs().max() < 1e-6


def test_local_mean_force_matches_manual_construction():
    p = pot.AlkaneParams(n_atoms=5, beta=1.0, sigma=2.3)
    cv = JointDihedralCV2D()
    q = _rand_pentane(32, 5)
    F = pot.forces(q, p)
    f, phi, g, geo = cv.local_mean_force(q, F, p.beta)
    Ginv = geo["Ginv"]
    Fdotg = (F[:, None] * g).sum(dim=(-2, -1))
    manual = -torch.einsum("pab,pb->pa", Ginv, Fdotg) - (1.0 / p.beta) * geo["div_v"]
    assert (f - manual).abs().max() < 1e-12


def test_bias_force_projects_correctly():
    # applying bias A'_a along CV a produces + sum_a A'_a grad phi_a
    cv = JointDihedralCV2D()
    q = _rand_pentane(16, 3)
    geo = cv.geometry(q)
    g = geo["g"]
    A = torch.tensor([[1.3, -0.7]]).expand(16, 2).contiguous()
    bf = abf_bias_force_2d(g, A)
    manual = A[:, 0, None, None] * g[:, 0] + A[:, 1, None, None] * g[:, 1]
    assert (bf - manual).abs().max() < 1e-12


def test_cpu_gpu_parity():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    p = pot.AlkaneParams(n_atoms=5, beta=1.0, sigma=2.3)
    cv = JointDihedralCV2D()
    q = _rand_pentane(64, 9)
    f_cpu, _, _, _ = cv.local_mean_force(q, pot.forces(q, p), p.beta)
    qg = q.cuda()
    f_gpu, _, _, _ = cv.local_mean_force(qg, pot.forces(qg, p), p.beta)
    assert (f_cpu - f_gpu.cpu()).abs().max() < 1e-7


if __name__ == "__main__":
    import subprocess
    raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-q"]))
