"""Geometry + generalized-force validation for the alkane CV.

Run: CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_alkanes_geometry.py -q
"""
import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
torch.set_default_dtype(torch.float64)

from alkanes import geometry as geom  # noqa: E402
from alkanes import cv as cvmod  # noqa: E402
from alkanes import potentials as pot  # noqa: E402

PI = math.pi


def _rand_dihedrals(n, n_dih, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(n, n_dih, generator=g) * 2 - 1) * (PI - 1e-3)


# --------------------------- dihedral range & wrapping ---------------------------
def test_dihedral_range():
    q = geom.place_chain(_rand_dihedrals(200, 1, 1), n_atoms=4)
    phi = geom.signed_dihedral(q, 0, 1, 2, 3)
    assert torch.all(phi >= -PI - 1e-9) and torch.all(phi < PI + 1e-9)


def test_wrap_and_circular_diff():
    a = torch.tensor([3.0, -3.0, 0.1, PI - 0.01])
    assert torch.all(geom.wrap_to_pi(a) >= -PI) and torch.all(geom.wrap_to_pi(a) < PI)
    # circular diff across the branch cut is small
    d = geom.circular_diff(torch.tensor(PI - 0.01), torch.tensor(-PI + 0.01))
    assert abs(d.item()) < 0.05


def test_periodic_continuity():
    # phi sweeping past +pi wraps continuously to -pi
    targets = torch.linspace(PI - 0.05, -PI + 0.05, 5).reshape(-1, 1)
    q = geom.place_chain(targets, n_atoms=4)
    phi = geom.signed_dihedral(q, 0, 1, 2, 3)
    assert geom.circular_diff(phi, targets[:, 0]).abs().max() < 1e-9


# --------------------------- invariances ---------------------------
def test_translation_rotation_reflection():
    q = geom.place_chain(torch.tensor([[0.9, -1.3]]), n_atoms=5)
    phi = geom.signed_dihedral(q, 0, 1, 2, 3)
    # translation
    qT = q + torch.tensor([2.0, -3.0, 1.0])
    assert (geom.signed_dihedral(qT, 0, 1, 2, 3) - phi).abs().item() < 1e-10
    # proper rotation
    A = torch.linalg.qr(torch.randn(3, 3, generator=torch.Generator().manual_seed(3)))[0]
    if torch.det(A) < 0:
        A[:, 0] = -A[:, 0]
    qR = q @ A.T
    assert (geom.signed_dihedral(qR, 0, 1, 2, 3) - phi).abs().item() < 1e-10
    # reflection flips the sign
    qRef = q.clone()
    qRef[..., 2] = -qRef[..., 2]
    assert (geom.signed_dihedral(qRef, 0, 1, 2, 3) + phi).abs().item() < 1e-10


# --------------------------- builder round-trip & known conformers ---------------------------
def test_place_chain_roundtrip_butane():
    t = _rand_dihedrals(300, 1, 7)
    q = geom.place_chain(t, n_atoms=4)
    phi = geom.signed_dihedral(q, 0, 1, 2, 3)
    assert geom.circular_diff(phi, t[:, 0]).abs().max() < 1e-9


def test_place_chain_roundtrip_pentane():
    t = _rand_dihedrals(300, 2, 9)
    q = geom.place_chain(t, n_atoms=5)
    phi1 = geom.signed_dihedral(q, 0, 1, 2, 3)
    phi2 = geom.signed_dihedral(q, 1, 2, 3, 4)
    assert geom.circular_diff(phi1, t[:, 0]).abs().max() < 1e-9
    assert geom.circular_diff(phi2, t[:, 1]).abs().max() < 1e-9


def test_trans_is_planar_zigzag():
    q = geom.place_chain(torch.tensor([[0.0]]), n_atoms=4)[0]
    assert q[:, 2].abs().max() < 1e-9        # coplanar (trans zig-zag)
    # bonds equal the equilibrium value
    for a in range(3):
        assert abs((q[a + 1] - q[a]).norm().item() - 1.0) < 1e-9
    # interior C-C-C angle is pi - theta0 = 112 deg (bend convention)
    for (i, j, k) in [(0, 1, 2), (1, 2, 3)]:
        u = q[i] - q[j]; v = q[k] - q[j]
        interior = math.acos((u @ v).item() / (u.norm() * v.norm()).item())
        assert abs(interior - (math.pi - 1.187)) < 1e-6


def test_bend_angle_at_equilibrium_matches_theta0():
    # place_chain builds bend = theta0 => angle_energy is at its minimum (grad ~ 0)
    p = pot.AlkaneParams(n_atoms=5)
    q = geom.place_chain(torch.zeros(4, 2), n_atoms=5)
    # bond+angle-only Cartesian force must vanish at the built (equilibrium) geometry
    p_ba = pot.AlkaneParams(n_atoms=5, c1=0.0, c2=0.0, c3=0.0, decouple=True)
    assert pot.forces(q, p_ba).abs().max().item() < 1e-6


# --------------------------- V4 torsion minima / barriers ---------------------------
def test_V4_minima_barriers():
    p = pot.AlkaneParams(n_atoms=4)
    assert abs(pot.V4(torch.tensor(0.0), p).item()) < 1e-9            # trans = 0
    gauche = math.radians(116.57)
    assert pot.V4(torch.tensor(gauche), p).item() < pot.V4(torch.tensor(math.radians(90.0)), p).item()
    barrier = pot.V4(torch.tensor(math.radians(61.6)), p).item()
    assert 5.0 < barrier < 6.0                                        # trans<->gauche ~5.5
    assert pot.V4(torch.tensor(PI), p).item() > 7.0                   # cis barrier ~7.64


def test_V4_prime_matches_fd():
    p = pot.AlkaneParams(n_atoms=4)
    phi = torch.linspace(-PI + 0.1, PI - 0.1, 50)
    eps = 1e-6
    fd = (pot.V4(phi + eps, p) - pot.V4(phi - eps, p)) / (2 * eps)
    assert (pot.V4_prime(phi, p) - fd).abs().max() < 1e-6


# --------------------------- generalized-force geometric term ---------------------------
def test_grad_and_divv_vs_finite_difference():
    cv = cvmod.DihedralCV((0, 1, 2, 3))
    q = geom.place_chain(_rand_dihedrals(16, 1, 11), n_atoms=4)
    phi, grad_full, div_v = cv.geometry(q)
    # grad vs FD
    eps = 1e-6
    B = q.shape[0]
    gfd = torch.zeros(B, 4, 3)
    for a in range(4):
        for c in range(3):
            qp = q.clone(); qp[:, a, c] += eps
            qm = q.clone(); qm[:, a, c] -= eps
            gfd[:, a, c] = geom.circular_diff(cv.value(qp), cv.value(qm)) / (2 * eps)
    assert (grad_full[:, :4, :] - gfd).abs().max() < 1e-6
    # div(v) vs FD of v = grad/|grad|^2 (component-wise divergence)
    def vfield(qq):
        _, g, _ = cv.geometry(qq)
        gg = (g * g).sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
        return g / gg
    div_fd = torch.zeros(B)
    for a in range(4):
        for c in range(3):
            qp = q.clone(); qp[:, a, c] += eps
            qm = q.clone(); qm[:, a, c] -= eps
            div_fd += (vfield(qp)[:, a, c] - vfield(qm)[:, a, c]) / (2 * eps)
    assert (div_v - div_fd).abs().max() < 1e-4


# --------------------------- forces vs finite differences ---------------------------
@pytest.mark.parametrize("n_atoms,decouple", [(4, False), (5, False), (5, True)])
def test_forces_vs_fd(n_atoms, decouple):
    p = pot.AlkaneParams(n_atoms=n_atoms, decouple=decouple)
    n_dih = n_atoms - 3
    q = geom.place_chain(_rand_dihedrals(8, n_dih, 5), n_atoms=n_atoms)
    # perturb off equilibrium so bond/angle/LJ gradients are non-trivial
    q = q + 0.05 * torch.randn(q.shape, generator=torch.Generator().manual_seed(2))
    F = pot.forces(q, p)
    eps = 1e-6
    Ffd = torch.zeros_like(q)
    for a in range(n_atoms):
        for c in range(3):
            qp = q.clone(); qp[:, a, c] += eps
            qm = q.clone(); qm[:, a, c] -= eps
            Ffd[:, a, c] = -(pot.total_energy(qp, p) - pot.total_energy(qm, p)) / (2 * eps)
    assert (F - Ffd).abs().max() < 1e-5


# --------------------------- decoupled projection identity ---------------------------
def test_decoupled_projection_at_equilibrium():
    # At rigid equilibrium bonds/angles, grad(V).v == V4'(phi) exactly (no perp force).
    p = pot.AlkaneParams(n_atoms=4, decouple=True)
    cv = cvmod.DihedralCV((0, 1, 2, 3))
    q = geom.place_chain(_rand_dihedrals(32, 1, 13), n_atoms=4)   # exact eq bonds/angles
    F = pot.forces(q, p)
    _, grad_full, _ = cv.geometry(q)
    gg = (grad_full * grad_full).sum(dim=(-2, -1)).clamp_min(1e-12)
    gradV_dot_v = -(F * grad_full).sum(dim=(-2, -1)) / gg
    phi = cv.value(q)
    assert (gradV_dot_v - pot.V4_prime(phi, p)).abs().max() < 1e-6


# --------------------------- nonbonded exclusion convention ---------------------------
def test_nonbonded_exclusions():
    assert pot.AlkaneParams(n_atoms=4).nonbonded_pairs() == []       # butane: none
    assert pot.AlkaneParams(n_atoms=5).nonbonded_pairs() == [(0, 4)]  # pentane: 1-5 only


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_cpu_gpu_parity_forces_and_cv():
    # For a GIVEN configuration the physical forces and the generalized CV force must
    # agree bit-closely on CPU vs GPU in float64 (only RNG-driven dynamics differ).
    p = pot.AlkaneParams(n_atoms=5, beta=1.0, sigma=2.3)
    cv = cvmod.DihedralCV((0, 1, 2, 3))
    q_cpu = geom.place_chain(_rand_dihedrals(64, 2, 4), n_atoms=5) + \
        0.05 * torch.randn(64, 5, 3, generator=torch.Generator().manual_seed(1))
    q_gpu = q_cpu.cuda()
    F_cpu = pot.forces(q_cpu, p); F_gpu = pot.forces(q_gpu, p).cpu()
    assert (F_cpu - F_gpu).abs().max() < 1e-8
    fc, _, _ = cv.local_mean_force(q_cpu, F_cpu, p.beta)
    fg, _, _ = cv.local_mean_force(q_gpu, pot.forces(q_gpu, p), p.beta)
    assert (fc - fg.cpu()).abs().max() < 1e-7


if __name__ == "__main__":
    import subprocess
    raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-q"]))
