"""United-atom alkane potential: bonds + angles + RB torsion + LJ (with exclusions).

Model (Ryckaert--Bellemans / Cancès--Legoll--Stoltz, reduced units)
-------------------------------------------------------------------
``V = V_bond + V_angle + V_torsion + V_nonbonded`` with

    V_bond   = sum_bonds   (k0/2)(r - d0)^2
    V_angle  = sum_angles  (k_theta/2)(theta - theta0)^2
    V_torsion= sum_dihedrals V4(phi)
    V4(phi)  = c1(1-cos phi) + 2 c2(1-cos^2 phi) + c3(1+3cos phi-4cos^3 phi)
    V_nb     = sum_{included pairs} 4 eps [(sigma/r)^12 - (sigma/r)^6]

Nonbonded exclusion convention (documented + tested)
----------------------------------------------------
Ryckaert--Bellemans convention: **exclude every pair separated by <= 3 bonds**
(1-2 bonded, 1-3 angle, 1-4 the pair the dihedral already describes); LJ acts only
between atoms separated by >= 4 bonds.  Consequences, which drive the science:

* **Butane (4 sites):** the only non-bonded pair (1-4) is 3 bonds apart and hence
  excluded, so butane has **no LJ term**.  Its dihedral free energy is exactly
  ``F(phi1) = V4(phi1) + C`` (the bond/angle internal-coordinate Jacobian is
  phi-independent).  Butane is the clean easy control.
* **Pentane (5 sites):** the single pair 1-5 is 4 bonds apart and is **included**;
  this 1-5 LJ is the "pentane effect" that couples phi1 and phi2 and makes phi2 a
  hidden slow coordinate when only phi1 is biased.

``decouple=True`` turns the LJ term off (the B0/P0 exact-reference switch).
Torsions are never coupled to each other except through V_nb.

Forces are ``-grad V`` by autograd (exact; small molecule so this is cheap and
avoids hand-coded bond/angle/torsion/LJ gradients).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import torch

from . import geometry as geom

EPS = 1.0e-12


@dataclass(frozen=True)
class AlkaneParams:
    n_atoms: int = 4            # 4 butane, 5 pentane
    d0: float = 1.0
    k0: float = 1000.0
    theta0: float = 1.187
    k_theta: float = 208.0
    c1: float = 1.18
    c2: float = -0.23
    c3: float = 2.64
    sigma: float = 1.0
    epsilon: float = 1.0
    beta: float = 1.0
    decouple: bool = False      # True => LJ off (exact decoupled reference switch)
    r_min_factor: float = 0.60  # floor on LJ pair distance as a fraction of sigma
    force_clip: float = 0.0     # 0 => no clip; else clip per-atom force norm

    @property
    def n_dihedrals(self) -> int:
        return self.n_atoms - 3

    def dihedral_atom_indices(self) -> List[Tuple[int, int, int, int]]:
        return [(a, a + 1, a + 2, a + 3) for a in range(self.n_dihedrals)]

    def bond_pairs(self) -> List[Tuple[int, int]]:
        return [(a, a + 1) for a in range(self.n_atoms - 1)]

    def angle_triples(self) -> List[Tuple[int, int, int]]:
        return [(a, a + 1, a + 2) for a in range(self.n_atoms - 2)]

    def nonbonded_pairs(self) -> List[Tuple[int, int]]:
        """Pairs separated by >= 4 bonds (RB exclusion of 1-2, 1-3, 1-4)."""
        pairs = []
        for i in range(self.n_atoms):
            for j in range(i + 1, self.n_atoms):
                if (j - i) >= 4:
                    pairs.append((i, j))
        return pairs


# ---------------------------------------------------------------------------
# Analytic torsion (also used for references and tests)
# ---------------------------------------------------------------------------
def V4(phi, p: AlkaneParams):
    cp = torch.cos(phi) if isinstance(phi, torch.Tensor) else __import__("math").cos(phi)
    return (p.c1 * (1.0 - cp)
            + 2.0 * p.c2 * (1.0 - cp ** 2)
            + p.c3 * (1.0 + 3.0 * cp - 4.0 * cp ** 3))


def V4_prime(phi, p: AlkaneParams):
    """dV4/dphi (analytic)."""
    cp = torch.cos(phi)
    sp = torch.sin(phi)
    # d/dphi of V4 = (c1 + 4 c2 cp - 3 c3 - 12 c3 cp^2? ) * ... derive via chain rule
    # V4 = c1(1-cp) + 2c2(1-cp^2) + c3(1+3cp-4cp^3); dcp/dphi = -sp
    dV_dcp = -p.c1 - 4.0 * p.c2 * cp + p.c3 * (3.0 - 12.0 * cp ** 2)
    return dV_dcp * (-sp)


# ---------------------------------------------------------------------------
# Energy terms (batched; q is (B, n_atoms, 3))
# ---------------------------------------------------------------------------
def bond_energy(q, p: AlkaneParams):
    e = q.new_zeros(q.shape[0])
    for (i, j) in p.bond_pairs():
        r = torch.linalg.norm(q[:, i, :] - q[:, j, :], dim=-1)
        e = e + 0.5 * p.k0 * (r - p.d0) ** 2
    return e


def angle_energy(q, p: AlkaneParams):
    """Harmonic bend term on the angle between successive *forward* bond vectors.

    theta0 = 1.187 rad is the bend (= pi - interior angle); pi - 1.187 = 112.0 deg is
    the physical C-C-C interior angle.  Using the forward-bond-vector (bend)
    convention consistently with :func:`geometry.place_chain`.
    """
    e = q.new_zeros(q.shape[0])
    for (i, j, k) in p.angle_triples():
        b1 = q[:, j, :] - q[:, i, :]        # forward bond i->j
        b2 = q[:, k, :] - q[:, j, :]        # forward bond j->k
        b1n = torch.linalg.norm(b1, dim=-1).clamp_min(EPS)
        b2n = torch.linalg.norm(b2, dim=-1).clamp_min(EPS)
        cos_t = ((b1 * b2).sum(-1) / (b1n * b2n)).clamp(-1.0 + 1e-9, 1.0 - 1e-9)
        theta = torch.arccos(cos_t)
        e = e + 0.5 * p.k_theta * (theta - p.theta0) ** 2
    return e


def torsion_energy(q, p: AlkaneParams):
    e = q.new_zeros(q.shape[0])
    for (i, j, k, l) in p.dihedral_atom_indices():
        phi = geom.signed_dihedral(q, i, j, k, l)
        e = e + V4(phi, p)
    return e


def nonbonded_energy(q, p: AlkaneParams):
    e = q.new_zeros(q.shape[0])
    if p.decouple:
        return e
    r_floor = p.r_min_factor * p.sigma
    for (i, j) in p.nonbonded_pairs():
        r = torch.linalg.norm(q[:, i, :] - q[:, j, :], dim=-1).clamp_min(r_floor)
        inv6 = (p.sigma / r) ** 6
        e = e + 4.0 * p.epsilon * (inv6 ** 2 - inv6)
    return e


def total_energy(q, p: AlkaneParams):
    return bond_energy(q, p) + angle_energy(q, p) + torsion_energy(q, p) + nonbonded_energy(q, p)


def forces(q, p: AlkaneParams):
    """Physical force ``-grad V`` by autograd; returns (B, n_atoms, 3)."""
    q = q.detach().requires_grad_(True)
    e = total_energy(q, p).sum()
    (g,) = torch.autograd.grad(e, q, create_graph=False)
    f = -g
    if p.force_clip and p.force_clip > 0:
        norm = torch.linalg.norm(f, dim=-1, keepdim=True)
        scale = torch.clamp(p.force_clip / norm.clamp_min(EPS), max=1.0)
        f = f * scale
    return f.detach()
