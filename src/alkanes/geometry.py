"""Signed dihedral geometry for united-atom alkanes (3-D, batched, float64-safe).

A dihedral is degenerate in 2-D (four coplanar points give phi in {0, pi}), so the
alkanes live in R^3: coordinates are ``q`` of shape ``(..., n_atoms, 3)``.

Conventions
-----------
The dihedral is returned in the **Ryckaert--Bellemans / polymer convention** in
which the *trans* (all-anti, extended planar zig-zag) conformation is ``phi = 0``
and *cis* (eclipsed) is ``phi = +-pi``.  This is the convention in which the RB
torsion ``V4`` of :mod:`alkanes.potentials` has its global minimum at ``phi = 0``.
It equals the IUPAC dihedral shifted by ``pi`` (``atan2(-y, -x)`` instead of
``atan2(y, x)``).  ``phi`` is wrapped to ``[-pi, pi)``.

The signed dihedral uses the robust "Praxeolitic" atan2 construction (no arccos),
with small floors on the vector norms so nearly-collinear geometries do not blow up
(they are rare under the stiff bond-angle potential and are reported, not silently
mangled).  Gradients of ``phi`` are exact under autograd (validated vs finite
differences and vs ``torch.func.hessian``).
"""
from __future__ import annotations

import math

import torch

PI = math.pi
TWO_PI = 2.0 * math.pi
_NORM_FLOOR = 1.0e-10


def wrap_to_pi(a):
    """Map angle(s) to the periodic interval ``[-pi, pi)``."""
    return torch.remainder(a + PI, TWO_PI) - PI


def circular_diff(a, b):
    """Signed circular difference ``wrap(a - b) = atan2(sin(a-b), cos(a-b))``."""
    d = a - b
    return torch.atan2(torch.sin(d), torch.cos(d))


def signed_dihedral(q, i=0, j=1, k=2, l=3, convention="rb"):
    """Signed dihedral ``phi(q_i, q_j, q_k, q_l)`` in ``[-pi, pi)``.

    ``q`` has shape ``(..., n_atoms, 3)``; ``i,j,k,l`` select the four atoms.
    ``convention='rb'`` (default) puts *trans* at 0; ``'iupac'`` puts *trans* at pi.
    """
    p0 = q[..., i, :]
    p1 = q[..., j, :]
    p2 = q[..., k, :]
    p3 = q[..., l, :]
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1n = b1 / torch.linalg.norm(b1, dim=-1, keepdim=True).clamp_min(_NORM_FLOOR)
    # components of b0, b2 perpendicular to the central bond b1
    v = b0 - (b0 * b1n).sum(-1, keepdim=True) * b1n
    w = b2 - (b2 * b1n).sum(-1, keepdim=True) * b1n
    x = (v * w).sum(-1)
    y = (torch.linalg.cross(b1n, v, dim=-1) * w).sum(-1)
    if convention == "rb":
        # trans = 0: shift the IUPAC angle by pi <=> negate both atan2 args
        return torch.atan2(-y, -x)
    return torch.atan2(y, x)


def remove_com(q):
    """Subtract the (unit-mass) centre of mass over the atom axis.

    Kills the translational zero modes without touching any internal coordinate
    (bond lengths, angles, dihedrals are all translation-invariant).  Rotational
    invariance is intrinsic to ``V`` and the dihedral CV and is left free.
    """
    return q - q.mean(dim=-2, keepdim=True)


# ---------------------------------------------------------------------------
# Canonical chain builder (NeRF / Z-matrix), the exact inverse of signed_dihedral
# in the RB convention. Used to generate reference geometries and test inputs.
# ---------------------------------------------------------------------------
def _place_next(A, B, C, bond, angle, dihedral):
    """Place atom D given placed A,B,C and internal coords (bond CD, angle BCD, dihedral ABCD).

    ``dihedral`` is in the RB (trans=0) convention, matching :func:`signed_dihedral`.
    Standard Natural-Extension-Reference-Frame construction. All args broadcast over
    a leading batch dimension; A,B,C are ``(...,3)`` and bond/angle/dihedral ``(...,)``.
    """
    def _n(x):
        return x / torch.linalg.norm(x, dim=-1, keepdim=True).clamp_min(_NORM_FLOOR)

    bc = _n(C - B)
    n = _n(torch.linalg.cross(B - A, bc, dim=-1))
    m = torch.linalg.cross(n, bc, dim=-1)
    b = bond[..., None]
    th = angle[..., None]
    # RB-convention dihedral. The overall handedness of this NeRF frame is opposite
    # to signed_dihedral's, so we negate chi (calibrated by round-trip test: without
    # this, place_chain(chi) yields signed_dihedral == -chi).
    chi = -dihedral[..., None]
    # +cos(th) on the bc axis => the *bend* angle (between forward bond vectors C-B
    # and D-C) equals ``angle`` (th), matching potentials.angle_energy; the interior
    # C-C-C angle is then pi-th. The in-plane (m,n) terms set the dihedral (calibrated
    # by round-trip). The dihedral is invariant to the bc-axis sign.
    d2 = (torch.cos(th) * bc
          - torch.sin(th) * torch.cos(chi) * m
          + torch.sin(th) * torch.sin(chi) * n)
    return C + b * d2


def place_chain_internal(bonds, angles, dihedrals, n_atoms, device=None, dtype=torch.float64):
    """Build chains from explicit internal coordinates (variable bonds/angles).

    ``bonds``   : ``(B, n_atoms-1)`` bond lengths r_{a,a+1}.
    ``angles``  : ``(B, n_atoms-2)`` bond angles theta_{a,a+1,a+2}.
    ``dihedrals``: ``(B, n_atoms-3)`` RB-convention dihedrals.

    The exact inverse of the internal-coordinate readout (bonds, angles,
    :func:`signed_dihedral`), used to build the internal-coordinate reference
    ensemble.  COM removed.
    """
    bonds = torch.as_tensor(bonds, dtype=dtype, device=device)
    angles = torch.as_tensor(angles, dtype=dtype, device=device)
    dihedrals = torch.as_tensor(dihedrals, dtype=dtype, device=device)
    B = bonds.shape[0]
    dev = bonds.device
    q = torch.zeros(B, n_atoms, 3, dtype=dtype, device=dev)
    q[:, 1, 0] = bonds[:, 0]
    a1 = angles[:, 0]                        # bend angle at atom 1
    q[:, 2, :] = q[:, 1, :] + torch.stack(
        [bonds[:, 1] * torch.cos(a1), bonds[:, 1] * torch.sin(a1),
         torch.zeros_like(a1)], dim=-1)
    for a in range(3, n_atoms):
        q[:, a, :] = _place_next(q[:, a - 3, :], q[:, a - 2, :], q[:, a - 1, :],
                                 bonds[:, a - 1], angles[:, a - 2], dihedrals[:, a - 3])
    return remove_com(q)


def place_chain(dihedrals, n_atoms, d0=1.0, theta0=1.187, device=None, dtype=torch.float64):
    """Build alkane chains with equilibrium bonds/angles and the given dihedral(s).

    Parameters
    ----------
    dihedrals : tensor ``(B, n_dih)`` in the RB convention, ``n_dih = n_atoms - 3``.
    n_atoms   : 4 (butane) or 5 (pentane).

    Returns ``q`` of shape ``(B, n_atoms, 3)`` with COM removed.  By construction
    ``signed_dihedral`` recovers the input dihedrals (validated in the tests).
    """
    dihedrals = torch.as_tensor(dihedrals, dtype=dtype, device=device)
    if dihedrals.ndim == 1:
        dihedrals = dihedrals[:, None]
    B = dihedrals.shape[0]
    n_dih = n_atoms - 3
    assert dihedrals.shape[1] == n_dih, f"expected {n_dih} dihedrals, got {dihedrals.shape[1]}"
    dev = dihedrals.device
    q = torch.zeros(B, n_atoms, 3, dtype=dtype, device=dev)
    bond = torch.full((B,), float(d0), dtype=dtype, device=dev)
    ang = torch.full((B,), float(theta0), dtype=dtype, device=dev)
    # seed the first three atoms in a fixed plane realising bond d0 and angle theta0
    q[:, 0, :] = torch.tensor([0.0, 0.0, 0.0], dtype=dtype, device=dev)
    q[:, 1, :] = torch.tensor([d0, 0.0, 0.0], dtype=dtype, device=dev)
    # atom2 at bond d0 from atom1 with bend angle theta0 at atom1 (xy-plane)
    off = torch.stack([d0 * torch.cos(ang), d0 * torch.sin(ang),
                       torch.zeros_like(ang)], dim=-1)
    q[:, 2, :] = q[:, 1, :] + off
    for a in range(3, n_atoms):
        q[:, a, :] = _place_next(q[:, a - 3, :], q[:, a - 2, :], q[:, a - 1, :],
                                 bond, ang, dihedrals[:, a - 3])
    return remove_com(q)
