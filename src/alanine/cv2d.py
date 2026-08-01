"""IUPAC-convention joint ``(phi, psi)`` CV on ``T^2`` for the peptide backbone.

The repo's :class:`alkanes.cv2d.JointDihedralCV2D` computes the exact den Otter vector
generalized mean force for a pair of dihedrals, but it inherits ``convention='rb'`` from
:func:`alkanes.cv._phi4` -- *trans at 0*, which is the alkane torsion convention.  Peptides use
**IUPAC: trans at pi**.

The two conventions differ by exactly a constant shift of ``pi``:

    phi_IUPAC = wrap(phi_rb + pi),

because ``rb`` returns ``atan2(-y, -x)`` and IUPAC returns ``atan2(y, x)``.  A constant shift has
zero derivative, so **gradients, Hessians, the Gram matrix, the dual fields and the local mean
force are all numerically identical**; only the CV values and hence the grid labels move.  This
wrapper therefore adds the shift to the reported angles and leaves every geometric quantity
untouched, rather than duplicating the (validated) autodiff machinery.

Consequence for the torus grid: with IUPAC labels the cell-centred grid on ``[-pi, pi)`` places
C7eq near ``(-79 deg, +56 deg)`` and C7ax near ``(+64 deg, -41 deg)``, matching the literature
Ramachandran layout.
"""
from __future__ import annotations

import math

import torch

from alkanes.cv2d import JointDihedralCV2D, abf_bias_force_2d  # noqa: F401  (re-exported)

PI = math.pi
TWO_PI = 2.0 * PI


def wrap_to_pi(a):
    """Wrap angles into ``[-pi, pi)``."""
    return (a + PI) % TWO_PI - PI


def rb_to_iupac(a):
    """Convert an ``rb`` (trans at 0) angle to IUPAC (trans at pi)."""
    return wrap_to_pi(a + PI)


def iupac_to_rb(a):
    """Inverse of :func:`rb_to_iupac` (the shift is its own inverse up to wrapping)."""
    return wrap_to_pi(a - PI)


class BackboneCV2D(JointDihedralCV2D):
    """``(phi, psi)`` with IUPAC values; all geometry inherited unchanged.

    ``atoms_a``/``atoms_b`` are REQUIRED -- the base class defaults to pentane's
    ``(0,1,2,3)/(1,2,3,4)`` with ``n_atoms=5`` and never reads ``self.n_atoms``, so a forgotten
    override would silently compute the wrong dihedrals on a correctly shaped tensor.
    """

    def __init__(self, atoms_a, atoms_b, n_atoms, ridge=0.0, reg_threshold=1e-8):
        if atoms_a is None or atoms_b is None or n_atoms is None:
            raise ValueError("atom indices and n_atoms are required (no pentane defaults here)")
        super().__init__(atoms_a=atoms_a, atoms_b=atoms_b, n_atoms=n_atoms,
                         ridge=ridge, reg_threshold=reg_threshold)

    # -- values (IUPAC) ------------------------------------------------------
    # This is the ONLY override needed.  The parent's grad_only(), geometry() and
    # local_mean_force() all obtain their angles through ``self.values(q)``
    # (alkanes/cv2d.py:99, :134, :161-166), so overriding values() propagates the convention
    # to every consumer exactly once.  Shifting again in those methods would double-convert
    # back to 'rb' -- measured as phi = +105.05 deg instead of -74.95 deg.
    def values(self, q):
        p1, p2 = super().values(q)
        return rb_to_iupac(p1), rb_to_iupac(p2)
