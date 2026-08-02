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

from alkanes.cv2d import (JointDihedralCV2D, abf_bias_force_2d,  # noqa: F401 re-exported
                          sym2x2_eigvals)

PI = math.pi
TWO_PI = 2.0 * PI
EPS = 1.0e-12


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


# ---------------------------------------------------------------------------------------------
class FastBackboneCV2D(BackboneCV2D):
    """Union-block den Otter mean force: identical maths, ~19x smaller Hessian tensors.

    ``alkanes.cv2d.JointDihedralCV2D`` scatters each 12x12 dihedral Hessian into the FULL
    ``(B, 2, 3A, 3A)`` coordinate layout -- 66x66 for alanine -- and then contracts
    ``einsum("pbi,pcij,pdj->pbcd")`` over all 66 coordinates.  But phi = (4,6,8,14) and
    psi = (6,8,14,16) touch only the **5-atom union {4,6,8,14,16} = 15 coordinates**; everywhere
    else ``g`` and ``H`` are exactly zero.  Restricting to that block leaves ``G``, ``tr H``,
    ``T_bcd``, ``div w`` and hence the local mean force **bit-identical** while shrinking the
    Hessian by (66/15)^2 = 19x and the contraction by ~39x.

    Measured on this system: ``local_mean_force`` is 31-35 ms at B = 2048-8192 in the dense form
    versus ~5 ms for the physical force, so it dominates the step ~7:1.  This class removes that.

    ``geometry()`` returns the per-CV gradient in the COMPACT layout ``g (B, 2, n_union, 3)``;
    use :meth:`scatter_bias` to push a CV-space bias into full Cartesian coordinates.
    """

    def __init__(self, atoms_a, atoms_b, n_atoms, ridge=0.0, reg_threshold=1e-8):
        super().__init__(atoms_a, atoms_b, n_atoms, ridge=ridge, reg_threshold=reg_threshold)
        self.union = sorted(set(self.atoms[0]) | set(self.atoms[1]))
        self.n_union = len(self.union)
        self.nc = 3 * self.n_union
        pos = {a: k for k, a in enumerate(self.union)}
        self._cidx = [torch.tensor([pos[a] * 3 + c for a in atoms for c in range(3)],
                                   dtype=torch.long) for atoms in self.atoms]
        self._uidx = torch.tensor(self.union, dtype=torch.long)

    def _grad_hess_union(self, q):
        from alkanes.cv import _grad_phi4, _hess_phi4
        B = q.shape[0]
        nc = self.nc
        gflat = q.new_zeros(B, 2, nc)
        H = q.new_zeros(B, 2, nc, nc)
        for a, atoms in enumerate(self.atoms):
            sub = q[:, atoms, :].reshape(B, 12).detach()
            ga = _grad_phi4(sub)                                    # (B,12)
            Ha = _hess_phi4(sub)                                    # (B,12,12)
            idx = self._cidx[a].to(q.device)
            gflat[:, a].index_copy_(1, idx, ga)
            rows = idx[:, None].expand(12, 12).reshape(-1)
            cols = idx[None, :].expand(12, 12).reshape(-1)
            H[:, a].reshape(B, nc * nc).scatter_add_(
                1, (rows * nc + cols)[None, :].expand(B, -1), Ha.reshape(B, 144))
        return gflat, H

    def geometry(self, q):
        B = q.shape[0]
        gflat, H = self._grad_hess_union(q)
        G = torch.einsum("pbi,pci->pbc", gflat, gflat)
        lam_min, lam_max = sym2x2_eigvals(G)
        det = G[:, 0, 0] * G[:, 1, 1] - G[:, 0, 1] * G[:, 1, 0]
        eye = torch.eye(2, device=q.device, dtype=q.dtype)[None]
        bad = lam_min < self.reg_threshold
        Greg = G + self.ridge * eye + torch.where(bad[:, None, None], 1e-6 * eye,
                                                  torch.zeros_like(eye))
        if self._reg_counter is None:
            self._reg_counter = torch.zeros((), device=q.device, dtype=torch.long)
        self._reg_counter = self._reg_counter + bad.sum()
        Ginv = torch.linalg.inv(Greg)
        lap = torch.diagonal(H, dim1=-2, dim2=-1).sum(-1)
        T = torch.einsum("pbi,pcij,pdj->pbcd", gflat, H, gflat)
        U = T + T.transpose(-1, -2)
        div_v = (torch.einsum("pab,pb->pa", Ginv, lap)
                 - torch.einsum("pac,pdb,pbcd->pa", Ginv, Ginv, U))
        p1, p2 = self.values(q)
        return {"phi": torch.stack([p1, p2], dim=-1).detach(),
                "g": gflat.reshape(B, 2, self.n_union, 3).detach(),
                "gflat": gflat.detach(), "div_v": div_v.detach(),
                "G": G.detach(), "Ginv": Ginv.detach(),
                "lam_min": lam_min.detach(), "lam_max": lam_max.detach(),
                "cond": (lam_max / lam_min.clamp_min(EPS)).detach(), "det": det.detach()}

    def local_mean_force(self, q, physical_forces, beta):
        geo = self.geometry(q)
        gu = geo["g"]                                               # (B,2,n_union,3)
        Fu = physical_forces[:, self._uidx.to(q.device), :]         # (B,n_union,3)
        Fdotg = (Fu[:, None] * gu).sum(dim=(-2, -1))                # (B,2)
        f = -torch.einsum("pab,pb->pa", geo["Ginv"], Fdotg) - (1.0 / beta) * geo["div_v"]
        return f, geo["phi"], gu, geo

    def grad_only(self, q):
        from alkanes.cv import _grad_phi4
        B = q.shape[0]
        g = q.new_zeros(B, 2, self.n_union, 3)
        pos = {a: k for k, a in enumerate(self.union)}
        for a, atoms in enumerate(self.atoms):
            sub = q[:, atoms, :].reshape(B, 12).detach()
            ga = _grad_phi4(sub).reshape(B, 4, 3)
            for t, at in enumerate(atoms):
                g[:, a, pos[at]] += ga[:, t]
        p1, p2 = self.values(q)
        return torch.stack([p1, p2], dim=-1).detach(), g.detach()

    def scatter_bias(self, g_compact, c1, c2, n_atoms):
        """CV-space bias ``(c1, c2)`` -> full Cartesian force ``(B, n_atoms, 3)``."""
        B = g_compact.shape[0]
        loc = (c1[:, None, None] * g_compact[:, 0] + c2[:, None, None] * g_compact[:, 1])
        out = g_compact.new_zeros(B, n_atoms, 3)
        out.index_copy_(1, self._uidx.to(g_compact.device), loc)
        return out
