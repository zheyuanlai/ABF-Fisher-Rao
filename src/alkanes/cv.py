"""Collective variable and the correct ABF generalized mean force for a dihedral.

For a nonlinear CV ``xi = phi(q)`` the local mean force is NOT ``partial V / partial
phi``.  Using any vector field ``v`` with ``v . grad(phi) = 1`` the exact local mean
force whose conditional average is ``dF/dphi`` is

    f_loc(q) = grad(V) . v  -  beta^{-1} div(v),                       (den Otter)

and we take the natural choice ``v = grad(phi) / |grad(phi)|^2``, for which

    div(v) = Laplacian(phi)/|grad(phi)|^2  -  2 (grad(phi)^T H grad(phi))/|grad(phi)|^4,

with ``H`` the Hessian of ``phi``.  The geometric term ``-beta^{-1} div(v)`` does not
vanish for a dihedral (it does for the linear toy CV ``xi = x``).  ``grad(phi)`` and
``H`` are obtained by exact autodiff (``torch.func``), validated against finite
differences and against the analytic reduction on the decoupled model (B0/P0).

Only the four atoms of the dihedral enter ``phi``; the gradient/Hessian are computed
on those 12 coordinates and scattered back into the full ``(n_atoms, 3)`` layout, so
pentane's 5th atom (irrelevant to ``phi1``) costs nothing here.

The ABF biasing force applied to the Cartesian coordinates is ``+A'(phi) grad(phi)``
(a conservative force ``-grad[-A(phi(q))]``); when ``A = F`` it flattens the phi
marginal exactly.
"""
from __future__ import annotations

import torch
from torch.func import grad, hessian, vmap

from . import geometry as geom

EPS = 1.0e-12


def _phi4(qflat):
    """Scalar RB dihedral of a flattened 4-atom coordinate vector ``(12,)``."""
    q = qflat.reshape(4, 3)
    return geom.signed_dihedral(q, 0, 1, 2, 3, convention="rb")


_grad_phi4 = vmap(grad(_phi4))
_hess_phi4 = vmap(hessian(_phi4))


class DihedralCV:
    """A dihedral collective variable over four consecutive atoms ``atoms``."""

    def __init__(self, atoms=(0, 1, 2, 3)):
        self.atoms = tuple(atoms)
        assert len(self.atoms) == 4

    def value(self, q):
        """phi(q) in [-pi, pi); ``q`` is ``(B, n_atoms, 3)``."""
        i, j, k, l = self.atoms
        return geom.signed_dihedral(q, i, j, k, l, convention="rb")

    def geometry(self, q):
        """Return ``(phi, grad_full, div_v)`` for the batch.

        ``grad_full`` has shape ``(B, n_atoms, 3)`` (zero off the four CV atoms);
        ``div_v`` has shape ``(B,)``.  Exact autodiff; outputs are detached.
        """
        B, n_atoms, _ = q.shape
        sub = q[:, self.atoms, :].reshape(B, 12).detach()
        g = _grad_phi4(sub)                       # (B, 12)
        H = _hess_phi4(sub)                        # (B, 12, 12)
        gg = (g * g).sum(-1)                       # |grad phi|^2, (B,)
        lap = torch.diagonal(H, dim1=-2, dim2=-1).sum(-1)          # Laplacian(phi)
        gHg = torch.einsum("bi,bij,bj->b", g, H, g)
        div_v = lap / gg.clamp_min(EPS) - 2.0 * gHg / gg.clamp_min(EPS) ** 2
        phi = self.value(q)
        grad_full = q.new_zeros(B, n_atoms, 3)
        grad_full[:, self.atoms, :] = g.reshape(B, 4, 3)
        return phi.detach(), grad_full.detach(), div_v.detach()

    def local_mean_force(self, q, physical_forces, beta):
        """Instantaneous local mean force ``f_loc = grad(V).v - beta^{-1} div(v)``.

        ``physical_forces = -grad V`` of shape ``(B, n_atoms, 3)``.  With
        ``v = grad(phi)/|grad phi|^2``, ``grad(V).v = -(F . grad(phi))/|grad phi|^2``.
        Returns ``(f_loc, phi, grad_full)``.
        """
        phi, grad_full, div_v = self.geometry(q)
        gg = (grad_full * grad_full).sum(dim=(-2, -1)).clamp_min(EPS)   # |grad phi|^2
        gradV_dot_v = -(physical_forces * grad_full).sum(dim=(-2, -1)) / gg
        f_loc = gradV_dot_v - (1.0 / beta) * div_v
        return f_loc, phi, grad_full


def abf_bias_force(grad_full, mean_force_at_phi):
    """Cartesian ABF/OPES bias force ``+A'(phi) grad(phi)`` -> ``(B, n_atoms, 3)``."""
    return mean_force_at_phi[:, None, None] * grad_full
