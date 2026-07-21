"""End-to-end distance collective variable ``R = |q_j - q_i|`` and its ABF
generalized mean force (united-atom alkanes, 3-D, batched, float64-safe).

For the non-periodic distance CV ``xi = R(q) = |q_j - q_i|`` with ``r = q_j - q_i`` and
``e = r/R`` we have the elementary exact geometry

    grad_{q_j} R = e,   grad_{q_i} R = -e,   |grad R|^2 = 2,

and, taking the den Otter vector field ``v = grad R / |grad R|^2 = grad R / 2`` (so that
``v . grad R = 1``), the geometric term is exactly

    div v = (1/2) Laplacian(R) = (1/2)(4/R) = 2/R,

because ``H_R grad R = 0`` (``|grad R|^2`` is constant so its gradient
``2 H_R grad R`` vanishes), which kills the ``-2 (grad^T H grad)/|grad|^4`` term.  Hence
the instantaneous local mean force whose conditional average is ``dF/dR`` is

    f_R(q) = grad V . v - beta^{-1} div v
           = (1/2) e . (grad_{q_j} V - grad_{q_i} V) - 2/(beta R).

The analytic gradient/divergence are used in the dynamics; a ``torch.func`` autodiff
path (:meth:`DistanceCV.geometry_autodiff`) validates them.  R is bounded away from 0
under the stiff bond potential (pentane R15 ~ [1.5, 3.9]), so no floor is needed, but a
small one guards pathological inputs and is reported, not silently applied.

The ABF/OPES bias force applied to the Cartesian coordinates is ``+A'(R) grad R`` (a
conservative force flattening the R marginal when ``A = F``), identical channel to the
dihedral CV.
"""
from __future__ import annotations

import torch
from torch.func import grad, hessian, vmap

EPS = 1.0e-12


def _dist2(qflat):
    """Scalar distance between the two atoms of a flattened 6-vector ``(6,)``."""
    d = qflat[3:6] - qflat[0:3]
    return torch.sqrt((d * d).sum().clamp_min(EPS ** 2))


_grad_dist2 = vmap(grad(_dist2))
_hess_dist2 = vmap(hessian(_dist2))


class DistanceCV:
    """Euclidean distance between atoms ``i`` and ``j`` of a chain."""

    def __init__(self, i, j):
        assert i != j
        self.i = int(i)
        self.j = int(j)

    def value(self, q):
        """R(q) = |q_j - q_i|; ``q`` is ``(B, n_atoms, 3)`` -> ``(B,)``."""
        r = q[..., self.j, :] - q[..., self.i, :]
        return torch.linalg.norm(r, dim=-1)

    def geometry(self, q):
        """Analytic ``(R, grad_full (B,n_atoms,3), div_v (B,))``; outputs detached."""
        B, n_atoms, _ = q.shape
        r = (q[:, self.j, :] - q[:, self.i, :]).detach()          # (B,3)
        R = torch.linalg.norm(r, dim=-1).clamp_min(EPS)           # (B,)
        e = r / R[:, None]
        grad_full = q.new_zeros(B, n_atoms, 3)
        grad_full[:, self.j, :] = e
        grad_full[:, self.i, :] = -e
        div_v = 2.0 / R                                           # (B,)
        return R.detach(), grad_full.detach(), div_v.detach()

    def geometry_autodiff(self, q):
        """Autodiff ``(R, grad_full, div_v)`` (validation reference for :meth:`geometry`).

        ``div_v = Laplacian(R)/|grad R|^2 - 2 (grad^T H grad)/|grad R|^4`` with the exact
        6-coordinate Hessian; equals ``2/R`` analytically.
        """
        B, n_atoms, _ = q.shape
        sub = torch.stack([q[:, self.i, :], q[:, self.j, :]], dim=1).reshape(B, 6).detach()
        g = _grad_dist2(sub)                                      # (B,6)
        H = _hess_dist2(sub)                                      # (B,6,6)
        gg = (g * g).sum(-1)                                      # ~2
        lap = torch.diagonal(H, dim1=-2, dim2=-1).sum(-1)
        gHg = torch.einsum("bi,bij,bj->b", g, H, g)
        div_v = lap / gg.clamp_min(EPS) - 2.0 * gHg / gg.clamp_min(EPS) ** 2
        R = self.value(q)
        grad_full = q.new_zeros(B, n_atoms, 3)
        grad_full[:, self.i, :] = g[:, 0:3]
        grad_full[:, self.j, :] = g[:, 3:6]
        return R.detach(), grad_full.detach(), div_v.detach()

    def local_mean_force(self, q, physical_forces, beta):
        """``f_R = grad V . v - beta^{-1} div v`` with ``v = grad R / |grad R|^2``.

        ``physical_forces = -grad V`` of shape ``(B, n_atoms, 3)``.  Returns
        ``(f_loc (B,), R (B,), grad_full (B,n_atoms,3))``.
        """
        R, grad_full, div_v = self.geometry(q)
        gg = (grad_full * grad_full).sum(dim=(-2, -1)).clamp_min(EPS)   # = 2
        gradV_dot_v = -(physical_forces * grad_full).sum(dim=(-2, -1)) / gg
        f_loc = gradV_dot_v - (1.0 / beta) * div_v
        return f_loc, R, grad_full


def dist_bias_force(grad_full, mean_force_at_R):
    """Cartesian ABF/OPES bias force ``+A'(R) grad R`` -> ``(B, n_atoms, 3)``."""
    return mean_force_at_R[:, None, None] * grad_full
