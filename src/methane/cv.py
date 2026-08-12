"""The methane collective variable ``xi = |Q_2 - Q_1|`` under minimum image, and its mean force.

This is ``alkanes.distance_cv.DistanceCV`` with one change -- the separation is taken under the
minimum-image convention -- so the geometry and the local mean force are the *same* expressions
the alkane and deca studies already validated against autodiff:

    grad_{Q_2} xi = e,   grad_{Q_1} xi = -e,   |grad xi|^2 = 2,   div v = 2/r

    f_loc = (1/2) e.(grad_{Q_2} V - grad_{Q_1} V) - 2/(beta r)
          = (1/2) (F_1 - F_2).e                  - 2/(beta r)

The wrap is load-bearing, not decorative
----------------------------------------
An earlier version of this docstring claimed the wrap was a no-op because the soft walls hold
``r <= 0.90 nm`` against a half-box of 1.245 nm.  **That is wrong, and a guard caught it on the
first run:** configurations arriving from OpenMM are wrapped into the primary cell, so the two
methanes routinely sit on opposite faces of the box and the raw difference is ``~L`` rather than
``~r``.  Without the wrap the CV would read a nonsense separation for a large fraction of the
population.

The invariant that *is* worth asserting is the one that makes the coordinate unambiguous --
the minimum-image distance must stay below ``L/2`` -- and :meth:`separation_is_unambiguous`
asserts exactly that.

The chemistry PMF differs from ``F`` by the radial-entropy term (SPEC §2.3):

    W(r) = F(r) + 2 beta^-1 log r + C ,      W'(r) = F'(r) + 2/(beta r)

so ``W'`` is the bare average force along the line of centres, and the two are kept together
everywhere -- ``F`` is what ABF flattens, ``W`` is what the methane literature can be compared to.
"""
from __future__ import annotations

import torch

EPS = 1.0e-12


class PeriodicDistanceCV:
    """Minimum-image distance between two sites of a periodic box of side ``L``."""

    def __init__(self, i, j, box_nm):
        assert i != j
        self.i = int(i)
        self.j = int(j)
        self.L = float(box_nm)

    def _delta(self, q):
        d = q[..., self.j, :] - q[..., self.i, :]
        return d - self.L * torch.round(d / self.L)

    def value(self, q):
        """``xi(q)`` for ``q`` of shape ``(B, N, 3)`` -> ``(B,)``."""
        return torch.linalg.norm(self._delta(q), dim=-1)

    def separation_is_unambiguous(self, q, margin=0.98):
        """``True`` if every minimum-image separation is safely below ``L/2``.

        This is the condition that makes ``xi`` well defined: at exactly ``L/2`` the pair is
        equidistant from two images and the coordinate is degenerate.  ``margin`` keeps a little
        room below the boundary.
        """
        return bool((self.value(q) < margin * 0.5 * self.L).all())

    def geometry(self, q):
        """Analytic ``(r, grad_full (B, N, 3), div_v (B,))``; outputs detached."""
        B, N, _ = q.shape
        d = self._delta(q).detach()
        r = torch.linalg.norm(d, dim=-1).clamp_min(EPS)
        e = d / r[:, None]
        grad_full = q.new_zeros(B, N, 3)
        grad_full[:, self.j, :] = e
        grad_full[:, self.i, :] = -e
        return r.detach(), grad_full.detach(), (2.0 / r).detach()

    def local_mean_force(self, q, physical_forces, beta):
        """``f_loc = grad V . v - beta^-1 div v`` with ``v = grad xi / |grad xi|^2``.

        ``physical_forces = -grad V`` of shape ``(B, N, 3)``.  Returns
        ``(f_loc (B,), r (B,), grad_full (B, N, 3))``.
        """
        r, grad_full, div_v = self.geometry(q)
        gg = (grad_full * grad_full).sum(dim=(-2, -1)).clamp_min(EPS)          # = 2
        gradV_dot_v = -(physical_forces * grad_full).sum(dim=(-2, -1)) / gg
        return gradV_dot_v - div_v / beta, r, grad_full

    def bias_force(self, grad_full, mean_force_at_r):
        """Cartesian ABF/mFR bias force ``+A'(r) grad xi``.

        Equal and opposite on the two methanes, so the bias applies no net translational force to
        the pair.
        """
        return mean_force_at_r[:, None, None] * grad_full


def W_from_F(F, grid, beta):
    """Radial PMF from the reaction-coordinate free energy: ``W = F + 2 beta^-1 log r``."""
    return F + (2.0 / beta) * torch.log(grid.clamp_min(EPS))


def Wprime_from_Fprime(Fprime, grid, beta):
    """``W'(r) = F'(r) + 2/(beta r)`` -- the geometric term cancels, leaving the bare force."""
    return Fprime + 2.0 / (beta * grid.clamp_min(EPS))
