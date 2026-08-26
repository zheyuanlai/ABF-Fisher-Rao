"""Probability mass as a cell-level state, projected onto replicas on demand.

Frozen protocol: ``docs/QR_DECOUPLING_PREREGISTRATION.md``.

v4-A's :class:`abffr.persistent_mass.PersistentMass` carried a multiplicative
weight per *particle*, which made ``w_i`` a path functional: two replicas at the
same current ``xi`` could differ by 8e6 in one narrow bin, and ABF's derivation
needs the biasing factor to be constant on the current fibre.  That is why those
weights were barred from the accumulator.

Here the Fisher--Rao state lives on cells instead::

    M_j  <- M_j^{1-theta} q_j^theta          (exact finite-time FR step)
    w_i  =  M_j / n_j        for every replica in cell j

``M_j`` may carry as much FR history as it likes; the projection onto particles
is by current cell only, so ``w_i == w_k`` whenever ``xi_i`` and ``xi_k`` share a
cell.  Fibre constancy is a property of the representation, not a promise.

The campaign runs ``theta = 1`` -- the mass is projected exactly onto the target
at each opportunity.  ``theta < 1`` is the genuine Fisher--Rao flow and is kept
here because the reduction is the point: the geometry is what licenses acting on
the marginal alone, and ``theta`` is a rate on that flow.  It is not swept,
because the mass does not enter the free-energy estimator and so cannot change
the primary endpoint -- a knob that cannot move the headline is a knob that
should not be tuned.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

EPS = 1e-300


@dataclass
class CellMass:
    """Fisher--Rao probability mass over allocation cells."""

    n_cells: int
    theta: float = 1.0

    def __post_init__(self):
        if not 0.0 < float(self.theta) <= 1.0:
            raise ValueError("theta must lie in (0, 1]")
        self.log_M = np.full(int(self.n_cells), -np.log(float(self.n_cells)))

    @property
    def mass(self) -> np.ndarray:
        m = np.exp(self.log_M - self.log_M.max())
        return m / m.sum()

    def fr_step(self, log_q: np.ndarray) -> None:
        """``M <- M^(1-theta) q^theta``, normalised.  Kept in log space throughout.

        With ``theta = 1`` this is an exact projection onto the target; with
        ``theta < 1`` it is the exact solution of the Fisher--Rao flow after
        reaction time ``-log(1-theta) / gamma``.
        """
        log_q = np.asarray(log_q, dtype=float)
        if log_q.shape != self.log_M.shape:
            raise ValueError("target has the wrong number of cells")
        if not np.isfinite(log_q).all():
            raise FloatingPointError(
                "non-finite log target; failing closed rather than clipping it")
        th = float(self.theta)
        new = (1.0 - th) * self.log_M + th * log_q
        self.log_M = new - _logsumexp(new)

    def project(self, cell_of_particle: np.ndarray) -> np.ndarray:
        """Per-replica weights ``w_i = M_j / n_j``, normalised over the population.

        Occupied cells alone carry the mass: a cell the population has left
        cannot have its mass represented, and silently leaving that mass in the
        normalisation would understate every live replica's weight.  The
        renormalisation makes the represented measure conditional on the
        discovered support, which is what a particle method can actually claim.
        """
        cell = np.asarray(cell_of_particle, dtype=int)
        counts = np.bincount(cell, minlength=self.n_cells).astype(float)
        M = self.mass
        live = counts > 0
        if not live.any():
            raise ValueError("empty population")
        share = np.zeros(self.n_cells)
        total = M[live].sum()
        share[live] = M[live] / (total if total > EPS else live.sum())
        w = share[cell] / counts[cell]
        return w / w.sum()


def log_target_from_free_energy(A_cell: np.ndarray, beta: float) -> np.ndarray:
    """``log q_j = -beta A_j`` up to the constant that normalisation removes.

    ``A`` is the running ABF estimate on cells, never a reference free energy.
    """
    A = np.asarray(A_cell, dtype=float)
    if not np.isfinite(A).all():
        raise FloatingPointError("non-finite free energy in the FR target")
    return -float(beta) * A


def _logsumexp(x: np.ndarray) -> float:
    m = float(np.max(x))
    return m + float(np.log(np.sum(np.exp(x - m))))
