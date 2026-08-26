"""Replica-count allocation: the object clean-v2 conflated with probability mass.

Frozen protocol: ``docs/QR_DECOUPLING_PREREGISTRATION.md``.

Equal-weight birth--death represents a region's probability mass by its replica
count, which silently imposes ``r = q``.  This module computes ``r`` as its own
object.  The mass ``q`` lives in :mod:`abffr.cell_mass`; nothing here reads it
except through the optional ESS constraint below.

The risk model
--------------
The endpoint is the centred domain-RMS error of the *free energy*, and ``F`` is
built from ``F'`` by cumulative trapezoid.  An error in the mean force at ``z``
therefore reaches the endpoint through the integration and centring operators,
not directly, so the bins do not have equal influence.  Writing ``F = H f`` for
the cumulative-trapezoid matrix, ``C`` for centring on the evaluation mask and
``W`` for the mask quadrature weights, and assuming ``Cov(f_hat)`` diagonal::

    E[e_F^2] = sum_j a_j Var[f_hat_j],    a_j = (1/L) [H^T C^T W C H]_jj

:func:`leverage` returns ``a``.  It is pure grid geometry plus the mask: no free
energy, no landscape, no reference.  That is the whole reason the evaluation
mask must be fixed a priori by geometry -- under a thermal scope such as R12 the
mask depends on ``F_ref`` and ``a`` would be an oracle quantity.

With ``Var[f_hat_j] ~ Gamma_j / (K T r_j)`` the allocation problem is
``min_r sum_j g_j / r_j`` subject to ``sum_j r_j = 1``, where ``g_j = a_j
Gamma_j``, and Cauchy--Schwarz gives the Neyman solution ``r_j ∝ sqrt(g_j)``.

Two multiplicative factors, two arms
------------------------------------
``g = a * Gamma`` is a product of a *static geometric* factor and an *online
statistical* one, and they are separated on purpose::

    A4a   r ∝ sqrt(a)          leverage only -- static, needs no estimation
    A4b   r ∝ sqrt(a Gamma)    leverage x measured difficulty

because ``a`` alone is already strongly non-uniform (14x across cells on the
clean-v2 grid, and exactly zero outside the mask).  An arm that used the product
only could not say which factor earned a gain, and the campaign's tie prediction
lives in ``A4b == A4a`` where ``Gamma`` is flat -- not in ``A4 == uniform``,
which ``a`` alone already falsifies.

The floor is structural, not a knob
-----------------------------------
``a_j`` vanishes outside the evaluation mask, so an unfloored ``r`` asks for
regions the reflecting dynamics still visit to hold no replicas at all.  That is
not implementable and not desirable: the bias there would never be learned and
particles would enter a region whose forces are wrong.  Every allocation arm --
including uniform -- is therefore mixed with the same fixed floor, so that the
floor cannot become an arm-specific tuning degree of freedom.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

#: Smallest replica fraction any cell may be assigned, as a multiple of uniform.
#: Frozen for the campaign and shared by every arm.
FLOOR_FRACTION = 0.25


def cumulative_trapezoid_matrix(n: int, dx: float) -> np.ndarray:
    """``H`` with ``H @ f == cumulative_trapezoid(f, dx=dx, initial=0)``.

    Built as a matrix rather than applied as a routine because the leverage is a
    diagonal of ``H^T (...) H`` and writing the quadratic form out is what makes
    the assumption ``Cov(f_hat)`` diagonal visible instead of implicit.
    """
    H = np.zeros((n, n), dtype=float)
    for g in range(1, n):
        H[g, :g] += 0.5 * dx
        H[g, 1:g + 1] += 0.5 * dx
    return H


def leverage(x_grid: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """``a_j``: how much a unit of mean-force variance at ``j`` costs the endpoint.

    Zero outside ``mask`` on both sides, for two different reasons: bins past the
    mask never enter ``F`` there, and bins before it shift ``F`` on the whole mask
    by a constant, which the centring removes.
    """
    x = np.asarray(x_grid, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if x.ndim != 1 or mask.shape != x.shape:
        raise ValueError("x_grid and mask must be 1-D and the same length")
    idx = np.flatnonzero(mask)
    if idx.size < 3:
        raise ValueError("evaluation mask needs at least 3 grid points")
    dx = float(x[1] - x[0])
    n = x.size

    H = cumulative_trapezoid_matrix(n, dx)
    C = np.eye(n)
    C[np.ix_(idx, idx)] -= 1.0 / idx.size          # subtract the mask mean

    w = np.zeros(n)                                 # trapezoid weights on mask
    w[idx[1:-1]] = dx
    w[idx[0]] = w[idx[-1]] = 0.5 * dx
    L = float(x[idx[-1]] - x[idx[0]])

    M = C @ H
    a = np.einsum("gj,g,gj->j", M, w / L, M)
    return np.maximum(a, 0.0)                       # kill round-off negatives


def cell_reduce(values: np.ndarray, cell_of_grid: np.ndarray,
                n_cells: int) -> np.ndarray:
    """Sum a per-grid-point quantity onto allocation cells."""
    out = np.zeros(int(n_cells), dtype=float)
    np.add.at(out, np.asarray(cell_of_grid, dtype=int), np.asarray(values, float))
    return out


def apply_floor(r: np.ndarray, floor_fraction: float = FLOOR_FRACTION) -> np.ndarray:
    """Mix ``r`` with uniform so no cell falls below ``floor_fraction / J``.

    A convex mixture rather than a clamp: clamping renormalises the untouched
    cells by a factor that depends on how many cells were clamped, so the same
    arm would get a different profile as the floor bound in more places.
    """
    r = np.asarray(r, dtype=float)
    if np.any(r < 0):
        raise ValueError("replica density must be non-negative")
    total = r.sum()
    if not np.isfinite(total) or total <= 0:
        return np.full(r.size, 1.0 / r.size)
    r = r / total
    eps = float(floor_fraction)
    if not 0.0 <= eps <= 1.0:
        raise ValueError("floor_fraction must lie in [0, 1]")
    return (1.0 - eps) * r + eps / r.size


# --------------------------------------------------------------------------- #
# The allocation rules
# --------------------------------------------------------------------------- #
def r_uniform(n_cells: int) -> np.ndarray:
    """A3 -- count balancing.  The incumbent, not a straw man."""
    return np.full(int(n_cells), 1.0 / int(n_cells))


def r_neyman(g: np.ndarray) -> np.ndarray:
    """``r ∝ sqrt(g)``, the unconstrained optimum of ``sum_j g_j / r_j``."""
    g = np.maximum(np.asarray(g, dtype=float), 0.0)
    s = np.sqrt(g)
    total = s.sum()
    if not np.isfinite(total) or total <= 0:
        return r_uniform(g.size)
    return s / total


def mass_ess_fraction(q: np.ndarray, r: np.ndarray) -> float:
    """``ESS_w / K`` for cell masses ``q`` carried by replica fractions ``r``.

    Every replica in cell ``j`` carries ``w = q_j / (K r_j)``, so
    ``sum_i w_i^2 = (1/K) sum_j q_j^2 / r_j`` and the fraction is its reciprocal.
    Cells with no target mass contribute nothing; cells with mass but no replica
    make the representation degenerate, which is reported as ESS 0 rather than
    as an exception.
    """
    q = np.asarray(q, dtype=float)
    r = np.asarray(r, dtype=float)
    live = q > 0
    if not live.any():
        return 1.0
    if np.any(r[live] <= 0):
        return 0.0
    return float(1.0 / np.sum(q[live] ** 2 / r[live]))


@dataclass(frozen=True)
class ConstrainedAllocation:
    r: np.ndarray
    lam: float                 # the multiplier the bisection found
    ess_fraction: float        # achieved ESS_w / K
    constraint_active: bool    # False when the Neyman optimum already complied


def r_ess_constrained(g: np.ndarray, q: np.ndarray, rho: float,
                      tol: float = 1e-10,
                      max_iter: int = 200) -> ConstrainedAllocation:
    """A5 -- ``r ∝ sqrt(g + lam q^2)`` with ``lam`` set so ``ESS_w / K >= rho``.

    ``lam = 0`` recovers A4b and ``lam -> infinity`` recovers ``r = q``, which is
    what equal-weight birth--death imposed all along: clean-v2 was this family at
    ``rho = 1``.  ``lam`` is found by bisection, not tuned.

    Because A5 minimises the same risk as A4b under an extra constraint, its
    predicted risk is *weakly worse* -- A5 cannot beat A4b on the free-energy
    endpoint except through model error or noise.  The constraint buys fidelity
    of the physical-mass representation, which is a separate reported endpoint.
    """
    g = np.maximum(np.asarray(g, dtype=float), 0.0)
    q = np.asarray(q, dtype=float)
    if q.shape != g.shape:
        raise ValueError("g and q must have the same number of cells")
    rho = float(rho)
    if not 0.0 < rho <= 1.0:
        raise ValueError("rho must lie in (0, 1]")
    qn = q / q.sum() if q.sum() > 0 else np.full(q.size, 1.0 / q.size)

    def alloc(lam: float) -> np.ndarray:
        return r_neyman(g + lam * qn ** 2)

    r0 = alloc(0.0)
    ess0 = mass_ess_fraction(qn, r0)
    if ess0 >= rho:
        return ConstrainedAllocation(r0, 0.0, ess0, False)

    # ESS_w(lam) increases monotonically to 1 as r -> q, so bracket then bisect.
    hi = 1.0
    scale = max(float(g.max()), 1e-300)
    for _ in range(max_iter):
        if mass_ess_fraction(qn, alloc(hi * scale)) >= rho:
            break
        hi *= 4.0
    else:                                    # pragma: no cover - rho <= 1 always reachable
        r = qn.copy()
        return ConstrainedAllocation(r, float("inf"), mass_ess_fraction(qn, r), True)

    lo, hi = 0.0, hi * scale
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if mass_ess_fraction(qn, alloc(mid)) >= rho:
            hi = mid
        else:
            lo = mid
        if hi - lo <= tol * max(hi, 1.0):
            break
    r = alloc(hi)
    return ConstrainedAllocation(r, float(hi), mass_ess_fraction(qn, r), True)


def predicted_risk(g: np.ndarray, r: np.ndarray) -> float:
    """``sum_j g_j / r_j`` -- the quantity the resampling gate compares."""
    g = np.asarray(g, dtype=float)
    r = np.asarray(r, dtype=float)
    live = g > 0
    if not live.any():
        return 0.0
    if np.any(r[live] <= 0):
        return float("inf")
    return float(np.sum(g[live] / r[live]))


def desired_counts(r: np.ndarray, n_particles: int,
                   occupied: Optional[np.ndarray] = None) -> np.ndarray:
    """Integer target counts summing to ``K``, by largest-remainder.

    ``occupied`` marks cells that currently hold at least one replica.  A cell
    with no replicas gets none: resampling may not create support the dynamics
    have not discovered, which is the line between ABF (discovery) and
    reallocation (establishment).  The mass that rule strands is redistributed
    over the occupied cells rather than dropped, so the counts still sum to
    ``K``.
    """
    r = np.asarray(r, dtype=float).copy()
    K = int(n_particles)
    if occupied is not None:
        occupied = np.asarray(occupied, dtype=bool)
        if not occupied.any():
            raise ValueError("no occupied cell: the population is empty")
        r[~occupied] = 0.0
    total = r.sum()
    if total <= 0:
        r = np.ones_like(r) if occupied is None else occupied.astype(float)
        total = r.sum()
    r = r / total

    exact = r * K
    base = np.floor(exact).astype(int)
    short = K - int(base.sum())
    if short > 0:
        order = np.argsort(-(exact - base), kind="stable")
        base[order[:short]] += 1
    return base
