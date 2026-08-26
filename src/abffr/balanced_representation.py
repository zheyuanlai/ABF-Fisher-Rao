"""Cell-wise resampling that spends as little genealogy as the counts allow.

Frozen protocol: ``docs/QR_DECOUPLING_PREREGISTRATION.md``.

:mod:`abffr.representation` resamples the whole population against a weight
vector.  That is the right operator when mass and count are the same object.
Once they are separated, resampling has a different job: move the *counts* from
``n_j`` to ``n_j*`` while destroying as little independent information as
possible, and never move a configuration across a cell.

The genealogy penalty
---------------------
Under a conditional spectral gap ``lambda_j``, two descendants of one parent that
have propagated independently for ``Delta`` carry covariance bounded by
``exp(-2 lambda_j Delta) sigma_j^2``, so a cell average over ``n`` descendants
drawn from parents with multiplicities ``m_a`` obeys::

    Var <= (sigma^2 / n) [ 1 + exp(-2 lambda Delta) D / n ],   D = sum_a m_a(m_a-1)

Two consequences are implemented here.  :func:`balanced_offspring` minimises
``D`` exactly over integer multiplicities, and :func:`rejuvenation_steps` turns
the bound into the hold time that keeps the penalty under ``eps_gene`` -- which
is what replaces v4-A's fixed ``L_hold = 500``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class CellResample:
    src: np.ndarray            # new[i] = old[src[i]], global particle indices
    is_clone: np.ndarray       # output slots that are NEW clones
    duplicate_pairs: float     # D = sum_a m_a (m_a - 1), summed over cells
    n_replacements: int        # parents that lost their last slot


def balanced_offspring(n_parents: int, n_children: int) -> np.ndarray:
    """Multiplicities minimising ``sum_a m_a (m_a - 1)`` for a fixed total.

    ``sum m_a`` is fixed, so minimising ``sum m_a(m_a-1)`` is minimising
    ``sum m_a^2``, which for integers is achieved exactly by making the
    multiplicities as equal as possible: with ``n_children = a n_parents + b``,
    ``b`` parents get ``a + 1`` and the rest get ``a``.  Any other integer vector
    can be improved by moving one child from a larger to a smaller entry, which
    strictly decreases the sum.
    """
    n_parents, n_children = int(n_parents), int(n_children)
    if n_parents < 0 or n_children < 0:
        raise ValueError("counts must be non-negative")
    if n_parents == 0:
        if n_children:
            raise ValueError("cannot create children with no parents in the cell")
        return np.zeros(0, dtype=int)
    base, extra = divmod(n_children, n_parents)
    m = np.full(n_parents, base, dtype=int)
    m[:extra] += 1
    return m


def duplicate_pairs(m: np.ndarray) -> int:
    """``D = sum_a m_a (m_a - 1)`` -- the genealogy cost of a multiplicity vector."""
    m = np.asarray(m, dtype=int)
    return int((m * (m - 1)).sum())


def rejuvenation_steps(D: float, n_children: int, tau: float, dt: float,
                       eps_gene: float = 0.1) -> int:
    """Steps a clone must propagate before its observations may be used.

    Inverting ``exp(-2 lambda Delta) D / n <= eps`` with ``lambda ~ 1 / tau``::

        Delta = (tau / 2) log( D / (eps n) )_+

    ``tau`` is the online integrated autocorrelation time for the cell, so a cell
    that mixes fast releases its clones quickly and a slow one holds them.  A
    fixed hold cannot do that, and this campaign varies conditional mixing by
    16x on purpose.
    """
    n_children = int(n_children)
    if n_children <= 0 or D <= 0:
        return 0
    if not (tau > 0 and dt > 0):
        raise ValueError("tau and dt must be positive")
    ratio = float(D) / (float(eps_gene) * n_children)
    if ratio <= 1.0:
        return 0
    return int(math.ceil(0.5 * float(tau) * math.log(ratio) / float(dt)))


def resample_cells(cell_of_particle: np.ndarray, target_counts: np.ndarray,
                   rng: np.random.Generator) -> CellResample:
    """Move counts to ``target_counts`` cell by cell, minimising genealogy.

    Configurations never move between cells: a cell short of replicas duplicates
    its own, a cell over its target drops some of its own.  Which parents are
    duplicated or dropped is chosen uniformly at random within the cell -- never
    by force, hidden coordinate or weight, since selecting on anything that
    correlates with the fibre would bias the conditional law the mean force is an
    expectation over.
    """
    cell_of_particle = np.asarray(cell_of_particle, dtype=int)
    target_counts = np.asarray(target_counts, dtype=int)
    K = cell_of_particle.size
    if target_counts.sum() != K:
        raise ValueError(
            f"target counts sum to {target_counts.sum()}, not the population {K}")

    src = np.empty(K, dtype=int)
    is_clone = np.zeros(K, dtype=bool)
    D_total = 0
    n_repl = 0
    out = 0
    for j in range(target_counts.size):
        members = np.flatnonzero(cell_of_particle == j)
        want = int(target_counts[j])
        if members.size == 0:
            if want:
                raise ValueError(
                    f"cell {j} is empty but was assigned {want} replicas; "
                    f"resampling may not create undiscovered support")
            continue
        if want == 0:
            n_repl += members.size
            continue
        if want <= members.size:
            keep = rng.choice(members, size=want, replace=False)
            src[out:out + want] = keep
            n_repl += members.size - want
            out += want
            continue

        m = balanced_offspring(members.size, want)
        order = rng.permutation(members.size)      # who gets the extra child
        m = m[np.argsort(order, kind="stable")]
        D_total += duplicate_pairs(m)
        for parent, mult in zip(members, m):
            src[out] = parent                      # the continuation
            out += 1
            for _ in range(int(mult) - 1):
                src[out] = parent
                is_clone[out] = True
                out += 1
    if out != K:                                   # pragma: no cover - guarded above
        raise AssertionError(f"filled {out} of {K} slots")
    return CellResample(src=src, is_clone=is_clone,
                        duplicate_pairs=float(D_total), n_replacements=int(n_repl))


def resample_benefit(g: np.ndarray, r_now: np.ndarray,
                     r_star: np.ndarray) -> float:
    """Relative predicted risk reduction, ``(R_now - R_star) / R_now``.

    The gate that decides whether a count change is worth its genealogy: an
    opportunity is not an obligation, and small fluctuations in ``Gamma_hat``
    must not be able to churn the population.
    """
    from .allocation import predicted_risk
    R_now = predicted_risk(g, r_now)
    R_star = predicted_risk(g, r_star)
    if not np.isfinite(R_now) or R_now <= 0:
        return 0.0 if not np.isfinite(R_star) else 1.0
    return float((R_now - R_star) / R_now)
