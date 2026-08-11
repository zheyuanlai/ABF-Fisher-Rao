"""Selection rules for deca-alanine: Fisher--Rao, and the prior art it must be measured against.

**One sign convention, stated once, because inverting it would silently reverse a method.**

Throughout this project a particle's ``score`` means

    score > 0  ->  OVER-represented  ->  dies      (``death_w = max(score, 0)``)
    score < 0  ->  UNDER-represented ->  gives birth (``birth_w = max(-score, 0)``)

which is what :func:`alanine.dynamics.birth_death_full_state` consumes.  The literature does
not share this convention -- Chapter 6 of Lelievre--Rousset--Stoltz writes its selection
function ``S`` so that **positive means multiply**.  The Laplacian rule below is therefore
negated on the way in.  That negation is the single most dangerous line in this module.

The three rules
---------------
**Fisher--Rao** (the proposed method).  ``score = log p(z) - log q(z) - KL(p||q)``, the
Wasserstein--Fisher--Rao birth--death direction toward the target ``q``.  Unbounded in both
directions until the clip.

**Book Laplacian** (Lelievre--Rousset--Stoltz Ch. 6, applied there to this very WCA dimer).
``S = c * d2p/dz2 / p``, which formally shifts the marginal diffusion coefficient from
``beta^-1`` to ``beta^-1 + c``.  Particles sitting where the density is convex -- the valleys
of ``p`` -- multiply.

**Count balancing** (Remark 6.10; also Comer et al.'s ``1/(bin count + 1)`` rule in NAMD).
``S = c * (1 - p/p_bar)``, i.e. push every bin toward the mean density.

Why the third one is not a footnote
-----------------------------------
A parallel v1 audit found plain count balancing *matching* the Fisher--Rao score on WCA
(``fr_uniform - count_bal = +1.53 %``, p = 0.17) and *beating* it on pentane ``R15``.  It also
identified the structural reason the two differ: count balancing saturates.  Its birth weight
``max(1 - p/p_bar, 0)`` is bounded by 1, so a 10x and a 10 000x deficit receive almost the same
push, whereas the Fisher--Rao log-ratio is unbounded until the clip and is systematically ~2x
more aggressive at the emptiest bins.  Whether that extra aggression helps or hurts is exactly
what v2 Q1 asks, so both rules are implemented faithfully rather than approximated into each
other.
"""
from __future__ import annotations

import torch

from alkanes import interval as iv

EPS = 1.0e-12


def _recentered_clipped(raw, clip):
    """Zero-mean the score across walkers, then clip.

    Recentering matters: birth--death conserves population only if the score has zero mean, and
    an uncentred score biases the whole ensemble toward death or birth.  Clipping is a guard on
    the tail of the log-ratio, not a regulariser.
    """
    s = raw - raw.mean(dim=-1, keepdim=True)
    return torch.clamp(s, -clip, clip)


def _density(z, grid, dz, R_lo, R_hi, K_kde, p_grid=None):
    """Reflected-KDE marginal of the walker positions, or a caller-supplied density.

    ``p_grid`` exists so the rules can be exercised against a *known* density.  Probing a
    density valley with walkers is self-defeating -- the probes create their own KDE peak at
    the very point being probed, so the point is concave, not convex, and a correct rule looks
    inverted.  That is not a hypothetical: it is how the first version of the Laplacian test
    failed.
    """
    if p_grid is not None:
        return p_grid
    counts = iv.bin_counts(z, grid.numel(), R_lo, R_hi)
    return iv.normalize_density(iv.smooth(counts, K_kde), dz)


def fisher_rao_score(z, grid, dz, R_lo, R_hi, K_kde, q_grid, clip, p_grid=None):
    """``log p(z) - log q(z) - KL(p||q)``.  Returns ``(score, p_grid, kl)``."""
    p_grid = _density(z, grid, dz, R_lo, R_hi, K_kde, p_grid)
    p_at = iv.interval_interp(p_grid, grid, z)
    q_at = iv.interval_interp(q_grid, grid, z)
    log_ratio = torch.log(p_grid.clamp_min(EPS)) - torch.log(q_grid.clamp_min(EPS))
    kl = (p_grid * log_ratio).sum(-1) * dz
    raw = torch.log(p_at.clamp_min(EPS)) - torch.log(q_at.clamp_min(EPS)) - kl[:, None]
    return _recentered_clipped(raw, clip), p_grid, kl


def count_balancing_score(z, grid, dz, R_lo, R_hi, K_kde, clip, c=1.0, support_mask=None,
                          p_grid=None):
    """``c * (p(z)/p_bar - 1)``  -- negated relative to the book's ``S`` per the sign rule.

    ``p_bar`` is the mean density over the *supported* part of the domain.  Using the whole
    domain would let a region the walkers have never reached drag ``p_bar`` down and make every
    occupied bin look over-represented.
    """
    p_grid = _density(z, grid, dz, R_lo, R_hi, K_kde, p_grid)
    if support_mask is None:
        p_bar = p_grid.mean(dim=-1, keepdim=True)
    else:
        m = support_mask.to(p_grid.dtype)
        p_bar = (p_grid * m).sum(-1, keepdim=True) / m.sum().clamp_min(1.0)
    p_at = iv.interval_interp(p_grid, grid, z)
    raw = c * (p_at / p_bar.clamp_min(EPS) - 1.0)
    return _recentered_clipped(raw, clip), p_grid


def book_laplacian_score(z, grid, dz, R_lo, R_hi, K_kde, clip, c=1.0, p_grid=None):
    """``-c * (d2p/dz2) / p``  -- the Chapter 6 rule, negated per the sign rule above.

    Where ``p`` is **convex** (``d2p/dz2 > 0``, the valleys of the density) the book's ``S`` is
    positive and the population multiplies; under this project's convention that is a negative
    score, hence the leading minus.  Where ``p`` is concave -- the top of a mode -- walkers die.
    The net effect is the extra ``c d2p/dz2`` term in the marginal evolution, i.e. the diffusion
    coefficient shifting from ``beta^-1`` to ``beta^-1 + c``.

    The second derivative is taken on the smoothed density by central differences, with
    one-sided differences at the two domain edges.  Smoothing first is not optional: a second
    derivative of a raw histogram is dominated by counting noise, which would make the rule a
    random-turnover generator and quietly turn this baseline into a second sham.
    """
    p_grid = _density(z, grid, dz, R_lo, R_hi, K_kde, p_grid)
    d2 = torch.zeros_like(p_grid)
    d2[:, 1:-1] = (p_grid[:, 2:] - 2.0 * p_grid[:, 1:-1] + p_grid[:, :-2]) / (dz * dz)
    d2[:, 0] = d2[:, 1]
    d2[:, -1] = d2[:, -2]
    ratio = d2 / p_grid.clamp_min(EPS)
    raw_at = iv.interval_interp(-c * ratio, grid, z)
    return _recentered_clipped(raw_at, clip), p_grid


def sham_score(score, generator):
    """Matched-turnover, random-direction control for a given FR-family score.

    Preserves the *magnitude* distribution of the partner arm's score exactly -- so the expected
    number of birth--death events matches -- while destroying the direction by randomly
    permuting the scores across walkers.  What survives is turnover; what is removed is the
    claim that the score points anywhere in particular.

    One sham per FR arm, per the v1 method rule: a single shared sham cannot control two arms
    whose turnover differs.
    """
    R, N = score.shape
    out = torch.empty_like(score)
    for r in range(R):
        perm = torch.randperm(N, generator=generator, device=score.device)
        out[r] = score[r].index_select(0, perm)
    return out


#: ``method -> (is_selection_arm, needs_target, is_sham)``
METHODS = {
    "abf":              (False, False, False),
    "mfr_practical":    (True,  True,  False),
    "mfr_oracle":       (True,  True,  False),
    "mfr_sham":         (True,  True,  True),
    "book_laplacian":   (True,  False, False),
    "count_balancing":  (True,  False, False),
}
SELECTION_METHODS = tuple(k for k, v in METHODS.items() if v[0])
ORACLE_METHODS = ("mfr_oracle",)


def assert_no_reference_leakage(method, reference_free_energy):
    """Structural gate: only an explicitly-named oracle arm may ever hold the reference."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}")
    if method in ORACLE_METHODS:
        if reference_free_energy is None:
            raise ValueError(f"{method!r} requires the reference free energy.")
        return
    if reference_free_energy is not None:
        raise AssertionError(
            f"NO-REFERENCE-LEAKAGE VIOLATION: method={method!r} received a reference free "
            f"energy; only {ORACLE_METHODS} may.")
