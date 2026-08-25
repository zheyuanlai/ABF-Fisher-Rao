"""v3 Fisher--Rao operators: one score, two discretizations, frozen semantics.

Frozen protocol: ``docs/V3_PREREGISTRATION.md`` (v3.1) with Amendments 1-3.

The continuum flow, for a target ``q`` held fixed across one FR opportunity::

    d p_tau / d tau = -p_tau [ log(p_tau/q) - E_{p_tau} log(p_tau/q) ]

Two particle realizations are provided.  **Both consume the same**
:class:`FRScore` **object**; neither recomputes the score, because two
independently-written score expressions are how a discretization comparison
turns into a comparison of two different flows.

``bd_standard``
    The reference birth--death scheme.  Particle ``i`` carries event rate
    ``|S_i|``; on an event, ``S_i > 0`` means over-represented (``i`` dies, a
    uniformly chosen other replica duplicates) and ``S_i < 0`` means
    under-represented (``i`` duplicates, a uniformly chosen other replica dies).
    Event probability ``1 - exp(-|S_i| dtau)``.  Particles above the 90th
    percentile of ``|S|`` are **not** clipped -- they simply get probability
    above ``p_max``.  That is the whole difference from v2.

``ft_step``
    The exact finite-time map ``p+ propto p^(1-theta) q^theta``, realized by
    weights ``(q/p)^theta`` and one systematic resampling.  Step size from the
    ESS governor.  Amendment 3: ``theta = 1 - exp(-tau)``, so ``theta`` is not
    FR time and a dose-matched comparison must convert.

Every operator returns a ``src`` index vector with ``new[i] = old[src[i]]``.
The caller applies it to positions, ancestry and hold-out counters alike, which
is what keeps genealogy bookkeeping identical across the two schemes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

EPS_LOG = 1e-300


@dataclass(frozen=True)
class FRScore:
    """The centered Fisher--Rao score for one cloud of ``K`` particles.

    ``S_i = r_i - mean(r)`` with ``r_i = log p_hat(z_i) - log q(z_i)``.  The mean
    is the empirical form of ``E_p[log(p/q)]``; subtracting it is what makes the
    birth--death rates conserve population in expectation.  No clipping is
    applied here or anywhere downstream.
    """

    log_p: torch.Tensor
    log_q: torch.Tensor

    @property
    def r(self) -> torch.Tensor:
        return self.log_p - self.log_q

    @property
    def S(self) -> torch.Tensor:
        r = self.r
        return r - r.mean()

    @property
    def a(self) -> torch.Tensor:
        """``log q - log p``: the FT weight exponent, i.e. ``-r``."""
        return -self.r

    def quantiles(self, levels: torch.Tensor) -> torch.Tensor:
        return torch.quantile(self.S, levels)


def bd_timestep(score: FRScore, p_max: float, quantile: float = 0.90) -> float:
    """FR time for one BD opportunity from a bulk event-probability cap.

    ``p_max`` is the event probability for a particle *at* the given quantile of
    ``|S|``; the upper tail exceeds it, faithfully rather than by truncation.
    """
    if not 0.0 < p_max < 1.0:
        raise ValueError("p_max must lie in (0, 1)")
    scale = torch.quantile(score.S.abs(), quantile)
    if float(scale) <= 0.0:
        return 0.0
    return float(-torch.log(torch.tensor(1.0 - p_max, dtype=score.S.dtype)) / scale)


def bd_standard(score: FRScore, dtau: float,
                generator: torch.Generator) -> Tuple[torch.Tensor, int]:
    """One standard birth--death opportunity.  Returns ``(src, n_events)``.

    Frozen conventions (v3.1 Phase II): all probabilities come from the pre-FR
    ensemble; accepted events are executed in a randomized order drawn from the
    FR stream; the uniform partner is chosen at execution time; ``S`` is never
    recomputed inside the opportunity.  ``src`` always points at an *original*
    index, so an event never produces a copy-of-a-copy.
    """
    K = score.S.numel()
    src = torch.arange(K, device=score.S.device)
    if dtau <= 0.0:
        return src, 0

    S = score.S
    prob = 1.0 - torch.exp(-S.abs() * dtau)
    draw = torch.rand(K, generator=generator, device=S.device, dtype=S.dtype)
    fired = torch.nonzero(draw < prob, as_tuple=False).flatten()
    if fired.numel() == 0:
        return src, 0

    order = torch.randperm(fired.numel(), generator=generator, device=S.device)
    fired = fired[order]
    # One uniform partner draw per event, consumed in the same randomized order.
    partner_u = torch.rand(fired.numel(), generator=generator,
                           device=S.device, dtype=S.dtype)

    for k in range(fired.numel()):
        i = int(fired[k])
        # Uniform over the K-1 slots other than i.
        j = int(partner_u[k] * (K - 1))
        if j >= i:
            j += 1
        if float(S[i]) > 0.0:
            src[i] = src[j]        # i is over-represented: it dies, j duplicates
        else:
            src[j] = src[i]        # i is under-represented: it duplicates, j dies
    return src, int(fired.numel())


def ess_of_theta(a: torch.Tensor, theta: float) -> float:
    """``ESS(theta) = 1 / sum_i w_i^2`` for ``w propto exp(theta * a)``.

    Non-increasing in ``theta`` by Amendment 3's theorem
    (``d/dtheta log ESS = 2[m(theta) - m(2theta)] <= 0``); a violation is an
    engineering anomaly, not a cloud property.
    """
    logw = theta * a
    logw = logw - logw.max()
    w = torch.exp(logw)
    w = w / w.sum()
    return float(1.0 / (w * w).sum())


def ess_governor(a: torch.Tensor, rho: float, K: int,
                 tol: float = 1e-6, max_iter: int = 60,
                 logger=None) -> float:
    """Largest ``theta`` in [0, 1] with ``ESS(theta) >= rho * K``.

    Bisection, valid because ESS is non-increasing.  Monotonicity is spot-checked
    and any violation is reported loudly rather than absorbed by a silent scan.
    """
    if not 0.0 < rho <= 1.0:
        raise ValueError("rho must lie in (0, 1]")
    target = rho * K
    if ess_of_theta(a, 1.0) >= target:
        return 1.0
    if ess_of_theta(a, 0.0) < target:
        return 0.0                      # cannot move at all (rho too strict)

    probe = [ess_of_theta(a, t) for t in (0.0, 0.25, 0.5, 0.75, 1.0)]
    if any(probe[i + 1] > probe[i] + 1e-9 for i in range(len(probe) - 1)):
        msg = ("ESS(theta) is not monotone on a fixed cloud -- this contradicts "
               f"Amendment 3 and indicates numerical pathology: {probe}")
        if logger is not None:
            logger.error(msg)
        else:                                     # never swallow this silently
            raise RuntimeError(msg)

    lo, hi = 0.0, 1.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if ess_of_theta(a, mid) >= target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return lo


def systematic_resample(weights: torch.Tensor,
                        generator: torch.Generator) -> torch.Tensor:
    """Systematic resampling to a fixed population.  Returns parent indices.

    With exactly uniform weights this is the identity map (one child per parent),
    which is what makes the ``p_hat == q`` inertness gate exact rather than
    statistical.
    """
    K = weights.numel()
    u = torch.rand(1, generator=generator, device=weights.device,
                   dtype=weights.dtype)
    positions = (u + torch.arange(K, device=weights.device,
                                  dtype=weights.dtype)) / K
    cumsum = torch.cumsum(weights, dim=0)
    cumsum[-1] = 1.0                      # guard against float drift at the end
    return torch.searchsorted(cumsum, positions.contiguous()).clamp_max(K - 1)


def ft_step(score: FRScore, rho: float, generator: torch.Generator,
            logger=None) -> Tuple[torch.Tensor, float, float]:
    """One exact finite-time FR step.  Returns ``(src, theta, ess)``.

    ``theta = 0`` bypasses the resampler entirely, so the zero-strength identity
    gate holds by construction rather than by tolerance.
    """
    K = score.log_p.numel()
    a = score.a
    theta = ess_governor(a, rho, K, logger=logger)
    ess = ess_of_theta(a, theta)
    if theta <= 0.0:
        return torch.arange(K, device=a.device), 0.0, ess
    logw = theta * a
    logw = logw - logw.max()
    w = torch.exp(logw)
    w = w / w.sum()
    return systematic_resample(w, generator), theta, ess


def theta_from_dtau(dtau: float) -> float:
    """Amendment 3: FR time to FT interpolation parameter."""
    import math
    return 1.0 - math.exp(-dtau)


def dtau_from_theta(theta: float) -> float:
    """Amendment 3, inverse.  ``theta -> 1`` is infinite FR time."""
    import math
    if theta >= 1.0:
        return float("inf")
    return -math.log(1.0 - theta)


def offspring_counts(src: torch.Tensor, K: int) -> torch.Tensor:
    return torch.bincount(src, minlength=K)


def replacement_count(src: torch.Tensor, K: int) -> int:
    """``R = K - #{i : N_i > 0} = sum_i (N_i - 1)_+`` -- excess children.

    Simultaneously the number of parents eliminated, which is what makes it
    comparable to BD's death count.
    """
    counts = offspring_counts(src, K)
    return int(K - int((counts > 0).sum()))


def clone_mask(src: torch.Tensor) -> torch.Tensor:
    """Which output slots are *new clones* rather than continuations.

    For each parent with ``N_i >= 1`` the first output slot holding it (in slot
    order) is the continuation; any further slot is a new clone.  Without this
    convention a global FT resampling would place nearly the whole ensemble in
    hold-out, which is not what the policy means.
    """
    K = src.numel()
    seen = torch.zeros(K, dtype=torch.bool, device=src.device)
    is_clone = torch.zeros(K, dtype=torch.bool, device=src.device)
    for i in range(K):
        parent = int(src[i])
        if seen[parent]:
            is_clone[i] = True
        else:
            seen[parent] = True
    return is_clone


def apply_holdout(hold: torch.Tensor, src: torch.Tensor,
                  is_clone: torch.Tensor, hold_steps: int) -> torch.Tensor:
    """Propagate hold-out counters through a reallocation.

    Continuations inherit whatever the parent had left; new clones get a fresh
    ``hold_steps``, including a clone of an already-held-out replica.
    """
    new_hold = hold[src].clone()
    new_hold[is_clone] = int(hold_steps)
    return new_hold
