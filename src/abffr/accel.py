"""Time-to-accuracy metrics: the clean-v2 primary endpoint.

Frozen protocol: ``docs/CLEAN_V2_PREREGISTRATION.md``.

The v2/v3 campaigns selected on *final* error.  That was the wrong endpoint for
the question actually being asked: plain ABF is a convergent estimator, so given
enough time it is supposed to be excellent, and an accelerator's advantage is
supposed to shrink.  The clean-v2 endpoint is instead

    tau_eps = inf { t : e(t) <= eps },      S_eps = E[tau_eps^ABF] / E[tau_eps^FR]

with ``S_eps > 1`` meaning ABF+FR reached the same accuracy in less physical
simulation time.  ``S_eps = 1.5`` reads as "the same free-energy accuracy for
two thirds of the simulation time".

Three details carry most of the statistical weight, and all three are here
rather than in a script so a gate can pin them:

*Persistence.*  A single downward fluctuation of a noisy error curve is not
convergence.  ``tau`` requires the error to sit at or below the threshold for
``consecutive`` saved frames (default 3), and is reported at the *first* frame of
that run.  A consequence worth stating out loud: a threshold first reached in
the last ``consecutive - 1`` frames cannot start a qualifying run, and is
censored.

*Censoring, not deletion, and the bias has a direction.*  A seed that never
reaches the threshold is right-censored at ``T`` rather than dropped.  Dropping
it would compare the subset of ABF seeds that converged against the subset of FR
seeds that converged -- the "no-data-reads-as-PASS" defect with extra steps.  The
statistic is therefore the *restricted* speedup at horizon ``T``

    S^(T) = E[min(tau_base, T)] / E[min(tau_arm, T)],

which answers "within the budget T, who got there first?".  It is **not** an
estimate of the unrestricted ``E[tau_base]/E[tau_arm]``, and it is **not**
unconditionally conservative.  Restriction replaces a censored ``tau`` by ``T``,
the smallest value it could have had, so:

* censoring only in the **arm** shrinks the denominator and **inflates** S^(T);
* censoring only in the **baseline** shrinks the numerator and **deflates** it;
* censoring on both sides leaves the direction indeterminate.

The one safe case is the useful one: with **no arm censoring**, S^(T) is a lower
bound on the unrestricted speedup, so a positive result there is conservative.
Every summary carries the hit fraction ``P(tau <= T)`` and the censored count on
each side precisely so a reader can tell which case they are in; a headline
speedup computed with more arm censoring than baseline censoring is inflated by
construction and :func:`confirms` refuses it.

*The count travels with the ratio.*  Every summary returned here carries the
number of seeds and the number censored on each side.  A speedup computed from
three surviving seeds and one from thirty-two are different claims.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

INF = float("inf")


# --------------------------------------------------------------------------- #
# Hitting times
# --------------------------------------------------------------------------- #
def hitting_time(times: Sequence[float], errors: Sequence[float], eps: float,
                 consecutive: int = 3) -> float:
    """First time the error is at or below ``eps`` for ``consecutive`` frames.

    Returns ``inf`` when no such run exists (right-censored).  Non-finite errors
    break a run: a NaN frame is not evidence that the threshold was met.
    """
    t = np.asarray(times, dtype=float)
    e = np.asarray(errors, dtype=float)
    if t.shape != e.shape:
        raise ValueError("times and errors must have the same length")
    k = int(consecutive)
    if k < 1:
        raise ValueError("consecutive must be >= 1")
    ok = np.isfinite(e) & (e <= float(eps))
    run = 0
    for j in range(ok.size):
        run = run + 1 if ok[j] else 0
        if run >= k:
            return float(t[j - k + 1])
    return INF


def restricted_hitting_time(times: Sequence[float], errors: Sequence[float],
                            eps: float, horizon: float,
                            consecutive: int = 3) -> "Hit":
    """``min(tau, T)`` plus the censoring flag, as one object."""
    tau = hitting_time(times, errors, eps, consecutive=consecutive)
    censored = not np.isfinite(tau)
    return Hit(tau=tau, restricted=float(min(tau, float(horizon))),
               censored=censored)


@dataclass(frozen=True)
class Hit:
    tau: float           # inf when the threshold was never held
    restricted: float    # min(tau, T)
    censored: bool


# --------------------------------------------------------------------------- #
# Speedup
# --------------------------------------------------------------------------- #
@dataclass
class Speedup:
    """``S_eps`` with everything needed to read it honestly."""

    s: float
    mean_base: float
    mean_arm: float
    n_base: int
    n_arm: int
    n_censored_base: int
    n_censored_arm: int
    ci_lo: float = float("nan")
    ci_hi: float = float("nan")
    n_boot: int = 0

    @property
    def excludes_one(self) -> bool:
        """Whether the bootstrap CI lies strictly above 1."""
        return bool(np.isfinite(self.ci_lo) and self.ci_lo > 1.0)

    @property
    def hit_fraction_base(self) -> float:
        """``P(tau <= T)`` for the baseline: the share that actually converged."""
        return 1.0 - self.n_censored_base / max(self.n_base, 1)

    @property
    def hit_fraction_arm(self) -> float:
        return 1.0 - self.n_censored_arm / max(self.n_arm, 1)

    @property
    def censoring_inflates(self) -> bool:
        """Whether restriction biases this ratio in the arm's favour.

        True when the arm is censored more often than the baseline, in which
        case S^(T) overstates the unrestricted speedup and may not be reported
        as a headline number.
        """
        return self.n_censored_arm > self.n_censored_base

    def to_row(self) -> Dict:
        row = asdict(self)
        row["excludes_one"] = self.excludes_one
        row["hit_fraction_base"] = self.hit_fraction_base
        row["hit_fraction_arm"] = self.hit_fraction_arm
        row["censoring_inflates"] = self.censoring_inflates
        return row


def speedup(base: Sequence[Hit], arm: Sequence[Hit]) -> Speedup:
    """``S^(T) = E[min(tau_base, T)] / E[min(tau_arm, T)]``.

    The restricted speedup at horizon ``T``, not an estimate of the unrestricted
    ratio.  See the module docstring for which way censoring biases it.
    """
    b = np.asarray([h.restricted for h in base], dtype=float)
    a = np.asarray([h.restricted for h in arm], dtype=float)
    if b.size == 0 or a.size == 0:
        raise ValueError("speedup needs at least one run on each side")
    mb, ma = float(b.mean()), float(a.mean())
    return Speedup(
        s=(mb / ma) if ma > 0 else INF,
        mean_base=mb, mean_arm=ma, n_base=b.size, n_arm=a.size,
        n_censored_base=int(sum(h.censored for h in base)),
        n_censored_arm=int(sum(h.censored for h in arm)))


def paired_bootstrap_speedup(base: Sequence[Hit], arm: Sequence[Hit],
                             n_boot: int = 10000, seed: int = 20260826,
                             alpha: float = 0.05) -> Speedup:
    """Paired bootstrap over matched seeds.

    ``base[i]`` and ``arm[i]`` must be the *same* seed: the arms share initial
    conditions and Langevin noise, so resampling seeds jointly is what removes
    the shared physical variability the pairing bought.  Resampling them
    independently would throw that away and widen the interval for nothing.
    """
    if len(base) != len(arm):
        raise ValueError(
            f"paired bootstrap needs matched seeds; got {len(base)} baseline "
            f"and {len(arm)} arm runs")
    point = speedup(base, arm)
    n = len(base)
    b = np.asarray([h.restricted for h in base], dtype=float)
    a = np.asarray([h.restricted for h in arm], dtype=float)
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, n, size=(int(n_boot), n))
    mb = b[idx].mean(axis=1)
    ma = a[idx].mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(ma > 0, mb / ma, INF)
    finite = ratios[np.isfinite(ratios)]
    if finite.size:
        point.ci_lo = float(np.quantile(finite, alpha / 2.0))
        point.ci_hi = float(np.quantile(finite, 1.0 - alpha / 2.0))
    point.n_boot = int(n_boot)
    return point


# --------------------------------------------------------------------------- #
# Threshold freezing
# --------------------------------------------------------------------------- #
def value_at_fraction(times: Sequence[float], errors: Sequence[float],
                      fraction: float, horizon: float) -> float:
    """The error at the saved frame nearest ``fraction * T`` (no interpolation).

    Reading the *saved* frame rather than an interpolant means the frozen
    threshold is a number that actually occurred in a run.
    """
    t = np.asarray(times, dtype=float)
    e = np.asarray(errors, dtype=float)
    j = int(np.argmin(np.abs(t - float(fraction) * float(horizon))))
    return float(e[j])


def freeze_thresholds(curves: Iterable, fractions: Sequence[float],
                      horizon: float) -> List[float]:
    """Median across baseline seeds of the ABF error at each fraction of ``T``.

    ``curves`` is an iterable of ``(times, errors)`` pairs, one per plain-ABF
    calibration seed.  The rule is deliberately mechanical and is executed
    *before* any FR run is looked at, so no threshold can be chosen because it
    flatters an arm.
    """
    pairs = [(np.asarray(t, float), np.asarray(e, float)) for t, e in curves]
    if not pairs:
        raise ValueError("freeze_thresholds needs at least one baseline curve")
    out = []
    for f in fractions:
        vals = [value_at_fraction(t, e, f, horizon) for t, e in pairs]
        vals = [v for v in vals if np.isfinite(v)]
        if not vals:
            raise ValueError(
                f"every baseline seed had a non-finite error at {f:g}T; a "
                f"threshold cannot be frozen from no data")
        out.append(float(np.median(vals)))
    return out


# --------------------------------------------------------------------------- #
# Cost of acceleration
# --------------------------------------------------------------------------- #
def accel_cost(s: float, replacement_fraction: float) -> float:
    """``C_accel = (S - 1) / replacement fraction`` -- reported, never optimised.

    Birth--death necessarily spends genealogy; the question this answers is how
    much acceleration was bought per unit of population turnover.  It is a
    secondary descriptor, and the protocol forbids selecting a schedule on it.
    """
    r = float(replacement_fraction)
    return float("nan") if r <= 0 else (float(s) - 1.0) / r


# --------------------------------------------------------------------------- #
# Pre-declared decision criteria
# --------------------------------------------------------------------------- #
#: Scope in which the primary endpoint is evaluated.  ``R12`` is the
#: dimensionless thermal scope ``beta (F_ref - min F_ref) <= 12``.  On this
#: benchmark it spans ``x`` in ``[-1.74, +1.69]``: both basins *and the whole
#: barrier*, excluding only the reflecting-wall strips where the reference free
#: energy exceeds ``12 kT`` and no sampler has data.  It is not a scope chosen to
#: flatter this method -- it was frozen for the v3 campaign, before the question
#: this campaign asks existed -- and it cannot hide barrier damage, which is the
#: one thing an unflattened target is most likely to cause.
PRIMARY_SCOPE = "R12"
SECONDARY_SCOPES = ("legacy", "full")

#: Fractions of ``T`` at which the plain-ABF calibration freezes a threshold:
#: one moderate, one stringent.
THRESHOLD_FRACTIONS = (0.4, 0.6)

#: Frames the error must hold below a threshold before ``tau`` is recorded.
CONSECUTIVE_FRAMES = 3

PILOT_MIN_S_F = 1.15          # both F thresholds
PILOT_MIN_S_FPRIME = 1.10     # at least one F' threshold
SLOWDOWN_S = 0.95             # below this counts as a clear slowdown


def pilot_promising(s_F: Sequence[float], s_Fp: Sequence[float]) -> bool:
    """Stage-2 screen: a schedule worth carrying into fresh seeds.

    Acceleration-first by construction -- final error and AUC appear nowhere.
    Both free-energy thresholds must clear ``1.15``; the mean force must clear
    ``1.10`` at *one* threshold and must not show a clear slowdown at the other.
    """
    if len(s_F) != 2 or len(s_Fp) != 2:
        raise ValueError("both thresholds are required for the pilot screen")
    if not all(np.isfinite(v) and v >= PILOT_MIN_S_F for v in s_F):
        return False
    if not any(np.isfinite(v) and v >= PILOT_MIN_S_FPRIME for v in s_Fp):
        return False
    return not any(np.isfinite(v) and v < SLOWDOWN_S for v in s_Fp)


def confirms(s_F: Sequence[Speedup], s_Fp: Sequence[Speedup]) -> bool:
    """Stage-3 verdict on fresh seeds, with the bootstrap doing the work.

    Both free-energy speedups must reach ``1.15`` *and* have a paired-bootstrap
    95% CI strictly above one.  The mean force must exceed one at both
    thresholds with no clear slowdown; it is a supporting endpoint because ABF
    learns ``F'``, but the claim is about ``F``.

    A free-energy threshold at which the *arm* is censored more often than the
    baseline cannot carry the verdict: restriction inflates S^(T) exactly there,
    so a "confirmation" would partly be an artefact of the arm failing to
    converge.  Such a run is not a negative result -- it is an unresolved one,
    and the horizon is the thing to fix.
    """
    if len(s_F) != 2 or len(s_Fp) != 2:
        raise ValueError("both thresholds are required for the verdict")
    if any(v.censoring_inflates for v in s_F):
        return False
    if not all(v.s >= PILOT_MIN_S_F and v.excludes_one for v in s_F):
        return False
    return all(v.s > 1.0 for v in s_Fp) and not any(
        v.s < SLOWDOWN_S for v in s_Fp)


def rank_key(s_F2: float, replacement_fraction: float, fr_every: int,
             gamma: float):
    """Frozen tie-breaking order for schedule selection.

    Rank by the stringent free-energy speedup; then, among schedules within 5%
    of the leader, prefer fewer replacements, then a larger ``fr_every``, then a
    smaller ``gamma`` -- i.e. the *sparsest intervention that buys essentially
    the same acceleration*.  Returned as a sort key (ascending = better) so the
    ordering lives in one place and a script cannot reimplement it differently.
    """
    return (float(replacement_fraction), -int(fr_every), float(gamma),
            -float(s_F2))


def within_tolerance(best: float, other: float, tol: float = 0.05) -> bool:
    """Whether ``other`` is within ``tol`` (relative) of the leading speedup."""
    if not np.isfinite(best) or best <= 0:
        return False
    return (best - float(other)) / best <= float(tol)
