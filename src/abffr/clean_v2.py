"""Clean-v2: intermittent physical-target Fisher--Rao as an ABF *accelerator*.

Frozen protocol: ``docs/CLEAN_V2_PREREGISTRATION.md``.

The research question this module exists to answer is **not** whether ABF+FR has
a better stationary distribution or a lower asymptotic error.  It is::

    Does intermittent physical-target Fisher--Rao birth--death make ABF reach a
    given free-energy accuracy in less physical simulation time?

so the headline quantity is a *time-to-accuracy ratio* (:mod:`abffr.accel`), not
a final error.  FR is switched off before the end of every run, which makes the
arm and its baseline literally the same plain-ABF algorithm over the last
segment; the ``t -> infinity`` limit is therefore ABF's, by construction.

The target
----------
The physical (unflattened) reaction-coordinate marginal is
``q(z) propto exp(-beta F(z))``.  ``F`` is unknown, so the deployable target
uses the *current* ABF free-energy estimate ``A_t = integral Fhat'_t``::

    q_t(z) = exp(-beta A_t(z)) / integral exp(-beta A_t(u)) du.

There is no interpolation, no cap, no tempering, no flat/physical mixture and no
EMA: ABF's accumulator is already a cumulative estimator, and a second smoothed
free-energy memory is a second thing to tune.

The score, without ever exponentiating
--------------------------------------
Since ``log q_t(z) = -beta A_t(z) + C_t`` and the Fisher--Rao score is centered,
the normalisation ``C_t`` cancels identically::

    r_i = log phat_t(z_i) + beta A_t(z_i),
    S_i = r_i - (1/K) sum_j r_j.

That is exactly ``log(phat/q_t)`` minus its empirical mean, computed with no
normalising integral, no free-energy gauge convention and no ``exp(-100)``
underflow.  It is also *gauge invariant by construction*: ``A_t -> A_t + C``
shifts every ``r_i`` by ``beta C``, which the centering removes (Gate C).

``phat_t`` is only ever evaluated *at particle positions*, where a KDE is bounded
below by its own self-contribution ``1/(K eta sqrt(2 pi))``.  The score therefore
cannot be produced by a floored density: the ``EPS`` guard below is a NaN
backstop, and the engine records whether it ever binds.

The operator
------------
The standard particle birth--death realization, :func:`abffr.fr_v3.bd_standard`
-- ``S_i > 0`` means over-represented relative to the physical target, so ``i``
dies and a uniformly chosen replica duplicates; ``S_i < 0`` means
under-represented, so ``i`` duplicates and a uniformly chosen replica dies; the
event probability is ``1 - exp(-gamma |S_i| dtau_FR)`` with **no** cap on the
tail and **no** ceiling on the number of events.  Population is exactly fixed.
It is reused verbatim rather than re-derived here, because two independently
written realisations of one operator is how a discretisation drifts.

The FR clock is interval-scaled, ``dtau_FR = L_FR dt``, so that over a fixed
physical window the *total* nominal Fisher--Rao reaction time is approximately
the window length whatever ``L_FR`` is.  Varying ``L_FR`` then studies
"frequent and weak" against "sparse and strong" rather than handing one arm more
integrated FR dose than another.

The two scientific knobs are ``L_FR`` (``fr_every``) and ``gamma``.  Everything
else in this module is frozen, and :func:`validate_config` refuses to run a
configuration that reintroduces a knob the protocol removed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from . import fr_v3, torch_utils as tu

# NaN backstop for log(phat) only.  See the module docstring: at particle
# positions a KDE is bounded below by its own self-contribution, so a binding
# floor is an engineering anomaly and the engine reports it as one.
EPS = 1e-300

#: FR target names the clean-v2 protocol admits.  ``physical`` is the deployable
#: method (``A_t`` = the running ABF estimate); ``physical_oracle`` is a
#: diagnostic that substitutes ``F_ref`` for ``A_t`` and is never a candidate.
TARGETS = ("none", "physical", "physical_oracle")

#: Knobs the post-mortems removed.  Each maps to the reason, which is what the
#: error message prints -- a config that reintroduces one must argue with the
#: finding, not merely with a validator.
BANNED_FR_KEYS = {
    "score_clip": (
        "score clipping truncated 55.4% of particle scores in v2 and compressed "
        "a 27.19-nat raw span to 8.11 nats; the clipped operator is not "
        "birth-death"),
    "max_event_fraction": (
        "the 10% event cap demonstrably bound and silently changed the FR dose; "
        "if FR is too strong, reduce gamma -- do not truncate the operator"),
    "target_ema_alpha": (
        "ABF's accumulator is already a cumulative estimator; a second smoothed "
        "free-energy memory is a second thing to tune"),
}

#: Knobs that must be present-but-inert if present at all.
ZERO_FR_KEYS = ("ramp_fraction", "jitter")


@dataclass(frozen=True)
class CleanV2:
    """The validated clean-v2 switch.  Carries no tunable state by design."""

    enabled: bool = True

    def __post_init__(self):
        if not self.enabled:
            raise ValueError("CleanV2 is only constructed when enabled")


def validate_config(cfg: Dict) -> None:
    """Fail loud on anything the clean-v2 protocol froze out.

    This is deliberately a *rejection*, not a default: "we removed the knob"
    then becomes a checkable property of the config on disk rather than a
    promise about what the engine happens to read.
    """
    fr = cfg.get("fr", {}) or {}
    for key, why in BANNED_FR_KEYS.items():
        if key in fr:
            raise ValueError(
                f"clean_v2 forbids fr.{key} (found {fr[key]!r}): {why}. "
                f"Remove the key; there is no 'off' value for it.")
    for key in ZERO_FR_KEYS:
        if key in fr and float(fr[key]) != 0.0:
            raise ValueError(
                f"clean_v2 requires fr.{key} == 0 (found {fr[key]!r}); the FR "
                f"operator is applied as written, without a ramp or jitter.")
    if not bool(fr.get("interval_scaled_clock", True)):
        raise ValueError(
            "clean_v2 requires fr.interval_scaled_clock: true so that dtau_FR = "
            "L_FR * dt and fr_every does not silently rescale the FR dose")
    abf = cfg.get("abf", {}) or {}
    if "ema_alpha" in abf:
        raise ValueError(
            "clean_v2 forbids abf.ema_alpha: there is no smoothed free-energy "
            "process in the protocol, and leaving an inert knob in the config "
            "is how a reader concludes the target was smoothed after all")

    # Selection on final or integrated error is the endpoint the campaign moved
    # away from; the generic selector still ranks on it, so it must not be able
    # to write a file from a clean-v2 stage.
    if bool((cfg.get("selection", {}) or {}).get("write_generic_best", True)):
        raise ValueError(
            "clean_v2 requires selection.write_generic_best: false -- schedules "
            "are selected on time-to-accuracy (scripts/analyze_clean_v2.py), "
            "never on final or integrated L2 error")

    for block in ("v3", "v4"):
        if cfg.get(block):
            raise ValueError(
                f"clean_v2 replaces the {block} campaign; remove the '{block}:' "
                f"block (see docs/CLEAN_V2_PREREGISTRATION.md, 'What is retired')")

    bad = sorted(set(fr.get("target_types", ["physical"])) - set(TARGETS))
    if bad:
        raise ValueError(
            f"clean_v2 admits only {list(TARGETS)} as FR targets; got {bad}. "
            f"There is no target family, interpolation or tempering in v2-clean.")

    # A_t must be current at every FR opportunity: an opportunity that lands
    # between two estimator refreshes would build its target from a stale free
    # energy, and the staleness would scale with update_every rather than with
    # anything scientific.
    update_every = max(1, int((cfg.get("abf", {}) or {}).get("update_every", 1)))
    for every in fr.get("fr_every_values", []):
        if int(every) % update_every:
            raise ValueError(
                f"clean_v2 requires fr_every ({every}) to be a multiple of "
                f"abf.update_every ({update_every}) so that A_t is current at "
                f"every FR opportunity")

    # The three-phase structure is the claim's shape, not a tuning choice.
    # Phase I must be non-empty because exp(-beta A_t) is not a credible physical
    # target at t = 0, and Phase III must be non-empty because "the arm and its
    # baseline are the same plain-ABF algorithm at the end" is the sentence that
    # licenses using a physical target at all.  The v3 campaign lost every FR arm
    # to exactly this: a window that silently ran to the end of the run.
    burnins = [float(b) for b in fr.get("burnin_fractions", [])]
    for burn in burnins:
        if not 0.0 < burn < 1.0:
            raise ValueError(
                "clean_v2 requires a strictly positive pure-ABF burn-in: "
                "exp(-beta A_t) is not a credible physical target at t = 0")
    durations = fr.get("duration_fractions")
    stops = ([b + float(d) for b in burnins for d in durations]
             if durations is not None
             else [float(s) for s in fr.get("stop_fractions", [1.0])])
    for stop in stops:
        if stop >= 1.0:
            raise ValueError(
                f"clean_v2 requires the FR window to close strictly before the "
                f"end of the run (got stop_fraction {stop:g}).  Phase III is "
                f"what makes the long-time limit ABF's by construction; with "
                f"burnin_fractions {burnins} you want "
                f"duration_fractions <= {1.0 - max(burnins or [0.0]):g} "
                f"(exclusive).")


def from_config(cfg: Dict) -> Optional[CleanV2]:
    """Return the validated switch, or ``None`` when clean-v2 is not selected."""
    block = cfg.get("clean_v2", {}) or {}
    if not bool(block.get("enabled", False)):
        return None
    validate_config(cfg)
    return CleanV2(enabled=True)


# --------------------------------------------------------------------------- #
# Schedule
# --------------------------------------------------------------------------- #
def window_steps(n_steps: int, burnin_fraction: float,
                 stop_fraction: float) -> tuple:
    """``(burn, stop)`` in integration steps.  FR fires on ``[burn, stop)``."""
    if not 0.0 <= burnin_fraction <= stop_fraction <= 1.0:
        raise ValueError("FR schedule must satisfy 0 <= burnin <= stop <= 1")
    return (int(round(burnin_fraction * n_steps)),
            int(round(stop_fraction * n_steps)))


def firing_steps(n_steps: int, burnin_fraction: float, stop_fraction: float,
                 fr_every: int) -> List[int]:
    """Every step index at which an FR opportunity fires.

    Phase I is ``[0, burn)`` (pure ABF), Phase II is ``[burn, stop)`` (ABF plus
    a pulse every ``fr_every`` steps, the first landing exactly on ``burn``) and
    Phase III is ``[stop, n_steps]`` (pure ABF again).  The half-open window is
    the whole of Gate F: no pulse before ``t_burn``, none at or after ``t_off``.

    This is the *specification* of the schedule; the engine reproduces it inline
    and the gate compares the two, so a drift in either shows up as a failure
    rather than as a plot.
    """
    burn, stop = window_steps(n_steps, burnin_fraction, stop_fraction)
    every = int(fr_every)
    if every <= 0:
        raise ValueError("fr_every must be a positive number of steps")
    return [s for s in range(1, int(n_steps) + 1)
            if burn <= s < stop and (s - burn) % every == 0]


def dtau(gamma: float, fr_every: int, dt: float) -> float:
    """Effective FR time for one opportunity: ``gamma * L_FR * dt``.

    ``gamma`` is folded into ``dtau`` so the operator sees a single reaction
    time and the event probability is exactly ``1 - exp(-gamma |S| L_FR dt)``.
    """
    return float(gamma) * int(fr_every) * float(dt)


# --------------------------------------------------------------------------- #
# Target and score
# --------------------------------------------------------------------------- #
def log_target_at(A_at: torch.Tensor, beta: float) -> torch.Tensor:
    """``log q_t`` up to the additive constant that the centering removes."""
    return -float(beta) * A_at


def score(p_hat: torch.Tensor, A_grid: torch.Tensor, X: torch.Tensor,
          x0: float, dx: float, beta: float):
    """The clean-v2 Fisher--Rao score for one batch.

    Returns ``(S, log_p_at, log_q_at, floored)`` where ``S`` is ``(B, N)``
    mean-centered and ``floored`` is the per-row fraction of particles whose
    ``log phat`` hit the NaN backstop (expected to be exactly zero).
    """
    p_at = tu.interp1d(p_hat, X, x0, dx)
    floored = (p_at <= EPS).to(p_at.dtype).mean(dim=1)
    log_p_at = torch.log(p_at.clamp_min(EPS))
    log_q_at = log_target_at(tu.interp1d(A_grid, X, x0, dx), beta)
    r = log_p_at - log_q_at
    return r - r.mean(dim=1, keepdim=True), log_p_at, log_q_at, floored


def row_score(log_p_at: torch.Tensor, log_q_at: torch.Tensor) -> fr_v3.FRScore:
    """Wrap one row's log-densities in the operator's score object."""
    return fr_v3.FRScore(log_p=log_p_at, log_q=log_q_at)


def event_probability(S: torch.Tensor, dtau_eff: float) -> torch.Tensor:
    """``1 - exp(-|S| dtau_eff)`` -- the operator's probability, uncapped.

    Written once, here, so the engine's Gate-E diagnostic and the operator
    cannot disagree about what probability was applied.
    """
    return 1.0 - torch.exp(-S.abs() * float(dtau_eff))


def target_grid(A_grid: torch.Tensor, beta: float, dx: float) -> torch.Tensor:
    """``q_t`` rendered on the profile grid.  **Diagnostics and figures only.**

    The algorithm never evaluates this: it works from ``log phat + beta A`` at
    particle positions.  Rendering exists so the mechanism figure can show
    ``phat_t``, ``q_t`` and the reference physical marginal on one axis.
    """
    e = -float(beta) * A_grid
    e = e - e.max(dim=1, keepdim=True).values
    return tu.normalize_density(torch.exp(e), dx)
