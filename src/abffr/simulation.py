"""Particle-based ABF (+ optional Fisher--Rao birth--death) dynamics.

Reaction coordinate ``xi(x, y) = x``; local mean force ``f(x, y) = dV/dx``.

Biased overdamped Langevin dynamics with the ABF bias force ``+F'_hat_t(x)``::

    dX_t = [ -dV/dx(X_t, Y_t) + F'_hat_t(X_t) ] dt + sqrt(2/beta) dW^x_t
    dY_t = [ -dV/dy(X_t, Y_t)                 ] dt + sqrt(2/beta) dW^y_t

``F'_hat_t(x) ~= E[ dV/dx(X, Y) | X = x ]`` is a kernel running-average estimate
of the mean force.  ``B_t(x) = integral F'_hat_t`` is the current bias potential.

A single engine :func:`run_simulation` implements all methods; the thin wrappers
:func:`abf_only`, :func:`abf_fr_estimated`, :func:`abf_fr_uniform`,
:func:`abf_fr_oracle` fix the ``target_type``.

The Fisher--Rao birth--death step (when active) nudges the particle x-marginal
``p_hat_t(x)`` toward a target ``q_t(x)``:

* estimated : q_t(x) proportional to exp(-beta (F_hat_target(x) - B_t(x)))
* uniform   : q_t(x) = Uniform on the x-grid
* oracle    : q_t(x) proportional to exp(-beta (F_ref(x) - B_t(x)))   [diagnostic]
* physical  : q_t(x) proportional to exp(-beta Fhat_EMA_t(x))
* physical_oracle : q_t(x) proportional to exp(-beta F_ref(x))          [diagnostic]
* self      : q_t(x) = p_hat_t(x)   [negative control, score ~ 0]
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np
import scipy.integrate as integrate

from . import potentials

EPS = 1e-30


# --------------------------------------------------------------------------- #
# Low-level numerical helpers
# --------------------------------------------------------------------------- #
def gaussian_kernel(u, bw):
    """Unit-normalised 1-D Gaussian kernel ``K_bw(u)``."""
    return np.exp(-0.5 * (u / bw) ** 2) / (bw * np.sqrt(2.0 * np.pi))


def reflect_1d(q, lo, hi):
    """Reflect coordinates into ``[lo, hi]`` (specular boundary conditions)."""
    q = np.asarray(q, dtype=float).copy()
    span = hi - lo
    q -= lo
    q = np.mod(q, 2.0 * span)
    q[q > span] = 2.0 * span - q[q > span]
    q += lo
    return q


def kde_marginal(x_eval, X, xmin, xmax, eta):
    """KDE estimate of the x-marginal ``p_hat`` with boundary reflections.

    Mirror images about ``xmin`` and ``xmax`` remove the KDE's edge bias.  The
    result is floored at ``EPS`` so downstream logs are finite.
    """
    x_eval = np.atleast_1d(np.asarray(x_eval, dtype=float))
    n = len(X)
    X_left = 2.0 * xmin - X
    X_right = 2.0 * xmax - X
    X_all = np.concatenate([X_left, X, X_right])
    diff = x_eval[:, None] - X_all[None, :]
    p = np.sum(gaussian_kernel(diff, eta), axis=1) / n
    return np.maximum(p, EPS)


def integrate_mean_force(Fprime, x_grid):
    """Antiderivative of ``Fprime`` (trapezoid), pinned to 0 at x nearest 0."""
    F = integrate.cumulative_trapezoid(Fprime, x_grid, initial=0.0)
    idx0 = int(np.argmin(np.abs(x_grid)))
    return F - F[idx0]


def gamma_schedule(step, gamma_max, fr_burnin, gamma_ramp_steps):
    """Soft-ramped Fisher--Rao rate at integration step ``step``."""
    if step < fr_burnin:
        return 0.0
    if gamma_ramp_steps <= 0:
        return gamma_max
    s = (step - fr_burnin) / gamma_ramp_steps
    return gamma_max * (1.0 - np.exp(-max(s, 0.0)))


# --------------------------------------------------------------------------- #
# Fisher--Rao target densities and score
# --------------------------------------------------------------------------- #
def estimated_fr_target(x_grid, Fhat_target_grid, B_grid, beta, eps=1e-12):
    """q(x) proportional to exp(-beta (F_hat_target(x) - B(x))), normalised on ``x_grid``."""
    exponent = -beta * (Fhat_target_grid - B_grid)
    exponent = exponent - np.max(exponent)
    q = np.exp(exponent)
    q = q / np.maximum(np.trapezoid(q, x_grid), eps)
    q = np.maximum(q, eps)
    q = q / np.maximum(np.trapezoid(q, x_grid), eps)
    return q


def physical_fr_target(x_grid, F_grid, beta):
    """Normalized physical marginal proportional to exp(-beta F)."""
    exponent = -float(beta) * np.asarray(F_grid, dtype=float)
    exponent = exponent - np.max(exponent)
    q = np.exp(exponent)
    return q / np.maximum(np.trapezoid(q, x_grid), EPS)


def uniform_fr_target(x_grid):
    """Uniform density on ``[x_grid[0], x_grid[-1]]``."""
    width = x_grid[-1] - x_grid[0]
    return np.full_like(x_grid, 1.0 / max(width, EPS))


def fr_score(X, p_at_X, p_grid, q_grid, x_grid, score_clip=None):
    """Mean-zero Fisher--Rao particle score against a grid target.

    The continuum KL centering is followed by particle-mean recentering. When
    clipping is requested, clip/recenter is iterated to keep the fixed-population
    discretisation mass preserving while bounding extreme scores.
    """
    Zp = np.maximum(np.trapezoid(p_grid, x_grid), EPS)
    Zq = np.maximum(np.trapezoid(q_grid, x_grid), EPS)
    p_g = p_grid / Zp
    q_g = q_grid / Zq
    p_part = p_at_X / Zp

    q_part = np.maximum(np.interp(X, x_grid, q_g), EPS)
    log_ratio_grid = np.log(np.maximum(p_g, EPS)) - np.log(np.maximum(q_g, EPS))
    baseline = np.trapezoid(p_g * log_ratio_grid, x_grid)
    S = np.log(np.maximum(p_part, EPS)) - np.log(q_part) - baseline
    S = S - np.mean(S)
    if score_clip is not None:
        for _ in range(3):
            S = np.clip(S, -float(score_clip), float(score_clip))
            S = S - np.mean(S)
    return S


def resample_fixed_N(X, Y, ancestors, S, gamma, dt, rng,
                     max_event_fraction=None):
    """Fixed-N death/clone resampling with a replacement-fraction cap.

    Positive-score particles are death candidates. Each realised dead slot is
    replaced by a clone sampled from the negative-score pool with probability
    proportional to minus S. Source arrays are never mutated during sampling.
    """
    N = len(X)
    positive = S > 0
    p_death = np.where(
        positive,
        np.clip(1.0 - np.exp(-float(gamma) * S * float(dt)), 0.0, 1.0),
        0.0,
    )
    die = positive & (rng.uniform(size=N) < p_death)

    if max_event_fraction is not None:
        cap = int(np.floor(float(max_event_fraction) * N))
        if cap <= 0:
            die[:] = False
        elif int(die.sum()) > cap:
            chosen = rng.choice(np.flatnonzero(die), size=cap, replace=False)
            die[:] = False
            die[chosen] = True

    birth_weights = np.where(S < 0, -S, 0.0)
    total_birth_weight = float(np.sum(birth_weights))
    if not np.isfinite(total_birth_weight) or total_birth_weight <= EPS:
        die[:] = False

    dead = np.flatnonzero(die)
    n_replace = int(dead.size)
    X_new = np.asarray(X).copy()
    Y_new = np.asarray(Y).copy()
    ancestors_new = np.asarray(ancestors).copy()
    if n_replace:
        source = rng.choice(
            N, size=n_replace, replace=True,
            p=birth_weights / total_birth_weight)
        X_new[dead] = np.asarray(X)[source]
        Y_new[dead] = np.asarray(Y)[source]
        ancestors_new[dead] = np.asarray(ancestors)[source]

    info = dict(n_die=n_replace, n_clone=n_replace, n_events=n_replace)
    return X_new, Y_new, ancestors_new, info


# --------------------------------------------------------------------------- #
# Initial conditions
# --------------------------------------------------------------------------- #
def _init_positions(rng, n, lo, hi, mode):
    """Draw ``n`` initial coordinates in ``[lo, hi]``.

    ``mode``:
      * ``"uniform"`` -- uniform over the whole interval.
      * ``"mixed"``   -- half near the left third, half near the right third,
        so that without exploration help the two basins are both seeded
        (a fair starting point for studying exploration).
      * ``"left"`` / ``"right"`` -- all particles in one third.
    """
    if mode == "uniform":
        return rng.uniform(lo, hi, size=n)
    span = hi - lo
    if mode == "mixed":
        n_left = n // 2
        left = rng.uniform(lo + 0.10 * span, lo + 0.40 * span, size=n_left)
        right = rng.uniform(lo + 0.60 * span, lo + 0.90 * span, size=n - n_left)
        return np.concatenate([left, right])
    if mode == "left":
        return rng.uniform(lo + 0.10 * span, lo + 0.40 * span, size=n)
    if mode == "right":
        return rng.uniform(lo + 0.60 * span, lo + 0.90 * span, size=n)
    raise ValueError(f"Unknown init mode {mode!r}")


# --------------------------------------------------------------------------- #
# Core engine
# --------------------------------------------------------------------------- #
def run_simulation(
    *,
    target_type: str,
    beta: float,
    dt: float,
    n_steps: int,
    n_particles: int,
    eval_every: int,
    x_grid: np.ndarray,
    F_ref: np.ndarray,
    Fprime_ref: np.ndarray,
    domain: Dict[str, float],
    h: float,
    eta: float,
    min_count: float = 1.0,
    ema_alpha: float = 0.05,
    gamma: float = 0.0,
    burnin_fraction: float = 0.0,
    stop_fraction: float = 1.0,
    ramp_fraction: float = 0.1,
    fr_every: int = 5,
    score_clip: Optional[float] = 5.0,
    max_event_fraction: Optional[float] = 0.10,
    x_init_mode: str = "mixed",
    y_init_mode: str = "mixed",
    x_barrier: float = 0.0,
    observation_order: str = "pre_propagation",
    update_every: int = 1,
    x_tilt: float = 0.0,
    interval_scaled_clock: bool = False,
    rng_init: Optional[np.random.Generator] = None,
    rng_noise: Optional[np.random.Generator] = None,
    rng_fr: Optional[np.random.Generator] = None,
    V_func: Callable = potentials.potential_xy,
    dVdx_func: Callable = potentials.dVdx_xy,
    dVdy_func: Callable = potentials.dVdy_xy,
) -> Dict:
    """Run one ABF(+FR) simulation and return aligned diagnostics.

    post_propagation implements the v2 estimator protection:
    propagate -> accumulate -> optional FR. The legacy order remains available
    for regression. FR is active on completed steps in
    [burnin_fraction*n_steps, stop_fraction*n_steps).
    """
    allowed = {
        "none", "estimated", "uniform", "oracle", "self",
        "physical", "physical_oracle",
    }
    if target_type not in allowed:
        raise ValueError(f"Unknown target_type {target_type!r}")
    if fr_every <= 0:
        raise ValueError("fr_every must be positive")
    if update_every <= 0:
        raise ValueError("update_every must be positive")
    if observation_order not in {"pre_propagation", "post_propagation"}:
        raise ValueError(
            "observation_order must be 'pre_propagation' or 'post_propagation'")
    if not 0.0 <= burnin_fraction <= stop_fraction <= 1.0:
        raise ValueError(
            "FR schedule must satisfy 0 <= burnin_fraction <= stop_fraction <= 1")

    if rng_init is None:
        rng_init = np.random.default_rng()
    if rng_noise is None:
        rng_noise = np.random.default_rng()
    if rng_fr is None:
        rng_fr = np.random.default_rng()

    xmin, xmax = domain["x_min"], domain["x_max"]
    ymin, ymax = domain["y_min"], domain["y_max"]
    n_grid = len(x_grid)

    fr_burnin = int(round(burnin_fraction * n_steps))
    fr_stop = int(round(stop_fraction * n_steps))
    gamma_ramp_steps = int(round(ramp_fraction * n_steps))
    fr_active = (target_type != "none") and (gamma > 0.0)
    ema_alpha_eff = 1.0 - (1.0 - float(ema_alpha)) ** int(update_every)

    X = _init_positions(rng_init, n_particles, xmin, xmax, x_init_mode)
    Y = _init_positions(rng_init, n_particles, ymin, ymax, y_init_mode)
    ancestors = np.arange(n_particles)
    noise_scale = np.sqrt(2.0 * dt / beta)

    num_acc = np.zeros(n_grid)
    den_acc = np.zeros(n_grid)
    Fprime_hat = np.zeros(n_grid)
    F_hat = np.zeros(n_grid)
    Fhat_target_grid = np.zeros(n_grid)

    diag = _empty_diag(target_type)
    win_events = 0
    win_fr_steps = 0
    win_event_fracs = []
    win_score_means, win_score_stds = [], []
    win_score_mins, win_score_maxs = [], []
    cumulative_fr_events = 0
    cumulative_replacements = 0
    barrier_crossings = 0
    prev_sign = np.sign(X - x_barrier)

    def accumulate(X_obs, Y_obs):
        nonlocal num_acc, den_acc
        diff = x_grid[:, None] - X_obs[None, :]
        weights = gaussian_kernel(diff, h)
        force = dVdx_func(X_obs, Y_obs) + float(x_tilt)
        num_acc += np.sum(weights * force[None, :], axis=1)
        den_acc += np.sum(weights, axis=1)

    def update_profiles():
        nonlocal Fprime_hat, F_hat, Fhat_target_grid
        Fprime_hat = num_acc / (den_acc + min_count + EPS)
        F_hat = integrate_mean_force(Fprime_hat, x_grid)
        Fhat_target_grid = (
            (1.0 - ema_alpha_eff) * Fhat_target_grid
            + ema_alpha_eff * F_hat)

    def save_snapshot(completed_step, gamma_eff):
        nonlocal win_events, win_fr_steps, win_event_fracs
        nonlocal win_score_means, win_score_stds
        nonlocal win_score_mins, win_score_maxs
        p_hat_grid = kde_marginal(x_grid, X, xmin, xmax, eta)
        idx0 = int(np.argmin(np.abs(x_grid)))
        F_hat_centered = F_hat - F_hat[idx0]
        q_save = _build_target(
            target_type, x_grid, Fhat_target_grid, F_hat, beta,
            X, xmin, xmax, eta, F_ref)
        counts = np.bincount(ancestors, minlength=n_particles).astype(float)
        ess = counts.sum() ** 2 / max(float(np.sum(counts ** 2)), 1.0)
        max_mult = int(counts.max())
        diag["steps"].append(int(completed_step))
        diag["times"].append(float(completed_step * dt))
        diag["Fprime_hat"].append(Fprime_hat.copy())
        diag["F_hat"].append(F_hat_centered.copy())
        diag["p_hat_grid"].append(p_hat_grid.copy())
        diag["q_target_grid"].append(q_save.copy())
        diag["X_snap"].append(X.copy())
        diag["Y_snap"].append(Y.copy())
        diag["barrier_crossings"].append(int(barrier_crossings))
        diag["n_unique_ancestors"].append(int(np.count_nonzero(counts)))
        diag["ancestor_ess"].append(float(ess))
        diag["max_clone_multiplicity"].append(max_mult)
        diag["max_clone_weight"].append(float(max_mult / n_particles))
        diag["gamma_eff"].append(float(gamma_eff))
        diag["fr_applied"].append(bool(win_fr_steps > 0))
        diag["fr_event_fraction"].append(
            float(np.mean(win_event_fracs)) if win_event_fracs else 0.0)
        diag["fr_event_fraction_max"].append(
            float(np.max(win_event_fracs)) if win_event_fracs else 0.0)
        diag["fr_events_total"].append(int(win_events))
        diag["score_mean"].append(
            float(np.mean(win_score_means)) if win_score_means else np.nan)
        diag["score_std"].append(
            float(np.mean(win_score_stds)) if win_score_stds else np.nan)
        diag["score_min"].append(
            float(np.min(win_score_mins)) if win_score_mins else np.nan)
        diag["score_max"].append(
            float(np.max(win_score_maxs)) if win_score_maxs else np.nan)
        diag["cumulative_fr_events"].append(int(cumulative_fr_events))
        diag["cumulative_replacements"].append(int(cumulative_replacements))
        win_events = 0
        win_fr_steps = 0
        win_event_fracs = []
        win_score_means, win_score_stds = [], []
        win_score_mins, win_score_maxs = [], []

    # A genuine time-zero snapshot makes full-run AUCs cover [0, T].
    save_snapshot(0, 0.0)

    for step in range(n_steps):
        next_step = step + 1

        if observation_order == "pre_propagation":
            accumulate(X, Y)
            if step % update_every == 0:
                update_profiles()

        force_x = dVdx_func(X, Y) + float(x_tilt)
        force_y = dVdy_func(X, Y)
        abf_at_X = np.interp(X, x_grid, Fprime_hat)
        X_prop = X + (-force_x + abf_at_X) * dt \
            + noise_scale * rng_noise.standard_normal(n_particles)
        Y_prop = Y + (-force_y) * dt \
            + noise_scale * rng_noise.standard_normal(n_particles)
        X_prop = reflect_1d(X_prop, xmin, xmax)
        Y_prop = reflect_1d(Y_prop, ymin, ymax)

        new_sign = np.sign(X_prop - x_barrier)
        crossed = (new_sign != prev_sign) & (new_sign != 0) & (prev_sign != 0)
        barrier_crossings += int(np.count_nonzero(crossed))

        if observation_order == "post_propagation":
            accumulate(X_prop, Y_prop)
            if next_step % update_every == 0:
                update_profiles()

        in_window = (
            fr_active and fr_burnin <= next_step < fr_stop)
        gamma_eff = (
            gamma_schedule(next_step, gamma, fr_burnin, gamma_ramp_steps)
            if in_window else 0.0)
        do_fr = (
            in_window
            and ((next_step - fr_burnin) % fr_every == 0)
            and gamma_eff > 0.0)

        if do_fr:
            p_grid = kde_marginal(x_grid, X_prop, xmin, xmax, eta)
            p_at_X = kde_marginal(X_prop, X_prop, xmin, xmax, eta)
            q_grid = _build_target(
                target_type, x_grid, Fhat_target_grid, F_hat, beta,
                X_prop, xmin, xmax, eta, F_ref)
            S = fr_score(
                X_prop, p_at_X, p_grid, q_grid, x_grid,
                score_clip=score_clip)
            clock_dt = dt * fr_every if interval_scaled_clock else dt
            X, Y, ancestors, info = resample_fixed_N(
                X_prop, Y_prop, ancestors, S, gamma_eff, clock_dt, rng_fr,
                max_event_fraction=max_event_fraction)
            win_events += info["n_events"]
            win_fr_steps += 1
            win_event_fracs.append(info["n_events"] / n_particles)
            win_score_means.append(float(np.mean(S)))
            win_score_stds.append(float(np.std(S)))
            win_score_mins.append(float(np.min(S)))
            win_score_maxs.append(float(np.max(S)))
            cumulative_fr_events += 1
            cumulative_replacements += info["n_events"]
        else:
            X, Y = X_prop, Y_prop

        prev_sign = np.sign(X - x_barrier)

        if next_step % eval_every == 0 or next_step == n_steps:
            save_snapshot(next_step, gamma_eff)

    diag["fr_burnin"] = fr_burnin
    diag["fr_stop"] = fr_stop
    diag["fr_every"] = int(fr_every)
    diag["n_steps"] = int(n_steps)
    diag["dt"] = float(dt)
    diag["observation_order"] = observation_order
    diag["interval_scaled_clock"] = bool(interval_scaled_clock)
    diag["x_tilt"] = float(x_tilt)
    return diag


def _build_target(target_type, x_grid, Fhat_target_grid, B_grid, beta,
                  X, xmin, xmax, eta, F_ref):
    """Construct the normalized FR target density q on x_grid."""
    if target_type == "uniform":
        return uniform_fr_target(x_grid)
    if target_type == "oracle":
        return estimated_fr_target(x_grid, F_ref, B_grid, beta)
    if target_type == "physical":
        return physical_fr_target(x_grid, Fhat_target_grid, beta)
    if target_type == "physical_oracle":
        return physical_fr_target(x_grid, F_ref, beta)
    if target_type == "self":
        q = kde_marginal(x_grid, X, xmin, xmax, eta)
        return q / np.maximum(np.trapezoid(q, x_grid), EPS)
    # Legacy estimated target; none saves the same diagnostic grid.
    return estimated_fr_target(x_grid, Fhat_target_grid, B_grid, beta)


def _empty_diag(target_type):
    return dict(
        target_type=target_type,
        steps=[], times=[],
        Fprime_hat=[], F_hat=[], p_hat_grid=[], q_target_grid=[],
        X_snap=[], Y_snap=[],
        barrier_crossings=[], n_unique_ancestors=[], ancestor_ess=[],
        max_clone_multiplicity=[], max_clone_weight=[], gamma_eff=[],
        fr_applied=[], fr_event_fraction=[], fr_event_fraction_max=[],
        fr_events_total=[], score_mean=[], score_std=[], score_min=[], score_max=[],
        cumulative_fr_events=[], cumulative_replacements=[],
    )


# --------------------------------------------------------------------------- #
# Thin method wrappers (fix the FR target type)
# --------------------------------------------------------------------------- #
def abf_only(**kwargs):
    kwargs.pop("target_type", None)
    return run_simulation(target_type="none", **kwargs)


def abf_fr_estimated(**kwargs):
    kwargs.pop("target_type", None)
    return run_simulation(target_type="estimated", **kwargs)


def abf_fr_uniform(**kwargs):
    kwargs.pop("target_type", None)
    return run_simulation(target_type="uniform", **kwargs)


def abf_fr_oracle(**kwargs):
    kwargs.pop("target_type", None)
    return run_simulation(target_type="oracle", **kwargs)


def abf_fr_physical(**kwargs):
    kwargs.pop("target_type", None)
    return run_simulation(target_type="physical", **kwargs)


def abf_fr_physical_oracle(**kwargs):
    kwargs.pop("target_type", None)
    return run_simulation(target_type="physical_oracle", **kwargs)


METHOD_DISPATCH = {
    "abf_only": abf_only,
    "abf_fr_estimated": abf_fr_estimated,
    "abf_fr_uniform": abf_fr_uniform,
    "abf_fr_oracle": abf_fr_oracle,
    "abf_fr_physical": abf_fr_physical,
    "abf_fr_physical_oracle": abf_fr_physical_oracle,
}
