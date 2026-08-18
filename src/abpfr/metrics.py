"""Post-hoc metrics on saved time series (numpy; nothing here touches the simulator).

Primary endpoints of the campaign (docs/PREREGISTRATION_GATEWAY.md):

* e_F(t)  -- L2 free-energy error with the optimal additive constant,
* I_F(T)  -- integrated error (area under e_F),
* tau_eps -- time-to-accuracy with a persistence window (right-censored -> nan),
* S_eps   -- paired speedup tau^baseline / tau^arm.

Everything consumes plain arrays so stored runs can always be rescored.
"""
from __future__ import annotations

import numpy as np


def l2_error_gauge(F_hat, F_ref, eval_mask):
    """RMS over the eval window after subtracting the optimal additive constant.

    For an unweighted L2 the optimal constant is the mean of the difference, so this is
    the mean-centered interior RMS.  F_hat: (..., G); F_ref: (G,) or broadcastable.
    """
    d = (np.asarray(F_hat) - np.asarray(F_ref))[..., eval_mask]
    d = d - d.mean(axis=-1, keepdims=True)
    return np.sqrt((d * d).mean(axis=-1))


def integrated_error(times, err):
    """I_F(T) = integral of e_F dt (trapezoid).  err: (..., n_saves) -> (...)."""
    return np.trapezoid(err, np.asarray(times), axis=-1)


def time_to_accuracy(times, err, eps, persist_frac=0.2):
    """First t with err <= eps over the whole trailing window [t, t + persist_frac*T].

    Returns nan when never attained (right-censored; callers must keep such runs
    visible, not drop them).  err: (n_saves,).
    """
    times = np.asarray(times, dtype=float)
    err = np.asarray(err, dtype=float)
    T = times[-1]
    below = err <= eps
    for i in range(len(times)):
        j = np.searchsorted(times, times[i] + persist_frac * T, side="right")
        j = max(j, i + 1)
        if below[i:j].all() and (j > i):
            # the window must be fully observed unless it runs past the end of the data
            return float(times[i])
    return float("nan")


def speedup(tau_base, tau_arm):
    """S_eps = tau^baseline / tau^arm, elementwise; nan-censored inputs give nan."""
    tb, ta = np.asarray(tau_base, float), np.asarray(tau_arm, float)
    return tb / ta


def relative_error_curve(err_arm, err_base):
    """R_F(t) = e_F^arm(t) / e_F^baseline(t) (matched seeds, same time axis)."""
    return np.asarray(err_arm) / np.asarray(err_base)


def cosine_modes(profile_err, x, eval_lo, eval_hi, k_max=4):
    """Project a (gauge-centered) bias-error profile onto low Neumann cosine modes.

    The gateway domain is reflecting, not periodic, so the natural low-frequency basis
    on the eval window [lo, hi] is cos(k pi (x - lo)/L).  profile_err: (..., G).
    Returns (..., k_max) coefficients a_k = (2/L) * integral E(x) cos(...) dx, k>=1.
    """
    x = np.asarray(x)
    m = (x >= eval_lo) & (x <= eval_hi)
    xs = x[m]
    L = xs[-1] - xs[0]
    E = np.asarray(profile_err)[..., m]
    E = E - E.mean(axis=-1, keepdims=True)
    out = []
    for k in range(1, k_max + 1):
        basis = np.cos(k * np.pi * (xs - xs[0]) / L)
        out.append((2.0 / L) * np.trapezoid(E * basis, xs, axis=-1))
    return np.stack(out, axis=-1)


def kl_to_uniform_np(p, dx, volume):
    """KL(p||u) from a stored marginal.  p: (..., G) -> (...)."""
    p = np.clip(np.asarray(p), 1e-300, None)
    integ = p * (np.log(p) - np.log(1.0 / volume))
    return np.trapezoid(integ, dx=dx, axis=-1)


def tv_to_uniform_np(p, dx, volume):
    p = np.asarray(p)
    return 0.5 * np.trapezoid(np.abs(p - 1.0 / volume), dx=dx, axis=-1)


def paired_bootstrap_ci(values, stat=np.median, n_boot=10_000, alpha=0.05, seed=20260818):
    """Bootstrap CI for a statistic of per-seed paired values (e.g. differences/ratios).

    nan entries (censored seeds) are kept OUT of the resampling only if the caller
    removed them; by default nan propagates so censoring stays visible.
    """
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(v)
    stats = np.array([stat(v[rng.integers(0, n, n)]) for _ in range(n_boot)])
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(stat(v)), float(lo), float(hi)
