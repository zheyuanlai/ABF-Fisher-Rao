"""Brownian dynamics for molecules, free and constrained to Sigma(z).

Mobility is M^{-1} (friction proportional to mass), so with h = dt/gamma

    free        q <- q - h M^{-1} grad V + sqrt(2 h / beta) M^{-1/2} eta
    constrained q~ = q + P_M(q) [ -h M^{-1} grad V + sqrt(2h/beta) M^{-1/2} eta ]
                q' = SHAKE(q~ -> Sigma(z))

Both leave the right measure invariant up to the usual O(h) projected-Euler
bias, which the campaign measures rather than assumes.  P_M is the M-orthogonal
tangent projector, so the constrained scheme is plain projected Brownian motion
in mass-weighted coordinates and therefore samples the RIGID measure
e^{-beta V} sigma^M_Sigma; thermodynamic integration reweights it by
(det G)^{-1/2} at deposit time (see geom.py).

The inner loops are wrapped by torch.compile(mode='reduce-overhead'), which
replays them as CUDA graphs.  These steps are launch-bound, not flop-bound: a
single uncompiled step costs ~4 ms of kernel launches for ANY batch size, and
the compiled one ~0.1 ms, so the batch axis is close to free -- put every
replica, seed and hyper-parameter configuration in it.
"""
from __future__ import annotations

import torch

from .ff import _wrap
from .geom import TorsionCV, _grad_at, _scatter


def free_step(top, q, h, beta, drift_cap=None):
    """`drift_cap` bounds the deterministic displacement at C thermal steps.

    At equilibrium the drift is ~0.14 noise amplitudes, so a cap at C = 20 can
    only bind on a force two orders of magnitude out of equilibrium -- which is
    exactly what a rigidly rotated peptide seed carries before it relaxes.  Off
    by default; the alkanes never need it.
    """
    g = top.grad(q)
    minv = (1.0 / top.mass).view(-1, 1)
    noise = torch.randn(q.shape, device=q.device, dtype=q.dtype)
    amp = ((2.0 * h / beta) ** 0.5) * torch.sqrt(minv)
    drift = -h * minv * g
    if drift_cap is not None:
        drift = torch.clamp(drift, -drift_cap * amp, drift_cap * amp)
    return q + drift + amp * noise


def constrained_step(top, cv: TorsionCV, q, z, h, beta, n_newton: int = 4,
                     bias_force=None, drift_cap=None):
    """One projected Euler-Maruyama step on Sigma(z) followed by SHAKE."""
    g = top.grad(q)
    if bias_force is not None:
        g = g + bias_force
    minv = (1.0 / top.mass).view(-1, 1)
    noise = torch.randn(q.shape, device=q.device, dtype=q.dtype)
    amp = ((2.0 * h / beta) ** 0.5) * torch.sqrt(minv)
    drift = -h * minv * g
    if drift_cap is not None:
        drift = torch.clamp(drift, -drift_cap * amp, drift_cap * amp)
    step = drift + amp * noise
    gs = cv.grad_local(q)
    G = cv.gram_from_grad(gs)
    step = step.clone()
    sup = cv.support
    step[..., sup, :] = cv.tangent_project_local(gs, G, step[..., sup, :])
    qt = q + step
    qp, lam = cv.project(qt, z, n_newton=n_newton, n_outer=1)
    return qp


class Compiled:
    """Lazily torch.compile'd step functions, keyed by call signature."""

    def __init__(self, enable=True, mode="reduce-overhead"):
        self.enable = enable
        self.mode = mode
        self._cache = {}

    def get(self, fn, key):
        if not self.enable:
            return fn
        if key not in self._cache:
            self._cache[key] = torch.compile(fn, mode=self.mode, dynamic=False)
        return self._cache[key]


COMPILER = Compiled(enable=True)
