"""Wasserstein (horizontal OT) lift + constrained gate repair for ethane in flexible ZIF-8 --
the operators of docs/ZIF8_OT_REPAIR.md (Z1-Z3 mechanism study; Z5 arms later).

ADDITIVE: nothing in core_zif8 changes.  The CV is linear in the guest coordinates,
xi = (COM_guest - win_center) . n  (phi = wrap(2 pi xi / L)), so

  * LIFT   translate the WHOLE ethane by (xi' - xi) n: bond, orientation, every framework
           atom and every velocity untouched -- the guest sits at xi' while the gate is still
           adapted to xi (the ZIF-8 analogue of the WCA solvent-shell carry-over).
  * REPAIR constrained BAOAB at fixed xi': one full BAOAB step of the outer dynamics (physical
           force only, framework-COM pinned, same dt / gamma / T), then the guest COM is
           re-projected along n and the guest COM velocity component along n removed.  With a
           linear constraint the constraint manifold is a hyperplane, |grad xi| is constant, and
           the projected measure is the exact conditional p(. | xi = xi') with the tangent-space
           Maxwellian; every framework atom, the six-ring aperture, the guest orientation and
           bond can move.  Nothing is deposited; every inner step costs one force evaluation.

Units: A, ps, kJ/mol.  ``f_xi`` = k_phi * f_loc(phi) is the local mean force in kJ/mol/A whose
conditional mean at fixed xi is dF/dxi.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from zif8.core_zif8 import ZIF8System, ZIF8SimConfig


def guest_com_shift(system: ZIF8System, q, xi_target):
    """Return q with the guest translated along n so that xi(q) == xi_target (exact, linear CV)."""
    d = (xi_target - system.xi_value(q))[:, None] * system.normal[None, :]      # (B, 3)
    out = q.clone()
    out[:, system.n_frame:] = q[:, system.n_frame:] + d[:, None, :]
    return out


def lift_guest(system: ZIF8System, q, xi_new):
    """The OT lift: whole-ethane translation to xi_new; framework and velocities untouched."""
    return guest_com_shift(system, q, xi_new)


def project_guest_velocity(system: ZIF8System, v):
    """Remove the guest mass-weighted COM velocity component along n (tangent-space Maxwellian)."""
    vg = v[:, system.n_frame:]
    vcom_n = ((vg * system.mass_w[None, :, None]).sum(1) * system.normal[None, :]).sum(-1)   # (B,)
    out = v.clone()
    out[:, system.n_frame:] = vg - vcom_n[:, None, None] * system.normal[None, None, :]
    return out


class ConstrainedBAOAB:
    """Fixed-xi constrained BAOAB for a batch of replicas, each with its own xi_fixed."""

    def __init__(self, system: ZIF8System, sim: ZIF8SimConfig, gen):
        self.s, self.sim, self.gen = system, sim, gen
        self.m = system.mass[None, :, None]
        self.c1 = math.exp(-sim.gamma * sim.dt)
        self.c2 = math.sqrt(1.0 - self.c1 * self.c1)
        self.vsig = torch.sqrt(system.kT / system.mass)[None, :, None]

    def step(self, q, v, F, xi_fixed):
        """One BAOAB step + projection.  ``F`` is the force at ``q``; returns (q, v, F_new)."""
        s, dt = self.s, self.sim.dt
        v = v + (0.5 * dt) * F / self.m
        q = q + (0.5 * dt) * v
        noise = torch.randn(q.shape, generator=self.gen, device=q.device, dtype=q.dtype)
        v = s.pin_frame_com(self.c1 * v + self.c2 * self.vsig * noise)
        q = q + (0.5 * dt) * v
        q = guest_com_shift(s, q, xi_fixed)                 # position projection (exact)
        F = s.forces(q)
        v = v + (0.5 * dt) * F / self.m
        v = project_guest_velocity(s, v)                    # velocity projection
        return q, v, F

    def run(self, q, v, xi_fixed, n_steps, F=None, xi_schedule=None, record=None, record_every=1):
        """Advance ``n_steps``.  ``xi_schedule(k)`` -> per-replica target at inner step k (pulling);
        ``record(k, q, v, F)`` is called before step k when ``k % record_every == 0``."""
        s = self.s
        if F is None:
            F = s.forces(q)
        for k in range(n_steps):
            if record is not None and k % record_every == 0:
                record(k, q, v, F)
            xt = xi_fixed if xi_schedule is None else xi_schedule(k)
            q, v, F = self.step(q, v, F, xt)
        return q, v, F


def local_mean_force_xi(system: ZIF8System, q, F, clip_A=None):
    """Estimator's local mean force in kJ/mol/A (the sampler clips at 8 x abf_force_clip)."""
    f_phi, _ = system.cv_local_mean_force(q, F)
    f_xi = f_phi * system.k_phi
    if clip_A is not None:
        f_xi = f_xi.clamp(-clip_A, clip_A)
    return f_xi


def reference_mean_force(ref):
    """F'_ref(xi) in kJ/mol/A on the reference grid by periodic central differences, plus a
    linear interpolator on the circle."""
    xi = np.asarray(ref["xi_grid"], float); F = np.asarray(ref["F"], float)
    dxi = xi[1] - xi[0]
    Fp = (np.roll(F, -1) - np.roll(F, 1)) / (2.0 * dxi)
    L = float(ref["period"])

    def interp(x):
        x = np.asarray(x, float)
        xw = (x - xi[0]) % L + xi[0]
        return np.interp(xw, np.concatenate([xi, [xi[0] + L]]), np.concatenate([Fp, [Fp[0]]]))
    return xi, Fp, interp


def gate_pdf(a_gate, edges):
    """Normalised histogram of A_gate samples on ``edges`` (out-of-range dropped, count returned)."""
    a = np.asarray(a_gate, float).ravel()
    h, _ = np.histogram(a, bins=edges)
    return h / max(h.sum(), 1), int(a.size - h.sum())


def tv(p, q):
    return 0.5 * float(np.abs(p - q).sum())


def integrated_autocorr(x, dt, max_lag=None):
    """Integrated autocorrelation time of a (T, B) series (mean over replicas of the ACF), by
    the initial-positive-sequence rule; returns (tau, acf)."""
    x = np.asarray(x, float); x = x - x.mean(0, keepdims=True)
    T = x.shape[0]; max_lag = max_lag or T // 2
    var = (x * x).mean()
    if var <= 0:
        return float("nan"), np.zeros(1)
    acf = np.array([(x[: T - k] * x[k:]).mean() / var for k in range(max_lag)])
    cut = np.nonzero(acf <= 0)[0]
    kc = int(cut[0]) if cut.size else max_lag
    return float(dt * (0.5 + acf[1:kc].sum())), acf
