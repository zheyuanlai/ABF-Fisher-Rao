"""Wasserstein (horizontal OT) reallocation along the WCA dimer coordinate + constrained fibre
repair -- the ``ot=`` option of :func:`wca_abffr_core.run_sampler_gpu`.

ADDITIVE: nothing here runs unless the sampler is handed an :class:`OTConfig`; every accepted
path stays byte-identical.

One OT opportunity (on the FR schedule: ``fr_start_steps`` + k ``fr_every``):

  1. ``z_i = xi(q_i)``; sort; uniform quantiles ``u_i = z_lo + (i - 1/2)(z_hi - z_lo)/N`` on the
     sampler's own z-domain (the same domain the uniform-FR target lives on);
  2. rank-matched displacement interpolation with a per-event cap,
        ``z_i' = z_i + clip(alpha (u_(rank i) - z_i), -dz_max, dz_max)``
     (the 1-D W2-optimal coupling; identity for alpha = 0; ranks preserved; no RNG);
  3. LIFT: ``project_dimer_to_z(q, z')`` -- midpoint and direction kept, bath untouched;
  4. REPAIR (``c_repair > 0``): ``m_i = ceil(c_repair * tau_f(z_i') / dt)`` steps of the
     reference-consistent projected constrained scheme (``frozen_dimer_relax(scheme=
     'projected')``: all particles move, dimer re-projected to z_i' each step), from its own
     generator; nothing is deposited during repair; every inner step is charged.

Deposit-free diagnostics, binned on the grid by destination z: the estimator's local-mean-
force sample at the lifted state (``pre``) and after repair (``post``), so the analyzer can
form the injected conditional bias ``<f_pre | z> - F'_ref(z)`` and its post-repair residual
without the sampler ever reading the reference.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class OTConfig:
    alpha: float                    # displacement-interpolation fraction toward the uniform quantiles
    dz_max: float                   # per-event displacement cap in z units (2 grid bins = 0.0176)
    c_repair: float                 # repair duration = c_repair * tau_f(z'); 0 => T0 (no repair)
    tau_grid: tuple                 # tau_f on the sampler grid (the W0 map, floored/capped)
    scheme: str = "projected"       # inner scheme; 'projected' = the TI reference's own operator
    inner_seed_offset: int = 8_000_000
    min_move: float = 1.0e-9        # walkers moved by less than this are not repaired


def uniform_quantiles(n, z_lo, z_hi, device, dtype):
    i = torch.arange(n, device=device, dtype=dtype)
    return z_lo + (i + 0.5) * (z_hi - z_lo) / float(n)


def ot_displacement(z, alpha, dz_max, u):
    """``z' = z + clip(alpha (u_rank - z), +-dz_max)`` with each walker keeping its identity.
    ``z`` (N,), ``u`` (N,) sorted.  Deterministic; alpha = 0 returns ``z`` exactly."""
    zs, order = torch.sort(z, stable=True)
    step = torch.clamp(alpha * (u - zs), -dz_max, dz_max)
    return torch.empty_like(z).scatter_(0, order, zs + step)


def ot_lift(q, params, sim, ot, u, core):
    """One OT event on the whole population.  Returns ``(q_lifted, z_old, z_new)``."""
    z = core.reaction_coordinate(q, params)
    z_new = ot_displacement(z, float(ot.alpha), float(ot.dz_max), u)
    return core.project_dimer_to_z(q, z_new, params), z, z_new
