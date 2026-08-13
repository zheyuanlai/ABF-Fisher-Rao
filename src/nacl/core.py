"""Frozen configuration and shared helpers for the NaCl ABF screen (SPEC §7).

**This module deliberately contains no sampler loop.**  ``scripts/nacl_screen.py`` owns the one
loop, because the screen packs all four fixed-compute cells into a single batch and a
single-cell version would be a second implementation of the same physics -- the exact shape
that lets a fix land in one copy and silently miss the other.  An earlier ``run_screen_cell``
lived here, was called by nothing, and was removed for that reason.

The frozen choices these helpers encode:

* **All 8 ensemble seeds of a cell run as ONE batch in ONE process** (`(S, N)` walkers flat as
  ``B = S*N``), each ensemble with its own ABF estimator and bias.  A cell at ``N = 8`` walkers
  would otherwise run the GPU at ~2 % occupancy, and the WCA determinism trap requires whole
  blocks in one process anyway.  The estimator arrays carry a leading ensemble axis, which
  ``alkanes.interval`` supports natively.  Ensemble labels are the preregistered seeds
  4000-4007; noise is one master stream (recorded in the manifest -- ensembles are independent
  by construction, and only within-process contrasts are quotable, per the WCA finding).
* **The Colvars ``fullSamples`` ramp, exactly**: applied bias factor 0 below
  ``fullSamples/2`` effective samples, linear to 1 at ``fullSamples`` (the published
  ``fullSamples 500``).  The estimate keeps the full mean force; only the applied bias ramps.
  No global warmup ramp -- the published protocol has none.
* **Out-of-domain samples are DROPPED, not clamped.**  The published walls sit exactly at the
  domain edges, so excursions past them do happen; ``interval.bin_counts`` would clamp them
  into the edge bins and pollute the boundary mean force.  Masked accumulation instead.
* **Walls as published**: harmonic at 0.20 / R_hi nm with k = 41 840 kJ/mol/nm^2 (1 kcal/mol
  over one 0.1-A colvar width squared).
* **Minimum-image unambiguity** is asserted at every diagnostic save with margin 0.995
  (domain top sits at ~97 % of L/2; SPEC §1.3).

`T_hit`, `T_est`, bias-aware targets and every gate verdict are computed AFTERWARDS from the
saved traces, never in the sampler.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from alkanes import interval as iv
from methane.cv import PeriodicDistanceCV, W_from_F, Wprime_from_Fprime
from methane.dynamics import BAOAB, RigidWaterConstraints

from . import system as nsys
from .observables import HydrationDescriptors

EPS = 1.0e-12


@dataclass
class NaClSimConfig:
    """One fixed-compute cell: S ensembles x N walkers x T = 100 ns / N per ensemble."""
    n_ensembles: int = 8
    n_walkers: int = 64                    #: per ensemble
    n_steps: int = 781_250                 #: T/dt; set by the driver from the cell
    dt: float = 0.002                      #: from the dynamics gate
    temperature: float = nsys.TEMPERATURE_K
    gamma: float = nsys.GAMMA_PS
    box_nm: float = 2.89                   #: frozen L, set by the driver

    R_lo: float = nsys.R_LO_NM
    R_hi: float = nsys.R_HI_NM             #: possibly truncated; set by the driver
    n_grid: int = nsys.N_GRID
    wall_lo: float = nsys.WALL_LO_NM
    wall_hi: float = nsys.WALL_HI_NM
    k_wall: float = nsys.K_WALL_KJ_NM2

    abf_bandwidth: float = 0.012
    kde_bandwidth: float = 0.012
    full_samples: float = float(nsys.FULL_SAMPLES)
    abf_force_clip: float = 4_000.0
    abf_bias_scale: float = 1.0

    save_every: int = 5_000                #: 10 ps at 2 fs
    xi_trace_every: int = 250              #: 0.5 ps
    y_trace_every: int = 500               #: 1 ps
    rng_seed: int = 74000
    chunk: int = 256
    seed_labels: tuple = tuple(range(4000, 4008))

    @property
    def beta(self):
        from methane.dynamics import KB_KJ_PER_MOL_K
        return 1.0 / (KB_KJ_PER_MOL_K * self.temperature)


def wall_force(r, sim):
    lo = (r < sim.wall_lo).to(r.dtype) * (sim.wall_lo - r)
    hi = (r > sim.wall_hi).to(r.dtype) * (sim.wall_hi - r)
    return sim.k_wall * (lo + hi)


def colvars_trust(eff, full_samples):
    """The Colvars ABF ramp: 0 below fullSamples/2, linear to 1 at fullSamples."""
    half = 0.5 * full_samples
    return ((eff - half) / max(half, EPS)).clamp(0.0, 1.0)


def masked_bin_sum(r, values, mask, n_grid, lo, hi):
    """bin_sum with out-of-domain samples contributing nothing (not clamped)."""
    return iv.bin_sum(r, values * mask, n_grid, lo, hi)


def assert_distinct_solvent(init_positions, ion_index=(0, 1), tol_nm=1e-3):
    q = np.asarray(init_positions)
    water = np.setdiff1d(np.arange(q.shape[1]), np.asarray(ion_index))
    ref = q[0, water]
    spread = np.abs(q[:, water] - ref[None]).max(axis=(1, 2))
    if float(spread[1:].min()) < tol_nm:
        raise RuntimeError("initial population contains cloned solvent environments "
                           f"(min max-deviation {float(spread[1:].min()):.2e} nm)")
    return float(spread[1:].min())
