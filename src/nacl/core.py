"""Multiple-walker ABF for NaCl in water -- the fixed-compute screen sampler (SPEC §7).

Structure follows ``methane.core`` deliberately; the differences are NaCl-frozen and listed:

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


def run_screen_cell(engine, sim: NaClSimConfig, init_positions, device="cuda",
                    dtype=torch.float32, verbose=True, progress_every=20_000):
    """One fixed-compute cell: ``sim.n_ensembles`` independent ABF ensembles, one batch.

    ``init_positions`` is ``(S, N, n_sites, 3)`` with independently equilibrated solvent.
    Returns per-ensemble profiles and traces.
    """
    S, N = sim.n_ensembles, sim.n_walkers
    B = S * N
    beta = sim.beta
    grid, dz = iv.interval_grid(sim.n_grid, sim.R_lo, sim.R_hi, device=device, dtype=dtype)
    K_abf = iv.gaussian_kernel_matrix(grid, sim.abf_bandwidth)
    K_kde = iv.reflected_kernel_matrix(grid, sim.kde_bandwidth, sim.R_lo, sim.R_hi)

    cv = PeriodicDistanceCV(0, 1, sim.box_nm)
    hyd = HydrationDescriptors(engine.params["waters"], sim.box_nm, device=device)
    cons = RigidWaterConstraints(engine.params["waters"], nsys.rigid_water_lengths(),
                                 engine.params["mass"], device=device, dtype=dtype)
    integ = BAOAB(lambda q: engine.energy_forces(q, chunk=sim.chunk), engine.params["mass"],
                  cons, sim.dt, sim.temperature, sim.gamma, device=device, dtype=dtype)

    gen = torch.Generator(device=device).manual_seed(int(sim.rng_seed))
    q = torch.as_tensor(np.asarray(init_positions), device=device,
                        dtype=dtype).reshape(B, -1, 3).clone()
    v = integ.maxwell_velocities(q, generator=gen)
    _, f = engine.energy_forces(q, chunk=sim.chunk)

    fsum = torch.zeros(S, sim.n_grid, device=device, dtype=dtype)
    csum = torch.zeros(S, sim.n_grid, device=device, dtype=dtype)

    diag = {k: [] for k in ("steps", "times", "mean_force", "pmf", "p_hat", "eff_counts",
                            "occupancy", "temperature")}
    xi_trace, xi_steps = [], []
    y_trace, y_xi, y_steps = [], [], []

    t0 = time.perf_counter()
    for step in range(sim.n_steps + 1):
        f_loc, r_flat, grad_full = cv.local_mean_force(q, f, beta)
        f_loc = torch.clamp(f_loc, -8.0 * sim.abf_force_clip, 8.0 * sim.abf_force_clip)
        r = r_flat.view(S, N)

        in_dom = ((r >= sim.R_lo) & (r <= sim.R_hi)).to(dtype)
        fsum += masked_bin_sum(r, f_loc.view(S, N), in_dom, sim.n_grid, sim.R_lo, sim.R_hi)
        csum += masked_bin_sum(r, torch.ones_like(r), in_dom, sim.n_grid, sim.R_lo, sim.R_hi)

        mf_profile = iv.mean_force_profile(fsum, csum, K_abf)          # (S, n_grid)
        eff = iv.effective_counts(csum, K_abf)
        mf_bias_profile = mf_profile * colvars_trust(eff, sim.full_samples)

        if step % sim.xi_trace_every == 0:
            xi_trace.append(r.to(torch.float32).cpu().numpy())
            xi_steps.append(step)
        if step % sim.y_trace_every == 0:
            y_trace.append(hyd.Y(q).to(torch.float32).cpu().numpy())
            y_xi.append(r_flat.to(torch.float32).cpu().numpy())
            y_steps.append(step)
        if step % sim.save_every == 0 or step == sim.n_steps:
            if not bool((r_flat < 0.995 * 0.5 * sim.box_nm).all()):
                raise RuntimeError("an ion pair reached 99.5% of L/2; xi is degenerate there "
                                   f"(r_max = {float(r_flat.max()):.4f} nm)")
            A_hat = iv.free_energy_from_mean_force(mf_bias_profile, grid, dz)
            p_grid = iv.kde_marginal(r, K_kde, sim.n_grid, dz, sim.R_lo, sim.R_hi)
            diag["steps"].append(step)
            diag["times"].append(step * sim.dt)
            diag["mean_force"].append(mf_profile.detach().cpu().numpy())
            diag["pmf"].append(A_hat.detach().cpu().numpy())
            diag["p_hat"].append(p_grid.detach().cpu().numpy())
            diag["eff_counts"].append(eff.detach().cpu().numpy())
            diag["occupancy"].append(iv.bin_counts(r, sim.n_grid, sim.R_lo, sim.R_hi)
                                     .detach().cpu().numpy())
            diag["temperature"].append(float(integ.temperature(v).mean()))
            if verbose and step % progress_every == 0:
                el = time.perf_counter() - t0
                print(f"  step {step:8d}/{sim.n_steps}  t = {step*sim.dt:9.2f} ps  "
                      f"T = {diag['temperature'][-1]:6.1f} K  "
                      f"r in [{float(r.min()):.3f}, {float(r.max()):.3f}]  "
                      f"({el:7.0f}s)", flush=True)

        if step == sim.n_steps:
            break

        def _bias_at(q_new, _prof=mf_bias_profile, _S=S, _N=N):
            r_new, grad_new, _ = cv.geometry(q_new)
            mf_new = iv.interval_interp(_prof, grid, r_new.view(_S, _N)) \
                .clamp(-sim.abf_force_clip, sim.abf_force_clip)
            g_tot = sim.abf_bias_scale * mf_new.reshape(-1) + wall_force(r_new, sim)
            return cv.bias_force(grad_new, g_tot)

        _, f = integ.step(q, v, f, bias_fn=_bias_at, generator=gen)

    mf_profile = iv.mean_force_profile(fsum, csum, K_abf)
    eff = iv.effective_counts(csum, K_abf)
    A_hat = iv.free_energy_from_mean_force(mf_profile * colvars_trust(eff, sim.full_samples),
                                           grid, dz)
    grid64 = grid.to(torch.float64)
    out = dict(
        grid=grid.cpu().numpy(), dz=dz,
        seed_labels=np.asarray(sim.seed_labels[:S]),
        mean_force=mf_profile.detach().cpu().numpy(),
        pmf=A_hat.detach().cpu().numpy(),
        W_pmf=W_from_F(A_hat.to(torch.float64), grid64, beta).cpu().numpy(),
        W_mean_force=Wprime_from_Fprime(mf_profile.to(torch.float64), grid64, beta)
        .cpu().numpy(),
        eff_counts=eff.detach().cpu().numpy(),
        xi_trace=np.asarray(xi_trace), xi_steps=np.asarray(xi_steps),
        y_trace=np.asarray(y_trace), y_xi=np.asarray(y_xi), y_steps=np.asarray(y_steps),
        wall_seconds=time.perf_counter() - t0,
    )
    for k, val in diag.items():
        out[f"diag_{k}"] = np.asarray(val)
    return out
