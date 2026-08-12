"""Multiple-walker ABF for the methane pair in explicit SPC/E water.

Structure follows ``deca.core`` deliberately: same estimator module (``alkanes.interval``), same
``fullSamples`` trust ramp, same soft walls, same diagnostic schema -- so the methane screen is
comparable to the rest of the campaign rather than a new thing that happens to be about methane.

**This module runs the screen, and computes no regime classification.**  ``T_hit``, ``T_est``,
the bias-aware targets and every gate verdict are derived afterwards from the saved ``xi`` trace,
``n_gap`` trace and bias profile.  Keeping them out of the sampler is what makes it impossible to
tune a gate against a result that is still running.

Two guards carried over because the campaign paid for them
----------------------------------------------------------
* **``abf_min_count`` is applied, not merely declared.**  ``mean_force_profile`` guards only
  ``den > EPS``, so a bin holding one sample contributes that single instantaneous local mean
  force as its conditional average; integrating that noise drives the population irreversibly to
  one end.  Amendment 5 Defect 2 -- declared and never read -- cost a retracted deca screen.
  Only the **applied bias** is ramped; the estimate keeps the full mean force.
* **Walkers are not clones.**  ``init_positions`` must already carry independently equilibrated
  solvent (SPEC §6.1); this module does not manufacture a population from one configuration, and
  :func:`assert_distinct_solvent` refuses one that looks cloned.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from alkanes import interval as iv

from .cv import PeriodicDistanceCV, W_from_F, Wprime_from_Fprime
from .dynamics import BAOAB, RigidWaterConstraints, water_molecules

EPS = 1.0e-12

#: The screen runs ABF only.  Selection arms are added after Gate 0/A/B/C license them.
METHODS = ("abf",)


@dataclass
class MethaneSimConfig:
    """Everything frozen by SPEC §2, §6.2 and §6.3."""
    n_walkers: int = 512
    n_steps: int = 400_000                 #: 200 ps at dt = 0.5 fs
    dt: float = 0.0005
    temperature: float = 298.0
    gamma: float = 1.0
    box_nm: float = 2.490832

    R_lo: float = 0.33                     #: evaluation domain, SPEC §2
    R_hi: float = 0.90
    n_grid: int = 115                      #: odd -- no Nyquist row
    wall_lo: float = 0.34
    wall_hi: float = 0.89
    k_wall: float = 20_000.0               #: kJ/mol/nm^2

    abf_bandwidth: float = 0.012           #: selected by the frozen rule of SPEC §6.2
    kde_bandwidth: float = 0.012
    abf_min_count: float = 100.0           #: the fullSamples guard -- APPLIED
    abf_force_clip: float = 4_000.0
    abf_warmup_steps: int = 20_000
    abf_bias_scale: float = 1.0
    estimator_burn_in_steps: int = 0

    save_every: int = 5_000
    xi_trace_every: int = 500
    ngap_every: int = 2_000
    rng_seed: int = 5000
    chunk: int = 128
    dtype: str = "float32"

    @property
    def beta(self):
        from .dynamics import KB_KJ_PER_MOL_K
        return 1.0 / (KB_KJ_PER_MOL_K * self.temperature)


def wall_force(r, sim):
    """Soft harmonic walls, identical on every arm (the R15 convention).

    Returns the *generalised* force along ``xi``; converted to Cartesian by the CV.
    """
    lo = (r < sim.wall_lo).to(r.dtype) * (sim.wall_lo - r)
    hi = (r > sim.wall_hi).to(r.dtype) * (sim.wall_hi - r)
    return sim.k_wall * (lo + hi)


def assert_distinct_solvent(init_positions, methane_index, tol_nm=1e-3):
    """Refuse a population that is many copies of one solvent configuration (SPEC §6.1).

    "Many walkers" that are initially "many clones" contaminates the mechanism test at exactly
    the point it matters, so this is an assertion rather than a note.
    """
    q = np.asarray(init_positions)
    if q.ndim != 3:
        raise ValueError(f"expected (N, sites, 3), got {q.shape}")
    water = np.setdiff1d(np.arange(q.shape[1]), np.asarray(methane_index))
    ref = q[0, water]
    spread = np.abs(q[:, water] - ref[None]).max(axis=(1, 2))
    if float(spread[1:].min()) < tol_nm:
        raise RuntimeError(
            "initial population contains walkers whose solvent is identical to walker 0 "
            f"(min max-deviation {float(spread[1:].min()):.2e} nm). SPEC §6.1 requires each "
            "walker to carry an independently equilibrated solvent environment.")
    return float(spread[1:].min())


def run_screen(engine, sim: MethaneSimConfig, init_positions, topology, device="cuda",
               dtype=torch.float32, verbose=True, progress_every=20_000):
    """One ABF-only ensemble of ``sim.n_walkers`` walkers.

    ``engine`` is a :class:`methane.nonbonded.MethaneNonbonded`; ``init_positions`` is
    ``(n_walkers, n_sites, 3)`` in nm with independently equilibrated solvent.
    Returns profiles, traces and the diagnostics the gate analysis consumes.
    """
    from .observables import n_gap_batch

    N = sim.n_walkers
    beta = sim.beta
    grid, dz = iv.interval_grid(sim.n_grid, sim.R_lo, sim.R_hi, device=device, dtype=dtype)
    K_abf = iv.gaussian_kernel_matrix(grid, sim.abf_bandwidth)
    K_kde = iv.reflected_kernel_matrix(grid, sim.kde_bandwidth, sim.R_lo, sim.R_hi)

    mi = engine.methane_index
    cv = PeriodicDistanceCV(int(mi[0]), int(mi[1]), sim.box_nm)
    oxy = np.flatnonzero((~engine.params["is_methane"]) & (engine.params["epsilon"] > 0))

    from . import system as msys
    cons = RigidWaterConstraints(water_molecules(topology),
                                 [msys.R_OH_NM, msys.R_OH_NM, msys.r_HH_nm()],
                                 engine.params["mass"], device=device, dtype=dtype)
    integ = BAOAB(lambda q: engine.energy_forces(q, chunk=sim.chunk), engine.params["mass"],
                  cons, sim.dt, sim.temperature, sim.gamma, device=device, dtype=dtype)

    gen = torch.Generator(device=device).manual_seed(int(sim.rng_seed))
    q = torch.as_tensor(np.asarray(init_positions), device=device, dtype=dtype).clone()
    if q.shape[0] != N:
        raise ValueError(f"init_positions has {q.shape[0]} walkers, config says {N}")
    v = integ.maxwell_velocities(q, generator=gen)
    _, f = engine.energy_forces(q, chunk=sim.chunk)

    fsum = torch.zeros(sim.n_grid, device=device, dtype=dtype)
    csum = torch.zeros(sim.n_grid, device=device, dtype=dtype)

    diag = {k: [] for k in ("steps", "times", "mean_force", "pmf", "p_hat", "eff_counts",
                            "occupancy", "temperature")}
    xi_trace, xi_steps = [], []
    ngap_trace, ngap_xi, ngap_steps = [], [], []
    wrap_checked = False

    t0 = time.perf_counter()
    for step in range(sim.n_steps + 1):
        f_loc, r, grad_full = cv.local_mean_force(q, f, beta)
        if not wrap_checked:
            if not cv.separation_is_unambiguous(q):
                raise RuntimeError(
                    f"a methane pair sits within 2% of L/2 = {0.5*sim.box_nm:.3f} nm, where the "
                    "minimum-image separation is degenerate and xi is not well defined")
            wrap_checked = True
        f_loc = torch.clamp(f_loc, -8.0 * sim.abf_force_clip, 8.0 * sim.abf_force_clip)

        if step >= sim.estimator_burn_in_steps:
            fsum += iv.bin_sum(r, f_loc, sim.n_grid, sim.R_lo, sim.R_hi)
            csum += iv.bin_counts(r, sim.n_grid, sim.R_lo, sim.R_hi)

        mf_profile = iv.mean_force_profile(fsum, csum, K_abf)
        eff = iv.effective_counts(csum, K_abf)
        trust = (eff / max(sim.abf_min_count, EPS)).clamp(0.0, 1.0)
        mf_bias_profile = mf_profile * trust

        ramp = min(1.0, step / max(sim.abf_warmup_steps, 1))
        scale = sim.abf_bias_scale * ramp

        if step % sim.xi_trace_every == 0:
            xi_trace.append(r.to(torch.float32).cpu().numpy())
            xi_steps.append(step)
        if step % sim.ngap_every == 0:
            ngap_trace.append(n_gap_batch(q.detach().cpu().numpy().astype(np.float64),
                                          mi, oxy, sim.box_nm).astype(np.float32))
            ngap_xi.append(r.to(torch.float32).cpu().numpy())
            ngap_steps.append(step)
        if step % sim.save_every == 0 or step == sim.n_steps:
            A_hat = iv.free_energy_from_mean_force(mf_bias_profile, grid, dz)
            p_grid = iv.kde_marginal(r.unsqueeze(0), K_kde, sim.n_grid, dz,
                                     sim.R_lo, sim.R_hi).squeeze(0)
            diag["steps"].append(step)
            diag["times"].append(step * sim.dt)
            diag["mean_force"].append(mf_profile.detach().cpu().numpy())
            diag["pmf"].append(A_hat.detach().cpu().numpy())
            diag["p_hat"].append(p_grid.detach().cpu().numpy())
            diag["eff_counts"].append(eff.detach().cpu().numpy())
            diag["occupancy"].append(
                iv.bin_counts(r, sim.n_grid, sim.R_lo, sim.R_hi).detach().cpu().numpy())
            diag["temperature"].append(float(integ.temperature(v).mean()))
            if verbose and step % progress_every == 0:
                el = time.perf_counter() - t0
                print(f"  step {step:7d}/{sim.n_steps}  t = {step*sim.dt:7.2f} ps  "
                      f"T = {diag['temperature'][-1]:6.1f} K  "
                      f"r in [{float(r.min()):.3f}, {float(r.max()):.3f}]  "
                      f"({el:6.0f}s)", flush=True)

        if step == sim.n_steps:
            break

        # The bias must be re-evaluated at the *new* positions inside the step -- BAOAB adds it
        # to the force it computes after the A-O-A block.  Passing the pre-step `bias` tensor
        # would apply a one-step-stale force, which is a small, smooth, entirely plausible error
        # that shows up only as a slightly wrong PMF.  The estimator profile is held fixed within
        # the step (it is only updated once per step, above), so this is a pure position update.
        def _bias_at(q_new, _prof=mf_bias_profile, _scale=scale):
            r_new, grad_new, _ = cv.geometry(q_new)
            mf_new = iv.interval_interp(_prof, grid, r_new).clamp(-sim.abf_force_clip,
                                                                  sim.abf_force_clip)
            return cv.bias_force(grad_new, _scale * mf_new + wall_force(r_new, sim))

        _, f = integ.step(q, v, f, bias_fn=_bias_at, generator=gen)

    mf_profile = iv.mean_force_profile(fsum, csum, K_abf)
    eff = iv.effective_counts(csum, K_abf)
    trust = (eff / max(sim.abf_min_count, EPS)).clamp(0.0, 1.0)
    A_hat = iv.free_energy_from_mean_force(mf_profile * trust, grid, dz)

    out = dict(
        grid=grid.cpu().numpy(), dz=dz,
        mean_force=mf_profile.detach().cpu().numpy(),
        pmf=A_hat.detach().cpu().numpy(),
        W_pmf=W_from_F(A_hat, grid, beta).detach().cpu().numpy(),
        W_mean_force=Wprime_from_Fprime(mf_profile, grid, beta).detach().cpu().numpy(),
        eff_counts=eff.detach().cpu().numpy(),
        xi_trace=np.asarray(xi_trace), xi_steps=np.asarray(xi_steps),
        ngap_trace=np.asarray(ngap_trace), ngap_xi=np.asarray(ngap_xi),
        ngap_steps=np.asarray(ngap_steps),
        wall_seconds=time.perf_counter() - t0,
    )
    for k, val in diag.items():
        out[f"diag_{k}"] = np.asarray(val)
    return out
