"""Batched ABF (+ optional Fisher--Rao birth--death) sampler for deca-alanine.

One process, one GPU, leading dimensions ``(seed, walker, atom, 3)``.  Every arm of a stage
shares one batch, one noise stream and one force-evaluation budget, which is what makes the
per-seed comparisons paired rather than merely matched.

What this module deliberately does NOT do
-----------------------------------------
It never sees the reference free energy unless the arm is explicitly an oracle
(:func:`deca.selection.assert_no_reference_leakage` is a structural gate, not a convention),
and it computes **no** regime classification.  Everything reference-dependent is computed in
analysis from the saved ``xi`` trace, the saved counts and the saved bias profile.

Relationship to the rest of the repository
------------------------------------------
The interval grid, reflected KDE and kernel estimator come unchanged from
:mod:`alkanes.interval`; the CV geometry and generalized mean force from
:mod:`alkanes.distance_cv`; BAOAB and full-state birth--death from :mod:`alanine.dynamics`;
the online FR stopping rule from :mod:`eri.fr_controller` (engine-agnostic, time in ps here).

2026-10 revision (``docs/DECA_FR_2026_PROTOCOL.md``)
----------------------------------------------------
* **Bias-force consistency.**  The two BAOAB half-kicks now evaluate the *same* frozen,
  trust-weighted mean-force field (``mf_bias_profile``).  The 2026-08 code applied the
  trust ramp in the first half-kick only; the second half-kick interpolated the unweighted
  profile, so every sparse bin still received half of an unguarded bias.  The old behaviour is
  reproducible with ``legacy_unweighted_second_kick=True`` for the historical replay and for
  nothing else.
* **Histogram estimator** (``abf_estimator="histogram"``): per-cell sums, piecewise-constant
  bias (own cell, no interpolation, no smoothing), trust ``min(1, C_j / C_min)``.
* **Wall-corrected uniform FR target** (method ``fr_uniform``):
  ``q(z) proportional to exp(-beta V_wall(z))`` -- the limiting marginal of a converged ABF run
  under the soft walls.  Uses no estimate of F and no reference.
* **Online FR stop** (``fr_controller``): the R2 occupancy rule of ``eri.fr_controller``, one
  controller per seed, fed only cumulative counts, birth/death histograms and lineage shares.
* **Clone deposits**: a freshly cloned walker does not deposit an ABF sample until it has been
  propagated once (``clone_deposit_skip``).
* Event log and partial saves for long runs.

Full-state cloning is used, not position-only: a birth copies position, cached force and
genealogy label and draws a fresh Maxwell momentum.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
import torch

from alanine.dynamics import BAOAB, birth_death_full_state, check_finite, make_seed_streams
from alkanes import interval as iv
from alkanes.distance_cv import DistanceCV, dist_bias_force

from . import system as dsys
from .labels import DecaLabels
from .selection import (METHODS, SELECTION_METHODS, assert_no_reference_leakage,
                        book_laplacian_score, count_balancing_score, fisher_rao_score,
                        sham_score)

EPS = 1.0e-12
KB = 0.008314462618          # kJ/mol/K
ESTIMATORS = ("kernel", "histogram")
TARGET_METHODS = ("mfr_practical", "mfr_oracle", "mfr_sham", "fr_uniform")


@dataclass
class DecaSimConfig:
    """Frozen physical model (identical to the 2026-08 study); estimator and FR options added.

    The default budget is still the *historical* one -- 16 walkers x 0.5 ns -- so that
    ``DecaSimConfig()`` with ``legacy_unweighted_second_kick=True`` reproduces the closed
    screen exactly.
    """
    # --- dynamics (frozen physical model) ---
    dt: float = 0.001                      # ps
    gamma: float = 1.0                     # ps^-1
    temperature: float = 300.0             # K
    n_walkers: int = 16                    # K
    n_steps: int = 500_000                 # 0.5 ns per walker
    rng_seed: int = 20260811

    # --- CV interval and estimator ---
    R_lo: float = 1.20
    R_hi: float = 3.60
    wall_lo: float = 1.25
    wall_hi: float = 3.55
    k_wall: float = 4000.0                 # kJ/mol/nm^2
    n_grid: int = 129                      # ODD: no Nyquist row exists
    abf_estimator: str = "kernel"          # "kernel" | "histogram"
    abf_bandwidth: float = 0.04            # nm (kernel estimator only)
    kde_bandwidth: float = 0.06            # nm (FR marginal KDE)
    abf_min_count: float = 100.0           # trust threshold C_min (effective counts)

    # --- ABF application ---
    abf_bias_scale: float = 1.0
    abf_warmup_steps: int = 10_000
    abf_force_clip: float = 2000.0         # kJ/mol/nm -- a GUARD, not a regulariser
    estimator_burn_in_steps: int = 0
    legacy_unweighted_second_kick: bool = False   # replay of the 2026-08 defect ONLY

    # --- Fisher--Rao selection ---
    fr_rate: float = 0.0                   # ps^-1 (p_die = 1 - exp(-rate |S| dt_FR))
    score_clip: float = 2.0
    fr_start_steps: int = 50_000           # 50 ps
    fr_every: int = 500                    # 0.5 ps
    fr_stop_steps: int = -1                # fixed stop (exclusive); -1 = to the horizon
    fr_controller: Optional[dict] = None   # online stop rule spec (eri.fr_controller), or None
    fr_window: tuple = (2.0, 2.8)          # nm, monitoring region of the occupancy rule
    fr_support_interval: tuple = (1.30, 3.50)   # nm, bins the rule requires support on
    clone_deposit_skip: bool = True
    target_ema_rate: float = 0.005         # legacy estimated-target arms only
    max_event_fraction: float = 0.05
    c_book: float = 1.0
    c_count: float = 1.0

    # --- diagnostics ---
    save_every: int = 5_000
    xi_trace_every: int = 100
    label_every: int = 5_000
    partial_save_every: int = 0            # steps; 0 = never

    @property
    def beta(self):
        return 1.0 / (KB * self.temperature)

    def config_hash(self):
        import hashlib, json
        return hashlib.md5(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]

    def validate(self):
        if self.abf_estimator not in ESTIMATORS:
            raise ValueError(f"abf_estimator must be one of {ESTIMATORS}")
        if self.legacy_unweighted_second_kick and self.abf_estimator != "kernel":
            raise ValueError("the legacy replay flag only applies to the kernel estimator")
        if self.fr_controller is not None and self.fr_stop_steps >= 0:
            raise ValueError("fr_controller and a fixed fr_stop_steps are exclusive")


# --------------------------------------------------------------------------- pieces
def _wall_gforce(R, sim):
    """Generalized force along R from the soft harmonic walls (``-dV_wall/dR``)."""
    f = torch.zeros_like(R)
    f = torch.where(R > sim.wall_hi, -sim.k_wall * (R - sim.wall_hi), f)
    f = torch.where(R < sim.wall_lo, -sim.k_wall * (R - sim.wall_lo), f)
    return f


def wall_potential(grid, sim):
    """``V_wall(z)`` on the grid, kJ/mol."""
    lo = (grid - sim.wall_lo).clamp(max=0.0)
    hi = (grid - sim.wall_hi).clamp(min=0.0)
    return 0.5 * sim.k_wall * (lo * lo + hi * hi)


def wall_uniform_target(grid, dz, sim):
    """``q(z) proportional to exp(-beta V_wall(z))``, normalised; shape ``(n_grid,)``.

    Constant on ``[wall_lo, wall_hi]`` and decaying smoothly into the wall regions: the
    marginal a perfectly flattened ABF run would have.  Reference-free by construction.
    """
    return iv.normalize_density(torch.exp(-sim.beta * wall_potential(grid, sim))[None], dz)[0]


def free_energy_from_histogram_force(mf, dz):
    """``F`` at the cell centres from a piecewise-constant mean force, centred on its mean.

    ``F(e_{j+1}) = F(e_j) + Gamma_j dz`` with ``F(e_0) = 0``; the centre value of cell ``j`` is
    ``F(e_j) + Gamma_j dz / 2``.  Works with a leading batch dimension.
    """
    right = torch.cumsum(mf * dz, dim=-1)
    F = right - 0.5 * mf * dz
    return F - F.mean(-1, keepdim=True)


def _fr_target(method, grid, dz, F_ema, B_n, oracle, beta, q_wall):
    """The target marginal ``q`` ``(R, n_grid)`` for the Fisher--Rao family; ``None`` otherwise."""
    if method not in TARGET_METHODS:
        return None
    if method == "fr_uniform":
        return q_wall.expand(B_n.shape[0], -1)
    if method == "mfr_oracle":
        log_q = -beta * (oracle[None, :] - B_n)
    else:
        if F_ema is None:
            return None
        log_q = -beta * (F_ema - B_n)
    log_q = log_q - log_q.max(-1, keepdim=True).values
    return iv.normalize_density(torch.exp(log_q), dz)


def _ancestor_stats(anc_np, N):
    """``(ess (R,), n_unique (R,), max_frac (R,))`` from ancestor labels ``(R, N)``."""
    ess, nuq, mxf = [], [], []
    for r in range(anc_np.shape[0]):
        cnt = np.bincount(anc_np[r], minlength=N).astype(float) / N
        ess.append(1.0 / np.sum(cnt ** 2))
        nuq.append(int((cnt > 0).sum()))
        mxf.append(float(cnt.max()))
    return np.array(ess), np.array(nuq), np.array(mxf)


# --------------------------------------------------------------------------- sampler
def run_sampler_deca(method, engine, sim: DecaSimConfig, seeds, init_positions,
                     reference_free_energy=None, device="cuda", dtype=torch.float64,
                     verbose=True, progress_every=100_000, partial_path=None):
    """Run ``R = len(seeds)`` matched-seed ensembles of ``method``.

    ``init_positions`` is ``(R, N, 112, 3)`` in nm, already relaxed and structurally validated.
    Returns a dict of profiles, traces, event log and genealogy diagnostics.  ``partial_path``
    (optional) receives an ``npz`` of everything accumulated so far every
    ``sim.partial_save_every`` steps, so a job killed at the walltime leaves usable data.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; known: {sorted(METHODS)}")
    sim.validate()
    assert_no_reference_leakage(method, reference_free_energy)
    is_sel = method in SELECTION_METHODS
    hist = sim.abf_estimator == "histogram"

    R, N, A = len(seeds), sim.n_walkers, engine.n_atoms
    beta = sim.beta
    grid, dz = iv.interval_grid(sim.n_grid, sim.R_lo, sim.R_hi, device=device, dtype=dtype)
    K_abf = None if hist else iv.gaussian_kernel_matrix(grid, sim.abf_bandwidth)
    K_kde = iv.reflected_kernel_matrix(grid, sim.kde_bandwidth, sim.R_lo, sim.R_hi)
    q_wall = wall_uniform_target(grid, dz, sim)

    i_cv, j_cv = dsys.terminal_carbonyls(dsys.N_RES)
    cv = DistanceCV(i_cv, j_cv)
    labels = DecaLabels(device=device, dtype=dtype)

    gen_dyn = torch.Generator(device=device).manual_seed(int(sim.rng_seed))
    gen_sel = torch.Generator(device=device).manual_seed(int(sim.rng_seed) + 987654321)
    gens = make_seed_streams(sim.rng_seed, R, device)

    oracle = None
    if method == "mfr_oracle":
        oracle = torch.as_tensor(reference_free_energy, device=device, dtype=dtype)
        oracle = oracle - oracle.mean()

    q = torch.as_tensor(np.asarray(init_positions), device=device, dtype=dtype).reshape(R, N, A, 3)
    integ = BAOAB(engine.masses, dt=sim.dt, gamma=sim.gamma, temperature=sim.temperature,
                  force_fn=engine.forces, device=device, dtype=dtype)
    v = integ.maxwell((R, N, A, 3), gen_dyn, device, dtype)
    f = engine.forces(q.reshape(R * N, A, 3)).reshape(R, N, A, 3)

    fsum = torch.zeros(R, sim.n_grid, device=device, dtype=dtype)
    csum = torch.zeros(R, sim.n_grid, device=device, dtype=dtype)
    dep_w = torch.ones(R, N, device=device, dtype=dtype)      # deposit weight (0 right after a clone)
    F_ema = None
    ancestors = torch.arange(N, device=device).expand(R, N).clone()
    total_repl = torch.zeros(R, dtype=torch.long)
    birth_hist = torch.zeros(R, sim.n_grid, device=device, dtype=dtype)
    death_hist = torch.zeros(R, sim.n_grid, device=device, dtype=dtype)
    n_clipped, n_scored = 0, 0
    ev_step, ev_seed, ev_slot, ev_src = [], [], [], []

    # ---- FR stop: fixed, or the online controller (one per seed) ------------------------
    big = 2 ** 62
    fr_stop_step = np.full(R, big, dtype=np.int64)
    if sim.fr_stop_steps >= 0:
        fr_stop_step[:] = int(sim.fr_stop_steps)
    ctls, check_steps, interior_idx = None, None, None
    if is_sel and sim.fr_rate > 0.0 and sim.fr_controller is not None:
        from eri.fr_controller import OnlineStop
        spec = dict(sim.fr_controller)
        g_np = grid.cpu().numpy()
        interior = (g_np >= sim.fr_support_interval[0]) & (g_np <= sim.fr_support_interval[1])
        interior_idx = torch.as_tensor(np.flatnonzero(interior), device=device)
        wmask = ((g_np >= sim.fr_window[0]) & (g_np <= sim.fr_window[1]))[interior]
        check_steps = int(round(spec["check_every"] / sim.dt))
        if abs(check_steps * sim.dt - spec["check_every"]) > 1e-9:
            raise ValueError("fr_controller.check_every must be a multiple of dt")
        if abs(round(spec["W"] / spec["check_every"]) * spec["check_every"] - spec["W"]) > 1e-9:
            raise ValueError("fr_controller.W must be a multiple of check_every")
        if int(interior.sum()) % int(spec.get("R", 1)) != 0:
            raise ValueError("fr_controller.R must divide the number of interior bins")
        ctls = [OnlineStop(spec, N, sim.fr_start_steps * sim.dt, window_mask=wmask)
                for _ in range(R)]

    diag = {k: [] for k in ("steps", "times", "mean_force", "pmf", "bias_pmf", "p_hat", "q_target",
                            "eff_counts", "ancestor_ess", "n_unique_ancestor",
                            "max_ancestor_frac", "repl_cumulative", "pq_l2", "kl_pq", "pq_tv",
                            "fr_active")}
    xi_trace, xi_trace_steps = [], []
    label_trace = {k: [] for k in ("n_hbonds", "alpha_frac", "rg", "ca_rmsd_helix", "y")}
    label_trace_xi, label_steps = [], []
    t0 = time.perf_counter()

    def _assemble(final):
        out = dict(method=method, grid=grid.cpu().numpy(), dz=float(dz),
                   R_lo=sim.R_lo, R_hi=sim.R_hi, seeds=np.asarray(seeds),
                   config=asdict(sim), config_hash=sim.config_hash(),
                   runtime_seconds=time.perf_counter() - t0, complete=bool(final),
                   total_replacement_events=total_repl.numpy().copy(),
                   birth_hist=birth_hist.cpu().numpy(), death_hist=death_hist.cpu().numpy(),
                   final_counts=csum.cpu().numpy(), final_force_sum=fsum.cpu().numpy(),
                   final_eff_counts=(csum if hist else iv.effective_counts(csum, K_abf)).cpu().numpy(),
                   F_target_ema=(F_ema.cpu().numpy() if F_ema is not None else None),
                   q_wall=q_wall.cpu().numpy(),
                   score_clip_fraction=(n_clipped / max(n_scored, 1)),
                   fr_stop_step=fr_stop_step.copy(),
                   ev_step=np.asarray(ev_step, dtype=np.int64), ev_seed=np.asarray(ev_seed, dtype=np.int32),
                   ev_slot=np.asarray(ev_slot, dtype=np.int32), ev_src=np.asarray(ev_src, dtype=np.int32),
                   xi_trace=np.stack(xi_trace), xi_trace_steps=np.asarray(xi_trace_steps),
                   label_steps=np.asarray(label_steps),
                   label_xi=np.stack(label_trace_xi) if label_trace_xi else np.zeros((0, R, N), np.float32),
                   **{f"label_{k}": (np.stack(vv) if vv else np.zeros((0, R, N), np.float32))
                      for k, vv in label_trace.items()})
        for k in diag:
            out[k] = np.asarray(diag[k])
        if ctls is not None:
            out["controller"] = [dict(t1=c.t1, t_stop=c.t_stop, reason=c.reason, log=c.log)
                                 for c in ctls]
        return out

    for step in range(sim.n_steps + 1):
        qf = q.reshape(R * N, A, 3)
        Ff = f.reshape(R * N, A, 3)
        f_loc, R_f, grad_f = cv.local_mean_force(qf, Ff, beta)
        Rv = R_f.reshape(R, N)
        f_loc = torch.clamp(f_loc, -8.0 * sim.abf_force_clip, 8.0 * sim.abf_force_clip).reshape(R, N)
        grad_full = grad_f.reshape(R, N, A, 3)

        if step >= sim.estimator_burn_in_steps:
            fsum += iv.bin_sum(Rv, f_loc * dep_w, sim.n_grid, sim.R_lo, sim.R_hi)
            csum += iv.bin_sum(Rv, dep_w, sim.n_grid, sim.R_lo, sim.R_hi)
        if sim.clone_deposit_skip:
            dep_w.fill_(1.0)

        # --- estimator: the STORED profile keeps the full conditional mean force -------------
        if hist:
            mf_profile = torch.where(csum > 0, fsum / csum.clamp_min(EPS), torch.zeros_like(fsum))
            A_hat = free_energy_from_histogram_force(mf_profile, dz)
            eff = csum
        else:
            mf_profile = iv.mean_force_profile(fsum, csum, K_abf)
            A_hat = iv.free_energy_from_mean_force(mf_profile, grid, dz)
            eff = iv.effective_counts(csum, K_abf)

        # --- per-bin trust ramp on the APPLIED bias (NAMD fullSamples) -----------------------
        trust = (eff / max(sim.abf_min_count, 1e-12)).clamp(0.0, 1.0)
        mf_bias_profile = mf_profile * trust
        ramp = min(1.0, step / max(sim.abf_warmup_steps, 1))
        abf_scale = sim.abf_bias_scale * ramp
        if hist:
            B_n = abf_scale * free_energy_from_histogram_force(mf_bias_profile, dz)
            idx = iv.bin_index(Rv, sim.n_grid, sim.R_lo, sim.R_hi)
            mf_at = torch.gather(mf_bias_profile, -1, idx)
        else:
            B_n = abf_scale * iv.free_energy_from_mean_force(mf_bias_profile, grid, dz)
            mf_at = iv.interval_interp(mf_bias_profile, grid, Rv)
        mf_at = mf_at.clamp(-sim.abf_force_clip, sim.abf_force_clip)
        applied = abf_scale * mf_at + _wall_gforce(Rv, sim)
        bias_force = dist_bias_force(grad_full.reshape(R * N, A, 3),
                                     applied.reshape(R * N)).reshape(R, N, A, 3)

        if method in ("mfr_practical", "mfr_sham") and (step + 1) >= sim.fr_start_steps:
            F_ema = A_hat.clone() if F_ema is None else \
                (1 - sim.target_ema_rate) * F_ema + sim.target_ema_rate * A_hat
            F_ema = F_ema - F_ema.mean(-1, keepdim=True)

        # ---------------------------------------------------------------- diagnostics
        if step % sim.xi_trace_every == 0:
            xi_trace.append(Rv.to(torch.float32).cpu().numpy())
            xi_trace_steps.append(step)
        if step % sim.label_every == 0:
            L = labels.all_labels(qf)
            for k in label_trace:
                label_trace[k].append(L[k].to(torch.float32).cpu().numpy().reshape(R, N))
            label_trace_xi.append(Rv.to(torch.float32).cpu().numpy())
            label_steps.append(step)
        if step % sim.save_every == 0 or step == sim.n_steps:
            p_grid = iv.kde_marginal(Rv, K_kde, sim.n_grid, dz, sim.R_lo, sim.R_hi)
            q_grid = _fr_target(method, grid, dz, F_ema, B_n, oracle, beta, q_wall)
            diag["steps"].append(step)
            diag["times"].append(step * sim.dt)
            diag["mean_force"].append(mf_profile.cpu().numpy())
            diag["pmf"].append(A_hat.cpu().numpy())
            diag["bias_pmf"].append(B_n.cpu().numpy())
            diag["p_hat"].append(p_grid.cpu().numpy())
            diag["eff_counts"].append(eff.cpu().numpy())
            diag["repl_cumulative"].append(total_repl.numpy().copy())
            diag["fr_active"].append((step < fr_stop_step).astype(np.int8)
                                     if (is_sel and sim.fr_rate > 0 and step >= sim.fr_start_steps)
                                     else np.zeros(R, np.int8))
            qg = q_grid if q_grid is not None else q_wall.expand(R, -1)
            lr = torch.log(p_grid.clamp_min(EPS)) - torch.log(qg.clamp_min(EPS))
            diag["q_target"].append(qg.cpu().numpy())
            diag["pq_l2"].append((((p_grid - qg) ** 2).sum(-1) * dz).sqrt().cpu().numpy())
            diag["kl_pq"].append(((p_grid * lr).sum(-1) * dz).cpu().numpy())
            diag["pq_tv"].append(iv.interval_tv(p_grid, qg, dz).cpu().numpy())
            ess, nuq, mxf = _ancestor_stats(ancestors.cpu().numpy(), N)
            diag["ancestor_ess"].append(ess)
            diag["n_unique_ancestor"].append(nuq)
            diag["max_ancestor_frac"].append(mxf)

        if step == sim.n_steps:
            break

        # ---------------------------------------------------------------- integrate (BAOAB)
        v = v + (0.5 * sim.dt) * (f + bias_force) / integ.m
        q = q + (0.5 * sim.dt) * v
        v = integ.c1 * v + integ.c2 * torch.randn(v.shape, generator=gen_dyn, device=device,
                                                  dtype=dtype) * integ.sigma
        q = q + (0.5 * sim.dt) * v
        f = engine.forces(q.reshape(R * N, A, 3)).reshape(R, N, A, 3)
        f_new_loc, R_new, grad_new = cv.local_mean_force(q.reshape(R * N, A, 3),
                                                         f.reshape(R * N, A, 3), beta)
        Rn = R_new.reshape(R, N)
        # Second half-kick: the SAME frozen trust-weighted field, evaluated at the new positions.
        if hist:
            idx2 = iv.bin_index(Rn, sim.n_grid, sim.R_lo, sim.R_hi)
            mf_at2 = torch.gather(mf_bias_profile, -1, idx2)
        elif sim.legacy_unweighted_second_kick:
            mf_at2 = iv.interval_interp(mf_profile, grid, Rn)          # the 2026-08 defect
        else:
            mf_at2 = iv.interval_interp(mf_bias_profile, grid, Rn)
        mf_at2 = mf_at2.clamp(-sim.abf_force_clip, sim.abf_force_clip)
        applied2 = abf_scale * mf_at2 + _wall_gforce(Rn, sim)
        bias2 = dist_bias_force(grad_new.reshape(R * N, A, 3),
                                applied2.reshape(R * N)).reshape(R, N, A, 3)
        v = v + (0.5 * sim.dt) * (f + bias2) / integ.m

        if step % 20_000 == 0:
            check_finite(step, ("q", q), ("v", v), tag=f"deca_{method}")

        # ---------------------------------------------------------------- selection
        nxt = step + 1
        active = nxt < fr_stop_step                                     # (R,) numpy bool
        if (is_sel and sim.fr_rate > 0.0 and nxt >= sim.fr_start_steps and active.any()
                and (nxt - sim.fr_start_steps) % max(int(sim.fr_every), 1) == 0):
            score = None
            if method in TARGET_METHODS:
                q_grid = _fr_target(method, grid, dz, F_ema, B_n, oracle, beta, q_wall)
                if q_grid is not None:
                    score, _, _ = fisher_rao_score(Rn, grid, dz, sim.R_lo, sim.R_hi,
                                                   K_kde, q_grid, sim.score_clip)
                    if method == "mfr_sham":
                        score = sham_score(score, gen_sel)
            elif method == "count_balancing":
                score, _ = count_balancing_score(Rn, grid, dz, sim.R_lo, sim.R_hi, K_kde,
                                                 sim.score_clip, c=sim.c_count)
            elif method == "book_laplacian":
                score, _ = book_laplacian_score(Rn, grid, dz, sim.R_lo, sim.R_hi, K_kde,
                                                sim.score_clip, c=sim.c_book)
            if score is not None:
                if not active.all():
                    score = score * torch.as_tensor(active, device=device, dtype=dtype)[:, None]
                n_clipped += int((score.abs() >= sim.score_clip - 1e-9).sum())
                n_scored += int(active.sum()) * N
                q, v, f, ancestors, n_ev, deaths, births = birth_death_full_state(
                    q, v, f, score, ancestors, gens, sim.fr_rate,
                    sim.fr_every * sim.dt, sim.max_event_fraction, integ)
                total_repl += n_ev
                for r in range(R):
                    if deaths[r] is not None and deaths[r].numel() > 0:
                        zb = Rn[r].index_select(0, births[r])
                        zd = Rn[r].index_select(0, deaths[r])
                        birth_hist[r] += iv.bin_counts(zb[None], sim.n_grid, sim.R_lo, sim.R_hi)[0]
                        death_hist[r] += iv.bin_counts(zd[None], sim.n_grid, sim.R_lo, sim.R_hi)[0]
                        if sim.clone_deposit_skip:
                            dep_w[r, deaths[r]] = 0.0
                        d_np, s_np = deaths[r].cpu().numpy(), births[r].cpu().numpy()
                        ev_step.extend([nxt] * len(d_np))
                        ev_seed.extend([r] * len(d_np))
                        ev_slot.extend(d_np.tolist())
                        ev_src.extend(s_np.tolist())

        # ---------------------------------------------------------------- online stop rule
        if (ctls is not None and nxt >= sim.fr_start_steps
                and (nxt - sim.fr_start_steps) % check_steps == 0):
            anc_np = ancestors.cpu().numpy()
            ess, _, mxf = _ancestor_stats(anc_np, N)
            BH = birth_hist.index_select(1, interior_idx).cpu().numpy()
            DH = death_hist.index_select(1, interior_idx).cpu().numpy()
            C = csum.index_select(1, interior_idx).cpu().numpy()
            M = fsum.index_select(1, interior_idx).cpu().numpy()
            t_now = nxt * sim.dt
            for r in range(R):
                if nxt >= fr_stop_step[r]:
                    continue
                ts = ctls[r].observe(t_now, BH[r], DH[r], C[r], M[r], ess[r] / N, mxf[r])
                if ts is not None:
                    fr_stop_step[r] = min(fr_stop_step[r], int(round(ts / sim.dt)))

        if verbose and progress_every and nxt % progress_every == 0:
            el = time.perf_counter() - t0
            fr = nxt / sim.n_steps
            print(f"    {method:14s} {100*fr:5.1f}%  {el/60:6.1f} min, "
                  f"~{el/fr*(1-fr)/60:6.1f} left  repl={int(total_repl.sum())}  "
                  f"stopped={int((fr_stop_step <= nxt).sum())}/{R}", flush=True)
        if partial_path and sim.partial_save_every and nxt % sim.partial_save_every == 0 \
                and nxt < sim.n_steps:
            part = _assemble(final=False)
            np.savez_compressed(partial_path, **{k: v_ for k, v_ in part.items()
                                                 if isinstance(v_, (np.ndarray, np.generic))},
                                config_json=np.array(__import__("json").dumps(part["config"])))

    out = _assemble(final=True)
    if verbose:
        print(f"  {method:14s} R={R} N={N}: {out['runtime_seconds']/60:.1f} min, "
              f"repl={int(total_repl.sum())}, clip={out['score_clip_fraction']:.4f}", flush=True)
    return out
