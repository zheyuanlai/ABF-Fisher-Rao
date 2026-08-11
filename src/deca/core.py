"""Batched ABF (+ optional directed selection) sampler for deca-alanine.

One process, one GPU, leading dimensions ``(seed, walker, atom, 3)``.  Every arm of a stage
shares one batch, one noise stream and one force-evaluation budget, which is what makes the
per-seed comparisons paired rather than merely matched.

What this module deliberately does NOT do
-----------------------------------------
It never sees the reference free energy unless the arm is explicitly an oracle
(:func:`deca.selection.assert_no_reference_leakage` is a structural gate, not a convention),
and it computes **no** regime classification.  ``T_hit``, the bias-aware target populations
``Q*_k(t)`` and the establishment gate are all reference-dependent, so they are computed in
analysis from the saved ``xi`` trace and the saved bias profile.  Keeping them out of the
sampler is what lets the same code produce the ABF-only screen that decides whether any
selection arm is licensed at all.

Relationship to the rest of the repository
------------------------------------------
The estimator, interval grid, reflected KDE and free-energy integration come unchanged from
:mod:`alkanes.interval`; the CV geometry and generalized mean force from
:mod:`alkanes.distance_cv`; BAOAB and full-state birth--death from :mod:`alanine.dynamics`.
The genuinely new pieces are the selection rules in :mod:`deca.selection` and the structural
labels in :mod:`deca.labels`.

Full-state cloning is used, not position-only.  With BAOAB a position-only clone would leave
the child carrying the *killed* replica's momentum; here a birth copies position, cached force
and genealogy label and draws a **fresh Maxwell momentum**, which is exact (the canonical
density factorises) and decorrelates the child's velocity immediately at no cost.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass

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


@dataclass
class DecaSimConfig:
    """Frozen per §6.2/§6.5 of ``docs/V2_PREREGISTRATION.md``.

    The budget is the *historical* one -- 16 walkers x 0.5 ns = 8 ns aggregate per ensemble --
    not a budget invented around mFR.
    """
    # --- dynamics (frozen physical model, identical to the alanine and valine studies) ---
    dt: float = 0.001                      # ps
    gamma: float = 1.0                     # ps^-1
    temperature: float = 300.0             # K
    n_walkers: int = 16                    # N
    n_steps: int = 500_000                 # 0.5 ns per walker
    rng_seed: int = 20260811

    # --- CV interval and estimator ---
    R_lo: float = 1.20
    R_hi: float = 3.60
    wall_lo: float = 1.25
    wall_hi: float = 3.55
    k_wall: float = 4000.0                 # kJ/mol/nm^2
    n_grid: int = 129                      # ODD: no Nyquist row exists
    abf_bandwidth: float = 0.04            # nm
    kde_bandwidth: float = 0.06            # nm
    abf_min_count: float = 100.0

    # --- ABF application ---
    abf_bias_scale: float = 1.0
    abf_warmup_steps: int = 10_000
    abf_force_clip: float = 2000.0         # kJ/mol/nm -- a GUARD, not a regulariser
    estimator_burn_in_steps: int = 0

    # --- selection (all arms; rates are set by the §3 calibration, never transferred) ---
    fr_rate: float = 0.0
    score_clip: float = 2.0
    fr_start_steps: int = 50_000
    fr_every: int = 500                    # 0.5 ps
    target_ema_rate: float = 0.005
    max_event_fraction: float = 0.05
    c_book: float = 1.0
    c_count: float = 1.0

    # --- diagnostics ---
    save_every: int = 5_000
    xi_trace_every: int = 100              # fine enough to resolve T_hit
    label_every: int = 5_000

    @property
    def beta(self):
        return 1.0 / (KB * self.temperature)

    def config_hash(self):
        import hashlib, json
        return hashlib.md5(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]


def _wall_gforce(R, sim):
    """Generalized force along R from the soft harmonic walls (``-dV_wall/dR``).

    Identical walls on every arm.  They keep the biased dynamics inside the interval where the
    estimator is reliable; an arm that wandered outside would be compared on a different domain.
    """
    f = torch.zeros_like(R)
    f = torch.where(R > sim.wall_hi, -sim.k_wall * (R - sim.wall_hi), f)
    f = torch.where(R < sim.wall_lo, -sim.k_wall * (R - sim.wall_lo), f)
    return f


def _fr_target(method, grid, dz, F_ema, B_n, oracle, beta):
    """The target marginal ``q`` for the Fisher--Rao family.  ``None`` for non-target arms."""
    if method not in ("mfr_practical", "mfr_oracle", "mfr_sham"):
        return None
    if method == "mfr_oracle":
        log_q = -beta * (oracle[None, :] - B_n)
    else:
        if F_ema is None:
            return None
        log_q = -beta * (F_ema - B_n)
    log_q = log_q - log_q.max(-1, keepdim=True).values
    return iv.normalize_density(torch.exp(log_q), dz)


def run_sampler_deca(method, engine, sim: DecaSimConfig, seeds, init_positions,
                     reference_free_energy=None, device="cuda", dtype=torch.float64,
                     verbose=True, progress_every=100_000):
    """Run ``R = len(seeds)`` matched-seed ensembles of ``method``.

    ``init_positions`` is ``(R, N, 112, 3)`` in nm, already relaxed and structurally validated.
    Returns a dict of profiles, traces and genealogy diagnostics.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; known: {sorted(METHODS)}")
    assert_no_reference_leakage(method, reference_free_energy)
    is_sel = method in SELECTION_METHODS

    R, N, A = len(seeds), sim.n_walkers, engine.n_atoms
    beta = sim.beta
    grid, dz = iv.interval_grid(sim.n_grid, sim.R_lo, sim.R_hi, device=device, dtype=dtype)
    K_abf = iv.gaussian_kernel_matrix(grid, sim.abf_bandwidth)
    K_kde = iv.reflected_kernel_matrix(grid, sim.kde_bandwidth, sim.R_lo, sim.R_hi)

    i_cv, j_cv = dsys.terminal_carbonyls(dsys.N_RES)
    cv = DistanceCV(i_cv, j_cv)
    labels = DecaLabels(device=device, dtype=dtype)

    gen_dyn = torch.Generator(device=device).manual_seed(int(sim.rng_seed))
    gen_sel = torch.Generator(device=device).manual_seed(int(sim.rng_seed) + 987654321)
    gens = make_seed_streams(sim.rng_seed, R, device)

    oracle = None
    if method in ("mfr_oracle",):
        oracle = torch.as_tensor(reference_free_energy, device=device, dtype=dtype)
        oracle = oracle - oracle.mean()

    q = torch.as_tensor(np.asarray(init_positions), device=device, dtype=dtype).reshape(R, N, A, 3)
    integ = BAOAB(engine.masses, dt=sim.dt, gamma=sim.gamma, temperature=sim.temperature,
                  force_fn=engine.forces, device=device, dtype=dtype)
    v = integ.maxwell((R, N, A, 3), gen_dyn, device, dtype)
    f = engine.forces(q.reshape(R * N, A, 3)).reshape(R, N, A, 3)

    fsum = torch.zeros(R, sim.n_grid, device=device, dtype=dtype)
    csum = torch.zeros(R, sim.n_grid, device=device, dtype=dtype)
    F_ema = None
    ancestors = torch.arange(N, device=device).expand(R, N).clone()
    total_repl = torch.zeros(R, dtype=torch.long)
    birth_hist = torch.zeros(R, sim.n_grid, device=device, dtype=dtype)
    death_hist = torch.zeros(R, sim.n_grid, device=device, dtype=dtype)
    n_clipped, n_scored = 0, 0

    diag = {k: [] for k in ("steps", "times", "mean_force", "pmf", "p_hat", "q_target",
                            "eff_counts", "ancestor_ess", "n_unique_ancestor",
                            "max_ancestor_frac", "repl_cumulative", "pq_l2", "kl_pq")}
    xi_trace, xi_trace_steps = [], []
    label_trace = {k: [] for k in ("n_hbonds", "alpha_frac", "rg", "ca_rmsd_helix", "y")}
    label_trace_xi, label_steps = [], []

    t0 = time.perf_counter()
    for step in range(sim.n_steps + 1):
        qf = q.reshape(R * N, A, 3)
        Ff = f.reshape(R * N, A, 3)
        f_loc, R_f, grad_f = cv.local_mean_force(qf, Ff, beta)
        Rv = R_f.reshape(R, N)
        f_loc = torch.clamp(f_loc, -8.0 * sim.abf_force_clip, 8.0 * sim.abf_force_clip).reshape(R, N)
        grad_full = grad_f.reshape(R, N, A, 3)

        if step >= sim.estimator_burn_in_steps:
            fsum += iv.bin_sum(Rv, f_loc, sim.n_grid, sim.R_lo, sim.R_hi)
            csum += iv.bin_counts(Rv, sim.n_grid, sim.R_lo, sim.R_hi)

        mf_profile = iv.mean_force_profile(fsum, csum, K_abf)
        A_hat = iv.free_energy_from_mean_force(mf_profile, grid, dz)
        ramp = min(1.0, step / max(sim.abf_warmup_steps, 1))
        abf_scale = sim.abf_bias_scale * ramp
        B_n = abf_scale * A_hat
        mf_at = iv.interval_interp(mf_profile, grid, Rv).clamp(-sim.abf_force_clip,
                                                               sim.abf_force_clip)
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
            q_grid = _fr_target(method, grid, dz, F_ema, B_n, oracle, beta)
            diag["steps"].append(step)
            diag["times"].append(step * sim.dt)
            diag["mean_force"].append(mf_profile.cpu().numpy())
            diag["pmf"].append(A_hat.cpu().numpy())
            diag["p_hat"].append(p_grid.cpu().numpy())
            diag["eff_counts"].append(iv.effective_counts(csum, K_abf).cpu().numpy())
            diag["repl_cumulative"].append(total_repl.numpy().copy())
            if q_grid is not None:
                lr = torch.log(p_grid.clamp_min(EPS)) - torch.log(q_grid.clamp_min(EPS))
                diag["q_target"].append(q_grid.cpu().numpy())
                diag["pq_l2"].append((((p_grid - q_grid) ** 2).sum(-1) * dz).sqrt().cpu().numpy())
                diag["kl_pq"].append(((p_grid * lr).sum(-1) * dz).cpu().numpy())
            else:
                diag["q_target"].append(np.full((R, sim.n_grid), np.nan))
                diag["pq_l2"].append(np.full(R, np.nan))
                diag["kl_pq"].append(np.full(R, np.nan))
            anc = ancestors.cpu().numpy()
            ess, nuq, mxf = [], [], []
            for r in range(R):
                cnt = np.bincount(anc[r], minlength=N).astype(float) / N
                ess.append(1.0 / np.sum(cnt ** 2))
                nuq.append(int((cnt > 0).sum()))
                mxf.append(float(cnt.max()))
            diag["ancestor_ess"].append(np.array(ess))
            diag["n_unique_ancestor"].append(np.array(nuq))
            diag["max_ancestor_frac"].append(np.array(mxf))

        if step == sim.n_steps:
            break

        # ---------------------------------------------------------------- integrate
        v = v + (0.5 * sim.dt) * (f + bias_force) / integ.m
        q = q + (0.5 * sim.dt) * v
        v = integ.c1 * v + integ.c2 * torch.randn(v.shape, generator=gen_dyn, device=device,
                                                  dtype=dtype) * integ.sigma
        q = q + (0.5 * sim.dt) * v
        f = engine.forces(q.reshape(R * N, A, 3)).reshape(R, N, A, 3)
        f_new_loc, R_new, grad_new = cv.local_mean_force(q.reshape(R * N, A, 3),
                                                         f.reshape(R * N, A, 3), beta)
        Rn = R_new.reshape(R, N)
        mf_at2 = iv.interval_interp(mf_profile, grid, Rn).clamp(-sim.abf_force_clip,
                                                                sim.abf_force_clip)
        applied2 = abf_scale * mf_at2 + _wall_gforce(Rn, sim)
        bias2 = dist_bias_force(grad_new.reshape(R * N, A, 3),
                                applied2.reshape(R * N)).reshape(R, N, A, 3)
        v = v + (0.5 * sim.dt) * (f + bias2) / integ.m

        if step % 20_000 == 0:
            check_finite(step, ("q", q), ("v", v), tag=f"deca_{method}")

        # ---------------------------------------------------------------- selection
        nxt = step + 1
        if (is_sel and sim.fr_rate > 0.0 and nxt >= sim.fr_start_steps
                and (nxt - sim.fr_start_steps) % max(int(sim.fr_every), 1) == 0):
            score = None
            if method in ("mfr_practical", "mfr_oracle", "mfr_sham"):
                q_grid = _fr_target(method, grid, dz, F_ema, B_n, oracle, beta)
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
                n_clipped += int((score.abs() >= sim.score_clip - 1e-9).sum())
                n_scored += score.numel()
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

        if verbose and progress_every and nxt % progress_every == 0:
            el = time.perf_counter() - t0
            fr = nxt / sim.n_steps
            print(f"    {method:16s} {100*fr:5.1f}%  {el/60:6.1f} min, "
                  f"~{el/fr*(1-fr)/60:6.1f} left  repl={int(total_repl.sum())}", flush=True)

    out = dict(method=method, grid=grid.cpu().numpy(), dz=float(dz),
               R_lo=sim.R_lo, R_hi=sim.R_hi, seeds=np.asarray(seeds),
               config=asdict(sim), config_hash=sim.config_hash(),
               runtime_seconds=time.perf_counter() - t0,
               total_replacement_events=total_repl.numpy(),
               birth_hist=birth_hist.cpu().numpy(), death_hist=death_hist.cpu().numpy(),
               final_eff_counts=iv.effective_counts(csum, K_abf).cpu().numpy(),
               F_target_ema=(F_ema.cpu().numpy() if F_ema is not None else None),
               score_clip_fraction=(n_clipped / max(n_scored, 1)),
               xi_trace=np.stack(xi_trace), xi_trace_steps=np.asarray(xi_trace_steps),
               label_steps=np.asarray(label_steps),
               label_xi=np.stack(label_trace_xi),
               **{f"label_{k}": np.stack(v) for k, v in label_trace.items()})
    for k in diag:
        out[k] = np.asarray(diag[k])
    if verbose:
        print(f"  {method:16s} R={R} N={N}: {out['runtime_seconds']/60:.1f} min, "
              f"repl={int(total_repl.sum())}, clip={out['score_clip_fraction']:.4f}", flush=True)
    return out
