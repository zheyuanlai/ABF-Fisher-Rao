"""Batched ABF (+ Fisher--Rao birth--death) sampler for united-atom alkanes.

Mirrors ``wca_abffr_core.run_sampler_gpu`` but for a periodic dihedral CV, and runs
``R`` seeds of one method in a single GPU process (leading batch dim ``R``) so the
overhead-bound autodiff CV force is amortised.  Coordinates are ``q`` of shape
``(R, N, n_atoms, 3)``.

Methods: ``abf`` (baseline), ``fr_estimated`` (deployable), ``fr_uniform``
(ablation), ``fr_oracle`` (diagnostic; the only method that may see the reference).

Matched seeds across methods: Langevin noise + initial conditions are drawn from a
``gen_dyn`` generator seeded by a *method-independent* ``rng_seed``, and FR
birth--death RNG from a separate ``gen_fr`` stream, so ABF and every FR method see
identical dynamics noise per seed (an improvement over a single global stream).

No boundary walls are needed (phi is periodic).  COM drift is removed every step
(translational zero mode); rotation is left free (phi and V are rotation invariant).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import numpy as np
import torch

from . import geometry as geom
from . import periodic as per
from . import potentials as pot
from .cv import DihedralCV, abf_bias_force

EPS = 1.0e-12
FR_METHODS = ("fr_estimated", "fr_uniform", "fr_oracle")
ESTIMATED_TARGET_METHODS = ("fr_estimated",)
ALL_METHODS = ("abf",) + FR_METHODS
PI = math.pi


@dataclass
class AlkaneSimConfig:
    # dynamics
    dt: float = 5.0e-4
    n_steps: int = 100_000
    n_replicas: int = 512
    save_every: int = 2_000
    rng_seed: int = 20260719          # method-independent (matched seeds)
    # CV grid / estimator
    n_grid: int = 180
    abf_bandwidth: float = 0.05       # radians
    kde_bandwidth: float = 0.10       # radians (for the FR marginal p_hat)
    # ABF application
    abf_bias_scale: float = 1.0
    abf_warmup_steps: int = 10_000
    abf_force_clip: float = 60.0      # clip |A'(phi)| like WCA
    estimator_burn_in_steps: int = 10_000
    # Fisher--Rao
    fr_rate: float = 0.10
    score_clip: float = 2.0
    fr_start_steps: int = 20_000
    fr_every: int = 5
    target_ema_rate: float = 0.005
    max_event_fraction: float = 0.02
    # 2-D joint diagnostics (pentane)
    n_grid2: int = 48                 # coarse grid for the (phi1,phi2) joint histogram
    # basin barrier (|phi|<barrier => trans); gauche beyond, split by sign
    basin_barrier: float = math.radians(61.6)

    def config_hash(self):
        import hashlib, json
        return hashlib.md5(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]


def basin_index(phi, barrier):
    """0=trans (|phi|<barrier), 1=G+ (phi>=barrier), 2=G- (phi<=-barrier)."""
    idx = torch.zeros_like(phi, dtype=torch.long)
    idx = torch.where(phi >= barrier, torch.ones_like(idx), idx)
    idx = torch.where(phi <= -barrier, 2 * torch.ones_like(idx), idx)
    return idx


def _fr_target(method, grid, dphi, F_ema, B_n, oracle, beta):
    """Build FR target q_n(phi) (R,n_grid) or None (abf / not-yet-started)."""
    if method == "abf":
        return None
    if method == "fr_uniform":
        R = B_n.shape[0] if B_n is not None else 1
        return per.normalize_density(torch.ones(R, grid.numel(), device=grid.device, dtype=grid.dtype), dphi)
    if method == "fr_oracle":
        log_q = -beta * (oracle[None, :] - B_n)
        log_q = log_q - log_q.max(-1, keepdim=True).values
        return per.normalize_density(torch.exp(log_q), dphi)
    # fr_estimated
    if F_ema is None:
        return None
    log_q = -beta * (F_ema - B_n)
    log_q = log_q - log_q.max(-1, keepdim=True).values
    return per.normalize_density(torch.exp(log_q), dphi)


def _recentered_clipped_score(raw, clip):
    s = raw - raw.mean(-1, keepdim=True)
    for _ in range(3):
        s = torch.clamp(s, -clip, clip)
        s = s - s.mean(-1, keepdim=True)
    return s


def _fr_score(phi, grid, dphi, K_kde, q_grid, kde_bw, clip):
    """FR log-ratio score per replica: S = log p_hat - log q - KL, clipped+recentred.

    ``phi`` (R,N), ``q_grid`` (R,n_grid). Returns (score (R,N), p_grid (R,n_grid), kl (R,))."""
    counts = per.bin_counts(phi, grid.numel())
    p_grid = per.normalize_density(per.smooth(counts, K_kde), dphi)
    p_at = per.circular_interp(p_grid, grid, phi)
    q_at = per.circular_interp(q_grid, grid, phi)
    log_ratio = torch.log(p_grid.clamp_min(EPS)) - torch.log(q_grid.clamp_min(EPS))
    kl = (p_grid * log_ratio).sum(-1) * dphi
    raw = torch.log(p_at.clamp_min(EPS)) - torch.log(q_at.clamp_min(EPS)) - kl[:, None]
    return _recentered_clipped_score(raw, clip), p_grid, kl


def _birth_death(q, score, ancestors, sim, gen_fr):
    """Fixed-population birth--death per run (loops over R). Clones full configs.

    ``q`` (R,N,A,3), ``score`` (R,N), ``ancestors`` (R,N). Returns
    (q_new, ancestors_new, n_repl (R,), death_idx list, birth_src list)."""
    R, N = score.shape
    dt_eff = sim.dt * max(int(sim.fr_every), 1)
    max_events = int(sim.max_event_fraction * N)
    q_new = q.clone()
    anc_new = ancestors.clone()
    n_repl = torch.zeros(R, dtype=torch.long, device=q.device)
    deaths = [None] * R
    births = [None] * R
    if max_events < 1 or sim.fr_rate <= 0.0:
        return q_new, anc_new, n_repl, deaths, births
    death_w = torch.clamp(score, min=0.0)
    birth_w = torch.clamp(-score, min=0.0)
    death_prob = torch.where(death_w > 0, 1.0 - torch.exp(-sim.fr_rate * death_w * dt_eff),
                             torch.zeros_like(death_w))
    u = torch.rand(R, N, generator=gen_fr, device=q.device, dtype=q.dtype)
    fire = u < death_prob
    for r in range(R):
        if birth_w[r].sum() <= EPS or death_w[r].sum() <= EPS:
            continue
        di = torch.nonzero(fire[r], as_tuple=False).flatten()
        n = int(di.numel())
        if n == 0:
            continue
        if n > max_events:
            perm = torch.randperm(n, generator=gen_fr, device=q.device)[:max_events]
            di = di[perm]; n = max_events
        src = torch.multinomial(birth_w[r], n, replacement=True, generator=gen_fr)
        q_new[r, di] = q[r, src]
        anc_new[r, di] = ancestors[r, src]
        n_repl[r] = n
        deaths[r] = di; births[r] = src
    return q_new, anc_new, n_repl, deaths, births


def _ancestor_stats(ancestors, N):
    """ESS, n_unique, max_fraction per run from an (R,N) ancestor-label tensor."""
    R = ancestors.shape[0]
    ess = torch.zeros(R, dtype=torch.float64)
    nuq = torch.zeros(R, dtype=torch.long)
    maxf = torch.zeros(R, dtype=torch.float64)
    for r in range(R):
        counts = torch.bincount(ancestors[r], minlength=N).to(torch.float64)
        w = counts / counts.sum().clamp_min(1.0)
        ess[r] = 1.0 / (w * w).sum().clamp_min(EPS)
        nuq[r] = int((counts > 0).sum())
        maxf[r] = (counts.max() / counts.sum().clamp_min(1.0))
    return ess.numpy(), nuq.numpy(), maxf.numpy()


def assert_no_reference_leakage(method, oracle_free_energy):
    if method == "fr_oracle":
        if oracle_free_energy is None:
            raise ValueError("fr_oracle requires the reference free energy.")
        return
    if oracle_free_energy is not None:
        raise AssertionError(
            f"NO-REFERENCE-LEAKAGE VIOLATION: method={method!r} received a reference "
            "free energy; only fr_oracle may.")


def run_sampler(method, params: pot.AlkaneParams, sim: AlkaneSimConfig, seeds,
                cv: DihedralCV, device, dtype=torch.float64, initial_dihedrals=None,
                oracle_free_energy=None, collect_pentane=False, verbose=True):
    """Run ``R=len(seeds)`` matched-seed replicas of ``method`` in one process.

    ``initial_dihedrals`` : per-seed init.  Either ``(n_dih,)`` broadcast to all
    seeds/replicas (localized init) or a callable ``(R, N)->(R,N,n_dih)`` sampler
    (dispersed init).  If None, all replicas start at the trans state.
    Returns a dict of per-run (leading dim R) metrics/time-series/profiles.
    """
    if method not in ALL_METHODS:
        raise ValueError(f"unknown method {method!r}")
    assert_no_reference_leakage(method, oracle_free_energy)
    is_fr = method in FR_METHODS
    R = len(seeds)
    N = sim.n_replicas
    A = params.n_atoms
    n_dih = params.n_dihedrals
    beta = params.beta
    grid, dphi = per.periodic_grid(sim.n_grid, device=device, dtype=dtype)
    K_abf = per.wrapped_gaussian_kernel_matrix(grid, sim.abf_bandwidth)
    K_kde = per.wrapped_gaussian_kernel_matrix(grid, sim.kde_bandwidth)
    gen_dyn = torch.Generator(device=device).manual_seed(int(sim.rng_seed))
    gen_fr = torch.Generator(device=device).manual_seed(int(sim.rng_seed) + 987654321)

    oracle = None
    if method == "fr_oracle":
        oracle = torch.as_tensor(oracle_free_energy, device=device, dtype=dtype)
        oracle = oracle - oracle.mean()

    # ---- initial conditions ----
    if initial_dihedrals is None:
        init = torch.zeros(R, N, n_dih, device=device, dtype=dtype)
    elif callable(initial_dihedrals):
        init = initial_dihedrals(R, N, gen_dyn).to(device=device, dtype=dtype)
    else:
        d0 = torch.as_tensor(initial_dihedrals, device=device, dtype=dtype).reshape(1, 1, n_dih)
        init = d0.expand(R, N, n_dih).clone()
    q = geom.place_chain(init.reshape(R * N, n_dih), A, d0=params.d0, theta0=params.theta0,
                         device=device, dtype=dtype).reshape(R, N, A, 3)
    # small jitter so replicas decorrelate off the exact lattice
    q = q + 1e-3 * torch.randn(q.shape, generator=gen_dyn, device=device, dtype=dtype)
    q = geom.remove_com(q)

    noise_scale = math.sqrt(2.0 * sim.dt / beta)
    # estimator accumulators (per run)
    fsum = torch.zeros(R, sim.n_grid, device=device, dtype=dtype)
    csum = torch.zeros(R, sim.n_grid, device=device, dtype=dtype)
    fsum_prod = torch.zeros(R, sim.n_grid, device=device, dtype=dtype)
    csum_prod = torch.zeros(R, sim.n_grid, device=device, dtype=dtype)
    F_ema = None
    ancestors = (torch.arange(N, device=device).expand(R, N).clone() if is_fr else None)
    total_repl = torch.zeros(R, dtype=torch.long)

    # pentane joint histogram accumulator (post burn-in)
    do_pent = collect_pentane and A >= 5
    if do_pent:
        g2, dphi2 = per.periodic_grid(sim.n_grid2, device=device, dtype=dtype)
        joint_hist = torch.zeros(R, sim.n_grid2, sim.n_grid2, device=device, dtype=dtype)
        cv2 = DihedralCV((1, 2, 3, 4))
    # transition tracking (phi1 basins) per replica
    prev_basin = None
    trans_counts = torch.zeros(R, dtype=torch.long, device=device)
    # round trips: a replica completes one when it returns to T after visiting a G
    # basin (T -> G -> T). ``has_left_T`` is set on any G step, cleared on returning to T.
    has_left_T = torch.zeros(R, N, dtype=torch.bool, device=device)
    rep_roundtrips = torch.zeros(R, N, dtype=torch.long, device=device)
    # birth/death phi histograms
    birth_hist = torch.zeros(R, sim.n_grid, device=device, dtype=dtype)
    death_hist = torch.zeros(R, sim.n_grid, device=device, dtype=dtype)
    # score aggregates
    score_std_sum = np.zeros(R); score_absmax = np.zeros(R); n_score = 0

    diag = {k: [] for k in ["steps", "times", "mean_force", "pmf", "p_hat", "q_target",
                            "eff_counts", "ancestor_ess", "n_unique_ancestor",
                            "max_ancestor_frac", "repl_cumulative", "pq_l2", "kl_pq",
                            "frac_T", "frac_Gp", "frac_Gm",
                            "frac2_T", "frac2_Gp", "frac2_Gm"]}
    t0 = time.perf_counter()
    for step in range(sim.n_steps + 1):
        qf = q.reshape(R * N, A, 3)
        F = pot.forces(qf, params)
        f_loc, phi_f, grad_f = cv.local_mean_force(qf, F, beta)
        phi = phi_f.reshape(R, N)
        f_loc = torch.clamp(f_loc, -sim.abf_force_clip * 8, sim.abf_force_clip * 8).reshape(R, N)
        grad_full = grad_f.reshape(R, N, A, 3)

        # accumulate estimator
        fsum += per.bin_sum(phi, f_loc, sim.n_grid)
        csum += per.bin_counts(phi, sim.n_grid)
        if step >= sim.estimator_burn_in_steps:
            fsum_prod += per.bin_sum(phi, f_loc, sim.n_grid)
            csum_prod += per.bin_counts(phi, sim.n_grid)
            if do_pent:
                phi2 = cv2.value(qf).reshape(R, N)
                i1 = torch.floor((phi + PI) / dphi2).long().clamp(0, sim.n_grid2 - 1)
                i2 = torch.floor((phi2 + PI) / dphi2).long().clamp(0, sim.n_grid2 - 1)
                lin = i1 * sim.n_grid2 + i2
                joint_hist.view(R, -1).scatter_add_(1, lin, torch.ones_like(phi))

        # ABF bias from current profile
        mf_profile = per.mean_force_profile(fsum, csum, K_abf)      # (R,n_grid)
        A_hat = per.free_energy_from_mean_force(mf_profile, grid, dphi)
        ramp = min(1.0, step / max(sim.abf_warmup_steps, 1))
        abf_scale = sim.abf_bias_scale * ramp
        B_n = abf_scale * A_hat
        mf_at = per.circular_interp(mf_profile, grid, phi).clamp(-sim.abf_force_clip, sim.abf_force_clip)
        bias_at = abf_scale * mf_at
        bias_force = abf_bias_force(grad_full.reshape(R * N, A, 3),
                                    bias_at.reshape(R * N)).reshape(R, N, A, 3)

        # EMA target
        if method in ESTIMATED_TARGET_METHODS and (step + 1) >= sim.fr_start_steps:
            if F_ema is None:
                F_ema = A_hat.clone()
            else:
                rt = sim.target_ema_rate
                F_ema = (1 - rt) * F_ema + rt * A_hat
            F_ema = F_ema - F_ema.mean(-1, keepdim=True)

        # transition + basin bookkeeping
        cur_basin = basin_index(phi, sim.basin_barrier)
        cur_T = cur_basin == 0
        cur_G = ~cur_T
        if prev_basin is not None:
            trans_counts += (cur_basin != prev_basin).sum(-1)
            rep_roundtrips += (has_left_T & cur_T).long()      # T -> G -> T completed
        has_left_T = torch.where(cur_T, torch.zeros_like(has_left_T), has_left_T | cur_G)
        prev_basin = cur_basin

        # ---- diagnostics ----
        if step % sim.save_every == 0 or step == sim.n_steps:
            rep_est_f = fsum_prod if csum_prod.sum() > 0 else fsum
            rep_est_c = csum_prod if csum_prod.sum() > 0 else csum
            mf_rep = per.mean_force_profile(rep_est_f, rep_est_c, K_abf)
            pmf_rep = per.free_energy_from_mean_force(mf_rep, grid, dphi)
            diag["steps"].append(step); diag["times"].append(step * sim.dt)
            diag["mean_force"].append(mf_rep.cpu().numpy())
            diag["pmf"].append(pmf_rep.cpu().numpy())
            diag["eff_counts"].append(per.effective_counts(csum, K_abf).cpu().numpy())
            diag["repl_cumulative"].append(total_repl.numpy().copy())
            p_grid = per.kde_marginal(phi, K_kde, sim.n_grid, dphi)
            diag["p_hat"].append(p_grid.cpu().numpy())
            q_grid = _fr_target(method, grid, dphi, F_ema, B_n, oracle, beta)
            if q_grid is not None:
                lr = torch.log(p_grid.clamp_min(EPS)) - torch.log(q_grid.clamp_min(EPS))
                diag["q_target"].append(q_grid.cpu().numpy())
                diag["pq_l2"].append((((p_grid - q_grid) ** 2).sum(-1) * dphi).sqrt().cpu().numpy())
                diag["kl_pq"].append(((p_grid * lr).sum(-1) * dphi).cpu().numpy())
            else:
                diag["q_target"].append(np.full((R, sim.n_grid), np.nan))
                diag["pq_l2"].append(np.full(R, np.nan)); diag["kl_pq"].append(np.full(R, np.nan))
            if is_fr:
                ess, nuq, maxf = _ancestor_stats(ancestors.cpu(), N)
            else:
                ess, nuq, maxf = np.full(R, np.nan), np.full(R, N), np.full(R, np.nan)
            diag["ancestor_ess"].append(ess); diag["n_unique_ancestor"].append(nuq)
            diag["max_ancestor_frac"].append(maxf)
            fr_ = (cur_basin == 0).float().mean(-1).cpu().numpy()
            diag["frac_T"].append(fr_)
            diag["frac_Gp"].append((cur_basin == 1).float().mean(-1).cpu().numpy())
            diag["frac_Gm"].append((cur_basin == 2).float().mean(-1).cpu().numpy())
            if do_pent:
                phi2 = cv2.value(qf).reshape(R, N)
                b2 = basin_index(phi2, sim.basin_barrier)
                diag["frac2_T"].append((b2 == 0).float().mean(-1).cpu().numpy())
                diag["frac2_Gp"].append((b2 == 1).float().mean(-1).cpu().numpy())
                diag["frac2_Gm"].append((b2 == 2).float().mean(-1).cpu().numpy())
            else:
                for k in ("frac2_T", "frac2_Gp", "frac2_Gm"):
                    diag[k].append(np.full(R, np.nan))

        if step == sim.n_steps:
            break

        # ---- Langevin step ----
        noise = torch.randn(q.shape, generator=gen_dyn, device=device, dtype=dtype)
        q = q + sim.dt * (F.reshape(R, N, A, 3) + bias_force) + noise_scale * noise
        q = geom.remove_com(q)

        # ---- Fisher--Rao birth--death ----
        if is_fr:
            nxt = step + 1
            if nxt >= sim.fr_start_steps and (nxt - sim.fr_start_steps) % max(int(sim.fr_every), 1) == 0:
                phi_new = cv.value(q.reshape(R * N, A, 3)).reshape(R, N)
                q_grid = _fr_target(method, grid, dphi, F_ema, B_n, oracle, beta)
                if q_grid is not None:
                    score, p_fr, kl = _fr_score(phi_new, grid, dphi, K_kde, q_grid,
                                                sim.kde_bandwidth, sim.score_clip)
                    ss = score.detach().cpu().numpy()
                    score_std_sum += ss.std(axis=1); score_absmax = np.maximum(score_absmax, np.abs(ss).max(axis=1))
                    n_score += 1
                    q, ancestors, n_repl, deaths, births = _birth_death(q, score, ancestors, sim, gen_fr)
                    total_repl += n_repl.cpu()
                    for r in range(R):
                        if deaths[r] is not None and deaths[r].numel() > 0:
                            zb = phi_new[r].index_select(0, births[r])
                            zd = phi_new[r].index_select(0, deaths[r])
                            birth_hist[r] += per.bin_counts(zb[None, :], sim.n_grid)[0]
                            death_hist[r] += per.bin_counts(zd[None, :], sim.n_grid)[0]
                            # keep transition bookkeeping consistent under cloning
                            for arr in (has_left_T, rep_roundtrips):
                                arr[r, deaths[r]] = arr[r].index_select(0, births[r])
                            if prev_basin is not None:
                                prev_basin[r, deaths[r]] = prev_basin[r].index_select(0, births[r])

    out = {"method": method, "grid": grid.cpu().numpy(), "dphi": float(dphi),
           "runtime_seconds": time.perf_counter() - t0,
           "total_replacement_events": total_repl.numpy(),
           "n_transitions": trans_counts.cpu().numpy(),
           "n_round_trips": rep_roundtrips.sum(-1).cpu().numpy(),
           "birth_hist": birth_hist.cpu().numpy(), "death_hist": death_hist.cpu().numpy(),
           "F_target_ema": (F_ema.cpu().numpy() if F_ema is not None else None),
           "fr_score_std": (score_std_sum / max(n_score, 1)),
           "fr_score_absmax": score_absmax,
           "final_eff_counts": per.effective_counts(csum, K_abf).cpu().numpy()}
    for k in diag:
        out[k] = np.asarray(diag[k])
    if do_pent:
        out["joint_hist"] = joint_hist.cpu().numpy()
        out["grid2"] = g2.cpu().numpy(); out["dphi2"] = float(dphi2)
    if verbose:
        print(f"  {method:12s} R={R} N={N}: {out['runtime_seconds']:.1f}s "
              f"repl={out['total_replacement_events'].sum()} trans={out['n_transitions'].sum()}")
    return out


def run_frozen_bias(params: pot.AlkaneParams, sim: AlkaneSimConfig, learned_mean_force,
                    seeds, cv: DihedralCV, device, dtype=torch.float64,
                    initial_dihedrals=None, verbose=True):
    # NOTE: no @torch.no_grad here -- pot.forces / cv.geometry manage their own
    # autograd internally (a global no_grad would break them); outputs are detached.
    """Frozen-bias validation: fix the learned bias B(phi), run fresh dynamics with no
    ABF update and no birth--death, accumulate the biased marginal p_B, and reconstruct
    ``F_recon(phi) = B(phi) - beta^{-1} log p_B(phi) + C``.

    ``learned_mean_force`` is ``(n_grid,)`` or ``(R, n_grid)``. No reference is read, so
    this is deployable for any learned bias (abf / fr_* alike).
    """
    R = len(seeds); N = sim.n_replicas; Aat = params.n_atoms; n_dih = params.n_dihedrals
    beta = params.beta
    grid, dphi = per.periodic_grid(sim.n_grid, device=device, dtype=dtype)
    K_kde = per.wrapped_gaussian_kernel_matrix(grid, sim.kde_bandwidth)
    mf = torch.as_tensor(np.asarray(learned_mean_force), device=device, dtype=dtype)
    if mf.ndim == 1:
        mf = mf[None, :].expand(R, -1).contiguous()
    B_grid = per.free_energy_from_mean_force(mf, grid, dphi)
    gen = torch.Generator(device=device).manual_seed(int(sim.rng_seed) + 555)
    if initial_dihedrals is None:
        init = torch.zeros(R, N, n_dih, device=device, dtype=dtype)
    else:
        d0 = torch.as_tensor(initial_dihedrals, device=device, dtype=dtype).reshape(1, 1, n_dih)
        init = d0.expand(R, N, n_dih).clone()
    q = geom.place_chain(init.reshape(R * N, n_dih), Aat, d0=params.d0, theta0=params.theta0,
                         device=device, dtype=dtype).reshape(R, N, Aat, 3)
    q = geom.remove_com(q + 1e-3 * torch.randn(q.shape, generator=gen, device=device, dtype=dtype))
    noise_scale = math.sqrt(2.0 * sim.dt / beta)
    p_accum = torch.zeros(R, sim.n_grid, device=device, dtype=dtype)
    n_marg = 0
    t0 = time.perf_counter()
    for step in range(sim.n_steps + 1):
        qf = q.reshape(R * N, Aat, 3)
        F = pot.forces(qf, params)
        phi, grad_full, _ = cv.geometry(qf)
        phi = phi.reshape(R, N)
        bias_at = per.circular_interp(mf, grid, phi).clamp(-sim.abf_force_clip, sim.abf_force_clip)
        bias_force = abf_bias_force(grad_full, bias_at.reshape(R * N)).reshape(R, N, Aat, 3)
        if step >= sim.estimator_burn_in_steps:
            p_accum += per.kde_marginal(phi, K_kde, sim.n_grid, dphi)
            n_marg += 1
        if step == sim.n_steps:
            break
        noise = torch.randn(q.shape, generator=gen, device=device, dtype=dtype)
        q = geom.remove_com(q + sim.dt * (F.reshape(R, N, Aat, 3) + bias_force) + noise_scale * noise)
    p_B = per.normalize_density(p_accum / max(n_marg, 1), dphi)
    F_recon = B_grid - (1.0 / beta) * torch.log(p_B.clamp_min(EPS))
    F_recon = F_recon - F_recon.mean(-1, keepdim=True)
    if verbose:
        print(f"  frozen-bias R={R}: {time.perf_counter()-t0:.1f}s")
    return {"grid": grid.cpu().numpy(), "p_B": p_B.cpu().numpy(),
            "F_recon": F_recon.cpu().numpy(), "B": B_grid.cpu().numpy(),
            "learned_mean_force": mf.cpu().numpy()}
