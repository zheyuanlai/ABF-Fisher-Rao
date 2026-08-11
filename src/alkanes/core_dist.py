"""Batched ABF (+ Fisher--Rao birth--death) sampler for a NON-periodic distance CV
(pentane R15 = |q5-q1|, butane R14 = |q4-q1|).

Mirrors :func:`alkanes.core.run_sampler` but on a bounded interval ``[R_lo, R_hi]``:
the estimator/KDE/interpolation/PMF come from :mod:`alkanes.interval` (Euclidean, not
circular), identical soft harmonic walls are applied to every method to keep the biased
dynamics inside a reliable interval, and the hidden-conditional diagnostic is the joint
``(phi1,phi2)`` torsion distribution within R bins (the analogue of ``p(phi2|phi1)`` for
the dihedral study).  The birth--death / ancestor / score / no-leakage machinery is reused
unchanged from :mod:`alkanes.core` (it is CV-agnostic).

``q`` has shape ``(R, N, n_atoms, 3)``; ``R`` seeds of one method run in one GPU process.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, asdict

import numpy as np
import torch

from . import geometry as geom
from . import interval as iv
from . import potentials as pot
from . import density2d as d2
from .distance_cv import DistanceCV, dist_bias_force
from .core import (_birth_death, _ancestor_stats, _recentered_clipped_score,
                   assert_no_reference_leakage, FR_METHODS, ESTIMATED_TARGET_METHODS, ALL_METHODS)

EPS = 1.0e-12
PI = math.pi


@dataclass
class DistSimConfig:
    # dynamics
    dt: float = 5.0e-4
    n_steps: int = 100_000
    n_replicas: int = 768
    save_every: int = 5_000
    rng_seed: int = 20260719
    # interval / estimator
    R_lo: float = 1.4
    R_hi: float = 3.7
    wall_lo: float = 1.45
    wall_hi: float = 3.65
    k_wall: float = 200.0
    n_grid: int = 256
    abf_bandwidth: float = 0.04
    kde_bandwidth: float = 0.06
    # ABF application
    abf_bias_scale: float = 1.0
    abf_warmup_steps: int = 5_000
    abf_force_clip: float = 60.0
    #: Support floor before a bin's mean force is trusted to APPLY bias (the standard ABF
    #: `fullSamples` guard).  **Defaults to 0.0 = disabled, which reproduces frozen v1 exactly**;
    #: v1 is immutable and every published R15 number was produced with no guard.
    #:
    #: This field exists because v1 is internally inconsistent: `Sim2DConfig` carries
    #: `abf_min_count = 5.0` and `core2d._project_bias` masks untrusted cells, while
    #: `DistSimConfig` had no such field at all and `jobs_cv._dist_sim` never passed one.  The
    #: 2-D torsion cell therefore ran WITH the guard and R15 ran WITHOUT it, so the two v1
    #: classifications were never on equal footing.  The v2 audit in
    #: `results/v2_validity_audits/r15_abf_guard/` turns it on and changes nothing else.
    abf_min_count: float = 0.0
    estimator_burn_in_steps: int = 6_000
    # Fisher--Rao (hyperparameters are on the normalised CV in [0,1] internally where noted)
    fr_rate: float = 0.02
    score_clip: float = 2.0
    fr_start_steps: int = 10_000
    fr_every: int = 5
    target_ema_rate: float = 0.005
    max_event_fraction: float = 0.01
    # conditional torsion diagnostic
    n_grid2: int = 48
    n_rbins: int = 12
    basin_barrier: float = math.radians(61.6)

    def config_hash(self):
        import hashlib, json
        return hashlib.md5(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]


def _wall_gforce(R, sim):
    """Generalized force along R from the soft harmonic walls (= -dV_wall/dR)."""
    f = torch.zeros_like(R)
    hi = R > sim.wall_hi
    lo = R < sim.wall_lo
    f = torch.where(hi, -sim.k_wall * (R - sim.wall_hi), f)
    f = torch.where(lo, -sim.k_wall * (R - sim.wall_lo), f)   # (R<wall_lo) => positive push out
    return f


def _fr_target(method, grid, dz, R_lo, R_hi, F_ema, B_n, oracle, beta):
    if method == "abf":
        return None
    if method == "fr_uniform":
        R = B_n.shape[0] if B_n is not None else 1
        return iv.normalize_density(torch.ones(R, grid.numel(), device=grid.device, dtype=grid.dtype), dz)
    if method == "fr_oracle":
        log_q = -beta * (oracle[None, :] - B_n)
        log_q = log_q - log_q.max(-1, keepdim=True).values
        return iv.normalize_density(torch.exp(log_q), dz)
    if F_ema is None:
        return None
    log_q = -beta * (F_ema - B_n)
    log_q = log_q - log_q.max(-1, keepdim=True).values
    return iv.normalize_density(torch.exp(log_q), dz)


def _fr_score(R, grid, dz, R_lo, R_hi, K_kde, q_grid, clip):
    counts = iv.bin_counts(R, grid.numel(), R_lo, R_hi)
    p_grid = iv.normalize_density(iv.smooth(counts, K_kde), dz)
    p_at = iv.interval_interp(p_grid, grid, R)
    q_at = iv.interval_interp(q_grid, grid, R)
    log_ratio = torch.log(p_grid.clamp_min(EPS)) - torch.log(q_grid.clamp_min(EPS))
    kl = (p_grid * log_ratio).sum(-1) * dz
    raw = torch.log(p_at.clamp_min(EPS)) - torch.log(q_at.clamp_min(EPS)) - kl[:, None]
    return _recentered_clipped_score(raw, clip), p_grid, kl


def run_sampler_dist(method, params: pot.AlkaneParams, sim: DistSimConfig, seeds,
                     cv: DistanceCV, device, dtype=torch.float64, initial_dihedrals=None,
                     oracle_free_energy=None, collect_conditional=True, verbose=True):
    """Run ``R=len(seeds)`` matched-seed replicas of ``method`` on the distance CV."""
    if method not in ALL_METHODS:
        raise ValueError(f"unknown method {method!r}")
    assert_no_reference_leakage(method, oracle_free_energy)
    is_fr = method in FR_METHODS
    R = len(seeds); N = sim.n_replicas; A = params.n_atoms; n_dih = params.n_dihedrals
    beta = params.beta
    grid, dz = iv.interval_grid(sim.n_grid, sim.R_lo, sim.R_hi, device=device, dtype=dtype)
    K_abf = iv.gaussian_kernel_matrix(grid, sim.abf_bandwidth)
    K_kde = iv.reflected_kernel_matrix(grid, sim.kde_bandwidth, sim.R_lo, sim.R_hi)
    gen_dyn = torch.Generator(device=device).manual_seed(int(sim.rng_seed))
    gen_fr = torch.Generator(device=device).manual_seed(int(sim.rng_seed) + 987654321)

    oracle = None
    if method == "fr_oracle":
        oracle = torch.as_tensor(oracle_free_energy, device=device, dtype=dtype)
        oracle = oracle - oracle.mean()

    # initial conditions (dihedral-space init -> Cartesian)
    if initial_dihedrals is None:
        init = torch.zeros(R, N, n_dih, device=device, dtype=dtype)
    elif callable(initial_dihedrals):
        init = initial_dihedrals(R, N, gen_dyn).to(device=device, dtype=dtype)
    else:
        d0 = torch.as_tensor(initial_dihedrals, device=device, dtype=dtype).reshape(1, 1, n_dih)
        init = d0.expand(R, N, n_dih).clone()
    q = geom.place_chain(init.reshape(R * N, n_dih), A, d0=params.d0, theta0=params.theta0,
                         device=device, dtype=dtype).reshape(R, N, A, 3)
    q = geom.remove_com(q + 1e-3 * torch.randn(q.shape, generator=gen_dyn, device=device, dtype=dtype))

    noise_scale = math.sqrt(2.0 * sim.dt / beta)
    fsum = torch.zeros(R, sim.n_grid, device=device, dtype=dtype)
    csum = torch.zeros(R, sim.n_grid, device=device, dtype=dtype)
    fsum_prod = torch.zeros(R, sim.n_grid, device=device, dtype=dtype)
    csum_prod = torch.zeros(R, sim.n_grid, device=device, dtype=dtype)
    F_ema = None
    ancestors = (torch.arange(N, device=device).expand(R, N).clone() if is_fr else None)
    total_repl = torch.zeros(R, dtype=torch.long)

    # conditional torsion accumulator: per R-bin joint (phi1,phi2) histogram
    do_cond = collect_conditional and n_dih >= 2
    if do_cond:
        g1c, g2c, dphi1c, dphi2c = d2.torus_grid(sim.n_grid2, sim.n_grid2, device=device, dtype=dtype)
        cond_hist = torch.zeros(R, sim.n_rbins, sim.n_grid2, sim.n_grid2, device=device, dtype=dtype)
        cond_edges = torch.linspace(sim.R_lo, sim.R_hi, sim.n_rbins + 1, device=device, dtype=dtype)
        atoms2 = (1, 2, 3, 4)
    # extension basins: compact / intermediate / extended by R terciles of the support
    R_c1 = sim.R_lo + (sim.R_hi - sim.R_lo) / 3.0
    R_c2 = sim.R_lo + 2.0 * (sim.R_hi - sim.R_lo) / 3.0
    prev_ext = None
    trans_counts = torch.zeros(R, dtype=torch.long, device=device)
    has_left = torch.zeros(R, N, dtype=torch.bool, device=device)
    rep_roundtrips = torch.zeros(R, N, dtype=torch.long, device=device)
    first_discovery = {k: torch.full((R,), -1, dtype=torch.long, device=device)
                       for k in ("compact", "intermediate", "extended")}
    birth_hist = torch.zeros(R, sim.n_grid, device=device, dtype=dtype)
    death_hist = torch.zeros(R, sim.n_grid, device=device, dtype=dtype)
    score_std_sum = np.zeros(R); score_absmax = np.zeros(R); n_score = 0

    diag = {k: [] for k in ["steps", "times", "mean_force", "pmf", "p_hat", "q_target",
                            "eff_counts", "ancestor_ess", "n_unique_ancestor",
                            "max_ancestor_frac", "repl_cumulative", "pq_l2", "kl_pq",
                            "frac_compact", "frac_inter", "frac_extended"]}
    t0 = time.perf_counter()
    for step in range(sim.n_steps + 1):
        qf = q.reshape(R * N, A, 3)
        F = pot.forces(qf, params)
        f_loc, R_f, grad_f = cv.local_mean_force(qf, F, beta)
        Rv = R_f.reshape(R, N)
        f_loc = torch.clamp(f_loc, -sim.abf_force_clip * 8, sim.abf_force_clip * 8).reshape(R, N)
        grad_full = grad_f.reshape(R, N, A, 3)

        fsum += iv.bin_sum(Rv, f_loc, sim.n_grid, sim.R_lo, sim.R_hi)
        csum += iv.bin_counts(Rv, sim.n_grid, sim.R_lo, sim.R_hi)
        if step >= sim.estimator_burn_in_steps:
            fsum_prod += iv.bin_sum(Rv, f_loc, sim.n_grid, sim.R_lo, sim.R_hi)
            csum_prod += iv.bin_counts(Rv, sim.n_grid, sim.R_lo, sim.R_hi)
            if do_cond:
                phi1 = geom.signed_dihedral(qf, 0, 1, 2, 3).reshape(R, N)
                phi2 = geom.signed_dihedral(qf, 1, 2, 3, 4).reshape(R, N)
                bin_id = (torch.bucketize(Rv, cond_edges) - 1).clamp(0, sim.n_rbins - 1)
                i1 = torch.floor((phi1 + PI) / dphi1c).long().clamp(0, sim.n_grid2 - 1)
                i2 = torch.floor((phi2 + PI) / dphi2c).long().clamp(0, sim.n_grid2 - 1)
                lin = bin_id * (sim.n_grid2 * sim.n_grid2) + i1 * sim.n_grid2 + i2
                cond_hist.view(R, -1).scatter_add_(1, lin, torch.ones_like(phi1))

        mf_profile = iv.mean_force_profile(fsum, csum, K_abf)
        A_hat = iv.free_energy_from_mean_force(mf_profile, grid, dz)
        ramp = min(1.0, step / max(sim.abf_warmup_steps, 1))
        abf_scale = sim.abf_bias_scale * ramp
        # `fullSamples` guard on the APPLIED bias only; the stored estimator keeps the full mean
        # force.  `abf_min_count <= 0` is a no-op and reproduces frozen v1 bit-for-bit.
        if sim.abf_min_count > 0.0:
            trust = (iv.effective_counts(csum, K_abf) / sim.abf_min_count).clamp(0.0, 1.0)
            mf_bias_profile = mf_profile * trust
        else:
            mf_bias_profile = mf_profile
        B_n = abf_scale * iv.free_energy_from_mean_force(mf_bias_profile, grid, dz)
        mf_at = iv.interval_interp(mf_bias_profile, grid, Rv).clamp(-sim.abf_force_clip,
                                                                    sim.abf_force_clip)
        bias_at = abf_scale * mf_at
        wall_g = _wall_gforce(Rv, sim)
        applied = (bias_at + wall_g)
        bias_force = dist_bias_force(grad_full.reshape(R * N, A, 3),
                                     applied.reshape(R * N)).reshape(R, N, A, 3)

        if method in ESTIMATED_TARGET_METHODS and (step + 1) >= sim.fr_start_steps:
            if F_ema is None:
                F_ema = A_hat.clone()
            else:
                rt = sim.target_ema_rate
                F_ema = (1 - rt) * F_ema + rt * A_hat
            F_ema = F_ema - F_ema.mean(-1, keepdim=True)

        # extension-basin bookkeeping
        cur_ext = torch.zeros_like(Rv, dtype=torch.long)
        cur_ext = torch.where(Rv >= R_c2, torch.ones_like(cur_ext), cur_ext)      # extended=1
        cur_ext = torch.where(Rv <= R_c1, 2 * torch.ones_like(cur_ext), cur_ext)  # compact=2 else inter=0
        if prev_ext is not None:
            trans_counts += (cur_ext != prev_ext).sum(-1)
            is_ext = cur_ext == 1
            rep_roundtrips += (has_left & is_ext).long()
        has_left = torch.where(cur_ext == 1, torch.zeros_like(has_left), has_left | (cur_ext == 2))
        prev_ext = cur_ext
        for nm, msk in (("compact", Rv <= R_c1), ("intermediate", (Rv > R_c1) & (Rv < R_c2)),
                        ("extended", Rv >= R_c2)):
            seen = msk.any(-1)
            fd = first_discovery[nm]
            first_discovery[nm] = torch.where((fd < 0) & seen, torch.full_like(fd, step), fd)

        if step % sim.save_every == 0 or step == sim.n_steps:
            rep_est_f = fsum_prod if csum_prod.sum() > 0 else fsum
            rep_est_c = csum_prod if csum_prod.sum() > 0 else csum
            mf_rep = iv.mean_force_profile(rep_est_f, rep_est_c, K_abf)
            pmf_rep = iv.free_energy_from_mean_force(mf_rep, grid, dz)
            diag["steps"].append(step); diag["times"].append(step * sim.dt)
            diag["mean_force"].append(mf_rep.cpu().numpy())
            diag["pmf"].append(pmf_rep.cpu().numpy())
            diag["eff_counts"].append(iv.effective_counts(csum, K_abf).cpu().numpy())
            diag["repl_cumulative"].append(total_repl.numpy().copy())
            p_grid = iv.kde_marginal(Rv, K_kde, sim.n_grid, dz, sim.R_lo, sim.R_hi)
            diag["p_hat"].append(p_grid.cpu().numpy())
            q_grid = _fr_target(method, grid, dz, sim.R_lo, sim.R_hi, F_ema, B_n, oracle, beta)
            if q_grid is not None:
                lr = torch.log(p_grid.clamp_min(EPS)) - torch.log(q_grid.clamp_min(EPS))
                diag["q_target"].append(q_grid.cpu().numpy())
                diag["pq_l2"].append((((p_grid - q_grid) ** 2).sum(-1) * dz).sqrt().cpu().numpy())
                diag["kl_pq"].append(((p_grid * lr).sum(-1) * dz).cpu().numpy())
            else:
                diag["q_target"].append(np.full((R, sim.n_grid), np.nan))
                diag["pq_l2"].append(np.full(R, np.nan)); diag["kl_pq"].append(np.full(R, np.nan))
            if is_fr:
                ess, nuq, maxf = _ancestor_stats(ancestors.cpu(), N)
            else:
                ess, nuq, maxf = np.full(R, np.nan), np.full(R, N), np.full(R, np.nan)
            diag["ancestor_ess"].append(ess); diag["n_unique_ancestor"].append(nuq)
            diag["max_ancestor_frac"].append(maxf)
            diag["frac_compact"].append((cur_ext == 2).float().mean(-1).cpu().numpy())
            diag["frac_inter"].append((cur_ext == 0).float().mean(-1).cpu().numpy())
            diag["frac_extended"].append((cur_ext == 1).float().mean(-1).cpu().numpy())

        if step == sim.n_steps:
            break

        noise = torch.randn(q.shape, generator=gen_dyn, device=device, dtype=dtype)
        q = q + sim.dt * (F.reshape(R, N, A, 3) + bias_force) + noise_scale * noise
        q = geom.remove_com(q)

        if is_fr:
            nxt = step + 1
            if nxt >= sim.fr_start_steps and (nxt - sim.fr_start_steps) % max(int(sim.fr_every), 1) == 0:
                R_new = cv.value(q.reshape(R * N, A, 3)).reshape(R, N)
                q_grid = _fr_target(method, grid, dz, sim.R_lo, sim.R_hi, F_ema, B_n, oracle, beta)
                if q_grid is not None:
                    score, p_fr, kl = _fr_score(R_new, grid, dz, sim.R_lo, sim.R_hi, K_kde,
                                                q_grid, sim.score_clip)
                    ss = score.detach().cpu().numpy()
                    score_std_sum += ss.std(axis=1); score_absmax = np.maximum(score_absmax, np.abs(ss).max(axis=1))
                    n_score += 1
                    q, ancestors, n_repl, deaths, births = _birth_death(q, score, ancestors, sim, gen_fr)
                    total_repl += n_repl.cpu()
                    for r in range(R):
                        if deaths[r] is not None and deaths[r].numel() > 0:
                            zb = R_new[r].index_select(0, births[r])
                            zd = R_new[r].index_select(0, deaths[r])
                            birth_hist[r] += iv.bin_counts(zb[None, :], sim.n_grid, sim.R_lo, sim.R_hi)[0]
                            death_hist[r] += iv.bin_counts(zd[None, :], sim.n_grid, sim.R_lo, sim.R_hi)[0]
                            for arr in (has_left, rep_roundtrips):
                                arr[r, deaths[r]] = arr[r].index_select(0, births[r])
                            if prev_ext is not None:
                                prev_ext[r, deaths[r]] = prev_ext[r].index_select(0, births[r])

    out = {"method": method, "grid": grid.cpu().numpy(), "dz": float(dz),
           "R_lo": sim.R_lo, "R_hi": sim.R_hi,
           "runtime_seconds": time.perf_counter() - t0,
           "total_replacement_events": total_repl.numpy(),
           "n_transitions": trans_counts.cpu().numpy(),
           "n_round_trips": rep_roundtrips.sum(-1).cpu().numpy(),
           "first_discovery": {k: v.cpu().numpy() for k, v in first_discovery.items()},
           "birth_hist": birth_hist.cpu().numpy(), "death_hist": death_hist.cpu().numpy(),
           "F_target_ema": (F_ema.cpu().numpy() if F_ema is not None else None),
           "fr_score_std": (score_std_sum / max(n_score, 1)),
           "fr_score_absmax": score_absmax,
           "final_eff_counts": iv.effective_counts(csum, K_abf).cpu().numpy()}
    for k in diag:
        out[k] = np.asarray(diag[k])
    if do_cond:
        out["cond_hist"] = cond_hist.cpu().numpy()
        out["cond_grid1"] = g1c.cpu().numpy(); out["cond_grid2"] = g2c.cpu().numpy()
        out["cond_dphi"] = float(dphi1c)
        out["cond_edges"] = cond_edges.cpu().numpy()
    if verbose:
        print(f"  {method:12s} R={R} N={N}: {out['runtime_seconds']:.1f}s "
              f"repl={out['total_replacement_events'].sum()} trans={out['n_transitions'].sum()}")
    return out


def run_frozen_bias_dist(params: pot.AlkaneParams, sim: DistSimConfig, learned_mean_force,
                         seeds, cv: DistanceCV, device, dtype=torch.float64,
                         initial_dihedrals=None, verbose=True):
    """Frozen-bias validation for the distance CV: fix the learned bias ``B(R)``, run fresh
    dynamics with NO ABF update and NO birth--death (soft walls kept identical), accumulate
    the biased marginal ``p_B(R)``, reconstruct ``F_recon = B - beta^{-1} log p_B + C``.
    ``learned_mean_force`` is ``(n_grid,)`` or ``(R, n_grid)``. No reference is read.
    """
    R = len(seeds); N = sim.n_replicas; A = params.n_atoms; n_dih = params.n_dihedrals
    beta = params.beta
    grid, dz = iv.interval_grid(sim.n_grid, sim.R_lo, sim.R_hi, device=device, dtype=dtype)
    K_kde = iv.reflected_kernel_matrix(grid, sim.kde_bandwidth, sim.R_lo, sim.R_hi)
    mf = torch.as_tensor(np.asarray(learned_mean_force), device=device, dtype=dtype)
    if mf.ndim == 1:
        mf = mf[None, :].expand(R, -1).contiguous()
    B_grid = iv.free_energy_from_mean_force(mf, grid, dz)
    gen = torch.Generator(device=device).manual_seed(int(sim.rng_seed) + 555)
    if initial_dihedrals is None:
        init = torch.zeros(R, N, n_dih, device=device, dtype=dtype)
    else:
        d0 = torch.as_tensor(initial_dihedrals, device=device, dtype=dtype).reshape(1, 1, n_dih)
        init = d0.expand(R, N, n_dih).clone()
    q = geom.place_chain(init.reshape(R * N, n_dih), A, d0=params.d0, theta0=params.theta0,
                         device=device, dtype=dtype).reshape(R, N, A, 3)
    q = geom.remove_com(q + 1e-3 * torch.randn(q.shape, generator=gen, device=device, dtype=dtype))
    noise_scale = math.sqrt(2.0 * sim.dt / beta)
    p_accum = torch.zeros(R, sim.n_grid, device=device, dtype=dtype); n_marg = 0
    t0 = time.perf_counter()
    for step in range(sim.n_steps + 1):
        qf = q.reshape(R * N, A, 3)
        F = pot.forces(qf, params)
        Rv0, grad_full, _ = cv.geometry(qf)
        Rv = Rv0.reshape(R, N)
        bias_at = iv.interval_interp(mf, grid, Rv).clamp(-sim.abf_force_clip, sim.abf_force_clip)
        wall_g = _wall_gforce(Rv, sim)
        bias_force = dist_bias_force(grad_full, (bias_at + wall_g).reshape(R * N)).reshape(R, N, A, 3)
        if step >= sim.estimator_burn_in_steps:
            p_accum += iv.kde_marginal(Rv, K_kde, sim.n_grid, dz, sim.R_lo, sim.R_hi); n_marg += 1
        if step == sim.n_steps:
            break
        noise = torch.randn(q.shape, generator=gen, device=device, dtype=dtype)
        q = geom.remove_com(q + sim.dt * (F.reshape(R, N, A, 3) + bias_force) + noise_scale * noise)
    p_B = iv.normalize_density(p_accum / max(n_marg, 1), dz)
    F_recon = B_grid - (1.0 / beta) * torch.log(p_B.clamp_min(EPS))
    F_recon = F_recon - F_recon.mean(-1, keepdim=True)
    if verbose:
        print(f"  dist frozen-bias R={R}: {time.perf_counter()-t0:.1f}s")
    return {"grid": grid.cpu().numpy(), "p_B": p_B.cpu().numpy(),
            "F_recon": F_recon.cpu().numpy(), "B": B_grid.cpu().numpy(),
            "learned_mean_force": mf.cpu().numpy()}
