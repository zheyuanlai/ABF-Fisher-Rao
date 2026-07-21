"""Batched 2-D ABF (+ Fisher--Rao birth--death) sampler on the torus CV
``xi = (phi1, phi2)`` for pentane (the mandatory two-dimensional extension).

The same joint CV drives ABF *and* FR (``xi_ABF = xi_FR = (phi1,phi2)``), so FR cannot
alter ``p(phi2|phi1)`` behind ABF's back.  Each step:

  1. exact vector local mean force ``f_a`` (den Otter; :mod:`alkanes.cv2d`);
  2. scatter ``count, f1, f2`` onto the torus grid;
  3. separable wrapped-Gaussian smoothing -> raw mean-force fields ``g=(g1,g2)``;
  4. FFT Poisson (Hodge) projection ``g -> B`` (conservative scalar bias) and ``grad B``;
  5. apply the conservative bias force ``+ sum_a (grad B)_a grad phi_a``;
  6. (FR) build the joint marginal ``p_hat``, target ``q``, centred 2-D score, and
     fixed-N kill-and-clone of whole molecular configurations.

Methods: ``abf``, ``fr_estimated`` (deployable), ``fr_uniform`` (ablation), ``fr_oracle``
(diagnostic; the only method that may see the reference).  Birth--death / ancestor / score
/ no-leakage helpers are reused from :mod:`alkanes.core`.  ``q`` is ``(R,N,n_atoms,3)``.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, asdict

import numpy as np
import torch

from . import geometry as geom
from . import potentials as pot
from . import density2d as d2
from . import poisson2d as ps
from .cv2d import JointDihedralCV2D, abf_bias_force_2d
from .core import (_birth_death, _ancestor_stats, assert_no_reference_leakage,
                   FR_METHODS, ESTIMATED_TARGET_METHODS, ALL_METHODS)

EPS = 1.0e-12
PI = math.pi
TWO_PI = 2.0 * math.pi


@dataclass
class Sim2DConfig:
    dt: float = 5.0e-4
    n_steps: int = 100_000
    n_replicas: int = 2048
    save_every: int = 5_000
    rng_seed: int = 20260719
    # torus grid / estimator
    n_grid: int = 48                 # per-axis (square torus)
    abf_bandwidth: float = 0.20      # radians (separable smoothing of the mean-force field)
    kde_bandwidth: float = 0.30      # radians (joint marginal p_hat for FR)
    # ABF application
    abf_bias_scale: float = 1.0
    abf_warmup_steps: int = 5_000
    abf_force_clip: float = 60.0
    abf_min_count: float = 5.0       # cell support floor before its bias is trusted
    estimator_burn_in_steps: int = 6_000
    estimator_stride: int = 1        # accumulate the (Hessian) mean force every k steps
    # Fisher--Rao
    fr_rate: float = 0.01
    score_clip: float = 2.0
    fr_start_steps: int = 10_000
    fr_every: int = 5
    target_ema_rate: float = 0.005
    max_event_fraction: float = 0.005
    density_ema: float = 0.0         # 0 => instantaneous p_hat; >0 => EMA joint marginal
    # basins
    basin_barrier: float = math.radians(61.6)

    def config_hash(self):
        import hashlib, json
        return hashlib.md5(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]


def _basin1d(phi, barrier):
    idx = torch.zeros_like(phi, dtype=torch.long)
    idx = torch.where(phi >= barrier, torch.ones_like(idx), idx)
    idx = torch.where(phi <= -barrier, 2 * torch.ones_like(idx), idx)
    return idx


def _joint_basin(phi1, phi2, barrier):
    return 3 * _basin1d(phi1, barrier) + _basin1d(phi2, barrier)     # 0..8


def _project_bias(f1_sum, f2_sum, count, K1, K2, dz1, dz2, min_count):
    """Smoothed mean-force field -> conservative bias ``B`` and ``grad B`` via Poisson.

    Cells with < ``min_count`` smoothed support contribute no force (masked to 0 before
    projection), so the projection is not driven by empty-cell noise.
    """
    g1, g2, den = d2.mean_force_fields(f1_sum, f2_sum, count, K1, K2)
    trust = (den >= min_count)
    g1 = torch.where(trust, g1, torch.zeros_like(g1))
    g2 = torch.where(trust, g2, torch.zeros_like(g2))
    B, gB1, gB2 = ps.poisson_projection(g1, g2, dz1, dz2)
    return B, gB1, gB2, g1, g2


def _fr_target(method, F_ema, B, oracle, beta, dz1, dz2):
    if method == "abf":
        return None
    if method == "fr_uniform":
        R, n1, n2 = B.shape
        return d2.normalize2(torch.ones(R, n1, n2, device=B.device, dtype=B.dtype), dz1, dz2)
    if method == "fr_oracle":
        log_q = -beta * (oracle[None] - B)
        log_q = log_q - log_q.amax(dim=(-2, -1), keepdim=True)
        return d2.normalize2(torch.exp(log_q), dz1, dz2)
    if F_ema is None:
        return None
    log_q = -beta * (F_ema - B)
    log_q = log_q - log_q.amax(dim=(-2, -1), keepdim=True)
    return d2.normalize2(torch.exp(log_q), dz1, dz2)


def run_sampler_2d(method, params: pot.AlkaneParams, sim: Sim2DConfig, seeds,
                   cv: JointDihedralCV2D, device, dtype=torch.float64, initial_dihedrals=None,
                   oracle_free_energy=None, verbose=True):
    """Run ``R=len(seeds)`` matched-seed replicas of ``method`` on the 2-D torsion CV."""
    if method not in ALL_METHODS:
        raise ValueError(f"unknown method {method!r}")
    assert_no_reference_leakage(method, oracle_free_energy)
    is_fr = method in FR_METHODS
    R = len(seeds); N = sim.n_replicas; A = params.n_atoms; n_dih = params.n_dihedrals
    assert n_dih >= 2, "2-D CV requires pentane (>=2 dihedrals)"
    beta = params.beta
    n1 = n2 = sim.n_grid
    g1c, g2c, dz1, dz2 = d2.torus_grid(n1, n2, device=device, dtype=dtype)
    K1, K2 = d2.kernels(g1c, g2c, sim.abf_bandwidth, sim.abf_bandwidth)
    Kk1, Kk2 = d2.kernels(g1c, g2c, sim.kde_bandwidth, sim.kde_bandwidth)
    gen_dyn = torch.Generator(device=device).manual_seed(int(sim.rng_seed))
    gen_fr = torch.Generator(device=device).manual_seed(int(sim.rng_seed) + 987654321)

    oracle = None
    if method == "fr_oracle":
        oracle = torch.as_tensor(oracle_free_energy, device=device, dtype=dtype)
        oracle = oracle - oracle.mean()

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
    f1s = torch.zeros(R, n1, n2, device=device, dtype=dtype)
    f2s = torch.zeros(R, n1, n2, device=device, dtype=dtype)
    csum = torch.zeros(R, n1, n2, device=device, dtype=dtype)
    joint_hist = torch.zeros(R, n1, n2, device=device, dtype=dtype)      # post burn-in
    F_ema = None
    p_ema = None
    ancestors = (torch.arange(N, device=device).expand(R, N).clone() if is_fr else None)
    total_repl = torch.zeros(R, dtype=torch.long)

    # basin bookkeeping (9 joint basins)
    prev_basin = None
    trans_counts = torch.zeros(R, dtype=torch.long, device=device)
    trans_matrix = torch.zeros(R, 9, 9, dtype=torch.long, device=device)
    has_left_TT = torch.zeros(R, N, dtype=torch.bool, device=device)
    rep_roundtrips = torch.zeros(R, N, dtype=torch.long, device=device)
    first_discovery = torch.full((R, 9), -1, dtype=torch.long, device=device)
    birth_hist = torch.zeros(R, n1, n2, device=device, dtype=dtype)
    death_hist = torch.zeros(R, n1, n2, device=device, dtype=dtype)
    # Gram / curl running diagnostics (accumulated ON DEVICE; no per-step host sync)
    cond_sum_g = torch.zeros(R, device=device, dtype=dtype)
    cond_max_g = torch.zeros(R, device=device, dtype=dtype)
    lam_min_min_g = torch.full((R,), float("inf"), device=device, dtype=dtype)
    n_cfg = 0
    score_std_sum = np.zeros(R); score_absmax = np.zeros(R); n_score = 0

    diag = {k: [] for k in ["steps", "times", "pmf", "p_hat", "q_target", "curl_pre",
                            "ancestor_ess", "n_unique_ancestor", "max_ancestor_frac",
                            "repl_cumulative", "kl_pq", "n_basins_visited",
                            "gram_cond_mean", "gram_cond_max"]}
    # cached bias fields (recomputed only on estimator steps; applied every step)
    B_raw = torch.zeros(R, n1, n2, device=device, dtype=dtype)
    gB1 = torch.zeros(R, n1, n2, device=device, dtype=dtype)
    gB2 = torch.zeros(R, n1, n2, device=device, dtype=dtype)
    g1f = torch.zeros(R, n1, n2, device=device, dtype=dtype)
    g2f = torch.zeros(R, n1, n2, device=device, dtype=dtype)
    stride = max(int(sim.estimator_stride), 1)
    t0 = time.perf_counter()
    for step in range(sim.n_steps + 1):
        qf = q.reshape(R * N, A, 3)
        F = pot.forces(qf, params)
        # --- mean-force estimator accumulation (strided: the expensive Hessian path) ---
        if step % stride == 0:
            f, phi, gfull, geo = cv.local_mean_force(qf, F, beta)  # f,phi (RN,2), g (RN,2,A,3)
            phi1 = phi[:, 0].reshape(R, N); phi2 = phi[:, 1].reshape(R, N)
            f = torch.clamp(f, -sim.abf_force_clip * 8, sim.abf_force_clip * 8)
            f1 = f[:, 0].reshape(R, N); f2 = f[:, 1].reshape(R, N)
            cnd = geo["cond"].reshape(R, N)
            cond_sum_g += cnd.mean(1); cond_max_g = torch.maximum(cond_max_g, cnd.amax(1))
            lam_min_min_g = torch.minimum(lam_min_min_g, geo["lam_min"].reshape(R, N).amin(1))
            n_cfg += 1
            f1s += d2.scatter_sum(phi1, phi2, f1, n1, n2, dz1, dz2)
            f2s += d2.scatter_sum(phi1, phi2, f2, n1, n2, dz1, dz2)
            csum += d2.scatter_counts(phi1, phi2, n1, n2, dz1, dz2)
            B_raw, gB1, gB2, g1f, g2f = _project_bias(f1s, f2s, csum, K1, K2, dz1, dz2, sim.abf_min_count)
            if method in ESTIMATED_TARGET_METHODS and (step + 1) >= sim.fr_start_steps:
                if F_ema is None:
                    F_ema = B_raw.clone()
                else:
                    rt = sim.target_ema_rate
                    F_ema = (1 - rt) * F_ema + rt * B_raw
                F_ema = F_ema - F_ema.mean(dim=(-2, -1), keepdim=True)
        else:
            phi, gfull = cv.grad_only(qf)                          # cheap: gradient only
            phi1 = phi[:, 0].reshape(R, N); phi2 = phi[:, 1].reshape(R, N)

        # --- bias force from the cached projected field (applied every step) ---
        ramp = min(1.0, step / max(sim.abf_warmup_steps, 1))
        abf_scale = sim.abf_bias_scale * ramp
        B = abf_scale * B_raw
        gB1s = abf_scale * gB1; gB2s = abf_scale * gB2
        bias_at1 = d2.bilinear_interp2(gB1s, g1c, g2c, dz1, dz2, phi1, phi2).clamp(-sim.abf_force_clip, sim.abf_force_clip)
        bias_at2 = d2.bilinear_interp2(gB2s, g1c, g2c, dz1, dz2, phi1, phi2).clamp(-sim.abf_force_clip, sim.abf_force_clip)
        bias_at = torch.stack([bias_at1.reshape(R * N), bias_at2.reshape(R * N)], dim=-1)  # (RN,2)
        bias_force = abf_bias_force_2d(gfull, bias_at).reshape(R, N, A, 3)
        if step >= sim.estimator_burn_in_steps:
            joint_hist += d2.scatter_counts(phi1, phi2, n1, n2, dz1, dz2)

        # basin bookkeeping
        cur_basin = _joint_basin(phi1, phi2, sim.basin_barrier)        # (R,N)
        cur_TT = cur_basin == 0
        if prev_basin is not None:
            changed = cur_basin != prev_basin
            trans_counts += changed.sum(-1)
            # transition matrix, fully vectorised (no nonzero/host sync): add `changed`
            # (0/1) at (r, prev, cur); unchanged rows land on the diagonal with weight 0.
            r_idx = torch.arange(R, device=device)[:, None].expand(R, N)
            lin = (r_idx * 81 + prev_basin * 9 + cur_basin).reshape(-1)
            trans_matrix.view(-1).scatter_add_(0, lin, changed.reshape(-1).long())
            rep_roundtrips += (has_left_TT & cur_TT).long()
        has_left_TT = torch.where(cur_TT, torch.zeros_like(has_left_TT), has_left_TT | (~cur_TT))
        prev_basin = cur_basin
        for b in range(9):
            seen = (cur_basin == b).any(-1)
            fd = first_discovery[:, b]
            first_discovery[:, b] = torch.where((fd < 0) & seen, torch.full_like(fd, step), fd)

        if step % sim.save_every == 0 or step == sim.n_steps:
            p_hat = d2.kde2(phi1, phi2, Kk1, Kk2, n1, n2, dz1, dz2)
            diag["steps"].append(step); diag["times"].append(step * sim.dt)
            diag["pmf"].append(B_raw.cpu().numpy())
            diag["p_hat"].append(p_hat.cpu().numpy())
            diag["curl_pre"].append(ps.curl_norm(g1f, g2f, dz1, dz2).cpu().numpy())
            diag["gram_cond_mean"].append((cond_sum_g / max(n_cfg, 1)).cpu().numpy())
            diag["gram_cond_max"].append(cond_max_g.cpu().numpy())
            q_grid = _fr_target(method, F_ema, B, oracle, beta, dz1, dz2)
            if q_grid is not None:
                diag["q_target"].append(q_grid.cpu().numpy())
                diag["kl_pq"].append(d2.kl2(p_hat, q_grid, dz1, dz2).cpu().numpy())
            else:
                diag["q_target"].append(np.full((R, n1, n2), np.nan))
                diag["kl_pq"].append(np.full(R, np.nan))
            if is_fr:
                ess, nuq, maxf = _ancestor_stats(ancestors.cpu(), N)
            else:
                ess, nuq, maxf = np.full(R, np.nan), np.full(R, N), np.full(R, np.nan)
            diag["ancestor_ess"].append(ess); diag["n_unique_ancestor"].append(nuq)
            diag["max_ancestor_frac"].append(maxf)
            diag["repl_cumulative"].append(total_repl.numpy().copy())
            nb = np.array([(first_discovery[r] >= 0).sum().item() for r in range(R)])
            diag["n_basins_visited"].append(nb)

        if step == sim.n_steps:
            break

        noise = torch.randn(q.shape, generator=gen_dyn, device=device, dtype=dtype)
        q = q + sim.dt * (F.reshape(R, N, A, 3) + bias_force) + noise_scale * noise
        q = geom.remove_com(q)

        if is_fr:
            nxt = step + 1
            if nxt >= sim.fr_start_steps and (nxt - sim.fr_start_steps) % max(int(sim.fr_every), 1) == 0:
                ph = cv.values(q.reshape(R * N, A, 3))
                p1n = ph[0].reshape(R, N); p2n = ph[1].reshape(R, N)
                p_inst = d2.kde2(p1n, p2n, Kk1, Kk2, n1, n2, dz1, dz2)
                if sim.density_ema > 0:
                    p_ema = p_inst if p_ema is None else (1 - sim.density_ema) * p_ema + sim.density_ema * p_inst
                    p_hat = d2.normalize2(p_ema, dz1, dz2)
                else:
                    p_hat = p_inst
                q_grid = _fr_target(method, F_ema, B, oracle, beta, dz1, dz2)
                if q_grid is not None:
                    score, kl = d2.fr_score_2d(p1n, p2n, p_hat, q_grid, g1c, g2c, dz1, dz2, sim.score_clip)
                    ss = score.detach().cpu().numpy()
                    score_std_sum += ss.std(axis=1); score_absmax = np.maximum(score_absmax, np.abs(ss).max(axis=1))
                    n_score += 1
                    q, ancestors, n_repl, deaths, births = _birth_death(q, score, ancestors, sim, gen_fr)
                    total_repl += n_repl.cpu()
                    for r in range(R):
                        if deaths[r] is not None and deaths[r].numel() > 0:
                            zb1 = p1n[r].index_select(0, births[r]); zb2 = p2n[r].index_select(0, births[r])
                            zd1 = p1n[r].index_select(0, deaths[r]); zd2 = p2n[r].index_select(0, deaths[r])
                            birth_hist[r] += d2.scatter_counts(zb1[None], zb2[None], n1, n2, dz1, dz2)[0]
                            death_hist[r] += d2.scatter_counts(zd1[None], zd2[None], n1, n2, dz1, dz2)[0]
                            for arr in (has_left_TT, rep_roundtrips):
                                arr[r, deaths[r]] = arr[r].index_select(0, births[r])
                            if prev_basin is not None:
                                prev_basin[r, deaths[r]] = prev_basin[r].index_select(0, births[r])

    out = {"method": method, "grid1": g1c.cpu().numpy(), "grid2": g2c.cpu().numpy(),
           "dz1": float(dz1), "dz2": float(dz2), "n_grid": n1,
           "runtime_seconds": time.perf_counter() - t0,
           "total_replacement_events": total_repl.numpy(),
           "n_transitions": trans_counts.cpu().numpy(),
           "n_round_trips": rep_roundtrips.sum(-1).cpu().numpy(),
           "trans_matrix": trans_matrix.cpu().numpy(),
           "first_discovery": first_discovery.cpu().numpy(),
           "birth_hist": birth_hist.cpu().numpy(), "death_hist": death_hist.cpu().numpy(),
           "joint_hist": joint_hist.cpu().numpy(),
           "F_target_ema": (F_ema.cpu().numpy() if F_ema is not None else None),
           "fr_score_std": (score_std_sum / max(n_score, 1)), "fr_score_absmax": score_absmax,
           "final_pmf": (B_raw.cpu().numpy()),
           "gram_reg_activations": int(cv.reg_activation_count()),
           "gram_lam_min_min": lam_min_min_g.cpu().numpy()}
    for k in diag:
        out[k] = np.asarray(diag[k])
    if verbose:
        print(f"  2D {method:12s} R={R} N={N}: {out['runtime_seconds']:.1f}s "
              f"repl={out['total_replacement_events'].sum()} trans={out['n_transitions'].sum()} "
              f"reg={out['gram_reg_activations']}")
    return out


def run_frozen_bias_2d(params, sim: Sim2DConfig, learned_pmf, seeds, cv, device,
                       dtype=torch.float64, initial_dihedrals=None, verbose=True):
    """Frozen-bias validation: fix the learned 2-D bias ``B(z)``, run fresh dynamics with no
    ABF update and no birth--death, accumulate the joint biased marginal ``p_B``, reconstruct
    ``F_recon = B - beta^{-1} log p_B + C``.  ``learned_pmf`` is ``(n1,n2)`` or ``(R,n1,n2)``.
    """
    R = len(seeds); N = sim.n_replicas; A = params.n_atoms; n_dih = params.n_dihedrals
    beta = params.beta; n1 = n2 = sim.n_grid
    g1c, g2c, dz1, dz2 = d2.torus_grid(n1, n2, device=device, dtype=dtype)
    Kk1, Kk2 = d2.kernels(g1c, g2c, sim.kde_bandwidth, sim.kde_bandwidth)
    B = torch.as_tensor(np.asarray(learned_pmf), device=device, dtype=dtype)
    if B.ndim == 2:
        B = B[None].expand(R, -1, -1).contiguous()
    B = B - B.mean(dim=(-2, -1), keepdim=True)
    gB1, gB2 = ps.spectral_gradient(B, dz1, dz2)
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
    p_accum = torch.zeros(R, n1, n2, device=device, dtype=dtype); n_marg = 0
    t0 = time.perf_counter()
    for step in range(sim.n_steps + 1):
        qf = q.reshape(R * N, A, 3)
        F = pot.forces(qf, params)
        geo = cv.geometry(qf)
        phi = geo["phi"]; gfull = geo["g"]
        phi1 = phi[:, 0].reshape(R, N); phi2 = phi[:, 1].reshape(R, N)
        b1 = d2.bilinear_interp2(gB1, g1c, g2c, dz1, dz2, phi1, phi2).clamp(-sim.abf_force_clip, sim.abf_force_clip)
        b2 = d2.bilinear_interp2(gB2, g1c, g2c, dz1, dz2, phi1, phi2).clamp(-sim.abf_force_clip, sim.abf_force_clip)
        bias_at = torch.stack([b1.reshape(R * N), b2.reshape(R * N)], dim=-1)
        bias_force = abf_bias_force_2d(gfull, bias_at).reshape(R, N, A, 3)
        if step >= sim.estimator_burn_in_steps:
            p_accum += d2.kde2(phi1, phi2, Kk1, Kk2, n1, n2, dz1, dz2); n_marg += 1
        if step == sim.n_steps:
            break
        noise = torch.randn(q.shape, generator=gen, device=device, dtype=dtype)
        q = geom.remove_com(q + sim.dt * (F.reshape(R, N, A, 3) + bias_force) + noise_scale * noise)
    p_B = d2.normalize2(p_accum / max(n_marg, 1), dz1, dz2)
    F_recon = B - (1.0 / beta) * torch.log(p_B.clamp_min(EPS))
    F_recon = F_recon - F_recon.mean(dim=(-2, -1), keepdim=True)
    if verbose:
        print(f"  2D frozen-bias R={R}: {time.perf_counter()-t0:.1f}s")
    return {"grid1": g1c.cpu().numpy(), "grid2": g2c.cpu().numpy(),
            "p_B": p_B.cpu().numpy(), "F_recon": F_recon.cpu().numpy(), "B": B.cpu().numpy()}
