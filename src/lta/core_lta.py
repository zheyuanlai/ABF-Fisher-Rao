"""Ethane in rigid all-silica LTA: the campaign's molecular entropic-barrier system.

Physics (frozen in configs/uniform_campaign/lta_prereg.json before production):

  * framework   IZA DLS76-optimized SiO2 LTA (cache/lta/framework.npz), RIGID,
                2x2x2 pseudo-cells, box L = 2a with a = 11.919 A, PBC.  Only the
                384 framework O atoms interact with the guest (standard for
                alkane-zeolite force fields; Si is screened).
  * guest       one TraPPE-UA ethane per replica: two CH3 pseudo-atoms, harmonic
                bond r0 = 1.54 A (stiff k, BD-integrable stand-in for TraPPE's
                fixed bond), no charges.
  * cross LJ    CH3-O: eps/kB = 93.0 K, sigma = 3.48 A (TraPPE-zeolite family,
                Dubbeldam et al.), truncated+shifted at rc = 10 A < L/2.
  * dynamics    overdamped Langevin, q <- q + dt*F + sqrt(2 dt / beta) eta --
                the SAME convention as the alkanes/WCA engines.  Equilibrium
                (hence F(z), U(z), -TS(z)) is independent of this choice; the
                campaign budget axis is force evaluations, not physical ps.
  * CV          phi = wrap((2 pi / a) * x_COM) in [-pi, pi): the ethane COM
                position along the cage-center line, mapped onto the circle so
                the WINDOW plane (x = 0 mod a) sits at phi = 0 and the
                alpha-cage centers at phi = +-pi.  z = phi * a / (2 pi) in
                [-a/2, a/2).  The CV is LINEAR in coordinates, so the den Otter
                geometric term vanishes identically.

The ABF estimator, KDE marginal, FR score, birth-death and genealogy machinery
are IMPORTED from the closed alkanes engine (same periodic-circle conventions),
not re-implemented: `alkanes.periodic` and `alkanes.core._fr_target/_fr_score/
_birth_death/_ancestor_stats` are the tested implementations.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from alkanes import periodic as per
from alkanes.core import (_ancestor_stats, _birth_death, _fr_score,
                          assert_no_reference_leakage)

EPS = 1.0e-12
KB = 0.008314462618        # kJ/mol/K
FR_METHODS = ("fr_estimated", "fr_uniform", "fr_oracle")
ESTIMATED_TARGET_METHODS = ("fr_estimated",)
ALL_METHODS = ("abf",) + FR_METHODS
PI = math.pi


@dataclass
class LTAParams:
    framework_npz: str = "cache/lta/framework.npz"
    temperature: float = 300.0            # K
    eps_go: float = 93.0 * KB             # CH3-O epsilon, kJ/mol
    sigma_go: float = 3.48                # A
    rc: float = 10.0                      # A
    r0_bond: float = 1.54                 # A
    k_bond: float = 400.0                 # kJ/mol/A^2  (0.5 k (r-r0)^2)
    n_beads: int = 2

    @property
    def beta(self):
        return 1.0 / (KB * self.temperature)


@dataclass
class LTASimConfig:
    dt: float = 2.0e-4
    n_steps: int = 300_000
    n_replicas: int = 1024
    save_every: int = 3_000
    rng_seed: int = 20260829
    # CV grid / estimator (dimensionless circle units, alkanes conventions)
    n_grid: int = 180
    abf_bandwidth: float = 0.05
    kde_bandwidth: float = 0.10
    abf_bias_scale: float = 1.0
    abf_warmup_steps: int = 20_000
    abf_force_clip: float = 60.0
    estimator_burn_in_steps: int = 20_000
    # Fisher--Rao (rate frozen by SAFETY-ONLY calibration before production)
    fr_rate: float = 0.10
    score_clip: float = 2.0
    fr_start_steps: int = 40_000
    fr_every: int = 5
    target_ema_rate: float = 0.005
    max_event_fraction: float = 0.02
    # region bookkeeping (diagnostics only): |z| < window_half -> window,
    # |z| > cage_min -> cage, in A
    window_half: float = 1.5
    cage_min: float = 4.0

    def config_hash(self):
        import hashlib, json
        return hashlib.md5(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]


class LTASystem:
    """Rigid framework + batched ethane physics on one device."""

    def __init__(self, params: LTAParams, device, dtype=torch.float64, root="."):
        import os
        self.p = params
        z = np.load(os.path.join(root, params.framework_npz), allow_pickle=True)
        self.a = float(z["a_pseudo"])
        self.L = float(z["box"])
        assert self.p.rc < self.L / 2, "cutoff must fit the minimum image"
        self.o_pos = torch.as_tensor(z["o_pos"], device=device, dtype=dtype)
        self.device, self.dtype = device, dtype
        # shifted LJ constant
        sr6 = (params.sigma_go / params.rc) ** 6
        self.v_rc = 4.0 * params.eps_go * (sr6 * sr6 - sr6)

    def _lj_pairs(self, q):
        """Min-image bead->O displacement d (B,2,nO,3) and r^2, cutoff mask."""
        d = q[:, :, None, :] - self.o_pos[None, None, :, :]
        d = d - self.L * torch.round(d / self.L)
        r2 = (d * d).sum(-1)
        mask = r2 < self.p.rc ** 2
        return d, r2, mask

    def forces(self, q):
        """Physical forces on the beads; ``q`` (B,2,3) UNWRAPPED. Returns (B,2,3)."""
        p = self.p
        F = torch.zeros_like(q)
        # bond (no min-image: coordinates are unwrapped, the molecule never straddles)
        dr = q[:, 0, :] - q[:, 1, :]
        r = dr.norm(dim=-1, keepdim=True).clamp_min(EPS)
        fb = -p.k_bond * (r - p.r0_bond) * dr / r
        F[:, 0, :] += fb
        F[:, 1, :] -= fb
        # LJ to framework O
        d, r2, mask = self._lj_pairs(q)
        inv_r2 = torch.where(mask, 1.0 / r2.clamp_min(EPS), torch.zeros_like(r2))
        sr6 = (p.sigma_go ** 2 * inv_r2) ** 3
        coef = 24.0 * p.eps_go * (2.0 * sr6 * sr6 - sr6) * inv_r2     # (B,2,nO)
        F += (coef[..., None] * d).sum(dim=2)
        return F

    def potential_energy(self, q):
        """Total potential (bond + shifted LJ), (B,) in kJ/mol."""
        p = self.p
        dr = q[:, 0, :] - q[:, 1, :]
        r = dr.norm(dim=-1)
        e = 0.5 * p.k_bond * (r - p.r0_bond) ** 2
        _, r2, mask = self._lj_pairs(q)
        inv_r2 = torch.where(mask, 1.0 / r2.clamp_min(EPS), torch.zeros_like(r2))
        sr6 = (p.sigma_go ** 2 * inv_r2) ** 3
        v = 4.0 * p.eps_go * (sr6 * sr6 - sr6) - self.v_rc
        e = e + torch.where(mask, v, torch.zeros_like(v)).sum(dim=(1, 2))
        return e

    # ---- CV: phi = wrap((2 pi / a) x_COM), window at phi=0, cages at +-pi ----
    def cv_value(self, q):
        x = q[..., 0].mean(dim=-1)                     # (B,) COM x (equal masses)
        phi = (2.0 * PI / self.a) * x
        return torch.remainder(phi + PI, 2.0 * PI) - PI

    def cv_local_mean_force(self, q, F):
        """(f_loc, phi, grad_full) with the alkanes sign/interface conventions.

        Linear CV: f_loc = -(F . grad phi)/|grad phi|^2, no geometric term.
        grad phi has x-components pi/a on each bead; |grad phi|^2 = 2 pi^2/a^2,
        so f_loc = -(a / 2 pi) (F1x + F2x).  grad_full is the TRUE grad(phi)
        because ``abf_bias_force`` applies +A'(phi) * grad(phi).
        """
        f_loc = -(self.a / (2.0 * PI)) * (F[:, 0, 0] + F[:, 1, 0])
        grad_full = torch.zeros_like(q)
        grad_full[:, :, 0] = PI / self.a
        return f_loc, self.cv_value(q), grad_full

    def initial_conditions(self, R, N, gen):
        """Ethane at a random alpha-cage center, random orientation, COM jitter."""
        S = round(self.L / self.a)
        cages = torch.tensor([[i + 0.5, j + 0.5, k + 0.5] for i in range(S)
                              for j in range(S) for k in range(S)],
                             device=self.device, dtype=self.dtype) * self.a
        pick = torch.randint(0, cages.shape[0], (R * N,), generator=gen,
                             device=self.device)
        com = cages[pick] + 0.5 * torch.randn(R * N, 3, generator=gen,
                                              device=self.device, dtype=self.dtype)
        u = torch.randn(R * N, 3, generator=gen, device=self.device, dtype=self.dtype)
        u = u / u.norm(dim=-1, keepdim=True).clamp_min(EPS)
        half = 0.5 * self.p.r0_bond
        q = torch.stack([com + half * u, com - half * u], dim=1)
        return q.reshape(R, N, 2, 3)


def region_index(phi, a, window_half, cage_min):
    """0 = cage (|z| > cage_min), 1 = neck, 2 = window (|z| < window_half)."""
    z = phi.abs() * a / (2.0 * PI)
    # note: cages sit at phi = +-pi, i.e. LARGE |z'| where z' = a/2 - |z| ...
    # define via distance from the window plane: d_win = |z|
    idx = torch.ones_like(phi, dtype=torch.long)
    idx = torch.where(z < window_half, torch.full_like(idx, 2), idx)
    idx = torch.where(z > cage_min, torch.zeros_like(idx), idx)
    return idx


def _fr_target_lta(method, grid, dphi, F_ema, B_n, oracle, beta):
    """Same law as alkanes.core._fr_target (uniform normalized on the circle)."""
    if method == "abf":
        return None
    if method == "fr_uniform":
        R = B_n.shape[0] if B_n is not None else 1
        return per.normalize_density(
            torch.ones(R, grid.numel(), device=grid.device, dtype=grid.dtype), dphi)
    if method == "fr_oracle":
        log_q = -beta * (oracle[None, :] - B_n)
        log_q = log_q - log_q.max(-1, keepdim=True).values
        return per.normalize_density(torch.exp(log_q), dphi)
    if F_ema is None:
        return None
    log_q = -beta * (F_ema - B_n)
    log_q = log_q - log_q.max(-1, keepdim=True).values
    return per.normalize_density(torch.exp(log_q), dphi)


def run_sampler(method, system: LTASystem, sim: LTASimConfig, seeds,
                oracle_free_energy=None, verbose=True):
    """R = len(seeds) matched-seed replicas of ``method`` in one process.

    Mirrors alkanes.core.run_sampler: same estimator, same FR machinery, same
    output keys (plus ``u_of_z`` accumulators for the entropy decomposition).
    """
    if method not in ALL_METHODS:
        raise ValueError(f"unknown method {method!r}")
    assert_no_reference_leakage(method, oracle_free_energy)
    is_fr = method in FR_METHODS
    device, dtype = system.device, system.dtype
    R, N = len(seeds), sim.n_replicas
    beta = system.p.beta
    grid, dphi = per.periodic_grid(sim.n_grid, device=device, dtype=dtype)
    K_abf = per.wrapped_gaussian_kernel_matrix(grid, sim.abf_bandwidth)
    K_kde = per.wrapped_gaussian_kernel_matrix(grid, sim.kde_bandwidth)
    gen_dyn = torch.Generator(device=device).manual_seed(int(sim.rng_seed))
    gen_fr = torch.Generator(device=device).manual_seed(int(sim.rng_seed) + 987654321)

    oracle = None
    if method == "fr_oracle":
        oracle = torch.as_tensor(oracle_free_energy, device=device, dtype=dtype)
        oracle = oracle - oracle.mean()

    q = system.initial_conditions(R, N, gen_dyn)

    noise_scale = math.sqrt(2.0 * sim.dt / beta)
    fsum = torch.zeros(R, sim.n_grid, device=device, dtype=dtype)
    csum = torch.zeros(R, sim.n_grid, device=device, dtype=dtype)
    fsum_prod = torch.zeros(R, sim.n_grid, device=device, dtype=dtype)
    csum_prod = torch.zeros(R, sim.n_grid, device=device, dtype=dtype)
    usum_prod = torch.zeros(R, sim.n_grid, device=device, dtype=dtype)
    F_ema = None
    ancestors = (torch.arange(N, device=device).expand(R, N).clone() if is_fr else None)
    total_repl = torch.zeros(R, dtype=torch.long)
    prev_reg = None
    trans_counts = torch.zeros(R, dtype=torch.long, device=device)
    has_left_cage = torch.zeros(R, N, dtype=torch.bool, device=device)
    rep_crossings = torch.zeros(R, N, dtype=torch.long, device=device)
    birth_hist = torch.zeros(R, sim.n_grid, device=device, dtype=dtype)
    death_hist = torch.zeros(R, sim.n_grid, device=device, dtype=dtype)
    score_std_sum = np.zeros(R); score_absmax = np.zeros(R); n_score = 0

    diag = {k: [] for k in ["steps", "times", "mean_force", "pmf", "p_hat", "q_target",
                            "eff_counts", "ancestor_ess", "n_unique_ancestor",
                            "max_ancestor_frac", "repl_cumulative", "pq_l2", "kl_pq",
                            "kl_uniform", "frac_cage", "frac_neck", "frac_window"]}
    t0 = time.perf_counter()
    for step in range(sim.n_steps + 1):
        qf = q.reshape(R * N, 2, 3)
        F = system.forces(qf)
        f_loc, phi_f, grad_f = system.cv_local_mean_force(qf, F)
        phi = phi_f.reshape(R, N)
        f_loc = torch.clamp(f_loc, -sim.abf_force_clip * 8,
                            sim.abf_force_clip * 8).reshape(R, N)

        fsum += per.bin_sum(phi, f_loc, sim.n_grid)
        csum += per.bin_counts(phi, sim.n_grid)
        if step >= sim.estimator_burn_in_steps:
            fsum_prod += per.bin_sum(phi, f_loc, sim.n_grid)
            csum_prod += per.bin_counts(phi, sim.n_grid)
            u_now = system.potential_energy(qf).reshape(R, N)
            usum_prod += per.bin_sum(phi, u_now, sim.n_grid)

        mf_profile = per.mean_force_profile(fsum, csum, K_abf)
        A_hat = per.free_energy_from_mean_force(mf_profile, grid, dphi)
        ramp = min(1.0, step / max(sim.abf_warmup_steps, 1))
        abf_scale = sim.abf_bias_scale * ramp
        B_n = abf_scale * A_hat
        mf_at = per.circular_interp(mf_profile, grid, phi).clamp(
            -sim.abf_force_clip, sim.abf_force_clip)
        bias_at = abf_scale * mf_at
        bias_force = (bias_at.reshape(R * N)[:, None, None]
                      * grad_f).reshape(R, N, 2, 3)

        if method in ESTIMATED_TARGET_METHODS and (step + 1) >= sim.fr_start_steps:
            if F_ema is None:
                F_ema = A_hat.clone()
            else:
                rt = sim.target_ema_rate
                F_ema = (1 - rt) * F_ema + rt * A_hat
            F_ema = F_ema - F_ema.mean(-1, keepdim=True)

        cur_reg = region_index(phi, system.a, sim.window_half, sim.cage_min)
        in_cage = cur_reg == 0
        if prev_reg is not None:
            trans_counts += (cur_reg != prev_reg).sum(-1)
            rep_crossings += (has_left_cage & in_cage).long()
        has_left_cage = torch.where(in_cage, torch.zeros_like(has_left_cage),
                                    has_left_cage | (cur_reg == 2))
        prev_reg = cur_reg

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
            u_dens = 1.0 / (2.0 * PI)
            diag["kl_uniform"].append(
                ((p_grid * (torch.log(p_grid.clamp_min(EPS)) - math.log(u_dens)))
                 .sum(-1) * dphi).cpu().numpy())
            q_grid = _fr_target_lta(method, grid, dphi, F_ema, B_n, oracle, beta)
            if q_grid is not None:
                lr = torch.log(p_grid.clamp_min(EPS)) - torch.log(q_grid.clamp_min(EPS))
                diag["q_target"].append(q_grid.cpu().numpy())
                diag["pq_l2"].append((((p_grid - q_grid) ** 2).sum(-1) * dphi)
                                     .sqrt().cpu().numpy())
                diag["kl_pq"].append(((p_grid * lr).sum(-1) * dphi).cpu().numpy())
            else:
                diag["q_target"].append(np.full((R, sim.n_grid), np.nan))
                diag["pq_l2"].append(np.full(R, np.nan))
                diag["kl_pq"].append(np.full(R, np.nan))
            if is_fr:
                ess, nuq, maxf = _ancestor_stats(ancestors.cpu(), N)
            else:
                ess, nuq, maxf = np.full(R, np.nan), np.full(R, N), np.full(R, np.nan)
            diag["ancestor_ess"].append(ess)
            diag["n_unique_ancestor"].append(nuq)
            diag["max_ancestor_frac"].append(maxf)
            diag["frac_cage"].append((cur_reg == 0).float().mean(-1).cpu().numpy())
            diag["frac_neck"].append((cur_reg == 1).float().mean(-1).cpu().numpy())
            diag["frac_window"].append((cur_reg == 2).float().mean(-1).cpu().numpy())

        if step == sim.n_steps:
            break

        noise = torch.randn(q.shape, generator=gen_dyn, device=device, dtype=dtype)
        q = q + sim.dt * (F.reshape(R, N, 2, 3) + bias_force) + noise_scale * noise

        if is_fr:
            nxt = step + 1
            if nxt >= sim.fr_start_steps and \
                    (nxt - sim.fr_start_steps) % max(int(sim.fr_every), 1) == 0:
                phi_new = system.cv_value(q.reshape(R * N, 2, 3)).reshape(R, N)
                q_grid = _fr_target_lta(method, grid, dphi, F_ema, B_n, oracle, beta)
                if q_grid is not None:
                    score, p_fr, kl = _fr_score(phi_new, grid, dphi, K_kde, q_grid,
                                                sim.kde_bandwidth, sim.score_clip)
                    ss = score.detach().cpu().numpy()
                    score_std_sum += ss.std(axis=1)
                    score_absmax = np.maximum(score_absmax, np.abs(ss).max(axis=1))
                    n_score += 1
                    q, ancestors, n_repl, deaths, births = _birth_death(
                        q, score, ancestors, sim, gen_fr)
                    total_repl += n_repl.cpu()
                    for r in range(R):
                        if deaths[r] is not None and deaths[r].numel() > 0:
                            zb = phi_new[r].index_select(0, births[r])
                            zd = phi_new[r].index_select(0, deaths[r])
                            birth_hist[r] += per.bin_counts(zb[None, :], sim.n_grid)[0]
                            death_hist[r] += per.bin_counts(zd[None, :], sim.n_grid)[0]
                            for arr in (has_left_cage, rep_crossings):
                                arr[r, deaths[r]] = arr[r].index_select(0, births[r])
                            if prev_reg is not None:
                                prev_reg[r, deaths[r]] = prev_reg[r].index_select(0, births[r])

    u_of_z = (usum_prod / csum_prod.clamp_min(1.0)).cpu().numpy()
    out = {"method": method, "grid": grid.cpu().numpy(), "dphi": float(dphi),
           "a_pseudo": system.a, "box": system.L,
           "runtime_seconds": time.perf_counter() - t0,
           "total_replacement_events": total_repl.numpy(),
           "n_transitions": trans_counts.cpu().numpy(),
           "n_cage_crossings": rep_crossings.sum(-1).cpu().numpy(),
           "birth_hist": birth_hist.cpu().numpy(),
           "death_hist": death_hist.cpu().numpy(),
           "F_target_ema": (F_ema.cpu().numpy() if F_ema is not None else None),
           "fr_score_std": (score_std_sum / max(n_score, 1)),
           "fr_score_absmax": score_absmax,
           "u_of_z": u_of_z,
           "u_counts": csum_prod.cpu().numpy(),
           "final_eff_counts": per.effective_counts(csum, K_abf).cpu().numpy()}
    for k in diag:
        out[k] = np.asarray(diag[k])
    if verbose:
        print(f"  {method:12s} R={R} N={N}: {out['runtime_seconds']:.1f}s "
              f"repl={out['total_replacement_events'].sum()} "
              f"crossings={out['n_cage_crossings'].sum()}", flush=True)
    return out


def run_umbrella(system: LTASystem, sim: LTASimConfig, centers, kappa,
                 n_steps, n_replicas, burn_in, sample_every, seed, verbose=True):
    """Umbrella sampling for the independent reference: harmonic windows on phi.

    All windows run in ONE batch: (W, n_replicas) molecules, window w biased by
    0.5*kappa*wrap(phi - centers[w])^2.  Returns per-window phi samples and
    unbiased potential energies for WHAM + the U(z) conditional.
    """
    device, dtype = system.device, system.dtype
    W = len(centers)
    c = torch.as_tensor(centers, device=device, dtype=dtype).reshape(W, 1)
    gen = torch.Generator(device=device).manual_seed(int(seed))
    q = system.initial_conditions(W, n_replicas, gen)
    # start each window's molecules near its center: shift COM x to the center
    phi0 = system.cv_value(q.reshape(W * n_replicas, 2, 3)).reshape(W, n_replicas)
    dx = ((c - phi0 + PI) % (2.0 * PI) - PI) * system.a / (2.0 * PI)
    q[..., 0] = q[..., 0] + dx[..., None]

    beta = system.p.beta
    noise_scale = math.sqrt(2.0 * sim.dt / beta)
    soft_start = 5_000        # clamp forces early: window-centred inits can clash
    phis, us = [], []
    t0 = time.perf_counter()
    for step in range(n_steps):
        qf = q.reshape(W * n_replicas, 2, 3)
        F = system.forces(qf)
        if step < soft_start:
            fn = F.norm(dim=-1, keepdim=True).clamp_min(EPS)
            F = F * torch.clamp(fn, max=500.0) / fn
        phi = system.cv_value(qf).reshape(W, n_replicas)
        dphi_c = (phi - c + PI) % (2.0 * PI) - PI
        # umbrella generalized force -kappa*dphi_c, applied via grad(phi)
        gphi = torch.zeros_like(qf)
        gphi[:, :, 0] = PI / system.a
        Fu = (-kappa * dphi_c).reshape(W * n_replicas)[:, None, None] * gphi
        noise = torch.randn(q.shape, generator=gen, device=device, dtype=dtype)
        q = q + sim.dt * (F + Fu).reshape(W, n_replicas, 2, 3) + noise_scale * noise
        if step >= burn_in and step % sample_every == 0:
            phis.append(phi.detach().cpu().numpy().copy())
            us.append(system.potential_energy(q.reshape(W * n_replicas, 2, 3))
                      .reshape(W, n_replicas).cpu().numpy().copy())
    if verbose:
        print(f"  umbrella: {W} windows x {n_replicas} replicas, {n_steps} steps "
              f"-> {len(phis)} frames in {time.perf_counter() - t0:.1f}s", flush=True)
    return np.array(phis), np.array(us)      # (T, W, n) each


def wham_1d(phi_samples, centers, kappa, beta, n_bins=180):
    """Standard periodic histogram WHAM. Returns (grid, F(phi), p(phi), bin_hist).

    ``phi_samples`` (T, W, n): samples of window w. Iterates the WHAM equations
    to convergence; F is set to mean zero.
    """
    W = phi_samples.shape[1]
    edges = np.linspace(-PI, PI, n_bins + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])
    dz = edges[1] - edges[0]
    hist = np.zeros((W, n_bins))
    for w in range(W):
        h, _ = np.histogram(phi_samples[:, w, :].ravel(), bins=edges)
        hist[w] = h
    n_w = hist.sum(axis=1)                                  # samples per window
    d = (mids[None, :] - np.asarray(centers)[:, None] + PI) % (2 * PI) - PI
    bias = 0.5 * kappa * d * d                              # (W, n_bins)
    f_w = np.zeros(W)
    for _ in range(20000):
        denom = (n_w[:, None] * np.exp(beta * (f_w[:, None] - bias))).sum(axis=0)
        p = hist.sum(axis=0) / np.maximum(denom, 1e-300)
        p = p / (p.sum() * dz)
        f_new = -np.log(np.maximum((np.exp(-beta * bias) * p[None, :] * dz), 1e-300)
                        .sum(axis=1)) / beta
        if np.abs(f_new - f_new.mean() - (f_w - f_w.mean())).max() < 1e-10:
            f_w = f_new
            break
        f_w = f_new
    F = -np.log(np.maximum(p, 1e-300)) / beta
    F = F - F.mean()
    return mids, F, p, hist


def conditional_u(phi_samples, u_samples, n_bins=180):
    """U(phi) = <V | phi> pooled over windows.

    Within a narrow phi-bin the umbrella bias is (to first order) a function of
    phi alone, so the conditional distribution of everything else at fixed phi
    is unbiased; the pooled per-bin average of V is a consistent estimator of
    the conditional mean.  Bins with no samples return NaN.
    """
    edges = np.linspace(-PI, PI, n_bins + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])
    ph = phi_samples.ravel()
    uu = u_samples.ravel()
    idx = np.clip(np.digitize(ph, edges) - 1, 0, n_bins - 1)
    s = np.bincount(idx, weights=uu, minlength=n_bins)
    c = np.bincount(idx, minlength=n_bins)
    return mids, np.where(c > 0, s / np.maximum(c, 1), np.nan), c
