"""Well-tempered OPES_METAD baselines for the CV-extension experiments:
a NON-periodic interval OPES (distance CV R15/R14) and a 2-D torus OPES (joint torsion).

Both reuse the OPES_METAD math of :mod:`alkanes.opes` (Invernizzi & Parrinello 2020) --
grow a reweighted CV density by depositing walker kernels with weight ``w = exp(beta A)``,
bias ``A_n = (1-1/gamma) beta^{-1} log( u + epsilon )`` with ``u = rho / rho_flat`` and
``epsilon = exp(-beta*BARRIER/(1-1/gamma))``, apply ``-grad A`` through the same
``+f grad(xi)`` channel as ABF (equal force-evaluation match).  ``gamma=inf`` => flat
target; ``gamma_from_barrier`` sets ``gamma = beta*BARRIER``.  Two estimates reported:
OPES-native reweight ``F=-beta^{-1} log rho`` and a common mean-force reconstruction.
The reference is never consulted (structural no-leakage: no oracle argument).

Interval OPES uses a plain Gaussian kernel on ``[lo,hi]`` (soft walls keep support inside);
torus OPES uses separable wrapped-Gaussian kernels and a spectral bias gradient.
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
from . import poisson2d as ps
from .distance_cv import DistanceCV, dist_bias_force
from .cv2d import JointDihedralCV2D, abf_bias_force_2d
from .core_dist import DistSimConfig, _wall_gforce
from .core2d import Sim2DConfig, _joint_basin

EPS = 1.0e-12
PI = math.pi
TWO_PI = 2.0 * math.pi


# ===========================================================================
# 1-D interval OPES (distance CV)
# ===========================================================================
@dataclass
class IntervalOPESConfig:
    n_grid: int = 256
    beta: float = 1.0
    R_lo: float = 1.4
    R_hi: float = 3.7
    barrier: float = 8.0
    pace: int = 500
    sigma: float = 0.08           # kernel bandwidth in CV units
    gamma: float = float("inf")
    gamma_from_barrier: bool = True
    bias_force_clip: float = 60.0
    warmup_steps: int = 5_000

    def effective_gamma(self):
        g = self.gamma
        if (g is None) or (g == float("inf")) or (g <= 1.0):
            if self.gamma_from_barrier and math.isinf(g):
                gb = self.beta * self.barrier
                return gb if gb > 1.0 else float("inf")
            return float("inf")
        return float(g)


class BatchedIntervalOPES:
    def __init__(self, cfg: IntervalOPESConfig, R, device, dtype=torch.float64):
        self.cfg = cfg; self.R = R; self.device = device; self.dtype = dtype
        self.grid, self.dz = iv.interval_grid(cfg.n_grid, cfg.R_lo, cfg.R_hi, device=device, dtype=dtype)
        self.K = iv.gaussian_kernel_matrix(self.grid, cfg.sigma)
        self.L = cfg.R_hi - cfg.R_lo
        self.beta = float(cfg.beta)
        self.gamma = cfg.effective_gamma()
        self.prefactor = 1.0 if math.isinf(self.gamma) else (1.0 - 1.0 / self.gamma)
        self.epsilon = float(math.exp(-self.beta * cfg.barrier / max(self.prefactor, EPS)))
        self.num = torch.zeros(R, cfg.n_grid, device=device, dtype=dtype)
        self.wsum = torch.zeros(R, device=device, dtype=torch.float64)
        self.w2sum = torch.zeros(R, device=device, dtype=torch.float64)
        self.n_deposits = 0; self.n_samples = 0
        self._rebuild()

    def _rebuild(self):
        if self.n_deposits == 0:
            self._rho = torch.full((self.R, self.cfg.n_grid), 1.0 / self.L, device=self.device, dtype=self.dtype)
            self._bias = torch.zeros_like(self._rho); self._force = torch.zeros_like(self._rho); return
        raw = self.num / self.wsum[:, None].clamp_min(EPS).to(self.dtype)
        self._rho = iv.normalize_density(raw, self.dz).clamp_min(EPS)
        u = self._rho * self.L
        A = self.prefactor * (1.0 / self.beta) * torch.log(u + self.epsilon)
        A = A - A.max(-1, keepdim=True).values
        self._bias = A
        f = torch.zeros_like(A); g = self.grid
        f[:, 1:-1] = -(A[:, 2:] - A[:, :-2]) / (g[2:] - g[:-2])
        f[:, 0] = -(A[:, 1] - A[:, 0]) / (g[1] - g[0]); f[:, -1] = -(A[:, -1] - A[:, -2]) / (g[-1] - g[-2])
        self._force = torch.clamp(f, -self.cfg.bias_force_clip, self.cfg.bias_force_clip)

    def _weights(self, R):
        if self.n_deposits == 0:
            return torch.ones_like(R)
        A_at = iv.interval_interp(self._bias, self.grid, R)
        return torch.exp(torch.clamp(self.beta * A_at, max=50.0))

    def deposit(self, Rv):
        w = self._weights(Rv).detach()
        counts_w = iv.bin_sum(Rv, w, self.cfg.n_grid, self.cfg.R_lo, self.cfg.R_hi)
        self.num += iv.smooth(counts_w, self.K)
        self.wsum += w.sum(-1).to(torch.float64); self.w2sum += (w.to(torch.float64) ** 2).sum(-1)
        self.n_deposits += 1; self.n_samples += Rv.shape[-1]; self._rebuild()

    def bias_force_at(self, Rv, step=None):
        f = iv.interval_interp(self._force, self.grid, Rv)
        if step is not None and self.cfg.warmup_steps > 0:
            f = f * min(1.0, float(step) / float(self.cfg.warmup_steps))
        return f

    def free_energy(self):
        Fz = -(1.0 / self.beta) * torch.log(self._rho.clamp_min(EPS))
        return Fz - Fz.mean(-1, keepdim=True)

    def neff_frac(self):
        neff = torch.where(self.w2sum > 0, self.wsum ** 2 / self.w2sum.clamp_min(EPS), torch.zeros_like(self.wsum))
        return (neff / max(self.n_samples, 1)).cpu().numpy()

    def n_kernels(self):
        thr = self.num.max(-1, keepdim=True).values * 1e-4
        return (self.num > thr).sum(-1).cpu().numpy()


def run_opes_dist(params: pot.AlkaneParams, sim: DistSimConfig, opes_cfg: IntervalOPESConfig,
                  seeds, cv: DistanceCV, device, dtype=torch.float64, initial_dihedrals=None,
                  collect_conditional=True, verbose=True):
    R = len(seeds); N = sim.n_replicas; A = params.n_atoms; n_dih = params.n_dihedrals
    beta = params.beta
    grid, dz = iv.interval_grid(sim.n_grid, sim.R_lo, sim.R_hi, device=device, dtype=dtype)
    K_abf = iv.gaussian_kernel_matrix(grid, sim.abf_bandwidth)
    K_kde = iv.reflected_kernel_matrix(grid, sim.kde_bandwidth, sim.R_lo, sim.R_hi)
    gen_dyn = torch.Generator(device=device).manual_seed(int(sim.rng_seed))
    opes = BatchedIntervalOPES(opes_cfg, R, device, dtype)

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
    fsum_prod = torch.zeros(R, sim.n_grid, device=device, dtype=dtype)
    csum_prod = torch.zeros(R, sim.n_grid, device=device, dtype=dtype)
    do_cond = collect_conditional and n_dih >= 2
    if do_cond:
        g1c, g2c, dphi1c, dphi2c = d2.torus_grid(sim.n_grid2, sim.n_grid2, device=device, dtype=dtype)
        cond_hist = torch.zeros(R, sim.n_rbins, sim.n_grid2, sim.n_grid2, device=device, dtype=dtype)
        cond_edges = torch.linspace(sim.R_lo, sim.R_hi, sim.n_rbins + 1, device=device, dtype=dtype)
    R_c1 = sim.R_lo + (sim.R_hi - sim.R_lo) / 3.0
    R_c2 = sim.R_lo + 2.0 * (sim.R_hi - sim.R_lo) / 3.0
    trans_counts = torch.zeros(R, dtype=torch.long, device=device); prev_ext = None

    diag = {k: [] for k in ["steps", "times", "mean_force", "pmf", "mean_force_reweight",
                            "pmf_reweight", "p_hat", "eff_counts", "ancestor_ess",
                            "n_unique_ancestor", "max_ancestor_frac", "repl_cumulative",
                            "pq_l2", "kl_pq", "frac_compact", "frac_inter", "frac_extended",
                            "neff_frac", "n_kernels"]}
    t0 = time.perf_counter()
    for step in range(sim.n_steps + 1):
        qf = q.reshape(R * N, A, 3)
        F = pot.forces(qf, params)
        f_loc, R_f, grad_f = cv.local_mean_force(qf, F, beta)
        Rv = R_f.reshape(R, N)
        f_loc = torch.clamp(f_loc, -sim.abf_force_clip * 8, sim.abf_force_clip * 8).reshape(R, N)
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
        bias_at = opes.bias_force_at(Rv, step=step).clamp(-sim.abf_force_clip, sim.abf_force_clip)
        wall_g = _wall_gforce(Rv, sim)
        bias_force = dist_bias_force(grad_f, (bias_at + wall_g).reshape(R * N)).reshape(R, N, A, 3)

        cur_ext = torch.zeros_like(Rv, dtype=torch.long)
        cur_ext = torch.where(Rv >= R_c2, torch.ones_like(cur_ext), cur_ext)
        cur_ext = torch.where(Rv <= R_c1, 2 * torch.ones_like(cur_ext), cur_ext)
        if prev_ext is not None:
            trans_counts += (cur_ext != prev_ext).sum(-1)
        prev_ext = cur_ext

        if step % sim.save_every == 0 or step == sim.n_steps:
            mf_c = iv.mean_force_profile(fsum_prod, csum_prod, K_abf)
            F_c = iv.free_energy_from_mean_force(mf_c, grid, dz)
            F_native = opes.free_energy()
            mf_native = torch.zeros_like(F_native); g = grid
            mf_native[:, 1:-1] = (F_native[:, 2:] - F_native[:, :-2]) / (g[2:] - g[:-2])
            diag["steps"].append(step); diag["times"].append(step * sim.dt)
            diag["mean_force"].append(mf_c.cpu().numpy()); diag["pmf"].append(F_c.cpu().numpy())
            diag["mean_force_reweight"].append(mf_native.cpu().numpy()); diag["pmf_reweight"].append(F_native.cpu().numpy())
            diag["eff_counts"].append(iv.effective_counts(csum_prod, K_abf).cpu().numpy())
            diag["repl_cumulative"].append(np.zeros(R))
            p_grid = iv.kde_marginal(Rv, K_kde, sim.n_grid, dz, sim.R_lo, sim.R_hi)
            diag["p_hat"].append(p_grid.cpu().numpy())
            diag["pq_l2"].append(np.full(R, np.nan)); diag["kl_pq"].append(np.full(R, np.nan))
            diag["ancestor_ess"].append(np.full(R, np.nan)); diag["n_unique_ancestor"].append(np.full(R, N))
            diag["max_ancestor_frac"].append(np.full(R, np.nan))
            diag["neff_frac"].append(opes.neff_frac()); diag["n_kernels"].append(opes.n_kernels())
            diag["frac_compact"].append((cur_ext == 2).float().mean(-1).cpu().numpy())
            diag["frac_inter"].append((cur_ext == 0).float().mean(-1).cpu().numpy())
            diag["frac_extended"].append((cur_ext == 1).float().mean(-1).cpu().numpy())
        if step == sim.n_steps:
            break
        noise = torch.randn(q.shape, generator=gen_dyn, device=device, dtype=dtype)
        q = geom.remove_com(q + sim.dt * (F.reshape(R, N, A, 3) + bias_force) + noise_scale * noise)
        if (step + 1) % max(int(opes_cfg.pace), 1) == 0:
            opes.deposit(cv.value(q.reshape(R * N, A, 3)).reshape(R, N))

    out = {"method": "opes", "grid": grid.cpu().numpy(), "dz": float(dz),
           "R_lo": sim.R_lo, "R_hi": sim.R_hi, "runtime_seconds": time.perf_counter() - t0,
           "total_replacement_events": np.zeros(R, dtype=int), "n_transitions": trans_counts.cpu().numpy(),
           "n_round_trips": np.zeros(R, dtype=int),
           "first_discovery": {k: np.full(R, -1) for k in ("compact", "intermediate", "extended")},
           "birth_hist": np.zeros((R, sim.n_grid)), "death_hist": np.zeros((R, sim.n_grid)),
           "F_target_ema": None, "fr_score_std": np.full(R, np.nan), "fr_score_absmax": np.full(R, np.nan),
           "final_eff_counts": iv.effective_counts(csum_prod, K_abf).cpu().numpy(),
           "final_neff_frac": opes.neff_frac(), "final_n_kernels": opes.n_kernels()}
    for k in diag:
        out[k] = np.asarray(diag[k])
    if do_cond:
        out["cond_hist"] = cond_hist.cpu().numpy()
        out["cond_grid1"] = g1c.cpu().numpy(); out["cond_grid2"] = g2c.cpu().numpy()
        out["cond_dphi"] = float(dphi1c); out["cond_edges"] = cond_edges.cpu().numpy()
    if verbose:
        print(f"  opes-dist R={R} N={N}: {out['runtime_seconds']:.1f}s neff={np.median(opes.neff_frac()):.2f}")
    return out


# ===========================================================================
# 2-D torus OPES (joint torsion CV)
# ===========================================================================
@dataclass
class TorusOPESConfig:
    n_grid: int = 48
    beta: float = 1.0
    barrier: float = 8.0
    pace: int = 500
    sigma: float = 0.30           # kernel bandwidth (radians, both axes)
    gamma: float = float("inf")
    gamma_from_barrier: bool = True
    bias_force_clip: float = 60.0
    warmup_steps: int = 5_000

    def effective_gamma(self):
        g = self.gamma
        if (g is None) or (g == float("inf")) or (g <= 1.0):
            if self.gamma_from_barrier and math.isinf(g):
                gb = self.beta * self.barrier
                return gb if gb > 1.0 else float("inf")
            return float("inf")
        return float(g)


class BatchedTorusOPES:
    def __init__(self, cfg: TorusOPESConfig, R, device, dtype=torch.float64):
        self.cfg = cfg; self.R = R; self.device = device; self.dtype = dtype
        self.g1, self.g2, self.dz1, self.dz2 = d2.torus_grid(cfg.n_grid, cfg.n_grid, device=device, dtype=dtype)
        self.K1, self.K2 = d2.kernels(self.g1, self.g2, cfg.sigma, cfg.sigma)
        self.area = TWO_PI * TWO_PI
        self.beta = float(cfg.beta)
        self.gamma = cfg.effective_gamma()
        self.prefactor = 1.0 if math.isinf(self.gamma) else (1.0 - 1.0 / self.gamma)
        self.epsilon = float(math.exp(-self.beta * cfg.barrier / max(self.prefactor, EPS)))
        n = cfg.n_grid
        self.num = torch.zeros(R, n, n, device=device, dtype=dtype)
        self.wsum = torch.zeros(R, device=device, dtype=torch.float64)
        self.w2sum = torch.zeros(R, device=device, dtype=torch.float64)
        self.n_deposits = 0; self.n_samples = 0
        self._rebuild()

    def _rebuild(self):
        n = self.cfg.n_grid
        if self.n_deposits == 0:
            self._rho = torch.full((self.R, n, n), 1.0 / self.area, device=self.device, dtype=self.dtype)
            self._bias = torch.zeros_like(self._rho)
            self._f1 = torch.zeros_like(self._rho); self._f2 = torch.zeros_like(self._rho); return
        raw = self.num / self.wsum[:, None, None].clamp_min(EPS).to(self.dtype)
        self._rho = d2.normalize2(raw, self.dz1, self.dz2).clamp_min(EPS)
        u = self._rho * self.area
        A = self.prefactor * (1.0 / self.beta) * torch.log(u + self.epsilon)
        A = A - A.amax(dim=(-2, -1), keepdim=True)
        self._bias = A
        gA1, gA2 = ps.spectral_gradient(A, self.dz1, self.dz2)
        self._f1 = torch.clamp(-gA1, -self.cfg.bias_force_clip, self.cfg.bias_force_clip)
        self._f2 = torch.clamp(-gA2, -self.cfg.bias_force_clip, self.cfg.bias_force_clip)

    def _weights(self, phi1, phi2):
        if self.n_deposits == 0:
            return torch.ones_like(phi1)
        A_at = d2.bilinear_interp2(self._bias, self.g1, self.g2, self.dz1, self.dz2, phi1, phi2)
        return torch.exp(torch.clamp(self.beta * A_at, max=50.0))

    def deposit(self, phi1, phi2):
        w = self._weights(phi1, phi2).detach()
        counts_w = d2.scatter_sum(phi1, phi2, w, self.cfg.n_grid, self.cfg.n_grid, self.dz1, self.dz2)
        self.num += d2.smooth2(counts_w, self.K1, self.K2)
        self.wsum += w.sum(-1).to(torch.float64); self.w2sum += (w.to(torch.float64) ** 2).sum(-1)
        self.n_deposits += 1; self.n_samples += phi1.shape[-1]; self._rebuild()

    def bias_force_at(self, phi1, phi2, step=None):
        f1 = d2.bilinear_interp2(self._f1, self.g1, self.g2, self.dz1, self.dz2, phi1, phi2)
        f2 = d2.bilinear_interp2(self._f2, self.g1, self.g2, self.dz1, self.dz2, phi1, phi2)
        if step is not None and self.cfg.warmup_steps > 0:
            s = min(1.0, float(step) / float(self.cfg.warmup_steps)); f1 = f1 * s; f2 = f2 * s
        return f1, f2

    def free_energy(self):
        Fz = -(1.0 / self.beta) * torch.log(self._rho.clamp_min(EPS))
        return Fz - Fz.mean(dim=(-2, -1), keepdim=True)

    def neff_frac(self):
        neff = torch.where(self.w2sum > 0, self.wsum ** 2 / self.w2sum.clamp_min(EPS), torch.zeros_like(self.wsum))
        return (neff / max(self.n_samples, 1)).cpu().numpy()

    def n_kernels(self):
        thr = self.num.amax(dim=(-2, -1), keepdim=True) * 1e-4
        return (self.num > thr).sum(dim=(-2, -1)).cpu().numpy()


def run_opes_2d(params: pot.AlkaneParams, sim: Sim2DConfig, opes_cfg: TorusOPESConfig,
                seeds, cv: JointDihedralCV2D, device, dtype=torch.float64, initial_dihedrals=None,
                verbose=True):
    R = len(seeds); N = sim.n_replicas; A = params.n_atoms; n_dih = params.n_dihedrals
    beta = params.beta; n1 = n2 = sim.n_grid
    g1c, g2c, dz1, dz2 = d2.torus_grid(n1, n2, device=device, dtype=dtype)
    K1, K2 = d2.kernels(g1c, g2c, sim.abf_bandwidth, sim.abf_bandwidth)
    Kk1, Kk2 = d2.kernels(g1c, g2c, sim.kde_bandwidth, sim.kde_bandwidth)
    gen_dyn = torch.Generator(device=device).manual_seed(int(sim.rng_seed))
    opes = BatchedTorusOPES(opes_cfg, R, device, dtype)

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
    joint_hist = torch.zeros(R, n1, n2, device=device, dtype=dtype)
    prev_basin = None; trans_counts = torch.zeros(R, dtype=torch.long, device=device)
    first_discovery = torch.full((R, 9), -1, dtype=torch.long, device=device)

    diag = {k: [] for k in ["steps", "times", "pmf", "pmf_reweight", "p_hat",
                            "ancestor_ess", "n_unique_ancestor", "max_ancestor_frac",
                            "repl_cumulative", "kl_pq", "n_basins_visited", "neff_frac", "n_kernels"]}
    t0 = time.perf_counter()
    for step in range(sim.n_steps + 1):
        qf = q.reshape(R * N, A, 3)
        F = pot.forces(qf, params)
        f, phi, gfull, geo = cv.local_mean_force(qf, F, beta)
        phi1 = phi[:, 0].reshape(R, N); phi2 = phi[:, 1].reshape(R, N)
        f = torch.clamp(f, -sim.abf_force_clip * 8, sim.abf_force_clip * 8)
        f1 = f[:, 0].reshape(R, N); f2 = f[:, 1].reshape(R, N)
        if step >= sim.estimator_burn_in_steps:
            f1s += d2.scatter_sum(phi1, phi2, f1, n1, n2, dz1, dz2)
            f2s += d2.scatter_sum(phi1, phi2, f2, n1, n2, dz1, dz2)
            csum += d2.scatter_counts(phi1, phi2, n1, n2, dz1, dz2)
            joint_hist += d2.scatter_counts(phi1, phi2, n1, n2, dz1, dz2)
        b1, b2 = opes.bias_force_at(phi1, phi2, step=step)
        bias_at = torch.stack([b1.clamp(-sim.abf_force_clip, sim.abf_force_clip).reshape(R * N),
                               b2.clamp(-sim.abf_force_clip, sim.abf_force_clip).reshape(R * N)], dim=-1)
        bias_force = abf_bias_force_2d(gfull, bias_at).reshape(R, N, A, 3)

        cur_basin = _joint_basin(phi1, phi2, sim.basin_barrier)
        if prev_basin is not None:
            trans_counts += (cur_basin != prev_basin).sum(-1)
        prev_basin = cur_basin
        for b in range(9):
            seen = (cur_basin == b).any(-1); fd = first_discovery[:, b]
            first_discovery[:, b] = torch.where((fd < 0) & seen, torch.full_like(fd, step), fd)

        if step % sim.save_every == 0 or step == sim.n_steps:
            _, _, den = d2.mean_force_fields(f1s, f2s, csum, K1, K2)
            g1f, g2f, _ = d2.mean_force_fields(f1s, f2s, csum, K1, K2)
            B_c, _, _ = ps.poisson_projection(g1f, g2f, dz1, dz2)
            p_hat = d2.kde2(phi1, phi2, Kk1, Kk2, n1, n2, dz1, dz2)
            diag["steps"].append(step); diag["times"].append(step * sim.dt)
            diag["pmf"].append(B_c.cpu().numpy()); diag["pmf_reweight"].append(opes.free_energy().cpu().numpy())
            diag["p_hat"].append(p_hat.cpu().numpy())
            diag["ancestor_ess"].append(np.full(R, np.nan)); diag["n_unique_ancestor"].append(np.full(R, N))
            diag["max_ancestor_frac"].append(np.full(R, np.nan)); diag["repl_cumulative"].append(np.zeros(R))
            diag["kl_pq"].append(np.full(R, np.nan))
            diag["n_basins_visited"].append(np.array([(first_discovery[r] >= 0).sum().item() for r in range(R)]))
            diag["neff_frac"].append(opes.neff_frac()); diag["n_kernels"].append(opes.n_kernels())
        if step == sim.n_steps:
            break
        noise = torch.randn(q.shape, generator=gen_dyn, device=device, dtype=dtype)
        q = geom.remove_com(q + sim.dt * (F.reshape(R, N, A, 3) + bias_force) + noise_scale * noise)
        if (step + 1) % max(int(opes_cfg.pace), 1) == 0:
            ph = cv.values(q.reshape(R * N, A, 3))
            opes.deposit(ph[0].reshape(R, N), ph[1].reshape(R, N))

    g1f, g2f, _ = d2.mean_force_fields(f1s, f2s, csum, K1, K2)
    B_final, _, _ = ps.poisson_projection(g1f, g2f, dz1, dz2)
    out = {"method": "opes", "grid1": g1c.cpu().numpy(), "grid2": g2c.cpu().numpy(),
           "dz1": float(dz1), "dz2": float(dz2), "n_grid": n1, "runtime_seconds": time.perf_counter() - t0,
           "total_replacement_events": np.zeros(R, dtype=int), "n_transitions": trans_counts.cpu().numpy(),
           "n_round_trips": np.zeros(R, dtype=int), "trans_matrix": np.zeros((R, 9, 9), dtype=int),
           "first_discovery": first_discovery.cpu().numpy(),
           "birth_hist": np.zeros((R, n1, n2)), "death_hist": np.zeros((R, n1, n2)),
           "joint_hist": joint_hist.cpu().numpy(), "F_target_ema": None,
           "fr_score_std": np.full(R, np.nan), "fr_score_absmax": np.full(R, np.nan),
           "final_pmf": B_final.cpu().numpy(), "final_pmf_reweight": opes.free_energy().cpu().numpy(),
           "gram_reg_activations": 0, "gram_lam_min_min": np.full(R, np.nan),
           "final_neff_frac": opes.neff_frac(), "final_n_kernels": opes.n_kernels()}
    for k in diag:
        out[k] = np.asarray(diag[k])
    if verbose:
        print(f"  2D-opes R={R} N={N}: {out['runtime_seconds']:.1f}s trans={out['n_transitions'].sum()} "
              f"neff={np.median(opes.neff_frac()):.2f}")
    return out
