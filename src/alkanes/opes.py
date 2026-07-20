"""Periodic well-tempered OPES_METAD for a dihedral CV (batched over R seeds).

Reuses the OPES_METAD math of ``src/opes_core.py`` (Invernizzi & Parrinello 2020) but
with a *periodic* (wrapped-Gaussian / von Mises) KDE on ``phi in [-pi, pi)``, so the
existing OPES core is left untouched.  ``R`` independent OPES states share one GPU
process (leading batch dim ``R``), matched to the ABF/mFR seed-batched sampler.

Bias:  A_n(phi) = (1 - 1/gamma) beta^{-1} log( p_tilde_n(phi) * 2pi + epsilon ),
with epsilon = exp(-beta*BARRIER/(1-1/gamma)) and the reweighted marginal grown by
depositing walker samples with weight w = exp(beta*A_{n-1}).  gamma=inf => flat
target.  The applied biasing mean force along phi is -A_n'(phi), added to the
dynamics via the same ``+f grad(phi)`` channel as ABF (equal force-evaluation match).

Two free-energy estimates are reported: the OPES-native reweight
``F = -beta^{-1} log p_tilde`` and a common mean-force estimate (integrate the local
mean force accumulated under the OPES-biased dynamics), the latter being the exact
analogue of the ABF/mFR reconstruction so OPES differs only in the biasing strategy.
The reference is never consulted (no-leakage).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, asdict

import numpy as np
import torch

from . import geometry as geom
from . import periodic as per
from . import potentials as pot
from .cv import DihedralCV, abf_bias_force
from .core import AlkaneSimConfig, basin_index

EPS = 1.0e-12
PI = math.pi


@dataclass
class PeriodicOPESConfig:
    n_grid: int = 180
    beta: float = 1.0
    barrier: float = 8.0
    pace: int = 500
    sigma: float = 0.20           # kernel bandwidth in radians
    gamma: float = float("inf")   # inf => flat target
    gamma_from_barrier: bool = True
    bias_force_clip: float = 60.0
    warmup_steps: int = 10_000

    def effective_gamma(self):
        g = self.gamma
        if (g is None) or (g == float("inf")) or (g <= 1.0):
            if self.gamma_from_barrier and math.isinf(g):
                gb = self.beta * self.barrier
                return gb if gb > 1.0 else float("inf")
            return float("inf")
        return float(g)

    def config_hash(self):
        import hashlib, json
        return hashlib.md5(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]


class BatchedPeriodicOPES:
    """R independent periodic OPES states on a shared grid."""

    def __init__(self, cfg: PeriodicOPESConfig, R, device, dtype=torch.float64):
        self.cfg = cfg
        self.R = R
        self.device = device
        self.dtype = dtype
        self.grid, self.dphi = per.periodic_grid(cfg.n_grid, device=device, dtype=dtype)
        self.K = per.wrapped_gaussian_kernel_matrix(self.grid, cfg.sigma)
        self.beta = float(cfg.beta)
        self.gamma = cfg.effective_gamma()
        self.prefactor = 1.0 if math.isinf(self.gamma) else (1.0 - 1.0 / self.gamma)
        denom = max(self.prefactor, EPS)
        self.epsilon = float(math.exp(-self.beta * cfg.barrier / denom))
        self.num = torch.zeros(R, cfg.n_grid, device=device, dtype=dtype)
        self.wsum = torch.zeros(R, device=device, dtype=torch.float64)
        self.w2sum = torch.zeros(R, device=device, dtype=torch.float64)
        self.n_deposits = 0
        self.n_samples = 0
        self._rebuild()

    def _rebuild(self):
        if self.n_deposits == 0:
            self._rho = torch.full((self.R, self.cfg.n_grid), 1.0 / (2 * PI),
                                   device=self.device, dtype=self.dtype)
            self._bias = torch.zeros_like(self._rho)
            self._force = torch.zeros_like(self._rho)
            return
        raw = self.num / self.wsum[:, None].clamp_min(EPS).to(self.dtype)
        rho = per.normalize_density(raw, self.dphi)
        self._rho = rho.clamp_min(EPS)
        u = self._rho * (2 * PI)
        A = self.prefactor * (1.0 / self.beta) * torch.log(u + self.epsilon)
        A = A - A.max(-1, keepdim=True).values
        self._bias = A
        f = torch.zeros_like(A)
        g = self.grid
        f[:, 1:-1] = -(A[:, 2:] - A[:, :-2]) / (g[2:] - g[:-2])
        f[:, 0] = -(A[:, 1] - A[:, 0]) / (g[1] - g[0])
        f[:, -1] = -(A[:, -1] - A[:, -2]) / (g[-1] - g[-2])
        self._force = torch.clamp(f, -self.cfg.bias_force_clip, self.cfg.bias_force_clip)

    def _weights(self, phi):
        if self.n_deposits == 0:
            return torch.ones_like(phi)
        A_at = per.circular_interp(self._bias, self.grid, phi)
        return torch.exp(torch.clamp(self.beta * A_at, max=50.0))

    def deposit(self, phi):
        w = self._weights(phi).detach()
        counts_w = per.bin_sum(phi, w, self.cfg.n_grid)
        self.num += per.smooth(counts_w, self.K)
        self.wsum += w.sum(-1).to(torch.float64)
        self.w2sum += (w.to(torch.float64) ** 2).sum(-1)
        self.n_deposits += 1
        self.n_samples += phi.shape[-1]
        self._rebuild()

    def bias_force_at(self, phi, step=None):
        f = per.circular_interp(self._force, self.grid, phi)
        if step is not None and self.cfg.warmup_steps > 0:
            f = f * min(1.0, float(step) / float(self.cfg.warmup_steps))
        return f

    def free_energy(self):
        Fz = -(1.0 / self.beta) * torch.log(self._rho.clamp_min(EPS))
        return Fz - Fz.mean(-1, keepdim=True)

    def neff_frac(self):
        ws = self.wsum; w2 = self.w2sum
        neff = torch.where(w2 > 0, ws * ws / w2.clamp_min(EPS), torch.zeros_like(ws))
        return (neff / max(self.n_samples, 1)).cpu().numpy()

    def n_kernels(self):
        thr = self.num.max(-1, keepdim=True).values * 1e-4
        return (self.num > thr).sum(-1).cpu().numpy()


def run_opes(params: pot.AlkaneParams, sim: AlkaneSimConfig, opes_cfg: PeriodicOPESConfig,
             seeds, cv: DihedralCV, device, dtype=torch.float64, initial_dihedrals=None,
             collect_pentane=False, verbose=True):
    """Multi-walker periodic OPES on the alkane; diag schema matches ``core.run_sampler``."""
    R = len(seeds); N = sim.n_replicas; Aat = params.n_atoms; n_dih = params.n_dihedrals
    beta = params.beta
    grid, dphi = per.periodic_grid(sim.n_grid, device=device, dtype=dtype)
    K_abf = per.wrapped_gaussian_kernel_matrix(grid, sim.abf_bandwidth)
    K_kde = per.wrapped_gaussian_kernel_matrix(grid, sim.kde_bandwidth)
    gen_dyn = torch.Generator(device=device).manual_seed(int(sim.rng_seed))
    opes = BatchedPeriodicOPES(opes_cfg, R, device, dtype)

    if initial_dihedrals is None:
        init = torch.zeros(R, N, n_dih, device=device, dtype=dtype)
    elif callable(initial_dihedrals):
        init = initial_dihedrals(R, N, gen_dyn).to(device=device, dtype=dtype)
    else:
        d0 = torch.as_tensor(initial_dihedrals, device=device, dtype=dtype).reshape(1, 1, n_dih)
        init = d0.expand(R, N, n_dih).clone()
    q = geom.place_chain(init.reshape(R * N, n_dih), Aat, d0=params.d0, theta0=params.theta0,
                         device=device, dtype=dtype).reshape(R, N, Aat, 3)
    q = geom.remove_com(q + 1e-3 * torch.randn(q.shape, generator=gen_dyn, device=device, dtype=dtype))
    noise_scale = math.sqrt(2.0 * sim.dt / beta)

    fsum_prod = torch.zeros(R, sim.n_grid, device=device, dtype=dtype)
    csum_prod = torch.zeros(R, sim.n_grid, device=device, dtype=dtype)
    do_pent = collect_pentane and Aat >= 5
    if do_pent:
        g2, dphi2 = per.periodic_grid(sim.n_grid2, device=device, dtype=dtype)
        joint_hist = torch.zeros(R, sim.n_grid2, sim.n_grid2, device=device, dtype=dtype)
        cv2 = DihedralCV((1, 2, 3, 4))
    prev_basin = None
    trans_counts = torch.zeros(R, dtype=torch.long, device=device)

    diag = {k: [] for k in ["steps", "times", "mean_force", "pmf", "mean_force_reweight",
                            "pmf_reweight", "p_hat", "eff_counts", "ancestor_ess",
                            "n_unique_ancestor", "max_ancestor_frac", "repl_cumulative",
                            "pq_l2", "kl_pq", "frac_T", "frac_Gp", "frac_Gm",
                            "frac2_T", "frac2_Gp", "frac2_Gm", "neff_frac", "n_kernels"]}
    t0 = time.perf_counter()
    for step in range(sim.n_steps + 1):
        qf = q.reshape(R * N, Aat, 3)
        F = pot.forces(qf, params)
        f_loc, phi_f, grad_f = cv.local_mean_force(qf, F, beta)
        phi = phi_f.reshape(R, N)
        f_loc = torch.clamp(f_loc, -sim.abf_force_clip * 8, sim.abf_force_clip * 8).reshape(R, N)
        if step >= sim.estimator_burn_in_steps:
            fsum_prod += per.bin_sum(phi, f_loc, sim.n_grid)
            csum_prod += per.bin_counts(phi, sim.n_grid)
            if do_pent:
                phi2 = cv2.value(qf).reshape(R, N)
                i1 = torch.floor((phi + PI) / dphi2).long().clamp(0, sim.n_grid2 - 1)
                i2 = torch.floor((phi2 + PI) / dphi2).long().clamp(0, sim.n_grid2 - 1)
                joint_hist.view(R, -1).scatter_add_(1, i1 * sim.n_grid2 + i2, torch.ones_like(phi))

        bias_at = opes.bias_force_at(phi, step=step).clamp(-sim.abf_force_clip, sim.abf_force_clip)
        bias_force = abf_bias_force(grad_f, bias_at.reshape(R * N)).reshape(R, N, Aat, 3)

        cur_basin = basin_index(phi, sim.basin_barrier)
        if prev_basin is not None:
            trans_counts += (cur_basin != prev_basin).sum(-1)
        prev_basin = cur_basin

        if step % sim.save_every == 0 or step == sim.n_steps:
            mf_c = per.mean_force_profile(fsum_prod, csum_prod, K_abf)
            F_c = per.free_energy_from_mean_force(mf_c, grid, dphi)
            F_native = opes.free_energy()
            mf_native = torch.zeros_like(F_native)
            g = grid
            mf_native[:, 1:-1] = (F_native[:, 2:] - F_native[:, :-2]) / (g[2:] - g[:-2])
            diag["steps"].append(step); diag["times"].append(step * sim.dt)
            diag["mean_force"].append(mf_c.cpu().numpy()); diag["pmf"].append(F_c.cpu().numpy())
            diag["mean_force_reweight"].append(mf_native.cpu().numpy())
            diag["pmf_reweight"].append(F_native.cpu().numpy())
            diag["eff_counts"].append(per.effective_counts(csum_prod, K_abf).cpu().numpy())
            diag["repl_cumulative"].append(np.zeros(R))
            p_grid = per.kde_marginal(phi, K_kde, sim.n_grid, dphi)
            diag["p_hat"].append(p_grid.cpu().numpy())
            diag["pq_l2"].append(np.full(R, np.nan)); diag["kl_pq"].append(np.full(R, np.nan))
            diag["ancestor_ess"].append(np.full(R, np.nan))
            diag["n_unique_ancestor"].append(np.full(R, N)); diag["max_ancestor_frac"].append(np.full(R, np.nan))
            diag["neff_frac"].append(opes.neff_frac()); diag["n_kernels"].append(opes.n_kernels())
            diag["frac_T"].append((cur_basin == 0).float().mean(-1).cpu().numpy())
            diag["frac_Gp"].append((cur_basin == 1).float().mean(-1).cpu().numpy())
            diag["frac_Gm"].append((cur_basin == 2).float().mean(-1).cpu().numpy())
            if do_pent:
                b2 = basin_index(cv2.value(qf).reshape(R, N), sim.basin_barrier)
                diag["frac2_T"].append((b2 == 0).float().mean(-1).cpu().numpy())
                diag["frac2_Gp"].append((b2 == 1).float().mean(-1).cpu().numpy())
                diag["frac2_Gm"].append((b2 == 2).float().mean(-1).cpu().numpy())
            else:
                for k in ("frac2_T", "frac2_Gp", "frac2_Gm"):
                    diag[k].append(np.full(R, np.nan))
        if step == sim.n_steps:
            break
        noise = torch.randn(q.shape, generator=gen_dyn, device=device, dtype=dtype)
        q = geom.remove_com(q + sim.dt * (F.reshape(R, N, Aat, 3) + bias_force) + noise_scale * noise)
        if (step + 1) % max(int(opes_cfg.pace), 1) == 0:
            opes.deposit(cv.value(q.reshape(R * N, Aat, 3)).reshape(R, N))

    out = {"method": "opes", "grid": grid.cpu().numpy(), "dphi": float(dphi),
           "runtime_seconds": time.perf_counter() - t0,
           "total_replacement_events": np.zeros(R, dtype=int),
           "n_transitions": trans_counts.cpu().numpy(),
           "n_round_trips": np.zeros(R, dtype=int),
           "birth_hist": np.zeros((R, sim.n_grid)), "death_hist": np.zeros((R, sim.n_grid)),
           "F_target_ema": None, "fr_score_std": np.full(R, np.nan),
           "fr_score_absmax": np.full(R, np.nan),
           "final_eff_counts": per.effective_counts(csum_prod, K_abf).cpu().numpy(),
           "final_neff_frac": opes.neff_frac(), "final_n_kernels": opes.n_kernels()}
    for k in diag:
        out[k] = np.asarray(diag[k])
    if do_pent:
        out["joint_hist"] = joint_hist.cpu().numpy()
        out["grid2"] = g2.cpu().numpy(); out["dphi2"] = float(dphi2)
    if verbose:
        print(f"  opes R={R} N={N}: {out['runtime_seconds']:.1f}s trans={out['n_transitions'].sum()} "
              f"neff={np.median(opes.neff_frac()):.2f}")
    return out
