"""Ethene/propene through a CHA 8-ring: the olefin/zeolite stage of the campaign.

Model (frozen in configs/uniform_campaign/cha_prereg.json before production):

  * framework   IZA all-silica CHA (R-3m expanded, orthorhombic 2x1x2 supercell,
                cache/cha/framework.npz), RIGID; only the 576 framework O
                interact with the guest.  This is the type-0 (acid-site-free)
                8-ring environment in the repo's LTA modelling convention -- a
                model system for the SAMPLING question, NOT a reproduction of
                the flexible-framework H-SAPO-34 force field of Cnudde et al.;
                rigid frameworks overestimate window barriers and the frozen
                regime classifier, not the literature, decides each cell.
  * guests      TraPPE-UA olefins, no charges:
                  ethene   CH2=CH2      eps/kB 85.0 K, sigma 3.675 A x2,
                           bond 1.33 A (harmonic k=400 kJ/mol/A^2)
                  propene  CH2=CH-CH3   (85.0/3.675), (47.0/3.73), (98.0/3.75),
                           bonds 1.33/1.54, angle 119.7 deg k=585 kJ/mol/rad^2
  * cross LJ    Lorentz-Berthelot to TraPPE-zeolite O (53.0 K, 3.30 A),
                truncated+shifted at rc=10 A.
  * confinement two-cage sphere-union wall: E += k/2 relu(min(dA,dB)-R_cage)^2
                on the COM (R_cage=6.0, k=100).  Measured on the framework:
                clips 0.0% of the two cages, admits 0.6% side-cage volume --
                declared, identical across arms and reference.
  * dynamics    overdamped Langevin (BD), dt=2e-4, the campaign convention;
                equilibrium F/U/S are dynamics-independent.
  * CV          xi = (COM - ring_centroid) . n_ring, LINEAR, ring plane fixed
                from the rigid framework (best-fit SVD plane of the 8 ring O).
                Non-periodic; the sphere union bounds the range naturally.

The FR birth-death and genealogy machinery is IMPORTED from the closed alkanes
engine; the 1-D estimator here is the straightforward non-periodic analogue of
the alkanes/LTA code (Gaussian-smoothed binned mean force, trapezoid F).
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from alkanes.core import _ancestor_stats, _birth_death, assert_no_reference_leakage

EPS = 1.0e-12
KB = 0.008314462618          # kJ/mol/K
FR_METHODS = ("fr_estimated", "fr_uniform", "fr_oracle")
ALL_METHODS = ("abf",) + FR_METHODS

GUESTS = {
    "ethene": dict(
        eps_K=[85.0, 85.0], sigma=[3.675, 3.675], mass=[14.027, 14.027],
        bonds=[(0, 1, 1.33)], angles=[]),
    "propene": dict(
        eps_K=[85.0, 47.0, 98.0], sigma=[3.675, 3.73, 3.75],
        mass=[14.027, 13.019, 15.035],
        bonds=[(0, 1, 1.33), (1, 2, 1.54)],
        angles=[(0, 1, 2, math.radians(119.7))]),
}
O_EPS_K, O_SIGMA = 53.0, 3.30           # TraPPE-zeolite framework oxygen
K_BOND = 400.0                          # kJ/mol/A^2  (0.5 k dx^2)
K_ANGLE = 585.0                         # kJ/mol/rad^2
RC = 10.0
K_CONF = 100.0


@dataclass
class CHASimConfig:
    dt: float = 2.0e-4
    n_steps: int = 400_000
    n_replicas: int = 1024
    save_every: int = 4_000
    rng_seed: int = 20260830
    # xi grid / estimator (Angstrom units, non-periodic)
    xi_lo: float = -11.0
    xi_hi: float = 10.5
    n_grid: int = 180
    abf_bandwidth: float = 0.15
    kde_bandwidth: float = 0.25
    abf_bias_scale: float = 1.0
    abf_warmup_steps: int = 25_000
    abf_force_clip: float = 30.0        # kJ/mol/A on the CV force
    estimator_burn_in_steps: int = 25_000
    abf_min_count: float = 20.0
    # Fisher--Rao (rate frozen by SAFETY-ONLY calibration before production)
    fr_rate: float = 0.10
    score_clip: float = 2.0
    fr_start_steps: int = 25_000        # = warmup end (the LTA sweep lesson)
    fr_every: int = 5
    target_ema_rate: float = 0.005
    max_event_fraction: float = 0.02
    # region bookkeeping (diagnostics): window if |xi| < window_half, else cage
    window_half: float = 1.5

    def config_hash(self):
        import hashlib, json
        return hashlib.md5(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]


class CHASystem:
    """Rigid CHA + one batched TraPPE-UA olefin per replica."""

    def __init__(self, guest, temperature, device, dtype=torch.float64, root=".",
                 compile=None):
        # compile=True (default on CUDA): route forces AND energies through one
        # torch.compile'd kernel -- measured 13x on H200, parity vs eager 5e-7
        # (f64 roundoff).  ONE engine path for reference/screen/calibration/
        # production alike.
        import os
        z = np.load(os.path.join(root, "cache/cha/framework.npz"), allow_pickle=True)
        self.guest = guest
        g = GUESTS[guest]
        self.n_beads = len(g["mass"])
        self.temperature = float(temperature)
        self.beta = 1.0 / (KB * self.temperature)
        self.device, self.dtype = device, dtype
        self.box = torch.as_tensor(z["box"], device=device, dtype=dtype)
        self.o_pos = torch.as_tensor(z["o_pos"], device=device, dtype=dtype)
        self.center = torch.as_tensor(z["window_center"], device=device, dtype=dtype)
        self.normal = torch.as_tensor(z["window_normal"], device=device, dtype=dtype)
        self.cage_A = torch.as_tensor(z["cage_A"], device=device, dtype=dtype)
        self.cage_B = torch.as_tensor(z["cage_B"], device=device, dtype=dtype)
        self.xi_A, self.xi_B = float(z["xi_A"]), float(z["xi_B"])
        self.R_cage = float(z["R_cage"])
        m = torch.as_tensor(g["mass"], device=device, dtype=dtype)
        self.mass_w = (m / m.sum())                      # COM weights (n_beads,)
        eps_g = torch.as_tensor(g["eps_K"], device=device, dtype=dtype) * KB
        sig_g = torch.as_tensor(g["sigma"], device=device, dtype=dtype)
        self.eps_x = torch.sqrt(eps_g * (O_EPS_K * KB))  # (n_beads,)
        self.sig_x = 0.5 * (sig_g + O_SIGMA)
        sr6 = (self.sig_x / RC) ** 6
        self.v_rc = 4.0 * self.eps_x * (sr6 * sr6 - sr6)  # per-bead shift
        self.bonds = g["bonds"]
        self.angles = g["angles"]
        if compile is None:
            compile = (torch.device(device).type == "cuda")
        self._terms = (torch.compile(self._energy_terms, dynamic=False)
                       if compile else self._energy_terms)

    def _min_image(self, d):
        return d - self.box * torch.round(d / self.box)

    def _energy_terms(self, q):
        """(E_bonded+conf (B,), E_nonbond (B,)) for q (B, n_beads, 3), unwrapped."""
        Eb = torch.zeros(q.shape[0], device=q.device, dtype=q.dtype)
        for (i, j, r0) in self.bonds:
            r = (q[:, i] - q[:, j]).norm(dim=-1)
            Eb = Eb + 0.5 * K_BOND * (r - r0) ** 2
        for (i, j, k, th0) in self.angles:
            a = q[:, i] - q[:, j]
            b = q[:, k] - q[:, j]
            cth = (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(EPS)
            th = torch.arccos(cth.clamp(-1 + 1e-12, 1 - 1e-12))
            Eb = Eb + 0.5 * K_ANGLE * (th - th0) ** 2
        # confinement on the COM
        com = (q * self.mass_w[None, :, None]).sum(dim=1)
        dA = self._min_image(com - self.cage_A).norm(dim=-1)
        dB = self._min_image(com - self.cage_B).norm(dim=-1)
        Eb = Eb + 0.5 * K_CONF * torch.relu(torch.minimum(dA, dB) - self.R_cage) ** 2
        # guest-framework LJ (truncated + shifted)
        d = q[:, :, None, :] - self.o_pos[None, None, :, :]
        d = self._min_image(d)
        r2 = (d * d).sum(-1)
        mask = r2 < RC ** 2
        inv_r2 = torch.where(mask, 1.0 / r2.clamp_min(EPS), torch.zeros_like(r2))
        sr6 = (self.sig_x[None, :, None] ** 2 * inv_r2) ** 3
        v = 4.0 * self.eps_x[None, :, None] * (sr6 * sr6 - sr6) - self.v_rc[None, :, None]
        Enb = torch.where(mask, v, torch.zeros_like(v)).sum(dim=(1, 2))
        return Eb, Enb

    def potential_energy(self, q, split=False):
        Eb, Enb = self._terms(q)
        if split:
            return Eb + Enb, Enb
        return Eb + Enb

    def forces(self, q):
        qg = q.detach().requires_grad_(True)
        Eb, Enb = self._terms(qg)
        E = (Eb + Enb).sum()
        (F,) = torch.autograd.grad(E, qg)
        return -F.detach()

    # ---- CV: xi = (COM - center) . n, min-imaged relative to the window ----
    def cv_value(self, q):
        com = (q * self.mass_w[None, :, None]).sum(dim=1)
        rel = self._min_image(com - self.center)
        return (rel * self.normal).sum(-1)

    def cv_grad(self, q):
        """grad(xi): bead i gets (m_i/M) * n; |grad xi|^2 = sum_i (m_i/M)^2."""
        g = torch.zeros_like(q)
        g += self.mass_w[None, :, None] * self.normal[None, None, :]
        return g

    def cv_local_mean_force(self, q, F):
        """f_loc = -(F . grad xi)/|grad xi|^2 (linear CV, no geometric term)."""
        gg = float((self.mass_w ** 2).sum())
        fdot = (F * self.mass_w[None, :, None] * self.normal[None, None, :]).sum(dim=(1, 2))
        return -fdot / gg, self.cv_value(q), None

    def initial_conditions(self, R, N, gen, side="A"):
        """Guest near a cage centre, random orientation, small COM jitter."""
        c = self.cage_A if side == "A" else self.cage_B
        com = c[None, :] + 0.8 * torch.randn(R * N, 3, generator=gen,
                                             device=self.device, dtype=self.dtype)
        u = torch.randn(R * N, 3, generator=gen, device=self.device, dtype=self.dtype)
        u = u / u.norm(dim=-1, keepdim=True).clamp_min(EPS)
        if self.n_beads == 2:
            q = torch.stack([com + 0.665 * u, com - 0.665 * u], dim=1)
        else:
            # rough propene geometry along u with the methyl kicked sideways
            v = torch.randn(R * N, 3, generator=gen, device=self.device, dtype=self.dtype)
            v = v - (v * u).sum(-1, keepdim=True) * u
            v = v / v.norm(dim=-1, keepdim=True).clamp_min(EPS)
            b0 = com + 1.0 * u
            b1 = com - 0.33 * u
            b2 = b1 - 1.54 * (0.5 * u - 0.866 * v)
            q = torch.stack([b0, b1, b2], dim=1)
        return q.reshape(R, N, self.n_beads, 3)


# ---------------- non-periodic 1-D estimator helpers ----------------
def grid_1d(sim, device, dtype):
    edges = torch.linspace(sim.xi_lo, sim.xi_hi, sim.n_grid + 1, device=device, dtype=dtype)
    mids = 0.5 * (edges[:-1] + edges[1:])
    return mids, float(edges[1] - edges[0])


def kernel_matrix(mids, bw):
    d = mids[:, None] - mids[None, :]
    K = torch.exp(-0.5 * (d / bw) ** 2)
    return K


def bin_index(xi, sim):
    i = torch.floor((xi - sim.xi_lo) / (sim.xi_hi - sim.xi_lo) * sim.n_grid).long()
    return i.clamp(0, sim.n_grid - 1)


def bin_counts(xi, sim):
    R, N = xi.shape
    out = torch.zeros(R, sim.n_grid, device=xi.device, dtype=xi.dtype)
    out.scatter_add_(1, bin_index(xi, sim), torch.ones_like(xi))
    return out


def bin_sum(xi, w, sim):
    R, N = xi.shape
    out = torch.zeros(R, sim.n_grid, device=xi.device, dtype=xi.dtype)
    out.scatter_add_(1, bin_index(xi, sim), w)
    return out


def smooth(rows, K):
    return rows @ K


def mean_force_profile(fsum, csum, K, min_count):
    num = smooth(fsum, K)
    den = smooth(csum, K)
    return num / (den + min_count + EPS)


def free_energy_from_mean_force(mf, dz):
    F = torch.cumsum(0.5 * (mf[:, 1:] + mf[:, :-1]) * dz, dim=1)
    F = torch.cat([torch.zeros_like(F[:, :1]), F], dim=1)
    return F - F.mean(dim=1, keepdim=True)


def interp_profile(prof, mids, xi):
    """Linear interpolation of (R,G) profiles at (R,N) points, clamped ends."""
    R, G = prof.shape
    lo, hi = float(mids[0]), float(mids[-1])
    dz = (hi - lo) / (G - 1)
    t = ((xi - lo) / dz).clamp(0.0, G - 1 - 1e-9)
    i0 = t.floor().long()
    frac = t - i0.to(xi.dtype)
    p0 = torch.gather(prof, 1, i0)
    p1 = torch.gather(prof, 1, (i0 + 1).clamp(max=G - 1))
    return p0 * (1 - frac) + p1 * frac


def kde_marginal(xi, K, sim, dz):
    p = smooth(bin_counts(xi, sim), K)
    return p / (p.sum(dim=1, keepdim=True) * dz).clamp_min(EPS)


def _recentered_clipped_score(raw, clip):
    s = raw - raw.mean(-1, keepdim=True)
    for _ in range(3):
        s = torch.clamp(s, -clip, clip)
        s = s - s.mean(-1, keepdim=True)
    return s


def fr_score_1d(xi, mids, dz, K_kde, q_grid, sim, clip):
    p_grid = kde_marginal(xi, K_kde, sim, dz)
    p_at = interp_profile(p_grid, mids, xi)
    q_at = interp_profile(q_grid, mids, xi)
    lr = torch.log(p_grid.clamp_min(EPS)) - torch.log(q_grid.clamp_min(EPS))
    kl = (p_grid * lr).sum(-1) * dz
    raw = (torch.log(p_at.clamp_min(EPS)) - torch.log(q_at.clamp_min(EPS))
           - kl[:, None])
    return _recentered_clipped_score(raw, clip), p_grid, kl


def fr_target_1d(method, sim, mids, dz, F_ema, B_n, oracle, beta):
    R = B_n.shape[0]
    if method == "abf":
        return None
    if method == "fr_uniform":
        q = torch.ones(R, sim.n_grid, device=mids.device, dtype=mids.dtype)
        return q / (q.sum(dim=1, keepdim=True) * dz)
    if method == "fr_oracle":
        log_q = -beta * (oracle[None, :] - B_n)
    else:
        if F_ema is None:
            return None
        log_q = -beta * (F_ema - B_n)
    log_q = log_q - log_q.max(-1, keepdim=True).values
    q = torch.exp(log_q)
    return q / (q.sum(dim=1, keepdim=True) * dz).clamp_min(EPS)


# --------------------------------- sampler ---------------------------------
def run_sampler(method, system: CHASystem, sim: CHASimConfig, seeds,
                oracle_free_energy=None, init_side="A", verbose=True):
    """R = len(seeds) matched-seed replicas of ``method`` in one process.

    Same conventions as the LTA engine: seeds are labels, pairing across arms
    comes from the shared rng_seed streams; FR consumes an independent stream.
    """
    if method not in ALL_METHODS:
        raise ValueError(f"unknown method {method!r}")
    assert_no_reference_leakage(method, oracle_free_energy)
    is_fr = method in FR_METHODS
    device, dtype = system.device, system.dtype
    R, N = len(seeds), sim.n_replicas
    beta = system.beta
    mids, dz = grid_1d(sim, device, dtype)
    K_abf = kernel_matrix(mids, sim.abf_bandwidth)
    K_kde = kernel_matrix(mids, sim.kde_bandwidth)
    gen_dyn = torch.Generator(device=device).manual_seed(int(sim.rng_seed))
    gen_fr = torch.Generator(device=device).manual_seed(int(sim.rng_seed) + 987654321)

    oracle = None
    if method == "fr_oracle":
        oracle = torch.as_tensor(oracle_free_energy, device=device, dtype=dtype)
        oracle = oracle - oracle.mean()

    q = system.initial_conditions(R, N, gen_dyn, side=init_side)
    noise_scale = math.sqrt(2.0 * sim.dt / beta)
    nb = system.n_beads

    fsum = torch.zeros(R, sim.n_grid, device=device, dtype=dtype)
    csum = torch.zeros(R, sim.n_grid, device=device, dtype=dtype)
    fsum_p = torch.zeros(R, sim.n_grid, device=device, dtype=dtype)
    csum_p = torch.zeros(R, sim.n_grid, device=device, dtype=dtype)
    usum_p = torch.zeros(R, sim.n_grid, device=device, dtype=dtype)
    unbsum_p = torch.zeros(R, sim.n_grid, device=device, dtype=dtype)
    F_ema = None
    ancestors = (torch.arange(N, device=device).expand(R, N).clone() if is_fr else None)
    total_repl = torch.zeros(R, dtype=torch.long)
    prev_reg = None
    prev_side = None
    crossings = torch.zeros(R, N, dtype=torch.long, device=device)
    score_std_sum = np.zeros(R); score_absmax = np.zeros(R); n_score = 0

    diag = {k: [] for k in ["steps", "times", "mean_force", "pmf", "p_hat", "q_target",
                            "eff_counts", "ancestor_ess", "n_unique_ancestor",
                            "max_ancestor_frac", "repl_cumulative", "kl_uniform",
                            "kl_pq", "frac_A", "frac_win", "frac_B", "n_visited_bins"]}
    soft_start = 2_000        # clash-safe start: random placements can touch the wall
    t0 = time.perf_counter()
    for step in range(sim.n_steps + 1):
        qf = q.reshape(R * N, nb, 3)
        F = system.forces(qf)
        if step < soft_start:
            fn = F.norm(dim=-1, keepdim=True).clamp_min(EPS)
            F = F * torch.clamp(fn, max=500.0) / fn
        f_loc, xi_f, _ = system.cv_local_mean_force(qf, F)
        xi = xi_f.reshape(R, N)
        f_loc = f_loc.reshape(R, N).clamp(-sim.abf_force_clip * 8, sim.abf_force_clip * 8)

        fsum += bin_sum(xi, f_loc, sim)
        csum += bin_counts(xi, sim)
        if step >= sim.estimator_burn_in_steps:
            fsum_p += bin_sum(xi, f_loc, sim)
            csum_p += bin_counts(xi, sim)
            if step % 5 == 0:      # U(xi) conditionals at 1-in-5 stride (declared)
                et, en = system.potential_energy(qf, split=True)
                usum_p += bin_sum(xi, et.reshape(R, N), sim)
                unbsum_p += bin_sum(xi, en.reshape(R, N), sim)

        mf = mean_force_profile(fsum, csum, K_abf, sim.abf_min_count)
        A_hat = free_energy_from_mean_force(mf, dz)
        ramp = min(1.0, step / max(sim.abf_warmup_steps, 1))
        B_n = sim.abf_bias_scale * ramp * A_hat
        mf_at = interp_profile(mf, mids, xi).clamp(-sim.abf_force_clip, sim.abf_force_clip)
        bias_gen = (sim.abf_bias_scale * ramp) * mf_at        # +A'(xi) generalized
        bias_force = (bias_gen.reshape(R * N)[:, None, None]
                      * system.mass_w[None, :, None]
                      * system.normal[None, None, :]).reshape(R, N, nb, 3)

        if method == "fr_estimated" and (step + 1) >= sim.fr_start_steps:
            if F_ema is None:
                F_ema = A_hat.clone()
            else:
                rt = sim.target_ema_rate
                F_ema = (1 - rt) * F_ema + rt * A_hat
            F_ema = F_ema - F_ema.mean(-1, keepdim=True)

        reg = torch.ones_like(xi, dtype=torch.long)           # 1 = window band
        reg = torch.where(xi < -sim.window_half, torch.zeros_like(reg), reg)
        reg = torch.where(xi > sim.window_half, 2 * torch.ones_like(reg), reg)
        in_cage = reg != 1
        side_now = torch.where(in_cage, reg, torch.full_like(reg, -1))
        if prev_reg is None:
            prev_side = side_now.clone()
        else:
            # a completed cage-to-cage transit: last cage side differs from the
            # cage just arrived in (window passages in between do not reset it)
            arrived = in_cage & (prev_side >= 0) & (side_now >= 0) & (prev_side != side_now)
            crossings += arrived.long()
        prev_reg = reg
        prev_side = torch.where(in_cage, side_now, prev_side)

        if step % sim.save_every == 0 or step == sim.n_steps:
            est_f = fsum_p if csum_p.sum() > 0 else fsum
            est_c = csum_p if csum_p.sum() > 0 else csum
            mf_rep = mean_force_profile(est_f, est_c, K_abf, sim.abf_min_count)
            pmf_rep = free_energy_from_mean_force(mf_rep, dz)
            diag["steps"].append(step); diag["times"].append(step * sim.dt)
            diag["mean_force"].append(mf_rep.cpu().numpy())
            diag["pmf"].append(pmf_rep.cpu().numpy())
            diag["eff_counts"].append(smooth(csum, K_abf).cpu().numpy())
            diag["repl_cumulative"].append(total_repl.numpy().copy())
            p_grid = kde_marginal(xi, K_kde, sim, dz)
            diag["p_hat"].append(p_grid.cpu().numpy())
            u_dens = 1.0 / (sim.xi_hi - sim.xi_lo)
            diag["kl_uniform"].append(((p_grid * (torch.log(p_grid.clamp_min(EPS))
                                                  - math.log(u_dens))).sum(-1) * dz)
                                      .cpu().numpy())
            q_grid = fr_target_1d(method, sim, mids, dz, F_ema, B_n, oracle, beta)
            if q_grid is not None:
                lr = torch.log(p_grid.clamp_min(EPS)) - torch.log(q_grid.clamp_min(EPS))
                diag["q_target"].append(q_grid.cpu().numpy())
                diag["kl_pq"].append(((p_grid * lr).sum(-1) * dz).cpu().numpy())
            else:
                diag["q_target"].append(np.full((R, sim.n_grid), np.nan))
                diag["kl_pq"].append(np.full(R, np.nan))
            if is_fr:
                ess, nuq, maxf = _ancestor_stats(ancestors.cpu(), N)
            else:
                ess, nuq, maxf = np.full(R, np.nan), np.full(R, N), np.full(R, np.nan)
            diag["ancestor_ess"].append(ess)
            diag["n_unique_ancestor"].append(nuq)
            diag["max_ancestor_frac"].append(maxf)
            diag["frac_A"].append((reg == 0).float().mean(-1).cpu().numpy())
            diag["frac_win"].append((reg == 1).float().mean(-1).cpu().numpy())
            diag["frac_B"].append((reg == 2).float().mean(-1).cpu().numpy())
            diag["n_visited_bins"].append((csum > 0).sum(-1).cpu().numpy())

        if step == sim.n_steps:
            break

        noise = torch.randn(q.shape, generator=gen_dyn, device=device, dtype=dtype)
        q = q + sim.dt * (F.reshape(R, N, nb, 3) + bias_force) + noise_scale * noise

        if is_fr:
            nxt = step + 1
            if nxt >= sim.fr_start_steps and \
                    (nxt - sim.fr_start_steps) % max(int(sim.fr_every), 1) == 0:
                xi_new = system.cv_value(q.reshape(R * N, nb, 3)).reshape(R, N)
                q_grid = fr_target_1d(method, sim, mids, dz, F_ema, B_n, oracle, beta)
                if q_grid is not None:
                    score, p_fr, kl = fr_score_1d(xi_new, mids, dz, K_kde, q_grid,
                                                  sim, sim.score_clip)
                    ss = score.detach().cpu().numpy()
                    score_std_sum += ss.std(axis=1)
                    score_absmax = np.maximum(score_absmax, np.abs(ss).max(axis=1))
                    n_score += 1
                    q, ancestors, n_repl, deaths, births = _birth_death(
                        q, score, ancestors, sim, gen_fr)
                    total_repl += n_repl.cpu()
                    for r in range(R):
                        if deaths[r] is not None and deaths[r].numel() > 0:
                            crossings[r, deaths[r]] = crossings[r].index_select(0, births[r])
                            prev_side[r, deaths[r]] = prev_side[r].index_select(0, births[r])
                            prev_reg[r, deaths[r]] = prev_reg[r].index_select(0, births[r])

    u_of_xi = (usum_p / csum_p.clamp_min(1.0)).cpu().numpy()
    unb_of_xi = (unbsum_p / csum_p.clamp_min(1.0)).cpu().numpy()
    out = {"method": method, "grid": mids.cpu().numpy(), "dz": dz,
           "xi_A": system.xi_A, "xi_B": system.xi_B, "guest": system.guest,
           "temperature": system.temperature,
           "runtime_seconds": time.perf_counter() - t0,
           "total_replacement_events": total_repl.numpy(),
           "n_crossings": crossings.sum(-1).cpu().numpy(),
           "F_target_ema": (F_ema.cpu().numpy() if F_ema is not None else None),
           "fr_score_std": (score_std_sum / max(n_score, 1)),
           "fr_score_absmax": score_absmax,
           "u_of_xi": u_of_xi, "u_nonbond_of_xi": unb_of_xi,
           "u_counts": csum_p.cpu().numpy()}
    for k in diag:
        out[k] = np.asarray(diag[k])
    if verbose:
        print(f"  {method:12s} {system.guest} T={system.temperature:g} R={R} N={N}: "
              f"{out['runtime_seconds']:.1f}s repl={out['total_replacement_events'].sum()} "
              f"crossings={out['n_crossings'].sum()}", flush=True)
    return out


# --------------------------------- umbrella ---------------------------------
def run_umbrella(system: CHASystem, sim: CHASimConfig, centers, kappa,
                 n_steps, n_replicas, burn_in, sample_every, seed, verbose=True):
    """Harmonic umbrella windows on xi for the independent reference."""
    device, dtype = system.device, system.dtype
    W = len(centers)
    c = torch.as_tensor(centers, device=device, dtype=dtype).reshape(W, 1)
    gen = torch.Generator(device=device).manual_seed(int(seed))
    half = W // 2
    qA = system.initial_conditions(1, half * n_replicas, gen, side="A")
    qB = system.initial_conditions(1, (W - half) * n_replicas, gen, side="B")
    q = torch.cat([qA.reshape(half, n_replicas, system.n_beads, 3),
                   qB.reshape(W - half, n_replicas, system.n_beads, 3)], dim=0)
    # order windows so the first half starts from cage A (low xi), rest from B
    order = torch.argsort(c.flatten())
    c = c.flatten()[order].reshape(W, 1)
    nb = system.n_beads
    noise_scale = math.sqrt(2.0 * sim.dt / system.beta)
    soft_start = 5_000
    phis, us, unbs = [], [], []
    t0 = time.perf_counter()
    gg = float((system.mass_w ** 2).sum())
    for step in range(n_steps):
        qf = q.reshape(W * n_replicas, nb, 3)
        F = system.forces(qf)
        if step < soft_start:
            fn = F.norm(dim=-1, keepdim=True).clamp_min(EPS)
            F = F * torch.clamp(fn, max=500.0) / fn
        xi = system.cv_value(qf).reshape(W, n_replicas)
        Fu_gen = (-kappa * (xi - c)).reshape(W * n_replicas)
        Fu = (Fu_gen[:, None, None] * system.mass_w[None, :, None]
              * system.normal[None, None, :])
        noise = torch.randn(q.shape, generator=gen, device=device, dtype=dtype)
        q = q + sim.dt * (F + Fu).reshape(W, n_replicas, nb, 3) + noise_scale * noise
        if step >= burn_in and step % sample_every == 0:
            phis.append(xi.detach().cpu().numpy().copy())
            et, en = system.potential_energy(q.reshape(W * n_replicas, nb, 3), split=True)
            us.append(et.reshape(W, n_replicas).cpu().numpy().copy())
            unbs.append(en.reshape(W, n_replicas).cpu().numpy().copy())
    if verbose:
        print(f"  umbrella: {W} windows x {n_replicas}, {n_steps} steps "
              f"-> {len(phis)} frames in {time.perf_counter() - t0:.1f}s", flush=True)
    return np.array(phis), np.array(us), np.array(unbs)


def wham_1d_line(phi_samples, centers, kappa, beta, lo, hi, n_bins=180):
    """Non-periodic histogram WHAM on [lo, hi]."""
    W = phi_samples.shape[1]
    edges = np.linspace(lo, hi, n_bins + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])
    dzb = edges[1] - edges[0]
    hist = np.zeros((W, n_bins))
    for w in range(W):
        h, _ = np.histogram(phi_samples[:, w, :].ravel(), bins=edges)
        hist[w] = h
    n_w = hist.sum(axis=1)
    bias = 0.5 * kappa * (mids[None, :] - np.asarray(centers)[:, None]) ** 2
    f_w = np.zeros(W)
    for _ in range(20000):
        denom = (n_w[:, None] * np.exp(beta * (f_w[:, None] - bias))).sum(axis=0)
        p = hist.sum(axis=0) / np.maximum(denom, 1e-300)
        p = p / max(p.sum() * dzb, 1e-300)
        f_new = -np.log(np.maximum((np.exp(-beta * bias) * p[None, :] * dzb), 1e-300)
                        .sum(axis=1)) / beta
        if np.abs(f_new - f_new.mean() - (f_w - f_w.mean())).max() < 1e-10:
            f_w = f_new
            break
        f_w = f_new
    with np.errstate(divide="ignore"):
        F = -np.log(np.maximum(p, 1e-300)) / beta
    F = F - np.nanmin(F)
    return mids, F, p, hist


def conditional_u_line(phi_samples, u_samples, lo, hi, n_bins=180):
    edges = np.linspace(lo, hi, n_bins + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])
    ph, uu = phi_samples.ravel(), u_samples.ravel()
    idx = np.clip(np.digitize(ph, edges) - 1, 0, n_bins - 1)
    s = np.bincount(idx, weights=uu, minlength=n_bins)
    cnt = np.bincount(idx, minlength=n_bins)
    return mids, np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan), cnt
