"""Production core for the WCA-dimer ABF vs ABF+Fisher-Rao study.

Refactored and extended from ``scratch/abffr_core.py`` (the validated demo core).
Adds the diagnostic FR target variants (``fr_uniform``, ``fr_oracle``), Stage-C
mechanism diagnostics (RC marginal, FR target, per-bin effective counts, region
fractions, birth/death locations, ancestor ESS) and a no-oracle-leakage audit.

Method names (``method`` argument to :func:`run_sampler_gpu`):
    abf           ABF only baseline.
    fr_estimated  ABF + FR with an ONLINE EMA estimated target. The practical method.
    fr_uniform    ABF + FR with a uniform-in-z target. Naive ablation (diagnostic).
    fr_oracle     ABF + FR with the TI reference target. Positive control (diagnostic).

CRITICAL ALGORITHMIC RULE
-------------------------
Only ``fr_oracle`` may receive ``oracle_free_energy``. ``fr_estimated`` (and ``abf``
and ``fr_uniform``) MUST NOT receive it; the sampler asserts this. The TI reference
is used ONLY for post-hoc L2 evaluation, and for the ``fr_oracle`` diagnostic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, replace, asdict

import numpy as np
import torch
import torch.nn.functional as F

EPS = 1.0e-12

FR_METHODS = ("fr_estimated", "fr_uniform", "fr_oracle", "fr_estimated_adaptive")
# Matched-sham arms.  They resample -- at their partner's event times, with their partner's
# realised counts -- but the direction is randomised, so they carry no target and compute no
# score.  Each replays a per-event count sequence recorded from its own partner; a sham
# shadowing the oracle is not a control for the practical arm, because the two arms build
# different targets and therefore fire different numbers of events.
SHAM_METHODS = ("sham_practical", "sham_oracle")
#: Prior-art directed selection, added in v2 for Q1. Both depend on the CV ALONE, so the
#: invariance theorem d/dt p(y|xi)|_sel = 0 covers them exactly as it covers Fisher-Rao:
#: they redistribute the marginal and supply no selection pressure at fixed xi.
#:   book_laplacian   Lelievre-Rousset-Stoltz Ch.6:  S = c * d2p/dz2 / p
#:   count_balancing  Remark 6.10 / Comer et al. NAMD: S = c * (1 - p/p_bar)
PRIOR_SELECTION_METHODS = ("book_laplacian", "count_balancing")
SHAM_PARTNER = {"sham_practical": "fr_estimated", "sham_oracle": "fr_oracle"}
ALL_METHODS = ("abf",) + FR_METHODS + SHAM_METHODS + PRIOR_SELECTION_METHODS
# Methods that maintain the online EMA estimated target (never see the TI reference).
ESTIMATED_TARGET_METHODS = ("fr_estimated", "fr_estimated_adaptive")


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = choose_device()
DTYPE = torch.float32
if DEVICE.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


@dataclass(frozen=True)
class DimerWCAParams:
    n_dim: int = 10
    a: float = 1.5
    sigma: float = 1.0
    epsilon: float = 1.0
    h: float = 2.0
    w: float = 2.0
    beta: float = 1.0
    min_r: float = 0.65
    force_clip: float = 250.0

    @property
    def n_particles(self):
        return self.n_dim * self.n_dim

    @property
    def box_length(self):
        return self.n_dim * self.a

    @property
    def r0(self):
        return 2.0 ** (1.0 / 6.0) * self.sigma

    @property
    def wca_cutoff(self):
        return self.r0


@dataclass(frozen=True)
class SimConfig:
    n_replicas: int = 1024
    dt: float = 2.0e-3
    n_steps: int = 250_000
    save_every: int = 2500
    seed: int = 42

    z_min: float = -0.2
    z_max: float = 1.2
    n_grid: int = 160

    abf_bandwidth: float = 0.025
    kde_bandwidth: float = 0.070
    abf_smooth_sigma: float = 0.50

    mean_force_sample_clip: float = 500.0
    use_clipped_force_for_mean_force: bool = True
    abf_bias_scale: float = 1.0
    abf_edge_extrapolate: bool = True
    boundary_wall_strength: float = 80.0
    abf_force_clip: float = 40.0
    abf_warmup_steps: int = 10_000
    estimator_burn_in_steps: int = 10_000

    # Fisher-Rao (online estimated target + diagnostic variants).
    fr_rate: float = 0.10
    score_clip: float = 2.0
    fr_start_steps: int = 20_000
    fr_every: int = 5
    target_ema_rate: float = 0.005
    max_event_fraction: float = 0.02
    #: Selection intensity `c` for the prior-art arms (book_laplacian, count_balancing).
    #: Read only by those methods; every other arm is unaffected, so v1 stays bit-identical.
    prior_c: float = 1.0
    # Ancestry window for the genealogy-health gate, matching the gateway confirmatory
    # (`ess_window_steps: 4000`). The run-long ancestral ESS this file already records
    # decays monotonically toward zero for ANY birth-death process, so a fixed 0.30 floor
    # on it is really a cap on run length; the windowed statistic is the one the gate means.
    ess_window_steps: int = 4000

    # Region boundaries (compact / transition / stretched) for diagnostics.
    transition_lo: float = 0.25
    transition_hi: float = 0.75
    # Interior evaluation window (exclude wall-affected edges) as a fraction.
    eval_z_lo: float = 0.0
    eval_z_hi: float = 1.0

    # ---- Adaptive diversity-aware FR (method "fr_estimated_adaptive") ----
    # Defaults below leave the *non-adaptive* methods byte-identical; they are read
    # only when method == "fr_estimated_adaptive".  At each FR event the effective
    # rate is base_fr_rate * support_gate * diversity_gate * event_gate, clipped to
    # [adaptive_min_rate, adaptive_max_rate].  All gates use ONLINE quantities only
    # (no TI / oracle leakage).
    adaptive_fr_enabled: bool = False           # informational; method name selects it
    adaptive_base_fr_rate: float = 0.20
    # Support gate mode + EMA-smoothed ramp [support_lo, support_hi].
    # "marginal_uniform" (recommended default): ramp on L2(p_hat, uniform), the
    #   ABF-driven non-flatness of the z-marginal. This is EXOGENOUS to the FR
    #   firing (unlike p_hat-vs-target, which FR actively shrinks, creating a
    #   feedback loop) and separates the starved beta=1 cells (>= ~0.115) from the
    #   already-accurate cells (<= ~0.10) in the production data.
    # "marginal_mismatch": L2(p_hat, q_target) -- kept for ablation; noisier and
    #   reduced by FR itself, so a less stable gate.
    # "marginal_tv": TV(p_hat, uniform).  "neff_transition": kernel-denominator
    #   deficit (weak; ABF balances support across z).
    adaptive_support_mode: str = "marginal_uniform"
    adaptive_support_lo: float = 0.115          # signal <= lo  => support_gate 0
    adaptive_support_hi: float = 0.14           # signal >= hi  => support_gate 1
    adaptive_support_ema_rate: float = 0.02     # EMA smoothing of the online signal
    # Hold the rate at zero for this many steps AFTER fr_start so the cumulative
    # marginal settles before the gate acts (kills the early-transient false firing
    # in already-accurate cells; the marginal keeps accumulating during the hold).
    adaptive_gate_warmup_steps: int = 20000
    adaptive_neff_target_strategy: str = "relative_median"  # | "absolute"
    adaptive_neff_target_frac: float = 0.5      # relative-median multiplier
    adaptive_neff_target_abs: float = 0.0       # absolute Neff threshold (if "absolute")
    adaptive_tv_scale: float = 0.30             # TV scale for "marginal_tv" mode
    adaptive_ess_full_threshold: float = 0.25   # ESS_frac >= => diversity_gate 1
    adaptive_ess_stop_threshold: float = 0.10   # ESS_frac <= => diversity_gate 0
    adaptive_event_backoff_threshold: float = 0.80  # frac of max_event_fraction
    adaptive_event_backoff_factor: float = 0.50
    adaptive_min_rate: float = 0.0
    adaptive_max_rate: float = 0.20

    def config_hash(self) -> str:
        return hashlib.md5(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]


@dataclass(frozen=True)
class TIConfig:
    z_min: float = -0.2
    z_max: float = 1.2
    n_z: int = 51
    n_replicas: int = 128
    dt: float = 2.0e-3
    n_thermalization: int = 5_000
    n_steps: int = 50_000
    sample_every: int = 20
    seed: int = 314159
    smooth_sigma: float = 1.0
    z_chunk_size: int = 8
# === END PART 1 ===


def as_tensor(x, device=DEVICE, dtype=DTYPE):
    return torch.as_tensor(x, device=device, dtype=dtype)


def wrap_positions(q, L):
    return torch.remainder(q, L)


def minimum_image(delta, L):
    return delta - L * torch.round(delta / L)


def clip_forces(forces, force_clip):
    norm = torch.linalg.norm(forces, dim=-1, keepdim=True)
    scale = torch.clamp(force_clip / torch.clamp(norm, min=EPS), max=1.0)
    return forces * scale


class WCADimerEngine:
    """GPU pair-list force engine for the dimer in WCA solvent."""

    def __init__(self, params, device=DEVICE, dtype=DTYPE):
        self.params = params
        self.device = device
        self.dtype = dtype
        self.N = params.n_particles
        self.L = float(params.box_length)
        self.sigma = float(params.sigma)
        self.epsilon = float(params.epsilon)
        self.h = float(params.h)
        self.w = float(params.w)
        self.r0 = float(params.r0)
        self.cutoff = float(params.wca_cutoff)
        self.min_r = float(params.min_r)

        pair_i, pair_j = torch.triu_indices(self.N, self.N, offset=1, device=device)
        keep = ~((pair_i == 0) & (pair_j == 1))
        self.pair_i = pair_i[keep].long()
        self.pair_j = pair_j[keep].long()
        self.n_pairs = int(self.pair_i.numel())
        # Optional fused all-pairs force (env WCA_DENSE_FORCE=1; 2026-09-05, M3 speed-up).  Same
        # physics and the same float32 arithmetic per pair; the pair sum is a fused reduction over
        # a (B, N, N, 2) tensor instead of 4950 scatter-adds with atomics per replica, compiled by
        # torch.compile into one kernel (~8x faster force at B = 1024).  Not bitwise equal to the
        # scatter path (summation order), agreement ~1e-6 relative on thermalised configurations;
        # WCA arms were never bitwise-paired (the scatter path's atomics are order-nondeterministic
        # themselves), so the pairing semantics (same init, same noise stream) are unchanged.
        # ``compute_energy=True`` always takes the scatter path.  Recorded as ``force_impl``.
        # STATUS 2026-09-05: NOT deployed -- under this flag the bit-identity tests
        # (zero-budget relaxation == plain ABF, instrumentation inert) fail, i.e. a run's arithmetic
        # is not reproducible across sampler variants; the step is CPU/launch-bound anyway (~12 %
        # gain), so concurrency, not this path, is the speed lever.  Opt-in only.
        self.force_impl = "scatter"
        if os.environ.get("WCA_DENSE_FORCE", "") == "1":
            m = torch.ones(self.N, self.N, device=device, dtype=torch.bool)
            m.fill_diagonal_(False); m[0, 1] = False; m[1, 0] = False
            self._pair_mask = m.to(dtype)
            # dynamic=True: ONE kernel for every batch size, so the arithmetic of a run cannot
            # change when a relaxation sub-batch triggers a recompile (bit-identity tests)
            self._force_dense_c = torch.compile(self._force_dense, dynamic=True)
            self.force_impl = "dense_compiled"

    def _force_dense(self, q):
        d = q[:, :, None, :] - q[:, None, :, :]                                   # q_i - q_j, (B,N,N,2)
        d = d - self.L * torch.round(d / self.L)
        r = torch.linalg.norm(d, dim=-1)
        r_safe = torch.clamp(r, min=self.min_r * self.sigma)
        inv = self.sigma / r_safe
        inv6 = inv ** 6
        inv12 = inv6 ** 2
        dVdr = 4.0 * self.epsilon * (-12.0 * inv12 / r_safe + 6.0 * inv6 / r_safe)
        act = (r <= self.cutoff).to(q.dtype) * self._pair_mask
        f_pair = (-dVdr / torch.clamp(r_safe, min=EPS) * act).unsqueeze(-1) * d
        forces = f_pair.sum(2)
        d01 = minimum_image(q[:, 0, :] - q[:, 1, :], self.L)
        r01 = torch.linalg.norm(d01, dim=1).clamp_min(EPS)
        u = (r01 - self.r0 - self.w) / self.w
        dVdr_dim = -4.0 * self.h * u * (1.0 - u ** 2) / self.w
        f01 = (-dVdr_dim / r01).unsqueeze(-1) * d01
        forces[:, 0, :] += f01
        forces[:, 1, :] -= f01
        return forces

    def force(self, q, compute_energy=False):
        if self.force_impl == "dense_compiled" and not compute_energy:
            return self._force_dense_c(q)
        B = q.shape[0]
        qi = q.index_select(1, self.pair_i)
        qj = q.index_select(1, self.pair_j)
        delta = minimum_image(qi - qj, self.L)
        r = torch.linalg.norm(delta, dim=-1)
        r_safe = torch.clamp(r, min=self.min_r * self.sigma)

        active = r <= self.cutoff
        inv = self.sigma / r_safe
        inv6 = inv ** 6
        inv12 = inv6 ** 2
        dVdr = 4.0 * self.epsilon * (-12.0 * inv12 / r_safe + 6.0 * inv6 / r_safe)
        dVdr = torch.where(active, dVdr, torch.zeros_like(dVdr))
        f_pair = (-dVdr / torch.clamp(r_safe, min=EPS)).unsqueeze(-1) * delta

        forces = torch.zeros_like(q)
        idx_i = self.pair_i.view(1, -1, 1).expand(B, -1, 2)
        idx_j = self.pair_j.view(1, -1, 1).expand(B, -1, 2)
        forces.scatter_add_(1, idx_i, f_pair)
        forces.scatter_add_(1, idx_j, -f_pair)

        d01 = minimum_image(q[:, 0, :] - q[:, 1, :], self.L)
        r01 = torch.linalg.norm(d01, dim=1).clamp_min(EPS)
        u = (r01 - self.r0 - self.w) / self.w
        dVdr_dim = -4.0 * self.h * u * (1.0 - u ** 2) / self.w
        f01 = (-dVdr_dim / r01).unsqueeze(-1) * d01
        forces[:, 0, :] += f01
        forces[:, 1, :] -= f01

        if not compute_energy:
            return forces

        V_wca = 4.0 * self.epsilon * (inv12 - inv6) + self.epsilon
        V_wca = torch.where(active, V_wca, torch.zeros_like(V_wca)).sum(dim=1)
        V_dim = self.h * (1.0 - u ** 2) ** 2
        return V_wca + V_dim, forces


# ---------------------------------------------------------------------------
# Geometry, local mean force, initial conditions
# ---------------------------------------------------------------------------
def dimer_displacement_and_length(q, params):
    d01 = minimum_image(q[:, 0, :] - q[:, 1, :], params.box_length)
    r01 = torch.linalg.norm(d01, dim=1).clamp_min(EPS)
    return d01, r01


def reaction_coordinate(q, params):
    _, r01 = dimer_displacement_and_length(q, params)
    return (r01 - params.r0) / (2.0 * params.w)


def grad_xi_dimer(q, params):
    d01, r01 = dimer_displacement_and_length(q, params)
    grad0 = d01 / (2.0 * params.w * r01[:, None])
    return grad0, -grad0


def local_mean_force(q, physical_forces, params):
    d01, r01 = dimer_displacement_and_length(q, params)
    grad_difference = physical_forces[:, 1, :] - physical_forces[:, 0, :]
    energetic = (params.w / r01) * torch.sum(d01 * grad_difference, dim=1)
    entropic = (2.0 * params.w) / (params.beta * r01)
    return energetic - entropic


def add_abf_force(q, forces, mean_force_at_z, params):
    grad0, grad1 = grad_xi_dimer(q, params)
    out = forces.clone()
    out[:, 0, :] += mean_force_at_z[:, None] * grad0
    out[:, 1, :] += mean_force_at_z[:, None] * grad1
    return out


def add_reaction_coordinate_wall_force(q, forces, z, sim, params):
    if sim.boundary_wall_strength <= 0.0:
        return forces
    upper_excess = torch.clamp(z - sim.z_max, min=0.0)
    lower_excess = torch.clamp(z - sim.z_min, max=0.0)
    dU_wall_dz = sim.boundary_wall_strength * (upper_excess + lower_excess)
    return add_abf_force(q, forces, -dU_wall_dz, params)


def lattice_initial_conditions(params, n_replicas, device=DEVICE, dtype=DTYPE, seed=0, jitter=0.015):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    coords = [((0.5 + i) * params.a, (0.5 + j) * params.a)
              for i in range(params.n_dim) for j in range(params.n_dim)]
    base = torch.tensor(coords, device=device, dtype=dtype)
    q = base.unsqueeze(0).repeat(n_replicas, 1, 1)
    shifts = torch.rand((n_replicas, 1, 2), device=device, dtype=dtype, generator=g) * params.box_length
    q = wrap_positions(q + shifts, params.box_length)
    if jitter > 0.0:
        q[:, 2:, :] = wrap_positions(
            q[:, 2:, :] + jitter * torch.randn(q[:, 2:, :].shape, device=device, dtype=dtype, generator=g),
            params.box_length,
        )
    q[:, 1, :] = wrap_positions(q[:, 0, :] + torch.tensor([0.0, params.r0], device=device, dtype=dtype), params.box_length)
    return q


def project_dimer_to_z(q, z, params):
    z = torch.as_tensor(z, device=q.device, dtype=q.dtype)
    if z.ndim == 0:
        z = z.expand(q.shape[0])
    target_r = params.r0 + 2.0 * params.w * z
    d01, r01 = dimer_displacement_and_length(q, params)
    direction = d01 / r01[:, None]
    midpoint = q[:, 1, :] + 0.5 * d01
    target_d = target_r[:, None] * direction
    out = q.clone()
    out[:, 0, :] = midpoint + 0.5 * target_d
    out[:, 1, :] = midpoint - 0.5 * target_d
    return wrap_positions(out, params.box_length)


# ---------------------------------------------------------------------------
# KDE / ABF helpers
# ---------------------------------------------------------------------------
def gaussian_kernel_torch(diff, bandwidth):
    bw = float(max(bandwidth, EPS))
    return torch.exp(-0.5 * (diff / bw) ** 2) / (bw * math.sqrt(2.0 * math.pi))


def smooth_profile_torch(y, sigma_grid_points):
    sigma = float(sigma_grid_points)
    if sigma <= 0:
        return y.clone()
    radius = max(1, int(math.ceil(4.0 * sigma)))
    x = torch.arange(-radius, radius + 1, device=y.device, dtype=y.dtype)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()
    yp = F.pad(y.view(1, 1, -1), (radius, radius), mode="replicate")
    return F.conv1d(yp, kernel.view(1, 1, -1)).view(-1)


def cumulative_trapezoid_torch(y, x):
    out = torch.zeros_like(y)
    out[1:] = torch.cumsum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1]), dim=0)
    return out


def normalize_profile_zero_at_midpoint_torch(profile, grid, midpoint=0.5):
    idx = torch.argmin(torch.abs(grid - midpoint))
    return profile - profile[idx]


def trapz_torch(y, x):
    return torch.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1]))


def normalize_density_on_grid_torch(p_grid, grid, eps=EPS):
    p_grid = torch.clamp(p_grid, min=eps)
    mass = trapz_torch(p_grid, grid)
    return p_grid / torch.clamp(mass, min=eps)


def kde_1d_torch(eval_points, samples, bandwidth, z_min, z_max):
    reflected = torch.cat([samples, 2.0 * z_min - samples, 2.0 * z_max - samples])
    diff = eval_points[:, None] - reflected[None, :]
    p = gaussian_kernel_torch(diff, bandwidth).sum(dim=1) / max(samples.numel(), 1)
    return torch.clamp(p, min=EPS)


def free_energy_from_density_torch(p_grid, grid, beta):
    Fz = -(1.0 / beta) * torch.log(torch.clamp(p_grid, min=EPS))
    return normalize_profile_zero_at_midpoint_torch(Fz, grid)


def mean_force_from_free_energy_torch(free_energy, grid):
    out = torch.empty_like(free_energy)
    out[1:-1] = (free_energy[2:] - free_energy[:-2]) / (grid[2:] - grid[:-2])
    out[0] = (free_energy[1] - free_energy[0]) / (grid[1] - grid[0])
    out[-1] = (free_energy[-1] - free_energy[-2]) / (grid[-1] - grid[-2])
    return out


def interp_uniform_grid(profile, grid, z, outside_value=0.0):
    dz = grid[1] - grid[0]
    x = (z - grid[0]) / dz
    idx0 = torch.floor(x).long()
    inside = (idx0 >= 0) & (idx0 < grid.numel() - 1)
    idx0c = idx0.clamp(0, grid.numel() - 2)
    frac = (x - idx0c.to(z.dtype)).clamp(0.0, 1.0)
    val = (1.0 - frac) * profile[idx0c] + frac * profile[idx0c + 1]
    return torch.where(inside, val, torch.full_like(val, outside_value))


def interp_uniform_grid_edge(profile, grid, z):
    dz = grid[1] - grid[0]
    x = (z - grid[0]) / dz
    idx0 = torch.floor(x).long()
    idx0c = idx0.clamp(0, grid.numel() - 2)
    frac = (x - idx0c.to(z.dtype)).clamp(0.0, 1.0)
    val = (1.0 - frac) * profile[idx0c] + frac * profile[idx0c + 1]
    val = torch.where(z < grid[0], profile[0].expand_as(val), val)
    val = torch.where(z > grid[-1], profile[-1].expand_as(val), val)
    return val


class TorchKernelABFEstimator:
    def __init__(self, z_grid, bandwidth, smooth_sigma=0.0, edge_extrapolate=False):
        self.z_grid = z_grid
        self.bandwidth = float(bandwidth)
        self.smooth_sigma = float(smooth_sigma)
        self.edge_extrapolate = bool(edge_extrapolate)
        self.num = torch.zeros_like(z_grid)
        self.den = torch.zeros_like(z_grid)
        self.n_updates = 0

    def update(self, z_samples, force_samples):
        weights = gaussian_kernel_torch(self.z_grid[:, None] - z_samples[None, :], self.bandwidth)
        self.num += torch.sum(weights * force_samples[None, :], dim=1)
        self.den += torch.sum(weights, dim=1)
        self.n_updates += int(z_samples.numel())

    def mean_force_profile(self):
        raw = self.num / torch.clamp(self.den, min=EPS)
        raw = torch.where(self.den > EPS, raw, torch.zeros_like(raw))
        return smooth_profile_torch(raw, self.smooth_sigma)

    def evaluate(self, z_samples):
        profile = self.mean_force_profile()
        if self.edge_extrapolate:
            return interp_uniform_grid_edge(profile, self.z_grid, z_samples)
        return interp_uniform_grid(profile, self.z_grid, z_samples, outside_value=0.0)

    def pmf_profile(self):
        pmf = cumulative_trapezoid_torch(self.mean_force_profile(), self.z_grid)
        return normalize_profile_zero_at_midpoint_torch(pmf, self.z_grid)

    def effective_counts(self):
        """Kernel-accumulated weight per grid bin (proxy for N_eff(z_j))."""
        return self.den.clone()


class ReadoutBank:
    """Report-only read-out estimators at extra bandwidths + RAW binned accumulators.

    The engine's kernel estimator accumulates weights AT the sample positions, so a
    finished run cannot be re-scored at another bandwidth (nothing raw exists on
    disk; ``effective_counts`` is already smoothed).  This bank runs additional
    ``TorchKernelABFEstimator`` instances alongside, mirroring the engine's own
    two-phase convention (all-steps estimator until the burn-in ends, post-burn-in
    estimator afterwards) so every read-out is exactly what the production profile
    would have been at that bandwidth.  ``h <= 0`` requests the raw binned sums
    (nearest grid node, one-hot matmul so the sums are deterministic); those are
    reported unsmoothed so any read-out kernel can be applied offline.

    INERT: nothing here is evaluated for the bias force and no RNG is consumed, so
    the dynamics are byte-identical with or without a bank.
    """

    def __init__(self, grid, sim, bandwidths):
        self.grid = grid
        self.G = int(grid.numel())
        self.z_min = float(grid[0])
        self.dz = float((grid[-1] - grid[0]) / max(self.G - 1, 1))
        self.hs = [float(h) for h in bandwidths]
        self.kernels = {h: [TorchKernelABFEstimator(grid, h, sim.abf_smooth_sigma),
                            TorchKernelABFEstimator(grid, h, sim.abf_smooth_sigma)]
                        for h in self.hs if h > 0}
        self.raw = None
        if any(h <= 0 for h in self.hs):
            self.raw = [[torch.zeros_like(grid), torch.zeros_like(grid)],   # fsum: all / prod
                        [torch.zeros_like(grid), torch.zeros_like(grid)]]   # csum: all / prod
        self.n_prod = 0
        self._ar = torch.arange(self.G, device=grid.device)

    def update(self, z, f, in_production):
        phases = (0, 1) if in_production else (0,)
        for ests in self.kernels.values():
            for p in phases:
                ests[p].update(z, f)
        if self.raw is not None:
            idx = torch.round((z - self.z_min) / self.dz).long().clamp_(0, self.G - 1)
            onehot = (idx[:, None] == self._ar[None, :]).to(f.dtype)          # (N, G)
            fs = f @ onehot
            cs = onehot.sum(0)
            for p in phases:
                self.raw[0][p] += fs
                self.raw[1][p] += cs
        if in_production:
            self.n_prod += 1

    def report(self):
        """dict: h -> mean-force profile (engine convention incl. smooth_sigma);
        'raw_fsum'/'raw_csum' -> unsmoothed binned sums of the reported phase."""
        p = 1 if self.n_prod > 0 else 0
        out = {h: to_numpy(ests[p].mean_force_profile()) for h, ests in self.kernels.items()}
        if self.raw is not None:
            out["raw_fsum"] = to_numpy(self.raw[0][p])
            out["raw_csum"] = to_numpy(self.raw[1][p])
        return out


# ---------------------------------------------------------------------------
# Targeted solvent relaxation (targeted-relax campaign): online sensitivity field,
# budgeted water-filling allocation, frozen-dimer inner relaxation.  ADDITIVE: nothing
# here runs unless ``run_sampler_gpu`` is handed a ``RelaxConfig`` or asked to record
# the sensitivity accumulator, so every accepted path stays byte-identical.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RelaxConfig:
    """Targeted constrained solvent relaxation at fixed dimer coordinate.

    At every FR opportunity (same schedule as birth-death, applied AFTER it), each replica
    ``i`` receives an inner relaxation of ``m_i`` steps with its dimer frozen and the solvent
    evolving under the physical potential, allocated by water-filling
    ``t_i = tau_i/2 [log(2 a_i / (lambda tau_i))]_+`` with importance ``a_i`` = the ONLINE
    conditional variance of the local mean force at the replica's z (never the reference),
    ``tau_i`` = the frozen local force-correlation time map, and the multiplier set so that
    ``sum_i m_i = rho x n_replicas x fr_every`` replica-steps (rho = notional extra cost).
    ``target='random'`` keeps the same multiset of durations and permutes it across replicas
    (the cost-matched control).  Inner steps deposit nothing into any estimator.
    """
    rho: float
    tau_grid: tuple                     # tau_f on the sim z-grid (already floored/capped)
    target: str = "sensitivity"         # 'sensitivity' | 'random'
    inner_seed_offset: int = 7_000_000  # inner noise stream: its own generator, never the global one
    min_count: float = 2.0              # bins need C > 1 for a variance
    # Inner integration scheme (amendment A2): 'frozen' = dimer atoms fixed, solvent moves (W1);
    # 'projected' = every particle moves and the dimer is re-projected to its z after each step --
    # the TI reference's own scheme, hence the same discretised stationary law as the reference.
    scheme: str = "frozen"


class SensitivityAccumulator:
    """Raw binned sums ``C, Sf, Sf2`` of the local mean force (nearest grid node, one-hot
    matmul, deterministic) and the ONLINE sensitivity profile built from them.

    ``profile()``: per-bin variance FIRST (with the finite-sample correction ``C/(C-1)``;
    bins with ``C <= 1`` contribute nothing), then count-weighted Gaussian-kernel smoothing at
    the inherited ABF bandwidth, then the inherited ``smooth_sigma`` grid smoothing -- i.e. the
    same pipeline as the mean-force profile, applied to a variance.  This is the estimator the
    gateway D0 audit found floor-free (smoothing the moments first leaks the mean-force
    gradient across the kernel window into the variance).  Report-only unless a RelaxConfig
    reads it; consumes no RNG.
    """

    def __init__(self, grid):
        self.grid = grid
        self.G = int(grid.numel())
        self.z_min = float(grid[0])
        self.dz = float((grid[-1] - grid[0]) / max(self.G - 1, 1))
        self.C = torch.zeros_like(grid)
        self.Sf = torch.zeros_like(grid)
        self.Sf2 = torch.zeros_like(grid)
        self._ar = torch.arange(self.G, device=grid.device)
        self._K = {}

    def update(self, z, f):
        idx = torch.round((z - self.z_min) / self.dz).long().clamp_(0, self.G - 1)
        onehot = (idx[:, None] == self._ar[None, :]).to(f.dtype)
        self.C += onehot.sum(0)
        self.Sf += f @ onehot
        self.Sf2 += (f * f) @ onehot

    def bin_variance(self):
        C = self.C
        ok = C > 1.0
        Cs = torch.clamp(C, min=1.0)
        v = (self.Sf2 / Cs - (self.Sf / Cs) ** 2) * (Cs / torch.clamp(Cs - 1.0, min=1.0))
        return torch.where(ok, torch.clamp(v, min=0.0), torch.zeros_like(v)), ok

    def profile(self, bandwidth, smooth_sigma):
        v, ok = self.bin_variance()
        w = torch.where(ok, self.C, torch.zeros_like(self.C))
        K = self._K.get(float(bandwidth))
        if K is None:
            K = gaussian_kernel_torch(self.grid[:, None] - self.grid[None, :], bandwidth)   # (G, G), cached
            self._K[float(bandwidth)] = K
        num = K @ (w * v)
        den = K @ w
        vz = torch.where(den > EPS, num / torch.clamp(den, min=EPS), torch.zeros_like(num))
        return torch.clamp(smooth_profile_torch(vz, smooth_sigma), min=0.0)

    def state(self):
        return dict(C=to_numpy(self.C), Sf=to_numpy(self.Sf), Sf2=to_numpy(self.Sf2))


def water_filling_durations(a, tau, B_time, n_iter=48):
    """``t_i = tau_i/2 [log(2 a_i / (lambda tau_i))]_+`` with ``sum_i t_i = B_time`` (bisection on
    ``log lambda``; the total is monotone in it).  Replicas with ``a_i <= 0`` get 0; a zero budget
    or no positive importance gives all zeros.  ``a, tau`` (N,), ``B_time`` float.  Returns (N,)."""
    valid = a > 0
    if B_time <= 0.0 or not bool(valid.any()):
        return torch.zeros_like(a)
    ninf = torch.full_like(a, -float("inf"))
    logr = torch.where(valid, torch.log(torch.clamp(2.0 * a, min=EPS) / tau), ninf)
    hi = logr.max()
    tau_min = torch.where(valid, tau, torch.full_like(tau, float("inf"))).min()
    lo = torch.where(valid, logr, torch.full_like(a, float("inf"))).min() - 2.0 * B_time / tau_min - 1.0
    for _ in range(n_iter):                      # sync-free bisection: lo/hi stay on the device
        mid = 0.5 * (lo + hi)
        cost = (0.5 * torch.clamp(logr - mid, min=0.0) * tau).sum()
        over = cost > B_time
        lo = torch.where(over, mid, lo)
        hi = torch.where(over, hi, mid)
    c = 0.5 * torch.clamp(logr - 0.5 * (lo + hi), min=0.0)
    return torch.where(valid, c * tau, torch.zeros_like(a))


def integer_steps_largest_remainder(t, dt, budget_steps):
    """Integer inner steps ``m_i`` from durations: if ``sum t_i/dt`` exceeds the integer budget the
    durations are rescaled to it first; then floor, then +1 to the largest remainders so that
    ``sum m_i = min(budget_steps, round(sum t_i / dt))``.  Deterministic; never exceeds the budget;
    one host sync."""
    x = t / dt
    total = float(x.sum().item())
    budget = int(min(int(budget_steps), int(round(total))))
    if budget <= 0 or total <= 0:
        return torch.zeros_like(t, dtype=torch.long)
    if total > budget:
        x = x * (budget / total)
    m = torch.floor(x).long()
    left = budget - int(m.sum().item())
    if left > 0:
        order = torch.argsort(x - m.to(x.dtype), descending=True)
        m[order[:left]] += 1
    return torch.clamp(m, min=0)


@torch.inference_mode()
def frozen_dimer_relax(engine, params, sim, q, m_steps, gen, scheme="frozen"):
    """Inner relaxation with the dimer FROZEN: particles 0 and 1 do not move; every solvent
    particle evolves under the full physical potential (including the fixed dimer's forces),
    temperature, time step and periodic box of the outer dynamics.  Replica ``i`` is advanced for
    exactly ``m_steps[i]`` steps: the batch is sorted by ``m`` and shrinks as replicas finish, so
    the force is never evaluated for a replica that is not being advanced.  Nothing is deposited
    anywhere; noise comes from ``gen`` (never the global stream).  Returns (q_new, replica_steps)."""
    B = q.shape[0]
    if B == 0:
        return q, 0
    order = torch.argsort(m_steps, descending=True)
    qs = q.index_select(0, order).clone()
    ms = m_steps.index_select(0, order)
    z_fixed = reaction_coordinate(qs, params) if scheme == "projected" else None
    noise_scale = math.sqrt(2.0 * sim.dt / params.beta)
    kmax = int(ms.max().item()) if ms.numel() else 0
    counts = torch.bincount(ms.clamp(min=0), minlength=kmax + 1)
    # n_active(k) = number of replicas with m > k = B - (number with m <= k); one host sync
    cum_le = torch.cumsum(counts, 0).tolist()
    done = 0
    for k in range(kmax):
        n_act = B - int(cum_le[k])
        if n_act <= 0:
            break
        qa = qs[:n_act]
        f = clip_forces(engine.force(qa, compute_energy=False), params.force_clip)
        noise = torch.randn(qa.shape, device=qa.device, dtype=qa.dtype, generator=gen)
        qa_new = qa + sim.dt * f + noise_scale * noise
        if scheme == "projected":
            # the TI scheme: all particles move, then the dimer is re-projected to its z
            qs[:n_act] = project_dimer_to_z(wrap_positions(qa_new, params.box_length), z_fixed[:n_act], params)
        else:
            qa_new[:, :2, :] = qa[:, :2, :]                   # dimer frozen exactly
            qs[:n_act] = wrap_positions(qa_new, params.box_length)
        done += n_act
    q_new = torch.empty_like(q)
    q_new[order] = qs
    return q_new, done


@torch.inference_mode()
def constrained_force_series(engine, params, sim, q0, z_k, n_eq, n_prod, gen, record_every=1):
    """W0-B instrument: project each replica's dimer to ``z_k``, run the frozen-dimer solvent
    dynamics for ``n_eq + n_prod`` steps and record the (estimator-clipped) local mean force
    every ``record_every`` production steps.  Returns ``f`` (B, n_rec) as float64 numpy."""
    q = project_dimer_to_z(q0, torch.as_tensor(z_k, device=q0.device, dtype=q0.dtype), params)
    noise_scale = math.sqrt(2.0 * sim.dt / params.beta)
    rec = []
    for step in range(n_eq + n_prod):
        f_raw = engine.force(q, compute_energy=False)
        f_phys = clip_forces(f_raw, params.force_clip)
        if step >= n_eq and (step - n_eq) % record_every == 0:
            fl = local_mean_force(q, f_phys if sim.use_clipped_force_for_mean_force else f_raw, params)
            rec.append(torch.clamp(fl, -sim.mean_force_sample_clip, sim.mean_force_sample_clip).to(torch.float64))
        noise = torch.randn(q.shape, device=q.device, dtype=q.dtype, generator=gen)
        q_new = q + sim.dt * f_phys + noise_scale * noise
        q_new[:, :2, :] = q[:, :2, :]
        q = wrap_positions(q_new, params.box_length)
    return to_numpy(torch.stack(rec, dim=1))


def autocorrelation_time(f, dt, max_lag=None):
    """Integrated force-correlation time from series ``f`` (B, T): replica-averaged normalised
    autocorrelation ``rho(k)``, integrated over the initial positive sequence,
    ``tau = dt (1/2 + sum_{k>=1}^{K} rho(k))`` with ``K`` the last lag before ``rho`` first drops
    to <= 0.  Returns (tau, rho) with ``rho`` the curve up to ``max_lag``."""
    f = np.asarray(f, dtype=np.float64)
    B, T = f.shape
    d = f - f.mean(axis=1, keepdims=True)
    L = int(max_lag or T // 2)
    n = 1
    while n < 2 * T:
        n *= 2
    Fd = np.fft.rfft(d, n=n, axis=1)
    ac = np.fft.irfft(Fd * np.conj(Fd), n=n, axis=1)[:, :L] / (T - np.arange(L))[None, :]
    c = ac.mean(axis=0)
    if c[0] <= 0:
        return float("nan"), np.zeros(L)
    rho = c / c[0]
    neg = np.nonzero(rho <= 0.0)[0]
    K = int(neg[0]) if neg.size else L
    tau = dt * (0.5 + rho[1:K].sum())
    return float(tau), rho


# ---------------------------------------------------------------------------
# Fisher-Rao target densities (estimated / uniform / oracle) and score
# ---------------------------------------------------------------------------
def recentered_clipped_score_torch(raw_score, score_clip):
    """Mean-zero, clipped score. Idempotent: applying twice == applying once.

    Ends on a recenter (not a bare clamp) so a second application is a guaranteed
    no-op; the score is normalized once in fr_score_torch and the birth-death step
    is defensive without shifting the death/birth partition near S_i=0.
    """
    score = raw_score - raw_score.mean()
    for _ in range(3):
        score = torch.clamp(score, -score_clip, score_clip)
        score = score - score.mean()
    return score


def fr_target_estimated_torch(grid, F_target_ema, current_bias_profile, beta, eps=EPS):
    """ESTIMATED-target marginal q_n(z) ~ exp[-beta (F_target_ema(z) - B_n(z))].

    F_target_ema is the ONLINE EMA of the unscaled ABF free energy A_hat_n, and
    B_n = current_bias_profile = abf_scale * A_hat_n. NO TI / oracle input.
    """
    if F_target_ema is None:
        raise ValueError("fr_estimated requires F_target_ema (init at fr_start_steps).")
    if current_bias_profile is None:
        raise ValueError("fr_estimated requires the current ABF bias B_n.")
    log_q = -beta * (F_target_ema - current_bias_profile)
    log_q = log_q - torch.max(log_q)
    return normalize_density_on_grid_torch(torch.exp(log_q), grid, eps=eps)


def fr_target_uniform_torch(grid, eps=EPS):
    """Naive ablation: uniform target on the z-grid."""
    return normalize_density_on_grid_torch(torch.ones_like(grid), grid, eps=eps)


def fr_target_oracle_torch(grid, oracle_free_energy, current_bias_profile, beta, eps=EPS):
    """DIAGNOSTIC oracle target q_n(z) ~ exp[-beta (F_ref(z) - B_n(z))].

    Uses the (unknown in practice) TI reference free energy. Positive control only.
    """
    if oracle_free_energy is None:
        raise ValueError("fr_oracle requires oracle_free_energy (the TI reference).")
    if current_bias_profile is None:
        raise ValueError("fr_oracle requires the current ABF bias B_n.")
    log_q = -beta * (oracle_free_energy - current_bias_profile)
    log_q = log_q - torch.max(log_q)
    return normalize_density_on_grid_torch(torch.exp(log_q), grid, eps=eps)


def fr_score_torch(z_samples, grid, sim, q_grid, eps=EPS):
    """Log-density-ratio FR score against a prebuilt target q_n.

    S_i = log p_hat(Z_i) - log q_n(Z_i) - KL(p_hat || q_n), then clip + recenter.
    Returns (score, p_grid, q_grid, kl_pq).
    """
    p_grid = normalize_density_on_grid_torch(
        kde_1d_torch(grid, z_samples, sim.kde_bandwidth, sim.z_min, sim.z_max), grid, eps=eps
    )
    p_at = interp_uniform_grid_edge(p_grid, grid, z_samples)
    q_at = interp_uniform_grid_edge(q_grid, grid, z_samples)
    log_ratio_grid = torch.log(torch.clamp(p_grid, min=eps)) - torch.log(torch.clamp(q_grid, min=eps))
    kl_pq = trapz_torch(p_grid * log_ratio_grid, grid)
    raw_score = (
        torch.log(torch.clamp(p_at, min=eps))
        - torch.log(torch.clamp(q_at, min=eps))
        - kl_pq
    )
    score = recentered_clipped_score_torch(raw_score, sim.score_clip)
    return score, p_grid, q_grid, kl_pq


def prior_selection_score_torch(method, z_samples, grid, sim, c=1.0, eps=EPS):
    """Chapter-6 Laplacian selection and count balancing, on this project's sign convention.

    **Sign.** `fixed_population_birth_death_torch` kills positive scores. Chapter 6 writes its
    selection function `S` so that POSITIVE means MULTIPLY, so both rules are negated here.
    Getting this backwards would run to completion and mean the opposite of the claim.

      book_laplacian   S_book  = c * d2p/dz2 / p        -> score = -S_book
                       multiplies where p is CONVEX (density valleys), which is what shifts the
                       marginal diffusion from beta^-1 to beta^-1 + c
      count_balancing  S_count = c * (1 - p/p_bar)      -> score = -S_count = c*(p/p_bar - 1)
                       kills the over-represented; saturating, unlike the FR log-ratio

    The second derivative is taken on the KDE-smoothed density: a second derivative of a raw
    histogram is dominated by counting noise, which would silently turn this baseline into a
    second sham.
    """
    p_grid = normalize_density_on_grid_torch(
        kde_1d_torch(grid, z_samples, sim.kde_bandwidth, sim.z_min, sim.z_max), grid, eps=eps
    )
    if method == "count_balancing":
        p_bar = p_grid.mean()
        raw_grid = c * (p_grid / torch.clamp(p_bar, min=eps) - 1.0)
    elif method == "book_laplacian":
        dz = float(grid[1] - grid[0])
        d2 = torch.zeros_like(p_grid)
        d2[1:-1] = (p_grid[2:] - 2.0 * p_grid[1:-1] + p_grid[:-2]) / (dz * dz)
        d2[0], d2[-1] = d2[1], d2[-2]
        raw_grid = -c * d2 / torch.clamp(p_grid, min=eps)
    else:
        raise ValueError(method)
    raw = interp_uniform_grid_edge(raw_grid, grid, z_samples)
    return recentered_clipped_score_torch(raw, sim.score_clip), p_grid


def fixed_population_birth_death_torch(q, score, sim, ancestors=None, fr_interval=None,
                                       fr_rate_override=None):
    """Fixed-population birth-death with a max_event_fraction safeguard.

    Over-represented replicas (positive score) die; each is replaced by a clone of
    an under-represented replica (birth weights ~ max(-S_i, 0)). Deaths are capped
    at max_event_fraction * n_replicas (random subselection). The whole replica
    configuration is copied. If ``ancestors`` (long tensor, one label per replica)
    is given it is updated in place-style and returned so lineage can be tracked.
    Returns (q_new, ancestors_new, stats) where stats holds replacement count and
    the z-free birth/death replica indices.
    """
    R = q.shape[0]
    # score is already mean-zero+clipped by fr_score_torch; recenter is now
    # idempotent so this is a defensive no-op (safe for any ad-hoc caller).
    score = recentered_clipped_score_torch(score, sim.score_clip)
    interval = sim.fr_every if fr_interval is None else fr_interval
    dt_eff = sim.dt * max(int(interval), 1)
    fr_rate = float(sim.fr_rate if fr_rate_override is None else fr_rate_override)

    empty = torch.empty(0, dtype=torch.long, device=q.device)
    # max_event_fraction <= 0 (or so small that floor(frac*R) == 0) disables all
    # events, matching the convention "fraction -> 0 ⇒ zero replacements".
    max_events = int(sim.max_event_fraction * R)
    no_event = {"replacement": 0, "death_idx": empty, "birth_src": empty}
    if max_events < 1 or fr_rate <= 0.0:
        # max_event_fraction -> 0 OR an adaptive rate gated to zero disables events.
        return q.clone(), (None if ancestors is None else ancestors.clone()), no_event

    death_weights = torch.clamp(score, min=0.0)
    birth_weights = torch.clamp(-score, min=0.0)
    death_mass = torch.sum(death_weights)
    birth_mass = torch.sum(birth_weights)
    if bool(((death_mass <= EPS) | (birth_mass <= EPS)).detach().cpu().item()):
        return q.clone(), (None if ancestors is None else ancestors.clone()), no_event

    death_prob = torch.where(
        death_weights > 0.0,
        1.0 - torch.exp(-fr_rate * death_weights * dt_eff),
        torch.zeros_like(death_weights),
    )
    death_indices = torch.nonzero(
        torch.rand(R, device=q.device, dtype=q.dtype) < death_prob, as_tuple=False
    ).flatten()
    n_events = int(death_indices.numel())
    if n_events == 0:
        return q.clone(), (None if ancestors is None else ancestors.clone()), no_event

    if n_events > max_events:
        perm = torch.randperm(n_events, device=q.device)[:max_events]
        death_indices = death_indices[perm]
        n_events = max_events

    clone_sources = torch.multinomial(birth_weights, n_events, replacement=True)
    q_new = q.clone()
    q_new[death_indices] = q.index_select(0, clone_sources)
    ancestors_new = None
    if ancestors is not None:
        ancestors_new = ancestors.clone()
        ancestors_new[death_indices] = ancestors.index_select(0, clone_sources)
    return q_new, ancestors_new, {
        "replacement": int(n_events), "death_idx": death_indices, "birth_src": clone_sources}


def uniform_birth_death_torch(q, n_events, ancestors=None, generator=None):
    """Kill ``n_events`` replicas uniformly at random and refill from survivors.

    The matched-sham replacement step.  It is handed a count -- the count its partner FR arm
    actually realised at this event -- and performs the same number of replacements at the
    same time, choosing *which* replicas die and which are copied uniformly at random.

    Everything that distinguishes it from ``fixed_population_birth_death_torch`` is the
    selection: no score, no death weights, no birth weights.  The population size, the copy
    semantics (whole replica configuration), the lineage bookkeeping and the event count are
    identical, so a difference in outcome is attributable to the Fisher-Rao *direction*.

    Clone sources are drawn from the survivors only, matching the FR arm, where a replica
    that dies has zero birth weight and so cannot be its own source.
    """
    R = q.shape[0]
    empty = torch.empty(0, dtype=torch.long, device=q.device)
    n_events = int(n_events)
    if n_events < 1:
        return (q.clone(), None if ancestors is None else ancestors.clone(),
                {"replacement": 0, "death_idx": empty, "birth_src": empty})
    n_events = min(n_events, R - 1)          # at least one survivor must remain
    perm = torch.randperm(R, device=q.device, generator=generator)
    death_indices = perm[:n_events]
    survivors = perm[n_events:]
    pick = torch.randint(survivors.numel(), (n_events,), device=q.device,
                         generator=generator)
    clone_sources = survivors[pick]
    q_new = q.clone()
    q_new[death_indices] = q.index_select(0, clone_sources)
    ancestors_new = None
    if ancestors is not None:
        ancestors_new = ancestors.clone()
        ancestors_new[death_indices] = ancestors.index_select(0, clone_sources)
    return q_new, ancestors_new, {"replacement": int(n_events),
                                  "death_idx": death_indices,
                                  "birth_src": clone_sources}


# ---------------------------------------------------------------------------
# Adaptive diversity-aware FR schedule (deployable; ONLINE quantities only).
# ---------------------------------------------------------------------------
def _clip01(x):
    return float(min(1.0, max(0.0, x)))


def support_signal(sim, p_grid, q_grid, grid):
    """The raw (un-smoothed) online support signal for the adaptive support gate.

    Depends on ``sim.adaptive_support_mode``:
      marginal_uniform  : L2(p_hat, uniform) on the eval window (ABF-driven,
                          exogenous to FR; the recommended default).
      marginal_mismatch : L2(p_hat, q_target) (reduced by FR -> noisier gate).
      marginal_tv       : total variation TV(p_hat, uniform) on the eval window.
      neff_transition   : handled in adaptive_fr_rate (uses the ABF denominator).
    Returns a float, or NaN if not applicable for the mode.
    """
    g = to_numpy(grid)
    emask = (g >= sim.eval_z_lo) & (g <= sim.eval_z_hi)
    ge = g[emask]
    if not emask.any() or p_grid is None:
        return float("nan")
    p = np.clip(to_numpy(p_grid).astype(float)[emask], 0.0, None)
    zc = np.trapezoid(p, ge)
    if not (np.isfinite(zc) and zc > 0):
        return float("nan")
    p = p / zc
    mode = sim.adaptive_support_mode
    if mode == "marginal_mismatch" and q_grid is not None:
        q = np.clip(to_numpy(q_grid).astype(float)[emask], 0.0, None)
        zq = np.trapezoid(q, ge)
        if not (np.isfinite(zq) and zq > 0):
            return float("nan")
        q = q / zq
        return float(np.sqrt(np.trapezoid((p - q) ** 2, ge) / (ge[-1] - ge[0])))
    uni = np.ones_like(p)
    uni = uni / np.trapezoid(uni, ge)
    if mode == "marginal_tv":
        return 0.5 * float(np.trapezoid(np.abs(p - uni), ge))
    # marginal_uniform (default)
    return float(np.sqrt(np.trapezoid((p - uni) ** 2, ge) / (ge[-1] - ge[0])))


def adaptive_fr_rate(sim, eff_counts, grid, ess_frac, last_event_fraction,
                     support_ema=None, p_grid=None):
    """Effective FR rate for the adaptive method at one FR event.

    rate_eff = base_fr_rate * support_gate * diversity_gate * event_gate, clipped
    to [adaptive_min_rate, adaptive_max_rate].  Uses ONLY online quantities (no TI):

      support_gate   : for the marginal_* modes, a ramp on the EMA-smoothed support
        signal (see :func:`support_signal`), clip((support_ema - lo)/(hi - lo),0,1);
        for neff_transition, the kernel-denominator deficit vs a relative-median (or
        absolute) target.
      diversity_gate : tapered in ancestor ESS fraction; 1 above ess_full, 0 below
        ess_stop, linear between -- self-limits when genealogy collapses.
      event_gate     : backoff if the previous realised event fraction is near the
        max_event_fraction cap.

    Returns (rate_eff, gates).
    """
    g = to_numpy(grid)
    tmask = (g >= sim.transition_lo) & (g <= sim.transition_hi)
    emask = (g >= sim.eval_z_lo) & (g <= sim.eval_z_hi)
    ec = to_numpy(eff_counts).astype(float)
    neff_tr = float(np.nanmean(ec[tmask])) if tmask.any() else float("nan")
    neff_med = float(np.nanmedian(ec[emask])) if emask.any() else float("nan")

    mode = sim.adaptive_support_mode
    if mode in ("marginal_uniform", "marginal_mismatch", "marginal_tv"):
        lo, hi = sim.adaptive_support_lo, sim.adaptive_support_hi
        if support_ema is not None and np.isfinite(support_ema):
            support_gate = _clip01((support_ema - lo) / max(hi - lo, EPS))
        else:
            support_gate = 0.0
        support_deficit = support_gate
    else:  # neff_transition
        if sim.adaptive_neff_target_strategy == "absolute":
            neff_target = float(sim.adaptive_neff_target_abs)
        else:
            neff_target = float(sim.adaptive_neff_target_frac) * neff_med
        if np.isfinite(neff_target) and neff_target > 0 and np.isfinite(neff_tr):
            support_deficit = _clip01((neff_target - neff_tr) / max(neff_target, EPS))
        else:
            support_deficit = 0.0
        support_gate = support_deficit

    # diversity gate (taper in ESS fraction)
    full, stop = sim.adaptive_ess_full_threshold, sim.adaptive_ess_stop_threshold
    if not np.isfinite(ess_frac):
        diversity_gate = 1.0
    elif ess_frac >= full:
        diversity_gate = 1.0
    elif ess_frac <= stop:
        diversity_gate = 0.0
    else:
        diversity_gate = _clip01((ess_frac - stop) / max(full - stop, EPS))

    # event backoff gate
    cap = sim.max_event_fraction
    if (np.isfinite(last_event_fraction)
            and last_event_fraction > sim.adaptive_event_backoff_threshold * cap):
        event_gate = float(sim.adaptive_event_backoff_factor)
    else:
        event_gate = 1.0

    rate = sim.adaptive_base_fr_rate * support_gate * diversity_gate * event_gate
    rate_eff = float(min(sim.adaptive_max_rate, max(sim.adaptive_min_rate, rate)))
    gates = dict(rate_eff=rate_eff, support_gate=float(support_gate),
                 diversity_gate=float(diversity_gate), event_gate=float(event_gate),
                 neff_transition=neff_tr, neff_median=neff_med,
                 support_ema=(float(support_ema) if support_ema is not None
                              and np.isfinite(support_ema) else float("nan")),
                 support_deficit=float(support_deficit), ess_frac=float(ess_frac),
                 last_event_fraction=float(last_event_fraction))
    return rate_eff, gates


def score_statistics(score, score_clip):
    """Mean-zero score summary for logging: std, abs-max, and clip fraction."""
    s = to_numpy(score).astype(float)
    if s.size == 0:
        return dict(score_std=float("nan"), score_absmax=float("nan"),
                    score_clip_fraction=float("nan"))
    clip_frac = float(np.mean(np.abs(s) >= (float(score_clip) - 1e-6)))
    return dict(score_std=float(np.std(s)), score_absmax=float(np.max(np.abs(s))),
                score_clip_fraction=clip_frac)


# ---------------------------------------------------------------------------
# Diagnostics helpers
# ---------------------------------------------------------------------------
def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def profile_l2_error_np(profile, reference, grid, mask=None):
    p = np.asarray(profile, dtype=float)
    r = np.asarray(reference, dtype=float)
    g = np.asarray(grid, dtype=float)
    if mask is not None:
        p, r, g = p[mask], r[mask], g[mask]
    return math.sqrt(np.trapezoid((p - r) ** 2, g) / (g[-1] - g[0]))


def align_additive_constant_np(profile, reference, grid, mask=None):
    profile = np.asarray(profile, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if mask is None:
        return profile - np.mean(profile - reference)
    return profile - np.mean(profile[mask] - reference[mask])


def region_masks_np(grid, sim):
    g = np.asarray(grid, dtype=float)
    compact = g < sim.transition_lo
    transition = (g >= sim.transition_lo) & (g <= sim.transition_hi)
    stretched = g > sim.transition_hi
    return {"compact": compact, "transition": transition, "stretched": stretched}


def eval_window_mask_np(grid, sim):
    g = np.asarray(grid, dtype=float)
    return (g >= sim.eval_z_lo) & (g <= sim.eval_z_hi)


def region_fractions_torch(z, sim):
    """Fraction of replicas in compact/transition/stretched regions."""
    total = float(z.numel())
    compact = float((z < sim.transition_lo).sum().item()) / total
    transition = float(((z >= sim.transition_lo) & (z <= sim.transition_hi)).sum().item()) / total
    stretched = float((z > sim.transition_hi).sum().item()) / total
    return compact, transition, stretched


def ancestor_ess(ancestors, n_replicas):
    """ESS_ancestor = 1 / sum_a w_a^2 where w_a is the fraction descended from a."""
    if ancestors is None:
        return float("nan"), 0
    counts = torch.bincount(ancestors, minlength=n_replicas).to(torch.float64)
    w = counts / counts.sum().clamp_min(1.0)
    ess = 1.0 / torch.clamp((w * w).sum(), min=EPS)
    n_unique = int((counts > 0).sum().item())
    return float(ess.item()), n_unique


def ancestor_max_fraction(ancestors, n_replicas):
    """Largest fraction of the current population descended from a single ancestor.

    Read-only genealogy diagnostic (no RNG consumed): a high value flags
    genealogical collapse even when ESS or unique-ancestor count look healthy.
    """
    if ancestors is None:
        return float("nan")
    counts = torch.bincount(ancestors, minlength=n_replicas).to(torch.float64)
    return float((counts.max() / counts.sum().clamp_min(1.0)).item())


def region_l2_errors(profile, reference, grid, sim):
    """L2 of (profile-reference) over compact/transition/stretched regions.

    Each region mask is intersected with the interior evaluation window so all
    three regions share the SAME interior domain as the headline l2_f / l2_fp
    (otherwise compact/stretched would include wall-affected edge bins that the
    headline metric deliberately excludes). profile/reference are arrays on `grid`.
    """
    masks = region_masks_np(grid, sim)
    eval_mask = eval_window_mask_np(grid, sim)
    out = {}
    for name, m in masks.items():
        m = m & eval_mask
        if m.sum() < 2:
            out[name] = float("nan")
            continue
        out[name] = profile_l2_error_np(profile, reference, grid, mask=m)
    return out


# ---------------------------------------------------------------------------
# No-oracle-leakage audit
# ---------------------------------------------------------------------------
def assert_no_oracle_leakage(method, oracle_free_energy):
    """Hard guard: only fr_oracle may receive an oracle/TI reference target.

    fr_estimated (the practical method), abf, and fr_uniform MUST NOT receive it.
    """
    if method == "fr_oracle":
        if oracle_free_energy is None:
            raise ValueError("fr_oracle requires oracle_free_energy (the TI reference).")
        return
    if oracle_free_energy is not None:
        raise AssertionError(
            f"NO-ORACLE-LEAKAGE VIOLATION: method='{method}' received oracle_free_energy. "
            "Only 'fr_oracle' may use the TI reference as a target.")


# ---------------------------------------------------------------------------
# Main sampler. abf / fr_estimated / fr_uniform / fr_oracle.
# ---------------------------------------------------------------------------
@torch.inference_mode()
def run_sampler_gpu(method, params, sim, engine, initial_q=None,
                    oracle_free_energy=None, collect_diagnostics=True, verbose=True,
                    track_crossings=False, replay_counts=None, readout_bandwidths=None,
                    relax=None, sensitivity_record=False, ot=None):
    """Run one sampler on the GPU.

    ``readout_bandwidths`` (default None: byte-identical to before) attaches a
    :class:`ReadoutBank` -- report-only estimators at those bandwidths (``0`` = raw
    binned sums) recorded at every save as ``diag["readout_mean_force"][h]`` and
    ``diag["raw_fsum"]``/``diag["raw_csum"]``.  Never touches the bias or the RNG.

    method in {abf, fr_estimated, fr_uniform, fr_oracle}. The FR target is built
    per-variant; only fr_oracle reads oracle_free_energy (the TI reference). The
    EMA estimated target is maintained ONLY for fr_estimated and never sees TI.

    ``track_crossings`` (default False, so existing callers are byte-identical and
    pay no overhead) enables a read-only per-replica barrier-crossing counter at
    the transition midpoint z_b=(transition_lo+transition_hi)/2. It counts
    compact->stretched and stretched->compact crossings and per-replica round
    trips; on a birth-death clone only the side label is inherited from the clone
    source, so a teleporting clone is not miscounted as a physical crossing. The
    per-slot crossing tallies are not duplicated. No RNG is consumed, so dynamics
    are unchanged.
    """
    if method not in ALL_METHODS:
        raise ValueError(f"method must be one of {ALL_METHODS}, got {method!r}")
    assert_no_oracle_leakage(method, oracle_free_energy)
    is_sham = method in SHAM_METHODS
    is_prior = method in PRIOR_SELECTION_METHODS
    is_fr = (method in FR_METHODS) or is_sham or is_prior
    if is_sham:
        # A sham with no schedule to replay would silently become plain ABF and still print
        # a verdict, so this is an error rather than a default.
        if replay_counts is None:
            raise ValueError(
                f"{method!r} requires replay_counts: the per-event replacement counts "
                f"realised by {SHAM_PARTNER[method]!r} on the same seed. Without them the "
                f"arm is not intensity-matched and is not the control it claims to be.")
        replay_counts = [int(x) for x in np.asarray(replay_counts).ravel().tolist()]
    elif replay_counts is not None:
        raise ValueError(f"replay_counts is only meaningful for {SHAM_METHODS}")
    fr_event_counts = []          # per FR opportunity; what a sham later replays
    sham_event_idx = 0

    torch.manual_seed(sim.seed)
    q = (lattice_initial_conditions(params, sim.n_replicas, engine.device, engine.dtype, seed=sim.seed)
         if initial_q is None else initial_q.clone())
    grid = torch.linspace(sim.z_min, sim.z_max, sim.n_grid, device=engine.device, dtype=engine.dtype)
    oracle_t = None
    if method == "fr_oracle":
        oracle_t = torch.as_tensor(oracle_free_energy, device=engine.device, dtype=engine.dtype)
        oracle_t = normalize_profile_zero_at_midpoint_torch(oracle_t, grid)

    bias_estimator = TorchKernelABFEstimator(grid, sim.abf_bandwidth, sim.abf_smooth_sigma, edge_extrapolate=sim.abf_edge_extrapolate)
    production_estimator = TorchKernelABFEstimator(grid, sim.abf_bandwidth, sim.abf_smooth_sigma, edge_extrapolate=sim.abf_edge_extrapolate)
    readout = ReadoutBank(grid, sim, readout_bandwidths) if readout_bandwidths else None
    # Targeted relaxation / sensitivity instrumentation (additive; nothing below runs otherwise)
    sens = SensitivityAccumulator(grid) if (relax is not None or sensitivity_record) else None
    if relax is not None:
        assert relax.target in ("sensitivity", "random"), relax.target
        assert relax.scheme in ("frozen", "projected"), relax.scheme
        assert 0.0 <= float(relax.rho) < float("inf")
        tau_grid_t = torch.as_tensor(np.asarray(relax.tau_grid, dtype=np.float64), device=engine.device,
                                     dtype=engine.dtype)
        assert tau_grid_t.numel() == sim.n_grid and bool((tau_grid_t > 0).all()), "tau map must be positive on the grid"
        gen_relax = torch.Generator(device=engine.device)
        gen_relax.manual_seed(int(sim.seed) + int(relax.inner_seed_offset))
        relax_budget_steps = int(round(float(relax.rho) * sim.n_replicas * max(int(sim.fr_every), 1)))
        relax_budget_time = relax_budget_steps * sim.dt
        relax_steps_total, relax_events, relax_inner_wall = 0, 0, 0.0
        relax_hist = torch.zeros(sim.n_grid, device=engine.device, dtype=torch.float64)
        _rx_steps_acc, _rx_active_acc, _rx_n_acc = 0, 0.0, 0
    if ot is not None:
        # Wasserstein reallocation + constrained fibre repair (wca_ot_repair).  ADDITIVE: absent
        # unless an OTConfig is handed in.  OT consumes no RNG; repair has its own generator.
        from wca_ot_repair import ABSDZ_EDGES, OTConfig, ot_displacement, uniform_quantiles
        assert isinstance(ot, OTConfig)
        assert ot.scheme in ("frozen", "projected"), ot.scheme
        assert float(ot.alpha) >= 0.0 and float(ot.dz_max) > 0.0 and float(ot.c_repair) >= 0.0
        ot_tau_t = torch.as_tensor(np.asarray(ot.tau_grid, dtype=np.float64), device=engine.device, dtype=engine.dtype)
        assert ot_tau_t.numel() == sim.n_grid and bool((ot_tau_t > 0).all()), "tau map must be positive on the grid"
        gen_ot = torch.Generator(device=engine.device)
        gen_ot.manual_seed(int(sim.seed) + int(ot.inner_seed_offset))
        u_ot = uniform_quantiles(sim.n_replicas, sim.z_min, sim.z_max, engine.device, engine.dtype)
        ot_steps_total, ot_events, ot_inner_wall, ot_diag_evals, _ot_steps_acc = 0, 0, 0.0, 0, 0
        ot_absdz_sum = torch.zeros((), device=engine.device, dtype=torch.float64)
        ot_absdz_max = torch.zeros((), device=engine.device, dtype=torch.float64)
        ot_capped = torch.zeros((), device=engine.device, dtype=torch.float64)
        ot_moved = torch.zeros((), device=engine.device, dtype=torch.float64)
        ot_C_pre = torch.zeros(sim.n_grid, device=engine.device, dtype=torch.float64)
        ot_Sf_pre = torch.zeros_like(ot_C_pre)
        ot_C_post = torch.zeros_like(ot_C_pre)
        ot_Sf_post = torch.zeros_like(ot_C_pre)
        ot_pending = None        # walkers moved at the last opportunity: their next deposit is the 'post' sample
        # M3 additions: per-save |dz| / cap series, and the deposit-after-event table binned by (z, |dz|)
        ot_dz_edges = torch.tensor(ABSDZ_EDGES[1:-1], device=engine.device, dtype=engine.dtype)   # bucketize on the finite interior edges
        n_dzb = len(ABSDZ_EDGES) - 1
        ot_C2_post = torch.zeros(sim.n_grid * n_dzb, device=engine.device, dtype=torch.float64)
        ot_Sf2_post = torch.zeros_like(ot_C2_post)
        ot_pending_dzb = None
        _ot_absdz_acc = torch.zeros((), device=engine.device, dtype=torch.float64)
        _ot_capped_acc = torch.zeros((), device=engine.device, dtype=torch.float64)
        _ot_events_acc = 0
    noise_scale = math.sqrt(2.0 * sim.dt / params.beta)
    total_replacement_events = 0
    F_target_ema = None
    ancestors = (torch.arange(sim.n_replicas, device=engine.device, dtype=torch.long) if is_fr else None)
    # Windowed ancestry: same labels, but reset every `ess_window_steps` so the ESS measures
    # coalescence over a FIXED horizon instead of over the whole run. Updated from the death /
    # birth indices the birth-death routines already return, so it consumes no RNG and cannot
    # perturb the dynamics -- purely a read-off. `min_ancestor_ess_window` is the quantity the
    # §3.3 gate `ESS_anc/N >= 0.30` was calibrated against.
    anc_win = (ancestors.clone() if is_fr else None)
    min_ess_window = float(sim.n_replicas)
    _win = max(int(getattr(sim, "ess_window_steps", 4000)), 1)

    def _track_window(stats, step_now):
        """Apply the realised replacement permutation to the windowed labels, then reset."""
        nonlocal anc_win, min_ess_window
        if anc_win is None:
            return
        di, bs = stats.get("death_idx"), stats.get("birth_src")
        if di is not None and int(di.numel()) > 0:
            anc_win[di] = anc_win.index_select(0, bs)
            e, _ = ancestor_ess(anc_win, sim.n_replicas)
            min_ess_window = min(min_ess_window, e)
        if step_now % _win == 0:
            anc_win = torch.arange(sim.n_replicas, device=engine.device, dtype=torch.long)

    # birth/death z-location histograms accumulated over the run
    birth_hist = np.zeros(sim.n_grid, dtype=np.float64)
    death_hist = np.zeros(sim.n_grid, dtype=np.float64)

    # Adaptive-FR state + per-event log (only populated for fr_estimated_adaptive).
    is_adaptive = method == "fr_estimated_adaptive"
    last_event_fraction = 0.0
    support_ema = None           # EMA of the adaptive support signal over FR events
    # Cumulative z-marginal (low-variance) for the adaptive marginal_uniform gate:
    # broadly Gaussian-smoothed (kde bandwidth) so cold-system sharp peaks are washed
    # out and only the broad transition-region under-sampling (starvation) survives.
    marg_hist = (torch.zeros(sim.n_grid, device=engine.device, dtype=engine.dtype)
                 if is_adaptive else None)
    _dz = (sim.z_max - sim.z_min) / max(sim.n_grid - 1, 1)
    _kde_sigma_pts = sim.kde_bandwidth / max(_dz, EPS)
    adaptive_log = {k: [] for k in ["step", "fr_rate_eff", "support_gate",
                                    "diversity_gate", "event_gate", "neff_transition",
                                    "support_ema", "ess_frac", "event_fraction",
                                    "score_std", "score_absmax", "score_clip_fraction"]}
    # Aggregate score statistics over all FR events (any FR method).
    _score_std_sum = _score_absmax_max = _score_clipfrac_sum = 0.0
    _n_score_events = 0

    # Optional read-only barrier-crossing tracker (no effect on dynamics/RNG).
    # Counters are kept on-GPU and summed once at the end to avoid a per-step
    # host sync (the .item() that would otherwise serialise every step).
    z_barrier = 0.5 * (sim.transition_lo + sim.transition_hi)
    side = None                 # bool: True == stretched side (z > z_barrier)
    n_c2s_total = (torch.zeros((), device=engine.device, dtype=torch.long)
                   if track_crossings else 0)   # compact -> stretched (population sum)
    n_s2c_total = (torch.zeros((), device=engine.device, dtype=torch.long)
                   if track_crossings else 0)   # stretched -> compact (population sum)
    rep_c2s = (torch.zeros(sim.n_replicas, device=engine.device, dtype=torch.long)
               if track_crossings else None)
    rep_s2c = (torch.zeros(sim.n_replicas, device=engine.device, dtype=torch.long)
               if track_crossings else None)

    diag = {k: [] for k in ["steps", "times", "mean_force", "pmf",
                             "p_hat", "q_target", "pq_l2", "kl_pq", "eff_counts",
                             "frac_compact", "frac_transition", "frac_stretched",
                             "ancestor_ess", "n_unique_ancestor", "max_ancestor_frac",
                             "repl_cumulative"]}
    if readout is not None:
        diag["readout_mean_force"] = {h: [] for h in readout.hs if h > 0}
        if readout.raw is not None:
            diag["raw_fsum"], diag["raw_csum"] = [], []
    if sens is not None:
        diag["vhat"] = []
        if sensitivity_record:
            diag["sens_C"], diag["sens_Sf"], diag["sens_Sf2"] = [], [], []
    if relax is not None:
        diag["relax_steps"], diag["relax_active_frac"] = [], []

    t0 = time.perf_counter()
    for step in range(sim.n_steps + 1):
        forces_raw = engine.force(q, compute_energy=False)
        forces_physical = clip_forces(forces_raw, params.force_clip)
        z = reaction_coordinate(q, params)

        if track_crossings:
            cur_side = z > z_barrier
            if side is None:
                side = cur_side
            else:
                up = (~side) & cur_side       # compact -> stretched
                down = side & (~cur_side)     # stretched -> compact
                n_c2s_total += up.sum()       # GPU accumulate (no host sync)
                n_s2c_total += down.sum()
                rep_c2s += up.long()
                rep_s2c += down.long()
                side = cur_side

        if is_adaptive and step >= sim.estimator_burn_in_steps:
            # accumulate the cumulative z-marginal (count histogram) for the gate
            idx = ((z - sim.z_min) / max(_dz, EPS)).long().clamp_(0, sim.n_grid - 1)
            marg_hist.scatter_add_(0, idx, torch.ones_like(z))

        mean_force_input = forces_physical if sim.use_clipped_force_for_mean_force else forces_raw
        f_local = local_mean_force(q, mean_force_input, params)
        f_local = torch.clamp(f_local, -sim.mean_force_sample_clip, sim.mean_force_sample_clip)
        if ot is not None and ot_pending is not None:
            # what the moved walkers deposit on their first outer step after the event (T0: the
            # injected sample; TR: the post-repair residual).  Pure read-off, binned by z.
            bidx_post = torch.round((z - sim.z_min) / _dz).long().clamp_(0, sim.n_grid - 1)
            w_post = ot_pending.to(torch.float64)
            ot_C_post.scatter_add_(0, bidx_post, w_post)
            ot_Sf_post.scatter_add_(0, bidx_post, w_post * f_local.to(torch.float64))
            lin2 = bidx_post * n_dzb + ot_pending_dzb
            ot_C2_post.scatter_add_(0, lin2, w_post)
            ot_Sf2_post.scatter_add_(0, lin2, w_post * f_local.to(torch.float64))
            ot_pending = None
        bias_estimator.update(z, f_local)
        if step >= sim.estimator_burn_in_steps:
            production_estimator.update(z, f_local)
        if readout is not None:
            readout.update(z, f_local, step >= sim.estimator_burn_in_steps)
        if sens is not None:
            sens.update(z, f_local)
        ramp = min(1.0, step / max(sim.abf_warmup_steps, 1))
        abf_scale = sim.abf_bias_scale * ramp
        abf_at_z = abf_scale * torch.clamp(bias_estimator.evaluate(z), -sim.abf_force_clip, sim.abf_force_clip)
        A_hat = bias_estimator.pmf_profile()                 # unscaled A_hat_n(z)
        current_bias_profile = abf_scale * A_hat             # B_n(z)
        # Online EMA target maintained for the estimated-target methods (incl.
        # the adaptive variant), starting at fr_start. Never sees the TI reference.
        if method in ESTIMATED_TARGET_METHODS and (step + 1) >= sim.fr_start_steps:
            if F_target_ema is None:
                F_target_ema = A_hat.clone()
            else:
                r = sim.target_ema_rate
                F_target_ema = (1.0 - r) * F_target_ema + r * A_hat
            F_target_ema = normalize_profile_zero_at_midpoint_torch(F_target_ema, grid)

        transport = clip_forces(add_abf_force(q, forces_physical, abf_at_z, params), params.force_clip)
        transport = clip_forces(add_reaction_coordinate_wall_force(q, transport, z, sim, params), params.force_clip)

        if step % sim.save_every == 0 or step == sim.n_steps:
            report_estimator = production_estimator if production_estimator.n_updates > 0 else bias_estimator
            diag["steps"].append(step)
            diag["times"].append(step * sim.dt)
            diag["mean_force"].append(to_numpy(report_estimator.mean_force_profile()))
            diag["pmf"].append(to_numpy(report_estimator.pmf_profile()))
            diag["repl_cumulative"].append(total_replacement_events)
            if readout is not None:
                rep = readout.report()
                for h in diag["readout_mean_force"]:
                    diag["readout_mean_force"][h].append(rep[h])
                if readout.raw is not None:
                    diag["raw_fsum"].append(rep["raw_fsum"])
                    diag["raw_csum"].append(rep["raw_csum"])
            if sens is not None:
                diag["vhat"].append(to_numpy(sens.profile(sim.abf_bandwidth, sim.abf_smooth_sigma)))
                if sensitivity_record:
                    st_ = sens.state()
                    diag["sens_C"].append(st_["C"]); diag["sens_Sf"].append(st_["Sf"]); diag["sens_Sf2"].append(st_["Sf2"])
            if relax is not None:
                diag["relax_steps"].append(int(_rx_steps_acc))
                diag["relax_active_frac"].append(_rx_active_acc / max(_rx_n_acc, 1))
                _rx_steps_acc, _rx_active_acc, _rx_n_acc = 0, 0.0, 0
            if ot is not None:
                diag.setdefault("ot_steps", []).append(int(_ot_steps_acc))
                _ot_steps_acc = 0
                nev = max(_ot_events_acc, 1)
                diag.setdefault("ot_absdz_t", []).append(float(_ot_absdz_acc.item()) / float(sim.n_replicas * nev) if _ot_events_acc else float("nan"))
                diag.setdefault("ot_capped_t", []).append(float(_ot_capped_acc.item()) / float(sim.n_replicas * nev) if _ot_events_acc else float("nan"))
                _ot_absdz_acc.zero_(); _ot_capped_acc.zero_(); _ot_events_acc = 0
            if collect_diagnostics:
                fc, ft, fs = region_fractions_torch(z, sim)
                diag["frac_compact"].append(fc)
                diag["frac_transition"].append(ft)
                diag["frac_stretched"].append(fs)
                diag["eff_counts"].append(to_numpy(report_estimator.effective_counts()))
                p_grid = normalize_density_on_grid_torch(
                    kde_1d_torch(grid, z, sim.kde_bandwidth, sim.z_min, sim.z_max), grid)
                diag["p_hat"].append(to_numpy(p_grid))
                # build the *current* FR target for logging (no dynamics effect here)
                q_grid = _build_fr_target(method, grid, F_target_ema, current_bias_profile,
                                          oracle_t, params.beta)
                if q_grid is not None:
                    log_ratio = torch.log(torch.clamp(p_grid, min=EPS)) - torch.log(torch.clamp(q_grid, min=EPS))
                    diag["q_target"].append(to_numpy(q_grid))
                    diag["pq_l2"].append(float(math.sqrt(trapz_torch((p_grid - q_grid) ** 2, grid).item())))
                    diag["kl_pq"].append(float(trapz_torch(p_grid * log_ratio, grid).item()))
                else:
                    diag["q_target"].append(np.full(sim.n_grid, np.nan))
                    diag["pq_l2"].append(float("nan"))
                    diag["kl_pq"].append(float("nan"))
                if is_fr:
                    ess, nuq = ancestor_ess(ancestors, sim.n_replicas)
                    diag["ancestor_ess"].append(ess)
                    diag["n_unique_ancestor"].append(nuq)
                    diag["max_ancestor_frac"].append(ancestor_max_fraction(ancestors, sim.n_replicas))
                else:
                    diag["ancestor_ess"].append(float("nan"))
                    diag["n_unique_ancestor"].append(sim.n_replicas)
                    diag["max_ancestor_frac"].append(float("nan"))

        if step == sim.n_steps:
            break

        q = wrap_positions(q + sim.dt * transport + noise_scale * torch.randn_like(q), params.box_length)
        if is_fr:
            next_step = step + 1
            do_fr = (next_step >= sim.fr_start_steps
                     and (next_step - sim.fr_start_steps) % max(int(sim.fr_every), 1) == 0)
            if do_fr:
                z_new = reaction_coordinate(q, params)
                if is_sham:
                    # Replay the partner's realised count at this same opportunity index,
                    # with the direction randomised.  No target and no score are built, so a
                    # sham cannot read a reference even by accident.  The opportunity index
                    # is what makes the timing match: `do_fr` depends only on the step
                    # schedule, which is identical for every arm.
                    K = (replay_counts[sham_event_idx]
                         if sham_event_idx < len(replay_counts) else 0)
                    sham_event_idx += 1
                    q, ancestors, stats = uniform_birth_death_torch(q, K,
                                                                    ancestors=ancestors)
                    _track_window(stats, next_step)
                    total_replacement_events += stats["replacement"]
                    last_event_fraction = stats["replacement"] / float(sim.n_replicas)
                    fr_event_counts.append(int(stats["replacement"]))
                    if track_crossings and side is not None and stats["replacement"] > 0:
                        di, bs = stats["death_idx"], stats["birth_src"]
                        side[di] = side.index_select(0, bs)
                    if collect_diagnostics and stats["replacement"] > 0:
                        zb = z_new.index_select(0, stats["birth_src"])
                        zd = z_new.index_select(0, stats["death_idx"])
                        birth_hist += np.histogram(to_numpy(zb), bins=sim.n_grid,
                                                   range=(sim.z_min, sim.z_max))[0]
                        death_hist += np.histogram(to_numpy(zd), bins=sim.n_grid,
                                                   range=(sim.z_min, sim.z_max))[0]
                    q_grid = None
                else:
                    q_grid = _build_fr_target(method, grid, F_target_ema,
                                              current_bias_profile, oracle_t, params.beta)
                if is_prior:
                    score, p_fr = prior_selection_score_torch(
                        method, z_new, grid, sim, c=float(getattr(sim, "prior_c", 1.0)))
                    q_fr, kl_pq = None, float("nan")
                    q_grid = p_fr        # non-None so the shared branch below runs
                if q_grid is not None:
                    if not is_prior:
                        score, p_fr, q_fr, kl_pq = fr_score_torch(z_new, grid, sim, q_grid)
                    # aggregate score stats (cheap; used by Part B safety diagnostics)
                    sstat = score_statistics(score, sim.score_clip)
                    _score_std_sum += sstat["score_std"]
                    _score_absmax_max = max(_score_absmax_max, sstat["score_absmax"])
                    _score_clipfrac_sum += sstat["score_clip_fraction"]
                    _n_score_events += 1
                    # adaptive: compute the effective rate from ONLINE gates
                    rate_override = None
                    if is_adaptive:
                        ess_now, _ = ancestor_ess(ancestors, sim.n_replicas)
                        ess_frac = ess_now / sim.n_replicas if np.isfinite(ess_now) else float("nan")
                        # online support signal (mode-dependent) + EMA. For
                        # marginal_uniform use the LOW-VARIANCE cumulative marginal
                        # (broadly smoothed), not the noisy single-snapshot p_fr.
                        if sim.adaptive_support_mode == "marginal_uniform":
                            p_cum = normalize_density_on_grid_torch(
                                smooth_profile_torch(marg_hist, _kde_sigma_pts), grid)
                            sig = support_signal(sim, p_cum, None, grid)
                        else:
                            sig = support_signal(sim, p_fr, q_fr, grid)
                        if np.isfinite(sig):
                            if support_ema is None:
                                support_ema = sig
                            else:
                                r = sim.adaptive_support_ema_rate
                                support_ema = (1.0 - r) * support_ema + r * sig
                        rate_override, gates = adaptive_fr_rate(
                            sim, bias_estimator.effective_counts(), grid, ess_frac,
                            last_event_fraction, support_ema=support_ema, p_grid=p_fr)
                        # gate warmup: hold rate at zero until the cumulative marginal
                        # has settled (avoids early-transient false firing).
                        if (next_step - sim.fr_start_steps) < sim.adaptive_gate_warmup_steps:
                            rate_override = 0.0
                            gates["rate_eff"] = 0.0
                    q, ancestors, stats = fixed_population_birth_death_torch(
                        q, score, sim, ancestors=ancestors, fr_interval=sim.fr_every,
                        fr_rate_override=rate_override)
                    _track_window(stats, next_step)
                    total_replacement_events += stats["replacement"]
                    last_event_fraction = stats["replacement"] / float(sim.n_replicas)
                    # Recorded per FR opportunity so a matched sham can replay this exact
                    # sequence later; the sham runs in a separate process, so the schedule
                    # has to travel through the artifact.
                    fr_event_counts.append(int(stats["replacement"]))
                    if is_adaptive:
                        adaptive_log["step"].append(int(next_step))
                        adaptive_log["fr_rate_eff"].append(gates["rate_eff"])
                        adaptive_log["support_gate"].append(gates["support_gate"])
                        adaptive_log["diversity_gate"].append(gates["diversity_gate"])
                        adaptive_log["event_gate"].append(gates["event_gate"])
                        adaptive_log["neff_transition"].append(gates["neff_transition"])
                        adaptive_log["support_ema"].append(gates["support_ema"])
                        adaptive_log["ess_frac"].append(gates["ess_frac"])
                        adaptive_log["event_fraction"].append(last_event_fraction)
                        adaptive_log["score_std"].append(sstat["score_std"])
                        adaptive_log["score_absmax"].append(sstat["score_absmax"])
                        adaptive_log["score_clip_fraction"].append(sstat["score_clip_fraction"])
                    if track_crossings and side is not None and stats["replacement"] > 0:
                        # a clone inherits only the source's SIDE, so the teleport is
                        # not miscounted as a barrier crossing on the next step. The
                        # per-slot crossing tallies are NOT copied, so they stay an
                        # un-duplicated population tally (sum == global counters) and
                        # round_trips <= min(compact->stretched, stretched->compact).
                        di, bs = stats["death_idx"], stats["birth_src"]
                        side[di] = side.index_select(0, bs)
                    if collect_diagnostics and stats["replacement"] > 0:
                        # record births/deaths at their PRE-replacement z (z_new).
                        zb = z_new.index_select(0, stats["birth_src"])
                        zd = z_new.index_select(0, stats["death_idx"])
                        birth_hist += np.histogram(to_numpy(zb), bins=sim.n_grid,
                                                   range=(sim.z_min, sim.z_max))[0]
                        death_hist += np.histogram(to_numpy(zd), bins=sim.n_grid,
                                                   range=(sim.z_min, sim.z_max))[0]
                elif not is_sham:
                    # The target is not ready yet: a real opportunity with zero events. It
                    # must still occupy a slot, or a sham replaying the sequence would shift
                    # every later count onto the wrong opportunity.
                    fr_event_counts.append(0)

        if ot is not None:
            # Wasserstein reallocation along z + constrained fibre repair, on the FR schedule and
            # AFTER any birth-death.  Lift = project_dimer_to_z (midpoint and direction kept, bath
            # untouched); repair = the reference-consistent projected scheme for
            # ceil(c tau_f(z')/dt) steps per moved walker (own generator, nothing deposited, every
            # inner step charged).  'pre' = the lifted state's f_loc sample (deposit-free, one
            # diagnostic force evaluation, not charged; recorded only when there is a repair to
            # compare against); 'post' = the first outer deposit after the event (via ot_pending).
            next_step = step + 1
            do_ot = (next_step >= sim.fr_start_steps
                     and (next_step - sim.fr_start_steps) % max(int(sim.fr_every), 1) == 0)
            if do_ot:
                if engine.device.type == "cuda":
                    torch.cuda.synchronize()
                tw = time.perf_counter()
                z_old = reaction_coordinate(q, params)
                z_new = ot_displacement(z_old, float(ot.alpha), float(ot.dz_max), u_ot)
                dzabs = (z_new - z_old).abs()
                moved = dzabs > float(ot.min_move)
                repair_mask = torch.ones_like(moved) if bool(ot.repair_all) else moved
                q = project_dimer_to_z(q, z_new, params)
                ot_absdz_sum += dzabs.sum().to(torch.float64)
                ot_absdz_max = torch.maximum(ot_absdz_max, dzabs.max().to(torch.float64))
                _capped_now = (dzabs >= float(ot.dz_max) - 1e-12).sum().to(torch.float64)
                ot_capped += _capped_now
                ot_moved += moved.sum().to(torch.float64)
                _ot_absdz_acc += dzabs.sum().to(torch.float64); _ot_capped_acc += _capped_now; _ot_events_acc += 1
                ot_pending_dzb = torch.bucketize(dzabs, ot_dz_edges).clamp_(0, n_dzb - 1)
                if float(ot.c_repair) > 0.0:
                    bidx = torch.round((z_new - sim.z_min) / _dz).long().clamp_(0, sim.n_grid - 1)
                    wmv = moved.to(torch.float64)
                    f_pre = engine.force(q, compute_energy=False)
                    f_pre_in = clip_forces(f_pre, params.force_clip) if sim.use_clipped_force_for_mean_force else f_pre
                    fl_pre = torch.clamp(local_mean_force(q, f_pre_in, params),
                                         -sim.mean_force_sample_clip, sim.mean_force_sample_clip)
                    ot_C_pre.scatter_add_(0, bidx, wmv)
                    ot_Sf_pre.scatter_add_(0, bidx, wmv * fl_pre.to(torch.float64))
                    ot_diag_evals += 1
                    tau_i = interp_uniform_grid_edge(ot_tau_t, grid, z_new)
                    # ceil with a guard so that an exact integer (e.g. 0.02/0.002) is not pushed up by rounding
                    m_i = torch.ceil(float(ot.c_repair) * tau_i / sim.dt - 1e-6).long() * repair_mask.long()
                    idx_act = torch.nonzero(m_i > 0, as_tuple=False).flatten()
                    if int(idx_act.numel()) > 0:
                        q_act, done = frozen_dimer_relax(engine, params, sim, q.index_select(0, idx_act),
                                                         m_i.index_select(0, idx_act), gen_ot, scheme=ot.scheme)
                        q = q.clone()
                        q[idx_act] = q_act
                        ot_steps_total += done
                        _ot_steps_acc += done
                ot_pending = moved | repair_mask
                ot_events += 1
                if engine.device.type == "cuda":
                    torch.cuda.synchronize()
                ot_inner_wall += time.perf_counter() - tw

        if relax is not None:
            # Targeted constrained solvent relaxation, on the FR schedule and AFTER any
            # birth-death, so the post-selection population is the one being rejuvenated.
            # Importance = the ONLINE sensitivity field at the replica's current z; cost weight
            # = the frozen tau map; budget = rho x N x fr_every replica-steps.  The inner steps
            # deposit nothing (no C/Sf/Sf2, no bank, no FR, no ancestry, no error).
            next_step = step + 1
            do_relax = (next_step >= sim.fr_start_steps
                        and (next_step - sim.fr_start_steps) % max(int(sim.fr_every), 1) == 0)
            if do_relax and relax_budget_steps > 0:
                z_now = reaction_coordinate(q, params)
                vhat = sens.profile(sim.abf_bandwidth, sim.abf_smooth_sigma)
                a_imp = torch.clamp(interp_uniform_grid_edge(vhat, grid, z_now), min=0.0)
                tau_i = interp_uniform_grid_edge(tau_grid_t, grid, z_now)
                t_i = water_filling_durations(a_imp, tau_i, relax_budget_time)
                m_i = integer_steps_largest_remainder(t_i, sim.dt, relax_budget_steps)
                if relax.target == "random":
                    perm = torch.randperm(sim.n_replicas, device=engine.device, generator=gen_relax)
                    m_i = m_i.index_select(0, perm)
                active = m_i > 0
                n_active = int(active.sum().item())
                if n_active > 0:
                    if engine.device.type == "cuda":
                        torch.cuda.synchronize()
                    tw = time.perf_counter()
                    idx_act = torch.nonzero(active, as_tuple=False).flatten()
                    q_act, done = frozen_dimer_relax(engine, params, sim, q.index_select(0, idx_act),
                                                     m_i.index_select(0, idx_act), gen_relax, scheme=relax.scheme)
                    q = q.clone()
                    q[idx_act] = q_act
                    if engine.device.type == "cuda":
                        torch.cuda.synchronize()
                    relax_inner_wall += time.perf_counter() - tw
                    relax_steps_total += done
                    _rx_steps_acc += done
                    bidx = torch.round((z_now - sim.z_min) / _dz).long().clamp_(0, sim.n_grid - 1)
                    relax_hist.scatter_add_(0, bidx, m_i.to(torch.float64))
                relax_events += 1
                _rx_active_acc += n_active / float(sim.n_replicas)
                _rx_n_acc += 1

    diag["runtime_seconds"] = time.perf_counter() - t0
    diag["method"] = method
    diag["grid"] = to_numpy(grid)
    diag["total_replacement_events"] = total_replacement_events
    diag["fr_event_counts"] = np.asarray(fr_event_counts, dtype=np.int64)
    diag["sham_partner"] = SHAM_PARTNER.get(method)
    diag["sham_replayed_events"] = int(sham_event_idx) if is_sham else 0
    diag["F_target_ema"] = to_numpy(F_target_ema) if F_target_ema is not None else None
    diag["birth_hist"] = birth_hist
    diag["death_hist"] = death_hist
    diag["hist_edges"] = np.linspace(sim.z_min, sim.z_max, sim.n_grid + 1)
    # aggregate score statistics over FR events (NaN for abf / no events)
    if _n_score_events > 0:
        diag["fr_score_std"] = _score_std_sum / _n_score_events
        diag["fr_score_absmax"] = _score_absmax_max
        diag["fr_score_clip_fraction"] = _score_clipfrac_sum / _n_score_events
    else:
        diag["fr_score_std"] = float("nan")
        diag["fr_score_absmax"] = float("nan")
        diag["fr_score_clip_fraction"] = float("nan")
    # adaptive per-event log (empty arrays for non-adaptive methods)
    diag["adaptive_log"] = {k: np.asarray(v, dtype=float) for k, v in adaptive_log.items()}
    if track_crossings:
        c2s = int(n_c2s_total.item())
        s2c = int(n_s2c_total.item())
        diag["n_compact_to_stretched"] = c2s
        diag["n_stretched_to_compact"] = s2c
        diag["n_barrier_crossings"] = c2s + s2c
        # a completed round trip needs both an up and a down leg in the same replica
        diag["n_round_trips"] = int(torch.minimum(rep_c2s, rep_s2c).sum().item())
    else:
        diag["n_compact_to_stretched"] = -1
        diag["n_stretched_to_compact"] = -1
        diag["n_barrier_crossings"] = -1
        diag["n_round_trips"] = -1
    for key in ["steps", "times", "mean_force", "pmf", "p_hat", "q_target", "eff_counts",
                "pq_l2", "kl_pq", "frac_compact", "frac_transition", "frac_stretched",
                "ancestor_ess", "n_unique_ancestor", "max_ancestor_frac", "repl_cumulative"]:
        diag[key] = np.asarray(diag[key])
    # The §3.3 gate statistic. NaN for `abf`, which has no genealogy at all.
    diag["min_ancestor_ess_window"] = (float(min_ess_window) if is_fr else float("nan"))
    diag["ess_window_steps"] = int(_win)
    if sens is not None:
        diag["vhat"] = np.asarray(diag["vhat"], dtype=np.float64)
        diag["final_vhat"] = diag["vhat"][-1]
        if sensitivity_record:
            for k in ("sens_C", "sens_Sf", "sens_Sf2"):
                diag[k] = np.asarray(diag[k], dtype=np.float64)
            diag["final_q"] = to_numpy(q).astype(np.float32)
    if relax is not None:
        diag["relax_rho"] = float(relax.rho)
        diag["relax_target"] = relax.target
        diag["relax_scheme"] = relax.scheme
        diag["relax_budget_steps_per_opportunity"] = int(relax_budget_steps)
        diag["relax_steps_total"] = int(relax_steps_total)
        diag["relax_cost_ratio"] = relax_steps_total / float(sim.n_replicas * sim.n_steps)
        diag["relax_n_opportunities"] = int(relax_events)
        diag["relax_inner_wall_seconds"] = float(relax_inner_wall)
        diag["relax_budget_hist"] = to_numpy(relax_hist)
        diag["relax_steps"] = np.asarray(diag["relax_steps"], dtype=np.int64)
        diag["relax_active_frac"] = np.asarray(diag["relax_active_frac"], dtype=np.float64)
        diag["tau_grid"] = np.asarray(relax.tau_grid, dtype=np.float64)
    if ot is not None:
        n_ev = max(ot_events, 1)
        diag["ot_alpha"], diag["ot_dz_max"], diag["ot_c_repair"] = float(ot.alpha), float(ot.dz_max), float(ot.c_repair)
        diag["ot_scheme"] = ot.scheme
        diag["ot_n_opportunities"] = int(ot_events)
        diag["ot_moved_frac"] = float(ot_moved.item()) / float(sim.n_replicas * n_ev)
        diag["ot_absdz_mean"] = float(ot_absdz_sum.item()) / float(sim.n_replicas * n_ev)
        diag["ot_absdz_max"] = float(ot_absdz_max.item())
        diag["ot_capped_frac"] = float(ot_capped.item()) / float(sim.n_replicas * n_ev)
        diag["ot_diag_force_evals"] = int(ot_diag_evals)
        diag["ot_C_pre"], diag["ot_Sf_pre"] = to_numpy(ot_C_pre), to_numpy(ot_Sf_pre)
        diag["ot_C_post"], diag["ot_Sf_post"] = to_numpy(ot_C_post), to_numpy(ot_Sf_post)
        diag["ot_C2_post"] = to_numpy(ot_C2_post).reshape(sim.n_grid, n_dzb)
        diag["ot_Sf2_post"] = to_numpy(ot_Sf2_post).reshape(sim.n_grid, n_dzb)
        diag["ot_absdz_edges"] = np.asarray(ABSDZ_EDGES, dtype=np.float64)
        diag["ot_repair_all"] = bool(ot.repair_all)
        diag["ot_absdz_t"] = np.asarray(diag.get("ot_absdz_t", []), dtype=np.float64)
        diag["ot_capped_t"] = np.asarray(diag.get("ot_capped_t", []), dtype=np.float64)
        diag["ot_steps"] = np.asarray(diag.get("ot_steps", []), dtype=np.int64)
        if relax is None:
            # the compute-accounting keys the campaign analyzers already read (exact: every inner
            # replica-step of the repair is charged; the 'pre' diagnostic evaluations are not)
            diag["relax_steps_total"] = int(ot_steps_total)
            diag["relax_cost_ratio"] = ot_steps_total / float(sim.n_replicas * sim.n_steps)
            diag["relax_inner_wall_seconds"] = float(ot_inner_wall)
            diag["relax_steps"] = diag["ot_steps"].copy()
            diag["relax_n_opportunities"] = int(ot_events)
            diag["relax_scheme"] = ot.scheme
            diag["tau_grid"] = np.asarray(ot.tau_grid, dtype=np.float64)
    if verbose:
        extra = f", replacements {total_replacement_events}" if is_fr else ""
        print(f"{method:13s}: {diag['runtime_seconds']:.1f}s{extra}")
    return diag


def _build_fr_target(method, grid, F_target_ema, current_bias_profile, oracle_t, beta):
    """Construct q_n(z) for the given FR method, or None for abf / not-yet-started."""
    if method == "abf":
        return None
    if method == "fr_uniform":
        return fr_target_uniform_torch(grid)
    if method in ESTIMATED_TARGET_METHODS:
        if F_target_ema is None:
            return None
        return fr_target_estimated_torch(grid, F_target_ema, current_bias_profile, beta)
    if method == "fr_oracle":
        return fr_target_oracle_torch(grid, oracle_t, current_bias_profile, beta)
    if method in PRIOR_SELECTION_METHODS:
        return None          # these build their own score; no target exists
    if method in SHAM_METHODS:
        # A sham has no target by construction -- it replays its partner's event counts and
        # picks uniformly at random. Returning None (rather than raising) is what keeps the
        # diagnostics call site honest: there is nothing to log, so it logs NaN.
        return None
    raise ValueError(method)


# ---------------------------------------------------------------------------
# Frozen-bias validation (Part F): run under a FIXED learned bias, no ABF/FR.
# ---------------------------------------------------------------------------
@torch.inference_mode()
def run_frozen_bias_gpu(params, sim, engine, learned_mean_force, learned_pmf,
                        initial_q=None, burn_in_steps=None, verbose=True):
    """Sample under a FIXED bias B(z) and reconstruct F from the biased marginal.

    The learned bias is held constant (no ABF update, no Fisher--Rao birth--death):
    at every step the biasing force ``+learned_mean_force(z)`` (edge-extrapolated
    interpolation on the z-grid, clipped like the online run) is added, exactly the
    converged ABF transport but frozen. Particles explore the residual landscape
    ``F(z) - B(z)`` where ``B(z) = learned_pmf(z)`` is the integral of the learned
    mean force. The biased reaction-coordinate marginal ``p_B(z)`` is accumulated by
    KDE after a burn-in, and the free energy is reconstructed as

        F_recon(z) = B(z) - beta^{-1} log p_B(z) + C,

    with C fixed later by centring on the evaluation window. No TI reference is read
    here, so the frozen test is deployable for any learned bias (abf / fr_* alike).

    Parameters
    ----------
    learned_mean_force, learned_pmf : 1-D arrays on the z-grid (length sim.n_grid),
        the stage-1 ``final_mean_force`` and ``final_pmf``.
    """
    grid = torch.linspace(sim.z_min, sim.z_max, sim.n_grid, device=engine.device, dtype=engine.dtype)
    mf_grid = torch.as_tensor(np.asarray(learned_mean_force, dtype=np.float64),
                              device=engine.device, dtype=engine.dtype)
    B_grid = torch.as_tensor(np.asarray(learned_pmf, dtype=np.float64),
                             device=engine.device, dtype=engine.dtype)
    burn_in = int(sim.estimator_burn_in_steps if burn_in_steps is None else burn_in_steps)

    torch.manual_seed(sim.seed)
    q = (lattice_initial_conditions(params, sim.n_replicas, engine.device, engine.dtype, seed=sim.seed)
         if initial_q is None else initial_q.clone())
    noise_scale = math.sqrt(2.0 * sim.dt / params.beta)

    p_accum = torch.zeros_like(grid)
    n_marg = 0
    t0 = time.perf_counter()
    for step in range(sim.n_steps + 1):
        forces_raw = engine.force(q, compute_energy=False)
        forces_physical = clip_forces(forces_raw, params.force_clip)
        z = reaction_coordinate(q, params)
        # frozen bias force at the current z (edge-extrapolated, force-clipped)
        bias_at_z = torch.clamp(interp_uniform_grid_edge(mf_grid, grid, z),
                                -sim.abf_force_clip, sim.abf_force_clip)
        transport = clip_forces(add_abf_force(q, forces_physical, bias_at_z, params), params.force_clip)
        transport = clip_forces(add_reaction_coordinate_wall_force(q, transport, z, sim, params), params.force_clip)
        if step >= burn_in and (step % max(int(sim.fr_every), 1) == 0):
            p_accum += kde_1d_torch(grid, z, sim.kde_bandwidth, sim.z_min, sim.z_max)
            n_marg += 1
        if step == sim.n_steps:
            break
        q = wrap_positions(q + sim.dt * transport + noise_scale * torch.randn_like(q), params.box_length)

    p_B = normalize_density_on_grid_torch(p_accum / max(n_marg, 1), grid)
    beta = params.beta
    log_pB = torch.log(torch.clamp(p_B, min=EPS))
    F_recon = B_grid - (1.0 / beta) * log_pB
    F_recon = normalize_profile_zero_at_midpoint_torch(F_recon, grid)
    out = dict(
        grid=to_numpy(grid), p_B=to_numpy(p_B), F_recon=to_numpy(F_recon),
        learned_pmf=to_numpy(B_grid), learned_mean_force=to_numpy(mf_grid),
        n_marginal_snapshots=n_marg, runtime_seconds=time.perf_counter() - t0,
    )
    if verbose:
        print(f"frozen-bias : {out['runtime_seconds']:.1f}s ({n_marg} marginal snapshots)")
    return out


# ---------------------------------------------------------------------------
# TI reference (EVALUATION ONLY) + cache
# ---------------------------------------------------------------------------
@torch.inference_mode()
def constrained_ti_reference_gpu(params, sim, ti, engine, eval_grid_np=None, verbose=True):
    torch.manual_seed(ti.seed)
    z_all = torch.linspace(ti.z_min, ti.z_max, ti.n_z, device=engine.device, dtype=engine.dtype)
    mf = torch.zeros(ti.n_z, device=engine.device, dtype=engine.dtype)
    count = torch.zeros(ti.n_z, device=engine.device, dtype=engine.dtype)
    noise_scale = math.sqrt(2.0 * ti.dt / params.beta)

    t0 = time.perf_counter()
    for start in range(0, ti.n_z, ti.z_chunk_size):
        end = min(start + ti.z_chunk_size, ti.n_z)
        z_chunk = z_all[start:end]
        C = z_chunk.numel()
        q0 = lattice_initial_conditions(params, C * ti.n_replicas, engine.device, engine.dtype, seed=ti.seed + start)
        z_batch = z_chunk.repeat_interleave(ti.n_replicas)
        q = project_dimer_to_z(q0, z_batch, params)
        total_steps = ti.n_thermalization + ti.n_steps
        for step in range(total_steps):
            forces_raw = engine.force(q, compute_energy=False)
            transport = clip_forces(forces_raw, params.force_clip)
            q = wrap_positions(q + ti.dt * transport + noise_scale * torch.randn_like(q), params.box_length)
            q = project_dimer_to_z(q, z_batch, params)
            if step >= ti.n_thermalization and (step - ti.n_thermalization) % ti.sample_every == 0:
                sample_forces = engine.force(q, compute_energy=False)
                if sim.use_clipped_force_for_mean_force:
                    sample_forces = clip_forces(sample_forces, params.force_clip)
                f_local = local_mean_force(q, sample_forces, params).view(C, ti.n_replicas)
                mf[start:end] += f_local.sum(dim=1)
                count[start:end] += ti.n_replicas
        if verbose:
            print(f"TI chunk {start:02d}:{end:02d} done, elapsed {(time.perf_counter()-t0)/60:.1f} min")

    mean_force = mf / torch.clamp(count, min=1.0)
    mean_force_smooth = smooth_profile_torch(mean_force, ti.smooth_sigma)
    free_energy = normalize_profile_zero_at_midpoint_torch(cumulative_trapezoid_torch(mean_force_smooth, z_all), z_all)

    if eval_grid_np is None:
        eval_grid = torch.linspace(sim.z_min, sim.z_max, sim.n_grid, device=engine.device, dtype=engine.dtype)
    else:
        eval_grid = torch.as_tensor(eval_grid_np, device=engine.device, dtype=engine.dtype)
    mf_eval = interp_uniform_grid(mean_force_smooth, z_all, eval_grid, outside_value=0.0)
    fe_eval = interp_uniform_grid(free_energy, z_all, eval_grid, outside_value=float(free_energy[0].item()))
    fe_eval = normalize_profile_zero_at_midpoint_torch(fe_eval, eval_grid)
    return {
        "label": "thermodynamic integration reference (evaluation only)",
        "grid": to_numpy(eval_grid),
        "mean_force": to_numpy(mf_eval),
        "free_energy": to_numpy(fe_eval),
        "z_ti": to_numpy(z_all),
        "runtime_seconds": time.perf_counter() - t0,
    }


def load_or_compute_ti_reference(path, params, sim, ti, engine, verbose=True):
    """Load the cached TI reference if the grid matches, else compute and cache it.

    EVALUATION ONLY. Never feeds the Fisher-Rao target (except fr_oracle diagnostic).
    """
    eval_grid_np = np.linspace(sim.z_min, sim.z_max, sim.n_grid)
    if os.path.exists(path):
        d = np.load(path, allow_pickle=True)
        grid_ok = (d["grid"].shape[0] == sim.n_grid
                   and abs(float(d["grid"][0]) - sim.z_min) < 1e-5
                   and abs(float(d["grid"][-1]) - sim.z_max) < 1e-5)
        if grid_ok:
            if verbose:
                print(f"Loaded cached TI reference from {path}")
            return {"label": str(d["label"]), "grid": d["grid"],
                    "mean_force": d["mean_force"], "free_energy": d["free_energy"], "z_ti": d["z_ti"]}
        if verbose:
            print(f"Cache {path} grid mismatch; recomputing.")
    ref = constrained_ti_reference_gpu(params, sim, ti, engine, eval_grid_np=eval_grid_np, verbose=verbose)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, label=ref["label"], grid=ref["grid"], mean_force=ref["mean_force"],
             free_energy=ref["free_energy"], z_ti=ref["z_ti"])
    if verbose:
        print(f"Saved TI reference to {path} (runtime {ref['runtime_seconds']/60:.1f} min)")
    return ref


# ---------------------------------------------------------------------------
# Evaluation metrics vs reference (additive-constant aligned on interior window)
# ---------------------------------------------------------------------------
def final_l2_errors(diag, reference, sim):
    grid = reference["grid"]
    mask = eval_window_mask_np(grid, sim)
    mf, fe = diag["mean_force"][-1], diag["pmf"][-1]
    fe_al = align_additive_constant_np(fe, reference["free_energy"], grid, mask=mask)
    l2_fp = profile_l2_error_np(mf, reference["mean_force"], grid, mask=mask)
    l2_f = profile_l2_error_np(fe_al, reference["free_energy"], grid, mask=mask)
    reg_f = region_l2_errors(fe_al, reference["free_energy"], grid, sim)
    reg_fp = region_l2_errors(mf, reference["mean_force"], grid, sim)
    return {"l2_f": l2_f, "l2_fp": l2_fp,
            "l2_f_compact": reg_f["compact"], "l2_f_transition": reg_f["transition"],
            "l2_f_stretched": reg_f["stretched"],
            "l2_fp_compact": reg_fp["compact"], "l2_fp_transition": reg_fp["transition"],
            "l2_fp_stretched": reg_fp["stretched"]}


def timeseries_l2(diag, reference, sim):
    """L2(F) and L2(F') at every saved time; plus integrated L2(F) over time."""
    grid = reference["grid"]
    mask = eval_window_mask_np(grid, sim)
    times = np.asarray(diag["times"], dtype=float)
    l2_f_t, l2_fp_t = [], []
    for k in range(len(times)):
        fe_al = align_additive_constant_np(diag["pmf"][k], reference["free_energy"], grid, mask=mask)
        l2_f_t.append(profile_l2_error_np(fe_al, reference["free_energy"], grid, mask=mask))
        l2_fp_t.append(profile_l2_error_np(diag["mean_force"][k], reference["mean_force"], grid, mask=mask))
    l2_f_t = np.asarray(l2_f_t)
    l2_fp_t = np.asarray(l2_fp_t)
    integrated_f = float(np.trapezoid(l2_f_t, times)) if len(times) > 1 else float("nan")
    return {"times": times, "l2_f_t": l2_f_t, "l2_fp_t": l2_fp_t, "integrated_l2_f": integrated_f}
