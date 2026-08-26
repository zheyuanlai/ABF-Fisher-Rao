"""Batched PyTorch ABF(+Fisher--Rao) engine (GPU backend).

This is the fast backend for the 2D ``xi(x, y) = x`` study.  It simulates a
*batch* of independent runs at once: particle tensors have shape ``(B, N)`` with
``B`` = runs/seeds/configs in the batch and ``N`` = particles per run.  The
science is identical to the CPU reference (:mod:`abffr.simulation`):

* same potential / reaction coordinate (:func:`abffr.potentials.*_torch`),
* same biased overdamped dynamics,
* same ABF target ``F'(x) = E[dV/dx | X = x]``,
* same Fisher--Rao target densities and score,
* same metrics (each run produces a ``diag`` dict with the *identical* structure
  to :func:`abffr.simulation.run_simulation`, so :mod:`abffr.metrics` and
  :mod:`abffr.diagnostics` are reused verbatim).

Two estimator modes are provided:

``binned_smooth`` (production)
    Bin particles onto the x-grid (``scatter_add``) and Gaussian-smooth the
    count/force histograms -- an ``O(N + G)`` per-step estimator that is a
    Riemann-sum approximation of the CPU kernel estimator (see
    :mod:`abffr.torch_utils`).
``kernel_reference`` (validation only, slow)
    The exact ``O(G*N)`` Nadaraya--Watson kernel estimator and reflected KDE,
    a faithful GPU port of the CPU engine, used to bound the binning error.

A batch must share ``target_type``, ``eta``, ``fr_every``, ``burnin_fraction``
and ``stop_fraction`` (so the FR firing schedule and smoothing bandwidth are
common); ``gamma`` and ``seed`` may vary per row. Grouping is handled by
:mod:`abffr.parallel`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch

from . import (clean_v2 as cv2, family as fam, fibre_diagnostics as fib,
               fr_v3, persistent_mass as pmass, potentials,
               representation as rep, torch_utils as tu)
from .io_utils import RunSpec, make_rng_streams
from .simulation import _init_positions  # reuse CPU init for matched seeds

EPS = 1e-12

# Score-shape diagnostics: levels and their column suffixes stay paired here so
# the recorded quantile cannot drift from its label.
SCORE_QUANTILE_LEVELS = (0.01, 0.10, 0.50, 0.90, 0.99)
SCORE_QUANTILE_LABELS = ("q01", "q10", "q50", "q90", "q99")


@dataclass
class BatchResult:
    diags: List[Dict]          # one CPU-style diag dict per row
    runtime_seconds: float     # wall-clock for the whole batch
    device: str
    fr_opportunities: Optional[List[int]] = None   # v3: exact firing steps
    v3_events: Optional[List[Dict]] = None         # Amendment 4c diagnostics
    v4_events: Optional[List[Dict]] = None         # v4-A mass/representation
    clean_events: Optional[List[Dict]] = None      # clean-v2 per-FR-pulse rows


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _init_batch_positions(specs, n_particles, domain, x_mode, y_mode,
                          device, dtype):
    """Per-row initial conditions, matched to the CPU init stream by seed.

    Using :func:`abffr.io_utils.make_rng_streams` (the same SeedSequence split
    as the CPU engine) means a given ``seed`` has *identical* initial conditions
    across methods and across the CPU/torch backends -- the basis for clean
    matched-seed comparisons.
    """
    xmin, xmax = domain["x_min"], domain["x_max"]
    ymin, ymax = domain["y_min"], domain["y_max"]
    X = np.empty((len(specs), n_particles), dtype=np.float64)
    Y = np.empty((len(specs), n_particles), dtype=np.float64)
    for b, spec in enumerate(specs):
        rng_init, _, _ = make_rng_streams(spec.seed)
        X[b] = _init_positions(rng_init, n_particles, xmin, xmax, x_mode)
        Y[b] = _init_positions(rng_init, n_particles, ymin, ymax, y_mode)
    return (torch.as_tensor(X, device=device, dtype=dtype),
            torch.as_tensor(Y, device=device, dtype=dtype))


class _MatchedNoiseBank:
    """Chunked, configuration-independent Langevin variates.

    Each seed owns one generator keyed only by (base_seed, seed). Repeated rows
    with the same matched seed receive identical additive noise irrespective of
    method, batch composition, row order, or sharding. Chunking avoids both a
    Python RNG call per row/step and a full-run GPU buffer.
    """

    def __init__(self, specs, n_particles, n_steps, device, dtype, base_seed,
                 chunk_steps=1024):
        self.n_particles = int(n_particles)
        self.n_steps = int(n_steps)
        self.device = device
        self.dtype = dtype
        self.chunk_steps = max(int(chunk_steps), 1)
        seeds = sorted({int(s.seed) for s in specs})
        seed_to_row = {seed: i for i, seed in enumerate(seeds)}
        self.row_index = torch.as_tensor(
            [seed_to_row[int(s.seed)] for s in specs], device=device,
            dtype=torch.long)
        self.generators = [
            tu.make_generator(tu.stable_seed("langevin", base_seed, seed), device)
            for seed in seeds
        ]
        self.chunk = None
        self.chunk_start = -1

    def at(self, step):
        if (self.chunk is None or step < self.chunk_start
                or step >= self.chunk_start + self.chunk.shape[1]):
            length = min(self.chunk_steps, self.n_steps - int(step))
            self.chunk = torch.stack([
                torch.randn((length, 2, self.n_particles), generator=g,
                            device=self.device, dtype=self.dtype)
                for g in self.generators
            ], dim=0)
            self.chunk_start = int(step)
        noise = self.chunk.index_select(
            0, self.row_index)[:, step - self.chunk_start]
        return noise[:, 0], noise[:, 1]


def _empty_diag(target_type):
    return dict(
        target_type=target_type,
        steps=[], times=[],
        Fprime_hat=[], F_hat=[], p_hat_grid=[], q_target_grid=[],
        X_snap=[], Y_snap=[],
        barrier_crossings=[], n_unique_ancestors=[], gamma_eff=[],
        fr_applied=[], fr_event_fraction=[], fr_event_fraction_max=[],
        fr_events_total=[], score_mean=[], score_std=[], score_min=[],
        score_max=[], cumulative_fr_events=[], cumulative_replacements=[],
        ancestor_ess=[], max_clone_multiplicity=[], max_clone_weight=[],
        target_l2=[], score_clipped_fraction=[],
        # The Amendment 4c per-opportunity diagnostics are NOT snapshot columns:
        # they live one row per FR opportunity in the v3_events CSV.  Declaring
        # them here advertised a schema that was always empty.
        **{f"score_raw_{lab}": [] for lab in SCORE_QUANTILE_LABELS},
        **{f"score_applied_{lab}": [] for lab in SCORE_QUANTILE_LABELS},
    )


def _kernel_estimator(x_grid_t, X, Y, h, x_tilt=0.0):
    """Exact O(G*N) Nadaraya--Watson contribution for one step (per row).

    Returns ``(num_contrib, den_contrib)`` each ``(B, G)`` matching the CPU
    ``weights`` accumulation, for the ``kernel_reference`` validation mode.
    """
    # (B, G, N): kernel of every grid node against every particle.
    diff = x_grid_t.view(1, -1, 1) - X.unsqueeze(1)
    w = torch.exp(-0.5 * (diff / h) ** 2) / (h * np.sqrt(2.0 * np.pi))
    dvdx = potentials.dVdx_xy_torch(X, Y) + float(x_tilt)
    return (w * dvdx.unsqueeze(1)).sum(-1), w.sum(-1)


def _kde_reflected(x_grid_t, X, eta, xmin, xmax):
    """Reflected-boundary KDE marginal on the grid (``kernel_reference`` mode).

    Mirror images about ``xmin``/``xmax`` remove KDE edge bias, matching
    :func:`abffr.simulation.kde_marginal`.  Returns ``(B, G)``.
    """
    N = X.shape[1]
    X_all = torch.cat([2.0 * xmin - X, X, 2.0 * xmax - X], dim=1)
    diff = x_grid_t.view(1, -1, 1) - X_all.unsqueeze(1)
    w = torch.exp(-0.5 * (diff / eta) ** 2) / (eta * np.sqrt(2.0 * np.pi))
    return w.sum(-1) / N


# --------------------------------------------------------------------------- #
# Fisher--Rao target densities and score (batched, on grid)
# --------------------------------------------------------------------------- #
def _build_target(target_type, Fhat_target, B_grid, F_ref_grid, p_hat,
                  beta, dx, width):
    """Build a normalized batched FR target on the reaction-coordinate grid."""
    allowed = {"none", "estimated", "uniform", "oracle", "self",
               "physical", "physical_oracle"}
    if target_type not in allowed:
        raise ValueError(f"Unknown target_type {target_type!r}")
    if target_type == "uniform":
        B, G = p_hat.shape
        return torch.full((B, G), 1.0 / max(width, EPS),
                          device=p_hat.device, dtype=p_hat.dtype)
    if target_type == "self":
        return tu.normalize_density(p_hat, dx)

    uses_oracle = target_type in {"oracle", "physical_oracle"}
    F_target = F_ref_grid.expand_as(p_hat) if uses_oracle else Fhat_target
    if target_type in {"physical", "physical_oracle"}:
        exponent = -beta * F_target
    else:
        exponent = -beta * (F_target - B_grid)
    exponent = exponent - exponent.max(dim=1, keepdim=True).values
    q = tu.normalize_density(torch.exp(exponent), dx)
    if target_type in {"physical", "physical_oracle"}:
        return q
    q = q.clamp_min(EPS)
    return tu.normalize_density(q, dx)


def _score_quantiles(S, levels):
    """Per-row quantiles of a ``(B, N)`` score, returned as ``(B, len(levels))``."""
    return torch.quantile(S, levels, dim=1).transpose(0, 1)


def _fr_score(X, p_hat, q_grid, x0, dx, beta, score_clip):
    """Batched Fisher--Rao score (mean-zero), matching ``simulation.fr_score``.

    Returns ``(S, S_raw)``. ``S_raw`` is the mean-centered score before any
    clipping; only the diagnostics read it.
    """
    Zp = tu.trapezoid(p_hat, dx).clamp_min(EPS).unsqueeze(1)
    Zq = tu.trapezoid(q_grid, dx).clamp_min(EPS).unsqueeze(1)
    p_g = p_hat / Zp
    q_g = q_grid / Zq
    log_ratio_grid = torch.log(p_g.clamp_min(EPS)) - torch.log(q_g.clamp_min(EPS))
    baseline = tu.trapezoid(p_g * log_ratio_grid, dx).unsqueeze(1)  # KL(p||q)

    p_part = tu.interp1d(p_g, X, x0, dx).clamp_min(EPS)
    q_part = tu.interp1d(q_g, X, x0, dx).clamp_min(EPS)
    S = torch.log(p_part) - torch.log(q_part) - baseline
    S = S - S.mean(dim=1, keepdim=True)
    S_raw = S
    if score_clip is not None:
        for _ in range(3):
            S = S.clamp(-float(score_clip), float(score_clip))
            S = S - S.mean(dim=1, keepdim=True)
    return S, S_raw


def _resample(X, Y, ancestors, S, gamma_eff, dt, max_event_fraction,
              generators, jitter, noise_scale):
    """Fixed-N birth--death with one independent FR stream per run."""
    B, N = X.shape
    X_new, Y_new, anc_new = X.clone(), Y.clone(), ancestors.clone()
    n_death = torch.zeros(B, device=X.device, dtype=torch.long)
    cap = N if max_event_fraction is None else int(
        np.floor(float(max_event_fraction) * N))
    if cap <= 0:
        return X_new, Y_new, anc_new, n_death

    for b, gen in enumerate(generators):
        if float(gamma_eff[b].detach().cpu()) <= 0.0:
            continue
        score = S[b]
        positive = score > 0
        birth_weight = torch.where(score < 0, -score, torch.zeros_like(score))
        if not bool((birth_weight.sum() > EPS).detach().cpu()):
            continue
        prob = torch.where(
            positive,
            (1.0 - torch.exp(
                -gamma_eff[b] * float(dt) * score)).clamp(0.0, 1.0),
            torch.zeros_like(score))
        die = torch.nonzero(
            positive & (torch.rand((N,), generator=gen, device=X.device,
                                   dtype=X.dtype) < prob),
            as_tuple=False).flatten()
        if die.numel() > cap:
            die = die[torch.randperm(
                die.numel(), generator=gen, device=X.device)[:cap]]
        n = int(die.numel())
        if n == 0:
            continue
        source = torch.multinomial(
            birth_weight, n, replacement=True, generator=gen)
        X_new[b, die] = X[b, source]
        Y_new[b, die] = Y[b, source]
        anc_new[b, die] = ancestors[b, source]
        if jitter and jitter > 0.0:
            X_new[b, die] += (
                jitter * noise_scale
                * torch.randn((n,), generator=gen, device=X.device,
                              dtype=X.dtype))
            Y_new[b, die] += (
                jitter * noise_scale
                * torch.randn((n,), generator=gen, device=Y.device,
                              dtype=Y.dtype))
        n_death[b] = n
    return X_new, Y_new, anc_new, n_death


# --------------------------------------------------------------------------- #
# Core batched engine
# --------------------------------------------------------------------------- #
def run_batch(
    specs: List[RunSpec],
    *,
    cfg: Dict,
    x_grid: np.ndarray,
    F_ref: np.ndarray,
    Fprime_ref: np.ndarray,
    ev,
    device: torch.device,
    dtype: torch.dtype,
    estimator: str = "binned_smooth",
    base_seed: int = 0,
) -> BatchResult:
    """Simulate one schedule-homogeneous batch of independent runs."""
    if not specs:
        return BatchResult([], 0.0, str(device))
    target_type = specs[0].target_type
    eta = float(specs[0].eta)
    fr_every = int(specs[0].fr_every)
    burnin_fraction = float(specs[0].burnin_fraction)
    stop_fraction = float(getattr(specs[0], "stop_fraction", 1.0))
    if not 0.0 <= burnin_fraction <= stop_fraction <= 1.0:
        raise ValueError("FR schedule must satisfy 0 <= burnin <= stop <= 1")
    for s in specs:
        key = (s.target_type, float(s.eta), int(s.fr_every),
               float(s.burnin_fraction),
               float(getattr(s, "stop_fraction", 1.0)))
        if key != (target_type, eta, fr_every, burnin_fraction, stop_fraction):
            raise ValueError(
                "run_batch requires a target/eta/every/on/off-homogeneous batch")

    sim = cfg["simulation"]
    abf = cfg["abf"]
    fr = cfg.get("fr", {})
    beta = float(sim["beta"]); dt = float(sim["dt"])
    n_steps = int(sim["n_steps"]); n_particles = int(sim["n_particles"])
    eval_every = int(sim["eval_every"])
    domain = cfg["domain"]
    xmin, xmax = float(domain["x_min"]), float(domain["x_max"])
    ymin, ymax = float(domain["y_min"]), float(domain["y_max"])
    width = xmax - xmin

    h = float(abf["h"])
    update_every = max(1, int(abf.get("update_every", 1)))
    min_count = float(abf.get("min_count", 1.0))
    # Estimated-target free-energy EMA: the study spec names this
    # ``fr.target_ema_alpha``; fall back to the CPU engine's ``abf.ema_alpha``
    # (both default to 0.05, so the CPU/torch estimated target stays consistent).
    ema_alpha = float(fr.get("target_ema_alpha", abf.get("ema_alpha", 0.05)))
    # EMA cadence correction: applying the per-step alpha only every
    # ``update_every`` steps would slow the EMA; use the matched decay.
    ema_alpha_eff = 1.0 - (1.0 - ema_alpha) ** update_every

    # v3 arm (docs/V3_PREREGISTRATION.md).  When absent every line below that
    # tests ``scheme is None`` takes the frozen v2 path unchanged.
    v3cfg = cfg.get("v3", {}) or {}
    scheme = fam.scheme_from_config(v3cfg)
    v3_operator = v3cfg.get("operator", "none") if scheme is not None else "none"
    v3_rho = float(v3cfg.get("rho", 0.85))
    v3_p_max = float(v3cfg.get("p_max", 0.05))
    v3_stride = int(v3cfg.get("fr_stride", 500))
    v3_clone_policy = v3cfg.get("clone_policy", "exact")
    v3_hold_steps = int(v3cfg.get("hold_steps", 500))
    v3_oracle_target = bool(v3cfg.get("oracle_target", False))
    # The v3 window is owned by the v3 block, NOT by the RunSpec.  io_utils
    # hardcodes burnin_fraction=0 / stop_fraction=1 for the ``abf_only`` method,
    # so reading the window from the spec silently gave every v3 arm the whole
    # run instead of [0.2T, 0.8T].
    v3_burnin_fraction = float(v3cfg.get("burnin_fraction", 0.2))
    v3_stop_fraction = float(v3cfg.get("stop_fraction", 0.8))
    if not 0.0 <= v3_burnin_fraction <= v3_stop_fraction <= 1.0:
        raise ValueError("v3 window must satisfy 0 <= burnin <= stop <= 1")
    if v3_clone_policy not in {"exact", "holdout", "oracle_refresh"}:
        raise ValueError(f"unknown v3.clone_policy: {v3_clone_policy!r}")

    ramp_fraction = float(fr.get("ramp_fraction", 0.1))
    score_clip = fr.get("score_clip", 5.0)
    max_event_fraction = fr.get("max_event_fraction", 0.10)
    jitter = float(fr.get("jitter", 0.0))
    interval_scaled_clock = bool(fr.get("interval_scaled_clock", False))

    # --- clean-v2 (docs/CLEAN_V2_PREREGISTRATION.md) -------------------------
    # ``from_config`` validates first, and it *rejects* rather than defaults:
    # a config carrying score_clip / max_event_fraction / target_ema_alpha, or a
    # v3/v4 block, cannot reach this line.  The assignments below therefore
    # restate a guarantee rather than establish one -- but they make the retired
    # knobs unreachable from the clean path even if the config gate is edited.
    clean = cv2.from_config(cfg)
    if clean is not None:
        assert scheme is None, "clean-v2 and the v3 bias family are exclusive"
        if target_type not in cv2.TARGETS:
            raise ValueError(
                f"clean_v2 admits only {list(cv2.TARGETS)}; got {target_type!r}")
        score_clip = None
        max_event_fraction = None
        jitter = 0.0
        ramp_fraction = 0.0     # consumed by ramp_steps below -> gamma_eff = gamma
        interval_scaled_clock = True

    noise_chunk_steps = int(fr.get("noise_chunk_steps", 1024))
    observation_order = abf.get("observation_order", "pre_propagation")
    if observation_order not in {"pre_propagation", "post_propagation"}:
        raise ValueError(
            "abf.observation_order must be pre_propagation or post_propagation")
    x_tilt = float(cfg.get("potential", {}).get("x_tilt", 0.0))
    x_init_mode = sim.get("x_init_mode", "mixed")
    y_init_mode = sim.get("y_init_mode", "mixed")
    x_barrier = float(getattr(ev, "x_barrier", 0.0))

    B = len(specs)
    G = len(x_grid)
    x_grid_t = torch.as_tensor(np.asarray(x_grid), device=device, dtype=dtype)
    x0 = float(x_grid[0]); dx = tu.grid_spacing(x_grid_t)
    idx0 = int(np.argmin(np.abs(np.asarray(x_grid))))
    F_ref_t = torch.as_tensor(np.asarray(F_ref), device=device,
                              dtype=dtype).view(1, G)

    fr_enabled = (target_type != "none")
    fr_burnin = int(round(burnin_fraction * n_steps))
    fr_stop = int(round(stop_fraction * n_steps))
    ramp_steps = int(round(ramp_fraction * n_steps))
    gamma_vec = torch.as_tensor([float(s.gamma) for s in specs], device=device,
                                dtype=dtype)

    # Smoothing kernels for the binned-smooth estimator.
    k_h, r_h = tu.gaussian_kernel1d(h, dx, device, dtype)
    k_eta, r_eta = tu.gaussian_kernel1d(eta, dx, device, dtype)
    use_kernel_ref = (estimator == "kernel_reference")

    X, Y = _init_batch_positions(specs, n_particles, domain, x_init_mode,
                                 y_init_mode, device, dtype)
    noise_bank = _MatchedNoiseBank(
        specs, n_particles, n_steps, device, dtype, base_seed,
        chunk_steps=noise_chunk_steps)
    fr_generators = [
        tu.make_generator(tu.stable_seed("fr", base_seed, s.run_id), device)
        for s in specs
    ]
    # Appendix A.6: a third, independent stream.  The MD bank stays keyed by
    # (seed, step, slot) alone, so FR changes which configuration occupies a
    # slot but never which Langevin variates that slot will receive.
    oracle_generators = [
        tu.make_generator(tu.stable_seed("oracle", base_seed, s.run_id), device)
        for s in specs
    ]
    # --- v4-A mass sidecar ---------------------------------------------------
    # The bias comes from the v3 block, so an arm's bias is identical to its
    # control's by construction.  This block adds only the mass process.
    v4cfg = cfg.get("v4", {}) or {}
    v4_on = bool(v4cfg.get("enabled", False))
    v4_arm = int(v4cfg.get("arm", 4))            # 3 = never resample, 4 = triggered
    v4_theta = float(v4cfg.get("theta", 1.0))
    v4_rho = float(v4cfg.get("rho_resample", 0.5))
    v4_hold = int(v4cfg.get("hold_steps", 500))
    if v4_on and v3_operator != "none":
        raise ValueError("v4-A replaces the v3 FR operator; set v3.operator=none")
    if v4_on and scheme is None:
        raise ValueError("v4-A needs the v3 block for its bias family")
    if v4_on and clean is not None:
        raise ValueError("clean-v2 replaces the v4-A mass sidecar")
    masses = [pmass.PersistentMass(n_particles, device=device, dtype=dtype)
              for _ in range(B)] if v4_on else []
    v4_rows = []

    v3_burn = int(round(v3_burnin_fraction * n_steps))
    v3_stop = int(round(v3_stop_fraction * n_steps))
    hold = torch.zeros((B, n_particles), device=device, dtype=torch.long)
    held_out_active = ((scheme is not None and v3_clone_policy == "holdout")
                       or bool(cfg.get("v4", {}) and cfg["v4"].get("enabled")))
    v3_opportunities = []          # asserted as a whole array by the A.3 gate
    v3_rows = []                   # Amendment 4c per-opportunity diagnostics
    ancestors = torch.arange(
        n_particles, device=device).expand(B, n_particles).contiguous()
    noise_scale = float(np.sqrt(2.0 * dt / beta))

    oracle_refresh_y = None
    if scheme is not None and v3_clone_policy == "oracle_refresh":
        # Grid inverse-CDF for pi(y | x) prop exp(-beta V(x, y)).  The x-tilt is
        # a function of x alone, so it cancels in the conditional.
        ny_cond = int(v3cfg.get("oracle_ny", 401))
        y_cond = torch.linspace(ymin, ymax, ny_cond, device=device, dtype=dtype)
        logw = -beta * potentials.potential_xy_torch(
            x_grid_t.view(G, 1).expand(G, ny_cond),
            y_cond.view(1, ny_cond).expand(G, ny_cond))
        logw = logw - logw.max(dim=1, keepdim=True).values
        w_cond = torch.exp(logw)
        cdf_cond = torch.cumsum(w_cond, dim=1)
        cdf_cond = (cdf_cond / cdf_cond[:, -1:].clamp_min(EPS)).contiguous()

        def oracle_refresh_y(x_children, generator):
            ix = tu.nearest_index(x_children.view(1, -1), x0, dx, G).view(-1)
            u = torch.rand(x_children.numel(), generator=generator,
                           device=device, dtype=dtype)
            j = torch.searchsorted(
                cdf_cond[ix], u.unsqueeze(1)).view(-1).clamp_max(ny_cond - 1)
            return y_cond[j]

    # ABF accumulators and current grid estimates.
    C_acc = torch.zeros((B, G), device=device, dtype=dtype)
    S_acc = torch.zeros((B, G), device=device, dtype=dtype)
    Fprime_hat = torch.zeros((B, G), device=device, dtype=dtype)
    F_hat = torch.zeros((B, G), device=device, dtype=dtype)
    Fhat_target = torch.zeros((B, G), device=device, dtype=dtype)

    barrier_crossings = torch.zeros(B, device=device, dtype=torch.long)
    prev_sign = torch.sign(X - x_barrier)

    # Windowed FR accumulators (reset every snapshot), all (B,) device tensors.
    win_n = torch.zeros(B, device=device, dtype=torch.long)
    win_events = torch.zeros(B, device=device, dtype=torch.long)
    win_frac_sum = torch.zeros(B, device=device, dtype=dtype)
    win_frac_max = torch.zeros(B, device=device, dtype=dtype)
    win_smean_sum = torch.zeros(B, device=device, dtype=dtype)
    win_sstd_sum = torch.zeros(B, device=device, dtype=dtype)
    win_smin = torch.full((B,), float("inf"), device=device, dtype=dtype)
    win_smax = torch.full((B,), float("-inf"), device=device, dtype=dtype)
    q_levels = torch.tensor(
        SCORE_QUANTILE_LEVELS, device=device, dtype=dtype)
    win_sq_raw = torch.zeros((B, q_levels.numel()), device=device, dtype=dtype)
    win_sq_app = torch.zeros((B, q_levels.numel()), device=device, dtype=dtype)
    win_clip_sum = torch.zeros(B, device=device, dtype=dtype)
    win_target_l2 = torch.zeros(B, device=device, dtype=dtype)
    cumulative_fr_events = torch.zeros(B, device=device, dtype=torch.long)
    cumulative_replacements = torch.zeros(B, device=device, dtype=torch.long)

    diags = [_empty_diag(target_type) for _ in range(B)]

    # One-slot cell holding the applied bias-force grid, refreshed with the
    # estimator.  Starts at zero exactly as the v2 path does, since F_hat is
    # zero before the first update and every family multiplier is finite.
    nonlocal_bias = [torch.zeros((B, G), device=device, dtype=dtype)]

    def recompute_grid():
        nonlocal Fprime_hat, F_hat
        if use_kernel_ref:
            # S_acc / C_acc hold the exact kernel-accumulated numerator
            # (force-weighted) and denominator (weights); no smoothing needed.
            Fprime_hat = S_acc / (C_acc + min_count + EPS)
        else:
            num_s = tu.smooth_grid(S_acc, k_h, r_h, dx)
            den_s = tu.smooth_grid(C_acc, k_h, r_h, dx)
            Fprime_hat = num_s / (den_s + min_count + EPS)
        F_hat = tu.center_at_index(tu.cumulative_trapezoid(Fprime_hat, dx), idx0)
        # v3 carrier: A_t is exactly this F_hat, and the applied bias force is
        # A' * (1 - g'(A)).  The multiplier comes from the Scheme; no family
        # formula is written in this module.
        nonlocal_bias[0] = (
            Fprime_hat if scheme is None
            else Fprime_hat * scheme.force_family.bias_force_multiplier(F_hat, beta))

    def current_p_hat(Xc):
        if use_kernel_ref:
            p = _kde_reflected(x_grid_t, Xc, eta, xmin, xmax)
        else:
            hist = tu.scatter_grid(tu.nearest_index(Xc, x0, dx, G), G)
            p = tu.smooth_grid(hist, k_eta, r_eta, dx) / n_particles
        return tu.normalize_density(p, dx)

    def carrier_A():
        """``A_t`` for the clean-v2 target: the running ABF estimate itself.

        ``physical_oracle`` substitutes ``F_ref``.  That arm is a diagnostic --
        it answers "is A_t already accurate enough by t_burn to build the
        physical target?" -- and is never a candidate method.
        """
        return F_ref_t.expand(B, G) if target_type == "physical_oracle" else F_hat

    def render_target(p_snap):
        """``q_t`` on the profile grid, for diagnostics and figures only.

        The clean-v2 algorithm never evaluates a normalised target: it works
        from ``log phat + beta A`` at particle positions.  For ``abf_only`` this
        renders the target the baseline's own estimate would imply, which is
        what makes the mechanism panel comparable across arms; nothing consumes
        it.
        """
        if clean is not None:
            return cv2.target_grid(carrier_A(), beta, dx)
        if scheme is not None and scheme.target_family is not None:
            return (fam.oracle_target(F_ref_t.expand(B, G), F_hat,
                                      scheme.target_family, beta, dx)
                    if v3_oracle_target
                    else scheme.target_family.target(F_hat, beta, dx))
        return _build_target(target_type, Fhat_target, F_hat, F_ref_t,
                             p_snap, beta, dx, width)

    clean_rows = []
    p_initial = current_p_hat(X)
    q_initial = render_target(p_initial)
    zero_gamma = torch.zeros_like(gamma_vec)
    _record_snapshot(
        diags, 0, 0.0, Fprime_hat, F_hat, p_initial, q_initial,
        X, Y, ancestors, barrier_crossings, zero_gamma,
        win_n, win_events, win_frac_sum, win_frac_max, win_smean_sum,
        win_sstd_sum, win_smin, win_smax, win_sq_raw, win_sq_app,
        win_clip_sum, win_target_l2,
        cumulative_fr_events, cumulative_replacements, n_particles)

    t0 = time.time()
    for step in range(n_steps):
        next_step = step + 1
        dvdx = potentials.dVdx_xy_torch(X, Y) + x_tilt
        dvdy = potentials.dVdy_xy_torch(X, Y)

        if observation_order == "pre_propagation":
            if use_kernel_ref:
                num_c, den_c = _kernel_estimator(
                    x_grid_t, X, Y, h, x_tilt=x_tilt)
                S_acc += num_c
                C_acc += den_c
            else:
                idx = tu.nearest_index(X, x0, dx, G)
                C_acc += tu.scatter_grid(idx, G)
                S_acc += tu.scatter_grid(idx, G, dvdx)
            if step % update_every == 0:
                recompute_grid()
                if clean is None:       # clean-v2 targets A_t itself, not an EMA
                    Fhat_target = (
                        (1.0 - ema_alpha_eff) * Fhat_target
                        + ema_alpha_eff * F_hat)

        bias_grid = Fprime_hat if scheme is None else nonlocal_bias[0]
        abf_at_X = tu.interp1d(bias_grid, X, x0, dx)
        noise_x, noise_y = noise_bank.at(step)
        X_prop = tu.reflect_into(
            X + (-dvdx + abf_at_X) * dt + noise_scale * noise_x,
            xmin, xmax)
        Y_prop = tu.reflect_into(
            Y + (-dvdy) * dt + noise_scale * noise_y,
            ymin, ymax)
        tu.assert_finite("X_prop", X_prop)
        tu.assert_finite("Y_prop", Y_prop)

        new_sign = torch.sign(X_prop - x_barrier)
        crossed = (new_sign != prev_sign) & (new_sign != 0) & (prev_sign != 0)
        barrier_crossings += crossed.sum(dim=1)

        if observation_order == "post_propagation":
            if use_kernel_ref:
                num_c, den_c = _kernel_estimator(
                    x_grid_t, X_prop, Y_prop, h, x_tilt=x_tilt)
                S_acc += num_c
                C_acc += den_c
            else:
                dvdx_prop = potentials.dVdx_xy_torch(X_prop, Y_prop) + x_tilt
                idx = tu.nearest_index(X_prop, x0, dx, G)
                if held_out_active:
                    # A.4: a held-out replica propagates normally but deposits
                    # nothing.  FR never touches the accumulators directly; it
                    # only changes which configurations are eligible to speak.
                    elig = (hold == 0).to(dtype)
                    C_acc += tu.scatter_grid(idx, G, elig)
                    S_acc += tu.scatter_grid(idx, G, dvdx_prop * elig)
                else:
                    C_acc += tu.scatter_grid(idx, G)
                    S_acc += tu.scatter_grid(idx, G, dvdx_prop)
            if next_step % update_every == 0:
                recompute_grid()
                if clean is None:       # clean-v2 targets A_t itself, not an EMA
                    Fhat_target = (
                        (1.0 - ema_alpha_eff) * Fhat_target
                        + ema_alpha_eff * F_hat)

        in_window = (
            fr_enabled and fr_burnin <= next_step < fr_stop)
        do_fr = (
            in_window
            and ((next_step - fr_burnin) % fr_every == 0))
        if in_window:
            if ramp_steps <= 0:
                gamma_eff = gamma_vec
            else:
                s_ramp = max(
                    (next_step - fr_burnin) / ramp_steps, 0.0)
                gamma_eff = gamma_vec * (1.0 - np.exp(-s_ramp))
        else:
            gamma_eff = torch.zeros_like(gamma_vec)

        do_fr_v4 = (
            v4_on
            and v3_burn <= next_step <= v3_stop
            and ((next_step - v3_burn) % v3_stride == 0))
        if do_fr_v4:
            # Oracle target: q propto exp(-beta(F_ref + B_t)) (v4-A holds the
            # target fixed at the oracle so target error cannot excuse an
            # operator failure).
            beta_B = scheme.force_family.beta_bias_potential(F_hat, beta)
            log_q_grid = -(beta * F_ref_t.expand(B, G) + beta_B)
            log_q_grid = log_q_grid - log_q_grid.max(dim=1, keepdim=True).values
            X_new, Y_new = X_prop.clone(), Y_prop.clone()
            anc_new, hold_new = ancestors.clone(), hold.clone()
            for b in range(B):
                zb = X_prop[b]
                m = masses[b]
                log_q_at = tu.interp1d(log_q_grid[b:b + 1], zb.view(1, -1),
                                       x0, dx).view(-1)
                log_p_at = m.log_density_at(zb, eta)
                ess_before = m.ess()
                # MASS ONLY: no positions move, no accumulator is touched.
                m.fr_update(log_q_at, log_p_at, theta=v4_theta)
                ess_after = m.ess()
                row = dict(step=int(next_step), row=b,
                           ess_w_before=ess_before / n_particles,
                           ess_w_after=ess_after / n_particles,
                           w_max_after=m.w_max(),
                           log_w_span=float(m.log_w.max() - m.log_w.min()),
                           would_resample=bool(m.needs_resample(v4_rho)),
                           resampled=False, n_replacements=0)
                ess_anc_mass, m_max = m.mass_ancestry(anc_new[b])
                row.update(ess_anc_mass=ess_anc_mass / n_particles, m_max=m_max)
                # Arm 3 never resamples, whatever the degeneracy (frozen).
                if v4_arm == 4 and m.needs_resample(v4_rho):
                    probe = torch.tensor([-1.05, 0.0, 1.0], dtype=dtype,
                                         device=device)
                    row["fibre_ess_before"] = float(
                        m.fibre_ess(zb, probe, h).min())
                    xb = zb.detach().cpu().numpy()
                    yb = Y_prop[b].detach().cpu().numpy()
                    row.update({f"pre_{k}": v for k, v in
                                fib.fibre_report(xb, yb, beta).items()})
                    r = rep.resample(m.weights, fr_generators[b])
                    X_new[b] = X_prop[b][r.src]
                    Y_new[b] = Y_prop[b][r.src]
                    anc_new[b] = ancestors[b][r.src]
                    hold_new[b] = rep.apply_holdout(hold[b], r.src, r.is_clone,
                                                    v4_hold)
                    m.take_indices(r.src)
                    m.reset_uniform()
                    row.update(resampled=True, n_replacements=r.n_replacements)
                    row.update({f"post_{k}": v for k, v in
                                fib.fibre_report(X_new[b].detach().cpu().numpy(),
                                                 Y_new[b].detach().cpu().numpy(),
                                                 beta).items()})
                ess_c, c_max = rep.count_ancestry(anc_new[b], n_particles)
                row.update(ess_anc_count=ess_c / n_particles, c_max=c_max)
                v4_rows.append(row)
            X, Y, ancestors, hold = X_new, Y_new, anc_new, hold_new
            cumulative_fr_events += torch.ones(B, device=device, dtype=torch.long)

        # v3 opportunities: Appendix A.3 includes BOTH endpoints of the window.
        do_fr_v3 = (
            v3_operator != "none"
            and v3_burn <= next_step <= v3_stop
            and ((next_step - v3_burn) % v3_stride == 0))
        if do_fr_v3:
            v3_opportunities.append(int(next_step))
            p_hat = current_p_hat(X_prop)
            if v3_oracle_target:
                q_grid = fam.oracle_target(
                    F_ref_t.expand(B, G), F_hat, scheme.target_family, beta, dx)
            else:
                q_grid = scheme.target_family.target(F_hat, beta, dx)
            log_p_part = torch.log(
                tu.interp1d(p_hat, X_prop, x0, dx).clamp_min(EPS))
            log_q_part = torch.log(
                tu.interp1d(q_grid, X_prop, x0, dx).clamp_min(EPS))

            kl_before = _kl_grid(p_hat, q_grid, dx)
            ess_anc_b, wmax_b = _anc_stats(ancestors, n_particles)

            X_new, Y_new = X_prop.clone(), Y_prop.clone()
            anc_new, hold_new = ancestors.clone(), hold.clone()
            ndeath = torch.zeros(B, device=device, dtype=torch.long)
            S_all = torch.zeros_like(X_prop)
            th_b = torch.zeros(B, device=device, dtype=dtype)
            essw_b = torch.zeros(B, device=device, dtype=dtype)
            dtau_b = torch.zeros(B, device=device, dtype=dtype)
            q90_b = torch.zeros(B, device=device, dtype=dtype)
            pev_mean_b = torch.zeros(B, device=device, dtype=dtype)
            pev_max_b = torch.zeros(B, device=device, dtype=dtype)
            for b in range(B):
                score = fr_v3.FRScore(log_p=log_p_part[b], log_q=log_q_part[b])
                S_all[b] = score.S
                if v3_operator == "bd":
                    dtau = fr_v3.bd_timestep(score, v3_p_max)
                    src, _ = fr_v3.bd_standard(score, dtau, fr_generators[b])
                    p_event = 1.0 - torch.exp(-score.S.abs() * dtau)
                    dtau_b[b] = dtau
                    q90_b[b] = torch.quantile(score.S.abs(), 0.90)
                    pev_mean_b[b] = p_event.mean()
                    pev_max_b[b] = p_event.max()
                else:
                    src, theta_b, essw = fr_v3.ft_step(
                        score, v3_rho, fr_generators[b])
                    th_b[b] = theta_b
                    essw_b[b] = essw / n_particles
                is_clone = fr_v3.clone_mask(src)
                X_new[b] = X_prop[b][src]
                Y_new[b] = Y_prop[b][src]
                anc_new[b] = ancestors[b][src]
                hold_new[b] = (
                    fr_v3.apply_holdout(hold[b], src, is_clone, v3_hold_steps)
                    if v3_clone_policy == "holdout" else hold[b][src])
                if v3_clone_policy == "oracle_refresh" and bool(is_clone.any()):
                    # A.5: the fibre coordinate only.  x, ancestry and the
                    # offspring count are left exactly as FR set them.
                    Y_new[b][is_clone] = oracle_refresh_y(
                        X_new[b][is_clone], oracle_generators[b])
                ndeath[b] = fr_v3.replacement_count(src, n_particles)
            X, Y, ancestors, hold = X_new, Y_new, anc_new, hold_new

            kl_after = _kl_grid(current_p_hat(X), q_grid, dx)
            ess_anc_a, wmax_a = _anc_stats(ancestors, n_particles)
            # Amendment 4c: carrier error and the exact consistency residue.
            F_ref_b = F_ref_t.expand(B, G)
            gauge = (F_hat - F_ref_b).mean(dim=1, keepdim=True)
            carrier_err = torch.sqrt(
                tu.trapezoid((F_hat - F_ref_b - gauge) ** 2, dx).clamp_min(0.0))
            beta_B = scheme.force_family.beta_bias_potential(F_hat, beta)
            p_star = fam.stationary_marginal(F_ref_b, beta_B, beta, dx)
            dcons = _kl_grid(p_star, q_grid, dx)
            v3_rows.append(dict(
                step=int(next_step),
                theta=th_b.detach().cpu().numpy().copy(),
                ess_w=essw_b.detach().cpu().numpy().copy(),
                dtau=dtau_b.detach().cpu().numpy().copy(),
                q90=q90_b.detach().cpu().numpy().copy(),
                pev_mean=pev_mean_b.detach().cpu().numpy().copy(),
                pev_max=pev_max_b.detach().cpu().numpy().copy(),
                repl=ndeath.detach().cpu().numpy().copy(),
                kl_before=kl_before.detach().cpu().numpy().copy(),
                kl_after=kl_after.detach().cpu().numpy().copy(),
                ess_anc_before=ess_anc_b.detach().cpu().numpy().copy(),
                ess_anc_after=ess_anc_a.detach().cpu().numpy().copy(),
                wmax_before=wmax_b.detach().cpu().numpy().copy(),
                wmax_after=wmax_a.detach().cpu().numpy().copy(),
                carrier_err=carrier_err.detach().cpu().numpy().copy(),
                dcons=dcons.detach().cpu().numpy().copy()))

            ones = torch.ones(B, device=device, dtype=dtype)
            frac = ndeath.to(dtype) / n_particles
            win_n += torch.ones(B, device=device, dtype=torch.long)
            win_events += ndeath
            win_frac_sum += frac
            win_frac_max = torch.maximum(win_frac_max, frac)
            win_smean_sum += S_all.mean(dim=1)
            win_sstd_sum += S_all.std(dim=1)
            win_smin = torch.minimum(win_smin, S_all.amin(dim=1))
            win_smax = torch.maximum(win_smax, S_all.amax(dim=1))
            win_sq_raw += _score_quantiles(S_all, q_levels)
            win_sq_app += _score_quantiles(S_all, q_levels)   # v3 never clips
            win_target_l2 += _marginal_l2(p_hat, q_grid, dx) * ones
            cumulative_fr_events += torch.ones(B, device=device,
                                               dtype=torch.long)
            cumulative_replacements += ndeath
        elif scheme is not None:
            X, Y = X_prop, Y_prop

        active = gamma_eff > 0
        if clean is not None:
            # ---- clean-v2 pulse: physical target, standard birth--death ----
            # Order matters and is frozen: the ABF accumulators were already
            # fed from (X_prop, Y_prop) above, so a replica created here
            # deposits nothing at this step.  A clone is not an observation; it
            # first speaks after its next physical propagation (Gate B).
            if do_fr and bool(active.any().detach().cpu()):
                p_hat = current_p_hat(X_prop)
                A_grid = carrier_A()
                S_all, log_p_at, log_q_at, floored = cv2.score(
                    p_hat, A_grid, X_prop, x0, dx, beta)
                q_grid = cv2.target_grid(A_grid, beta, dx)   # diagnostics only
                kl_before = _kl_grid(p_hat, q_grid, dx)
                ess_anc_b, wmax_b = _anc_stats(ancestors, n_particles)

                X_new, Y_new = X_prop.clone(), Y_prop.clone()
                anc_new = ancestors.clone()
                ndeath = torch.zeros(B, device=device, dtype=torch.long)
                dtau_b = torch.zeros(B, device=device, dtype=dtype)
                pev_mean_b = torch.zeros(B, device=device, dtype=dtype)
                pev_max_b = torch.zeros(B, device=device, dtype=dtype)
                gamma_list = gamma_eff.detach().cpu().tolist()
                active_list = active.detach().cpu().tolist()
                for b in range(B):
                    if not active_list[b]:
                        continue
                    dtau_eff = cv2.dtau(gamma_list[b], fr_every, dt)
                    p_event = cv2.event_probability(S_all[b], dtau_eff)
                    dtau_b[b] = dtau_eff
                    pev_mean_b[b] = p_event.mean()
                    pev_max_b[b] = p_event.max()
                    # The operator is fr_v3.bd_standard, unchanged: uncapped
                    # probability, uniformly chosen partner, fixed population.
                    src, _ = fr_v3.bd_standard(
                        cv2.row_score(log_p_at[b], log_q_at[b]), dtau_eff,
                        fr_generators[b])
                    X_new[b] = X_prop[b][src]
                    Y_new[b] = Y_prop[b][src]
                    anc_new[b] = ancestors[b][src]
                    ndeath[b] = fr_v3.replacement_count(src, n_particles)
                X, Y, ancestors = X_new, Y_new, anc_new

                kl_after = _kl_grid(current_p_hat(X), q_grid, dx)
                ess_anc_a, wmax_a = _anc_stats(ancestors, n_particles)
                active_i = active.to(torch.long)
                active_f = active.to(dtype)
                frac = ndeath.to(dtype) / n_particles
                s_min = S_all.amin(dim=1)
                s_max = S_all.amax(dim=1)
                sq = _score_quantiles(S_all, q_levels)
                win_n += active_i
                win_events += ndeath
                win_frac_sum += frac
                win_frac_max = torch.maximum(win_frac_max, frac)
                win_smean_sum += S_all.mean(dim=1) * active_f
                win_sstd_sum += S_all.std(dim=1) * active_f
                win_smin = torch.where(
                    active, torch.minimum(win_smin, s_min), win_smin)
                win_smax = torch.where(
                    active, torch.maximum(win_smax, s_max), win_smax)
                # Gate D: raw and applied are the *same* tensor, not two
                # tensors that happen to agree, and win_clip_sum is left at
                # zero because nothing clips.
                win_sq_raw += sq * active_f.unsqueeze(1)
                win_sq_app += sq * active_f.unsqueeze(1)
                win_target_l2 += _marginal_l2(p_hat, q_grid, dx) * active_f
                cumulative_fr_events += active_i
                cumulative_replacements += ndeath

                def _np(t):
                    return t.detach().cpu().numpy().copy()

                row = dict(
                    step=int(next_step), t=float(next_step * dt),
                    dtau=_np(dtau_b), p_event_mean=_np(pev_mean_b),
                    p_event_max=_np(pev_max_b), repl=_np(ndeath),
                    event_fraction=_np(frac),
                    s_min=_np(s_min), s_max=_np(s_max),
                    s_absmax=_np(S_all.abs().amax(dim=1)),
                    kl_before=_np(kl_before), kl_after=_np(kl_after),
                    ess_anc_before=_np(ess_anc_b), ess_anc_after=_np(ess_anc_a),
                    wmax_before=_np(wmax_b), wmax_after=_np(wmax_a),
                    logp_floored_fraction=_np(floored))
                for j, lab in enumerate(SCORE_QUANTILE_LABELS):
                    row[f"s_{lab}"] = _np(sq[:, j])
                clean_rows.append(row)
            else:
                X, Y = X_prop, Y_prop
        elif scheme is not None:
            pass                        # v3 arms never take the v2 FR path
        elif do_fr and bool(active.any().detach().cpu()):
            p_hat = current_p_hat(X_prop)
            q_grid = _build_target(
                target_type, Fhat_target, F_hat, F_ref_t,
                p_hat, beta, dx, width)
            S, S_raw = _fr_score(
                X_prop, p_hat, q_grid, x0, dx, beta, score_clip)
            clock_dt = dt * fr_every if interval_scaled_clock else dt
            X, Y, ancestors, ndeath = _resample(
                X_prop, Y_prop, ancestors, S, gamma_eff, clock_dt,
                max_event_fraction, fr_generators, jitter, noise_scale)

            active_i = active.to(torch.long)
            active_f = active.to(dtype)
            frac = ndeath.to(dtype) / n_particles
            win_n += active_i
            win_events += ndeath
            win_frac_sum += frac
            win_frac_max = torch.maximum(win_frac_max, frac)
            win_smean_sum += S.mean(dim=1) * active_f
            win_sstd_sum += S.std(dim=1) * active_f
            win_smin = torch.where(
                active, torch.minimum(win_smin, S.amin(dim=1)), win_smin)
            win_smax = torch.where(
                active, torch.maximum(win_smax, S.amax(dim=1)), win_smax)
            win_sq_raw += _score_quantiles(
                S_raw, q_levels) * active_f.unsqueeze(1)
            win_sq_app += _score_quantiles(
                S, q_levels) * active_f.unsqueeze(1)
            if score_clip is not None:
                win_clip_sum += (
                    S_raw.abs() > float(score_clip)
                ).to(dtype).mean(dim=1) * active_f
            win_target_l2 += _marginal_l2(
                p_hat, q_grid, dx) * active_f
            cumulative_fr_events += active_i
            cumulative_replacements += ndeath
        elif scheme is None and not v4_on:
            X, Y = X_prop, Y_prop
        elif v4_on and not do_fr_v4:
            X, Y = X_prop, Y_prop

        if held_out_active:
            hold = (hold - 1).clamp_min(0)     # A.4: decrement after depositing

        prev_sign = torch.sign(X - x_barrier)

        if next_step % eval_every == 0 or next_step == n_steps:
            if observation_order == "pre_propagation":
                recompute_grid()
            p_snap = current_p_hat(X)
            q_snap = render_target(p_snap)
            _record_snapshot(
                diags, next_step, next_step * dt, Fprime_hat, F_hat,
                p_snap, q_snap, X, Y, ancestors, barrier_crossings,
                gamma_eff, win_n, win_events, win_frac_sum, win_frac_max,
                win_smean_sum, win_sstd_sum, win_smin, win_smax,
                win_sq_raw, win_sq_app, win_clip_sum,
                win_target_l2, cumulative_fr_events,
                cumulative_replacements, n_particles)
            win_n = torch.zeros(B, device=device, dtype=torch.long)
            win_events = torch.zeros(B, device=device, dtype=torch.long)
            win_frac_sum = torch.zeros(B, device=device, dtype=dtype)
            win_frac_max = torch.zeros(B, device=device, dtype=dtype)
            win_smean_sum = torch.zeros(B, device=device, dtype=dtype)
            win_sstd_sum = torch.zeros(B, device=device, dtype=dtype)
            win_smin = torch.full(
                (B,), float("inf"), device=device, dtype=dtype)
            win_smax = torch.full(
                (B,), float("-inf"), device=device, dtype=dtype)
            win_sq_raw = torch.zeros(
                (B, q_levels.numel()), device=device, dtype=dtype)
            win_sq_app = torch.zeros(
                (B, q_levels.numel()), device=device, dtype=dtype)
            win_clip_sum = torch.zeros(B, device=device, dtype=dtype)
            win_target_l2 = torch.zeros(B, device=device, dtype=dtype)

    if device.type == "cuda":
        torch.cuda.synchronize()
    runtime = time.time() - t0

    for d in diags:
        d["fr_burnin"] = fr_burnin
        d["fr_stop"] = fr_stop
        d["fr_every"] = int(fr_every)
        d["n_steps"] = int(n_steps)
        d["dt"] = float(dt)
        d["observation_order"] = observation_order
        d["interval_scaled_clock"] = interval_scaled_clock
        d["x_tilt"] = x_tilt
        d["clean_v2"] = clean is not None
        d["score_clip"] = score_clip
        d["max_event_fraction"] = max_event_fraction
        if clean is not None:
            # The schedule as *specified*, so a gate can compare it with the
            # opportunities the engine actually took rather than trust either.
            # ``abf_only`` has no FR schedule at all -- io_utils gives it the
            # whole run at stride 1, and materialising that as a list of every
            # step would advertise a schedule the arm does not have.
            d["fr_firing_steps"] = (
                cv2.firing_steps(n_steps, burnin_fraction, stop_fraction,
                                 fr_every) if fr_enabled else [])
    return BatchResult(diags, runtime, str(device),
                       fr_opportunities=v3_opportunities,
                       v3_events=v3_rows, v4_events=v4_rows,
                       clean_events=clean_rows)


def _kl_grid(p_hat, q_grid, dx):
    """KL(p || q) on the profile grid, per row."""
    p = p_hat.clamp_min(EPS)
    q = q_grid.clamp_min(EPS)
    return tu.trapezoid(p * (torch.log(p) - torch.log(q)), dx)


def _anc_stats(ancestors, n_particles):
    """(ESS_anc / K, w_max) per row -- the genealogy pair, aggregated by ancestor.

    Amendment 4a: this is a different object from the weight ESS the governor
    controls, and rho places no bound on it.
    """
    B = ancestors.shape[0]
    ess = torch.zeros(B, device=ancestors.device, dtype=torch.float64)
    wmax = torch.zeros(B, device=ancestors.device, dtype=torch.float64)
    for b in range(B):
        c = torch.bincount(ancestors[b], minlength=n_particles).double()
        w = c / float(n_particles)
        ess[b] = 1.0 / (w * w).sum().clamp_min(EPS)
        wmax[b] = w.max()
    return ess / float(n_particles), wmax


def _marginal_l2(p_hat, q_grid, dx):
    """L2 distance between (renormalised) ``p_hat`` and ``q`` per row -> (B,)."""
    p = p_hat / tu.trapezoid(p_hat, dx).clamp_min(EPS).unsqueeze(1)
    q = q_grid / tu.trapezoid(q_grid, dx).clamp_min(EPS).unsqueeze(1)
    width = (p.shape[1] - 1) * dx
    return torch.sqrt((tu.trapezoid((p - q) ** 2, dx) / max(width, EPS)).clamp_min(0.0))


def _record_snapshot(diags, step, t, Fprime_hat, F_hat, p_snap, q_snap, X, Y,
                     ancestors, barrier_crossings, gamma_eff, win_n, win_events,
                     win_frac_sum, win_frac_max, win_smean_sum, win_sstd_sum,
                     win_smin, win_smax, win_sq_raw, win_sq_app, win_clip_sum,
                     win_target_l2, cumulative_fr_events,
                     cumulative_replacements, n_particles):
    """Move per-row snapshot data to numpy and append to each run's diag.

    This is the *only* GPU->CPU transfer inside the time loop and it happens at
    ``eval_every`` cadence on grid-sized arrays plus the particle snapshot.
    """
    Fp = Fprime_hat.detach().cpu().numpy()
    F = F_hat.detach().cpu().numpy()
    P = p_snap.detach().cpu().numpy()
    Q = q_snap.detach().cpu().numpy()
    Xn = X.detach().cpu().numpy()
    Yn = Y.detach().cpu().numpy()
    anc = ancestors.detach().cpu().numpy()
    bc = barrier_crossings.detach().cpu().numpy()
    ge = gamma_eff.detach().cpu().numpy()
    wn = win_n.detach().cpu().numpy()
    we = win_events.detach().cpu().numpy()
    wfs = win_frac_sum.detach().cpu().numpy()
    wfm = win_frac_max.detach().cpu().numpy()
    wsm = win_smean_sum.detach().cpu().numpy()
    wss = win_sstd_sum.detach().cpu().numpy()
    wmin = win_smin.detach().cpu().numpy()
    wmax = win_smax.detach().cpu().numpy()
    wqr = win_sq_raw.detach().cpu().numpy()
    wqa = win_sq_app.detach().cpu().numpy()
    wcf = win_clip_sum.detach().cpu().numpy()
    wtl = win_target_l2.detach().cpu().numpy()
    cfe = cumulative_fr_events.detach().cpu().numpy()
    crep = cumulative_replacements.detach().cpu().numpy()

    for b, d in enumerate(diags):
        nfr = int(wn[b])
        d["steps"].append(int(step)); d["times"].append(float(t))
        d["Fprime_hat"].append(Fp[b].copy()); d["F_hat"].append(F[b].copy())
        d["p_hat_grid"].append(P[b].copy()); d["q_target_grid"].append(Q[b].copy())
        d["X_snap"].append(Xn[b].copy()); d["Y_snap"].append(Yn[b].copy())
        d["barrier_crossings"].append(int(bc[b]))
        anc_b = anc[b]
        counts = np.bincount(anc_b, minlength=n_particles).astype(np.float64)
        nz = counts[counts > 0]
        d["n_unique_ancestors"].append(int((counts > 0).sum()))
        d["ancestor_ess"].append(float(nz.sum() ** 2 / np.maximum((nz ** 2).sum(), 1.0)))
        d["max_clone_multiplicity"].append(int(counts.max()))
        d["max_clone_weight"].append(float(counts.max() / n_particles))
        d["gamma_eff"].append(float(ge[b]))
        d["fr_applied"].append(bool(nfr > 0))
        d["fr_event_fraction"].append(float(wfs[b] / nfr) if nfr else 0.0)
        d["fr_event_fraction_max"].append(float(wfm[b]) if nfr else 0.0)
        d["fr_events_total"].append(int(we[b]))
        d["score_mean"].append(float(wsm[b] / nfr) if nfr else float("nan"))
        d["score_std"].append(float(wss[b] / nfr) if nfr else float("nan"))
        d["score_min"].append(float(wmin[b]) if nfr else float("nan"))
        d["score_max"].append(float(wmax[b]) if nfr else float("nan"))
        for j, lab in enumerate(SCORE_QUANTILE_LABELS):
            d[f"score_raw_{lab}"].append(
                float(wqr[b, j] / nfr) if nfr else float("nan"))
            d[f"score_applied_{lab}"].append(
                float(wqa[b, j] / nfr) if nfr else float("nan"))
        d["score_clipped_fraction"].append(
            float(wcf[b] / nfr) if nfr else float("nan"))
        d["target_l2"].append(float(wtl[b] / nfr) if nfr else float("nan"))
        d["cumulative_fr_events"].append(int(cfe[b]))
        d["cumulative_replacements"].append(int(crep[b]))
