"""Serial one-walker ABF at equal force-evaluation budget (WCA closeout, Part H).

ADDITIVE companion to ``wca_followup_jobs.py`` / ``wca_phase_jobs.py``. It answers the
advisor/reviewer fairness concern behind the WCA study: an ``N``-replica ABF/mFR run at
time ``T`` spends ``N x`` the force evaluations of ordinary ABF at time ``T``. The fair
control is a **single-walker serial ABF run at the same total force-evaluation budget**

    budget = n_replicas * n_steps    (per physical M).

For the base parallel run ``N_parallel=1024, nsteps_base=120000`` the exact equal-budget
serial control is ``N_serial=1, nsteps_serial = 1024*120000 = 122,880,000`` steps.

Why a dedicated engine (not ``core.run_sampler_gpu``)
-----------------------------------------------------
``core.TorchKernelABFEstimator`` keeps ONE shared ``(n_grid,)`` histogram: all N replicas
of a run pool into a single ABF mean-force estimate. A serial control must instead be a set
of INDEPENDENT one-walker trajectories, each with its OWN ABF accumulators. To exploit the
H200 we batch ``G`` such trajectories (the seeds of one physics cell) in the leading tensor
dimension and give the estimator a ``(G, n_grid)`` accumulator -- G genuinely independent
one-walker ABF runs advancing together, never sharing force information. Everything else
(the WCA force engine, geometry, local mean force, ABF force, reaction-coordinate wall,
region/metric/reference code) is reused verbatim from ``wca_abffr_core`` so the control is
scientifically identical to ordinary ABF, only with N=1. There is NO Fisher-Rao / birth-death
/ oracle / adaptive machinery here.

Because a serial run is enormous the loop is chunked and CHECKPOINTED: a checkpoint holds the
current step, configuration ``q``, per-trajectory bias+production ABF accumulators, the RNG
state, the running diagnostics (crossings, region-time, z-marginal), and the budget-ladder
snapshot series, so an interrupted run resumes without losing ABF accumulators. Snapshots
taken along the single long run yield the budget ladder as a by-product. No particle
trajectories are ever written to disk.
"""
from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, asdict, field

import numpy as np
import torch

import wca_abffr_core as core
import wca_phase_jobs as pj

EPS = core.EPS
BASE_PARALLEL_REPLICAS = 1024
BASE_PARALLEL_STEPS = 120000
BASE_BUDGET = BASE_PARALLEL_REPLICAS * BASE_PARALLEL_STEPS   # 122,880,000
DEFAULT_LADDER_FRACS = (1.0 / 1024, 1.0 / 256, 1.0 / 64, 1.0 / 16, 1.0 / 4, 1.0)


# ---------------------------------------------------------------------------
# Spec: one serial one-walker trajectory (seed) of a physics cell.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SerialRunSpec:
    study: str
    stage: str
    method: str          # always "serial_abf"
    seed: int            # trajectory label
    n_steps: int         # target steps for THIS cell (budget = 1 * n_steps)
    n_replicas: int      # always 1 (a one-walker control)
    # ---- physics ----
    beta: float
    h: float
    w: float
    n_dim: int
    a: float
    sigma: float
    epsilon: float

    @property
    def M(self) -> int:
        return int(self.n_dim) * int(self.n_dim)

    @property
    def budget(self) -> int:
        return int(self.n_replicas) * int(self.n_steps)

    def physics_tag(self) -> str:
        return (f"b{self.beta:g}_h{self.h:g}_w{self.w:g}"
                f"_n{int(self.n_dim)}_a{self.a:g}")

    def spec_hash(self) -> str:
        return hashlib.md5(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]

    def run_id(self) -> str:
        return (f"{self.stage}__{self.method}__{self.physics_tag()}"
                f"__seed{self.seed}__N{self.n_replicas}__T{self.n_steps}__{self.spec_hash()}")


# ---------------------------------------------------------------------------
# physics/base -> core objects (reuse the phase-study builders + TI cache).
# ---------------------------------------------------------------------------
def build_params(spec: SerialRunSpec) -> "core.DimerWCAParams":
    return core.DimerWCAParams(
        n_dim=int(spec.n_dim), a=float(spec.a), sigma=float(spec.sigma),
        epsilon=float(spec.epsilon), h=float(spec.h), w=float(spec.w), beta=float(spec.beta))


def build_sim(spec: SerialRunSpec, base: dict) -> "core.SimConfig":
    """SimConfig from the YAML base block (ABF knobs). n_replicas=1: the ABF machinery
    (bandwidths, warmup ramp, burn-in, wall, clips, eval window) is identical to ordinary
    ABF; FR knobs are irrelevant for the serial control."""
    kw = dict(base)
    kw.pop("ti", None)
    kw.update(n_replicas=1, n_steps=int(spec.n_steps), seed=int(spec.seed))
    valid = core.SimConfig.__dataclass_fields__.keys()
    kw = {k: v for k, v in kw.items() if k in valid}
    return core.SimConfig(**kw)


def engine_key(spec: SerialRunSpec):
    return (int(spec.n_dim), float(spec.a), float(spec.sigma),
            float(spec.epsilon), float(spec.h), float(spec.w))


def get_engine(spec: SerialRunSpec, engines: dict):
    k = engine_key(spec)
    if k not in engines:
        engines[k] = core.WCADimerEngine(build_params(spec), core.DEVICE, core.DTYPE)
    return engines[k]


def get_reference(spec: SerialRunSpec, base: dict, engine, cache_dir, verbose=False):
    """TI reference for this physics (same cache key as the phase/followup studies, so the
    already-cached references in cache/phase are reused and never recomputed)."""
    sim = build_sim(spec, base)
    params = build_params(spec)
    ti = pj.build_ti_config(base, sim)
    path = pj.ti_cache_path(cache_dir, spec, sim.n_grid)
    return core.load_or_compute_ti_reference(path, params, sim, ti, engine, verbose=verbose)


# ---------------------------------------------------------------------------
# Per-run IO (one .npz per trajectory; idempotent skip -- same convention as followup).
# ---------------------------------------------------------------------------
def run_npz_path(raw_dir: str, spec: SerialRunSpec) -> str:
    return os.path.join(raw_dir, spec.run_id() + ".npz")


def run_is_valid(path: str) -> bool:
    if not os.path.exists(path):
        return False
    try:
        d = np.load(path, allow_pickle=True)
        # A *complete* run (reached its target budget) with a finite, non-NaN L2 is valid;
        # a partial emit (complete=False) is analyzable but still resumed, so NOT "valid".
        ok = ("l2_f" in d.files and np.isfinite(float(d["l2_f"]))
              and bool(d.get("complete", np.array(False)))
              and not bool(d.get("had_nan", np.array(False))))
        d.close()
        return bool(ok)
    except Exception:
        return False


def save_run(path: str, out: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **out)
    os.replace(tmp, path)


def save_failure(raw_dir: str, group_id: str, specs, exc: Exception):
    fail_dir = os.path.join(raw_dir, "_failures")
    os.makedirs(fail_dir, exist_ok=True)
    path = os.path.join(fail_dir, group_id + ".json")
    with open(path, "w") as fh:
        json.dump({"group_id": group_id, "specs": [asdict(s) for s in specs],
                   "error": repr(exc), "time": time.time()}, fh, indent=2)
    return path


# ---------------------------------------------------------------------------
# Batched (G, n_grid) grid helpers -- per-trajectory analogues of the 1-D core helpers.
# ---------------------------------------------------------------------------
def _smooth_batched(y, sigma_grid_points):
    """Gaussian smooth each row of y (G, n_grid) independently (replicate-padded conv1d)."""
    sigma = float(sigma_grid_points)
    if sigma <= 0:
        return y.clone()
    radius = max(1, int(math.ceil(4.0 * sigma)))
    x = torch.arange(-radius, radius + 1, device=y.device, dtype=y.dtype)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()
    yp = torch.nn.functional.pad(y.unsqueeze(1), (radius, radius), mode="replicate")
    return torch.nn.functional.conv1d(yp, kernel.view(1, 1, -1)).squeeze(1)


def _cumtrapz_batched(y, grid):
    """Cumulative trapezoid of each row of y (G, n_grid) along the grid axis."""
    dz = (grid[1:] - grid[:-1])[None, :]
    out = torch.zeros_like(y)
    out[:, 1:] = torch.cumsum(0.5 * (y[:, 1:] + y[:, :-1]) * dz, dim=1)
    return out


def _normalize_midpoint_batched(profile, grid, midpoint=0.5):
    idx = int(torch.argmin(torch.abs(grid - midpoint)).item())
    return profile - profile[:, idx:idx + 1]


def _interp_edge_batched(profile, grid, z):
    """Per-row linear interp of profile (G, n_grid) at z (G,) with edge-hold outside."""
    dz = grid[1] - grid[0]
    x = (z - grid[0]) / dz
    idx0 = torch.floor(x).long().clamp(0, grid.numel() - 2)
    frac = (x - idx0.to(z.dtype)).clamp(0.0, 1.0)
    lo = torch.gather(profile, 1, idx0.unsqueeze(1)).squeeze(1)
    hi = torch.gather(profile, 1, (idx0 + 1).unsqueeze(1)).squeeze(1)
    val = (1.0 - frac) * lo + frac * hi
    val = torch.where(z < grid[0], profile[:, 0], val)
    val = torch.where(z > grid[-1], profile[:, -1], val)
    return val


class BatchedKernelABFEstimator:
    """Per-trajectory kernel ABF estimator: num/den have shape (G, n_grid). With N=1 sample
    per trajectory per step, each trajectory accumulates into its OWN row only. Numerically
    identical to G independent ``core.TorchKernelABFEstimator`` instances."""

    def __init__(self, z_grid, G, bandwidth, smooth_sigma=0.0, edge_extrapolate=False):
        self.z_grid = z_grid
        self.G = int(G)
        self.bandwidth = float(bandwidth)
        self.smooth_sigma = float(smooth_sigma)
        self.edge_extrapolate = bool(edge_extrapolate)
        self.num = torch.zeros(G, z_grid.numel(), device=z_grid.device, dtype=z_grid.dtype)
        self.den = torch.zeros(G, z_grid.numel(), device=z_grid.device, dtype=z_grid.dtype)
        self.n_updates = 0

    def update(self, z, f):                              # z (G,), f (G,)
        w = core.gaussian_kernel_torch(self.z_grid[None, :] - z[:, None], self.bandwidth)
        self.num += w * f[:, None]
        self.den += w
        self.n_updates += 1

    def mean_force_profile(self):                        # (G, n_grid)
        raw = self.num / torch.clamp(self.den, min=EPS)
        raw = torch.where(self.den > EPS, raw, torch.zeros_like(raw))
        return _smooth_batched(raw, self.smooth_sigma)

    def evaluate(self, z):                               # (G,)
        profile = self.mean_force_profile()
        return _interp_edge_batched(profile, self.z_grid, z)

    def pmf_profile(self):                               # (G, n_grid)
        pmf = _cumtrapz_batched(self.mean_force_profile(), self.z_grid)
        return _normalize_midpoint_batched(pmf, self.z_grid)

    def effective_counts(self):
        return self.den.clone()

    # -- checkpoint round-trip --
    def state(self):
        return dict(num=core.to_numpy(self.num), den=core.to_numpy(self.den),
                    n_updates=int(self.n_updates))

    def load(self, st):
        self.num = torch.as_tensor(st["num"], device=self.z_grid.device, dtype=self.z_grid.dtype)
        self.den = torch.as_tensor(st["den"], device=self.z_grid.device, dtype=self.z_grid.dtype)
        self.n_updates = int(st["n_updates"])


# ---------------------------------------------------------------------------
# Snapshot / ladder schedule.
# ---------------------------------------------------------------------------
def snapshot_schedule(target_steps, base_budget=BASE_BUDGET, ladder_fracs=DEFAULT_LADDER_FRACS,
                      n_log=40, snap_start=10000):
    """Steps at which to record a budget-ladder snapshot for a run of ``target_steps``:
    the ladder fractions of the *base* equal budget (so ladder points line up across cells)
    plus a log-spaced grid, all clipped to [snap_start, target] and always including target."""
    steps = set()
    ladder = set()
    for fr in ladder_fracs:
        s = int(round(fr * base_budget))
        if snap_start <= s <= target_steps:
            steps.add(s)
            ladder.add(s)
    lo = max(int(snap_start), 1)
    if target_steps > lo and n_log > 0:
        for v in np.geomspace(lo, target_steps, int(n_log)):
            s = int(round(v))
            if snap_start <= s <= target_steps:
                steps.add(s)
    steps.add(int(target_steps))
    return sorted(steps), sorted(ladder | {int(target_steps)})


# ---------------------------------------------------------------------------
# Checkpoint IO (one checkpoint per batched group).
# ---------------------------------------------------------------------------
def checkpoint_path(ckpt_dir, group_id):
    return os.path.join(ckpt_dir, group_id + ".ckpt.npz")


def _save_checkpoint(path, blob):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez(tmp, **{k: np.asarray(v, dtype=object) if isinstance(v, (dict, list)) else v
                     for k, v in blob.items()})
    os.replace(tmp, path)


def _load_checkpoint(path):
    if not os.path.exists(path):
        return None
    try:
        d = np.load(path, allow_pickle=True)
        return {k: d[k].item() if d[k].dtype == object and d[k].ndim == 0 else d[k] for k in d.files}
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Batched serial ABF: G independent one-walker trajectories of ONE physics.
# ---------------------------------------------------------------------------
@torch.inference_mode()
def run_serial_abf_batched(specs, base, engine, ref, ckpt_path, *,
                           checkpoint_every=1_000_000, base_budget=BASE_BUDGET,
                           ladder_fracs=DEFAULT_LADDER_FRACS, n_log_snapshots=40,
                           progress_every=2_000_000, verbose=True, job_seed=None,
                           bench_steps=None, stop_after_step=None, emit_fn=None):
    """Advance G = len(specs) independent one-walker ABF trajectories (same physics, distinct
    seeds) together, each with its own ABF accumulators. Chunked + checkpointed; resumes from
    ``ckpt_path`` if present. Returns a list of per-trajectory output dicts (one per spec).

    ``bench_steps`` (if given) overrides the target and disables checkpoint/final assembly --
    it is used only by the benchmark path to time raw ms/step.

    ``stop_after_step`` (if given) gracefully stops after that step, writing a checkpoint and
    returning ``None`` instead of the per-trajectory outputs (i.e. "not done yet"). It enables
    a deterministic resume test and an optional wall-time-friendly partial run; the periodic
    checkpoint already makes any interruption resumable regardless of this flag.

    ``emit_fn`` (if given) is called with the current per-trajectory outputs at every
    checkpoint (``complete=False``), so a still-running multi-day job is already analyzable up
    to its accumulated budget (the budget ladder). The final call at completion has
    ``complete=True``.
    """
    assert len({s.physics_tag() for s in specs}) == 1, "batched group must share physics"
    G = len(specs)
    params = build_params(specs[0])
    sim = build_sim(specs[0], base)
    dev, dt = engine.device, engine.dtype
    target = int(bench_steps) if bench_steps else int(specs[0].n_steps)
    seeds = [int(s.seed) for s in specs]
    if job_seed is None:
        job_seed = 20260703 + (int(hashlib.md5(specs[0].physics_tag().encode()).hexdigest(), 16) % 100000)

    grid = torch.linspace(sim.z_min, sim.z_max, sim.n_grid, device=dev, dtype=dt)
    grid_np = core.to_numpy(grid)
    ref_fe = np.asarray(ref["free_energy"], float)
    ref_mf = np.asarray(ref["mean_force"], float)
    eval_mask = core.eval_window_mask_np(grid_np, sim)
    noise_scale = math.sqrt(2.0 * sim.dt / params.beta)
    dz = float((sim.z_max - sim.z_min) / max(sim.n_grid - 1, 1))
    z_barrier = 0.5 * (sim.transition_lo + sim.transition_hi)

    bias_est = BatchedKernelABFEstimator(grid, G, sim.abf_bandwidth, sim.abf_smooth_sigma,
                                         edge_extrapolate=sim.abf_edge_extrapolate)
    prod_est = BatchedKernelABFEstimator(grid, G, sim.abf_bandwidth, sim.abf_smooth_sigma,
                                         edge_extrapolate=sim.abf_edge_extrapolate)

    snap_steps, ladder_steps = snapshot_schedule(
        target, base_budget=base_budget, ladder_fracs=ladder_fracs,
        n_log=(0 if bench_steps else n_log_snapshots),
        snap_start=int(sim.estimator_burn_in_steps))
    snap_set = set(snap_steps)

    # -- diagnostics accumulated on-device --
    region_time = torch.zeros(G, 3, device=dev, dtype=dt)      # compact/transition/stretched step counts
    marg_hist = torch.zeros(G, sim.n_grid, device=dev, dtype=dt)
    n_post_burnin = torch.zeros((), device=dev, dtype=dt)
    up_cross = torch.zeros(G, device=dev, dtype=torch.long)
    dn_cross = torch.zeros(G, device=dev, dtype=torch.long)
    side = None
    snap_budget, snap_l2f, snap_l2fp = [], [], []             # snap_l2f/fp: list of (G,) arrays
    wall_accum = 0.0
    start_step = 0

    # -- resume --
    ck = None if bench_steps else _load_checkpoint(ckpt_path)
    if ck is not None and int(ck["target"]) == target and list(ck["seeds"]) == seeds:
        q = torch.as_tensor(ck["q"], device=dev, dtype=dt)
        bias_est.load(ck["bias_state"]); prod_est.load(ck["prod_state"])
        region_time = torch.as_tensor(ck["region_time"], device=dev, dtype=dt)
        marg_hist = torch.as_tensor(ck["marg_hist"], device=dev, dtype=dt)
        n_post_burnin = torch.as_tensor(float(ck["n_post_burnin"]), device=dev, dtype=dt)
        up_cross = torch.as_tensor(ck["up_cross"], device=dev, dtype=torch.long)
        dn_cross = torch.as_tensor(ck["dn_cross"], device=dev, dtype=torch.long)
        sd = ck["side"]
        side = None if (np.ndim(sd) == 0 and sd is None) else torch.as_tensor(sd, device=dev, dtype=torch.bool)
        sb = np.asarray(ck["snap_budget"], dtype=np.int64)
        snap_budget = [int(x) for x in sb]
        snap_l2f = [np.asarray(r, float) for r in np.asarray(ck["snap_l2f"], float)] if sb.size else []
        snap_l2fp = [np.asarray(r, float) for r in np.asarray(ck["snap_l2fp"], float)] if sb.size else []
        wall_accum = float(ck["wall_accum"])
        start_step = int(ck["resume_step"])
        torch.random.set_rng_state(torch.as_tensor(ck["cpu_rng"], dtype=torch.uint8))
        if dev.type == "cuda" and "cuda_rng" in ck:
            torch.cuda.set_rng_state(torch.as_tensor(ck["cuda_rng"], dtype=torch.uint8), dev)
        if verbose:
            print(f"[serial] resume {specs[0].physics_tag()} from step {start_step}/{target} "
                  f"({len(snap_budget)} snapshots restored)")
    else:
        torch.manual_seed(job_seed)
        q = torch.stack([core.lattice_initial_conditions(params, 1, dev, dt, seed=s)[0] for s in seeds], dim=0)

    def _take_snapshot(step):
        est = prod_est if prod_est.n_updates > 0 else bias_est
        mf = core.to_numpy(est.mean_force_profile())        # (G, n_grid)
        pmf = core.to_numpy(est.pmf_profile())
        l2f = np.empty(G); l2fp = np.empty(G)
        for g in range(G):
            fe_al = core.align_additive_constant_np(pmf[g], ref_fe, grid_np, mask=eval_mask)
            l2f[g] = core.profile_l2_error_np(fe_al, ref_fe, grid_np, mask=eval_mask)
            l2fp[g] = core.profile_l2_error_np(mf[g], ref_mf, grid_np, mask=eval_mask)
        snap_budget.append(int(step)); snap_l2f.append(l2f); snap_l2fp.append(l2fp)

    def _write_checkpoint(resume_step):
        blob = dict(
            physics_tag=specs[0].physics_tag(), seeds=np.asarray(seeds, np.int64),
            target=target, resume_step=int(resume_step), job_seed=int(job_seed),
            q=core.to_numpy(q), bias_state=bias_est.state(), prod_state=prod_est.state(),
            region_time=core.to_numpy(region_time), marg_hist=core.to_numpy(marg_hist),
            n_post_burnin=float(n_post_burnin.item()),
            up_cross=core.to_numpy(up_cross), dn_cross=core.to_numpy(dn_cross),
            side=(core.to_numpy(side) if side is not None else None),
            snap_budget=np.asarray(snap_budget, np.int64),
            snap_l2f=np.asarray(snap_l2f, float) if snap_l2f else np.zeros((0, G)),
            snap_l2fp=np.asarray(snap_l2fp, float) if snap_l2fp else np.zeros((0, G)),
            wall_accum=float(wall_accum),
            cpu_rng=core.to_numpy(torch.random.get_rng_state()),
        )
        if dev.type == "cuda":
            blob["cuda_rng"] = core.to_numpy(torch.cuda.get_rng_state(dev))
        _save_checkpoint(ckpt_path, blob)

    def _assemble(step_reached, done):
        """Build per-trajectory output dicts from the CURRENT estimator/diagnostics/snapshot
        state. ``done=False`` (partial emit at a checkpoint) marks the npz incomplete so the
        runner still resumes it; ``budget_reached`` is the actual budget the metrics reflect."""
        est = prod_est if prod_est.n_updates > 0 else bias_est
        mf_f = core.to_numpy(est.mean_force_profile())
        pmf_f = core.to_numpy(est.pmf_profile())
        rt = core.to_numpy(region_time)
        marg = core.to_numpy(marg_hist)
        up = core.to_numpy(up_cross); dn = core.to_numpy(dn_cross)
        budget_arr = np.asarray(snap_budget, np.int64)
        l2f_arr = np.asarray(snap_l2f, float) if snap_l2f else np.zeros((0, G))
        l2fp_arr = np.asarray(snap_l2fp, float) if snap_l2fp else np.zeros((0, G))
        outs = []
        for g, spec in enumerate(specs):
            fe_al = core.align_additive_constant_np(pmf_f[g], ref_fe, grid_np, mask=eval_mask)
            l2_f = core.profile_l2_error_np(fe_al, ref_fe, grid_np, mask=eval_mask)
            l2_fp = core.profile_l2_error_np(mf_f[g], ref_mf, grid_np, mask=eval_mask)
            reg_f = core.region_l2_errors(fe_al, ref_fe, grid_np, sim)
            reg_fp = core.region_l2_errors(mf_f[g], ref_mf, grid_np, sim)
            tot = float(rt[g].sum())
            fr_c = rt[g, 0] / tot if tot > 0 else float("nan")
            fr_t = rt[g, 1] / tot if tot > 0 else float("nan")
            fr_s = rt[g, 2] / tot if tot > 0 else float("nan")
            mg = marg[g].copy()
            mg_mass = mg.sum() * dz
            marginal = mg / mg_mass if mg_mass > 0 else np.full_like(mg, np.nan)
            budget_g = budget_arr.astype(float)
            l2f_g = l2f_arr[:, g] if l2f_arr.size else np.array([])
            l2fp_g = l2fp_arr[:, g] if l2fp_arr.size else np.array([])
            times_g = budget_g * sim.dt
            integ_time = float(np.trapezoid(l2f_g, times_g)) if l2f_g.size > 1 else float("nan")
            integ_budget = float(np.trapezoid(l2f_g, budget_g)) if l2f_g.size > 1 else float("nan")
            span = (budget_g[-1] - budget_g[0]) if budget_g.size > 1 else 0.0
            auc_budget = (integ_budget / span) if span > 0 else float("nan")
            nanflag = bool(not np.isfinite(l2_f) or np.isnan(mf_f[g]).any())
            out = {
                "run_id": spec.run_id(), "spec_hash": spec.spec_hash(),
                "spec_json": json.dumps(asdict(spec), sort_keys=True),
                "study": spec.study, "stage": spec.stage, "mode": "serial",
                "name": "serial_abf", "method": "serial_abf", "seed": spec.seed,
                "n_steps": spec.n_steps, "n_replicas": spec.n_replicas, "budget": spec.budget,
                "budget_reached": int(step_reached), "complete": bool(done),
                "beta": spec.beta, "h": spec.h, "w": spec.w, "n_dim": spec.n_dim, "M": spec.M,
                "a": spec.a, "sigma": spec.sigma, "epsilon": spec.epsilon, "beta_h": spec.beta * spec.h,
                "job_seed": int(job_seed), "core_version": "wca_serial_abf_v1",
                "config_hash": sim.config_hash(), "device": str(core.DEVICE),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "wall_seconds": float(wall_accum), "runtime_seconds": float(wall_accum),
                "had_nan": nanflag,
                "l2_f": l2_f, "l2_fp": l2_fp,
                "l2_f_compact": reg_f["compact"], "l2_f_transition": reg_f["transition"],
                "l2_f_stretched": reg_f["stretched"],
                "l2_fp_compact": reg_fp["compact"], "l2_fp_transition": reg_fp["transition"],
                "l2_fp_stretched": reg_fp["stretched"],
                "frac_compact": fr_c, "frac_transition": fr_t, "frac_stretched": fr_s,
                "n_up_crossings": int(up[g]), "n_down_crossings": int(dn[g]),
                "n_barrier_crossings": int(up[g] + dn[g]), "n_round_trips": int(min(up[g], dn[g])),
                "integrated_l2_f": integ_time, "integrated_l2_f_budget": integ_budget,
                "budget_auc_normalized_l2_f": auc_budget,
                "grid": grid_np, "ref_free_energy": ref_fe, "ref_mean_force": ref_mf,
                "final_mean_force": mf_f[g], "final_pmf": fe_al, "final_marginal": marginal,
                "snap_budget": budget_g, "l2_f_t": l2f_g, "l2_fp_t": l2fp_g,
                "times_t": times_g, "ladder_steps": np.asarray(ladder_steps, np.int64),
                "dt": sim.dt,
            }
            outs.append(out)
        return outs

    t0 = time.perf_counter()
    for step in range(start_step, target + 1):
        forces_raw = engine.force(q, compute_energy=False)
        forces_physical = core.clip_forces(forces_raw, params.force_clip)
        z = core.reaction_coordinate(q, params)

        # per-trajectory barrier crossings (read-only; no RNG)
        cur_side = z > z_barrier
        if side is None:
            side = cur_side
        else:
            up_cross += ((~side) & cur_side).long()
            dn_cross += (side & (~cur_side)).long()
            side = cur_side

        if step >= sim.estimator_burn_in_steps:
            reg = torch.stack([(z < sim.transition_lo),
                               (z >= sim.transition_lo) & (z <= sim.transition_hi),
                               (z > sim.transition_hi)], dim=1).to(dt)
            region_time += reg
            idx = ((z - sim.z_min) / max(dz, EPS)).long().clamp_(0, sim.n_grid - 1)
            marg_hist.scatter_add_(1, idx.unsqueeze(1), torch.ones(G, 1, device=dev, dtype=dt))
            n_post_burnin += 1.0

        mean_force_input = forces_physical if sim.use_clipped_force_for_mean_force else forces_raw
        f_local = core.local_mean_force(q, mean_force_input, params)
        f_local = torch.clamp(f_local, -sim.mean_force_sample_clip, sim.mean_force_sample_clip)
        bias_est.update(z, f_local)
        if step >= sim.estimator_burn_in_steps:
            prod_est.update(z, f_local)

        ramp = min(1.0, step / max(sim.abf_warmup_steps, 1))
        abf_scale = sim.abf_bias_scale * ramp
        abf_at_z = abf_scale * torch.clamp(bias_est.evaluate(z), -sim.abf_force_clip, sim.abf_force_clip)
        transport = core.clip_forces(core.add_abf_force(q, forces_physical, abf_at_z, params), params.force_clip)
        transport = core.clip_forces(
            core.add_reaction_coordinate_wall_force(q, transport, z, sim, params), params.force_clip)

        if (not bench_steps) and step in snap_set:
            _take_snapshot(step)

        if step == target:
            break

        q = core.wrap_positions(q + sim.dt * transport + noise_scale * torch.randn_like(q), params.box_length)

        if stop_after_step is not None and (step + 1) >= int(stop_after_step):
            wall_accum += time.perf_counter() - t0
            _write_checkpoint(step + 1)
            if emit_fn is not None:
                emit_fn(_assemble(step + 1, False))
            if verbose:
                print(f"[serial] {specs[0].physics_tag()} graceful stop at step {step+1}/{target} "
                      f"(checkpoint written; not done)")
            return None

        if (not bench_steps) and checkpoint_every and ((step + 1) % checkpoint_every == 0):
            wall_accum += time.perf_counter() - t0
            _write_checkpoint(step + 1)
            if emit_fn is not None:
                emit_fn(_assemble(step + 1, False))     # partial npz -> ladder analyzable mid-run
            t0 = time.perf_counter()
            if verbose:
                print(f"[serial] {specs[0].physics_tag()} checkpoint at step {step+1}/{target} "
                      f"(wall {wall_accum/3600:.2f}h)")
        if verbose and progress_every and ((step + 1) % progress_every == 0):
            print(f"[serial] {specs[0].physics_tag()} step {step+1}/{target}")

    wall_accum += time.perf_counter() - t0
    if bench_steps:
        return {"wall_seconds": wall_accum, "n_steps": target, "G": G,
                "ms_per_step": 1e3 * wall_accum / max(target, 1)}
    return _assemble(target, True)


# ---------------------------------------------------------------------------
# Config -> SerialRunSpec expansion + batched-group grouping.
# ---------------------------------------------------------------------------
PHYSICS_AXES = ("beta", "h", "w", "n_dim", "a", "sigma", "epsilon")


def load_yaml(path: str) -> dict:
    import yaml
    with open(path) as fh:
        return yaml.safe_load(fh)


def effective_base(cfg: dict, stage: str) -> dict:
    base = dict(cfg.get("base", {}))
    st = cfg.get("stages", {}).get(stage, {})
    base.update(st.get("base_overrides", {}))
    return base


def serial_settings(cfg: dict) -> dict:
    s = dict(cfg.get("serial", {}))
    return dict(
        base_budget=int(s.get("base_budget", BASE_BUDGET)),
        ladder_fracs=tuple(s.get("ladder_fracs", DEFAULT_LADDER_FRACS)),
        n_log_snapshots=int(s.get("n_log_snapshots", 40)),
        checkpoint_every_steps=int(s.get("checkpoint_every_steps", 1_000_000)),
    )


def _cell_physics(cfg: dict, cell: dict) -> dict:
    defaults = dict(cfg.get("system_defaults", {}))
    for k, v in dict(n_dim=10, a=1.5, sigma=1.0, epsilon=1.0, w=2.0, beta=1.0, h=2.0).items():
        defaults.setdefault(k, v)
    phys = dict(defaults)
    phys.update({k: cell[k] for k in PHYSICS_AXES if k in cell})
    return {k: phys[k] for k in PHYSICS_AXES}


def expand_stage(cfg: dict, stage: str) -> list:
    """Expand one stage into per-trajectory SerialRunSpecs. Each cell carries its own
    ``n_steps`` (target budget, since N=1) and ``seeds`` list; a cell without them falls
    back to the stage defaults."""
    st = cfg["stages"][stage]
    study = cfg.get("experiment_name", "wca_serial_abf")
    default_seeds = list(st.get("seeds", [0, 1, 2, 3, 4]))
    default_nsteps = int(st.get("n_steps", BASE_BUDGET))
    specs, seen = [], set()
    cells = st.get("cells", [{}])
    for cell in cells:
        phys = _cell_physics(cfg, cell)
        seeds = list(cell.get("seeds", default_seeds))
        nsteps = int(cell.get("n_steps", default_nsteps))
        for s in seeds:
            spec = SerialRunSpec(
                study=study, stage=stage, method="serial_abf", seed=int(s),
                n_steps=nsteps, n_replicas=1,
                beta=float(phys["beta"]), h=float(phys["h"]), w=float(phys["w"]),
                n_dim=int(phys["n_dim"]), a=float(phys["a"]), sigma=float(phys["sigma"]),
                epsilon=float(phys["epsilon"]))
            if spec.run_id() not in seen:
                seen.add(spec.run_id())
                specs.append(spec)
    return specs


def group_id(physics_tag: str, n_steps: int, stage: str) -> str:
    return f"{stage}__{physics_tag}__T{int(n_steps)}"


def group_specs(specs) -> dict:
    """Group per-trajectory specs into batched jobs keyed by (physics, target). Each group
    is one batched GPU job (all its seeds advance together, each an independent one-walker
    ABF trajectory with its own accumulators)."""
    groups = {}
    for s in specs:
        groups.setdefault((s.physics_tag(), int(s.n_steps), s.stage), []).append(s)
    for k in groups:
        groups[k] = sorted(groups[k], key=lambda s: s.seed)
    return groups


def distinct_physics(specs) -> list:
    out, seen = [], set()
    for s in specs:
        if s.physics_tag() not in seen:
            seen.add(s.physics_tag())
            out.append(s)
    return out
