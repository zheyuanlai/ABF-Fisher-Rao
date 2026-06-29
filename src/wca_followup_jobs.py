"""Job expansion + single-run pipeline for the WCA *follow-up* studies.

This is an ADDITIVE companion to ``wca_phase_jobs.py``. It deliberately does NOT
reuse :class:`wca_phase_jobs.PhaseRunSpec`, because adding fields to that frozen
dataclass would change its ``spec_hash`` and orphan the 224 completed phase-diagram
runs. Instead it defines :class:`FollowupRunSpec`, a superset that also carries the
adaptive-FR knobs, an equal-compute ``budget`` view, and a ``mode`` switch
(``sample`` vs ``frozen``) for the frozen-bias validation. Physics -> core objects,
the TI-reference cache, and the L2 metrics are reused verbatim from
``wca_abffr_core`` / ``wca_phase_jobs`` so a follow-up cell at a given physics is
directly comparable to the corresponding phase-diagram cell.

Three study families share this module (selected by the YAML ``mode``):

  sample  : representative-cell seed expansion (Part C), the adaptive method
            (Part D), and equal-compute ABF baselines (Part E). One sampler run
            per (cell, method, seed, budget); writes the full phase-diagram metric
            set plus the adaptive per-event log and aggregate score statistics.
  frozen  : frozen-bias validation (Part F). Loads a stage-1 learned bias
            (seed-averaged final mean force + PMF for a (cell, source method)),
            then runs FIXED-bias dynamics with independent seeds and reconstructs
            F(z) = B(z) - beta^{-1} log p_B(z) + C. Deployable: no TI is read by
            the dynamics; TI is used only for the post-hoc L2 of the reconstruction.

Particle counts (as in the phase study): M = n_dim^2 PHYSICAL bath particles;
N = n_replicas independent ABF/FR replicas.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import time
from dataclasses import dataclass, asdict, field

import numpy as np

import wca_abffr_core as core
import wca_phase_jobs as pj


# ---------------------------------------------------------------------------
# FollowupRunSpec
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FollowupRunSpec:
    study: str           # e.g. "representative" / "equal_compute" / "frozen_bias"
    stage: str
    mode: str            # "sample" | "frozen"
    name: str            # method label
    method: str          # abf / fr_estimated / fr_uniform / fr_oracle / fr_estimated_adaptive
    seed: int
    n_steps: int
    n_replicas: int
    save_every: int
    # ---- physics ----
    beta: float
    h: float
    w: float
    n_dim: int
    a: float
    sigma: float
    epsilon: float
    # ---- Fisher-Rao knobs ----
    fr_rate: float
    target_ema_rate: float
    max_event_fraction: float
    fr_every: int
    fr_start_steps: int
    score_clip: float
    # ---- adaptive-FR knobs (used only when method == fr_estimated_adaptive) ----
    adaptive_base_fr_rate: float = 0.20
    adaptive_support_mode: str = "marginal_uniform"
    adaptive_support_lo: float = 0.115
    adaptive_support_hi: float = 0.14
    adaptive_support_ema_rate: float = 0.02
    adaptive_gate_warmup_steps: int = 20000
    adaptive_neff_target_strategy: str = "relative_median"
    adaptive_neff_target_frac: float = 0.5
    adaptive_tv_scale: float = 0.30
    adaptive_ess_full_threshold: float = 0.25
    adaptive_ess_stop_threshold: float = 0.10
    adaptive_event_backoff_threshold: float = 0.80
    adaptive_event_backoff_factor: float = 0.50
    adaptive_min_rate: float = 0.0
    adaptive_max_rate: float = 0.20
    # ---- frozen-bias only ----
    frozen_source_method: str = ""     # which learned bias to freeze ("" for sample)
    frozen_source_study: str = ""      # output_root of the source (sample) study

    @property
    def M(self) -> int:
        return int(self.n_dim) * int(self.n_dim)

    @property
    def budget(self) -> int:
        """Force-evaluation budget = N * n_steps (per physical M)."""
        return int(self.n_replicas) * int(self.n_steps)

    def physics_tag(self) -> str:
        return (f"b{self.beta:g}_h{self.h:g}_w{self.w:g}"
                f"_n{int(self.n_dim)}_a{self.a:g}")

    def spec_hash(self) -> str:
        return hashlib.md5(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]

    def run_id(self) -> str:
        src = f"__from-{self.frozen_source_method}" if self.mode == "frozen" else ""
        return (f"{self.stage}__{self.name}{src}__{self.physics_tag()}"
                f"__seed{self.seed}__N{self.n_replicas}__T{self.n_steps}__{self.spec_hash()}")


# ---------------------------------------------------------------------------
# physics -> core SimConfig (adds adaptive knobs to pj.build_sim's behaviour)
# ---------------------------------------------------------------------------
def build_sim_followup(spec: FollowupRunSpec, base: dict) -> "core.SimConfig":
    kw = dict(base)
    kw.pop("ti", None)
    kw.update(
        n_replicas=int(spec.n_replicas), n_steps=int(spec.n_steps), seed=int(spec.seed),
        save_every=int(spec.save_every), fr_rate=float(spec.fr_rate),
        target_ema_rate=float(spec.target_ema_rate),
        max_event_fraction=float(spec.max_event_fraction), fr_every=int(spec.fr_every),
        fr_start_steps=int(spec.fr_start_steps), score_clip=float(spec.score_clip),
        # adaptive
        adaptive_fr_enabled=(spec.method == "fr_estimated_adaptive"),
        adaptive_base_fr_rate=float(spec.adaptive_base_fr_rate),
        adaptive_support_mode=str(spec.adaptive_support_mode),
        adaptive_support_lo=float(spec.adaptive_support_lo),
        adaptive_support_hi=float(spec.adaptive_support_hi),
        adaptive_support_ema_rate=float(spec.adaptive_support_ema_rate),
        adaptive_gate_warmup_steps=int(spec.adaptive_gate_warmup_steps),
        adaptive_neff_target_strategy=str(spec.adaptive_neff_target_strategy),
        adaptive_neff_target_frac=float(spec.adaptive_neff_target_frac),
        adaptive_tv_scale=float(spec.adaptive_tv_scale),
        adaptive_ess_full_threshold=float(spec.adaptive_ess_full_threshold),
        adaptive_ess_stop_threshold=float(spec.adaptive_ess_stop_threshold),
        adaptive_event_backoff_threshold=float(spec.adaptive_event_backoff_threshold),
        adaptive_event_backoff_factor=float(spec.adaptive_event_backoff_factor),
        adaptive_min_rate=float(spec.adaptive_min_rate),
        adaptive_max_rate=float(spec.adaptive_max_rate),
    )
    valid = core.SimConfig.__dataclass_fields__.keys()
    kw = {k: v for k, v in kw.items() if k in valid}
    return core.SimConfig(**kw)


def build_params(spec: FollowupRunSpec) -> "core.DimerWCAParams":
    return core.DimerWCAParams(
        n_dim=int(spec.n_dim), a=float(spec.a), sigma=float(spec.sigma),
        epsilon=float(spec.epsilon), h=float(spec.h), w=float(spec.w), beta=float(spec.beta))


def engine_key(spec: FollowupRunSpec):
    return (int(spec.n_dim), float(spec.a), float(spec.sigma),
            float(spec.epsilon), float(spec.h), float(spec.w))


def get_engine(spec: FollowupRunSpec, engines: dict):
    k = engine_key(spec)
    if k not in engines:
        engines[k] = core.WCADimerEngine(build_params(spec), core.DEVICE, core.DTYPE)
    return engines[k]


def get_reference(spec: FollowupRunSpec, base: dict, engine, cache_dir, verbose=False):
    """TI reference for this spec's physics (memoised; identical cache key to the
    phase study so cached references are shared, never recomputed)."""
    sim = build_sim_followup(spec, base)
    params = build_params(spec)
    ti = pj.build_ti_config(base, sim)
    path = pj.ti_cache_path(cache_dir, spec, sim.n_grid)
    return core.load_or_compute_ti_reference(path, params, sim, ti, engine, verbose=verbose)


# ---------------------------------------------------------------------------
# Per-run IO (one .npz per run; idempotent skip).
# ---------------------------------------------------------------------------
def run_npz_path(raw_dir: str, spec: FollowupRunSpec) -> str:
    return os.path.join(raw_dir, spec.run_id() + ".npz")


def run_is_valid(path: str) -> bool:
    if not os.path.exists(path):
        return False
    try:
        d = np.load(path, allow_pickle=True)
        ok = ("l2_f" in d.files and np.isfinite(float(d["l2_f"]))
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


def save_failure(raw_dir: str, spec: FollowupRunSpec, exc: Exception):
    fail_dir = os.path.join(raw_dir, "_failures")
    os.makedirs(fail_dir, exist_ok=True)
    path = os.path.join(fail_dir, spec.run_id() + ".json")
    with open(path, "w") as fh:
        json.dump({"run_id": spec.run_id(), "spec": asdict(spec),
                   "error": repr(exc), "time": time.time()}, fh, indent=2)
    return path


# ---------------------------------------------------------------------------
# SAMPLE run (Parts C/D/E): full sampler + adaptive log + score stats + budget.
# ---------------------------------------------------------------------------
def execute_sample_run(spec: FollowupRunSpec, base: dict, engine, cache_dir, verbose=False):
    params = build_params(spec)
    sim = build_sim_followup(spec, base)
    ref = get_reference(spec, base, engine, cache_dir, verbose=verbose)
    ic = core.lattice_initial_conditions(params, sim.n_replicas, engine.device, engine.dtype, seed=sim.seed)
    oracle_fe = ref["free_energy"] if spec.method == "fr_oracle" else None

    t0 = time.perf_counter()
    diag = core.run_sampler_gpu(spec.method, params, sim, engine, initial_q=ic,
                                oracle_free_energy=oracle_fe, collect_diagnostics=True,
                                verbose=verbose, track_crossings=True)
    fin = core.final_l2_errors(diag, ref, sim)
    ts = core.timeseries_l2(diag, ref, sim)

    grid = ref["grid"]
    mask = core.eval_window_mask_np(grid, sim)
    p_hat = np.asarray(diag["p_hat"][-1], dtype=float) if len(diag["p_hat"]) else np.full_like(grid, np.nan)
    uniform = np.full_like(grid, 1.0 / (grid[-1] - grid[0]))
    marginal_l2_uniform = core.profile_l2_error_np(p_hat, uniform, grid, mask=mask)
    fe = np.asarray(ref["free_energy"], dtype=float)
    log_pref = -float(spec.beta) * (fe - fe.max())
    pref = np.exp(log_pref)
    pref = pref / np.trapezoid(pref, grid)
    marginal_l2_ref = core.profile_l2_error_np(p_hat, pref, grid, mask=mask)
    q_final = (np.asarray(diag["q_target"][-1], dtype=float)
               if len(diag["q_target"]) else np.full_like(grid, np.nan))
    marginal_l2_target = (core.profile_l2_error_np(p_hat, q_final, grid, mask=mask)
                          if np.isfinite(q_final).all() else float("nan"))

    fre = pj.fr_event_stats(spec, diag["steps"], diag["repl_cumulative"], sim.n_replicas)

    maf = np.asarray(diag["max_ancestor_frac"], dtype=float)
    ess = np.asarray(diag["ancestor_ess"], dtype=float)
    final_max_anc = float(maf[-1]) if maf.size else float("nan")
    max_anc_over_time = float(np.nanmax(maf)) if np.isfinite(maf).any() else float("nan")
    min_ess = float(np.nanmin(ess)) if np.isfinite(ess).any() else float("nan")

    had_nan = bool(np.isnan(diag["mean_force"][-1]).any() or np.isnan(diag["pmf"][-1]).any()
                   or not np.isfinite(fin["l2_f"]))

    alog = diag.get("adaptive_log", {})
    out = {
        "run_id": spec.run_id(), "spec_hash": spec.spec_hash(),
        "spec_json": json.dumps(asdict(spec), sort_keys=True),
        "study": spec.study, "stage": spec.stage, "mode": spec.mode,
        "name": spec.name, "method": spec.method, "seed": spec.seed,
        "n_steps": spec.n_steps, "n_replicas": spec.n_replicas, "budget": spec.budget,
        "beta": spec.beta, "h": spec.h, "w": spec.w, "n_dim": spec.n_dim, "M": spec.M,
        "a": spec.a, "sigma": spec.sigma, "epsilon": spec.epsilon, "beta_h": spec.beta * spec.h,
        "fr_rate": spec.fr_rate, "target_ema_rate": spec.target_ema_rate,
        "max_event_fraction": spec.max_event_fraction, "fr_every": spec.fr_every,
        "fr_start_steps": spec.fr_start_steps, "score_clip": spec.score_clip,
        "adaptive_support_mode": spec.adaptive_support_mode,
        "adaptive_base_fr_rate": spec.adaptive_base_fr_rate,
        "config_hash": sim.config_hash(), "core_version": "wca_followup_v1",
        "runtime_seconds": diag["runtime_seconds"], "wall_seconds": time.perf_counter() - t0,
        "device": str(core.DEVICE), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "had_nan": had_nan, "total_replacement_events": diag["total_replacement_events"],
        "l2_f": fin["l2_f"], "l2_fp": fin["l2_fp"], "integrated_l2_f": ts["integrated_l2_f"],
        "l2_f_compact": fin["l2_f_compact"], "l2_f_transition": fin["l2_f_transition"],
        "l2_f_stretched": fin["l2_f_stretched"],
        "l2_fp_compact": fin["l2_fp_compact"], "l2_fp_transition": fin["l2_fp_transition"],
        "l2_fp_stretched": fin["l2_fp_stretched"],
        "marginal_l2_uniform": marginal_l2_uniform, "marginal_l2_target": marginal_l2_target,
        "marginal_l2_ref": marginal_l2_ref,
        "n_compact_to_stretched": diag["n_compact_to_stretched"],
        "n_stretched_to_compact": diag["n_stretched_to_compact"],
        "n_barrier_crossings": diag["n_barrier_crossings"], "n_round_trips": diag["n_round_trips"],
        "fr_event_fraction": fre["fr_event_fraction"], "max_fr_event_fraction": fre["max_fr_event_fraction"],
        "n_fr_applications": fre["n_fr_applications"],
        "deaths_per_fr_application": fre["deaths_per_fr_application"],
        "clones_per_fr_application": fre["clones_per_fr_application"],
        # aggregate score statistics (new in core)
        "fr_score_std": diag.get("fr_score_std", float("nan")),
        "fr_score_absmax": diag.get("fr_score_absmax", float("nan")),
        "fr_score_clip_fraction": diag.get("fr_score_clip_fraction", float("nan")),
        "final_ancestor_ess": float(diag["ancestor_ess"][-1]),
        "final_n_unique_ancestor": int(diag["n_unique_ancestor"][-1]),
        "final_max_ancestor_frac": final_max_anc, "max_ancestor_frac_over_time": max_anc_over_time,
        "min_ancestor_ess": min_ess,
        "grid": ref["grid"], "ref_free_energy": ref["free_energy"], "ref_mean_force": ref["mean_force"],
        "ref_p_boltzmann": pref,
        "times": ts["times"], "l2_f_t": ts["l2_f_t"], "l2_fp_t": ts["l2_fp_t"],
        "repl_cumulative": diag["repl_cumulative"],
        "frac_compact": diag["frac_compact"], "frac_transition": diag["frac_transition"],
        "frac_stretched": diag["frac_stretched"],
        "ancestor_ess_t": diag["ancestor_ess"], "max_ancestor_frac_t": diag["max_ancestor_frac"],
        "n_unique_ancestor_t": diag["n_unique_ancestor"],
        "pq_l2_t": diag["pq_l2"], "kl_pq_t": diag["kl_pq"],
        "final_mean_force": diag["mean_force"][-1], "final_pmf": diag["pmf"][-1],
        "final_p_hat": p_hat, "final_q_target": q_final,
        "final_eff_counts": (diag["eff_counts"][-1] if len(diag["eff_counts"]) else np.full_like(grid, np.nan)),
        "F_target_ema": (diag["F_target_ema"] if diag["F_target_ema"] is not None else np.full_like(grid, np.nan)),
        "birth_hist": diag["birth_hist"], "death_hist": diag["death_hist"], "hist_edges": diag["hist_edges"],
        # adaptive per-event log
        "adaptive_log_step": alog.get("step", np.array([])),
        "adaptive_log_fr_rate_eff": alog.get("fr_rate_eff", np.array([])),
        "adaptive_log_support_gate": alog.get("support_gate", np.array([])),
        "adaptive_log_diversity_gate": alog.get("diversity_gate", np.array([])),
        "adaptive_log_event_gate": alog.get("event_gate", np.array([])),
        "adaptive_log_support_ema": alog.get("support_ema", np.array([])),
        "adaptive_log_ess_frac": alog.get("ess_frac", np.array([])),
        "adaptive_log_event_fraction": alog.get("event_fraction", np.array([])),
        "adaptive_log_score_std": alog.get("score_std", np.array([])),
        "adaptive_log_score_clip_fraction": alog.get("score_clip_fraction", np.array([])),
    }
    return out


# ---------------------------------------------------------------------------
# FROZEN run (Part F): freeze a stage-1 learned bias, reconstruct F.
# ---------------------------------------------------------------------------
def load_learned_bias(source_raw_dir, physics_tag, source_method, n_replicas=None, n_steps=None):
    """Seed-average the stage-1 learned (final_mean_force, final_pmf) for one
    (cell, source method). Returns (mean_force, pmf, grid, n_sources, source_ids)."""
    pats = sorted(glob.glob(os.path.join(source_raw_dir, f"*{source_method}*{physics_tag}*.npz")))
    mfs, pmfs, grids, ids = [], [], [], []
    for p in pats:
        try:
            d = np.load(p, allow_pickle=True)
        except Exception:
            continue
        # only SAMPLE-mode runs carry a learned bias; never a frozen-stage output
        if "final_mean_force" not in d.files or "final_pmf" not in d.files:
            continue
        if "mode" in d.files and str(d["mode"]) == "frozen":
            continue
        if str(d.get("method", "")) != source_method:
            continue
        if _tag_from_npz(d) != physics_tag:
            continue
        if n_replicas is not None and int(d["n_replicas"]) != int(n_replicas):
            continue
        if n_steps is not None and int(d["n_steps"]) != int(n_steps):
            continue
        if bool(d.get("had_nan", np.array(False))):
            continue
        mfs.append(np.asarray(d["final_mean_force"], float))
        pmfs.append(np.asarray(d["final_pmf"], float))
        grids.append(np.asarray(d["grid"], float))
        ids.append(str(d["run_id"]))
    if not mfs:
        return None
    mf = np.nanmean(np.vstack(mfs), axis=0)
    pmf = np.nanmean(np.vstack(pmfs), axis=0)
    return mf, pmf, grids[0], len(mfs), ids


def _tag_from_npz(d):
    def v(k):
        x = d[k]
        return x.item() if isinstance(x, np.ndarray) and x.ndim == 0 else x
    return (f"b{v('beta'):g}_h{v('h'):g}_w{v('w'):g}_n{int(v('n_dim'))}_a{v('a'):g}")


def execute_frozen_run(spec: FollowupRunSpec, base: dict, engine, cache_dir,
                       source_raw_dir, verbose=False):
    params = build_params(spec)
    sim = build_sim_followup(spec, base)
    ref = get_reference(spec, base, engine, cache_dir, verbose=verbose)
    grid = ref["grid"]

    learned = load_learned_bias(source_raw_dir, spec.physics_tag(), spec.frozen_source_method)
    if learned is None:
        raise FileNotFoundError(
            f"no stage-1 learned bias for {spec.physics_tag()} method={spec.frozen_source_method} "
            f"in {source_raw_dir}")
    mf, pmf, src_grid, n_src, src_ids = learned

    ic = core.lattice_initial_conditions(params, sim.n_replicas, engine.device, engine.dtype, seed=sim.seed)
    t0 = time.perf_counter()
    fz = core.run_frozen_bias_gpu(params, sim, engine, mf, pmf, initial_q=ic, verbose=verbose)

    # reconstruct vs TI reference (eval only)
    mask = core.eval_window_mask_np(grid, sim)
    F_recon = np.asarray(fz["F_recon"], float)
    F_recon_al = core.align_additive_constant_np(F_recon, ref["free_energy"], grid, mask=mask)
    l2_f = core.profile_l2_error_np(F_recon_al, ref["free_energy"], grid, mask=mask)
    # the frozen online L2 of the LEARNED bias itself (B(z)=pmf aligned to ref)
    pmf_al = core.align_additive_constant_np(np.asarray(pmf, float), ref["free_energy"], grid, mask=mask)
    learned_l2_f = core.profile_l2_error_np(pmf_al, ref["free_energy"], grid, mask=mask)

    had_nan = bool(not np.isfinite(l2_f) or np.isnan(F_recon).any())
    out = {
        "run_id": spec.run_id(), "spec_hash": spec.spec_hash(),
        "spec_json": json.dumps(asdict(spec), sort_keys=True),
        "study": spec.study, "stage": spec.stage, "mode": "frozen",
        "name": spec.name, "method": spec.method,
        "frozen_source_method": spec.frozen_source_method, "n_bias_sources": int(n_src),
        "seed": spec.seed, "n_steps": spec.n_steps, "n_replicas": spec.n_replicas,
        "beta": spec.beta, "h": spec.h, "w": spec.w, "n_dim": spec.n_dim, "M": spec.M,
        "a": spec.a, "sigma": spec.sigma, "epsilon": spec.epsilon, "beta_h": spec.beta * spec.h,
        "config_hash": sim.config_hash(), "core_version": "wca_followup_v1",
        "runtime_seconds": fz["runtime_seconds"], "wall_seconds": time.perf_counter() - t0,
        "device": str(core.DEVICE), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "had_nan": had_nan,
        # headline: reconstructed-F error from the frozen bias, and the learned-bias error
        "l2_f": l2_f, "frozen_recon_l2_f": l2_f, "learned_bias_l2_f": learned_l2_f,
        "n_marginal_snapshots": int(fz["n_marginal_snapshots"]),
        "grid": grid, "ref_free_energy": ref["free_energy"], "ref_mean_force": ref["mean_force"],
        "p_B": fz["p_B"], "F_recon": F_recon_al, "learned_pmf": pmf_al,
        "learned_mean_force": np.asarray(mf, float),
        "source_run_ids": np.asarray(src_ids, dtype=object),
    }
    return out


# ---------------------------------------------------------------------------
# Stage -> FollowupRunSpec expansion.
# ---------------------------------------------------------------------------
PHYSICS_AXES = ("beta", "h", "w", "n_dim", "a", "sigma", "epsilon")
_ADAPTIVE_FIELDS = (
    "adaptive_base_fr_rate", "adaptive_support_mode", "adaptive_support_lo",
    "adaptive_support_hi", "adaptive_support_ema_rate", "adaptive_gate_warmup_steps",
    "adaptive_neff_target_strategy",
    "adaptive_neff_target_frac", "adaptive_tv_scale", "adaptive_ess_full_threshold",
    "adaptive_ess_stop_threshold", "adaptive_event_backoff_threshold",
    "adaptive_event_backoff_factor", "adaptive_min_rate", "adaptive_max_rate")


def effective_base(cfg: dict, stage: str) -> dict:
    base = dict(cfg.get("base", {}))
    st = cfg.get("stages", {}).get(stage, {})
    base.update(st.get("base_overrides", {}))
    return base


def _method_knobs(cfg: dict, name: str):
    base = cfg.get("base", {})
    m = cfg["methods"][name]
    knobs = dict(
        fr_rate=m.get("fr_rate", base.get("fr_rate", 0.0)),
        target_ema_rate=m.get("target_ema_rate", base.get("target_ema_rate", 0.005)),
        max_event_fraction=m.get("max_event_fraction", base.get("max_event_fraction", 0.02)),
        fr_every=m.get("fr_every", base.get("fr_every", 5)),
        fr_start_steps=m.get("fr_start_steps", base.get("fr_start_steps", 20000)),
        score_clip=m.get("score_clip", base.get("score_clip", 2.0)),
    )
    adaptive = dict(cfg.get("adaptive_defaults", {}))
    adaptive.update({k: m[k] for k in _ADAPTIVE_FIELDS if k in m})
    return m["type"], knobs, adaptive


def _physics_settings(cfg: dict, stage: str) -> list:
    st = cfg["stages"][stage]
    defaults = dict(cfg.get("system_defaults", {}))
    for k, v in dict(n_dim=10, a=1.5, sigma=1.0, epsilon=1.0, w=2.0, beta=1.0, h=2.0).items():
        defaults.setdefault(k, v)
    settings, seen = [], set()

    def _add(d):
        phys = dict(defaults)
        phys.update(d)
        phys = {k: phys[k] for k in PHYSICS_AXES}
        key = tuple(round(float(phys[k]), 9) if k != "n_dim" else int(phys[k]) for k in PHYSICS_AXES)
        if key not in seen:
            seen.add(key)
            settings.append(phys)

    grid = st.get("grid", {})
    if grid:
        import itertools
        axes = [k for k in PHYSICS_AXES if k in grid]
        for combo in itertools.product(*[grid[k] for k in axes]):
            _add(dict(zip(axes, combo)))
    for cell in st.get("cells", []):
        _add(cell)
    if not grid and not st.get("cells"):
        _add({})
    return settings


def _mk_spec(study, stage, mode, name, mtype, knobs, adaptive, seed, n_steps,
             n_replicas, save_every, phys, frozen_source_method="", frozen_source_study=""):
    # Adaptive knobs enter the spec (and therefore the run_id) ONLY for the
    # adaptive method, so retuning them never orphans the abf / fixed-FR runs.
    if mtype == "fr_estimated_adaptive":
        ad = {k: adaptive[k] for k in _ADAPTIVE_FIELDS if k in adaptive}
    else:
        ad = {}
    return FollowupRunSpec(
        study=study, stage=stage, mode=mode, name=name, method=mtype, seed=int(seed),
        n_steps=int(n_steps), n_replicas=int(n_replicas), save_every=int(save_every),
        beta=float(phys["beta"]), h=float(phys["h"]), w=float(phys["w"]),
        n_dim=int(phys["n_dim"]), a=float(phys["a"]), sigma=float(phys["sigma"]),
        epsilon=float(phys["epsilon"]),
        fr_rate=float(knobs["fr_rate"]), target_ema_rate=float(knobs["target_ema_rate"]),
        max_event_fraction=float(knobs["max_event_fraction"]), fr_every=int(knobs["fr_every"]),
        fr_start_steps=int(knobs["fr_start_steps"]), score_clip=float(knobs["score_clip"]),
        frozen_source_method=frozen_source_method, frozen_source_study=frozen_source_study,
        **ad)


def expand_stage(cfg: dict, stage: str) -> list:
    """Expand one stage into deduplicated FollowupRunSpecs.

    sample mode : {physics} x {methods} x {seeds} x {budgets} (budget = (N, n_steps)
                  pairs; if absent uses the stage n_steps/n_replicas).
    frozen mode : {physics} x {source methods} x {frozen seeds}.
    """
    st = cfg["stages"][stage]
    study = cfg.get("experiment_name", "followup")
    mode = st.get("mode", cfg.get("mode", "sample"))
    base = effective_base(cfg, stage)
    save_every = int(base.get("save_every", 2500))
    physics = _physics_settings(cfg, stage)
    specs, seen = [], set()

    if mode == "frozen":
        n_steps = int(st.get("n_steps", 120000))
        n_replicas = int(st.get("n_replicas", 512))
        fseeds = list(st.get("seeds", [0, 1, 2, 3, 4, 5, 6, 7]))
        src_methods = list(st.get("source_methods", ["abf", "fr_estimated", "fr_estimated_adaptive"]))
        src_study = st.get("source_study", cfg.get("source_study", ""))
        # frozen runs do not resample; give them inert FR knobs
        knobs = dict(fr_rate=0.0, target_ema_rate=0.005, max_event_fraction=0.0,
                     fr_every=int(base.get("fr_every", 5)), fr_start_steps=int(1e12), score_clip=2.0)
        for phys in physics:
            for sm in src_methods:
                for seed in fseeds:
                    s = _mk_spec(study, stage, "frozen", f"frozen_{sm}", "abf", knobs, {},
                                 seed, n_steps, n_replicas, save_every, phys,
                                 frozen_source_method=sm, frozen_source_study=src_study)
                    if s.run_id() not in seen:
                        seen.add(s.run_id())
                        specs.append(s)
        return specs

    # sample mode
    seeds = list(st.get("seeds", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
    methods = list(st.get("methods", ["abf", "fr_estimated", "fr_estimated_adaptive"]))
    # budgets: list of {n_replicas, n_steps, label}; default single (stage values)
    budgets = st.get("budgets")
    if not budgets:
        budgets = [dict(n_replicas=int(st.get("n_replicas", 1024)),
                        n_steps=int(st.get("n_steps", 120000)))]
    for phys in physics:
        for mname in methods:
            mtype, knobs, adaptive = _method_knobs(cfg, mname)
            for b in budgets:
                nrep = int(b.get("n_replicas", st.get("n_replicas", 1024)))
                nstp = int(b.get("n_steps", st.get("n_steps", 120000)))
                for seed in seeds:
                    s = _mk_spec(study, stage, "sample", mname, mtype, knobs, adaptive,
                                 seed, nstp, nrep, save_every, phys)
                    if s.run_id() not in seen:
                        seen.add(s.run_id())
                        specs.append(s)
    return specs


def distinct_physics(specs) -> list:
    out, seen = [], set()
    for s in specs:
        k = s.physics_tag()
        if k not in seen:
            seen.add(k)
            out.append(s)
    return out


def load_yaml(path: str) -> dict:
    import yaml
    with open(path) as fh:
        return yaml.safe_load(fh)
