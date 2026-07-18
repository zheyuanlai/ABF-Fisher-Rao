"""OPES run specs + per-run execution (own IDs; resume-safe, additive).

Deliberately independent of ``FollowupRunSpec``: OPES has its own hyperparameters
(barrier / pace / sigma / gamma) and its own ``spec_hash`` / ``run_id``, so adding
OPES never perturbs the existing follow-up run_ids (whose hash is baked into every
npz filename).  Physics tagging matches the phase/follow-up studies exactly, so the
cached TI references are shared and never recomputed.

``execute_opes_run`` returns the SAME ``out`` dict schema as
``wca_followup_jobs.execute_sample_run`` (FR-only fields set to neutral values),
so the existing analysis/summary code consumes OPES rows unchanged.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, asdict, field

import numpy as np

import wca_abffr_core as core
import wca_phase_jobs as pj
import wca_followup_jobs as fj
import opes_wca as ow


@dataclass
class OPESRunSpec:
    study: str
    stage: str
    name: str                # method label (e.g. "opes")
    seed: int
    n_steps: int
    n_replicas: int
    save_every: int
    # ---- physics (identical fields/semantics to FollowupRunSpec) ----
    beta: float
    h: float
    w: float
    n_dim: int
    a: float
    sigma_wca: float
    epsilon: float
    # ---- OPES knobs ----
    barrier: float
    pace: int
    sigma: float             # kernel bandwidth in CV units
    gamma: float             # bias factor; inf => flat-target ablation
    gamma_from_barrier: bool = True
    sigma_mode: str = "fixed"
    warmup_steps: int = 8000
    method: str = "opes"
    mode: str = "sample"

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
        g = "inf" if self.gamma == float("inf") else f"{self.gamma:g}"
        return (f"{self.stage}__{self.name}__{self.physics_tag()}"
                f"__seed{self.seed}__N{self.n_replicas}__T{self.n_steps}"
                f"__b{self.barrier:g}_p{self.pace}_s{self.sigma:g}_g{g}__{self.spec_hash()}")


def build_params(spec: OPESRunSpec) -> "core.DimerWCAParams":
    return core.DimerWCAParams(
        n_dim=int(spec.n_dim), a=float(spec.a), sigma=float(spec.sigma_wca),
        epsilon=float(spec.epsilon), h=float(spec.h), w=float(spec.w), beta=float(spec.beta))


def build_sim(spec: OPESRunSpec, base: dict) -> "core.SimConfig":
    """SimConfig for OPES: reuse the follow-up base (grid, dt, walls, bandwidths, TI)
    but with OPES n_replicas/n_steps/seed/save_every. FR knobs are irrelevant here."""
    kw = dict(base)
    kw.pop("ti", None)
    kw.update(n_replicas=int(spec.n_replicas), n_steps=int(spec.n_steps),
              seed=int(spec.seed), save_every=int(spec.save_every))
    valid = core.SimConfig.__dataclass_fields__.keys()
    kw = {k: v for k, v in kw.items() if k in valid}
    return core.SimConfig(**kw)


def engine_key(spec: OPESRunSpec):
    return (int(spec.n_dim), float(spec.a), float(spec.sigma_wca),
            float(spec.epsilon), float(spec.h), float(spec.w))


def get_engine(spec: OPESRunSpec, engines: dict):
    k = engine_key(spec)
    if k not in engines:
        engines[k] = core.WCADimerEngine(build_params(spec), core.DEVICE, core.DTYPE)
    return engines[k]


def get_reference(spec: OPESRunSpec, base: dict, engine, cache_dir, verbose=False):
    """Shared TI reference (identical cache key to phase/follow-up: physics_tag+grid)."""
    sim = build_sim(spec, base)
    params = build_params(spec)
    ti = pj.build_ti_config(base, sim)
    path = pj.ti_cache_path(cache_dir, spec, sim.n_grid)
    return core.load_or_compute_ti_reference(path, params, sim, ti, engine, verbose=verbose)


def run_npz_path(raw_dir: str, spec: OPESRunSpec) -> str:
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


def save_failure(raw_dir: str, spec: OPESRunSpec, exc: Exception):
    fail_dir = os.path.join(raw_dir, "_failures")
    os.makedirs(fail_dir, exist_ok=True)
    path = os.path.join(fail_dir, spec.run_id() + ".json")
    with open(path, "w") as fh:
        json.dump({"run_id": spec.run_id(), "spec": asdict(spec),
                   "error": repr(exc), "time": time.time()}, fh, indent=2)
    return path


def execute_opes_run(spec: OPESRunSpec, base: dict, engine, cache_dir, verbose=False):
    """Run one OPES sampler; emit the same `out` schema as execute_sample_run."""
    params = build_params(spec)
    sim = build_sim(spec, base)
    ref = get_reference(spec, base, engine, cache_dir, verbose=verbose)
    ic = core.lattice_initial_conditions(params, sim.n_replicas, engine.device, engine.dtype, seed=sim.seed)

    opes_cfg = ow.build_opes_config(
        sim, params, barrier=spec.barrier, pace=spec.pace, sigma=spec.sigma,
        gamma=spec.gamma, gamma_from_barrier=spec.gamma_from_barrier,
        sigma_mode=spec.sigma_mode, warmup_steps=spec.warmup_steps)

    t0 = time.perf_counter()
    diag = ow.run_opes_gpu(params, sim, opes_cfg, engine, initial_q=ic,
                           collect_diagnostics=True, verbose=verbose, track_crossings=True)
    fin = core.final_l2_errors(diag, ref, sim)
    ts = core.timeseries_l2(diag, ref, sim)
    # secondary: native OPES reweight estimator L2 (diagnostic column)
    grid_r = ref["grid"]; mask_r = core.eval_window_mask_np(grid_r, sim)
    if len(diag.get("pmf_reweight", [])):
        fe_rw = core.align_additive_constant_np(diag["pmf_reweight"][-1], ref["free_energy"], grid_r, mask=mask_r)
        l2_f_reweight = core.profile_l2_error_np(fe_rw, ref["free_energy"], grid_r, mask=mask_r)
        l2_fp_reweight = core.profile_l2_error_np(diag["mean_force_reweight"][-1], ref["mean_force"], grid_r, mask=mask_r)
    else:
        l2_f_reweight = l2_fp_reweight = float("nan")

    grid = ref["grid"]
    mask = core.eval_window_mask_np(grid, sim)
    p_hat = np.asarray(diag["p_hat"][-1], dtype=float) if len(diag["p_hat"]) else np.full_like(grid, np.nan)
    uniform = np.full_like(grid, 1.0 / (grid[-1] - grid[0]))
    marginal_l2_uniform = core.profile_l2_error_np(p_hat, uniform, grid, mask=mask)
    fe = np.asarray(ref["free_energy"], dtype=float)
    log_pref = -float(spec.beta) * (fe - fe.max())
    pref = np.exp(log_pref); pref = pref / np.trapezoid(pref, grid)
    marginal_l2_ref = core.profile_l2_error_np(p_hat, pref, grid, mask=mask)

    had_nan = bool(np.isnan(diag["mean_force"][-1]).any() or np.isnan(diag["pmf"][-1]).any()
                   or not np.isfinite(fin["l2_f"]))
    olog = diag.get("opes_log", {})
    g = "inf" if spec.gamma == float("inf") else f"{spec.gamma:g}"
    out = {
        "run_id": spec.run_id(), "spec_hash": spec.spec_hash(),
        "spec_json": json.dumps(asdict(spec), sort_keys=True),
        "study": spec.study, "stage": spec.stage, "mode": spec.mode,
        "name": spec.name, "method": spec.method, "seed": spec.seed,
        "n_steps": spec.n_steps, "n_replicas": spec.n_replicas, "budget": spec.budget,
        "beta": spec.beta, "h": spec.h, "w": spec.w, "n_dim": spec.n_dim, "M": spec.M,
        "a": spec.a, "sigma": spec.sigma_wca, "epsilon": spec.epsilon, "beta_h": spec.beta * spec.h,
        # OPES hyperparameters (analysis columns)
        "opes_barrier": spec.barrier, "opes_pace": spec.pace, "opes_sigma": spec.sigma,
        "opes_gamma": g, "opes_gamma_from_barrier": spec.gamma_from_barrier,
        "opes_sigma_mode": spec.sigma_mode, "opes_warmup_steps": spec.warmup_steps,
        "config_hash": sim.config_hash(), "core_version": "opes_v1",
        "runtime_seconds": diag["runtime_seconds"], "wall_seconds": time.perf_counter() - t0,
        "device": str(core.DEVICE), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "had_nan": had_nan, "total_replacement_events": 0,
        "l2_f": fin["l2_f"], "l2_fp": fin["l2_fp"], "integrated_l2_f": ts["integrated_l2_f"],
        "l2_f_reweight": l2_f_reweight, "l2_fp_reweight": l2_fp_reweight, "opes_estimator": "meanforce",
        "l2_f_compact": fin["l2_f_compact"], "l2_f_transition": fin["l2_f_transition"],
        "l2_f_stretched": fin["l2_f_stretched"],
        "l2_fp_compact": fin["l2_fp_compact"], "l2_fp_transition": fin["l2_fp_transition"],
        "l2_fp_stretched": fin["l2_fp_stretched"],
        "marginal_l2_uniform": marginal_l2_uniform, "marginal_l2_ref": marginal_l2_ref,
        "marginal_l2_target": float("nan"),
        "n_compact_to_stretched": diag["n_compact_to_stretched"],
        "n_stretched_to_compact": diag["n_stretched_to_compact"],
        "n_barrier_crossings": diag["n_barrier_crossings"], "n_round_trips": diag["n_round_trips"],
        # FR-only fields -> neutral values so the schema matches
        "fr_rate": float("nan"), "fr_event_fraction": 0.0, "max_fr_event_fraction": 0.0,
        "n_fr_applications": 0, "deaths_per_fr_application": 0.0, "clones_per_fr_application": 0.0,
        "fr_score_std": float("nan"), "fr_score_absmax": float("nan"), "fr_score_clip_fraction": float("nan"),
        "final_ancestor_ess": float("nan"), "final_n_unique_ancestor": int(spec.n_replicas),
        "final_max_ancestor_frac": float("nan"), "max_ancestor_frac_over_time": float("nan"),
        "min_ancestor_ess": float("nan"),
        # OPES reweight-quality diagnostics (analogue of ancestor-ESS)
        "opes_neff_frac_final": float(olog["neff_frac"][-1]) if len(olog.get("neff_frac", [])) else float("nan"),
        "opes_neff_frac_min": float(np.nanmin(olog["neff_frac"])) if len(olog.get("neff_frac", [])) else float("nan"),
        "opes_n_kernels_final": float(olog["n_kernels"][-1]) if len(olog.get("n_kernels", [])) else float("nan"),
        "opes_bias_range_final": float(olog["bias_range"][-1]) if len(olog.get("bias_range", [])) else float("nan"),
        # profiles + time series (same keys as execute_sample_run)
        "grid": ref["grid"], "ref_free_energy": ref["free_energy"], "ref_mean_force": ref["mean_force"],
        "ref_p_boltzmann": pref,
        "times": ts["times"], "l2_f_t": ts["l2_f_t"], "l2_fp_t": ts["l2_fp_t"],
        "repl_cumulative": diag["repl_cumulative"],
        "frac_compact": diag["frac_compact"], "frac_transition": diag["frac_transition"],
        "frac_stretched": diag["frac_stretched"],
        "final_mean_force": diag["mean_force"][-1], "final_pmf": diag["pmf"][-1],
        "final_p_hat": p_hat,
        "final_eff_counts": (diag["eff_counts"][-1] if len(diag["eff_counts"]) else np.full_like(grid, np.nan)),
        # OPES per-deposit log
        "opes_log_step": olog.get("step", np.array([])),
        "opes_log_neff_frac": olog.get("neff_frac", np.array([])),
        "opes_log_n_kernels": olog.get("n_kernels", np.array([])),
        "opes_log_zed": olog.get("zed", np.array([])),
        "opes_log_sigma_cur": olog.get("sigma_cur", np.array([])),
        "opes_log_max_bias": olog.get("max_bias", np.array([])),
        "opes_log_bias_range": olog.get("bias_range", np.array([])),
    }
    return out


# ---------------------------------------------------------------------------
# Config expansion (reuses the follow-up base/physics/yaml helpers).
# ---------------------------------------------------------------------------
load_yaml = fj.load_yaml
effective_base = fj.effective_base
distinct_physics = fj.distinct_physics


def _opes_knobs(cfg: dict, name: str):
    d = dict(cfg.get("opes_defaults", {}))
    m = cfg["methods"][name]
    def g(key, default):
        return m.get(key, d.get(key, default))
    gamma = g("gamma", float("inf"))
    gamma = float("inf") if (gamma is None or gamma == float("inf") or str(gamma).lower() in (".inf", "inf")) else float(gamma)
    return dict(
        barrier=float(g("barrier", 4.0)), pace=int(g("pace", 500)),
        sigma=float(g("sigma", 0.05)), gamma=gamma,
        gamma_from_barrier=bool(g("gamma_from_barrier", True)),
        sigma_mode=str(g("sigma_mode", "fixed")), warmup_steps=int(g("warmup_steps", 10000)),
    )


def expand_stage(cfg: dict, stage: str) -> list:
    """Expand one OPES stage into deduplicated OPESRunSpecs.

    {physics} x {methods} x {barrier/pace/sigma grids} x {seeds}. Optional per-stage
    grids (barriers/paces/sigmas) sweep the OPES hyperparameters for tuning.
    """
    st = cfg["stages"][stage]
    study = cfg.get("experiment_name", "opes")
    base = effective_base(cfg, stage)
    save_every = int(base.get("save_every", 2500))
    physics = fj._physics_settings(cfg, stage)
    seeds = list(st.get("seeds", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
    methods = list(st.get("methods", ["opes"]))
    n_steps = int(st.get("n_steps", 120000))
    n_replicas = int(st.get("n_replicas", 1024))
    # tuning grids (optional); default to [None] => use the method/base knob value
    barriers = st.get("barriers", [None])
    paces = st.get("paces", [None])
    sigmas = st.get("sigmas", [None])
    specs, seen = [], set()
    for phys in physics:
        for mname in methods:
            k = _opes_knobs(cfg, mname)
            for barrier in barriers:
                for pace in paces:
                    for sigma in sigmas:
                        for seed in seeds:
                            s = OPESRunSpec(
                                study=study, stage=stage, name=mname, seed=int(seed),
                                n_steps=n_steps, n_replicas=n_replicas, save_every=save_every,
                                beta=float(phys["beta"]), h=float(phys["h"]), w=float(phys["w"]),
                                n_dim=int(phys["n_dim"]), a=float(phys["a"]),
                                sigma_wca=float(phys["sigma"]), epsilon=float(phys["epsilon"]),
                                barrier=float(barrier) if barrier is not None else k["barrier"],
                                pace=int(pace) if pace is not None else k["pace"],
                                sigma=float(sigma) if sigma is not None else k["sigma"],
                                gamma=k["gamma"], gamma_from_barrier=k["gamma_from_barrier"],
                                sigma_mode=k["sigma_mode"], warmup_steps=k["warmup_steps"])
                            if s.run_id() not in seen:
                                seen.add(s.run_id())
                                specs.append(s)
    return specs
