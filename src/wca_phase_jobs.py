"""Job expansion, run IO, and the single-run pipeline for the WCA *phase diagram*.

This is the phase-diagram analogue of ``wca_jobs.py``. The crucial difference is
that a :class:`PhaseRunSpec` carries the FULL physical parameter set --- inverse
temperature ``beta``, barrier height ``h``, well width ``w``, physical particle
count via the lattice side ``n_dim`` (so M = n_dim^2 physical particles), and the
lattice spacing ``a`` --- whereas the original production ``RunSpec`` varied only
``a`` and used the engine defaults for everything else. The phase-diagram study
therefore needs its own TI reference *per physical setting*, cached by the whole
physics tag rather than by ``a`` alone.

The sampler, estimators, FR machinery, and L2 metrics are reused verbatim from
``wca_abffr_core`` so a phase-diagram cell at the anchor physics
(beta=1, h=2, w=2, n_dim=10, a=1.5) is directly comparable to the existing
single-setting production study. ``run_sampler_gpu(..., track_crossings=True)`` is
used to obtain barrier-crossing / round-trip counts; this flag defaults to False
for the existing study, which is therefore untouched.

A *job* is one fully-specified run (one method, one seed, one physics cell, one
budget). Jobs are deterministic in their PhaseRunSpec, hashed for idempotency, and
saved one-.npz-per-run so interrupted sweeps never lose completed work.

Important constants distinguishing the two particle counts:
    M = n_dim**2     PHYSICAL particles in the WCA bath (2 are the dimer).
    N = n_replicas   independent ABF / FR replicas of the whole system.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, asdict

import numpy as np

import wca_abffr_core as core


# ---------------------------------------------------------------------------
# PhaseRunSpec: a single fully-specified phase-diagram run.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PhaseRunSpec:
    stage: str
    name: str            # method label (e.g. "fr_estimated")
    method: str          # sampler type: abf / fr_estimated / fr_uniform / fr_oracle
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
    prior_c: float = 1.0

    @property
    def M(self) -> int:
        """Physical particle count = n_dim^2."""
        return int(self.n_dim) * int(self.n_dim)

    def physics_tag(self) -> str:
        return (f"b{self.beta:g}_h{self.h:g}_w{self.w:g}"
                f"_n{int(self.n_dim)}_a{self.a:g}")

    def spec_hash(self) -> str:
        return hashlib.md5(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]

    def run_id(self) -> str:
        return (f"{self.stage}__{self.name}__{self.physics_tag()}"
                f"__seed{self.seed}__N{self.n_replicas}__T{self.n_steps}__{self.spec_hash()}")


# ---------------------------------------------------------------------------
# Physics -> core objects.
# ---------------------------------------------------------------------------
def build_params(spec: PhaseRunSpec) -> "core.DimerWCAParams":
    return core.DimerWCAParams(
        n_dim=int(spec.n_dim), a=float(spec.a), sigma=float(spec.sigma),
        epsilon=float(spec.epsilon), h=float(spec.h), w=float(spec.w), beta=float(spec.beta))


def build_sim(spec: PhaseRunSpec, base: dict) -> "core.SimConfig":
    """SimConfig from the YAML base block, overridden by this spec.

    Note: physics (beta/h/w/n_dim/a) lives in DimerWCAParams, NOT SimConfig.
    """
    kw = dict(base)
    kw.pop("ti", None)  # TI sub-block is not a SimConfig field
    kw.update(
        n_replicas=int(spec.n_replicas), n_steps=int(spec.n_steps), seed=int(spec.seed),
        save_every=int(spec.save_every), fr_rate=float(spec.fr_rate),
        target_ema_rate=float(spec.target_ema_rate),
        max_event_fraction=float(spec.max_event_fraction),
        prior_c=float(getattr(spec, 'prior_c', 1.0)), fr_every=int(spec.fr_every),
        fr_start_steps=int(spec.fr_start_steps), score_clip=float(spec.score_clip),
    )
    valid = core.SimConfig.__dataclass_fields__.keys()
    kw = {k: v for k, v in kw.items() if k in valid}
    return core.SimConfig(**kw)


def engine_key(spec: PhaseRunSpec):
    """Engine depends on geometry+potential (NOT beta, which enters elsewhere)."""
    return (int(spec.n_dim), float(spec.a), float(spec.sigma),
            float(spec.epsilon), float(spec.h), float(spec.w))


def get_engine(spec: PhaseRunSpec, engines: dict):
    k = engine_key(spec)
    if k not in engines:
        engines[k] = core.WCADimerEngine(build_params(spec), core.DEVICE, core.DTYPE)
    return engines[k]


# ---------------------------------------------------------------------------
# TI reference cache, keyed by the WHOLE physical setting (eval-only).
# ---------------------------------------------------------------------------
def ti_cache_path(cache_dir: str, spec: PhaseRunSpec, n_grid: int) -> str:
    return os.path.join(cache_dir, f"wca_ti_{spec.physics_tag()}_g{n_grid}.npz")


def build_ti_config(base: dict, sim: "core.SimConfig") -> "core.TIConfig":
    ti = dict(base.get("ti", {}))
    ti.setdefault("z_min", sim.z_min)
    ti.setdefault("z_max", sim.z_max)
    ti.setdefault("dt", sim.dt)
    valid = core.TIConfig.__dataclass_fields__.keys()
    ti = {k: v for k, v in ti.items() if k in valid}
    return core.TIConfig(**ti)


_REF_CACHE: dict = {}


def get_reference(spec: PhaseRunSpec, base: dict, engine, cache_dir="cache/phase", verbose=False):
    """Load/compute the TI reference for this spec's physics (memoised per process)."""
    sim = build_sim(spec, base)
    key = (spec.physics_tag(), int(sim.n_grid), float(sim.z_min), float(sim.z_max))
    if key in _REF_CACHE:
        return _REF_CACHE[key]
    params = build_params(spec)
    ti = build_ti_config(base, sim)
    path = ti_cache_path(cache_dir, spec, sim.n_grid)
    ref = core.load_or_compute_ti_reference(path, params, sim, ti, engine, verbose=verbose)
    _REF_CACHE[key] = ref
    return ref


# ---------------------------------------------------------------------------
# Per-run output files (one .npz per run -> interrupt-safe, idempotent).
# ---------------------------------------------------------------------------
def run_npz_path(raw_dir: str, spec: PhaseRunSpec) -> str:
    return os.path.join(raw_dir, spec.run_id() + ".npz")


def run_is_valid(path: str) -> bool:
    if not os.path.exists(path):
        return False
    try:
        d = np.load(path, allow_pickle=True)
        ok = ("l2_f" in d.files and "times" in d.files
              and np.isfinite(float(d["l2_f"])) and not bool(d.get("had_nan", np.array(False))))
        d.close()
        return bool(ok)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Extra phase-diagram metrics (RC marginal errors, FR event stats).
# ---------------------------------------------------------------------------
def n_fr_applications(spec: PhaseRunSpec) -> int:
    """Number of birth-death applications scheduled over the run."""
    if spec.method not in core.FR_METHODS:
        return 0
    fr_start, fr_every, n_steps = int(spec.fr_start_steps), max(int(spec.fr_every), 1), int(spec.n_steps)
    if n_steps < fr_start:
        return 0
    return (n_steps - fr_start) // fr_every + 1


def fr_event_stats(spec: PhaseRunSpec, steps, repl_cumulative, n_replicas):
    """Mean / max realised fraction of the population replaced per FR application.

    The mean is total replacements / (N * #applications). The max is taken over
    save-intervals: (delta replacements in interval) / (N * #applications in
    interval), i.e. the most aggressive sustained resampling window.
    """
    apps_total = n_fr_applications(spec)
    total_repl = float(repl_cumulative[-1]) if len(repl_cumulative) else 0.0
    mean_frac = (total_repl / (n_replicas * apps_total)) if apps_total > 0 else 0.0
    fr_start, fr_every = int(spec.fr_start_steps), max(int(spec.fr_every), 1)

    def apps_in(lo, hi):  # #do_fr steps with next_step in (lo, hi]
        lo2 = max(lo + 1, fr_start)
        if hi < lo2:
            return 0
        # smallest next_step >= lo2 that is congruent to fr_start mod fr_every
        rem = (lo2 - fr_start) % fr_every
        first = lo2 + ((fr_every - rem) % fr_every)
        if first > hi:
            return 0
        return (hi - first) // fr_every + 1

    max_frac = 0.0
    steps = np.asarray(steps, dtype=int)
    repl_cumulative = np.asarray(repl_cumulative, dtype=float)
    for k in range(1, len(steps)):
        apps_k = apps_in(int(steps[k - 1]), int(steps[k]))
        if apps_k <= 0:
            continue
        d_repl = repl_cumulative[k] - repl_cumulative[k - 1]
        frac_k = d_repl / (n_replicas * apps_k)
        max_frac = max(max_frac, frac_k)
    deaths_per_app = (total_repl / apps_total) if apps_total > 0 else 0.0
    return dict(fr_event_fraction=mean_frac, max_fr_event_fraction=max_frac,
                n_fr_applications=apps_total, deaths_per_fr_application=deaths_per_app,
                clones_per_fr_application=deaths_per_app)


def execute_run(spec: PhaseRunSpec, base: dict, engine, cache_dir="cache/phase", verbose=False,
                replay_counts=None, store_profiles=False, readout_bandwidths=None):
    """Run one phase-diagram job; return a flat dict of scalars + arrays to save.

    ``replay_counts`` is required by the matched-sham methods and rejected by every other
    one: it is the per-FR-opportunity replacement count its partner arm realised on the same
    seed, which the sham reproduces with the direction randomised.
    """
    params = build_params(spec)
    sim = build_sim(spec, base)
    ref = get_reference(spec, base, engine, cache_dir=cache_dir, verbose=verbose)
    ic = core.lattice_initial_conditions(params, sim.n_replicas, engine.device, engine.dtype, seed=sim.seed)
    oracle_fe = ref["free_energy"] if spec.method == "fr_oracle" else None

    t0 = time.perf_counter()
    diag = core.run_sampler_gpu(spec.method, params, sim, engine, initial_q=ic,
                                oracle_free_energy=oracle_fe, collect_diagnostics=True,
                                verbose=verbose, track_crossings=True,
                                replay_counts=replay_counts,
                                readout_bandwidths=readout_bandwidths)
    fin = core.final_l2_errors(diag, ref, sim)
    ts = core.timeseries_l2(diag, ref, sim)

    # RC marginal errors (p_hat vs uniform, FR target, Boltzmann p_ref)
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

    fre = fr_event_stats(spec, diag["steps"], diag["repl_cumulative"], sim.n_replicas)

    # Case IX retained only scored scalars, so its headline could never be rescored against
    # a corrected reference -- the whole reason Stage C needs fresh dynamics. Storing the
    # profile time series (49 x 160 float32 ~ 31 kB/run) removes that trap for good.
    # Purely ADDITIVE: no existing key changes, so v1 artifacts stay valid.
    profiles = {}
    if store_profiles:
        profiles = dict(
            pmf_t=np.asarray(diag["pmf"], dtype=np.float32),
            mean_force_t=np.asarray(diag["mean_force"], dtype=np.float32),
            profile_steps=np.asarray(diag["steps"]),
            profile_times=np.asarray(diag["times"]),
            reference_free_energy=np.asarray(ref["free_energy"], dtype=np.float64),
            reference_mean_force=np.asarray(ref["mean_force"], dtype=np.float64),
            reference_label=str(ref.get("label", "")))

    # genealogy summaries
    maf = np.asarray(diag["max_ancestor_frac"], dtype=float)
    ess = np.asarray(diag["ancestor_ess"], dtype=float)
    final_max_anc = float(maf[-1]) if maf.size else float("nan")
    max_anc_over_time = float(np.nanmax(maf)) if np.isfinite(maf).any() else float("nan")
    min_ess = float(np.nanmin(ess)) if np.isfinite(ess).any() else float("nan")

    had_nan = bool(np.isnan(diag["mean_force"][-1]).any() or np.isnan(diag["pmf"][-1]).any()
                   or not np.isfinite(fin["l2_f"]))

    out = {
        # identity / metadata
        "run_id": spec.run_id(), "spec_hash": spec.spec_hash(),
        "spec_json": json.dumps(asdict(spec), sort_keys=True),
        "stage": spec.stage, "name": spec.name, "method": spec.method, "seed": spec.seed,
        "n_steps": spec.n_steps, "n_replicas": spec.n_replicas,
        # physics
        "beta": spec.beta, "h": spec.h, "w": spec.w, "n_dim": spec.n_dim, "M": spec.M,
        "a": spec.a, "sigma": spec.sigma, "epsilon": spec.epsilon, "beta_h": spec.beta * spec.h,
        # FR knobs
        "fr_rate": spec.fr_rate, "target_ema_rate": spec.target_ema_rate,
        "max_event_fraction": spec.max_event_fraction, "fr_every": spec.fr_every,
        "fr_start_steps": spec.fr_start_steps, "score_clip": spec.score_clip,
        "config_hash": sim.config_hash(), "core_version": "wca_phase_v1",
        "runtime_seconds": diag["runtime_seconds"], "wall_seconds": time.perf_counter() - t0,
        "device": str(core.DEVICE),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "had_nan": had_nan, "total_replacement_events": diag["total_replacement_events"],
        # final scalar metrics
        "l2_f": fin["l2_f"], "l2_fp": fin["l2_fp"], "integrated_l2_f": ts["integrated_l2_f"],
        "l2_f_compact": fin["l2_f_compact"], "l2_f_transition": fin["l2_f_transition"],
        "l2_f_stretched": fin["l2_f_stretched"],
        "l2_fp_compact": fin["l2_fp_compact"], "l2_fp_transition": fin["l2_fp_transition"],
        "l2_fp_stretched": fin["l2_fp_stretched"],
        "marginal_l2_uniform": marginal_l2_uniform, "marginal_l2_target": marginal_l2_target,
        "marginal_l2_ref": marginal_l2_ref,
        # exploration / transitions
        "n_compact_to_stretched": diag["n_compact_to_stretched"],
        "n_stretched_to_compact": diag["n_stretched_to_compact"],
        "n_barrier_crossings": diag["n_barrier_crossings"],
        "n_round_trips": diag["n_round_trips"],
        # FR event stats
        "fr_event_fraction": fre["fr_event_fraction"],
        "max_fr_event_fraction": fre["max_fr_event_fraction"],
        "n_fr_applications": fre["n_fr_applications"],
        "deaths_per_fr_application": fre["deaths_per_fr_application"],
        "clones_per_fr_application": fre["clones_per_fr_application"],
        # genealogy
        "final_ancestor_ess": float(diag["ancestor_ess"][-1]),
        "final_n_unique_ancestor": int(diag["n_unique_ancestor"][-1]),
        "final_max_ancestor_frac": final_max_anc, "max_ancestor_frac_over_time": max_anc_over_time,
        "min_ancestor_ess": min_ess,
        # The §3.3 gate statistic, measured over a fixed ancestry window rather than over the
        # whole run. `min_ancestor_ess` above traces lineage from t=0, so it decays
        # monotonically with run length for any birth-death process and cannot be compared to
        # a fixed 0.30 floor; the gateway confirmatory that set that floor used a 4000-step
        # window. Both are stored so the two are never conflated again.
        "min_ancestor_ess_window": float(diag.get("min_ancestor_ess_window", float("nan"))),
        "ess_window_steps": int(diag.get("ess_window_steps", 0)),
        # grid + reference (small; duplicated per run for self-containment)
        "grid": ref["grid"], "ref_free_energy": ref["free_energy"], "ref_mean_force": ref["mean_force"],
        "ref_p_boltzmann": pref,
        # timeseries
        "times": ts["times"], "l2_f_t": ts["l2_f_t"], "l2_fp_t": ts["l2_fp_t"],
        "repl_cumulative": diag["repl_cumulative"],
        "frac_compact": diag["frac_compact"], "frac_transition": diag["frac_transition"],
        "frac_stretched": diag["frac_stretched"],
        "ancestor_ess_t": diag["ancestor_ess"], "max_ancestor_frac_t": diag["max_ancestor_frac"],
        "n_unique_ancestor_t": diag["n_unique_ancestor"],
        "pq_l2_t": diag["pq_l2"], "kl_pq_t": diag["kl_pq"],
        # final profiles + mechanism
        "final_mean_force": diag["mean_force"][-1], "final_pmf": diag["pmf"][-1],
        "final_p_hat": p_hat, "final_q_target": q_final,
        "final_eff_counts": (diag["eff_counts"][-1] if len(diag["eff_counts"]) else np.full_like(grid, np.nan)),
        "F_target_ema": (diag["F_target_ema"] if diag["F_target_ema"] is not None else np.full_like(grid, np.nan)),
        "birth_hist": diag["birth_hist"], "death_hist": diag["death_hist"], "hist_edges": diag["hist_edges"],
        # the per-opportunity replacement schedule, so a matched sham can replay this run
        "fr_event_counts": diag["fr_event_counts"],
        "sham_partner": (diag["sham_partner"] or ""),
        "sham_replayed_events": diag["sham_replayed_events"],
    }
    out.update(profiles)
    if readout_bandwidths:
        # Read-out bank (inert diagnostics): extra-bandwidth profiles + raw binned sums,
        # so the read-out bandwidth can be swept OFFLINE at fixed dynamics.
        out["readout_bandwidths"] = np.asarray([float(h) for h in readout_bandwidths])
        out["abf_bandwidth_online"] = float(sim.abf_bandwidth)
        out["abf_smooth_sigma"] = float(sim.abf_smooth_sigma)
        for h, series in diag.get("readout_mean_force", {}).items():
            out[f"readout_mean_force_t__h{h:g}"] = np.asarray(series, dtype=np.float64)
        if "raw_fsum" in diag:
            out["raw_fsum_t"] = np.asarray(diag["raw_fsum"], dtype=np.float64)
            out["raw_csum_t"] = np.asarray(diag["raw_csum"], dtype=np.float64)
    return out


def save_run(path: str, out: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **out)
    os.replace(tmp, path)  # atomic -> a half-written file is never seen as valid


def load_run(path: str) -> dict:
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def save_failure(raw_dir: str, spec: PhaseRunSpec, exc: Exception):
    """Write a small failure JSON so a crashed run is recorded, not silently lost."""
    fail_dir = os.path.join(raw_dir, "_failures")
    os.makedirs(fail_dir, exist_ok=True)
    path = os.path.join(fail_dir, spec.run_id() + ".json")
    with open(path, "w") as fh:
        json.dump({"run_id": spec.run_id(), "spec": asdict(spec),
                   "error": repr(exc), "time": time.time()}, fh, indent=2)
    return path


# ---------------------------------------------------------------------------
# Stage -> PhaseRunSpec expansion.
# ---------------------------------------------------------------------------
PHYSICS_AXES = ("beta", "h", "w", "n_dim", "a", "sigma", "epsilon")


def effective_base(cfg: dict, stage: str) -> dict:
    base = dict(cfg.get("base", {}))
    st = cfg.get("stages", {}).get(stage, {})
    base.update(st.get("base_overrides", {}))
    return base


def _method_knobs(cfg: dict, name: str):
    base = cfg.get("base", {})
    m = cfg["methods"][name]
    knobs = dict(
        fr_rate=m.get("fr_rate", 0.0),
        target_ema_rate=m.get("target_ema_rate", base.get("target_ema_rate", 0.005)),
        max_event_fraction=m.get("max_event_fraction", base.get("max_event_fraction", 0.02)),
        fr_every=m.get("fr_every", base.get("fr_every", 5)),
        fr_start_steps=m.get("fr_start_steps", base.get("fr_start_steps", 20000)),
        score_clip=m.get("score_clip", base.get("score_clip", 2.0)),
    )
    return m["type"], knobs


def _physics_settings(cfg: dict, stage: str) -> list:
    """Build the list of physics dicts (factorial grid + explicit extra points)."""
    st = cfg["stages"][stage]
    defaults = dict(cfg.get("system_defaults", {}))
    for k, v in dict(n_dim=10, a=1.5, sigma=1.0, epsilon=1.0, w=2.0, beta=1.0, h=2.0).items():
        defaults.setdefault(k, v)
    settings = []
    seen = set()

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
        axes = [k for k in PHYSICS_AXES if k in grid]
        import itertools
        for combo in itertools.product(*[grid[k] for k in axes]):
            _add(dict(zip(axes, combo)))
    else:
        _add({})  # single default cell if no grid given
    for extra in st.get("extra_points", []):
        _add(extra)
    return settings


def _mk_spec(stage, name, mtype, knobs, seed, n_steps, n_replicas, save_every, phys) -> PhaseRunSpec:
    return PhaseRunSpec(
        stage=stage, name=name, method=mtype, seed=int(seed), n_steps=int(n_steps),
        n_replicas=int(n_replicas), save_every=int(save_every),
        beta=float(phys["beta"]), h=float(phys["h"]), w=float(phys["w"]),
        n_dim=int(phys["n_dim"]), a=float(phys["a"]), sigma=float(phys["sigma"]),
        epsilon=float(phys["epsilon"]),
        fr_rate=float(knobs["fr_rate"]), target_ema_rate=float(knobs["target_ema_rate"]),
        max_event_fraction=float(knobs["max_event_fraction"]), fr_every=int(knobs["fr_every"]),
        fr_start_steps=int(knobs["fr_start_steps"]), score_clip=float(knobs["score_clip"]))


def expand_stage(cfg: dict, stage: str) -> list:
    """Expand one stage into a deduplicated list of PhaseRunSpecs.

    Cartesian product of {physics settings} x {methods} x {seeds}.
    """
    st = cfg["stages"][stage]
    base = effective_base(cfg, stage)
    save_every = int(base.get("save_every", 2500))
    knob_ov = dict(st.get("knob_overrides", {}))
    n_steps = int(st.get("n_steps", 250000))
    n_replicas = int(st.get("n_replicas", 1024))
    seeds = list(st.get("seeds", [0, 1, 2, 3, 4]))
    methods = list(st.get("methods", ["abf", "fr_estimated"]))
    physics = _physics_settings(cfg, stage)

    specs, seen = [], set()
    for phys in physics:
        for mname in methods:
            mtype, knobs = _method_knobs(cfg, mname)
            knobs.update(knob_ov)
            for seed in seeds:
                s = _mk_spec(stage, mname, mtype, knobs, seed, n_steps, n_replicas, save_every, phys)
                rid = s.run_id()
                if rid not in seen:
                    seen.add(rid)
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
