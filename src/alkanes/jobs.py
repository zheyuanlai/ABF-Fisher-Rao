"""Job expansion, reference caching, single-job execution and IO for the alkanes.

One *job* = one (molecule, method, physics cell, initialisation, stage) running a
batch of matched ``seeds`` in a single GPU process (seed-batched; leading dim R).
Jobs are deterministic in their :class:`AlkaneRunSpec`, hashed for idempotency, and
saved one atomic ``.npz`` per job so interrupted sweeps never lose work.

References (evaluation only) are cached per physics under ``cache/alkanes/``.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, asdict, field
from typing import List, Tuple

import numpy as np
import torch

from . import core, metrics, opes as opesmod, periodic as per, potentials as pot, reference as refmod
from .cv import DihedralCV

PI = math.pi
GAUCHE = math.radians(116.57)


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AlkaneRunSpec:
    stage: str
    molecule: str            # "butane" | "pentane"
    name: str                # method label
    method: str              # abf | fr_estimated | fr_uniform | fr_oracle | opes
    init_mode: str           # "trans" | "dispersed"
    seeds: Tuple[int, ...]
    # physics
    beta: float
    sigma: float
    epsilon: float
    decouple: bool
    # dynamics / sim
    dt: float
    n_steps: int
    n_replicas: int
    save_every: int
    rng_seed: int
    n_grid: int
    n_grid2: int
    abf_bandwidth: float
    kde_bandwidth: float
    abf_warmup_steps: int
    estimator_burn_in_steps: int
    abf_force_clip: float
    force_clip: float
    # FR
    fr_rate: float
    target_ema_rate: float
    max_event_fraction: float
    fr_every: int
    fr_start_steps: int
    score_clip: float
    # OPES
    opes_barrier: float
    opes_pace: int
    opes_sigma: float
    opes_gamma: float
    # reference sampling
    ref_n_samples: int = 40000

    @property
    def n_atoms(self):
        return 4 if self.molecule == "butane" else 5

    def physics_tag(self):
        return (f"{self.molecule}_b{self.beta:g}_s{self.sigma:g}"
                f"_{'dec' if self.decouple else 'full'}_g{self.n_grid}")

    def spec_hash(self):
        return hashlib.md5(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]

    def run_id(self):
        return (f"{self.stage}__{self.molecule}__{self.name}__{self.init_mode}"
                f"__b{self.beta:g}__s{self.sigma:g}__N{self.n_replicas}__T{self.n_steps}"
                f"__ns{len(self.seeds)}__{self.spec_hash()}")


def build_params(spec: AlkaneRunSpec) -> pot.AlkaneParams:
    return pot.AlkaneParams(n_atoms=spec.n_atoms, beta=spec.beta, sigma=spec.sigma,
                            epsilon=spec.epsilon, decouple=spec.decouple,
                            force_clip=spec.force_clip)


def build_sim(spec: AlkaneRunSpec) -> core.AlkaneSimConfig:
    return core.AlkaneSimConfig(
        dt=spec.dt, n_steps=spec.n_steps, n_replicas=spec.n_replicas,
        save_every=spec.save_every, rng_seed=spec.rng_seed, n_grid=spec.n_grid,
        n_grid2=spec.n_grid2, abf_bandwidth=spec.abf_bandwidth, kde_bandwidth=spec.kde_bandwidth,
        abf_warmup_steps=spec.abf_warmup_steps, estimator_burn_in_steps=spec.estimator_burn_in_steps,
        abf_force_clip=spec.abf_force_clip, fr_rate=spec.fr_rate,
        target_ema_rate=spec.target_ema_rate, max_event_fraction=spec.max_event_fraction,
        fr_every=spec.fr_every, fr_start_steps=spec.fr_start_steps, score_clip=spec.score_clip)


def build_opes_cfg(spec: AlkaneRunSpec) -> opesmod.PeriodicOPESConfig:
    return opesmod.PeriodicOPESConfig(
        n_grid=spec.n_grid, beta=spec.beta, barrier=spec.opes_barrier, pace=spec.opes_pace,
        sigma=spec.opes_sigma, gamma=spec.opes_gamma, gamma_from_barrier=True,
        bias_force_clip=spec.abf_force_clip, warmup_steps=spec.abf_warmup_steps)


# ---------------------------------------------------------------------------
# Reference cache (evaluation only)
# ---------------------------------------------------------------------------
_REF_CACHE: dict = {}


def reference_path(cache_dir, spec: AlkaneRunSpec):
    return os.path.join(cache_dir, f"ref_{spec.physics_tag()}_g2{spec.n_grid2}_ns{spec.ref_n_samples}.npz")


def build_reference(spec: AlkaneRunSpec, device, cache_dir="cache/alkanes", verbose=False):
    """Independent reference: 1-D F(phi1)/F'(phi1) on the sim grid, plus (pentane)
    the joint + conditional on the coarse grid.  Cached per physics."""
    key = (spec.physics_tag(), spec.n_grid2, spec.ref_n_samples)
    if key in _REF_CACHE:
        return _REF_CACHE[key]
    path = reference_path(cache_dir, spec)
    if os.path.exists(path):
        d = np.load(path, allow_pickle=True)
        ref = {k: d[k] for k in d.files}
        _REF_CACHE[key] = ref
        return ref
    p = build_params(spec)
    grid, dphi = per.periodic_grid(spec.n_grid, device=device, dtype=torch.float64)
    ref = {"grid": grid.cpu().numpy(), "dphi": float(dphi)}
    if spec.molecule == "butane" or spec.decouple:
        # butane full == V4 (no LJ); decoupled == V4 always
        F = pot.V4(grid, p) - pot.V4(grid, p).mean()
        ref["F"] = F.cpu().numpy()
        ref["Fprime"] = pot.V4_prime(grid, p).cpu().numpy()
    else:
        # pentane full: QMC joint marginalised for the 1-D reference
        g2, dphi2 = per.periodic_grid(spec.n_grid2, device=device, dtype=torch.float64)
        Rj = refmod.qmc_reference_pentane(grid, g2, p, n_samples=spec.ref_n_samples,
                                          seed=987 + spec.n_grid, device=device)
        F1 = refmod.marginalize_joint_to_phi1(Rj["F"], grid.cpu(), g2.cpu(), spec.beta)
        ref["F"] = F1 - F1.mean()
        # periodic derivative of F1 on the grid
        Fp = np.zeros_like(F1)
        Fp[1:-1] = (F1[2:] - F1[:-2]) / (grid.cpu().numpy()[2:] - grid.cpu().numpy()[:-2])
        Fp[0] = (F1[1] - F1[-1]) / (2 * dphi); Fp[-1] = (F1[0] - F1[-2]) / (2 * dphi)
        ref["Fprime"] = Fp
    if spec.molecule == "pentane":
        g2, dphi2 = per.periodic_grid(spec.n_grid2, device=device, dtype=torch.float64)
        pcell = build_params(spec)
        Rc = refmod.qmc_reference_pentane(g2, g2, pcell, n_samples=spec.ref_n_samples,
                                          seed=123 + spec.n_grid2, device=device)
        ref["joint_F"] = Rc["F"]
        ref["grid2"] = g2.cpu().numpy(); ref["dphi2"] = float(dphi2)
        ref["cond"] = refmod.conditional_phi2_given_phi1(Rc["F"], g2.cpu(), spec.beta)
        # reference phi1 weight from the coarse joint
        w = np.exp(-spec.beta * (Rc["F"] - Rc["F"].min())).sum(1)
        ref["joint_weight"] = w / w.sum()
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(path, **ref)
    _REF_CACHE[key] = ref
    if verbose:
        print(f"[ref] built {os.path.relpath(path)}")
    return ref


# ---------------------------------------------------------------------------
# Initialisation callables
# ---------------------------------------------------------------------------
def make_init(spec: AlkaneRunSpec):
    n_dih = spec.n_atoms - 3
    if spec.init_mode == "trans":
        return [0.0] * n_dih
    # dispersed: each dihedral uniformly among {trans, G+, G-} per replica
    centers = torch.tensor([0.0, GAUCHE, -GAUCHE], dtype=torch.float64)

    def sampler(R, N, gen):
        idx = torch.randint(0, 3, (R, N, n_dih), generator=gen, device=gen.device)
        return centers.to(gen.device)[idx]
    return sampler


# ---------------------------------------------------------------------------
# Execute one job
# ---------------------------------------------------------------------------
def execute_run(spec: AlkaneRunSpec, device, cache_dir="cache/alkanes", verbose=False):
    params = build_params(spec)
    sim = build_sim(spec)
    cv = DihedralCV((0, 1, 2, 3))
    ref = build_reference(spec, device, cache_dir=cache_dir, verbose=verbose)
    init = make_init(spec)
    is_pent = spec.molecule == "pentane"
    oracle = ref["F"] if spec.method == "fr_oracle" else None

    t0 = time.perf_counter()
    if spec.method == "opes":
        diag = opesmod.run_opes(params, sim, build_opes_cfg(spec), list(spec.seeds), cv,
                                device, initial_dihedrals=init, collect_pentane=is_pent, verbose=verbose)
    else:
        diag = core.run_sampler(spec.method, params, sim, list(spec.seeds), cv, device,
                                initial_dihedrals=init, oracle_free_energy=oracle,
                                collect_pentane=is_pent, verbose=verbose)

    R = len(spec.seeds)
    grid = ref["grid"]; dphi = float(ref["dphi"])
    times = np.asarray(diag["times"], float)
    per_seed = []
    for r in range(R):
        pmf_series = diag["pmf"][:, r, :]
        mf_series = diag["mean_force"][:, r, :]
        pm = metrics.profile_metrics(pmf_series, mf_series, times, ref["F"], ref["Fprime"], dphi)
        mm = metrics.marginal_metrics(diag["p_hat"][-1, r], grid, dphi, spec.beta, ref["F"])
        rec = {"seed": int(spec.seeds[r]),
               "final_l2_F": pm["final_l2_F"], "final_l2_Fp": pm["final_l2_Fp"],
               "integrated_l2_F": pm["integrated_l2_F"], "integrated_l2_Fp": pm["integrated_l2_Fp"],
               "early_l2_F": pm["early_l2_F"], "mid_l2_F": pm["mid_l2_F"],
               "n_transitions": int(diag["n_transitions"][r]),
               "n_round_trips": int(diag["n_round_trips"][r]),
               "total_replacements": int(diag["total_replacement_events"][r]),
               "final_ancestor_ess": float(diag["ancestor_ess"][-1, r]),
               "min_ancestor_ess": (float(np.nanmin(diag["ancestor_ess"][:, r]))
                                    if np.isfinite(diag["ancestor_ess"][:, r]).any() else float("nan")),
               "final_max_ancestor_frac": float(diag["max_ancestor_frac"][-1, r]),
               "final_n_unique_ancestor": int(diag["n_unique_ancestor"][-1, r]),
               "fr_score_std": float(np.asarray(diag["fr_score_std"]).reshape(-1)[r] if np.ndim(diag["fr_score_std"]) else np.nan),
               "fr_score_absmax": float(np.asarray(diag["fr_score_absmax"]).reshape(-1)[r] if np.ndim(diag["fr_score_absmax"]) else np.nan),
               "final_frac_T": float(diag["frac_T"][-1, r]),
               "final_frac_Gp": float(diag["frac_Gp"][-1, r]),
               "final_frac_Gm": float(diag["frac_Gm"][-1, r])}
        rec.update({k: v for k, v in mm.items()})
        if spec.method in core.FR_METHODS:
            fe = metrics.fr_event_metrics(diag["total_replacement_events"][r],
                                          diag["repl_cumulative"][:, r], diag["steps"],
                                          spec.n_replicas, spec.fr_start_steps, spec.fr_every, spec.n_steps)
            rec.update(fe)
        if spec.method == "opes":
            rec["final_neff_frac"] = float(np.asarray(diag["final_neff_frac"]).reshape(-1)[r])
            rec["final_n_kernels"] = int(np.asarray(diag["final_n_kernels"]).reshape(-1)[r])
            # native reweight final L2
            rec["final_l2_F_reweight"] = metrics._circ_l2(diag["pmf_reweight"][-1, r], ref["F"], dphi)
        if is_pent and "joint_hist" in diag:
            cm = metrics.conditional_metrics(diag["joint_hist"][r], ref["grid2"], float(ref["dphi2"]),
                                             ref["cond"], ref["joint_weight"], sim.basin_barrier)
            jb = metrics.joint_basin_visits(diag["joint_hist"][r], ref["grid2"], sim.basin_barrier)
            rec.update({k: v for k, v in cm.items() if not k.endswith("_bins")})
            rec.update(jb)
        per_seed.append(rec)

    had_nan = bool(not np.isfinite(diag["pmf"][-1]).all())
    out = {
        "run_id": spec.run_id(), "spec_hash": spec.spec_hash(),
        "spec_json": json.dumps(asdict(spec), sort_keys=True),
        "stage": spec.stage, "molecule": spec.molecule, "name": spec.name,
        "method": spec.method, "init_mode": spec.init_mode,
        "beta": spec.beta, "sigma": spec.sigma, "decouple": spec.decouple,
        "n_steps": spec.n_steps, "n_replicas": spec.n_replicas, "seeds": np.asarray(spec.seeds),
        "runtime_seconds": diag["runtime_seconds"], "wall_seconds": time.perf_counter() - t0,
        "device": str(device), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "had_nan": had_nan, "config_hash": sim.config_hash(), "core_version": "alkanes_v1",
        # reference + final profiles (median over seeds for compactness + per-seed profiles)
        "grid": grid, "ref_F": ref["F"], "ref_Fprime": ref["Fprime"], "dphi": dphi,
        "times": times,
        "final_pmf": diag["pmf"][-1], "final_mean_force": diag["mean_force"][-1],
        "final_p_hat": diag["p_hat"][-1],
        "final_q_target": (diag["q_target"][-1] if "q_target" in diag else np.full((R, spec.n_grid), np.nan)),
        "F_target_ema": (diag["F_target_ema"] if diag["F_target_ema"] is not None else np.full((R, spec.n_grid), np.nan)),
        "birth_hist": diag["birth_hist"], "death_hist": diag["death_hist"],
        "l2_F_t": np.stack([metrics.profile_metrics(diag["pmf"][:, r, :], diag["mean_force"][:, r, :],
                                                    times, ref["F"], ref["Fprime"], dphi)["l2_F_series"]
                            for r in range(R)]),
        "per_seed": json.dumps(per_seed),
    }
    if spec.method == "opes":
        out["final_pmf_reweight"] = diag["pmf_reweight"][-1]
    if is_pent and "joint_hist" in diag:
        out["joint_hist"] = diag["joint_hist"]
        out["ref_joint_F"] = ref["joint_F"]; out["ref_cond"] = ref["cond"]
        out["grid2"] = ref["grid2"]; out["joint_weight"] = ref["joint_weight"]
    return out, per_seed


# ---------------------------------------------------------------------------
# IO / resume
# ---------------------------------------------------------------------------
def run_npz_path(raw_dir, spec: AlkaneRunSpec):
    return os.path.join(raw_dir, spec.run_id() + ".npz")


def run_is_valid(path):
    if not os.path.exists(path):
        return False
    try:
        d = np.load(path, allow_pickle=True)
        ok = ("per_seed" in d.files and not bool(d.get("had_nan", np.array(False))))
        d.close()
        return bool(ok)
    except Exception:
        return False


def save_run(path, out):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **out)
    os.replace(tmp, path)


def save_failure(raw_dir, spec, exc):
    fail_dir = os.path.join(raw_dir, "_failures")
    os.makedirs(fail_dir, exist_ok=True)
    path = os.path.join(fail_dir, spec.run_id() + ".json")
    with open(path, "w") as fh:
        json.dump({"run_id": spec.run_id(), "spec": asdict(spec), "error": repr(exc)}, fh, indent=2)
    return path


# ---------------------------------------------------------------------------
# Stage -> specs
# ---------------------------------------------------------------------------
def _method_knobs(cfg, name):
    base = cfg.get("method_defaults", {})
    m = dict(base); m.update(cfg["methods"][name])
    return m


def expand_stage(cfg, stage):
    st = cfg["stages"][stage]
    base = dict(cfg.get("base", {}))
    base.update(st.get("base_overrides", {}))
    seeds = tuple(st["seeds"])
    rng_seed = int(st.get("rng_seed", 20260719))
    specs = []
    for cell in st["cells"]:
        c = dict(base); c.update(cell)
        for mname in st["methods"]:
            mk = _method_knobs(cfg, mname)
            spec = AlkaneRunSpec(
                stage=stage, molecule=c["molecule"], name=mname, method=mk["type"],
                init_mode=c.get("init_mode", "trans"), seeds=seeds,
                beta=float(c["beta"]), sigma=float(c.get("sigma", 2.3)),
                epsilon=float(c.get("epsilon", 1.0)), decouple=bool(c.get("decouple", False)),
                dt=float(c.get("dt", 5e-4)), n_steps=int(c["n_steps"]),
                n_replicas=int(c["n_replicas"]), save_every=int(c.get("save_every", 2000)),
                rng_seed=rng_seed, n_grid=int(c.get("n_grid", 180)), n_grid2=int(c.get("n_grid2", 48)),
                abf_bandwidth=float(c.get("abf_bandwidth", 0.05)),
                kde_bandwidth=float(c.get("kde_bandwidth", 0.10)),
                abf_warmup_steps=int(c.get("abf_warmup_steps", 10000)),
                estimator_burn_in_steps=int(c.get("estimator_burn_in_steps", 12000)),
                abf_force_clip=float(c.get("abf_force_clip", 60.0)),
                force_clip=float(c.get("force_clip", 200.0)),
                fr_rate=float(mk.get("fr_rate", 0.0)),
                target_ema_rate=float(mk.get("target_ema_rate", 0.005)),
                max_event_fraction=float(mk.get("max_event_fraction", 0.02)),
                fr_every=int(mk.get("fr_every", 5)),
                fr_start_steps=int(mk.get("fr_start_steps", 20000)),
                score_clip=float(mk.get("score_clip", 2.0)),
                opes_barrier=float(mk.get("opes_barrier", 8.0)),
                opes_pace=int(mk.get("opes_pace", 500)),
                opes_sigma=float(mk.get("opes_sigma", 0.2)),
                opes_gamma=float(mk.get("opes_gamma", float("inf"))),
                ref_n_samples=int(c.get("ref_n_samples", 40000)))
            specs.append(spec)
    return specs


def load_yaml(path):
    import yaml
    with open(path) as fh:
        return yaml.safe_load(fh)
