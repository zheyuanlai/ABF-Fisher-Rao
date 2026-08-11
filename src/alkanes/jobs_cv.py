"""Job expansion, reference caching, single-job execution and IO for the CV-extension
experiments (distance CV R15/R14 and joint torsion CV (phi1,phi2)).

One *job* = one (kind, molecule, cv, method, physics cell, initialisation, stage) running a
batch of matched ``seeds`` in a single GPU process.  Deterministic in its :class:`CVRunSpec`,
hashed for idempotency, one atomic ``.npz`` per job (resume-safe).  References are cached per
physics under ``cache/alkanes_cv/``.  Reuses the resume/IO helpers of :mod:`alkanes.jobs`.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, asdict
from typing import Tuple

import numpy as np
import torch

from . import (core_dist as cd, core2d as c2, opes_cv as opesmod, metrics as M,
               metrics_cv as MC, reference as refmod, reference_cv as rcmod,
               periodic as per, potentials as pot)
from .distance_cv import DistanceCV
from .cv2d import JointDihedralCV2D
from .jobs import run_is_valid, save_run, save_failure  # reuse generic IO

PI = math.pi
GAUCHE = math.radians(116.57)


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CVRunSpec:
    kind: str                 # "dist" | "joint2d"
    stage: str
    molecule: str             # "butane" | "pentane"
    name: str                 # method label
    method: str               # abf | fr_estimated | fr_uniform | fr_oracle | opes
    init_mode: str            # "trans" | "compact" | "dispersed"
    seeds: Tuple[int, ...]
    # physics
    beta: float
    sigma: float
    epsilon: float
    decouple: bool
    # dynamics
    dt: float
    n_steps: int
    n_replicas: int
    save_every: int
    rng_seed: int
    abf_warmup_steps: int
    estimator_burn_in_steps: int
    abf_force_clip: float
    force_clip: float
    # distance-CV geometry / estimator
    cv_i: int = 0
    cv_j: int = 4
    R_lo: float = 1.4
    R_hi: float = 3.7
    wall_lo: float = 1.45
    wall_hi: float = 3.65
    k_wall: float = 200.0
    dist_n_grid: int = 256
    dist_abf_bandwidth: float = 0.04
    dist_kde_bandwidth: float = 0.06
    n_rbins: int = 12
    thermal_delta: float = 10.0
    # 2-D estimator
    grid2d: int = 48
    abf_bandwidth2d: float = 0.20
    kde_bandwidth2d: float = 0.30
    abf_min_count: float = 5.0
    #: Distance-CV `fullSamples` guard. 0.0 = disabled = frozen v1 behaviour; see
    #: alkanes.core_dist.DistSimConfig.abf_min_count for why it is a separate field.
    abf_min_count_dist: float = 0.0
    density_ema: float = 0.0
    estimator_stride: int = 1
    # conditional torsion grid
    n_grid2: int = 48
    # FR
    fr_rate: float = 0.0
    target_ema_rate: float = 0.005
    max_event_fraction: float = 0.01
    fr_every: int = 5
    fr_start_steps: int = 10000
    score_clip: float = 2.0
    # OPES
    opes_barrier: float = 8.0
    opes_pace: int = 500
    opes_sigma: float = 0.20
    opes_gamma: float = float("inf")
    # reference sampling
    ref_n_samples: int = 400000

    @property
    def n_atoms(self):
        return 4 if self.molecule == "butane" else 5

    def physics_tag(self):
        base = f"{self.molecule}_b{self.beta:g}_s{self.sigma:g}_{'dec' if self.decouple else 'full'}"
        if self.kind == "dist":
            return f"{base}_R{self.cv_i}{self.cv_j}_lo{self.R_lo:g}_hi{self.R_hi:g}_g{self.dist_n_grid}"
        return f"{base}_2d_g{self.grid2d}"

    def spec_hash(self):
        return hashlib.md5(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]

    def run_id(self):
        return (f"{self.stage}__{self.kind}__{self.molecule}__{self.name}__{self.init_mode}"
                f"__b{self.beta:g}__N{self.n_replicas}__T{self.n_steps}"
                f"__ns{len(self.seeds)}__{self.spec_hash()}")


def build_params(spec: CVRunSpec) -> pot.AlkaneParams:
    return pot.AlkaneParams(n_atoms=spec.n_atoms, beta=spec.beta, sigma=spec.sigma,
                            epsilon=spec.epsilon, decouple=spec.decouple, force_clip=spec.force_clip)


# ---------------------------------------------------------------------------
# References (evaluation only), cached per physics
# ---------------------------------------------------------------------------
_REF_CACHE: dict = {}


def build_dist_reference(spec: CVRunSpec, device, cache_dir="cache/alkanes_cv", verbose=False):
    key = ("dist", spec.physics_tag(), spec.ref_n_samples, spec.n_grid2, spec.n_rbins)
    if key in _REF_CACHE:
        return _REF_CACHE[key]
    path = os.path.join(cache_dir, f"ref_{spec.physics_tag()}_ns{spec.ref_n_samples}_g2{spec.n_grid2}.npz")
    if os.path.exists(path):
        d = np.load(path, allow_pickle=True); ref = {k: d[k] for k in d.files}; _REF_CACHE[key] = ref
        return ref
    p = build_params(spec)
    ref = rcmod.distance_reference(p, spec.cv_i, spec.cv_j, R_lo=spec.R_lo, R_hi=spec.R_hi,
                                   n_grid=spec.dist_n_grid, n_samples=spec.ref_n_samples,
                                   seed=987 + spec.dist_n_grid, device=device,
                                   bandwidth=spec.dist_abf_bandwidth, proposal="v4",
                                   n_cond_bins=spec.n_rbins, n_grid2=spec.n_grid2)
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(path, **{k: v for k, v in ref.items() if not isinstance(v, str)},
             _strs=json.dumps({k: v for k, v in ref.items() if isinstance(v, str)}))
    _REF_CACHE[key] = ref
    if verbose:
        print(f"[ref] built {os.path.relpath(path)} ESSfrac={ref['ess_frac']:.3f}")
    return ref


def build_2d_reference(spec: CVRunSpec, device, cache_dir="cache/alkanes_cv", verbose=False):
    key = ("2d", spec.physics_tag(), spec.ref_n_samples)
    if key in _REF_CACHE:
        return _REF_CACHE[key]
    path = os.path.join(cache_dir, f"ref2d_{spec.physics_tag()}_ns{spec.ref_n_samples}.npz")
    if os.path.exists(path):
        d = np.load(path, allow_pickle=True); ref = {k: d[k] for k in d.files}; _REF_CACHE[key] = ref
        return ref
    p = build_params(spec)
    grid, dphi = per.periodic_grid(spec.grid2d, device=device, dtype=torch.float64)
    if spec.decouple:
        F = (pot.V4(grid[:, None], p) + pot.V4(grid[None, :], p))
        F = (F - F.mean()).cpu().numpy()
    else:
        Rj = refmod.qmc_reference_pentane(grid, grid, p, n_samples=spec.ref_n_samples,
                                          seed=41 + spec.grid2d, device=device)
        F = Rj["F"]
    ref = {"grid1": grid.cpu().numpy(), "grid2": grid.cpu().numpy(), "dphi": float(dphi),
           "joint_F": F}
    beta = spec.beta
    ref["cond"] = refmod.conditional_phi2_given_phi1(F, grid.cpu(), beta)
    w = np.exp(-beta * (F - F.min())).sum(1); ref["joint_weight"] = w / w.sum()
    # equilibrium weight (normalised joint Boltzmann)
    pw = np.exp(-beta * (F - F.min())); ref["eq_weight"] = pw / pw.sum()
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(path, **ref)
    _REF_CACHE[key] = ref
    if verbose:
        print(f"[ref] built {os.path.relpath(path)}")
    return ref


# ---------------------------------------------------------------------------
# Initialisation callables
# ---------------------------------------------------------------------------
def make_init(spec: CVRunSpec):
    n_dih = spec.n_atoms - 3
    if spec.init_mode == "trans":
        return [0.0] * n_dih
    if spec.init_mode == "compact":
        return [GAUCHE] * n_dih                 # all-gauche+ (compact)
    centers = torch.tensor([0.0, GAUCHE, -GAUCHE], dtype=torch.float64)

    def sampler(R, N, gen):
        idx = torch.randint(0, 3, (R, N, n_dih), generator=gen, device=gen.device)
        return centers.to(gen.device)[idx]
    return sampler


# ---------------------------------------------------------------------------
# Execute one job
# ---------------------------------------------------------------------------
def _dist_sim(spec: CVRunSpec) -> cd.DistSimConfig:
    return cd.DistSimConfig(
        dt=spec.dt, n_steps=spec.n_steps, n_replicas=spec.n_replicas, save_every=spec.save_every,
        rng_seed=spec.rng_seed, R_lo=spec.R_lo, R_hi=spec.R_hi, wall_lo=spec.wall_lo,
        wall_hi=spec.wall_hi, k_wall=spec.k_wall, n_grid=spec.dist_n_grid,
        abf_bandwidth=spec.dist_abf_bandwidth, kde_bandwidth=spec.dist_kde_bandwidth,
        abf_warmup_steps=spec.abf_warmup_steps, abf_force_clip=spec.abf_force_clip,
        abf_min_count=spec.abf_min_count_dist,
        estimator_burn_in_steps=spec.estimator_burn_in_steps, fr_rate=spec.fr_rate,
        score_clip=spec.score_clip, fr_start_steps=spec.fr_start_steps, fr_every=spec.fr_every,
        target_ema_rate=spec.target_ema_rate, max_event_fraction=spec.max_event_fraction,
        n_grid2=spec.n_grid2, n_rbins=spec.n_rbins)


def _sim2d(spec: CVRunSpec) -> c2.Sim2DConfig:
    return c2.Sim2DConfig(
        dt=spec.dt, n_steps=spec.n_steps, n_replicas=spec.n_replicas, save_every=spec.save_every,
        rng_seed=spec.rng_seed, n_grid=spec.grid2d, abf_bandwidth=spec.abf_bandwidth2d,
        kde_bandwidth=spec.kde_bandwidth2d, abf_warmup_steps=spec.abf_warmup_steps,
        abf_force_clip=spec.abf_force_clip, abf_min_count=spec.abf_min_count,
        estimator_burn_in_steps=spec.estimator_burn_in_steps, estimator_stride=spec.estimator_stride,
        fr_rate=spec.fr_rate, score_clip=spec.score_clip, fr_start_steps=spec.fr_start_steps,
        fr_every=spec.fr_every, target_ema_rate=spec.target_ema_rate,
        max_event_fraction=spec.max_event_fraction, density_ema=spec.density_ema)


def execute_dist(spec: CVRunSpec, device, cache_dir="cache/alkanes_cv", verbose=False):
    p = build_params(spec); sim = _dist_sim(spec)
    cv = DistanceCV(spec.cv_i, spec.cv_j)
    ref = build_dist_reference(spec, device, cache_dir=cache_dir, verbose=verbose)
    init = make_init(spec)
    oracle = ref["F"] if spec.method == "fr_oracle" else None
    t0 = time.perf_counter()
    if spec.method == "opes":
        ocfg = opesmod.IntervalOPESConfig(n_grid=spec.dist_n_grid, beta=spec.beta, R_lo=spec.R_lo,
                                          R_hi=spec.R_hi, barrier=spec.opes_barrier, pace=spec.opes_pace,
                                          sigma=spec.opes_sigma, gamma=spec.opes_gamma,
                                          bias_force_clip=spec.abf_force_clip, warmup_steps=spec.abf_warmup_steps)
        diag = opesmod.run_opes_dist(p, sim, ocfg, list(spec.seeds), cv, device,
                                     initial_dihedrals=init, collect_conditional=(spec.molecule == "pentane"), verbose=verbose)
    else:
        diag = cd.run_sampler_dist(spec.method, p, sim, list(spec.seeds), cv, device,
                                   initial_dihedrals=init, oracle_free_energy=oracle,
                                   collect_conditional=(spec.molecule == "pentane"), verbose=verbose)
    R = len(spec.seeds)
    grid = ref["grid"]; dz = float(ref["dz"]); F_ref = ref["F"]; Fp_ref = ref["Fprime"]
    thermal_mask = (F_ref - F_ref.min()) <= spec.thermal_delta
    times = np.asarray(diag["times"], float)
    per_seed = []
    for r in range(R):
        pm = MC.dist_profile_metrics(diag["pmf"][:, r, :], diag["mean_force"][:, r, :], times,
                                     F_ref, Fp_ref, dz, thermal_mask)
        sm = MC.dist_support_metrics(diag["final_eff_counts"][r], thermal_mask)
        rec = {"seed": int(spec.seeds[r]), "final_l2_F": pm["final_l2_F"],
               "integrated_l2_F": pm["integrated_l2_F"], "final_l2_Fp": pm["final_l2_Fp"],
               "integrated_l2_Fp": pm["integrated_l2_Fp"], "early_l2_F": pm["early_l2_F"],
               "mid_l2_F": pm["mid_l2_F"], **sm,
               "n_transitions": int(diag["n_transitions"][r]),
               "n_round_trips": int(diag["n_round_trips"][r]),
               "fd_compact": int(diag["first_discovery"]["compact"][r]),
               "fd_intermediate": int(diag["first_discovery"]["intermediate"][r]),
               "fd_extended": int(diag["first_discovery"]["extended"][r]),
               "total_replacements": int(diag["total_replacement_events"][r]),
               "final_ancestor_ess": float(diag["ancestor_ess"][-1, r]),
               "min_ancestor_ess": (float(np.nanmin(diag["ancestor_ess"][:, r])) if np.isfinite(diag["ancestor_ess"][:, r]).any() else float("nan")),
               "final_max_ancestor_frac": float(diag["max_ancestor_frac"][-1, r]),
               "final_frac_compact": float(diag["frac_compact"][-1, r]),
               "final_frac_extended": float(diag["frac_extended"][-1, r])}
        if spec.molecule == "pentane" and "cond_hist" in diag and "cond_dens" in ref:
            cm = MC.dist_conditional_metrics(diag["cond_hist"][r], ref["cond_dens"], ref["cond_weight"],
                                             ref["cond_grid1"], float(ref["cond_dphi"]),
                                             math.radians(61.6))
            rec.update(cm)
        if spec.method in cd.FR_METHODS:
            fe = M.fr_event_metrics(diag["total_replacement_events"][r], diag["repl_cumulative"][:, r],
                                    diag["steps"], spec.n_replicas, spec.fr_start_steps, spec.fr_every, spec.n_steps)
            rec.update(fe)
        if spec.method == "opes":
            rec["final_neff_frac"] = float(np.asarray(diag["final_neff_frac"]).reshape(-1)[r])
            rec["final_n_kernels"] = int(np.asarray(diag["final_n_kernels"]).reshape(-1)[r])
            rec["final_l2_F_reweight"] = MC._interval_l2(diag["pmf_reweight"][-1, r], F_ref, dz, thermal_mask)
        per_seed.append(rec)
    had_nan = bool(not np.isfinite(diag["pmf"][-1]).all())
    l2_F_t = np.stack([MC.dist_profile_metrics(diag["pmf"][:, r, :], diag["mean_force"][:, r, :],
                       times, F_ref, Fp_ref, dz, thermal_mask)["l2_F_series"] for r in range(R)])
    F_range = float(F_ref[thermal_mask].max() - F_ref[thermal_mask].min())
    out = _base_out(spec, diag, device, per_seed, had_nan, t0)
    out.update({"l2_F_t": l2_F_t, "F_range_thermal": F_range,
                "grid": grid, "ref_F": F_ref, "ref_Fprime": Fp_ref, "dz": dz, "times": times,
                "R_lo": spec.R_lo, "R_hi": spec.R_hi, "thermal_delta": spec.thermal_delta,
                "final_pmf": diag["pmf"][-1], "final_mean_force": diag["mean_force"][-1],
                "final_p_hat": diag["p_hat"][-1], "final_eff_counts": diag["final_eff_counts"]})
    if spec.method == "opes":
        out["final_pmf_reweight"] = diag["pmf_reweight"][-1]
    if "cond_hist" in diag:
        out["cond_hist"] = diag["cond_hist"]; out["ref_cond_dens"] = ref.get("cond_dens")
    return out, per_seed


def execute_2d(spec: CVRunSpec, device, cache_dir="cache/alkanes_cv", verbose=False):
    p = build_params(spec); sim = _sim2d(spec)
    cv = JointDihedralCV2D()
    ref = build_2d_reference(spec, device, cache_dir=cache_dir, verbose=verbose)
    init = make_init(spec)
    oracle = ref["joint_F"] if spec.method == "fr_oracle" else None
    t0 = time.perf_counter()
    if spec.method == "opes":
        ocfg = opesmod.TorusOPESConfig(n_grid=spec.grid2d, beta=spec.beta, barrier=spec.opes_barrier,
                                       pace=spec.opes_pace, sigma=spec.opes_sigma, gamma=spec.opes_gamma,
                                       bias_force_clip=spec.abf_force_clip, warmup_steps=spec.abf_warmup_steps)
        diag = opesmod.run_opes_2d(p, sim, ocfg, list(spec.seeds), cv, device, initial_dihedrals=init, verbose=verbose)
    else:
        diag = c2.run_sampler_2d(spec.method, p, sim, list(spec.seeds), cv, device,
                                 initial_dihedrals=init, oracle_free_energy=oracle, verbose=verbose)
    R = len(spec.seeds)
    grid = ref["grid1"]; dphi = float(ref["dphi"]); F_ref = ref["joint_F"]
    thermal_mask = (F_ref - F_ref.min()) <= spec.thermal_delta
    eq_weight = ref["eq_weight"]
    times = np.asarray(diag["times"], float)
    barrier = sim.basin_barrier
    per_seed = []
    for r in range(R):
        pm = MC.joint_profile_metrics(diag["pmf"][:, r], times, F_ref, dphi, dphi, thermal_mask, eq_weight)
        mfe = MC.meanforce_vector_error(diag["final_pmf"][r], F_ref, dphi, dphi, thermal_mask)
        # basin/conditional fidelity from the RECONSTRUCTED F_hat (NOT the biased histogram,
        # which ABF flattens toward uniform); n_basins_visited stays a discovery signal.
        fid = MC.reconstructed_fidelity(diag["final_pmf"][r], F_ref, grid, spec.beta, barrier)
        jb = M.joint_basin_visits(diag["joint_hist"][r], grid, barrier)   # discovery only
        rec = {"seed": int(spec.seeds[r]), "final_l2_F": pm["final_l2_F"],
               "integrated_l2_F": pm["integrated_l2_F"], "final_l2_F_eqw": pm["final_l2_F_eqw"],
               "early_l2_F": pm["early_l2_F"], "mid_l2_F": pm["mid_l2_F"],
               "meanforce_vec_err": mfe,
               "basin_occupancy_tv": fid["basin_occupancy_tv"],
               "cond_tv_weighted": fid["cond_tv_weighted"], "cond_kl_weighted": fid["cond_kl_weighted"],
               "n_basins_visited": int(jb["n_basins_visited"]),
               "n_transitions": int(diag["n_transitions"][r]),
               "n_round_trips": int(diag["n_round_trips"][r]),
               "total_replacements": int(diag["total_replacement_events"][r]),
               "final_ancestor_ess": float(diag["ancestor_ess"][-1, r]),
               "min_ancestor_ess": (float(np.nanmin(diag["ancestor_ess"][:, r])) if np.isfinite(diag["ancestor_ess"][:, r]).any() else float("nan")),
               "final_max_ancestor_frac": float(diag["max_ancestor_frac"][-1, r]),
               "gram_lam_min_min": float(diag["gram_lam_min_min"][r]),
               "curl_pre_final": (float(diag["curl_pre"][-1, r])
                                  if ("curl_pre" in diag and np.asarray(diag["curl_pre"]).size) else float("nan"))}
        if spec.method in c2.FR_METHODS:
            fe = M.fr_event_metrics(diag["total_replacement_events"][r], diag["repl_cumulative"][:, r],
                                    diag["steps"], spec.n_replicas, spec.fr_start_steps, spec.fr_every, spec.n_steps)
            rec.update(fe)
        if spec.method == "opes":
            rec["final_neff_frac"] = float(np.asarray(diag["final_neff_frac"]).reshape(-1)[r])
            rec["final_n_kernels"] = int(np.asarray(diag["final_n_kernels"]).reshape(-1)[r])
            rec["final_l2_F_reweight"] = MC.l2_2d_np(diag["final_pmf_reweight"][r], F_ref, dphi, dphi, thermal_mask)
        per_seed.append(rec)
    had_nan = bool(not np.isfinite(diag["final_pmf"]).all())
    l2_F_t = np.stack([MC.joint_profile_metrics(diag["pmf"][:, r], times, F_ref, dphi, dphi,
                       thermal_mask, eq_weight)["l2_F_series"] for r in range(R)])
    F_range = float(F_ref[thermal_mask].max() - F_ref[thermal_mask].min())
    out = _base_out(spec, diag, device, per_seed, had_nan, t0)
    out.update({"l2_F_t": l2_F_t, "F_range_thermal": F_range,
                "grid1": grid, "grid2": grid, "dphi": dphi, "times": times,
                "ref_joint_F": F_ref, "ref_cond": ref["cond"], "joint_weight": ref["joint_weight"],
                "eq_weight": eq_weight, "thermal_delta": spec.thermal_delta,
                "final_pmf": diag["final_pmf"], "joint_hist": diag["joint_hist"],
                "birth_hist": diag["birth_hist"], "death_hist": diag["death_hist"],
                "trans_matrix": diag["trans_matrix"], "first_discovery": diag["first_discovery"],
                "gram_reg_activations": int(diag["gram_reg_activations"])})
    if spec.method == "opes":
        out["final_pmf_reweight"] = diag["final_pmf_reweight"]
    return out, per_seed


_BASIN_KEYS = ["basin_T_T", "basin_T_G+", "basin_T_G-", "basin_G+_T", "basin_G+_G+",
               "basin_G+_G-", "basin_G-_T", "basin_G-_G+", "basin_G-_G-"]


def _ref_basin_probs(F, grid, beta, barrier):
    g = np.asarray(grid, float)
    T = np.abs(g) < barrier; Gp = g >= barrier; Gm = g <= -barrier
    masks = [T, Gp, Gm]
    w = np.exp(-beta * (F - F.min()))
    out = []
    for m1 in masks:
        for m2 in masks:
            out.append(w[np.ix_(m1, m2)].sum())
    out = np.array(out); return out / max(out.sum(), 1e-12)


def _base_out(spec, diag, device, per_seed, had_nan, t0):
    return {"run_id": spec.run_id(), "spec_hash": spec.spec_hash(),
            "spec_json": json.dumps(asdict(spec), sort_keys=True), "kind": spec.kind,
            "stage": spec.stage, "molecule": spec.molecule, "name": spec.name, "method": spec.method,
            "init_mode": spec.init_mode, "beta": spec.beta, "sigma": spec.sigma, "decouple": spec.decouple,
            "n_steps": spec.n_steps, "n_replicas": spec.n_replicas, "seeds": np.asarray(spec.seeds),
            "runtime_seconds": diag["runtime_seconds"], "wall_seconds": time.perf_counter() - t0,
            "device": str(device), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "had_nan": had_nan, "core_version": "alkanes_cv_v1", "per_seed": json.dumps(per_seed)}


def execute_run(spec: CVRunSpec, device, cache_dir="cache/alkanes_cv", verbose=False):
    if spec.kind == "dist":
        return execute_dist(spec, device, cache_dir=cache_dir, verbose=verbose)
    return execute_2d(spec, device, cache_dir=cache_dir, verbose=verbose)


def run_npz_path(raw_dir, spec: CVRunSpec):
    return os.path.join(raw_dir, spec.run_id() + ".npz")


# ---------------------------------------------------------------------------
# Stage -> specs
# ---------------------------------------------------------------------------
def _method_knobs(cfg, name):
    base = dict(cfg.get("method_defaults", {})); base.update(cfg["methods"][name]); return base


def expand_stage(cfg, stage):
    st = cfg["stages"][stage]
    base = dict(cfg.get("base", {})); base.update(st.get("base_overrides", {}))
    seeds = tuple(st["seeds"]); rng_seed = int(st.get("rng_seed", 20260719))
    specs = []
    for cell in st["cells"]:
        c = dict(base); c.update(cell)
        for mname in st["methods"]:
            mk = _method_knobs(cfg, mname)
            spec = CVRunSpec(
                kind=c["kind"], stage=stage, molecule=c["molecule"], name=mname, method=mk["type"],
                init_mode=c.get("init_mode", "trans"), seeds=seeds,
                beta=float(c["beta"]), sigma=float(c.get("sigma", 2.3)), epsilon=float(c.get("epsilon", 1.0)),
                decouple=bool(c.get("decouple", False)), dt=float(c.get("dt", 5e-4)),
                n_steps=int(c["n_steps"]), n_replicas=int(c["n_replicas"]),
                save_every=int(c.get("save_every", 5000)), rng_seed=rng_seed,
                abf_warmup_steps=int(c.get("abf_warmup_steps", 5000)),
                estimator_burn_in_steps=int(c.get("estimator_burn_in_steps", 6000)),
                abf_force_clip=float(c.get("abf_force_clip", 60.0)), force_clip=float(c.get("force_clip", 200.0)),
                cv_i=int(c.get("cv_i", 0)), cv_j=int(c.get("cv_j", 4)),
                R_lo=float(c.get("R_lo", 1.4)), R_hi=float(c.get("R_hi", 3.7)),
                wall_lo=float(c.get("wall_lo", 1.45)), wall_hi=float(c.get("wall_hi", 3.65)),
                k_wall=float(c.get("k_wall", 200.0)), dist_n_grid=int(c.get("dist_n_grid", 256)),
                dist_abf_bandwidth=float(c.get("dist_abf_bandwidth", 0.04)),
                dist_kde_bandwidth=float(c.get("dist_kde_bandwidth", 0.06)),
                n_rbins=int(c.get("n_rbins", 12)), thermal_delta=float(c.get("thermal_delta", 10.0)),
                grid2d=int(c.get("grid2d", 48)), abf_bandwidth2d=float(c.get("abf_bandwidth2d", 0.20)),
                kde_bandwidth2d=float(c.get("kde_bandwidth2d", 0.30)),
                abf_min_count=float(c.get("abf_min_count", 5.0)),
                abf_min_count_dist=float(c.get("abf_min_count_dist", 0.0)), density_ema=float(mk.get("density_ema", 0.0)),
                estimator_stride=int(c.get("estimator_stride", 1)), n_grid2=int(c.get("n_grid2", 48)),
                fr_rate=float(mk.get("fr_rate", 0.0)), target_ema_rate=float(mk.get("target_ema_rate", 0.005)),
                max_event_fraction=float(mk.get("max_event_fraction", 0.01)), fr_every=int(mk.get("fr_every", 5)),
                fr_start_steps=int(mk.get("fr_start_steps", c.get("fr_start_steps", 10000))),
                score_clip=float(mk.get("score_clip", 2.0)),
                opes_barrier=float(mk.get("opes_barrier", 8.0)), opes_pace=int(mk.get("opes_pace", 500)),
                opes_sigma=float(mk.get("opes_sigma", 0.20)), opes_gamma=float(mk.get("opes_gamma", float("inf"))),
                ref_n_samples=int(c.get("ref_n_samples", 400000)))
            specs.append(spec)
    return specs


def load_yaml(path):
    import yaml
    with open(path) as fh:
        return yaml.safe_load(fh)
