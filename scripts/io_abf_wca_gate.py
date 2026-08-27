"""WCA reference gate, then an A0-only difficulty screening.

Frozen protocol: ``docs/IO_ABF_OVERNIGHT_PREREGISTRATION.md`` section 7, item 4.

The gate asks one question: does the high-precision reference cover *the setup a
confirmatory comparison would run* -- physical parameters, CV, grid, evaluation
mask, integration convention -- well enough that a speedup scored against it
means what it says?  If not, A0 diagnostics may still run; a speedup may not be
reported.  That rule is not negotiable after seeing the answer, so the gate is a
separate script from the campaign and runs first.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import wca_abffr_core as core                                    # noqa: E402
import wca_jobs as jobs                                          # noqa: E402
from abffr import io_abf                                         # noqa: E402

OUT = os.path.join(ROOT, "results", "io_abf_overnight", "wca")
HP = os.path.join(ROOT, "results", "v2_validity_audits", "wca_hp_reference",
                  "wca_hp_reference.npz")
HP_AUDIT = os.path.join(ROOT, "results", "v2_validity_audits",
                        "wca_hp_reference", "summary.json")


def reference_gate():
    """Compare the cached reference the pipeline would use against the HP build."""
    params, sim = core.DimerWCAParams(), core.SimConfig()
    grid = np.linspace(sim.z_min, sim.z_max, sim.n_grid)
    emask = core.eval_window_mask_np(grid, sim)
    cached_path = jobs.ti_cache_path(os.path.join(ROOT, "cache"), params.a, sim.n_grid)

    out = dict(
        cached_path=os.path.relpath(cached_path, ROOT),
        hp_path=os.path.relpath(HP, ROOT),
        cached_exists=os.path.exists(cached_path), hp_exists=os.path.exists(HP),
        physics=dict(n_dim=params.n_dim, a=params.a, sigma=params.sigma,
                     epsilon=params.epsilon, h=params.h, w=params.w,
                     beta=params.beta),
        grid=dict(z_min=sim.z_min, z_max=sim.z_max, n_grid=sim.n_grid),
        eval_mask=dict(lo=sim.eval_z_lo, hi=sim.eval_z_hi,
                       n_points=int(emask.sum())),
        checks={}, blocking=[])

    if not out["hp_exists"]:
        out["blocking"].append("no high-precision reference on disk")
        out["verdict"] = "FAIL"
        return out

    with np.load(HP, allow_pickle=True) as d:
        hp = {k: d[k] for k in d.files}
    with open(HP_AUDIT) as fh:
        audit = json.load(fh)

    # 1. physics
    ok_phys = all(abs(float(audit["cell"][k]) - float(out["physics"][k])) < 1e-12
                  for k in audit["cell"])
    out["checks"]["physics_match"] = bool(ok_phys)
    out["hp_cell"] = audit["cell"]
    if not ok_phys:
        out["blocking"].append("HP reference was built at different physical parameters")

    # 2. grid / CV
    ok_grid = (hp["grid"].shape == grid.shape
               and np.allclose(hp["grid"], grid, atol=1e-6))
    out["checks"]["grid_match"] = bool(ok_grid)
    if not ok_grid:
        out["blocking"].append("HP reference grid is not the sampler's grid")

    # 3. how much does the correction move the score, against the effect it scored?
    Fn, Fc = hp["free_energy"], hp["cached_free_energy"]
    dF = Fn - Fc
    l2_shift = float(np.sqrt(np.mean(dF[emask] ** 2)))
    out["checks"]["l2_reference_shift_on_eval_mask"] = l2_shift
    out["checks"]["l2_reference_shift_reported_by_audit"] = float(
        audit["l2_F_new_minus_cached"])
    out["checks"]["max_abs_mean_force_delta"] = float(
        audit["max_abs_delta_mean_force"])
    out["checks"]["Fprime_at_0.25_new_vs_cached"] = [
        float(audit["Fp_at_0p25_new"]), float(audit["Fp_at_0p25_cached"])]
    out["checks"]["sigma_of_that_discrepancy"] = float(
        (audit["Fp_at_0p25_cached"] - audit["Fp_at_0p25_new"])
        / audit["se_prep_at_0p25"])

    # 4. is the HP build itself dense enough to be quoted pointwise?
    n_ti = int(hp["z_ti"].size)
    spacing = float(np.median(np.diff(hp["z_ti"])))
    grid_dz = float(grid[1] - grid[0])
    out["checks"]["hp_n_z"] = n_ti
    out["checks"]["hp_z_spacing"] = spacing
    out["checks"]["eval_grid_spacing"] = grid_dz
    out["checks"]["hp_points_per_eval_point"] = spacing / grid_dz
    dense_enough = spacing <= 2.0 * grid_dz
    out["checks"]["hp_dense_enough_for_pointwise_Fprime"] = bool(dense_enough)
    if not dense_enough:
        out["blocking"].append(
            f"HP reference has {n_ti} z-values at spacing {spacing:.3f} against an "
            f"evaluation grid at {grid_dz:.4f}; F' is interpolated by a factor "
            f"{spacing / grid_dz:.1f} and the audit itself states a denser build "
            f"is needed to quote a corrected F' pointwise")

    # 5. the decisive ratio: correction size against the effect it would score
    abf_err = 0.0901            # Case IX median final L2(F) for the ABF arm
    effect = 0.2283 * abf_err   # the -22.83 % effect the reference was used to measure
    out["checks"]["case_ix_abf_error"] = abf_err
    out["checks"]["case_ix_effect_size"] = effect
    out["checks"]["shift_over_effect"] = float(
        audit["l2_F_new_minus_cached"] / effect)
    if audit["l2_F_new_minus_cached"] > effect:
        out["blocking"].append(
            f"the reference correction ({audit['l2_F_new_minus_cached']:.4f}) is "
            f"{audit['l2_F_new_minus_cached'] / effect:.1f}x the effect size it "
            f"would be used to score")

    out["verdict"] = "FAIL" if out["blocking"] else "PASS"
    out["consequence"] = (
        "A0 diagnostics only; no WCA speedup may be reported tonight."
        if out["verdict"] == "FAIL" else
        "WCA confirmatory comparison is licensed.")
    return out


def probe():
    """Rule R-OBS for WCA: a short A0 run at full density, then the lag search."""
    device = core.choose_device()
    params, sim = core.DimerWCAParams(), core.SimConfig()
    sim_p = core.replace(sim, n_steps=40_000, seed=900)
    engine = core.WCADimerEngine(params, device=device, dtype=core.DTYPE)
    cfg = io_abf.IOConfig(n_cells=io_abf.cells_for_walkers(sim.n_replicas),
                          obs_every=1, opportunity_every=sim_p.n_steps // 4,
                          history_capacity=4000)
    t0 = time.time()
    diag = core.run_sampler_gpu("abf", params, sim_p, engine, verbose=False,
                                io={"arm": "A0", "cfg": cfg, "keep_series": True})
    out = io_abf.probe_obs_every(diag["io_series"][None], dense_obs_every=1)
    out.update(system="wca", probe_steps=int(sim_p.n_steps), dt=float(sim.dt),
               structural_obs_every=io_abf.cadence_for_run(sim.n_steps)["obs_every"],
               obs_interval_time=float(out["obs_every"]) * sim.dt,
               wall_seconds=time.time() - t0)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "r_obs.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(json.dumps(out, indent=2, default=str))
    return out


def screening(n_seeds, gpu_tag, seeds=None):
    """A0 only, instrumented.  Measures Gamma; changes no dynamics."""
    device = core.choose_device()
    params, sim = core.DimerWCAParams(), core.SimConfig()
    engine = core.WCADimerEngine(params, device=device, dtype=core.DTYPE)
    rp = os.path.join(OUT, "r_obs.json")
    if not os.path.exists(rp):
        raise SystemExit("[wca] run --mode probe first (rule R-OBS)")
    with open(rp) as fh:
        cad = io_abf.cadence_for_run(sim.n_steps,
                                     obs_every_override=int(json.load(fh)["obs_every"]))
    cfg = io_abf.IOConfig(n_cells=io_abf.cells_for_walkers(sim.n_replicas), **cad)
    seeds = seeds if seeds is not None else list(range(1000, 1000 + n_seeds))
    os.makedirs(os.path.join(OUT, "screening"), exist_ok=True)
    print(f"[wca] A0 screening: {len(seeds)} seeds, J={cfg.n_cells}, "
          f"obs_every={cfg.obs_every}, opp_every={cfg.opportunity_every}", flush=True)
    for sd in seeds:
        path = os.path.join(OUT, "screening", f"A0__seed{sd:04d}.npz")
        if os.path.exists(path):
            print(f"  seed {sd}: exists, skipped", flush=True)
            continue
        s = core.replace(sim, seed=int(sd))
        t0 = time.time()
        diag = core.run_sampler_gpu("abf", params, s, engine, verbose=False,
                                    io={"arm": "A0", "cfg": cfg})
        keep = {k: v for k, v in diag.items()
                if k.startswith("io_") or k in ("steps", "times", "mean_force",
                                                "pmf", "eff_counts", "p_hat",
                                                "frac_compact", "frac_transition",
                                                "frac_stretched")}
        keep["seed"] = sd
        keep["wall_seconds"] = time.time() - t0
        np.savez_compressed(path, **{k: np.asarray(v) for k, v in keep.items()})
        print(f"  seed {sd}: {time.time() - t0:.0f}s -> {os.path.basename(path)}",
              flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["gate", "probe", "screen"], required=True)
    ap.add_argument("--n-seeds", type=int, default=6)
    ap.add_argument("--seeds", type=str, default="")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.mode == "probe":
        probe()
    elif a.mode == "gate":
        out = reference_gate()
        with open(os.path.join(OUT, "reference_gate.json"), "w") as fh:
            json.dump(out, fh, indent=2, default=str)
        print(json.dumps(out, indent=2, default=str))
    else:
        seeds = ([int(x) for x in a.seeds.split(",")] if a.seeds else None)
        screening(a.n_seeds, "", seeds)


if __name__ == "__main__":
    main()
