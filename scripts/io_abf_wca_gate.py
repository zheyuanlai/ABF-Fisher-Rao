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
#: The CURRENT reference. `wca_hp_v3/` is the full-resolution rebuild -- 160
#: acquisition z-values on the evaluation grid itself, no smoothing, PCHIP -- and
#: `cache/phase_hp_v3/` is its drop-in cache, which the accepted five-arm runner
#: already defaults to. The 41-point `wca_hp_reference/` build is SUPERSEDED and
#: is loaded here only to quote the correction it established.
HP = os.path.join(ROOT, "results", "v2_validity_audits", "wca_hp_v3",
                  "wca_hp_v2.npz")
HP_META = os.path.join(ROOT, "results", "v2_validity_audits", "wca_hp_v3",
                       "metadata.json")
HP_CACHE = os.path.join(ROOT, "cache", "phase_hp_v3",
                        "wca_ti_b1_h2_w2_n10_a1.5_g160.npz")
GATE0 = os.path.join(ROOT, "results", "v2_validity_audits", "wca_gate0_hpv3",
                     "verdict.json")
#: Superseded; quoted, never scored against.
HP_V1 = os.path.join(ROOT, "results", "v2_validity_audits", "wca_hp_reference",
                     "summary.json")

#: Case IX's own numbers, used only to size the reference's uncertainty against
#: the effect a WCA comparison would be trying to resolve.
CASEIX_ABF_ERROR = 0.0901


def reference_gate():
    """Does the CURRENT reference cover what a confirmatory comparison would run?

    The question is not "has the reference ever been wrong" -- it has, and that is
    settled -- but whether the reference a *new* campaign would score against is
    accurate enough for the effect it would resolve.  Those are different
    questions, and answering the first when you meant the second is how a usable
    benchmark gets retired by its own audit.
    """
    params, sim = core.DimerWCAParams(), core.SimConfig()
    grid = np.linspace(sim.z_min, sim.z_max, sim.n_grid)
    emask = core.eval_window_mask_np(grid, sim)

    out = dict(
        reference_used=os.path.relpath(HP, ROOT),
        reference_cache=os.path.relpath(HP_CACHE, ROOT),
        superseded_build=os.path.relpath(os.path.dirname(HP_V1), ROOT),
        physics=dict(n_dim=params.n_dim, a=params.a, sigma=params.sigma,
                     epsilon=params.epsilon, h=params.h, w=params.w,
                     beta=params.beta),
        grid=dict(z_min=sim.z_min, z_max=sim.z_max, n_grid=sim.n_grid),
        eval_mask=dict(lo=sim.eval_z_lo, hi=sim.eval_z_hi,
                       n_points=int(emask.sum())),
        checks={}, blocking=[], caveats=[])

    for path, label in ((HP, "hp_v3 npz"), (HP_META, "hp_v3 metadata"),
                        (HP_CACHE, "hp_v3 cache"), (GATE0, "gate 0 verdict")):
        if not os.path.exists(path):
            out["blocking"].append(f"missing {label}: {path}")
    if out["blocking"]:
        out["verdict"] = "FAIL"
        return out

    with np.load(HP, allow_pickle=True) as d:
        hp = {k: d[k] for k in d.files}
    with open(HP_META) as fh:
        meta = json.load(fh)
    with open(GATE0) as fh:
        g0 = json.load(fh)
    with open(HP_V1) as fh:
        v1 = json.load(fh)

    # 1. physics
    ok_phys = all(abs(float(meta["cell"][k]) - float(out["physics"][k])) < 1e-12
                  for k in meta["cell"])
    out["checks"]["physics_match"] = bool(ok_phys)
    if not ok_phys:
        out["blocking"].append("reference built at different physical parameters")

    # 2. grid and CV
    ok_grid = (hp["grid"].shape == grid.shape
               and np.allclose(hp["grid"], grid, atol=1e-6))
    out["checks"]["grid_match"] = bool(ok_grid)
    if not ok_grid:
        out["blocking"].append("reference grid is not the sampler's grid")

    # 3. acquisition resolution -- the defect that retired the 41-point build
    n_acq = int(meta["n_z_acquisition"])
    dz_acq = float(np.median(np.diff(np.asarray(meta["z_acquisition"]))))
    dz_eval = float(grid[1] - grid[0])
    out["checks"]["n_z_acquisition"] = n_acq
    out["checks"]["dz_acquisition"] = dz_acq
    out["checks"]["dz_evaluation"] = dz_eval
    out["checks"]["interpolation_factor"] = dz_acq / dz_eval
    out["checks"]["smoothing_applied"] = bool(meta["smoothing_applied"])
    resolved = dz_acq <= 1.05 * dz_eval and not meta["smoothing_applied"]
    out["checks"]["acquired_at_evaluation_resolution"] = bool(resolved)
    if not resolved:
        out["blocking"].append(
            f"acquisition dz {dz_acq:.4f} against evaluation dz {dz_eval:.4f}, "
            f"or smoothing applied")

    # 4. the reference's OWN uncertainty, against the effect to be resolved.
    #    se on F' propagated to F in the worst case, i.e. fully correlated across
    #    z, which is the mode a per-preparation systematic would take.
    se_fp = float(meta["max_se_prep"])
    span = float(sim.eval_z_hi - sim.eval_z_lo)
    zc = grid[emask]
    F_unc_rms = se_fp * float(np.sqrt(np.mean((zc - zc.mean()) ** 2)))
    out["checks"]["max_se_prep_on_Fprime"] = se_fp
    out["checks"]["max_se_replica_on_Fprime"] = float(meta["max_se_replica"])
    out["checks"]["worst_case_F_uncertainty_rms"] = F_unc_rms
    out["checks"]["case_ix_abf_final_error"] = CASEIX_ABF_ERROR
    ratio = F_unc_rms / CASEIX_ABF_ERROR
    out["checks"]["reference_uncertainty_over_abf_error"] = ratio
    if ratio > 0.5:
        out["blocking"].append(
            f"reference F uncertainty ({F_unc_rms:.4f}) is {ratio:.2f} of the "
            f"ABF error it would score against")
    elif ratio > 0.15:
        out["caveats"].append(
            f"reference F uncertainty is {ratio:.0%} of ABF's own final error "
            f"under a fully-correlated worst case; it is common to all arms and "
            f"largely cancels in a paired threshold-crossing endpoint, but it "
            f"bounds how fine an effect may be claimed")

    # 5. Gate 0 -- does the conditional equilibrate at fixed z?  Without this the
    #    system is conditional-equilibration-limited and allocation is the wrong
    #    tool regardless of how good the reference is.
    out["checks"]["gate0_pass"] = bool(g0["gate0_pass"])
    out["checks"]["gate0_rel_spread_all"] = float(g0["rel_spread_all"])
    out["checks"]["gate0_rel_spread_transition"] = float(g0["rel_spread_transition"])
    if not g0["gate0_pass"]:
        out["blocking"].append("Gate 0 fails against this reference")

    # 6. context: what the correction was, and that the defective cache is not used
    out["checks"]["superseded_l2_correction"] = float(v1["l2_F_new_minus_cached"])
    out["checks"]["hp_v3_l2_vs_cached"] = float(meta["l2_F_vs_cached"])
    out["checks"]["cached_reference_is_defective"] = True
    out["caveats"].append(
        "the DEFAULT cache `cache/wca_ti_reference.npz` is the defective build; "
        "any WCA run in this campaign must be pointed at cache/phase_hp_v3")

    out["verdict"] = "FAIL" if out["blocking"] else "PASS"
    out["consequence"] = (
        "WCA A0/A6b/A6c confirmatory comparison is licensed, scored against "
        "cache/phase_hp_v3, subject to the caveats above."
        if out["verdict"] == "PASS" else
        "A0 diagnostics only; no WCA speedup may be reported.")
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


def arms_phase(phase, seeds):
    """A0 / A6b / A6c on paired seeds.

    Pairing is by construction rather than by bookkeeping: ``run_sampler_gpu``
    seeds torch once from ``sim.seed`` and IO-ABF draws no randomness at all, so
    two arms on the same seed see the identical initial lattice and the identical
    Langevin noise sequence.  Trajectories diverge because the *force* differs,
    which is the only thing that is allowed to differ.
    """
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
    d = os.path.join(OUT, phase)
    os.makedirs(d, exist_ok=True)
    print(f"[wca] {phase}: {len(seeds)} paired seeds x 3 arms, J={cfg.n_cells}, "
          f"obs_every={cfg.obs_every}", flush=True)
    # Seed-major so an interrupted run leaves whole paired seeds, never a
    # half-paired one: an unpaired arm is worse than a missing arm, because it
    # looks like data.
    for sd in seeds:
        for arm in io_abf.ARMS:
            path = os.path.join(d, f"{arm}__seed{sd:04d}.npz")
            if os.path.exists(path):
                print(f"  {arm} seed {sd}: exists, skipped", flush=True)
                continue
            s_i = core.replace(sim, seed=int(sd))
            t0 = time.time()
            diag = core.run_sampler_gpu("abf", params, s_i, engine, verbose=False,
                                        io={"arm": arm, "cfg": cfg})
            keep = {k: v for k, v in diag.items()
                    if k.startswith("io_") or k in ("steps", "times", "mean_force",
                                                    "pmf", "eff_counts", "p_hat",
                                                    "frac_compact", "frac_transition",
                                                    "frac_stretched")}
            keep["seed"] = sd
            keep["wall_seconds"] = time.time() - t0
            np.savez_compressed(path, **{k: np.asarray(v) for k, v in keep.items()})
            print(f"  {arm} seed {sd}: {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["gate", "probe", "screen", "pilot",
                                       "confirmatory"], required=True)
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
    elif a.mode == "screen":
        seeds = ([int(x) for x in a.seeds.split(",")] if a.seeds else None)
        screening(a.n_seeds, "", seeds)
    else:
        seeds = ([int(x) for x in a.seeds.split(",")] if a.seeds
                 else (list(range(2000, 2008)) if a.mode == "pilot"
                       else list(range(3000, 3032))))
        arms_phase(a.mode, seeds)


if __name__ == "__main__":
    main()
