#!/usr/bin/env python3
"""Select the one clean-v2 schedule that goes to fresh seeds, and emit its configs.

The selection rule is frozen in :mod:`abffr.accel` and is applied here without
discretion:

1. keep only schedules that pass the pre-declared acceleration screen;
2. rank by the stringent free-energy speedup ``S_{F,2}``;
3. among schedules within 5% of the leader, prefer fewer replacements, then a
   larger ``fr_every``, then a smaller ``gamma``

-- that is, the *sparsest intervention that buys essentially the same
acceleration*.

The Stage-3 and Stage-4 configs do not exist in the repository until this script
writes them.  That is the point: a confirmation config cannot be authored before
the pilot has spoken, so "no retuning after seeing confirmatory seeds" is
enforced by the order in which files come into existence rather than by
resolve.  Both outputs refuse to overwrite without ``--force``.

Example
-------
  python scripts/select_clean_v2_schedule.py \
      --acceleration results/clean_v2/stage2_pilot/pilot/acceleration.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from abffr import accel, clean_v2  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CONF_SEEDS = list(range(3000, 3032))
LONG_SEEDS = list(range(4000, 4008))


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--acceleration", required=True,
                   help="acceleration.csv written by analyze_clean_v2.py.")
    p.add_argument("--config-dir", default=os.path.join(ROOT, "configs", "clean_v2"))
    p.add_argument("--out-json", default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--ignore-screen", action="store_true",
                   help="Rank even when no schedule passed the screen, and write "
                        "ONLY the JSON record -- no Stage 3/4 config.  Use it to "
                        "document what the best cell of a negative pilot was; it "
                        "does not authorise a confirmation run, and now cannot "
                        "produce the files that would start one.")
    return p.parse_args(argv)


def _template(name, output_root, seeds, gamma, fr_every, n_steps, methods,
              batch=24, note=""):
    return {
        "experiment_name": name,
        "output_root": output_root,
        "backend": "torch",
        "device": "cuda",
        "dtype": "float64",
        "batch_size_configs": batch,
        "clean_v2": {"enabled": True},
        "selection": {"write_generic_best": False},
        "potential": {"x_tilt": 0.1021665783},
        "domain": {"x_min": -3.0, "x_max": 3.0, "y_min": -2.5, "y_max": 3.5,
                   "nx_ref": 801, "ny_ref": 801, "nx_profile": 401},
        "simulation": {
            "beta": 4.0, "dt": 0.002, "n_steps": int(n_steps),
            "n_particles": 256, "eval_every": 500, "seeds": list(seeds),
            "x_init_mode": "uniform", "y_init_mode": "uniform"},
        "abf": {"estimator": "binned_smooth",
                "observation_order": "post_propagation",
                "h": 0.05, "update_every": 10, "min_count": 1.0},
        "fr": {
            "enabled": True,
            "target_types": sorted({
                "physical" if m == "abf_fr_physical" else "physical_oracle"
                for m in methods if m != "abf_only"}),
            "gamma_values": [float(gamma)],
            "eta_values": [0.10],
            "burnin_fractions": [0.20],
            "duration_fractions": [0.60],
            "fr_every_values": [int(fr_every)],
            "interval_scaled_clock": True,
            "noise_chunk_steps": 1000},
        "methods": list(methods),
        "_note": note,
    }


def _write(path, cfg, force, header):
    if os.path.exists(path) and not force:
        raise SystemExit(
            f"{path} already exists; refusing to overwrite a frozen "
            f"confirmation config.  Pass --force only if the pilot itself was "
            f"re-run.")
    with open(path, "w") as fh:
        fh.write(header)
        yaml.safe_dump(cfg, fh, sort_keys=False)
    print(f"[select] wrote {os.path.relpath(path)}")


def main(argv=None):
    args = parse_args(argv)
    df = pd.read_csv(args.acceleration)
    df = df[df["target_type"] == "physical"]      # the oracle is never a candidate
    if df.empty:
        raise SystemExit("no deployable (physical-target) schedules in the pilot")

    passed = df[df["promising"].astype(bool)]
    if passed.empty:
        msg = ("No pilot schedule met the pre-declared acceleration screen "
               f"(S_F >= {accel.PILOT_MIN_S_F} at both thresholds). "
               "Under the frozen protocol this is Case C: physical-target "
               "intermittent FR does not accelerate this benchmark at this "
               "dose/schedule.  Do not invent a new target.")
        if not args.ignore_screen:
            raise SystemExit(msg + "\n(Use --ignore-screen only to record the "
                                   "best cell of a negative pilot.)")
        print("[select] WARNING: " + msg)
        passed = df
        screened = False
    else:
        screened = True

    best = float(passed["S_F_2"].max())
    pool = passed[[accel.within_tolerance(best, v) for v in passed["S_F_2"]]]
    pool = pool.assign(_key=[
        accel.rank_key(r["S_F_2"], r["replacement_fraction"], r["fr_every"],
                       r["gamma"]) for _, r in pool.iterrows()])
    chosen = pool.sort_values("_key").iloc[0]

    n_passed = int(df["promising"].astype(bool).sum())
    print(f"[select] {len(df)} schedules analysed, {n_passed} passed the "
          f"screen{'' if screened else ' (screen OVERRIDDEN: ranking all)'}, "
          f"{len(pool)} within 5% of the leading S_F,2={best:.3f}")
    print(f"[select] CHOSEN gamma={chosen['gamma']:g} fr_every={int(chosen['fr_every'])} "
          f"S_F,1={chosen['S_F_1']:.3f} S_F,2={chosen['S_F_2']:.3f} "
          f"replacements/pulse={chosen['replacement_fraction']:.3f}")
    if bool(chosen.get("inflated_F_1")) or bool(chosen.get("inflated_F_2")):
        # pilot_promising already refuses these, so reaching here means the
        # screen was overridden.  Say it out loud rather than let a boundary
        # case pass quietly.
        print("[select] WARNING: the chosen cell has MORE arm censoring than "
              "baseline censoring at a free-energy threshold; its S is inflated "
              "there and it is not a valid pilot winner.")
    at_edge = float(chosen["gamma"]) in (0.002, 0.05)
    if at_edge:
        print(f"[select] NOTE: gamma={chosen['gamma']:g} sits at an edge of the "
              f"frozen grid, so the dose optimum is boundary-limited.  The "
              f"protocol does NOT extend the grid: this campaign tests whether "
              f"the method accelerates, not where the optimal gamma is.  Report "
              f"the schedule as boundary-limited and take it to fresh seeds.")

    n_steps = 50000                     # the frozen Stage-3 horizon
    n_opportunities = len(clean_v2.firing_steps(
        n_steps, float(chosen["burnin_fraction"]),
        float(chosen["stop_fraction"]), int(chosen["fr_every"])))
    payload = {
        "gamma": float(chosen["gamma"]),
        "fr_every": int(chosen["fr_every"]),
        "burnin_fraction": float(chosen["burnin_fraction"]),
        "stop_fraction": float(chosen["stop_fraction"]),
        "scope": str(chosen["scope"]),
        "pilot_S_F_1": float(chosen["S_F_1"]),
        "pilot_S_F_2": float(chosen["S_F_2"]),
        "pilot_replacement_fraction": float(chosen["replacement_fraction"]),
        "pilot_passed_screen": bool(chosen["promising"]),
        "gamma_at_grid_edge": bool(float(chosen["gamma"]) in (0.002, 0.05)),
        "n_schedules_considered": int(len(df)),
        "n_passed_screen": n_passed,
        "screen_overridden": not screened,
        "source": os.path.relpath(args.acceleration),
        # Dimensionless transfer quantities: what a harder benchmark inherits.
        # 500 MD steps do not mean the same thing on two simulators, so the
        # schedule travels as run fractions, a pulse count and an integrated
        # dose -- never as an absolute stride.
        "transfer": {
            "burnin_fraction": float(chosen["burnin_fraction"]),
            "active_window_fraction": float(chosen["stop_fraction"])
                                      - float(chosen["burnin_fraction"]),
            "n_fr_opportunities": n_opportunities,
            "integrated_fr_dose": float(chosen["gamma"]) * (
                float(chosen["stop_fraction"]) - float(chosen["burnin_fraction"])
            ) * n_steps * 0.002,
        },
    }
    out_json = args.out_json or os.path.join(
        os.path.dirname(args.acceleration), "selected_schedule.json")
    if os.path.exists(out_json) and not args.force:
        raise SystemExit(f"{out_json} exists; pass --force to re-select.")
    with open(out_json, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[select] wrote {os.path.relpath(out_json)}")

    if not screened:
        print("[select] --ignore-screen: the JSON record is written, but NO "
              "Stage 3/4 config.  A negative pilot does not get a confirmation "
              "run, and the files that would start one are not created.")
        return 0

    hdr3 = (
        "# Clean-v2 Stage 3: fresh-seed confirmation.  GENERATED by\n"
        "# scripts/select_clean_v2_schedule.py from the Stage-2 pilot -- do not\n"
        "# hand-edit the schedule.  One schedule, three arms, 32 fresh seeds\n"
        "# (3000-3031), disjoint from every earlier stage.  The oracle arm is a\n"
        "# diagnostic and is never a candidate method.\n"
        f"# Selected from: {os.path.relpath(args.acceleration)}\n"
        f"# Pilot S_F,1={chosen['S_F_1']:.4f}  S_F,2={chosen['S_F_2']:.4f}\n")
    hdr4 = (
        "# Clean-v2 Stage 4: long-horizon sanity, 2T with FR still off after\n"
        "# 0.8 of the ORIGINAL T.  GENERATED by\n"
        "# scripts/select_clean_v2_schedule.py -- do not hand-edit.\n"
        "# The expected picture is the FR curve dropping first and plain ABF\n"
        "# catching up: acceleration changes the convergence time, not the\n"
        "# free-energy limit.\n")

    cfg3 = _template(
        "clean_v2_stage3_confirmation", "results/clean_v2/stage3_confirmation",
        CONF_SEEDS, payload["gamma"], payload["fr_every"], n_steps,
        ["abf_only", "abf_fr_physical", "abf_fr_physical_oracle"],
        note="Stage 3: fresh-seed confirmation of the frozen schedule.")
    # Stage 4 doubles the horizon while keeping FR off after the ORIGINAL 0.8T,
    # so the window fractions halve.
    cfg4 = _template(
        "clean_v2_stage4_long_horizon", "results/clean_v2/stage4_long_horizon",
        LONG_SEEDS, payload["gamma"], payload["fr_every"], 2 * n_steps,
        ["abf_only", "abf_fr_physical"], batch=16,
        note="Stage 4: 2T sanity run; FR window is 0.10-0.40 of 2T, i.e. the "
             "same physical [0.2T, 0.8T] as Stage 3.")
    cfg4["fr"]["burnin_fractions"] = [0.10]
    cfg4["fr"]["duration_fractions"] = [0.30]

    _write(os.path.join(args.config_dir, "stage3_confirmation.yaml"), cfg3,
           args.force, hdr3)
    _write(os.path.join(args.config_dir, "stage4_long_horizon.yaml"), cfg4,
           args.force, hdr4)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
