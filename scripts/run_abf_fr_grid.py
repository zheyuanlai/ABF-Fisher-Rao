#!/usr/bin/env python3
"""Run the ABF / ABF+FR method-and-hyperparameter grid for xi(x, y) = x.

Stages
------
  --stage smoke   tiny grid from a smoke config; writes into <root>/tuning/
  --stage tuning  full grid from a tuning config;  writes into <root>/tuning/
  --stage eval    runs the configs selected in tuning/best_configs.csv; writes
                  into <root>/eval/  (does NOT do its own hyperparameter sweep)

Outputs (PREFIX = "tuning" for smoke/tuning, "eval" for eval), under the stage dir:
  PREFIX_runs_long.csv              per (run, snapshot) time series
  PREFIX_final_summary.csv          per run final + time-integrated metrics
  PREFIX_profiles.csv               per run final profiles F'(x), F(x), p(x), q(x)
  PREFIX_fr_events.csv              per (run, snapshot) Fisher--Rao event stats
  PREFIX_conditional_diagnostics.csv  conditional Y|X diagnostics
  tuning_config_summary.csv         (tuning/smoke only) per-config seed aggregates
  best_configs.csv                  (tuning/smoke only) selected configs for eval

Examples
--------
  python scripts/run_abf_fr_grid.py --config configs/two_dim_xi_x_smoke.yaml --stage smoke
  python scripts/run_abf_fr_grid.py --config configs/two_dim_xi_x_tuning.yaml --stage tuning

This script never runs the large tuning grid on its own -- you must pass the
tuning config and --stage tuning explicitly.  Use --dry-run to preview the run
list and --max-runs to cap it.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from abffr import (  # noqa: E402
    diagnostics,
    io_utils,
    metrics,
    reference,
    simulation,
)
from abffr.io_utils import RunSpec  # noqa: E402

CHECKPOINT_EVERY = 10  # rewrite CSVs every N completed runs (crash safety)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="Path to YAML config.")
    p.add_argument("--stage", required=True, choices=["smoke", "tuning", "eval"])
    p.add_argument("--output-root", default=None, help="Override output_root.")
    p.add_argument("--seeds", default=None,
                   help="Comma-separated seed override, e.g. '0,1,2'.")
    p.add_argument("--max-runs", type=int, default=None,
                   help="Cap the number of runs (safety).")
    p.add_argument("--dry-run", action="store_true",
                   help="List the planned runs and exit without simulating.")
    p.add_argument("--conditional-snapshots", default="final",
                   choices=["final", "all"],
                   help="Which snapshots to evaluate Y|X diagnostics on.")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def _progress(iterable, total, quiet):
    """tqdm if available, else a plain pass-through (optional dependency)."""
    if quiet:
        return iterable
    try:
        from tqdm import tqdm
        return tqdm(iterable, total=total)
    except Exception:
        return iterable


def build_specs_for_stage(cfg, stage, seeds, stage_root):
    """Construct the list of RunSpec for the stage."""
    if stage in ("smoke", "tuning"):
        return io_utils.build_run_specs(cfg, seeds)
    # eval: read the selected configs from tuning/best_configs.csv
    best_path = os.path.join(io_utils.stage_dir(cfg, "tuning"), "best_configs.csv")
    if not os.path.exists(best_path):
        print(f"[run_abf_fr_grid] ERROR: {os.path.relpath(best_path)} not found.\n"
              f"  Run the tuning stage first to produce best_configs.csv, e.g.\n"
              f"    python scripts/run_abf_fr_grid.py --config <tuning.yaml> --stage tuning")
        return []
    best = pd.read_csv(best_path)
    best = best[best.get("selected", True) == True] if "selected" in best else best
    specs = []
    seen = set()
    # Always include the ABF-only baseline for matched-seed comparisons.
    for seed in seeds:
        specs.append(RunSpec("abf_only", "none", seed, 0.0,
                             float(cfg["fr"]["eta_values"][0]), 0.0, 1))
    for _, r in best.iterrows():
        if r["method"] == "abf_only":
            continue
        key = (r["method"], r["target_type"], float(r["gamma"]), float(r["eta"]),
               float(r["burnin_fraction"]), int(r["fr_every"]))
        if key in seen:
            continue
        seen.add(key)
        for seed in seeds:
            specs.append(RunSpec(r["method"], r["target_type"], seed,
                                 float(r["gamma"]), float(r["eta"]),
                                 float(r["burnin_fraction"]), int(r["fr_every"])))
    return specs


def simulate_one(spec: RunSpec, cfg, x_grid, ref, ev):
    """Run one simulation and return (diag, summary_dict)."""
    sim = cfg["simulation"]
    abf = cfg["abf"]
    fr = cfg["fr"]
    rng_init, rng_noise, rng_fr = io_utils.make_rng_streams(spec.seed)
    diag = simulation.run_simulation(
        target_type=spec.target_type,
        beta=float(sim["beta"]), dt=float(sim["dt"]),
        n_steps=int(sim["n_steps"]), n_particles=int(sim["n_particles"]),
        eval_every=int(sim["eval_every"]),
        x_grid=x_grid, F_ref=ref["F_ref"], Fprime_ref=ref["Fprime_ref"],
        domain=cfg["domain"],
        h=float(abf["h"]), eta=float(spec.eta),
        min_count=float(abf.get("min_count", 1.0)),
        ema_alpha=float(abf.get("ema_alpha", 0.05)),
        gamma=float(spec.gamma), burnin_fraction=float(spec.burnin_fraction),
        ramp_fraction=float(fr.get("ramp_fraction", 0.1)),
        fr_every=int(spec.fr_every),
        score_clip=fr.get("score_clip", 5.0),
        max_event_fraction=fr.get("max_event_fraction", 0.10),
        x_init_mode=sim.get("x_init_mode", "mixed"),
        y_init_mode=sim.get("y_init_mode", "mixed"),
        x_barrier=ev.x_barrier,
        rng_init=rng_init, rng_noise=rng_noise, rng_fr=rng_fr,
    )
    summary = metrics.final_summary(diag, x_grid, ref["F_ref"], ref["Fprime_ref"], ev)
    return diag, summary


def _iqr(s):
    s = np.asarray(s, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return float("nan")
    return float(np.percentile(s, 75) - np.percentile(s, 25))


def summarize_configs(final_df):
    """Aggregate per-run final metrics over seeds, grouped by config."""
    rows = []
    keys = ["config_id", "method", "target_type", "gamma", "eta",
            "burnin_fraction", "fr_every"]
    for cid, g in final_df.groupby("config_id"):
        row = {k: g[k].iloc[0] for k in keys}
        row["n_seeds"] = len(g)
        for col in ["final_l2_F", "final_l2_Fprime", "integrated_l2_F",
                    "integrated_l2_Fprime", "final_marginal_l2_uniform",
                    "final_marginal_l2_target", "mean_fr_event_fraction",
                    "max_fr_event_fraction", "barrier_crossings",
                    "frac_left", "frac_barrier", "frac_right"]:
            row[f"median_{col}"] = float(np.nanmedian(g[col]))
            row[f"iqr_{col}"] = _iqr(g[col])
        row["frac_nan"] = float(np.mean(g["any_nan"]))
        rows.append(row)
    return pd.DataFrame(rows)


def select_best_configs(config_df, cfg):
    """Select the best config per FR target type (+ ABF-only) for eval.

    Primary metric : median over seeds of integral ||F_hat_t - F_ref|| dt.
    Secondary      : median over seeds of final ||F_hat_T - F_ref||.
    Safety filters : no NaNs; typical FR event fraction within max_event_fraction;
                     max FR event fraction not grossly above the cap.
    """
    cap = float(cfg["fr"].get("max_event_fraction", 0.10))
    rows = []
    for target in ["none", "estimated", "uniform", "oracle", "self"]:
        sub = config_df[config_df["target_type"] == target].copy()
        if sub.empty:
            continue
        # Safety filter (ABF-only has no FR events, so it always passes).
        passed = (
            (sub["frac_nan"] == 0.0)
            & (sub["median_mean_fr_event_fraction"] <= cap + 1e-9)
            & (sub["median_max_fr_event_fraction"] <= 1.5 * cap + 1e-9)
        )
        sub["passed_safety"] = passed
        pool = sub[passed] if passed.any() else sub
        pool = pool.sort_values(
            ["median_integrated_l2_F", "median_final_l2_F"]).reset_index(drop=True)
        for rank, (_, r) in enumerate(pool.iterrows(), start=1):
            rows.append(dict(
                rank_within_target=rank,
                selected=(rank == 1),
                method=r["method"], target_type=r["target_type"],
                gamma=float(r["gamma"]), eta=float(r["eta"]),
                burnin_fraction=float(r["burnin_fraction"]),
                fr_every=int(r["fr_every"]),
                median_integrated_l2_F=float(r["median_integrated_l2_F"]),
                median_final_l2_F=float(r["median_final_l2_F"]),
                median_final_l2_Fprime=float(r["median_final_l2_Fprime"]),
                median_mean_fr_event_fraction=float(r["median_mean_fr_event_fraction"]),
                passed_safety=bool(r["passed_safety"]),
                n_seeds=int(r["n_seeds"]),
            ))
    return pd.DataFrame(rows)


def main(argv=None):
    args = parse_args(argv)
    cfg = io_utils.load_config(args.config)
    if args.output_root:
        cfg["output_root"] = args.output_root

    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds
             else [int(s) for s in cfg["simulation"]["seeds"]])
    stage_root = io_utils.stage_dir(cfg, args.stage)
    prefix = io_utils.stage_prefix(args.stage)

    # Reference profiles on the simulation x-grid (recomputed for exact
    # grid consistency; reference_profile.csv is for figures/inspection only).
    beta = float(cfg["simulation"]["beta"])
    x_grid = reference.profile_grid(cfg)
    y_quad = np.linspace(cfg["domain"]["y_min"], cfg["domain"]["y_max"],
                         int(cfg["domain"]["ny_ref"]))
    ref = reference.compute_reference(x_grid, y_quad, beta)
    ev = metrics.EvalConfig.from_domain(cfg["domain"])

    if not os.path.exists(os.path.join(io_utils.reference_dir(cfg),
                                       "reference_profile.csv")):
        print("[run_abf_fr_grid] NOTE: reference_profile.csv not found; "
              "run scripts/run_reference_2d.py for reference figures/CSV. "
              "(Reference profiles were recomputed internally for this run.)")

    specs = build_specs_for_stage(cfg, args.stage, seeds, stage_root)
    if not specs:
        return 1
    if args.max_runs is not None:
        specs = specs[:args.max_runs]

    print(f"[run_abf_fr_grid] stage={args.stage}  seeds={seeds}  "
          f"n_runs={len(specs)}  out={os.path.relpath(stage_root)}")
    if args.dry_run:
        for s in specs:
            print("   ", s.run_id)
        print(f"[run_abf_fr_grid] dry-run: {len(specs)} runs planned (not executed).")
        return 0

    cond_idx = None if args.conditional_snapshots == "final" else "all"

    long_rows, final_rows, profile_rows, fr_rows, cond_rows = [], [], [], [], []
    t_start = time.time()

    def checkpoint():
        _write_csv(long_rows, stage_root, f"{prefix}_runs_long.csv")
        _write_csv(final_rows, stage_root, f"{prefix}_final_summary.csv")
        _write_csv(profile_rows, stage_root, f"{prefix}_profiles.csv")
        _write_csv(fr_rows, stage_root, f"{prefix}_fr_events.csv")
        _write_csv(cond_rows, stage_root, f"{prefix}_conditional_diagnostics.csv")

    for i, spec in enumerate(_progress(specs, len(specs), args.quiet)):
        t0 = time.time()
        diag, summary = simulate_one(spec, cfg, x_grid, ref, ev)
        meta = spec.to_row()

        # Per-snapshot long rows + FR event rows.
        ts = metrics.time_series_metrics(diag, x_grid, ref["F_ref"],
                                         ref["Fprime_ref"], ev)
        for r in ts:
            long_rows.append({**meta, **r})
            fr_rows.append({**meta, **{k: r[k] for k in (
                "step", "t", "gamma_eff", "fr_applied", "fr_event_fraction",
                "fr_event_fraction_max", "fr_events_total", "score_mean",
                "score_std", "score_min", "score_max", "n_unique_ancestors")}})

        # Final summary row.
        final_rows.append({**meta, **summary})

        # Final profiles.
        Fp = diag["Fprime_hat"][-1]; F = diag["F_hat"][-1]
        p = diag["p_hat_grid"][-1]; q = diag["q_target_grid"][-1]
        for j in range(len(x_grid)):
            profile_rows.append({**meta, "x": float(x_grid[j]),
                                 "Fprime_hat": float(Fp[j]), "F_hat": float(F[j]),
                                 "p_hat": float(p[j]), "q_target": float(q[j])})

        # Conditional Y|X diagnostics.
        snap_idx = None if cond_idx is None else list(range(len(diag["steps"])))
        cmeta = dict(method=spec.method, target_type=spec.target_type,
                     seed=int(spec.seed))
        cond_rows.extend(diagnostics.conditional_diagnostics(
            diag, cmeta, beta, cfg["domain"], snapshot_indices=snap_idx))

        if not args.quiet:
            print(f"  [{i+1}/{len(specs)}] {spec.run_id}  "
                  f"l2_F={summary['final_l2_F']:.3f} "
                  f"l2_Fp={summary['final_l2_Fprime']:.3f} "
                  f"xcross={summary['barrier_crossings']} "
                  f"fr_evt={summary['mean_fr_event_fraction']:.4f} "
                  f"({time.time()-t0:.1f}s)")

        if (i + 1) % CHECKPOINT_EVERY == 0:
            checkpoint()

    checkpoint()

    # Config aggregation + best-config selection (tuning/smoke only).
    if args.stage in ("smoke", "tuning") and final_rows:
        final_df = pd.DataFrame(final_rows)
        config_df = summarize_configs(final_df)
        config_df.to_csv(os.path.join(stage_root, "tuning_config_summary.csv"),
                         index=False)
        best_df = select_best_configs(config_df, cfg)
        best_df.to_csv(os.path.join(stage_root, "best_configs.csv"), index=False)
        print(f"[run_abf_fr_grid] wrote tuning_config_summary.csv "
              f"({len(config_df)} configs) and best_configs.csv "
              f"({int(best_df['selected'].sum()) if len(best_df) else 0} selected)")

    n_nan = int(pd.DataFrame(final_rows)["any_nan"].sum()) if final_rows else 0
    print(f"[run_abf_fr_grid] DONE  {len(specs)} runs in "
          f"{time.time()-t_start:.1f}s, runs_with_nan={n_nan}")
    print(f"[run_abf_fr_grid] outputs under {os.path.relpath(stage_root)}/")
    return 0


def _write_csv(rows, out_dir, name):
    if not rows:
        return
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, name), index=False)


if __name__ == "__main__":
    raise SystemExit(main())
