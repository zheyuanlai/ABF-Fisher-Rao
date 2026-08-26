#!/usr/bin/env python3
"""Run one stage of the clean-v2 campaign.

Clean-v2 is ABF plus *intermittent* physical-target Fisher--Rao birth--death,
frozen in ``docs/CLEAN_V2_PREREGISTRATION.md``.  The scientific claim is
acceleration -- less physical simulation time to a prescribed free-energy
accuracy -- so nothing in this script selects, ranks or reports on final error.

Stages
------
``gates``         Stage 0, the engineering gates.  Not scientific data.
``calibration``   Stage 1, plain ABF only.  Its product is the frozen thresholds.
``pilot``         Stage 2, the 3 x 3 (L_FR, gamma) schedule map plus baseline.
``confirmation``  Stage 3, fresh seeds, one frozen schedule, three arms.
``long_horizon``  Stage 4, 2T sanity run showing plain ABF catch up.

Examples
--------
  python scripts/run_reference_2d.py --config configs/clean_v2/stage1_calibration.yaml
  python scripts/run_clean_v2.py --config configs/clean_v2/stage1_calibration.yaml \
      --stage calibration

  # tiny local check of the whole path on CPU
  python scripts/run_clean_v2.py --config configs/clean_v2/stage0_gates.yaml \
      --stage gates --device cpu

The config gate runs before anything else: a config that still carries
``score_clip``, ``max_event_fraction``, ``target_ema_alpha``, ``abf.ema_alpha``
or a ``v3:``/``v4:`` block is rejected rather than defaulted, so "those knobs are
gone" is a property of the file on disk.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from abffr import clean_v2, io_utils, parallel  # noqa: E402

STAGES = ["gates", "calibration", "pilot", "confirmation", "long_horizon"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--stage", required=True, choices=STAGES)
    p.add_argument("--tag", default=None,
                   help="CSV/output tag (default 'main'); use one tag per process.")
    p.add_argument("--merge-only", action="store_true",
                   help="Skip simulation; merge tagged CSVs and write the "
                        "per-config summary.")
    p.add_argument("--resume", dest="resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--force", action="store_true", help="Re-run completed run_ids.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-runs", type=int, default=None)
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--conditional-snapshots", default="final",
                   choices=["final", "all"])
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default=None, choices=["float32", "float64"])
    p.add_argument("--batch-size-configs", type=int, default=None)
    p.add_argument("--n-steps", type=int, default=None)
    p.add_argument("--n-particles", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--seeds", default=None, help="Comma-separated seed override.")
    p.add_argument("--output-root", default=None)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = io_utils.load_config(args.config)
    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds else None)
    io_utils.apply_cli_overrides(
        cfg, device=args.device, dtype=args.dtype,
        batch_size_configs=args.batch_size_configs, n_steps=args.n_steps,
        n_particles=args.n_particles, eval_every=args.eval_every, seeds=seeds,
        output_root=args.output_root)

    # Fail here, loudly, rather than three hours into a grid.
    if clean_v2.from_config(cfg) is None:
        raise SystemExit(
            f"{args.config} does not set 'clean_v2: {{enabled: true}}'. "
            f"run_clean_v2.py refuses to run a legacy v2/v3/v4 config; use "
            f"scripts/run_abf_fr_grid_torch.py for those.")

    stage_root = io_utils.stage_dir(cfg, args.stage)
    prefix = io_utils.stage_prefix(args.stage)

    if args.merge_only:
        parallel.merge_stage_csvs(stage_root, prefix)
        parallel.write_config_summaries(stage_root, prefix, cfg)
        print(f"[clean_v2] merge-only done; outputs under "
              f"{os.path.relpath(stage_root)}/")
        return 0

    setup = parallel.prepare_stage(cfg, args.stage, require_csv=True)
    run_seeds = seeds if seeds is not None else [
        int(s) for s in cfg["simulation"]["seeds"]]
    specs = io_utils.build_run_specs(cfg, run_seeds)
    if args.max_runs is not None:
        specs = specs[:args.max_runs]
    tag = args.tag or "main"

    sim = cfg["simulation"]
    horizon = float(sim["n_steps"]) * float(sim["dt"])
    print(f"[clean_v2] stage={args.stage} device={setup['device']} "
          f"dtype={setup['dtype']} T={horizon:g} n_runs={len(specs)} "
          f"configs={len({s.config_id for s in specs})} tag={tag} "
          f"out={os.path.relpath(stage_root)}")
    for spec in sorted({s.config_id for s in specs}):
        print(f"    {spec}")
    if args.dry_run:
        print(f"[clean_v2] dry-run: {len(specs)} runs planned (not executed).")
        return 0

    summary = parallel.run_specs(
        specs, cfg=cfg, stage_root=stage_root, prefix=prefix,
        x_grid=setup["x_grid"], ref=setup["ref"], ev=setup["ev"],
        device=setup["device"], dtype=setup["dtype"],
        estimator=cfg.get("abf", {}).get("estimator", "binned_smooth"),
        batch_size=int(cfg.get("batch_size_configs", 16)),
        base_seed=args.base_seed, tag=tag, resume=args.resume,
        force=args.force, conditional=args.conditional_snapshots)

    parallel.merge_stage_csvs(stage_root, prefix)
    parallel.write_config_summaries(stage_root, prefix, cfg)
    print(f"[clean_v2] DONE done={summary['n_done']} failed={summary['n_failed']} "
          f"skipped={summary['n_skipped']} nan={summary['n_nan']} in "
          f"{summary['wall_seconds']:.1f}s")
    return 1 if summary["n_failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
