#!/usr/bin/env python3
"""Run the guarded information-target pulse--release campaign.

The runner accepts the established batch-execution flags and the calibration
or confirmation stage. It validates the information-target protocol before
building any run specs and, for the oracle campaign, requires the configured
gamma to match the mechanism-only calibration receipt.
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

from abffr import information_target, io_utils, parallel  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True,
                        choices=["calibration", "confirmation"])
    parser.add_argument("--tag", default=None)
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--resume", dest="resume", action="store_true",
                        default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--conditional-snapshots", default="final",
                        choices=["final", "all"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None,
                        choices=["float32", "float64"])
    parser.add_argument("--batch-size-configs", type=int, default=None)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--n-particles", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--output-root", default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = io_utils.load_config(args.config)
    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds else None)
    io_utils.apply_cli_overrides(
        cfg, device=args.device, dtype=args.dtype,
        batch_size_configs=args.batch_size_configs, n_steps=args.n_steps,
        n_particles=args.n_particles, eval_every=args.eval_every, seeds=seeds,
        output_root=args.output_root)

    steps = information_target.validate_config(cfg)
    kind = cfg["information_target"]["kind"]
    if kind == information_target.ORACLE_CAMPAIGN_KIND:
        receipt = information_target.validate_dose_receipt(cfg, ROOT)
        print(f"[information-target] dose_receipt={os.path.relpath(receipt, ROOT)}")
    print(
        f"[information-target] kind={kind} firing_steps={steps}; "
        "operator=fr_v3.bd_standard; target=frozen")

    stage_root = io_utils.stage_dir(cfg, args.stage)
    prefix = io_utils.stage_prefix(args.stage)
    if args.merge_only:
        parallel.merge_stage_csvs(stage_root, prefix)
        parallel.write_config_summaries(stage_root, prefix, cfg)
        print(f"[information-target] merge-only done under "
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
    print(f"[information-target] stage={args.stage} device={setup['device']} "
          f"dtype={setup['dtype']} T={horizon:g} n_runs={len(specs)} "
          f"configs={len({s.config_id for s in specs})} tag={tag} "
          f"out={os.path.relpath(stage_root)}")
    for config_id in sorted({s.config_id for s in specs}):
        print(f"    {config_id}")
    if args.dry_run:
        print(f"[information-target] dry-run: {len(specs)} runs planned")
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
    print(f"[information-target] DONE done={summary['n_done']} "
          f"failed={summary['n_failed']} skipped={summary['n_skipped']} "
          f"nan={summary['n_nan']} in {summary['wall_seconds']:.1f}s")
    return 1 if summary["n_failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
