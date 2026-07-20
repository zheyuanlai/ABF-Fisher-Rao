#!/usr/bin/env python3
"""Run the united-atom alkane ABF vs marginal-Fisher--Rao study from a YAML config.

One .npz per job under ``<output_root>/raw/``; valid results are skipped on restart
(checkpoint/resume). Seed-batched: each job runs all its seeds in one GPU process.

GPU POLICY: uses exactly the ONE GPU exposed via CUDA_VISIBLE_DEVICES (must be from
{4,5,6,7}); production asserts torch.cuda.device_count() == 1.

Examples
--------
  python scripts/run_alkanes.py --config configs/alkanes/smoke.yaml --stage smoke --dry-run
  CUDA_VISIBLE_DEVICES=7 python scripts/run_alkanes.py --config configs/alkanes/smoke.yaml --stage smoke
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from alkanes import jobs as J  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--stage", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--max-runs", type=int, default=None)
    p.add_argument("--only-method", default=None, help="comma list of method names to run")
    p.add_argument("--require-single-gpu", action="store_true",
                   help="assert exactly one visible CUDA device (production safety)")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def _est_seconds(specs):
    # ~0.028 s/step for a seed-batched job (H200, float64), roughly batch-flat
    return sum(0.028 * (s.n_steps + 1) for s in specs)


def main(argv=None):
    args = parse_args(argv)
    cfg = J.load_yaml(args.config)
    out_root = cfg["output_root"]
    raw_dir = os.path.join(out_root, "raw")
    cache_dir = cfg.get("cache_dir", "cache/alkanes")
    os.makedirs(raw_dir, exist_ok=True); os.makedirs(cache_dir, exist_ok=True)

    specs = J.expand_stage(cfg, args.stage)
    if args.only_method:
        keep = set(args.only_method.split(","))
        specs = [s for s in specs if s.name in keep]
    specs = sorted(specs, key=lambda s: s.run_id())

    if args.dry_run:
        todo = [s for s in specs if not J.run_is_valid(J.run_npz_path(raw_dir, s))]
        print("=" * 74)
        print(f"DRY RUN stage={args.stage} config={cfg.get('experiment_name','?')}")
        print("=" * 74)
        cells = sorted({s.physics_tag() for s in specs})
        print(f"  jobs (method x cell)   : {len(specs)}  (done={len(specs)-len(todo)}, todo={len(todo)})")
        print(f"  physics cells          : {len(cells)}")
        for c in cells:
            print(f"     {c}")
        print(f"  methods                : {sorted({s.name for s in specs})}")
        print(f"  seeds/job              : {len(specs[0].seeds)}  n_steps={specs[0].n_steps} N={specs[0].n_replicas}")
        print(f"  est. compute (todo)    : {_est_seconds(todo)/3600:.2f} GPU-h")
        print(f"  output                 : {raw_dir}")
        return 0

    if args.device == "cuda":
        assert torch.cuda.is_available(), "CUDA not available"
        if args.require_single_gpu:
            assert torch.cuda.device_count() == 1, \
                f"expected exactly 1 visible GPU, saw {torch.cuda.device_count()} (set CUDA_VISIBLE_DEVICES)"
    device = args.device
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
    print(f"[run] stage={args.stage} jobs={len(specs)} device={device} CUDA_VISIBLE_DEVICES={cvd}")

    if args.max_runs:
        specs = specs[:args.max_runs]
    n_done = n_skip = n_fail = n_nan = 0
    for k, spec in enumerate(specs):
        path = J.run_npz_path(raw_dir, spec)
        if not args.overwrite and J.run_is_valid(path):
            n_skip += 1
            continue
        try:
            out, per_seed = J.execute_run(spec, device, cache_dir=cache_dir, verbose=args.verbose)
            J.save_run(path, out)
            n_done += 1
            if bool(out["had_nan"]):
                n_nan += 1
            med_F = float(np.median([s["final_l2_F"] for s in per_seed]))
            print(f"  [{k+1}/{len(specs)}] {spec.molecule} {spec.name} b{spec.beta:g} "
                  f"{spec.init_mode}: medL2(F)={med_F:.4f} "
                  f"repl={out['birth_hist'].sum():.0f} "
                  f"{'NAN!' if out['had_nan'] else ''} ({out['wall_seconds']:.0f}s)")
        except Exception as exc:
            n_fail += 1
            fpath = J.save_failure(raw_dir, spec, exc)
            import traceback; traceback.print_exc()
            print(f"  [{k+1}/{len(specs)}] {spec.run_id()} FAILED: {exc!r} -> {fpath}")
    print(f"[run] DONE done={n_done} skipped={n_skip} failed={n_fail} nan={n_nan}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
