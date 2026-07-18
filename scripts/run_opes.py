#!/usr/bin/env python3
"""Run an OPES_METAD study (WCA dimer) from a YAML config. Additive companion to
``run_wca_followup.py`` -- OWN output_root, OWN run_ids, shares the cached TI
references (cache/phase) keyed by physics.

One ``.npz`` per run under ``<output_root>/raw/`` (idempotent skip unless
--overwrite). Shard a stage across at most TWO GPUs from the allow-list {4,5,7}
with --shard/--num-shards. Writes a manifest per stage.

Examples
--------
python scripts/run_opes.py --config configs/opes_wca.yaml --stage representative --dry-run
CUDA_VISIBLE_DEVICES=4 python scripts/run_opes.py --config configs/opes_wca.yaml --stage validate
CUDA_VISIBLE_DEVICES=4 python scripts/run_opes.py ... --shard 0 --num-shards 2
CUDA_VISIBLE_DEVICES=5 python scripts/run_opes.py ... --shard 1 --num-shards 2
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np  # noqa: E402
import wca_abffr_core as core  # noqa: E402
import opes_jobs as oj  # noqa: E402
import wca_phase_jobs as pj  # noqa: E402


def _ms_per_step(n_dim, n_replicas=1024):
    return 2.0 * (n_replicas / 1024.0) * (1.0 if n_dim <= 10 else (n_dim / 10) ** 4)


def _runtime_estimate(specs):
    return sum(_ms_per_step(s.n_dim, s.n_replicas) * 1e-3 * (s.n_steps + 1) for s in specs)


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--stage", required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-runs", type=int, default=None)
    p.add_argument("--seeds", default=None)
    p.add_argument("--cells", default=None)
    p.add_argument("--methods", default=None)
    p.add_argument("--precompute-references", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def _filter(specs, args):
    if args.seeds:
        keep = {int(x) for x in args.seeds.split(",")}
        specs = [s for s in specs if s.seed in keep]
    if args.cells:
        subs = [c.strip() for c in args.cells.split(",") if c.strip()]
        specs = [s for s in specs if any(c in s.physics_tag() for c in subs)]
    if args.methods:
        keep = {m.strip() for m in args.methods.split(",")}
        specs = [s for s in specs if (s.name in keep or s.method in keep)]
    return specs


def dry_run_summary(cfg, stage, specs, raw_dir):
    n_done = sum(1 for s in specs if oj.run_is_valid(oj.run_npz_path(raw_dir, s)))
    todo = [s for s in specs if not oj.run_is_valid(oj.run_npz_path(raw_dir, s))]
    secs = _runtime_estimate(todo)
    print("=" * 78)
    print(f"DRY RUN  study={cfg.get('experiment_name','?')} stage={stage} (OPES)")
    print("=" * 78)
    print(f"  cells   : {sorted({s.physics_tag() for s in specs})}")
    print(f"  methods : {sorted({s.name for s in specs})}")
    print(f"  seeds   : {sorted({s.seed for s in specs})}")
    print(f"  barriers: {sorted({s.barrier for s in specs})}")
    print(f"  paces   : {sorted({s.pace for s in specs})}")
    print(f"  sigmas  : {sorted({s.sigma for s in specs})}")
    print(f"  total runs: {len(specs)}  (done={n_done}, todo={len(todo)})")
    print(f"  TI references (distinct physics): {len(oj.distinct_physics(specs))}")
    print(f"  est. compute for TODO: {secs/3600:.2f} GPU-h (~{secs/3600/2:.2f} h on 2 GPUs)")
    print(f"  output: {raw_dir}")
    print("=" * 78)


def precompute_references(specs, base, cache_dir, verbose, shard, num_shards):
    engines = {}
    reps = oj.distinct_physics(specs)
    if num_shards > 1:
        reps = [s for i, s in enumerate(reps) if i % num_shards == shard]
    print(f"[refs] shard {shard}/{num_shards}: {len(reps)} TI references -> {cache_dir}")
    for k, s in enumerate(reps):
        eng = oj.get_engine(s, engines)
        path = pj.ti_cache_path(cache_dir, s, oj.build_sim(s, base).n_grid)
        existed = os.path.exists(path)
        oj.get_reference(s, base, eng, cache_dir, verbose=verbose)
        print(f"  [{k+1}/{len(reps)}] {s.physics_tag()} {'(cached)' if existed else '(computed)'}")


def write_manifest(out_dir, cfg, stage, specs, args):
    os.makedirs(out_dir, exist_ok=True)
    man = dict(script="run_opes.py", git_commit=_git_commit(), argv=sys.argv,
               config=os.path.abspath(args.config), experiment_name=cfg.get("experiment_name"),
               stage=stage, code_version="opes_v1", timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
               cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""), device=str(core.DEVICE),
               cells=sorted({s.physics_tag() for s in specs}), methods=sorted({s.name for s in specs}),
               seeds=sorted({s.seed for s in specs}), n_runs=len(specs),
               shard=args.shard, num_shards=args.num_shards)
    path = os.path.join(out_dir, f"manifest_opes_{stage}_shard{args.shard}of{args.num_shards}.json")
    with open(path, "w") as fh:
        json.dump(man, fh, indent=2, default=str)
    print(f"[manifest] wrote {path}")


def main(argv=None):
    args = parse_args(argv)
    cfg = oj.load_yaml(args.config)
    raw_dir = os.path.join(cfg["output_root"], "raw")
    cache_dir = cfg.get("cache_dir", "cache/phase")
    os.makedirs(raw_dir, exist_ok=True); os.makedirs(cache_dir, exist_ok=True)
    base = oj.effective_base(cfg, args.stage)
    all_specs = sorted(oj.expand_stage(cfg, args.stage), key=lambda s: s.run_id())
    all_specs = _filter(all_specs, args)

    if args.dry_run:
        dry_run_summary(cfg, args.stage, all_specs, raw_dir); return 0
    if args.precompute_references:
        precompute_references(all_specs, base, cache_dir, args.verbose, args.shard, args.num_shards); return 0

    write_manifest(cfg["output_root"], cfg, args.stage, all_specs, args)
    specs = all_specs
    if args.num_shards > 1:
        specs = [s for i, s in enumerate(specs) if i % args.num_shards == args.shard]
    if args.max_runs:
        specs = specs[:args.max_runs]

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
    print(f"[run] OPES study={cfg.get('experiment_name')} stage={args.stage} "
          f"shard={args.shard}/{args.num_shards} jobs={len(specs)} device={core.DEVICE} CVD={cvd}")
    engines = {}
    n_done = n_skip = n_fail = n_nan = 0
    for k, spec in enumerate(specs):
        path = oj.run_npz_path(raw_dir, spec)
        if not args.overwrite and oj.run_is_valid(path):
            n_skip += 1; continue
        try:
            engine = oj.get_engine(spec, engines)
            out = oj.execute_opes_run(spec, base, engine, cache_dir, verbose=args.verbose)
            oj.save_run(path, out)
            n_done += 1
            if bool(out["had_nan"]):
                n_nan += 1
            print(f"  [{k+1}/{len(specs)}] {spec.name} {spec.physics_tag()} seed{spec.seed} "
                  f"b{spec.barrier:g}p{spec.pace}s{spec.sigma:g}: L2(F)={out['l2_f']:.4f} "
                  f"L2(Fp)={out['l2_fp']:.4f} neff={out['opes_neff_frac_final']:.2f} "
                  f"rtrip={out['n_round_trips']} {'NAN!' if out['had_nan'] else ''} ({out['wall_seconds']:.0f}s)")
        except Exception as exc:
            n_fail += 1
            fpath = oj.save_failure(raw_dir, spec, exc)
            print(f"  [{k+1}/{len(specs)}] {spec.run_id()} FAILED: {exc!r} -> {fpath}")
    print(f"[run] DONE done={n_done} skipped={n_skip} failed={n_fail} nan={n_nan}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
