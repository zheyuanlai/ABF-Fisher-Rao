#!/usr/bin/env python3
"""Run the WCA serial one-walker ABF control at equal force-evaluation budget (Part H).

Additive companion to ``run_wca_followup.py``. Each *batched job* advances all the seeds of
one physics cell together (independent one-walker ABF trajectories, each with its own ABF
accumulators). Jobs are chunked + checkpointed, so an interrupted run resumes without losing
ABF accumulators; a valid per-seed ``.npz`` marks a trajectory done (idempotent skip).

One process per invocation. Shard the batched jobs across at most TWO GPUs (4-7 only) with
``--shard/--num-shards`` (the launcher sets these). Writes a manifest per stage.

Examples
--------
# workload + runtime summary (no compute)
python scripts/run_wca_serial_abf.py --config configs/wca_serial_abf_equal_budget.yaml \
    --stage production --dry-run

# benchmark: real ms/step + extrapolated wall clock for the exact 122.88M-step run
CUDA_VISIBLE_DEVICES=4 python scripts/run_wca_serial_abf.py \
    --config configs/wca_serial_abf_equal_budget.yaml --stage benchmark --benchmark

# mechanical smoke (small checkpoint cadence exercises checkpoint + resume)
CUDA_VISIBLE_DEVICES=4 python scripts/run_wca_serial_abf.py \
    --config configs/wca_serial_abf_equal_budget.yaml --stage smoke --checkpoint-every 5000

# production, one cell on one GPU
CUDA_VISIBLE_DEVICES=4 python scripts/run_wca_serial_abf.py \
    --config configs/wca_serial_abf_equal_budget.yaml --stage production --cells b1_h2
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np  # noqa: E402
import wca_abffr_core as core  # noqa: E402
import wca_serial_abf as sa  # noqa: E402


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
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--benchmark", action="store_true",
                   help="time raw ms/step on one group and print extrapolated wall clock")
    p.add_argument("--bench-steps", type=int, default=None,
                   help="override the benchmark step count (default: stage n_steps)")
    p.add_argument("--precompute-references", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--cells", default=None,
                   help="comma list of physics_tag substrings to keep (e.g. b1_h2,b4_h1)")
    p.add_argument("--seeds", default=None, help="comma list to override the stage seeds")
    p.add_argument("--checkpoint-every", type=int, default=None,
                   help="override serial.checkpoint_every_steps")
    p.add_argument("--max-steps", type=int, default=None,
                   help="cap each cell's target steps (for quick tests)")
    p.add_argument("--ms-per-step", type=float, default=0.5,
                   help="ms/step assumption for the dry-run runtime estimate")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def _filter(specs, args):
    if args.seeds:
        keep = {int(x) for x in args.seeds.split(",")}
        specs = [s for s in specs if s.seed in keep]
    if args.cells:
        subs = [c.strip() for c in args.cells.split(",") if c.strip()]
        specs = [s for s in specs if any(c in s.physics_tag() for c in subs)]
    if args.max_steps:
        specs = [dataclasses.replace(s, n_steps=min(int(s.n_steps), int(args.max_steps))) for s in specs]
    return specs


def dry_run_summary(cfg, stage, specs, raw_dir, ms_per_step):
    groups = sa.group_specs(specs)
    print("=" * 78)
    print(f"DRY RUN  study={cfg.get('experiment_name','?')} stage={stage} (serial one-walker ABF)")
    print("=" * 78)
    total_secs = 0.0
    for (tag, nsteps, _st), gspecs in sorted(groups.items()):
        done = sum(1 for s in gspecs if sa.run_is_valid(sa.run_npz_path(raw_dir, s)))
        secs = ms_per_step * 1e-3 * (nsteps + 1)      # batched: wall ~ one trajectory's loop
        total_secs += 0.0 if done == len(gspecs) else secs
        print(f"  {tag:22s} T={nsteps:>11,d} budget={nsteps:>13,d} "
              f"seeds={len(gspecs):2d} done={done}/{len(gspecs)}  ~{secs/3600:.2f} h/job")
    print(f"  batched jobs: {len(groups)}   (one process/GPU; shard across <=2 GPUs)")
    print(f"  est. remaining wall @ {ms_per_step:.2f} ms/step: {total_secs/3600:.2f} h "
          f"(serial across shards); measure with --benchmark")
    print(f"  TI refs (distinct physics): {len(sa.distinct_physics(specs))} (cached in {cfg.get('cache_dir')})")
    print(f"  output: {raw_dir}")
    print("=" * 78)


def precompute_references(specs, base, cache_dir, verbose, shard, num_shards):
    engines = {}
    reps = sa.distinct_physics(specs)
    if num_shards > 1:
        reps = [s for i, s in enumerate(reps) if i % num_shards == shard]
    print(f"[refs] shard {shard}/{num_shards}: {len(reps)} TI references <- {cache_dir}")
    for k, s in enumerate(reps):
        eng = sa.get_engine(s, engines)
        sa.get_reference(s, base, eng, cache_dir, verbose=verbose)
        print(f"  [{k+1}/{len(reps)}] {s.physics_tag()} ok")


def run_benchmark(cfg, stage, specs, base, cache_dir, ser, args):
    groups = sa.group_specs(specs)
    (tag, nsteps, _st), gspecs = sorted(groups.items())[0]
    bench_steps = int(args.bench_steps or min(nsteps, 1_000_000))
    engines = {}
    engine = sa.get_engine(gspecs[0], engines)
    ref = sa.get_reference(gspecs[0], base, engine, cache_dir, verbose=args.verbose)
    print(f"[bench] {tag}: {len(gspecs)} trajectories x {bench_steps:,d} steps on {core.DEVICE} "
          f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')})")
    res = sa.run_serial_abf_batched(
        gspecs, base, engine, ref, ckpt_path="/dev/null", bench_steps=bench_steps,
        verbose=False, base_budget=ser["base_budget"], ladder_fracs=ser["ladder_fracs"],
        n_log_snapshots=ser["n_log_snapshots"])
    mps = res["ms_per_step"]
    print(f"[bench] {res['wall_seconds']:.1f}s wall  =>  {mps:.3f} ms/step  (G={res['G']})")
    for label, T in [("exact NT (122.88M)", sa.BASE_BUDGET), ("1/4 ladder (30.72M)", sa.BASE_BUDGET // 4)]:
        print(f"[bench]   extrapolated {label:22s}: {mps*T/1e3/3600:.2f} h wall / batched job")
    return 0


def write_manifest(out_dir, cfg, stage, specs, args):
    os.makedirs(out_dir, exist_ok=True)
    groups = sa.group_specs(specs)
    man = dict(
        script="run_wca_serial_abf.py", git_commit=_git_commit(), argv=sys.argv,
        config=os.path.abspath(args.config), experiment_name=cfg.get("experiment_name"),
        stage=stage, mode="serial", code_version="wca_serial_abf_v1",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""), device=str(core.DEVICE),
        cells=sorted({s.physics_tag() for s in specs}),
        seeds=sorted({s.seed for s in specs}),
        targets=sorted({int(s.n_steps) for s in specs}),
        n_trajectories=len(specs), n_batched_jobs=len(groups),
        shard=args.shard, num_shards=args.num_shards,
    )
    path = os.path.join(out_dir, f"manifest_{stage}_shard{args.shard}of{args.num_shards}.json")
    with open(path, "w") as fh:
        json.dump(man, fh, indent=2, default=str)
    print(f"[manifest] wrote {path}")


def main(argv=None):
    args = parse_args(argv)
    cfg = sa.load_yaml(args.config)
    raw_dir = os.path.join(cfg["output_root"], "raw")
    ckpt_dir = os.path.join(cfg["output_root"], "checkpoints")
    cache_dir = cfg.get("cache_dir", "cache/phase")
    log_dir = os.path.join(cfg["output_root"], "logs")
    for d in (raw_dir, ckpt_dir, cache_dir, log_dir):
        os.makedirs(d, exist_ok=True)

    base = sa.effective_base(cfg, args.stage)
    ser = sa.serial_settings(cfg)
    if args.checkpoint_every:
        ser["checkpoint_every_steps"] = int(args.checkpoint_every)

    specs = _filter(sa.expand_stage(cfg, args.stage), args)
    if not specs:
        print("[run] no specs after filtering; nothing to do")
        return 0

    if args.dry_run:
        dry_run_summary(cfg, args.stage, specs, raw_dir, args.ms_per_step)
        return 0
    if args.precompute_references:
        precompute_references(specs, base, cache_dir, args.verbose, args.shard, args.num_shards)
        return 0
    if args.benchmark:
        return run_benchmark(cfg, args.stage, specs, base, cache_dir, ser, args)

    write_manifest(cfg["output_root"], cfg, args.stage, specs, args)

    groups = sorted(sa.group_specs(specs).items())
    if args.num_shards > 1:
        groups = [g for i, g in enumerate(groups) if i % args.num_shards == args.shard]

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
    print(f"[run] study={cfg.get('experiment_name')} stage={args.stage} "
          f"shard={args.shard}/{args.num_shards} jobs={len(groups)} device={core.DEVICE} "
          f"CUDA_VISIBLE_DEVICES={cvd} checkpoint_every={ser['checkpoint_every_steps']}")

    engines = {}
    n_done = n_skip = n_fail = 0
    run_kwargs = dict(checkpoint_every=ser["checkpoint_every_steps"],
                      base_budget=ser["base_budget"], ladder_fracs=ser["ladder_fracs"],
                      n_log_snapshots=ser["n_log_snapshots"], verbose=True)
    for (tag, nsteps, st), gspecs in groups:
        gid = sa.group_id(tag, nsteps, st)
        paths = {s.seed: sa.run_npz_path(raw_dir, s) for s in gspecs}
        if not args.overwrite and all(sa.run_is_valid(p) for p in paths.values()):
            n_skip += len(gspecs)
            print(f"  [skip] {gid} ({len(gspecs)} trajectories already valid)")
            continue
        ck_path = sa.checkpoint_path(ckpt_dir, gid)
        try:
            engine = sa.get_engine(gspecs[0], engines)
            ref = sa.get_reference(gspecs[0], base, engine, cache_dir, verbose=args.verbose)
            t0 = time.perf_counter()

            def _emit(partial_outs, _paths=paths, _gs=gspecs):     # partial npz at each checkpoint
                for spec, out in zip(_gs, partial_outs):
                    sa.save_run(_paths[spec.seed], out)

            outs = sa.run_serial_abf_batched(gspecs, base, engine, ref, ck_path,
                                             emit_fn=_emit, **run_kwargs)
            if outs is None:                 # graceful partial stop -> keep checkpoint
                print(f"  [partial] {gid} stopped early; checkpoint kept for resume")
                continue
            for spec, out in zip(gspecs, outs):
                sa.save_run(paths[spec.seed], out)
            n_done += len(outs)
            l2s = ", ".join(f"s{o['seed']}:{o['l2_f']:.4f}" for o in outs)
            print(f"  [done] {gid}  L2(F)=[{l2s}]  ({(time.perf_counter()-t0)/3600:.2f}h this run)")
            if os.path.exists(ck_path):      # completed -> drop the checkpoint
                os.remove(ck_path)
        except Exception as exc:
            n_fail += len(gspecs)
            fpath = sa.save_failure(raw_dir, gid, gspecs, exc)
            print(f"  [FAIL] {gid}: {exc!r} -> {fpath}")
    print(f"[run] DONE trajectories done={n_done} skipped={n_skip} failed={n_fail}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
