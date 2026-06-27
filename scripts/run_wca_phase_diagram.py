#!/usr/bin/env python3
"""Run the WCA-dimer *phase diagram* study from a YAML config.

Sweeps physical parameters (inverse temperature beta, barrier height h, well width
w, physical particle count M = n_dim^2, lattice spacing a) on a grid and compares
ABF against marginal-Fisher--Rao ABF (estimated / uniform / oracle targets). One
.npz per run is written under ``<output_root>/raw/`` so an interrupted sweep never
loses completed work; valid existing results are skipped unless ``--overwrite``
(checkpoint/resume). Sharding splits a stage across at most TWO GPUs/processes.

This script never uses more than the GPUs you expose with CUDA_VISIBLE_DEVICES; it
runs ONE process per invocation. Use ``scripts/run_wca_phase_diagram_h200.sh`` to
fan a stage out across one or two GPUs.

Examples
--------
# dry-run workload summary (no compute)
python scripts/run_wca_phase_diagram.py --config configs/wca_phase_diagram_pilot.yaml \
    --stage pilot --dry-run

# whole stage on one visible GPU
CUDA_VISIBLE_DEVICES=4 python scripts/run_wca_phase_diagram.py \
    --config configs/wca_phase_diagram_pilot.yaml --stage pilot

# split across two GPUs (run each line on its own GPU)
CUDA_VISIBLE_DEVICES=4 python scripts/run_wca_phase_diagram.py ... --shard 0 --num-shards 2
CUDA_VISIBLE_DEVICES=7 python scripts/run_wca_phase_diagram.py ... --shard 1 --num-shards 2
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np  # noqa: E402
import wca_abffr_core as core  # noqa: E402
import wca_phase_jobs as pj  # noqa: E402


# per-step ms estimates measured on the H200 (full sampler incl. FR + crossing
# diagnostics), used only for the dry-run time estimate. Keyed by n_dim, linearly
# interpolated otherwise.
_MS_PER_STEP = {7: 2.0, 10: 2.0, 14: 4.0}


def _ms_per_step(n_dim: int) -> float:
    keys = sorted(_MS_PER_STEP)
    if n_dim <= keys[0]:
        return _MS_PER_STEP[keys[0]]
    if n_dim >= keys[-1]:
        # extrapolate ~ n_pairs ~ n_dim^4 beyond the table
        return _MS_PER_STEP[keys[-1]] * (n_dim / keys[-1]) ** 4
    for lo, hi in zip(keys, keys[1:]):
        if lo <= n_dim <= hi:
            t = (n_dim - lo) / (hi - lo)
            return (1 - t) * _MS_PER_STEP[lo] + t * _MS_PER_STEP[hi]
    return _MS_PER_STEP[keys[-1]]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--stage", required=True, help="stage name in the YAML stages block")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--num-gpus", type=int, default=1, choices=(1, 2),
                   help="informational; the launcher uses this to set --num-shards")
    p.add_argument("--device", default="cuda", help="cuda|cpu (informational; uses core.DEVICE)")
    p.add_argument("--batch-size-configs", type=int, default=1,
                   help="compatibility knob for batching independent configs; current H200 benchmark recommends 1 process/config per GPU")
    p.add_argument("--overwrite", action="store_true", help="recompute even valid results")
    p.add_argument("--dry-run", action="store_true", help="print workload summary, do not run")
    p.add_argument("--max-runs", type=int, default=None, help="cap number of jobs (debug/safety)")
    p.add_argument("--precompute-references", action="store_true",
                   help="only compute the TI references for this stage, then exit")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def _runtime_estimate(specs):
    secs = 0.0
    for s in specs:
        secs += _ms_per_step(s.n_dim) * 1e-3 * (s.n_steps + 1)
    return secs


def dry_run_summary(cfg, stage, specs, raw_dir, num_gpus):
    physics = pj._physics_settings(cfg, stage)
    n_done = sum(1 for s in specs if pj.run_is_valid(pj.run_npz_path(raw_dir, s)))
    n_todo = len(specs) - n_done
    todo_specs = [s for s in specs if not pj.run_is_valid(pj.run_npz_path(raw_dir, s))]
    total_secs = _runtime_estimate(todo_specs)
    n_refs = len(pj.distinct_physics(specs))
    print("=" * 78)
    print(f"DRY RUN  stage={stage}  config={cfg.get('experiment_name', '?')}")
    print("=" * 78)
    print(f"  physical parameter settings : {len(physics)}")
    methods = sorted({s.name for s in specs})
    seeds = sorted({s.seed for s in specs})
    print(f"  methods                     : {methods}")
    print(f"  seeds                       : {seeds}  ({len(seeds)} per cell)")
    print(f"  n_steps / n_replicas        : {specs[0].n_steps} / {specs[0].n_replicas}")
    print(f"  total method/seed/cell runs : {len(specs)}  (done={n_done}, todo={n_todo})")
    print(f"  TI references (1 per cell)  : {n_refs}")
    print(f"  output directory            : {raw_dir}")
    print(f"  planned GPUs                : {num_gpus} (NEVER above 2)")
    print("  batch_size_configs          : 1 recommended (independent processes serialize on one H200)" )
    print(f"  est. compute for TODO runs  : {total_secs/3600:.2f} GPU-h "
          f"(~{total_secs/3600/max(num_gpus,1):.2f} h wall on {num_gpus} GPU)")
    print(f"  (+ TI references not yet cached, ~5 min each at full TI settings)")
    print("-" * 78)
    print("  per-cell physics:")
    for ph in physics:
        M = int(ph["n_dim"]) ** 2
        print(f"    beta={ph['beta']:g} h={ph['h']:g} w={ph['w']:g} "
              f"n_dim={int(ph['n_dim'])} (M={M}) a={ph['a']:g}  beta*h={ph['beta']*ph['h']:g}")
    print("=" * 78)


def precompute_references(all_specs, base, cache_dir, verbose, shard=0, num_shards=1):
    """Compute/load the TI references for the DISTINCT physics cells.

    Sharding is by distinct-physics stride (NOT run stride) so two GPUs never race
    on the same cache file -- each computes a disjoint set of references.
    """
    engines = {}
    reps = pj.distinct_physics(all_specs)
    if num_shards > 1:
        reps = [s for i, s in enumerate(reps) if i % num_shards == shard]
    print(f"[refs] shard {shard}/{num_shards}: {len(reps)} TI references -> {cache_dir}")
    for k, s in enumerate(reps):
        eng = pj.get_engine(s, engines)
        path = pj.ti_cache_path(cache_dir, s, pj.build_sim(s, base).n_grid)
        existed = os.path.exists(path)
        pj.get_reference(s, base, eng, cache_dir=cache_dir, verbose=verbose)
        print(f"  [{k+1}/{len(reps)}] {s.physics_tag()} {'(cached)' if existed else '(computed)'}")


def main(argv=None):
    args = parse_args(argv)
    cfg = pj.load_yaml(args.config)
    raw_dir = os.path.join(cfg["output_root"], "raw")
    cache_dir = cfg.get("cache_dir", "cache/phase")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    all_specs = sorted(pj.expand_stage(cfg, args.stage), key=lambda s: s.run_id())
    base = pj.effective_base(cfg, args.stage)

    if args.dry_run:
        dry_run_summary(cfg, args.stage, all_specs, raw_dir, args.num_gpus)
        return 0

    if args.precompute_references:
        # reference sharding is by distinct physics (uses the UNSHARDED spec list)
        precompute_references(all_specs, base, cache_dir, args.verbose,
                              shard=args.shard, num_shards=args.num_shards)
        return 0

    # shard runs by run-id stride so each shard runs a disjoint set
    specs = all_specs
    if args.num_shards > 1:
        specs = [s for i, s in enumerate(specs) if i % args.num_shards == args.shard]
    if args.max_runs:
        specs = specs[:args.max_runs]

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
    print(f"[run] stage={args.stage} shard={args.shard}/{args.num_shards} "
          f"jobs={len(specs)} device={core.DEVICE} CUDA_VISIBLE_DEVICES={cvd}")

    engines: dict = {}
    n_done = n_skip = n_fail = n_nan = 0
    for k, spec in enumerate(specs):
        path = pj.run_npz_path(raw_dir, spec)
        if not args.overwrite and pj.run_is_valid(path):
            n_skip += 1
            continue
        try:
            engine = pj.get_engine(spec, engines)
            out = pj.execute_run(spec, base, engine, cache_dir=cache_dir, verbose=args.verbose)
            pj.save_run(path, out)
            n_done += 1
            if bool(out["had_nan"]):
                n_nan += 1
            print(f"  [{k+1}/{len(specs)}] {spec.name} {spec.physics_tag()} seed{spec.seed}: "
                  f"L2(F)={out['l2_f']:.4f} L2(Fp)={out['l2_fp']:.4f} "
                  f"intF={out['integrated_l2_f']:.3f} cross={out['n_barrier_crossings']} "
                  f"repl={int(out['total_replacement_events'])} "
                  f"essA={out['final_ancestor_ess']:.0f} maxAnc={out['final_max_ancestor_frac']:.3f} "
                  f"{'NAN!' if out['had_nan'] else ''} ({out['wall_seconds']:.0f}s)")
        except Exception as exc:  # keep the sweep alive; record the failure
            n_fail += 1
            fpath = pj.save_failure(raw_dir, spec, exc)
            print(f"  [{k+1}/{len(specs)}] {spec.run_id()} FAILED: {exc!r} -> {fpath}")
    print(f"[run] DONE done={n_done} skipped={n_skip} failed={n_fail} nan={n_nan}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
