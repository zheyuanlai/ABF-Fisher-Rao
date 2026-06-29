#!/usr/bin/env python3
"""Run a WCA *follow-up* study (representative cells / adaptive FR / equal-compute /
frozen-bias) from a YAML config. Additive companion to ``run_wca_phase_diagram.py``.

One ``.npz`` per run under ``<output_root>/raw/`` => interrupted sweeps never lose
completed work; valid existing results are skipped unless ``--overwrite``. One
process per invocation; shard a stage across at most TWO GPUs with --shard /
--num-shards (each shard a disjoint run set). Writes a ``manifest.json`` per stage
recording git commit, command, seeds, params, CUDA device, timestamp, code version.

Examples
--------
# workload summary (no compute)
python scripts/run_wca_followup.py --config configs/wca_representative.yaml --stage representative --dry-run

# whole stage on one visible GPU
CUDA_VISIBLE_DEVICES=0 python scripts/run_wca_followup.py \
    --config configs/wca_representative.yaml --stage representative

# only two cells, two seeds (smoke / validation)
CUDA_VISIBLE_DEVICES=0 python scripts/run_wca_followup.py \
    --config configs/wca_representative.yaml --stage representative \
    --cells b1_h2,b4_h1 --seeds 0,1 --max-runs 12

# split across two GPUs
CUDA_VISIBLE_DEVICES=0 python scripts/run_wca_followup.py ... --shard 0 --num-shards 2
CUDA_VISIBLE_DEVICES=4 python scripts/run_wca_followup.py ... --shard 1 --num-shards 2
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
import wca_followup_jobs as fj  # noqa: E402
import wca_phase_jobs as pj  # noqa: E402

_MS_PER_STEP = {7: 2.0, 10: 2.0, 14: 4.0}


def _ms_per_step(n_dim, n_replicas=1024):
    base = _MS_PER_STEP.get(int(n_dim), _MS_PER_STEP[10] * (n_dim / 10) ** 4 if n_dim > 14 else 2.0)
    return base * (n_replicas / 1024.0)


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
    p.add_argument("--max-gpus", type=int, default=1, choices=(1, 2),
                   help="informational; the launcher sets --num-shards accordingly")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-runs", type=int, default=None)
    p.add_argument("--seeds", default=None, help="comma list to override the stage seeds")
    p.add_argument("--cells", default=None,
                   help="comma list of physics_tag substrings to keep (e.g. b1_h2,b4_h1)")
    p.add_argument("--methods", default=None, help="comma list to restrict methods")
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
        specs = [s for s in specs if (s.name in keep or s.method in keep
                                      or s.frozen_source_method in keep)]
    return specs


def dry_run_summary(cfg, stage, specs, raw_dir, mode):
    n_done = sum(1 for s in specs if fj.run_is_valid(fj.run_npz_path(raw_dir, s)))
    todo = [s for s in specs if not fj.run_is_valid(fj.run_npz_path(raw_dir, s))]
    secs = _runtime_estimate(todo)
    print("=" * 78)
    print(f"DRY RUN  study={cfg.get('experiment_name','?')} stage={stage} mode={mode}")
    print("=" * 78)
    cells = sorted({s.physics_tag() for s in specs})
    methods = sorted({s.name for s in specs})
    seeds = sorted({s.seed for s in specs})
    budgets = sorted({(s.n_replicas, s.n_steps) for s in specs})
    print(f"  cells   ({len(cells)}): {cells}")
    print(f"  methods ({len(methods)}): {methods}")
    print(f"  seeds   ({len(seeds)}): {seeds}")
    print(f"  budgets (N,T): {budgets}")
    print(f"  total runs: {len(specs)}  (done={n_done}, todo={len(todo)})")
    print(f"  TI references (distinct physics): {len(fj.distinct_physics(specs))}")
    print(f"  est. compute for TODO: {secs/3600:.2f} GPU-h "
          f"(~{secs/3600/2:.2f} h wall on 2 GPUs)")
    print(f"  output: {raw_dir}")
    print("=" * 78)


def precompute_references(specs, base, cache_dir, verbose, shard, num_shards):
    engines = {}
    reps = fj.distinct_physics(specs)
    if num_shards > 1:
        reps = [s for i, s in enumerate(reps) if i % num_shards == shard]
    print(f"[refs] shard {shard}/{num_shards}: {len(reps)} TI references -> {cache_dir}")
    for k, s in enumerate(reps):
        eng = fj.get_engine(s, engines)
        path = pj.ti_cache_path(cache_dir, s, fj.build_sim_followup(s, base).n_grid)
        existed = os.path.exists(path)
        fj.get_reference(s, base, eng, cache_dir, verbose=verbose)
        print(f"  [{k+1}/{len(reps)}] {s.physics_tag()} {'(cached)' if existed else '(computed)'}")


def write_manifest(out_dir, cfg, stage, specs, args, mode, source_raw_dir):
    os.makedirs(out_dir, exist_ok=True)
    man = dict(
        script="run_wca_followup.py", git_commit=_git_commit(), argv=sys.argv,
        config=os.path.abspath(args.config), experiment_name=cfg.get("experiment_name"),
        stage=stage, mode=mode, code_version="wca_followup_v1",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        device=str(core.DEVICE),
        cells=sorted({s.physics_tag() for s in specs}),
        methods=sorted({s.name for s in specs}),
        seeds=sorted({s.seed for s in specs}),
        budgets=sorted({(s.n_replicas, s.n_steps) for s in specs}),
        n_runs=len(specs), source_raw_dir=source_raw_dir,
        shard=args.shard, num_shards=args.num_shards,
    )
    path = os.path.join(out_dir, f"manifest_{stage}_shard{args.shard}of{args.num_shards}.json")
    with open(path, "w") as fh:
        json.dump(man, fh, indent=2, default=str)
    print(f"[manifest] wrote {path}")


def main(argv=None):
    args = parse_args(argv)
    cfg = fj.load_yaml(args.config)
    raw_dir = os.path.join(cfg["output_root"], "raw")
    cache_dir = cfg.get("cache_dir", "cache/phase")
    log_dir = os.path.join(cfg["output_root"], "logs")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    st = cfg["stages"][args.stage]
    mode = st.get("mode", cfg.get("mode", "sample"))
    base = fj.effective_base(cfg, args.stage)

    all_specs = sorted(fj.expand_stage(cfg, args.stage), key=lambda s: s.run_id())
    all_specs = _filter(all_specs, args)

    # frozen mode needs the source study's raw dir
    source_raw_dir = ""
    if mode == "frozen":
        src_study = st.get("source_study", cfg.get("source_study", ""))
        source_raw_dir = os.path.join(src_study, "raw") if src_study else ""

    if args.dry_run:
        dry_run_summary(cfg, args.stage, all_specs, raw_dir, mode)
        return 0

    if args.precompute_references:
        precompute_references(all_specs, base, cache_dir, args.verbose,
                              args.shard, args.num_shards)
        return 0

    write_manifest(cfg["output_root"], cfg, args.stage, all_specs, args, mode, source_raw_dir)

    specs = all_specs
    if args.num_shards > 1:
        specs = [s for i, s in enumerate(specs) if i % args.num_shards == args.shard]
    if args.max_runs:
        specs = specs[:args.max_runs]

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
    print(f"[run] study={cfg.get('experiment_name')} stage={args.stage} mode={mode} "
          f"shard={args.shard}/{args.num_shards} jobs={len(specs)} device={core.DEVICE} "
          f"CUDA_VISIBLE_DEVICES={cvd}")
    if mode == "frozen" and not (source_raw_dir and os.path.isdir(source_raw_dir)):
        print(f"[run] FATAL: frozen mode needs source_study raw dir (got {source_raw_dir!r})")
        return 2

    engines = {}
    n_done = n_skip = n_fail = n_nan = 0
    for k, spec in enumerate(specs):
        path = fj.run_npz_path(raw_dir, spec)
        if not args.overwrite and fj.run_is_valid(path):
            n_skip += 1
            continue
        try:
            engine = fj.get_engine(spec, engines)
            if mode == "frozen":
                out = fj.execute_frozen_run(spec, base, engine, cache_dir,
                                            source_raw_dir, verbose=args.verbose)
            else:
                out = fj.execute_sample_run(spec, base, engine, cache_dir, verbose=args.verbose)
            fj.save_run(path, out)
            n_done += 1
            if bool(out["had_nan"]):
                n_nan += 1
            extra = ""
            if mode == "frozen":
                extra = (f"recon_L2(F)={out['frozen_recon_l2_f']:.4f} "
                         f"learned_L2(F)={out['learned_bias_l2_f']:.4f} src={out['frozen_source_method']}")
            else:
                extra = (f"L2(F)={out['l2_f']:.4f} L2(Fp)={out['l2_fp']:.4f} "
                         f"repl={int(out['total_replacement_events'])} "
                         f"essA={out['final_ancestor_ess']:.0f}")
            print(f"  [{k+1}/{len(specs)}] {spec.name} {spec.physics_tag()} "
                  f"seed{spec.seed} N{spec.n_replicas} T{spec.n_steps}: {extra} "
                  f"{'NAN!' if out['had_nan'] else ''} ({out['wall_seconds']:.0f}s)")
        except Exception as exc:
            n_fail += 1
            fpath = fj.save_failure(raw_dir, spec, exc)
            print(f"  [{k+1}/{len(specs)}] {spec.run_id()} FAILED: {exc!r} -> {fpath}")
    print(f"[run] DONE done={n_done} skipped={n_skip} failed={n_fail} nan={n_nan}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
