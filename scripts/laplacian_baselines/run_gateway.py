#!/usr/bin/env python3
"""Only the published LRS Laplacian baseline on the existing positive gateway cells.

Example (after validating the selection rule with the tests):
  NUMBA_NUM_THREADS=1 python scripts/laplacian_baselines/run_gateway.py --dry-run
  NUMBA_NUM_THREADS=1 python scripts/laplacian_baselines/run_gateway.py --only-N 2048 --seeds 8100 --workers 1
  NUMBA_NUM_THREADS=1 python scripts/laplacian_baselines/run_gateway.py --workers 16

The legacy equal-budget gateway ABF/FR data and their frozen configs are never overwritten.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


def _job(task):
    import numpy as np
    import gateway_ladder_numba as E

    cfg, N, seed, B, checkpoints, root_out = task
    n_steps = int(B) // int(N)
    if n_steps * N != B:
        raise ValueError(f"budget {B} is not divisible by {N}")
    path = ROOT / root_out / f"N{N}" / f"s{seed}_laplacian.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    if E.is_complete(str(path)):
        return dict(N=N, seed=seed, status="skipped", path=str(path.relative_to(ROOT)))
    # Use exactly the same save grid as scripts/equal_budget/run_ladder.py.
    steps, _ = E.budget_save_grid(n_steps)
    h = float(cfg["h"])
    extra = [int(round(t / h)) for t in checkpoints if 0 < round(t / h) <= n_steps]
    save = np.unique(np.concatenate([steps, np.asarray(extra, dtype=np.int64)]))
    start = time.monotonic()
    res = E.run_job(cfg, "laplacian", N, seed, n_steps, str(path), save_steps=save)
    return dict(N=N, seed=seed, status="complete", path=str(path.relative_to(ROOT)),
                n_force_evals=int(res["n_force_evals"]), wall_s=time.monotonic()-start)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/laplacian_baselines/gateway.json")
    ap.add_argument("--only-N", nargs="*", type=int, default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    with (ROOT / a.config).open() as f:
        p = json.load(f)
    ns = list(p["N_positive"]) if a.only_N is None else list(a.only_N)
    seeds = list(p["seeds"]) if a.seeds is None else list(a.seeds)
    if not set(ns).issubset(p["N_positive"]):
        raise ValueError(f"only established positive gateway cells {p['N_positive']} are permitted")
    if not set(seeds).issubset(p["seeds"]):
        raise ValueError("choose seeds from the frozen equal-budget gateway positive study")
    if a.workers < 1:
        raise ValueError("workers must be >= 1")
    tasks = [(p["engine_cfg"], N, seed, int(p["B"]), list(p["physical_checkpoints_t"]), p["out_dir"])
             for N in ns for seed in seeds]
    print(f"Gateway Laplacian only: {len(tasks)} runs; N={ns}, seeds={len(seeds)}", flush=True)
    if a.dry_run:
        for N in ns:
            print(f"  N={N}: steps/replica={int(p['B'])//N:,}, physical T={int(p['B'])/N*p['engine_cfg']['h']:.3f}")
        return
    log = ROOT / "results/laplacian_baselines/gateway_runs.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=a.workers) as pool, log.open("a") as f:
        futures = [pool.submit(_job, task) for task in tasks]
        for i, future in enumerate(as_completed(futures), start=1):
            rec = future.result()
            f.write(json.dumps(rec, sort_keys=True) + "\n")
            f.flush()
            print(f"{i}/{len(tasks)}: {rec['status']} N={rec['N']} seed={rec['seed']}", flush=True)


if __name__ == "__main__":
    main()
