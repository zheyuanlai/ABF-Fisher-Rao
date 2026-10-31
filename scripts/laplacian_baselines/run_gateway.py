#!/usr/bin/env python3
"""Entropic-gateway arms of the LRS Laplacian baseline (configs/laplacian_baselines/gateway.json).

  NUMBA_NUM_THREADS=1 python scripts/laplacian_baselines/run_gateway.py --dry-run
  NUMBA_NUM_THREADS=1 python scripts/laplacian_baselines/run_gateway.py --only-N 512 --seeds 8100 --workers 1
  NUMBA_NUM_THREADS=1 python scripts/laplacian_baselines/run_gateway.py --workers 16            # Laplacian, frozen scale
  NUMBA_NUM_THREADS=1 python scripts/laplacian_baselines/run_gateway.py --methods abf fr        # NSCC re-run of the arms

Every run uses the frozen equal-budget gateway engine_cfg, budget B, seeds and save grid (budget grid + physical
checkpoints, exactly scripts/equal_budget/run_ladder.py's) of configs/equal_budget_v2/gateway_production.json.
The Laplacian scale is laplacian_scale_by_N[N] of gateway.json when present (frozen by the pilot), else
engine_cfg.laplacian_scale; --scale overrides it and tags the file name (pilots).  scripts/laplacian_baselines/
run_all.py calls gateway_job() for the same work inside a mixed job pool.  The legacy equal-budget gateway data and
their frozen configs are never written to.
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
CONFIG = "configs/laplacian_baselines/gateway.json"


def load_config(path=CONFIG):
    with (ROOT / path).open() as f:
        return json.load(f)


def save_grid(E, cfg, n_steps, checkpoints):
    """Budget grid (200 uniform + 24 log-spaced u) union the frozen physical-time checkpoints <= T_N (run_ladder.py)."""
    import numpy as np
    steps, _ = E.budget_save_grid(n_steps)
    h = float(cfg["h"])
    extra = [int(round(t / h)) for t in checkpoints if 0 < round(t / h) <= n_steps]
    return np.unique(np.concatenate([steps, np.asarray(extra, dtype=np.int64)]))


def gateway_job(j, config=None):
    """One gateway arm: j = {"N", "seed", "method", "scale" (optional), "out" (repo-relative .npz)}.
    Returns (out, wall_s, "skip" | "done"); resumes from out + '.ckpt.npz'; never overwrites a complete result."""
    import gateway_ladder_numba as E
    p = config or load_config(j.get("config", CONFIG))
    cfg = dict(p["engine_cfg"])
    N, seed, method = int(j["N"]), int(j["seed"]), str(j["method"])
    if method == "laplacian":
        sc = j.get("scale")
        if sc is None:
            sc = p.get("laplacian_scale_by_N", {}).get(str(N), cfg["laplacian_scale"])
        cfg["laplacian_scale"] = float(sc)
    B = int(p["B"])
    n_steps = B // N
    if n_steps * N != B:
        raise ValueError(f"budget {B} is not divisible by {N}")
    path = ROOT / j["out"]
    path.parent.mkdir(parents=True, exist_ok=True)
    if E.is_complete(str(path)):
        return j["out"], 0.0, "skip"
    save = save_grid(E, cfg, n_steps, p["physical_checkpoints_t"])
    t0 = time.monotonic()
    E.run_job(cfg, method, N, seed, n_steps, str(path), save_steps=save)
    return j["out"], time.monotonic() - t0, "done"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--only-N", nargs="*", type=int, default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--methods", nargs="*", default=["laplacian"], choices=["abf", "fr", "laplacian"])
    ap.add_argument("--scale", type=float, default=None, help="Laplacian scale override; tags the file name (pilots)")
    ap.add_argument("--out-dir", default=None, help="repo-relative output root (default: config out_dir)")
    ap.add_argument("--any-seeds", action="store_true", help="allow seeds outside the frozen production list (pilots)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    p = load_config(a.config)
    ns = list(p["N_positive"]) if a.only_N is None else list(a.only_N)
    seeds = list(p["seeds"]) if a.seeds is None else list(a.seeds)
    if not set(ns).issubset(p["N_positive"]):
        raise ValueError(f"only established positive gateway cells {p['N_positive']} are permitted")
    if not a.any_seeds and not set(seeds).issubset(p["seeds"]):
        raise ValueError("choose seeds from the frozen equal-budget gateway positive study (or pass --any-seeds for pilots)")
    if a.workers < 1:
        raise ValueError("workers must be >= 1")
    out_dir = a.out_dir or p["out_dir"]
    tag = f"_sc{a.scale:g}" if a.scale is not None else ""
    jobs = [dict(N=N, seed=s, method=m, config=a.config, **({"scale": a.scale} if a.scale is not None else {}),
                 out=f"{out_dir}/N{N}/s{s}_{m}{tag}.npz") for N in ns for s in seeds for m in a.methods]
    print(f"Gateway arms {a.methods}: {len(jobs)} runs; N={ns}, seeds={len(seeds)}, out {out_dir}", flush=True)
    if a.dry_run:
        for N in ns:
            print(f"  N={N}: steps/replica={int(p['B'])//N:,}, physical T={int(p['B'])/N*p['engine_cfg']['h']:.3f}, "
                  f"laplacian_scale={p.get('laplacian_scale_by_N', {}).get(str(N), p['engine_cfg']['laplacian_scale']) if a.scale is None else a.scale}")
        return
    log = ROOT / out_dir / "runs.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=a.workers) as pool, log.open("a") as f:
        futures = [pool.submit(gateway_job, j) for j in jobs]
        for i, future in enumerate(as_completed(futures), start=1):
            out, wall, status = future.result()
            f.write(json.dumps(dict(out=out, wall_s=wall, status=status), sort_keys=True) + "\n")
            f.flush()
            print(f"{i}/{len(jobs)}: {status} {out} {wall:.0f}s", flush=True)


if __name__ == "__main__":
    main()
