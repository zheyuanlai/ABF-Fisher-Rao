"""Run one batch (R seeds of one method at one N) of the finite-loading engine and write its npz.

Skips the batch when the output already exists with the same signature (a resubmitted task resumes).
  python scripts/lta_loading/run_batch.py --method abf --N 256 --seeds 31000-31015 --n-steps 400000 \
      --out results/lta_loading/ladder/N256/abf.npz [--table ref/table.npy] [--set key=value ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from lta_loading import engine as LE  # noqa: E402


def parse_seeds(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def parse_set(items):
    cfg = {}
    for it in items or []:
        k, v = it.split("=", 1)
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        cfg[k] = v
    return cfg


def signature(args, cfg, table_sha):
    d = dict(method=args.method, N=args.N, seeds=parse_seeds(args.seeds), n_steps=args.n_steps, cfg=cfg,
             table_sha=table_sha, engine_sha=LE.source_sha256(), save_steps=args.save_steps)
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["abf", "fr", "frozen"])
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--n-steps", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--table", default=None, help=".npy bin forces for --method frozen")
    ap.add_argument("--save-steps", default=None, help="comma list; default: the budget save grid")
    ap.add_argument("--set", nargs="*", help="cfg overrides key=value (JSON values)")
    args = ap.parse_args()
    cfg = LE.resolve_cfg(**parse_set(args.set))
    table = None
    table_sha = None
    if args.table:
        table = np.load(args.table)
        table_sha = hashlib.sha256(table.tobytes()).hexdigest()
    sig = signature(args, cfg, table_sha)
    out_path = os.path.join(ROOT, args.out) if not os.path.isabs(args.out) else args.out
    if os.path.exists(out_path):
        _, meta = LE.load_npz(out_path)
        if meta.get("signature") == sig:
            print(f"SKIP {args.out} (complete, same signature)")
            return
        raise SystemExit(f"{args.out} exists with a different signature; move it away first")
    save_steps = None
    if args.save_steps:
        save_steps = [int(x) for x in args.save_steps.split(",")]
    seeds = parse_seeds(args.seeds)
    t0 = time.perf_counter()
    b = LE.Batch(seeds, args.method, args.N, args.n_steps, cfg=cfg, table=table, save_steps=save_steps)
    t_init = time.perf_counter() - t0
    b.preequilibrate()
    label = f"{args.method} N{args.N} R{len(seeds)}"
    print(f"{label}: init {t_init:.1f}s preeq {b.timers['preeq_s']:.1f}s, running {args.n_steps} steps", flush=True)
    out, meta = b.run(label=label)
    meta.update(signature=sig, args=vars(args), init_s=t_init, total_s=time.perf_counter() - t0,
                host=os.uname().nodename, pbs_jobid=os.environ.get("PBS_JOBID"))
    LE.save_npz(out_path, out, meta)
    print(f"DONE {args.out} run {meta['timers']['run_s']:.1f}s total {meta['total_s']:.1f}s "
          f"repl {int(out['sv_i'][-1, :, 0].sum())} cross/walker {out['sv_i'][-1, :, 1].mean() / args.N:.2f} "
          f"mc_acc {out['sv_i'][-1, :, 2].sum() / max(1, out['sv_i'][-1, :, 3].sum()):.3f}", flush=True)


if __name__ == "__main__":
    main()
