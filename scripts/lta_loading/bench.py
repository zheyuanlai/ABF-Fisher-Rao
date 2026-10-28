"""c_ss(N, method): steady-state seconds per step of single-seed runs on one A100 (PREREGISTRATION §8, Q3).

R = 1, seeds 30900-30902 (3 repeats), 40 000 steps: c_ss = wall(40 000) - wall(20 000) over 20 000 steps, the FR-active
window (FR starts at step 20 000).  Also records init and pre-equilibration seconds.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from lta_loading import engine as LE  # noqa: E402

LADDER = (64, 128, 256, 512, 1024)
SEEDS = (30900, 30901, 30902)


def main():
    out_dir = os.path.join(ROOT, "results", "lta_loading", "bench")
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    cfg = LE.resolve_cfg()
    for rep, seed in enumerate(SEEDS):
        for N in LADDER:
            for method in ("abf", "fr"):
                t0 = time.perf_counter()
                b = LE.Batch([seed], method, N, 40_000, cfg=cfg, save_steps=[0, 20_000, 30_000, 40_000],
                             verbose=False)
                t_init = time.perf_counter() - t0
                b.preequilibrate()
                out, meta = b.run()
                w = out["sv_wall"]
                row = dict(N=N, method=method, seed=seed, rep=rep, c_ss=(w[3] - w[1]) / 20_000.0,
                           c_pre=(w[1] - w[0]) / 20_000.0, init_s=t_init, preeq_s=b.timers["preeq_s"],
                           repl=int(out["sv_i"][-1, 0, 0]))
                rows.append(row)
                print(json.dumps(row), flush=True)
    summ = {}
    for N in LADDER:
        for m in ("abf", "fr"):
            v = [r["c_ss"] for r in rows if r["N"] == N and r["method"] == m]
            summ[f"{m}_N{N}"] = dict(median_c_ss=float(np.median(v)), all=v)
    with open(os.path.join(out_dir, "bench.json"), "w") as fh:
        json.dump(dict(rows=rows, summary=summ, engine_sha256=LE.source_sha256()), fh, indent=1)
    for k, v in summ.items():
        print(f"{k:10s} c_ss = {1e6 * v['median_c_ss']:.1f} us/step")


if __name__ == "__main__":
    main()
