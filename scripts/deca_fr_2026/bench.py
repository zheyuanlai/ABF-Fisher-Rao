"""A100 throughput of the deca-alanine sampler per batch size (cost planning only).

    python scripts/deca_fr_2026/bench.py --sizes 128 512 1024 2048 4096 --steps 300
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C                                                          # noqa: E402
from deca import system as dsys                                             # noqa: E402
from deca.core import run_sampler_deca                                      # noqa: E402
from deca.umbrella import relax_pool                                        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[128, 512, 1024, 2048, 4096])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--out", default="results/deca_fr_2026/validation/bench.json")
    args = ap.parse_args()
    dev = "cuda"
    engine = C.engine_for(dev, compiled=True)
    rows = []
    x = dsys.build_helix(-57.0, -47.0)
    for B in args.sizes:
        rng = np.random.default_rng(0)
        X0 = np.stack([x + 0.004 * rng.standard_normal(x.shape) for _ in range(B)])
        q = torch.as_tensor(X0, device=dev)
        q, ok, _ = relax_pool(engine, q)
        init = q.reshape(1, B, -1, 3).cpu().numpy()
        for est in ("kernel", "histogram"):
            sim = C.make_sim(dict(n_walkers=B, n_steps=args.steps, abf_estimator=est,
                                  save_every=10 ** 9, xi_trace_every=10 ** 9, label_every=10 ** 9,
                                  abf_warmup_steps=1))
            # warm-up (compilation for this shape), then the timed run
            run_sampler_deca("abf", engine, sim, [0], init, device=dev, verbose=False)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            run_sampler_deca("abf", engine, sim, [0], init, device=dev, verbose=False)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / args.steps
            rows.append(dict(B=B, estimator=est, ms_per_step=1e3 * dt,
                             us_per_state_step=1e6 * dt / B,
                             ns_per_day_aggregate=B * 1e-6 * 86400 / dt))
            print(f"  B={B:5d} {est:9s}: {1e3*dt:7.2f} ms/step  {1e6*dt/B:6.3f} us/state-step  "
                  f"{rows[-1]['ns_per_day_aggregate']:9.0f} ns/day", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(dict(device=torch.cuda.get_device_name(0), torch=torch.__version__,
                       rows=rows), fh, indent=1)


if __name__ == "__main__":
    main()
