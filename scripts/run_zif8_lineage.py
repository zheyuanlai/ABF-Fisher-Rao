#!/usr/bin/env python
"""Instrumented ABF and fr_uniform arms for the lineage mechanism experiment."""
from __future__ import annotations
import json, os, sys
import numpy as np, torch
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from zif8.core_zif8 import ZIF8SimConfig, ZIF8System, engine_kwargs, run_sampler  # noqa

OUT = os.path.join(ROOT, "results/information_campaign/lineage")
pre = json.load(open(os.path.join(ROOT, "configs/uniform_campaign/zif8_prereg.json")))
s = {k: v for k, v in pre["sampler"].items() if not k.startswith("_")}
rate = json.load(open(os.path.join(ROOT, "results/uniform_campaign/zif8/calibration/"
                                         "fr_rate_selection_T300.json")))["selected"]
os.makedirs(OUT, exist_ok=True)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
system = ZIF8System(300.0, dev, root=ROOT, **engine_kwargs(pre))
seeds = list(range(8))
for method in ("abf", "fr_uniform"):
    p = os.path.join(OUT, f"{method}.npz")
    if os.path.exists(p):
        print(f"skip {method}"); continue
    sim = ZIF8SimConfig(**s, rng_seed=20260990, fr_rate=float(rate))
    sim.n_steps = 300000                      # 150 ps, mechanism not re-measurement
    out = run_sampler(method, system, sim, seeds=seeds, init_pool=os.path.join(
        ROOT, "cache/zif8/init_pool_T300.npz"), verbose=True, progress_every=10,
        lineage_diagnostics=True)
    np.savez_compressed(p, **{k: v for k, v in out.items()
                              if isinstance(v, (np.ndarray, np.generic, int, float, str))})
    print(f"wrote {p}", flush=True)
