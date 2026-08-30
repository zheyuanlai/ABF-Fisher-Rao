#!/usr/bin/env python
"""SAFETY-ONLY FR-rate calibration for the ethane/ZIF-8 cell (uniform target).
No error metric is computed or read; the reference is never loaded.

Corrects a defect the olefin/CHA stage exposed: THAT calibration ran 150k
steps and selected rate 0.10, but production ran 400k steps and its ancestor
ESS/N landed at 0.221 -- below the 0.30 floor the calibration had certified.
Ancestor collapse accumulates with the number of birth-death events, i.e. with
the HORIZON, so a short calibration systematically over-certifies.  Here the
ladder runs the FULL production horizon (with fewer seeds/replicas, which the
genealogy statistics tolerate because they self-average over the population).

    CUDA_VISIBLE_DEVICES=3 python -u scripts/calibrate_zif8_fr.py --temperature 300
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from zif8.core_zif8 import (ZIF8SimConfig, ZIF8System, engine_kwargs,  # noqa: E402
                            run_sampler)

PREREG = os.path.join(ROOT, "configs/uniform_campaign/zif8_prereg.json")
OUT = os.path.join(ROOT, "results/uniform_campaign/zif8/calibration")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--temperature", type=float, default=300.0)
    ap.add_argument("--chunk", type=int, default=None)
    ap.add_argument("--n-replicas", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=None)
    a = ap.parse_args()
    pre = json.load(open(PREREG))
    rule = pre["success_rule"]
    cal = pre["fr_rate_rule"]
    tag = f"T{a.temperature:g}"
    os.makedirs(OUT, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ek = engine_kwargs(pre)
    if a.chunk:
        ek["chunk"] = a.chunk
    system = ZIF8System(a.temperature, device, root=ROOT, **ek)
    print(f"  engine: dtype {ek['dtype']}, force kernel "
          f"{ek['force_dtype'] or ek['dtype']}, chunk {ek['chunk']}")
    s = {k: v for k, v in pre["sampler"].items() if not k.startswith("_")}
    pool = os.path.join(ROOT, f"cache/zif8/init_pool_{tag}.npz")
    seeds = list(range(cal["seed_first"], cal["seed_first"]
                       + (a.seeds or cal["n_seeds"])))
    n_rep = a.n_replicas or cal["n_replicas"]

    print(f"ZIF-8 FR safety calibration {tag}: ladder {cal['ladder']}, "
          f"{len(seeds)} seeds x {n_rep} replicas at the FULL production "
          f"horizon ({s['n_steps']} steps)", flush=True)
    rows = []
    for rate in cal["ladder"]:
        t0 = time.time()
        sim = ZIF8SimConfig(**s, rng_seed=cal["rng_seed"], fr_rate=float(rate))
        sim.n_replicas = n_rep
        out = run_sampler("fr_uniform", system, sim, seeds=seeds,
                          init_pool=pool, verbose=False)
        ess = np.asarray(out["ancestor_ess"], float)
        wmax = np.asarray(out["max_ancestor_frac"], float)
        active = np.asarray(out["steps"]) >= sim.fr_start_steps
        N = sim.n_replicas
        ess_min = float(np.nanmin(ess[active]) / N)
        wmax_max = float(np.nanmax(wmax[active]))
        ev = float(out["total_replacement_events"].sum() / (len(seeds) * N))
        n_fr_ops = (sim.n_steps - sim.fr_start_steps) / max(sim.fr_every, 1)
        ev_frac = ev / max(n_fr_ops, 1)
        ok = (ess_min >= rule["ess_anc_over_N_min"]
              and wmax_max <= rule["wmax_max"]
              and ev_frac <= s["max_event_fraction"])
        rows.append(dict(rate=rate, ess_min=ess_min, wmax_max=wmax_max,
                         events_per_replica=ev, event_fraction=ev_frac,
                         ok=bool(ok), minutes=(time.time() - t0) / 60))
        print(f"  rate {rate:>6.3f}: min ESS/N {ess_min:.3f}  wmax {wmax_max:.4f}  "
              f"events/replica {ev:7.2f}  event frac {ev_frac:.4f}  ok={ok}  "
              f"[{rows[-1]['minutes']:.1f} min]", flush=True)

    safe = [r for r in rows if r["ok"]]
    sel = max(safe, key=lambda r: r["rate"]) if safe else None
    result = dict(ladder=rows, selected=(sel["rate"] if sel else None), cell=tag,
                  fr_start_steps=int(s["fr_start_steps"]),
                  n_steps=int(s["n_steps"]), n_replicas=n_rep, seeds=seeds,
                  note=("Safety-only; largest safe rate; no error metric read. "
                        "Run at the FULL production horizon -- see the module "
                        "docstring for why a short ladder over-certifies."))
    with open(os.path.join(OUT, f"fr_rate_selection_{tag}.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"selected fr_rate = {result['selected']} ({tag})")
    if result["selected"] is None:
        print("  NO SAFE RATE: the two-arm run is refused by the safety rule.")


if __name__ == "__main__":
    main()
