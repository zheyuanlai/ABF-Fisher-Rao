#!/usr/bin/env python
"""Z4: ABF-ONLY budget ladder at 300 K on the corrected baseline (h_bias 0.10 A), with the CORRECTED
hidden-gate diagnostic (unwrapped-xi band, gate_reference_v2).  One cell per process:

    CUDA_VISIBLE_DEVICES=1 python -u scripts/zif8_z4_budget_ladder.py --cell B3
cells: B1 64, B2 96, B3 128, B4 192 replicas x 150 ps, 8 seed labels (rng 20260950 of the ZIF-8 prereg screen).
Classification: the frozen ZIF-8 screen classifier (T_cover / T_marg / T_gate, relative rule) imported
from scripts/run_zif8_screen.py, T_gate against cache/zif8/gate_reference_v2_T300.npz.
-> results/ot_repair_campaign/zif8/Z4/{cell}.npz, {cell}.json
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np, torch
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from zif8.core_zif8 import ZIF8SimConfig, ZIF8System, engine_kwargs, run_sampler
from run_zif8_screen import classify
CELLS = {"B1": 64, "B2": 96, "B3": 128, "B4": 192}
OUT = os.path.join(ROOT, "results/ot_repair_campaign/zif8/Z4")
PRE = json.load(open(os.path.join(ROOT, "configs/uniform_campaign/zif8_prereg.json")))
CORR = json.load(open(os.path.join(ROOT, "configs/information_campaign/corrected_baseline_prereg.json")))


def sampler_config(n_replicas, n_steps=300_000, rng_seed=None, **over):
    s = {k: v for k, v in PRE["sampler"].items() if not k.startswith("_")}
    s["abf_bandwidth_A"] = CORR["corrected_baseline"]["h_bias_A"]          # corrected baseline
    s["n_replicas"] = n_replicas; s["n_steps"] = n_steps; s["gate_band_unwrapped"] = True
    s.update(over)
    return ZIF8SimConfig(**s, rng_seed=(rng_seed if rng_seed is not None else PRE["screen"]["rng_seed"]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cell", required=True, choices=sorted(CELLS)); ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    dev = torch.device("cuda"); system = ZIF8System(300.0, dev, root=ROOT, **engine_kwargs(PRE))
    sim = sampler_config(CELLS[a.cell], n_steps=(3000 if a.quick else 300_000))
    if a.quick:
        sim.abf_warmup_steps = sim.estimator_burn_in_steps = sim.fr_start_steps = 600; sim.save_every = 300
    seeds = list(range(PRE["screen"]["n_seeds"]))
    ref2 = np.load(os.path.join(ROOT, "cache/zif8/gate_reference_v2_T300.npz"), allow_pickle=True)
    print(f"Z4 {a.cell}: {len(seeds)} seeds x {sim.n_replicas} replicas x {sim.n_steps * sim.dt:.0f} ps, h_bias {sim.abf_bandwidth_A} A, unwrapped gate band", flush=True)
    t0 = time.time()
    out = run_sampler("abf", system, sim, seeds=seeds, init_pool=os.path.join(ROOT, "cache/zif8/init_pool_T300.npz"), verbose=True, progress_every=25)
    cls = classify(out, sim, PRE, np.asarray(ref2["gate_hist_window_xi"], float))
    T = cls["T"]
    print(f"  {a.cell}: T_cover {cls['T_cover']:.1f} ps ({cls['T_cover'] / T:.2f} T)  T_marg {cls['T_marg']:.1f} ({cls['T_marg'] / T:.2f} T, {cls['marg_status']})  "
          f"T_gate {cls['T_gate']:.1f} ({cls['T_gate'] / T:.2f} T, {cls['gate_status']})  transits {cls['transit_events']}  VERDICT {cls['verdict']}  [{(time.time() - t0) / 60:.0f} min]", flush=True)
    payload = {k: v for k, v in out.items() if isinstance(v, (np.ndarray, np.generic, int, float, str))}
    np.savez_compressed(os.path.join(OUT, f"{a.cell}.npz"), **payload)
    json.dump(dict(cls, cell=a.cell, n_replicas=sim.n_replicas, n_steps=sim.n_steps, dt=sim.dt, h_bias=sim.abf_bandwidth_A, seeds=seeds, rng_seed=sim.rng_seed,
                   gate_reference="cache/zif8/gate_reference_v2_T300.npz", wall_min=(time.time() - t0) / 60), open(os.path.join(OUT, f"{a.cell}.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
