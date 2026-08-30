#!/usr/bin/env python
"""SAFETY-ONLY FR-rate calibration for one olefin/CHA cell (uniform target).
No error metric is computed or read; the reference is never loaded.

    CUDA_VISIBLE_DEVICES=3 python -u scripts/calibrate_cha_fr.py --guest ethene --temperature 450
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from cha.core_cha import CHASimConfig, CHASystem, run_sampler  # noqa: E402

PREREG = os.path.join(ROOT, "configs/uniform_campaign/cha_prereg.json")
OUT = os.path.join(ROOT, "results/uniform_campaign/cha/calibration")
LADDER = (0.02, 0.05, 0.10, 0.20)
CAL_STEPS = 150_000
CAL_SEEDS = [900, 901]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--guest", required=True)
    ap.add_argument("--temperature", type=float, required=True)
    a = ap.parse_args()
    pre = json.load(open(PREREG))
    rule = pre["success_rule"]
    tag = f"{a.guest}_{a.temperature:g}"
    os.makedirs(OUT, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    system = CHASystem(a.guest, a.temperature, device, root=ROOT)
    s = {k: v for k, v in pre["sampler"].items() if not k.startswith("_")}

    rows = []
    for rate in LADDER:
        sim = CHASimConfig(**s, rng_seed=20260930 + int(a.temperature))
        sim.n_steps = CAL_STEPS
        sim.fr_rate = rate
        out = run_sampler("fr_uniform", system, sim, seeds=CAL_SEEDS, verbose=False)
        ess = np.asarray(out["ancestor_ess"], dtype=float)
        wmax = np.asarray(out["max_ancestor_frac"], dtype=float)
        active = np.asarray(out["steps"]) >= sim.fr_start_steps
        N = sim.n_replicas
        ess_min = float(np.nanmin(ess[active]) / N)
        wmax_max = float(np.nanmax(wmax[active]))
        ev = float(out["total_replacement_events"].sum() / (len(CAL_SEEDS) * N))
        okay = (ess_min >= rule["ess_anc_over_N_min"]
                and wmax_max <= rule["wmax_max"])
        rows.append(dict(rate=rate, ess_min=ess_min, wmax_max=wmax_max,
                         events_per_replica=ev, ok=bool(okay)))
        print(f"  rate {rate:>5.2f}: min ESS/N {ess_min:.3f}  wmax {wmax_max:.4f}  "
              f"events/replica {ev:.3f}  ok={okay}", flush=True)

    safe = [r for r in rows if r["ok"]]
    sel = max(safe, key=lambda r: r["rate"]) if safe else None
    result = dict(ladder=rows, selected=(sel["rate"] if sel else None), cell=tag,
                  fr_start_steps=int(s["fr_start_steps"]),
                  note="Safety-only; largest safe rate; no error metric read.")
    with open(os.path.join(OUT, f"fr_rate_selection_{tag}.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"selected fr_rate = {result['selected']} ({tag})")


if __name__ == "__main__":
    main()
