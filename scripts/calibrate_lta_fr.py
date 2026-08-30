#!/usr/bin/env python
"""SAFETY-ONLY FR-rate calibration for ethane/LTA (uniform target).

Runs a short fr_uniform ladder and selects the LARGEST rate satisfying the
frozen genealogy/event floors (configs/uniform_campaign/lta_prereg.json).
No error metric is computed, read, or stored here -- by construction this file
never loads the reference.

    CUDA_VISIBLE_DEVICES=3 python -u scripts/calibrate_lta_fr.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from lta.core_lta import LTAParams, LTASimConfig, LTASystem, run_sampler  # noqa: E402

PREREG = os.path.join(ROOT, "configs/uniform_campaign/lta_prereg.json")
OUT = os.path.join(ROOT, "results/uniform_campaign/lta/calibration")

LADDER = (0.02, 0.05, 0.10, 0.20)
CAL_STEPS = 120_000       # calibration horizon: FR active for 80k steps
CAL_SEEDS = [900, 901]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--prereg", default=PREREG,
                    help="v1 prereg (default) or the temperature-sweep prereg")
    ap.add_argument("--temperature", type=float, default=None,
                    help="sweep mode: calibrate this T using the sweep prereg's sampler")
    a = ap.parse_args()
    pre = json.load(open(a.prereg))
    rule = pre["success_rule"]
    temp = (a.temperature if a.temperature is not None
            else pre["system"]["temperature_K"])
    tag = f"_T{temp:g}" if a.temperature is not None else ""
    os.makedirs(OUT, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = LTAParams(temperature=temp)
    system = LTASystem(params, device, root=ROOT)

    rows = []
    for rate in LADDER:
        s = dict(pre["sampler"])
        sim = LTASimConfig(n_steps=CAL_STEPS, n_replicas=s["n_replicas"],
                           save_every=s["save_every"], n_grid=s["n_grid"],
                           abf_bandwidth=s["abf_bandwidth"],
                           kde_bandwidth=s["kde_bandwidth"],
                           abf_warmup_steps=s["abf_warmup_steps"],
                           abf_force_clip=s["abf_force_clip"],
                           estimator_burn_in_steps=s["estimator_burn_in_steps"],
                           fr_start_steps=s["fr_start_steps"],
                           fr_every=s["fr_every"], score_clip=s["score_clip"],
                           max_event_fraction=s["max_event_fraction"],
                           target_ema_rate=s["target_ema_rate"],
                           fr_rate=rate, rng_seed=20260830 + int(temp))
        out = run_sampler("fr_uniform", system, sim, seeds=CAL_SEEDS, verbose=False)
        ess = np.asarray(out["ancestor_ess"], dtype=float)      # (T, R)
        wmax = np.asarray(out["max_ancestor_frac"], dtype=float)
        active = np.asarray(out["steps"]) >= sim.fr_start_steps
        N = sim.n_replicas
        ess_min = float(np.nanmin(ess[active]) / N)
        wmax_max = float(np.nanmax(wmax[active]))
        n_opps = (CAL_STEPS - sim.fr_start_steps) / sim.fr_every
        ev_cum = float(out["total_replacement_events"].sum()
                       / (len(CAL_SEEDS) * N))
        # The frozen gates are the two unambiguous genealogy floors; per-event
        # turnover is already capped structurally by max_event_fraction=0.02.
        # events/replica is reported, not gated.
        okay = (ess_min >= rule["ess_anc_over_N_min"]
                and wmax_max <= rule["wmax_max"])
        rows.append(dict(rate=rate, ess_min=ess_min, wmax_max=wmax_max,
                         events_per_replica=ev_cum, ok=bool(okay)))
        print(f"  rate {rate:>5.2f}: min ESS/N {ess_min:.3f}  wmax {wmax_max:.4f}  "
              f"events/replica {ev_cum:.3f}  ok={okay}", flush=True)

    safe = [r for r in rows if r["ok"]]
    sel = max(safe, key=lambda r: r["rate"]) if safe else None
    result = dict(ladder=rows, selected=(sel["rate"] if sel else None),
                  temperature_K=temp, fr_start_steps=int(s["fr_start_steps"]),
                  rule=dict(ess_min=0.30, wmax_max=0.05),
                  note=("Selected on genealogy safety only; no error metric was "
                        "computed or read. Among safe rates the LARGEST is chosen "
                        "so the mechanism is exercised."))
    out_name = f"fr_rate_selection{tag}.json"
    with open(os.path.join(OUT, out_name), "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"selected fr_rate = {result['selected']} (T={temp:g} K)")
    print(f"wrote {OUT}/{out_name}")


if __name__ == "__main__":
    main()
