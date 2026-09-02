#!/usr/bin/env python
"""Transport-refresh campaign, Stage A2: EXACT dose matching (blind refinement ladder).

Preregistration: configs/transport_campaign/gateway_transport_refresh_prereg.json.
Same 16 calibration rows and the same batch_seed as Stage A (identical Langevin noise, so
fr_uniform's J_KL reproduces); ladder {0.0030, 0.00325, 0.0035, 0.00375, 0.0040};
alpha** = argmin |log(J_KL(ot)/J_KL(fr))|.  BLIND: no error metric computed, printed or stored.

    CUDA_VISIBLE_DEVICES=3 python -u scripts/run_gateway_transport_refresh_calibration.py   # GPU 3 ONLY
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_gateway_horizontal_calibration import (  # noqa: E402
    dose, run_ladder, git_rev, TAKEN, BATCH_SEED, KEEP, SCALARS, BASE_PREREG, CORR_PREREG, STEP1)

PREREG = os.path.join(ROOT, "configs/transport_campaign/gateway_transport_refresh_prereg.json")
OUT_DIR = os.path.join(ROOT, "results", "transport_campaign", "gateway_horizontal", "calibration_refine")


def select_alpha_exact(ladder, ratio):
    a = min(ladder, key=lambda al: abs(np.log(ratio[al])))
    return a, "argmin |log ratio| over the refinement ladder"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    pre = json.load(open(PREREG))
    base = json.load(open(BASE_PREREG)); corr = json.load(open(CORR_PREREG)); step1 = json.load(open(STEP1))
    sampler, cell = base["sampler"], base["cell"]
    h_bias = float(corr["corrected_baseline"]["h_bias"])
    assert abs(h_bias - float(step1["h_bias_corrected"])) < 1e-12
    gamma = float(corr["rate"]["gamma"])
    st = pre["stage_A2_refinement"]
    seeds = list(range(st["seeds"]["first"], st["seeds"]["first"] + st["seeds"]["count"]))
    assert seeds == list(range(480, 488)), "Stage A2 must reuse Stage A's calibration labels"
    assert not (set(seeds) & TAKEN)
    ladder = [float(x) for x in st["ladder"]]
    rows = [(init, sd) for init in ["left", "one_right"] for sd in seeds]
    if torch.cuda.is_available():
        assert torch.cuda.device_count() == 1, "pin exactly one GPU"
    os.makedirs(a.out, exist_ok=True)
    print("Transport-refresh campaign, Stage A2: exact dose refinement (BLIND to every error metric)")
    print(f"  rows {len(rows)} (seeds {seeds[0]}-{seeds[-1]}); gamma {gamma:g}; ladder {ladder}; batch_seed {BATCH_SEED}")
    if a.dry_run:
        return
    t0 = time.time()
    recs, arms = run_ladder(rows, ladder, sampler, cell, h_bias, gamma)
    J = {m.name: np.array([dose(r) for r in recs if r["method"] == m.name]) for m in arms}
    med = {k: float(np.median(v)) for k, v in J.items()}
    ratio = {al: med[f"ot_{al:g}"] / med["fr_uniform"] for al in ladder}
    a_star, rule = select_alpha_exact(ladder, ratio)
    within = abs(ratio[a_star] - 1.0) <= 0.05
    print(f"  {len(recs)} rows in {time.time() - t0:.0f}s; median J_KL: abf {med['abf']:.4f}, fr_uniform {med['fr_uniform']:.4f}; "
          + ", ".join(f"ot_{al:g} {med[f'ot_{al:g}']:.4f} (x{ratio[al]:.3f})" for al in ladder), flush=True)
    print(f"  ALPHA SELECTED: alpha** = {a_star:g}  [{rule}]  ratio {ratio[a_star]:.3f}  within +-5%: {within}", flush=True)
    t_axis = np.asarray(recs[0]["t"], float)
    desc = {}
    for m in arms:
        rr = [r for r in recs if r["method"] == m.name]
        dc = np.array([np.trapezoid(np.nan_to_num(np.asarray(r["dcond_t"], float)), t_axis) for r in rr])
        desc[m.name] = dict(median_J_KL=med[m.name], median_int_Dcond=float(np.median(dc)))
        if m.transport == "horizontal_ot":
            desc[m.name].update(median_mean_absdx=float(np.median([np.mean(r["ot_absdx_t"][1:]) for r in rr])),
                                median_max_Dmove=float(np.median([np.max(r["dmove_max_t"]) for r in rr])))
    sel = dict(prereg=os.path.relpath(PREREG, ROOT), rule=rule, alpha_star=float(a_star), ratio_at_alpha_star=float(ratio[a_star]),
               within_5pct=bool(within), ladder=list(ladder), median_J_KL=med, ratio={f"{al:g}": ratio[al] for al in ladder},
               descriptive=desc, seeds=seeds, n_rows=len(rows), batch_seed=BATCH_SEED, gamma=gamma, h_bias=h_bias,
               git_rev=git_rev(), wall_seconds=time.time() - t0, blind="no error metric computed, printed or stored")
    with open(os.path.join(a.out, "alpha_selection.json"), "w") as fh:
        json.dump(sel, fh, indent=2, default=float)
    npz = {k: np.stack([np.asarray(r[k]) for r in recs]) for k in KEEP}
    for k in SCALARS:
        npz[k] = np.array([r[k] for r in recs])
    np.savez_compressed(os.path.join(a.out, "raw.npz"), **npz)
    prov = dict(script=os.path.basename(__file__), git_rev=git_rev(), host=socket.gethostname(),
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                device=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
                prereg=os.path.relpath(PREREG, ROOT), seeds=seeds, ladder=ladder, arms=[m.name for m in arms],
                batch_seed=BATCH_SEED, wall_seconds=time.time() - t0, kept_fields=list(KEEP) + list(SCALARS))
    with open(os.path.join(a.out, "provenance.json"), "w") as fh:
        json.dump(prov, fh, indent=2, default=float)
    print(f"\ndone in {time.time() - t0:.0f}s -> {os.path.relpath(a.out, ROOT)}/{{raw.npz, alpha_selection.json}}")


if __name__ == "__main__":
    main()
