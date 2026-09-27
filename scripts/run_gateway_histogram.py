#!/usr/bin/env python
"""Entropic gateway: histogram online estimator ladder vs the accepted kernel estimator.

Prereg: configs/gateway_histogram/prereg.json.  One batch per estimator, both arms inside it
(shared init + noise); every batch uses the same batch_seed and row order so the ABF rows of
different estimators see identical Langevin noise too.

    CUDA_VISIBLE_DEVICES=3 python -u scripts/run_gateway_histogram.py
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import socket
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import gateway_core as gw  # noqa: E402
from run_gateway_bandwidth_audit import build_config  # noqa: E402

PREREG = os.path.join(ROOT, "configs/gateway_histogram/prereg.json")
BASE_PREREG = os.path.join(ROOT, "results/gateway_anchor/CONFIRMATORY_PREREGISTRATION.json")
CORR_PREREG = os.path.join(ROOT, "configs/information_campaign/gateway_corrected_confirmation_prereg.json")
OUT_DIR = os.path.join(ROOT, "results", "gateway_histogram")
TAKEN = (set(range(16)) | set(range(100, 132)) | set(range(300, 316)) | set(range(400, 416)) | set(range(480, 488))
         | set(range(500, 572)) | set(range(580, 588)) | set(range(800, 832)) | set(range(1200, 1208)))
KEYS = ["t", "P_regions", "Q_regions", "l2_f_t", "l2_fp_t", "ess_t", "wmax_t",
        "x_grid", "F_hat", "Fp_hat", "F_ref", "Fp_ref", "Sf_t", "C_t"]


def git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def save(path, recs, extra):
    npz = {k: np.stack([r[k] for r in recs]) for k in KEYS if k in recs[0]}
    for k in ("Mh_t", "Ch_t", "hist_edges", "Fp_bins"):
        if k in recs[0]:
            npz[k] = np.stack([r[k] for r in recs])
    for k in recs[0]:
        if k not in npz and k != "config" and not isinstance(recs[0][k], np.ndarray):
            npz[k] = np.array([r[k] for r in recs])
    npz["config_json"] = np.array([json.dumps(r["config"], sort_keys=True) for r in recs])
    npz.update({k: np.array(v) for k, v in extra.items()})
    np.savez_compressed(path, **npz)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--only", default=None, help="comma-separated estimator names")
    a = ap.parse_args()
    pre = json.load(open(PREREG))
    base, corr = json.load(open(BASE_PREREG)), json.load(open(CORR_PREREG))
    sampler, cell = base["sampler"], base["cell"]
    assert pre["cell"] == cell
    gamma = float(corr["rate"]["gamma"]); h_bias = float(corr["corrected_baseline"]["h_bias"])
    seeds = list(range(pre["seeds"]["first"], pre["seeds"]["first"] + pre["seeds"]["n"]))
    assert not (set(seeds) & TAKEN), "label collision"
    if torch.cuda.is_available():
        assert torch.cuda.device_count() == 1, "pin exactly one GPU"
    os.makedirs(a.out, exist_ok=True)
    rows = [(init, sd) for init in pre["seeds"]["inits"] for sd in seeds]
    arms = [gw.ABF, dataclasses.replace(gw.FR_UNIFORM, gamma=gamma)]
    ladder = pre["estimator_ladder"]
    if a.only:
        keep = set(a.only.split(","))
        ladder = [e for e in ladder if e["name"] in keep]
    print(f"gateway histogram ladder: {len(ladder)} estimators x {len(rows)} rows x 2 arms; gamma {gamma:g}; "
          f"seeds {seeds[0]}-{seeds[-1]} x {pre['seeds']['inits']}", flush=True)
    t_start = time.time()
    for est in ladder:
        t0 = time.time()
        cfgs = []
        for init, _ in rows:
            c = build_config(sampler, cell, init, h_bias if est["estimator"] == "kernel" else float(est.get("h", h_bias)))
            cfgs.append(dataclasses.replace(c, estimator=est["estimator"], n_bins=int(est.get("n_bins", 0))))
        spec = gw.BatchSpec(configs=cfgs, seeds=[sd for _, sd in rows], methods=arms, batch_seed=61_000)
        recs = gw.simulate_batch(spec, store_accumulators=True)
        fr = [r for r in recs if r["method"] == "fr_uniform"]
        save(os.path.join(a.out, f"{est['name']}.npz"), recs,
             dict(estimator=est["name"], gamma=gamma, h_bias=h_bias, batch_seed=61_000))
        print(f"  {est['name']:12s} {len(recs)} runs in {time.time() - t0:.0f}s; FR median repl_fraction "
              f"{np.median([r['repl_fraction'] for r in fr]):.4f}, min ESS/N {np.median([r['min_ess_frac'] for r in fr]):.3f}, "
              f"max wmax {np.median([r['max_wmax'] for r in fr]):.4f}", flush=True)
    prov = dict(script=os.path.basename(__file__), git_rev=git_rev(), host=socket.gethostname(),
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                device=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
                prereg=os.path.relpath(PREREG, ROOT), seeds=seeds, wall_seconds=time.time() - t_start,
                ladder=[e["name"] for e in ladder])
    with open(os.path.join(a.out, "provenance.json"), "w") as fh:
        json.dump(prov, fh, indent=2)
    print(f"done in {time.time() - t_start:.0f}s -> {os.path.relpath(a.out, ROOT)}")


if __name__ == "__main__":
    main()
