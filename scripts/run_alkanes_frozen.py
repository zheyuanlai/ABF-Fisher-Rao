#!/usr/bin/env python3
"""Frozen-bias validation (Part F) for representative alkane cells.

For each (molecule, method in {abf, fr_estimated}) run in a completed production stage,
take the learned mean force B'(phi) per seed, FREEZE it, run fresh dynamics with no ABF
update and no birth--death, and reconstruct F_recon = B - beta^{-1} log p_B + C. Reports
learned-bias L2 and reconstructed-F L2 vs the reference (evaluation only).

Writes results/alkanes/<stage>_frozen/frozen_summary.csv and frozen_profiles.npz.

GPU: single visible device from {4,5,6,7}.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
torch.set_default_dtype(torch.float64)
from alkanes import core, jobs as J, metrics, potentials as pot  # noqa: E402
from alkanes.cv import DihedralCV  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--methods", default="abf,fr_estimated")
    ap.add_argument("--n-steps", type=int, default=25000)
    ap.add_argument("--n-replicas", type=int, default=None, help="override N (smaller => faster CPU run)")
    ap.add_argument("--max-seeds", type=int, default=8)
    ap.add_argument("--cells", default=None, help="substring filter on physics tag")
    args = ap.parse_args(argv)
    cfg = J.load_yaml(args.config)
    raw_dir = os.path.join(cfg["output_root"], "raw")
    out_dir = os.path.join(cfg["output_root"], f"{args.stage}_frozen")
    os.makedirs(out_dir, exist_ok=True)
    device = args.device
    want = set(args.methods.split(","))
    rows = []
    profiles = {}
    for path in sorted(glob.glob(os.path.join(raw_dir, f"{args.stage}__*.npz"))):
        d = np.load(path, allow_pickle=True)
        if str(d["name"]) not in want or str(d["init_mode"]) != "trans":
            continue
        spec = json.loads(str(d["spec_json"]))
        tag = f"{spec['molecule']}_b{spec['beta']:g}"
        if args.cells and args.cells not in tag:
            continue
        params = pot.AlkaneParams(n_atoms=(4 if spec["molecule"] == "butane" else 5),
                                  beta=spec["beta"], sigma=spec["sigma"],
                                  decouple=spec["decouple"], force_clip=spec["force_clip"])
        n_rep = int(args.n_replicas) if args.n_replicas else spec["n_replicas"]
        sim = core.AlkaneSimConfig(dt=spec["dt"], n_steps=args.n_steps, n_replicas=n_rep,
                                   save_every=args.n_steps, rng_seed=spec["rng_seed"] + 1,
                                   n_grid=spec["n_grid"], abf_bandwidth=spec["abf_bandwidth"],
                                   kde_bandwidth=spec["kde_bandwidth"],
                                   estimator_burn_in_steps=args.n_steps // 3,
                                   abf_force_clip=spec["abf_force_clip"])
        seeds = list(d["seeds"])[:args.max_seeds]
        learned = d["final_mean_force"][:len(seeds)]       # (R, n_grid) per seed
        cv = DihedralCV((0, 1, 2, 3))
        fb = core.run_frozen_bias(params, sim, learned, seeds, cv, device,
                                  initial_dihedrals=[0.0] * (params.n_atoms - 3), verbose=True)
        grid = d["grid"]; ref_F = d["ref_F"]; ref_Fp = d["ref_Fprime"]; dphi = float(d["dphi"])
        for r in range(len(seeds)):
            # learned-bias quality = online final F vs ref; frozen quality = reconstructed F vs ref
            l2_learned = metrics._circ_l2(d["final_pmf"][r], ref_F, dphi)
            l2_recon = metrics._circ_l2(fb["F_recon"][r], ref_F, dphi)
            rows.append(dict(molecule=spec["molecule"], beta=spec["beta"], method=str(d["name"]),
                             seed=int(seeds[r]), l2_online_F=l2_learned, l2_frozen_recon_F=l2_recon))
        profiles[f"{tag}_{str(d['name'])}"] = dict(grid=grid, ref_F=ref_F,
                                                   F_recon=fb["F_recon"], p_B=fb["p_B"])
        print(f"[frozen] {tag} {str(d['name'])}: median recon L2={np.median([rows[-1]['l2_frozen_recon_F']]):.4f}")
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(out_dir, "frozen_summary.csv"), index=False)
        np.savez(os.path.join(out_dir, "frozen_profiles.npz"),
                 **{k: json.dumps({kk: vv.tolist() for kk, vv in v.items()}) for k, v in profiles.items()})
        print(df.groupby(["molecule", "beta", "method"])[["l2_online_F", "l2_frozen_recon_F"]].median().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
