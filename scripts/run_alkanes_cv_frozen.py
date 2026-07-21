#!/usr/bin/env python3
"""Frozen-bias validation for the 2-D torsion CV (fairness control B).

For each completed ABF / mFR run of a target cell, freeze the learned 2-D bias ``B(z)``,
run fresh independent dynamics with NO ABF update and NO birth--death, accumulate the joint
biased marginal ``p_B(phi1,phi2)``, and reconstruct
``F_recon = B - beta^{-1} log p_B + C``. Comparing ``F_recon`` with the reference certifies
that the learned bias (not a resampling artifact) carries the free-energy information, and
localizes any online mean-force floor. No reference enters the dynamics.

GPU: single visible device from {4,5,6,7}.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
torch.set_default_dtype(torch.float64)
from alkanes import core2d as c2, core_dist as cd, jobs_cv as J, metrics_cv as MC, potentials as pot  # noqa: E402
from alkanes.cv2d import JointDihedralCV2D  # noqa: E402
from alkanes.distance_cv import DistanceCV  # noqa: E402
from alkanes import interval as iv  # noqa: E402
import math  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--cell-contains", default="b2_trans", help="substring of run_id cell to select")
    ap.add_argument("--methods", default="abf,fr_estimated")
    ap.add_argument("--n-steps", type=int, default=40000)
    ap.add_argument("--n-replicas", type=int, default=2048)
    ap.add_argument("--seeds", type=int, nargs="+", default=[201, 202, 203, 204])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    dev = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    if args.device == "cuda":
        assert torch.cuda.device_count() == 1
    cfg = J.load_yaml(args.config)
    root = cfg["output_root"]; raw_dir = os.path.join(root, "raw")
    out_dir = os.path.join(root, "frozen"); os.makedirs(out_dir, exist_ok=True)
    keep = set(args.methods.split(","))

    results = []
    for path in sorted(glob.glob(os.path.join(raw_dir, "*.npz"))):
        d = np.load(path, allow_pickle=True)
        if "kind" not in d.files:
            continue
        kind = str(d["kind"])
        run_name = str(d["name"]) if "name" in d.files else str(d["method"])
        if run_name not in keep or args.cell_contains not in str(d["run_id"]):
            continue
        if args.stage and "stage" in d.files and str(d["stage"]) != args.stage:
            continue
        spec = json.loads(str(d["spec_json"]))
        if kind == "joint2d":
            p = pot.AlkaneParams(n_atoms=5, beta=float(d["beta"]), sigma=float(d["sigma"]),
                                 decouple=bool(d["decouple"]), force_clip=float(spec["force_clip"]))
            sim = c2.Sim2DConfig(dt=float(spec["dt"]), n_steps=args.n_steps, n_replicas=args.n_replicas,
                                 save_every=args.n_steps, rng_seed=20260719, n_grid=int(spec["grid2d"]),
                                 abf_bandwidth=float(spec["abf_bandwidth2d"]), kde_bandwidth=float(spec["kde_bandwidth2d"]),
                                 abf_force_clip=float(spec["abf_force_clip"]), estimator_burn_in_steps=args.n_steps // 4)
            cv = JointDihedralCV2D()
            init = J.make_init(J.CVRunSpec(**spec))
            fb = c2.run_frozen_bias_2d(p, sim, d["final_pmf"].mean(0), args.seeds, cv, dev, initial_dihedrals=init, verbose=True)
            F_ref = d["ref_joint_F"]; dphi = float(d["dphi"]); grid = d["grid1"]
            mask = (F_ref - F_ref.min()) <= float(d["thermal_delta"])
            l2_recon = MC.l2_2d_np(fb["F_recon"].mean(0), F_ref, dphi, dphi, mask)
            l2_online = MC.l2_2d_np(d["final_pmf"].mean(0), F_ref, dphi, dphi, mask)
        else:  # dist (R15/R14)
            p = pot.AlkaneParams(n_atoms=int(4 if str(d["molecule"]) == "butane" else 5),
                                 beta=float(d["beta"]), sigma=float(d["sigma"]),
                                 decouple=bool(d["decouple"]), force_clip=float(spec["force_clip"]))
            sim = cd.DistSimConfig(dt=float(spec["dt"]), n_steps=args.n_steps, n_replicas=args.n_replicas,
                                   save_every=args.n_steps, rng_seed=20260719, R_lo=float(spec["R_lo"]),
                                   R_hi=float(spec["R_hi"]), wall_lo=float(spec["wall_lo"]), wall_hi=float(spec["wall_hi"]),
                                   k_wall=float(spec["k_wall"]), n_grid=int(spec["dist_n_grid"]),
                                   abf_bandwidth=float(spec["dist_abf_bandwidth"]), kde_bandwidth=float(spec["dist_kde_bandwidth"]),
                                   abf_force_clip=float(spec["abf_force_clip"]), estimator_burn_in_steps=args.n_steps // 4)
            cv = DistanceCV(int(spec["cv_i"]), int(spec["cv_j"]))
            init = J.make_init(J.CVRunSpec(**spec))
            fb = cd.run_frozen_bias_dist(p, sim, d["final_mean_force"].mean(0), args.seeds, cv, dev, initial_dihedrals=init, verbose=True)
            F_ref = d["ref_F"]; dz = float(d["dz"]); grid = d["grid"]
            mask = (F_ref - F_ref.min()) <= float(d["thermal_delta"])
            l2_recon = MC._interval_l2(fb["F_recon"].mean(0), F_ref, dz, mask)
            l2_online = MC._interval_l2(d["final_pmf"].mean(0), F_ref, dz, mask)
        rec = {"kind": kind, "name": run_name, "method": str(d["method"]),
               "stage": (str(d["stage"]) if "stage" in d.files else ""), "cell": args.cell_contains,
               "beta": float(d["beta"]), "l2_recon": float(l2_recon), "l2_online": float(l2_online)}
        results.append(rec)
        np.savez(os.path.join(out_dir, f"frozen_{kind}_{run_name}_{args.cell_contains}_b{float(d['beta']):g}.npz"),
                 F_recon=fb["F_recon"], p_B=fb["p_B"], B=fb["B"], F_ref=F_ref, grid=grid, **rec)
        print(f"[frozen] {kind} {run_name:14s} b{float(d['beta']):g}: recon L2={l2_recon:.4f} online L2={l2_online:.4f}")
    with open(os.path.join(out_dir, "frozen_summary.json"), "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"[frozen] wrote {out_dir}/frozen_summary.json ({len(results)} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
