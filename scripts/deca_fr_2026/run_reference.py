"""One independent build of the deca-alanine umbrella + MBAR reference (2026-10 rebuild).

    python scripts/deca_fr_2026/run_reference.py --build-index 0 --out results/deca_fr_2026/reference

The 2026-08 reference (results/deca/reference/reference_summary.json: 3 builds x 96 windows x 32
replicas, 2 ns/replica, span 179.6 kJ/mol, between-build L2 0.604, drift 3.18 kJ/mol) was built
on an H200; its raw profile never reached NSCC.  It is rebuilt here with the SAME frozen
``UmbrellaConfig`` (same seeds per build), one build per process so the three builds can run on
three GPUs, plus the per-cell conditional mean-force accumulation that gives an independent
``F'_ref``.  ``merge_reference.py`` does MBAR, consensus, drift, Gate A and the acceptance test.
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
from deca.umbrella import UmbrellaConfig, run_umbrella, window_centers       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-index", type=int, required=True)
    ap.add_argument("--out", default="results/deca_fr_2026/reference")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-prod-steps", type=int, default=2_000_000)      # 2 ns, as accepted in 2026-08
    ap.add_argument("--checkpoint-every", type=int, default=500_000)    # 0.5 ns
    # smoke-only overrides
    ap.add_argument("--n-windows", type=int, default=None)
    ap.add_argument("--n-rep", type=int, default=None)
    ap.add_argument("--n-pull-steps", type=int, default=None)
    ap.add_argument("--n-equil-steps", type=int, default=None)
    ap.add_argument("--sample-every", type=int, default=None)
    ap.add_argument("--win-lo", type=float, default=None)
    ap.add_argument("--win-hi", type=float, default=None)
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("no CUDA device visible")

    over = {k: v for k, v in dict(
        n_prod_steps=args.n_prod_steps, n_windows=args.n_windows, n_rep=args.n_rep,
        n_pull_steps=args.n_pull_steps, n_equil_steps=args.n_equil_steps,
        sample_every=args.sample_every, win_lo=args.win_lo, win_hi=args.win_hi).items()
        if v is not None}
    if over:
        print(f"!! NON-DEFAULT UMBRELLA CONFIG (smoke only): {over}", flush=True)
    cfg = UmbrellaConfig(**over)
    b = int(args.build_index)
    raw_dir = os.path.join(args.out, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    engine = C.engine_for(args.device, compiled=True)
    prov = C.provenance_umbrella(cfg, engine, args.device, b) if hasattr(C, "provenance_umbrella") \
        else dict(config=cfg.__dict__, config_hash=cfg.config_hash(), parameter_hash=engine.parameter_hash(),
                  torch=torch.__version__, git=C.git_commit(), build_index=b,
                  device=torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu")
    with open(os.path.join(raw_dir, f"build{b}_provenance.json"), "w") as fh:
        json.dump(prov, fh, indent=2)

    ckpt_path = os.path.join(raw_dir, f"build{b}_checkpoints.npz")
    ckpts = []

    def on_checkpoint(step, snap):
        ns = step * cfg.dt / 1000.0
        ckpts.append(dict(ns=ns, n_sample=int(snap["xi"].shape[0]),
                          floc_sum=snap["floc_sum"].copy(), floc_cnt=snap["floc_cnt"].copy()))
        # the xi/y samples so far: MBAR at every checkpoint is done in the merge step from the
        # final sample arrays (samples are appended in time order, so a prefix IS the checkpoint)
        np.savez_compressed(ckpt_path, ns=np.array([c["ns"] for c in ckpts]),
                            n_sample=np.array([c["n_sample"] for c in ckpts]),
                            floc_sum=np.stack([c["floc_sum"] for c in ckpts]),
                            floc_cnt=np.stack([c["floc_cnt"] for c in ckpts]))
        print(f"    [checkpoint {ns:.2f} ns/replica, build {b}] saved", flush=True)
        return False

    print(f"build {b}: {cfg.n_windows} windows x {cfg.n_rep} replicas = {cfg.n_windows*cfg.n_rep} "
          f"states, {cfg.n_prod_steps*cfg.dt/1000:.2f} ns production", flush=True)
    out = run_umbrella(engine, cfg, build_index=b, n_builds=1, device=args.device,
                       dtype=torch.float64, progress_every=100_000,
                       checkpoint_every=args.checkpoint_every, on_checkpoint=on_checkpoint)
    np.savez_compressed(
        os.path.join(raw_dir, f"build{b}.npz"),
        xi=out["xi"].astype(np.float32), y=out["y"].astype(np.int8), keep=out["keep"],
        n_hbonds=out["n_hbonds"].astype(np.float32), alpha_frac=out["alpha_frac"].astype(np.float32),
        rg=out["rg"].astype(np.float32), ca_rmsd_helix=out["ca_rmsd_helix"].astype(np.float32),
        floc_sum=out["floc_sum"], floc_cnt=out["floc_cnt"], centers=out["centers"],
        pull_error_nm=out["pull_error_nm"], n_sample=out["n_sample"],
        runtime_seconds=out["runtime_seconds"], build_index=b,
        n_excluded=out["n_excluded"], n_fail_cis_pull=out["n_fail_cis_pull"],
        n_fail_chirality_pull=out["n_fail_chirality_pull"],
        final_thermal_pass=out["final_thermal_pass"], final_n_cis=out["final_n_cis"],
        config_json=np.array(json.dumps(out["config"])))
    print(f"build {b} DONE: {out['runtime_seconds']/3600:.2f} h, {out['n_sample']} samples/replica, "
          f"{out['n_excluded']} replicas excluded", flush=True)


if __name__ == "__main__":
    main()
