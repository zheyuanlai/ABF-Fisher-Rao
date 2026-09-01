#!/usr/bin/env python
"""Is the ZIF-8 endpoint error kernel-SMOOTHING BIAS, controlled by bandwidth,
rather than a sample-allocation problem?

Five independent investigations in this project have now concluded that
kernel-ABF free-energy endpoints are bias-dominated, most recently the
information-clock audit (98-99% of the squared error is not variance).  If that
is right, the estimator's bandwidth h -- not where the walkers sit -- is the
dominant lever, and no reallocation scheme can compete with simply choosing h.

ONE ABF-only trajectory is enough: the raw binned accumulators are saved, so h
can be swept entirely offline at fixed dynamics.  That isolates the READ-OUT
bias from any change in sampling.

    CUDA_VISIBLE_DEVICES=3 python -u scripts/zif8_bandwidth_sweep.py
"""
from __future__ import annotations

import argparse, json, os, sys
import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from alkanes import periodic as per                                    # noqa: E402
from zif8.core_zif8 import (ZIF8SimConfig, ZIF8System, engine_kwargs,   # noqa: E402
                            mean_force_regularized, run_sampler)

PREREG = os.path.join(ROOT, "configs/uniform_campaign/zif8_prereg.json")
OUT = os.path.join(ROOT, "results/information_campaign")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--temperature", type=float, default=300.0)
    ap.add_argument("--steps", type=int, default=300000)      # 150 ps
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--h-bias", type=float, default=None,
                    help="ONLINE bandwidth for the bias force (A). "
                         "Default = the prereg value.")
    ap.add_argument("--bandwidths", type=float, nargs="*",
                    default=[0.40, 0.30, 0.20, 0.15, 0.10, 0.07, 0.05, 0.03, 0.02])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    pre = json.load(open(PREREG))
    s = {k: v for k, v in pre["sampler"].items() if not k.startswith("_")}
    tag = f"T{a.temperature:g}"
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    system = ZIF8System(a.temperature, dev, root=ROOT, **engine_kwargs(pre))
    sim = ZIF8SimConfig(**s, rng_seed=20260980)
    sim.n_steps = a.steps
    if a.h_bias is not None:
        sim.abf_bandwidth_A = float(a.h_bias)
    tag = f"{tag}_hb{sim.abf_bandwidth_A:g}" if a.h_bias is not None else tag
    pool = os.path.join(ROOT, f"cache/zif8/init_pool_T{a.temperature:g}.npz")
    raw_path = os.path.join(OUT, f"zif8_raw_accumulators_{tag}.npz")

    if os.path.exists(raw_path):
        print(f"reusing {raw_path}")
        z = np.load(raw_path)
        fsum, csum = z["raw_fsum"], z["raw_csum"]
    else:
        print(f"ABF-only, {a.seeds} seeds x {sim.n_replicas} replicas, "
              f"{sim.n_steps} steps ({sim.n_steps*sim.dt:.0f} ps), "
              f"saving RAW accumulators", flush=True)
        out = run_sampler("abf", system, sim, seeds=list(range(a.seeds)),
                          init_pool=pool, verbose=True, progress_every=10)
        fsum, csum = out["raw_fsum"], out["raw_csum"]
        np.savez_compressed(raw_path, raw_fsum=fsum, raw_csum=csum,
                            n_steps=sim.n_steps, n_replicas=sim.n_replicas,
                            seeds=np.arange(a.seeds))
        print(f"wrote {raw_path}")

    ref = np.load(os.path.join(ROOT, f"results/uniform_campaign/zif8/reference/"
                                     f"reference_T{a.temperature:g}.npz"), allow_pickle=True)
    F_ref = np.asarray(ref["F"], float); kT = float(ref["kT"])
    G = fsum.shape[-1]
    grid, dphi = per.periodic_grid(G, dtype=torch.float64)
    kf = system.k_phi
    fs = torch.as_tensor(fsum, dtype=torch.float64)
    cs = torch.as_tensor(csum, dtype=torch.float64)

    print(f"\n{'h (A)':>7} {'h (bins)':>9} {'e_F med':>9} {'e_F sd':>8} "
          f"{'bias':>8} {'sd(seed)':>9} {'barrier':>9} {'vs ref':>8}")
    rows = []
    for h in a.bandwidths:
        K = per.wrapped_gaussian_kernel_matrix(grid, h * kf)
        mf = mean_force_regularized(fs, cs, K, s["abf_min_count"])
        F = per.free_energy_from_mean_force(mf, grid, dphi).numpy()
        d = F - F_ref[None, :]; d = d - d.mean(-1, keepdims=True)
        eF = np.sqrt((d * d).mean(-1))
        bias = np.sqrt((d.mean(0) ** 2).mean())
        sd = np.sqrt(d.var(0, ddof=1).mean())
        bar = (F.max(-1) - F.min(-1))
        bref = F_ref.max() - F_ref.min()
        rows.append(dict(h_A=h, h_bins=h * kf / dphi, eF=float(np.median(eF)),
                         eF_sd=float(eF.std()), bias=float(bias), seed_sd=float(sd),
                         barrier=float(np.median(bar)),
                         barrier_err_pct=float(100 * (np.median(bar) / bref - 1))))
        r = rows[-1]
        print(f"{h:7.3f} {r['h_bins']:9.2f} {r['eF']:9.4f} {r['eF_sd']:8.4f} "
              f"{r['bias']:8.4f} {r['seed_sd']:9.4f} {r['barrier']:9.3f} "
              f"{r['barrier_err_pct']:+7.2f}%")
    best = min(rows, key=lambda r: r["eF"])
    prod = [r for r in rows if abs(r["h_A"] - s["abf_bandwidth_A"]) < 1e-9]
    print(f"\n  production bandwidth h = {s['abf_bandwidth_A']} A -> "
          f"e_F {prod[0]['eF']:.4f} kJ/mol" if prod else "")
    print(f"  best in sweep    h = {best['h_A']} A -> e_F {best['eF']:.4f} kJ/mol")
    if prod:
        print(f"  ENDPOINT-MSE FACTOR from bandwidth alone: "
              f"{(prod[0]['eF']/best['eF'])**2:.1f}x")
        print(f"  (uniform mFR moved the same endpoint by a factor of "
              f"{1.0367**2:.2f}x, in the WRONG direction)")
    with open(os.path.join(OUT, f"zif8_bandwidth_sweep_{tag}.json"), "w") as fh:
        json.dump(dict(rows=rows, production_h=s["abf_bandwidth_A"],
                       n_steps=int(sim.n_steps), n_seeds=a.seeds), fh, indent=2)
    print(f"  wrote zif8_bandwidth_sweep_{tag}.json")


if __name__ == "__main__":
    main()
