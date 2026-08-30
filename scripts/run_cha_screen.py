#!/usr/bin/env python
"""ABF-only screen for the olefin/CHA cells, classified by the FROZEN rules in
configs/uniform_campaign/cha_prereg.json (T_cover / T_marg / L_marg).  No FR arm
exists here; the classification alone licenses (or refuses) the two-arm run.

    CUDA_VISIBLE_DEVICES=3 python -u scripts/run_cha_screen.py --guest ethene --temperature 450
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
OUT = os.path.join(ROOT, "results/uniform_campaign/cha/screen")
SCREEN_SEEDS = list(range(8))
SCREEN_RNG = 20260915


def classify(out, sim, xi_A, xi_B):
    t = np.asarray(out["times"], dtype=float)
    T = t[-1]
    fA = np.asarray(out["frac_A"], dtype=float).mean(1)
    fB = np.asarray(out["frac_B"], dtype=float).mean(1)
    grid = out["grid"]
    scoring = (grid >= xi_A - 1.0) & (grid <= xi_B + 1.0)
    # per-save visited-bin count is monotone; use it for coverage
    nvis = np.asarray(out["n_visited_bins"], dtype=float).mean(1)
    n_scoring = int(scoring.sum())
    p_hat = np.asarray(out["p_hat"], dtype=float).mean(1)     # (T, G)
    dz = float(out["dz"])
    u = np.zeros_like(grid); u[scoring] = 1.0 / (scoring.sum() * dz)
    tv = 0.5 * np.abs(p_hat[:, scoring] - u[None, scoring]).sum(1) * dz

    both = (fA >= 0.05) & (fB >= 0.05) & (nvis >= n_scoring)
    T_cover = float(t[np.argmax(both)]) if both.any() else float("inf")
    T_marg = float("inf")
    hold = max(1, int(0.1 * len(t)))
    for i in range(len(t)):
        if (tv[i:i + hold] < 0.10).all():
            T_marg = float(t[i]); break
    if T_marg < 0.25 * T:
        verdict = "abf_sufficient"
    elif T_cover < 0.25 * T and 0.25 * T <= T_marg <= 0.8 * T:
        verdict = "establishment_limited"
    elif T_cover > 0.5 * T or not both.any():
        verdict = "discovery_limited"
    else:
        verdict = "intermediate"
    return dict(T=T, T_cover=T_cover, T_marg=T_marg,
                L_marg=(T_marg - T_cover if np.isfinite(T_marg) and np.isfinite(T_cover)
                        else float("inf")),
                tv_final=float(tv[-1]), fA_final=float(fA[-1]), fB_final=float(fB[-1]),
                crossings=int(np.asarray(out["n_crossings"]).sum()),
                unvisited_scoring_bins=int(n_scoring - nvis[-1]),
                verdict=verdict)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--guest", required=True)
    ap.add_argument("--temperature", type=float, required=True)
    a = ap.parse_args()
    pre = json.load(open(PREREG))
    tag = f"{a.guest}_{a.temperature:g}"
    os.makedirs(OUT, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    system = CHASystem(a.guest, a.temperature, device, root=ROOT)
    s = pre["sampler"]
    sim = CHASimConfig(**{k: v for k, v in s.items() if not k.startswith("_")},
                       rng_seed=SCREEN_RNG)
    print(f"CHA ABF-only screen: {tag}, {len(SCREEN_SEEDS)} labels, "
          f"{sim.n_steps} steps", flush=True)
    out = run_sampler("abf", system, sim, seeds=SCREEN_SEEDS, verbose=True)
    cls = classify(out, sim, system.xi_A, system.xi_B)
    print(f"  T_cover={cls['T_cover']:.1f}  T_marg={cls['T_marg']:.1f}  "
          f"L_marg={cls['L_marg']:.1f}  (T={cls['T']:.0f})  tv_final={cls['tv_final']:.3f}")
    print(f"  crossings={cls['crossings']}  unvisited scoring bins="
          f"{cls['unvisited_scoring_bins']}  fA/fB final "
          f"{cls['fA_final']:.2f}/{cls['fB_final']:.2f}")
    print(f"  VERDICT: {cls['verdict']}")
    np.savez_compressed(os.path.join(OUT, f"screen_{tag}.npz"),
                        **{k: v for k, v in out.items()
                           if isinstance(v, (np.ndarray, np.generic, int, float, str))})
    with open(os.path.join(OUT, f"screen_{tag}.json"), "w") as fh:
        json.dump(dict(cls, tag=tag, seeds=SCREEN_SEEDS, rng_seed=SCREEN_RNG,
                       config_hash=sim.config_hash()), fh, indent=2)
    print(f"  wrote screen_{tag}.npz/.json")


if __name__ == "__main__":
    main()
