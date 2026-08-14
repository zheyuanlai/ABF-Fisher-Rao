"""Merge seed-split screen halves into one cell file for the gate analysis.

The 8 preregistered seeds (4000-4007) were run as two 4-seed processes because the engine is
fastest at B=256 and collapses above B=512 (measured: 1023 vs 124 ns/day).  Seeds are
independent ensembles -- each carries its own ABF estimator and bias and never sees another's
samples -- so concatenating along the seed axis reconstructs exactly the object a single
8-seed process would have produced for every per-seed statistic Gate B and Gate C use.

What this may NOT be used for: an ARM comparison (mFR vs ABF vs sham).  Those must share one
process, because absolute levels are not reproducible across processes (the WCA finding) and
only within-block paired contrasts are quotable.  Gate B/C count seeds, so they are safe.

Refuses unless the union is exactly the preregistered block, once each.

Usage:
    python scripts/nacl_screen_merge.py --parts results/nacl/screen results/nacl/screen_B \
                                        --out results/nacl/screen_merged
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

EXPECTED = list(range(4000, 4008))
# axis along which each array carries the seed dimension
SEED_AXIS = dict(mean_force=0, pmf=0, eff_counts=0, xi_trace=1, y_trace=1,
                 diag_occupancy=1, diag_pmf=1, diag_mean_force=1, diag_p_hat=1,
                 diag_eff_counts=1, diag_out_of_domain=1, final_positions=0,
                 W_pmf=0, W_mean_force=0)
SHARED = ("grid", "dz", "N", "T_ns", "n_steps", "dt_ps", "box_L_nm", "R_hi_nm",
          "xi_steps", "y_steps", "diag_times", "diag_steps")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", required=True)
    ap.add_argument("--out", default="results/nacl/screen_merged")
    ap.add_argument("--cell", default="cell_N64.npz")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    parts = [np.load(os.path.join(p, args.cell)) for p in args.parts]
    seeds = np.concatenate([z["seed_labels"] for z in parts])
    order = np.argsort(seeds)
    if sorted(seeds.tolist()) != EXPECTED:
        raise SystemExit(f"merged seeds {sorted(seeds.tolist())} != preregistered block "
                         f"{EXPECTED}; refusing (a partial or duplicated block makes the "
                         "6-of-8 thresholds meaningless)")
    for key in SHARED:
        if key in parts[0]:
            ref = np.asarray(parts[0][key])
            for z in parts[1:]:
                if not np.array_equal(ref, np.asarray(z[key])):
                    raise SystemExit(f"halves disagree on shared field '{key}'; they were not "
                                     "run with the same frozen settings")

    out = {k: np.asarray(parts[0][k]) for k in SHARED if k in parts[0]}
    out["seed_labels"] = seeds[order]
    for key, axis in SEED_AXIS.items():
        if key not in parts[0]:
            continue
        arrs = [np.asarray(z[key]) for z in parts]
        if any(a.shape[axis] != len(z["seed_labels"]) for a, z in zip(arrs, parts)):
            raise SystemExit(f"'{key}' does not carry the seed dimension on axis {axis}")
        merged = np.concatenate(arrs, axis=axis)
        out[key] = np.take(merged, order, axis=axis)

    np.savez_compressed(os.path.join(args.out, args.cell), **out)
    json.dump(dict(stage="nacl_screen_merged", parts=list(args.parts),
                   seeds=out["seed_labels"].tolist(),
                   note="seed-axis concatenation; safe for per-seed Gate B/C statistics, NOT "
                        "for arm comparisons (those must share one process)"),
              open(os.path.join(args.out, "manifest.json"), "w"), indent=2)
    print(f"merged seeds {out['seed_labels'].tolist()} -> {args.out}/{args.cell}")


if __name__ == "__main__":
    main()
