"""Merge the two 4-seed halves of a screen cell into the 8-seed cell the gates require.

The N=64 cell was split across two GPUs as seeds 4000-4003 and 4004-4007 (independent
ensembles; Gate B/C statistics are per-seed counts, so a SEED split is safe where an ARM split
would not be). The gate analysis wants one cell with all eight, so this concatenates along the
seed axis.

Everything that must match between halves is ASSERTED rather than assumed -- grid, N, T, dt,
box, domain, checkpoint schedule -- because a silent mismatch here would produce an 8-seed cell
whose halves mean different things, and the gates would report a verdict over a population that
never existed. The seed sets are additionally required to be disjoint and to union to exactly
the preregistered 4000-4007.

Usage:
    python scripts/nacl_merge_halves.py --a results/nacl/screen --b results/nacl/screen_B \
        --out results/nacl/screen_merged --cell N64
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

SEED_AXIS_1D = ("mean_force", "pmf", "W_pmf", "W_mean_force", "eff_counts")   # (S, n_grid)
SEED_AXIS_2D = ("xi_trace",)                                                  # (F, S, N)
DIAG_SEED_AXIS = ("diag_mean_force", "diag_pmf", "diag_p_hat", "diag_eff_counts",
                  "diag_occupancy")                                           # (cp, S, n_grid)
SCALAR_MATCH = ("N", "T_ns", "n_steps", "dt_ps", "box_L_nm", "R_hi_nm", "dz")
ARRAY_MATCH = ("grid", "diag_times", "diag_steps", "xi_steps", "y_steps")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cell", default="N64")
    ap.add_argument("--expect-seeds", default="4000,4001,4002,4003,4004,4005,4006,4007")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    A = dict(np.load(os.path.join(args.a, f"cell_{args.cell}.npz")))
    B = dict(np.load(os.path.join(args.b, f"cell_{args.cell}.npz")))

    # ---- everything that must agree, asserted -------------------------------------------
    for k in SCALAR_MATCH:
        if k in A and k in B and not np.allclose(A[k], B[k]):
            raise SystemExit(f"halves disagree on {k}: {A[k]} vs {B[k]}")
    for k in ARRAY_MATCH:
        if k in A and k in B:
            if A[k].shape != B[k].shape or not np.allclose(A[k], B[k]):
                raise SystemExit(f"halves disagree on {k} (shape {A[k].shape} vs {B[k].shape})")

    sa, sb = np.asarray(A["seed_labels"]).ravel(), np.asarray(B["seed_labels"]).ravel()
    if set(sa) & set(sb):
        raise SystemExit(f"seed sets overlap: {sorted(set(sa) & set(sb))}")
    merged_seeds = np.concatenate([sa, sb])
    expect = np.array([int(x) for x in args.expect_seeds.split(",")])
    if sorted(merged_seeds.tolist()) != sorted(expect.tolist()):
        raise SystemExit(f"merged seeds {sorted(merged_seeds.tolist())} != preregistered "
                         f"{sorted(expect.tolist())}")

    out = {}
    for k in A:
        if k in SEED_AXIS_1D or k in DIAG_SEED_AXIS:
            ax = 0 if k in SEED_AXIS_1D else 1
            out[k] = np.concatenate([A[k], B[k]], axis=ax)
        elif k in SEED_AXIS_2D:
            out[k] = np.concatenate([A[k], B[k]], axis=1)
        elif k == "seed_labels":
            out[k] = merged_seeds
        elif k == "y_trace":                    # (F, S*N, 3) -- flat walker axis
            out[k] = np.concatenate([A[k], B[k]], axis=1)
        elif k == "final_positions":
            out[k] = np.concatenate([A[k], B[k]], axis=0)
        elif k == "diag_out_of_domain":         # (cp,) scalar per checkpoint: average halves
            out[k] = 0.5 * (np.asarray(A[k]) + np.asarray(B[k]))
        else:
            out[k] = A[k]                        # identical by the assertions above

    # ---- shape check: the seed axis really is 8 everywhere it should be -----------------
    S = len(merged_seeds)
    for k in SEED_AXIS_1D:
        if k in out and out[k].shape[0] != S:
            raise SystemExit(f"{k} has seed axis {out[k].shape[0]}, expected {S}")
    for k in DIAG_SEED_AXIS:
        if k in out and out[k].shape[1] != S:
            raise SystemExit(f"{k} has seed axis {out[k].shape[1]}, expected {S}")
    if out["xi_trace"].shape[1] != S:
        raise SystemExit(f"xi_trace seed axis {out['xi_trace'].shape[1]}, expected {S}")

    path = os.path.join(args.out, f"cell_{args.cell}.npz")
    np.savez_compressed(path, **out)
    man = dict(merged_from=[os.path.abspath(args.a), os.path.abspath(args.b)],
               seeds=merged_seeds.tolist(), n_seeds=int(S), cell=args.cell,
               a_manifest=json.load(open(os.path.join(args.a, "manifest.json")))
               if os.path.exists(os.path.join(args.a, "manifest.json")) else None,
               b_manifest=json.load(open(os.path.join(args.b, "manifest.json")))
               if os.path.exists(os.path.join(args.b, "manifest.json")) else None)
    with open(os.path.join(args.out, "merge_manifest.json"), "w") as fh:
        json.dump(man, fh, indent=2, default=str)
    print(f"merged {len(sa)} + {len(sb)} seeds -> {S}: {merged_seeds.tolist()}")
    print(f"xi_trace {out['xi_trace'].shape}, diag_occupancy {out['diag_occupancy'].shape}")
    print(f"-> {path}")


if __name__ == "__main__":
    main()
