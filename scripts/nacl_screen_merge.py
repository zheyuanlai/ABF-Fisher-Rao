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
#: Fields that may legitimately be AVERAGED across halves (per-checkpoint scalar diagnostics,
#: not per-seed data). Everything else must be classifiable; see the refusal below.
MEANABLE = ("diag_out_of_domain",)


def classify(key, arrays, parts, S_total, N):
    """Infer how a field carries the seed dimension, from its SHAPE rather than a hard-coded
    table.  A table is a claim about every field's layout that silently goes stale when the
    sampler adds one; inference plus an explicit report is auditable.  Returns
    (kind, axis) with kind in {seed, walker, shared, mean}."""
    a0, z0 = arrays[0], parts[0]
    s0 = len(z0["seed_labels"])
    if a0.ndim >= 1 and a0.shape[0] == s0:
        return "seed", 0
    if a0.ndim >= 2 and a0.shape[1] == s0:
        return "seed", 1
    if a0.ndim >= 1 and a0.shape[0] == s0 * N:
        return "walker", 0
    if a0.ndim >= 2 and a0.shape[1] == s0 * N:
        return "walker", 1
    if all(np.array_equal(a0, a) for a in arrays[1:]):
        return "shared", None
    return "mean", None


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
    N = int(parts[0]["N"])
    S_total = len(seeds)

    out, report = {}, {}
    for key in parts[0].files:
        if key == "seed_labels":
            continue
        arrays = [np.asarray(z[key]) for z in parts]
        kind, axis = classify(key, arrays, parts, S_total, N)
        report[key] = f"{kind}" + (f" axis {axis}" if axis is not None else "")
        if kind == "seed":
            out[key] = np.take(np.concatenate(arrays, axis=axis), order, axis=axis)
        elif kind == "walker":
            blocks = [a.reshape(*a.shape[:axis], len(z["seed_labels"]), N, *a.shape[axis+1:])
                      for a, z in zip(arrays, parts)]
            merged = np.take(np.concatenate(blocks, axis=axis), order, axis=axis)
            out[key] = merged.reshape(*merged.shape[:axis], S_total * N,
                                      *merged.shape[axis+2:])
        elif kind == "shared":
            out[key] = arrays[0]
        elif key in MEANABLE:                    # per-checkpoint scalar diagnostics
            out[key] = np.mean(np.stack(arrays), axis=0)
        else:
            # AVERAGING AN UNRECOGNISED FIELD IS A SILENT CORRUPTION. classify() infers the
            # layout from shape, so a field it does not recognise -- a new one the sampler
            # adds, or one whose leading dimension coincidentally equals the half-block seed
            # count -- would land here and be averaged across halves, giving a merged cell in
            # which that field means something neither half meant. Nothing raises, and the
            # merge reports success. Averaging is therefore allowed only for fields explicitly
            # declared meanable; anything else stops the merge.
            raise SystemExit(
                f"cannot classify field '{key}' (shape {arrays[0].shape}, kind={kind}). "
                f"Refusing to average it into the merged cell: averaging a field whose layout "
                f"is unknown corrupts it silently. Add it to MEANABLE if it really is a "
                f"per-checkpoint scalar, or extend classify() -- do not let it fall through.")
    out["seed_labels"] = seeds[order]

    for k, v in sorted(report.items()):
        print(f"   {k:22s} {v}")
    np.savez_compressed(os.path.join(args.out, args.cell), **out)
    json.dump(dict(stage="nacl_screen_merged", parts=list(args.parts),
                   seeds=out["seed_labels"].tolist(), layout=report,
                   note="seed-axis concatenation; safe for per-seed Gate B/C statistics, NOT "
                        "for arm comparisons (those must share one process)"),
              open(os.path.join(args.out, "manifest.json"), "w"), indent=2)
    print(f"merged seeds {out['seed_labels'].tolist()} -> {args.out}/{args.cell}")


if __name__ == "__main__":
    main()
