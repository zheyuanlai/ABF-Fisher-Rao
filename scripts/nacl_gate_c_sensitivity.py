"""What deficit does Gate C actually need in order to fire? Planted in the REAL traces.

Gate C nominally tests "occupancy < 0.5 Q* sustained over 0.2 T". The `lambda >= 16` power floor
was derived from `0.5 lambda >= 2 sqrt(lambda)` -- the point where a 50 % deficit is 2 sigma **on
one checkpoint**. That is a single-checkpoint criterion standing in for a CONTIGUOUS-SPAN test,
and the two are not the same quantity: at a 50 % deficit the mean sits exactly ON the threshold,
so noise lifts roughly half the checkpoints above it and a long contiguous run rarely forms.
The methane session measured this on their system and found the gate needs ~60 %, not 50 %.

This replicates it on NaCl. The deficit is planted in the **real** occupancy traces, not in
synthetic data, because the correlation structure of the trace is the entire quantity in
question -- synthetic samples would answer a different question. Procedure: scale one basin's
per-checkpoint counts by `f`, redistribute the removed population across the other grid points
so each checkpoint's total is preserved (the targets are a partition and must stay one), then
run the UNMODIFIED gate.

Reports the smallest planted deficit at which the gate fires on >= HIT_SEEDS of 8.

    python scripts/nacl_gate_c_sensitivity.py --cell N64 --state SSIP
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import nacl_gates as ng
from nacl import system as nsys


def plant(occ, msk, f, out_weights=None):
    """Scale the masked basin's counts by f, redistributing the removed mass over the rest so
    every checkpoint's total is unchanged. A deficit that also shrank the total would change
    the normalisation P uses and would not be the deficit the gate reads.

    **Redistribution must NOT be proportional to the existing outside counts.** Where a
    checkpoint has no walkers outside the basin, that scheme has nowhere to put the removed mass,
    silently drops it, and the basin's SHARE rises to 1.0 -- planting a 90 % deficit and getting
    P = 1.0000, the exact opposite of the intended effect. Scattered inflated checkpoints then
    break the contiguous run and the gate reports "no deficit" for a state that was emptied.
    Measured on both NaCl cells before this was fixed.

    So the removed mass is placed on the outside grid points by `out_weights` -- the bias-aware
    TARGET outside the basin, which is strictly positive everywhere the physics allows walkers
    and is what "those walkers went somewhere else" actually means. Falls back to uniform over
    the outside points if no weights are supplied.
    """
    out = np.array(occ, dtype=float, copy=True)
    tot_before = out.sum(axis=2)
    inside = out[:, :, msk]
    removed = inside.sum(axis=2) * (1.0 - f)
    out[:, :, msk] = inside * f
    if out_weights is None:
        w = np.ones(int((~msk).sum()), dtype=float)
        w = np.broadcast_to(w / w.sum(), out[:, :, ~msk].shape)
    else:
        w = out_weights[:, :, ~msk]
        w = w / np.maximum(w.sum(axis=2, keepdims=True), 1e-300)
    out[:, :, ~msk] = out[:, :, ~msk] + removed[:, :, None] * w
    # The invariant the docstring claims, ASSERTED. Not asserting it is what let the broken
    # redistribution ship a result in the first place.
    tot_after = out.sum(axis=2)
    bad = np.abs(tot_after - tot_before) > 1e-6 * np.maximum(tot_before, 1.0)
    if bad.any():
        raise RuntimeError(f"planting changed the per-checkpoint total at {int(bad.sum())} "
                           f"checkpoint-seeds; P's normalisation would be corrupted")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", default="results/nacl/screen_all")
    ap.add_argument("--ref", default="results/nacl/reference")
    ap.add_argument("--cell", default="N64")
    ap.add_argument("--state", default="SSIP")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rep = json.load(open(os.path.join(args.ref, "reference_report.json")))
    ref = np.load(os.path.join(args.ref, "reference.npz"))
    d = np.load(os.path.join(args.screen, f"cell_{args.cell}.npz"))
    grid = d["grid"]
    F_ref = np.interp(grid, ref["r_nm"], ref["F_ref"])
    beta = nsys.beta_per_kJ()
    T_ps = float(d["T_ns"]) * 1000.0
    msk = ng.basin_masks(grid, rep["basins"])[args.state]

    base = ng.gate_c(d["diag_occupancy"], d["diag_pmf"], d["diag_times"], grid,
                     F_ref, rep["basins"], beta, T_ps)[args.state]
    lam = base["power"]["lambda_expected_walkers"]
    analytic = base["power"]["detectable_deficit_frac"]
    print(f"cell {args.cell}, state {args.state}: lambda = {lam:.2f}, "
          f"analytic 2-sigma detectable deficit = {100*analytic:.0f} %")
    print(f"unplanted: {base['n_seeds_deficient']}/8 seeds deficient "
          f"(gate fires at >= {ng.HIT_SEEDS}/8)\n")
    print(f"{'planted deficit':>16} {'seeds firing':>13}   verdict")

    # bias-aware target per checkpoint/seed: where the removed walkers go
    dz = float(grid[1] - grid[0])
    pmf = d["diag_pmf"]
    W = np.empty_like(np.asarray(d["diag_occupancy"], dtype=float))
    for c in range(W.shape[0]):
        for s in range(W.shape[1]):
            B = pmf[c, s]
            w_ = np.exp(-beta * (F_ref - B - (F_ref - B).min()))
            W[c, s] = w_ / w_.sum()

    rows, fired_at = [], None
    for deficit in (0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90):
        occ = plant(d["diag_occupancy"], msk, 1.0 - deficit, out_weights=W)
        r = ng.gate_c(occ, d["diag_pmf"], d["diag_times"], grid, F_ref,
                      rep["basins"], beta, T_ps)[args.state]
        n = r["n_seeds_deficient"]
        hit = n >= ng.HIT_SEEDS
        if hit and fired_at is None:
            fired_at = deficit
        rows.append(dict(planted_deficit=deficit, n_seeds_deficient=int(n), fires=bool(hit)))
        print(f"{100*deficit:14.0f} % {n:9d}/8      {'FIRES' if hit else '-'}")

    print(f"\nGate C needs a planted deficit of >= {100*fired_at:.0f} %"
          if fired_at else "\nGate C did NOT fire at any planted deficit up to 90 %")
    if fired_at:
        print(f"against the {100*ng.DEFICIT_RATIO:.0f} % it nominally tests and the "
              f"{100*analytic:.0f} % the analytic power argument implies "
              f"({fired_at/ng.DEFICIT_RATIO:.1f}x nominal).")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(dict(cell=args.cell, state=args.state, lambda_=lam,
                           analytic_detectable=analytic, nominal=ng.DEFICIT_RATIO,
                           first_firing_deficit=fired_at, ladder=rows,
                           limitation="planted deficit is STATIONARY -- a real establishment "
                                      "failure that decays as the bias fills in would fire "
                                      "less readily, so this is a FLOOR on the detection "
                                      "threshold, not a characterisation of it"), fh, indent=2)
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
