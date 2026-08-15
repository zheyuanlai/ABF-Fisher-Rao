"""Does the basin that carries the verdict actually sample its interior, or just integrate to 1?

Gate C compares BASIN-INTEGRATED occupancy against the bias-aware target. SSIP spans
[0.34, 1.40] nm -- 88 % of the domain -- and carries the entire N = 64 verdict. Any failure that
redistributes walkers WITHIN that basin while preserving its integral is invisible to Gate C:
walkers jammed against the outer soft wall would still make SSIP "hold its target population".

This is the NaCl instance of a general form the C60 session stated after finding a relaxation
guard set at the explosion scale (1e6 kJ/mol/nm) that could not fire on a sterically jammed
water at 2-5e4: **a guard placed at the catastrophic scale cannot protect against the metastable
scale, and "passes the coarse check" quietly reads as "sampled correctly".** A basin-integrated
ratio is the coarse check; this is the fine one.

Reports, over the judged window and all seeds, the per-grid-point occupancy against the
per-grid-point target, split into quarters of the basin plus a shape mismatch TV and an explicit
outer-wall fraction. A jam shows as a rising P/Q toward the wall and a TV that is not small.

    python scripts/nacl_audit_within_basin.py --screen results/nacl/screen_merged --cell N64
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


def profiles(d, F_ref, beta):
    """Per-grid-point occupancy P(r) and bias-aware target Q(r), averaged over the judged
    window and all seeds. Both are normalised per checkpoint BEFORE averaging, so a checkpoint
    with more samples cannot dominate the shape."""
    occ, pmf = d["diag_occupancy"], d["diag_pmf"]
    times = np.asarray(d["diag_times"], float)
    T_ps = float(d["T_ns"]) * 1000.0
    P = np.zeros(occ.shape[2]); Q = np.zeros(occ.shape[2]); n = 0
    for c in np.flatnonzero(times >= 0.5 * T_ps):
        for s in range(occ.shape[1]):
            B = pmf[c, s]
            w = np.exp(-beta * (F_ref - B - (F_ref - B).min()))
            Q += w / w.sum()
            cnt = occ[c, s]
            P += cnt / max(cnt.sum(), 1e-300)
            n += 1
    return P / n, Q / n


def audit_basin(grid, P, Q, msk, n_quarters=4, n_edge=3, min_target_share=0.01):
    idx = np.flatnonzero(msk)
    out = dict(basin_nm=[float(grid[idx[0]]), float(grid[idx[-1]])],
               n_grid_points=int(msk.sum()),
               integrated_ratio=float(P[msk].sum() / max(Q[msk].sum(), 1e-300)),
               quarters=[], )
    # A RATIO NEEDS BOTH ARGUMENTS' POPULATIONS. A quarter whose target mass is negligible
    # produces a spectacular ratio out of nothing: measured here, CIP's inner quarter reported
    # P/Q = 1274 from 0.0175 % of walkers against a target 1000x smaller, while the basin's
    # integrated ratio was unchanged to four decimals. Carry the masses so the reader can see
    # whether a ratio has any weight behind it, and suppress the ratio outright when it does not.
    for sub in np.array_split(idx, n_quarters):
        pm, qm = float(P[sub].sum()), float(Q[sub].sum())
        share_p = pm / max(float(P[msk].sum()), 1e-300)
        share_q = qm / max(float(Q[msk].sum()), 1e-300)
        meaningful = share_q >= min_target_share
        out["quarters"].append(dict(
            r_nm=[float(grid[sub][0]), float(grid[sub][-1])],
            ratio=(float(pm / max(qm, 1e-300)) if meaningful else None),
            occupancy_share=share_p, target_share=share_q,
            ratio_meaningful=bool(meaningful),
            note=None if meaningful else
            (f"target carries {100*share_q:.4f} % of the basin (< {100*min_target_share:.2f} %); "
             f"a ratio here is division by ~0 and is reported as null, not as a finding")))
    p = P[msk] / max(P[msk].sum(), 1e-300)
    q = Q[msk] / max(Q[msk].sum(), 1e-300)
    out["shape_TV"] = float(0.5 * np.abs(p - q).sum())
    out["outer_edge_fraction"] = float(p[-n_edge:].sum())
    out["outer_edge_target"] = float(q[-n_edge:].sum())
    out["outer_edge_excess"] = float(out["outer_edge_fraction"] / max(out["outer_edge_target"], 1e-300))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", default="results/nacl/screen_merged")
    ap.add_argument("--ref", default="results/nacl/reference")
    ap.add_argument("--cell", default="N64")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rep = json.load(open(os.path.join(args.ref, "reference_report.json")))
    ref = np.load(os.path.join(args.ref, "reference.npz"))
    d = np.load(os.path.join(args.screen, f"cell_{args.cell}.npz"))
    grid = d["grid"]
    F_ref = np.interp(grid, ref["r_nm"], ref["F_ref"])
    P, Q = profiles(d, F_ref, nsys.beta_per_kJ())

    report = {}
    for lab, msk in ng.basin_masks(grid, rep["basins"]).items():
        a = audit_basin(grid, P, Q, msk)
        report[lab] = a
        print(f"{lab}: [{a['basin_nm'][0]:.2f}, {a['basin_nm'][1]:.2f}] nm, "
              f"{a['n_grid_points']}/{len(grid)} points, integrated P/Q = "
              f"{a['integrated_ratio']:.4f}   <- what Gate C sees")
        for i, qd in enumerate(a["quarters"]):
            r = (f"{qd['ratio']:6.3f}" if qd["ratio_meaningful"] else "  n/a ")
            print(f"    q{i+1} [{qd['r_nm'][0]:.2f},{qd['r_nm'][1]:.2f}]  P/Q = {r}"
                  f"   (holds {100*qd['occupancy_share']:5.2f}% of walkers, "
                  f"{100*qd['target_share']:5.2f}% of target)"
                  + ("" if qd["ratio_meaningful"] else "  <- ratio suppressed: no target mass"))
        print(f"    within-basin shape TV = {a['shape_TV']:.4f}; outermost 3 points hold "
              f"{100*a['outer_edge_fraction']:.2f}% vs {100*a['outer_edge_target']:.2f}% of target "
              f"({a['outer_edge_excess']:.2f}x)\n")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
