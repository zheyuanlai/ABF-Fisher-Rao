"""Does the N=64 verdict's CIP claim actually have the power it assumes?

RESULT_N64.md rests the cell on SSIP (lambda 61.4, powered) and reports CIP NON-BINDING
(lambda 1.57) -- correctly, because "occupancy < 0.5 Q*" on a state holding ~2 walkers is
arithmetically "is CIP empty right now". But the write-up then CLEARS CIP with a different
statistic: "what clears CIP is the time-averaged occupancy, which averages 79 checkpoints x 8
seeds and is well estimated". That sentence carries the verdict for CIP and **no error bar was
ever computed for it.** An unpowered gate cannot support "no deficit" any more than it supports
"deficit", so if the replacement statistic is not itself powered, the honest N=64 outcome for
CIP is UNKNOWN, and the cell is ABF-sufficient only on SSIP.

This script tests the replacement statistic on its own terms.

The gate does not ask "is CIP empty at time t". It asks whether a deficit is **sustained for
DEFICIT_FRACTION * T = 312.5 ps**. So the powered form of the same question averages occupancy
over a 312.5 ps sliding window (~31 checkpoints) instead of reading one checkpoint. The error
bar needs no autocorrelation model and no Poisson assumption: the 8 seeds are independent
ensembles, so the spread of the per-seed windowed means across seeds IS the standard error.

Reports, for every sliding window and for CIP and SSIP both (SSIP is the positive control -- a
guard that only ever refuses is indistinguishable from one that works):

    ratio = <P>_window / <Q>_window   mean over 8 seeds +- SEM across seeds

and asks whether the worst window is separated from the 0.5 deficit threshold by 2 SEM. If the
worst window's upper 2-sigma bound is still above 0.5, no sustained deficit of the size the gate
tests is compatible with the data, and the CIP claim is powered. If it straddles 0.5, it is not,
and RESULT_N64.md must say CIP is UNKNOWN rather than clear.
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


def windowed_ratios(diag_occ, diag_pmf, diag_times, grid, F_ref, msk, beta, T_ps):
    """Per-seed mean P and mean Q inside each sliding window of the required deficit length."""
    n_cp, S, _ = diag_occ.shape
    times = np.asarray(diag_times, float)
    dz = float(grid[1] - grid[0])
    need = ng.DEFICIT_FRACTION * T_ps
    second = np.flatnonzero(times >= 0.5 * T_ps)

    P = np.zeros((n_cp, S))
    Q = np.zeros((n_cp, S))
    for c in range(n_cp):
        for s in range(S):
            B_t = diag_pmf[c, s]
            w = np.exp(-beta * (F_ref - B_t - (F_ref - B_t).min()))
            Q[c, s] = float(w[msk].sum() * dz / max(w.sum() * dz, 1e-300))
            counts = diag_occ[c, s]
            P[c, s] = float(counts[msk].sum() / max(counts.sum(), 1e-300))

    wins = []
    for i0 in second:
        # The window must SPAN at least `need`. Taking only checkpoints with
        # times <= t0 + need gives a span short by up to one checkpoint spacing, so requiring
        # span >= need on that set discards every window except one that happens to land
        # exactly -- 1 window scanned instead of 48, reported as a pass. Include the first
        # checkpoint at or past t0 + need so the span genuinely covers the required duration.
        idx = np.flatnonzero(times >= times[i0])
        past = np.flatnonzero(times[idx] >= times[i0] + need - 1e-9)
        if past.size == 0:
            continue                                   # window runs off the end of the run
        j = idx[: past[0] + 1]
        pbar, qbar = P[j].mean(axis=0), Q[j].mean(axis=0)      # (S,) per-seed window means
        wins.append(dict(t0=float(times[j[0]]), t1=float(times[j[-1]]),
                         n_cp=int(len(j)), ratio_per_seed=(pbar / np.maximum(qbar, 1e-300))))
    return wins, P, Q


def verdict_from_windows(wins, threshold):
    """Worst sliding window, and what it does or does not exclude.

    Shared by the driver and its tests on purpose: a test that reimplements the statistic it is
    checking passes against its own copy, which has already happened once in this repo.
    """
    S = len(wins[0]["ratio_per_seed"])
    rows = []
    for w in wins:
        r = w["ratio_per_seed"]
        m = float(r.mean())
        sem = float(r.std(ddof=1) / np.sqrt(S))       # seeds are INDEPENDENT ensembles
        rows.append(dict(ratio=m, sem=sem, hi=m + 2 * sem, lo=m - 2 * sem, window=w))
    worst = min(rows, key=lambda d: d["hi"])          # window closest to the threshold
    worst_seed = float(min(float(w["ratio_per_seed"].min()) for w in wins))
    return dict(n_windows=len(wins), n_seeds=S, worst_single_seed=worst_seed,
                SUSTAINED_DEFICIT=bool(worst["hi"] < threshold),
                ESTABLISHED_WITH_POWER=bool(worst["lo"] > threshold), **worst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", default="results/nacl/screen_merged")
    ap.add_argument("--ref", default="results/nacl/reference")
    ap.add_argument("--cell", default="N64")
    ap.add_argument("--out", default="results/nacl/screen_merged/cip_power_audit.json")
    args = ap.parse_args()

    rep = json.load(open(os.path.join(args.ref, "reference_report.json")))
    ref = np.load(os.path.join(args.ref, "reference.npz"))
    basins, beta = rep["basins"], nsys.beta_per_kJ()

    d = np.load(os.path.join(args.screen, f"cell_{args.cell}.npz"))
    T_ps = float(d["T_ns"]) * 1000.0
    grid = d["grid"]
    F_ref = np.interp(grid, ref["r_nm"], ref["F_ref"])
    masks = ng.basin_masks(grid, basins)

    print(f"cell {args.cell}: T = {T_ps:.1f} ps, required sustained deficit = "
          f"{ng.DEFICIT_FRACTION * T_ps:.1f} ps, threshold ratio = {ng.DEFICIT_RATIO}\n")

    out = {}
    for lab, msk in masks.items():
        wins, P, Q = windowed_ratios(d["diag_occupancy"], d["diag_pmf"], d["diag_times"],
                                     grid, F_ref, msk, beta, T_ps)
        if not wins:
            print(f"{lab}: no complete window fits in the judged half -- NOT COMPUTABLE")
            out[lab] = dict(COMPUTABLE=False)
            continue
        v = verdict_from_windows(wins, ng.DEFICIT_RATIO)
        S, m, sem, hi, lo = v["n_seeds"], v["ratio"], v["sem"], v["hi"], v["lo"]
        w, worst_seed = v["window"], v["worst_single_seed"]
        excluded, clear = v["SUSTAINED_DEFICIT"], v["ESTABLISHED_WITH_POWER"]
        print(f"{lab}:  windows = {len(wins)}, {w['n_cp']} checkpoints each, {S} seeds")
        print(f"   worst window [{w['t0']:.0f}, {w['t1']:.0f}] ps: "
              f"ratio = {m:.3f} +- {sem:.3f} (2-sigma band [{lo:.3f}, {hi:.3f}])")
        print(f"   worst single seed-window ratio = {worst_seed:.3f}")
        print(f"   -> {'DEFICIT (sustained, significant)' if excluded else ''}"
              f"{'ESTABLISHED at 2 sigma: a sustained 50 % deficit is EXCLUDED' if clear else ''}"
              f"{'INCONCLUSIVE: the 2-sigma band straddles 0.5 -- NOT powered' if not (excluded or clear) else ''}\n")
        out[lab] = dict(COMPUTABLE=True, n_windows=len(wins), n_seeds=S,
                        checkpoints_per_window=int(w["n_cp"]),
                        worst_window_ps=[w["t0"], w["t1"]], worst_window_ratio=m,
                        worst_window_sem=sem, two_sigma=[lo, hi],
                        worst_single_seed_window_ratio=v["worst_single_seed"],
                        threshold=ng.DEFICIT_RATIO,
                        SUSTAINED_DEFICIT=bool(excluded),
                        ESTABLISHED_WITH_POWER=bool(clear),
                        statistic="mean P/Q over a sliding window of the required deficit "
                                  "length; SEM from the spread across independent seeds")
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", args.out), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
