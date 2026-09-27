#!/usr/bin/env python
"""POST-HOC diagnostic for the entropic-bottleneck histogram campaign (not a preregistered endpoint).

Question raised by the A1 calibration: the kernel ABF (h = 0.07) ends at e_F(T) = 0.21 with the whole
error in the outer 0.1 of the evaluation window (a frontier stall at the steep walls), while the
histogram ABF converges to its discretisation floor.  Is the stall a property of the kernel estimator
at any online bandwidth?  Reads the ABF-only batches of the calibration (kernel h = 0.07, histogram
ladder) and the diag_kernel_h stage (kernel h = 0.14, 0.035, 0.0175), same 8 seeds and noise, and
tabulates: final e_F on the full window and on the interior |x| <= 1.3, the F' error at the window
edge nodes x = +-1.5, the final KDE walker density at x = +-1.5, and the maximum applied bias force.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(SCRIPTS, "..")
sys.path.insert(0, SCRIPTS)
from analyze_histogram_abf_entropic import load_batch, by_method, med_iqr   # noqa: E402

RES = os.path.join(ROOT, "results", "histogram_abf", "entropic_bottleneck")


def stats(rows):
    seeds = sorted(rows)
    x = np.asarray(rows[seeds[0]]["x_grid"], dtype=float)
    mi = (x >= -1.3) & (x <= 1.3)
    i_lo, i_hi = int(np.argmin(np.abs(x + 1.5))), int(np.argmin(np.abs(x - 1.5)))
    eF, eFi, dFp_lo, dFp_hi, p_lo, p_hi, mb = [], [], [], [], [], [], []
    for s in seeds:
        r = rows[s]
        d = np.asarray(r["F_hat"]) - np.asarray(r["F_ref"])
        eF.append(float(r["final_l2_f"]))
        di = d[mi] - d[mi].mean(); eFi.append(float(np.sqrt(np.mean(di * di))))
        if "Fp_bins" in r:        # histogram: P0 value of the bin containing the edge node
            edges = np.asarray(r["hist_edges"]); G = np.asarray(r["Fp_bins"]); n = len(G)
            j = lambda xx: int(np.clip(np.floor((xx + 1.8) / (edges[1] - edges[0]) + 1e-9), 0, n - 1))
            dFp_lo.append(float(G[j(-1.5)] - r["Fp_ref"][i_lo])); dFp_hi.append(float(G[j(1.5)] - r["Fp_ref"][i_hi]))
        else:
            dFp_lo.append(float(r["Fp_hat"][i_lo] - r["Fp_ref"][i_lo])); dFp_hi.append(float(r["Fp_hat"][i_hi] - r["Fp_ref"][i_hi]))
        p_lo.append(float(r["p_hat"][i_lo])); p_hi.append(float(r["p_hat"][i_hi])); mb.append(float(r["max_abs_bias_force"]))
    return dict(n=len(seeds), final_l2_f=med_iqr(eF), final_l2_f_interior=med_iqr(eFi), dFp_at_minus1p5=med_iqr(dFp_lo),
                dFp_at_plus1p5=med_iqr(dFp_hi), p_hat_at_minus1p5=med_iqr(p_lo), p_hat_at_plus1p5=med_iqr(p_hi), max_abs_bias=med_iqr(mb))


def main():
    arms = []
    for f in sorted(glob.glob(os.path.join(RES, "diagnostics", "kernel_h", "raw", "kernel_h*.npz"))) + [os.path.join(RES, "calibration", "raw", "kernel.npz")]:
        rows, meta = load_batch(f)
        arms.append((f"kernel h={meta['config']['h']:g}", by_method(rows)["abf"], meta["config"]["h"]))
    arms.sort(key=lambda a: -a[2])
    for nb in (45, 60, 90, 180, 360):
        rows, meta = load_batch(os.path.join(RES, "calibration", "raw", f"hist_nbins{nb}.npz"))
        arms.append((f"histogram Delta={3.6 / nb:.3f}", by_method(rows)["abf"], None))
    out = {}
    print("POST-HOC diagnostic: ABF only, 8 calibration seeds, identical noise (medians)")
    print(f"  {'estimator':>22} {'e_F(T)':>8} {'e_F int':>8} {'F\' err@-1.5':>11} {'F\' err@+1.5':>11} {'p(-1.5)':>8} {'p(+1.5)':>8} {'max|bias|':>9}")
    for name, rows, _ in arms:
        s = stats(rows); out[name] = s
        print(f"  {name:>22} {s['final_l2_f']['median']:8.4f} {s['final_l2_f_interior']['median']:8.4f} {s['dFp_at_minus1p5']['median']:+11.2f} "
              f"{s['dFp_at_plus1p5']['median']:+11.2f} {s['p_hat_at_minus1p5']['median']:8.3f} {s['p_hat_at_plus1p5']['median']:8.3f} {s['max_abs_bias']['median']:9.2f}")
    os.makedirs(os.path.join(RES, "diagnostics", "kernel_h"), exist_ok=True)
    json.dump(dict(post_hoc=True, note=__doc__, arms=out), open(os.path.join(RES, "diagnostics", "kernel_h", "summary.json"), "w"), indent=2, default=float)
    print(f"  wrote {os.path.relpath(os.path.join(RES, 'diagnostics', 'kernel_h', 'summary.json'), ROOT)}")


if __name__ == "__main__":
    main()
