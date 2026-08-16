"""Figures for the NaCl screen stage: Gate C occupancy, the lambda ladder, and the two
instrument calibrations.

This is the panel set the report was missing. `plot_nacl_reference.py` covers the reference
stage (F/W, Gate 0 family spread, the external check, the RDFs); nothing covered the screen,
so the verdict-carrying evidence existed only as numbers in the text.

The occupancy ratio P/Q* is recomputed here with `nacl_gates`' own basin masks and the same
bias-aware target expression the gate evaluates, rather than a second implementation of it --
two implementations of one spec is exactly the defect class this study keeps finding.

Usage:
    python scripts/plot_nacl_gate_c.py --screen results/nacl/screen_all \\
        --ref results/nacl/reference --out report/figures
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
import numpy as np                                               # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nacl import system as nsys                                  # noqa: E402
import nacl_gates as G                                           # noqa: E402

#: The powered state -- the one the verdict rests on. CIP is non-binding in both cells and is
#: cleared separately by the windowed audit, so it is not drawn as if it carried the result.
POWERED = "SSIP"
CELL_COLOR = {64: "C0", 32: "C1"}


def occupancy_ratio(d, F_ref_on_grid, basins, beta, label):
    """P_k(t)/Q*_k(t) per checkpoint per seed, second-half mask, using the gate's own target.

    NOTE the denominator. `nacl_gates` uses two different ones for two different quantities,
    and they are easy to swap by accident:

        deficit test   P = counts[mask].sum() / counts.sum()      <- share of IN-DOMAIN walkers
        power block    lambda = Q* x diag_occ[0,0].sum()          <- expected count out of N

    Both are right for what they measure -- a share against a share, and an expected count
    against a count threshold -- but only the first is the ratio Gate C thresholds at 0.5, so
    it is the one plotted here. Using N instead shifts every ratio down by the out-of-domain
    fraction (about 1.3 % typical, ~3 % at the worst checkpoint) and would have put a figure
    on the page disagreeing with the report's own 0.866 / 0.832.
    """
    occ, pmf, times = d["diag_occupancy"], d["diag_pmf"], np.asarray(d["diag_times"], float)
    grid = d["grid"]
    msk = G.basin_masks(grid, basins)[label]
    n_cp, S, _ = occ.shape
    ratio = np.full((n_cp, S), np.nan)
    for c in range(n_cp):
        for s in range(S):
            B_t = pmf[c, s]
            w = np.exp(-beta * (F_ref_on_grid - B_t - (F_ref_on_grid - B_t).min()))
            q = float(w[msk].sum() / w.sum())
            counts = occ[c, s]
            p = float(counts[msk].sum() / max(counts.sum(), 1e-300))
            ratio[c, s] = p / q if q > 0 else np.nan
    T_ps = float(d["T_ns"]) * 1000.0
    return times, ratio, times >= 0.5 * T_ps, T_ps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", default="results/nacl/screen_all")
    ap.add_argument("--ref", default="results/nacl/reference")
    ap.add_argument("--out", default="report/figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rep = json.load(open(os.path.join(args.ref, "reference_report.json")))
    ref = np.load(os.path.join(args.ref, "reference.npz"))
    basins, beta = rep["basins"], nsys.beta_per_kJ()
    gates = json.load(open(os.path.join(args.screen, "gates_report.json")))

    fig, ax = plt.subplots(2, 2, figsize=(11.5, 8.4))

    # --- (a) the direct evidence: P/Q* on the powered state, every seed, both cells ----------
    a = ax[0, 0]
    mins = {}
    for path in sorted(glob.glob(os.path.join(args.screen, "cell_N*.npz"))):
        d = np.load(path)
        N = int(d["N"])
        F_ref_on_grid = np.interp(d["grid"], ref["r_nm"], ref["F_ref"])
        t, ratio, sh, T_ps = occupancy_ratio(d, F_ref_on_grid, basins, beta, POWERED)
        col = CELL_COLOR.get(N, "0.5")
        for s in range(ratio.shape[1]):
            a.plot(t / 1000.0, ratio[:, s], color=col, lw=0.7, alpha=0.55,
                   label=f"$N={N}$, {ratio.shape[1]} seeds" if s == 0 else None)
        a.axvspan(0.5 * T_ps / 1000.0, T_ps / 1000.0, color=col, alpha=0.05)
        mins[N] = float(np.nanmin(ratio[sh]))
    a.axhline(0.5, color="crimson", lw=1.4, ls="--")
    a.text(0.02, 0.52, "Gate C deficit threshold $0.5\\,Q^*$", color="crimson", fontsize=8.5,
           transform=a.get_yaxis_transform(), va="bottom")
    a.axhline(1.0, color="0.3", lw=0.8, ls=":")
    for N, v in sorted(mins.items()):
        a.axhline(v, color=CELL_COLOR.get(N, "0.5"), lw=1.0, ls="-.", alpha=0.9)
        a.text(0.995, v - 0.038, f"$N={N}$ worst second-half checkpoint: {v:.3f}",
               transform=a.get_yaxis_transform(), ha="right", va="center",
               fontsize=8, color=CELL_COLOR.get(N, "0.5"))
    a.set_xlabel("time (ns)")
    a.set_ylabel(r"occupancy / bias-aware target, $P/Q^*$")
    a.set_title(f"(a) {POWERED} -- the powered state that carries the verdict")
    a.set_ylim(0.3, 1.45)
    a.legend(frameon=False, fontsize=8.5, loc="lower right")

    # --- (b) the lambda ladder, and the two cells struck before they ran --------------------
    a = ax[0, 1]
    meas = {}
    for cell, res in gates.get("cells", {}).items():
        N = int(cell.lstrip("N"))
        for lab in ("CIP", POWERED):
            p = res.get("gate_C", {}).get(lab, {}).get("power")
            if p:
                meas.setdefault(lab, {})[N] = p["lambda_min_over_window"]
    # Projections for the struck cells are Q* x N arithmetic off the measured N = 64 cell, and
    # are drawn hollow because that is all they are -- no data exists at N = 8 or 16.
    proj = {lab: {n: v[64] / 64.0 * n for n in (8, 16)} for lab, v in meas.items() if 64 in v}
    for lab, mk, cl in ((POWERED, "o", "C0"), ("CIP", "s", "C3")):
        if lab not in meas:
            continue
        ns = sorted(meas[lab])
        a.plot(ns, [meas[lab][n] for n in ns], mk + "-", color=cl, ms=8, lw=1.6,
               label=f"{lab}, measured")
        pn = sorted(proj.get(lab, {}))
        if pn:
            a.plot(pn, [proj[lab][n] for n in pn], mk, color=cl, ms=8, mfc="none", ls=":",
                   lw=1.2, label=f"{lab}, projected (struck a priori)")
            a.plot([pn[-1], ns[0]], [proj[lab][pn[-1]], meas[lab][ns[0]]], ls=":", lw=1.2,
                   color=cl)
    a.axhline(G.LAMBDA_MIN, color="crimson", lw=1.4, ls="--")
    a.text(0.98, G.LAMBDA_MIN * 1.15, f"power floor $\\lambda \\geq {G.LAMBDA_MIN:.0f}$",
           color="crimson", fontsize=8.5, transform=a.get_yaxis_transform(), ha="right")
    a.axvspan(6.5, 24, color="0.5", alpha=0.13)
    a.text(11, 0.35, "struck a priori:\n$Q^*\\!\\leq\\!1$ so $\\lambda\\!<\\!N$,\n"
                     "unreachable at any sampling", fontsize=7.8, ha="center", color="0.25")
    a.set_xscale("log", base=2); a.set_yscale("log")
    a.set_xticks([8, 16, 32, 64]); a.set_xticklabels(["8", "16", "32", "64"])
    a.set_xlabel("$N$ (walkers sharing one bias; $N\\times T$ fixed at 100 ns)")
    a.set_ylabel(r"$\lambda = \min_t Q^*(t)\,N$  (expected walkers)")
    a.set_title("(b) why only two cells could be classified at all")
    a.legend(frameon=False, fontsize=7.8, loc="upper left")

    # --- (c) detection calibration: planted deficits in the REAL traces ----------------------
    a = ax[1, 0]
    for N, cl in CELL_COLOR.items():
        f = os.path.join(args.screen, f"gate_c_sensitivity_N{N}_{POWERED}.json")
        if not os.path.exists(f):
            continue
        s = json.load(open(f))
        lad = s["ladder"]
        x = [100 * r["planted_deficit"] for r in lad]
        y = [r["n_seeds_deficient"] for r in lad]
        dy = 0.11 if N == 64 else -0.11   # the two ladders coincide exactly; dodge to show both
        a.plot(x, np.asarray(y, float) + dy, "o-", color=cl, ms=5, lw=1.5,
               label=f"$N={N}$  (ladders coincide; offset to show both)")
        a.axvline(100 * s["analytic_detectable"], color=cl, ls=":", lw=1.2)
        a.text(100 * s["analytic_detectable"] - 0.7, 7.9,
               f"{100 * s['analytic_detectable']:.0f}%", color=cl, fontsize=7.8,
               rotation=90, va="top", ha="right")
    a.axvline(100 * G.DEFICIT_FRACTION * 2.5, color="crimson", lw=1.4, ls="--")
    a.text(51.4, 2.7, "the deficit the gate\nnominally tests for\n(fires on 0 of 8)",
           color="crimson", fontsize=8.2, va="center")
    a.text(0.025, 0.985,
           "dotted: what counting noise predicts the gate should\n"
           "resolve. It OVERSTATES the sensitivity about twofold --\n"
           "detection is set by the contiguity span, not by $\\lambda$.",
           transform=a.transAxes, fontsize=7.6, color="0.25", va="top")
    a.set_xlabel("planted stationary deficit (%)")
    a.set_ylabel("seeds on which Gate C fires (of 8)")
    a.set_title("(c) the gate is silent at the shortfall it tests for")
    a.set_ylim(-0.9, 9.6)
    a.legend(frameon=False, fontsize=7.6, loc="lower right")

    # --- (d) span frontier: the preregistered choice is dominated ----------------------------
    a = ax[1, 1]
    fr = json.load(open(os.path.join(args.screen, "gate_c_span_frontier.json")))
    lad = fr["ladder"]
    x = [r["span"] for r in lad]
    det = [r["detections"] for r in lad]
    fp = [r["false_positives"] for r in lad]
    n = lad[0]["n_state_seeds"]
    a.plot(x, det, "o-", color="C2", ms=6, lw=1.6, label=f"detections of a planted 50% deficit (of {n})")
    a.plot(x, fp, "s-", color="C3", ms=6, lw=1.6, label=f"false positives on the REAL traces (of {n})")
    pre = [r["span"] for r in lad if r["is_preregistered"]][0]
    a.axvline(pre, color="crimson", lw=1.4, ls="--")
    a.text(pre * 0.94, 0.62 * max(det), "preregistered\nspan (used for\nthe verdict)",
           color="crimson", fontsize=8.2, ha="right")
    a.axvline(fr["frontier_span"], color="0.25", lw=1.2, ls="-.")
    a.text(fr["frontier_span"] * 1.05, 0.52 * max(det),
           "efficient frontier:\nsame zero false positives,\n"
           f"{max(r['detections'] for r in lad if r['span']==fr['frontier_span'])} of {n} detected",
           color="0.25", fontsize=8.2, va="top")
    a.set_xscale("log")
    a.set_xticks(x); a.set_xticklabels([str(v) for v in x])
    a.set_xlabel("contiguity span (fraction of $T$)")
    a.set_ylabel("count over 32 state-seeds")
    a.set_title("(d) the preregistered span is dominated")
    a.legend(frameon=False, fontsize=7.4, loc="upper center")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(args.out, f"fig_nacl_02_gate_c.{ext}"), dpi=180)
    print(f"-> {args.out}/fig_nacl_02_gate_c.png / .pdf")
    print(f"   worst second-half P/Q* on {POWERED}: "
          + ", ".join(f"N={k}: {v:.4f}" for k, v in sorted(mins.items())))


if __name__ == "__main__":
    main()
