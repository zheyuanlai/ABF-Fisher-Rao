"""Within-basin occupancy audit for NaCl: are the basins sampled inside, or only right on
aggregate?

Gate C is basin-integrated and the outer basin spans 88 % of the domain, so an internal
redistribution that preserved the integral would be invisible to it. This splits each basin into
quarters and asks the same question one level down.

The suppressed bar is the point of the left-hand pair as much as the drawn ones. CIP's innermost
quarter carries 0.0004 % of the basin target, and the first version of this audit reported a
ratio of 1274 there -- 0.0175 % of walkers over a target a thousand times smaller. A ratio needs
both arguments' populations. The tool now refuses the ratio where the target has none, and the
figure draws that refusal rather than quietly omitting the quarter.

Usage:
    python scripts/plot_nacl_within_basin.py --screen results/nacl/screen_all \\
        --out report/figures
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
import numpy as np                                               # noqa: E402

CELL_COLOR = {64: "C0", 32: "C1"}
DEFICIT_RATIO = 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", default="results/nacl/screen_all")
    ap.add_argument("--out", default="report/figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    audits = {}
    for N in (64, 32):
        p = os.path.join(args.screen, f"within_basin_audit_N{N}.json")
        if os.path.exists(p):
            audits[N] = json.load(open(p))
    if not audits:
        raise SystemExit("no within_basin_audit_N*.json found")

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))

    for a, state in zip(ax, ("SSIP", "CIP")):
        ref = audits[max(audits)][state]
        nq = len(ref["quarters"])
        idx = np.arange(nq)
        width = 0.36
        for j, (N, aud) in enumerate(sorted(audits.items(), reverse=True)):
            q = aud[state]["quarters"]
            vals = [x["ratio"] if x["ratio_meaningful"] else np.nan for x in q]
            pos = idx + (j - 0.5) * width
            a.bar(pos, vals, width, color=CELL_COLOR.get(N, "0.5"), alpha=0.85,
                  label=f"$N={N}$")
            for k, x in enumerate(q):
                if not x["ratio_meaningful"]:
                    # Draw the REFUSAL, not a gap: an empty box where a bar would be.
                    a.bar(pos[k], 1.0, width, facecolor="none", edgecolor="0.55",
                          hatch="///", lw=0.9)
        a.axhline(1.0, color="0.25", lw=1.0, ls=":")
        a.axhline(DEFICIT_RATIO, color="crimson", lw=1.4, ls="--")
        a.text(nq - 0.45, DEFICIT_RATIO + 0.03, "Gate C deficit threshold",
               color="crimson", fontsize=8, ha="right")
        lab = []
        for x in ref["quarters"]:
            lo, hi = x["r_nm"]
            lab.append(f"{lo:.2f}\u2013\n{hi:.2f}")
        a.set_xticks(idx); a.set_xticklabels(lab, fontsize=8)
        a.set_xlabel("quarter of the basin ($r$, nm)")
        a.set_ylabel("occupancy share / target share")
        a.set_title(state, fontsize=11)
        it = "\n".join(
            f"$N={N}$:  integrated {audits[N][state]['integrated_ratio']:.3f},  "
            f"shape TV {audits[N][state]['shape_TV']:.3f}"
            for N in sorted(audits, reverse=True))
        a.text(0.985, 0.985, it, transform=a.transAxes, fontsize=8.2, color="0.2",
               ha="right", va="top")
        a.set_ylim(0, max(2.1, np.nanmax([x["ratio"] or 0 for N in audits
                                          for x in audits[N][state]["quarters"]]) * 1.35))
        a.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=2)

    ax[0].text(0.5, 0.80,
               "gentle monotone gradient, no wall accumulation:\n"
               "the verdict-carrying basin is not a coarse-graining artifact",
               transform=ax[0].transAxes, fontsize=8, color="0.25", ha="center", va="top")
    ax[1].text(0.335, 0.665,
               "hatched: target carries $0.0004\\,\\%$ of the basin.\n"
               "A ratio needs both arguments' populations, so it is\n"
               "refused here rather than reported (it once read $1274$).",
               transform=ax[1].transAxes, fontsize=8, color="0.25", ha="left", va="top")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(args.out, f"fig_nacl_03_within_basin.{ext}"), dpi=180)
    print(f"-> {args.out}/fig_nacl_03_within_basin.png / .pdf")
    for N in sorted(audits, reverse=True):
        for st in ("SSIP", "CIP"):
            q = [x["ratio"] for x in audits[N][st]["quarters"] if x["ratio_meaningful"]]
            print(f"   N={N} {st}: quarters {min(q):.3f}-{max(q):.3f}, "
                  f"integrated {audits[N][st]['integrated_ratio']:.4f}, "
                  f"shape TV {audits[N][st]['shape_TV']:.4f}")


if __name__ == "__main__":
    main()
