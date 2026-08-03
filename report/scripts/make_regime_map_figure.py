#!/usr/bin/env python
"""Render the unified regime map from the authoritative closure table.

The axes are deliberately **categorical**. The study supports a discovery/establishment
classification per benchmark; it does not support continuous adequacy scores, and inventing
coordinates to make a scatter plot would dress up a two-way classification as a measurement.
So each axis has two states and systems are laid out inside their cell.

Reads:  results/closure/v1_regime_map.csv   (written by scripts/build_closure_inventory.py)
Writes: report/figures/fig_regime_map.png
"""
from __future__ import annotations

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle            # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "results/closure/v1_regime_map.csv")
OUT = os.path.join(ROOT, "report/figures/fig_regime_map.png")

# Short labels for the figure; the full CV strings live in the table.
LABEL = {
    ("Butane", "phi1"): "butane $\\varphi_1$",
    ("Pentane", "phi1"): "pentane $\\varphi_1$",
    ("Pentane", "R15"): "pentane $R_{15}$",
    ("Pentane", "(phi1,phi2)"): "pentane $(\\varphi_1,\\varphi_2)$",
    ("Alanine dipeptide", "(phi,psi)"): "alanine $(\\varphi,\\psi)$",
    ("Valine dipeptide", "(phi,chi1)"): "valine $(\\varphi,\\chi_1)$",
    ("Metastability model", "x"): "metastability $x$",
    ("Entropic bottleneck", "x, beta<=4"): "bottleneck $x$, $\\beta\\leq 4$",
    ("Entropic bottleneck", "x, beta=8"): "bottleneck $x$, $\\beta=8$",
    ("Entropic gateway", "x"): "gateway $x$",
    ("WCA dimer", "bond coordinate"): "WCA dimer",
}
# Molecular systems are marked, because "does this transfer to a real molecule?" is the
# question a reader brings to this figure.
MOLECULAR = {"Butane", "Pentane", "Alanine dipeptide", "Valine dipeptide", "WCA dimer"}

CELL = {  # regime -> (column, row); column = discovery adequate?, row = establishment adequate?
    "discovery-limited": (0, 0),
    "establishment-limited": (1, 0),
    "ABF-sufficient": (1, 1),
}
FACE = {"discovery-limited": "#f2dede", "establishment-limited": "#d9ead3",
        "ABF-sufficient": "#e8eaf2"}
EDGE = {"discovery-limited": "#a94442", "establishment-limited": "#3d7a2a",
        "ABF-sufficient": "#5b6191"}


def main():
    with open(SRC, newline="") as fh:
        rows = list(csv.DictReader(fh))

    groups = {k: [] for k in CELL}
    for r in rows:
        reg = r["regime"]
        if reg not in groups:
            raise SystemExit(f"unmapped regime {reg!r}; the figure must cover every row")
        groups[reg].append((LABEL.get((r["system"], r["cv"]),
                                      f"{r['system']} {r['cv']}"),
                            r["system"] in MOLECULAR))

    fig, ax = plt.subplots(figsize=(9.2, 6.4))
    for reg, (cx, cy) in CELL.items():
        ax.add_patch(Rectangle((cx, cy), 1, 1, facecolor=FACE[reg],
                               edgecolor=EDGE[reg], linewidth=1.4, zorder=0))
    # The empty cell: adequate establishment cannot precede inadequate discovery.
    ax.add_patch(Rectangle((0, 1), 1, 1, facecolor="#f6f6f6", edgecolor="#bbbbbb",
                           linewidth=1.0, linestyle=(0, (4, 3)), zorder=0))
    ax.text(0.5, 1.5, "not reachable\n(a state cannot be well\nestablished before it\n"
                      "has been discovered)",
            ha="center", va="center", fontsize=9.5, color="#888888", style="italic")

    # The region mFR serves.
    ax.add_patch(FancyBboxPatch((1.035, 0.035), 0.93, 0.93,
                                boxstyle="round,pad=0.012,rounding_size=0.05",
                                facecolor="none", edgecolor=EDGE["establishment-limited"],
                                linewidth=2.6, zorder=3))

    titles = {"discovery-limited": "discovery-limited\nmFR cannot act",
              "establishment-limited": "establishment-limited\nmFR helps here",
              "ABF-sufficient": "ABF-sufficient\nmFR unnecessary or neutral"}
    for reg, (cx, cy) in CELL.items():
        ax.text(cx + 0.5, cy + 0.90, titles[reg], ha="center", va="top",
                fontsize=11.5, fontweight="bold", color=EDGE[reg], linespacing=1.35)
        items = groups[reg]
        top = cy + 0.66
        for i, (name, molecular) in enumerate(items):
            ax.text(cx + 0.5, top - i * 0.088,
                    ("$\\bullet$  " if molecular else "$\\circ$  ") + name,
                    ha="center", va="top", fontsize=10.2, zorder=4)

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["inadequate\n(zero-visit or too few\nindependent discoveries)",
                        "adequate\n(relevant states reached\nearly in the run)"], fontsize=10)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["inadequate\n(populated slowly\nrelative to the budget)",
                        "adequate\n(populated early)"], fontsize=10, rotation=90,
                       va="center", ha="center")
    ax.tick_params(axis="y", pad=26)
    ax.set_xlabel("Discovery adequacy within the run's budget", fontsize=12, labelpad=10)
    ax.set_ylabel("Post-discovery establishment adequacy", fontsize=12, labelpad=34)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.text(1.0, 2.06, "$\\bullet$ molecular system    $\\circ$ model potential",
            ha="center", va="bottom", fontsize=9.5, color="#555555")

    fig.tight_layout()
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"wrote {OUT}  ({sum(len(v) for v in groups.values())} benchmarks placed)")


if __name__ == "__main__":
    main()
