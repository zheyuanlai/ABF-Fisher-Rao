#!/usr/bin/env python
"""Presentation figure: does a SHORTER ABF burn-in before FR help?

The FR-start timing experiment (docs/FR_START_TIMING.md, frozen 2026-09-04) asks whether the
alanine and pentane-R15 nulls were artefacts of starting FR too late, as the LTA sweep suggested
when moving FR from 40 000 to 20 000 steps turned -0.21% into -14.8%.  It sweeps the FR start
time at two doses on three cells, 16 paired seeds each.

One panel per cell: the preregistered primary contrast, paired median Delta I_F against a fresh
ABF baseline on the common window, versus the FR start time.  Positive is WORSE than ABF.  The
end of the ABF warm-up -- the shortest burn-in the LTA rule would pick -- is marked.  Arms that
break the preregistered genealogy floors are drawn hollow and are not a usable configuration.

Numbers are read from the frozen analysis summaries (results/fr_start_timing/analysis/*_summary.json)
and six headline entries are asserted against RESULTS.md.  The sibling *_arms.csv files are NOT used:
their window fields are written unquoted as "[5,100]", so the embedded comma shifts every column after
W1_window by three, and `floors_ok` read by name there returns an event fraction.

    python scripts/plot_fr_start_timing_presentation.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ANA = os.path.join(ROOT, "results/fr_start_timing/analysis")
OUT = os.path.join(ANA, "figures")

C_LOW = "#2a78d6"    # frozen (gentle) rate
C_HIGH = "#eb6834"   # raised dose
C_INK = "#0b0b0b"
C_INK2 = "#52514e"
C_GRID = "#e4e3df"

# (title, x label, warm-up end, x of the closed arm, rows-source, dose keys)
PANELS = [
    dict(cell=None, title="Alanine dipeptide", xlabel="FR start  (ps)", warmup=5.0, closed=20.0,
         src="alanine", xkey="fr_start_ps", doses=(0.02, 0.15), noise=2.0),
    dict(cell="b1.4", title=r"Pentane R15,  $\beta = 1.4$", xlabel="FR start  (t.u.)", warmup=2.5,
         closed=6.0, src="r15", xkey="fr_start_tu", doses=(0.02, 0.10), noise=None),
    dict(cell="b1.6", title=r"Pentane R15,  $\beta = 1.6$", xlabel="FR start  (t.u.)", warmup=2.5,
         closed=6.0, src="r15", xkey="fr_start_tu", doses=(0.02, 0.10), noise=None),
]


def read(name):
    """arms of one system's frozen summary, as a list of dicts (typed, unlike the CSV)."""
    d = json.load(open(os.path.join(ANA, f"{name}_summary.json")))
    return [dict(v, key=k) for k, v in d["arms"].items()]


def series(rows, panel, rate):
    """(x, median%, lo%, hi%, floors_ok) for the swept FR arms at one dose, sorted by start."""
    out = []
    for r in rows:
        if panel["cell"] is not None and r.get("cell") != panel["cell"]:
            continue
        if r["method"] != "fr_uniform" or r["arm"].startswith("campaign"):
            continue
        if abs(float(r["fr_rate"]) - rate) > 1e-12:
            continue
        out.append((float(r[panel["xkey"]]),
                    100 * float(r["dIF_W1_median"]),
                    100 * float(r["dIF_W1_lo"]),
                    100 * float(r["dIF_W1_hi"]),
                    bool(r["floors_ok"])))
    out.sort(key=lambda v: v[0])
    return out


def main():
    data = {"alanine": read("alanine"), "r15": read("r15")}

    # ---- assert the headline entries of RESULTS.md ----
    def find(src, cell, rate, x, xkey):
        for r in data[src]:
            if cell is not None and r.get("cell") != cell:
                continue
            if r["method"] != "fr_uniform" or r["arm"].startswith("campaign"):
                continue
            if abs(float(r["fr_rate"]) - rate) < 1e-12 and abs(float(r[xkey]) - x) < 1e-9:
                return 100 * float(r["dIF_W1_median"])
        raise AssertionError(f"arm not found: {src} {cell} rate {rate} at {x}")
    headline = [("alanine", None, 0.02, 5.0, "fr_start_ps", +2.12),
                ("alanine", None, 0.15, 5.0, "fr_start_ps", +10.53),
                ("alanine", None, 0.02, 20.0, "fr_start_ps", -0.13),
                ("alanine", None, 0.15, 20.0, "fr_start_ps", +1.59),
                ("r15", "b1.4", 0.02, 2.5, "fr_start_tu", +0.58),
                ("r15", "b1.6", 0.10, 2.5, "fr_start_tu", +9.50)]
    for src, cell, rate, x, xkey, want in headline:
        got = find(src, cell, rate, x, xkey)
        assert abs(got - want) < 0.01, f"{src} {cell} rate {rate} at {x}: {got:+.2f} vs RESULTS.md {want:+.2f}"
    print(f"self-check: {len(headline)} headline entries match RESULTS.md")

    plt.rcParams.update({
        "font.size": 14, "axes.labelsize": 16, "axes.titlesize": 17,
        "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13,
        "axes.edgecolor": C_INK2, "axes.linewidth": 0.8,
        "xtick.color": C_INK2, "ytick.color": C_INK2,
        "axes.labelcolor": C_INK, "text.color": C_INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.0), constrained_layout=True)
    for ax, panel in zip(axes, PANELS):
        rows = data[panel["src"]]
        if panel["noise"] is not None:
            ax.axhspan(-panel["noise"], panel["noise"], color=C_INK2, alpha=0.08, lw=0)
            ax.text(0.98, 0.02, "process-to-process noise floor", transform=ax.transAxes,
                    ha="right", va="bottom", color=C_INK2, fontsize=11)
        ax.axhline(0.0, color=C_INK, lw=1.2)
        ax.axvline(panel["warmup"], color=C_INK2, lw=1.0, ls=":")
        ax.text(panel["warmup"], 0.02, " warm-up ends", transform=ax.get_xaxis_transform(),
                va="bottom", ha="left", color=C_INK2, fontsize=11)
        for rate, colour in zip(panel["doses"], (C_LOW, C_HIGH)):
            sr = series(rows, panel, rate)
            if not sr:
                continue
            x = np.array([v[0] for v in sr]); y = np.array([v[1] for v in sr])
            lo = np.array([v[2] for v in sr]); hi = np.array([v[3] for v in sr])
            ok = np.array([v[4] for v in sr])
            ax.plot(x, y, color=colour, lw=2.2, zorder=3)
            ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="none", ecolor=colour, elinewidth=1.6,
                        capsize=4, zorder=4)
            ax.scatter(x[ok], y[ok], s=90, color=colour, zorder=5, edgecolor="white", linewidth=1.5)
            if (~ok).any():
                ax.scatter(x[~ok], y[~ok], s=90, facecolor="white", edgecolor=colour,
                           linewidth=2.2, zorder=5)
            ax.annotate(f"rate {rate:g}", xy=(x[-1], y[-1]), xytext=(9, 0),
                        textcoords="offset points", va="center", ha="left",
                        color=colour, fontsize=12, fontweight="bold", annotation_clip=False,
                        zorder=6, bbox=dict(facecolor="white", edgecolor="none", pad=1.0, alpha=0.9))
        xs = [v for rate in panel["doses"] for v in [p_[0] for p_ in series(rows, panel, rate)]]
        ax.set_xlim(min(xs) - 0.06 * (max(xs) - min(xs)), max(xs) + 0.34 * (max(xs) - min(xs)))
        ax.set_xlabel(panel["xlabel"])
        ax.set_title(panel["title"], loc="left", fontweight="bold")
        ax.grid(True, axis="y", color=C_GRID, lw=0.7)
        ax.set_axisbelow(True)
    axes[0].set_ylabel(r"$\Delta I_F$ vs ABF  (%)")
    axes[0].annotate("worse than ABF", xy=(0.98, 0.97), xycoords="axes fraction",
                     ha="right", va="top", color=C_INK2, fontsize=12)
    handles = [Line2D([], [], color=C_LOW, lw=2.2, marker="o", markersize=9,
                      markeredgecolor="white", markeredgewidth=1.5, label="frozen FR rate"),
               Line2D([], [], color=C_HIGH, lw=2.2, marker="o", markersize=9,
                      markeredgecolor="white", markeredgewidth=1.5, label="raised FR dose"),
               Line2D([], [], color=C_INK2, lw=0, marker="o", markersize=9, markerfacecolor="white",
                      markeredgecolor=C_INK2, markeredgewidth=2.2, label="genealogy floor broken")]
    fig.legend(handles=handles, loc="outside upper center", ncol=3, frameon=False)

    os.makedirs(OUT, exist_ok=True)
    base = os.path.join(OUT, "fig_fr_start_timing_presentation")
    fig.savefig(base + ".png", dpi=200)
    fig.savefig(base + ".pdf")
    print("saved", base + ".{png,pdf}")
    for panel in PANELS:
        for rate in panel["doses"]:
            s = series(data[panel["src"]], panel, rate)
            if s:
                pretty = "  ".join(f"{v[0]:g}:{v[1]:+.2f}%{'' if v[4] else '*'}" for v in s)
                print(f"  {panel['title']:<28} rate {rate:<5g} {pretty}")
    print("  * genealogy floor broken")


if __name__ == "__main__":
    main()
