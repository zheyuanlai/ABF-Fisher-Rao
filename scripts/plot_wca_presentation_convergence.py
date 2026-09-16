#!/usr/bin/env python
"""Presentation figure: WCA dimer convergence, ABF vs ABF+FR (uniform-FR campaign).

One row, two panels: free-energy error e_F(t) (left) and mean-force error e_F'(t)
(right), median over 16 paired seeds with the interquartile band, log-y, FR onset marked.

    python scripts/plot_wca_presentation_convergence.py
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RAW = os.path.join(ROOT, "results/uniform_campaign/wca/uniform/raw")
OUT = os.path.join(ROOT, "results/uniform_campaign/wca/figures")

C_ABF = "#2a78d6"   # blue
C_FR = "#eb6834"    # orange
C_INK = "#0b0b0b"
C_INK2 = "#52514e"
C_GRID = "#e4e3df"
T_FR = 40.0         # fr_start_steps 20000 x dt 0.002


def load():
    runs = {}
    for path in sorted(glob.glob(os.path.join(RAW, "*.npz"))):
        with np.load(path, allow_pickle=True) as z:
            spec = json.loads(str(z["spec_json"]))
            runs[(spec["method"], spec["seed"])] = {
                k: np.asarray(z[k]) for k in ("l2_f_t", "l2_fp_t", "times")}
    seeds = sorted({s for (m, s) in runs if m == "abf" and ("fr_uniform", s) in runs})
    return runs, seeds


def med_iqr(stack):
    a = np.asarray(stack)
    return np.median(a, 0), np.percentile(a, 25, 0), np.percentile(a, 75, 0)


def main():
    runs, seeds = load()
    t = runs[("abf", seeds[0])]["times"]
    keep = t > 0  # t=0 is the untrained estimator, not a convergence point

    plt.rcParams.update({
        "font.size": 14, "axes.labelsize": 16, "axes.titlesize": 17,
        "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 14,
        "axes.edgecolor": C_INK2, "axes.linewidth": 0.8,
        "xtick.color": C_INK2, "ytick.color": C_INK2,
        "axes.labelcolor": C_INK, "text.color": C_INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6), constrained_layout=True)
    panels = [
        ("l2_f_t", r"free energy error  $\|\hat F_t - F\|_{L^2}$", "Free energy"),
        ("l2_fp_t", r"mean force error  $\|\hat F'_t - F'\|_{L^2}$", "Mean force"),
    ]
    arms = [("abf", C_ABF, "ABF"), ("fr_uniform", C_FR, "ABF + FR")]

    for ax, (key, ylab, title) in zip(axes, panels):
        ends = {}
        for method, c, lab in arms:
            md, lo, hi = med_iqr([runs[(method, s)][key] for s in seeds])
            ax.fill_between(t[keep], lo[keep], hi[keep], color=c, alpha=0.18, lw=0)
            ax.plot(t[keep], md[keep], color=c, lw=2.4, label=lab)
            ends[method] = (c, lab, md[-1])
        # direct labels at the curve ends, pushed apart if the ends are close on the log axis
        gap = np.log10(ends["abf"][2] / ends["fr_uniform"][2])
        min_gap = 0.16  # decades needed for two 13 pt labels not to touch
        shift = max(0.0, (min_gap - gap) / 2)
        pct = 100 * (ends["fr_uniform"][2] / ends["abf"][2] - 1)
        for method, sgn in (("abf", +1), ("fr_uniform", -1)):
            c, lab, y = ends[method]
            text = lab if method == "abf" else f"{lab}  {pct:+.0f}%"
            ax.annotate(text, xy=(t[-1], y * 10 ** (sgn * shift)), xytext=(6, 0),
                        textcoords="offset points", va="center", ha="left",
                        color=c, fontsize=13, fontweight="bold")
        ax.axvline(T_FR, color=C_INK2, lw=1.0, ls=":")
        ax.text(T_FR, 1.0, " FR on", transform=ax.get_xaxis_transform(),
                va="top", ha="left", color=C_INK2, fontsize=12)
        ax.set_yscale("log")
        ax.set_xlim(0, t[-1] * 1.30)
        ax.set_xticks(np.arange(0, t[-1] + 1, 50))
        ax.set_xlabel("simulation time  $t$")
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(True, axis="y", color=C_GRID, lw=0.7)
        ax.set_axisbelow(True)

    axes[0].legend(frameon=False, loc="upper right")

    os.makedirs(OUT, exist_ok=True)
    base = os.path.join(OUT, "fig_wca_presentation_convergence")
    fig.savefig(base + ".png", dpi=200)
    fig.savefig(base + ".pdf")
    print("saved", base + ".{png,pdf}")

    # numbers for the caption
    for key, name in (("l2_f_t", "F"), ("l2_fp_t", "F'")):
        a = np.median([runs[("abf", s)][key][-1] for s in seeds])
        u = np.median([runs[("fr_uniform", s)][key][-1] for s in seeds])
        print(f"final {name}: ABF {a:.4f}  ABF+FR {u:.4f}  ({100*(u/a-1):+.1f}%)")


if __name__ == "__main__":
    main()
