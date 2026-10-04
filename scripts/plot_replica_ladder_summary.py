#!/usr/bin/env python
"""Two-system summary of the equal-budget replica ladders: FR effect vs N, and the best (N, T) split."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
C_ABF, C_FR, C_INK, C_MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#8a8984"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": C_MUTED,
                     "xtick.color": C_MUTED, "ytick.color": C_MUTED, "axes.titlesize": 9.5, "axes.titleweight": "bold",
                     "legend.frameon": False, "grid.color": "#e6e5e0"})
gw = json.load(open(os.path.join(ROOT, "results/gateway_replica_ladder/summary_emref.json")))["per_width"]["180"]["rows"]
wc = json.load(open(os.path.join(ROOT, "results/wca_replica_ladder/summary_dt0.0005_dynref.json")))["rows"]
fig, axs = plt.subplots(2, 2, figsize=(11, 7.4))
for col, (rows, name, nseed) in enumerate(((gw, "Entropic gateway (2-D), B = 2.048e8", 32), (wc, "WCA dimer (100 particles, dt 0.0005), B = 4.9e8", 16))):
    ax = axs[0, col]
    rr = [r for r in rows if "effect" in r]
    x = np.array([r["N"] for r in rr], float)
    for key, c, mk, lab, off in (("dIbar", C_INK, "o", "integrated error (1/B) ∫ e_F db", 0.94), ("dfin", C_MUTED, "D", "final error e_F(B)", 1.06)):
        v = np.array([r["effect"][key] for r in rr])
        ax.errorbar(x * off, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt=mk, color=c, ms=4.5, capsize=2, lw=1, label=lab)
        ax.plot(x * off, v[:, 0], color=c, lw=1, alpha=0.6)
    ax.axhline(0, color=C_MUTED, lw=0.8)
    ax.set_xscale("log", base=2); ax.grid(True, axis="y")
    if col == 1:
        ax.set_yscale("symlog", linthresh=50)
    ax.set_xticks([2, 4, 16, 64, 256, 1024, 2048]); ax.set_xticklabels(["2", "4", "16", "64", "256", "1k", "2k"])
    ax.set_xlabel("replicas N  (n_steps = B / N)"); ax.set_ylabel("ABF+FR vs ABF, paired median change (%)")
    ax.set_title(f"{name}\nFR effect vs N ({nseed} seeds, 95 % CI)", loc="left")
    ax.text(0.98, 0.04, "below 0 = FR helps", transform=ax.transAxes, ha="right", fontsize=7.5, color=C_MUTED)
    ax.legend(fontsize=7.5, loc="upper left" if col == 1 else "lower left")
    ax = axs[1, col]
    for m, c, lab in (("abf", C_ABF, "ABF"), ("fr", C_FR, "ABF + FR")):
        sel = [r for r in rows if m in r]
        xx = np.array([r["N"] for r in sel], float)
        for key, ls, mk, kind in (("Ibar", "-", "o", "integrated"), ("fin", (0, (4, 2)), "^", "final")):
            v = np.array([r[m][key] for r in sel])
            ax.plot(xx, v[:, 0], color=c, ls=ls, marker=mk, ms=4, label=f"{lab}, {kind}")
            ax.fill_between(xx, v[:, 1], v[:, 2], color=c, alpha=0.12, lw=0)
    ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.grid(True, which="major")
    ax.set_xticks([1, 4, 16, 64, 256, 1024, 2048]); ax.set_xticklabels(["1", "4", "16", "64", "256", "1k", "2k"])
    ax.set_xlabel("replicas N"); ax.set_ylabel("e_F at equal force-evaluation budget")
    ax.set_title("absolute error: which (N, T) split of the budget is best", loc="left")
    ax.legend(fontsize=7, loc="upper left")
fig.suptitle("Equal force-evaluation budget (N replicas x n_steps fixed): Fisher-Rao helps only with many replicas (gateway) or not at all (WCA),\n"
             "and few-replica / serial ABF is the best use of the budget in both systems.  e_F vs dt-consistent references.",
             fontsize=9.5, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.93))
os.makedirs(os.path.join(ROOT, "results/replica_ladder_summary"), exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(ROOT, f"results/replica_ladder_summary/fig_replica_ladder_two_systems.{ext}"), dpi=170)
