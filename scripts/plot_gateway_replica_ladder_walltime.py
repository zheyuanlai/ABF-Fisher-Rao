#!/usr/bin/env python
"""Companion view of the gateway replica ladder: the SAME runs against physical time t (= wall-clock
on parallel hardware, where N replicas cost one replica's time) instead of the normalised budget.

  python scripts/plot_gateway_replica_ladder_walltime.py [--reference em|analytic]
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "results", "gateway_replica_ladder")
C_ABF, C_FR, C_INK, C_MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#8a8984"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default="em", choices=["em", "analytic"])
    a = ap.parse_args()
    sfx = "_emref" if a.reference == "em" else ""
    d = json.load(open(os.path.join(ROOT, "configs", "gateway_replica_ladder", "design.json")))
    B, dt = d["budget"]["B_walker_steps"], d["cell"]["dt"]
    z = np.load(os.path.join(OUT, f"per_run_curves{sfx}.npz"))
    runs = {}
    for k in z.files:
        arm, Nk, sk = k.rsplit("_", 2)
        runs.setdefault((arm, int(Nk[1:])), []).append(z[k])

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": C_MUTED,
                         "xtick.color": C_MUTED, "ytick.color": C_MUTED, "axes.titlesize": 9.5, "axes.titleweight": "bold",
                         "legend.frameon": False, "grid.color": "#e6e5e0", "grid.linewidth": 0.6})
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axs[0]
    show = [2048, 256, 32, 1]
    lw = {2048: 2.2, 256: 1.7, 32: 1.3, 1: 1.0}
    alpha = {2048: 1.0, 256: 0.85, 32: 0.7, 1: 0.6}
    for N in show:
        T = B // N * dt
        for arm, col, ls in (("abf_h180", C_ABF, "-"), ("fr_h180", C_FR, "-")):
            if (arm, N) not in runs:
                continue
            A = np.stack(runs[(arm, N)])          # (seeds, 3, n_saves): u, eF, eFp
            t = A[0, 0] * T
            ax.plot(t, np.median(A[:, 1], 0), color=col, lw=lw[N], alpha=alpha[N], ls=ls)
        ax.text(T * 1.05, np.median(np.stack(runs[("abf_h180", N)])[:, 1, -1]), f"N={N}", fontsize=7.5, color=C_INK, va="center")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(True)
    ax.set_xlabel("physical time per replica  t   (= wall-clock if replicas run in parallel)")
    ax.set_ylabel("e_F  (median of 32 seeds)")
    ax.set_title("(a) the same runs on the physical-time axis", loc="left")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=C_ABF, lw=2, label="ABF"), Line2D([], [], color=C_FR, lw=2, label="ABF + FR"),
                       Line2D([], [], color=C_MUTED, lw=2.2, label="thicker = more replicas")], loc="lower left", fontsize=7.5)

    # (b) time-to-accuracy: first t after which e_F stays below eps, eps = median ABF e_F(B) at N = 2048
    ax = axs[1]
    eps = float(np.median(np.stack(runs[("abf_h180", 2048)])[:, 1, -1]))
    Ns = sorted({N for (_, N) in runs}, reverse=True)
    for arm, col, lab in (("abf_h180", C_ABF, "ABF"), ("fr_h180", C_FR, "ABF + FR")):
        xs, med, lo, hi = [], [], [], []
        for N in Ns:
            if (arm, N) not in runs:
                continue
            T = B // N * dt
            te = []
            for r in runs[(arm, N)]:
                u, e = r[0], r[1]
                ok = e <= eps
                if not ok[-1]:
                    te.append(np.inf); continue
                bad = np.where(~ok)[0]
                te.append(T * (u[0] if len(bad) == 0 else u[bad[-1] + 1]))
            te = np.array(te)
            xs.append(N); med.append(np.median(te)); lo.append(np.percentile(te, 25)); hi.append(np.percentile(te, 75))
        xs, med, lo, hi = map(np.array, (xs, med, lo, hi))
        fin = np.isfinite(med)
        ax.plot(xs[fin], med[fin], color=col, marker="o", ms=4, label=lab)
        ax.fill_between(xs[fin], np.where(np.isfinite(lo[fin]), lo[fin], np.nan), np.where(np.isfinite(hi[fin]), hi[fin], np.nan),
                        color=col, alpha=0.15, lw=0)
    ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.grid(True)
    ax.set_xticks([1, 4, 16, 64, 256, 1024]); ax.set_xticklabels(["1", "4", "16", "64", "256", "1k"])
    ax.set_xlabel("replicas N")
    ax.set_ylabel(f"physical time to reach e_F <= {eps:.4f}\n(persistently; median, IQR)")
    ax.set_title("(b) time-to-accuracy: FR shortens it most at large N", loc="left")
    ax.legend(fontsize=8)
    ref = "dt-consistent reference (post-hoc)" if a.reference == "em" else "analytic reference (preregistered)"
    fig.suptitle(f"Entropic gateway replica ladder on the physical-time axis (180 bins; {ref}). ABF at N = 2048 never reaches eps within T = 40.",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, "figures", f"fig_ladder_physical_time{sfx}.{ext}"), dpi=170)
    print("eps", eps)


if __name__ == "__main__":
    main()
