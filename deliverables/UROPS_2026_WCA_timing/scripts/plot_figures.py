#!/usr/bin/env python
"""Figures of the WCA Fisher-Rao timing study (from data/ only; run after analyze_campaign.py + export_figures.py).

  figures/main_threepanel.pdf           ABF vs FR from t = 0 (continuous) vs FR from t = 0 stopped at t = 20:
                                        e_F, e_F', e_xi (full domain); median + IQR; log y
  figures/timing_comparison.pdf         paired change vs ABF (median, 95 % CI) of I_F[0,40], I_F[20,80], e_F(240)
  figures/marginal_vs_free_energy.pdf   e_xi full domain, e_xi interior, edge-bin fraction, e_F
  figures/supp_all_arms.pdf             all seven arms, e_F / e_F' / e_xi medians
  figures/supp_genealogy.pdf            cumulative replacements and windowed ancestor ESS / K
Markers: t = 20 = end of the ABF warm-up AND reset of the reported estimator (production accumulators start; the
t = 20 save holds one step of deposits); FR stop markers where an arm stops.
"""
import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common_wca as CW  # noqa: E402
import export_figures as EX  # noqa: E402

FIG = os.path.join(CW.PKG, "figures")
STY = {"abf": ("#1f77b4", "-", "ABF"), "fr_start0": ("#ff7f0e", "--", "FR from t = 0 (continuous)"),
       "fr_start0_stop20": ("#2ca02c", "-.", "FR on 0 ≤ t < 20, then ABF"),
       "fr_start0_stop40": ("#17becf", (0, (5, 1, 1, 1)), "FR on 0 ≤ t < 40, then ABF"),
       "fr_start40": ("#d62728", ":", "FR from t = 40 (validated schedule)"),
       "fr_start5": ("#9467bd", (0, (3, 1)), "FR from t = 5"), "fr_start20": ("#8c564b", (0, (1, 1)), "FR from t = 20")}
YL = {"eF": r"$e_F(t)$ [kT]", "eFp": r"$e_{F'}(t)$ [kT per unit z]", "exi_full": r"$e_\xi(t)$, full domain [z$^{-1/2}$]",
      "exi_interior": r"$e_\xi(t)$, interior bins only [z$^{-1/2}$]", "edge_frac": "replicas in the two edge bins (fraction of K)"}
MD = {"CreationDate": None}


def band(ax, arm, key, lw=1.4, iqr=True):
    t, X, _ = EX.read(arm, key)
    q25, med, q75 = np.nanpercentile(X, [25, 50, 75], axis=1)
    c, ls, lab = STY[arm]
    if iqr:
        ax.fill_between(t, q25, q75, color=c, alpha=0.18, lw=0)
    ax.plot(t, med, color=c, ls=ls, lw=lw, label=lab)


def marks(ax, stop=(20.0,), start0=True):
    ax.axvline(20.0, color="0.45", lw=0.8, ls=":", zorder=0)
    ax.text(20.3, 0.97, "t = 20: ABF warm-up end +\nestimator reset", transform=ax.get_xaxis_transform(), fontsize=6.5,
            va="top", color="0.35")
    for s in stop:
        if s != 20.0:
            ax.axvline(s, color="#17becf", lw=0.8, ls="-.", zorder=0)


def fig_main(png):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))
    for ax, key in zip(axes, ("eF", "eFp", "exi_full")):
        for arm in ("abf", "fr_start0", "fr_start0_stop20"):
            band(ax, arm, key)
        ax.set_yscale("log")
        ax.set_xlim(0, 240)
        marks(ax)
        ax.axvline(20.0, color="#2ca02c", lw=0.9, ls="-.", alpha=0.7)
        ax.set_xlabel("t [reduced time units]")
        ax.set_ylabel(YL[key])
        ax.grid(alpha=0.25, which="both", lw=0.4)
    axes[0].set_title("(a) free energy", fontsize=10)
    axes[1].set_title("(b) mean force", fontsize=10)
    axes[2].set_title("(c) RC marginal (full domain)", fontsize=10)
    axes[0].legend(fontsize=7.5, loc="upper right")
    fig.suptitle("WCA dimer, validated dt = 1.25e-4, K = 256, 16 paired seeds (5600-5615): median and IQR; FR from t = 0 "
                 "(green stops at t = 20, dash-dot line)", fontsize=9.5)
    fig.tight_layout()
    save(fig, "main_threepanel", png)


def fig_timing(png):
    A = json.load(open(os.path.join(CW.PKG, "data", "summary", "analysis.json")))
    C = A["contrasts"]
    arms = [a for a in ("fr_start0", "fr_start5", "fr_start20", "fr_start40", "fr_start0_stop20", "fr_start0_stop40") if a in A["arms"]]
    metrics = (("I_F[0,40]", r"$I_F^{[0,40]}$"), ("I_F[20,80]", r"$I_F^{[20,80]}$"), ("e_F(240)", r"$e_F(240)$"))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9), sharey=True)
    y = np.arange(len(arms))[::-1]
    for ax, (m, lab) in zip(axes, metrics):
        for yi, arm in zip(y, arms):
            c = C[f"{arm} vs abf | {m}"]
            stopped = "stop" in arm
            col = "#2ca02c" if stopped else "#ff7f0e"
            ax.errorbar(100 * c["median"], yi, xerr=[[100 * (c["median"] - c["ci95"][0])], [100 * (c["ci95"][1] - c["median"])]],
                        fmt="s" if stopped else "o", color=col, ms=6, capsize=3, lw=1.3)
            ax.text(1.0, yi, f" {c['wins']}/{c['n']}", transform=ax.get_yaxis_transform(), fontsize=7, va="center")
        ax.axvline(0, color="0.3", lw=0.8)
        ax.set_xlabel(f"paired change vs ABF of {lab} [%]")
        ax.grid(alpha=0.25, lw=0.4)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([STY[a][2] for a in arms], fontsize=8)
    fig.suptitle("Median per-seed change vs ABF with 95 % bootstrap CI (16 paired seeds; right: seeds where FR is better); "
                 "circles = continuous FR, squares = early-stopped FR", fontsize=9)
    fig.tight_layout()
    save(fig, "timing_comparison", png)


def fig_marginal(png):
    keys = ("exi_full", "exi_interior", "edge_frac", "eF")
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.9))
    for ax, key in zip(axes, keys):
        for arm in ("abf", "fr_start0", "fr_start0_stop20", "fr_start40"):
            band(ax, arm, key)
        if key != "edge_frac":
            ax.set_yscale("log")
        ax.set_xlim(0, 240)
        marks(ax)
        ax.set_xlabel("t [reduced time units]")
        ax.set_ylabel(YL.get(key, key))
        ax.grid(alpha=0.25, which="both", lw=0.4)
    axes[0].legend(fontsize=7, loc="upper right")
    titles = ("(a) marginal error, full domain", "(b) marginal error, interior bins (diagnostic)",
              "(c) replicas held at the soft walls (edge bins)", "(d) free-energy error")
    for ax, ti in zip(axes, titles):
        ax.set_title(ti, fontsize=9.5)
    fig.suptitle("Marginal uniformity vs free-energy accuracy (median and IQR over 16 paired seeds)", fontsize=9.5)
    fig.tight_layout()
    save(fig, "marginal_vs_free_energy", png)


def fig_supp(png):
    arms = [a["name"] for a in CW.frozen()["arms"] if a["run"]]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))
    for ax, key in zip(axes, ("eF", "eFp", "exi_full")):
        for arm in arms:
            band(ax, arm, key, lw=1.1, iqr=False)
        ax.set_yscale("log")
        ax.set_xlim(0, 240)
        marks(ax, stop=(20.0, 40.0))
        ax.set_xlabel("t [reduced time units]")
        ax.set_ylabel(YL[key])
        ax.grid(alpha=0.25, which="both", lw=0.4)
    axes[0].legend(fontsize=6.5, loc="upper right")
    fig.suptitle("All seven arms (medians over 16 paired seeds)", fontsize=9.5)
    fig.tight_layout()
    save(fig, "supp_all_arms", png)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    for arm in arms:
        if arm == "abf":
            continue
        band(axes[0], arm, "repl_cum", lw=1.1, iqr=False)
        band(axes[1], arm, "ess_window", lw=1.1, iqr=False)
    axes[0].set_ylabel("cumulative FR replacements")
    axes[1].set_ylabel("windowed ancestor ESS / K (8 t.u. windows)")
    for ax in axes:
        ax.set_xlim(0, 240)
        ax.set_xlabel("t [reduced time units]")
        ax.grid(alpha=0.25, lw=0.4)
    axes[0].legend(fontsize=6.5)
    fig.suptitle("Genealogy diagnostics (medians); after an FR stop the windowed labels are no longer reset", fontsize=9.5)
    fig.tight_layout()
    save(fig, "supp_genealogy", png)


def save(fig, name, png):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, f"{name}.pdf"), metadata=MD)
    if png:
        os.makedirs(png, exist_ok=True)
        fig.savefig(os.path.join(png, f"{name}.png"), dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--png", default=None)
    a = ap.parse_args()
    fig_main(a.png)
    fig_timing(a.png)
    fig_marginal(a.png)
    fig_supp(a.png)
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
