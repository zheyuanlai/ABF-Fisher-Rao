#!/usr/bin/env python
"""Information-conversion audit: the preregistered causal figures.

Reads only saved CSV/JSON outputs.  Skips gracefully any figure whose stage
did not run (e.g. a Stage-0D stop leaves only the stage-0 panels).
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.3, "figure.constrained_layout.use": True,
})

CELL_COLORS = {"K2": "tab:blue", "K3": "tab:orange"}


def savefig(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  wrote {name}")


def fig_stage0(root, stage, cells, out):
    """C_j, a_j V_j, pi*, and asymptotic-vs-finite-horizon opportunity."""
    ref = json.load(open(os.path.join(root, "reference",
                                      "reference_difficulty.json")))["cells"]
    fig, axes = plt.subplots(3, len(cells), figsize=(9, 7), sharex=True)
    for c, cell in enumerate(cells):
        df = pd.read_csv(os.path.join(root, stage, f"{cell}_stage0_cells.csv"))
        one = df[df.seed == df.seed.min()]
        j = one.j.values
        med = df.groupby("j").agg(C=("C", "median"), pi=("pi_star", "median"),
                                  occ=("occ0", "median")).reset_index()
        av = one.a.values * one.V.values
        axes[0, c].bar(j, med.C, color="0.6")
        axes[0, c].set_title(f"{cell}: raw counts $C_j$ at $t_b$ (median seed)")
        ax2 = axes[1, c]
        ax2.bar(j, av / max(av.max(), 1e-300), color=CELL_COLORS[cell],
                alpha=0.7)
        ax2.set_title(r"$a_j V_j$ (normalised)")
        ax3 = axes[2, c]
        ax3.bar(j, med.pi, color=CELL_COLORS[cell], alpha=0.7,
                label=r"$\pi^\star$ (median)")
        ax3.step(j, med.occ, where="mid", color="k", lw=1,
                 label="occupancy at $t_b$")
        ax3.axhline(1.0 / 256, color="r", ls=":", lw=1, label="1/K floor")
        ax3.set_title(r"$\pi^\star$ vs occupancy")
        ax3.set_xlabel("allocation cell $j$")
        if c == 0:
            ax3.legend(fontsize=7)
        _ = ref  # provenance loaded to fail loudly if missing
    savefig(fig, out, f"{stage}_stage0_target")

    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    for cell in cells:
        s0 = pd.read_csv(os.path.join(root, stage, f"{cell}_stage0.csv"))
        ax.scatter(1.0 - s0.R_asym_ratio, s0.G_ideal, s=22,
                   color=CELL_COLORS[cell], label=cell)
    ax.axhline(0.10, color="r", ls="--", lw=1, label="Stage-0D gate (0.10)")
    ax.set_xlabel("asymptotic oracle opportunity  $1 - R^{asym}_{opt}/R^{asym}_{unif}$")
    ax.set_ylabel("finite-horizon opportunity  $G_{ideal}$")
    ax.legend(fontsize=8)
    savefig(fig, out, f"{stage}_opportunity_asym_vs_finite")


def fig_frontier(root, stage, cells, out):
    runs = {}
    for cell in cells:
        p = os.path.join(root, stage, f"{cell}_runs.csv")
        if not os.path.exists(p):
            return
        runs[cell] = pd.read_csv(p)

    # dose frontier: target movement vs genealogy
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for cell in cells:
        df = runs[cell]
        fr = df[df.arm != "abf"].copy()
        fr["kl_ratio"] = fr.kl_post / fr.kl_pre
        g = fr.groupby("p90")
        m = g.agg(kl=("kl_ratio", "median"), ess=("ess_anc", "median"),
                  tvf=("tv_future", "median")).reset_index()
        abf_tvf = df[df.arm == "abf"].groupby("seed").tv_future.median().median()
        axes[0].plot(m.kl, m.ess, "o-", color=CELL_COLORS[cell], label=cell)
        for _, r in m.iterrows():
            axes[0].annotate(f"{r.p90:g}", (r.kl, r.ess), fontsize=7,
                             xytext=(3, 3), textcoords="offset points")
        axes[1].plot(m.p90, m.tvf / abf_tvf, "o-", color=CELL_COLORS[cell],
                     label=cell)
        # risk ratio with paired bootstrap
        rr, lo, hi = [], [], []
        for d, sub in fr.groupby("p90"):
            a = df[df.arm == "abf"].set_index("seed").sort_index()
            f = sub.set_index("seed").sort_index()
            x_, y_ = f.R_s.values, a.R_s.values
            rng = np.random.default_rng(0)
            idx = rng.integers(0, x_.size, size=(10_000, x_.size))
            boot = x_[idx].mean(1) / np.maximum(y_[idx].mean(1), 1e-300)
            rr.append(x_.mean() / y_.mean())
            lo.append(np.percentile(boot, 2.5))
            hi.append(np.percentile(boot, 97.5))
        d_ = sorted(fr.p90.unique())
        axes[2].errorbar(d_, rr,
                         yerr=[np.array(rr) - lo, np.array(hi) - rr],
                         fmt="o-", color=CELL_COLORS[cell], capsize=3,
                         label=cell)
    axes[0].axhline(0.90, color="r", ls="--", lw=1)
    axes[0].axvline(0.90, color="r", ls="--", lw=1)
    axes[0].set_xlabel(r"KL$_{post}$/KL$_{pre}$ (median)")
    axes[0].set_ylabel("ancestor ESS / K (median)")
    axes[0].set_title("dose frontier: movement vs genealogy")
    axes[1].axhline(0.90, color="r", ls="--", lw=1)
    axes[1].set_xlabel(r"$p_{90}$")
    axes[1].set_ylabel(r"TV($r_{future},\pi^\star$) ratio FR/ABF")
    axes[1].set_title("future allocation movement")
    axes[2].axhline(1.0, color="k", lw=0.8)
    axes[2].axhline(0.90, color="r", ls="--", lw=1)
    axes[2].set_xlabel(r"$p_{90}$")
    axes[2].set_ylabel(r"$\bar R_{FR}/\bar R_{ABF}$ (95% paired CI)")
    axes[2].set_title("realized information risk (PRIMARY)")
    for ax in axes:
        ax.legend(fontsize=7)
    savefig(fig, out, f"{stage}_dose_frontier")

    # per-cell predicted deficit vs realized improvement
    fig, axes = plt.subplots(1, len(cells), figsize=(8.6, 3.6))
    for c, cell in enumerate(cells):
        cr = pd.read_csv(os.path.join(root, stage, f"{cell}_cellruns.csv"))
        s0 = pd.read_csv(os.path.join(root, stage,
                                      f"{cell}_stage0_cells.csv"))
        C_by = s0.set_index(["seed", "j"]).C
        rows = []
        for (seed, j), grp in cr.groupby(["seed", "j"]):
            a = grp[grp.arm == "abf"]
            if a.empty or not (grp.a > 0).any():
                continue
            aj = float(grp.a.iloc[0])
            if aj <= 0:
                continue
            fref = float(grp.f_ref.iloc[0])
            e_abf = aj * (float(a.fhat.iloc[0]) - fref) ** 2
            pi_j = float(grp.pi_star.iloc[0])
            deficit = pi_j - float(C_by.loc[(seed, j)] / C_by.loc[seed].sum())
            for _, r in grp[grp.arm != "abf"].iterrows():
                e_fr = aj * (r.fhat - fref) ** 2
                rows.append(dict(j=j, seed=seed, arm=r.arm, deficit=deficit,
                                 improvement=e_abf - e_fr))
        if not rows:
            continue
        dd = pd.DataFrame(rows)
        m = dd.groupby("j").agg(deficit=("deficit", "median"),
                                imp=("improvement", "median"))
        ax = axes[c] if len(cells) > 1 else axes
        ax.scatter(m.deficit, m.imp, s=20, color=CELL_COLORS[cell])
        ax.axhline(0, color="k", lw=0.8)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel(r"predicted deficit  $\pi^\star_j - C_j/\sum C$")
        ax.set_ylabel(r"realized improvement  $a_j[(e^{ABF}_j)^2-(e^{FR}_j)^2]$")
        ax.set_title(cell)
    savefig(fig, out, f"{stage}_deficit_vs_improvement")


def fig_continuation(root, cells, out):
    p = os.path.join(root, "continuation")
    if not os.path.isdir(p):
        return
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "aic", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "analyze_info_conversion.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    thr = json.load(open("results/qr_decoupling/thresholds.json"))
    fig, axes = plt.subplots(1, len(cells), figsize=(9, 3.6), sharey=True)
    for c, cell in enumerate(cells):
        prof = mod.stitch_profiles(root, cell)
        ax = axes[c] if len(cells) > 1 else axes
        for arm, sub in prof.groupby("arm"):
            m = sub.groupby("t").e_F.median()
            ax.plot(m.index, m.values, label=arm,
                    color="k" if arm == "abf" else CELL_COLORS[cell])
        for name, ls in (("eps_1", ":"), ("eps_2", "--")):
            ax.axhline(thr[cell][name], color="r", ls=ls, lw=1)
        ax.set_yscale("log")
        ax.set_xlabel("t")
        ax.set_title(cell)
        ax.legend(fontsize=7)
    axes[0].set_ylabel(r"$e_F(t)$ (median)")
    savefig(fig, out, "continuation_eF_curves")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/information_conversion")
    ap.add_argument("--cells", nargs="+", default=["K2", "K3"])
    args = ap.parse_args()
    out = os.path.join(args.root, "figures")
    os.makedirs(out, exist_ok=True)
    for stage in ("pilot", "confirm"):
        if os.path.isdir(os.path.join(args.root, stage)):
            fig_stage0(args.root, stage, args.cells, out)
            fig_frontier(args.root, stage, args.cells, out)
    fig_continuation(args.root, args.cells, out)


if __name__ == "__main__":
    main()
