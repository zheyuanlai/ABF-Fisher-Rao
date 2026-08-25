#!/usr/bin/env python
"""Figures for the v3 pilot, with every caption derived from the data.

Frozen protocol: docs/V3_PREREGISTRATION.md -- "Figures are generated from saved
CSVs with data-derived captions (no literals)."  The v2 analysis violated that
with hardcoded "0/54" and hardcoded schedule parameters that could disagree with
the panel beside them; nothing here may contain a literal that the data could
contradict.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from publication_style import (add_panel_labels, apply_publication_style,  # noqa: E402
                              save_figure, style_for_series)


def _load(root: pathlib.Path) -> pd.DataFrame:
    return pd.read_csv(root / "production_gpu" / "production_gpu_runs_long.csv")


def _median_iqr(df: pd.DataFrame, col: str):
    g = df.groupby("step")[col]
    return g.median(), g.quantile(0.25), g.quantile(0.75)


def fig_convergence(frames, eps, out, arms):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for k, (col, key1, key2, name) in enumerate(
            [("l2_F_R12", "eps_F_1", "eps_F_2", r"$e_F$"),
             ("l2_Fprime_R12", "eps_Fprime_1", "eps_Fprime_2", r"$e_{F'}$")]):
        ax = axes[k]
        for i, arm in enumerate(arms):
            if arm not in frames:
                continue
            med, lo, hi = _median_iqr(frames[arm], col)
            color, ls, _ = style_for_series(i)
            ax.plot(med.index, med.values, color=color, ls=ls, label=arm, lw=1.4)
            ax.fill_between(med.index, lo.values, hi.values, color=color, alpha=0.13,
                            linewidth=0)
        for key, style in ((key1, ":"), (key2, "--")):
            ax.axhline(eps[key], color="0.35", ls=style, lw=0.9)
            ax.annotate(f"{key} = {eps[key]:.4f}", xy=(med.index[-1], eps[key]),
                        xytext=(-4, 3), textcoords="offset points",
                        ha="right", fontsize=7, color="0.35")
        ax.set_yscale("log"); ax.set_xlabel("ABF iterations")
        ax.set_ylabel(f"{name} on scope R12")
    axes[0].legend(fontsize=7, frameon=False, loc="upper right")
    n_seeds = frames[arms[0]]["seed"].nunique()
    fig.suptitle(f"Median and IQR over {n_seeds} matched seeds; thresholds frozen "
                 f"from plain ABF alone", fontsize=8, y=1.02)
    add_panel_labels(axes)
    return save_figure(fig, out / "fig_v3_convergence", tight=True)


def fig_mechanism(frames, retention, out, arms):
    """Dose, ancestry and per-opportunity retention -- the P3 mechanism panel."""
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.5))
    for i, arm in enumerate(arms):
        if arm not in frames:
            continue
        df = frames[arm]
        color, ls, _ = style_for_series(i)
        dose = []
        for _, g in df.groupby("seed"):
            g = g.sort_values("step")
            dose.append(np.diff(g["cumulative_replacements"].to_numpy()))
        dose = np.median(np.vstack(dose), axis=0)
        steps = np.sort(df["step"].unique())[1:]
        axes[0].plot(steps, dose, color=color, ls=ls, lw=1.3, label=arm)
        med, lo, hi = _median_iqr(df, "ancestor_ess")
        n_part = int(df["n_unique_ancestors"].max())
        axes[1].plot(med.index, med.values / n_part, color=color, ls=ls, lw=1.3)
        axes[1].fill_between(med.index, lo.values / n_part, hi.values / n_part,
                             color=color, alpha=0.13, linewidth=0)
        if arm in retention:
            g = np.asarray(retention[arm])
            axes[2].plot(np.arange(1, len(g) + 1), g, color=color, ls=ls, lw=1.3)
    axes[0].set_xlabel("ABF iterations"); axes[0].set_ylabel("replacements per FR opportunity")
    axes[1].set_xlabel("ABF iterations"); axes[1].set_ylabel(r"ancestral ESS / $K$")
    axes[1].axhline(0.5, color="0.35", ls="--", lw=0.9)
    axes[1].annotate("genealogy gate", xy=(0.02, 0.52), xycoords=("axes fraction", "data"),
                     fontsize=7, color="0.35")
    axes[2].set_xlabel("FR opportunity index")
    axes[2].set_ylabel(r"retention $G_t=\mathrm{ESS}^+_{\rm anc}/\mathrm{ESS}^-_{\rm anc}$")
    axes[2].axhline(1.0, color="0.35", ls=":", lw=0.9)
    axes[0].legend(fontsize=7, frameon=False)
    n_opp = max((len(v) for v in retention.values()), default=0)
    fig.suptitle(f"Self-limitation diagnostics over {n_opp} FR opportunities "
                 f"(median over seeds)", fontsize=8, y=1.02)
    add_panel_labels(axes)
    return save_figure(fig, out / "fig_v3_mechanism", tight=True)


def fig_decomposition(gates, out):
    """R_total = R_shape * R_FR, so the headline cannot hide the factorization."""
    cols = [c for c in gates.columns if c.startswith("SvsCtrl_F_2")]
    sub = gates.dropna(subset=cols) if cols else gates.iloc[0:0]
    if sub.empty:
        return None
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    y = np.arange(len(sub))
    ax.barh(y - 0.22, sub["Rshape_F_2"], height=0.4, label=r"$R_{\rm shape}$ (bias shape)")
    ax.barh(y + 0.22, sub["SvsCtrl_F_2"], height=0.4, label=r"$R_{\rm FR}$ (FR increment)")
    ax.plot(sub["S_F_2"], y, "k.", ms=7, label=r"$R_{\rm total}$")
    ax.axvline(1.0, color="0.35", lw=0.9)
    ax.axvline(1.10, color="0.35", ls="--", lw=0.9)
    ax.set_yticks(y); ax.set_yticklabels(sub["arm"], fontsize=7)
    ax.set_xlabel(r"speedup at $\varepsilon_{F,2}$")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title(r"$R_{\rm total}=R_{\rm shape}\times R_{\rm FR}$ "
                 "(dashed line: advancement threshold)", fontsize=8)
    return save_figure(fig, out / "fig_v3_decomposition", tight=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="results/v3/arms/arm_manifest.json")
    ap.add_argument("--baseline", default="results/v3/plain_abf")
    ap.add_argument("--analysis", default="results/v3/pilot_analysis")
    ap.add_argument("--thresholds", default="results/v3/V3_THRESHOLDS.json")
    ap.add_argument("--out", default="results/v3/pilot_analysis/figures")
    ap.add_argument("--arms", default=None, help="comma-separated subset to plot")
    args = ap.parse_args(argv)

    apply_publication_style()
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    eps = {k: v["value"] for k, v in
           json.loads(pathlib.Path(args.thresholds).read_text())["thresholds"].items()}
    manifest = json.loads(pathlib.Path(args.manifest).read_text())["arms"]

    frames = {"plain_abf": _load(pathlib.Path(args.baseline))}
    for a in manifest:
        root = pathlib.Path(a["output_root"])
        if (root / "production_gpu" / "production_gpu_runs_long.csv").exists():
            frames[a["arm"]] = _load(root)
    arms = (args.arms.split(",") if args.arms
            else [a for a in ["plain_abf", "C_capped12_noFR", "C_capped12_FT_rho0.85",
                              "P_FT_rho0.85"] if a in frames])
    ret_path = pathlib.Path(args.analysis) / "retention_trajectories.json"
    retention = json.loads(ret_path.read_text()) if ret_path.exists() else {}

    print("convergence ->", fig_convergence(frames, eps, out, arms))
    print("mechanism   ->", fig_mechanism(frames, retention, out,
                                          [a for a in arms if a != "plain_abf"]))
    gpath = pathlib.Path(args.analysis) / "v3_pilot_gates.csv"
    if gpath.exists():
        r = fig_decomposition(pd.read_csv(gpath), out)
        print("decomposition ->", r if r else "(no Track-C candidates yet)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
