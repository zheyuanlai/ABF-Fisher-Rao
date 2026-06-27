#!/usr/bin/env python3
"""Manuscript-quality figures for the WCA phase-diagram study.

Reads the summary CSVs written by ``analyze_wca_phase_diagram.py`` and renders
figures into ``<output_root>/figures_<stage>/``.  The script intentionally uses
only the experiment environment's core scientific stack (csv/numpy/matplotlib),
not pandas, so it runs in the same ``ddlpm`` environment as the sampler.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import sys
from collections import Counter, defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import wca_phase_jobs as pj  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 150, "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 8.5,
    "axes.grid": True, "grid.alpha": 0.25,
})

METHOD_LABEL = {"abf": "ABF only", "fr_estimated": "mFR--ABF (estimated)",
                "fr_uniform": "mFR--ABF (uniform)", "fr_oracle": "mFR--ABF (oracle)"}
METHOD_COLOR = {"abf": "#444444", "fr_estimated": "#1f77b4",
                "fr_uniform": "#2ca02c", "fr_oracle": "#d62728"}


def _parse(v: str):
    if v == "":
        return np.nan
    low = v.lower()
    if low == "nan":
        return np.nan
    if low == "inf":
        return np.inf
    if low == "-inf":
        return -np.inf
    try:
        f = float(v)
    except ValueError:
        return v
    return f


def read_csv_rows(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as fh:
        return [{k: _parse(v) for k, v in row.items()} for row in csv.DictReader(fh)]


def _finite(values) -> np.ndarray:
    arr = np.asarray([float(v) for v in values if isinstance(v, (int, float, np.floating))], dtype=float)
    return arr[np.isfinite(arr)]


def _median(values):
    arr = _finite(values)
    return float(np.median(arr)) if arr.size else np.nan


def _quantile(values, q):
    arr = _finite(values)
    return float(np.quantile(arr, q)) if arr.size else np.nan


def _mode(rows, key):
    vals = [r[key] for r in rows if key in r and not (isinstance(r[key], float) and np.isnan(r[key]))]
    if not vals:
        return np.nan
    return Counter(vals).most_common(1)[0][0]


def _unique(rows, key):
    vals = {float(r[key]) for r in rows if key in r and not (isinstance(r[key], float) and np.isnan(r[key]))}
    return sorted(vals)


def _rows(rows, **conds):
    out = rows
    for k, v in conds.items():
        if isinstance(v, (int, float, np.floating)):
            out = [r for r in out if k in r and np.isclose(float(r[k]), float(v))]
        else:
            out = [r for r in out if r.get(k) == v]
    return out


def _pivot(rows, value):
    """Pivot a per-cell table into a (h rows, beta cols) matrix at the modal M."""
    if not rows:
        return None
    M0 = _mode(rows, "M")
    d = _rows(rows, M=M0)
    betas = _unique(d, "beta")
    hs = _unique(d, "h")
    mat = np.full((len(hs), len(betas)), np.nan)
    for i, hh in enumerate(hs):
        for j, bb in enumerate(betas):
            sub = _rows(d, h=hh, beta=bb)
            if sub and value in sub[0]:
                mat[i, j] = float(sub[0][value])
    return dict(mat=mat, betas=betas, hs=hs, M=M0)


def _heatmap(ax, piv, title, cmap, norm=None, fmt="{:.3f}", cbar_label=""):
    mat, betas, hs = piv["mat"], piv["betas"], piv["hs"]
    im = ax.imshow(mat, origin="lower", aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(len(betas)), [f"{b:g}" for b in betas])
    ax.set_yticks(range(len(hs)), [f"{h:g}" for h in hs])
    ax.set_xlabel(r"inverse temperature $\beta$")
    ax.set_ylabel(r"barrier height $h$")
    ax.set_title(title)
    ax.grid(False)
    for i in range(len(hs)):
        for j in range(len(betas)):
            if np.isfinite(mat[i, j]):
                v = mat[i, j]
                try:
                    lum = im.norm(v) if im.norm else 0.5
                except Exception:
                    lum = 0.5
                color = "white" if (lum is not None and lum > 0.6) else "black"
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8.5, color=color)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if cbar_label:
        cb.set_label(cbar_label)
    return im


def _regimes(main):
    """Pick easy / anchor / hard cells at the modal M by beta*h."""
    M0 = _mode(main, "M")
    d = _rows(main, M=M0)
    if not d:
        return []
    easy = min(d, key=lambda r: float(r["beta"]) * float(r["h"]))
    hard = max(d, key=lambda r: float(r["beta"]) * float(r["h"]))
    anchors = [r for r in d if np.isclose(float(r["beta"]), 1.0) and np.isclose(float(r["h"]), 2.0)]
    anchor = anchors[0] if anchors else min(d, key=lambda r: abs(float(r["beta"]) * float(r["h"]) - 2.0))
    out, seen = [], set()
    for label, row in [("easy", easy), ("anchor", anchor), ("hard", hard)]:
        tag = row["physics_tag"]
        if tag not in seen:
            seen.add(tag); out.append((label, row))
    return out


def fig_heatmaps(S, figdir):
    main = S["main_table"]
    piv_abf = _pivot(main, "abf_l2_f")
    piv_fr = _pivot(main, "fr_est_l2_f")
    piv_R = _pivot(main, "R_est")
    if piv_abf is None or piv_fr is None or piv_R is None:
        return None
    vals = np.concatenate([_finite(piv_abf["mat"].ravel()), _finite(piv_fr["mat"].ravel())])
    lognorm = LogNorm(vmin=max(float(vals.min()), 1e-4), vmax=float(vals.max()))
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    _heatmap(axs[0], piv_abf, f"ABF final $L^2(F)$  (M={int(piv_abf['M'])})", "viridis_r", lognorm,
             cbar_label="$L^2(F)$")
    _heatmap(axs[1], piv_fr, "mFR--ABF (estimated) final $L^2(F)$", "viridis_r", lognorm,
             cbar_label="$L^2(F)$")
    Rmat = piv_R["mat"]
    rvals = _finite(Rmat.ravel())
    rnorm = TwoSlopeNorm(vmin=min(0.8, float(rvals.min())), vcenter=1.0, vmax=max(1.2, float(rvals.max())))
    _heatmap(axs[2], piv_R, r"improvement ratio $R=L^2(F)_{\rm ABF}/L^2(F)_{\rm mFR}$",
             "RdBu", rnorm, fmt="{:.2f}", cbar_label="R  (>1 = mFR better)")
    fig.suptitle("WCA phase diagram: ABF vs marginal-Fisher--Rao ABF over $(\\beta, h)$", y=1.02)
    fig.tight_layout()
    p = os.path.join(figdir, "fig_wca_phase_01_heatmaps.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def fig_integrated_R(S, figdir):
    est = [r for r in S["improvement"] if r.get("method") == "fr_estimated"]
    piv = _pivot(est, "R_integrated_median")
    if piv is None:
        return None
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    Rmat = piv["mat"]; rvals = _finite(Rmat.ravel())
    rnorm = TwoSlopeNorm(vmin=min(0.8, float(rvals.min())), vcenter=1.0, vmax=max(1.2, float(rvals.max())))
    _heatmap(ax, piv, r"integrated-error ratio $R_{\rm int}$ (mFR estimated)", "RdBu", rnorm,
             fmt="{:.2f}", cbar_label="$R_{int}$ (>1 = mFR better)")
    fig.tight_layout()
    p = os.path.join(figdir, "fig_wca_phase_02_integrated_R.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def fig_transitions(S, figdir):
    main = S["main_table"]
    M0 = _mode(main, "M")
    d = sorted(_rows(main, M=M0), key=lambda r: (float(r["beta"]), float(r["h"])))
    if not d:
        return None
    labels = [f"$\\beta$={r['beta']:g},h={r['h']:g}" for r in d]
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(max(7, 0.7 * len(d)), 4.2))
    ax.bar(x - 0.2, [r["abf_barrier_crossings"] for r in d], 0.4, label="ABF", color=METHOD_COLOR["abf"])
    ax.bar(x + 0.2, [r["fr_est_barrier_crossings"] for r in d], 0.4, label="mFR--ABF (est)",
           color=METHOD_COLOR["fr_estimated"])
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.set_ylabel("barrier crossings (median over seeds)")
    ax.set_title(f"Reaction-coordinate barrier crossings (M={int(M0)})")
    ax.legend(); fig.tight_layout()
    p = os.path.join(figdir, "fig_wca_phase_03_transitions.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def fig_fr_event_frac(S, figdir):
    est = [r for r in S["config_summary"] if r.get("method") == "fr_estimated"]
    piv = _pivot(est, "fr_event_fraction_median")
    if piv is None:
        return None
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    _heatmap(ax, piv, "FR event fraction per application (estimated)", "magma",
             fmt="{:.4f}", cbar_label="mean fraction of population resampled")
    fig.tight_layout()
    p = os.path.join(figdir, "fig_wca_phase_04_fr_event_frac.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def _group_series(rows, value_key):
    by_t = defaultdict(list)
    for r in rows:
        by_t[float(r["t"])].append(r.get(value_key, np.nan))
    t = np.array(sorted(by_t), dtype=float)
    med = np.array([_median(by_t[x]) for x in t])
    q25 = np.array([_quantile(by_t[x], 0.25) for x in t])
    q75 = np.array([_quantile(by_t[x], 0.75) for x in t])
    return t, med, q25, q75


def fig_genealogy(S, figdir):
    fr = [r for r in S["runs_long"] if r.get("method") == "fr_estimated"]
    if not fr:
        return None
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    for label, row in _regimes(S["main_table"]):
        tag = row["physics_tag"]
        sub = [r for r in fr if r.get("physics_tag") == tag]
        if not sub:
            continue
        t, ess, _, _ = _group_series(sub, "ancestor_ess")
        _, maf, _, _ = _group_series(sub, "max_ancestor_frac")
        lbl = f"{label} ($\\beta$={row['beta']:g},h={row['h']:g})"
        axs[0].plot(t, ess, label=lbl)
        axs[1].plot(t, maf, label=lbl)
    axs[0].set_xlabel("time"); axs[0].set_ylabel("ancestor ESS")
    axs[0].set_title("Genealogical diversity (ancestor ESS)"); axs[0].legend()
    axs[1].set_xlabel("time"); axs[1].set_ylabel("max ancestor fraction")
    axs[1].set_title("Genealogical collapse check"); axs[1].legend()
    fig.tight_layout()
    p = os.path.join(figdir, "fig_wca_phase_05_genealogy.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def fig_error_vs_time(S, figdir):
    regimes = _regimes(S["main_table"])
    fig, axs = plt.subplots(1, len(regimes), figsize=(5 * len(regimes), 4.2), squeeze=False)
    axs = axs[0]
    for ax, (label, row) in zip(axs, regimes):
        tag = row["physics_tag"]
        for method in ["abf", "fr_estimated", "fr_uniform", "fr_oracle"]:
            sub = [r for r in S["runs_long"] if r.get("physics_tag") == tag and r.get("method") == method]
            if not sub:
                continue
            t, med, q25, q75 = _group_series(sub, "l2_f")
            ax.plot(t, med, label=METHOD_LABEL[method], color=METHOD_COLOR[method])
            ax.fill_between(t, q25, q75, color=METHOD_COLOR[method], alpha=0.15)
        ax.set_yscale("log")
        ax.set_xlabel("time"); ax.set_ylabel("$L^2(F)$")
        ax.set_title(f"{label}: $\\beta$={row['beta']:g}, h={row['h']:g} "
                     f"($\\beta h$={float(row['beta'])*float(row['h']):g})")
        ax.legend()
    fig.suptitle("Free-energy error vs time across regimes", y=1.02)
    fig.tight_layout()
    p = os.path.join(figdir, "fig_wca_phase_06_error_vs_time.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def fig_profiles(S, figdir):
    regimes = _regimes(S["main_table"])
    fig, axs = plt.subplots(1, len(regimes), figsize=(5 * len(regimes), 4.2), squeeze=False)
    axs = axs[0]
    for ax, (label, row) in zip(axs, regimes):
        tag = row["physics_tag"]
        sub = [r for r in S["profiles"] if r.get("physics_tag") == tag]
        if not sub:
            continue
        ref = sorted([r for r in sub if r.get("method") == "abf"], key=lambda r: float(r["z"]))
        ax.plot([r["z"] for r in ref], [r["ref_F"] for r in ref], "k--", lw=2, label="TI reference")
        for method in ["abf", "fr_estimated", "fr_uniform", "fr_oracle"]:
            m = sorted([r for r in sub if r.get("method") == method], key=lambda r: float(r["z"]))
            if not m:
                continue
            ax.plot([r["z"] for r in m], [r["F"] for r in m], label=METHOD_LABEL[method], color=METHOD_COLOR[method])
        ax.set_xlabel("reaction coordinate $z$"); ax.set_ylabel("$F(z)$")
        ax.set_title(f"{label}: $\\beta$={row['beta']:g}, h={row['h']:g}")
        ax.legend()
    fig.suptitle("Final free-energy profiles vs TI reference", y=1.02)
    fig.tight_layout()
    p = os.path.join(figdir, "fig_wca_phase_07_profiles.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def fig_gain_vs_difficulty(S, figdir):
    ir = S["improvement"]
    if not ir:
        return None
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    M0 = _mode(ir, "M")
    d = _rows(ir, M=M0)
    for method in ["fr_estimated", "fr_uniform", "fr_oracle"]:
        sub = sorted([r for r in d if r.get("method") == method], key=lambda r: float(r["beta_h"]))
        if not sub:
            continue
        axs[0].plot([r["beta_h"] for r in sub], [r["median_gain_pct_F"] for r in sub], "o-",
                    label=METHOD_LABEL[method], color=METHOD_COLOR[method])
    axs[0].axhline(0, color="k", lw=0.8, ls=":")
    axs[0].set_xlabel(r"difficulty $\beta\,h$"); axs[0].set_ylabel("median gain in $L^2(F)$ over ABF (%)")
    axs[0].set_title(f"Gain vs difficulty (M={int(M0)})"); axs[0].legend()

    est = [r for r in ir if r.get("method") == "fr_estimated"]
    masses = _unique(est, "M")
    if len(masses) > 1:
        grouped = defaultdict(list)
        for r in est:
            grouped[(float(r["beta"]), float(r["h"]))].append(r)
        for (b, h), sub in sorted(grouped.items()):
            if len({r["M"] for r in sub}) > 1:
                sub = sorted(sub, key=lambda r: float(r["M"]))
                axs[1].plot([r["M"] for r in sub], [r["median_gain_pct_F"] for r in sub], "s-",
                            color=METHOD_COLOR["fr_estimated"], label=f"$\\beta$={b:g}, h={h:g}")
        axs[1].axhline(0, color="k", lw=0.8, ls=":")
        axs[1].set_xlabel("physical particle count $M=n_{\\rm dim}^2$")
        axs[1].set_ylabel("median gain in $L^2(F)$ (%)")
        axs[1].set_title("Gain vs system size (estimated)"); axs[1].legend()
    else:
        for method in ["fr_estimated", "fr_oracle"]:
            sub = sorted([r for r in d if r.get("method") == method], key=lambda r: float(r["beta_h"]))
            if not sub:
                continue
            axs[1].plot([r["beta_h"] for r in sub], [r["R_final_median"] for r in sub], "o-",
                        label=METHOD_LABEL[method], color=METHOD_COLOR[method])
        axs[1].axhline(1, color="k", lw=0.8, ls=":")
        axs[1].set_xlabel(r"difficulty $\beta\,h$"); axs[1].set_ylabel("improvement ratio R")
        axs[1].set_title("Improvement ratio vs difficulty"); axs[1].legend()
    fig.tight_layout()
    p = os.path.join(figdir, "fig_wca_phase_08_gain_vs_difficulty.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", default=None, help="figures_<stage> subdir name")
    ap.add_argument("--summaries", default=None)
    ap.add_argument("--figdir", default=None)
    ap.add_argument("--report-figdir", default=None,
                    help="also copy headline composites here (e.g. report/figures)")
    args = ap.parse_args(argv)
    cfg = pj.load_yaml(args.config)
    sums = args.summaries or os.path.join(cfg["output_root"], "summaries")
    stage = args.stage or "all"
    figdir = args.figdir or os.path.join(cfg["output_root"], f"figures_{stage}")
    os.makedirs(figdir, exist_ok=True)

    S = dict(
        main_table=read_csv_rows(os.path.join(sums, "phase_main_table.csv")),
        improvement=read_csv_rows(os.path.join(sums, "phase_improvement_ratios.csv")),
        config_summary=read_csv_rows(os.path.join(sums, "phase_config_summary.csv")),
        runs_long=read_csv_rows(os.path.join(sums, "phase_runs_long.csv")),
        profiles=read_csv_rows(os.path.join(sums, "phase_profiles.csv")),
        fr_events=read_csv_rows(os.path.join(sums, "phase_fr_events.csv")),
        genealogy=read_csv_rows(os.path.join(sums, "phase_genealogy.csv")),
    )
    if not S["main_table"]:
        print("[plot] no main table; nothing to do")
        return 1

    print(f"[plot] figdir={figdir}")
    made = []
    for fn in [fig_heatmaps, fig_integrated_R, fig_transitions, fig_fr_event_frac,
               fig_genealogy, fig_error_vs_time, fig_profiles, fig_gain_vs_difficulty]:
        try:
            p = fn(S, figdir)
            if p:
                made.append(p); print(f"  wrote {p}")
        except Exception as exc:
            print(f"  FAILED {fn.__name__}: {exc!r}")

    if args.report_figdir and made:
        os.makedirs(args.report_figdir, exist_ok=True)
        for p in made:
            shutil.copy2(p, os.path.join(args.report_figdir, os.path.basename(p)))
        print(f"[plot] copied {len(made)} figures to {args.report_figdir}")
    print("[plot] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
