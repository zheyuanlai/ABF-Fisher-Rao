#!/usr/bin/env python3
"""Figures for the WCA follow-up studies (representative / equal-compute / frozen).

Reads ``<output_root>/raw/*.npz`` directly (so it can draw seed IQR bands and
profiles) and writes PNGs to ``<output_root>/figures_<stage>/`` and, with
--report-figdir, copies the manuscript figures to report/figures.

  representative ->
    fig_wca_fixed_vs_adaptive.png   per cell: ABF / fixed-mFR / adaptive-mFR final
                                    L2(F) with IQR; matched-seed gain bars.
    fig_wca_adaptive_schedule.png   adaptive gate trajectories (rate_eff, support,
                                    diversity, mismatch EMA) for a starved vs easy cell.
    fig_wca_rep_error_vs_time.png   error-vs-time with IQR bands per cell/method.
  equal_compute ->
    fig_wca_equal_compute.png       final & integrated L2(F) vs force-eval budget.
  frozen_bias ->
    fig_wca_frozen_bias.png         reconstructed F(z) vs TI reference + online/frozen L2.

Usage:
  python scripts/plot_wca_followup.py --config configs/wca_representative.yaml --stage representative \
      --report-figdir report/figures
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np  # noqa: E402
import wca_followup_jobs as fj  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

METHOD_COLOR = {"abf": "#444444", "fr_estimated": "#2c7fb8", "fr_uniform": "#7fbf7b",
                "fr_oracle": "#984ea3", "fr_estimated_adaptive": "#d95f0e"}
METHOD_LABEL = {"abf": "ABF", "fr_estimated": "fixed mFR", "fr_uniform": "uniform mFR",
                "fr_oracle": "oracle mFR", "fr_estimated_adaptive": "adaptive mFR"}


def _val(d, k, default=None):
    if k not in d.files:
        return default
    v = d[k]
    if isinstance(v, np.ndarray) and v.ndim == 0:
        v = v.item()
    if isinstance(v, bytes):
        v = v.decode()
    return v


def _tag(d):
    return (f"b{_val(d,'beta'):g}_h{_val(d,'h'):g}_w{_val(d,'w'):g}"
            f"_n{int(_val(d,'n_dim'))}_a{_val(d,'a'):g}")


def load(raw_dir, stages=None):
    out = []
    for p in sorted(glob.glob(os.path.join(raw_dir, "*.npz"))):
        try:
            d = np.load(p, allow_pickle=True)
        except Exception:
            continue
        if "l2_f" not in d.files:
            continue
        if stages and str(_val(d, "stage")) not in stages:
            continue
        out.append(d)
    return out


def _med_iqr(vals):
    a = np.asarray(vals, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan"), float("nan"), float("nan")
    return np.median(a), np.percentile(a, 25), np.percentile(a, 75)


def _save(fig, fig_dir, name, report_figdir=None, report_name=None):
    os.makedirs(fig_dir, exist_ok=True)
    path = os.path.join(fig_dir, name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}")
    if report_figdir and report_name:
        os.makedirs(report_figdir, exist_ok=True)
        shutil.copy(path, os.path.join(report_figdir, report_name))
        print(f"    copied -> {os.path.join(report_figdir, report_name)}")


# --------------------------------------------------------------------------- #
def plot_representative(runs, fig_dir, report_figdir=None):
    cells = sorted({_tag(d) for d in runs})
    methods = ["abf", "fr_estimated", "fr_estimated_adaptive", "fr_uniform", "fr_oracle"]
    present = [m for m in methods if any(str(_val(d, "method")) == m for d in runs)]

    # ---- 1) fixed vs adaptive: final L2(F) (IQR) + matched gain bars ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(cells))
    width = 0.8 / max(len(present), 1)
    for i, m in enumerate(present):
        meds, los, his = [], [], []
        for tag in cells:
            vals = [float(_val(d, "l2_f")) for d in runs
                    if _tag(d) == tag and str(_val(d, "method")) == m]
            med, lo, hi = _med_iqr(vals)
            meds.append(med); los.append(med - lo); his.append(hi - med)
        ax1.bar(x + i * width, meds, width, yerr=[los, his], capsize=2,
                color=METHOD_COLOR.get(m), label=METHOD_LABEL.get(m, m))
    ax1.set_xticks(x + width * (len(present) - 1) / 2)
    ax1.set_xticklabels([t.replace("_w2", "").replace("_a1.5", "") for t in cells], rotation=30, ha="right", fontsize=8)
    ax1.set_ylabel("final $L_2(F)$ (median, IQR)")
    ax1.set_title("Per-cell accuracy")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3, axis="y")

    # matched gain for fixed vs adaptive
    def matched_gain(m, tag):
        gains = []
        for d in runs:
            if _tag(d) != tag or str(_val(d, "method")) != m:
                continue
            base = [b for b in runs if _tag(b) == tag and str(_val(b, "method")) == "abf"
                    and int(_val(b, "seed")) == int(_val(d, "seed"))
                    and int(_val(b, "n_replicas")) == int(_val(d, "n_replicas"))
                    and int(_val(b, "n_steps")) == int(_val(d, "n_steps"))]
            if not base:
                continue
            bf, ff = float(_val(base[0], "l2_f")), float(_val(d, "l2_f"))
            if bf > 0:
                gains.append(100.0 * (bf - ff) / bf)
        return _med_iqr(gains)
    for i, m in enumerate(["fr_estimated", "fr_estimated_adaptive"]):
        if m not in present:
            continue
        meds, los, his = [], [], []
        for tag in cells:
            med, lo, hi = matched_gain(m, tag)
            meds.append(med); los.append(med - lo); his.append(hi - med)
        ax2.bar(x + i * 0.4, meds, 0.4, yerr=[los, his], capsize=2,
                color=METHOD_COLOR.get(m), label=METHOD_LABEL.get(m))
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_xticks(x + 0.2)
    ax2.set_xticklabels([t.replace("_w2", "").replace("_a1.5", "") for t in cells], rotation=30, ha="right", fontsize=8)
    ax2.set_ylabel("matched-seed median gain over ABF (%)")
    ax2.set_title("Fixed vs adaptive mFR gain")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    _save(fig, fig_dir, "fig_wca_fixed_vs_adaptive.png", report_figdir, "fig_wca_phase_09_fixed_vs_adaptive.png")

    # ---- 2) adaptive schedule trajectories (starved vs easy) ----
    adruns = [d for d in runs if str(_val(d, "method")) == "fr_estimated_adaptive"
              and np.asarray(_val(d, "adaptive_log_step", np.array([]))).size > 0]
    if adruns:
        # pick highest-ABF-error and lowest-ABF-error cell
        abf_err = {}
        for tag in cells:
            v = [float(_val(d, "l2_f")) for d in runs if _tag(d) == tag and str(_val(d, "method")) == "abf"]
            if v:
                abf_err[tag] = np.median(v)
        if abf_err:
            starved = max(abf_err, key=abf_err.get)
            easy = min(abf_err, key=abf_err.get)
            fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
            for ax, tag, title in [(axes[0], starved, f"starved {starved.split('_w2')[0]}"),
                                   (axes[1], easy, f"easy {easy.split('_w2')[0]}")]:
                sel = [d for d in adruns if _tag(d) == tag]
                if not sel:
                    ax.set_title(title + " (no adaptive run)"); continue
                d = sel[0]
                step = np.asarray(_val(d, "adaptive_log_step"), float)
                for key, c in [("fr_rate_eff", "#d95f0e"), ("support_gate", "#2c7fb8"),
                               ("diversity_gate", "#7fbf7b"), ("support_ema", "#984ea3")]:
                    y = np.asarray(_val(d, f"adaptive_log_{key}", np.array([])), float)
                    if y.size == step.size:
                        ax.plot(step, y, lw=1.4, color=c, label=key)
                ax.set_title(title); ax.set_xlabel("step"); ax.grid(alpha=0.3)
            axes[0].set_ylabel("gate value / rate")
            axes[0].legend(fontsize=8, loc="upper right")
            fig.suptitle("Adaptive FR schedule: gates self-limit in the easy regime")
            fig.tight_layout()
            _save(fig, fig_dir, "fig_wca_adaptive_schedule.png", report_figdir,
                  "fig_wca_phase_10_adaptive_schedule.png")

    # ---- 3) error vs time (IQR) for fixed vs adaptive vs abf ----
    fig, ax = plt.subplots(figsize=(8, 5.2))
    abf_err = {}
    for tag in cells:
        v = [float(_val(d, "l2_f")) for d in runs if _tag(d) == tag and str(_val(d, "method")) == "abf"]
        if v:
            abf_err[tag] = np.median(v)
    sel_cells = []
    if abf_err:
        sel_cells = [max(abf_err, key=abf_err.get), min(abf_err, key=abf_err.get)]
    for tag in sel_cells:
        lbl = "starved" if tag == sel_cells[0] else "easy"
        for m, ls in [("abf", "--"), ("fr_estimated", "-"), ("fr_estimated_adaptive", ":")]:
            sel = [d for d in runs if _tag(d) == tag and str(_val(d, "method")) == m]
            if not sel:
                continue
            t = np.asarray(_val(sel[0], "times"), float)
            mats = [np.asarray(_val(d, "l2_f_t"), float) for d in sel
                    if np.asarray(_val(d, "l2_f_t"), float).size == t.size]
            if not mats:
                continue
            A = np.vstack(mats)
            med = np.nanmedian(A, 0)
            line, = ax.plot(t, med, ls, lw=1.6, label=f"{lbl}:{METHOD_LABEL.get(m,m)}")
            ax.fill_between(t, np.nanpercentile(A, 25, 0), np.nanpercentile(A, 75, 0),
                            alpha=0.12, color=line.get_color())
    ax.set_xlabel("time"); ax.set_ylabel("$L_2(F)$")
    ax.set_title("Error vs time: adaptive tracks fixed mFR when starved, ABF when easy")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, fig_dir, "fig_wca_rep_error_vs_time.png", report_figdir,
          "fig_wca_phase_11_rep_error_vs_time.png")


# --------------------------------------------------------------------------- #
def plot_equal_compute(runs, fig_dir, report_figdir=None):
    cells = sorted({_tag(d) for d in runs})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    markers = {"abf": "o", "fr_estimated": "s"}
    for tag in cells:
        for m in ["abf", "fr_estimated"]:
            pts = {}
            for d in runs:
                if _tag(d) != tag or str(_val(d, "method")) != m:
                    continue
                b = int(_val(d, "budget"))
                pts.setdefault(b, {"final": [], "integ": []})
                pts[b]["final"].append(float(_val(d, "l2_f")))
                pts[b]["integ"].append(float(_val(d, "integrated_l2_f")))
            if not pts:
                continue
            budgets = sorted(pts)
            fmed = [np.median(pts[b]["final"]) for b in budgets]
            imed = [np.median(pts[b]["integ"]) for b in budgets]
            lbl = f"{tag.split('_w2')[0]}:{METHOD_LABEL.get(m,m)}"
            ax1.plot(budgets, fmed, marker=markers[m], lw=1.4, label=lbl)
            ax2.plot(budgets, imed, marker=markers[m], lw=1.4, label=lbl)
    for ax, t in [(ax1, "final $L_2(F)$ vs budget"), (ax2, "integrated $L_2(F)$ vs budget")]:
        ax.set_xlabel("force-eval budget $N\\cdot n_{\\rm steps}$")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(t); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7)
    ax1.set_ylabel("$L_2(F)$")
    fig.tight_layout()
    _save(fig, fig_dir, "fig_wca_equal_compute.png", report_figdir, "fig_wca_phase_12_equal_compute.png")


# --------------------------------------------------------------------------- #
def plot_frozen(runs, fig_dir, report_figdir=None):
    cells = sorted({_tag(d) for d in runs})
    srcs = sorted({str(_val(d, "frozen_source_method")) for d in runs})
    fig, axes = plt.subplots(1, max(len(cells), 1), figsize=(6.2 * max(len(cells), 1), 5), squeeze=False)
    for ci, tag in enumerate(cells):
        ax = axes[0][ci]
        ref_done = False
        for sm in srcs:
            sel = [d for d in runs if _tag(d) == tag and str(_val(d, "frozen_source_method")) == sm]
            if not sel:
                continue
            grid = np.asarray(_val(sel[0], "grid"), float)
            if not ref_done:
                ax.plot(grid, np.asarray(_val(sel[0], "ref_free_energy"), float), "k--", lw=1.6, label="TI reference")
                ref_done = True
            recon = np.vstack([np.asarray(_val(d, "F_recon"), float) for d in sel])
            med = np.nanmedian(recon, 0)
            line, = ax.plot(grid, med, lw=1.5, color=METHOD_COLOR.get(sm), label=f"frozen {METHOD_LABEL.get(sm,sm)}")
            ax.fill_between(grid, np.nanpercentile(recon, 25, 0), np.nanpercentile(recon, 75, 0),
                            alpha=0.12, color=line.get_color())
        ax.set_xlim(-0.1, 1.1)
        ax.set_xlabel("z"); ax.set_ylabel("F(z)")
        ax.set_title(f"frozen-bias reconstruction {tag.split('_w2')[0]}")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, fig_dir, "fig_wca_frozen_bias.png", report_figdir, "fig_wca_phase_13_frozen_bias.png")


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", default=None)
    ap.add_argument("--stages", nargs="*", default=None)
    ap.add_argument("--raw", default=None)
    ap.add_argument("--report-figdir", default=None)
    args = ap.parse_args(argv)
    cfg = fj.load_yaml(args.config)
    study = cfg.get("experiment_name", "followup")
    raw_dir = args.raw or os.path.join(cfg["output_root"], "raw")
    stages = args.stages or ([args.stage] if args.stage else None)
    fig_dir = os.path.join(cfg["output_root"], f"figures_{(stages[0] if stages else 'all')}")

    runs = load(raw_dir, stages)
    if not runs:
        print("[plot-followup] no runs found")
        return 1
    is_frozen = any(str(_val(d, "mode", "sample")) == "frozen" for d in runs)
    print(f"[plot-followup] study={study} loaded {len(runs)} runs (frozen={is_frozen}) -> {fig_dir}")

    if is_frozen:
        plot_frozen(runs, fig_dir, args.report_figdir)
    elif "equal_compute" in study:
        plot_equal_compute(runs, fig_dir, args.report_figdir)
    else:
        plot_representative(runs, fig_dir, args.report_figdir)
    print("[plot-followup] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
