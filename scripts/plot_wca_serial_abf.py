#!/usr/bin/env python3
"""Figure for the WCA serial one-walker ABF closeout (Part H).

fig_wca_serial_abf_equal_budget.png: one panel per representative cell (starved /
intermediate / easy). x = force-evaluation budget N*n_steps (log); y = L2(F) (log).
Serial one-walker ABF is drawn as a median+IQR CURVE over its accumulated budget (so one
can see whether it plateaus or catches up); the parallel ABF shapes, the deployable mFR, and
any doubled-budget ABF are drawn as points at their budgets (median + IQR over seeds), read
from the existing equal-compute summary.

Usage:
  python scripts/plot_wca_serial_abf.py \
      --config configs/wca_serial_abf_equal_budget.yaml --stages production \
      --equal-compute results/wca_equal_compute/summaries/equal_compute_summary.csv \
      --report-figdir report/figures
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import wca_serial_abf as sa  # noqa: E402

CELL_ORDER = ["b1_h2_w2_n10_a1.5", "b2_h6_w2_n10_a1.5", "b4_h1_w2_n10_a1.5"]
CELL_TITLE = {"b1_h2_w2_n10_a1.5": "starved  $\\beta{=}1, h{=}2$",
              "b2_h6_w2_n10_a1.5": "intermediate  $\\beta{=}2, h{=}6$",
              "b4_h1_w2_n10_a1.5": "easy  $\\beta{=}4, h{=}1$"}
METHOD_STYLE = {"abf": ("#1f77b4", "o", "parallel ABF"),
                "fr_estimated": ("#d62728", "s", "parallel mFR")}


def _val(d, k, default=None):
    if k not in d.files:
        return default
    x = d[k]
    return x.item() if isinstance(x, np.ndarray) and x.ndim == 0 else x


def _tag(d):
    return (f"b{float(_val(d,'beta')):g}_h{float(_val(d,'h')):g}_w{float(_val(d,'w')):g}"
            f"_n{int(_val(d,'n_dim'))}_a{float(_val(d,'a')):g}")


def load_serial(raw_dir, stages):
    out = {}
    for p in sorted(glob.glob(os.path.join(raw_dir, "*.npz"))):
        try:
            d = np.load(p, allow_pickle=True)
        except Exception:
            continue
        if stages and str(_val(d, "stage")) not in stages:
            continue
        out.setdefault(_tag(d), []).append(d)
    return out


def load_equal_compute(path):
    rows = {}
    if not (path and os.path.exists(path)):
        return rows
    for r in csv.DictReader(open(path)):
        rows.setdefault(r["physics_tag"], []).append(r)
    return rows


def _serial_curve(ds):
    """Median + IQR of L2(F) over seeds at each snapshot budget (shared grid)."""
    budgets = np.asarray(_val(ds[0], "snap_budget"), float)
    mat = []
    for d in ds:
        b = np.asarray(_val(d, "snap_budget"), float)
        y = np.asarray(_val(d, "l2_f_t"), float)
        if b.shape == budgets.shape and np.allclose(b, budgets):
            mat.append(y)
    if not mat:
        return budgets, None, None, None
    mat = np.vstack(mat)
    return (budgets, np.nanmedian(mat, 0),
            np.nanpercentile(mat, 25, 0), np.nanpercentile(mat, 75, 0))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--stages", nargs="+", default=["production"])
    ap.add_argument("--equal-compute",
                    default="results/wca_equal_compute/summaries/equal_compute_summary.csv")
    ap.add_argument("--report-figdir", default=None)
    args = ap.parse_args(argv)

    cfg = sa.load_yaml(args.config)
    root = cfg["output_root"]
    fig_dir = os.path.join(root, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    serial = load_serial(os.path.join(root, "raw"), set(args.stages))
    parallel = load_equal_compute(args.equal_compute)
    cells = [c for c in CELL_ORDER if c in serial or c in parallel]
    if not cells:
        cells = sorted(set(serial) | set(parallel))
    n = max(len(cells), 1)

    fig, axes = plt.subplots(1, n, figsize=(6.0 * n, 5.0), squeeze=False)
    for ci, tag in enumerate(cells):
        ax = axes[0][ci]
        # serial curve
        if tag in serial:
            b, med, lo, hi = _serial_curve(serial[tag])
            if med is not None:
                ax.plot(b, med, color="#2ca02c", lw=1.8, marker=".", ms=4,
                        label="serial ABF ($N{=}1$)", zorder=5)
                ax.fill_between(b, lo, hi, color="#2ca02c", alpha=0.15, zorder=1)
                ax.plot(b[-1], med[-1], color="#2ca02c", marker="*", ms=14, zorder=6)
        # parallel points
        for r in parallel.get(tag, []):
            m = r["method"]
            if m not in METHOD_STYLE:
                continue
            color, marker, _ = METHOD_STYLE[m]
            try:
                budget = float(r["budget"]); med = float(r["l2_f_median"])
                q25 = float(r["l2_f_q25"]); q75 = float(r["l2_f_q75"])
                nrep = int(r["n_replicas"])
            except Exception:
                continue
            jitter = {512: 0.82, 1024: 1.0, 2048: 1.22}.get(nrep, 1.0)
            ax.errorbar(budget * jitter, med, yerr=[[max(med - q25, 0)], [max(q75 - med, 0)]],
                        color=color, marker=marker, ms=8, capsize=3, lw=1.2, zorder=4)
        # legend proxies (dedup)
        handles = [plt.Line2D([], [], color="#2ca02c", lw=1.8, marker=".", label="serial ABF ($N{=}1$)")]
        for m, (color, marker, lbl) in METHOD_STYLE.items():
            if any(rr["method"] == m for rr in parallel.get(tag, [])):
                handles.append(plt.Line2D([], [], color=color, marker=marker, ls="none", label=lbl))
        ax.axvline(sa.BASE_BUDGET, color="grey", ls=":", lw=1.0, alpha=0.7)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"force-eval budget $N\cdot n_{\rm steps}$")
        if ci == 0:
            ax.set_ylabel(r"$L_2(F)$")
        ax.set_title(CELL_TITLE.get(tag, tag.split("_w2")[0]))
        ax.grid(alpha=0.3, which="both")
        ax.legend(handles=handles, fontsize=8, loc="best")
    fig.suptitle("Serial one-walker ABF vs parallel ABF/mFR at equal force-evaluation budget", y=1.02)
    fig.tight_layout()

    name = "fig_wca_serial_abf_equal_budget.png"
    out = os.path.join(fig_dir, name)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[fig] {out}")
    if args.report_figdir:
        os.makedirs(args.report_figdir, exist_ok=True)
        rp = os.path.join(args.report_figdir, name)
        fig.savefig(rp, dpi=150, bbox_inches="tight")
        print(f"[fig] {rp}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
