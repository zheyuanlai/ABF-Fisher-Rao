#!/usr/bin/env python3
"""Aggregate the WCA serial one-walker ABF control (Part H) and merge it with the existing
equal-compute summary onto a single force-evaluation-budget axis.

Reads ``results/wca_serial_abf/raw/*.npz`` (per-trajectory) and, optionally,
``results/wca_equal_compute/summaries/equal_compute_summary.csv`` (parallel ABF/mFR at equal
budget). Writes:

  results/wca_serial_abf/summaries/serial_abf_run_summary.csv   per-trajectory scalar metrics
  results/wca_serial_abf/summaries/serial_abf_summary.csv       per-cell median/IQR over seeds
  results/wca_serial_abf/summaries/serial_equal_compute_merged.csv   serial + parallel, one axis
  report/tables/wca_serial_abf.tex          compact comparison table
  report/tables/wca_serial_abf_numbers.tex  \\newcommand macros for in-text numbers (WCAserial*)

Usage:
  python scripts/analyze_wca_serial_abf.py \
      --config configs/wca_serial_abf_equal_budget.yaml --stages production \
      --equal-compute results/wca_equal_compute/summaries/equal_compute_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np  # noqa: E402
import wca_serial_abf as sa  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT_TABLES = os.path.join(HERE, "..", "report", "tables")
BASE_BUDGET = sa.BASE_BUDGET


def _val(d, k, default=None):
    if k not in d.files:
        return default
    x = d[k]
    return x.item() if isinstance(x, np.ndarray) and x.ndim == 0 else x


def _iqr(a):
    a = np.asarray([x for x in a if x is not None and np.isfinite(x)], float)
    if a.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(np.median(a)), float(np.percentile(a, 25)), float(np.percentile(a, 75))


def _tag(d):
    return (f"b{float(_val(d,'beta')):g}_h{float(_val(d,'h')):g}_w{float(_val(d,'w')):g}"
            f"_n{int(_val(d,'n_dim'))}_a{float(_val(d,'a')):g}")


def write_csv(rows, path, cols=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        open(path, "w").close()
        print(f"[csv] {path} (empty)")
        return
    cols = cols or list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    print(f"[csv] {path} ({len(rows)} rows)")


def load_runs(raw_dir, stages):
    runs = []
    for p in sorted(glob.glob(os.path.join(raw_dir, "*.npz"))):
        try:
            d = np.load(p, allow_pickle=True)
        except Exception:
            continue
        if stages and str(_val(d, "stage")) not in stages:
            continue
        runs.append(d)
    return runs


RUN_SCALARS = [
    "run_id", "stage", "name", "method", "seed", "n_steps", "n_replicas", "budget",
    "budget_reached", "complete", "beta", "h", "w", "n_dim", "M", "a", "beta_h",
    "l2_f", "l2_fp", "l2_f_transition", "l2_fp_transition",
    "l2_f_compact", "l2_f_stretched",
    "frac_compact", "frac_transition", "frac_stretched",
    "n_barrier_crossings", "n_round_trips", "n_up_crossings", "n_down_crossings",
    "integrated_l2_f", "integrated_l2_f_budget", "budget_auc_normalized_l2_f",
    "wall_seconds", "had_nan",
]
AGG_METRICS = [
    "l2_f", "l2_fp", "l2_f_transition", "frac_transition", "n_barrier_crossings",
    "n_round_trips", "integrated_l2_f", "integrated_l2_f_budget",
    "budget_auc_normalized_l2_f", "wall_seconds",
]


def run_rows(runs):
    rows = []
    for d in runs:
        row = {k: _val(d, k) for k in RUN_SCALARS}
        row["physics_tag"] = _tag(d)
        rows.append(row)
    return rows


def _reached(d):
    br = _val(d, "budget_reached")
    return int(br) if br is not None else int(_val(d, "n_steps"))


def cell_summary(runs):
    groups = {}
    for d in runs:
        groups.setdefault((_tag(d), int(_val(d, "n_steps"))), []).append(d)
    out = []
    for (tag, target), ds in sorted(groups.items()):
        d0 = ds[0]
        reached = int(np.median([_reached(x) for x in ds]))    # batched seeds share the step count
        complete = all(bool(_val(x, "complete", False)) for x in ds)
        # For a one-walker control budget == steps actually run.
        rec = dict(physics_tag=tag, method="serial_abf", n_replicas=1, n_steps=reached,
                   budget=reached, target_steps=target, complete=int(complete),
                   beta=float(_val(d0, "beta")), h=float(_val(d0, "h")),
                   M=int(_val(d0, "M")), beta_h=float(_val(d0, "beta_h")), n_seeds=len(ds))
        for m in AGG_METRICS:
            med, lo, hi = _iqr([_val(x, m) for x in ds])
            rec[f"{m}_median"] = med
            rec[f"{m}_q25"] = lo
            rec[f"{m}_q75"] = hi
        out.append(rec)
    return out


# --------------------------------------------------------------------------- #
# Merge with the existing equal-compute summary.
# --------------------------------------------------------------------------- #
MERGE_COLS = ["source", "physics_tag", "method", "n_replicas", "n_steps", "budget",
              "beta", "h", "M", "beta_h", "n_seeds",
              "l2_f_median", "l2_f_q25", "l2_f_q75",
              "l2_fp_median", "l2_fp_q25", "l2_fp_q75",
              "integrated_l2_f_median", "integrated_l2_f_q25", "integrated_l2_f_q75",
              "complete", "target_steps"]


def _merge_row(source, r):
    out = {c: r.get(c, "") for c in MERGE_COLS}
    out["source"] = source
    return out


def merged_rows(serial_cells, equal_compute_csv):
    rows = [_merge_row("serial", r) for r in serial_cells]
    if equal_compute_csv and os.path.exists(equal_compute_csv):
        for r in csv.DictReader(open(equal_compute_csv)):
            rows.append(_merge_row("equal_compute", r))
    return rows


def _base_abf_median(equal_compute_csv):
    """(physics_tag) -> base parallel ABF (N=1024, n_steps=120000) median L2(F)."""
    out = {}
    if not (equal_compute_csv and os.path.exists(equal_compute_csv)):
        return out
    for r in csv.DictReader(open(equal_compute_csv)):
        try:
            if r["method"] == "abf" and int(r["n_replicas"]) == 1024 and int(r["n_steps"]) == 120000:
                out[r["physics_tag"]] = float(r["l2_f_median"])
        except Exception:
            pass
    return out


def _base_mfr_median(equal_compute_csv):
    out = {}
    if not (equal_compute_csv and os.path.exists(equal_compute_csv)):
        return out
    for r in csv.DictReader(open(equal_compute_csv)):
        try:
            if r["method"] == "fr_estimated" and int(r["n_replicas"]) == 1024 and int(r["n_steps"]) == 120000:
                out[r["physics_tag"]] = float(r["l2_f_median"])
        except Exception:
            pass
    return out


# --------------------------------------------------------------------------- #
# LaTeX assets.
# --------------------------------------------------------------------------- #
def _f(x, nd=4):
    try:
        v = float(x)
        return f"{v:.{nd}f}" if np.isfinite(v) else "--"
    except Exception:
        return "--"


CELL_LABEL = {"b1_h2_w2_n10_a1.5": "starved $\\beta{=}1,h{=}2$",
              "b2_h6_w2_n10_a1.5": "intermediate $\\beta{=}2,h{=}6$",
              "b4_h1_w2_n10_a1.5": "easy $\\beta{=}4,h{=}1$"}


def latex_table(merged, serial_runs_by_cell, base_abf, out_path):
    lines = [r"% Auto-generated by analyze_wca_serial_abf.py -- do not edit.",
             r"\begin{table}[t]", r"\centering", r"\small",
             r"\caption{Serial one-walker ABF at equal force-evaluation budget versus parallel"
             r" ABF/mFR. ``gain'' is the median $\Ltwo(F)$ improvement over the base parallel"
             r" ABF ($N{=}1024$, $120$k); the serial win count is the number of serial seeds"
             r" beating that base-ABF median. Budget $=N\cdot n_{\mathrm{steps}}$.}",
             r"\label{tab:wca_serial_abf}",
             r"\begin{tabular}{llrrrrrrr}", r"\toprule",
             r"cell & method & $N$ & $n_{\rm steps}$ & budget & $\Ltwo(F)$ & $\Ltwo(F')$"
             r" & gain\% & wins \\", r"\midrule"]
    by_cell = {}
    for r in merged:
        by_cell.setdefault(r["physics_tag"], []).append(r)
    for tag in sorted(by_cell):
        base = base_abf.get(tag, float("nan"))
        block = sorted(by_cell[tag], key=lambda r: (r["source"] != "serial", str(r["method"]),
                                                    int(r.get("n_steps") or 0)))
        first = True
        for r in block:
            try:
                l2 = float(r["l2_f_median"])
            except Exception:
                continue
            gain = 100.0 * (base - l2) / base if np.isfinite(base) and base > 0 else float("nan")
            wins = "--"
            if r["source"] == "serial":
                seeds = serial_runs_by_cell.get(tag, [])
                if np.isfinite(base):
                    k = sum(1 for v in seeds if np.isfinite(v) and v < base)
                    wins = f"{k}/{len(seeds)}"
            cell = CELL_LABEL.get(tag, tag.split("_w2")[0]) if first else ""
            meth = {"serial_abf": "serial ABF ($N{=}1$)", "abf": "parallel ABF",
                    "fr_estimated": "parallel mFR"}.get(str(r["method"]), str(r["method"]))
            lines.append(
                f"{cell} & {meth} & {int(r['n_replicas'])} & {int(r['n_steps']):,} & "
                f"{int(r['budget']):.2e} & {_f(l2)} & {_f(r.get('l2_fp_median'))} & "
                f"{_f(gain,1)} & {wins} \\\\".replace(",", r"{,}"))
            first = False
        lines.append(r"\midrule")
    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"
    else:
        lines.append(r"\bottomrule")
    lines += [r"\end{tabular}", r"\end{table}", ""]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"[tex] {out_path}")


STARVED = "b1_h2_w2_n10_a1.5"


def latex_numbers(serial_cells, base_abf, base_mfr, out_path):
    """WCAserial* macros for in-text numbers (placeholders '--' when data is missing)."""
    def cell(tag):
        for r in serial_cells:
            if r["physics_tag"] == tag:
                return r
        return None
    s = cell(STARVED)
    serial_l2 = s["l2_f_median"] if s else float("nan")
    serial_budget = s["budget"] if s else BASE_BUDGET
    base_l2 = base_abf.get(STARVED, float("nan"))
    mfr_l2 = base_mfr.get(STARVED, float("nan"))
    # did the configured serial run for the starved anchor finish its target budget?
    complete = bool(s and int(s.get("complete", 0)))
    # gain of serial over base ABF; ratio of serial to mFR
    serial_gain = (100.0 * (base_l2 - serial_l2) / base_l2
                   if np.isfinite(base_l2) and base_l2 > 0 and np.isfinite(serial_l2) else float("nan"))
    mfr_over_serial = (serial_l2 / mfr_l2 if np.isfinite(serial_l2) and np.isfinite(mfr_l2)
                       and mfr_l2 > 0 else float("nan"))

    def m(name, val, nd=4):
        return f"\\newcommand{{\\{name}}}{{{_f(val, nd)}}}"

    lines = [
        r"% Auto-generated by analyze_wca_serial_abf.py -- do not edit.",
        m("WCAserialStarvedFinalLtwo", serial_l2),
        m("WCAserialStarvedBaseABFLtwo", base_l2),
        m("WCAserialStarvedBaseMFRLtwo", mfr_l2),
        m("WCAserialStarvedGainVsBaseABF", serial_gain, 1),
        m("WCAserialStarvedMFRoverSerialRatio", mfr_over_serial, 2),
        f"\\newcommand{{\\WCAserialStarvedBudget}}{{{int(serial_budget):.2e}}}",
        f"\\newcommand{{\\WCAserialExactComplete}}{{{'complete' if complete else 'in progress'}}}",
        "",
    ]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"[tex] {out_path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--stages", nargs="+", default=["production"])
    ap.add_argument("--equal-compute",
                    default="results/wca_equal_compute/summaries/equal_compute_summary.csv")
    args = ap.parse_args(argv)

    cfg = sa.load_yaml(args.config)
    root = cfg["output_root"]
    raw_dir = os.path.join(root, "raw")
    sum_dir = os.path.join(root, "summaries")
    os.makedirs(sum_dir, exist_ok=True)

    runs = load_runs(raw_dir, set(args.stages))
    print(f"[analyze] {len(runs)} serial trajectories from {raw_dir} (stages={args.stages})")

    rr = run_rows(runs)
    write_csv(rr, os.path.join(sum_dir, "serial_abf_run_summary.csv"), cols=RUN_SCALARS + ["physics_tag"])
    cells = cell_summary(runs)
    write_csv(cells, os.path.join(sum_dir, "serial_abf_summary.csv"))

    merged = merged_rows(cells, args.equal_compute)
    write_csv(merged, os.path.join(sum_dir, "serial_equal_compute_merged.csv"), cols=MERGE_COLS)

    base_abf = _base_abf_median(args.equal_compute)
    base_mfr = _base_mfr_median(args.equal_compute)
    serial_by_cell = {}
    for d in runs:
        serial_by_cell.setdefault(_tag(d), []).append(float(_val(d, "l2_f")))
    latex_table(merged, serial_by_cell, base_abf, os.path.join(REPORT_TABLES, "wca_serial_abf.tex"))
    latex_numbers(cells, base_abf, base_mfr, os.path.join(REPORT_TABLES, "wca_serial_abf_numbers.tex"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
