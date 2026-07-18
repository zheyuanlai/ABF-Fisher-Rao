#!/usr/bin/env python3
"""Analyze OPES runs and build the head-to-head comparison vs ABF / mFR.

Reads results/opes_wca/raw/*.npz and writes, under <output_root>/summaries/:
  * opes_run_summary.csv        one row per run (all scalar columns)
  * opes_cell_summary.csv       per (cell, method, hyperparams) median/IQR over seeds
  * opes_vs_baselines.csv       per cell: OPES vs ABF vs mFR median L2(F)/L2(Fp)
  * opes_tuning.csv             (tune stage) per (cell, barrier, pace) median L2(F)

The baseline (ABF / fr_estimated) medians are read from the representative study's
representative_cells_summary.csv so the comparison is against the SAME cells/budget.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

import numpy as np


def _val(d, k, default=np.nan):
    if k not in d.files:
        return default
    v = d[k]
    try:
        return float(v)
    except Exception:
        return v.item() if getattr(v, "ndim", 1) == 0 else default


def _iqr(a):
    a = np.asarray([x for x in a if np.isfinite(x)], dtype=float)
    if a.size == 0:
        return (float("nan"),) * 3
    return float(np.median(a)), float(np.percentile(a, 25)), float(np.percentile(a, 75))


def write_csv(rows, path, cols=None):
    if not rows:
        print(f"[skip] no rows for {path}")
        return
    cols = cols or list(rows[0].keys())
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"[write] {path}  ({len(rows)} rows)")


SCALARS = ["run_id", "study", "stage", "name", "method", "seed", "n_steps", "n_replicas",
           "budget", "beta", "h", "w", "n_dim", "M", "a", "beta_h",
           "opes_barrier", "opes_pace", "opes_sigma", "opes_gamma", "opes_estimator",
           "l2_f", "l2_fp", "l2_f_reweight", "l2_fp_reweight", "integrated_l2_f",
           "l2_f_transition", "l2_fp_transition", "marginal_l2_uniform", "marginal_l2_ref",
           "n_barrier_crossings", "n_round_trips", "opes_neff_frac_final",
           "opes_neff_frac_min", "opes_n_kernels_final", "opes_bias_range_final",
           "had_nan", "runtime_seconds", "physics_tag"]


def _tag(d):
    return (f"b{_val(d,'beta'):g}_h{_val(d,'h'):g}_w{_val(d,'w'):g}"
            f"_n{int(_val(d,'n_dim'))}_a{_val(d,'a'):g}")


def load_runs(raw_dir):
    runs = []
    for path in sorted(glob.glob(os.path.join(raw_dir, "*.npz"))):
        try:
            d = np.load(path, allow_pickle=True)
            if "l2_f" in d.files:
                runs.append(d)
        except Exception as e:
            print(f"[warn] skip {path}: {e}")
    return runs


def run_rows(runs):
    rows = []
    for d in runs:
        row = {k: _val(d, k) for k in SCALARS}
        row["physics_tag"] = _tag(d)
        rows.append(row)
    return rows


def cell_summary(runs, metrics=("l2_f", "l2_fp", "l2_f_reweight", "marginal_l2_ref",
                                "n_round_trips", "opes_neff_frac_final")):
    groups = {}
    for d in runs:
        # include stage so tuning seeds (20/21) are never pooled with production (0-9)
        key = (_tag(d), str(_val(d, "name")), str(_val(d, "stage")),
               f"{_val(d,'opes_barrier'):g}", f"{int(_val(d,'opes_pace'))}",
               f"{_val(d,'opes_sigma'):g}", str(_val(d, "opes_gamma")))
        groups.setdefault(key, []).append(d)
    rows = []
    for (tag, name, stage, barrier, pace, sigma, gamma), ds in sorted(groups.items()):
        rec = dict(physics_tag=tag, method=name, stage=stage, opes_barrier=barrier,
                   opes_pace=pace, opes_sigma=sigma, opes_gamma=gamma, n_seeds=len(ds),
                   beta=_val(ds[0], "beta"), h=_val(ds[0], "h"), beta_h=_val(ds[0], "beta_h"))
        for m in metrics:
            med, q25, q75 = _iqr([_val(x, m) for x in ds])
            rec[f"{m}_median"], rec[f"{m}_q25"], rec[f"{m}_q75"] = med, q25, q75
        rows.append(rec)
    return rows


def load_baseline(rep_summary_path):
    """(physics_tag, method) -> (l2_f_median, l2_fp_median) from representative summary."""
    out = {}
    if not os.path.exists(rep_summary_path):
        print(f"[warn] baseline summary not found: {rep_summary_path}")
        return out
    with open(rep_summary_path) as fh:
        for r in csv.DictReader(fh):
            out[(r["physics_tag"], r["method"])] = (
                float(r.get("l2_f_median", "nan")), float(r.get("l2_fp_median", "nan")))
    return out


def vs_baselines(runs, baseline, stage="representative"):
    """Per cell: PRODUCTION OPES median vs ABF vs fr_estimated.

    Uses ONLY held-out production runs (stage=representative, seeds 0-9 at the
    locked hyperparameters) so the comparison is not contaminated by the tuning
    seeds (20/21) that were used to *select* the barrier. Falls back to all
    'opes' runs only if no production runs exist yet (keeps partial-data reports
    working), flagging that case via the 'stage_used' column.
    """
    prod = [d for d in runs if str(_val(d, "name")) == "opes" and str(_val(d, "stage")) == stage]
    used = stage
    if not prod:
        prod = [d for d in runs if str(_val(d, "name")) == "opes"]
        used = "all(tuning-fallback)"
    by_cell = {}
    for d in prod:
        tag = _tag(d)
        hk = (f"{_val(d,'opes_barrier'):g}", f"{int(_val(d,'opes_pace'))}", f"{_val(d,'opes_sigma'):g}")
        by_cell.setdefault(tag, {}).setdefault(hk, []).append(d)
    rows = []
    for tag, hgroups in sorted(by_cell.items()):
        best_hk, best_med, best_fp = None, float("inf"), float("nan")
        for hk, ds in hgroups.items():
            med = _iqr([_val(x, "l2_f") for x in ds])[0]
            if np.isfinite(med) and med < best_med:
                best_med, best_hk = med, hk
                best_fp = _iqr([_val(x, "l2_fp") for x in ds])[0]
        abf = baseline.get((tag, "abf"), (float("nan"), float("nan")))
        mfr = baseline.get((tag, "fr_estimated"), (float("nan"), float("nan")))
        rows.append(dict(
            physics_tag=tag, opes_best_barrier=best_hk[0] if best_hk else "",
            opes_best_pace=best_hk[1] if best_hk else "", opes_best_sigma=best_hk[2] if best_hk else "",
            opes_l2_f=round(best_med, 5), opes_l2_fp=round(best_fp, 5),
            abf_l2_f=round(abf[0], 5), abf_l2_fp=round(abf[1], 5),
            mfr_l2_f=round(mfr[0], 5), mfr_l2_fp=round(mfr[1], 5),
            opes_vs_abf_pct=(round(100 * (abf[0] - best_med) / abf[0], 1) if np.isfinite(abf[0]) and abf[0] else float("nan")),
            opes_vs_mfr_pct=(round(100 * (mfr[0] - best_med) / mfr[0], 1) if np.isfinite(mfr[0]) and mfr[0] else float("nan")),
            n_opes_seeds=(len(hgroups[best_hk]) if best_hk else 0), stage_used=used))
    return rows


def tuning_table(runs):
    rows = []
    groups = {}
    for d in runs:
        if str(_val(d, "stage")) != "tune":
            continue
        key = (_tag(d), f"{_val(d,'opes_barrier'):g}", f"{int(_val(d,'opes_pace'))}")
        groups.setdefault(key, []).append(d)
    for (tag, barrier, pace), ds in sorted(groups.items()):
        med, q25, q75 = _iqr([_val(x, "l2_f") for x in ds])
        rows.append(dict(physics_tag=tag, opes_barrier=barrier, opes_pace=pace, n_seeds=len(ds),
                         l2_f_median=round(med, 5), l2_f_q25=round(q25, 5), l2_f_q75=round(q75, 5),
                         l2_fp_median=round(_iqr([_val(x, "l2_fp") for x in ds])[0], 5),
                         neff_frac_median=round(_iqr([_val(x, "opes_neff_frac_final") for x in ds])[0], 3),
                         round_trips_median=round(_iqr([_val(x, "n_round_trips") for x in ds])[0], 0)))
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-root", default="results/opes_wca")
    p.add_argument("--baseline-summary",
                   default="results/wca_representative/summaries/representative_cells_summary.csv")
    args = p.parse_args(argv)
    raw_dir = os.path.join(args.output_root, "raw")
    sum_dir = os.path.join(args.output_root, "summaries")
    runs = load_runs(raw_dir)
    print(f"[load] {len(runs)} OPES runs from {raw_dir}")
    if not runs:
        return 0
    write_csv(run_rows(runs), os.path.join(sum_dir, "opes_run_summary.csv"), cols=SCALARS)
    write_csv(cell_summary(runs), os.path.join(sum_dir, "opes_cell_summary.csv"))
    baseline = load_baseline(args.baseline_summary)
    write_csv(vs_baselines(runs, baseline), os.path.join(sum_dir, "opes_vs_baselines.csv"))
    write_csv(tuning_table(runs), os.path.join(sum_dir, "opes_tuning.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
