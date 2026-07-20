#!/usr/bin/env python3
"""Aggregate alkane raw runs into summary CSVs + matched-seed paired statistics.

Outputs (under <output_root>/summaries/):
  alkanes_runs_long.csv      one row per (job, seed): all per-seed metrics + physics
  alkanes_config_summary.csv per (cell, method): median / IQR over seeds
  alkanes_paired.csv         matched-seed deltas of each method vs ABF + bootstrap CI
  alkanes_equivalence.csv    butane practical-equivalence test (10% margin) per cell
  alkanes_main.csv           headline comparison table

No GPU. Reads <output_root>/raw/*.npz.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from alkanes import jobs as J  # noqa: E402

RNG = np.random.default_rng(20260719)


def load_long(raw_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(raw_dir, "*.npz"))):
        try:
            d = np.load(path, allow_pickle=True)
        except Exception:
            continue
        if "per_seed" not in d.files:
            continue
        per_seed = json.loads(str(d["per_seed"]))
        meta = dict(molecule=str(d["molecule"]), method=str(d["method"]), name=str(d["name"]),
                    init_mode=str(d["init_mode"]), beta=float(d["beta"]), sigma=float(d["sigma"]),
                    decouple=bool(d["decouple"]), stage=str(d["stage"]), n_steps=int(d["n_steps"]),
                    n_replicas=int(d["n_replicas"]), run_id=str(d["run_id"]),
                    wall_seconds=float(d["wall_seconds"]))
        cell = f"{meta['molecule']}_b{meta['beta']:g}_s{meta['sigma']:g}_{'dec' if meta['decouple'] else 'full'}_{meta['init_mode']}"
        for rec in per_seed:
            row = dict(meta); row.update(rec); row["cell"] = cell
            rows.append(row)
    return pd.DataFrame(rows)


def config_summary(df):
    metrics = [c for c in df.columns if df[c].dtype.kind in "fi" and c not in
               ("beta", "sigma", "n_steps", "n_replicas", "seed", "wall_seconds")]
    g = df.groupby(["cell", "molecule", "beta", "sigma", "decouple", "init_mode", "name", "method"])
    out = g[metrics].median().add_suffix("_med")
    iqr = g[metrics].quantile(0.75) - g[metrics].quantile(0.25)
    out = out.join(iqr[metrics].add_suffix("_iqr"))
    out["n_seeds"] = g.size()
    return out.reset_index()


def _boot_ci(x, n=10000):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    idx = RNG.integers(0, len(x), size=(n, len(x)))
    bs = x[idx].mean(1)
    return float(np.median(x)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def paired(df, key="final_l2_F"):
    """Matched-seed delta of each method vs ABF for the given metric (per cell)."""
    rows = []
    for cell, sub in df.groupby("cell"):
        abf = sub[sub.method == "abf"].set_index("seed")[key]
        if abf.empty:
            continue
        for method, m in sub.groupby("name"):
            if method == "abf":
                continue
            mm = m.set_index("seed")[key]
            common = abf.index.intersection(mm.index)
            if len(common) == 0:
                continue
            a = abf.loc[common].values
            b = mm.loc[common].values
            # positive delta => method WORSE than ABF (higher L2); relative change
            delta = b - a
            rel = (b - a) / np.where(np.abs(a) > 1e-12, np.abs(a), np.nan)
            med, lo, hi = _boot_ci(delta)
            rmed, rlo, rhi = _boot_ci(rel)
            win = float(np.mean(b < a))     # fraction of seeds where method beats ABF
            rows.append(dict(cell=cell, method=method, metric=key, n_pairs=len(common),
                             abf_med=float(np.median(a)), method_med=float(np.median(b)),
                             delta_med=med, delta_lo=lo, delta_hi=hi,
                             rel_med=rmed, rel_lo=rlo, rel_hi=rhi, win_rate=win))
    return pd.DataFrame(rows)


def equivalence(df, margin=0.10):
    """Butane practical-equivalence: is |rel change of mFR vs ABF| within +-margin?

    Uses the two-one-sided (TOST) idea via the bootstrap CI of the relative change in
    final and integrated L2(F): equivalent if the 95% CI lies within [-margin, margin].
    """
    rows = []
    for key in ("final_l2_F", "integrated_l2_F"):
        p = paired(df, key)
        for _, r in p.iterrows():
            within = (r["rel_lo"] >= -margin) and (r["rel_hi"] <= margin)
            harmful = r["rel_lo"] > margin      # CI entirely above +margin => worse
            better = r["rel_hi"] < -margin      # CI entirely below -margin => better
            verdict = ("equivalent" if within else "harmful" if harmful
                       else "improved" if better else "inconclusive")
            rows.append(dict(cell=r["cell"], method=r["method"], metric=key,
                             rel_med=r["rel_med"], rel_lo=r["rel_lo"], rel_hi=r["rel_hi"],
                             margin=margin, verdict=verdict, win_rate=r["win_rate"]))
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--margin", type=float, default=0.10)
    args = ap.parse_args(argv)
    cfg = J.load_yaml(args.config)
    root = cfg["output_root"]
    raw_dir = os.path.join(root, "raw")
    out_dir = os.path.join(root, "summaries")
    os.makedirs(out_dir, exist_ok=True)
    df = load_long(raw_dir)
    if df.empty:
        print("no runs found in", raw_dir); return 1
    df.to_csv(os.path.join(out_dir, "alkanes_runs_long.csv"), index=False)
    config_summary(df).to_csv(os.path.join(out_dir, "alkanes_config_summary.csv"), index=False)
    pair_all = pd.concat([paired(df, k) for k in ("final_l2_F", "integrated_l2_F", "final_l2_Fp")],
                         ignore_index=True)
    pair_all.to_csv(os.path.join(out_dir, "alkanes_paired.csv"), index=False)
    equivalence(df, args.margin).to_csv(os.path.join(out_dir, "alkanes_equivalence.csv"), index=False)
    # main headline table: median final/integrated L2(F) per cell x method
    main_cols = ["final_l2_F", "integrated_l2_F", "final_l2_Fp", "n_transitions",
                 "n_round_trips", "final_ancestor_ess", "fr_event_fraction",
                 "cond_tv_weighted", "cond_basin_err_weighted", "n_basins_visited"]
    have = [c for c in main_cols if c in df.columns]
    main = df.groupby(["cell", "name"])[have].median().reset_index()
    main.to_csv(os.path.join(out_dir, "alkanes_main.csv"), index=False)
    print(f"[analyze] {len(df)} run-seeds -> {out_dir}")
    print(main.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
