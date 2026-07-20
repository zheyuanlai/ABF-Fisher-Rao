#!/usr/bin/env python3
"""Select frozen hyperparameters from the tuning stage (pre-registered rule).

Rule: smallest MEDIAN integrated L2(F) on the tuning seeds, subject to stability
(no NaN) and genealogy safety (ancestor ESS not collapsed below a floor, event cap
not chronically binding). mFR rate selected on the PENTANE cell (the demanding case);
OPES config selected on pentane too. Also reports the best-final-L2 selection as a
labelled supplement. Prints the values to freeze into configs/alkanes/production.yaml.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from alkanes import jobs as J  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/alkanes/tuning.yaml")
    ap.add_argument("--ess-floor", type=float, default=0.15, help="min ancestor ESS fraction")
    args = ap.parse_args(argv)
    cfg = J.load_yaml(args.config)
    summ = os.path.join(cfg["output_root"], "summaries", "alkanes_config_summary.csv")
    if not os.path.exists(summ):
        print("run analyze_alkanes.py on the tuning config first"); return 1
    df = pd.read_csv(summ)

    def pick(sub, key):
        sub = sub.copy()
        # genealogy safety filter for FR rows
        if "final_ancestor_ess_med" in sub and "n_replicas" in sub:
            frac = sub["final_ancestor_ess_med"] / sub["n_replicas"]
            safe = sub[(frac.isna()) | (frac >= args.ess_floor)]
            sub = safe if len(safe) else sub
        sub = sub[np.isfinite(sub[key])]
        return sub.loc[sub[key].idxmin()] if len(sub) else None

    print("=" * 70)
    for mol in ("pentane", "butane"):
        fr = df[(df.molecule == mol) & (df.name.str.startswith("fr_r"))]
        if len(fr):
            print(f"\n[{mol}] mFR rate ladder (median over tuning seeds):")
            for _, r in fr.sort_values("name").iterrows():
                rate = float(r["name"].replace("fr_r", "")) / 100.0
                print(f"   rate={rate:.2f}: intL2F={r.get('integrated_l2_F_med', np.nan):.4f} "
                      f"finalL2F={r.get('final_l2_F_med', np.nan):.4f} "
                      f"ess={r.get('final_ancestor_ess_med', np.nan):.0f} "
                      f"evt={r.get('fr_event_fraction_med', np.nan):.4f} "
                      f"condTV={r.get('cond_tv_weighted_med', np.nan):.4f}")
            best = pick(fr, "integrated_l2_F_med")
            best_final = pick(fr, "final_l2_F_med")
            if best is not None:
                rate = float(best["name"].replace("fr_r", "")) / 100.0
                rate_f = float(best_final["name"].replace("fr_r", "")) / 100.0
                print(f"   -> PRIMARY (min int L2F): fr_rate={rate:.2f}")
                print(f"   -> supplement (min final L2F): fr_rate={rate_f:.2f}")
        op = df[(df.molecule == mol) & (df.name.str.startswith("opes"))]
        if len(op):
            print(f"[{mol}] OPES grid:")
            for _, r in op.sort_values("name").iterrows():
                print(f"   {r['name']}: intL2F={r.get('integrated_l2_F_med', np.nan):.4f} "
                      f"finalL2F={r.get('final_l2_F_med', np.nan):.4f}")
            bo = pick(op, "integrated_l2_F_med")
            if bo is not None:
                print(f"   -> OPES best: {bo['name']}")
    print("\nFreeze the PRIMARY selections into configs/alkanes/production.yaml methods block.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
