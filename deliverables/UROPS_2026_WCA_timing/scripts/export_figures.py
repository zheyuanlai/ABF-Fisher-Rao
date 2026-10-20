#!/usr/bin/env python
"""Export the UROPS-compatible three-panel CSVs and the master table from data/per_seed/ (run after
analyze_campaign.py).

Fixed, pre-specified comparisons against ABF (prefix -> FR arm), independent of the outcome:
  wca_start0 -> fr_start0, wca_stop20 -> fr_start0_stop20, wca_stop40 -> fr_start0_stop40, wca_start40 -> fr_start40,
  wca_start5 -> fr_start5, wca_start20 -> fr_start20 (secondary arms).
Each: data/threepanel/<prefix>_{eF,eFp,exi}.csv with t,abf_median,abf_q25,abf_q75,fr_median,fr_q25,fr_q75 (median
and 25 / 75 % percentiles across the 16 paired seeds at every saved checkpoint; exi = the FULL-domain marginal).
Master: data/summary/master_series.csv (t, arm, metric, median, q25, q75, n_seeds) for every arm and series.
"""
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common_wca as CW  # noqa: E402

PS = os.path.join(CW.PKG, "data", "per_seed")
TP = os.path.join(CW.PKG, "data", "threepanel")
SUM = os.path.join(CW.PKG, "data", "summary")
PREFIX = {"wca_start0": "fr_start0", "wca_stop20": "fr_start0_stop20", "wca_stop40": "fr_start0_stop40",
          "wca_start40": "fr_start40", "wca_start5": "fr_start5", "wca_start20": "fr_start20"}
PANELS = {"eF": "eF", "eFp": "eFp", "exi": "exi_full"}
SERIES = ("eF", "eFp", "exi_full", "exi_interior", "edge_frac", "frac_compact", "frac_transition", "frac_stretched",
          "repl_cum", "ess_window", "ess_run", "maxfam_run", "nuniq", "clip_frac_cum")
COLS = ("t", "abf_median", "abf_q25", "abf_q75", "fr_median", "fr_q25", "fr_q75")


def read(arm, k):
    with open(os.path.join(PS, f"{arm}__{k}.csv")) as fh:
        r = csv.reader(fh)
        head = next(r)
        a = np.array([[float(v) for v in row] for row in r])
    return a[:, 0], a[:, 1:], head[1:]


def q(x):
    return np.percentile(x, [25, 50, 75], axis=1)


def f(v):
    return f"{float(v):.10g}"


def main():
    os.makedirs(TP, exist_ok=True)
    arms = [a["name"] for a in CW.frozen()["arms"] if a["run"]]
    written = []
    for pre, arm in PREFIX.items():
        if arm not in arms:
            continue
        for panel, key in PANELS.items():
            t, A, sa = read("abf", key)
            t2, B, sb = read(arm, key)
            assert np.array_equal(t, t2) and sa == sb
            assert np.all(np.isfinite(A)) and np.all(np.isfinite(B)), f"{pre} {panel}: non-finite per-seed value"
            qa, qb = q(A), q(B)
            p = os.path.join(TP, f"{pre}_{panel}.csv")
            with open(p, "w") as fh:
                fh.write(",".join(COLS) + "\n")
                for i in range(t.size):
                    fh.write(",".join(f(v) for v in (t[i], qa[1][i], qa[0][i], qa[2][i], qb[1][i], qb[0][i], qb[2][i])) + "\n")
            written.append(os.path.relpath(p, CW.PKG))
    with open(os.path.join(SUM, "master_series.csv"), "w") as fh:
        fh.write("t,arm,metric,median,q25,q75,n_seeds\n")
        for arm in arms:
            for k in SERIES:
                t, X, s = read(arm, k)
                ok = np.isfinite(X).all(0) if k not in ("ess_window", "ess_run", "maxfam_run") or arm != "abf" else None
                if ok is None:
                    continue
                qq = np.nanpercentile(X, [25, 50, 75], axis=1)
                for i in range(t.size):
                    fh.write(f"{f(t[i])},{arm},{k},{f(qq[1][i])},{f(qq[0][i])},{f(qq[2][i])},{int(np.isfinite(X[i]).sum())}\n")
    print("wrote", len(written), "three-panel CSVs and data/summary/master_series.csv")


if __name__ == "__main__":
    main()
