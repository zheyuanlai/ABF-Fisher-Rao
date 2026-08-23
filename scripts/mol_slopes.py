"""Late-time log-log slope of e_F: does an arm have a bias floor or not?

-0.5 is pure statistics (an unbiased estimator with independent samples); 0 is a
bias plateau that no amount of compute removes.  This is the sharpest single
number the multi-budget protocol produces, and it is why the protocol requires
more than one budget.
"""
from __future__ import annotations

import argparse, glob, json, os, sys

import numpy as np


def slope(fe, e, span=4.0):
    """Fit over the last factor-`span` in force evaluations; a shorter window is
    too small a lever arm in log-fe to separate -0.5 from 0."""
    m = fe >= fe[-1] / span
    return float(np.polyfit(np.log(fe[m]), np.log(np.median(e[m], -1)), 1)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="PEN")
    ap.add_argument("--tag", default="confirm")
    ap.add_argument("--dir", default="results/mol/campaign")
    a = ap.parse_args()
    print(f"| arm | e_F at max budget | late-time d log e_F / d log fe | reading |")
    print("|---|---|---|---|")
    rows = []
    for p in sorted(glob.glob(os.path.join(a.dir, f"{a.system}_*_{a.tag}.npz"))):
        arm = os.path.basename(p)[len(a.system) + 1:-(len(a.tag) + 5)]
        d = np.load(p)
        n_cfg = int(d["n_cfg"]) if "n_cfg" in d else 1
        ns = d["e_F"].shape[1] // n_cfg
        e = d["e_F"].reshape(d["e_F"].shape[0], n_cfg, ns)[:, 0]
        s = slope(d["fe"], e)
        rows.append((float(np.median(e[-1])), arm, s))
    for v, arm, s in sorted(rows):
        tag = ("still converging" if s < -0.35 else
               "bias floor" if s > -0.12 else "partly bias-limited")
        print(f"| {arm} | {v:.4f} | {s:+.3f} | {tag} |")


if __name__ == "__main__":
    main()
