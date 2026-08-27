#!/usr/bin/env python
"""Freeze the Stage-2 accuracy thresholds from Stage-1 plain-ABF calibration.

Frozen protocol: ``docs/QR_DECOUPLING_PREREGISTRATION.md``.

    eps_1 = median_seeds e_F^A0(0.4 T)      lenient
    eps_2 = median_seeds e_F^A0(0.6 T)      stringent

One pair **per kappa cell**, because the cells differ in how hard sampling is
and a threshold shared across them would silently make one cell easy and
another impossible.  These must be written before any candidate arm's error is
inspected -- that ordering is the whole point of a calibration stage, and it is
what stops a threshold from being chosen where a margin happens to sit.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/qr_decoupling/stage1_calibration")
    ap.add_argument("--out", default="results/qr_decoupling/thresholds.json")
    ap.add_argument("--cells", nargs="+", default=["K0", "K1", "K2", "K3"])
    args = ap.parse_args()

    frozen = {}
    for cell in args.cells:
        path = os.path.join(args.root, cell, "profiles.csv")
        if not os.path.exists(path):
            print(f"{cell}: MISSING {path}")
            continue
        df = pd.read_csv(path)
        T = df.t.max()
        row = {}
        for name, frac in (("eps_1", 0.4), ("eps_2", 0.6)):
            t_target = frac * T
            k = df.t.unique()[np.argmin(np.abs(df.t.unique() - t_target))]
            at = df[df.t == k]
            row[name] = float(at.e_F.median())
            row[name + "_full"] = float(at.e_F_full.median())
            row[name + "_t"] = float(k)
        at_T = df[df.t == T]
        row["e_F_at_T_median"] = float(at_T.e_F.median())
        row["n_seeds"] = int(df.seed.nunique())
        row["T"] = float(T)
        frozen[cell] = row
        print(f"{cell}: eps_1={row['eps_1']:.4f} (t={row['eps_1_t']:.0f})  "
              f"eps_2={row['eps_2']:.4f} (t={row['eps_2_t']:.0f})  "
              f"e_F(T)={row['e_F_at_T_median']:.4f}  n={row['n_seeds']}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if os.path.exists(args.out):
        print(f"\nREFUSING to overwrite {args.out}: thresholds are frozen once. "
              f"Delete it deliberately if a recalibration is really intended.")
        return
    with open(args.out, "w") as fh:
        json.dump(frozen, fh, indent=2)
    print(f"\nfrozen -> {args.out}")


if __name__ == "__main__":
    main()
