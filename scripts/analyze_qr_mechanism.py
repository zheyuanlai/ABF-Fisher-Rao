#!/usr/bin/env python
"""Stage 0.5: does the realisation mechanism decide the sign, at identical r*?

Frozen protocol: ``docs/QR_DECOUPLING_PREREGISTRATION.md``, Amendment 1 / H7.

A4a and A6a carry the same ``r*``; so do A4b and A6b.  They differ only in how
that allocation is realised -- birth--death against the dynamics, or a bias that
makes it stationary.  Paired on seed, because the arms share matched noise and a
paired comparison is the one the design supports.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

ROOT = "results/qr_decoupling/stage05_mechanism"


def load(cell, arm):
    p = os.path.join(ROOT, cell, arm, "summary.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def paired_ratio(a, b, n_boot=10_000, seed=0):
    """``median(a/b)`` over shared seeds, with a paired bootstrap CI."""
    m = a.merge(b, on="seed", suffixes=("_a", "_b"))
    r = m.e_F_final_a.values / m.e_F_final_b.values
    rng = np.random.default_rng(seed)
    boot = [np.median(rng.choice(r, r.size, replace=True)) for _ in range(n_boot)]
    return float(np.median(r)), float(np.percentile(boot, 2.5)), \
        float(np.percentile(boot, 97.5)), len(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", default=["K0", "K2"])
    args = ap.parse_args()

    print(f"{'cell':>5} {'pair':>12} {'e_F bd':>9} {'e_F bias':>9} "
          f"{'ratio bd/bias':>14} {'95% CI':>20} {'ancESS bd':>10} {'repl bd':>8}")
    for cell in args.cells:
        for bd, bias in (("A4a", "A6a"), ("A4b", "A6b")):
            A, B = load(cell, bd), load(cell, bias)
            if A is None or B is None:
                print(f"{cell:>5} {bd+'/'+bias:>12}   (missing)")
                continue
            ratio, lo, hi, n = paired_ratio(A, B)
            print(f"{cell:>5} {bd+'/'+bias:>12} {A.e_F_final.median():9.4f} "
                  f"{B.e_F_final.median():9.4f} {ratio:14.2f} "
                  f"{f'[{lo:.2f}, {hi:.2f}]':>20} "
                  f"{A.ancestor_ess_final.median():10.1f} "
                  f"{A.n_replacements.median():8.0f}  (n={n})")
    print("\nH7 predicts ratio > 1 (birth-death worse) with the CI clear of 1.")
    print("Full-domain safety metric (Amendment 1 requires it beside the primary):")
    for cell in args.cells:
        for arm in ("A4a", "A6a", "A4b", "A6b"):
            d = load(cell, arm)
            if d is not None:
                print(f"  {cell} {arm:>4}: primary {d.e_F_final.median():.4f}  "
                      f"full-domain {d.e_F_full_final.median():.4f}")


if __name__ == "__main__":
    main()
