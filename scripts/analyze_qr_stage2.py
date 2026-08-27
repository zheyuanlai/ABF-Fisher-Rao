#!/usr/bin/env python
"""Stage 2: the confirmatory verdict, against the frozen thresholds.

Frozen protocol: ``docs/QR_DECOUPLING_PREREGISTRATION.md`` (Amendment 1).

Primary endpoint is **time-to-accuracy**, not error::

    tau_eps = first saved frame t_j with e_F(t_j), e_F(t_{j+1}), e_F(t_{j+2}) <= eps
    S_eps   = E[min(tau_baseline, T)] / E[min(tau_method, T)]

Three consecutive frames because a single dip below a threshold is noise, and
the restricted mean because runs that never arrive must still count -- dropping
them would score a method on the subset of seeds where it happened to work.

**Censoring is reported per arm and can veto a positive.** If the candidate
fails to reach a threshold more often than its baseline does, that threshold
returns no verdict however good the ratio looks: the clean-v2 amendment, which
exists because the censoring in that campaign flattered the arm under test.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

ROOT = "results/qr_decoupling/stage2"


def tau(df_seed, eps, T, n_consec=3):
    """First frame from which the error stays under ``eps`` for 3 frames."""
    t = df_seed.t.values
    e = df_seed.e_F.values
    for k in range(len(t) - n_consec + 1):
        if np.all(e[k:k + n_consec] <= eps):
            return float(t[k])
    return float(T)                       # censored at T, not dropped


def restricted_speedup(base, meth, eps, T, n_boot=10_000, seed=0):
    b = np.array([tau(base[base.seed == s], eps, T) for s in sorted(base.seed.unique())])
    m = np.array([tau(meth[meth.seed == s], eps, T) for s in sorted(meth.seed.unique())])
    n = min(b.size, m.size)
    b, m = b[:n], m[:n]
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)         # paired on seed
        boot.append(b[i].mean() / max(m[i].mean(), 1e-12))
    return (float(b.mean() / m.mean()), float(np.percentile(boot, 2.5)),
            float(np.percentile(boot, 97.5)),
            float(np.mean(b < T)), float(np.mean(m < T)), n)


def load(cell, arm):
    p = os.path.join(ROOT, cell, arm, "profiles.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", default=["K0", "K2", "K3", "K1"])
    ap.add_argument("--thresholds", default="results/qr_decoupling/thresholds.json")
    ap.add_argument("--pairs", nargs="+",
                    default=["A6a/A0", "A6b/A0", "A6c/A0", "A6b/A6a", "A6c/A6b"])
    args = ap.parse_args()
    thr = json.load(open(args.thresholds))

    print(f"{'cell':>5} {'contrast':>10} {'eps':>7} {'S':>7} {'95% CI':>16} "
          f"{'P(hit) base':>12} {'P(hit) meth':>12}  verdict")
    for cell in args.cells:
        if cell not in thr:
            continue
        T = thr[cell]["T"]
        for name in args.pairs:
            meth_a, base_a = name.split("/")
            meth, base = load(cell, meth_a), load(cell, base_a)
            if meth is None or base is None:
                continue
            for key in ("eps_1", "eps_2"):
                eps = thr[cell][key]
                S, lo, hi, pb, pm, n = restricted_speedup(base, meth, eps, T)
                if pm < pb - 1e-9:
                    verdict = "NO VERDICT (candidate censored more)"
                elif lo > 1.0:
                    verdict = "faster"
                elif hi < 1.0:
                    verdict = "SLOWER"
                else:
                    verdict = "tie"
                print(f"{cell:>5} {name:>10} {eps:7.4f} {S:7.3f} "
                      f"{f'[{lo:.2f}, {hi:.2f}]':>16} {pb:12.2f} {pm:12.2f}  "
                      f"{verdict}")
    print("\nH1 K0: A6b/A6a in [0.95, 1.05].  H2 K2/K3: A6b/A6a >= 1.10, CI > 1.")
    print("H3: A6b/A0 >= 1.15.  H4: A6c retains >= 80% of A6b's margin over A0.")


if __name__ == "__main__":
    main()
