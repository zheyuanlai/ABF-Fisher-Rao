"""C0 — rescore the STORED Case IX final PMFs against a new reference.

    python scripts/rescore_caseix_c0.py --reference cache/phase_hp_v3/wca_ti_b1_h2_w2_n10_a1.5_g160.npz

**This is not C1 and cannot replace C2.** The Case IX headline is the *time-integrated*
`I_F = int_0^T ||F_hat_t - F_ref||_L2 dt`, and the raw artifacts retain only the already-scored
`l2_f_t` scalars -- never `F_hat_t(z)`. That endpoint is unrecoverable without re-running the
dynamics.

What *is* stored is `final_pmf`, so the **final-time** error can be rescored for free:

    e_final = || F_hat_T - F_ref ||_L2

That is a different endpoint from the headline, and it is reported as such. Its value is
directional: it says whether swapping the reference leaves the arm ordering intact, collapses
mFR toward ABF, or reverses anything -- before spending an hour of GPU on C2.

Naming it C0 rather than C1 is deliberate: C1 (integrated-error rescoring) is impossible here,
and calling this C1 would imply the headline had been recomputed when it has not.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

ARMS = ("abf", "fr_estimated", "fr_oracle", "sham_practical", "sham_oracle")


def load_arm(raw_dir, arm):
    """Return {seed: final_pmf} for one arm."""
    out = {}
    for f in sorted(glob.glob(os.path.join(raw_dir, f"*__{arm}__*.npz"))):
        m = re.search(r"seed(\d+)", os.path.basename(f))
        if not m:
            continue
        z = np.load(f, allow_pickle=True)
        if "final_pmf" not in z.files:
            continue
        out[int(m.group(1))] = np.asarray(z["final_pmf"], dtype=np.float64)
    return out


def l2(a, b, dz, mask):
    """L2 distance after removing the additive constant on the evaluation mask."""
    a = a - a[mask].mean()
    b = b - b[mask].mean()
    return float(np.sqrt(((a - b)[mask] ** 2).sum() * dz))


def boot_median(x, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    m = np.median(np.asarray(x)[idx], axis=1)
    return float(np.median(x)), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results/wca_sham/sham/raw")
    ap.add_argument("--reference", required=True)
    ap.add_argument("--cached", default="cache/phase/wca_ti_b1_h2_w2_n10_a1.5_g160.npz")
    ap.add_argument("--out", default="results/v2_validity_audits/caseix_c0")
    ap.add_argument("--z-lo", type=float, default=-0.15, help="evaluation mask, edges excluded")
    ap.add_argument("--z-hi", type=float, default=1.15)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    new = np.load(args.reference, allow_pickle=True)
    old = np.load(args.cached, allow_pickle=True)
    grid = new["grid"]
    assert np.allclose(grid, old["grid"]), "reference grids differ; cannot compare like with like"
    dz = float(grid[1] - grid[0])
    mask = (grid >= args.z_lo) & (grid <= args.z_hi)
    F_new, F_old = new["free_energy"], old["free_energy"]
    print(f"reference: {args.reference}")
    print(f"  L2(F_new - F_old) on the mask: "
          f"{l2(F_new, F_old, dz, mask):.4f}   mask covers {int(mask.sum())}/{grid.size} points\n")

    arms = {a: load_arm(args.raw, a) for a in ARMS}
    arms = {a: v for a, v in arms.items() if v}
    seeds = sorted(set.intersection(*[set(v) for v in arms.values()]))
    print(f"arms {list(arms)}   paired seeds {len(seeds)}\n")

    e_old = {a: np.array([l2(arms[a][s], F_old, dz, mask) for s in seeds]) for a in arms}
    e_new = {a: np.array([l2(arms[a][s], F_new, dz, mask) for s in seeds]) for a in arms}

    print(f"{'arm':>16} {'e_final OLD':>12} {'e_final NEW':>12}")
    for a in arms:
        print(f"{a:>16} {np.median(e_old[a]):12.4f} {np.median(e_new[a]):12.4f}")

    def contrast(x, y, label):
        d = 100.0 * (np.asarray(x) - np.asarray(y)) / np.asarray(y)
        med, lo, hi = boot_median(d)
        wins = int((d < 0).sum())
        print(f"  {label:<34} {med:+8.2f} %   95% CI [{lo:+.2f}, {hi:+.2f}]   "
              f"{wins}/{len(d)} seeds better")
        return dict(median_pct=med, ci95=[lo, hi], wins=wins, n=len(d))

    res = {}
    print(f"\n--- FINAL-TIME contrasts, OLD reference (NOT the headline endpoint) ---")
    for arm, base, lab in (("fr_estimated", "abf", "mFR vs ABF"),
                           ("fr_estimated", "sham_practical", "mFR vs its own sham"),
                           ("sham_practical", "abf", "sham vs ABF")):
        if arm in e_old and base in e_old:
            res[f"old::{lab}"] = contrast(e_old[arm], e_old[base], lab)
    print(f"\n--- FINAL-TIME contrasts, NEW reference ---")
    for arm, base, lab in (("fr_estimated", "abf", "mFR vs ABF"),
                           ("fr_estimated", "sham_practical", "mFR vs its own sham"),
                           ("sham_practical", "abf", "sham vs ABF")):
        if arm in e_new and base in e_new:
            res[f"new::{lab}"] = contrast(e_new[arm], e_new[base], lab)

    ok = True
    for lab in ("mFR vs ABF", "mFR vs its own sham"):
        a, b = res.get(f"old::{lab}"), res.get(f"new::{lab}")
        if a and b and np.sign(a["median_pct"]) != np.sign(b["median_pct"]):
            ok = False
    print(f"\n  sign of every primary contrast preserved: {ok}")
    print(f"  NOTE: this is the FINAL-TIME endpoint. The headline -22.83 % is the TIME-INTEGRATED"
          f"\n        endpoint and is NOT recoverable from stored artifacts. C2 remains required.")

    with open(os.path.join(args.out, "c0_verdict.json"), "w") as fh:
        json.dump(dict(reference=args.reference, n_seeds=len(seeds), arms=list(arms),
                       mask=[args.z_lo, args.z_hi], contrasts=res,
                       signs_preserved=ok,
                       endpoint="final-time L2(F); NOT the time-integrated headline"), fh, indent=2)
    print(f"\nwrote {args.out}/c0_verdict.json")


if __name__ == "__main__":
    main()
