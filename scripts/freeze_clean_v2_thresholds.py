#!/usr/bin/env python3
"""Freeze the clean-v2 accuracy thresholds from the plain-ABF calibration.

The thresholds define what "reaching a given accuracy" *means*, so they are
frozen from Stage 1 -- which contains no FR run at all -- before any FR result
is looked at.  The rule is mechanical:

    eps_{F,1} = median over calibration seeds of e_F^ABF(0.4 T)
    eps_{F,2} = median over calibration seeds of e_F^ABF(0.6 T)

and the same for the mean force.  A threshold is therefore a number that
actually occurred in a plain-ABF run, at a pre-stated fraction of the horizon;
it cannot be tuned to make an arm look good.

Thresholds are frozen for the primary scope *and* both secondary scopes at once.
That is deliberate: if the primary scope were ever switched after seeing
results, the switch would be visible as a choice among values that were all
frozen at the same moment, rather than as a fresh calculation.

The output file is write-once.  ``--force`` exists for a genuine re-freeze (a
changed calibration stage), and it prints what it is discarding.

Example
-------
  python scripts/freeze_clean_v2_thresholds.py \
      --stage-root results/clean_v2/stage1_calibration/calibration \
      --out results/clean_v2/thresholds.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from abffr import accel  # noqa: E402


def _sha(path, n=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(n), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage-root", required=True,
                   help="Stage-1 output directory (contains *_runs_long.csv).")
    p.add_argument("--out", required=True, help="Destination JSON.")
    p.add_argument("--fractions", default=",".join(
        str(f) for f in accel.THRESHOLD_FRACTIONS))
    p.add_argument("--force", action="store_true",
                   help="Overwrite an existing frozen file (prints the diff).")
    return p.parse_args(argv)


def _load_long(stage_root):
    hits = [f for f in os.listdir(stage_root)
            if f.endswith("_runs_long.csv") and "__" not in f]
    if not hits:
        raise SystemExit(f"no merged *_runs_long.csv under {stage_root}")
    path = os.path.join(stage_root, sorted(hits)[0])
    return path, pd.read_csv(path)


def main(argv=None):
    args = parse_args(argv)
    fractions = [float(f) for f in args.fractions.split(",")]
    path, df = _load_long(args.stage_root)

    arms = sorted(df["method"].unique())
    if arms != ["abf_only"]:
        raise SystemExit(
            f"thresholds may only be frozen from a pure plain-ABF stage; "
            f"{os.path.relpath(path)} contains {arms}")

    horizon = float(df["t"].max())
    scopes = [accel.PRIMARY_SCOPE, *accel.SECONDARY_SCOPES]
    out = {
        "primary_scope": accel.PRIMARY_SCOPE,
        "fractions": fractions,
        "consecutive_frames": accel.CONSECUTIVE_FRAMES,
        "horizon": horizon,
        "n_seeds": int(df["seed"].nunique()),
        "seeds": sorted(int(s) for s in df["seed"].unique()),
        "source_csv": os.path.relpath(path),
        "source_sha256_16": _sha(path),
        "thresholds": {},
    }
    for scope in scopes:
        col_F = f"l2_F_{scope}"
        col_Fp = f"l2_Fprime_{scope}"
        if col_F not in df or col_Fp not in df:
            raise SystemExit(f"{path} has no columns for scope {scope!r}")
        entry = {}
        for label, col in (("F", col_F), ("Fprime", col_Fp)):
            curves = [(g["t"].to_numpy(), g[col].to_numpy())
                      for _, g in df.sort_values("t").groupby("seed")]
            entry[label] = accel.freeze_thresholds(curves, fractions, horizon)
            # A threshold is only useful if plain ABF actually reaches it inside
            # T often enough to resolve a difference.  Recording the baseline
            # hit fraction here, at freeze time, means a later low-resolution
            # result is read as "the horizon is too short" rather than as "the
            # threshold was hard".
            entry[f"{label}_hit_fraction"] = [
                float(np.mean([
                    np.isfinite(accel.hitting_time(
                        t, e, eps, consecutive=accel.CONSECUTIVE_FRAMES))
                    for t, e in curves]))
                for eps in entry[label]]
        out["thresholds"][scope] = entry

    if os.path.exists(args.out) and not args.force:
        raise SystemExit(
            f"{args.out} already exists.  Thresholds are write-once: a "
            f"re-freeze after seeing FR results is the defect this refusal "
            f"exists to prevent.  Pass --force only to re-freeze from a changed "
            f"calibration stage.")
    if os.path.exists(args.out):
        old = json.load(open(args.out))
        print("[freeze] --force: DISCARDING the existing frozen thresholds")
        print(f"    old: {json.dumps(old.get('thresholds'), sort_keys=True)}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)

    print(f"[freeze] {out['n_seeds']} plain-ABF calibration seeds, T={horizon:g}")
    print(f"[freeze] primary scope = {accel.PRIMARY_SCOPE}")
    for scope, entry in out["thresholds"].items():
        star = " *" if scope == accel.PRIMARY_SCOPE else "  "
        for label in ("F", "Fprime"):
            vals = ", ".join(
                f"{f:g}T -> {v:.5g} (ABF hits {h:.0%})"
                for f, v, h in zip(fractions, entry[label],
                                   entry[f"{label}_hit_fraction"]))
            print(f"{star} {scope:7s} e_{label:<7s} {vals}")
        worst = min(entry["F_hit_fraction"])
        if scope == accel.PRIMARY_SCOPE and worst < 0.5:
            print(f"   !! plain ABF reaches the stringent free-energy threshold "
                  f"in only {worst:.0%} of calibration seeds within T={horizon:g}. "
                  f"The HORIZON, not the method, is the binding constraint; any "
                  f"speedup measured against it will be restriction-dominated. "
                  f"Say this in the result rather than calling the threshold hard.")
    print(f"[freeze] wrote {os.path.relpath(args.out)} (write-once)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
