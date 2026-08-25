#!/usr/bin/env python
"""Freeze the v3 time-to-accuracy thresholds from plain ABF alone.

Frozen protocol: docs/V3_PREREGISTRATION.md -- "two thresholds fixed
method-blind from the pilot plain-ABF median curves at 60 % and 80 % of budget
and frozen before any FR curve is viewed".

The script refuses to run if any FR result is present in the input directory, and
refuses to overwrite an existing artifact.  Those two refusals are the whole
point: a threshold that can be recomputed after seeing FR data is not
method-blind, however sincerely it was first computed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd

BUDGET_FRACTIONS = (0.60, 0.80)      # frozen
SCOPE_NAME = "R12"                   # beta (F_ref - min F_ref) <= 12


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:                                   # pragma: no cover
        return "unknown"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-long", required=True,
                   help="plain-ABF runs_long CSV (no FR arms may be present)")
    p.add_argument("--out", default="results/v3/V3_THRESHOLDS.json")
    args = p.parse_args(argv)

    out = pathlib.Path(args.out)
    if out.exists():
        print(f"REFUSING: {out} already exists. The thresholds are frozen once; "
              f"recomputing them after any FR result exists would silently void "
              f"the method-blind guarantee.", file=sys.stderr)
        return 2

    src = pathlib.Path(args.runs_long)
    df = pd.read_csv(src)
    methods = sorted(df["method"].astype(str).unique())
    contaminated = [m for m in methods if m != "abf_only" and "fr" in m.lower()]
    if contaminated or methods != ["abf_only"]:
        print(f"REFUSING: input must contain plain ABF only; found {methods}",
              file=sys.stderr)
        return 2

    n_steps = int(df["step"].max())
    thresholds = {}
    # The frozen primary scope is R12, NOT the engine's default evaluation mask.
    for observable, col in (("F", "l2_F_R12"), ("Fprime", "l2_Fprime_R12")):
        if col not in df.columns:
            raise ValueError(
                f"{col} absent: these runs predate the scoped metrics, so "
                f"freezing from them would silently use the wrong scope")
        med = df.groupby("step")[col].median()
        for i, frac in enumerate(BUDGET_FRACTIONS, start=1):
            target_step = int(round(frac * n_steps))
            if target_step not in med.index:
                raise ValueError(
                    f"budget fraction {frac} maps to step {target_step}, "
                    f"which is not a saved frame")
            thresholds[f"eps_{observable}_{i}"] = dict(
                value=float(med.loc[target_step]),
                budget_fraction=frac, step=target_step,
                frame_index=int(list(med.index).index(target_step)))

    artifact = dict(
        thresholds=thresholds,
        scope=SCOPE_NAME,
        observables=["F", "Fprime"],
        budget_fractions=list(BUDGET_FRACTIONS),
        seeds=sorted(int(s) for s in df["seed"].unique()),
        n_steps=n_steps,
        source_csv=str(src), source_sha256=_sha256(src),
        analysis_script_sha256=_sha256(pathlib.Path(__file__)),
        git_commit=_git_commit(),
        created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        frozen_before_fr=True,
        # Amendment 5: the sequencing travels with the numbers.
        offline_fr_benchmark_viewed_before_freeze=True,
        online_fr_results_viewed_before_freeze=False,
        note=("Computed from plain-ABF median curves only. Never overwrite: the "
              "value of these numbers is that they could not have been chosen "
              "with knowledge of any FR result."))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2))
    print(f"[thresholds] wrote {out}")
    for k, v in thresholds.items():
        print(f"  {k:14s} = {v['value']:.6f}  (step {v['step']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
