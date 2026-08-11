"""Compare the guarded R15 ABF audit against the frozen v1 baseline, four-way.

    python scripts/compare_r15_guard_audit.py

v1 is immutable and is only read. The audit changes exactly one thing -- the standard ABF
`fullSamples` guard, `abf_min_count_dist = 200` -- and this script asks whether the v1
classification survives it.

Per **Amendment 8**, the classification is now **four-way**, so `Gate 0` is applied first: an
ABF baseline whose learned bias is untrustworthy is `conditional-equilibration-limited` or
`ABF-baseline-invalid`, and must not be forced into "discovery" or "establishment".
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

V1 = "results/alkanes_cv_extension/r15/summaries/cv_starvation.csv"
AUDIT_DIR = "results/v2_validity_audits/r15_abf_guard"

#: Amendment 7 Gate 0, transferred to R15. The learned bias must span a sane fraction of the
#: reference span, and no cell may park essentially the whole population in one region.
GATE0_SPAN_LO, GATE0_SPAN_HI = 0.75, 1.25


def _load(path):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def main():
    v1 = _load(V1)
    au = _load(os.path.join(AUDIT_DIR, "summaries", "cv_starvation.csv"))
    if v1 is None:
        raise SystemExit(f"missing frozen v1 baseline at {V1}")
    if au is None:
        raise SystemExit(f"missing audit summary; run the analyzer on {AUDIT_DIR} first")

    key = "cell"
    v1 = v1.set_index(key)
    au = au.set_index(key)
    # audit cells carry the same names; align on the shared set
    common = [c for c in v1.index if c in au.index]
    print(f"cells compared: {len(common)} of {len(v1.index)} v1 cells\n")

    cols = ["verdict", "norm_final_l2_F", "n_criteria_fired",
            "c1_normL2_above_floor", "c3_low_support_20pct", "c4_poor_mixing"]
    print(f"{'cell':<42} {'v1':>10} {'guarded':>10} {'v1 L2':>8} {'grd L2':>8} {'flip':>6}")
    flips = []
    for c in common:
        a, b = v1.loc[c], au.loc[c]
        flip = "YES" if a["verdict"] != b["verdict"] else ""
        if flip:
            flips.append((c, a["verdict"], b["verdict"]))
        print(f"{c:<42} {a['verdict']:>10} {b['verdict']:>10} "
              f"{a['norm_final_l2_F']:>8.4f} {b['norm_final_l2_F']:>8.4f} {flip:>6}")

    print("\n--- criteria that fired ---")
    for c in common:
        a, b = v1.loc[c], au.loc[c]
        fa = [k for k in cols[3:] if bool(a.get(k, False))]
        fb = [k for k in cols[3:] if bool(b.get(k, False))]
        if fa or fb:
            print(f"  {c}\n      v1      : {fa or ['none']}\n      guarded : {fb or ['none']}")

    verdict = dict(n_cells=len(common), flips=[dict(cell=c, v1=x, guarded=y)
                                              for c, x, y in flips])
    if not flips:
        verdict["conclusion"] = (
            "v1 R15 classification SURVIVES the ABF fullSamples guard; the discovery-limited "
            "pillar stands")
    else:
        verdict["conclusion"] = (
            "v1 R15 classification CHANGES under the guard; the discovery-limited pillar "
            "requires a corrected mFR/sham rerun before manuscript use")
    print(f"\n=== {verdict['conclusion']} ===")
    if flips:
        for c, x, y in flips:
            print(f"    {c}: {x} -> {y}")

    with open(os.path.join(AUDIT_DIR, "audit_verdict.json"), "w") as fh:
        json.dump(verdict, fh, indent=2)
    print(f"\nwrote {os.path.join(AUDIT_DIR, 'audit_verdict.json')}")


if __name__ == "__main__":
    main()
