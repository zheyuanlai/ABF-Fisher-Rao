#!/usr/bin/env python3
"""Select the smallest mechanism-eligible information-target FR dose.

Selection reads pulse mechanics only.  Free-energy and mean-force errors are
not inspected.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from abffr import information_target as it  # noqa: E402
from abffr import oracle_short_burst as osb  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage-root", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--summary-csv", default=None)
    return p.parse_args(argv)


def _find_pulses(stage_root: Path) -> Path:
    hits = sorted(
        p for p in stage_root.glob("*_fr_pulses.csv") if "__" not in p.name)
    if len(hits) != 1:
        raise SystemExit(
            f"expected one merged *_fr_pulses.csv under {stage_root}; got {hits}")
    return hits[0]


def main(argv=None):
    args = parse_args(argv)
    stage_root = Path(args.stage_root)
    pulse_path = _find_pulses(stage_root)
    pulses = pd.read_csv(pulse_path)
    required = {
        "gamma", "seed", "event_fraction", "kl_before", "kl_after",
        "ess_anc_after", "logp_floored_fraction", "dtau",
        "information_risk_ratio",
    }
    missing = sorted(required - set(pulses.columns))
    if missing:
        raise SystemExit(f"calibration pulse table misses columns {missing}")
    if not bool((pulses["information_risk_ratio"] <= 1.0 + 1e-12).all()):
        raise SystemExit("constructed information target does not lower predicted risk")

    rows = pulses.to_dict(orient="records")
    summaries = it.summarize_doses(rows)
    out = Path(args.out) if args.out else stage_root / "dose_selection.json"
    summary_csv = (
        Path(args.summary_csv) if args.summary_csv
        else stage_root / "dose_calibration_summary.csv")
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([s.to_row() for s in summaries]).to_csv(summary_csv, index=False)

    receipt = {
        "protocol": "information_target_oracle_campaign",
        "selection_uses_fec_outcomes": False,
        "target_gate": "information_risk_ratio <= 1",
        "source": os.path.relpath(pulse_path, ROOT),
        "criteria": {
            "median_event_fraction": [
                osb.MIN_EVENT_FRACTION, osb.MAX_EVENT_FRACTION],
            "max_median_kl_after_over_before": osb.MAX_MEDIAN_KL_RATIO,
            "min_kl_decrease_fraction": osb.MIN_KL_DECREASE_FRACTION,
            "min_median_single_pulse_ancestor_ess_fraction":
                osb.MIN_SINGLE_PULSE_ESS,
            "max_logp_floored_fraction": 0.0,
            "tie_break": "smallest eligible gamma",
        },
        "summaries": [s.to_row() for s in summaries],
    }
    try:
        selected, _ = it.select_dose(rows)
    except ValueError:
        receipt.update(status="NO_ELIGIBLE_DOSE", selected=None)
        code = 2
    else:
        receipt.update(status="SELECTED", selected=selected.to_row())
        code = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame([s.to_row() for s in summaries]).to_string(index=False))
    print(f"wrote {os.path.relpath(summary_csv, ROOT)}")
    print(f"wrote {os.path.relpath(out, ROOT)}")
    if receipt["selected"] is not None:
        print(f"SELECTED gamma={selected.gamma:g} dtau={selected.dtau:g}")
    else:
        print("NO_ELIGIBLE_DOSE: oracle campaign is not authorised")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
