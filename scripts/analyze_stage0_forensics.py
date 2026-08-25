#!/usr/bin/env python
"""Stage 0 forensic closeout of the v2 physical-target pulse campaign.

Frozen protocol: docs/V3_PREREGISTRATION.md, section "Stage 0".

This script decides nothing about method selection.  It answers three
mechanistic questions about what the v2 operator actually did:

1. Did ``score_clip=5`` collapse the Fisher--Rao score?  Compare the *raw*
   (pre-clip) score quantiles against the *applied* ones, and report the
   fraction of particles outside the clip.
2. Did ``max_event_fraction`` bind at high gamma and silently change the dose?
   Compare the realized event fraction against the cap.
3. Is the apparent gain attributable to evacuating the un-evaluated edge
   strips rather than to the target's shape?  Report edge-strip population.

Every variant is paired against its own matched-seed plain-ABF runs.  A
variant whose seeds do not match its baseline is refused, not averaged.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

QUANTILES = ("q01", "q10", "q50", "q90", "q99")
DEFAULT_VARIANTS = ("clip05", "clip20", "clipinf", "cap_fe500", "oracle")
# Declared in the v2 config; `clipinf` runs with no clip at all.
VARIANT_CLIP = {
    "clip05": 5.0, "clip20": 20.0, "clipinf": None,
    "cap_fe500": 5.0, "oracle": 5.0,
}
VARIANT_CAP = {
    "clip05": 0.10, "clip20": 0.10, "clipinf": 0.10,
    "cap_fe500": 1.0, "oracle": 0.10,
}


def _load(root: pathlib.Path, name: str) -> pd.DataFrame:
    path = root / "production_gpu" / f"production_gpu_{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing Stage 0 artifact: {path}")
    return pd.read_csv(path)


def _paired_gains(final: pd.DataFrame) -> pd.DataFrame:
    """Per-seed paired gain of the FR arm over its matched plain-ABF run."""
    base = final[final["method"] == "abf_only"]
    fr = final[final["method"] != "abf_only"]
    if base.empty or fr.empty:
        raise ValueError("a Stage 0 variant is missing one of its two arms")
    if base["seed"].duplicated().any():
        raise ValueError("duplicated plain-ABF seed in a Stage 0 variant")
    cols = ["seed", "integrated_l2_F", "integrated_l2_Fprime", "final_l2_F",
            "final_l2_Fprime"]
    merged = fr.merge(
        base[cols].rename(columns={
            "integrated_l2_F": "I_F_abf",
            "integrated_l2_Fprime": "I_Fp_abf",
            "final_l2_F": "final_F_abf",
            "final_l2_Fprime": "final_Fp_abf"}),
        on="seed", how="left", validate="many_to_one")
    if merged[["I_F_abf", "I_Fp_abf"]].isna().any().any():
        raise ValueError("an FR row lacks its matched-seed ABF baseline")
    merged["gain_I_F_pct"] = 100.0 * (1.0 - merged["integrated_l2_F"] / merged["I_F_abf"])
    merged["gain_I_Fp_pct"] = 100.0 * (1.0 - merged["integrated_l2_Fprime"] / merged["I_Fp_abf"])
    merged["final_F_ratio"] = merged["final_l2_F"] / merged["final_F_abf"]
    merged["final_Fp_ratio"] = merged["final_l2_Fprime"] / merged["final_Fp_abf"]
    return merged


def _score_shape(events: pd.DataFrame, variant: str) -> dict:
    """Median-over-(seed, event) raw and applied score quantiles."""
    active = events[events["fr_applied"] == True]  # noqa: E712 (pandas mask)
    if active.empty:
        raise ValueError(f"{variant}: no FR-active events found")
    out = {"n_active_rows": int(len(active)), "n_seeds": int(active["seed"].nunique())}
    for q in QUANTILES:
        raw_col, app_col = f"score_raw_{q}", f"score_applied_{q}"
        if raw_col not in active.columns:
            raise ValueError(
                f"{variant}: {raw_col} absent -- these runs predate the "
                f"score-shape instrumentation and cannot answer Stage 0")
        out[f"raw_{q}"] = float(active[raw_col].median())
        out[f"app_{q}"] = float(active[app_col].median())
    out["clipped_fraction"] = float(active["score_clipped_fraction"].median())
    out["raw_span_nats"] = out["raw_q99"] - out["raw_q01"]
    out["app_span_nats"] = out["app_q99"] - out["app_q01"]
    out["event_fraction_mean"] = float(active["event_fraction"].mean())
    out["event_fraction_max"] = float(active["event_fraction"].max())
    return out


def _edge_fraction(long: pd.DataFrame, at_t: float) -> float:
    """Population outside the evaluation mask, median over seeds, at time t."""
    fr = long[long["method"] != "abf_only"]
    snap = fr[np.isclose(fr["t"], at_t)]
    if snap.empty:
        return float("nan")
    outside = 1.0 - (snap["left_frac"] + snap["barrier_frac"] + snap["right_frac"])
    return float(outside.median())


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="results/physical_target_pulse_v2/stage0")
    p.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    args = p.parse_args(argv)

    root = pathlib.Path(args.root)
    wanted = [v.strip() for v in args.variants.split(",") if v.strip()]
    shape_rows, gain_rows = [], []
    missing = []

    for v in wanted:
        vroot = root / v
        if not (vroot / "production_gpu").exists():
            missing.append(v)
            continue
        final = _load(vroot, "final_summary")
        events = _load(vroot, "fr_events")
        long = _load(vroot, "runs_long")

        shape = _score_shape(events, v)
        shape.update(variant=v, score_clip=VARIANT_CLIP.get(v),
                     max_event_fraction=VARIANT_CAP.get(v))
        shape["cap_headroom"] = (
            VARIANT_CAP[v] - shape["event_fraction_max"]
            if v in VARIANT_CAP else float("nan"))
        shape_rows.append(shape)

        g = _paired_gains(final)
        gain_rows.append(dict(
            variant=v, n_seeds=int(g["seed"].nunique()),
            median_gain_I_F=float(g["gain_I_F_pct"].median()),
            median_gain_I_Fp=float(g["gain_I_Fp_pct"].median()),
            favorable_I_F=int((g["gain_I_F_pct"] > 0).sum()),
            median_final_F_ratio=float(g["final_F_ratio"].median()),
            median_final_Fp_ratio=float(g["final_Fp_ratio"].median()),
            median_ess_frac=float((g["final_ancestor_ess"] / 256.0).median()),
            median_replacements=float(g["cumulative_replacements"].median()),
            edge_frac_at_pulse_end=_edge_fraction(long, 30.0),
            edge_frac_at_T=_edge_fraction(long, 100.0)))

    if missing:
        print(f"[stage0] NOT YET AVAILABLE: {', '.join(missing)}", file=sys.stderr)
    if not shape_rows:
        print("[stage0] no completed variants; nothing to report.", file=sys.stderr)
        return 1

    shape_df = pd.DataFrame(shape_rows)
    gain_df = pd.DataFrame(gain_rows)

    pd.set_option("display.width", 200)
    print("\n=== Score shape: raw (pre-clip) vs applied, medians over seeds and events ===")
    print(shape_df[[
        "variant", "score_clip", "raw_q01", "raw_q50", "raw_q99", "raw_span_nats",
        "app_q01", "app_q99", "app_span_nats", "clipped_fraction",
    ]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n=== Event-cap binding ===")
    print(shape_df[[
        "variant", "max_event_fraction", "event_fraction_mean",
        "event_fraction_max", "cap_headroom",
    ]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== Paired outcome vs matched-seed plain ABF ===")
    print(gain_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    out = root / "stage0_forensics.csv"
    shape_df.merge(gain_df, on="variant", how="outer").to_csv(out, index=False)
    print(f"\n[stage0] wrote {out}")
    if missing:
        print(f"[stage0] report covers {len(shape_rows)}/{len(wanted)} variants; "
              f"absent: {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
