#!/usr/bin/env python3
"""Analyze the frozen oracle information-target campaign."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from abffr import accel, information_target as it, io_utils  # noqa: E402

BASE = "abf_only"
ARM = "abf_fr_information_oracle"
N_BOOT = 20000
BOOT_SEED = 20260828


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage-root", required=True)
    p.add_argument("--thresholds", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--summary-csv", default=None)
    return p.parse_args(argv)


def _load_one(stage_root: Path, suffix: str) -> pd.DataFrame:
    hits = sorted(
        p for p in stage_root.glob(f"*_{suffix}.csv") if "__" not in p.name)
    if len(hits) != 1:
        raise SystemExit(
            f"expected one merged *_{suffix}.csv under {stage_root}; got {hits}")
    return pd.read_csv(hits[0])


def _paired_hits(long_df, col, eps, horizon, consecutive):
    by = {
        (method, int(seed)): group.sort_values("t")
        for (method, seed), group in long_df.groupby(["method", "seed"])
    }
    seeds = sorted(
        set(long_df[long_df.method == BASE].seed.astype(int))
        & set(long_df[long_df.method == ARM].seed.astype(int)))
    base, arm = [], []
    for seed in seeds:
        for method, out in ((BASE, base), (ARM, arm)):
            group = by[(method, seed)]
            out.append(accel.restricted_hitting_time(
                group.t.to_numpy(), group[col].to_numpy(), eps, horizon,
                consecutive=consecutive))
    return seeds, base, arm


def _speed_rows(long_df, frozen, scope):
    horizon = float(frozen["horizon"])
    consecutive = int(frozen["consecutive_frames"])
    rows, objects = [], {}
    for label, stem in (("F", "l2_F"), ("Fprime", "l2_Fprime")):
        for k, eps in enumerate(frozen["thresholds"][scope][label], start=1):
            seeds, base, arm = _paired_hits(
                long_df, f"{stem}_{scope}", eps, horizon, consecutive)
            speed = accel.paired_bootstrap_speedup(
                base, arm, n_boot=N_BOOT, seed=BOOT_SEED + k)
            objects[(label, k)] = speed
            rows.append({
                "metric": label, "threshold_index": k,
                "epsilon": float(eps), "n_matched_seeds": len(seeds),
                "speedup": speed.s, "ci_lo": speed.ci_lo,
                "ci_hi": speed.ci_hi,
                "mean_restricted_tau_base": speed.mean_base,
                "mean_restricted_tau_arm": speed.mean_arm,
                "censored_base": speed.n_censored_base,
                "censored_arm": speed.n_censored_arm,
                "hit_fraction_base": speed.hit_fraction_base,
                "hit_fraction_arm": speed.hit_fraction_arm,
                "censoring_inflates": speed.censoring_inflates,
            })
    return rows, objects


def _target_is_frozen(pulses: pd.DataFrame) -> bool:
    mass_cols = sorted(c for c in pulses.columns if c.startswith("q_cell_"))
    if not mass_cols:
        return False
    for _, group in pulses.groupby("seed"):
        values = group.sort_values("step")[mass_cols].to_numpy(float)
        if not np.array_equal(values, np.repeat(values[:1], len(values), axis=0)):
            return False
    return True


def main(argv=None):
    args = parse_args(argv)
    stage_root = Path(args.stage_root)
    cfg = io_utils.load_config(args.config)
    expected_steps = it.validate_config(cfg)
    frozen = json.loads(Path(args.thresholds).read_text(encoding="utf-8"))
    scope = frozen["primary_scope"]
    long_df = _load_one(stage_root, "runs_long")
    pulses = _load_one(stage_root, "fr_pulses")
    final_df = _load_one(stage_root, "final_summary")

    speed_rows, speeds = _speed_rows(long_df, frozen, scope)
    speed_df = pd.DataFrame(speed_rows)
    final_t = float(long_df.t.max())
    final_long = long_df[np.isclose(long_df.t, final_t)]
    base_final = final_long[final_long.method == BASE]
    arm_final = final_long[final_long.method == ARM]
    final_error_ratio = (
        float(arm_final[f"l2_F_{scope}"].median())
        / float(base_final[f"l2_F_{scope}"].median()))
    n_particles = int(cfg["simulation"]["n_particles"])
    final_ess_fraction = float(
        final_df[final_df.method == ARM].final_ancestor_ess.median()
        / n_particles)

    pulses = pulses[pulses.method == ARM].sort_values(["seed", "step"]).copy()
    pulses["pulse_index"] = pulses.groupby("seed").cumcount() + 1
    counts = pulses.groupby("seed").size()
    steps_by_seed = pulses.groupby("seed").step.apply(list)
    pulse_count_pass = bool(
        len(counts) > 0 and (counts == len(expected_steps)).all()
        and all(value == expected_steps for value in steps_by_seed))
    pulses["kl_ratio"] = pulses.kl_after / pulses.kl_before
    risk_pass = bool(
        len(pulses) > 0
        and np.isfinite(pulses.information_risk_ratio).all()
        and float(pulses.information_risk_ratio.max()) <= 1.0 + 1e-12
        and _target_is_frozen(pulses))
    mechanism_pass = bool(
        len(pulses) > 0
        and float(pulses.kl_ratio.median()) < 1.0
        and float((pulses.kl_ratio < 1.0).mean()) >= 0.5
        and float(pulses.logp_floored_fraction.max()) == 0.0
        and risk_pass)
    genealogy_pass = final_ess_fraction >= 0.70
    censoring_pass = not any(
        speeds[("F", k)].censoring_inflates for k in (1, 2))
    acceleration_pass = all(
        speeds[("F", k)].s >= 1.15 and speeds[("F", k)].ci_lo > 1.0
        for k in (1, 2))
    endpoint_pass = final_error_ratio <= 1.05
    verdict = it.classify_campaign(
        acceleration_pass=acceleration_pass, endpoint_pass=endpoint_pass,
        genealogy_pass=genealogy_pass, mechanism_pass=mechanism_pass,
        pulse_count_pass=pulse_count_pass, censoring_pass=censoring_pass)

    result = {
        "protocol": "information_target_oracle_campaign",
        "verdict": verdict, "scope": scope,
        "n_matched_seeds": int(speed_df.n_matched_seeds.min()),
        "gamma": float(pulses.gamma.iloc[0]),
        "dtau_per_pulse": float(pulses.dtau.iloc[0]),
        "pulse_steps": expected_steps,
        "gates": {
            "acceleration_pass": acceleration_pass,
            "endpoint_pass": endpoint_pass,
            "genealogy_pass": genealogy_pass,
            "mechanism_pass": mechanism_pass,
            "target_risk_and_freeze_pass": risk_pass,
            "pulse_count_pass": pulse_count_pass,
            "censoring_pass": censoring_pass,
        },
        "threshold_speedups": speed_rows,
        "final_error_ratio": final_error_ratio,
        "final_ancestor_ess_fraction": final_ess_fraction,
        "overall_median_kl_ratio": float(pulses.kl_ratio.median()),
        "information_risk_ratio": float(
            pulses.information_risk_ratio.iloc[0]),
    }
    out = Path(args.out) if args.out else stage_root / "campaign_verdict.json"
    summary_csv = (
        Path(args.summary_csv) if args.summary_csv
        else stage_root / "campaign_speedups.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    speed_df.to_csv(summary_csv, index=False)
    print(speed_df.to_string(index=False))
    print(f"gates={result['gates']}")
    print(f"VERDICT: {verdict}")
    print(f"wrote {os.path.relpath(summary_csv, ROOT)}")
    print(f"wrote {os.path.relpath(out, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
