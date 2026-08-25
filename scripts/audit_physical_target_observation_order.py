#!/usr/bin/env python3
"""Matched ABF-only audit of legacy pre- versus protected post-propagation learning."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from abffr import io_utils, metrics, parallel, simulation_torch
from abffr.io_utils import RunSpec


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = io_utils.load_config(args.config)
    setup = parallel.prepare_stage(
        cfg, "production_gpu", require_csv=True, logger=print)
    seeds = [int(seed) for seed in cfg["simulation"]["seeds"]]
    eta = float(cfg["fr"]["eta_values"][0])
    specs = [
        RunSpec(
            "abf_only", "none", seed=seed, gamma=0.0, eta=eta,
            burnin_fraction=0.0, fr_every=1, stop_fraction=1.0)
        for seed in seeds
    ]
    rows = []
    for order in ["pre_propagation", "post_propagation"]:
        order_cfg = copy.deepcopy(cfg)
        order_cfg["abf"]["observation_order"] = order
        result = simulation_torch.run_batch(
            specs, cfg=order_cfg, x_grid=setup["x_grid"],
            F_ref=setup["ref"]["F_ref"],
            Fprime_ref=setup["ref"]["Fprime_ref"], ev=setup["ev"],
            device=setup["device"], dtype=setup["dtype"],
            estimator=order_cfg["abf"]["estimator"],
            base_seed=args.base_seed)
        for spec, diag in zip(specs, result.diags):
            summary = metrics.final_summary(
                diag, setup["x_grid"], setup["ref"]["F_ref"],
                setup["ref"]["Fprime_ref"], setup["ev"],
                p_ref=setup["ref"]["p_ref"])
            rows.append({
                "seed": int(spec.seed), "observation_order": order,
                "integrated_l2_F": summary["integrated_l2_F"],
                "integrated_l2_Fprime": summary["integrated_l2_Fprime"],
                "final_l2_F": summary["final_l2_F"],
                "final_l2_Fprime": summary["final_l2_Fprime"],
            })

    raw = pd.DataFrame(rows)
    pre = raw[raw["observation_order"] == "pre_propagation"].drop(
        columns="observation_order").add_suffix("_pre").rename(
            columns={"seed_pre": "seed"})
    post = raw[raw["observation_order"] == "post_propagation"].drop(
        columns="observation_order").add_suffix("_post").rename(
            columns={"seed_post": "seed"})
    paired = pre.merge(post, on="seed", validate="one_to_one")
    metric_names = [
        "integrated_l2_F", "integrated_l2_Fprime",
        "final_l2_F", "final_l2_Fprime"]
    summary = {
        "definition": "100 * (post / legacy_pre - 1); positive means post is larger",
        "n_seeds": len(seeds),
        "n_steps": int(cfg["simulation"]["n_steps"]),
        "observation_order_shared_within_pilot": True,
    }
    for name in metric_names:
        field = f"{name}_post_vs_pre_pct"
        paired[field] = 100.0 * (
            paired[f"{name}_post"] / paired[f"{name}_pre"] - 1.0)
        values = paired[field].to_numpy(float)
        summary[field] = {
            "median": float(np.median(values)),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
        }
    summary["auc_order_difference_below_one_percent"] = bool(
        abs(summary["integrated_l2_F_post_vs_pre_pct"]["median"]) < 1.0
        and abs(summary["integrated_l2_Fprime_post_vs_pre_pct"]["median"]) < 1.0)

    output_dir = Path(args.output_dir) if args.output_dir else (
        ROOT / cfg["output_root"] / "engineering")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(output_dir / "observation_order_raw.csv", index=False)
    paired.to_csv(output_dir / "observation_order_paired.csv", index=False)
    (output_dir / "observation_order_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"[order-audit] outputs: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
