"""Integrity regressions for the preregistered physical-target pilot analyzer."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_PATH = ROOT / "scripts" / "analyze_physical_target_pulse_v2.py"
_SPEC = importlib.util.spec_from_file_location(
    "analyze_physical_target_pulse_v2", ANALYSIS_PATH)
analysis = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(analysis)

from abffr import io_utils


def _smoke_frames():
    cfg = io_utils.load_config(
        ROOT / "configs" / "physical_target_pulse_v2_smoke.yaml")
    specs = io_utils.build_run_specs(cfg, cfg["simulation"]["seeds"])
    n_steps = int(cfg["simulation"]["n_steps"])
    every_eval = int(cfg["simulation"]["eval_every"])
    steps = list(range(0, n_steps + 1, every_eval))
    if steps[-1] != n_steps:
        steps.append(n_steps)
    final_rows = []
    long_rows = []
    for spec in specs:
        meta = spec.to_row()
        final_rows.append({
            **meta, "integrated_l2_F": 1.0,
            "integrated_l2_Fprime": 1.0, "final_ancestor_ess": 64.0,
            "final_max_clone_weight": 1.0 / 64.0,
            "cumulative_replacements": 0,
        })
        burn = int(round(spec.burnin_fraction * n_steps))
        stop = int(round(spec.stop_fraction * n_steps))
        for step in steps:
            if spec.gamma <= 0.0 or step < burn:
                events = 0
            else:
                last = min(step, stop - 1)
                events = (
                    0 if last < burn else 1 + (last - burn) // spec.fr_every)
            long_rows.append({
                **meta, "step": step, "t": 0.002 * step,
                "l2_F": 1.0, "l2_Fprime": 1.0,
                "marginal_l2_physical_ref": 1.0,
                "gamma_eff": (
                    spec.gamma if burn <= step < stop else 0.0),
                "cumulative_fr_events": events,
                "cumulative_replacements": 0,
            })
    return cfg, pd.DataFrame(final_rows), pd.DataFrame(long_rows)


def test_analyzer_requires_complete_long_run_and_snapshot_inventory():
    cfg, final, long = _smoke_frames()
    inventory = analysis._validate_inputs(
        final, long, cfg, allow_incomplete=False)
    assert inventory["bad_snapshot_grid_runs"] == 0
    assert inventory["missing_long_runs"] == 0

    missing_run_id = final["run_id"].iloc[-1]
    missing = long[long["run_id"] != missing_run_id]
    with pytest.raises(ValueError, match="long missing=1"):
        analysis._validate_inputs(final, missing, cfg, allow_incomplete=False)

    truncated = long.drop(long.index[-1])
    with pytest.raises(ValueError, match="bad snapshot grids=1"):
        analysis._validate_inputs(final, truncated, cfg, allow_incomplete=False)

    duplicated = pd.concat([long, long.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="long duplicates=1"):
        analysis._validate_inputs(final, duplicated, cfg, allow_incomplete=False)


def test_post_stop_integrity_detects_event_exactly_at_exclusive_stop():
    cfg, final, long = _smoke_frames()
    physical = final[final["target_type"] == "physical"]
    physical_ids = set(physical["run_id"])
    physical_long = long[long["run_id"].isin(physical_ids)].copy()
    n_steps = int(cfg["simulation"]["n_steps"])
    assert analysis._post_stop_integrity(physical, physical_long, n_steps)

    stop_step = int(round(
        float(physical["stop_fraction"].iloc[0]) * n_steps))
    bad = physical_long.copy()
    bad.loc[bad["step"] >= stop_step, "cumulative_fr_events"] += 1
    # The post-stop tail is still constant, but the analytic [on, off) event
    # count exposes the illegal event at the exclusive stop boundary.
    assert not analysis._post_stop_integrity(physical, bad, n_steps)
