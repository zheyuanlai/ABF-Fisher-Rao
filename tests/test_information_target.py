from __future__ import annotations

import copy
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from abffr import information_target as it  # noqa: E402
from abffr import parallel, reference, simulation_torch  # noqa: E402
from abffr.io_utils import RunSpec  # noqa: E402


DOMAIN = {
    "x_min": -3.0, "x_max": 3.0,
    "y_min": -2.5, "y_max": 3.5,
}


def _cfg(kind=it.ORACLE_CAMPAIGN_KIND):
    campaign = kind == it.ORACLE_CAMPAIGN_KIND
    return {
        "information_target": {
            "enabled": True,
            "kind": kind,
            "n_cells": 8,
            "report_min": -2.5,
            "report_max": 2.5,
            "min_expected_particles_per_cell": 1.0,
            "expected_firing_steps": [10, 15, 20] if campaign else [10],
            **({"calibrated_gamma": 0.1} if campaign else {}),
        },
        "selection": {"write_generic_best": False},
        "simulation": {
            "beta": 4.0, "dt": 0.001, "n_steps": 40 if campaign else 20,
            "n_particles": 32, "eval_every": 5,
            "x_init_mode": "uniform", "y_init_mode": "uniform",
        },
        "domain": dict(DOMAIN),
        "potential": {"x_tilt": 0.1021665783},
        "abf": {
            "estimator": "binned_smooth", "observation_order": "post_propagation",
            "h": 0.12, "update_every": 1, "min_count": 1.0,
        },
        "fr": {
            "target_types": ["information_oracle"],
            "gamma_values": [0.1] if campaign else [0.05, 0.1],
            "eta_values": [0.18],
            "burnin_fractions": [0.25 if campaign else 0.5],
            "duration_fractions": [0.375 if campaign else 0.25],
            "fr_every_values": [5],
            "interval_scaled_clock": True,
            "noise_chunk_steps": 7,
        },
        "methods": (["abf_only", "abf_fr_information_oracle"]
                    if campaign else ["abf_fr_information_oracle"]),
    }


def test_reference_force_variance_is_nonnegative_and_tilt_invariant():
    x = np.linspace(-3.0, 3.0, 101)
    y = np.linspace(-2.5, 3.5, 301)
    base = reference.compute_reference(x, y, beta=4.0, x_tilt=0.0)
    tilted = reference.compute_reference(x, y, beta=4.0, x_tilt=0.17)
    assert np.all(base["force_var_ref"] >= 0.0)
    np.testing.assert_allclose(
        base["force_var_ref"], tilted["force_var_ref"],
        rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(
        base["force_second_moment_ref"] - base["Fprime_ref"] ** 2,
        base["force_var_ref"], rtol=1e-12, atol=1e-12)


def test_information_target_is_normalized_covered_and_beats_uniform_risk():
    x = np.linspace(-3.0, 3.0, 401)
    variance = 0.2 + np.exp(-((x + 0.8) / 0.35) ** 2)
    target = it.build_target(
        x, variance, x_min=-3.0, x_max=3.0, n_cells=32,
        report_min=-2.5, report_max=2.5, n_particles=256)
    assert np.trapezoid(target.density, x) == pytest.approx(1.0, abs=1e-12)
    assert target.masses.sum() == pytest.approx(1.0, abs=1e-12)
    assert target.masses.min() >= 1.0 / 256 - 1e-14
    assert np.all(target.leverage >= 0.0)
    assert target.risk_ratio <= 1.0 + 1e-12
    assert np.ptp(target.masses) > 0.0


def test_information_score_is_centered_and_points_from_over_to_under_mass():
    p = torch.tensor([[0.8, 0.8, 0.2, 0.2]], dtype=torch.float64)
    q = torch.tensor([[0.2, 0.2, 0.8, 0.8]], dtype=torch.float64)
    x = torch.tensor([[-0.9, -0.6, 0.6, 0.9]], dtype=torch.float64)
    score, logp, logq, floored = it.score(p, q, x, -1.0, 2.0 / 3.0)
    assert float(score.mean()) == pytest.approx(0.0, abs=1e-14)
    assert torch.all(score[0, :2] > 0.0)  # overrepresented -> death sign
    assert torch.all(score[0, 2:] < 0.0)  # underrepresented -> birth sign
    assert float(floored[0]) == 0.0
    assert torch.allclose(it.row_score(logp[0], logq[0]).S, score[0])


def test_config_guard_pins_one_calibration_or_three_campaign_pulses():
    assert it.validate_config(_cfg(it.CALIBRATION_KIND)) == [10]
    assert it.validate_config(_cfg()) == [10, 15, 20]
    bad = copy.deepcopy(_cfg())
    bad["fr"]["score_clip"] = 5.0
    with pytest.raises(ValueError, match="forbids fr.score_clip"):
        it.validate_config(bad)


def test_oracle_information_target_runs_three_frozen_standard_bd_pulses():
    cfg = _cfg()
    x = np.linspace(-3.0, 3.0, 81)
    y = np.linspace(-2.5, 3.5, 161)
    ref = reference.compute_reference(
        x, y, beta=4.0, x_tilt=cfg["potential"]["x_tilt"])
    spec = RunSpec(
        method="abf_fr_information_oracle", target_type="information_oracle",
        seed=17, gamma=0.1, eta=0.18, burnin_fraction=0.25,
        stop_fraction=0.625, fr_every=5)
    result = simulation_torch.run_batch(
        [spec], cfg=cfg, x_grid=x, F_ref=ref["F_ref"],
        Fprime_ref=ref["Fprime_ref"], force_var_ref=ref["force_var_ref"],
        ev=SimpleNamespace(x_barrier=0.0), device=torch.device("cpu"),
        dtype=torch.float64, estimator="binned_smooth", base_seed=3)

    assert [row["step"] for row in result.clean_events] == [10, 15, 20]
    masses = [row["q_cell_masses"][0] for row in result.clean_events]
    np.testing.assert_array_equal(masses[0], masses[1])
    np.testing.assert_array_equal(masses[0], masses[2])
    assert result.clean_events[0]["information_risk_ratio"][0] <= 1.0 + 1e-12
    rows = parallel._clean_event_rows(spec.to_row(), result.clean_events, 0)
    assert len(rows) == 3
    assert rows[0]["information_risk_ratio"] <= 1.0 + 1e-12
    assert rows[0]["q_cell_00"] >= 1.0 / 32 - 1e-14
    assert rows[0]["q_cell_00"] == rows[1]["q_cell_00"]
    assert "force_var_cell_07" in rows[0]

    assert result.diags[0]["information_target"] is True
    assert result.diags[0]["fr_firing_steps"] == [10, 15, 20]
    assert result.diags[0]["cumulative_fr_events"][-1] == 3
