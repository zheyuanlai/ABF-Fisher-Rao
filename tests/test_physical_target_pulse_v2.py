"""Engineering gates for delayed, sparse, temporary physical-target FR."""
from __future__ import annotations

from types import SimpleNamespace
import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from abffr import io_utils, metrics, parallel, potentials, reference, simulation
from abffr import simulation_torch, torch_utils as tu
from abffr.io_utils import RunSpec


DOMAIN = {
    "x_min": -3.0,
    "x_max": 3.0,
    "y_min": -2.5,
    "y_max": 3.5,
}
TILT = 0.1021665783


def _reference(nx=121, ny=241, tilt=TILT):
    x = np.linspace(DOMAIN["x_min"], DOMAIN["x_max"], nx)
    y = np.linspace(DOMAIN["y_min"], DOMAIN["y_max"], ny)
    return x, reference.compute_reference(x, y, beta=4.0, x_tilt=tilt)


def _run_cpu(target, gamma, seed=7, **overrides):
    x, ref = _reference()
    kwargs = dict(
        target_type=target,
        beta=4.0,
        dt=0.001,
        n_steps=20,
        n_particles=32,
        eval_every=1,
        x_grid=x,
        F_ref=ref["F_ref"],
        Fprime_ref=ref["Fprime_ref"],
        domain=DOMAIN,
        h=0.12,
        eta=0.18,
        min_count=1.0,
        ema_alpha=0.05,
        gamma=gamma,
        burnin_fraction=0.2,
        stop_fraction=0.7,
        ramp_fraction=0.0,
        fr_every=2,
        score_clip=5.0,
        max_event_fraction=0.25,
        x_init_mode="uniform",
        y_init_mode="uniform",
        observation_order="post_propagation",
        update_every=1,
        x_tilt=TILT,
        interval_scaled_clock=True,
    )
    kwargs.update(overrides)
    kwargs["rng_init"], kwargs["rng_noise"], kwargs["rng_fr"] = (
        io_utils.make_rng_streams(seed))
    return simulation.run_simulation(**kwargs)


def _torch_cfg(n_steps=20, n_particles=32, eval_every=1):
    return {
        "simulation": {
            "beta": 4.0,
            "dt": 0.001,
            "n_steps": n_steps,
            "n_particles": n_particles,
            "eval_every": eval_every,
            "x_init_mode": "uniform",
            "y_init_mode": "uniform",
        },
        "domain": dict(DOMAIN),
        "potential": {"x_tilt": TILT},
        "abf": {
            "estimator": "kernel_reference",
            "observation_order": "post_propagation",
            "h": 0.12,
            "update_every": 1,
            "min_count": 1.0,
            "ema_alpha": 0.05,
        },
        "fr": {
            "ramp_fraction": 0.0,
            "score_clip": 5.0,
            "max_event_fraction": 0.25,
            "target_ema_alpha": 0.05,
            "interval_scaled_clock": True,
            "jitter": 0.0,
            "noise_chunk_steps": 7,
        },
    }


def test_physical_targets_normalize_and_ignore_additive_constant():
    x, ref = _reference(nx=401, ny=401)
    F = 0.3 * np.sin(x) + 0.1 * x
    B = np.cos(x)
    X = np.linspace(-2.0, 2.0, 20)

    q = simulation._build_target(
        "physical", x, F, B, 4.0, X, -3.0, 3.0, 0.15,
        ref["F_ref"])
    q_shift = simulation._build_target(
        "physical", x, F + 17.0, B, 4.0, X, -3.0, 3.0, 0.15,
        ref["F_ref"])
    q_oracle = simulation._build_target(
        "physical_oracle", x, F, B, 4.0, X, -3.0, 3.0, 0.15,
        ref["F_ref"])

    assert np.trapezoid(q, x) == pytest.approx(1.0, abs=1e-12)
    assert np.trapezoid(q_oracle, x) == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(q, q_shift, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(q_oracle, ref["p_ref"], rtol=1e-12, atol=1e-13)


def test_cpu_and_torch_physical_target_scores_agree_componentwise():
    x = np.linspace(-3.0, 3.0, 1201)
    dx = float(x[1] - x[0])
    X = np.array([-1.91, -1.3, -0.47, 0.18, 0.81, 1.37, 2.02])
    F = 0.2 * x ** 2 - 0.15 * x + 0.04 * np.sin(3.0 * x)
    B = 0.1 * np.cos(x)
    p_grid = simulation.kde_marginal(x, X, -3.0, 3.0, 0.22)
    p_at_x = simulation.kde_marginal(X, X, -3.0, 3.0, 0.22)
    q = simulation._build_target(
        "physical", x, F, B, 4.0, X, -3.0, 3.0, 0.22, F)
    score_cpu = simulation.fr_score(
        X, p_at_x, p_grid, q, x, score_clip=3.0)

    xt = torch.as_tensor(x, dtype=torch.float64)
    Xt = torch.as_tensor(X[None, :], dtype=torch.float64)
    Ft = torch.as_tensor(F[None, :], dtype=torch.float64)
    Bt = torch.as_tensor(B[None, :], dtype=torch.float64)
    pt = simulation_torch._kde_reflected(xt, Xt, 0.22, -3.0, 3.0)
    qt = simulation_torch._build_target(
        "physical", Ft, Bt, Ft, pt, 4.0, dx, 6.0)
    score_torch = simulation_torch._fr_score(
        Xt, pt, qt, -3.0, dx, 4.0, 3.0)[0].numpy()

    assert float(tu.trapezoid(qt, dx)[0]) == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(qt[0].numpy(), q, rtol=2e-11, atol=2e-12)
    np.testing.assert_allclose(score_torch, score_cpu, rtol=2e-3, atol=2e-3)


def test_completed_step_schedule_is_sparse_and_stops_exclusively():
    diag = _run_cpu(
        "physical", gamma=0.1, n_steps=10, eval_every=1,
        burnin_fraction=0.2, stop_fraction=0.7, fr_every=2)
    assert diag["steps"] == list(range(11))
    assert diag["cumulative_fr_events"][-1] == 3
    fired = [
        step for step, applied in zip(diag["steps"], diag["fr_applied"])
        if applied
    ]
    assert fired == [2, 4, 6]
    for step, gamma_eff, cumulative in zip(
            diag["steps"], diag["gamma_eff"], diag["cumulative_fr_events"]):
        if step >= 7:
            assert gamma_eff == 0.0
            assert cumulative == 3


def test_gamma_zero_physical_is_trajectory_identical_to_plain_abf():
    abf = _run_cpu("none", gamma=0.0, seed=19, n_steps=30, eval_every=5)
    physical = _run_cpu(
        "physical", gamma=0.0, seed=19, n_steps=30, eval_every=5)
    for key in ["X_snap", "Y_snap", "Fprime_hat", "F_hat", "p_hat_grid"]:
        np.testing.assert_array_equal(
            np.asarray(abf[key]), np.asarray(physical[key]))
    assert physical["cumulative_fr_events"][-1] == 0
    assert physical["cumulative_replacements"][-1] == 0


def test_resampling_does_not_enter_accumulator_before_propagation(monkeypatch):
    def forced_resample(X, Y, ancestors, S, gamma, dt, rng,
                        max_event_fraction=None):
        Xn = X.copy()
        Yn = Y.copy()
        an = ancestors.copy()
        Xn[-1], Yn[-1], an[-1] = X[0], Y[0], ancestors[0]
        return Xn, Yn, an, {"n_die": 1, "n_clone": 1, "n_events": 1}

    plain = _run_cpu(
        "none", gamma=0.0, seed=31, n_steps=6, eval_every=1,
        burnin_fraction=2 / 6, stop_fraction=5 / 6, fr_every=2)
    monkeypatch.setattr(simulation, "resample_fixed_N", forced_resample)
    pulsed = _run_cpu(
        "physical", gamma=1.0, seed=31, n_steps=6, eval_every=1,
        burnin_fraction=2 / 6, stop_fraction=5 / 6, fr_every=2)

    idx = pulsed["steps"].index(2)
    np.testing.assert_array_equal(
        pulsed["Fprime_hat"][idx], plain["Fprime_hat"][idx])
    np.testing.assert_array_equal(pulsed["F_hat"][idx], plain["F_hat"][idx])
    assert not np.array_equal(pulsed["X_snap"][idx], plain["X_snap"][idx])


def test_tilt_reference_identity_and_calibrated_well_mass():
    x = np.linspace(-3.0, 3.0, 1201)
    y = np.linspace(-2.5, 3.5, 1201)
    base = reference.compute_reference(x, y, beta=4.0, x_tilt=0.0)
    tilted = reference.compute_reference(x, y, beta=4.0, x_tilt=TILT)

    np.testing.assert_allclose(
        tilted["Fprime_ref"] - base["Fprime_ref"], TILT,
        rtol=1e-11, atol=1e-11)
    expected = TILT * (x - x.mean())
    np.testing.assert_allclose(
        tilted["F_ref"] - base["F_ref"], expected,
        rtol=1e-10, atol=1e-10)

    left = np.trapezoid(tilted["p_ref"][x <= 0], x[x <= 0])
    right = np.trapezoid(tilted["p_ref"][x >= 0], x[x >= 0])
    assert left / (left + right) == pytest.approx(0.70, abs=0.003)


def test_schedule_grid_expansion_and_identifiers():
    cfg = {
        "methods": ["abf_only", "abf_fr_physical"],
        "abf": {"eta": 0.1},
        "fr": {
            "target_types": ["physical"],
            "gamma_values": [0.02, 0.05, 0.10],
            "eta_values": [0.10],
            "burnin_fractions": [0.20, 0.40, 0.60],
            "duration_fractions": [0.10, 0.30],
            "fr_every_values": [20, 100, 500],
        },
    }
    specs = io_utils.build_run_specs(cfg, seeds=[0, 1])
    assert len(specs) == 110
    assert len({s.config_id for s in specs}) == 55
    physical = [s for s in specs if s.target_type == "physical"]
    assert all(s.stop_fraction == pytest.approx(
        s.burnin_fraction + (
            0.10 if np.isclose(
                s.stop_fraction - s.burnin_fraction, 0.10) else 0.30))
               for s in physical)
    a = RunSpec(
        "abf_fr_physical", "physical", 0, gamma=0.02, eta=0.1,
        burnin_fraction=0.2, fr_every=20, stop_fraction=0.3)
    b = RunSpec(
        "abf_fr_physical", "physical", 0, gamma=0.02, eta=0.1,
        burnin_fraction=0.2, fr_every=20, stop_fraction=0.5)
    assert a.config_id != b.config_id
    assert a.to_row()["stop_fraction"] == 0.3


def test_independent_physical_marginal_metric_and_diagnostics():
    x = np.linspace(-1.0, 1.0, 101)
    p_ref = np.exp(-x ** 2)
    p_ref /= np.trapezoid(p_ref, x)
    uniform = np.ones_like(x) / 2.0
    zeros = np.zeros_like(x)
    diag = {
        "steps": [0, 10],
        "times": [0.0, 2.0],
        "F_hat": [zeros, zeros],
        "Fprime_hat": [zeros, zeros],
        "p_hat_grid": [uniform, p_ref],
        "q_target_grid": [p_ref, p_ref],
        "X_snap": [np.array([-0.5, 0.5]), np.array([-0.5, 0.5])],
        "Y_snap": [np.zeros(2), np.zeros(2)],
        "barrier_crossings": [0, 0],
        "n_unique_ancestors": [2, 1],
        "ancestor_ess": [2.0, 1.6],
        "max_clone_multiplicity": [1, 2],
        "max_clone_weight": [0.5, 1.0],
        "cumulative_fr_events": [0, 3],
        "cumulative_replacements": [0, 2],
        "gamma_eff": [0.0, 0.0],
        "fr_applied": [False, False],
        "fr_event_fraction": [0.0, 0.0],
        "fr_event_fraction_max": [0.0, 0.0],
        "fr_events_total": [0, 0],
        "score_mean": [np.nan, np.nan],
        "score_std": [np.nan, np.nan],
        "score_min": [np.nan, np.nan],
        "score_max": [np.nan, np.nan],
    }
    ev = metrics.EvalConfig(
        eval_x_min=-1.0, eval_x_max=1.0, left_basin_min=-1.0,
        right_basin_max=1.0)
    rows = metrics.time_series_metrics(
        diag, x, zeros, zeros, ev, p_ref=p_ref)
    summary = metrics.final_summary(
        diag, x, zeros, zeros, ev, p_ref=p_ref)
    expected_auc = rows[0]["marginal_l2_physical_ref"]
    assert rows[-1]["marginal_l2_physical_ref"] == pytest.approx(0.0)
    assert summary["final_marginal_l2_physical_ref"] == pytest.approx(0.0)
    assert summary["integrated_marginal_l2_physical_ref"] == pytest.approx(
        expected_auc)
    assert summary["final_ancestor_ess"] == pytest.approx(1.6)
    assert summary["final_max_clone_weight"] == pytest.approx(1.0)
    assert summary["cumulative_fr_events"] == 3
    assert summary["cumulative_replacements"] == 2



def test_physical_marginal_metric_retains_full_domain_mass_error():
    x = np.linspace(-1.0, 1.0, 201)
    p_ref = np.full_like(x, 0.5)
    p = p_ref.copy()
    p[x < -0.6] += 0.2
    p[x > 0.6] -= 0.2
    mask = (x >= -0.5) & (x <= 0.5)

    # A conditional-interior metric would discard this edge mass error.
    assert metrics.marginal_l2_to_target(
        p, p_ref, x, mask) == pytest.approx(0.0)
    assert metrics.marginal_l2_to_physical_ref(
        p, p_ref, x, mask) > 0.05


def test_basin_deltaF_and_barrier_errors_are_evaluation_only():
    x = np.linspace(-2.0, 2.0, 401)
    F_ref = (x ** 2 - 1.0) ** 2
    F_hat = F_ref + 0.2 * x
    mask = np.ones_like(x, dtype=bool)
    out = metrics.free_energy_landmark_errors(
        F_hat, F_ref, x, mask, x_barrier=0.0)
    assert out["deltaF_ref"] == pytest.approx(0.0, abs=1e-12)
    assert out["deltaF_error"] == pytest.approx(0.4, abs=1e-12)
    assert out["barrier_height_error"] == pytest.approx(0.2, abs=1e-12)

def test_torch_gamma_zero_and_stop_gate():
    x, ref = _reference(nx=101, ny=201)
    cfg = _torch_cfg(n_steps=10, n_particles=24, eval_every=1)
    ev = SimpleNamespace(x_barrier=0.0)
    common = dict(
        seed=43, eta=0.18, burnin_fraction=0.2,
        fr_every=2, stop_fraction=0.7)
    plain_spec = RunSpec(
        "abf_only", "none", gamma=0.0, **common)
    physical_zero = RunSpec(
        "abf_fr_physical", "physical", gamma=0.0, **common)

    plain = simulation_torch.run_batch(
        [plain_spec], cfg=cfg, x_grid=x, F_ref=ref["F_ref"],
        Fprime_ref=ref["Fprime_ref"], ev=ev, device=torch.device("cpu"),
        dtype=torch.float64, estimator="kernel_reference", base_seed=91).diags[0]
    zero = simulation_torch.run_batch(
        [physical_zero], cfg=cfg, x_grid=x, F_ref=ref["F_ref"],
        Fprime_ref=ref["Fprime_ref"], ev=ev, device=torch.device("cpu"),
        dtype=torch.float64, estimator="kernel_reference", base_seed=91).diags[0]
    for key in ["X_snap", "Y_snap", "Fprime_hat", "F_hat", "p_hat_grid"]:
        np.testing.assert_array_equal(np.asarray(plain[key]), np.asarray(zero[key]))

    pulsed_spec = RunSpec(
        "abf_fr_physical", "physical", gamma=0.1, **common)
    pulsed = simulation_torch.run_batch(
        [pulsed_spec], cfg=cfg, x_grid=x, F_ref=ref["F_ref"],
        Fprime_ref=ref["Fprime_ref"], ev=ev, device=torch.device("cpu"),
        dtype=torch.float64, estimator="kernel_reference", base_seed=91).diags[0]
    assert pulsed["cumulative_fr_events"][-1] == 3
    for step, gamma_eff, cumulative in zip(
            pulsed["steps"], pulsed["gamma_eff"],
            pulsed["cumulative_fr_events"]):
        if step >= 7:
            assert gamma_eff == 0.0
            assert cumulative == 3



def test_resume_requires_post_flush_marker(tmp_path):
    spec = RunSpec(
        "abf_fr_physical", "physical", 3, gamma=0.05, eta=0.1,
        burnin_fraction=0.2, fr_every=20, stop_fraction=0.5)
    partial = tmp_path / "production_gpu_final_summary__main.csv"
    partial.write_text("run_id,final_l2_F\n" + spec.run_id + ",1.0\n")
    assert spec.run_id not in parallel.load_completed(
        str(tmp_path), "production_gpu")

    parallel._write_marker(
        str(tmp_path), spec, {"final_l2_F": 1.0, "final_l2_Fprime": 2.0})
    assert spec.run_id in parallel.load_completed(
        str(tmp_path), "production_gpu")

def test_binned_smooth_static_error_within_existing_tolerance():
    x = np.linspace(-3.0, 3.0, 401)
    xt = torch.as_tensor(x, dtype=torch.float64)
    dx = tu.grid_spacing(xt)
    x0 = float(x[0])
    h = 0.10
    min_count = 1.0
    rng = np.random.default_rng(5)
    X = torch.as_tensor(
        rng.uniform(-2.5, 2.5, (1, 600)), dtype=torch.float64)
    Y = torch.as_tensor(
        rng.uniform(-2.0, 3.0, (1, 600)), dtype=torch.float64)

    num_k, den_k = simulation_torch._kernel_estimator(
        xt, X, Y, h, x_tilt=TILT)
    Fp_k = num_k / (den_k + min_count)
    kernel, radius = tu.gaussian_kernel1d(
        h, dx, torch.device("cpu"), torch.float64)
    idx = tu.nearest_index(X, x0, dx, len(x))
    counts = tu.scatter_grid(idx, len(x))
    force = potentials.dVdx_xy_torch(X, Y) + TILT
    sums = tu.scatter_grid(idx, len(x), force)
    Fp_b = (
        tu.smooth_grid(sums, kernel, radius, dx)
        / (tu.smooth_grid(counts, kernel, radius, dx) + min_count))

    mask = (x >= -2.5) & (x <= 2.5)
    width = x[mask][-1] - x[mask][0]
    diff = np.sqrt(np.trapezoid(
        (Fp_b[0].numpy()[mask] - Fp_k[0].numpy()[mask]) ** 2,
        x[mask]) / width)
    scale = np.sqrt(np.trapezoid(
        Fp_k[0].numpy()[mask] ** 2, x[mask]) / width)
    assert diff / (scale + 1e-12) < 0.10


def test_binned_and_kernel_physical_targets_and_scores_agree():
    x = np.linspace(-3.0, 3.0, 401)
    xt = torch.as_tensor(x, dtype=torch.float64)
    dx = tu.grid_spacing(xt)
    x0 = float(x[0])
    idx0 = int(np.argmin(np.abs(x)))
    h, eta, min_count = 0.05, 0.10, 1.0
    rng = np.random.default_rng(17)
    X = torch.as_tensor(
        rng.uniform(-2.5, 2.5, (1, 2000)), dtype=torch.float64)
    Y = torch.as_tensor(
        rng.uniform(-2.0, 3.0, (1, 2000)), dtype=torch.float64)

    num_k, den_k = simulation_torch._kernel_estimator(
        xt, X, Y, h, x_tilt=TILT)
    Fp_k = num_k / (den_k + min_count)
    kernel_h, radius_h = tu.gaussian_kernel1d(
        h, dx, torch.device("cpu"), torch.float64)
    idx = tu.nearest_index(X, x0, dx, len(x))
    counts = tu.scatter_grid(idx, len(x))
    force = potentials.dVdx_xy_torch(X, Y) + TILT
    sums = tu.scatter_grid(idx, len(x), force)
    Fp_b = (
        tu.smooth_grid(sums, kernel_h, radius_h, dx)
        / (tu.smooth_grid(counts, kernel_h, radius_h, dx) + min_count))
    F_k = tu.center_at_index(tu.cumulative_trapezoid(Fp_k, dx), idx0)
    F_b = tu.center_at_index(tu.cumulative_trapezoid(Fp_b, dx), idx0)

    p_k = simulation_torch._kde_reflected(xt, X, eta, -3.0, 3.0)
    kernel_eta, radius_eta = tu.gaussian_kernel1d(
        eta, dx, torch.device("cpu"), torch.float64)
    p_b = tu.normalize_density(
        tu.smooth_grid(counts, kernel_eta, radius_eta, dx) / X.shape[1], dx)
    zeros = torch.zeros_like(F_k)
    q_k = simulation_torch._build_target(
        "physical", F_k, zeros, F_k, p_k, 4.0, dx, 6.0)
    q_b = simulation_torch._build_target(
        "physical", F_b, zeros, F_b, p_b, 4.0, dx, 6.0)
    score_k = simulation_torch._fr_score(
        X, p_k, q_k, x0, dx, 4.0, 5.0)
    score_b = simulation_torch._fr_score(
        X, p_b, q_b, x0, dx, 4.0, 5.0)

    q_rms = torch.sqrt(tu.trapezoid((q_b - q_k) ** 2, dx) / 6.0)
    q_scale = torch.sqrt(tu.trapezoid(q_k ** 2, dx) / 6.0)
    score_delta = score_b - score_k
    assert float(q_rms[0] / (q_scale[0] + 1e-12)) < 0.10
    assert float(torch.sqrt(torch.mean(score_delta ** 2))) < 0.03
    assert float(torch.max(torch.abs(score_delta))) < 0.12


def test_torch_matched_rng_is_rerun_batch_and_order_invariant():
    x, ref = _reference(nx=101, ny=201)
    cfg = _torch_cfg(n_steps=40, n_particles=48, eval_every=4)
    ev = SimpleNamespace(x_barrier=0.0)
    common = dict(
        eta=0.10, burnin_fraction=0.0, fr_every=2, stop_fraction=0.8)
    focus = RunSpec(
        "abf_fr_physical", "physical", seed=11, gamma=20.0, **common)
    other_a = RunSpec(
        "abf_fr_physical", "physical", seed=12, gamma=5.0, **common)
    other_b = RunSpec(
        "abf_fr_physical", "physical", seed=13, gamma=10.0, **common)

    def run(specs):
        return simulation_torch.run_batch(
            specs, cfg=cfg, x_grid=x, F_ref=ref["F_ref"],
            Fprime_ref=ref["Fprime_ref"], ev=ev, device=torch.device("cpu"),
            dtype=torch.float64, estimator="binned_smooth",
            base_seed=91).diags

    solo = run([focus])[0]
    repeat = run([focus])[0]
    mixed = run([other_a, focus, other_b])[1]
    reordered = run([other_b, focus, other_a])[1]
    exact_keys = [
        "X_snap", "Y_snap",
        "cumulative_fr_events", "cumulative_replacements"]
    near_keys = ["Fprime_hat", "F_hat", "p_hat_grid", "q_target_grid"]
    for key in exact_keys + near_keys:
        np.testing.assert_array_equal(np.asarray(solo[key]), np.asarray(repeat[key]))
    for candidate in [mixed, reordered]:
        for key in exact_keys:
            np.testing.assert_array_equal(
                np.asarray(solo[key]), np.asarray(candidate[key]))
        for key in near_keys:
            np.testing.assert_allclose(
                np.asarray(solo[key]), np.asarray(candidate[key]),
                rtol=2e-6, atol=1e-7)
    assert solo["cumulative_replacements"][-1] > 0


def test_preregistered_campaign_suppresses_generic_best_config(
        tmp_path, monkeypatch):
    final_path = tmp_path / "production_gpu_final_summary.csv"
    pd.DataFrame([{"run_id": "only"}]).to_csv(final_path, index=False)
    config_summary = pd.DataFrame([{"config_id": "only"}])
    monkeypatch.setattr(
        parallel, "summarize_configs", lambda _: config_summary.copy())

    def forbidden(*args, **kwargs):
        raise AssertionError("generic selector must not run")

    monkeypatch.setattr(parallel, "select_best_configs", forbidden)
    parallel.write_config_summaries(
        str(tmp_path), "production_gpu",
        {"selection": {"write_generic_best": False}}, logger=lambda _: None)
    assert (tmp_path / "production_gpu_config_summary.csv").exists()
    assert not (tmp_path / "best_configs.csv").exists()


def test_interrupted_run_level_resume_matches_uninterrupted_outputs(tmp_path):
    x, ref = _reference(nx=101, ny=201)
    cfg = _torch_cfg(n_steps=12, n_particles=24, eval_every=3)
    ev = metrics.EvalConfig.from_domain(DOMAIN)
    common = dict(
        eta=0.10, burnin_fraction=0.25, fr_every=2, stop_fraction=0.75)
    specs = [
        RunSpec("abf_fr_physical", "physical", seed=seed, gamma=gamma,
                **common)
        for seed, gamma in [(21, 0.05), (22, 0.10), (23, 0.20)]
    ]
    full_dir = tmp_path / "full"
    resume_dir = tmp_path / "resume"
    full_dir.mkdir()
    resume_dir.mkdir()
    kwargs = dict(
        cfg=cfg, prefix="production_gpu", x_grid=x, ref=ref, ev=ev,
        device=torch.device("cpu"), dtype=torch.float64,
        estimator="binned_smooth", batch_size=3, base_seed=73,
        tag="main", conditional="final", logger=lambda _: None)

    parallel.run_specs(
        specs, stage_root=str(full_dir), resume=True, force=False, **kwargs)
    parallel.merge_stage_csvs(
        str(full_dir), "production_gpu", logger=lambda _: None)

    first = parallel.run_specs(
        specs[:1], stage_root=str(resume_dir), resume=True, force=False,
        **kwargs)
    resumed = parallel.run_specs(
        specs, stage_root=str(resume_dir), resume=True, force=False, **kwargs)
    assert first["n_done"] == 1
    assert resumed["n_done"] == 2
    assert resumed["n_skipped"] == 1
    parallel.merge_stage_csvs(
        str(resume_dir), "production_gpu", logger=lambda _: None)

    sort_keys = {
        "runs_long": ["run_id", "step"],
        "final_summary": ["run_id"],
        "profiles": ["run_id", "x"],
        "fr_events": ["run_id", "step"],
    }
    for kind, keys in sort_keys.items():
        name = f"production_gpu_{kind}.csv"
        expected = pd.read_csv(full_dir / name).sort_values(keys).reset_index(drop=True)
        actual = pd.read_csv(resume_dir / name).sort_values(keys).reset_index(drop=True)
        if kind == "final_summary":
            expected = expected.drop(columns=["runtime_seconds"])
            actual = actual.drop(columns=["runtime_seconds"])
        pd.testing.assert_frame_equal(
            expected, actual, check_dtype=False, check_exact=False,
            rtol=2e-6, atol=1e-7)
