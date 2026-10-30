"""Regression gates for the 2026-10 deca-alanine revision (``docs/DECA_FR_2026_PROTOCOL.md``).

Physics-level invariants, CPU only, tiny runs.  Each one guards a defect that would run cleanly
and produce plausible numbers:

  * the two BAOAB half-kicks must apply the SAME trust-weighted field (the 2026-08 code
    applied the ramp in the first half-kick only);
  * ``abf_bias_scale = 0`` is unbiased Langevin under the walls, and an infinite trust threshold
    must coincide with it bit for bit;
  * the histogram estimator applies its own-cell value and integrates a known force exactly;
  * the wall-corrected uniform target is flat inside the walls and reference-free;
  * zero FR rate / before FR start: ``fr_uniform`` is bit-identical to ``abf``;
  * a fixed FR stop is honoured (no events after it) and a clone does not deposit until it has
    been propagated.

Run: CUDA_VISIBLE_DEVICES="" PYTHONPATH=src python -m pytest tests/test_deca_fr_2026.py -q
"""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
torch.set_default_dtype(torch.float64)

pytest.importorskip("openmm")

from alkanes import interval as iv                                          # noqa: E402
from deca import system as dsys                                            # noqa: E402
from deca.core import (DecaSimConfig, free_energy_from_histogram_force,     # noqa: E402
                       run_sampler_deca, wall_uniform_target)
from deca.engine import DecaEngine                                         # noqa: E402

R_LO, R_HI, NG = 1.20, 3.60, 65
KB = 0.008314462618


@pytest.fixture(scope="module")
def engine():
    _, _, system = dsys.make_system(10)
    return DecaEngine(system, device="cpu", dtype=torch.float64, compiled=False)


def _cfg(**kw):
    base = dict(n_walkers=8, n_steps=60, n_grid=NG, save_every=30, xi_trace_every=10,
                label_every=30, abf_warmup_steps=1, fr_start_steps=20, fr_every=10)
    base.update(kw)
    return DecaSimConfig(**base)


def _init(R, N, seed=0):
    g = np.random.default_rng(seed)
    x = dsys.build_helix(-57.0, -47.0)
    return np.stack([[x + 0.004 * g.standard_normal(x.shape) for _ in range(N)]
                     for _ in range(R)])


def _init_mixed(R, N, seed=0):
    """Six helical + two polyproline-like walkers per seed: the KDE marginal is then strongly
    non-uniform, so the recentred FR score is O(1) and birth--death events are certain.  With
    eight identical helices the recentred score is ~1e-3 and nothing ever fires."""
    g = np.random.default_rng(seed)
    xs = [dsys.build_helix(-57.0, -47.0)] * (N - 2) + [dsys.build_helix(-75.0, 145.0)] * 2
    return np.stack([[x + 0.004 * g.standard_normal(x.shape) for x in xs] for _ in range(R)])


def _run(engine, cfg, method="abf", R=1, N=8, mixed=False):
    x0 = _init_mixed(R, N) if mixed else _init(R, N)
    return run_sampler_deca(method, engine, cfg, list(range(R)), x0, device="cpu", verbose=False)


# =========================================================== both half-kicks use the guarded field
@pytest.mark.parametrize("estimator", ["kernel", "histogram"])
def test_infinite_trust_threshold_equals_zero_bias_scale(engine, estimator):
    """With C_min = inf the applied field is exactly zero in BOTH half-kicks, so the trajectory
    must coincide bit for bit with ``abf_bias_scale = 0`` (walls only).  The 2026-08 code fails
    this: its second half-kick interpolated the unweighted profile."""
    a = _run(engine, _cfg(abf_estimator=estimator, abf_min_count=1e300))
    b = _run(engine, _cfg(abf_estimator=estimator, abf_bias_scale=0.0))
    assert np.array_equal(a["xi_trace"], b["xi_trace"])


def test_legacy_flag_reproduces_the_defect(engine):
    """The replay flag must change the trajectory (it re-enables the unguarded second kick)."""
    a = _run(engine, _cfg(abf_min_count=1e300))
    c = _run(engine, _cfg(abf_min_count=1e300, legacy_unweighted_second_kick=True))
    assert not np.array_equal(a["xi_trace"], c["xi_trace"])
    with pytest.raises(ValueError):
        _run(engine, _cfg(abf_estimator="histogram", legacy_unweighted_second_kick=True))


def test_estimate_is_not_ramped_with_the_applied_bias(engine):
    out = _run(engine, _cfg(abf_min_count=1e300))
    mf = out["mean_force"][-1]
    assert np.isfinite(mf).all() and np.abs(mf).max() > 0.0


# =========================================================== histogram estimator
def test_histogram_integration_recovers_a_known_force():
    grid, dz = iv.interval_grid(257, R_LO, R_HI, device="cpu", dtype=torch.float64)
    F_true = 30.0 * (grid - 2.0) ** 2 - 8.0 * torch.sin(3.0 * grid)
    edges = torch.cat([grid - 0.5 * dz, grid[-1:] + 0.5 * dz])
    Fe = 30.0 * (edges - 2.0) ** 2 - 8.0 * torch.sin(3.0 * edges)
    gamma = ((Fe[1:] - Fe[:-1]) / dz)[None]              # exact cell averages of F'
    F_hat = free_energy_from_histogram_force(gamma, dz)[0]
    a = F_true - F_true.mean()
    b = F_hat - F_hat.mean()
    err = float((a - b).abs().max() / (a.max() - a.min()))
    assert err < 2e-3, f"integration error {err:.2e} of the span"


def test_histogram_arm_applies_its_own_cell_value(engine):
    """The stored profile is C-weighted sums per cell and the bias is piecewise constant."""
    out = _run(engine, _cfg(abf_estimator="histogram", abf_min_count=1.0))
    C, S = out["final_counts"][0], out["final_force_sum"][0]
    mf = out["mean_force"][-1][0]
    occupied = C > 0
    assert occupied.any()
    assert np.allclose(mf[occupied], S[occupied] / C[occupied])
    assert np.all(mf[~occupied] == 0.0)
    assert C.sum() == pytest.approx(out["xi_trace"].shape[-1] * (out["config"]["n_steps"] + 1))


# =========================================================== wall-corrected uniform target
def test_wall_uniform_target_is_flat_inside_the_walls():
    sim = DecaSimConfig(n_grid=129)
    grid, dz = iv.interval_grid(sim.n_grid, sim.R_lo, sim.R_hi, device="cpu", dtype=torch.float64)
    q = wall_uniform_target(grid, dz, sim)
    inside = (grid >= sim.wall_lo) & (grid <= sim.wall_hi)
    assert float(q.sum() * dz) == pytest.approx(1.0, abs=1e-12)
    assert torch.allclose(q[inside], q[inside][0])
    assert float(q[0]) < float(q[inside][0]) and float(q[-1]) < float(q[inside][0])
    # exactly exp(-beta V_wall) up to normalisation
    V = 0.5 * sim.k_wall * ((grid - sim.wall_lo).clamp(max=0) ** 2
                            + (grid - sim.wall_hi).clamp(min=0) ** 2)
    want = torch.exp(-sim.beta * V)
    assert torch.allclose(q / q[inside][0], want, atol=1e-12)


# =========================================================== arm equivalence
@pytest.mark.parametrize("estimator", ["kernel", "histogram"])
def test_zero_rate_fr_uniform_is_bit_identical_to_abf(engine, estimator):
    cfg = _cfg(fr_rate=0.0, abf_estimator=estimator)
    a = _run(engine, cfg, "abf", R=2)
    b = _run(engine, cfg, "fr_uniform", R=2)
    assert np.array_equal(a["xi_trace"], b["xi_trace"])
    assert np.array_equal(a["pmf"], b["pmf"])
    assert int(b["total_replacement_events"].sum()) == 0


def test_before_fr_start_fr_uniform_coincides_with_abf(engine):
    cfg = _cfg(fr_rate=50.0, fr_start_steps=10_000, n_steps=40)
    a = _run(engine, cfg, "abf", R=2)
    b = _run(engine, cfg, "fr_uniform", R=2)
    assert np.array_equal(a["xi_trace"], b["xi_trace"])
    assert int(b["total_replacement_events"].sum()) == 0


def test_fr_uniform_never_sees_a_reference(engine):
    with pytest.raises(AssertionError, match="NO-REFERENCE-LEAKAGE"):
        run_sampler_deca("fr_uniform", engine, _cfg(fr_rate=1.0), [0], _init(1, 8),
                         reference_free_energy=np.zeros(NG), device="cpu", verbose=False)


# =========================================================== events, stop, clone deposits
def test_fixed_stop_is_honoured_and_clones_skip_one_deposit(engine):
    cfg = _cfg(fr_rate=500.0, fr_start_steps=20, fr_every=5, fr_stop_steps=40,
               max_event_fraction=0.5, n_steps=80, save_every=10)
    out = _run(engine, cfg, "fr_uniform", mixed=True)
    n_ev = int(out["total_replacement_events"].sum())
    assert n_ev > 0, "the test needs some events to be meaningful"
    assert out["ev_step"].max() < 40
    assert (out["ev_step"] >= 20).all()
    assert out["fr_active"][out["steps"] >= 40].sum() == 0
    # each clone skips exactly one deposit
    expected = out["xi_trace"].shape[-1] * (cfg.n_steps + 1) - n_ev
    assert out["final_counts"][0].sum() == pytest.approx(expected)
    # population is conserved and the event log is consistent with the histograms
    assert out["birth_hist"].sum() == n_ev == out["death_hist"].sum() == out["ev_slot"].size


def test_online_controller_runs_and_stops_once(engine):
    spec = dict(name="R2test", trigger="occupancy", check_every=0.005, W=0.010, R=1, level=0.0,
                persist=1, stop_factor=1.0, support_count=0.0, stability_tol=1e9,
                stability_gate=False, essw_min=0.0, maxlin_min=1.0, maxlin_k=0.0)
    cfg = _cfg(fr_rate=500.0, fr_start_steps=20, fr_every=5, max_event_fraction=0.5,
               n_steps=100, save_every=10, fr_controller=spec,
               fr_support_interval=(1.30, 3.50))
    out = _run(engine, cfg, "fr_uniform", mixed=True)
    ctl = out["controller"][0]
    assert int(out["total_replacement_events"].sum()) > 0
    # level = 0 is established at the first check that has a trailing window (t_start + W)
    assert ctl["reason"] == "established"
    assert ctl["t_stop"] == pytest.approx(0.030)
    assert out["fr_stop_step"][0] == 30
    # the decision is taken AFTER the selection of the same step (as in the ERI engine, where the
    # chunk up to the check time has already run), so an event at step 30 is allowed, none after
    assert out["ev_step"].max() <= 30
    assert (out["ev_step"] <= out["fr_stop_step"][0]).all()
    with pytest.raises(ValueError):
        _run(engine, _cfg(fr_rate=1.0, fr_controller=spec, fr_stop_steps=10), "fr_uniform")
