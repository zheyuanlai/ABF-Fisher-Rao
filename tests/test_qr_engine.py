"""Gate 0D and the engine-level arm gates for the q-r decoupling campaign.

Frozen protocol: ``docs/QR_DECOUPLING_PREREGISTRATION.md``.

Stage 0 proved the allocation *objects* mean what the protocol says.  These
gates prove the *engine* applies them, which is a different claim and the one
v3 lost an entire campaign to: the estimator was tested, the pipeline was not.

The load-bearing one is 0D.  A2 updates the Fisher--Rao mass and nothing else,
so it must be trajectory-identical to plain ABF -- not close, identical.  If it
is not, probability mass has leaked into the physical dynamics or into the
accumulator, and every margin the campaign later measures would be
uninterpretable.  Identity pairs run **in one process**: the engine is not
bitwise reproducible across processes.

Run: CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_qr_engine.py -q
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from abffr import metrics, qr_arms as qra, reference, simulation_torch
from abffr.io_utils import RunSpec

DOMAIN = {"x_min": -3.0, "x_max": 3.0, "y_min": -2.5, "y_max": 3.5}
TILT = 0.1021665783
BETA = 4.0


def _cfg(arm=None, n_steps=1200, n_particles=64, n_cells=16, **qr_extra):
    cfg = {
        "simulation": {
            "beta": BETA, "dt": 0.002, "n_steps": n_steps,
            "n_particles": n_particles, "eval_every": 200,
            "x_init_mode": "uniform", "y_init_mode": "uniform",
        },
        "domain": dict(DOMAIN),
        "potential": {"x_tilt": TILT},
        "abf": {
            "estimator": "binned_smooth",
            "observation_order": "post_propagation",
            "h": 0.05, "update_every": 10, "min_count": 1.0,
        },
        "fr": {"enabled": False, "noise_chunk_steps": 64},
    }
    if arm is not None:
        cfg["qr"] = dict(
            enabled=True, arm=arm, n_cells=n_cells,
            opportunity_every=100, burnin_fraction=0.25, stop_fraction=0.75,
            history_capacity=200, **qr_extra)
    return cfg


def _run(cfg, seed=11, kappa=None):
    if kappa is not None:
        cfg = dict(cfg, kappa={"cell": kappa})
    x = np.linspace(DOMAIN["x_min"], DOMAIN["x_max"], 161)
    y = np.linspace(DOMAIN["y_min"], DOMAIN["y_max"], 321)
    ref = reference.compute_reference(x, y, beta=BETA, x_tilt=TILT)
    spec = RunSpec(method="abf_only", target_type="none", seed=seed, gamma=0.0,
                   eta=0.10, burnin_fraction=0.0, fr_every=1, stop_fraction=1.0)
    return simulation_torch.run_batch(
        [spec], cfg=cfg, x_grid=x, F_ref=ref["F_ref"],
        Fprime_ref=ref["Fprime_ref"],
        ev=metrics.EvalConfig.from_domain(DOMAIN), device=torch.device("cpu"),
        dtype=torch.float64, estimator="binned_smooth", base_seed=0)


# --------------------------------------------------------------------------- #
# Gate 0D -- the mass-only identity
# --------------------------------------------------------------------------- #
def test_gate_0D_mass_only_arm_is_identical_to_plain_abf():
    """A2 == A0 on the free energy, exactly.

    A2 runs the whole allocation machinery -- cells, Fisher--Rao mass update,
    per-opportunity bookkeeping -- and changes no position and no accumulator.
    It also consumes no randomness, so this is an identity and not a tolerance.
    """
    a0 = _run(_cfg(None))
    a2 = _run(_cfg("A2"))
    for key in ("F_hat", "Fprime_hat", "X_snap", "Y_snap"):
        s0 = np.asarray(a0.diags[0][key], dtype=float)
        s2 = np.asarray(a2.diags[0][key], dtype=float)
        assert s0.shape == s2.shape and s0.size > 1
        assert np.array_equal(s0, s2), (
            f"{key} differs: mass leaked into the dynamics or the estimator "
            f"(max |delta| = {np.max(np.abs(s0 - s2)):.3e})")


def test_gate_0D_positive_control_an_arm_that_resamples_does_differ():
    """The identity gate must be able to fail.

    If A3 also matched A0, the comparison would be measuring nothing -- which is
    exactly the shape of a gate that cannot fire.
    """
    a0 = _run(_cfg(None))
    a3 = _run(_cfg("A3", benefit_threshold=0.0))
    s0 = np.asarray(a0.diags[0]["X_snap"], dtype=float)
    s3 = np.asarray(a3.diags[0]["X_snap"], dtype=float)
    assert not np.array_equal(s0, s3)


def test_mass_only_arm_still_does_the_fisher_rao_work():
    """A2 being inert physically must not mean it is inert."""
    a2 = _run(_cfg("A2"))
    rows = a2.qr_events
    assert rows and all(r["arm"] == "A2" for r in rows)
    assert not any(r["resampled"] for r in rows)
    assert len({round(r["mass_ess"], 12) for r in rows}) > 1, (
        "the mass ESS never moved; the Fisher-Rao step is not running")


# --------------------------------------------------------------------------- #
# The arms differ in r, and in nothing else
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("arm", ["A3", "A4a", "A4b", "A5"])
def test_every_allocation_arm_runs_and_reports_its_decisions(arm):
    res = _run(_cfg(arm, benefit_threshold=0.0))
    rows = res.qr_events
    assert rows and all(r["arm"] == arm for r in rows)
    assert any(r["resampled"] for r in rows), "no arm should be inert here"
    assert all(r["n_occupied"] > 0 for r in rows)
    assert res.diags[0]["qr_arm"] == arm


def test_A5_reports_the_multiplier_and_holds_its_ess_floor():
    res = _run(_cfg("A5", rho=0.5, benefit_threshold=0.0))
    # Opportunities skipped by the cooldown never computed an allocation, so
    # they carry no ESS to report.
    rows = [r for r in res.qr_events
            if "ess_predicted" in r and np.isfinite(r["ess_predicted"])]
    assert rows, "A5 must report the ESS it achieved"
    assert all(r["ess_predicted"] >= 0.5 - 1e-6 for r in rows)
    assert any(r["lam"] > 0.0 for r in rows), (
        "lambda never engaged; either the constraint is inactive everywhere or "
        "the bisection is not running")


def test_A4a_needs_no_difficulty_estimator_and_A4b_does():
    """The arm split is real: A4a is static, A4b reads the online stream."""
    a4a = _run(_cfg("A4a", benefit_threshold=0.0))
    assert a4a.qr_events
    arm = qra.QRArm(qra.QRConfig(arm="A4a"), 64,
                    np.linspace(-3.0, 3.0, 161),
                    np.ones(161, dtype=bool), BETA, 0.002, 10)
    arm.observe(np.zeros(4), np.zeros(4), np.zeros(4))
    assert arm._s2_n == 0, "A4a must not pay for an estimator it does not read"


def test_benefit_gate_can_suppress_a_resampling():
    """An opportunity is not an obligation."""
    loose = _run(_cfg("A4b", benefit_threshold=0.0))
    tight = _run(_cfg("A4b", benefit_threshold=0.99))
    n_loose = sum(r["resampled"] for r in loose.qr_events)
    n_tight = sum(r["resampled"] for r in tight.qr_events)
    assert n_loose > n_tight == 0


def test_clones_are_held_out_of_the_accumulator():
    """A clone may not contribute the observation that justified creating it."""
    res = _run(_cfg("A4b", benefit_threshold=0.0))
    fired = [r for r in res.qr_events if r["resampled"]]
    assert fired, "nothing resampled, so the hold cannot be observed"
    assert any(r.get("hold_max", 0) > 0 for r in fired), (
        "no clone was ever held out; the rejuvenation rule is not applied")


# --------------------------------------------------------------------------- #
# Configuration is a rejection, not a default
# --------------------------------------------------------------------------- #
def test_unknown_qr_key_is_rejected():
    with pytest.raises(ValueError, match="unknown qr keys"):
        qra.config_from_dict({"qr": {"enabled": True, "arm": "A3",
                                     "gamma_boost": 2.0}})


def test_window_must_close_before_the_end_of_the_run():
    with pytest.raises(ValueError, match="close strictly before"):
        qra.QRConfig(arm="A3", stop_fraction=1.0)


def test_qr_refuses_to_share_a_run_with_another_allocation_scheme():
    cfg = _cfg("A3")
    cfg["v4"] = {"enabled": True}
    with pytest.raises(ValueError, match="one allocation scheme per run"):
        _run(cfg)


def test_qr_refuses_the_observation_order_it_cannot_read():
    cfg = _cfg("A3")
    cfg["abf"]["observation_order"] = "pre_propagation"
    with pytest.raises(ValueError, match="post_propagation"):
        _run(cfg)


def test_resampling_never_lands_on_unrejuvenated_clones():
    """The cooldown, and the failure it was measured against.

    Before it, A4b fired at 22 of 24 opportunities and drove ancestor ESS to 8
    of 256: r* jitters with the noise in Gamma_hat, the occupancy test fires
    every time, and genealogy compounds because each resampling lands on clones
    that are not yet independent.  The rejuvenation bound already says how long
    that is, so the cooldown introduces no new parameter.
    """
    res = _run(_cfg("A4b", n_steps=4000, benefit_threshold=0.0))
    rows = res.qr_events
    fired = [r for r in rows if r["resampled"]]
    assert fired, "nothing resampled, so the cooldown cannot be observed"
    for r in fired:
        later = [q for q in rows if q["step"] > r["step"]]
        inside = [q for q in later if q["step"] < r["cooldown_until"]]
        assert not any(q["resampled"] for q in inside), (
            f"a resampling at {r['step']} was followed by another before its "
            f"clones finished rejuvenating at {r['cooldown_until']}")


# --------------------------------------------------------------------------- #
# A6 -- the allocation held by the bias instead of by birth-death
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("arm", ["A6a", "A6b"])
def test_bias_held_arms_never_resample_and_keep_their_genealogy(arm):
    """A6 pays no genealogy at all: that is the entire point of it."""
    res = _run(_cfg(arm, n_steps=3000))
    rows = res.qr_events
    assert rows, "A6 must still take opportunities, to refresh the increment"
    assert not any(r["resampled"] for r in rows)
    anc = np.asarray(res.diags[0]["ancestor_ess"], dtype=float)
    assert anc[-1] == pytest.approx(anc[0]), (
        "ancestor ESS moved without a resampling")


def test_bias_held_arm_carries_the_same_r_star_as_its_birth_death_twin():
    """A6a and A4a must differ in mechanism only, never in target.

    If the targets differed, a margin between them would not isolate the
    question A6 exists to ask.
    """
    a4 = _run(_cfg("A4a", n_steps=3000, benefit_threshold=0.0))
    a6 = _run(_cfg("A6a", n_steps=3000))
    r4 = [r["r_star"] for r in a4.qr_events if "r_star" in r]
    r6 = [r["r_star"] for r in a6.qr_events if "r_star" in r]
    assert r4 and r6
    assert np.allclose(r4[0], r6[0], rtol=1e-12), "same r*, different mechanism"


def test_bias_increment_is_the_gradient_of_log_r_over_beta():
    """The increment must be the force that makes r* stationary, not a fudge.

    ``p ∝ exp(-beta(F - A))`` with ``A = Fhat + log r*/beta`` gives ``p ∝ r*``;
    the engine adds a *force*, so the term is ``d/dz log r* / beta``.
    """
    x = np.linspace(-3.0, 3.0, 401)
    arm = qra.QRArm(qra.QRConfig(arm="A6a", n_cells=16), 64, x,
                    (x >= -2.5) & (x <= 2.5), BETA, 0.002, 10)
    r = np.exp(-0.5 * np.linspace(-1.0, 1.0, 16))
    r = r / r.sum()
    inc = arm.bias_increment(r)
    centres = 0.5 * (arm.edges[1:] + arm.edges[:-1])
    want = np.gradient(np.interp(x, centres, np.log(r)), x) / BETA
    assert np.allclose(inc, want, rtol=1e-12)
    assert inc.shape == x.shape


def test_bias_held_arm_actually_moves_the_occupancy():
    """A gate that could not fail: A6 must change where the replicas are."""
    a0 = _run(_cfg(None, n_steps=3000))
    a6 = _run(_cfg("A6a", n_steps=3000))
    x0 = np.asarray(a0.diags[0]["X_snap"])[-1]
    x6 = np.asarray(a6.diags[0]["X_snap"])[-1]
    edges = np.linspace(-3.0, 3.0, 17)
    o0 = np.bincount(np.clip(np.digitize(x0, edges) - 1, 0, 15), minlength=16)
    o6 = np.bincount(np.clip(np.digitize(x6, edges) - 1, 0, 15), minlength=16)
    tv = 0.5 * np.abs(o0 / o0.sum() - o6 / o6.sum()).sum()
    assert tv > 0.05, f"the bias increment moved the occupancy by only {tv:.3f}"


# --------------------------------------------------------------------------- #
# A6c -- the main candidate, and whether Fisher-Rao is actually doing anything
# --------------------------------------------------------------------------- #
def test_A6c_holds_its_mass_ess_floor_without_resampling():
    res = _run(_cfg("A6c", n_steps=4000, rho=0.5))
    rows = [r for r in res.qr_events if "fr_active" in r]
    assert rows and not any(r["resampled"] for r in rows)
    assert all(r["ess_predicted"] >= 0.5 - 1e-6 for r in rows)


def test_A6c_records_whether_fisher_rao_was_active_at_all():
    """The diagnostic that decides what an A6c win may be attributed to.

    If ``lambda`` is zero almost always then ``r_A6c == r_A6b`` and the mass
    layer contributed nothing, however well A6c performs.  A campaign that could
    not tell those apart would be free to credit Fisher-Rao for a result it had
    no part in.
    """
    res = _run(_cfg("A6c", n_steps=4000, rho=0.5))
    rows = [r for r in res.qr_events if "fr_active" in r]
    for key in ("fr_active", "tv_to_unconstrained", "mass_ess_unconstrained"):
        assert all(key in r for r in rows), key
    assert np.mean([r["fr_active"] for r in rows]) > 0.0, (
        "the constraint never bound; A6c is A6b here and must be reported so")
    assert np.mean([r["mass_ess_unconstrained"] for r in rows]) < 0.5, (
        "if the unconstrained arm already meets rho, the constraint is vacuous")


def test_A6c_is_more_constrained_than_A6b_and_says_so():
    """A6c > A6b is not a hypothesis anyone should hold -- it is backwards."""
    a6b = _run(_cfg("A6b", n_steps=3000))
    a6c = _run(_cfg("A6c", n_steps=3000, rho=0.5))
    rb = np.array([r["r_star"] for r in a6b.qr_events if "r_star" in r])
    rc = np.array([r["r_star"] for r in a6c.qr_events if "r_star" in r])
    n = min(len(rb), len(rc))
    assert n > 0 and np.mean(
        [0.5 * np.abs(rb[i] - rc[i]).sum() for i in range(n)]) > 0.01, (
        "A6c and A6b produced the same allocation; the constraint is inactive")
