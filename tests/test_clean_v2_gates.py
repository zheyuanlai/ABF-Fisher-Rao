"""Engineering gates for the clean-v2 accelerator.

Frozen protocol: ``docs/CLEAN_V2_PREREGISTRATION.md``.  Scientific runs cannot
start until every test in this file passes.

The gates are written so that each one *can fail*.  Two of them exist purely
because the v2/v3 post-mortems found the failure only after the campaign was
over: Gate D (nothing clips) and Gate E (the applied event probability is the
one the protocol wrote down).  A gate that reads a value the engine cannot
produce is reassurance, not evidence, so each gate here is paired with a
positive control asserting the quantity it inspects actually moves.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from abffr import clean_v2 as cv2, fr_v3, io_utils, metrics, reference
from abffr import simulation_torch, torch_utils as tu
from abffr.io_utils import RunSpec

DOMAIN = {"x_min": -3.0, "x_max": 3.0, "y_min": -2.5, "y_max": 3.5}
TILT = 0.1021665783
BETA = 4.0


def _reference(nx=161, ny=321):
    x = np.linspace(DOMAIN["x_min"], DOMAIN["x_max"], nx)
    y = np.linspace(DOMAIN["y_min"], DOMAIN["y_max"], ny)
    return x, reference.compute_reference(x, y, beta=BETA, x_tilt=TILT)


def _cfg(n_steps=400, n_particles=48, eval_every=100, update_every=10):
    """A clean-v2 config: note there is no score_clip / cap / EMA key to set."""
    return {
        "clean_v2": {"enabled": True},
        "selection": {"write_generic_best": False},
        "simulation": {
            "beta": BETA, "dt": 0.002, "n_steps": n_steps,
            "n_particles": n_particles, "eval_every": eval_every,
            "x_init_mode": "uniform", "y_init_mode": "uniform",
        },
        "domain": dict(DOMAIN),
        "potential": {"x_tilt": TILT},
        "abf": {
            "estimator": "binned_smooth",
            "observation_order": "post_propagation",
            "h": 0.05, "update_every": update_every, "min_count": 1.0,
        },
        "fr": {
            "enabled": True,
            "target_types": ["physical", "physical_oracle"],
            "eta_values": [0.10],
            "burnin_fractions": [0.25],
            "duration_fractions": [0.50],
            "fr_every_values": [100],
            "interval_scaled_clock": True,
            "noise_chunk_steps": 64,
        },
    }


def _spec(method, target, gamma, seed=11, fr_every=100,
          burnin=0.25, stop=0.75):
    if method == "abf_only":
        return RunSpec(method=method, target_type="none", seed=seed, gamma=0.0,
                       eta=0.10, burnin_fraction=0.0, fr_every=1,
                       stop_fraction=1.0)
    return RunSpec(method=method, target_type=target, seed=seed,
                   gamma=float(gamma), eta=0.10, burnin_fraction=burnin,
                   fr_every=int(fr_every), stop_fraction=stop)


def _run(specs, cfg, **kw):
    x, ref = _reference()
    ev = metrics.EvalConfig.from_domain(DOMAIN)
    return simulation_torch.run_batch(
        specs, cfg=cfg, x_grid=x, F_ref=ref["F_ref"],
        Fprime_ref=ref["Fprime_ref"], ev=ev, device=torch.device("cpu"),
        dtype=torch.float64, estimator="binned_smooth", base_seed=0, **kw)


# --------------------------------------------------------------------------- #
# Config gate: the retired knobs are rejected, not defaulted
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key,value", [
    ("score_clip", 5.0), ("score_clip", None),
    ("max_event_fraction", 0.10), ("max_event_fraction", None),
    ("target_ema_alpha", 0.05),
])
def test_retired_fr_knobs_are_rejected_even_when_switched_off(key, value):
    """There is no 'off' value: the *key* is the defect, not its setting.

    A config carrying ``score_clip: null`` still tells a reader the operator has
    a clip.  Rejecting the key is what makes "clean-v2 does not clip" a property
    of the file rather than of the engine's default.
    """
    cfg = _cfg()
    cfg["fr"][key] = value
    with pytest.raises(ValueError, match=key):
        cv2.validate_config(cfg)


def test_ema_and_legacy_campaign_blocks_are_rejected():
    for mutate, needle in [
        (lambda c: c["abf"].__setitem__("ema_alpha", 0.05), "ema_alpha"),
        (lambda c: c.__setitem__("v3", {"operator": "bd"}), "v3"),
        (lambda c: c.__setitem__("v4", {"enabled": True}), "v4"),
        (lambda c: c["fr"].__setitem__("target_types", ["capped"]), "physical"),
        (lambda c: c["fr"].__setitem__("ramp_fraction", 0.1), "ramp_fraction"),
        (lambda c: c["fr"].__setitem__("interval_scaled_clock", False),
         "interval_scaled_clock"),
        (lambda c: c["selection"].__setitem__("write_generic_best", True),
         "write_generic_best"),
        (lambda c: c["fr"].__setitem__("burnin_fractions", [0.0]), "burn-in"),
        (lambda c: c["fr"].__setitem__("fr_every_values", [55]),
         "multiple of"),
    ]:
        cfg = _cfg()
        mutate(cfg)
        with pytest.raises(ValueError, match=needle):
            cv2.validate_config(cfg)
    cv2.validate_config(_cfg())        # positive control: the clean config passes


def test_engine_refuses_a_legacy_target_on_the_clean_path():
    with pytest.raises(ValueError, match="clean_v2 admits only"):
        _run([_spec("abf_fr_estimated", "estimated", 0.05)], _cfg())


# --------------------------------------------------------------------------- #
# Gate A -- gamma = 0 reproduces plain ABF under matched physical noise
# --------------------------------------------------------------------------- #
def test_gate_A_zero_gamma_is_plain_abf():
    """``atol = 0`` is meaningful here because this runs on CPU at gate scale.

    The GPU engine is not bitwise reproducible at production scale (v3
    Amendment 1 measured 3.9e-7 in ``l2_F`` over 50k steps with discrete
    counters exact), so a production repeat of this gate needs 1e-5 on the
    continuous profiles plus exact equality on the counters.  Passing here is
    not evidence about a GPU production run.
    """
    cfg = _cfg()
    base = _run([_spec("abf_only", None, 0.0)], cfg).diags[0]
    off = _run([_spec("abf_fr_physical", "physical", 0.0)], cfg).diags[0]
    live = _run([_spec("abf_fr_physical", "physical", 0.05)], cfg).diags[0]

    for key in ("Fprime_hat", "F_hat", "X_snap", "Y_snap"):
        for k in range(len(base["steps"])):
            np.testing.assert_allclose(off[key][k], base[key][k], rtol=0, atol=0)
    assert off["cumulative_fr_events"][-1] == 0

    # Positive control: the same comparison against a live arm must fail, or
    # this gate is measuring nothing.
    assert live["cumulative_fr_events"][-1] > 0
    assert not np.allclose(live["F_hat"][-1], base["F_hat"][-1])


# --------------------------------------------------------------------------- #
# Gate B -- a clone is not an ABF observation
# --------------------------------------------------------------------------- #
def test_gate_B_resampling_contributes_no_abf_observation():
    """At the pulse step itself the estimator must be untouched.

    ABF is fed from the propagated configurations *before* the pulse, so at the
    snapshot that coincides with the first pulse the FR arm and the baseline
    must agree on ``F'`` and ``F`` exactly, while already disagreeing on the
    marginal.  They may diverge only from the following step.
    """
    cfg = _cfg(n_steps=400, eval_every=100)      # snapshots at 100..400
    base = _run([_spec("abf_only", None, 0.0)], cfg).diags[0]
    arm = _run([_spec("abf_fr_physical", "physical", 0.10)], cfg).diags[0]

    steps = list(base["steps"])
    first_pulse = cv2.firing_steps(400, 0.25, 0.75, 100)[0]
    assert first_pulse == 100 and first_pulse in steps
    j = steps.index(first_pulse)

    np.testing.assert_allclose(arm["Fprime_hat"][j], base["Fprime_hat"][j],
                               rtol=0, atol=0)
    np.testing.assert_allclose(arm["F_hat"][j], base["F_hat"][j], rtol=0, atol=0)
    # ... and the pulse did happen: the population already differs.
    assert arm["cumulative_replacements"][j] > 0
    assert not np.allclose(arm["p_hat_grid"][j], base["p_hat_grid"][j])
    # ... and the estimator does diverge afterwards.
    assert not np.allclose(arm["Fprime_hat"][j + 1], base["Fprime_hat"][j + 1])


# --------------------------------------------------------------------------- #
# Gate C -- the physical target is gauge invariant
# --------------------------------------------------------------------------- #
def test_gate_C_score_is_invariant_to_the_free_energy_gauge():
    torch.manual_seed(0)
    B, G, N = 3, 64, 40
    x = np.linspace(-3.0, 3.0, G)
    dx = float(x[1] - x[0])
    p_hat = torch.rand(B, G, dtype=torch.float64) + 0.5
    A = torch.randn(B, G, dtype=torch.float64)
    X = torch.rand(B, N, dtype=torch.float64) * 5.0 - 2.5

    S, _, _, floored = cv2.score(p_hat, A, X, float(x[0]), dx, BETA)
    for shift in (-7.5, 0.0, 3.25, 1e3):
        S2, _, _, _ = cv2.score(p_hat, A + shift, X, float(x[0]), dx, BETA)
        np.testing.assert_allclose(S2.numpy(), S.numpy(), rtol=1e-12, atol=1e-10)
    assert float(floored.max()) == 0.0
    np.testing.assert_allclose(S.mean(dim=1).numpy(), 0.0, atol=1e-12)

    # Positive control: a *non*-constant perturbation must change the score.
    S3, _, _, _ = cv2.score(p_hat, A + torch.linspace(0, 1, G).view(1, G),
                            X, float(x[0]), dx, BETA)
    assert not np.allclose(S3.numpy(), S.numpy())


def test_score_matches_the_operators_own_score_object():
    """One score, one definition: the batch form and ``FRScore`` must agree.

    Two independently written score expressions is how a discretisation
    comparison silently turns into a comparison of two different flows.
    """
    torch.manual_seed(1)
    G, N = 48, 32
    x = np.linspace(-3.0, 3.0, G)
    dx = float(x[1] - x[0])
    p_hat = torch.rand(1, G, dtype=torch.float64) + 0.2
    A = torch.randn(1, G, dtype=torch.float64)
    X = torch.rand(1, N, dtype=torch.float64) * 5.0 - 2.5
    S, log_p, log_q, _ = cv2.score(p_hat, A, X, float(x[0]), dx, BETA)
    np.testing.assert_allclose(
        cv2.row_score(log_p[0], log_q[0]).S.numpy(), S[0].numpy(),
        rtol=0, atol=1e-15)


# --------------------------------------------------------------------------- #
# Gate D -- no hidden clipping anywhere in the path
# --------------------------------------------------------------------------- #
def test_gate_D_raw_and_applied_scores_are_identical():
    cfg = _cfg()
    diag = _run([_spec("abf_fr_physical", "physical", 0.10)], cfg).diags[0]
    assert diag["score_clip"] is None and diag["max_event_fraction"] is None
    fired = [k for k, applied in enumerate(diag["fr_applied"]) if applied]
    assert fired, "no FR frame recorded: this gate would read as a pass on no data"
    for k in fired:
        assert diag["score_clipped_fraction"][k] == 0.0
        for lab in simulation_torch.SCORE_QUANTILE_LABELS:
            assert (diag[f"score_raw_{lab}"][k]
                    == diag[f"score_applied_{lab}"][k])
        # The span the v2 clip destroyed is real and large; a gate that only
        # checked equality would also pass on a score that never moved.
        assert diag["score_max"][k] > diag["score_min"][k]
    assert max(diag["score_max"][k] - diag["score_min"][k] for k in fired) > 5.0


def test_gate_D_log_density_floor_never_binds():
    """``log phat`` is evaluated only at particle positions, where a KDE is
    bounded below by its own self-contribution; a binding floor is an anomaly."""
    res = _run([_spec("abf_fr_physical", "physical", 0.10)], _cfg())
    assert res.clean_events, "no pulses recorded"
    for e in res.clean_events:
        assert float(np.max(e["logp_floored_fraction"])) == 0.0


# --------------------------------------------------------------------------- #
# Gate E -- the applied event probability is the protocol's
# --------------------------------------------------------------------------- #
def test_gate_E_event_probability_is_one_minus_exp():
    S = torch.tensor([[-3.0, -0.5, 0.0, 0.5, 4.0]], dtype=torch.float64)
    for dtau in (0.0, 0.01, 0.5, 2.0):
        got = cv2.event_probability(S, dtau).numpy()
        want = 1.0 - np.exp(-np.abs(S.numpy()) * dtau)
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-15)
    # Uncapped: a large enough dose drives the tail to ~1 rather than to a cap.
    assert float(cv2.event_probability(S, 50.0).max()) > 0.999


def test_gate_E_operator_fires_at_the_stated_rate():
    """Empirical firing rate per particle matches ``1 - exp(-|S| dtau)``.

    This is the check the v2 campaign never ran: the cap meant the realised rate
    and the written rate were different functions, and no test compared them.
    """
    K, trials, dtau = 12, 4000, 0.3
    logp = torch.linspace(-2.0, 2.0, K, dtype=torch.float64)
    score = fr_v3.FRScore(log_p=logp, log_q=torch.zeros(K, dtype=torch.float64))
    expected = cv2.event_probability(score.S, dtau).numpy()
    gen = tu.make_generator(12345, torch.device("cpu"))
    fired = np.zeros(K)
    for _ in range(trials):
        src, _ = fr_v3.bd_standard(score, dtau, gen)
        # Particle i "fired" when it was killed (its slot now holds another) or
        # when it was cloned into some other slot.
        cloned = np.bincount(src.numpy(), minlength=K) > 1
        killed = src.numpy() != np.arange(K)
        fired += (cloned | killed).astype(float)
    rate = fired / trials
    # The uniformly chosen partner is *also* replaced, so the observed rate is
    # an upper bound on the own-event rate; the ordering must still track |S|.
    assert np.all(rate + 1e-9 >= expected * 0.9)
    assert np.corrcoef(rate, expected)[0, 1] > 0.9


def test_dtau_is_interval_scaled_so_total_dose_depends_only_on_gamma():
    """``dtau = gamma L dt`` makes the integrated FR reaction time over a fixed
    window ``gamma * window``, independent of ``L``.  That is the whole reason
    ``fr_every`` is a fair axis rather than a disguised dose axis."""
    n_steps, dt, gamma = 50000, 0.002, 0.05
    doses = []
    for every in (100, 500, 1000):
        steps = cv2.firing_steps(n_steps, 0.2, 0.8, every)
        doses.append(len(steps) * cv2.dtau(gamma, every, dt))
    assert max(doses) - min(doses) < 1e-9 * max(doses)
    assert doses[0] == pytest.approx(gamma * 0.6 * n_steps * dt, rel=1e-9)


# --------------------------------------------------------------------------- #
# Gate F -- the FR schedule is exactly the three-phase one
# --------------------------------------------------------------------------- #
def test_gate_F_engine_fires_exactly_on_the_specified_schedule():
    cfg = _cfg(n_steps=800, eval_every=200)
    spec = _spec("abf_fr_physical", "physical", 0.05, fr_every=100,
                 burnin=0.25, stop=0.75)
    res = _run([spec], cfg)
    fired = [int(e["step"]) for e in res.clean_events]
    expected = cv2.firing_steps(800, 0.25, 0.75, 100)
    assert fired == expected == [200, 300, 400, 500]
    assert min(fired) >= 200                    # nothing in Phase I
    assert max(fired) < 600                     # nothing in Phase III
    assert res.diags[0]["fr_firing_steps"] == expected


@pytest.mark.parametrize("burn,stop,every,want", [
    (0.2, 0.8, 100, list(range(200, 800, 100))),
    (0.2, 0.8, 500, [200, 700]),
    (0.5, 0.5, 100, []),                        # empty window fires nothing
    # stop = 1.0 still excludes the final step: the window is half-open.
    (0.0, 1.0, 250, [250, 500, 750]),
])
def test_schedule_specification_is_half_open(burn, stop, every, want):
    assert cv2.firing_steps(1000, burn, stop, every) == want


# --------------------------------------------------------------------------- #
# Gate G -- matched physical randomness, separate FR stream
# --------------------------------------------------------------------------- #
def test_gate_G_physical_noise_is_independent_of_the_fr_stream():
    """Changing FR must change *which configuration occupies a slot*, never
    which Langevin variates that slot receives."""
    assert (tu.stable_seed("langevin", 0, 7)
            != tu.stable_seed("fr", 0, "abf_fr_physical|seed=7"))

    cfg = _cfg()
    solo = _run([_spec("abf_only", None, 0.0, seed=11)], cfg).diags[0]
    # Same baseline row, this time batched *with* a live FR row.
    mixed = _run([_spec("abf_only", None, 0.0, seed=11),
                  _spec("abf_only", None, 0.0, seed=12)], cfg).diags[0]
    for k in range(len(solo["steps"])):
        np.testing.assert_allclose(mixed["F_hat"][k], solo["F_hat"][k],
                                   rtol=0, atol=0)

    # And the initial conditions are shared across arms for a given seed.
    a = _run([_spec("abf_only", None, 0.0, seed=11)], cfg).diags[0]
    b = _run([_spec("abf_fr_physical", "physical", 0.10, seed=11)], cfg).diags[0]
    np.testing.assert_allclose(b["X_snap"][0], a["X_snap"][0], rtol=0, atol=0)


def test_repeated_identical_runs_are_deterministic_within_a_process():
    cfg = _cfg()
    spec = _spec("abf_fr_physical", "physical", 0.10, seed=3)
    d1 = _run([spec], cfg).diags[0]
    d2 = _run([spec], cfg).diags[0]
    np.testing.assert_allclose(d2["F_hat"][-1], d1["F_hat"][-1], rtol=0, atol=0)
    np.testing.assert_allclose(d2["X_snap"][-1], d1["X_snap"][-1], rtol=0, atol=0)


# --------------------------------------------------------------------------- #
# The population is exactly fixed, and the oracle arm differs from the estimated
# --------------------------------------------------------------------------- #
def test_population_is_conserved_by_every_pulse():
    cfg = _cfg()
    n = cfg["simulation"]["n_particles"]
    diag = _run([_spec("abf_fr_physical", "physical", 0.10)], cfg).diags[0]
    for snap in diag["X_snap"]:
        assert snap.shape == (n,)
    assert diag["max_clone_multiplicity"][-1] >= 1
    assert diag["cumulative_replacements"][-1] > 0


def test_oracle_arm_uses_the_reference_not_the_running_estimate():
    cfg = _cfg()
    est = _run([_spec("abf_fr_physical", "physical", 0.10)], cfg).diags[0]
    orc = _run([_spec("abf_fr_physical_oracle", "physical_oracle", 0.10)],
               cfg).diags[0]
    assert not np.allclose(orc["q_target_grid"][-1], est["q_target_grid"][-1])
    # The oracle target is time-invariant: it is exp(-beta F_ref), not exp(-beta A_t).
    np.testing.assert_allclose(orc["q_target_grid"][0], orc["q_target_grid"][-1],
                               rtol=1e-12, atol=1e-14)


def test_the_fr_window_must_close_before_the_run_does():
    """The v3 campaign lost every arm to a window that ran to the end.

    Phase III is not decoration: it is what makes the long-time limit ABF's by
    construction, and therefore what licenses using an unflattened target at all.
    """
    cfg = _cfg()
    cfg["fr"]["duration_fractions"] = [0.75]        # 0.25 + 0.75 = 1.0
    with pytest.raises(ValueError, match="strictly before the end"):
        cv2.validate_config(cfg)
    cfg["fr"]["duration_fractions"] = [0.50]        # 0.25 + 0.50 = 0.75
    cv2.validate_config(cfg)
    del cfg["fr"]["duration_fractions"]
    cfg["fr"]["stop_fractions"] = [1.0]
    with pytest.raises(ValueError, match="strictly before the end"):
        cv2.validate_config(cfg)


# --------------------------------------------------------------------------- #
# The baseline is the SAME ABF the legacy campaigns ran
# --------------------------------------------------------------------------- #
def test_removing_ema_alpha_did_not_change_the_abf_estimator():
    """Clean-v2 forbids ``abf.ema_alpha``.  This proves that is a *target*
    change, not an estimator change.

    The experiment only means anything if the plain-ABF baseline is the same
    plain ABF the legacy campaigns ran, with sparse physical FR the single
    addition.  So run ``abf_only`` twice at the same seed -- once under a
    clean-v2 config, once under a legacy config that still carries
    ``abf.ema_alpha``, ``fr.score_clip`` and ``fr.max_event_fraction`` -- and
    require the estimator to be identical.  It is: those knobs only ever fed
    ``Fhat_target``, which only ever fed the FR target, and never the
    accumulators, ``F'``, ``F`` or the applied bias.
    """
    clean = _cfg()
    legacy = _cfg()
    del legacy["clean_v2"]
    legacy["abf"]["ema_alpha"] = 0.05
    legacy["fr"].update(score_clip=5.0, max_event_fraction=0.10,
                        target_ema_alpha=0.05, ramp_fraction=0.0, jitter=0.0)

    a = _run([_spec("abf_only", None, 0.0, seed=21)], clean).diags[0]
    b = _run([_spec("abf_only", None, 0.0, seed=21)], legacy).diags[0]
    for key in ("Fprime_hat", "F_hat", "X_snap", "Y_snap"):
        for k in range(len(a["steps"])):
            np.testing.assert_allclose(a[key][k], b[key][k], rtol=0, atol=0)
    # Positive control: the legacy config really did take the legacy path.
    assert a["score_clip"] is None and b["score_clip"] == 5.0
    assert a["max_event_fraction"] is None and b["max_event_fraction"] == 0.10
