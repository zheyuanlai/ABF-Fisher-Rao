"""Tests of the early-stopping / per-arm-stream extension of src/wca_numba.py (2026-10-20).

Design: deliverables/UROPS_2026_WCA_timing/PREREGISTRATION.md.  The ORIGINAL engine is loaded from
WCA_ORIG_ENGINE (default: the main checkout at commit 13e77275, src/wca_numba.py) to prove that the extension
is bitwise inert when it is not used.  Short deterministic jobs (N = 16 replicas, a few thousand steps, FR
active from early on with an enlarged rate so that events occur).
"""
import importlib.util
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np  # noqa: E402
import pytest  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import wca_numba as wn  # noqa: E402

ORIG = os.environ.get("WCA_ORIG_ENGINE", "/home/zheyuanlai/ABF-Fisher-Rao/src/wca_numba.py")
N, NS = 16, 3000
SAVE = np.arange(0, NS + 1, 100)
KEYS = ("M_bias", "C_bias", "M_prod", "C_prod", "M_rep", "C_rep", "z_hist", "repl_cumulative", "frac_regions",
        "ancestor_ess", "ancestor_ess_window", "max_ancestor_frac", "n_unique_ancestor", "n_event_opportunities",
        "n_cap_binding", "q_final", "n_c2s", "n_s2c")


def cfg(**kw):
    c = dict(wn.ACCEPTED_CFG, dt=1.25e-4, abf_warmup_steps=400, estimator_burn_in_steps=400, fr_start_steps=0,
             fr_every=80, ess_window_steps=800, fr_rate=5.0, max_event_fraction=0.25)
    c.update(kw)
    return c


def run(arms, c=None, **kw):
    return wn.run_ladder_point(3, N, NS, arms, cfg=c or cfg(), save_at=SAVE, cap_min=0, **kw)


def same(a, b, i=0, j=0, keys=KEYS):
    bad = [k for k in keys if not np.array_equal(np.asarray(a[k])[i], np.asarray(b[k])[j], equal_nan=True)]
    return bad


@pytest.fixture(scope="module")
def orig():
    if not os.path.exists(ORIG):
        pytest.skip(f"original engine not found: {ORIG}")
    spec = importlib.util.spec_from_file_location("wca_numba_orig", ORIG)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_backward_compatibility_bitwise_vs_original_engine(orig):
    """(1) stopping disabled = the original engine, bitwise, for ABF + FR + sham in one job and for an FR job."""
    arms = [("abf", False), ("fr", True), ("sham", "sham:fr")]
    a = run(arms)
    b = orig.run_ladder_point(3, N, NS, arms, cfg=cfg(), save_at=SAVE, cap_min=0)
    for i in range(3):
        assert not same(a, b, i, i), (arms[i], same(a, b, i, i))
    assert a["repl_cumulative"][1, -1] > 0, "the test must exercise FR events"
    c = run([("fr", True)], fr_stop_steps=None, fr_stream_ids=None)
    d = orig.run_ladder_point(3, N, NS, [("fr", True)], cfg=cfg(), save_at=SAVE, cap_min=0)
    assert not same(c, d)


def test_stream_id_zero_is_the_default_stream():
    a = run([("fr", True)])
    b = run([("fr", True)], fr_stream_ids=[0])
    assert not same(a, b)
    c = run([("fr", True)], fr_stream_ids=[3])
    assert same(a, c), "a different stream id must give a different FR history"


def test_arm_stream_independent_of_other_arms():
    """An arm's FR stream is fixed by its id, not by which other arms are in the job."""
    alone = run([("x", True)], fr_stream_ids=[4])
    with_others = run([("abf", False), ("y", True), ("x", True)], fr_stream_ids=[1, 4])
    assert not same(alone, with_others, 0, 2)


def test_pre_activation_identity():
    """(2) a delayed-FR arm equals ABF at every save before its first FR event (separate jobs, shared noise)."""
    k0 = 1600
    abf = run([("abf", False)])
    fr = run([("fr", True)], c=cfg(fr_start_steps=k0), fr_stream_ids=[2])
    pre = SAVE <= k0
    for k in ("M_rep", "C_rep", "z_hist"):
        assert np.array_equal(abf[k][0][pre], fr[k][0][pre]), k
    assert fr["repl_cumulative"][0][pre].max() == 0
    assert fr["repl_cumulative"][0, -1] > 0
    first = int(np.argmax(fr["repl_cumulative"][0] > 0))
    assert np.array_equal(abf["M_rep"][0][:first], fr["M_rep"][0][:first])


@pytest.mark.parametrize("k_stop", [800, 1600, 2400])
def test_common_prefix_and_no_events_after_stop(k_stop):
    """(3) continuous and stopped arms on the SAME FR stream are identical before the stop; (4) no FR event,
    no opportunity at or after the stop; (5) plain ABF continues (deposits keep growing by N per step)."""
    cont = run([("fr", True)], fr_stream_ids=[1])
    stop = run([("fr", True)], fr_stream_ids=[1], fr_stop_steps=k_stop)
    pre = SAVE < k_stop
    for k in ("M_bias", "C_bias", "M_rep", "C_rep", "z_hist", "repl_cumulative", "n_event_opportunities"):
        assert np.array_equal(cont[k][0][pre], stop[k][0][pre]), k
    post = SAVE >= k_stop
    r = stop["repl_cumulative"][0]
    assert np.all(r[post] == r[post][0]), "replacements after the stop"
    nev = stop["n_event_opportunities"][0]
    assert np.all(nev[post] == nev[post][0])
    assert cont["repl_cumulative"][0, -1] > r[-1], "the continuous arm must keep resampling"
    # the last opportunity of the stopped arm is next_step = largest multiple of fr_every below k_stop
    assert np.allclose(stop["C_bias"][0].sum(1), N * (SAVE + 1))      # deposits of states 0..s at the save at s
    assert stop["fr_stop_steps"] == k_stop


def test_stop_at_zero_is_plain_abf():
    """(5) FR stopped before it can act = the ABF arm bitwise (the post-stop propagation is plain ABF)."""
    abf = run([("abf", False)])
    fr0 = run([("fr", True)], fr_stream_ids=[1], fr_stop_steps=0)
    for k in ("M_bias", "C_bias", "M_prod", "C_prod", "z_hist", "q_final", "frac_regions"):
        assert np.array_equal(abf[k][0], fr0[k][0]), k
    assert fr0["repl_cumulative"][0, -1] == 0


def test_population_and_histogram_conservation_and_matched_physics():
    """(6, 7, 8) constant N, z_hist rows sum to N, identical cfg and force evaluations in every arm."""
    jobs = [run([("abf", False)]), run([("fr", True)], fr_stream_ids=[1]),
            run([("fr", True)], fr_stream_ids=[2], fr_stop_steps=1600)]
    for r in jobs:
        assert np.allclose(np.asarray(r["z_hist"])[0].sum(1), N)
        assert np.asarray(r["q_final"]).shape[1] == N
        assert r["cfg"] == jobs[0]["cfg"] and r["N"] == N and r["n_steps"] == NS


def test_clip_counters_monotone_and_inert(orig):
    a = run([("fr", True)])
    b = orig.run_ladder_point(3, N, NS, [("fr", True)], cfg=cfg(), save_at=SAVE, cap_min=0)
    c = np.asarray(a["bias_clip_cum"])[0]
    e = np.asarray(a["bias_eval_cum"])[0]
    assert np.all(np.diff(c) >= 0) and np.all(np.diff(e) > 0)
    assert np.isclose(c[-1] / e[-1], float(np.asarray(b["bias_clip_fraction"])[0]), rtol=0, atol=1e-3)


def test_accepted_cfg_unchanged():
    import importlib.util as iu
    spec = iu.spec_from_file_location("wca_numba_orig2", ORIG)
    if not os.path.exists(ORIG):
        pytest.skip("original engine not found")
    m = iu.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert wn.ACCEPTED_CFG == m.ACCEPTED_CFG and "fr_stop_steps" not in wn.ACCEPTED_CFG


def test_scorer_compatibility_with_exact_mc_reference():
    """(9) score_ladder_result accepts a stopped-arm result with the exact MC reference (main checkout data)."""
    main = os.environ.get("ABFFR_DATA_ROOT", "/home/zheyuanlai/ABF-Fisher-Rao")
    if not os.path.isdir(os.path.join(main, "results", "numerical_validation", "wca")):
        pytest.skip("exact MC chains not available")
    sys.path.insert(0, os.path.join(main, "scripts", "numerical_validation"))
    import rescore_wca_history as R
    ref, nmc = R.mc_reference()
    r = run([("fr", True)], fr_stream_ids=[1], fr_stop_steps=1600)
    o = wn.score_ladder_result(r, "fr", reference=ref)
    assert np.all(np.isfinite(o["l2_f_t"])) and np.all(np.isfinite(o["l2_fp_t"])) and nmc == 64
