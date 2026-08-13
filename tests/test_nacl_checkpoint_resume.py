"""A checkpoint is verified by resuming from it, not by existing.

The screen's loop couples three things that must agree at the checkpoint instant: the walker
state ``(q, v, f)``, the estimator accumulators, and the step counter.  Saving at the wrong
point in the loop desynchronises them by exactly one step -- the dynamics of that step never
happen and its samples are counted twice -- and the only symptom is a slightly wrong PMF.

This harness reproduces the loop's *bookkeeping* with trivial dynamics, so the off-by-one is
tested without an MD stack.  It fails on the placement the code originally had.
"""
from __future__ import annotations

import numpy as np
import pytest


def run_loop(n_steps, checkpoint_at=None, resume_from=None, save_after_accumulate=False,
             resume_plus_one=False):
    """Mimics the screen: accumulate from the current state, then advance it.

    ``save_after_accumulate`` and ``resume_plus_one`` reproduce the two defects the methane
    session found by verifying its own resume; both default off (the corrected behaviour).
    """
    if resume_from is None:
        q = np.zeros(3)                    # "positions"
        acc = np.zeros(3)                  # "estimator accumulators"
        trace = []
        start = 0
    else:
        q, acc, trace, saved_step = (resume_from["q"].copy(), resume_from["acc"].copy(),
                                     list(resume_from["trace"]), resume_from["step"])
        start = saved_step + 1 if resume_plus_one else saved_step

    ckpt = None
    for step in range(start, n_steps):
        if (not save_after_accumulate) and step == checkpoint_at:
            ckpt = dict(q=q.copy(), acc=acc.copy(), trace=list(trace), step=step)

        acc += q                           # this step's samples come from the CURRENT state
        trace.append(float(q.sum()))

        if save_after_accumulate and step == checkpoint_at:
            ckpt = dict(q=q.copy(), acc=acc.copy(), trace=list(trace), step=step)

        q = q + 1.0                        # "dynamics": advance the state

    return dict(q=q, acc=acc, trace=trace), ckpt


def test_resume_reproduces_an_uninterrupted_run_exactly():
    full, _ = run_loop(40)
    _, ckpt = run_loop(40, checkpoint_at=17)
    resumed, _ = run_loop(40, resume_from=ckpt)

    assert np.array_equal(resumed["q"], full["q"])
    assert np.array_equal(resumed["acc"], full["acc"]), "accumulators must match bit for bit"
    assert resumed["trace"] == full["trace"], "no trace frame gained or lost"


def test_saving_after_accumulation_skips_a_step_and_double_counts_it():
    """Defect 1: `(q, v, f)` are the step's *entry* state while the accumulators already hold
    its contribution, so resuming at step+1 loses one step of dynamics and re-counts its
    samples.  This is the placement the screen originally had."""
    full, _ = run_loop(40)
    _, bad = run_loop(40, checkpoint_at=17, save_after_accumulate=True)
    resumed, _ = run_loop(40, resume_from=bad, resume_plus_one=True)

    # the signature is NOT a missing frame -- the trace is the same length, with step 17's
    # frame duplicated -- so a length check would pass while the run is wrong
    assert len(resumed["trace"]) == len(full["trace"])
    assert resumed["trace"] != full["trace"]
    assert resumed["trace"][17] == resumed["trace"][18], "the entry state was re-sampled"
    assert not np.array_equal(resumed["acc"], full["acc"]), "samples double counted"
    assert float(full["q"][0] - resumed["q"][0]) == pytest.approx(1.0), \
        "exactly one step of dynamics never happened"


def test_adding_one_to_a_correctly_saved_step_drops_a_frame():
    """Defect 2: once the checkpoint stores the resume step directly, adding 1 on load skips a
    step -- the mirror of defect 1, and just as silent."""
    full, _ = run_loop(40)
    _, ckpt = run_loop(40, checkpoint_at=17)
    resumed, _ = run_loop(40, resume_from=ckpt, resume_plus_one=True)

    assert len(resumed["trace"]) == len(full["trace"]) - 1
    assert not np.array_equal(resumed["acc"], full["acc"])


def test_the_screen_saves_before_accumulating_and_resumes_at_the_saved_step():
    """Pins the two lines in the driver that this harness is a model of."""
    import os
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                            "scripts", "nacl_screen.py")).read()
    body = src.split("for step in range(start_step, max_steps + 1):", 1)[1]
    ckpt_pos = body.index("save_state(step)")
    accum_pos = body.index('s["fsum"] +=')
    assert ckpt_pos < accum_pos, "the checkpoint must precede this step's accumulation"
    assert 'start_step = int(z["step"])' in src
    assert 'start_step = int(z["step"]) + 1' not in src


def test_checkpoint_write_is_atomic():
    """A crash during a checkpoint write is a crash under load -- exactly the crash the
    checkpoint exists to survive -- so the naive write fails in the case it was written for."""
    import os
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                            "scripts", "nacl_screen.py")).read()
    assert "os.replace(tmp, state_path)" in src
    assert 'tmp = state_path + ".tmp"' in src
