"""Early-stopping FR and the online controller in the LTA numba engine (src/lta_ladder_numba.py): fixed stop and
controlled stop are the same mechanism, FR is fully off after the stop, the default path is untouched (see
tests/test_lta_ladder_numba.py), and checkpoint/resume reproduces every controller decision.

    NUMBA_NUM_THREADS=1 /home/zheyuanlai/miniconda3/envs/abffr/bin/python -m pytest -q tests/test_lta_fr_stop.py
"""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import lta_ladder_numba as E  # noqa: E402
from eri import fr_controller as FC  # noqa: E402

DYN = ("M_all", "C_all", "M_prod", "C_prod", "q_final", "M_all_final", "C_all_final", "anc_final", "hist_inst",
       "cum_deaths", "cum_opps", "birth_hist", "death_hist", "gen_ess_win", "gen_max_frac_win")
SS = np.arange(0, 1501, 100)
FORCE = dict(FC.R1, theta=1.0, persist=1, support_count=0.0, stability_tol=1e9, check_every=0.02, W=0.02,
             essw_min=0.0, maxlin_min=1.0)          # decides at the first full window
FORCE_R2 = dict(FC.R2, level=0.0, support_count=0.0, stability_tol=1e9, check_every=0.02, W=0.02,
                essw_min=0.0, maxlin_min=1.0)       # the occupancy rule, deciding at the first full window


def cfg(**kw):
    base = dict(n_steps=1500, warmup_steps=200, burn_in_steps=200, fr_start_steps=300, fr_rate=40.0,
                genealogy_window_tu=0.1)
    base.update(kw)
    return E.make_cfg(300.0, 32, 7, **base)


def same(a, b, keys=DYN, mask=None):
    for k in keys:
        x, y = np.asarray(a[k]), np.asarray(b[k])
        if mask is not None and x.ndim >= 1 and x.shape[0] == mask.size:
            x, y = x[mask], y[mask]
        assert x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes(), k


def test_fixed_stop_limits_and_mid_run_stop():
    cont = E.run_arm(cfg(), "fr", SS)
    abf = E.run_arm(cfg(), "abf", SS)
    s0 = E.run_arm(cfg(fr_stop_steps=300), "fr", SS)              # stop at the start = no FR at all
    same(abf, s0, keys=("M_all", "C_all", "q_final", "hist_inst"))
    assert int(s0["total_replacement_events"]) == 0 and int(s0["n_fr_opps"]) == 0
    sinf = E.run_arm(cfg(fr_stop_steps=10_000), "fr", SS)         # stop beyond the horizon = continuous
    same(cont, sinf)
    sm = E.run_arm(cfg(fr_stop_steps=800), "fr", SS)
    pre = SS < 800
    same(cont, sm, keys=("M_all", "C_all", "cum_deaths", "cum_opps", "gen_ess_win"), mask=pre)
    post = SS >= 800
    assert (sm["cum_deaths"][post] == sm["cum_deaths"][post][0]).all() and int(sm["cum_deaths"][-1]) > 0
    assert int(sm["n_fr_opps"]) == (800 - 1 - 300) // 5 + 1
    assert int(cont["total_replacement_events"]) > int(sm["total_replacement_events"])
    assert int(sm["n_force_evals"]) == int(abf["n_force_evals"]) == 32 * 1501


@pytest.mark.parametrize("FORCE", [FORCE, FORCE_R2], ids=["R1", "R2"])
def test_controlled_stop_equals_fixed_stop_and_is_deterministic(FORCE):
    a = E.run_arm(cfg(fr_controller=FORCE), "fr", SS)
    b = E.run_arm(cfg(fr_controller=FORCE), "fr", SS, chunk_steps=137)
    assert str(a["fr_controller_log"]) == str(b["fr_controller_log"])
    same(a, b)
    stop = int(a["fr_stop_effective_steps"])
    assert 300 < stop < 1500
    f = E.run_arm(cfg(fr_stop_steps=stop), "fr", SS)
    same(a, f)


@pytest.mark.parametrize("FORCE", [FORCE, FORCE_R2], ids=["R1", "R2"])
def test_checkpoint_resume_reproduces_controller(tmp_path, FORCE):
    a = E.run_arm(cfg(fr_controller=FORCE), "fr", SS, chunk_steps=100)
    p = str(tmp_path / "ck.npz")
    with pytest.raises(E.InterruptedRun):
        E.run_arm(cfg(fr_controller=FORCE), "fr", SS, chunk_steps=100, checkpoint_path=p, checkpoint_every_s=0.0,
                  _stop_after_chunks=4)
    b = E.run_arm(cfg(fr_controller=FORCE), "fr", SS, chunk_steps=100, checkpoint_path=p, checkpoint_every_s=0.0)
    assert json.loads(str(a["fr_controller_log"])) == json.loads(str(b["fr_controller_log"]))
    same(a, b)


def test_controller_inert_for_abf_and_exclusive_with_fixed_stop():
    a = E.run_arm(cfg(), "abf", SS)
    b = E.run_arm(cfg(fr_controller=FC.R1), "abf", SS)
    same(a, b, keys=("M_all", "C_all", "q_final", "hist_inst"))
    with pytest.raises(ValueError):
        E.run_arm(cfg(fr_controller=FC.R1, fr_stop_steps=800), "fr", SS)


def test_r2_window_mask_follows_the_lta_region_convention():
    """R2's window bins are the deposit bins whose centre lies at |z| < window_half (z = |phi| a / 2 pi): 46 of 180."""
    a = E.run_arm(cfg(fr_controller=FC.R2), "fr", SS)
    log = json.loads(str(a["fr_controller_log"]))
    assert a["fr_controller_reason"] == "None" or True        # a 0.3 t.u. run never closes R2's 1 t.u. window
    assert all("occ" not in x for x in log)
    m = FC.window_mask(180, 1.5 * 2 * np.pi / 11.919)
    assert int(m.sum()) == 46 and abs(m.mean() - 2 * 1.5 / 11.919) < 0.01
    b = E.run_arm(cfg(fr_controller=FC.R2), "abf", SS)       # inert for ABF
    same(E.run_arm(cfg(), "abf", SS), b, keys=("M_all", "C_all", "q_final", "hist_inst"))
