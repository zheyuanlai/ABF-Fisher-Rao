"""Movie snapshot record of the alanine engines (docs/ALANINE_MOVIES.md): bit-inert and consistent."""
import os
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src")); sys.path.insert(0, HERE)
torch.set_default_dtype(torch.float64)
pytest.importorskip("openmm")

from alanine.core1d_ala import BackboneCV1D, run_sampler_ala1d          # noqa: E402
from alanine.core2d_ala import run_sampler_ala                          # noqa: E402
from test_alanine_1d import _cfg as _cfg1d                              # noqa: E402
from test_alanine_sampler import REF, _cfg, _init_dispersed, rig        # noqa: E402,F401

needs_ref = pytest.mark.skipif(not os.path.exists(REF), reason="reference artifact not present")


def _check(out, n_steps, every, N, keys):
    n_snap = n_steps // every + 1
    assert out["snap_steps"].shape == (n_snap,) and out["snap_steps"][-1] == n_steps
    for k in keys:
        assert out[k].shape[:2] == (n_snap, N), (k, out[k].shape)
    assert out["snap_wid"].dtype == np.int32 and out["snap_basin"].dtype == np.int8
    die, birth = out["snap_ev_die"], out["snap_ev_birth"]
    assert die.shape == birth.shape and die.shape[1] == 3 and np.isfinite(die).all() and np.isfinite(birth).all()
    assert len(die) == int(out["total_events"][0])                      # every event of seed 0 recorded
    assert (die[:, 0] >= 1).all() and (die[:, 0] <= n_snap).all()      # slot = the snapshot it precedes
    # ids: births mint fresh ids, so the number of distinct ids grows by exactly the event count
    assert out["snap_wid"][-1].max() == N - 1 + len(die) and len(np.unique(out["snap_wid"][0])) == N


@needs_ref
def test_2d_snapshots_bit_inert_and_consistent(rig):
    tff, cv, X0, bm, _ = rig
    lab = bm.label_tensor(); init = _init_dispersed(X0, 2, 16)
    base = dict(fr_start_steps=0, fr_rate=200.0, n_steps=40, save_every=20, abf_min_count=2.0, abf_warmup_steps=0,
                abf_estimator="histogram")
    for m in ("abf", "fr_uniform"):
        a = run_sampler_ala(m, tff, cv, _cfg(**base), [0, 1], init, lab, "cpu", verbose=False)
        b = run_sampler_ala(m, tff, cv, _cfg(store_snapshots=10, **base), [0, 1], init, lab, "cpu", verbose=False)
        assert np.array_equal(a["final_pmf"], b["final_pmf"]) and np.array_equal(a["pmf"], b["pmf"])
        assert np.array_equal(a["total_events"], b["total_events"]) and "snap_phi" not in a
        _check(b, 40, 10, 16, ("snap_phi", "snap_psi", "snap_wid", "snap_basin"))
        assert b["snap_pmf"].shape == (5, 97, 97) and np.array_equal(b["snap_pmf"][-1], b["final_pmf"][0].astype(np.float32))
        if m == "fr_uniform":
            assert int(b["total_events"][0]) > 0
    assert _cfg(**base).config_hash() == _cfg(store_snapshots=10, **base).config_hash()


@needs_ref
def test_1d_snapshots_bit_inert_and_consistent(rig):
    tff, _, X0, bm, _ = rig
    lab = bm.label_tensor(); init = _init_dispersed(X0, 2, 16); cv = BackboneCV1D("psi")
    base = dict(cv="psi", fr_start_steps=0, fr_rate=200.0, n_steps=40, save_every=20)
    a = run_sampler_ala1d("fr_uniform", tff, cv, _cfg1d(**base), [0, 1], init, lab, "cpu", verbose=False)
    b = run_sampler_ala1d("fr_uniform", tff, cv, _cfg1d(store_snapshots=10, **base), [0, 1], init, lab, "cpu", verbose=False)
    assert np.array_equal(a["final_pmf"], b["final_pmf"]) and np.array_equal(a["total_events"], b["total_events"])
    _check(b, 40, 10, 16, ("snap_cv", "snap_hidden", "snap_wid", "snap_basin"))
    assert b["snap_pmf"].shape == (5, 97) and int(b["total_events"][0]) > 0
    # the recorded CV is psi and the hidden one is phi: compare with the 2-D engine's angles at step 0
    from alanine.cv2d import BackboneCV2D
    from alanine.system import PHI_ATOMS, PSI_ATOMS
    p2 = BackboneCV2D(PHI_ATOMS, PSI_ATOMS, n_atoms=22).values(init[0])
    np.testing.assert_allclose(b["snap_cv"][0], p2[1].numpy(), atol=1e-5)
    np.testing.assert_allclose(b["snap_hidden"][0], p2[0].numpy(), atol=1e-5)
