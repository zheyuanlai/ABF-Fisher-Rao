"""No-reference-leakage guards for the alkane methods (CPU, fast).

Run: CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_alkanes_noleak.py -q
"""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
torch.set_default_dtype(torch.float64)

from alkanes import core, potentials as pot, opes as opesmod  # noqa: E402
from alkanes.cv import DihedralCV  # noqa: E402


def _sim(**kw):
    base = dict(dt=5e-4, n_steps=60, n_replicas=8, save_every=30, rng_seed=1,
                abf_warmup_steps=10, fr_start_steps=20, estimator_burn_in_steps=10, n_grid=60)
    base.update(kw)
    return core.AlkaneSimConfig(**base)


P = pot.AlkaneParams(n_atoms=4, beta=1.0, decouple=True, force_clip=200.0)
CV = DihedralCV((0, 1, 2, 3))
REF = np.zeros(60)


@pytest.mark.parametrize("method", ["abf", "fr_estimated", "fr_uniform"])
def test_non_oracle_methods_reject_reference(method):
    with pytest.raises(AssertionError):
        core.run_sampler(method, P, _sim(), [0], CV, "cpu", oracle_free_energy=REF, verbose=False)


def test_oracle_requires_reference():
    with pytest.raises(ValueError):
        core.run_sampler("fr_oracle", P, _sim(), [0], CV, "cpu", oracle_free_energy=None, verbose=False)


def test_oracle_runs_with_reference():
    out = core.run_sampler("fr_oracle", P, _sim(), [0], CV, "cpu", oracle_free_energy=REF, verbose=False)
    assert np.isfinite(out["pmf"][-1]).all()


def test_assert_helper_direct():
    core.assert_no_reference_leakage("abf", None)          # ok
    with pytest.raises(AssertionError):
        core.assert_no_reference_leakage("fr_estimated", REF)
    with pytest.raises(ValueError):
        core.assert_no_reference_leakage("fr_oracle", None)


def test_opes_never_receives_reference():
    # run_opes signature has no oracle argument at all: structural no-leakage.
    import inspect
    assert "oracle" not in inspect.signature(opesmod.run_opes).parameters


if __name__ == "__main__":
    import subprocess
    raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-q"]))
