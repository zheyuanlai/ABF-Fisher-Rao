"""Initial-population modes of the 2026-10 study scripts (post-protocol diagnostics)."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "deca_fr_2026"))

import common as C                                   # noqa: E402
from deca import system as dsys                      # noqa: E402


def _R(x):
    i, j = dsys.terminal_carbonyls(dsys.N_RES)
    return np.linalg.norm(x[..., i, :] - x[..., j, :], axis=-1)


@pytest.fixture(scope="module")
def engine():
    return C.engine_for("cpu", compiled=False)


def test_init_mode_rejects_unknown(engine):
    with pytest.raises(ValueError):
        C.build_initial_population(engine, 1, 2, 0, "cpu", equil_ps=0.01, verbose=False, init_mode="beta")


def test_init_modes_start_on_the_intended_side(engine):
    helix, _ = C.build_initial_population(engine, 1, 4, 0, "cpu", equil_ps=0.02, verbose=False)
    ppii, rep = C.build_initial_population(engine, 1, 4, 0, "cpu", equil_ps=0.02, verbose=False,
                                           init_mode="ppii")
    mixed, _ = C.build_initial_population(engine, 1, 4, 0, "cpu", equil_ps=0.02, verbose=False,
                                          init_mode="mixed")
    Rh, Rp, Rm = _R(helix)[0], _R(ppii)[0], _R(mixed)[0]
    assert rep["init_mode"] == "ppii"
    assert Rh.max() < 2.0 and Rp.min() > 2.5
    assert (Rp.min() - Rh.max()) > 1.0
    # mixed: first half helical, second half extended
    assert Rm[:2].max() < 2.0 and Rm[2:].min() > 2.5


def test_helix_mode_is_the_protocol_default(engine):
    a, ra = C.build_initial_population(engine, 1, 3, 7, "cpu", equil_ps=0.02, verbose=False)
    b, rb = C.build_initial_population(engine, 1, 3, 7, "cpu", equil_ps=0.02, verbose=False,
                                       init_mode="helix")
    assert np.array_equal(a, b) and ra["init_mode"] == "helix"


def test_restrained_equilibration_holds_the_extended_start(engine):
    """Unrestrained, a PPII start drifts towards the helix; k = 3200 kJ/mol/nm^2 holds it."""
    free, _ = C.build_initial_population(engine, 1, 2, 3, "cpu", equil_ps=0.3, verbose=False,
                                         init_mode="ppii")
    held, rep = C.build_initial_population(engine, 1, 2, 3, "cpu", equil_ps=0.3, verbose=False,
                                           init_mode="ppii", restrain_k=3200.0)
    Rf, Rh = _R(free)[0], _R(held)[0]
    assert rep["restrain_k"] == 3200.0 and rep["restrain_R0"] is None
    assert np.abs(Rh - 3.21).max() < 0.08
    assert len(rep["R_after_equil_quantiles"]) == 5
    # the restraint must change the trajectory (same seed, same start)
    assert not np.allclose(Rf, Rh)
