"""Gates for the frozen orthogonal descriptor ``n_gap`` (SPEC_methane_water.md §5.1).

``n_gap`` carries three separate jobs -- Gate A's CV-visibility test, ``tau_perp``, and the
conditional-fidelity endpoint -- so a silent change in its meaning would move three results at
once.  Its parameters are frozen and these tests pin the properties the spec relies on.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from methane.observables import (R_CYL_NM, _switch, gap_geometry,  # noqa: E402
                                 n_gap, pair_distance)

L = 2.490832


def _config(r_nm, water_offsets):
    """Two methanes at separation ``r`` along x, with waters at given offsets from the midpoint."""
    mid = np.array([L / 2, L / 2, L / 2])
    pos = [mid - np.array([r_nm / 2, 0, 0]), mid + np.array([r_nm / 2, 0, 0])]
    pos += [mid + np.asarray(o, dtype=float) for o in water_offsets]
    return np.asarray(pos), np.array([0, 1]), np.arange(2, 2 + len(water_offsets))


def test_switch_is_one_at_zero_and_half_at_x0():
    assert _switch(0.0, 0.2) == pytest.approx(1.0)
    assert _switch(0.2, 0.2) == pytest.approx(0.5)
    assert _switch(0.4, 0.2) < 0.02
    # monotone decreasing, and finite at x0 (the naive rational form is 0/0 there)
    xs = np.linspace(0, 1, 200)
    v = _switch(xs, 0.2)
    assert np.all(np.isfinite(v)) and np.all(np.diff(v) <= 1e-12)


def test_empty_gap_reads_zero_and_a_centred_water_reads_one():
    pos, mi, oi = _config(0.7, [[0.0, 0.0, 1.0]])          # far away
    assert n_gap(pos, mi, oi, L) < 0.01
    pos, mi, oi = _config(0.7, [[0.0, 0.0, 0.0]])          # exactly at the midpoint
    assert n_gap(pos, mi, oi, L) == pytest.approx(1.0, abs=1e-9)


def test_radial_cutoff_excludes_water_beside_the_axis():
    pos, mi, oi = _config(0.7, [[0.0, R_CYL_NM, 0.0]])
    assert n_gap(pos, mi, oi, L) == pytest.approx(0.5, abs=1e-9)
    pos, mi, oi = _config(0.7, [[0.0, 2 * R_CYL_NM, 0.0]])
    assert n_gap(pos, mi, oi, L) < 0.02


def test_axial_halfwidth_scales_with_separation():
    """The cylinder ends on the methane centres, so the descriptor means the same at every r."""
    for r in (0.4, 0.6, 0.9):
        pos, mi, oi = _config(r, [[r / 2, 0.0, 0.0]])       # sitting on a methane centre
        assert n_gap(pos, mi, oi, L) == pytest.approx(0.5, abs=1e-9)


def test_geometry_and_cv_use_minimum_image():
    """A water placed across the periodic boundary must still be counted."""
    pos, mi, oi = _config(0.7, [[0.0, 0.0, 0.0]])
    shifted = pos.copy()
    shifted[oi[0]] += np.array([0.0, 0.0, L])              # one box along z
    assert n_gap(shifted, mi, oi, L) == pytest.approx(n_gap(pos, mi, oi, L), abs=1e-9)

    far = pos.copy()
    far[1] += np.array([L, 0.0, 0.0])
    assert pair_distance(far, mi, L) == pytest.approx(pair_distance(pos, mi, L), abs=1e-9)
    _, _, r = gap_geometry(far, mi, oi, L)
    assert r == pytest.approx(0.7, abs=1e-9)


def test_n_gap_is_smooth_in_the_water_position():
    """No jump discontinuities -- tau_perp is a correlation time and must not be dominated by
    boundary crossings, which is why a hard count is not used."""
    xs = np.linspace(-0.5, 0.5, 400)
    vals = []
    for x in xs:
        pos, mi, oi = _config(0.7, [[x, 0.0, 0.0]])
        vals.append(n_gap(pos, mi, oi, L))
    assert np.abs(np.diff(vals)).max() < 0.05
