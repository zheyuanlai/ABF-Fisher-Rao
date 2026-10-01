"""1-D (incomplete-CV) alanine engine (docs/ALANINE_1D_CV.md).

  * BackboneCV1D values == the 2-D engine's IUPAC angles; geometry == alkanes DihedralCV
  * 1-D adaptive box == brute force; PMF of a piecewise-constant force is its exact integral
  * marginal reference: sums the Boltzmann weight over the hidden axis, ignores +inf cells
  * abf == fr_uniform at fr_rate 0; FR fires with a dispersed init; both CVs run finite
  * histogram bias is the walker's OWN bin value; kernel path is the smoothed ratio

Run: CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_alanine_1d.py -q
"""
import os
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
torch.set_default_dtype(torch.float64)
pytest.importorskip("openmm")

from alanine.core1d_ala import (Ala1DSimConfig, BackboneCV1D, adaptive_box_mean_force_1d,  # noqa: E402
                                bias_value_at, box_sum1, marginal_reference_1d, run_sampler_ala1d)
from alanine.cv2d import BackboneCV2D                                       # noqa: E402
from alanine.system import PHI_ATOMS, PSI_ATOMS                             # noqa: E402
from alkanes import periodic as per                                         # noqa: E402
from alkanes.cv import DihedralCV                                           # noqa: E402
from test_alanine_sampler import REF, _init, _init_dispersed, rig           # noqa: E402,F401

needs_ref = pytest.mark.skipif(not os.path.exists(REF), reason="reference artifact not present")


def _cfg(**kw):
    base = dict(cv="phi", n_steps=60, n_replicas=16, save_every=30, abf_warmup_steps=20,
                fr_start_steps=20, fr_every=10, fr_rate=5.0, rng_seed=1234, max_event_fraction=0.5,
                lineage_reset_steps=0, abf_min_count=2.0)
    base.update(kw)
    return Ala1DSimConfig(**base)


def test_cv1d_values_and_geometry(rig):
    tff, cv2, X0, _, _ = rig
    q = torch.as_tensor(np.repeat(X0[None], 5, 0)) + 0.01 * torch.randn(5, 22, 3, generator=torch.Generator().manual_seed(1))
    p2, s2 = cv2.values(q)
    for name, ref, atoms in (("phi", p2, PHI_ATOMS), ("psi", s2, PSI_ATOMS)):
        cv = BackboneCV1D(name)
        assert torch.allclose(cv.value(q), ref, atol=1e-12)
        f = tff.forces(q)
        a = cv.local_mean_force(q, f, 0.4)
        b = DihedralCV(atoms).local_mean_force(q, f, 0.4)
        assert torch.equal(a[0], b[0]) and torch.equal(a[2], b[2])     # geometry untouched by the shift
    with pytest.raises(ValueError):
        BackboneCV1D("omega")


def test_box_and_adaptive_rule_1d():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, 11))
    for k in range(4):
        brute = sum(np.roll(x, s, axis=1) for s in range(-k, k + 1))
        np.testing.assert_allclose(box_sum1(torch.as_tensor(x), k).numpy(), brute, atol=1e-12)
    cs = torch.zeros(1, 9); fs = torch.zeros(1, 9)
    cs[0, 4] = 10; fs[0, 4] = 30
    cs[0, 0] = 3; fs[0, 0] = 9; cs[0, 8] = 7; fs[0, 8] = 7          # periodic neighbours
    g, den, lev = adaptive_box_mean_force_1d(fs, cs, 2, 10.0)
    assert int(lev[0, 4]) == 0 and abs(float(g[0, 4]) - 3.0) < 1e-12
    assert int(lev[0, 0]) == 1 and abs(float(g[0, 0]) - 1.6) < 1e-12 and float(den[0, 0]) == 10.0
    # bin 2: 3-box (1,2,3) empty, 5-box (0..4) = 3 + 10 = 13 -> level 2, force (9 + 30) / 13
    assert int(lev[0, 2]) == 2 and abs(float(g[0, 2]) - 39.0 / 13.0) < 1e-12
    # with only one level allowed, bin 2 has no trusted box: zero force, level -1
    g1, den1, lev1 = adaptive_box_mean_force_1d(fs, cs, 1, 10.0)
    assert int(lev1[0, 2]) == -1 and float(g1[0, 2]) == 0.0 and float(den1[0, 2]) == 0.0
    # fixed mode uses the level-2 box everywhere
    _, _, levf = adaptive_box_mean_force_1d(fs, cs, 2, 10.0, fixed=True)
    assert set(levf.unique().tolist()) <= {-1, 2} and int(levf[0, 4]) == 2


def test_pmf_is_exact_integral_and_own_bin_bias():
    n = 97
    grid, dphi = per.periodic_grid(n)
    prof = torch.sin(grid)[None] * 3.0                                  # F' = 3 sin -> F = -3 cos
    F = per.free_energy_from_mean_force(prof, grid, dphi)[0]
    # trapezoid between centres is exact for a P0 force; for sin it is O(dphi^2): check shape
    ref = -3.0 * torch.cos(grid); ref = ref - ref.mean()
    assert float((F - ref).abs().max()) < 3e-3
    phi = torch.tensor([[grid[10] + 0.3 * dphi, grid[50] - 0.49 * dphi]])
    v = bias_value_at(prof, grid, dphi, phi, "histogram")
    assert torch.equal(v[0, 0], prof[0, 10]) and torch.equal(v[0, 1], prof[0, 50])


def test_marginal_reference_ignores_unvisited_cells():
    kT = 2.5
    F = np.full((97, 97), np.inf); F[10:20, :] = 0.0; F[30, :] = np.inf
    F[15, 40:50] = -kT * np.log(2.0)                                     # doubles the weight there
    F1 = marginal_reference_1d(F, kT, axis_keep=1)
    assert np.isfinite(F1).all() and abs(F1[45] - F1[5] + kT * np.log(11.0 / 10.0)) < 1e-9
    F1p = marginal_reference_1d(F, kT, axis_keep=0)
    assert np.isinf(F1p[30]) and np.isinf(F1p[0]) and np.isfinite(F1p[15])


@needs_ref
@pytest.mark.parametrize("name", ["phi", "psi"])
def test_arms_equal_at_zero_rate_and_fr_fires(rig, name):
    tff, _, X0, bm, _ = rig
    lab = bm.label_tensor()
    cv = BackboneCV1D(name)
    init = _init_dispersed(X0, 2, 16)
    a = run_sampler_ala1d("abf", tff, cv, _cfg(cv=name, fr_rate=0.0), [0, 1], init, lab, "cpu", verbose=False)
    b = run_sampler_ala1d("fr_uniform", tff, cv, _cfg(cv=name, fr_rate=0.0), [0, 1], init, lab, "cpu", verbose=False)
    assert np.array_equal(a["final_pmf"], b["final_pmf"]) and int(b["total_events"].sum()) == 0
    assert np.isfinite(a["pmf"]).all() and a["pmf"].shape == (3, 2, 97) and a["joint_hist"].shape == (3, 2, 97, 97)
    assert np.array_equal(a["joint_hist"][-1].sum(axis=(1, 2)), np.full(2, 16.0))
    assert np.asarray(a["trust_frac"])[-1] > 0 and np.isfinite(a["hist_level_mean"][-1]).all()
    # a 1-D marginal of 16 dispersed walkers is already near-uniform (KL ~0.3-0.5): rate 5 expects
    # ~1 event in 60 steps, so the "FR fires" check uses a dose that fires with certainty
    c = run_sampler_ala1d("fr_uniform", tff, cv, _cfg(cv=name, fr_start_steps=0, fr_rate=200.0), [0, 1], init, lab, "cpu", verbose=False)
    assert int(c["total_events"].sum()) > 0
    k = run_sampler_ala1d("abf", tff, cv, _cfg(cv=name, abf_estimator="kernel"), [0, 1], init, lab, "cpu", verbose=False)
    assert np.isfinite(k["final_pmf"]).all() and np.isnan(k["hist_level_mean"]).all()
    with pytest.raises(ValueError):
        run_sampler_ala1d("abf", tff, BackboneCV1D("psi" if name == "phi" else "phi"), _cfg(cv=name), [0, 1], init, lab, "cpu", verbose=False)
