"""Adaptive-box histogram ABF estimator for alanine (docs/ALANINE_HISTOGRAM_ABF.md).

  * periodic box sums == brute force
  * the adaptive rule picks the SMALLEST level whose box count reaches min_count, returns M/C
    there, zero + level -1 where none; the fixed mode uses one level everywhere
  * the kernel path is BIT-IDENTICAL to the fixture generated at ebc123b before this code
    existed (abf and fr_uniform runs, tests/fixtures/ala_pre_histogram_fixture.npz)
  * legacy config_hash values unchanged; the new fields change it only when set
  * histogram smoke: finite, trusted cells appear, abf == fr_uniform at fr_rate 0
  * the FR block is estimator-independent: with no trusted cell (min_count huge) the kernel and
    histogram arms are the same run, events included

Run: CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_alanine_histogram.py -q
"""
import json
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

from alanine.core2d_ala import (AlaSimConfig, adaptive_box_mean_force,   # noqa: E402
                                box_sum2, run_sampler_ala)
from test_alanine_sampler import REF, _cfg, _init, _init_dispersed, rig  # noqa: E402,F401

FIX = os.path.join(HERE, "fixtures", "ala_pre_histogram_fixture.npz")
HASHES = os.path.join(HERE, "fixtures", "ala_pre_histogram_hashes.json")
needs_ref = pytest.mark.skipif(not os.path.exists(REF), reason="reference artifact not present")


# ------------------------------------------------------------------ estimator arithmetic
def _brute_box(x, k):
    R, n1, n2 = x.shape
    out = np.zeros_like(x)
    for a in range(-k, k + 1):
        for b in range(-k, k + 1):
            out += np.roll(np.roll(x, a, axis=1), b, axis=2)
    return out


def test_box_sum_matches_brute_force():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 7, 9))
    for k in range(4):
        got = box_sum2(torch.as_tensor(x), k).numpy()
        np.testing.assert_allclose(got, _brute_box(x, k), rtol=0, atol=1e-12)


def test_adaptive_rule_picks_smallest_trusted_level():
    n = 9
    csum = torch.zeros(1, n, n)
    f1s = torch.zeros(1, n, n)
    csum[0, 4, 4] = 10.0; f1s[0, 4, 4] = 30.0          # own cell trusted at min_count 10
    csum[0, 0, 0] = 3.0;  f1s[0, 0, 0] = 9.0           # needs the 3x3 box: neighbours below
    csum[0, 0, 1] = 4.0;  f1s[0, 0, 1] = 4.0
    csum[0, 8, 8] = 3.0;  f1s[0, 8, 8] = 3.0           # periodic neighbour of (0,0)
    g1, g2, den, level = adaptive_box_mean_force(f1s, torch.zeros_like(f1s), csum, 2, 10.0)
    assert int(level[0, 4, 4]) == 0 and abs(float(g1[0, 4, 4]) - 3.0) < 1e-12
    # (0,0): own count 3 < 10; 3x3 box = 3 + 4 + 3 = 10 -> level 1, force (9+4+3)/10
    assert int(level[0, 0, 0]) == 1 and abs(float(g1[0, 0, 0]) - 1.6) < 1e-12
    assert float(den[0, 0, 0]) == 10.0
    # a far cell: 5x5 box count 0 -> untrusted, zero force, level -1, den = coarsest count
    assert int(level[0, 4, 0]) == -1 and float(g1[0, 4, 0]) == 0.0 and float(den[0, 4, 0]) < 10.0
    assert torch.equal(g2, torch.zeros_like(g2))
    # fixed mode: level 2 everywhere it is trusted, never 0 or 1
    _, _, _, lev_f = adaptive_box_mean_force(f1s, torch.zeros_like(f1s), csum, 2, 10.0, fixed=True)
    assert set(lev_f.unique().tolist()) <= {-1, 2}
    assert int(lev_f[0, 4, 4]) == 2


def test_adaptive_equals_raw_ratio_when_threshold_is_zero():
    rng = np.random.default_rng(1)
    csum = torch.as_tensor(rng.integers(0, 5, size=(2, 6, 6)).astype(float))
    f1s = torch.as_tensor(rng.standard_normal((2, 6, 6))) * csum
    g1, _, den, level = adaptive_box_mean_force(f1s, torch.zeros_like(f1s), csum, 3, 0.0)
    assert int(level.max()) == 0 and int(level.min()) == 0
    expect = torch.where(csum > 0, f1s / csum.clamp_min(1e-300), torch.zeros_like(csum))
    np.testing.assert_allclose(g1.numpy(), expect.numpy(), atol=1e-12)


# ------------------------------------------------------------------ legacy path untouched
def test_legacy_config_hashes_unchanged():
    h = json.load(open(HASHES))
    assert AlaSimConfig().config_hash() == h["default"]
    assert _cfg().config_hash() == h["test"]
    assert _cfg(fr_start_steps=0, fr_rate=5.0).config_hash() == h["test_u"]
    assert _cfg(abf_estimator="histogram").config_hash() != h["test"]
    assert _cfg(abf_hist_levels=2).config_hash() != h["test"]
    assert _cfg(abf_hist_fixed=True).config_hash() != h["test"]


@needs_ref
def test_kernel_path_bit_identical_to_pre_histogram_fixture(rig):
    tff, cv, X0, bm, _ = rig
    lab = bm.label_tensor()
    fx = np.load(FIX)
    a = run_sampler_ala("abf", tff, cv, _cfg(), [0, 1], _init(X0, 2, 16), lab, "cpu", verbose=False)
    assert np.array_equal(a["final_pmf"], fx["abf_final_pmf"])
    assert np.array_equal(a["pmf"], fx["abf_pmf"])
    assert np.array_equal(a["basin_frac"], fx["abf_basin_frac"])
    assert np.array_equal(a["first_hit"], fx["abf_first_hit"])
    assert np.array_equal(np.asarray(a["trust_frac"]), fx["abf_trust_frac"])
    b = run_sampler_ala("fr_uniform", tff, cv, _cfg(fr_start_steps=0, fr_rate=5.0), [0, 1],
                        _init_dispersed(X0, 2, 16), lab, "cpu", verbose=False)
    assert np.array_equal(b["final_pmf"], fx["fru_final_pmf"])
    assert np.array_equal(b["total_events"], fx["fru_total_events"])
    assert np.array_equal(b["basin_frac"], fx["fru_basin_frac"])
    assert np.array_equal(b["ess_age"], fx["fru_ess_age"])
    assert np.array_equal(b["kl_uniform"], fx["fru_kl_uniform"])
    assert np.isnan(a["hist_level_mean"]).all()          # kernel arm: diagnostics are NaN


# ------------------------------------------------------------------ histogram arm behaviour
@needs_ref
def test_histogram_smoke_and_arms_equal_at_zero_rate(rig):
    tff, cv, X0, bm, _ = rig
    lab = bm.label_tensor()
    init = _init_dispersed(X0, 2, 16)
    sim = _cfg(abf_estimator="histogram", abf_min_count=2.0, abf_warmup_steps=0,
               fr_start_steps=0, fr_rate=0.0, n_steps=60, save_every=20)
    a = run_sampler_ala("abf", tff, cv, sim, [0, 1], init, lab, "cpu", verbose=False)
    b = run_sampler_ala("fr_uniform", tff, cv, sim, [0, 1], init, lab, "cpu", verbose=False)
    assert np.isfinite(a["final_pmf"]).all() and np.isfinite(a["pmf"]).all()
    assert np.asarray(a["trust_frac"])[-1] > 0.0
    assert np.isfinite(a["hist_level_mean"][-1]).all()
    assert (a["hist_untrusted_frac"][-1] < 1.0).all()
    assert np.array_equal(a["final_csum"].sum(axis=(1, 2)), np.full(2, 16.0 * 61))
    assert np.array_equal(a["final_pmf"], b["final_pmf"]) and int(b["total_events"].sum()) == 0
    # the projected bias is not identically zero once cells are trusted
    assert np.abs(a["final_pmf"]).max() > 0.0


@needs_ref
def test_histogram_differs_from_kernel_only_through_the_bias(rig):
    """With NO trusted cell (min_count huge) both estimators apply zero bias and the two arms
    are the same run, FR events included -> the FR block reads nothing estimator-specific."""
    tff, cv, X0, bm, _ = rig
    lab = bm.label_tensor()
    init = _init_dispersed(X0, 2, 16)
    k = _cfg(fr_start_steps=0, fr_rate=5.0, abf_min_count=1e9, n_steps=40, save_every=20)
    h = _cfg(fr_start_steps=0, fr_rate=5.0, abf_min_count=1e9, n_steps=40, save_every=20,
             abf_estimator="histogram", abf_warmup_steps=0)
    a = run_sampler_ala("fr_uniform", tff, cv, k, [0, 1], init, lab, "cpu", verbose=False)
    b = run_sampler_ala("fr_uniform", tff, cv, h, [0, 1], init, lab, "cpu", verbose=False)
    assert np.array_equal(a["total_events"], b["total_events"])
    assert np.array_equal(a["basin_frac"], b["basin_frac"])
    assert np.array_equal(a["final_pmf"], b["final_pmf"])   # both zero fields
    # and WITH trusted cells the histogram arm really diverges from the kernel arm
    k2 = _cfg(n_steps=60, abf_min_count=2.0, abf_warmup_steps=0)
    h2 = _cfg(n_steps=60, abf_min_count=2.0, abf_warmup_steps=0, abf_estimator="histogram")
    a2 = run_sampler_ala("abf", tff, cv, k2, [0, 1], init, lab, "cpu", verbose=False)
    b2 = run_sampler_ala("abf", tff, cv, h2, [0, 1], init, lab, "cpu", verbose=False)
    assert not np.array_equal(a2["final_pmf"], b2["final_pmf"])


# ------------------------------------------------------------------ histogram count rule frozen
HFIX = os.path.join(HERE, "fixtures", "ala_hist_count_fixture.npz")


@needs_ref
def test_histogram_count_rule_bit_identical_to_running_campaign(rig):
    """The count rule as launched in Stage A (fixture generated from that code): abf, fr_uniform
    and the fixed-box variant, plus their config hashes."""
    tff, cv, X0, bm, _ = rig
    lab = bm.label_tensor()
    fx = np.load(HFIX)
    init = _init_dispersed(X0, 2, 16)
    sa = _cfg(abf_estimator="histogram", abf_min_count=2.0, abf_warmup_steps=0, n_steps=60, save_every=20)
    a = run_sampler_ala("abf", tff, cv, sa, [0, 1], init, lab, "cpu", verbose=False)
    assert np.array_equal(a["final_pmf"], fx["abf_final_pmf"]) and np.array_equal(a["pmf"], fx["abf_pmf"])
    assert np.array_equal(a["hist_level_mean"], fx["abf_level"]) and np.array_equal(a["hist_untrusted_frac"], fx["abf_untrusted"])
    assert sa.config_hash() == str(fx["hash_a"])
    su = _cfg(abf_estimator="histogram", abf_min_count=2.0, abf_warmup_steps=0, n_steps=60, save_every=20,
              fr_start_steps=0, fr_rate=5.0)
    b = run_sampler_ala("fr_uniform", tff, cv, su, [0, 1], init, lab, "cpu", verbose=False)
    assert np.array_equal(b["final_pmf"], fx["fru_final_pmf"]) and np.array_equal(b["total_events"], fx["fru_total_events"])
    assert su.config_hash() == str(fx["hash_u"])
    sf = _cfg(abf_estimator="histogram", abf_hist_fixed=True, abf_hist_levels=2, abf_min_count=2.0,
              abf_warmup_steps=0, n_steps=60, save_every=20)
    c = run_sampler_ala("abf", tff, cv, sf, [0, 1], init, lab, "cpu", verbose=False)
    assert np.array_equal(c["final_pmf"], fx["fix_final_pmf"]) and sf.config_hash() == str(fx["hash_f"])


@needs_ref
def test_store_accumulators_is_inert_and_complete(rig):
    tff, cv, X0, bm, _ = rig
    lab = bm.label_tensor()
    init = _init_dispersed(X0, 2, 16)
    base = dict(abf_estimator="histogram", abf_min_count=2.0, abf_warmup_steps=0, n_steps=60, save_every=20)
    a = run_sampler_ala("abf", tff, cv, _cfg(**base), [0, 1], init, lab, "cpu", verbose=False)
    b = run_sampler_ala("abf", tff, cv, _cfg(store_accumulators=True, **base), [0, 1], init, lab, "cpu", verbose=False)
    assert np.array_equal(a["final_pmf"], b["final_pmf"]) and "acc_csum" not in a
    assert _cfg(**base).config_hash() == _cfg(store_accumulators=True, **base).config_hash()  # dropped at default only
    assert b["acc_csum"].shape == (4, 2, 97, 97) and b["acc_csum"].dtype == np.float32
    np.testing.assert_allclose(b["acc_csum"][-1], b["final_csum"], rtol=1e-6)
    np.testing.assert_allclose(b["acc_f1s"][-1], b["final_f1s"], rtol=1e-5, atol=1e-3)
    assert np.all(np.diff(b["acc_csum"].sum(axis=(2, 3)), axis=0) > 0)


# ------------------------------------------------------------------ visited-support target
def test_support_target_is_uniform_on_visited_cells_and_reduces_to_uniform():
    from alanine.core2d_ala import _support_target, _uniform_target
    from alkanes import density2d as d2
    n = 9
    g1, g2, dz1, dz2 = d2.torus_grid(n, n)
    cs = torch.zeros(2, n, n); cs[0, 2:5, 3:6] = 3.0; cs[1] = 1.0            # seed 0: 9 cells; seed 1: all
    q = _support_target(cs, 1.0, dz1, dz2)
    assert torch.allclose((q * dz1 * dz2).sum(dim=(-2, -1)), torch.ones(2))
    assert float(q[0, 2, 3]) > 0 and float(q[0, 0, 0]) == 0.0 and float(q[0].max()) == float(q[0, 2, 3])
    assert torch.allclose(q[1], _uniform_target(1, n, dz1, dz2, "cpu", torch.float64)[0])
    assert float(q[0, 2, 3]) == pytest.approx(1.0 / (9 * dz1 * dz2))
    empty = _support_target(torch.zeros(1, n, n), 1.0, dz1, dz2)             # nothing visited: uniform
    assert torch.allclose(empty[0], _uniform_target(1, n, dz1, dz2, "cpu", torch.float64)[0])


@needs_ref
def test_fr_support_runs_and_existing_methods_untouched(rig):
    tff, cv, X0, bm, _ = rig
    lab = bm.label_tensor()
    init = _init_dispersed(X0, 2, 16)
    su = _cfg(fr_start_steps=0, fr_rate=200.0, n_steps=40, save_every=20)
    a = run_sampler_ala("fr_support", tff, cv, su, [0, 1], init, lab, "cpu", verbose=False)
    assert np.isfinite(a["final_pmf"]).all() and int(a["total_events"].sum()) > 0
    assert np.isfinite(a["kl_support"]).all() and (a["kl_support"] <= a["kl_uniform"] + 1e-9).all()
    assert _cfg().config_hash() == json.load(open(HASHES))["test"]        # new field dropped at default
    with pytest.raises(AssertionError):
        run_sampler_ala("fr_support", tff, cv, su, [0, 1], init, lab, "cpu", reference_F=np.zeros((97, 97)), verbose=False)
