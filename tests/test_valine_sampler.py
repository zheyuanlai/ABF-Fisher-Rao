"""The Val (phi, chi1) CV in the accepted alanine sampler, dense vs union-block.

The union-block path is an OPTIMISATION OF AN EXACTLY KNOWN QUANTITY: restricting the den Otter
machinery to the atoms the two dihedrals actually touch leaves G, div_v, the local mean force and
the scattered bias force unchanged.  For Val that union is {4,6,8,10,12,20} -- 6 atoms, 18 of 84
coordinates -- so the Hessian contraction shrinks by (84/18)^2 ~ 22x.  The point of these tests
is that "unchanged" is measured rather than asserted, because the sampler applies ``gfull`` in
two *different memory layouts* depending on which class is passed, and mixing them up would
produce a plausible-looking but wrong bias force rather than an error.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from alanine.basins import BasinMap
from alanine.core2d_ala import AlaSimConfig, run_sampler_ala
from alanine.cv2d import BackboneCV2D, FastBackboneCV2D
from alanine.forcefield import TorchFF, extract_parameters
from valine.system import (CHI1_ATOMS, N_ATOMS, PHI_ATOMS, make_seed, make_system,
                           validate_seed)

BETA = 1.0 / (0.008314462618 * 300.0)


@pytest.fixture(scope="module")
def rig():
    _, _, system = make_system()
    P = extract_parameters(system)
    tff = TorchFF(P, device="cpu", dtype=torch.float64)
    X, e = make_seed((-80.0, 80.0, 180.0), system=system)
    validate_seed(system, X[None], np.radians([[-80.0, 80.0, 180.0]]), energy=[e])
    dense = BackboneCV2D(PHI_ATOMS, CHI1_ATOMS, n_atoms=N_ATOMS)
    fast = FastBackboneCV2D(PHI_ATOMS, CHI1_ATOMS, n_atoms=N_ATOMS)
    return tff, X, dense, fast


def test_union_is_the_six_atoms_the_two_dihedrals_touch(rig):
    _, _, _, fast = rig
    assert fast.union == [4, 6, 8, 10, 12, 20]
    assert fast.nc == 18                      # 18 of 84 coordinates


def test_mean_force_and_scatter_match_dense(rig):
    tff, X, dense, fast = rig
    rng = np.random.default_rng(20260801)
    x = torch.as_tensor(X[None] + 0.03 * rng.standard_normal((48, N_ATOMS, 3)))
    F = tff.forces(x)
    fs, ps_, gs, geos = dense.local_mean_force(x, F, BETA)
    ff, pf, gf, geof = fast.local_mean_force(x, F, BETA)
    assert (fs - ff).abs().max() < 1e-9
    assert (ps_ - pf).abs().max() == 0.0
    assert (geos["G"] - geof["G"]).abs().max() == 0.0
    assert (geos["div_v"] - geof["div_v"]).abs().max() < 1e-12

    # The layouts differ -- (B,2,28,3) against (B,2,6,3) -- so the sampler cannot simply index
    # one as the other.  What must agree is the SCATTERED Cartesian bias force.
    assert gs.shape[-2] == N_ATOMS and gf.shape[-2] == fast.n_union
    c1, c2 = torch.randn(48, dtype=torch.float64), torch.randn(48, dtype=torch.float64)
    dense_cart = c1[:, None, None] * gs[:, 0] + c2[:, None, None] * gs[:, 1]
    assert (dense_cart - fast.scatter_bias(gf, c1, c2, N_ATOMS)).abs().max() == 0.0


def test_sampler_agrees_between_layouts_over_a_short_run(rig):
    """End-to-end: the same seed, the same noise stream, both CV classes.

    Short on purpose.  The two paths contract different-sized tensors, so their rounding differs
    in the last bits and Langevin dynamics amplifies that exponentially -- over picoseconds the
    trajectories must diverge, and a test demanding agreement there would be testing chaos, not
    correctness.  50 steps is well inside the linear regime.
    """
    tff, X, dense, fast = rig
    labels = torch.zeros(21, 21, dtype=torch.long)      # one trivial region
    sim = AlaSimConfig(n_steps=50, n_replicas=8, save_every=50, n_grid=21,
                       abf_warmup_steps=10, abf_min_count=1.0)
    init = np.repeat(X[None], 8, 0)[None]      # (R=1, N=8, A, 3)
    kw = dict(device="cpu", dtype=torch.float64, reference_F=None, rare_basin=0, verbose=False)
    a = run_sampler_ala("abf", tff, dense, sim, [0], init, labels, **kw)
    b = run_sampler_ala("abf", tff, fast, sim, [0], init, labels, **kw)
    assert np.isfinite(a["final_pmf"]).all() and np.isfinite(b["final_pmf"]).all()
    assert np.abs(a["final_pmf"] - b["final_pmf"]).max() < 1e-8


def test_basin_map_gives_neutral_names_without_hints():
    """Alanine's Ramachandran boxes must not be able to name a chi1 rotamer.

    The well below sits at roughly (-69, +51) deg, inside alanine's C7eq box.  On the Val CV
    the second axis is chi1, so that basin is a *rotamer*, and inheriting the backbone name
    would put "C7eq" on it in every table downstream.
    """
    from alanine.basins import grid_deg
    g = np.radians(grid_deg(21))
    d1 = (g[:, None] - np.radians(-69.0) + np.pi) % (2 * np.pi) - np.pi
    d2 = (g[None, :] - np.radians(+51.0) + np.pi) % (2 * np.pi) - np.pi
    F = 20.0 * (1.0 - np.exp(-(d1 ** 2 + d2 ** 2) / 0.3))
    ok = np.ones_like(F, dtype=bool)
    assert BasinMap(F, ok, 2.494).names[0] == "C7eq"          # the failure mode
    assert BasinMap(F, ok, 2.494, name_hints=()).names[0] == "B0"   # what Val uses
