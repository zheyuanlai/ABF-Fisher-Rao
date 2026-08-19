"""Alanine port validation: ff14SB parity vs stored OpenMM fixtures, CV values and
gradients, reference/mask/conditional objects, watershed basins, the periodic-1D
layer, and engine smokes (pairing, determinism, thermostat, conditional
diagnostic) for both CV choices."""
import math

import numpy as np
import pytest
import torch

from conftest import DEVICE, DTYPE
from abpfr import grid1p as g1
from abpfr.events1p import fr_event1p
from abpfr.grid1p import Grid1P, ShusAccumulator1P
from abpfr.systems import alanine as ala
from abpfr.systems.gateway import Method

PI = math.pi


# -----------------------------------------------------------------------------
# force field parity (fixtures were generated with OpenMM at extraction time)
# -----------------------------------------------------------------------------
def test_ff_parity_vs_openmm_fixtures():
    import os
    z = np.load(os.path.join(ala.REF_DIR, "ala_parity_fixtures.npz"))
    tff = ala.TorchFF(DEVICE, DTYPE)
    X = torch.as_tensor(z["X"], dtype=DTYPE)
    E = tff.energy(X).numpy()
    F = tff.forces(X).numpy()
    # near-zero energies (the minimum, E ~ -91) inflate the relative measure;
    # the absolute deviation there is ~3e-6 kJ/mol (double-precision roundoff)
    rel_E = np.abs(E - z["E"]) / np.maximum(np.abs(z["E"]), 1.0)
    assert rel_E.max() < 1e-7, f"energy parity broken: {rel_E.max():.2e}"
    scale = np.abs(z["F"]).max(axis=(1, 2)) + 1.0
    rel_F = (np.abs(F - z["F"]).max(axis=(1, 2)) / scale)
    assert rel_F.max() < 1e-6, f"force parity broken: {rel_F.max():.2e}"


def test_forces_are_minus_grad_energy():
    tff = ala.TorchFF(DEVICE, DTYPE)
    q = tff.X0.unsqueeze(0).clone()
    f = tff.forces(q)
    eps = 1e-6
    for (i, d) in [(0, 0), (8, 1), (14, 2), (21, 0)]:
        qp, qm = q.clone(), q.clone()
        qp[0, i, d] += eps
        qm[0, i, d] -= eps
        num = -(tff.energy(qp) - tff.energy(qm)) / (2 * eps)
        assert abs(float(f[0, i, d]) - float(num)) < 1e-4


# -----------------------------------------------------------------------------
# collective variables
# -----------------------------------------------------------------------------
def signed_dihedral_np(x, idx):
    """The closed campaign's numpy reference implementation (IUPAC)."""
    p0, p1, p2, p3 = (x[..., i, :] for i in idx)
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1n = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
    v = b0 - (b0 * b1n).sum(-1, keepdims=True) * b1n
    w = b2 - (b2 * b1n).sum(-1, keepdims=True) * b1n
    return np.arctan2((np.cross(b1n, v) * w).sum(-1), (v * w).sum(-1))


def test_cv_values_match_reference_implementation():
    tff = ala.TorchFF(DEVICE, DTYPE)
    gen = torch.Generator().manual_seed(0)
    q = tff.X0.unsqueeze(0) + 0.01 * torch.randn((16, 22, 3), generator=gen,
                                                 dtype=DTYPE)
    phi, psi = ala.cv_values(q)
    x = q.numpy()
    assert np.abs(phi.numpy() - signed_dihedral_np(x, ala.PHI_ATOMS)).max() < 1e-12
    assert np.abs(psi.numpy() - signed_dihedral_np(x, ala.PSI_ATOMS)).max() < 1e-12
    # the minimised start is the C7eq region (reference global min (-74.2, 55.7) deg)
    phi0, psi0 = ala.cv_values(tff.X0.unsqueeze(0))
    assert abs(math.degrees(float(phi0)) - (-74.2)) < 25
    assert abs(math.degrees(float(psi0)) - 55.7) < 35


def test_cv_gradient_matches_finite_difference():
    tff = ala.TorchFF(DEVICE, DTYPE)
    q = tff.X0.unsqueeze(0).clone()
    for atoms in (ala.PHI_ATOMS, ala.PSI_ATOMS):
        val, g = ala.cv_value_grad(q, atoms)
        eps = 1e-6
        for (t, d) in [(0, 0), (1, 1), (2, 2), (3, 0)]:
            qp, qm = q.clone(), q.clone()
            qp[0, atoms[t], d] += eps
            qm[0, atoms[t], d] -= eps
            vp, _ = ala.cv_value_grad(qp, atoms)
            vm, _ = ala.cv_value_grad(qm, atoms)
            num = (float(vp) - float(vm)) / (2 * eps)
            assert abs(float(g[0, t, d]) - num) < 1e-5


# -----------------------------------------------------------------------------
# reference objects and basins
# -----------------------------------------------------------------------------
def test_reference_masks_and_conditionals():
    ref = ala.load_reference(DEVICE, DTYPE)
    assert ref["mask8"].sum() == 2239                 # frozen artifact content
    assert torch.isfinite(ref["F2"][ref["mask8"]]).all()
    assert torch.isfinite(ref["F1"]).all() and bool(ref["mask1"].any())
    # conditionals normalized where the column has mass
    col_mass = (ref["p_cond"].sum(dim=1) * ala.DZ)
    nz = col_mass > 0.5
    assert torch.allclose(col_mass[nz], torch.ones_like(col_mass[nz]), atol=1e-9)
    # engine grid nodes ARE the reference cell centres
    x1 = ala.ALA_GRID2.x1(DEVICE, DTYPE).numpy()
    centers = -PI + (np.arange(ala.N_GRID) + 0.5) * ala.DZ
    assert np.abs(x1 - centers).max() < 1e-12


def test_basin_labels_watershed():
    ref = ala.load_reference(DEVICE, DTYPE)
    lab, seeds = ala.basin_labels(ref["F2"], ref["mask8"])
    assert int(lab.max()) == 3 and int(lab.min()) == -1
    # every seed cell keeps its own label; the global min sits in basin 0
    for k, (i, j) in enumerate(seeds):
        assert int(lab[i, j]) == k
    F = ref["F2"].clone()
    F[~ref["mask8"]] = float("inf")
    gm = torch.nonzero(F == F[ref["mask8"]].min())[0]
    assert int(lab[gm[0], gm[1]]) == 0
    # labels only inside mask8
    assert bool((lab[~ref["mask8"].cpu()] == -1).all())


# -----------------------------------------------------------------------------
# periodic-1D layer
# -----------------------------------------------------------------------------
G1 = Grid1P(xmin=-PI, L=2 * PI, n=96)


def test_grid1p_kde_and_seam():
    k, r = g1.periodic_gaussian_kernel(0.25, G1.dx, G1.n, DEVICE, DTYPE)
    gen = torch.Generator().manual_seed(1)
    X = g1.wrap_periodic(0.3 * torch.randn((2, 3000), generator=gen, dtype=DTYPE),
                         -PI, 2 * PI)
    p = g1.binned_density1p(X, k, r, G1)
    assert torch.allclose(g1.integral1p(p, G1), torch.ones(2, dtype=DTYPE),
                          atol=1e-9)
    pa = g1.binned_density1p(torch.full((1, 400), PI - 1e-9, dtype=DTYPE), k, r, G1)
    pb = g1.binned_density1p(torch.full((1, 400), -PI, dtype=DTYPE), k, r, G1)
    assert torch.allclose(pa, pb, atol=1e-9)


def test_grid1p_interp_and_diff_periodic():
    x = G1.x(DEVICE, DTYPE)
    vals = torch.cos(x).unsqueeze(0)
    gen = torch.Generator().manual_seed(2)
    X = g1.wrap_periodic(2 * PI * torch.rand((1, 2000), generator=gen,
                                             dtype=DTYPE) - PI, -PI, 2 * PI)
    est = g1.interp1p(X, vals, G1)
    assert float((est - torch.cos(X)).abs().max()) < 3e-3
    d = g1.central_diff1p(vals, G1)
    assert float((d + torch.sin(x)).abs().max()) < 2e-3


def test_shus1p_sign_gauge_gain():
    mk = lambda gn=None: ShusAccumulator1P(
        1, G1, torch.ones((1, 1), dtype=DTYPE), 0.15, DEVICE, DTYPE,
        gain=None if gn is None else torch.full((1,), gn, dtype=DTYPE))
    gen = torch.Generator().manual_seed(3)
    X = g1.wrap_periodic(0.2 * torch.randn((1, 4000), generator=gen, dtype=DTYPE),
                         -PI, 2 * PI)
    s0, s1, sg = mk(), mk(1.0), mk(0.5)
    incs = []
    for s in (s0, s1, sg):
        s.deposit(X)
        incs.append(s.update(dt=1e-3, K=4000))
    assert torch.equal(s0.R, s1.R)
    assert torch.allclose(incs[2], 0.5 * incs[0], atol=1e-15)
    i0 = G1.n // 2                                    # x = 0 (visited)
    assert float(s0.F[0, i0]) < float(s0.F[0, 0])     # x = -pi (unvisited)
    s2 = mk()
    s2.R = s2.R * 11.0
    s2._refresh_bias()
    s2.deposit(X)
    s2.update(dt=1e-3, K=4000)
    assert torch.allclose(s0.Fp, s2.Fp, atol=1e-10)


def test_fr_event1p_estimator_protection_and_count():
    s = ShusAccumulator1P(1, G1, torch.ones((1, 1), dtype=DTYPE), 0.15,
                          DEVICE, DTYPE)
    gen = torch.Generator().manual_seed(4)
    X = g1.wrap_periodic(0.2 * torch.randn((1, 1024), generator=gen, dtype=DTYPE),
                         -PI, 2 * PI)
    s.deposit(X)
    s.update(dt=1e-3, K=1024)
    R_snap = s.R.clone()
    k, r = g1.periodic_gaussian_kernel(0.25, G1.dx, G1.n, DEVICE, DTYPE)
    sel, turn, th, ef = fr_event1p(
        X, torch.tensor([True]), torch.tensor([False]), torch.tensor([True]), 9,
        torch.tensor([0], dtype=torch.long), torch.tensor([1.0], dtype=DTYPE),
        torch.tensor([0.0], dtype=DTYPE), k, r, G1,
        torch.Generator().manual_seed(5))
    Xr = torch.gather(X, 1, sel)
    assert torch.equal(s.R, R_snap)                    # estimator untouched
    bw = 2 * PI / 9
    b = torch.remainder(((Xr + PI) / bw).long(), 9)
    cnt = torch.bincount(b[0], minlength=9).to(torch.float64)
    occ = cnt[cnt > 0]
    assert float(occ.max()) < 2.5 * float(occ.mean())  # occupied arcs equalized


# -----------------------------------------------------------------------------
# engine smokes
# -----------------------------------------------------------------------------
def tiny_cfg(**kw):
    base = dict(K=8, n_steps=400, block=20, n_saves=5, profile_every=2,
                ess_window_steps=200, eps_bw=0.15, eta_bw=0.30)
    base.update(kw)
    return ala.AlaConfig(**base)


@pytest.mark.parametrize("cv", ["phipsi", "phi"])
def test_engine_smoke_paired_deterministic(cv):
    cfg = tiny_cfg(cv=cv)
    a = ala.simulate_batch([cfg], [0], [Method("shus"), Method("b", g_shus=1.0)],
                           batch_seed=7, device=DEVICE, dtype=DTYPE)
    assert np.array_equal(a[0]["pmf_t"], a[1]["pmf_t"])          # paired + g=1
    b = ala.simulate_batch([cfg], [0], [Method("shus"), Method("b", g_shus=1.0)],
                           batch_seed=7, device=DEVICE, dtype=DTYPE)
    assert np.array_equal(a[0]["pmf_t"], b[0]["pmf_t"])          # deterministic
    r = a[0]
    assert np.isfinite(r["pmf_t"]).all() and np.isfinite(r["l2_f_t"]).all()
    assert 100.0 < r["temp_kin_t"][-1] < 700.0                   # thermostat sane
    assert abs(r["P_regions"][-1].sum() - 1.0) < 1e-9
    assert r["P_regions"][0, 0] > 0.9                            # starts in C7eq
    if cv == "phi":
        assert np.isfinite(r["e_cond_t"]).all()
        assert r["marginal2_t"].shape[-2:] == (97, 97)


def test_engine_fr_event_clones_full_state():
    cfg = tiny_cfg(cv="phipsi", n_steps=200, n_saves=3)
    fr = Method("fr", use_fr=True, theta=0.5, t_on_frac=0.0, t_off_frac=1.0,
                fr_every_blocks=2, alpha_ess=0.0)
    recs = ala.simulate_batch([cfg], [1], [Method("shus"), fr], batch_seed=11,
                              device=DEVICE, dtype=DTYPE)
    r_fr = next(r for r in recs if r["method"]["name"] == "fr")
    assert r_fr["event_turnover"].sum() > 0                      # events fired
    assert np.isfinite(r_fr["pmf_t"]).all()
    assert 50.0 < r_fr["temp_kin_t"][-1] < 900.0                 # fresh momenta sane
