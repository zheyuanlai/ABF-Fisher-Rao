"""Engineering tests for the molecular phase.  ~2 min on one GPU."""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rcwfr.mol import systems as S
from rcwfr.mol.dynamics import constrained_step, free_step
from rcwfr.mol.ff import (_wrap, angle, bond, dihedral, ideal_alkane,
                          rotate_about_bond, ua_alkane)
from rcwfr.mol.geom import TorsionCV
from rcwfr.mol.joint import JointRefresh

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DT = torch.float64


@pytest.fixture(scope="module")
def pen():
    torch.manual_seed(0)
    return S.pentane(DEV, DT)


def _rand_conf(sy, B=512, jit=0.02):
    nt = sy.top.tor_idx.shape[0]
    ph = (torch.rand(B, nt, device=DEV, dtype=DT) * 2 - 1) * math.pi
    return sy.ideal(ph) + jit * torch.randn(B, sy.top.n_atoms, 3, device=DEV, dtype=DT)


# --- force field and geometry ---------------------------------------------
def test_trappe_torsion_profile():
    """Butane: trans at 0, gauche at +-115 deg, the two barriers at 3.30/4.56."""
    top = ua_alkane(4, DEV, DT)
    ph = torch.linspace(-math.pi, math.pi, 361, device=DEV, dtype=DT).unsqueeze(-1)
    q = ideal_alkane(top, 4, ph, DEV, DT)
    E = top.energy(q) - top.energy(q).min()
    deg = torch.rad2deg(ph[:, 0])
    assert float(E[deg.abs() < 1e-6]) < 1e-9
    g = int(torch.argmin(torch.where(deg > 60, E, torch.full_like(E, 1e9))))
    assert 110 < float(deg[g]) < 125
    assert 0.80 < float(E[g]) < 0.90          # gauche well depth
    assert abs(float(E[-1]) - 4.556) < 0.01              # cis barrier
    assert top.lj_idx.numel() == 0                       # TraPPE excludes 1-4


def test_alkane_lj_exclusions():
    """Only 1-5 and beyond; pentane keeps exactly the CH3...CH3 contact."""
    assert ua_alkane(5, DEV, DT).lj_idx.cpu().numpy().tolist() == [[0, 4]]
    assert ua_alkane(6, DEV, DT).lj_idx.cpu().numpy().tolist() == [[0, 4], [0, 5], [1, 5]]


def test_rotation_is_exact_and_isolated(pen):
    """A distal-fragment rotation moves ONE dihedral and nothing else."""
    q = _rand_conf(pen, 256)
    p0 = dihedral(q, pen.top.tor_idx)
    d = (torch.rand(256, device=DEV, dtype=DT) * 2 - 1) * 3.0
    q1 = rotate_about_bond(q, *pen.y_bond, list(pen.y_movers), -d)
    ch = _wrap(dihedral(q1, pen.top.tor_idx) - p0)
    assert float((ch[:, 1] - d).abs().max()) < 1e-12
    assert float(ch[:, 0].abs().max()) < 1e-12
    assert float((bond(q1, pen.top.bond_idx) - bond(q, pen.top.bond_idx)).abs().max()) < 1e-12
    assert float((angle(q1, pen.top.ang_idx) - angle(q, pen.top.ang_idx)).abs().max()) < 1e-12


def test_cv_gradient_matches_finite_difference(pen):
    q = _rand_conf(pen, 8)
    g = pen.cv.grad(q)[:, 0]
    eps = 1e-6
    for (i, c) in [(0, 0), (2, 1), (3, 2)]:
        qp, qm = q.clone(), q.clone()
        qp[:, i, c] += eps; qm[:, i, c] -= eps
        fd = (pen.cv.value(qp)[:, 0] - pen.cv.value(qm)[:, 0]) / (2 * eps)
        assert float((fd - g[:, i, c]).abs().max()) < 1e-6


def test_shake_and_tangent_projection(pen):
    """Frozen-Jacobian SHAKE contracts at a rate O(|dz|): machine precision for a
    dynamics-sized step, and a gradient refresh for a lift-sized one."""
    q = _rand_conf(pen, 512)
    small = pen.cv.value(q) + (torch.rand(512, 1, device=DEV, dtype=DT) * 2 - 1) * 0.02
    qp, _ = pen.cv.project(q, small, n_newton=6, n_outer=1)
    assert float(pen.cv.dz_residual(pen.cv.value(qp), small).abs().max()) < 1e-13
    big = pen.cv.value(q) + (torch.rand(512, 1, device=DEV, dtype=DT) * 2 - 1) * 0.3
    qp, _ = pen.cv.project(q, big, n_newton=6, n_outer=2)
    assert float(pen.cv.dz_residual(pen.cv.value(qp), big).abs().max()) < 1e-12
    z = big
    gs = pen.cv.grad_local(qp)
    G = pen.cv.gram_from_grad(gs)
    v = torch.randn(512, pen.cv.S, 3, device=DEV, dtype=DT)
    pv = pen.cv.tangent_project_local(gs, G, v)
    assert float(torch.einsum("...kij,...ij->...k", gs, pv).abs().max()) < 1e-12


def test_hessian_paths_agree(pen):
    """The vectorised Hessian used by the mean force matches plain autograd."""
    q = _rand_conf(pen, 16)
    qs = q[..., pen.cv.support, :]
    H = pen.cv._hessian(qs)[:, 0]
    n = 3 * pen.cv.S
    x = qs.detach().reshape(16, n).requires_grad_(True)
    fn = lambda y: dihedral(y.reshape(-1, pen.cv.S, 3), pen.cv.idx_local)[..., 0].sum()
    (g1,) = torch.autograd.grad(fn(x), x, create_graph=True)
    rows = [torch.autograd.grad(g1[:, i].sum(), x, retain_graph=True)[0] for i in range(n)]
    assert float((H - torch.stack(rows, -2)).abs().max()) < 1e-8


# --- statistical correctness ----------------------------------------------
def test_constrained_dynamics_preserves_the_constraint(pen):
    q = _rand_conf(pen, 1024)
    z = pen.cv.value(q)
    for _ in range(200):
        q = constrained_step(pen.top, pen.cv, q, z, pen.h, pen.beta, n_newton=6)
    assert float(pen.cv.dz_residual(pen.cv.value(q), z).abs().max()) < 1e-10
    assert torch.isfinite(pen.top.energy(q)).all()


@pytest.mark.skipif(not os.path.exists("results/mol/ref/PEN_ref.npz"),
                    reason="needs the unbiased reference")
def test_metropolis_move_reaches_the_reference_conditional(pen):
    """The whole claim behind the practical arm: the move is exact whatever the
    proposal is.  The proposal here is deliberately WRONG -- a fixed skewed
    density that knows nothing about the potential -- and the ensemble it drives
    a cold start to is compared against the UNBIASED-MD reference conditional
    p(phi2 | phi1), which the move never sees.

    Natural relaxation of phi2 takes tau_y ~ 1.3e5 steps; the test allows 4e3.
    """
    torch.manual_seed(5)
    B, z0, nb = 16384, 0.0, 18
    ph = torch.zeros(B, 2, device=DEV, dtype=DT); ph[:, 0] = z0
    q = pen.ideal(ph)                                    # a DELTA in phi2
    z = torch.full((B, 1), z0, device=DEV, dtype=DT)
    full = TorsionCV(pen.top.tor_idx, pen.top.mass)
    # a deliberately wrong proposal: a fixed Gaussian bump plus a background
    yv = torch.linspace(-math.pi, math.pi, 361, device=DEV, dtype=DT)
    pdf = torch.exp(-0.5 * ((yv - 2.0) / 1.0) ** 2) + 0.25
    pdf = pdf / torch.trapezoid(pdf, yv)
    cdf = torch.cat([torch.zeros(1, device=DEV, dtype=DT),
                     torch.cumulative_trapezoid(pdf, yv)])
    cdf = cdf / cdf[-1]
    lp = lambda a: torch.log(pdf[torch.clamp(
        ((a + math.pi) / (2 * math.pi) * 360).long(), 0, 360)])
    acc = None
    for _ in range(200):
        for _k in range(20):
            q = constrained_step(pen.top, pen.cv, q, z, pen.h, pen.beta, n_newton=6)
        y = full.value(q)[:, 1]
        u = torch.rand(B, device=DEV, dtype=DT)
        j = torch.clamp(torch.searchsorted(cdf, u), 1, 360)
        t = (u - cdf[j - 1]) / (cdf[j] - cdf[j - 1]).clamp_min(1e-14)
        yp = yv[j - 1] + t * (yv[j] - yv[j - 1])
        qp = rotate_about_bond(q, *pen.y_bond, list(pen.y_movers), -_wrap(yp - y))
        logA = -pen.beta * (pen.top.energy(qp) - pen.top.energy(q)) + lp(y) - lp(yp)
        acc = torch.rand(B, device=DEV, dtype=DT) < torch.exp(torch.clamp(logA, max=0.0))
        q = torch.where(acc[:, None, None], qp, q)
    assert float(acc.to(DT).mean()) > 0.05, "proposal never accepted; test is vacuous"
    got = torch.histc(full.value(q)[:, 1], nb, -math.pi, math.pi).cpu().numpy()
    got = got / got.sum()
    d = np.load("results/mol/ref/PEN_ref.npz")
    H2 = d["H2"].sum(0)
    ctr = d["centers"]
    iz = int(np.argmin(np.abs(ctr - z0)))
    row = H2[iz - 2:iz + 3].sum(0)
    f = row.size // nb
    ref = row[: nb * f].reshape(nb, f).sum(1)
    ref = ref / ref.sum()
    tv = 0.5 * float(np.abs(got - ref).sum())
    assert tv < 0.05, f"MH-driven conditional is {tv:.3f} in TV from the reference"


def test_fixman_weight_is_benign(pen):
    q = _rand_conf(pen, 4096, jit=0.01)
    z = pen.cv.value(q)
    for _ in range(2000):
        q = constrained_step(pen.top, pen.cv, q, z, pen.h, pen.beta, n_newton=6)
    G = pen.cv.gram(q)[..., 0, 0]
    w = G ** -0.5
    ess = float((w.sum() ** 2) / (w.pow(2).sum() * w.numel()))
    assert ess > 0.9, ess


def test_joint_refresh_reproduces_its_table():
    torch.manual_seed(1)
    nz, n1, n2 = 8, 40, 40
    H = torch.rand(nz, n1, n2, device=DEV, dtype=DT) ** 3 + 1e-3
    J = JointRefresh(H, DEV, DT, smooth=1, eps_bg=0.0)
    B = 800_000                      # L1 sampling noise over 1600 cells ~ 0.036
    Zt = torch.full((B,), 0.4, device=DEV, dtype=DT)
    u = torch.rand(B, 2, device=DEV, dtype=DT)
    y = J.sample(Zt, u)
    iz = J._iz(Zt)[0]
    ref = (H[iz] / H[iz].sum()).cpu().numpy()
    got = np.histogram2d(y[:, 0].cpu().numpy(), y[:, 1].cpu().numpy(),
                         bins=[n1, n2], range=[[-math.pi, math.pi]] * 2)[0]
    got = got / got.sum()
    assert np.abs(got - ref).sum() < 0.05


def test_mean_force_is_independent_of_the_conjugate_field(pen):
    """Different mass matrices give different den Otter-Briels fields w, hence
    different mean-force ESTIMATORS -- but the same conditional average, which is
    what thermodynamic integration reads.  Pointwise they differ by O(1)."""
    torch.manual_seed(2)
    B, z0 = 8192, 1.0
    ph = torch.zeros(B, 2, device=DEV, dtype=DT); ph[:, 0] = z0
    ph[:, 1] = (torch.rand(B, device=DEV, dtype=DT) * 2 - 1) * math.pi
    q = pen.ideal(ph)
    z = torch.full((B, 1), z0, device=DEV, dtype=DT)
    cv2 = TorsionCV(pen.top.tor_idx[:1], torch.full_like(pen.top.mass, 7.0))
    m1, m2 = [], []
    for it in range(20_000):
        q = constrained_step(pen.top, pen.cv, q, z, pen.h, pen.beta, n_newton=6)
        if it >= 4_000 and it % 200 == 0:
            gV = pen.top.grad(q)
            f1, G1 = pen.cv.mean_force(q, gV, pen.beta)
            f2, G2 = cv2.mean_force(q, gV, pen.beta)
            w1, w2 = G1[..., 0, 0] ** -0.5, G2[..., 0, 0] ** -0.5
            m1.append(float((w1 * f1[:, 0]).sum() / w1.sum()))
            m2.append(float((w2 * f2[:, 0]).sum() / w2.sum()))
    a, b = np.array(m1), np.array(m2)
    se = np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
    assert abs(a.mean() - b.mean()) < max(4 * se, 0.05 * abs(a.mean())), \
        f"{a.mean():.4f} vs {b.mean():.4f} (s.e. {se:.4f})"


@pytest.mark.skipif(not os.path.exists("results/mol/gate1/ALA_ff_parity.json"),
                    reason="run scripts/mol_ala_gate.py first")
def test_alanine_forcefield_parity():
    import json
    r = json.load(open("results/mol/gate1/ALA_ff_parity.json"))
    assert r["max_rel_E"] < 1e-8 and r["max_rel_F"] < 1e-7


# --- two-dimensional reaction coordinates ---------------------------------
def _g2(bcx="periodic", nx=97, ny=97, wx=math.pi):
    from rcwfr.grid import Grid1D
    from rcwfr.mol.grid2d import Grid2D
    return Grid2D(Grid1D(-wx, wx, nx, -wx, wx, bcx),
                  Grid1D(-math.pi, math.pi, ny, -math.pi, math.pi, "periodic"))


def test_poisson_potential_is_exact_on_a_known_field():
    """F is the least-squares potential of the sampled mean force; on a field that
    IS a gradient it must return that potential to machine precision."""
    from rcwfr.mol.grid2d import gauge_l2_2d, poisson_potential
    g = _g2()
    X, Y = g.axes(DEV, DT)
    XX, YY = X.view(-1, 1), Y.view(1, -1)
    F = (2.0 * torch.cos(XX) + 1.3 * torch.sin(2 * YY)
         + 0.7 * torch.cos(XX - YY)).unsqueeze(0)
    fx = (-2.0 * torch.sin(XX) - 0.7 * torch.sin(XX - YY)).expand_as(F[0]).unsqueeze(0)
    fy = (2.6 * torch.cos(2 * YY) + 0.7 * torch.sin(XX - YY)).expand_as(F[0]).unsqueeze(0)
    Fh, curl = poisson_potential(torch.stack([fx, fy], -1), g)
    assert float(gauge_l2_2d(Fh, F, g.mask(DEV, DT))) < 1e-12
    assert float(curl) < 1e-12


def test_curl_is_reported_not_absorbed():
    """A non-gradient component cannot be explained by any F, so it must show up
    in curl_frac and must NOT contaminate the recovered potential."""
    from rcwfr.mol.grid2d import gauge_l2_2d, poisson_potential
    g = _g2()
    X, Y = g.axes(DEV, DT)
    XX, YY = X.view(-1, 1), Y.view(1, -1)
    F = (2.0 * torch.cos(XX) + 1.3 * torch.sin(2 * YY)).unsqueeze(0)
    fx = (-2.0 * torch.sin(XX)).expand_as(F[0]).unsqueeze(0)
    fy = (2.6 * torch.cos(2 * YY)).expand_as(F[0]).unsqueeze(0)
    cx = fx - 0.5 * torch.sin(YY).expand_as(fx[0]).unsqueeze(0)
    cy = fy + 0.5 * torch.sin(XX).expand_as(fy[0]).unsqueeze(0)
    Fh, curl = poisson_potential(torch.stack([cx, cy], -1), g)
    assert float(gauge_l2_2d(Fh, F, g.mask(DEV, DT))) < 1e-12
    assert float(curl) > 0.1


def test_reflecting_axis_uses_the_neumann_extension():
    from rcwfr.mol.grid2d import gauge_l2_2d, poisson_potential
    g = _g2("reflect", nx=97, wx=1.4)
    X, Y = g.axes(DEV, DT)
    XX, YY = X.view(-1, 1), Y.view(1, -1)
    k = math.pi / 2.8
    F = (2.0 * torch.cos(k * (XX + 1.4)) + 1.3 * torch.sin(2 * YY)).unsqueeze(0)
    fx = (-2.0 * k * torch.sin(k * (XX + 1.4))).expand_as(F[0]).unsqueeze(0)
    fy = (2.6 * torch.cos(2 * YY)).expand_as(F[0]).unsqueeze(0)
    Fh, _ = poisson_potential(torch.stack([fx, fy], -1), g)
    assert float(gauge_l2_2d(Fh, F, g.mask(DEV, DT))) < 1e-3


def test_snap_at_switch_makes_the_windows_uniform(pen):
    """The switch is only useful if stage B is a genuine stratification; check
    the replicas really land on an evenly spaced grid."""
    from rcwfr.mol.engines import MolCfg, run_constrained
    cfg = MolCfg(N=128, n_steps=6000, n_cond=20, dep_every=20, save_every=3000,
                 n_eq=1000, init="point", w_mode="sde", fr_rule="fr", kappa=0.6,
                 theta=0.3, lift="ymh_learned", t_switch=3000, snap_at_switch=True)
    out = run_constrained(pen, cfg, 2, seed=11)
    z = out["z_final"][..., 0]
    d = torch.diff(torch.sort(z, dim=1).values, dim=1)
    expect = pen.grid.volume / cfg.N
    assert float((d - expect).abs().max()) < 1e-9, float((d - expect).abs().max())
