"""WCA force-field audit (docs/numerical_validation/WCA_FORCE_AUDIT.md).

A1  the intended analytic force equals central differences of the intended energy (several eps,
    compact / transition / stretched dimer, several solvent states; coordinates whose stencil
    crosses a cutoff are excluded and the cutoff is checked separately for continuity);
A2  Newton's third law of the unclipped force; translation / periodic-image invariance;
A3  the PRODUCTION raw force (wca_numba.wca_force, both norm modes; the float64 torch engine) and the
    production energy equal the intended ones wherever no pair is inside min_r;
A4  min_r clamp: below min_r the production force is the gradient of an UNIMPLEMENTED quadratic
    continuation while the production energy is flat -- a force/energy inconsistency (characterised);
A5  per-particle force clip: not a gradient field (asymmetric Jacobian), breaks Newton's third law;
A6  the local mean force (Jacobian term included) equals dA/dz of the exact dimer-only free energy;
A7  the screened fast kernels equal the plain reference loops; single-particle and dimer-stretch MC
    energy differences equal total-energy differences;
A8  the EM_IMPL harness reproduces the production unbiased dynamics BITWISE (same noise);
A9  MC, MALA, EM and LM reproduce the analytic dimer-only density (statistical, chi-square).

Run:  CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_wca_force_audit.py -q
"""
import math
import os
import sys

import numpy as np
import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import wca_validation as V  # noqa: E402

PH = V.phys_array(V.DEFAULT_PHYS)
PH0 = V.phys_array(dict(V.DEFAULT_PHYS, wall=0.0))
L, RC, W, BETA = PH[0], PH[6], PH[4], PH[5]
RMIN = PH[10]


def restrained_config(z_target, seed, sweeps=3000):
    """An MC-equilibrated configuration with the dimer held near z_target by a stiff RC wall
    (z_min = z_max = z_target, k 2000 -> sd 0.02)."""
    phys = dict(z_min=z_target, z_max=z_target, wall=2000.0)
    o = V.chain("MC", V.lattice_q0(seed), n_steps=sweeps, seed=seed, phys=phys, mc_delta=0.3,
                mc_dimer_delta=0.1, mc_dimer_moves=10)
    return o["q_final"]


@pytest.fixture(scope="module")
def configs():
    out = {}
    for z, s in ((0.0, 11), (0.25, 12), (0.5, 13), (1.0, 14)):
        out[f"z{z}"] = restrained_config(z, s)
    out["free_a"] = V.chain("MC", V.lattice_q0(21), n_steps=3000, seed=21, mc_delta=0.4)["q_final"]
    out["free_b"] = V.chain("MC", V.lattice_q0(22), n_steps=3000, seed=22, mc_delta=0.4)["q_final"]
    return out


def ef(q, ph=PH):
    F = np.zeros_like(q)
    U, Uw, m2 = V.energy_force(np.ascontiguousarray(q), ph, F)
    return U, F, m2


def pair_distances(q):
    d = q[:, None, :] - q[None, :, :]
    d -= L * np.round(d / L)
    r = np.sqrt((d ** 2).sum(-1))
    r[np.diag_indices(len(q))] = np.inf
    r[0, 1] = r[1, 0] = np.inf          # the dimer pair is not a WCA pair
    return r


# ------------------------------------------------------------------------------------------- A1
@pytest.mark.parametrize("name", ["z0.0", "z0.25", "z0.5", "z1.0", "free_a", "free_b"])
def test_A1_gradient_vs_central_differences(configs, name):
    q = configs[name]
    U, F, _ = ef(q)
    r = pair_distances(q)
    worst = {}
    for eps in (1e-4, 1e-5, 1e-6):
        errs = []
        for p in range(q.shape[0]):
            if np.min(np.abs(r[p] - RC)) < 10 * eps:      # stencil would cross the cutoff kink of V''
                continue
            for k in range(2):
                qp = q.copy(); qp[p, k] += eps
                qm = q.copy(); qm[p, k] -= eps
                fd = -(V.energy_only(qp, PH) - V.energy_only(qm, PH)) / (2 * eps)
                errs.append(abs(F[p, k] - fd) / max(abs(F[p, k]), 1.0))
        worst[eps] = max(errs)
    assert worst[1e-5] < 1e-6, worst
    assert min(worst.values()) < 1e-7, worst


def test_A1_cutoff_continuity():
    """V_WCA and its derivative vanish at r_c (C1 potential); V'' jumps (so central differences
    straddling r_c are only first-order accurate, which is why A1 excludes those stencils)."""
    sig2, eps_, rc2 = 1.0, 1.0, RC * RC
    for d in (1e-6, 1e-9):
        V_in, s_in = V.wca_pair((RC - d) ** 2, sig2, eps_, rc2)
        assert abs(V_in) < 1e-6 * (d / 1e-6) ** 2 * 50 and abs(s_in * (RC - d)) < 1e-3 * d / 1e-6
    assert V.wca_pair(rc2, sig2, eps_, rc2) == (0.0, 0.0)


# ------------------------------------------------------------------------------------------- A2
def test_A2_newton_third_law_and_pbc(configs):
    rng = np.random.default_rng(0)
    for name, q in configs.items():
        U, F, _ = ef(q)
        assert np.abs(F.sum(0)).max() < 1e-10 * max(1.0, np.abs(F).max()), name
        qt = (q + rng.uniform(0, L, 2)) % L                       # global translation
        U2, F2, _ = ef(qt)
        assert abs(U2 - U) < 1e-10 * max(1, abs(U)) and np.abs(F2 - F).max() < 1e-9 * max(1, np.abs(F).max())
        qi = q.copy(); qi[5, 0] += L; qi[17, 1] -= 2 * L; qi[0, 0] -= L     # periodic images
        U3, F3, _ = ef(qi)
        assert abs(U3 - U) < 1e-10 * max(1, abs(U)) and np.abs(F3 - F).max() < 1e-9 * max(1, np.abs(F).max())
    # minimum image is valid: WCA cutoff < L/2, and a dimer as long as L/2 (where the minimum image of
    # the bond would become ambiguous) costs > 30 kT (double well + RC wall), i.e. is never visited
    assert RC < L / 2
    z_amb = (L / 2 - RC) / (2 * W)
    U_amb = V.dimer_terms(L / 2, PH[3], PH[4], PH[6], PH[7], PH[8], PH[9], PH[5])[0]
    assert z_amb > PH[9] and BETA * U_amb > 30.0, (z_amb, U_amb)


# ------------------------------------------------------------------------------------------- A3
def test_A3_production_force_and_energy_equal_intended(configs):
    for name, q in configs.items():
        r = pair_distances(q)
        assert r.min() > RMIN, "config must have every pair outside min_r"
        U, F, _ = ef(q, PH0)                          # the production force/energy carry no RC wall
        scale = max(1.0, np.abs(F).max())
        for fm in (True, False):
            Fp = V.production_force(q, fma=fm)
            assert np.abs(Fp - F).max() < 1e-12 * scale, (name, fm, np.abs(Fp - F).max())
        Ut, Ftorch = V.implemented_energy_torch(q)
        assert np.abs(Ftorch - F).max() < 1e-12 * scale
        assert abs(Ut - U) < 1e-12 * max(1.0, abs(U))


# ------------------------------------------------------------------------------------------- A4
def test_A4_min_r_clamp_force_energy_inconsistency():
    """Below min_r = 0.65 the production FORCE of a pair is -(V'(r_min)/r_min) d (it uses r_safe in
    V' but the true displacement d), i.e. -grad of U_q(r) = V(r_min) + V'(r_min)(r^2 - r_min^2)/(2 r_min),
    while the production ENERGY is V(r_safe) = V(r_min): flat.  So the implemented force is not the
    gradient of the implemented energy there (the force is conservative, but for a different,
    unimplemented energy).  Characterisation test: documents the defect, does not endorse it."""
    q = V.lattice_q0(5, jitter=0.0)                  # lattice spacing 1.5 > r_c: no active WCA pair
    q[12] = q[3]                                     # vacate the +x neighbour site of particle 2 ...
    q[3] = (q[2] + np.array([0.60, 0.0])) % L        # ... and put particle 3 at r = 0.60 < min_r from 2
    r = pair_distances(q)
    act = np.argwhere(np.triu(r < RC))
    assert act.tolist() == [[2, 3]], act
    Fp = V.production_force(q)
    Ut0, _ = V.implemented_energy_torch(q)
    h = 1e-6
    qp = q.copy(); qp[3, 0] += h
    qm = q.copy(); qm[3, 0] -= h
    fd_impl = -(V.implemented_energy_torch(qp)[0] - V.implemented_energy_torch(qm)[0]) / (2 * h)
    rmin = RMIN
    x = 1.0 / rmin
    dVdr_min = 4.0 * (-12.0 * x ** 12 / rmin + 6.0 * x ** 6 / rmin)
    expected = -(dVdr_min / rmin) * 0.60                     # x-force on particle 3 from particle 2
    # the clamped force on particle 3 points along +x with magnitude |V'(r_min)| r / r_min ~ 1.15e4
    assert abs(Fp[3, 0] - expected) < 1e-9 * abs(expected) and abs(expected) > 1e4
    # the dimer bond / other pairs contribute nothing to particle 3 here; the implemented energy is flat
    assert abs(fd_impl) < 1e-3, fd_impl
    # the force IS -grad of the quadratic continuation U_q
    Uq = lambda r: dVdr_min * (r * r - rmin * rmin) / (2 * rmin)
    fd_q = -(Uq(0.60 + h) - Uq(0.60 - h)) / (2 * h)
    assert abs(fd_q - Fp[3, 0]) < 1e-6 * abs(Fp[3, 0])


# ------------------------------------------------------------------------------------------- A5
def test_A5_force_clip_is_not_a_gradient_and_breaks_newton():
    """The production clip rescales each particle's TOTAL force to |F_i| <= 250 independently of the
    energy.  Where it binds, (a) sum_i F_i != 0 and (b) the Jacobian of the clipped field is not
    symmetric, so no potential has it as gradient (exact gradients have symmetric Jacobians)."""
    q = V.lattice_q0(6, jitter=0.0)
    c = 22                                            # site (2, 2); +x neighbour 32, +y neighbour 23
    q[32] = (q[c] + np.array([0.86, 0.0])) % L       # pair force ~272 along x
    q[23] = (q[c] + np.array([0.0, 0.84])) % L       # pair force ~382 along y
    r = pair_distances(q)
    act = sorted(map(tuple, np.argwhere(np.triu(r < RC)).tolist()))
    assert act == [(22, 23), (22, 32)], act
    Fc = V.production_clipped_force(q)
    Fr = V.production_force(q)
    assert np.linalg.norm(Fr[c]) > 250 and abs(np.linalg.norm(Fc[c]) - 250) < 1e-9
    assert np.abs(Fr.sum(0)).max() < 1e-9 * np.abs(Fr).max()
    assert np.abs(Fc.sum(0)).max() > 10.0                        # momentum not conserved
    h = 1e-6

    def J(i, a, k, b, field):
        qp = q.copy(); qp[k, b] += h
        qm = q.copy(); qm[k, b] -= h
        return (field(qp)[i, a] - field(qm)[i, a]) / (2 * h)
    # raw force: symmetric Jacobian (a gradient); clipped: asymmetric
    raw = lambda x: V.production_force(x)
    clp = lambda x: V.production_clipped_force(x)
    asym_raw = abs(J(c, 0, 23, 1, raw) - J(23, 1, c, 0, raw))
    asym_clp = abs(J(c, 0, 23, 1, clp) - J(23, 1, c, 0, clp))
    assert asym_raw < 1e-5 * max(1.0, abs(J(c, 0, 23, 1, raw))), asym_raw
    assert asym_clp > 10.0, asym_clp


# ------------------------------------------------------------------------------------------- A6
def test_A6_local_mean_force_is_dA_dz_for_the_dimer():
    """With no solvent, f(q) = (w/r) d.(F1 - F0) - 2w/(beta r) is deterministic and must equal
    dA/dz, A(z) = V_dim(r) + U_wall(z) - beta^-1 log r (2-D Jacobian), r = r_c + 2 w z."""
    phys = dict(n_dim=10)
    ph = V.phys_array(V.DEFAULT_PHYS)

    def A(z):
        r = RC + 2 * W * z
        return V.dimer_terms(r, ph[3], ph[4], ph[6], ph[7], ph[8], ph[9], ph[5])[0] - math.log(r) / BETA
    for z in (-0.25, -0.1, 0.0, 0.2, 0.25, 0.5, 0.8, 1.0, 1.15, 1.3):     # r > 0 needs z > -0.28
        r = RC + 2 * W * z
        th = 0.37
        q = np.array([[5.0 + r * math.cos(th), 5.0 + r * math.sin(th)], [5.0, 5.0]])
        F = np.zeros_like(q)
        V.energy_force(q, ph, F)
        dx, dy, rr, zz = V.dimer_geom(q, L, RC, W)
        f = V.local_mf(dx, dy, rr, F, W, BETA)
        h = 1e-6
        dA = (A(z + h) - A(z - h)) / (2 * h)
        assert abs(zz - z) < 1e-12 and abs(f - dA) < 1e-6 * max(1, abs(dA)), (z, f, dA)


# ------------------------------------------------------------------------------------------- A7
def test_A7_fast_kernels_and_mc_energy_differences(configs):
    rng = np.random.default_rng(1)
    for name, q in configs.items():
        F1 = np.zeros_like(q); F2 = np.zeros_like(q)
        a = V.energy_force(q, PH, F1)
        b = V.energy_force_ref(q, PH, F2)
        assert abs(a[0] - b[0]) < 1e-12 * max(1, abs(a[0])) and np.abs(F1 - F2).max() < 1e-11 * max(1, np.abs(F1).max())
        assert abs(a[0] - V.energy_only(q, PH)) < 1e-12 * max(1, abs(a[0]))
        for i in (0, 1, 2, 50):
            x, y = (q[i] + rng.uniform(-0.2, 0.2, 2)) % L
            assert abs(V.particle_energy(q, i, x, y, PH) - V.particle_energy_ref(q, i, x, y, PH)) < 1e-10
            qn = q.copy(); qn[i] = (x, y)
            dU_tot = V.energy_only(qn, PH) - V.energy_only(q, PH)
            dU_loc = V.particle_energy(q, i, x, y, PH) - V.particle_energy(q, i, q[i, 0], q[i, 1], PH)
            assert abs(dU_tot - dU_loc) < 1e-9 * max(1, abs(dU_tot))


def test_A7_mc_dimer_stretch_energy_bookkeeping(configs):
    """The dimer-stretch move's (Unew - Uold) equals the total-energy difference."""
    q = configs["z0.5"]
    dx, dy, r, z = V.dimer_geom(q, L, RC, W)
    for rn in (r - 0.05, r + 0.07):
        mx, my = q[1, 0] + 0.5 * dx, q[1, 1] + 0.5 * dy
        ux, uy = dx / r, dy / r
        qn = q.copy()
        qn[0] = ((mx + 0.5 * rn * ux) % L, (my + 0.5 * rn * uy) % L)
        qn[1] = ((mx - 0.5 * rn * ux) % L, (my - 0.5 * rn * uy) % L)
        dterm = lambda rr: V.dimer_terms(rr, PH[3], PH[4], PH[6], PH[7], PH[8], PH[9], PH[5])[0]
        Uold = V.particle_energy(q, 0, q[0, 0], q[0, 1], PH) + V.particle_energy(q, 1, q[1, 0], q[1, 1], PH) - dterm(r)
        Unew = V.particle_energy(qn, 0, qn[0, 0], qn[0, 1], PH) + V.particle_energy(qn, 1, qn[1, 0], qn[1, 1], PH) - dterm(rn)
        assert abs((Unew - Uold) - (V.energy_only(qn, PH) - V.energy_only(q, PH))) < 1e-9
        assert abs(V.dimer_geom(qn, L, RC, W)[2] - rn) < 1e-12


# ------------------------------------------------------------------------------------------- A8
def test_A8_em_impl_harness_is_the_production_dynamics():
    """Same initial state + same Gaussian stream => the harness's deposits (counts and the production
    clipped local mean force) equal wca_numba.simulate's bias accumulators BITWISE (unbiased branch,
    accepted clips), including steps where the force clip binds (dt 0.002)."""
    import wca_numba as wn
    for dt, n in ((0.002, 3000), (0.0005, 3000)):
        q0 = V.lattice_q0(7)
        o = V.chain("EM_IMPL", q0, dt=dt, n_steps=n, seed=99, diag_stride=1000)
        noise = np.random.Generator(np.random.PCG64(99)).standard_normal(n * 200).reshape(n, 200)
        cfg = dict(wn.ACCEPTED_CFG, abf_bias_scale=0.0, dt=dt)
        res = wn.run_ladder_point(1, 1, n, [("u", False)], cfg=cfg, q0=q0[None], ext_noise=noise,
                                  save_at=np.array([n - 1]))
        Cb = np.asarray(res["C_bias"])[0, 0]
        Mb = np.asarray(res["M_bias"])[0, 0]
        # the production estimator clamps out-of-window z into the edge bins; the harness keeps them apart

        def fold(a):
            b = a[1:-1].copy()
            b[0] += a[0]
            b[-1] += a[-1]
            return b
        assert np.array_equal(fold(o["C"]), Cb), (dt, np.abs(fold(o["C"]) - Cb).sum())
        assert np.allclose(fold(o["M_impl"]), Mb, rtol=1e-12, atol=1e-9), (dt, np.abs(fold(o["M_impl"]) - Mb).max())
        if dt == 0.002:
            assert o["n_clip_steps"] > 0          # the stress case really exercises the clip
        assert np.array_equal(o["q_final"], np.asarray(res["q_final"])[0, 0])      # same trajectory


# ------------------------------------------------------------------------------------------- A9
def _dimer_only_density(z_edges, dt_em=0.0):
    """Bin probabilities of p(z) ~ r exp(-beta U(r)), U = V_dim + U_wall.  With dt_em > 0: the
    first-order Euler-Maruyama invariant density p_dt ~ p (1 + dt f1), f1 = (beta/4)|grad U|^2 - (1/2) Lap U
    (unit mobility; for a function of r = |q0 - q1| in 2-D: |grad U|^2 = 2 U'^2, Lap U = 2 (U'' + U'/r)).
    The OU case fixes the |grad U|^2 coefficient (EM variance 1/(k(1 - k dt/2)))."""
    ph = V.phys_array(V.DEFAULT_PHYS)
    zz = np.linspace(z_edges[0], z_edges[-1], 200001)
    r = RC + 2 * W * zz
    U = np.array([V.dimer_terms(x, ph[3], ph[4], ph[6], ph[7], ph[8], ph[9], ph[5])[0] for x in r])
    dens = r * np.exp(-BETA * U)
    if dt_em:
        h = 1e-5
        Up = np.array([V.dimer_terms(x + h, ph[3], ph[4], ph[6], ph[7], ph[8], ph[9], ph[5])[0] for x in r])
        Um = np.array([V.dimer_terms(x - h, ph[3], ph[4], ph[6], ph[7], ph[8], ph[9], ph[5])[0] for x in r])
        d1 = (Up - Um) / (2 * h)
        d2 = (Up - 2 * U + Um) / h ** 2
        f1 = 0.25 * BETA * 2 * d1 ** 2 - 0.5 * 2 * (d2 + d1 / r)
        dens = dens * (1.0 + dt_em * f1)
    cdf = np.concatenate([[0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(zz))])
    probs = np.diff(np.interp(z_edges, zz, cdf))
    return probs / probs.sum()


@pytest.mark.parametrize("scheme,dt,n,exact", [("MC", 0.0, 1_000_000, True), ("MALA", 0.002, 1_000_000, True),
                                                ("EM_TRUE", 0.002, 1_000_000, False), ("LM_TRUE", 0.002, 1_000_000, False)])
def test_A9_dimer_only_density(scheme, dt, n, exact):
    """Dimer alone (P = 2): p(z) ~ r exp(-beta (V_dim + U_wall)).  48 independent chains.  Bins are
    strongly correlated within a well (a chain's well population fluctuates as a whole), so the test is
    on (i) the well population P(z < 0.5) and (ii) the within-well conditional shape on 16 coarse bins
    per well, each with the across-chain standard error (an earlier chi-square that treated the 32 coarse
    bins as independent rejected a CORRECT sampler).  Exact samplers: |t| < 4 on (i), max |t| < 4.5 over
    the 32 shape bins.  EM is compared with its FIRST-ORDER invariant density (``_dimer_only_density``
    with dt_em = dt; the exact density is off by up to ~3 % in the far compact tail at dt 0.002, the size
    the theory predicts), LM (second order) with the exact one, both with the same t thresholds.""" 
    q0 = np.array([[5.0, 5.0], [5.0, 5.0 + RC]])
    pe = _dimer_only_density(np.linspace(-0.2, 1.2, 161), dt_em=dt if scheme == "EM_TRUE" else 0.0)
    C = []
    for s in range(48):
        o = V.chain(scheme, q0, dt=dt if dt else 0.001, n_steps=n, seed=5000 + s, burn_in=n // 50,
                    mc_dimer_delta=1.0, mc_dimer_moves=1, mc_delta=0.6)
        C.append(o["C"][1:-1])
    C = np.array(C)
    frac = C / C.sum(1, keepdims=True)
    lo = frac[:, :80].sum(1)                                  # z < 0.5
    t_pop = (lo.mean() - pe[:80].sum()) / (lo.std(ddof=1) / np.sqrt(len(lo)))
    assert abs(t_pop) < 4.0, t_pop
    worst_t, worst_lr = 0.0, 0.0
    for sl in (slice(0, 80), slice(80, 160)):
        f = frac[:, sl] / frac[:, sl].sum(1, keepdims=True)
        e = pe[sl] / pe[sl].sum()
        fc = f.reshape(len(f), 16, -1).sum(2)
        ec = e.reshape(16, -1).sum(1)
        t = (fc.mean(0) - ec) / (fc.std(0, ddof=1) / np.sqrt(len(fc)))
        worst_t = max(worst_t, float(np.abs(t).max()))
        worst_lr = max(worst_lr, float(np.abs(np.log(fc.mean(0) / ec)).max()))
    assert worst_t < 4.5, (scheme, worst_t, worst_lr)
    if scheme == "EM_TRUE":      # and the EM tail deviation from the EXACT density is real (the test has power)
        pex = _dimer_only_density(np.linspace(-0.2, 1.2, 161))
        f = frac[:, :80] / frac[:, :80].sum(1, keepdims=True)
        e = pex[:80] / pex[:80].sum()
        assert abs(np.log(f[:, :5].sum(1).mean() / e[:5].sum())) > 0.01
