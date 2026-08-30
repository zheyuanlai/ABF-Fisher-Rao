"""Engine-correctness tests for the flexible ZIF-8 stage.

Scope note (declared): these tests validate the ENGINE MATH -- functional
forms, exclusions, PBC, the periodic CV, the integrator, cloning -- on a small
SYNTHETIC framework npz.  Parameter-vs-literature correctness is NOT testable
here: the topology/parameters are validated inside
scripts/build_zif8_framework.py, which reproduces the published 2x2x2 GROMACS
enumeration term by term, and the physical model is validated by the frozen
Stage-0 gates (lattice stability, gate-aperture statistics, dt gate).
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from alkanes import periodic as per  # noqa: E402
from zif8.core_zif8 import (GUEST, KB, TWO_PI, ZIF8SimConfig, ZIF8System,  # noqa: E402
                            gate_hist, js_divergence, run_sampler, wham_periodic)

DEVICE = torch.device("cpu")
BOX = 20.0
PERIOD = 8.0          # synthetic CV period along +x


# ------------------------------------------------------------------ fixture
def synthetic_framework(tmp_path, box=BOX, rc=6.0):
    """Tiny fake 'framework': 8 atoms, two bonded triangles + 2 loose ions,
    with every energy-term class populated."""
    pos = np.array([
        [8.0, 10.0, 10.0], [9.2, 10.6, 10.0], [8.6, 11.6, 10.4],   # triangle 1
        [12.0, 10.0, 10.0], [13.2, 10.4, 10.2], [12.6, 11.4, 9.8],  # triangle 2
        [10.0, 13.0, 10.0], [10.0, 7.0, 10.0],                      # ions
    ])
    A = len(pos)
    lj_scale = np.ones((A, A)); coul_scale = np.ones((A, A))
    bonds = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (1, 3)]
    for i, j in bonds:
        lj_scale[i, j] = lj_scale[j, i] = 0.0
        coul_scale[i, j] = coul_scale[j, i] = 0.0
    for i, j in [(0, 3), (2, 3), (1, 4), (1, 5)]:            # 1-3
        lj_scale[i, j] = lj_scale[j, i] = 0.0
        coul_scale[i, j] = coul_scale[j, i] = 0.0
    for i, j in [(0, 4), (0, 5), (2, 4), (2, 5)]:            # 1-4
        lj_scale[i, j] = lj_scale[j, i] = 0.5
        coul_scale[i, j] = coul_scale[j, i] = 5.0 / 6.0
    np.fill_diagonal(lj_scale, 0.0); np.fill_diagonal(coul_scale, 0.0)
    z = dict(
        box=np.array([box] * 3), rc=rc, dsf_alpha=0.2, pos=pos,
        mass_amu=np.array([12.0, 14.0, 12.0, 12.0, 14.0, 12.0, 65.0, 65.0]),
        charge_e=np.array([0.2, -0.3, 0.1, 0.1, -0.3, 0.2, 0.5, -0.5]),
        lj_eps_kj=np.full(A, 0.5), lj_sig_A=np.full(A, 3.0),
        lj_scale=lj_scale, coul_scale=coul_scale,
        bonds=np.array(bonds), bond_k=np.full(len(bonds), 900.0),
        bond_r0=np.array([1.35, 1.2, 1.35, 1.35, 1.2, 1.35, 3.9]),
        angles=np.array([[0, 1, 2], [3, 4, 5], [0, 1, 3]]),
        angle_k=np.array([500.0, 500.0, 300.0]),
        angle_th0=np.radians([60.0, 60.0, 120.0]),
        dihedrals=np.array([[0, 1, 3, 4]]), dih_k=np.array([8.0]),
        dih_n=np.array([2.0]), dih_delta=np.array([math.pi]),
        impropers=np.array([[2, 0, 1, 3]]), impr_k=np.array([40.0]),
        impr_psi0=np.array([0.0]),
        cage_A=np.array([10.0 - PERIOD / 2, 10.0, 10.0]),
        cage_B=np.array([10.0 + PERIOD / 2, 10.0, 10.0]),
        win_center=np.array([10.0, 10.0, 10.0]), win_normal=np.array([1.0, 0.0, 0.0]),
        period=PERIOD, xi_A=-PERIOD / 2, xi_B=PERIOD / 2,
        R_tube=4.0, k_wall=100.0,
        gate_zn_idx=np.array([0, 1, 2, 3, 4, 5]),
        gate_aperture_h=np.array([0, 1, 2, 3, 4, 5]),
        gate_methyl_c=np.array([6, 7, 6]),
        gate_tri=np.array([[0, 1, 2]] * 3 + [[3, 4, 5]] * 3),
        gate_aperture_crystal=np.full(6, 2.8),
    )
    path = os.path.join(tmp_path, "framework.npz")
    np.savez(path, **z)
    return path


def make_system(tmp_path, with_guest=True, T=300.0):
    fw = synthetic_framework(str(tmp_path))
    return ZIF8System(T, DEVICE, root="/", framework=fw.lstrip("/"),
                      with_guest=with_guest, compile=False)


def rand_config(system, B=3, jitter=0.1, seed=0):
    g = torch.Generator().manual_seed(seed)
    base = torch.zeros(B, system.n_atoms, 3, dtype=torch.float64)
    base[:, :system.n_frame] = system.pos0_frame[None]
    if system.with_guest:
        base[:, system.n_frame:] = torch.tensor([[3.0, 10.2, 10.1],
                                                 [4.5, 10.4, 10.3]])[None]
    return base + jitter * torch.randn(base.shape, generator=g, dtype=torch.float64)


def make_pool(tmp_path, system, P=32, jitter=0.05):
    q = rand_config(system, B=P, jitter=jitter, seed=7).numpy()
    path = os.path.join(str(tmp_path), "pool.npz")
    np.savez(path, q=q)
    return path


# ------------------------------------------------------------------- forces
def test_forces_match_fd(tmp_path):
    sys_ = make_system(tmp_path)
    q = rand_config(sys_, B=2)
    F = sys_.forces(q)
    h = 1.0e-6
    rng = np.random.default_rng(1)
    for _ in range(24):
        b = rng.integers(0, q.shape[0]); a = rng.integers(0, sys_.n_atoms)
        d = rng.integers(0, 3)
        qp = q.clone(); qp[b, a, d] += h
        qm = q.clone(); qm[b, a, d] -= h
        fd = -(sys_.potential_energy(qp)[b] - sys_.potential_energy(qm)[b]) / (2 * h)
        assert abs(float(F[b, a, d]) - float(fd)) < 1e-5 * max(1.0, abs(float(fd))), \
            f"FD mismatch at atom {a} dim {d}: {float(F[b, a, d])} vs {float(fd)}"


def test_translation_and_wrap_invariance(tmp_path):
    sys_nf = make_system(tmp_path, with_guest=False)
    q = rand_config(sys_nf, B=2)
    E0 = sys_nf.potential_energy(q)
    for shift in ([3.1, -7.7, 40.0], [20.0, 0.0, 0.0]):
        E1 = sys_nf.potential_energy(q + torch.tensor(shift, dtype=torch.float64))
        assert torch.allclose(E0, E1, atol=1e-9)
    # With the guest, the tube and the CV are LAB-FIXED on UNWRAPPED
    # coordinates, so moving the guest is physical, not a gauge choice.  What
    # must still be invariant: translating the FRAMEWORK by a lattice vector,
    # and re-wrapping any single framework atom.
    sys_ = make_system(tmp_path)
    qg = rand_config(sys_, B=2)
    E0 = sys_.potential_energy(qg)
    for shift in ([BOX, -BOX, 2 * BOX], [0.0, 0.0, -BOX]):
        q2 = qg.clone()
        q2[:, :sys_.n_frame] += torch.tensor(shift, dtype=torch.float64)
        assert torch.allclose(E0, sys_.potential_energy(q2), atol=1e-9)
    q2 = qg.clone(); q2[:, 6] += torch.tensor([0.0, BOX, -BOX], dtype=torch.float64)
    assert torch.allclose(E0, sys_.potential_energy(q2), atol=1e-9)


def test_frame_com_pinning(tmp_path):
    sys_ = make_system(tmp_path)
    g = torch.Generator().manual_seed(3)
    v = sys_.maxwell_velocities((5,), g)
    vp = sys_.pin_frame_com(v)
    nf = sys_.n_frame
    mf = sys_.mass[:nf]
    assert float(((vp[:, :nf] * mf[None, :, None]).sum(1) / mf.sum()).abs().max()) < 1e-12
    assert torch.allclose(v[:, nf:], vp[:, nf:])          # guest untouched


def test_two_body_lj_dsf_hand_formula(tmp_path):
    sys_ = make_system(tmp_path, with_guest=False)
    q = torch.zeros(1, 8, 3, dtype=torch.float64)
    q[0, :6] = sys_.pos0_frame[:6] + torch.tensor([0.0, -30.0, 0.0],
                                                  dtype=torch.float64)
    r = 3.4
    q[0, 6] = torch.tensor([2.0, 10.0, 10.0], dtype=torch.float64)
    q[0, 7] = torch.tensor([2.0 + r, 10.0, 10.0], dtype=torch.float64)
    q_ref = q.clone(); q_ref[0, 7, 0] = 2.0 + 5.9
    eps, sig, rc, al = 0.5, 3.0, 6.0, 0.2
    qq = 1389.35457644382 * 0.5 * (-0.5)

    def pair(rr):
        sr6, sr6c = (sig / rr) ** 6, (sig / rc) ** 6
        lj = 4 * eps * (sr6 ** 2 - sr6) - 4 * eps * (sr6c ** 2 - sr6c)
        e_rc = math.erfc(al * rc) / rc
        f_rc = (math.erfc(al * rc) / rc ** 2
                + 2 * al / math.sqrt(math.pi) * math.exp(-(al * rc) ** 2) / rc)
        return lj + qq * (math.erfc(al * rr) / rr - e_rc + f_rc * (rr - rc))
    dE = float(sys_.potential_energy(q) - sys_.potential_energy(q_ref))
    assert abs(dE - (pair(r) - pair(5.9))) < 1e-9


def test_bond_half_k_convention_exact(tmp_path):
    """E_bond = 1/2 k (r-r0)^2, isolated via the guest bond: guest parked where
    every host-guest pair is beyond rc in BOTH configs, COM (hence the tube
    energy) held fixed, guest-guest nonbonded excluded."""
    sys_ = make_system(tmp_path)

    def config(d):
        q = torch.zeros(1, sys_.n_atoms, 3, dtype=torch.float64)
        q[0, :sys_.n_frame] = sys_.pos0_frame
        q[0, sys_.n_frame + 0] = torch.tensor([0.4 - d / 2, 10.0, 10.0],
                                              dtype=torch.float64)
        q[0, sys_.n_frame + 1] = torch.tensor([0.4 + d / 2, 10.0, 10.0],
                                              dtype=torch.float64)
        return q
    dE = float(sys_.potential_energy(config(1.84))
               - sys_.potential_energy(config(1.54)))
    assert abs(dE - 0.5 * 400.0 * 0.3 ** 2) < 1e-9


def test_dihedral_form(tmp_path):
    sys_ = make_system(tmp_path, with_guest=False)
    q = rand_config(sys_, B=4, jitter=0.4, seed=3)
    phi = sys_._dihedral_angle(q, sys_.dihedrals)[:, 0].numpy()
    p = q[:, [0, 1, 3, 4]].numpy()
    for b in range(4):
        b1, b2, b3 = p[b, 1] - p[b, 0], p[b, 2] - p[b, 1], p[b, 3] - p[b, 2]
        n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
        x = np.dot(n1, n2)
        y = np.dot(np.cross(n1, n2), b2 / np.linalg.norm(b2))
        assert abs(math.atan2(y, x) - phi[b]) < 1e-10


def test_wall_energy_exact(tmp_path):
    """Radial flat-bottom tube isolated exactly: the guest is parked where every
    host-guest pair is beyond rc in BOTH configs, and xi is unchanged."""
    sys_ = make_system(tmp_path)

    def config(y):
        q = torch.zeros(1, sys_.n_atoms, 3, dtype=torch.float64)
        q[0, :sys_.n_frame] = sys_.pos0_frame
        q[0, sys_.n_frame + 0] = torch.tensor([0.4 - 0.77, y, 10.0],
                                              dtype=torch.float64)
        q[0, sys_.n_frame + 1] = torch.tensor([0.4 + 0.77, y, 10.0],
                                              dtype=torch.float64)
        return q
    dE = float(sys_.potential_energy(config(15.5)) - sys_.potential_energy(config(10.0)))
    assert abs(dE - 0.5 * 100.0 * (5.5 - 4.0) ** 2) < 1e-9


def test_no_axial_wall(tmp_path):
    """The CV is exactly periodic, so there must be NO axial confinement: the
    energy is unchanged by sliding the guest one full period along the axis."""
    sys_ = make_system(tmp_path)
    q = rand_config(sys_, B=2)
    E0 = sys_.potential_energy(q)
    for k in (1, 2, -3):
        q2 = q.clone()
        q2[:, sys_.n_frame:, 0] += k * PERIOD
        # the synthetic framework is not itself periodic with period 8, so
        # compare only the wall+CV: check phi, and that no wall term appeared
        assert torch.allclose(sys_.cv_value(q), sys_.cv_value(q2), atol=1e-9)
    # a purely axial displacement leaves the radial coordinate alone, so the
    # tube energy cannot change: verify on a guest far from the framework
    def far(x):
        qq = torch.zeros(1, sys_.n_atoms, 3, dtype=torch.float64)
        qq[0, :sys_.n_frame] = sys_.pos0_frame
        qq[0, sys_.n_frame + 0] = torch.tensor([x - 0.77, 10.0, 10.0], dtype=torch.float64)
        qq[0, sys_.n_frame + 1] = torch.tensor([x + 0.77, 10.0, 10.0], dtype=torch.float64)
        return qq
    assert abs(float(sys_.potential_energy(far(0.4))
                     - sys_.potential_energy(far(0.4 + 3 * BOX)))) < 1e-9


# ------------------------------------------------------------- periodic CV
def test_cv_periodicity_and_gradient(tmp_path):
    sys_ = make_system(tmp_path)
    q = rand_config(sys_, B=3)
    phi = sys_.cv_value(q)
    assert float(phi.abs().max()) <= math.pi + 1e-12
    xi = sys_.xi_value(q)
    expect = TWO_PI * xi / PERIOD
    expect = expect - TWO_PI * torch.round(expect / TWO_PI)
    assert torch.allclose(phi, expect, atol=1e-12)
    # exactly periodic in xi
    q2 = q.clone(); q2[:, sys_.n_frame:, 0] += PERIOD
    assert torch.allclose(phi, sys_.cv_value(q2), atol=1e-9)
    # d phi / d q_i = (2 pi / L) w_i n
    h = 1e-6
    for a in range(2):
        qp = q.clone(); qp[:, sys_.n_frame + a, 0] += h
        g = (sys_.cv_value(qp) - phi) / h
        assert torch.allclose(g, torch.full_like(g,
                              TWO_PI / PERIOD * float(sys_.mass_w[a])), atol=1e-6)
    # f_loc = -(F.grad phi)/|grad phi|^2
    F = torch.zeros_like(q); F[:, sys_.n_frame] = torch.tensor([2.0, 1.0, 0.0])
    f_loc, _ = sys_.cv_local_mean_force(q, F)
    kf = TWO_PI / PERIOD
    expect_f = -2.0 * float(sys_.mass_w[0]) * kf / (float((sys_.mass_w ** 2).sum()) * kf ** 2)
    assert torch.allclose(f_loc, torch.full_like(f_loc, expect_f), atol=1e-9)


def test_bias_cartesian_is_the_cv_gradient(tmp_path):
    """The applied bias force must equal bias_gen * grad phi, so that adding it
    is exactly a bias potential -bias_gen*phi to first order."""
    sys_ = make_system(tmp_path)
    q = rand_config(sys_, B=1)
    bg = torch.tensor([[1.7]], dtype=torch.float64)
    Fb = sys_.bias_cartesian(bg, 1, 1)
    assert float(Fb[:, :sys_.n_frame].abs().max()) == 0.0    # framework untouched
    h = 1e-7
    for a in range(2):
        for d in range(3):
            qp = q.clone(); qp[:, sys_.n_frame + a, d] += h
            dphi = float((sys_.cv_value(qp) - sys_.cv_value(q)) / h)
            assert abs(float(Fb[0, sys_.n_frame + a, d]) - 1.7 * dphi) < 1e-6


# --------------------------------------------------------------- integrator
def test_equipartition_and_smoke(tmp_path):
    sys_ = make_system(tmp_path)
    pool = make_pool(tmp_path, sys_, jitter=0.01)
    sim = ZIF8SimConfig(dt=0.0005, gamma=2.0, n_steps=14000, n_replicas=16,
                        save_every=500, abf_warmup_steps=10 ** 9,
                        abf_bias_scale=0.0, estimator_burn_in_steps=100,
                        n_grid=40, rng_seed=11, gate_every=100)
    out = run_sampler("abf", sys_, sim, seeds=[0, 1], init_pool=pool, verbose=False)
    tk = np.asarray(out["temp_kin"])[14:]
    assert abs(tk.mean() - 300.0) / 300.0 < 0.06, f"T_kin {tk.mean():.1f} != 300"
    assert np.isfinite(np.asarray(out["pmf"])).all()
    assert np.asarray(out["gate_mean"]).min() > 0


def test_fr_rate_zero_is_abf(tmp_path):
    sys_ = make_system(tmp_path)
    pool = make_pool(tmp_path, sys_)
    kw = dict(dt=0.0005, gamma=2.0, n_steps=800, n_replicas=8, save_every=200,
              abf_warmup_steps=200, estimator_burn_in_steps=200, n_grid=40,
              rng_seed=5, fr_start_steps=200, fr_every=5, gate_every=50)
    out_a = run_sampler("abf", sys_, ZIF8SimConfig(**kw), seeds=[0, 1],
                        init_pool=pool, verbose=False)
    out_0 = run_sampler("fr_uniform", sys_, ZIF8SimConfig(**kw, fr_rate=0.0),
                        seeds=[0, 1], init_pool=pool, verbose=False)
    assert np.allclose(np.asarray(out_a["pmf"]), np.asarray(out_0["pmf"]), atol=1e-10)
    assert int(np.asarray(out_0["total_replacement_events"]).sum()) == 0


def test_fr_cloning_invariants(tmp_path):
    sys_ = make_system(tmp_path)
    pool = make_pool(tmp_path, sys_)
    sim = ZIF8SimConfig(dt=0.0005, gamma=2.0, n_steps=1500, n_replicas=16,
                        save_every=300, abf_warmup_steps=200,
                        estimator_burn_in_steps=200, n_grid=40, rng_seed=6,
                        fr_start_steps=200, fr_every=5, fr_rate=50.0,
                        max_event_fraction=0.1, gate_every=50)
    out = run_sampler("fr_uniform", sys_, sim, seeds=[0, 1], init_pool=pool,
                      verbose=False)
    assert int(np.asarray(out["total_replacement_events"]).sum()) > 0
    assert np.nanmin(np.asarray(out["ancestor_ess"], dtype=float)) < 16.0
    nuq = np.asarray(out["n_unique_ancestor"])[-1]
    assert (nuq <= 16).all() and (nuq >= 1).all()


def test_no_reference_leakage_guard(tmp_path):
    sys_ = make_system(tmp_path)
    pool = make_pool(tmp_path, sys_)
    sim = ZIF8SimConfig(n_steps=10, n_replicas=4, save_every=10, n_grid=20)
    with pytest.raises(AssertionError):
        run_sampler("fr_uniform", sys_, sim, seeds=[0], init_pool=pool,
                    oracle_free_energy=np.zeros(20), verbose=False)


# ------------------------------------------------------- estimator identities
def test_free_energy_recovers_a_known_periodic_profile():
    """The circular ABF reconstruction must invert a known F'(phi)."""
    G = 144
    grid, dphi = per.periodic_grid(G, dtype=torch.float64)
    F_true = 3.0 * torch.cos(grid) + 1.5 * torch.sin(2 * grid)
    Fp = -3.0 * torch.sin(grid) + 3.0 * torch.cos(2 * grid)
    F_hat = per.free_energy_from_mean_force(Fp[None], grid, dphi)[0]
    F_true = F_true - F_true.mean()
    assert float((F_hat - F_true).abs().max()) < 5e-3


def test_wham_periodic_recovers_a_known_profile():
    """Sample a known periodic F from biased windows analytically and check the
    circular WHAM inversion."""
    rng = np.random.default_rng(0)
    beta = 1.0
    W, M, nb = 24, 40000, 72
    centers = np.linspace(-np.pi, np.pi, W, endpoint=False)
    kappa = 12.0
    F = lambda x: 2.0 * np.cos(x)
    samples = np.zeros((1, W, M))
    xs = np.linspace(-np.pi, np.pi, 4001)[:-1]
    for w, c in enumerate(centers):
        d = xs - c
        d -= 2 * np.pi * np.round(d / (2 * np.pi))
        logp = -beta * (F(xs) + 0.5 * kappa * d ** 2)
        p = np.exp(logp - logp.max()); p /= p.sum()
        samples[0, w] = rng.choice(xs, size=M, p=p)
    mids, Fw, _, _ = wham_periodic(samples, centers, kappa, beta, n_bins=nb)
    ref = F(mids); ref -= ref.min()
    Fw = Fw - Fw.mean() + ref.mean()
    assert float(np.abs(Fw - ref).max()) < 0.12, float(np.abs(Fw - ref).max())


# --------------------------------------------------------------- gate utils
def test_gate_hist_and_js():
    sim = ZIF8SimConfig(gate_lo=0.0, gate_hi=4.0, n_gate_bins=4)
    a = torch.tensor([[0.5, 1.5, 2.5, 3.5], [0.5, 0.6, 3.9, 9.0]], dtype=torch.float64)
    band = torch.tensor([[True, True, False, True], [True, True, True, False]])
    h = gate_hist(a, band, sim, torch.device("cpu"), torch.float64).numpy()
    assert np.allclose(h[0], [1, 1, 0, 1])
    assert np.allclose(h[1], [2, 0, 0, 1])
    p, qd = np.array([1.0, 0, 0, 0]), np.array([0.0, 1, 0, 0])
    assert abs(js_divergence(p, qd) - math.log(2.0)) < 1e-12
    assert js_divergence(p, p) < 1e-12


# ------------------------------------------------- real-artifact gates
REAL_NPZ = os.path.join(ROOT, "cache/zif8/framework.npz")
POOL = os.path.join(ROOT, "cache/zif8/init_pool_T300.npz")


@pytest.mark.skipif(not (torch.cuda.is_available() and os.path.exists(POOL)),
                    reason="needs the GPU and the Stage-0 init pool")
def test_gpu_arms_are_exactly_paired():
    """The campaign's primary endpoint is a PAIRED per-seed difference, so the
    two arms must be bit-identical until FR first intervenes.  On CUDA that is
    not free: torch.compile's default reductions are nondeterministic (measured
    6.1e-05 per force call in f32), which is how the WCA stage lost its
    pairing.  core_zif8 sets the inductor/determinism flags at import; this
    test is what keeps them there."""
    dev = torch.device("cuda")
    kw = dict(dt=0.0005, gamma=1.0, n_steps=1200, n_replicas=32, save_every=200,
              n_grid=48, abf_warmup_steps=100, estimator_burn_in_steps=100,
              fr_start_steps=600, fr_every=5, gate_every=25, rng_seed=5)
    s = ZIF8System(300.0, dev, root=ROOT, chunk=512, force_dtype=torch.float32)
    run = lambda m, **extra: run_sampler(m, s, ZIF8SimConfig(**kw, **extra),
                                         seeds=[0, 1], init_pool=POOL, verbose=False)
    a, a2 = run("abf"), run("abf")
    zero = run("fr_uniform", fr_rate=0.0)
    live = run("fr_uniform", fr_rate=20.0, max_event_fraction=0.1)
    pmf = lambda o: np.asarray(o["pmf"])
    assert np.array_equal(pmf(a), pmf(a2)), "same arm is not reproducible"
    assert np.array_equal(pmf(a), pmf(zero)), "fr_rate=0 does not reproduce ABF"
    pre = np.asarray(live["steps"]) < kw["fr_start_steps"]
    assert np.array_equal(pmf(a)[pre], pmf(live)[pre]), \
        "the arms differ BEFORE the first FR event"
    assert int(np.asarray(live["total_replacement_events"]).sum()) > 0
    assert not np.array_equal(pmf(a)[~pre], pmf(live)[~pre]), \
        "FR fired but changed nothing"


@pytest.mark.skipif(not os.path.exists(REAL_NPZ), reason="framework not built yet")
def test_real_framework_skeleton_is_at_the_force_field_minimum():
    """End-to-end check that the parameters landed on the right atoms: the
    Zn-N-C skeleton of the published crystal structure must already sit at the
    force field's minimum.  The C-H atoms do NOT (X-ray C-H ~0.95 A vs the FF
    r0 1.08 A), which is why minimize() exists."""
    s = ZIF8System(300.0, DEVICE, root=ROOT, with_guest=False, compile=False)
    q = s.pos0_frame[None].clone()
    fn = s.forces(q)[0].norm(dim=-1).numpy()
    types = np.load(REAL_NPZ, allow_pickle=True)["atom_type"]
    # Zn sits in a symmetric tetrahedral N cage, so its residual force is a
    # sharp signature that the Zn-N parameters landed on the right atoms.  It
    # holds at any lattice constant (the artifact scales with C-H, not a).
    assert fn[types == "Zn"].max() < 10.0, "Zn is not at the FF minimum"
    assert fn[np.isin(types, ("H2", "H3"))].min() > 10.0 * fn[types == "Zn"].max(), \
        "expected the X-ray C-H artifact to dominate the Zn residual"
    assert fn[types == "C1"].max() < 0.2 * fn[types == "H2"].max(), \
        "the methyl-bearing ring carbon should carry no C-H artifact"
    d = s._min_image(q[0][:, None, :] - q[0][None, :, :]).norm(dim=-1)
    d.fill_diagonal_(9e9)
    assert float(d.min()) > 0.9, "fused atoms in the framework"


@pytest.mark.skipif(not os.path.exists(REAL_NPZ), reason="framework not built yet")
def test_real_framework_minimization_relaxes_only_the_hydrogens():
    s = ZIF8System(300.0, DEVICE, root=ROOT, with_guest=False, compile=False)
    q0 = s.pos0_frame[None].clone()
    qm, fmax, E = s.minimize(q0, n_steps=3000, f_tol=5.0)
    types = np.load(REAL_NPZ, allow_pickle=True)["atom_type"]
    disp = (qm - q0)[0].norm(dim=-1).numpy()
    heavy = ~np.isin(types, ("H2", "H3"))
    print(f"minimized: fmax {fmax:.2f} kJ/mol/A, E {float(E[0]):.2f}; "
          f"heavy disp max {disp[heavy].max():.3f} A, H disp max {disp[~heavy].max():.3f}")
    assert fmax < 20.0, f"minimization did not converge (fmax {fmax})"
    assert disp[heavy].max() < 0.25, "the heavy-atom framework moved on relaxation"
    assert float(E[0]) < -1176.0, "minimization did not lower the energy"


@pytest.mark.skipif(not os.path.exists(REAL_NPZ), reason="framework not built yet")
def test_real_framework_gate_observable_matches_the_crystal():
    s = ZIF8System(300.0, DEVICE, root=ROOT, with_guest=False, compile=False)
    a_gate, theta = s.gate_observables(s.pos0_frame[None])
    assert abs(float(a_gate[0]) - s.gate_aperture_crystal) < 1e-6
    assert 0.0 <= float(theta[0]) <= 90.0


@pytest.mark.skipif(not os.path.exists(REAL_NPZ), reason="framework not built yet")
def test_real_gate_observable_is_continuous_under_jitter():
    """Opposite gate Zn are exactly a/2 apart in each cartesian component, so
    min-imaging the ring against a RING ATOM sits on the wrap boundary and any
    jitter flips it (A_gate jumped 2.85 -> 5.03 A).  It must be referenced to
    the fixed window centre instead: A_gate has to move CONTINUOUSLY."""
    s = ZIF8System(300.0, DEVICE, root=ROOT, with_guest=False, compile=False)
    g = torch.Generator().manual_seed(0)
    q0 = s.pos0_frame[None]
    for amp in (0.02, 0.05, 0.1, 0.2, 0.3):
        q = q0 + amp * torch.randn(8, *q0.shape[1:], generator=g,
                                   dtype=torch.float64)
        a_gate, theta = s.gate_observables(q)
        assert float((a_gate - s.gate_aperture_crystal).abs().max()) < 5.0 * amp + 0.05, \
            f"A_gate discontinuous at jitter {amp}: {a_gate.numpy()}"
        assert float(theta.max()) <= 90.0 and float(theta.min()) >= 0.0


@pytest.mark.skipif(not os.path.exists(REAL_NPZ), reason="framework not built yet")
def test_real_cv_is_periodic_over_the_channel():
    s = ZIF8System(300.0, DEVICE, root=ROOT, with_guest=True, compile=False)
    # nine axial positions spanning [-L, L] in steps of L/4, so index i and
    # index i+4 differ by EXACTLY one period
    q = torch.zeros(9, s.n_atoms, 3, dtype=torch.float64)
    q[:, :s.n_frame] = s.pos0_frame[None]
    xs = torch.linspace(-s.period, s.period, 9, dtype=torch.float64)
    com = s.center[None, :] + xs[:, None] * s.normal[None, :]
    q[:, s.n_frame + 0] = com - 0.77 * s.normal[None, :]
    q[:, s.n_frame + 1] = com + 0.77 * s.normal[None, :]
    phi = s.cv_value(q)
    expect = TWO_PI * xs / s.period
    assert float(per.circular_distance(phi, expect).abs().max()) < 1e-9
    # The guest is ON the axis so the tube is inactive, and the total energy is
    # periodic under the lattice translation to within the ONE declared
    # symmetry break: the CIF's ordered methyl rotamers are not body-centred
    # (max 0.79 A on H3).  The methyl is a free rotor in this force field, so
    # F(phi) is exactly periodic by symmetry -- only the instantaneous
    # potential is not, and by only ~0.001 kT.
    E = s.potential_energy(q)
    kT = KB * 300.0
    for i in range(5):
        assert abs(float(E[i] - E[i + 4])) < 0.01 * kT, (i, float(E[i]),
                                                         float(E[i + 4]))
    # and the profile is not flat -- there is a real barrier between the cages
    assert float(E.max() - E.min()) > 1.0
