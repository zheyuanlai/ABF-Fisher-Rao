"""Stage 0 engine-equivalence gate for the C60/TIP4P-Ew system -- SPEC_c60_water.md §3.1.

Mirrors ``test_nacl_engine.py`` deliberately: same 1e-6 tolerance, float64, the same OpenMM
**Reference**-platform oracle (measured 0.3 s/eval at this box -- affordable, deterministic,
and it cannot collide with a torch CUDA runtime in the same process, the measured NaCl
deadlock), and the same "if any test here fails, C60 does not run" stakes.  One deliberate
difference: the configuration pool implements the SPEC's ">= 16 configurations including
distinct solvent structures at the same separation" clause -- the clause the NaCl pool
under-implemented -- via independent perturbation draws at the same ``d``.

Configs are built from the frozen-box NPT configuration by re-placing the cages at each test
separation and pushing any clashing water radially off the nearest carbon (parity does not
need thermal configurations, only non-singular ones; both engines see identical coordinates).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from c60 import geometry, system as csys  # noqa: E402
from c60.nonbonded import C60Nonbonded  # noqa: E402

torch.set_default_dtype(torch.float64)

TOL = 1.0e-6                    # SPEC §3.1 == methane §3.2 == NaCl §3.1
FROZEN = os.path.join(os.path.dirname(__file__), "..", "results", "c60", "box",
                      "frozen_box.npz")

# NOTE: no separation may equal the 1.0 nm cutoff -- cage B is a pure translate of cage A,
# so at d = 1.00 all 60 equivalent-carbon pairs sit at r = 1.0 EXACTLY and per-engine float
# rounding of the inclusion test moves 60 x (LJ at cutoff) ~ 0.06 kJ between the engines
# (measured: one config at 1.09e-6 relative, nudge-immune, while all others sat at 1e-13).
SEPARATIONS = (0.908, 0.95, 0.968, 1.005, 1.05, 1.10, 1.20, 1.30,
               1.50, 1.70, 2.00, 2.20, 2.428)


def _oracle_platform():
    import openmm as mm
    return mm.Platform.getPlatformByName("Reference"), {}


@pytest.fixture(scope="module")
def built():
    import openmm as mm
    import openmm.unit as u

    if not os.path.exists(FROZEN):
        pytest.skip("frozen_box.npz missing -- run scripts/c60_npt_box.py first")
    fz = np.load(FROZEN)
    lx, lz = float(fz["lx_nm"]), float(fz["lz_nm"])
    base = np.asarray(fz["positions"], dtype=np.float64)

    mod = csys.build_modeller()          # deterministic (seeded); topology only
    box = [mm.Vec3(lx, 0, 0), mm.Vec3(0, lx, 0), mm.Vec3(0, 0, lz)] * u.nanometer
    system = csys.build_system(mod.topology, box_vectors=box,
                               pme_params=csys.pme_params())
    p = csys.site_parameters(system, mod.topology)
    base = csys.apply_constraints(system, mod.topology, base)

    platform, props = _oracle_platform()
    integ = mm.VerletIntegrator(1e-6)
    ctx = mm.Context(system, integ, platform, props)
    alpha, nx, ny, nz = csys.pme_params()

    engine = C60Nonbonded(system, mod.topology, (lx, lx, lz), alpha, (nx, ny, nz),
                          device="cpu", dtype=torch.float64)
    return dict(system=system, topology=mod.topology, ctx=ctx, params=p, base=base,
                lx=lx, lz=lz, engine=engine, oracle=platform.getName())


def _place(base, params, lx, lz, d, push_clashes=True):
    """Cages at separation ``d`` in the frozen box; clashing waters pushed off the carbons."""
    pos = base.copy()
    center = np.array([0.5 * lx, 0.5 * lx, 0.5 * lz])
    pos[params["carbon_index"]] = geometry.pair_positions(d, center)
    if not push_clashes:
        return pos
    carb = pos[params["carbon_index"]]
    for (o, h1, h2, m) in params["waters"]:
        dvec = pos[o] - carb
        L = np.array([lx, lx, lz])
        dvec -= L * np.round(dvec / L)
        r = np.linalg.norm(dvec, axis=1)
        k = int(np.argmin(r))
        if r[k] < 0.30:
            shift = (0.32 - r[k]) * dvec[k] / max(r[k], 1e-6)
            for s in (o, h1, h2, m):
                pos[s] += shift
    return pos


def _minimize(built, pos, d, max_iter=80):
    """Brief capped minimisation: removes pusher-created water overlaps whose ~1e9 kJ/mol
    LJ spikes make the total energy's float64 ulp as large as a finite-difference increment
    (measured: quantised FD slopes 76.29/83.92 at h = 2.5e-7).  Massless cages stay fixed
    (asserted); constraints and virtual sites are the oracle's own.

    The post-minimisation **nudge** is load-bearing: with an unswitched cutoff the minimiser
    can park a pair exactly on the r = 1.0 nm energy discontinuity, where float rounding
    decides inclusion differently per engine (measured: one config at 1.09e-6 relative energy
    while every other sat at ~1e-13).  A 2e-4 nm seeded perturbation, re-projected onto the
    constraint manifold, breaks the degeneracy; both engines still see identical coordinates."""
    import openmm as mm
    import openmm.unit as u
    ctx = built["ctx"]
    ctx.setPositions(pos * u.nanometer)
    ctx.applyConstraints(1e-10)
    ctx.computeVirtualSites()
    mm.LocalEnergyMinimizer.minimize(ctx, 10.0, max_iter)
    out = np.asarray(ctx.getState(getPositions=True).getPositions()
                     .value_in_unit(u.nanometer))
    rng = np.random.default_rng(int(round(d * 1e6)))
    noise = rng.normal(0.0, 2e-4, out.shape)
    noise[built["params"]["carbon_index"]] = 0.0
    ctx.setPositions((out + noise) * u.nanometer)
    ctx.applyConstraints(1e-10)
    ctx.computeVirtualSites()
    out = np.asarray(ctx.getState(getPositions=True).getPositions()
                     .value_in_unit(u.nanometer))
    assert abs(csys.xi_of(out, built["params"]["cage_a"], built["params"]["cage_b"]) - d) < 1e-9
    return out


_CONFIG_CACHE = {}
_CONFIG_DISK = os.path.join(os.path.dirname(FROZEN), "parity_test_configs.npz")


def _configs(built):
    """>= 16 configurations spanning the separation range and the energy range.

    Minimisation costs ~80 s/config on the Reference oracle, so the minimised pool is cached
    on disk keyed to the frozen box; a box change invalidates it.  The cache holds *inputs*
    to the parity comparison, never results.
    """
    if "configs" in _CONFIG_CACHE:
        return _CONFIG_CACHE["configs"]
    key = np.array([built["lx"], built["lz"], float(np.abs(built["base"]).sum()),
                    float(np.sum(SEPARATIONS))])
    if os.path.exists(_CONFIG_DISK):
        dz = np.load(_CONFIG_DISK)
        k_old = np.asarray(dz["key"])
        if k_old.shape == key.shape and np.allclose(k_old, key, rtol=0, atol=1e-9):
            out = [dz[f"c{k}"] for k in range(int(dz["n"]))]
            _CONFIG_CACHE["configs"] = out
            return out
    p, lx, lz = built["params"], built["lx"], built["lz"]
    rng = np.random.default_rng(20260814)
    out = []
    for d in SEPARATIONS:
        out.append(_minimize(built, _place(built["base"], p, lx, lz, d), d))
    # distinct solvent structures at the SAME separation (the SPEC clause)
    for scale in (0.004, 0.008):
        q = _place(built["base"], p, lx, lz, 1.005)
        noise = rng.normal(0.0, scale, q.shape)
        noise[p["carbon_index"]] = 0.0
        out.append(_minimize(built, q + noise, 1.005))
    q = _place(built["base"], p, lx, lz, 0.968)
    noise = rng.normal(0.0, 0.006, q.shape)
    noise[p["carbon_index"]] = 0.0
    out.append(_minimize(built, q + noise, 0.968))
    assert len(out) >= 16
    _CONFIG_CACHE["configs"] = out
    np.savez(_CONFIG_DISK, key=key, n=len(out),
             **{f"c{k}": c for k, c in enumerate(out)})
    return out


def _openmm_ef(built, pos):
    import openmm.unit as u
    ctx = built["ctx"]
    ctx.setPositions(pos * u.nanometer)
    ctx.computeVirtualSites()
    st = ctx.getState(getEnergy=True, getForces=True)
    e = st.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
    f = np.asarray(st.getForces().value_in_unit(u.kilojoule_per_mole / u.nanometer))
    return float(e), f


def _torch_ef(built, pos):
    # OpenMM's State force convention, measured per-row on the Reference platform
    # (2026-08-14): parent atoms carry REDISTRIBUTED forces (O row gains exactly w_O * F_M to
    # all printed digits), while the virtual-site row keeps the RAW M force rather than zero.
    # The comparison array is therefore redistribute(f_raw) with the M rows restored to raw.
    # ``redistribute`` alone (M row zeroed) is the *dynamics* convention -- massless sites get
    # no kick -- and its correctness is covered by the finite-difference gradient test.
    x = torch.as_tensor(pos)[None]
    built["engine"].compute_vsites(x)
    e, f_raw = built["engine"].energy_forces(x)
    f_red = built["engine"].redistribute(f_raw)
    m = built["engine"].waters[:, 3]
    f_cmp = f_red.clone()
    f_cmp[:, m, :] = f_raw[:, m, :]
    return float(e[0]), f_cmp[0].numpy(), f_red[0].numpy()


# --------------------------------------------------------------------------- tests
def test_frozen_params_match_openmm(built):
    p = built["params"]
    assert len(p["carbon_index"]) == 120
    assert p["waters"].shape == (csys.N_WATERS, 4)
    # site_parameters already asserts every frozen constant at 1e-12; reaching here is the pass


def test_lj_and_charge_sets_are_disjoint(built):
    p = built["params"]
    lj = p["epsilon"] > 0
    q = p["charge"] != 0
    assert not np.any(lj & q), "TIP4P-Ew structure violated: a site carries both LJ and charge"
    assert int(lj.sum()) == 120 + csys.N_WATERS
    assert int(q.sum()) == 3 * csys.N_WATERS


def test_cage_geometry_in_box(built):
    pos = built["base"]
    ca = pos[built["params"]["cage_a"]]
    cb = pos[built["params"]["cage_b"]]
    geometry.validate_cage(ca - ca.mean(0))
    geometry.validate_cage(cb - cb.mean(0))
    assert abs(geometry.facing_pentagon_registry(ca - ca.mean(0), cb - cb.mean(0)) - 36.0) < 1e-6


def test_vsite_placement_matches_openmm(built):
    import openmm.unit as u
    pos = _place(built["base"], built["params"], built["lx"], built["lz"], 1.20)
    ctx = built["ctx"]
    ctx.setPositions(pos * u.nanometer)
    ctx.computeVirtualSites()
    ref = np.asarray(ctx.getState(getPositions=True).getPositions()
                     .value_in_unit(u.nanometer))
    x = torch.as_tensor(pos)[None]
    built["engine"].compute_vsites(x)
    m = built["params"]["waters"][:, 3]
    assert np.abs(x[0].numpy()[m] - ref[m]).max() < 1e-12


def test_full_model_parity_over_an_energy_range(built):
    configs = _configs(built)
    assert len(configs) >= 16
    energies, rel_e, rel_f = [], [], []
    for pos in configs:
        e_mm, f_mm = _openmm_ef(built, pos)
        e_t, f_t, _ = _torch_ef(built, pos)
        energies.append(e_mm)
        rel_e.append(abs(e_t - e_mm) / max(1.0, abs(e_mm)))
        rel_f.append(np.abs(f_t - f_mm).max() / max(1.0, np.abs(f_mm).max()))
    assert max(energies) - min(energies) > 1e2, "configs do not span an energy range"
    assert max(rel_e) < TOL, f"energy parity fails: {max(rel_e):.3e}"
    assert max(rel_f) < TOL, f"force parity fails: {max(rel_f):.3e}"


def test_lj_only_parity(built):
    import openmm as mm
    import openmm.unit as u
    system2 = csys.build_system(built["topology"],
                                box_vectors=built["ctx"].getState()
                                .getPeriodicBoxVectors(),
                                pme_params=csys.pme_params())
    nbf = next(f for f in system2.getForces() if isinstance(f, mm.NonbondedForce))
    for i in range(nbf.getNumParticles()):
        _, sig, eps = nbf.getParticleParameters(i)
        nbf.setParticleParameters(i, 0.0, sig, eps)
    platform, props = _oracle_platform()
    ctx2 = mm.Context(system2, mm.VerletIntegrator(1e-6), platform, props)

    pos = _minimize(built, _place(built["base"], built["params"], built["lx"],
                                  built["lz"], 0.968), 0.968)
    ctx2.setPositions(pos * u.nanometer)
    ctx2.computeVirtualSites()
    st = ctx2.getState(getEnergy=True, getForces=True)
    e_mm = st.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
    f_mm = np.asarray(st.getForces().value_in_unit(u.kilojoule_per_mole / u.nanometer))
    del ctx2

    x = torch.as_tensor(pos)[None]
    # LJ-only: zero the Coulomb pair table so pair.energy_forces returns LJ alone
    eng = built["engine"]
    eng.compute_vsites(x)
    qq_save = eng.pair.q_qq.clone()
    eng.pair.q_qq.zero_()
    e_t, f_t = eng.pair.energy_forces(x)
    eng.pair.q_qq.copy_(qq_save)
    rel_e = abs(float(e_t[0]) - e_mm) / max(1.0, abs(e_mm))
    rel_f = np.abs(f_t[0].numpy() - f_mm).max() / max(1.0, np.abs(f_mm).max())
    assert rel_e < TOL and rel_f < TOL, f"LJ-only parity: e {rel_e:.3e}, f {rel_f:.3e}"


def test_electrostatics_only_parity(built):
    import openmm as mm
    import openmm.unit as u
    system2 = csys.build_system(built["topology"],
                                box_vectors=built["ctx"].getState()
                                .getPeriodicBoxVectors(),
                                pme_params=csys.pme_params())
    nbf = next(f for f in system2.getForces() if isinstance(f, mm.NonbondedForce))
    for i in range(nbf.getNumParticles()):
        q, sig, _ = nbf.getParticleParameters(i)
        nbf.setParticleParameters(i, q, sig, 0.0)
    platform, props = _oracle_platform()
    ctx2 = mm.Context(system2, mm.VerletIntegrator(1e-6), platform, props)

    pos = _minimize(built, _place(built["base"], built["params"], built["lx"],
                                  built["lz"], 1.30), 1.30)
    ctx2.setPositions(pos * u.nanometer)
    ctx2.computeVirtualSites()
    st = ctx2.getState(getEnergy=True, getForces=True)
    e_mm = st.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
    f_mm = np.asarray(st.getForces().value_in_unit(u.kilojoule_per_mole / u.nanometer))
    del ctx2

    eng = built["engine"]
    x = torch.as_tensor(pos)[None]
    eng.compute_vsites(x)
    sig_save = eng.pair.lj_eps.clone()
    eng.pair.lj_eps.zero_()
    e_r, f_r = eng.pair.energy_forces(x)
    eng.pair.lj_eps.copy_(sig_save)
    e_x, f_x = eng.pair.exclusion_correction(x)
    e_k, f_k = eng.recip.energy_forces(x)
    e_t = float((e_r + e_x + e_k + eng.e_self)[0])
    f_raw = f_r + f_x + f_k
    f_cmp = eng.redistribute(f_raw)
    m = eng.waters[:, 3]
    f_cmp[:, m, :] = f_raw[:, m, :]          # the OpenMM State convention (see _torch_ef)
    f_t = f_cmp[0].numpy()

    assert abs(e_x[0]) > 0.1 * abs(e_mm) or abs(e_mm) < 1.0, \
        "exclusion correction should be load-bearing"
    rel_e = abs(e_t - e_mm) / max(1.0, abs(e_mm))
    rel_f = np.abs(f_t - f_mm).max() / max(1.0, np.abs(f_mm).max())
    assert rel_e < TOL and rel_f < TOL, f"elec-only parity: e {rel_e:.3e}, f {rel_f:.3e}"


def test_batched_equals_single(built):
    p = built["params"]
    q1 = _place(built["base"], p, built["lx"], built["lz"], 1.005)
    q2 = _place(built["base"], p, built["lx"], built["lz"], 1.70)
    xb = torch.stack([torch.as_tensor(q1), torch.as_tensor(q2)])
    built["engine"].compute_vsites(xb)
    eb, fb = built["engine"].energy_forces(xb)
    for k, q in enumerate((q1, q2)):
        x = torch.as_tensor(q)[None]
        built["engine"].compute_vsites(x)
        e, f = built["engine"].energy_forces(x)
        assert abs(float(eb[k]) - float(e[0])) / max(1.0, abs(float(e[0]))) < 1e-9
        assert float((fb[k] - f[0]).abs().max()) < 1e-9 * max(1.0, float(f[0].abs().max()))


def test_forces_are_the_gradient_of_the_energy(built):
    pos = _minimize(built, _place(built["base"], built["params"], built["lx"],
                                  built["lz"], 1.10), 1.10)
    x = torch.as_tensor(pos)[None]
    built["engine"].compute_vsites(x)
    e0, f_raw = built["engine"].energy_forces(x)
    h = 1e-6
    rng = np.random.default_rng(7)
    # displace massive sites only (O/H); M is a dependent coordinate
    o_and_h = built["params"]["waters"][:, :3].reshape(-1)
    for idx in rng.choice(o_and_h, size=5, replace=False):
        for ax in (0, 2):
            xp = x.clone(); xp[0, idx, ax] += h
            built["engine"].compute_vsites(xp)
            xm = x.clone(); xm[0, idx, ax] -= h
            built["engine"].compute_vsites(xm)
            ep, _ = built["engine"].energy_forces(xp)
            em, _ = built["engine"].energy_forces(xm)
            fd = -(float(ep[0]) - float(em[0])) / (2 * h)
            f_red = built["engine"].redistribute(f_raw)
            f_an = float(f_red[0, idx, ax])
            assert abs(fd - f_an) < 5e-3 * max(1.0, abs(f_an)), \
                f"site {idx} axis {ax}: fd {fd:.6f} vs analytic {f_an:.6f}"


def test_xi_and_local_mean_force(built):
    """xi exact; f = (1/2)(F_Az - F_Bz) from OpenMM forces == torch; f == -dV/dxi by FD."""
    p = built["params"]
    d = 1.05
    pos = _minimize(built, _place(built["base"], p, built["lx"], built["lz"], d), d)
    x = torch.as_tensor(pos)[None]
    assert abs(float(built["engine"].xi(x)[0]) - d) < 1e-12

    e_mm, f_mm = _openmm_ef(built, pos)
    _, f_t, f_raw = _torch_ef(built, pos)
    f_est_mm = csys.local_mean_force(f_mm, p["cage_a"], p["cage_b"])
    f_est_t = float(built["engine"].local_mean_force(torch.as_tensor(f_t)[None])[0])
    assert abs(f_est_t - float(f_est_mm)) / max(1.0, abs(f_est_mm)) < TOL

    # finite-difference dV/dxi: cages displaced symmetrically +-h/2 on the SAME water
    # configuration (re-running the clash-pusher at d +- h could move a boundary water and
    # corrupt the difference)
    h = 2e-6
    pp = pos.copy(); pp[p["cage_a"], 2] -= 0.5 * h; pp[p["cage_b"], 2] += 0.5 * h
    pm = pos.copy(); pm[p["cage_a"], 2] += 0.5 * h; pm[p["cage_b"], 2] -= 0.5 * h
    ep = _openmm_ef(built, pp)[0]
    em = _openmm_ef(built, pm)[0]
    dV_dxi = (ep - em) / (2 * h)
    # f is the negative generalised force: F'(xi) sample = +dV/dxi
    assert abs(f_est_t - dV_dxi) < 5e-6 * max(1.0, abs(dV_dxi)), \
        f"estimator {f_est_t:.6f} vs FD dV/dxi {dV_dxi:.6f}"


def test_direct_cage_cage_energy_at_contact(built):
    """SPEC §5 reproduction clause (iii), definition fixed by Amendment 16.6: the
    **untruncated** vacuum C60-C60 LJ at the direct term's own minimum over d, vs the paper's
    -18.5 kJ/mol.  (The originally written 1.0 nm-truncated value at 0.968 nm reads -17.02 and
    tested the cage-geometry choice, not the §1.1 parameter derivation; measured 2026-08-14.)"""
    cage = geometry.c60_cage()
    sig, eps = csys.SIGMA_C_NM, csys.EPSILON_C_KJ

    def direct(d):
        a = cage + np.array([0, 0, -0.5 * d])
        b = cage + np.array([0, 0, +0.5 * d])
        r = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1).ravel()
        sr6 = (sig / r) ** 6
        return float((4 * eps * (sr6 ** 2 - sr6)).sum())

    ds = np.linspace(0.90, 1.05, 151)
    es = [direct(d) for d in ds]
    e_min = min(es)
    assert abs(e_min - (-18.5)) < 0.5, \
        f"direct cage-cage minimum {e_min:.2f} kJ/mol vs paper -18.5"
