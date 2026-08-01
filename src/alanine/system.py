"""Ace-Ala-Nme (22 atoms, AMBER ff14SB, vacuum): system construction and umbrella seeding.

Two distinct jobs, deliberately separated because conflating them corrupted the first
reference attempt:

  * :func:`build_positions` -- an internal-coordinate (NeRF) builder used ONLY to create the
    single reference minimum.  It rebuilds the whole molecule from scratch at each (phi, psi),
    and for a large part of the torus it places the ACE carbonyl O from the wrong reference
    frame, inverting an sp2 centre.  Restrained minimisation cannot repair an inverted planar
    centre.  Measured on a 24x24 lattice: **356/576 = 61.8 % of windows** ended at a median
    436 kJ/mol with ACE CH3-C-N at 163.7 deg (theta0 116.6) and O-C-N at 79.4 deg (theta0 122.9).
    The strain is nearly (phi,psi)-independent, so it yields a smooth, plausible-looking FES
    whose residual variation is ~8 kT of pure artifact.

  * :func:`seed_umbrella_lattice` -- the ONLY supported seeding path.  It takes one verified,
    minimised L-alanine structure and reaches every window by **rigid rotation about the phi and
    psi bonds**.  Bond lengths, bond angles, improper geometry and chirality are preserved by
    construction, because a rigid dihedral rotation changes nothing except the dihedral.

Every seed must pass :func:`validate_seed` before any dynamics.  This is Stage-0 gate V15.

Units are OpenMM md-units throughout: nm, kJ/mol, ps, amu, radians.
"""
from __future__ import annotations

import math

import numpy as np

# ACE(HH31 CH3 HH32 HH33 C O) ALA(N H CA HA CB HB1 HB2 HB3 C O) NME(N H CH3 HH31 HH32 HH33)
PHI_ATOMS = (4, 6, 8, 14)      # C(ACE)  N  CA  C(ALA)
PSI_ATOMS = (6, 8, 14, 16)     # N  CA  C(ALA)  N(NME)

#: atoms on the distal side of the N(6)-CA(8) bond, i.e. those a phi rotation moves
PHI_MOVING = (9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21)
#: atoms on the distal side of the CA(8)-C(14) bond, i.e. those a psi rotation moves
PSI_MOVING = (15, 16, 17, 18, 19, 20, 21)

FF_FILES = ("amber14/protein.ff14SB.xml",)

KB = 0.008314462618             # kJ/mol/K


# --------------------------------------------------------------------------- geometry helpers
def signed_dihedral_np(x, idx):
    """Signed dihedral (radians, IUPAC: trans at pi) of ``x[(i,j,k,l)]``; ``x`` is ``(...,22,3)``."""
    p0, p1, p2, p3 = (x[..., i, :] for i in idx)
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1n = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
    v = b0 - (b0 * b1n).sum(-1, keepdims=True) * b1n
    w = b2 - (b2 * b1n).sum(-1, keepdims=True) * b1n
    return np.arctan2((np.cross(b1n, v) * w).sum(-1), (v * w).sum(-1))


def chirality(x):
    """Signed volume ``(N-CA) . [(C-CA) x (CB-CA)]``.

    **POSITIVE means L-alanine (S at CA).**  Verified against an unambiguous CIP construction:
    with the lowest-priority substituent HA pointing away, N -> C -> CB must read
    counterclockwise for S.  (An earlier scratch helper used the complementary triple product
    and documented the sign backwards; that convention is not used here.)
    """
    CA, C, CB, N = 8, 14, 10, 6
    uN = x[..., N, :] - x[..., CA, :]
    uC = x[..., C, :] - x[..., CA, :]
    uB = x[..., CB, :] - x[..., CA, :]
    return (uN * np.cross(uC, uB)).sum(-1)


def rotate_about_bond(x, i, j, moving, angle):
    """Rotate ``moving`` atoms about the axis ``x[j]-x[i]`` by ``angle`` (radians).

    ``x`` is ``(B, 22, 3)``, ``angle`` is ``(B,)``.  Rodrigues rotation about the bond axis
    anchored at atom ``j``.  Rigid: every internal coordinate except the dihedral is invariant.
    """
    ax = x[:, j] - x[:, i]
    ax = ax / np.linalg.norm(ax, axis=-1, keepdims=True)
    o = x[:, j][:, None, :]
    p = x[:, list(moving)] - o
    c = np.cos(angle)[:, None, None]
    s = np.sin(angle)[:, None, None]
    k = ax[:, None, :]
    rot = p * c + np.cross(np.broadcast_to(k, p.shape), p) * s + k * (k * p).sum(-1, keepdims=True) * (1 - c)
    y = x.copy()
    y[:, list(moving)] = rot + o
    return y


# --------------------------------------------------------------------------- OpenMM system
def topology():
    import openmm.app as app
    from ._ala22_src import NAMES, BONDS
    top = app.Topology()
    ch = top.addChain()
    E = app.element
    elmap = {"H": "hydrogen", "C": "carbon", "N": "nitrogen", "O": "oxygen"}
    res, atoms = {}, []
    for rn, an, el in NAMES:
        if rn not in res:
            res[rn] = top.addResidue(rn, ch)
        atoms.append(top.addAtom(an, getattr(E, elmap[el]), res[rn]))
    for i, j in BONDS:
        top.addBond(atoms[i], atoms[j])
    return top


def make_system(ff_files=FF_FILES, constraints=None, hydrogen_mass=None):
    """Build the vacuum ff14SB system.  ``hydrogen_mass`` in amu enables HMR (None = off)."""
    import openmm.app as app
    import openmm.unit as u
    top = topology()
    ff = app.ForceField(*ff_files)
    kw = dict(nonbondedMethod=app.NoCutoff, constraints=constraints,
              rigidWater=False, removeCMMotion=False)
    if hydrogen_mass is not None:
        kw["hydrogenMass"] = hydrogen_mass * u.dalton
    return ff, top, ff.createSystem(top, **kw)


def build_positions(phi_deg=-80.0, psi_deg=80.0, cb_offset=-120.0):
    """NeRF builder -- **reference-minimum construction only, never umbrella seeding.**

    See the module docstring: this places the ACE carbonyl O from the wrong reference frame over
    a large part of the torus.  :func:`seed_umbrella_lattice` must be used instead.
    """
    from ._ala22_src import build_positions as _bp
    return _bp(phi_deg, psi_deg, cb_offset) * 0.1        # Angstrom -> nm


def reference_minimum(hydrogen_mass=None, max_iterations=10000):
    """Return ``(system, X0_nm)`` for the minimised L-alanine C7eq structure.

    This is the single structure every umbrella window is rigidly rotated from.
    """
    import openmm as mm
    import openmm.unit as u
    ff, top, system = make_system(hydrogen_mass=hydrogen_mass)
    ctx = mm.Context(system, mm.VerletIntegrator(0.001 * u.picoseconds),
                     mm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(build_positions(-80.0, 80.0))
    mm.LocalEnergyMinimizer.minimize(ctx, maxIterations=max_iterations)
    X0 = ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(u.nanometer)
    return system, np.asarray(X0)


def seed_umbrella_lattice(X0, centers):
    """Rigidly rotate one minimised structure ``X0 (22,3)`` to each ``centers (K,2)`` in radians.

    Returns ``(K, 22, 3)``.  phi is set first (rotating about N-CA), then psi is re-measured and
    set (rotating about CA-C), because the phi rotation also moves the psi atoms.
    """
    x = np.repeat(np.asarray(X0, dtype=float)[None], len(centers), axis=0)
    c = np.asarray(centers, dtype=float)
    x = rotate_about_bond(x, 6, 8, PHI_MOVING, c[:, 0] - signed_dihedral_np(x, PHI_ATOMS))
    x = rotate_about_bond(x, 8, 14, PSI_MOVING, c[:, 1] - signed_dihedral_np(x, PSI_ATOMS))
    return x


# --------------------------------------------------------------------------- Stage-0 gate V15
def angle_report(system, x):
    """Per-configuration ``HarmonicAngleForce`` energy (kJ/mol) and max angle deviation (deg).

    Evaluated directly from the extracted parameters, so no OpenMM Context is needed per frame.
    """
    from .forcefield import extract_parameters
    P = extract_parameters(system)
    idx, theta0, k = P["angles"]
    x = np.atleast_3d(np.asarray(x, dtype=float)).reshape(-1, x.shape[-2], 3)
    v1 = x[:, idx[:, 0]] - x[:, idx[:, 1]]
    v2 = x[:, idx[:, 2]] - x[:, idx[:, 1]]
    cs = (v1 * v2).sum(-1)
    sn = np.linalg.norm(np.cross(v1, v2), axis=-1)
    th = np.arctan2(sn, cs)
    dev = np.degrees(np.abs(th - theta0[None]))
    energy = (0.5 * k[None] * (th - theta0[None]) ** 2).sum(-1)
    return energy, dev.max(-1)


def relax_seeds(tff, x, centers, kappa, n_steps=800, lr=2.0e-5, max_disp=0.004):
    """Restrained steepest descent to relieve the STERIC clashes a rigid rotation creates.

    A rigid dihedral rotation preserves every bond and angle, but it can still drive non-bonded
    atoms into each other: measured over a 24x24 lattice, **18.6 % of rigidly rotated seeds sit
    above E_min + 200 kJ/mol**, peaking at 2.3e6 kJ/mol with forces to 4.7e8 kJ/mol/nm, while
    their bond angles and chirality are perfect.  Starting BAOAB from those explodes
    (kinetic temperature ~1e27 K) *without ever producing a NaN*, so a finiteness check alone
    does not catch it -- which is why :func:`validate_seed` also gates on total energy and force.

    Minimisation runs under the umbrella restraint so ``(phi, psi)`` stays at the window centre,
    and each step's displacement is capped so a huge initial force cannot throw the structure.

    ``tff`` is a :class:`alanine.forcefield.TorchFF`; ``x`` is ``(K, 22, 3)`` in nm.
    """
    import torch as _t
    from .reference import restraint_energy

    def _dihedral(y, idx):
        p0, p1, p2, p3 = (y[:, i] for i in idx)
        b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
        b1n = b1 / b1.norm(dim=-1, keepdim=True)
        v = b0 - (b0 * b1n).sum(-1, keepdim=True) * b1n
        w = b2 - (b2 * b1n).sum(-1, keepdim=True) * b1n
        return _t.atan2((_t.linalg.cross(b1n, v, dim=-1) * w).sum(-1), (v * w).sum(-1))

    y = _t.as_tensor(x).clone()
    c = _t.as_tensor(centers, device=y.device, dtype=y.dtype)
    for _ in range(int(n_steps)):
        yg = y.detach().requires_grad_(True)
        E = tff.energy(yg) + restraint_energy(_dihedral(yg, PHI_ATOMS),
                                              _dihedral(yg, PSI_ATOMS), c, kappa)
        g, = _t.autograd.grad(E.sum(), yg)
        d = -lr * g
        nrm = d.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        y = (y + d * _t.clamp(max_disp / nrm, max=1.0)).detach()
    return y


def validate_seed(system, x, centers, max_angle_energy=50.0, max_angle_dev_deg=15.0,
                  cv_tol_deg=1.0, energy=None, max_energy_above_min=200.0,
                  force_max=None, max_force=1.0e5):
    """Stage-0 gate V15 for umbrella seeds.  Returns ``(ok (K,), report dict)``.

    A seed passes iff: L chirality; harmonic-angle energy below ``max_angle_energy`` kJ/mol;
    max bond-angle deviation below ``max_angle_dev_deg``; and the recovered (phi, psi) is within
    ``cv_tol_deg`` of the requested centre.
    """
    x = np.asarray(x, dtype=float)
    aenergy, dev = angle_report(system, x)
    chir = chirality(x)
    phi = signed_dihedral_np(x, PHI_ATOMS)
    psi = signed_dihedral_np(x, PSI_ATOMS)
    c = np.asarray(centers, dtype=float)

    def wrap(a):
        return np.degrees(np.abs((a + np.pi) % (2 * np.pi) - np.pi))

    dphi = wrap(phi - c[:, 0])
    dpsi = wrap(psi - c[:, 1])
    ok = ((chir > 0) & (aenergy < max_angle_energy) & (dev < max_angle_dev_deg)
          & (dphi < cv_tol_deg) & (dpsi < cv_tol_deg))
    rep = dict(angle_energy=aenergy, max_angle_dev_deg=dev, chirality=chir,
               dphi_deg=dphi, dpsi_deg=dpsi,
               n_fail_chirality=int((chir <= 0).sum()),
               n_fail_angle_energy=int((aenergy >= max_angle_energy).sum()),
               n_fail_angle_dev=int((dev >= max_angle_dev_deg).sum()),
               n_fail_cv=int(((dphi >= cv_tol_deg) | (dpsi >= cv_tol_deg)).sum()))

    # Steric gate.  Bond angles and chirality can be perfect while non-bonded atoms overlap;
    # such a seed is finite but explodes on the first BAOAB step, so finiteness is not enough.
    if energy is not None:
        e = np.asarray(energy, dtype=float)
        finite_e = np.isfinite(e)
        thresh = (e[finite_e].min() + max_energy_above_min) if finite_e.any() else np.inf
        ok = ok & finite_e & (e <= thresh)
        rep["total_energy"] = e
        rep["energy_threshold"] = float(thresh)
        rep["n_fail_total_energy"] = int((~(finite_e & (e <= thresh))).sum())
    if force_max is not None:
        fm = np.asarray(force_max, dtype=float)
        ok = ok & np.isfinite(fm) & (fm <= max_force)
        rep["force_max"] = fm
        rep["n_fail_force"] = int((~(np.isfinite(fm) & (fm <= max_force))).sum())
    return ok, rep


def window_centers(n_per_axis):
    """Cell-centred ``(n^2, 2)`` lattice of umbrella centres on ``T^2`` (radians)."""
    c = -math.pi + (np.arange(n_per_axis) + 0.5) * (2 * math.pi / n_per_axis)
    A, B = np.meshgrid(c, c, indexing="ij")
    return np.stack([A.ravel(), B.ravel()], -1)
