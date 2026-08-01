"""Ace-L-Val-Nme (28 atoms, AMBER ff14SB, vacuum): system construction and seeding.

Physical model is inherited **verbatim** from the accepted alanine study
(`ALANINE_EXECUTION_DECISION.md` sec.4), so that ``Ala -> Val`` is the only change:

    vacuum, AMBER ff14SB, NO constraints, NO HMR
    BAOAB, dt = 1 fs, gamma = 1 ps^-1, T = 300 K, float64

"No constraints" is load-bearing and is *not* the same as "the same hydrogen-bond
constraints": :func:`alanine.forcefield.extract_parameters` raises if the system carries any
constraint, because the torch integrator implements no SHAKE.  dt = 1 fs is therefore
mandatory, not a choice.

Atom order follows the amber14 residue templates, ACE then VAL then NME::

    ACE(HH31 CH3 HH32 HH33 C O)
    VAL(N H CA HA CB HB CG1 HG11 HG12 HG13 CG2 HG21 HG22 HG23 C O)
    NME(N H CH3 HH31 HH32 HH33)

Backbone indices 4, 6, 8 coincide with alanine's; only the carbonyl C and the NME N move
(14 -> 20, 16 -> 22) to make room for the valine side chain.

Three dihedrals, all IUPAC-signed::

    phi  = (4, 6, 8, 20)    C(ACE)  N   CA  C(VAL)
    psi  = (6, 8, 20, 22)   N       CA  C   N(NME)
    chi1 = (6, 8, 10, 12)   N       CA  CB  CG1

**The three rigid rotations are exactly independent.**  Rotating about N-CA, CA-C or CA-CB
leaves the other two dihedrals invariant to machine precision (measured max drift 3e-15 deg,
max bond-length change 9e-16 A), because each rotation axis contains two of the four atoms
defining every other dihedral, and a dihedral is invariant under a rigid motion of its whole
4-tuple.  A 3-D seed lattice is therefore exact, not approximate.

Seeding is a three-step path, and every step is needed:

  1. :func:`build_positions` -- NeRF build.  **Reference construction only, never a lattice.**
     It inherits the alanine builder's defect: the ACE carbonyl O is placed from a reference
     frame that leaves an ACE HH32...O contact at 1.113 A and a raw energy near 6e4 kJ/mol.
     Measured identically for Ala (59890 kJ/mol) and Val (59799 kJ/mol), so this is inherited,
     not a valine bug.
  2. :func:`relieve_methyl_rotors` -- exhaustive coarse scan of the four methyl phases.
     Cheap, deterministic, and removes the rotor-addressable part of the clash.
  3. :func:`restrained_minimise` -- harmonic restraints on (phi, psi, chi1) while the rest
     relaxes.  This is the step that matters.  *Unrestrained* minimisation from the same
     builds drags the backbone across basins -- two of three rotamers started at (-80, 80)
     and finished near (+56, -38) -- which would have silently destroyed the 3-D seed lattice.

Every seed must pass :func:`validate_seed` before any dynamics.  Measured on the 12-seed
screening lattice: all 12 reach their targets within 0.6 deg, retain L chirality, keep both
peptide bonds trans, hold every sp2 centre planar to within 1.2 deg out-of-plane, and leave no
non-excluded contact below 2.05 A.

Units are OpenMM md-units throughout: nm, kJ/mol, ps, amu, radians -- except
:func:`build_positions`, which works in angstrom and is converted at its single call site.
"""
from __future__ import annotations

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as u

# --------------------------------------------------------------------------- topology tables
NAMES = [
    ('ACE', 'HH31', 'H'), ('ACE', 'CH3', 'C'), ('ACE', 'HH32', 'H'), ('ACE', 'HH33', 'H'),
    ('ACE', 'C', 'C'), ('ACE', 'O', 'O'),
    ('VAL', 'N', 'N'), ('VAL', 'H', 'H'), ('VAL', 'CA', 'C'), ('VAL', 'HA', 'H'),
    ('VAL', 'CB', 'C'), ('VAL', 'HB', 'H'),
    ('VAL', 'CG1', 'C'), ('VAL', 'HG11', 'H'), ('VAL', 'HG12', 'H'), ('VAL', 'HG13', 'H'),
    ('VAL', 'CG2', 'C'), ('VAL', 'HG21', 'H'), ('VAL', 'HG22', 'H'), ('VAL', 'HG23', 'H'),
    ('VAL', 'C', 'C'), ('VAL', 'O', 'O'),
    ('NME', 'N', 'N'), ('NME', 'H', 'H'), ('NME', 'CH3', 'C'),
    ('NME', 'HH31', 'H'), ('NME', 'HH32', 'H'), ('NME', 'HH33', 'H'),
]
BONDS = [
    (0, 1), (1, 2), (1, 3), (1, 4), (4, 5),
    (4, 6), (6, 7), (6, 8), (8, 9), (8, 10),
    (10, 11), (10, 12), (10, 16),
    (12, 13), (12, 14), (12, 15),
    (16, 17), (16, 18), (16, 19),
    (8, 20), (20, 21), (20, 22),
    (22, 23), (22, 24), (24, 25), (24, 26), (24, 27),
]
N_ATOMS = 28

PHI_ATOMS = (4, 6, 8, 20)
PSI_ATOMS = (6, 8, 20, 22)
CHI1_ATOMS = (6, 8, 10, 12)

#: distal to N(6)-CA(8) -- everything a phi rotation moves
PHI_MOVING = tuple(range(9, 28))
#: distal to CA(8)-C(20)
PSI_MOVING = (21, 22, 23, 24, 25, 26, 27)
#: distal to CA(8)-CB(10) -- a strict subset of PHI_MOVING
CHI1_MOVING = tuple(range(11, 20))

#: (axis_i, axis_j, moving, atoms) for each rotatable CV, keyed by name
ROTORS = {
    'phi':  (6, 8, PHI_MOVING, PHI_ATOMS),
    'psi':  (8, 20, PSI_MOVING, PSI_ATOMS),
    'chi1': (8, 10, CHI1_MOVING, CHI1_ATOMS),
}

#: (frame a, b, c ; hydrogen indices) for the four methyl rotors
METHYL_ROTORS = (
    ((6, 4, 1), (0, 2, 3)),          # ACE CH3
    ((20, 22, 24), (25, 26, 27)),    # NME CH3
    ((8, 10, 12), (13, 14, 15)),     # CG1
    ((8, 10, 16), (17, 18, 19)),     # CG2
)

#: sp2 centres as (centre, three substituents), checked for planarity by validate_seed
SP2_CENTRES = ((4, (1, 5, 6)), (20, (8, 21, 22)))
#: amide nitrogens, allowed to be mildly pyramidal
AMIDE_N = ((6, (4, 7, 8)), (22, (20, 23, 24)))
#: the two peptide bonds, expected trans
OMEGA_ATOMS = ((1, 4, 6, 8), (8, 20, 22, 24))

FF_FILES = ("amber14/protein.ff14SB.xml",)

KB = 0.008314462618            # kJ/mol/K


# --------------------------------------------------------------------------- geometry helpers
def signed_dihedral_np(x, idx):
    """Signed dihedral (radians, IUPAC: trans at pi) of ``x[(i,j,k,l)]``; ``x`` is ``(...,28,3)``."""
    p0, p1, p2, p3 = (x[..., i, :] for i in idx)
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1n = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
    v = b0 - (b0 * b1n).sum(-1, keepdims=True) * b1n
    w = b2 - (b2 * b1n).sum(-1, keepdims=True) * b1n
    return np.arctan2((np.cross(b1n, v) * w).sum(-1), (v * w).sum(-1))


def angles_np(x):
    """``(..., 3)`` array of (phi, psi, chi1) in radians."""
    return np.stack([signed_dihedral_np(x, PHI_ATOMS),
                     signed_dihedral_np(x, PSI_ATOMS),
                     signed_dihedral_np(x, CHI1_ATOMS)], axis=-1)


def chirality(x):
    """Signed volume ``(N-CA) . [(C-CA) x (CB-CA)]``.

    **POSITIVE means L-valine (S at CA)**, the same convention and the same construction as
    :func:`alanine.system.chirality`, with C renumbered 14 -> 20.

    Note CB itself is *not* a stereocentre: its two substituents CG1 and CG2 are both methyls,
    carrying identical ff14SB types (``protein-CT``) and identical charges (-0.3192).  Swapping
    the CG1/CG2 labels is therefore an exact symmetry of the energy and maps chi1 -> chi1 +/- 120
    deg.  It relabels the rotamers without moving a single atom, which is why the g+/g-/t naming
    must be read off the built topology rather than assumed.
    """
    CA, C, CB, N = 8, 20, 10, 6
    uN = x[..., N, :] - x[..., CA, :]
    uC = x[..., C, :] - x[..., CA, :]
    uB = x[..., CB, :] - x[..., CA, :]
    return (uN * np.cross(uC, uB)).sum(-1)


def rotate_about_bond(x, i, j, moving, angle):
    """Rotate ``moving`` atoms about the axis ``x[j]-x[i]`` by ``angle`` (radians).

    ``x`` is ``(B, 28, 3)``, ``angle`` is ``(B,)``.  Rodrigues rotation anchored at atom ``j``.
    Rigid: every internal coordinate except the target dihedral is invariant.
    """
    ax = x[:, j] - x[:, i]
    ax = ax / np.linalg.norm(ax, axis=-1, keepdims=True)
    o = x[:, j][:, None, :]
    p = x[:, list(moving)] - o
    c = np.cos(angle)[:, None, None]
    s = np.sin(angle)[:, None, None]
    dot = (p * ax[:, None, :]).sum(-1, keepdims=True)
    y = p * c + np.cross(np.broadcast_to(ax[:, None, :], p.shape), p) * s + ax[:, None, :] * dot * (1 - c)
    out = x.copy()
    out[:, list(moving)] = y + o
    return out


def _nerf(a, b, c, r, theta, phi):
    theta, phi = np.radians(theta), np.radians(phi)
    bc = c - b
    bc = bc / np.linalg.norm(bc)
    n = np.cross(b - a, bc)
    n = n / np.linalg.norm(n)
    m = np.cross(n, bc)
    return c + (-r * np.cos(theta)) * bc + (r * np.sin(theta) * np.cos(phi)) * m \
             + (r * np.sin(theta) * np.sin(phi)) * n


# --------------------------------------------------------------------------- construction
def build_positions(phi_deg=-80.0, psi_deg=80.0, chi1_deg=180.0, cb_offset=-120.0):
    """NeRF build in **angstrom**.  Seed construction only -- see module docstring.

    ``cb_offset = -120`` places CB and HA on either side of C about CA; combined with the
    IUPAC dihedral sign this is what makes :func:`chirality` positive (L).
    """
    X = np.zeros((N_ATOMS, 3))
    X[1] = [0., 0., 0.]
    X[4] = [1.522, 0., 0.]
    a = np.radians(116.6)
    X[6] = X[4] + 1.335 * np.array([-np.cos(a), np.sin(a), 0.])
    X[8] = _nerf(X[1], X[4], X[6], 1.449, 121.9, 180.0)                  # CA, omega trans
    X[20] = _nerf(X[4], X[6], X[8], 1.522, 110.4, phi_deg)               # VAL C
    X[22] = _nerf(X[6], X[8], X[20], 1.335, 116.6, psi_deg)              # NME N
    X[24] = _nerf(X[8], X[20], X[22], 1.449, 121.9, 180.0)               # NME CH3
    X[5] = _nerf(X[8], X[6], X[4], 1.229, 122.9, 180.0)                  # ACE O
    X[21] = _nerf(X[24], X[22], X[20], 1.229, 122.9, 180.0)              # VAL O
    X[10] = _nerf(X[4], X[6], X[8], 1.526, 110.5, phi_deg + cb_offset)   # CB
    X[9] = _nerf(X[4], X[6], X[8], 1.090, 108.0, phi_deg - cb_offset)    # HA
    X[7] = _nerf(X[1], X[4], X[6], 1.010, 119.0, 0.0)                    # H on VAL N
    X[23] = _nerf(X[8], X[20], X[22], 1.010, 119.0, 0.0)                 # H on NME N
    X[12] = _nerf(X[6], X[8], X[10], 1.526, 110.5, chi1_deg)             # CG1 -- defines chi1
    X[16] = _nerf(X[6], X[8], X[10], 1.526, 110.5, chi1_deg + 120.0)     # CG2
    X[11] = _nerf(X[6], X[8], X[10], 1.090, 108.0, chi1_deg + 240.0)     # HB
    for (fa, fb, fc), hs in METHYL_ROTORS:
        for j, idx in enumerate(hs):
            X[idx] = _nerf(X[fa], X[fb], X[fc], 1.090, 109.5, 60.0 + 120.0 * j)
    return X


def topology():
    top = app.Topology()
    ch = top.addChain()
    E = app.element
    el = {'H': 'hydrogen', 'C': 'carbon', 'N': 'nitrogen', 'O': 'oxygen'}
    res, atoms = {}, []
    for rn, an, e in NAMES:
        if rn not in res:
            res[rn] = top.addResidue(rn, ch)
        atoms.append(top.addAtom(an, getattr(E, el[e]), res[rn]))
    for i, j in BONDS:
        top.addBond(atoms[i], atoms[j])
    return top


def make_system(ff_files=FF_FILES, constraints=None, hydrogen_mass=None):
    """Build the OpenMM system.  Defaults reproduce the frozen physical model exactly."""
    top = topology()
    ff = app.ForceField(*ff_files)
    kw = dict(nonbondedMethod=app.NoCutoff, constraints=constraints,
              rigidWater=False, removeCMMotion=False)
    if hydrogen_mass is not None:
        kw["hydrogenMass"] = hydrogen_mass * u.dalton
    return ff, top, ff.createSystem(top, **kw)


# --------------------------------------------------------------------------- seeding
def _reference_context(system, platform="Reference"):
    return mm.Context(system, mm.VerletIntegrator(1.0 * u.femtosecond),
                      mm.Platform.getPlatformByName(platform))


def relieve_methyl_rotors(X_ang, system, n_phase=24, ctx=None):
    """Coarse exhaustive scan of the four methyl phases, one rotor at a time.

    Deterministic and order-dependent by construction; the point is only to hand the
    minimiser a less absurd starting structure, not to find a true minimum.
    """
    own = ctx is None
    if own:
        ctx = _reference_context(system)
    X = X_ang.copy()
    for (fa, fb, fc), hs in METHYL_ROTORS:
        best, best_e = X, np.inf
        for k in range(n_phase):
            ph = 360.0 * k / n_phase
            Y = X.copy()
            for j, idx in enumerate(hs):
                Y[idx] = _nerf(X[fa], X[fb], X[fc], 1.090, 109.5, ph + 120.0 * j)
            ctx.setPositions(Y * 0.1)
            e = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
                u.kilojoule_per_mole)
            if e < best_e:
                best_e, best = e, Y
        X = best
    return X


def restrained_minimise(system, X_ang, targets_deg, kappa=4184.0, max_iterations=20000):
    """Minimise with harmonic restraints on (phi, psi, chi1).

    ``kappa`` is in kJ/mol/rad^2; 4184 (= 1000 kcal) holds every tested seed to within
    0.6 deg of target.  The restraint is applied to a *copy* of the system, so the caller's
    system is never mutated and the returned energy is the true unrestrained potential.

    Returns ``(X_ang, energy_kJ_per_mol)``.
    """
    s2 = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system))
    tor = mm.CustomTorsionForce(
        "0.5*k*min(dtheta, 2*pi-dtheta)^2; dtheta = abs(theta - theta0); "
        "pi = 3.1415926535897932")
    tor.addPerTorsionParameter("k")
    tor.addPerTorsionParameter("theta0")
    for idx, t in zip((PHI_ATOMS, PSI_ATOMS, CHI1_ATOMS), targets_deg):
        tor.addTorsion(*idx, [float(kappa), float(np.radians(t))])
    s2.addForce(tor)
    ctx = _reference_context(s2)
    ctx.setPositions(X_ang * 0.1)
    mm.LocalEnergyMinimizer.minimize(ctx, maxIterations=max_iterations)
    Xm = ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(u.nanometer)
    ctx0 = _reference_context(system)
    ctx0.setPositions(Xm)
    e = ctx0.getState(getEnergy=True).getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
    return Xm * 10.0, e


def make_seed(targets_deg, system=None, kappa=4184.0):
    """Build one validated seed at ``(phi, psi, chi1)`` in degrees.  Returns ``(X_nm, energy)``."""
    if system is None:
        _, _, system = make_system()
    X = build_positions(*targets_deg)
    X = relieve_methyl_rotors(X, system)
    X, e = restrained_minimise(system, X, targets_deg, kappa=kappa)
    return X * 0.1, e


def seed_lattice(X0_nm, centers_rad):
    """Rigidly rotate one validated structure to each ``(phi, psi, chi1)`` in ``centers_rad``.

    ``X0_nm`` is ``(28,3)``, ``centers_rad`` is ``(K,3)``; returns ``(K,28,3)`` in nm.

    Bond lengths, bond angles, planarity and chirality are preserved exactly, because each
    step is a rigid rotation.  The three rotations commute (module docstring), so the order
    below is arbitrary; each angle is nevertheless re-measured immediately before its own
    rotation, matching the alanine idiom and making the function robust to a future CV whose
    rotations do *not* commute.
    """
    c = np.asarray(centers_rad, dtype=float)
    x = np.repeat(np.asarray(X0_nm, dtype=float)[None], c.shape[0], axis=0)
    for k, name in enumerate(('phi', 'chi1', 'psi')):
        i, j, moving, atoms = ROTORS[name]
        col = {'phi': 0, 'psi': 1, 'chi1': 2}[name]
        x = rotate_about_bond(x, i, j, moving, c[:, col] - signed_dihedral_np(x, atoms))
    return x


# --------------------------------------------------------------------------- validation
def _bond_angle(X, i, j, k):
    a, b = X[..., i, :] - X[..., j, :], X[..., k, :] - X[..., j, :]
    cosang = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))


def planarity_deficit(X):
    """``(..., 2)`` degrees by which each sp2 carbonyl carbon falls short of a planar 360."""
    out = []
    for c, (i, j, k) in SP2_CENTRES:
        out.append(360.0 - (_bond_angle(X, i, c, j) + _bond_angle(X, j, c, k)
                            + _bond_angle(X, i, c, k)))
    return np.stack(out, axis=-1)


def min_nonexcluded_distance(X_nm, system):
    """Closest pair in nm that ff14SB does not fully exclude (i.e. not 1-2 or 1-3 bonded)."""
    X = np.asarray(X_nm)
    d = np.linalg.norm(X[..., :, None, :] - X[..., None, :, :], axis=-1)
    idx = np.arange(X.shape[-2])
    d[..., idx, idx] = np.inf
    nb = [f for f in system.getForces() if isinstance(f, mm.NonbondedForce)][0]
    for k in range(nb.getNumExceptions()):
        i, j, qq, sig, eps = nb.getExceptionParameters(k)
        if eps.value_in_unit(u.kilojoule_per_mole) == 0.0:
            d[..., i, j] = d[..., j, i] = np.inf
    return d.min(axis=(-2, -1))


def validate_seed(system, X_nm, targets_rad, cv_tol_deg=1.0, min_contact_nm=0.180,
                  max_planarity_deficit_deg=3.0, max_omega_dev_deg=25.0,
                  require_L=True, energy=None, max_energy=0.0):
    """Stage-0 gate.  Raises ``ValueError`` on the first violated criterion.

    Thresholds are the alanine ones where they exist, and are otherwise set from the measured
    12-seed screening lattice with margin: worst observed CV error 0.6 deg, worst planarity
    deficit 1.01 deg, worst omega deviation 11.5 deg, closest contact 0.205 nm.
    """
    X = np.atleast_3d(np.asarray(X_nm, dtype=float)).reshape(-1, N_ATOMS, 3)
    tgt = np.asarray(targets_rad, dtype=float).reshape(-1, 3)
    if X.shape[0] != tgt.shape[0]:
        raise ValueError(f"{X.shape[0]} structures vs {tgt.shape[0]} targets")

    err = np.degrees((angles_np(X) - tgt + np.pi) % (2 * np.pi) - np.pi)
    if np.abs(err).max() > cv_tol_deg:
        b = np.unravel_index(np.abs(err).argmax(), err.shape)
        raise ValueError(f"seed {b[0]}: CV {('phi','psi','chi1')[b[1]]} off target by "
                         f"{err[b]:.3f} deg (tol {cv_tol_deg})")

    if require_L:
        ch = chirality(X)
        if (ch <= 0).any():
            raise ValueError(f"seed {int(np.argmin(ch))}: chirality {ch.min():.4f} <= 0 "
                             f"-- D-valine, or an inverted CA")

    pd = planarity_deficit(X)
    if pd.max() > max_planarity_deficit_deg:
        b = np.unravel_index(pd.argmax(), pd.shape)
        raise ValueError(f"seed {b[0]}: sp2 centre {SP2_CENTRES[b[1]][0]} non-planar by "
                         f"{pd[b]:.2f} deg -- restrained minimisation cannot repair this")

    om = np.stack([np.degrees(signed_dihedral_np(X, a)) for a in OMEGA_ATOMS], axis=-1)
    dev = np.abs(np.abs(om) - 180.0)
    if dev.max() > max_omega_dev_deg:
        b = np.unravel_index(dev.argmax(), dev.shape)
        raise ValueError(f"seed {b[0]}: omega_{b[1]+1} = {om[b]:.1f} deg, "
                         f"{dev[b]:.1f} deg from trans")

    dmin = min_nonexcluded_distance(X, system)
    if dmin.min() < min_contact_nm:
        raise ValueError(f"seed {int(np.argmin(dmin))}: closest non-excluded contact "
                         f"{dmin.min():.4f} nm < {min_contact_nm}")

    if energy is not None:
        e = np.atleast_1d(energy)
        if (e > max_energy).any():
            raise ValueError(f"seed {int(np.argmax(e))}: energy {e.max():.1f} kJ/mol "
                             f"> {max_energy}")
    return True
