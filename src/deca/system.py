"""Ace-(Ala)10-Nme (112 atoms, AMBER ff14SB, vacuum): construction and structural validation.

Why a builder at all
--------------------
No peptide builder is installed (``pdbfixer``, ``PeptideBuilder``, ``Bio`` and ``parmed`` are
all absent from the ``abffr`` environment), so the chain is built from internal coordinates.
That is exactly the operation the alanine module documents as dangerous: its vendored NeRF
builder places the ACE carbonyl O from the wrong reference frame and inverts an sp2 centre over
a large part of the ``(phi, psi)`` torus, producing a smooth, plausible-looking FES with ~8 kT
of pure artifact (see :mod:`alanine.system`).

This builder does not repeat that mistake, and it does not guess sign conventions either.
Every bond length, bond angle and decoration dihedral below was **measured off the minimised
Ace-Ala-Nme structure that already passed the alanine Stage-0 gates**.  The one convention the
vendored builder got wrong is visible in that measurement:

    dihedral(CA_next, N_next, C, O) = -0.72 deg      (vendored builder used 180)

so the carbonyl O here is placed anti to the alpha carbon of its own residue, which is what a
trans amide requires.  :func:`validate_structure` re-derives all of it from the built
coordinates and is a hard gate, not a diagnostic.

Chain construction is a single NeRF walk over the backbone ``CH3-C-[N-CA-C]x10-N-CH3``; every
other atom is then decorated from a frame entirely inside its own residue, so no atom's
placement depends on a torsion that has not been set yet.

Units: nm, kJ/mol, ps, amu, radians -- except the geometry tables below, which are in Angstrom
and degrees because that is how they are quoted in the literature, and are converted on use.
"""
from __future__ import annotations

import numpy as np

N_RES = 10

FF_FILES = ("amber14/protein.ff14SB.xml",)

# --------------------------------------------------------------------------- measured geometry
# Every value read off ``alanine.system.reference_minimum()`` -- the minimised C7eq structure
# (phi = -74.95, psi = +51.50) that the alanine study validated.  Using the *minimised* values
# rather than textbook ideals means the built chain starts near the force field's own optimum.
BOND_A = dict(N_CA=1.4685, CA_C=1.5514, C_N=1.3386, C_O=1.2298, N_H=1.0062,
              CA_CB=1.5372, CA_HA=1.0914, CB_HB=1.0910, C_CH3=1.5220, CH3_H=1.0900)
ANGLE_DEG = dict(N_CA_C=113.674, CA_C_N=117.535, C_N_CA=125.079, N_CA_CB=107.956,
                 N_CA_HA=107.822, CA_C_O=120.817, O_C_N=121.628, C_N_H=117.842,
                 CA_CB_HB=110.053, X_C_N=117.535, N_C_CH3=109.500)

#: ``dihedral(N, CA, C, O) - psi``.  A trans amide puts O anti to the next residue's N.
O_FROM_PSI_DEG = -178.39
#: ``dihedral(C_prev, N, CA, CB) - phi``.  Sets the L chirality at CA.
CB_FROM_PHI_DEG = -126.69
#: ``dihedral(C_prev, N, CA, HA) - phi``.
HA_FROM_PHI_DEG = +117.49
#: ``dihedral(CA_prev, C_prev, N, H)``.  Amide H trans to the carbonyl O across the C-N bond.
H_AMIDE_DEG = 0.54
#: ``dihedral(CA_next, N_next, C, O)`` for a cap carbonyl whose own CA frame is unavailable.
O_CAP_DEG = -0.72
#: ``dihedral(N, CA, CB, HB{1,2,3})`` -- the staggered methyl rotamer of the minimised structure.
HB_DEG = (64.92, -175.55, -56.01)
#: staggered methyl protons on the cap methyls, measured from the preceding heavy atom.
METHYL_DEG = (60.0, 180.0, 300.0)

KB = 0.008314462618  # kJ/mol/K


# --------------------------------------------------------------------------- geometry helpers
def nerf(a, b, c, r, theta_deg, phi_deg):
    """Place D with ``|c-D| = r``, ``angle(b,c,D) = theta``, ``dihedral(a,b,c,D) = phi``."""
    theta, phi = np.radians(theta_deg), np.radians(phi_deg)
    bc = c - b
    bc = bc / np.linalg.norm(bc)
    n = np.cross(b - a, bc)
    n = n / np.linalg.norm(n)
    m = np.cross(n, bc)
    return c + (-r * np.cos(theta)) * bc + (r * np.sin(theta) * np.cos(phi)) * m \
        + (r * np.sin(theta) * np.sin(phi)) * n


def dihedral_np(x, idx):
    """Signed dihedral in degrees of ``x[(i,j,k,l)]``; ``x`` is ``(..., n_atoms, 3)``."""
    p0, p1, p2, p3 = (x[..., i, :] for i in idx)
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1n = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
    v = b0 - (b0 * b1n).sum(-1, keepdims=True) * b1n
    w = b2 - (b2 * b1n).sum(-1, keepdims=True) * b1n
    return np.degrees(np.arctan2((np.cross(b1n, v) * w).sum(-1), (v * w).sum(-1)))


def angle_np(x, idx):
    """Bond angle in degrees of ``x[(i,j,k)]``."""
    i, j, k = idx
    v1, v2 = x[..., i, :] - x[..., j, :], x[..., k, :] - x[..., j, :]
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(v1, v2), axis=-1), (v1 * v2).sum(-1)))


# --------------------------------------------------------------------------- topology
def names_and_bonds(n_res=N_RES):
    """``(names, bonds)`` for Ace-(Ala)n-Nme in the amber14 within-residue atom order.

    ``names`` is a list of ``(resname, resseq, atomname, element)``; ``bonds`` a list of
    index pairs.  Atom order matches :mod:`alanine._ala22_src` residue by residue, which is
    the order the alanine parity gate was established in.
    """
    names, bonds = [], []

    def add(res, seq, atoms):
        base = len(names)
        for an, el in atoms:
            names.append((res, seq, an, el))
        return base

    ace = add("ACE", 0, [("HH31", "H"), ("CH3", "C"), ("HH32", "H"), ("HH33", "H"),
                         ("C", "C"), ("O", "O")])
    bonds += [(ace + 0, ace + 1), (ace + 1, ace + 2), (ace + 1, ace + 3),
              (ace + 1, ace + 4), (ace + 4, ace + 5)]
    prev_c = ace + 4
    for r in range(n_res):
        b = add("ALA", r + 1, [("N", "N"), ("H", "H"), ("CA", "C"), ("HA", "H"), ("CB", "C"),
                               ("HB1", "H"), ("HB2", "H"), ("HB3", "H"), ("C", "C"), ("O", "O")])
        bonds += [(prev_c, b + 0), (b + 0, b + 1), (b + 0, b + 2), (b + 2, b + 3),
                  (b + 2, b + 4), (b + 4, b + 5), (b + 4, b + 6), (b + 4, b + 7),
                  (b + 2, b + 8), (b + 8, b + 9)]
        prev_c = b + 8
    nme = add("NME", n_res + 1, [("N", "N"), ("H", "H"), ("CH3", "C"),
                                 ("HH31", "H"), ("HH32", "H"), ("HH33", "H")])
    bonds += [(prev_c, nme + 0), (nme + 0, nme + 1), (nme + 0, nme + 2),
              (nme + 2, nme + 3), (nme + 2, nme + 4), (nme + 2, nme + 5)]
    return names, bonds


def atom_index(n_res=N_RES):
    """``{(resseq, atomname): index}``.  ACE is residue 0, ALA i is residue i, NME is n_res+1."""
    names, _ = names_and_bonds(n_res)
    return {(seq, an): i for i, (_, seq, an, _) in enumerate(names)}


def terminal_carbonyls(n_res=N_RES):
    """``(i, j)`` for the two terminal carbonyl carbons -- the primary CV atoms.

    ACE C and the carbonyl C of the last alanine.  This is the end-to-end coordinate the
    deca-alanine ABF literature uses; the peptide's extension maps onto it monotonically.
    """
    I = atom_index(n_res)
    return I[(0, "C")], I[(n_res, "C")]


def topology(n_res=N_RES):
    """OpenMM ``Topology`` for Ace-(Ala)n-Nme."""
    import openmm.app as app
    names, bonds = names_and_bonds(n_res)
    top = app.Topology()
    ch = top.addChain()
    E = app.element
    elmap = {"H": "hydrogen", "C": "carbon", "N": "nitrogen", "O": "oxygen"}
    res, atoms = {}, []
    for rn, seq, an, el in names:
        if seq not in res:
            res[seq] = top.addResidue(rn, ch)
        atoms.append(top.addAtom(an, getattr(E, elmap[el]), res[seq]))
    for i, j in bonds:
        top.addBond(atoms[i], atoms[j])
    return top


def make_system(n_res=N_RES, ff_files=FF_FILES, constraints=None, hydrogen_mass=None):
    """Build the vacuum ff14SB system.  ``hydrogen_mass`` in amu enables HMR (None = off).

    ``constraints`` must stay ``None``: :func:`alanine.forcefield.extract_parameters` refuses a
    constrained system because the batched BAOAB integrator implements no SHAKE/RATTLE.
    """
    import openmm.app as app
    import openmm.unit as u
    top = topology(n_res)
    ff = app.ForceField(*ff_files)
    kw = dict(nonbondedMethod=app.NoCutoff, constraints=constraints,
              rigidWater=False, removeCMMotion=False)
    if hydrogen_mass is not None:
        kw["hydrogenMass"] = hydrogen_mass * u.dalton
    return ff, top, ff.createSystem(top, **kw)


# --------------------------------------------------------------------------- builder
def build_helix(phi_deg=-57.0, psi_deg=-47.0, omega_deg=180.0, n_res=N_RES):
    """Build the chain at uniform ``(phi, psi, omega)``.  Returns ``(n_atoms, 3)`` in **nm**.

    The default is the ideal right-handed alpha-helix.  Backbone first as one NeRF walk over
    ``CH3-C-[N-CA-C]xn-N-CH3``, then every remaining atom from a frame already placed.
    """
    I = atom_index(n_res)
    names, _ = names_and_bonds(n_res)
    X = np.full((len(names), 3), np.nan)

    B, A = BOND_A, ANGLE_DEG
    # --- backbone walk -------------------------------------------------------------------
    X[I[(0, "CH3")]] = [0.0, 0.0, 0.0]
    X[I[(0, "C")]] = [B["C_CH3"], 0.0, 0.0]
    a = np.radians(A["X_C_N"])
    X[I[(1, "N")]] = X[I[(0, "C")]] + B["C_N"] * np.array([-np.cos(a), np.sin(a), 0.0])

    chain = [I[(0, "CH3")], I[(0, "C")], I[(1, "N")]]
    for r in range(1, n_res + 1):
        n_i, ca_i, c_i = I[(r, "N")], I[(r, "CA")], I[(r, "C")]
        p, q, s = chain[-3], chain[-2], chain[-1]
        X[ca_i] = nerf(X[p], X[q], X[s], B["N_CA"], A["C_N_CA"], omega_deg)
        chain.append(ca_i)
        X[c_i] = nerf(X[q], X[s], X[ca_i], B["CA_C"], A["N_CA_C"], phi_deg)
        chain.append(c_i)
        nxt = I[(r + 1, "N")] if r < n_res else I[(n_res + 1, "N")]
        X[nxt] = nerf(X[s], X[ca_i], X[c_i], B["C_N"], A["CA_C_N"], psi_deg)
        chain.append(nxt)
    nme_n, nme_c = I[(n_res + 1, "N")], I[(n_res + 1, "CH3")]
    X[nme_c] = nerf(X[chain[-3]], X[chain[-2]], X[nme_n], B["N_CA"], A["C_N_CA"], omega_deg)

    # --- decorations ---------------------------------------------------------------------
    for r in range(1, n_res + 1):
        n_i, ca_i, c_i = I[(r, "N")], I[(r, "CA")], I[(r, "C")]
        c_prev = I[(0, "C")] if r == 1 else I[(r - 1, "C")]
        ca_prev = I[(0, "CH3")] if r == 1 else I[(r - 1, "CA")]
        X[I[(r, "O")]] = nerf(X[n_i], X[ca_i], X[c_i], B["C_O"], A["CA_C_O"],
                              psi_deg + O_FROM_PSI_DEG)
        X[I[(r, "H")]] = nerf(X[ca_prev], X[c_prev], X[n_i], B["N_H"], A["C_N_H"], H_AMIDE_DEG)
        X[I[(r, "CB")]] = nerf(X[c_prev], X[n_i], X[ca_i], B["CA_CB"], A["N_CA_CB"],
                               phi_deg + CB_FROM_PHI_DEG)
        X[I[(r, "HA")]] = nerf(X[c_prev], X[n_i], X[ca_i], B["CA_HA"], A["N_CA_HA"],
                               phi_deg + HA_FROM_PHI_DEG)
        for k, hb in enumerate(("HB1", "HB2", "HB3")):
            X[I[(r, hb)]] = nerf(X[n_i], X[ca_i], X[I[(r, "CB")]], B["CB_HB"],
                                 A["CA_CB_HB"], HB_DEG[k])

    # ACE carbonyl O: its own residue has no CA, so it is placed from the *next* residue's
    # frame, anti to the ACE methyl -- the convention the vendored dipeptide builder inverted.
    X[I[(0, "O")]] = nerf(X[I[(1, "CA")]], X[I[(1, "N")]], X[I[(0, "C")]],
                          B["C_O"], A["O_C_N"], O_CAP_DEG)
    for k, hh in enumerate(("HH31", "HH32", "HH33")):
        X[I[(0, hh)]] = nerf(X[I[(1, "N")]], X[I[(0, "C")]], X[I[(0, "CH3")]],
                             B["CH3_H"], A["N_C_CH3"], METHYL_DEG[k])
    X[I[(n_res + 1, "H")]] = nerf(X[I[(n_res, "CA")]], X[I[(n_res, "C")]], X[nme_n],
                                  B["N_H"], A["C_N_H"], H_AMIDE_DEG)
    for k, hh in enumerate(("HH31", "HH32", "HH33")):
        X[I[(n_res + 1, hh)]] = nerf(X[I[(n_res, "C")]], X[nme_n], X[nme_c],
                                     B["CH3_H"], A["N_C_CH3"], METHYL_DEG[k])

    if not np.isfinite(X).all():
        missing = [names[i] for i in np.where(~np.isfinite(X).all(-1))[0]]
        raise AssertionError(f"builder left atoms unplaced: {missing}")
    return X * 0.1  # Angstrom -> nm


# --------------------------------------------------------------------------- validation
def per_residue_chirality(x, n_res=N_RES):
    """Signed volume ``(N-CA) . [(C-CA) x (CB-CA)]`` per alanine.  **Positive means L.**

    Same convention as :func:`alanine.system.chirality`, which was verified against an
    unambiguous CIP construction.  ``x`` is ``(..., n_atoms, 3)``; returns ``(..., n_res)``.
    """
    I = atom_index(n_res)
    out = []
    for r in range(1, n_res + 1):
        ca, c, cb, n = I[(r, "CA")], I[(r, "C")], I[(r, "CB")], I[(r, "N")]
        uN = x[..., n, :] - x[..., ca, :]
        uC = x[..., c, :] - x[..., ca, :]
        uB = x[..., cb, :] - x[..., ca, :]
        out.append((uN * np.cross(uC, uB)).sum(-1))
    return np.stack(out, -1)


def backbone_dihedrals(x, n_res=N_RES):
    """``(phi, psi, omega)`` in degrees, each ``(..., n_res)``.

    ``omega[i]`` is the peptide bond *preceding* residue ``i+1``, i.e.
    ``CA_i - C_i - N_{i+1} - CA_{i+1}`` with the caps standing in as CA at the termini.
    """
    I = atom_index(n_res)
    phi, psi, omg = [], [], []
    for r in range(1, n_res + 1):
        c_prev = I[(0, "C")] if r == 1 else I[(r - 1, "C")]
        n_next = I[(r + 1, "N")] if r < n_res else I[(n_res + 1, "N")]
        ca_next = I[(r + 1, "CA")] if r < n_res else I[(n_res + 1, "CH3")]
        phi.append(dihedral_np(x, (c_prev, I[(r, "N")], I[(r, "CA")], I[(r, "C")])))
        psi.append(dihedral_np(x, (I[(r, "N")], I[(r, "CA")], I[(r, "C")], n_next)))
        omg.append(dihedral_np(x, (I[(r, "CA")], I[(r, "C")], n_next, ca_next)))
    return np.stack(phi, -1), np.stack(psi, -1), np.stack(omg, -1)


def validate_structure(system, x, n_res=N_RES, max_angle_dev_deg=8.0, max_bond_dev_frac=0.05,
                       min_omega_abs_deg=165.0):
    """Hard **builder** gate for freshly constructed or minimised structures.

    Do not apply this to configurations drawn from dynamics.  Its tolerances describe how far a
    *built* structure may sit from equilibrium geometry, and ordinary 300 K motion exceeds all
    of them: measured over 256 replicas after 20 ps of BAOAB, the median worst-bond deviation is
    7.0 %, the median worst-angle deviation 12.0 deg and the median smallest ``|omega|``
    163.5 deg, so 0/256 thermal configurations pass -- while chirality is violated 0 times and
    cis peptide bonds occur 0 times in 2560.  Use :func:`validate_thermal` at run time; it tests
    the clauses that separate a *broken* trajectory from a merely warm one.

    Returns ``(ok, report)``; ``ok`` is a single bool.

    Checks, in order of what actually goes wrong: L chirality at every alanine; every harmonic
    bond within ``max_bond_dev_frac`` of its equilibrium length; every harmonic angle within
    ``max_angle_dev_deg`` of its equilibrium value; every peptide bond trans.  A structure that
    fails any of these must never be minimised and used -- restrained minimisation cannot
    repair an inverted sp2 centre or a flipped chirality, it only hides them.
    """
    from alanine.forcefield import extract_parameters
    P = extract_parameters(system)
    x = np.asarray(x, dtype=float)
    single = x.ndim == 2
    xb = x[None] if single else x

    bi, b0, _ = P["bonds"]
    dlen = np.linalg.norm(xb[:, bi[:, 0]] - xb[:, bi[:, 1]], axis=-1)
    bond_dev = np.abs(dlen - b0[None]) / b0[None]

    ai, a0, _ = P["angles"]
    v1 = xb[:, ai[:, 0]] - xb[:, ai[:, 1]]
    v2 = xb[:, ai[:, 2]] - xb[:, ai[:, 1]]
    th = np.arctan2(np.linalg.norm(np.cross(v1, v2), axis=-1), (v1 * v2).sum(-1))
    ang_dev = np.degrees(np.abs(th - a0[None]))

    chir = per_residue_chirality(xb, n_res)
    _, _, omg = backbone_dihedrals(xb, n_res)

    rep = dict(
        max_bond_dev_frac=bond_dev.max(-1), max_angle_dev_deg=ang_dev.max(-1),
        min_chirality=chir.min(-1), min_abs_omega_deg=np.abs(omg).min(-1),
        n_res=n_res, n_atoms=xb.shape[1],
        n_fail_chirality=int((chir <= 0).any(-1).sum()),
    )
    ok = ((chir > 0).all(-1) & (bond_dev.max(-1) < max_bond_dev_frac)
          & (ang_dev.max(-1) < max_angle_dev_deg)
          & (np.abs(omg).min(-1) > min_omega_abs_deg))
    if single:
        rep = {k: (v[0] if isinstance(v, np.ndarray) and v.shape[:1] == (1,) else v)
               for k, v in rep.items()}
        return bool(ok[0]), rep
    return ok, rep


def validate_thermal(x, n_res=N_RES, min_omega_abs_deg=90.0):
    """Run-time integrity gate for configurations drawn from dynamics.

    Tests only what distinguishes a broken trajectory from a warm one, because bond, angle and
    ``omega`` fluctuations at 300 K are physics, not damage:

      * every coordinate finite;
      * L chirality at every alanine -- a chirality flip means the integrator drove an atom
        through a stereocentre and the configuration is no longer the molecule we are studying;
      * no cis peptide bond -- ``|omega| > 90 deg``.  ff14SB assigns a barrier, not a
        constraint, so a genuine cis isomer is possible in principle but is a different
        conformational species and must be counted, never silently averaged in.

    Returns ``(ok (...,), report)`` with a leading batch dimension preserved.
    """
    x = np.asarray(x, dtype=float)
    single = x.ndim == 2
    xb = x[None] if single else x
    finite = np.isfinite(xb).all(axis=(-2, -1))
    chir = per_residue_chirality(xb, n_res)
    _, _, omg = backbone_dihedrals(xb, n_res)
    n_cis = (np.abs(omg) < min_omega_abs_deg).sum(-1)
    ok = finite & (chir > 0).all(-1) & (n_cis == 0)
    rep = dict(finite=finite, min_chirality=chir.min(-1), n_cis_bonds=n_cis,
               min_abs_omega_deg=np.abs(omg).min(-1),
               n_fail_finite=int((~finite).sum()),
               n_fail_chirality=int((chir <= 0).any(-1).sum()),
               n_fail_cis=int((n_cis > 0).sum()))
    if single:
        return bool(ok[0]), rep
    return ok, rep
