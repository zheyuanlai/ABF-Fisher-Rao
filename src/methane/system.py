"""Methane pair in explicit SPC/E water: system construction and the OpenMM parity reference.

Frozen by ``docs/SPEC_methane_water.md`` §1.  This module owns the *definition* of the physical
system; nothing else in the methane package may restate a force-field number.

Two consumers, deliberately one builder
---------------------------------------
1. the OpenMM system, which is the **parity target** for the batched torch engine (SPEC §3.2)
   and the engine used for the one-off NPT box equilibration (SPEC §1.3);
2. the torch engine itself, which reads its parameters from :func:`site_parameters` -- the *same*
   arrays OpenMM was built from, so the two cannot drift apart silently.

If those two ever read different numbers the parity gate becomes meaningless, which is why the
tables below appear exactly once.

Units are the project's internal set: **nm, kJ/mol, ps, amu, K, elementary charge**.  Literature
values are quoted in the docstring in Angstrom and kcal/mol because that is how they are
published, and converted here once (``1 kcal/mol = 4.184 kJ/mol``).

Declared deviation (SPEC §1.1)
------------------------------
The methane--oxygen unlike-pair rule is **Lorentz--Berthelot, and it is our choice**, not a value
read from Asthagiri, Merchant & Pratt (JCP 128:244512, 2008).  That paper specifies the methane
model, the SPC/E solvent and the protocol; an explicit unlike-pair formula could not be verified
from it.  OpenMM applies Lorentz--Berthelot internally for ``NonbondedForce``, so the torch engine
must use the same rule -- :func:`unlike_pair` states it explicitly rather than inheriting it.

Two OpenMM conventions that would otherwise break parity
--------------------------------------------------------
* **PME parameters are set explicitly.**  Left alone, OpenMM derives ``alpha`` and the grid from
  ``ewaldErrorTolerance`` *at Context creation*, so they depend on the box and are not visible in
  the ``System``.  A torch engine matched against one box would silently mismatch another.
  :func:`build_system` pins them and records them in the manifest.
* **The analytic dispersion correction is off by default here.**  It is a function of volume and
  the particle parameters alone, so in NVT it is an additive constant contributing **zero force**
  -- it cannot affect any sampled distribution, any conditional mean force, or any free-energy
  difference along ``xi``.  It is switched *on* only for the NPT box equilibration, where it
  belongs in the pressure.  Carrying it into the parity gate would mean implementing a constant in
  torch for no physical return.
"""
from __future__ import annotations

import io

import numpy as np

KCAL_PER_KJ = 4.184

# ------------------------------------------------------------------ frozen model (SPEC §1)
#: United-atom methane, ``eps/k_B = 148 K``.  Asthagiri et al. 2008; TraPPE-UA agrees.
SIGMA_M_NM = 0.3730                     #: 3.73 Angstrom
EPSILON_M_KJ = 0.294 * KCAL_PER_KJ      #: 0.294 kcal/mol -> 1.230096 kJ/mol
MASS_METHANE_AMU = 16.043

#: SPC/E oxygen.  Berendsen, Grigera & Straatsma 1987.
#:
#: **The simulated value is AMBER's, not the nominal one, and the difference is deliberate.**
#: The literature quotes ``sigma_O = 3.166 Angstrom``; ``amber14/spce.xml`` stores the pair as an
#: R_min/A-B coefficient that unpacks to ``0.31657195050398826 nm`` -- 9.0e-5 relative below the
#: nominal value.  That is physically nothing and numerically everything: the engine-equivalence
#: gate of SPEC §3.2 demands ``1e-6``, so a torch engine written from ``0.3166`` would fail
#: parity by ~90x for a reason that is not a bug.  The value actually installed by the force
#: field is therefore the constant, and the nominal value is kept beside it for the record.
SIGMA_O_NM = 0.31657195050398826        #: amber14/spce.xml, as simulated
SIGMA_O_NOMINAL_NM = 0.3166             #: 3.166 Angstrom, as published
EPSILON_O_KJ = 0.1553 * KCAL_PER_KJ     #: 0.1553 kcal/mol -> 0.6497752 kJ/mol
Q_O_E = -0.8476
Q_H_E = +0.4238
R_OH_NM = 0.1000                        #: rigid
THETA_HOH_DEG = 109.47                  #: rigid

N_WATERS = 512
N_METHANES = 2
N_SITES = N_METHANES + 3 * N_WATERS     #: 1538

TEMPERATURE_K = 298.0
GAMMA_PS = 1.0
DT_PS = 0.0005                          #: 0.5 fs, SPEC §1 / Amendment 11.8
PRESSURE_BAR = 1.0                      #: NPT box equilibration only

CUTOFF_NM = 1.05
SWITCH_NM = 1.00

#: Ewald splitting and grid, pinned so parity does not depend on the box (see module docstring).
#: The values are OpenMM's own choice at ``ewaldErrorTolerance = 5e-4`` for a ~2.6 nm box; they
#: are frozen here so the torch engine has a fixed target.
PME_ALPHA_PER_NM = 2.503105604646158
PME_GRID = (20, 20, 20)
PME_SPLINE_ORDER = 5                    #: OpenMM's fixed B-spline order for PME

#: Force-field XML for the single-site methane.  Written inline because it is four lines of
#: parameters that must not diverge from the constants above; ``_check_xml_matches_constants``
#: is a hard gate that they have not.
METHANE_XML = f"""<ForceField>
 <AtomTypes>
  <Type name="MTH-C" class="MTH" element="C" mass="{MASS_METHANE_AMU}"/>
 </AtomTypes>
 <Residues>
  <Residue name="MTH"><Atom name="C1" type="MTH-C" charge="0.0"/></Residue>
 </Residues>
 <NonbondedForce coulomb14scale="0.833333" lj14scale="0.5">
  <Atom type="MTH-C" charge="0.0" sigma="{SIGMA_M_NM}" epsilon="{EPSILON_M_KJ}"/>
 </NonbondedForce>
</ForceField>"""


def kT_kJ(temperature_k=TEMPERATURE_K):
    """``kT`` in kJ/mol.  2.47771 kJ/mol at 298 K."""
    return 8.31446261815324e-3 * float(temperature_k)


def beta_per_kJ(temperature_k=TEMPERATURE_K):
    """``beta = 1/kT`` in mol/kJ."""
    return 1.0 / kT_kJ(temperature_k)


def unlike_pair(sigma_a, epsilon_a, sigma_b, epsilon_b):
    """Lorentz--Berthelot unlike-pair parameters -- **our declared choice** (SPEC §1.1).

    ``sigma = (sigma_a + sigma_b)/2``, ``epsilon = sqrt(eps_a eps_b)``.  For methane--oxygen this
    gives ``sigma_MO = 0.34480 nm``, ``eps_MO = 0.894022 kJ/mol`` (0.21368 kcal/mol).

    Stated as a function rather than as two constants because the torch engine applies it to
    every pair type, and because it is a deviation that must be visible at its point of use.
    """
    return 0.5 * (sigma_a + sigma_b), float(np.sqrt(epsilon_a * epsilon_b))


def r_HH_nm():
    """Rigid SPC/E H--H distance implied by ``r_OH`` and ``theta_HOH``: 0.163299 nm."""
    return 2.0 * R_OH_NM * float(np.sin(np.radians(THETA_HOH_DEG) / 2.0))


def _forcefield():
    import openmm.app as app
    return app.ForceField("amber14/spce.xml", io.StringIO(METHANE_XML))


def build_modeller(r0_nm=0.55, pad_box_nm=3.0, seed=None):
    """Two methanes separated by ``r0_nm`` solvated by **exactly** ``N_WATERS`` SPC/E waters.

    ``pad_box_nm`` is only the box handed to ``addSolvent``; the returned box is whatever OpenMM
    needed for ``numAdded = 512`` and is **not** the production box.  The production box comes
    from the NPT equilibration of SPEC §1.3 and is frozen from that measurement.

    ``Modeller.addSolvent`` takes no seed argument, so ``seed`` seeds the ``random`` module that
    OpenMM's solvation draws from.  Reproducibility of the solvated box is asserted by
    ``tests/test_methane_stage0.py``, not assumed.
    """
    import random

    import openmm as mm
    import openmm.app as app
    import openmm.unit as u

    top = app.Topology()
    chain = top.addChain()
    positions = []
    for dx in (-0.5 * r0_nm, +0.5 * r0_nm):
        res = top.addResidue("MTH", chain)
        top.addAtom("C1", app.element.carbon, res)
        positions.append(mm.Vec3(0.5 * pad_box_nm + dx, 0.5 * pad_box_nm, 0.5 * pad_box_nm))
    top.setUnitCellDimensions(mm.Vec3(pad_box_nm, pad_box_nm, pad_box_nm))

    if seed is not None:
        random.seed(int(seed))
    mod = app.Modeller(top, positions * u.nanometer)
    mod.addSolvent(_forcefield(), model="spce", numAdded=N_WATERS)

    n_w = sum(1 for r in mod.topology.residues() if r.name in ("HOH", "WAT"))
    if n_w != N_WATERS:
        raise RuntimeError(f"addSolvent produced {n_w} waters, expected {N_WATERS}")
    if mod.topology.getNumAtoms() != N_SITES:
        raise RuntimeError(f"built {mod.topology.getNumAtoms()} sites, expected {N_SITES}")
    return mod


def build_system(topology, box_vectors=None, dispersion_correction=False,
                 pin_pme=True, rigid_water=True):
    """The OpenMM ``System``: PME, switched LJ, rigid SPC/E, no flexible water terms.

    ``dispersion_correction`` is ``False`` by default -- see the module docstring; pass ``True``
    only for the NPT box equilibration.  ``pin_pme`` fixes ``alpha`` and the grid to the frozen
    values so parity does not depend on the box.
    """
    import openmm as mm
    import openmm.app as app
    import openmm.unit as u

    system = _forcefield().createSystem(
        topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=CUTOFF_NM * u.nanometer,
        switchDistance=SWITCH_NM * u.nanometer,
        constraints=None,
        rigidWater=rigid_water,
    )
    if box_vectors is not None:
        system.setDefaultPeriodicBoxVectors(*box_vectors)

    nbf = next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce))
    nbf.setUseDispersionCorrection(bool(dispersion_correction))
    if pin_pme:
        nbf.setPMEParameters(PME_ALPHA_PER_NM / u.nanometer, *PME_GRID)

    # Rigid SPC/E must carry no flexible internal terms; a nonzero count means the water model
    # was built flexible and every downstream constraint/equipartition statement would be wrong.
    for force in system.getForces():
        if isinstance(force, mm.HarmonicBondForce) and force.getNumBonds():
            raise RuntimeError(f"rigid water expected, found {force.getNumBonds()} bonds")
        if isinstance(force, mm.HarmonicAngleForce) and force.getNumAngles():
            raise RuntimeError(f"rigid water expected, found {force.getNumAngles()} angles")
    # Three constraints per water **in this topology**, not three times the frozen N_WATERS.
    # The production system has 512 waters, but SPEC §1.3's finite-size gate builds a 1024-water
    # box with the same builder, and a guard keyed to the module constant rejected it -- the same
    # number-versus-population confusion this campaign kept finding in its analysis code, here in
    # a validity check. The invariant that actually matters is "every water is rigid".
    n_waters = sum(1 for r in topology.residues() if r.name in ("HOH", "WAT"))
    expected_constraints = 3 * n_waters if rigid_water else 0
    if system.getNumConstraints() != expected_constraints:
        raise RuntimeError(f"{system.getNumConstraints()} constraints for {n_waters} waters, "
                           f"expected {expected_constraints}")
    return system


def site_parameters(system, topology):
    """Per-site ``(charge_e, sigma_nm, epsilon_kJ, mass_amu, is_methane)`` read **from OpenMM**.

    The torch engine consumes exactly this, so it cannot disagree with the parity target about a
    parameter.  Hydrogens carry OpenMM's placeholder ``sigma = 1.0, epsilon = 0``; epsilon is
    what matters and it is zero, but the sigma is normalised to ``0.0`` here so a Lorentz
    combination against it can never contribute a spurious radius.
    """
    import openmm as mm

    nbf = next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce))
    n = system.getNumParticles()
    q = np.zeros(n); sig = np.zeros(n); eps = np.zeros(n); mass = np.zeros(n)
    for i in range(n):
        qi, si, ei = nbf.getParticleParameters(i)
        q[i] = qi.value_in_unit_system(mm.unit.md_unit_system)
        sig[i] = si.value_in_unit_system(mm.unit.md_unit_system)
        eps[i] = ei.value_in_unit_system(mm.unit.md_unit_system)
        mass[i] = system.getParticleMass(i).value_in_unit_system(mm.unit.md_unit_system)
    sig[eps == 0.0] = 0.0

    is_methane = np.zeros(n, dtype=bool)
    for res in topology.residues():
        if res.name == "MTH":
            for atom in res.atoms():
                is_methane[atom.index] = True
    if int(is_methane.sum()) != N_METHANES:
        raise RuntimeError(f"found {int(is_methane.sum())} methane sites, expected {N_METHANES}")
    return dict(charge=q, sigma=sig, epsilon=eps, mass=mass, is_methane=is_methane,
                methane_index=np.flatnonzero(is_methane))


def exclusions(system):
    """Intramolecular exclusion pairs (``(i, j)`` array).  1536 = 512 waters x 3.

    SPC/E excludes all three intramolecular pairs entirely -- OpenMM stores them as exceptions
    with zero charge product and zero epsilon.  The torch engine must subtract the corresponding
    *reciprocal-space* contribution for each, which is the single most commonly wrong term in a
    hand-written PME and is therefore gated explicitly by the parity test.
    """
    import openmm as mm

    nbf = next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce))
    pairs = []
    for k in range(nbf.getNumExceptions()):
        i, j, qq, sig, eps = nbf.getExceptionParameters(k)
        qq = qq.value_in_unit_system(mm.unit.md_unit_system)
        eps = eps.value_in_unit_system(mm.unit.md_unit_system)
        if qq != 0.0 or eps != 0.0:
            raise RuntimeError(f"exception {k} is not a pure exclusion (qq={qq}, eps={eps})")
        pairs.append((i, j))
    return np.asarray(pairs, dtype=np.int64)


def validate_geometry(positions_nm, topology, tol_nm=1e-8):
    """Hard gate: every rigid water holds its O--H and H--H distances (SPEC §3.2).

    Returns ``(max_oh_error, max_hh_error)``; raises if either exceeds ``tol_nm``.
    """
    pos = np.asarray(positions_nm, dtype=np.float64)
    target_hh = r_HH_nm()
    max_oh = max_hh = 0.0
    for res in topology.residues():
        if res.name not in ("HOH", "WAT"):
            continue
        idx = {a.name: a.index for a in res.atoms()}
        o = pos[idx["O"]]
        h1, h2 = pos[idx["H1"]], pos[idx["H2"]]
        max_oh = max(max_oh, abs(np.linalg.norm(h1 - o) - R_OH_NM),
                     abs(np.linalg.norm(h2 - o) - R_OH_NM))
        max_hh = max(max_hh, abs(np.linalg.norm(h2 - h1) - target_hh))
    if max_oh > tol_nm or max_hh > tol_nm:
        raise RuntimeError(f"rigid-water violation: max|dOH| = {max_oh:.3e} nm, "
                           f"max|dHH| = {max_hh:.3e} nm (tol {tol_nm:.1e})")
    return max_oh, max_hh


def apply_constraints(system, topology, positions_nm, tol=1e-10):
    """Project ``positions_nm`` onto the rigid-water constraint manifold and validate.

    ``Modeller.addSolvent`` places waters from a stored pre-equilibrated box whose internal
    geometry carries only ~1e-4 nm of precision; OpenMM enforces the exact rigid geometry when a
    Context applies constraints, not at build time.  Every consumer that cares about the rigid
    geometry -- the parity gate, the torch engine's initial state, ``validate_geometry`` at its
    1e-8 tolerance -- must go through here first, otherwise it is testing the builder's rounding
    rather than the physics.
    """
    import openmm as mm
    import openmm.unit as u

    ctx = mm.Context(system, mm.VerletIntegrator(1e-6),
                     mm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(np.asarray(positions_nm) * u.nanometer)
    ctx.applyConstraints(tol)
    out = np.asarray(ctx.getState(getPositions=True)
                     .getPositions().value_in_unit(u.nanometer), dtype=np.float64)
    del ctx
    validate_geometry(out, topology, tol_nm=1e-8)
    return out


def manifest():
    """Every frozen number, for the run manifest.  What a reader needs to rebuild the model."""
    sig_mo, eps_mo = unlike_pair(SIGMA_M_NM, EPSILON_M_KJ, SIGMA_O_NM, EPSILON_O_KJ)
    return dict(
        n_waters=N_WATERS, n_methanes=N_METHANES, n_sites=N_SITES,
        sigma_M_nm=SIGMA_M_NM, epsilon_M_kJ=EPSILON_M_KJ, mass_methane_amu=MASS_METHANE_AMU,
        sigma_O_nm=SIGMA_O_NM, sigma_O_nominal_nm=SIGMA_O_NOMINAL_NM,
        epsilon_O_kJ=EPSILON_O_KJ, q_O_e=Q_O_E, q_H_e=Q_H_E,
        r_OH_nm=R_OH_NM, theta_HOH_deg=THETA_HOH_DEG, r_HH_nm=r_HH_nm(),
        sigma_MO_nm=sig_mo, epsilon_MO_kJ=eps_mo, mixing_rule="lorentz-berthelot (DECLARED CHOICE)",
        temperature_K=TEMPERATURE_K, gamma_ps=GAMMA_PS, dt_ps=DT_PS, kT_kJ=kT_kJ(),
        cutoff_nm=CUTOFF_NM, switch_nm=SWITCH_NM,
        pme_alpha_per_nm=PME_ALPHA_PER_NM, pme_grid=list(PME_GRID),
        pme_spline_order=PME_SPLINE_ORDER,
        dispersion_correction="off (NVT: zero force, additive constant); on for NPT only",
        spec="docs/SPEC_methane_water.md", amendment="V2_PREREGISTRATION.md Amendment 11",
    )
