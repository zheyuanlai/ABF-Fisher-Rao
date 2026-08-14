"""C60 pair in explicit TIP4P-Ew water: system construction and the OpenMM parity reference.

Frozen by ``docs/SPEC_c60_water.md`` §1.  This module owns the *definition* of the physical
system; nothing else in the c60 package may restate a force-field number.

Two consumers, deliberately one builder (the methane rule):

1. the OpenMM system -- the **parity target** for the batched torch engine (SPEC §3.1) and the
   engine for the one-off NPT box equilibration (SPEC §1.4);
2. the torch engine, which reads its parameters from :func:`site_parameters` -- the same arrays
   OpenMM was built from, so the two cannot drift apart silently.

Parameter derivation (SPEC §1.1)
--------------------------------
Zangi 2014 specifies the *cross* terms: ``sigma_CO = 0.319 nm``, ``epsilon_CO = 0.392 kJ/mol``
(Werder graphite/water), with C-C "extracted using the geometric combination rule".  The
per-particle carbon parameters are therefore **derived** so the geometric rule reproduces the
paper exactly:

    sigma_C   = sigma_CO^2 / sigma_O          epsilon_C = epsilon_CO^2 / epsilon_O

with the *stored* TIP4P-Ew values from ``amber14/tip4pew.xml`` (checked against the built
System at 1e-12, the methane ``sigma_O`` rule).  OpenMM's Lorentz-Berthelot arithmetic sigma
mean then puts ``sigma_CO`` at 0.3190104 nm -- 1.04e-5 nm above the paper -- while C-C and both
epsilons are exact.  Declared Deviation 1; ~1e-3 kJ/mol at contact, four orders below kT.

Intra-cage interactions are **excluded** (paper): all 2 x C(60,2) = 3540 pairs are pure
exclusions, overriding whatever the bond graph auto-generated.  Carbons are neutral and carry
zero charge, so the exclusions have no reciprocal-space correction.

The cages are rigid and **fixed**: every carbon has zero mass in the OpenMM system, and the
paper's convention (positions held fixed) is reproduced literally in the reference windows.
"""
from __future__ import annotations

import io
import json
import os

import numpy as np

from . import geometry

# ------------------------------------------------------------------ frozen model (SPEC §1)
SIGMA_CO_NM = 0.319                 #: Zangi 2014 / Werder 2003
EPSILON_CO_KJ = 0.392               #: Zangi 2014 / Werder 2003

#: TIP4P-Ew as stored by ``amber14/tip4pew.xml`` -- checked against the built System, never
#: retyped from the paper (Horn et al. 2004 quote 3.16435 A / 0.680946 kJ/mol; storage agrees).
SIGMA_O_NM = 0.31643500000000002
EPSILON_O_KJ = 0.68094600000000005
Q_H_E = 0.5242200000
Q_M_E = -1.0484400000
R_OH_NM = 0.09572
R_HH_NM = 0.15139006545247014       #: implied by theta_HOH = 104.52 deg, as OpenMM constrains
VSITE_WEIGHTS = (0.786646558, 0.106676721, 0.106676721)   #: ThreeParticleAverageSite (O,H1,H2)

#: Derived carbon parameters -- geometric rule exact (SPEC §1.1).
SIGMA_C_NM = SIGMA_CO_NM ** 2 / SIGMA_O_NM          # 0.32158673...
EPSILON_C_KJ = EPSILON_CO_KJ ** 2 / EPSILON_O_KJ    # 0.22566402...
MASS_C_AMU = geometry.MASS_C_AMU

N_WATERS = 1282
N_CARBONS = 120
N_SITES = N_CARBONS + 4 * N_WATERS      #: 5248

TEMPERATURE_K = 300.0
GAMMA_PS = 1.0
DT_PS = 0.002                           #: paper's 2 fs, subject to the SPEC §3.4 dt gate
PRESSURE_BAR = 1.0                      #: NPT box equilibration only

CUTOFF_NM = 1.0                         #: LJ and PME real space, paper; NO switching function

D_REF_NM = 2.428                        #: PMF anchor separation; the box is built and NPT-run here
XI_LO_NM = 0.908
XI_HI_NM = 2.428

#: Built box aspect A = Lz/Lx, frozen at build time (SPEC §1.4).  5.65/2.64 reproduces the
#: paper's Figure-3 z-extent and 1282 waters at ~1 g/cm^3.
BOX_ASPECT = 5.65 / 2.64

#: mass of one cage and the effective mass of the single solute DOF xi = Z_B - Z_A with the
#: midpoint fixed: KE = (1/2)(M/2) xidot^2  (SPEC §1.3)
MASS_CAGE_AMU = 60 * MASS_C_AMU
MU_XI_AMU = 0.5 * MASS_CAGE_AMU

C60_XML = f"""<ForceField>
 <AtomTypes>
  <Type name="C60-C" class="C60" element="C" mass="{MASS_C_AMU}"/>
 </AtomTypes>
 <Residues>
  <Residue name="C60">
   {{ATOMS_AND_BONDS}}
  </Residue>
 </Residues>
 <NonbondedForce coulomb14scale="0.833333" lj14scale="0.5">
  <Atom type="C60-C" charge="0.0" sigma="{SIGMA_C_NM!r}" epsilon="{EPSILON_C_KJ!r}"/>
 </NonbondedForce>
</ForceField>"""


PME_SPLINE_ORDER = 5                    #: OpenMM's fixed B-spline order

_PME_JSON = os.path.join(os.path.dirname(__file__), "..", "..",
                         "results", "c60", "box", "pme_params.json")


def pme_params():
    """Frozen PME ``(alpha_per_nm, nx, ny, nz)`` -- OpenMM's own choice at the frozen box
    (``ewaldErrorTolerance 5e-4``), measured once after the SPEC §1.4 box freeze and recorded
    in ``results/c60/box/pme_params.json`` so parity has a fixed target.  Raises before the
    box freeze: nothing downstream may run without it."""
    with open(_PME_JSON) as fh:
        d = json.load(fh)
    return float(d["alpha_per_nm"]), int(d["nx"]), int(d["ny"]), int(d["nz"])


def kT_kJ(temperature_k=TEMPERATURE_K):
    """``kT`` in kJ/mol.  2.494339 kJ/mol at 300 K."""
    return 8.31446261815324e-3 * float(temperature_k)


def beta_per_kJ(temperature_k=TEMPERATURE_K):
    return 1.0 / kT_kJ(temperature_k)


def cage_bonds():
    """The 90 bonded pairs of one cage, as local indices 0..59 (3-coordinated, checked)."""
    cage = geometry.c60_cage()
    d = np.linalg.norm(cage[:, None, :] - cage[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    iu = np.triu_indices(60, 1)
    pairs = [(int(i), int(j)) for i, j in zip(*iu) if d[i, j] < 0.185]
    if len(pairs) != 90:
        raise RuntimeError(f"{len(pairs)} cage bonds, expected 90")
    return pairs


def _c60_forcefield_xml():
    atoms = "\n   ".join(f'<Atom name="C{k}" type="C60-C"/>' for k in range(60))
    bonds = "\n   ".join(f'<Bond atomName1="C{i}" atomName2="C{j}"/>' for i, j in cage_bonds())
    return C60_XML.replace("{ATOMS_AND_BONDS}", atoms + "\n   " + bonds)


def _forcefield():
    import openmm.app as app
    return app.ForceField("amber14/tip4pew.xml", io.StringIO(_c60_forcefield_xml()))


def build_modeller(d_com_nm=D_REF_NM, lx0_nm=2.74, seed=20260814):
    """Two fixed cages at ``d_com_nm`` + **exactly** ``N_WATERS`` TIP4P-Ew waters.

    The initial box is ``(lx0, lx0, A*lx0)`` with the frozen aspect ``A``; ``addSolvent`` fills
    it slightly over target and a *seeded uniform* random deletion trims to exactly 1282 --
    a uniform density deficit the NPT stage then removes, rather than an anisotropic void.
    The returned box is **not** the production box; that comes from the SPEC §1.4 NPT freeze.
    """
    import openmm as mm
    import openmm.app as app
    import openmm.unit as u

    lz0 = BOX_ASPECT * lx0_nm
    center = (0.5 * lx0_nm, 0.5 * lx0_nm, 0.5 * lz0)

    top = app.Topology()
    chain = top.addChain()
    atoms = []
    for cage in range(2):
        res = top.addResidue("C60", chain)
        cage_atoms = [top.addAtom(f"C{k}", app.element.carbon, res) for k in range(60)]
        for i, j in cage_bonds():
            top.addBond(cage_atoms[i], cage_atoms[j])
        atoms.extend(cage_atoms)
    pos = geometry.pair_positions(d_com_nm, center)
    top.setPeriodicBoxVectors(np.diag([lx0_nm, lx0_nm, lz0]) * u.nanometer)

    mod = app.Modeller(top, pos * u.nanometer)
    mod.addSolvent(_forcefield(), model="tip4pew",
                   boxVectors=tuple(mm.Vec3(*row) for row in np.diag([lx0_nm, lx0_nm, lz0]))
                   * u.nanometer)

    waters = [r for r in mod.topology.residues() if r.name in ("HOH", "WAT")]
    if len(waters) < N_WATERS:
        raise RuntimeError(f"addSolvent produced {len(waters)} waters < {N_WATERS}; "
                           f"enlarge lx0_nm")
    rng = np.random.default_rng(seed)
    drop = rng.choice(len(waters), size=len(waters) - N_WATERS, replace=False)
    mod.delete([waters[int(k)] for k in drop])

    n_w = sum(1 for r in mod.topology.residues() if r.name in ("HOH", "WAT"))
    if n_w != N_WATERS or mod.topology.getNumAtoms() != N_SITES:
        raise RuntimeError(f"built {n_w} waters / {mod.topology.getNumAtoms()} sites, "
                           f"expected {N_WATERS} / {N_SITES}")
    return mod


def build_system(topology, box_vectors=None, dispersion_correction=False,
                 pme_params=None, fix_cages=True):
    """The OpenMM ``System``: PME, unswitched LJ at 1.0 nm, rigid TIP4P-Ew, fixed neutral cages.

    ``dispersion_correction`` is ``False`` by default (NVT: an additive constant with zero
    force); pass ``True`` only for the NPT box equilibration, where it belongs in the pressure
    (the paper applied tail corrections).  ``pme_params = (alpha_per_nm, nx, ny, nz)`` pins the
    reciprocal space once the box is frozen; ``None`` lets OpenMM choose (NPT only).
    """
    import openmm as mm
    import openmm.app as app
    import openmm.unit as u

    system = _forcefield().createSystem(
        topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=CUTOFF_NM * u.nanometer,
        constraints=None,
        rigidWater=True,
    )
    if box_vectors is not None:
        system.setDefaultPeriodicBoxVectors(*box_vectors)

    nbf = next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce))
    nbf.setUseDispersionCorrection(bool(dispersion_correction))
    nbf.setUseSwitchingFunction(False)                    # paper: plain 1.0 nm cutoff
    nbf.setEwaldErrorTolerance(5.0e-4)
    if pme_params is not None:
        alpha, nx, ny, nz = pme_params
        nbf.setPMEParameters(alpha / u.nanometer, int(nx), int(ny), int(nz))

    # ---- carbon indices, from the topology --------------------------------------------------
    carbon = [a.index for r in topology.residues() if r.name == "C60" for a in r.atoms()]
    if len(carbon) != N_CARBONS:
        raise RuntimeError(f"{len(carbon)} carbons, expected {N_CARBONS}")

    # ---- intra-cage pairs are pure exclusions (paper) ---------------------------------------
    # The bond graph auto-generated 1-2/1-3 exclusions and *scaled* 1-4 exceptions for the
    # cages; every intra-cage pair must instead be fully excluded.  addException(replace=True)
    # overrides the auto-generated entries and adds the rest.
    for cage in (carbon[:60], carbon[60:]):
        for a in range(60):
            for b in range(a + 1, 60):
                nbf.addException(cage[a], cage[b], 0.0, 1.0, 0.0, True)

    # ---- fixed cages: zero mass = immobile in OpenMM ----------------------------------------
    if fix_cages:
        for i in carbon:
            system.setParticleMass(i, 0.0 * u.amu)

    # ---- validity guards --------------------------------------------------------------------
    for force in system.getForces():
        if isinstance(force, mm.HarmonicBondForce) and force.getNumBonds():
            raise RuntimeError(f"found {force.getNumBonds()} bonded terms; the cage must be "
                               "rigid geometry, not a bonded model")
        if isinstance(force, mm.HarmonicAngleForce) and force.getNumAngles():
            raise RuntimeError("found angle terms; none expected")
    n_w = sum(1 for r in topology.residues() if r.name in ("HOH", "WAT"))
    if system.getNumConstraints() != 3 * n_w:
        raise RuntimeError(f"{system.getNumConstraints()} constraints for {n_w} waters")
    expected_exc = 6 * n_w + 2 * (60 * 59 // 2)
    if nbf.getNumExceptions() != expected_exc:
        raise RuntimeError(f"{nbf.getNumExceptions()} exceptions, expected {expected_exc}")
    return system


def site_parameters(system, topology):
    """Per-site parameters read **from OpenMM** -- the torch engine consumes exactly this.

    Returns charge/sigma/epsilon/mass arrays plus the index structure: carbon indices per cage,
    water site quadruples ``(O, H1, H2, M)``, and the virtual-site weights.  LJ-less sites get
    ``sigma = 0`` so a mixing rule against them can never contribute a spurious radius.
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

    carbon = np.asarray([a.index for r in topology.residues() if r.name == "C60"
                         for a in r.atoms()], dtype=np.int64)
    waters = []
    for res in topology.residues():
        if res.name in ("HOH", "WAT"):
            idx = {a.name: a.index for a in res.atoms()}
            waters.append((idx["O"], idx["H1"], idx["H2"], idx["M"]))
    waters = np.asarray(waters, dtype=np.int64)

    # virtual-site weights, from the System itself
    m_idx = waters[:, 3]
    w = None
    for m in m_idx:
        if not system.isVirtualSite(int(m)):
            raise RuntimeError(f"site {m} is not a virtual site")
        vs = system.getVirtualSite(int(m))
        if not isinstance(vs, mm.ThreeParticleAverageSite):
            raise RuntimeError(f"unexpected virtual site type {type(vs).__name__}")
        wk = tuple(vs.getWeight(k) for k in range(3))
        if w is None:
            w = wk
        elif wk != w:
            raise RuntimeError("inconsistent virtual-site weights across waters")

    # frozen-constant identity checks (the methane sigma_O rule)
    o = waters[:, 0]
    checks = [
        (float(np.abs(sig[o] - SIGMA_O_NM).max()), "sigma_O"),
        (float(np.abs(eps[o] - EPSILON_O_KJ).max()), "epsilon_O"),
        (float(np.abs(q[waters[:, 1]] - Q_H_E).max()), "q_H"),
        (float(np.abs(q[m_idx] - Q_M_E).max()), "q_M"),
        (float(np.abs(sig[carbon] - SIGMA_C_NM).max()), "sigma_C"),
        (float(np.abs(eps[carbon] - EPSILON_C_KJ).max()), "epsilon_C"),
        (float(np.abs(q[carbon]).max()), "q_C"),
        (float(max(abs(w[k] - VSITE_WEIGHTS[k]) for k in range(3))), "vsite weights"),
    ]
    for err, name in checks:
        if err > 1e-12:
            raise RuntimeError(f"frozen constant mismatch: {name} off by {err:.3e}")

    return dict(charge=q, sigma=sig, epsilon=eps, mass=mass,
                carbon_index=carbon, cage_a=carbon[:60], cage_b=carbon[60:],
                waters=waters, vsite_weights=np.asarray(w))


def exclusions(system):
    """All exclusion pairs; every exception must be a pure exclusion (asserted)."""
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


def xi_of(positions_nm, cage_a, cage_b):
    """``xi = Z_B - Z_A`` from cage COMs (equal masses: plain mean)."""
    pos = np.asarray(positions_nm, dtype=np.float64)
    return float(pos[cage_b, 2].mean() - pos[cage_a, 2].mean())


def local_mean_force(forces, cage_a, cage_b):
    """``f = (1/2)(F_A,z - F_B,z)`` -- the SPEC §2 estimator; physical forces only, no Jacobian."""
    f = np.asarray(forces, dtype=np.float64)
    return 0.5 * (f[..., cage_a, 2].sum(-1) - f[..., cage_b, 2].sum(-1))


def apply_constraints(system, topology, positions_nm, tol=1e-10):
    """Project onto the rigid-water manifold, recompute virtual sites, and validate."""
    import openmm as mm
    import openmm.unit as u

    ctx = mm.Context(system, mm.VerletIntegrator(1e-6),
                     mm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(np.asarray(positions_nm) * u.nanometer)
    ctx.applyConstraints(tol)
    ctx.computeVirtualSites()
    out = np.asarray(ctx.getState(getPositions=True)
                     .getPositions().value_in_unit(u.nanometer), dtype=np.float64)
    del ctx
    return out


def manifest():
    """Every frozen number a reader needs to rebuild the model."""
    return dict(
        n_waters=N_WATERS, n_carbons=N_CARBONS, n_sites=N_SITES,
        sigma_CO_nm=SIGMA_CO_NM, epsilon_CO_kJ=EPSILON_CO_KJ,
        sigma_C_nm=SIGMA_C_NM, epsilon_C_kJ=EPSILON_C_KJ,
        sigma_O_nm=SIGMA_O_NM, epsilon_O_kJ=EPSILON_O_KJ,
        q_H_e=Q_H_E, q_M_e=Q_M_E, r_OH_nm=R_OH_NM, r_HH_nm=R_HH_NM,
        vsite_weights=list(VSITE_WEIGHTS),
        combination="geometric-exact via derived per-particle C (SPEC 1.1, Deviation 1)",
        r66_nm=geometry.R_66_NM, r65_nm=geometry.R_65_NM, mass_C_amu=MASS_C_AMU,
        mu_xi_amu=MU_XI_AMU,
        temperature_K=TEMPERATURE_K, gamma_ps=GAMMA_PS, dt_ps=DT_PS, kT_kJ=kT_kJ(),
        cutoff_nm=CUTOFF_NM, switching="none (paper convention)",
        d_ref_nm=D_REF_NM, xi_domain=[XI_LO_NM, XI_HI_NM], box_aspect=BOX_ASPECT,
        dispersion_correction="off (NVT: zero force); on for NPT only",
        spec="docs/SPEC_c60_water.md", amendment="V2_PREREGISTRATION.md Amendment 16",
    )
