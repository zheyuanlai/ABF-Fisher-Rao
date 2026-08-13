"""NaCl/water system definition: the published Talmazan 2025 model, extracted, never retyped.

Frozen by ``docs/SPEC_nacl_water.md`` §1.  This module owns the *definition* of the physical
system; nothing else in the nacl package may restate a force-field number.  The numbers
themselves live in ``results/nacl/stage0/site_params.npz`` — written by
``scripts/nacl_stage0_extract.py`` from the OpenMM ``System`` built out of the published
PSF/parameter files — and are **loaded**, not declared, here.

Two builders, one source
------------------------
1. :func:`build_openmm_system` -- the parity target and the NPT box tool, built from the
   published CHARMM files with our pinned PME parameters;
2. the torch engine (``nacl.nonbonded.NaClNonbonded``), which consumes
   :func:`load_site_params` -- the same arrays the OpenMM parity target reports.

If those two ever read different numbers the parity gate is meaningless; the gate itself
asserts they do not.

Units: nm, kJ/mol, ps, amu, K, elementary charge (project-internal).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
SRC_TUTORIAL = REPO / "cache/talmazan2025/extracted/ABF_tutorial-main/NaCl"
STAGE0 = REPO / "results/nacl/stage0"

# ------------------------------------------------------------------ frozen protocol (SPEC §1)
TEMPERATURE_K = 300.0                   #: published langevinTemp
GAMMA_PS = 1.0                          #: published langevinDamping
DT_PS = 0.002                           #: published timestep; gated at 1 and 2 fs (SPEC §1.2)
DT_FALLBACK_PS = 0.001
PRESSURE_BAR = 1.0                      #: NPT box equilibration only

CUTOFF_NM = 1.20                        #: published cutoff 12 A
SWITCH_NM = 1.00                        #: published switchdist 10 A (OpenMM switch form, SPEC §1.1.7)

#: PME: alpha from OpenMM's rule at the published tolerance 1e-5 and the published cutoff, grid
#: ~1 A spacing as published (order 5 is OpenMM's fixed choice -- declared deviation from NAMD's
#: order 4).  Pinned in BOTH engines so parity does not depend on a box-derived default.
PME_TOLERANCE = 1.0e-5
PME_ALPHA_PER_NM = float(np.sqrt(-np.log(2.0 * PME_TOLERANCE)) / CUTOFF_NM)  # 2.741101...
PME_GRID = (30, 30, 30)
PME_SPLINE_ORDER = 5

N_IONS = 2
ION_INDEX = (0, 1)                      #: (Na, Cl) -- asserted against the PSF at load
N_WATERS = 821
N_SITES = N_IONS + 3 * N_WATERS         #: 2465

#: Published colvar domain and bins (abf.in): [2.0, 14.0] A, width 0.1 A, fullSamples 500.
R_LO_NM = 0.20
R_HI_NM = 1.40
N_GRID = 121                            #: odd; spacing 0.01 nm == published width
FULL_SAMPLES = 500
WALL_LO_NM = 0.20
WALL_HI_NM = 1.40
#: published harmonicWalls: 1 kcal/mol over one 0.1-A colvar width squared
K_WALL_KJ_NM2 = 1.0 * 4.184 / (0.01 ** 2)   # 41840 kJ/mol/nm^2


def kT_kJ(temperature_k=TEMPERATURE_K):
    return 8.31446261815324e-3 * float(temperature_k)


def beta_per_kJ(temperature_k=TEMPERATURE_K):
    return 1.0 / kT_kJ(temperature_k)


def load_site_params():
    """The frozen per-site arrays, exactly as extracted from the published model.

    Returns a dict with ``charge, sigma, epsilon, mass (2465,)``, ``exclusions (2463, 2)``,
    ``waters (821, 3)`` as ``(O, H1, H2)`` and ``box_nm`` (the tutorial's 11-ps NPT box --
    the *production* box comes from our own NPT run and is passed explicitly).
    """
    d = dict(np.load(STAGE0 / "site_params.npz"))
    n = len(d["charge"])
    if n != N_SITES or d["waters"].shape != (N_WATERS, 3) \
            or d["exclusions"].shape != (3 * N_WATERS, 2):
        raise RuntimeError(f"site_params.npz shape mismatch: n={n}, "
                           f"waters={d['waters'].shape}, excl={d['exclusions'].shape}")
    if not (d["charge"][0] == 1.0 and d["charge"][1] == -1.0):
        raise RuntimeError("ion order is not (Na+, Cl-)")
    if float(abs(d["charge"].sum())) > 1e-9:
        raise RuntimeError("system is not neutral")
    return d


def rigid_water_lengths(params=None):
    """``(r_OH, r_OH, r_HH)`` in nm for ``RigidWaterConstraints``, read from the manifest."""
    import json
    m = json.loads((STAGE0 / "model_manifest.json").read_text())
    return [m["system"]["r_OH_nm"], m["system"]["r_OH_nm"], m["system"]["r_HH_nm"]]


def build_openmm_system(box_nm, dispersion_correction=False, pin_pme=True):
    """The published model as an OpenMM ``System`` with our pinned PME -- the parity target.

    ``box_nm`` is the cubic box side.  Rigid water: full triangle constrained (3/water, as the
    published SETTLE water), frozen H-O-H angle force removed (SPEC §1.1.2).
    ``dispersion_correction`` off for NVT parity/production, on for the NPT box run.
    Returns ``(system, topology, psf)``.
    """
    import openmm as mm
    import openmm.app as app
    import openmm.unit as u

    psf = app.CharmmPsfFile(str(SRC_TUTORIAL / "solvate.psf"))
    top22 = SRC_TUTORIAL.parent / "Ethanol-Hydration/WTM-eABF/common/top_all22_prot.inp"
    params = app.CharmmParameterSet(str(top22), str(SRC_TUTORIAL / "par_all22_prot.inp"))
    psf.setBox(box_nm * u.nanometer, box_nm * u.nanometer, box_nm * u.nanometer)

    system = psf.createSystem(params, nonbondedMethod=app.PME,
                              nonbondedCutoff=CUTOFF_NM * u.nanometer,
                              switchDistance=SWITCH_NM * u.nanometer,
                              constraints=None, rigidWater=True)

    forces = {f.__class__.__name__: f for f in system.getForces()}
    if "CustomNonbondedForce" in forces:
        raise RuntimeError("pair-specific CustomNonbondedForce present (NBFIX?)")
    nbf = forces["NonbondedForce"]
    nbf.setUseDispersionCorrection(bool(dispersion_correction))
    if pin_pme:
        nbf.setPMEParameters(PME_ALPHA_PER_NM / u.nanometer, *PME_GRID)

    # frozen-angle removal, exactly as scripts/nacl_stage0_extract.py validated: every angle's
    # O-H pairs and the H-H pair are constrained, so the term is a constant on the manifold
    n_cons = system.getNumConstraints()
    if n_cons != 3 * N_WATERS:
        raise RuntimeError(f"{n_cons} constraints, expected {3 * N_WATERS}")
    for fi in range(system.getNumForces() - 1, -1, -1):
        if isinstance(system.getForce(fi), mm.HarmonicAngleForce):
            system.removeForce(fi)
    for name, counter in dict(HarmonicBondForce="getNumBonds",
                              PeriodicTorsionForce="getNumTorsions",
                              CustomTorsionForce="getNumTorsions",
                              CMAPTorsionForce="getNumTorsions").items():
        if name in forces and getattr(forces[name], counter)():
            raise RuntimeError(f"{name} carries terms; not a bare ions+rigid-water system")
    return system, psf.topology, psf


def assert_openmm_matches_frozen(system):
    """Hard gate: the OpenMM System the parity test runs against carries EXACTLY the frozen
    per-site parameters of ``site_params.npz``.  Run by the parity suite before comparing."""
    import openmm as mm

    p = load_site_params()
    nbf = next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce))
    for i in range(system.getNumParticles()):
        qi, si, ei = nbf.getParticleParameters(i)
        q = qi.value_in_unit_system(mm.unit.md_unit_system)
        s = si.value_in_unit_system(mm.unit.md_unit_system)
        e = ei.value_in_unit_system(mm.unit.md_unit_system)
        if abs(q - p["charge"][i]) > 0 or abs(s - p["sigma"][i]) > 0 or \
                abs(e - p["epsilon"][i]) > 0:
            raise RuntimeError(f"site {i}: OpenMM ({q},{s},{e}) != frozen "
                               f"({p['charge'][i]},{p['sigma'][i]},{p['epsilon'][i]})")
    return True


def apply_constraints(system, positions_nm, tol=1e-10):
    """Project positions onto the rigid-water manifold and validate against the System's own
    constraint list (methane precedent: consumers must not test the builder's rounding)."""
    import openmm as mm
    import openmm.unit as u

    ctx = mm.Context(system, mm.VerletIntegrator(1e-6),
                     mm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(np.asarray(positions_nm) * u.nanometer)
    ctx.applyConstraints(tol)
    out = np.asarray(ctx.getState(getPositions=True).getPositions()
                     .value_in_unit(u.nanometer), dtype=np.float64)
    del ctx
    worst = 0.0
    for k in range(system.getNumConstraints()):
        i, j, d = system.getConstraintParameters(k)
        d = d.value_in_unit(u.nanometer)
        worst = max(worst, abs(float(np.linalg.norm(out[j] - out[i])) - d))
    if worst > 1e-8:
        raise RuntimeError(f"constraint violation {worst:.3e} nm after projection")
    return out


def manifest(box_nm=None, dt_ps=None):
    """Frozen numbers for run manifests, so a result file describes the model it came from.

    ``dt_ps`` records the timestep actually used (the dynamics gate's choice); omitted, the
    published default is reported and labelled as such.
    """
    return dict(
        spec="docs/SPEC_nacl_water.md", amendment="V2_PREREGISTRATION.md Amendment 14",
        n_sites=N_SITES, n_waters=N_WATERS, ion_index=list(ION_INDEX),
        temperature_K=TEMPERATURE_K, gamma_ps=GAMMA_PS,
        dt_ps=(DT_PS if dt_ps is None else float(dt_ps)),
        dt_source=("published default (gate not applied)" if dt_ps is None
                   else "chosen by the dynamics gate"),
        kT_kJ=kT_kJ(), cutoff_nm=CUTOFF_NM, switch_nm=SWITCH_NM,
        pme_tolerance=PME_TOLERANCE, pme_alpha_per_nm=PME_ALPHA_PER_NM,
        pme_grid=list(PME_GRID), pme_spline_order=PME_SPLINE_ORDER,
        R_lo_nm=R_LO_NM, R_hi_nm=R_HI_NM, n_grid=N_GRID,
        wall_lo_nm=WALL_LO_NM, wall_hi_nm=WALL_HI_NM, k_wall_kJ_nm2=K_WALL_KJ_NM2,
        full_samples=FULL_SAMPLES, box_nm=box_nm,
        site_params="results/nacl/stage0/site_params.npz",
        model_manifest="results/nacl/stage0/model_manifest.json",
    )
