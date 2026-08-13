"""Stage 0A/0C — extract the published Talmazan 2025 NaCl model into an OpenMM System.

Source of truth: the Supporting-Information archive of Talmazan, Fu, Zhou, Henin, Gumbart &
Chipot, J. Phys. Chem. B 129, 9913-9928 (2025) -- the `NaCl/` tutorial directory.  Nothing in
this script types a force-field number by hand: the PSF, PDB and CHARMM parameter file shipped
with the paper are loaded verbatim, OpenMM builds the System, and every parameter the torch
engine will consume is read back OUT of that System (the methane sigma_O lesson, SPEC methane
Section 1).

Outputs (to results/nacl/stage0/):
  model_manifest.json   -- every frozen number, machine-readable
  site_params.npz       -- charge/sigma/epsilon/mass arrays + exclusion pairs, the torch input
  equilibrate_state.npz -- positions/velocities/box parsed from the NAMD binary restart files
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "cache/talmazan2025/extracted/ABF_tutorial-main/NaCl"
OUT = REPO / "results/nacl/stage0"

# Published NAMD protocol (abf.conf), recorded here for the manifest -- NOT retyped physics:
# these are protocol settings, and each is quoted from the file it came from.
PUBLISHED = dict(
    temperature_K=300.0,            # abf.conf: langevinTemp 300.0
    langevin_damping_ps=1.0,        # abf.conf: langevinDamping 1.0
    timestep_fs=2.0,                # abf.conf: timestep 2.0
    cutoff_A=12.0,                  # abf.conf: cutoff 12.0
    switch_A=10.0,                  # abf.conf: switchdist 10.0 (switching on)
    pme_tolerance=1.0e-5,           # abf.conf: PMETolerance 10e-6
    pme_interp_order=4,             # abf.conf: PMEInterpOrder 4
    pme_grid_spacing_A=1.0,         # abf.conf: PMEGridSpacing 1.0
    exclude="scaled1-4",            # abf.conf (no 1-4 pairs exist in this system)
    one_four_scaling=1.0,           # abf.conf: 1-4scaling 1.0
    rigidbonds="all",               # abf.conf: rigidbonds all
    ensemble="NPT (Langevin piston, 1 atm)",  # abf.conf: langevinpiston on
    run_steps=50_000_000,           # abf.conf: run 50000000  -> 100 ns at 2 fs
    colvar_lower_A=2.0,             # abf.in: lowerBoundary 2.0
    colvar_upper_A=14.0,            # abf.in: upperBoundary 14.0
    colvar_width_A=0.1,             # abf.in: width 0.1
    full_samples=500,               # abf.in: fullSamples 500
    wall_force_kcal=1.0,            # abf.in: harmonicWalls forceConstant 1.0
    hide_jacobian=False,            # abf.in: absent -> Colvars default -> PMF is F(r)
)


def read_namd_bin(path):
    """NAMD binary .coor/.vel: int32 natoms + natoms*3 float64, little-endian.  Units: Angstrom
    for .coor; NAMD-internal velocity units for .vel (Angstrom / (1/TIMEFACTOR fs))."""
    raw = Path(path).read_bytes()
    (n,) = struct.unpack("<i", raw[:4])
    data = np.frombuffer(raw[4:], dtype="<f8")
    if data.size != 3 * n:
        raise ValueError(f"{path}: {data.size} doubles for {n} atoms")
    return data.reshape(n, 3)


def read_xsc(path):
    for line in Path(path).read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        f = [float(t) for t in line.split()]
        return dict(step=int(f[0]), a=f[1], b=f[5], c=f[9])
    raise ValueError(f"no data line in {path}")


def main():
    import openmm as mm
    import openmm.app as app
    import openmm.unit as u

    OUT.mkdir(parents=True, exist_ok=True)

    xsc = read_xsc(SRC / "output/equilibrate.xsc")
    box_A = (xsc["a"], xsc["b"], xsc["c"])
    print(f"equilibrate.xsc: step {xsc['step']}, box {box_A} A")

    psf = app.CharmmPsfFile(str(SRC / "solvate.psf"))
    # The NaCl folder ships only the parameter file; the matching CHARMM22 topology (needed by
    # OpenMM for the MASS/atom-type records) is shipped elsewhere in the SAME SI archive.
    top22 = SRC.parent / "Ethanol-Hydration/WTM-eABF/common/top_all22_prot.inp"
    params = app.CharmmParameterSet(str(top22), str(SRC / "par_all22_prot.inp"))
    psf.setBox(*[b * u.angstrom for b in box_A])

    system = psf.createSystem(
        params,
        nonbondedMethod=app.PME,
        nonbondedCutoff=PUBLISHED["cutoff_A"] * u.angstrom,
        switchDistance=PUBLISHED["switch_A"] * u.angstrom,
        constraints=None,
        rigidWater=True,
    )

    forces = {f.__class__.__name__: f for f in system.getForces()}
    print("forces:", list(forces))
    nbf = forces["NonbondedForce"]
    # NBFIX would surface as a CustomNonbondedForce next to the NonbondedForce
    if "CustomNonbondedForce" in forces:
        raise RuntimeError("pair-specific CustomNonbondedForce present (NBFIX?)")
    # CharmmPsfFile adds empty bonded containers unconditionally; all must carry ZERO terms
    # except HarmonicAngleForce, which keeps one H-O-H angle per water even for rigid water.
    # NAMD keeps that term too, but `rigidbonds all` runs water by SETTLE, which fixes all
    # three distances -- the angle term is geometrically frozen and contributes zero force on
    # the constraint manifold.  We reproduce NAMD's *effective* water: assert the triangle is
    # fully constrained, then REMOVE the inert angle force (declared in the manifest).
    counters = dict(HarmonicBondForce="getNumBonds",
                    PeriodicTorsionForce="getNumTorsions", CustomTorsionForce="getNumTorsions",
                    CMAPTorsionForce="getNumTorsions")
    for name, counter in counters.items():
        if name in forces:
            n_terms = getattr(forces[name], counter)()
            if n_terms:
                raise RuntimeError(f"{name} carries {n_terms} terms; expected a bare "
                                   "ions+rigid-water system")

    nbf.setUseDispersionCorrection(False)  # NVT: additive constant, zero force (methane rule)

    # ---- rigid water: constraint pattern, and removal of the frozen angle term -------------
    cons_pairs = {}
    for k in range(system.getNumConstraints()):
        ci, cj, d = system.getConstraintParameters(k)
        cons_pairs[tuple(sorted((ci, cj)))] = d.value_in_unit(u.nanometer)

    angf = forces["HarmonicAngleForce"]
    theta0 = None
    for k in range(angf.getNumAngles()):
        ai, aj, ak_, th, kk = angf.getAngleParameters(k)
        th = th.value_in_unit(u.radian)
        if theta0 is None:
            theta0 = th
        elif abs(th - theta0) > 1e-12:
            raise RuntimeError("non-uniform angle terms; not a pure water system")
        for pair in ((aj, ai), (aj, ak_)):
            if tuple(sorted(pair)) not in cons_pairs:
                raise RuntimeError(f"angle {k}: O-H pair {pair} not constrained")
    n_hh_added = 0
    for res in psf.topology.residues():
        if res.name in ("TIP3", "HOH", "WAT"):
            idx = {a.name: a.index for a in res.atoms()}
            oh = cons_pairs[tuple(sorted((idx["O"], idx["H1"])))]
            hh_key = tuple(sorted((idx["H1"], idx["H2"])))
            if hh_key not in cons_pairs:
                # NAMD SETTLE fixes the full triangle; add the H-H constraint it implies
                r_hh_settle = 2.0 * oh * np.sin(0.5 * theta0)
                system.addConstraint(hh_key[0], hh_key[1], r_hh_settle * u.nanometer)
                cons_pairs[hh_key] = r_hh_settle
                n_hh_added += 1
    for fi in range(system.getNumForces() - 1, -1, -1):
        if isinstance(system.getForce(fi), mm.HarmonicAngleForce):
            system.removeForce(fi)
    print(f"rigid water: theta0 {np.degrees(theta0):.4f} deg, added {n_hh_added} H-H "
          f"constraints, removed frozen angle force")

    n = system.getNumParticles()
    q = np.zeros(n); sig = np.zeros(n); eps = np.zeros(n); mass = np.zeros(n)
    for i in range(n):
        qi, si, ei = nbf.getParticleParameters(i)
        q[i] = qi.value_in_unit_system(mm.unit.md_unit_system)
        sig[i] = si.value_in_unit_system(mm.unit.md_unit_system)
        eps[i] = ei.value_in_unit_system(mm.unit.md_unit_system)
        mass[i] = system.getParticleMass(i).value_in_unit_system(mm.unit.md_unit_system)

    excl = []
    for k in range(nbf.getNumExceptions()):
        i, j, qq, s_, e_ = nbf.getExceptionParameters(k)
        qq = qq.value_in_unit_system(mm.unit.md_unit_system)
        e_ = e_.value_in_unit_system(mm.unit.md_unit_system)
        if qq != 0.0 or e_ != 0.0:
            raise RuntimeError(f"exception {k} not a pure exclusion: qq={qq}, eps={e_}")
        excl.append((i, j))
    excl = np.asarray(excl, dtype=np.int64)

    atoms = list(psf.topology.atoms())
    names = [a.name for a in atoms]
    resnames = [a.residue.name for a in atoms]
    ion_idx = [i for i, r in enumerate(resnames) if r in ("SOD", "CLA")]
    assert names[0] == "SOD" and names[1] == "CLA" and ion_idx == [0, 1]
    n_waters = sum(1 for r in psf.topology.residues() if r.name in ("TIP3", "HOH", "WAT"))
    print(f"{n} particles: ions {ion_idx}, {n_waters} waters, "
          f"{system.getNumConstraints()} constraints, {len(excl)} exclusions")
    assert system.getNumConstraints() == 3 * n_waters
    assert len(excl) == 3 * n_waters
    total_q = q.sum()
    print(f"total charge {total_q:+.6f} e")
    assert abs(total_q) < 1e-9

    # water triplets (O, H1, H2) and their rigid geometry read from the constraints themselves
    cons = {}
    for k in range(system.getNumConstraints()):
        i, j, d = system.getConstraintParameters(k)
        cons[(i, j)] = d.value_in_unit(u.nanometer)
    d_vals = sorted(set(round(v, 10) for v in cons.values()))
    print("constraint lengths (nm):", d_vals)

    waters = []
    for res in psf.topology.residues():
        if res.name in ("TIP3", "HOH", "WAT"):
            idx = {a.name: a.index for a in res.atoms()}
            waters.append((idx["O"], idx["H1"], idx["H2"]))
    waters = np.asarray(waters, dtype=np.int64)
    r_oh = cons[(int(waters[0][0]), int(waters[0][1]))]
    r_hh = cons[(int(waters[0][1]), int(waters[0][2]))] if (int(waters[0][1]), int(waters[0][2])) in cons \
        else cons[(int(waters[0][2]), int(waters[0][1]))]

    # ---- parse the published equilibrated state --------------------------------------------
    pos_A = read_namd_bin(SRC / "output/equilibrate.coor")
    vel_namd = read_namd_bin(SRC / "output/equilibrate.vel")
    assert pos_A.shape == (n, 3)
    pos_nm = pos_A * 0.1

    # ---- single-point energy on the Reference platform, as the recorded anchor -------------
    ctx = mm.Context(system, mm.VerletIntegrator(1e-6), mm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(pos_nm * u.nanometer)
    ctx.applyConstraints(1e-10)
    pos_con = np.asarray(ctx.getState(getPositions=True).getPositions()
                         .value_in_unit(u.nanometer), dtype=np.float64)
    e0 = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
    pme_a, nx, ny, nz = nbf.getPMEParametersInContext(ctx)
    pme_alpha = pme_a.value_in_unit(u.nanometer ** -1) if u.is_quantity(pme_a) else float(pme_a)
    print(f"reference single-point energy {e0:.6f} kJ/mol")
    print(f"PME in context: alpha {pme_alpha} /nm, grid ({nx},{ny},{nz})")
    del ctx

    r_ion_A = float(np.linalg.norm(pos_A[1] - pos_A[0]))
    print(f"published equilibrated ion separation: {r_ion_A:.4f} A")

    np.savez(OUT / "site_params.npz",
             charge=q, sigma=sig, epsilon=eps, mass=mass,
             exclusions=excl, waters=waters,
             box_nm=np.array([b * 0.1 for b in box_A]))
    np.savez(OUT / "equilibrate_state.npz",
             positions_nm=pos_nm, positions_constrained_nm=pos_con,
             velocities_namd=vel_namd, box_nm=np.array([b * 0.1 for b in box_A]))

    manifest = dict(
        source=dict(
            paper="Talmazan, Fu, Zhou, Henin, Gumbart, Chipot, J. Phys. Chem. B 129, "
                  "9913-9928 (2025), doi 10.1021/acs.jpcb.5c04333",
            archive="NIHMS2186658-supplement-tutorial_files.zip (PMC13284794)",
            archive_sha256="f33a8fce86bc9fb7c85afd1647a81bb66d5bd9118de2bde721796ca988a5d94c",
            file_hashes="cache/talmazan2025/nacl_file_hashes.sha256",
        ),
        published_protocol=PUBLISHED,
        system=dict(
            n_particles=n, n_waters=int(n_waters), ion_indices=[0, 1],
            ion_names=["SOD", "CLA"], water_model="CHARMM TIP3P (H carries LJ)",
            total_charge_e=float(total_q),
            n_constraints=int(system.getNumConstraints()),
            n_exclusions=int(len(excl)),
            r_OH_nm=float(r_oh), r_HH_nm=float(r_hh),
            nbfix="none (asserted: no CustomNonbondedForce)",
        ),
        ions=dict(
            Na=dict(charge_e=float(q[0]), sigma_nm=float(sig[0]),
                    epsilon_kJ=float(eps[0]), mass_amu=float(mass[0])),
            Cl=dict(charge_e=float(q[1]), sigma_nm=float(sig[1]),
                    epsilon_kJ=float(eps[1]), mass_amu=float(mass[1])),
        ),
        water_sites=dict(
            O=dict(charge_e=float(q[2]), sigma_nm=float(sig[2]), epsilon_kJ=float(eps[2]),
                   mass_amu=float(mass[2])),
            H=dict(charge_e=float(q[3]), sigma_nm=float(sig[3]), epsilon_kJ=float(eps[3]),
                   mass_amu=float(mass[3])),
        ),
        equilibrated_state=dict(
            box_A=list(box_A), xsc_step=xsc["step"],
            ion_separation_A=r_ion_A,
            reference_energy_kJ=float(e0),
        ),
        pme_in_context=dict(alpha_per_nm=float(pme_alpha), grid=[int(nx), int(ny), int(nz)],
                            note="OpenMM default at ewaldErrorTolerance 5e-4; frozen values "
                                 "are pinned by src/nacl/system.py after this measurement"),
        openmm_version=mm.version.version,
    )
    (OUT / "model_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {OUT}/model_manifest.json")


if __name__ == "__main__":
    sys.exit(main())
