"""Stage 1 -- freeze the methane/SPC-E production box (SPEC_methane_water.md §1.3).

NPT is used **only** to fix the volume.  All reference, ABF and mFR production is NVT at the
frozen box, so the target measure is exactly the canonical measure of the book and no barostat
variable enters the mathematics.

    equilibrate NPT at 298 K / 1 bar  ->  measure <V> over a preregistered window
    ->  L = <V>^(1/3), frozen  ->  everything downstream is NVT at that L

Two details that are not cosmetic:

* **The analytic dispersion correction is ON here and only here.**  It is a function of volume, so
  it belongs in the pressure; in NVT it is an additive constant with zero force and is switched
  off (``methane.system``).
* **PME parameters are unpinned during NPT** because the box is changing, then **re-pinned for the
  frozen box** and written to the manifest.  ``methane.system`` pins them for parity, and a value
  chosen for a 2.61 nm box is not the value for a 2.49 nm one -- carrying the stale pair over
  would put the engine and OpenMM on different Ewald splittings while every test still passed.

Minimisation is mandatory: raw ``addSolvent`` output carries forces to 1e6 kJ/mol/nm and a single
0.5 fs kick against those injects ~500 nm/ps.

Usage:
    CUDA_VISIBLE_DEVICES=3 python scripts/methane_box.py --out results/methane/box
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import openmm as mm                                              # noqa: E402
import openmm.unit as u                                          # noqa: E402

from methane import system as msys                               # noqa: E402


def _platform(name=None):
    if name:
        return mm.Platform.getPlatformByName(name), {"Precision": "mixed"}
    for cand in ("CUDA", "OpenCL", "CPU"):
        try:
            p = mm.Platform.getPlatformByName(cand)
            return p, ({"Precision": "mixed"} if cand in ("CUDA", "OpenCL") else {})
        except Exception:
            continue
    raise RuntimeError("no OpenMM platform")


def gpu_state():
    """Record the device's idle state with every measurement (Amendment 12.4)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
        return dict(visible=os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"), nvidia_smi=out)
    except Exception as exc:                                      # pragma: no cover
        return dict(visible=os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"), error=str(exc))


def pme_parameters_for_box(topology, box_nm, platform_name=None):
    """Ask OpenMM which ``(alpha, nx, ny, nz)`` it selects for this box, and return them.

    Queried rather than reimplemented so the engine inherits OpenMM's own choice exactly.
    """
    system = msys.build_system(topology, pin_pme=False)
    system.setDefaultPeriodicBoxVectors(
        mm.Vec3(box_nm, 0, 0) * u.nanometer,
        mm.Vec3(0, box_nm, 0) * u.nanometer,
        mm.Vec3(0, 0, box_nm) * u.nanometer)
    nbf = next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce))
    plat, props = _platform(platform_name)
    ctx = mm.Context(system, mm.VerletIntegrator(1e-6), plat, props)
    alpha, nx, ny, nz = nbf.getPMEParametersInContext(ctx)
    del ctx
    alpha = alpha.value_in_unit(u.nanometer ** -1) if u.is_quantity(alpha) else float(alpha)
    return float(alpha), (int(nx), int(ny), int(nz))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/methane/box")
    ap.add_argument("--equil-ps", type=float, default=500.0, help="discarded")
    ap.add_argument("--prod-ps", type=float, default=1000.0, help="volume averaging window")
    ap.add_argument("--seed", type=int, default=20260812)
    ap.add_argument("--platform", default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    t_start = time.time()
    print(f"[build] solvating {msys.N_WATERS} SPC/E waters around 2 methanes ...", flush=True)
    mod = msys.build_modeller(r0_nm=0.55, seed=args.seed)
    pos0 = np.asarray(mod.positions.value_in_unit(u.nanometer))

    # NPT: dispersion correction ON (it belongs in the pressure), PME unpinned (box moves)
    system = msys.build_system(mod.topology, dispersion_correction=True, pin_pme=False)
    pos0 = msys.apply_constraints(system, mod.topology, pos0)

    barostat = mm.MonteCarloBarostat(msys.PRESSURE_BAR * u.bar,
                                     msys.TEMPERATURE_K * u.kelvin, 25)
    barostat.setRandomNumberSeed(args.seed)
    system.addForce(barostat)

    integ = mm.LangevinMiddleIntegrator(msys.TEMPERATURE_K * u.kelvin,
                                        msys.GAMMA_PS / u.picosecond,
                                        msys.DT_PS * u.picoseconds)
    integ.setRandomNumberSeed(args.seed)
    plat, props = _platform(args.platform)
    ctx = mm.Context(system, integ, plat, props)
    ctx.setPositions(pos0 * u.nanometer)
    print(f"[platform] {plat.getName()}  {props}", flush=True)

    e0 = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
    f0 = np.abs(np.asarray(ctx.getState(getForces=True).getForces()
                           .value_in_unit(u.kilojoule_per_mole / u.nanometer))).max()
    print(f"[minimise] before: E = {e0:.1f} kJ/mol   max|F| = {f0:.3e}", flush=True)
    mm.LocalEnergyMinimizer.minimize(ctx, 1.0, 5000)
    e1 = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
    f1 = np.abs(np.asarray(ctx.getState(getForces=True).getForces()
                           .value_in_unit(u.kilojoule_per_mole / u.nanometer))).max()
    print(f"[minimise] after : E = {e1:.1f} kJ/mol   max|F| = {f1:.3e}", flush=True)
    if f1 > 1e5:
        raise RuntimeError(f"minimisation left max|F| = {f1:.3e}; dynamics would explode")

    ctx.setVelocitiesToTemperature(msys.TEMPERATURE_K * u.kelvin, args.seed)

    n_equil = int(round(args.equil_ps / msys.DT_PS))
    n_prod = int(round(args.prod_ps / msys.DT_PS))
    sample_every = int(round(0.5 / msys.DT_PS))          # every 0.5 ps

    print(f"[npt] equilibrating {args.equil_ps:.0f} ps (discarded) ...", flush=True)
    t0 = time.time()
    integ.step(n_equil)
    print(f"[npt] equilibration done in {time.time()-t0:.0f}s", flush=True)

    print(f"[npt] production {args.prod_ps:.0f} ps, sampling V every 0.5 ps ...", flush=True)
    vols, temps = [], []
    dof = 3 * msys.N_SITES - 3 * msys.N_WATERS - 3
    t0 = time.time()
    for k in range(n_prod // sample_every):
        integ.step(sample_every)
        st = ctx.getState(getEnergy=True)
        bv = st.getPeriodicBoxVectors()
        vols.append(bv[0][0].value_in_unit(u.nanometer)
                    * bv[1][1].value_in_unit(u.nanometer)
                    * bv[2][2].value_in_unit(u.nanometer))
        ke = st.getKineticEnergy().value_in_unit(u.kilojoule_per_mole)
        temps.append(2 * ke / (dof * 8.31446261815324e-3))
    wall = time.time() - t0
    vols = np.asarray(vols)
    temps = np.asarray(temps)

    # block average for a usable error bar on <V>
    nb = 10
    blocks = np.asarray([b.mean() for b in np.array_split(vols, nb)])
    v_mean = float(vols.mean())
    v_err = float(blocks.std(ddof=1) / np.sqrt(nb))
    L = v_mean ** (1.0 / 3.0)
    L_err = v_err / (3.0 * v_mean ** (2.0 / 3.0))

    n_w = msys.N_WATERS
    mass_g = (n_w * 18.01528 + 2 * msys.MASS_METHANE_AMU) / 6.02214076e23
    density = mass_g / (v_mean * 1e-21)                          # g/cm^3

    print(f"\n[result] <V> = {v_mean:.4f} +- {v_err:.4f} nm^3", flush=True)
    print(f"[result] L   = {L:.6f} +- {L_err:.6f} nm     (L/2 = {L/2:.4f}, cutoff "
          f"{msys.CUTOFF_NM})", flush=True)
    print(f"[result] rho = {density:.4f} g/cm^3", flush=True)
    print(f"[result] T   = {temps.mean():.2f} +- {temps.std():.2f} K", flush=True)

    if L / 2.0 <= msys.CUTOFF_NM:
        raise RuntimeError(f"cutoff {msys.CUTOFF_NM} exceeds L/2 = {L/2:.4f}; minimum image broken")

    alpha, grid = pme_parameters_for_box(mod.topology, L, args.platform)
    print(f"[pme] for the frozen box: alpha = {alpha:.12f} /nm, grid = {grid}", flush=True)
    print(f"[pme] previously pinned : alpha = {msys.PME_ALPHA_PER_NM:.12f} /nm, "
          f"grid = {msys.PME_GRID}", flush=True)

    # final configuration, rescaled into the frozen box and re-constrained
    st = ctx.getState(getPositions=True)
    pos = np.asarray(st.getPositions().value_in_unit(u.nanometer))
    bv = st.getPeriodicBoxVectors()
    L_now = bv[0][0].value_in_unit(u.nanometer)
    pos = pos * (L / L_now)                                      # isotropic rescale
    frozen = msys.build_system(mod.topology, pin_pme=False)
    frozen.setDefaultPeriodicBoxVectors(mm.Vec3(L, 0, 0) * u.nanometer,
                                        mm.Vec3(0, L, 0) * u.nanometer,
                                        mm.Vec3(0, 0, L) * u.nanometer)
    pos = msys.apply_constraints(frozen, mod.topology, pos)

    np.savez_compressed(os.path.join(args.out, "box.npz"),
                        positions_nm=pos, volumes_nm3=vols, temperatures_K=temps)
    manifest = dict(
        stage="box", spec="docs/SPEC_methane_water.md §1.3",
        amendment="V2_PREREGISTRATION.md Amendments 11-13",
        box_L_nm=L, box_L_err_nm=L_err, volume_nm3=v_mean, volume_err_nm3=v_err,
        density_g_cm3=density, temperature_K=float(temps.mean()),
        temperature_sd_K=float(temps.std()),
        half_box_nm=L / 2.0, cutoff_nm=msys.CUTOFF_NM,
        pme_alpha_per_nm=alpha, pme_grid=list(grid),
        pme_alpha_previously_pinned=msys.PME_ALPHA_PER_NM,
        pme_grid_previously_pinned=list(msys.PME_GRID),
        equil_ps=args.equil_ps, prod_ps=args.prod_ps, seed=args.seed,
        dispersion_correction="ON for NPT only",
        minimised_max_force_before=float(f0), minimised_max_force_after=float(f1),
        platform=plat.getName(), wall_seconds=float(time.time() - t_start),
        production_ns_per_day=float(args.prod_ps / 1000.0 / (wall / 86400.0)),
        gpu=gpu_state(), model=msys.manifest(),
        openmm_version=mm.version.version, python=platform.python_version(),
        git_commit=subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                  text=True).stdout.strip(),
    )
    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\n[done] {time.time()-t_start:.0f}s total -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
