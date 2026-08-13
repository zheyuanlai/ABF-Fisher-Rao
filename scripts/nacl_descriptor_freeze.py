"""SPEC_nacl_water.md §3.2 — reference RDFs and the R0 descriptor freeze.

Runs BEFORE any ABF or screen data exist (the freeze protocol refuses to overwrite).  Ions are
held dissociated (harmonic restraint at 1.2 nm) so the shells are the bulk-ion ones; NVT at the
frozen box; g_NaO, g_ClO, g_ClH accumulated over 0.5 ns; first minima -> R0_ClH / R0_ClO via
``nacl.observables.freeze_descriptors``; sanity numbers recorded.

Usage:
    CUDA_VISIBLE_DEVICES=3 python scripts/nacl_descriptor_freeze.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import openmm as mm                                              # noqa: E402
import openmm.unit as u                                          # noqa: E402

from nacl import observables as nobs                             # noqa: E402
from nacl import system as nsys                                  # noqa: E402

OUT = nsys.REPO / "results/nacl/stage0"

EQUIL_PS = 100.0
PROD_PS = 500.0
SNAP_PS = 0.5
DT_PS = 0.002
R_HOLD_NM = 1.20
RDF_RMAX_NM = 0.80
RDF_DR_NM = 0.002


def main():
    box = json.loads((nsys.REPO / "results/nacl/box/box_manifest.json").read_text())
    L = float(box["L_nm"])

    system, topology, _ = nsys.build_openmm_system(L)
    hold = mm.CustomBondForce("0.5*k*(r-r0)^2")
    hold.addGlobalParameter("k", 20000.0)      # kJ/mol/nm^2
    hold.addGlobalParameter("r0", R_HOLD_NM)
    hold.addBond(0, 1, [])
    hold.setUsesPeriodicBoundaryConditions(True)
    system.addForce(hold)

    integ = mm.LangevinMiddleIntegrator(nsys.TEMPERATURE_K * u.kelvin,
                                        nsys.GAMMA_PS / u.picosecond, DT_PS * u.picosecond)
    integ.setRandomNumberSeed(20260814)
    ctx = mm.Context(system, integ, mm.Platform.getPlatformByName("CUDA"),
                     dict(Precision="mixed"))

    st0 = dict(np.load(nsys.REPO / "results/nacl/box/npt_final_state.npz"))
    pos = st0["positions_nm"].copy()
    # place the ions R_HOLD apart along their current axis, waters untouched; the restraint
    # plus equilibration relaxes any resulting overlap before statistics are taken
    d = pos[1] - pos[0]
    d = d - L * np.round(d / L)
    e = d / np.linalg.norm(d)
    mid = pos[0] + 0.5 * d
    pos[0] = mid - 0.5 * R_HOLD_NM * e
    pos[1] = mid + 0.5 * R_HOLD_NM * e
    ctx.setPositions(pos * u.nanometer)
    ctx.applyConstraints(1e-10)
    mm.LocalEnergyMinimizer.minimize(ctx, maxIterations=200)
    ctx.setVelocitiesToTemperature(nsys.TEMPERATURE_K * u.kelvin, 20260814)

    integ.step(int(EQUIL_PS / DT_PS))

    p = nsys.load_site_params()
    iO = p["waters"][:, 0]
    iH = p["waters"][:, 1:].reshape(-1)
    edges = np.arange(0.0, RDF_RMAX_NM + RDF_DR_NM, RDF_DR_NM)
    counts = dict(NaO=np.zeros(len(edges) - 1), ClO=np.zeros(len(edges) - 1),
                  ClH=np.zeros(len(edges) - 1))
    n_frames = 0
    t0 = time.perf_counter()
    for k in range(int(PROD_PS / SNAP_PS)):
        integ.step(int(SNAP_PS / DT_PS))
        x = np.asarray(ctx.getState(getPositions=True).getPositions()
                       .value_in_unit(u.nanometer))
        def dists(i_center, idx):
            dd = x[idx] - x[i_center]
            dd -= L * np.round(dd / L)
            return np.linalg.norm(dd, axis=1)
        counts["NaO"] += np.histogram(dists(0, iO), bins=edges)[0]
        counts["ClO"] += np.histogram(dists(1, iO), bins=edges)[0]
        counts["ClH"] += np.histogram(dists(1, iH), bins=edges)[0]
        n_frames += 1
        if k % 200 == 0:
            print(f"  {k * SNAP_PS:6.0f} ps  ({time.perf_counter() - t0:5.0f}s)", flush=True)

    r_mid = 0.5 * (edges[:-1] + edges[1:])
    shell = (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    rho = dict(NaO=len(iO) / L ** 3, ClO=len(iO) / L ** 3, ClH=len(iH) / L ** 3)
    g = {k: counts[k] / (n_frames * shell * rho[k]) for k in counts}

    np.savez(OUT / "reference_rdfs.npz", r_nm=r_mid, g_NaO=g["NaO"], g_ClO=g["ClO"],
             g_ClH=g["ClH"], n_frames=n_frames, L_nm=L)

    frozen = nobs.freeze_descriptors(r_mid, g["ClH"], g["ClO"])
    print("frozen:", frozen)

    # sanity record: first-shell peak positions
    peaks = {k: float(r_mid[np.argmax(g[k])]) for k in g}
    print("first-shell peaks (nm):", peaks)
    summary = dict(peaks_nm=peaks, frozen=frozen, n_frames=n_frames,
                   equil_ps=EQUIL_PS, prod_ps=PROD_PS, r_hold_nm=R_HOLD_NM,
                   gpu=os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
                   contended="methane screen on this device (wall-clock only)")
    (OUT / "rdf_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
