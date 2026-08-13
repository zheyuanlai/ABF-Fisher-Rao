"""SPEC_nacl_water.md §1.3 — the one-off NPT run that freezes the production NVT box.

OpenMM CUDA, MonteCarloBarostat at 1 bar / 300 K, dispersion correction ON (it belongs in the
pressure), 2 fs LangevinMiddle, starting from the published equilibrated restart.  0.5 ns
discard + 1.0 ns average of <V>; ``L = <V>^(1/3)`` is frozen into
``results/nacl/box/box_manifest.json`` and every downstream stage reads it from there.

Also applies the §1.3 finite-size gate: ``R_hi <= 0.97 (L/2)`` or the domain is truncated
(recorded here, decided before any reference exists).

Usage:
    CUDA_VISIBLE_DEVICES=3 python scripts/nacl_box_npt.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import openmm as mm                                              # noqa: E402
import openmm.app as app                                         # noqa: E402
import openmm.unit as u                                          # noqa: E402

from nacl import system as nsys                                  # noqa: E402

OUT = nsys.REPO / "results/nacl/box"

DISCARD_NS = 0.5
AVERAGE_NS = 1.0
DT_PS = 0.002
SAMPLE_EVERY_PS = 1.0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    state = dict(np.load(nsys.STAGE0 / "equilibrate_state.npz"))
    L0 = float(state["box_nm"][0])

    system, topology, _ = nsys.build_openmm_system(L0, dispersion_correction=True)
    system.addForce(mm.MonteCarloBarostat(nsys.PRESSURE_BAR * u.bar,
                                          nsys.TEMPERATURE_K * u.kelvin, 25))
    integ = mm.LangevinMiddleIntegrator(nsys.TEMPERATURE_K * u.kelvin,
                                        nsys.GAMMA_PS / u.picosecond, DT_PS * u.picosecond)
    integ.setRandomNumberSeed(20260813)
    platform = mm.Platform.getPlatformByName("CUDA")
    ctx = mm.Context(system, integ, platform, dict(Precision="mixed"))
    ctx.setPositions(state["positions_constrained_nm"] * u.nanometer)
    ctx.applyConstraints(1e-10)
    ctx.setVelocitiesToTemperature(nsys.TEMPERATURE_K * u.kelvin, 20260813)

    n_total = int(round((DISCARD_NS + AVERAGE_NS) * 1000.0 / DT_PS))
    n_chunk = int(round(SAMPLE_EVERY_PS / DT_PS))
    vols, temps = [], []
    t0 = time.perf_counter()
    for k in range(n_total // n_chunk):
        integ.step(n_chunk)
        st = ctx.getState(getEnergy=True)
        box = ctx.getState().getPeriodicBoxVectors()
        v = (box[0][0] * box[1][1] * box[2][2]).value_in_unit(u.nanometer ** 3)
        vols.append(v)
        ke = st.getKineticEnergy().value_in_unit(u.kilojoule_per_mole)
        ndof = 3 * nsys.N_SITES - system.getNumConstraints() - 3
        temps.append(2.0 * ke / (ndof * 8.31446261815324e-3))
        if k % 100 == 0:
            el = time.perf_counter() - t0
            print(f"  {k * SAMPLE_EVERY_PS:7.0f} ps  V = {v:8.4f} nm^3  "
                  f"L = {v ** (1/3):7.5f} nm  T = {temps[-1]:6.1f} K  ({el:5.0f}s)", flush=True)

    vols = np.asarray(vols)
    temps = np.asarray(temps)
    n_discard = int(DISCARD_NS * 1000.0 / SAMPLE_EVERY_PS)
    v_mean = float(vols[n_discard:].mean())
    v_sem = float(vols[n_discard:].std(ddof=1) / np.sqrt(len(vols[n_discard:])))
    L = v_mean ** (1.0 / 3.0)

    # ---- finite-size gate (SPEC §1.3) ------------------------------------------------------
    r_hi_max = 0.97 * 0.5 * L
    if nsys.R_HI_NM <= r_hi_max:
        domain = dict(R_lo_nm=nsys.R_LO_NM, R_hi_nm=nsys.R_HI_NM, truncated=False)
    else:
        # truncate to the largest grid edge below the gate; grid spacing is 0.01 nm
        r_hi_new = float(np.floor(r_hi_max / 0.01) * 0.01)
        domain = dict(R_lo_nm=nsys.R_LO_NM, R_hi_nm=r_hi_new, truncated=True,
                      rule="R_hi <= 0.97 L/2, floored to the 0.01 nm grid")
    print(f"L = {L:.6f} nm  (<V> = {v_mean:.4f} +- {v_sem:.4f} nm^3)")
    print(f"finite-size gate: R_hi <= {r_hi_max:.4f} nm -> {domain}")

    st = ctx.getState(getPositions=True, getVelocities=True)
    np.savez(OUT / "npt_final_state.npz",
             positions_nm=np.asarray(st.getPositions().value_in_unit(u.nanometer)),
             velocities=np.asarray(st.getVelocities().value_in_unit(u.nanometer / u.picosecond)),
             box_L_nm=np.array([L]), volume_trace_nm3=vols, temperature_trace=temps)

    manifest = dict(
        protocol=dict(discard_ns=DISCARD_NS, average_ns=AVERAGE_NS, dt_ps=DT_PS,
                      barostat="MonteCarloBarostat 1 bar / 300 K / 25",
                      dispersion_correction=True, start="published equilibrate state",
                      seed=20260813, platform="CUDA mixed",
                      gpu=os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
                      contended="methane screen running on this device (recorded, "
                                "affects wall-clock only)"),
        V_mean_nm3=v_mean, V_sem_nm3=v_sem, L_nm=L,
        T_mean_K=float(temps[n_discard:].mean()),
        finite_size_gate=domain,
        frozen="all reference/screen/production NVT at this L",
    )
    (OUT / "box_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {OUT}/box_manifest.json")


if __name__ == "__main__":
    main()
