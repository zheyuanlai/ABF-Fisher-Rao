"""NPT box equilibration for the C60 study -- SPEC_c60_water.md §1.4, run once.

Builds exactly 1282 TIP4P-Ew waters around the two fixed cages at d_ref = 2.428 nm, runs NPT
(300 K, 1 bar, isotropic MC barostat so the built aspect A = Lz/Lx is preserved), discards
0.5 ns, measures <V> over 1.0 ns, freezes (Lx, Ly=Lx, Lz=A Lx), and writes the frozen-box
starting configuration (molecule-COM rescaled final frame, cages re-placed analytically).

The barostat vs the fixed cages (both failure modes measured 2026-08-14):
  * OpenMM's MC barostat moves massless particles: an absolute-position guard fired at
    0.17 nm/10 ps.  Worse, it scales each carbon **individually** -- the cages are not
    "molecules" to the barostat because massless particles cannot carry constraints -- so the
    cage geometry dilated with the box (bond error 4.9e-3 nm by 100 ps).
  * Remedy: a **cage projector** -- every picosecond the cages are re-placed analytically,
    rigid and centred, at fixed fractional axial separation in the instantaneous box
    (exactly what rigid molecule-COM scaling would have done).  The pre-projection distortion
    accumulated per ps is recorded and asserted small (< 2e-3 nm), so a pathological barostat
    cannot hide behind the projector.  The frozen-box configuration re-places the cages at
    exactly d_ref, so downstream stages see d_ref precisely.
  * the frozen box must pass the SPEC §1.4 geometric guards.

Output: results/c60/box/{npt_trace.npz, frozen_box.npz, manifest.json, RESULT.md}

Usage:  CUDA_VISIBLE_DEVICES=3 python scripts/c60_npt_box.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from c60 import geometry, system as csys  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "c60", "box")

DISCARD_PS = 500.0
MEASURE_PS = 1000.0
SAMPLE_PS = 1.0
DT_PS = 0.002


def main():
    import openmm as mm
    import openmm.app as app
    import openmm.unit as u

    os.makedirs(OUT, exist_ok=True)
    dev = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if dev != "3":
        raise SystemExit(f"CUDA_VISIBLE_DEVICES={dev!r}; SPEC §11 pins this study to GPU 3")

    mod = csys.build_modeller()
    bv0 = mod.topology.getPeriodicBoxVectors().value_in_unit(u.nanometer)
    lx0, lz0 = float(bv0[0][0]), float(bv0[2][2])
    aspect = lz0 / lx0

    system = csys.build_system(mod.topology, dispersion_correction=True, fix_cages=True)
    system.addForce(mm.MonteCarloBarostat(csys.PRESSURE_BAR * u.bar,
                                          csys.TEMPERATURE_K * u.kelvin, 25))
    integ = mm.LangevinMiddleIntegrator(csys.TEMPERATURE_K * u.kelvin,
                                        csys.GAMMA_PS / u.picosecond, DT_PS * u.picoseconds)
    platform = mm.Platform.getPlatformByName("CUDA")
    ctx = mm.Context(system, integ, platform, dict(Precision="mixed"))
    ctx.setPositions(mod.positions)
    ctx.applyConstraints(1e-10)
    ctx.computeVirtualSites()
    ctx.setVelocitiesToTemperature(csys.TEMPERATURE_K * u.kelvin, 20260814)

    p = csys.site_parameters(system, mod.topology)
    carbon = p["carbon_index"]
    cage_a, cage_b = p["cage_a"], p["cage_b"]

    def cage_state():
        """(fractional COMs (2,3), d_com, max bond-length error) -- the true invariants."""
        pos = np.asarray(ctx.getState(getPositions=True).getPositions()
                         .value_in_unit(u.nanometer))
        box = ctx.getState().getPeriodicBoxVectors().value_in_unit(u.nanometer)
        L = np.array([float(box[0][0]), float(box[1][1]), float(box[2][2])])
        coms = np.stack([pos[cage_a].mean(0), pos[cage_b].mean(0)]) / L
        d = csys.xi_of(pos, cage_a, cage_b)
        b65, b66 = geometry.bond_lengths(pos[cage_a] - pos[cage_a].mean(0))
        err = max(abs(b65 - geometry.R_65_NM).max(), abs(b66 - geometry.R_66_NM).max())
        return coms, d, float(err)

    frac0, d0, err0 = cage_state()
    if err0 > 1e-6:
        raise RuntimeError(f"cage geometry already distorted at start: {err0:.2e} nm")
    frac_d_ref = csys.D_REF_NM / lz0            # fixed fractional axial separation

    def project_cages():
        """Re-place both cages rigid, centred, at fractional d in the current box."""
        st = ctx.getState(getPositions=True)
        pos = np.asarray(st.getPositions().value_in_unit(u.nanometer))
        box = st.getPeriodicBoxVectors().value_in_unit(u.nanometer)
        L = np.array([float(box[0][0]), float(box[1][1]), float(box[2][2])])
        _, _, err = cage_state()
        pos[carbon] = geometry.pair_positions(frac_d_ref * L[2], 0.5 * L)
        ctx.setPositions(pos * u.nanometer)
        return float(frac_d_ref * L[2]), err

    steps_per_sample = int(round(SAMPLE_PS / DT_PS))
    n_discard = int(round(DISCARD_PS / SAMPLE_PS))
    n_measure = int(round(MEASURE_PS / SAMPLE_PS))

    vols, temps, dcoms = [], [], []
    max_preproj_err = 0.0
    t0 = time.perf_counter()
    for k in range(n_discard + n_measure):
        integ.step(steps_per_sample)
        st = ctx.getState(getEnergy=True)
        box = ctx.getState().getPeriodicBoxVectors().value_in_unit(u.nanometer)
        v = float(box[0][0]) * float(box[1][1]) * float(box[2][2])
        vols.append(v)
        ke = st.getKineticEnergy().value_in_unit(u.kilojoule_per_mole)
        ndof = 6 * csys.N_WATERS - 3     # 3-site massive rigid waters, COM removed by OpenMM
        temps.append(2.0 * ke / (ndof * 8.31446261815324e-3))
        d_now, err = project_cages()
        dcoms.append(d_now)
        max_preproj_err = max(max_preproj_err, err)
        if err > 2e-3:
            raise RuntimeError(f"pre-projection cage distortion {err:.2e} nm at {k+1} ps "
                               "exceeds the 2e-3 nm guard; barostat behaviour changed")
        if (k + 1) % 100 == 0:
            print(f"  {k+1:5d}/{n_discard+n_measure} ps  V = {v:.4f} nm^3  "
                  f"T = {temps[-1]:.1f} K  d_com = {d_now:.4f} nm  "
                  f"preproj {err:.1e}  ({time.perf_counter()-t0:.0f}s)", flush=True)

    drift = max_preproj_err                # recorded in the manifest

    vols = np.asarray(vols); temps = np.asarray(temps)
    v_mean = float(vols[n_discard:].mean())
    v_sem = float(vols[n_discard:].std(ddof=1) / np.sqrt(len(vols[n_discard:]) / 10.0))
    lx = (v_mean / aspect) ** (1.0 / 3.0)
    lz = aspect * lx

    # ---- SPEC §1.4 geometric guards ---------------------------------------------------------
    R_cage = 0.35620837522777166
    guards = dict(
        half_lx_exceeds_cutoff=0.5 * lx > csys.CUTOFF_NM,
        z_image_water_gap_nm=lz - (csys.XI_HI_NM + 2 * R_cage),
        z_image_gap_ok=(lz - (csys.XI_HI_NM + 2 * R_cage)) > 2.0,
        interfacial_fits=0.5 * csys.XI_HI_NM + 1.082 < 0.5 * lz,
    )
    if not (guards["half_lx_exceeds_cutoff"] and guards["z_image_gap_ok"]
            and guards["interfacial_fits"]):
        raise RuntimeError(f"SPEC §1.4 geometric guard failed: {guards}")

    # ---- frozen-box configuration: molecule-COM rescale of the final frame ------------------
    st = ctx.getState(getPositions=True)
    pos = np.asarray(st.getPositions().value_in_unit(u.nanometer))
    box = st.getPeriodicBoxVectors().value_in_unit(u.nanometer)
    scale = np.array([lx / float(box[0][0]), lx / float(box[1][1]), lz / float(box[2][2])])
    out_pos = pos.copy()
    for (o, h1, h2, m) in p["waters"]:
        com = pos[[o, h1, h2]].mean(axis=0)      # COM by geometry is fine for a rigid scale
        shift = com * scale - com
        for s in (o, h1, h2, m):
            out_pos[s] = pos[s] + shift
    center = np.array([0.5 * lx, 0.5 * lx, 0.5 * lz])
    out_pos[carbon] = geometry.pair_positions(csys.D_REF_NM, center)

    np.savez(os.path.join(OUT, "npt_trace.npz"), volume_nm3=vols, temperature_K=temps,
             d_com_nm=np.asarray(dcoms), discard_ps=DISCARD_PS, sample_ps=SAMPLE_PS)
    np.savez(os.path.join(OUT, "frozen_box.npz"), positions=out_pos,
             lx_nm=lx, lz_nm=lz, aspect=aspect, d_com_nm=csys.D_REF_NM)

    manifest = dict(csys.manifest(), stage="npt_box", lx0_nm=lx0, lz0_nm=lz0,
                    v_mean_nm3=v_mean, v_sem_nm3=v_sem, frozen_lx_nm=lx, frozen_lz_nm=lz,
                    aspect=aspect, discard_ps=DISCARD_PS, measure_ps=MEASURE_PS,
                    temperature_mean_K=float(temps[n_discard:].mean()),
                    max_preprojection_cage_err_nm=drift,
                    d_com_range_nm=[float(min(dcoms)), float(max(dcoms))],
                    guards={k: (bool(v) if isinstance(v, (bool, np.bool_))
                                else float(v)) for k, v in guards.items()},
                    cuda_visible_devices=dev, platform="CUDA mixed",
                    wall_seconds=time.perf_counter() - t0)
    with open(os.path.join(OUT, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)

    with open(os.path.join(OUT, "RESULT.md"), "w") as fh:
        fh.write(f"""# NPT box freeze (SPEC §1.4)

* built {csys.N_WATERS} waters + 2 fixed cages at d_ref = {csys.D_REF_NM} nm,
  initial box {lx0:.4f} x {lx0:.4f} x {lz0:.4f} nm (aspect {aspect:.6f})
* NPT 300 K / 1 bar, MC barostat (isotropic), {DISCARD_PS:.0f} ps discard + {MEASURE_PS:.0f} ps measure
* <V> = {v_mean:.4f} +- {v_sem:.4f} nm^3 (block SEM), <T> = {float(temps[n_discard:].mean()):.2f} K
* **FROZEN: Lx = Ly = {lx:.6f} nm, Lz = {lz:.6f} nm**
* the barostat scales carbons individually (measured; massless => no constraints => not a
  molecule to it); the per-ps cage projector held the cages rigid at fixed fractional
  separation, max pre-projection distortion {drift:.2e} nm; d tracked the box within
  [{min(dcoms):.4f}, {max(dcoms):.4f}] nm and the frozen configuration re-places the cages at
  exactly d_ref = {csys.D_REF_NM} nm
* geometric guards: half-box vs cutoff {0.5*lx:.3f} > {csys.CUTOFF_NM}; z image water gap
  {guards['z_image_water_gap_nm']:.3f} nm > 2.0; interfacial shells fit in Lz/2: {guards['interfacial_fits']}
""")
    print(f"FROZEN Lx = {lx:.6f} nm, Lz = {lz:.6f} nm  (<V> = {v_mean:.4f} nm^3)", flush=True)


if __name__ == "__main__":
    main()
