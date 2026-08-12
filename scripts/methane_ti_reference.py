"""Stage 2 -- the primary methane reference: constrained-TI mean force (Amendment 12.2).

    F_ref(r) = C + integral fbar(s) ds ,   fbar(r) = < (1/2)(F_1 - F_2).e - 2/(beta r) >_{xi=r}
    W_ref(r) = F_ref(r) + 2 beta^-1 log r + C'

Why constrained rather than restrained
--------------------------------------
The methane-methane separation is held by a **rigid distance constraint**, not a stiff harmonic.
Because ``|grad xi|^2 = 2`` is constant for this CV, the den Otter field carries no Fixman/metric
correction, so the conditional average of the physical local mean force *is* ``F'(r)`` exactly --
no restraint width bias to bound and no restraint force to subtract.  OpenMM's ``getForces``
returns forces from ``Force`` objects only, never constraint forces, so what is accumulated is
already the physical mean force.

Wet / dry families are the Gate 0 instrument
--------------------------------------------
Each ``r`` is sampled from two deliberately opposed solvent preparations:

    dry : solvent bath equilibrated with the methanes in **contact**, gap empty
    wet : solvent bath equilibrated at the **solvent-separated** distance, gap filled

If the conditional ensemble at fixed ``r`` mixes within the budget, both families give the same
``fbar`` and Gate 0 passes.  If they do not, the difference *is* the conditional-equilibration
signal -- the same design that discharged Amendment 10's obligation for the WCA dimer
(``scripts/audit_wca_gate0.py``), transplanted.  Amendment 9: this controlled test is the
instrument, and a screen statistic is not a substitute for it.

Resumable: every (build, r) cell is written as it completes, and an existing cell is skipped.

Usage:
    CUDA_VISIBLE_DEVICES=3 python scripts/methane_ti_reference.py --out results/methane/ti
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import openmm as mm                                              # noqa: E402
import openmm.unit as u                                          # noqa: E402

from methane import system as msys                               # noqa: E402
from methane.observables import n_gap                            # noqa: E402

#: frozen TI grid, SPEC §4.2 -- 29 points
R_GRID_NM = np.round(np.arange(0.34, 0.9001, 0.02), 4)
R_DRY_PREP_NM = 0.38          #: contact bath
R_WET_PREP_NM = 0.80          #: solvent-separated bath


def _platform(name=None):
    for cand in ([name] if name else ["CUDA", "OpenCL", "CPU"]):
        try:
            p = mm.Platform.getPlatformByName(cand)
            return p, ({"Precision": "mixed"} if cand in ("CUDA", "OpenCL") else {})
        except Exception:
            continue
    raise RuntimeError("no OpenMM platform")


def build_constrained(topology, L, r_nm, methane_idx):
    """System with the two methanes rigidly constrained at ``r_nm``."""
    system = msys.build_system(topology, pin_pme=True)
    system.setDefaultPeriodicBoxVectors(mm.Vec3(L, 0, 0) * u.nanometer,
                                        mm.Vec3(0, L, 0) * u.nanometer,
                                        mm.Vec3(0, 0, L) * u.nanometer)
    system.addConstraint(int(methane_idx[0]), int(methane_idx[1]), float(r_nm) * u.nanometer)
    return system


def place_methanes(pos, methane_idx, r_nm, L):
    """Move the two methanes to separation ``r_nm``, symmetric about their current midpoint."""
    out = pos.copy()
    i, j = int(methane_idx[0]), int(methane_idx[1])
    d = out[j] - out[i]
    d -= L * np.round(d / L)
    e = d / max(np.linalg.norm(d), 1e-12)
    mid = out[i] + 0.5 * d
    out[i] = mid - 0.5 * r_nm * e
    out[j] = mid + 0.5 * r_nm * e
    return out


def local_mean_force(forces, pos, methane_idx, r_nm, beta):
    """``f_loc = (1/2)(F_1 - F_2).e - 2/(beta r)`` -- SPEC §2.2, identical to DistanceCV."""
    i, j = int(methane_idx[0]), int(methane_idx[1])
    d = pos[j] - pos[i]
    r = np.linalg.norm(d)
    e = d / r
    return 0.5 * float(np.dot(forces[i] - forces[j], e)) - 2.0 / (beta * r_nm), r


def run_cell(ctx, integ, pos0, methane_idx, r_nm, beta, equil_steps, prod_steps,
             sample_every, oxygen_idx, L, seed):
    """One replica at one ``r``: minimise, equilibrate, then accumulate ``f_loc`` and ``n_gap``."""
    ctx.setPositions(pos0 * u.nanometer)
    mm.LocalEnergyMinimizer.minimize(ctx, 5.0, 1000)
    ctx.setVelocitiesToTemperature(msys.TEMPERATURE_K * u.kelvin, seed)
    integ.step(equil_steps)

    f_acc, ng_acc = [], []
    n_batch = max(1, prod_steps // sample_every)
    for _ in range(n_batch):
        integ.step(sample_every)
        st = ctx.getState(getPositions=True, getForces=True)
        p = np.asarray(st.getPositions().value_in_unit(u.nanometer))
        f = np.asarray(st.getForces().value_in_unit(u.kilojoule_per_mole / u.nanometer))
        fl, _ = local_mean_force(f, p, methane_idx, r_nm, beta)
        f_acc.append(fl)
        ng_acc.append(n_gap(p, methane_idx, oxygen_idx, L))
    return np.asarray(f_acc), np.asarray(ng_acc)


def prepare_bath(topology, L, pos, methane_idx, r_prep, ps, platform_name, seed):
    """Equilibrate a solvent bath with the methanes held at ``r_prep``; return the final frame."""
    system = build_constrained(topology, L, r_prep, methane_idx)
    integ = mm.LangevinMiddleIntegrator(msys.TEMPERATURE_K * u.kelvin,
                                        msys.GAMMA_PS / u.picosecond,
                                        msys.DT_PS * u.picoseconds)
    integ.setRandomNumberSeed(seed)
    plat, props = _platform(platform_name)
    ctx = mm.Context(system, integ, plat, props)
    ctx.setPositions(place_methanes(pos, methane_idx, r_prep, L) * u.nanometer)
    mm.LocalEnergyMinimizer.minimize(ctx, 1.0, 3000)
    ctx.setVelocitiesToTemperature(msys.TEMPERATURE_K * u.kelvin, seed)
    integ.step(int(round(ps / msys.DT_PS)))
    out = np.asarray(ctx.getState(getPositions=True).getPositions()
                     .value_in_unit(u.nanometer))
    del ctx
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/methane/ti")
    ap.add_argument("--box", default="results/methane/box")
    ap.add_argument("--builds", type=int, default=3)
    ap.add_argument("--replicas", type=int, default=16, help="per r per build, split wet/dry")
    ap.add_argument("--equil-ps", type=float, default=50.0)
    ap.add_argument("--prod-ps", type=float, default=200.0)
    ap.add_argument("--bath-ps", type=float, default=200.0)
    ap.add_argument("--sample-ps", type=float, default=0.1)
    ap.add_argument("--platform", default=None)
    ap.add_argument("--r-subset", default=None, help="comma-separated r values (nm) for smoke runs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    man = json.load(open(os.path.join(args.box, "manifest.json")))
    L = float(man["box_L_nm"])
    pos_box = np.load(os.path.join(args.box, "box.npz"))["positions_nm"]
    beta = msys.beta_per_kJ()

    mod = msys.build_modeller(r0_nm=0.55, seed=man["seed"])
    p = msys.site_parameters(msys.build_system(mod.topology), mod.topology)
    methane_idx = p["methane_index"]
    oxygen_idx = np.flatnonzero((~p["is_methane"]) & (p["epsilon"] > 0))
    assert oxygen_idx.size == msys.N_WATERS

    r_grid = R_GRID_NM if args.r_subset is None else np.asarray(
        [float(s) for s in args.r_subset.split(",")])
    equil_steps = int(round(args.equil_ps / msys.DT_PS))
    prod_steps = int(round(args.prod_ps / msys.DT_PS))
    sample_every = int(round(args.sample_ps / msys.DT_PS))
    n_half = args.replicas // 2

    total_ns = len(r_grid) * args.replicas * args.builds * (args.equil_ps + args.prod_ps) / 1000.0
    print(f"[plan] {len(r_grid)} r-points x {args.replicas} replicas x {args.builds} builds "
          f"= {total_ns:.1f} ns aggregate", flush=True)
    print(f"[box]  L = {L:.6f} nm,  beta = {beta:.6f} mol/kJ", flush=True)

    t_start = time.time()
    for build in range(args.builds):
        print(f"\n[build {build}] preparing wet/dry solvent baths ...", flush=True)
        dry = prepare_bath(mod.topology, L, pos_box, methane_idx, R_DRY_PREP_NM,
                           args.bath_ps, args.platform, 9000 + build)
        wet = prepare_bath(mod.topology, L, pos_box, methane_idx, R_WET_PREP_NM,
                           args.bath_ps, args.platform, 9500 + build)
        ng_dry = n_gap(dry, methane_idx, oxygen_idx, L)
        ng_wet = n_gap(wet, methane_idx, oxygen_idx, L)
        print(f"[build {build}] bath n_gap: dry(r={R_DRY_PREP_NM}) = {ng_dry:.3f}   "
              f"wet(r={R_WET_PREP_NM}) = {ng_wet:.3f}", flush=True)

        for r_nm in r_grid:
            cell = os.path.join(args.out, f"build{build}_r{r_nm:.4f}.npz")
            if os.path.exists(cell):
                print(f"[build {build}] r = {r_nm:.4f}  (cached)", flush=True)
                continue
            t0 = time.time()
            system = build_constrained(mod.topology, L, r_nm, methane_idx)
            integ = mm.LangevinMiddleIntegrator(msys.TEMPERATURE_K * u.kelvin,
                                                msys.GAMMA_PS / u.picosecond,
                                                msys.DT_PS * u.picoseconds)
            plat, props = _platform(args.platform)
            ctx = mm.Context(system, integ, plat, props)

            fam, fbar, fsem, ngm = [], [], [], []
            for k in range(args.replicas):
                is_wet = k >= n_half
                seed = 100000 + build * 10000 + int(round(r_nm * 1000)) * 10 + k
                integ.setRandomNumberSeed(seed)
                start = place_methanes(wet if is_wet else dry, methane_idx, r_nm, L)
                fl, ng = run_cell(ctx, integ, start, methane_idx, r_nm, beta,
                                  equil_steps, prod_steps, sample_every, oxygen_idx, L, seed)
                fam.append(1 if is_wet else 0)
                fbar.append(float(fl.mean()))
                fsem.append(float(fl.std(ddof=1) / np.sqrt(len(fl))))
                ngm.append(float(ng.mean()))
            del ctx
            fam = np.asarray(fam); fbar = np.asarray(fbar)
            d_mean = fbar[fam == 0].mean(); w_mean = fbar[fam == 1].mean()
            pooled = fbar.mean()
            spread = abs(w_mean - d_mean)
            np.savez_compressed(cell, r_nm=r_nm, family=fam, fbar=fbar,
                                fsem=np.asarray(fsem), ngap=np.asarray(ngm),
                                build=build)
            print(f"[build {build}] r = {r_nm:.4f}  fbar = {pooled:9.3f}  "
                  f"dry {d_mean:9.3f}  wet {w_mean:9.3f}  |wet-dry| {spread:7.3f}  "
                  f"n_gap {np.mean(ngm):5.2f}  ({time.time()-t0:.0f}s)", flush=True)

    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(dict(
            stage="ti_reference", amendment="V2_PREREGISTRATION.md Amendment 12.2",
            r_grid_nm=[float(x) for x in r_grid], builds=args.builds,
            replicas_per_r=args.replicas, equil_ps=args.equil_ps, prod_ps=args.prod_ps,
            bath_ps=args.bath_ps, sample_ps=args.sample_ps,
            r_dry_prep_nm=R_DRY_PREP_NM, r_wet_prep_nm=R_WET_PREP_NM,
            box_L_nm=L, beta_per_kJ=beta, aggregate_ns=total_ns,
            wall_seconds=time.time() - t_start, box_manifest=man,
            git_commit=subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                      text=True).stdout.strip()), fh, indent=2)
    print(f"\n[done] {(time.time()-t_start)/3600:.2f} h -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
