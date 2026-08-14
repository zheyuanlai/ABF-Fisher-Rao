"""SPEC §1.3 finite-size gate: is the accepted reference sound at its outermost points?

Evaluates the constrained-TI mean force at ``r in {0.70, 0.80, 0.90} nm`` in a **1024-water box**
(``L ~ 3.14 nm``) and compares against the 512-water values already in the accepted reference.
The preregistered criterion:

    |<f_loc>_1024 - <f_loc>_512|  >  0.1 kT/nm  at any of the three  ->  truncate the domain

Why it matters here specifically: at ``L = 2.4908 nm`` the minimum-image half-box is 1.245 nm, so
``r = 0.90`` sits at **72 %** of it and is also the reference's largest wet/dry spread
(4.26 kJ/mol/nm). The interior is unaffected; the outer tail is where a 512-water box can lie.

**RUN LATE, AND THAT CHANGES WHAT IT CAN DO.** §1.3 specifies this check *"before the reference is
built"*, and the reference was built without it. Run now it cannot gate the domain *choice* --
that decision is already embedded in an accepted reference and a completed screen. It can only
report whether the outermost points of that reference are trustworthy. A failure is therefore
recorded as a caveat on those points plus an explicit judgement about whether the verdict moves,
**not** as a silent re-truncation of a domain that has already been used.

The verdict is not expected to move either way: methane's conclusion rests on Gate C, whose
deficit test is dominated by the interior where the population sits, and the outermost tercile
boundary is at 0.71 nm. That expectation is written here in advance so it cannot be presented
afterwards as a confirmed prediction.

Usage:
    CUDA_VISIBLE_DEVICES=<idle> python scripts/methane_finite_size.py --out results/methane/finite_size
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
import openmm.app as app                                         # noqa: E402
import openmm.unit as u                                          # noqa: E402

from methane import system as msys                               # noqa: E402
from methane.observables import n_gap                            # noqa: E402

R_POINTS_NM = (0.70, 0.80, 0.90)          #: SPEC §1.3, frozen
TOL_KT_PER_NM = 0.1                       #: SPEC §1.3, frozen
N_WATERS_LARGE = 1024


_PLATFORM_CACHE = {}


def _platform(prefer=None):
    """First platform that can actually create a Context, not merely the first that exists.

    ``getPlatformByName("CUDA")`` succeeds whenever the plugin is installed; the failure appears
    only at Context creation if no device is visible. Probing with a real Context is the check
    that matches how the platform is used.
    """
    if "p" in _PLATFORM_CACHE:
        return _PLATFORM_CACHE["p"]
    order = [prefer] if prefer else ["CUDA", "OpenCL", "CPU"]
    probe = mm.System()
    probe.addParticle(1.0)
    for cand in order:
        try:
            pl = mm.Platform.getPlatformByName(cand)
            props = {"Precision": "mixed"} if cand in ("CUDA", "OpenCL") else {}
            ctx = mm.Context(probe, mm.VerletIntegrator(1e-6), pl, props)
            del ctx
            _PLATFORM_CACHE["p"] = (pl, props)
            return pl, props
        except Exception:
            continue
    raise RuntimeError(f"no usable OpenMM platform among {order}")


def build_large_modeller(seed, r0_nm=0.55, pad_box_nm=3.6):
    """Two methanes solvated by exactly ``N_WATERS_LARGE`` waters -- the same builder, larger box."""
    import random
    top = app.Topology()
    chain = top.addChain()
    positions = []
    for dx in (-0.5 * r0_nm, +0.5 * r0_nm):
        res = top.addResidue("MTH", chain)
        top.addAtom("C1", app.element.carbon, res)
        positions.append(mm.Vec3(0.5 * pad_box_nm + dx, 0.5 * pad_box_nm, 0.5 * pad_box_nm))
    top.setUnitCellDimensions(mm.Vec3(pad_box_nm, pad_box_nm, pad_box_nm))
    random.seed(int(seed))
    mod = app.Modeller(top, positions * u.nanometer)
    mod.addSolvent(msys._forcefield(), model="spce", numAdded=N_WATERS_LARGE)
    n_w = sum(1 for r in mod.topology.residues() if r.name in ("HOH", "WAT"))
    if n_w != N_WATERS_LARGE:
        raise RuntimeError(f"addSolvent produced {n_w} waters, expected {N_WATERS_LARGE}")
    return mod


def npt_box(mod, seed, equil_ps, prod_ps):
    """Freeze the large box the same way SPEC §1.3 froze the production one."""
    system = msys.build_system(mod.topology, dispersion_correction=True, pin_pme=False)
    pos = msys.apply_constraints(
        system, mod.topology, np.asarray(mod.positions.value_in_unit(u.nanometer)))
    bar = mm.MonteCarloBarostat(msys.PRESSURE_BAR * u.bar, msys.TEMPERATURE_K * u.kelvin, 25)
    bar.setRandomNumberSeed(seed)
    system.addForce(bar)
    integ = mm.LangevinMiddleIntegrator(msys.TEMPERATURE_K * u.kelvin,
                                        msys.GAMMA_PS / u.picosecond, msys.DT_PS * u.picoseconds)
    integ.setRandomNumberSeed(seed)
    plat, props = _platform()
    ctx = mm.Context(system, integ, plat, props)
    ctx.setPositions(pos * u.nanometer)
    mm.LocalEnergyMinimizer.minimize(ctx, 1.0, 5000)
    ctx.setVelocitiesToTemperature(msys.TEMPERATURE_K * u.kelvin, seed)
    integ.step(int(round(equil_ps / msys.DT_PS)))
    vols = []
    every = int(round(0.5 / msys.DT_PS))
    for _ in range(int(round(prod_ps / msys.DT_PS)) // every):
        integ.step(every)
        bv = ctx.getState().getPeriodicBoxVectors()
        vols.append(bv[0][0].value_in_unit(u.nanometer) * bv[1][1].value_in_unit(u.nanometer)
                    * bv[2][2].value_in_unit(u.nanometer))
    vols = np.asarray(vols)
    L = float(vols.mean()) ** (1.0 / 3.0)
    st = ctx.getState(getPositions=True)
    pos = np.asarray(st.getPositions().value_in_unit(u.nanometer))
    L_now = st.getPeriodicBoxVectors()[0][0].value_in_unit(u.nanometer)
    del ctx
    return L, pos * (L / L_now), float(vols.mean()), float(vols.std())


def ti_point(topology, L, pos, mi, r_nm, equil_ps, prod_ps, sample_ps, n_rep, beta, seed0, oxy):
    """Constrained TI at one ``r``: the same estimator as the accepted reference."""
    system = msys.build_system(topology, pin_pme=True)
    system.setDefaultPeriodicBoxVectors(mm.Vec3(L, 0, 0) * u.nanometer,
                                        mm.Vec3(0, L, 0) * u.nanometer,
                                        mm.Vec3(0, 0, L) * u.nanometer)
    system.addConstraint(int(mi[0]), int(mi[1]), float(r_nm) * u.nanometer)
    plat, props = _platform()
    fbars, ngs = [], []
    for k in range(n_rep):
        seed = seed0 + k
        integ = mm.LangevinMiddleIntegrator(msys.TEMPERATURE_K * u.kelvin,
                                            msys.GAMMA_PS / u.picosecond,
                                            msys.DT_PS * u.picoseconds)
        integ.setRandomNumberSeed(seed)
        ctx = mm.Context(system, integ, plat, props)
        p = pos.copy()
        i, j = int(mi[0]), int(mi[1])
        d = p[j] - p[i]
        d -= L * np.round(d / L)
        e = d / np.linalg.norm(d)
        mid = p[i] + 0.5 * d
        p[i], p[j] = mid - 0.5 * r_nm * e, mid + 0.5 * r_nm * e
        ctx.setPositions(p * u.nanometer)
        mm.LocalEnergyMinimizer.minimize(ctx, 5.0, 2000)
        ctx.setVelocitiesToTemperature(msys.TEMPERATURE_K * u.kelvin, seed)
        integ.step(int(round(equil_ps / msys.DT_PS)))
        every = int(round(sample_ps / msys.DT_PS))
        acc, ng = [], []
        for _ in range(int(round(prod_ps / msys.DT_PS)) // every):
            integ.step(every)
            st = ctx.getState(getPositions=True, getForces=True)
            q = np.asarray(st.getPositions().value_in_unit(u.nanometer))
            f = np.asarray(st.getForces().value_in_unit(u.kilojoule_per_mole / u.nanometer))
            dd = q[j] - q[i]
            rr = np.linalg.norm(dd)
            ee = dd / rr
            acc.append(0.5 * float(np.dot(f[i] - f[j], ee)) - 2.0 / (beta * r_nm))
            ng.append(n_gap(q, mi, oxy, L))
        del ctx
        fbars.append(float(np.mean(acc)))
        ngs.append(float(np.mean(ng)))
    return np.asarray(fbars), np.asarray(ngs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/methane/finite_size")
    ap.add_argument("--ref", default="results/methane/ref")
    ap.add_argument("--replicas", type=int, default=16)
    ap.add_argument("--equil-ps", type=float, default=50.0)
    ap.add_argument("--prod-ps", type=float, default=200.0)
    ap.add_argument("--sample-ps", type=float, default=0.1)
    ap.add_argument("--box-equil-ps", type=float, default=500.0)
    ap.add_argument("--box-prod-ps", type=float, default=1000.0)
    ap.add_argument("--seed", type=int, default=20260814)
    ap.add_argument("--r-points", default=",".join(str(x) for x in R_POINTS_NM))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    ref = json.load(open(os.path.join(args.ref, "reference.json")))
    r_ref = np.asarray(ref["r_nm"])
    f_ref = np.asarray(ref["fbar"])
    kT = msys.kT_kJ()
    beta = msys.beta_per_kJ()
    r_points = [float(x) for x in args.r_points.split(",")]

    print(f"[plan] {N_WATERS_LARGE} waters, r = {r_points}, {args.replicas} replicas x "
          f"{args.prod_ps} ps; tolerance {TOL_KT_PER_NM} kT/nm = "
          f"{TOL_KT_PER_NM*kT:.4f} kJ/mol/nm", flush=True)

    mod = build_large_modeller(args.seed)
    print(f"[build] {mod.topology.getNumAtoms()} sites", flush=True)
    L, pos, vmean, vsd = npt_box(mod, args.seed, args.box_equil_ps, args.box_prod_ps)
    mass_g = (N_WATERS_LARGE * 18.01528 + 2 * msys.MASS_METHANE_AMU) / 6.02214076e23
    rho = mass_g / (vmean * 1e-21)
    print(f"[box] L = {L:.6f} nm  <V> = {vmean:.4f} +- {vsd:.4f} nm^3  rho = {rho:.4f} g/cm^3",
          flush=True)
    print(f"[box] half-box {L/2:.4f} nm; r = 0.90 is {0.90/(L/2)*100:.0f} % of it "
          f"(was 72 % at 512 waters)", flush=True)

    p = msys.site_parameters(msys.build_system(mod.topology), mod.topology)
    mi = p["methane_index"]
    oxy = np.flatnonzero((~p["is_methane"]) & (p["epsilon"] > 0))

    rows = []
    for idx, r_nm in enumerate(r_points):
        t1 = time.time()
        fb, ng = ti_point(mod.topology, L, pos, mi, r_nm, args.equil_ps, args.prod_ps,
                          args.sample_ps, args.replicas, beta, 4000 + 100 * idx, oxy)
        f512 = float(np.interp(r_nm, r_ref, f_ref))
        f1024 = float(fb.mean())
        sem = float(fb.std(ddof=1) / np.sqrt(len(fb)))
        d_kT = (f1024 - f512) / kT
        rows.append(dict(r_nm=r_nm, f_1024=f1024, sem_1024=sem, f_512=f512,
                         diff_kJ=f1024 - f512, diff_kT=d_kT,
                         passes=bool(abs(d_kT) <= TOL_KT_PER_NM), n_gap=float(ng.mean())))
        print(f"[r={r_nm:.2f}] f_1024 = {f1024:8.3f} +- {sem:.3f}   f_512 = {f512:8.3f}   "
              f"diff = {d_kT:+.4f} kT/nm   {'PASS' if abs(d_kT) <= TOL_KT_PER_NM else 'FAIL'}   "
              f"({(time.time()-t1)/60:.1f} min)", flush=True)
        np.savez_compressed(os.path.join(args.out, f"r{r_nm:.2f}.npz"), fbar=fb, ngap=ng,
                            r_nm=r_nm, L_nm=L)

    worst = max(rows, key=lambda x: abs(x["diff_kT"]))
    all_pass = all(x["passes"] for x in rows)
    print(f"\n[gate] worst |diff| = {abs(worst['diff_kT']):.4f} kT/nm at r = {worst['r_nm']:.2f} "
          f"(tolerance {TOL_KT_PER_NM})")
    print(f"[gate] FINITE-SIZE GATE: {'PASS' if all_pass else 'FAIL'}")
    if not all_pass:
        ok = [x["r_nm"] for x in rows if x["passes"]]
        print(f"[gate] points passing: {ok}. §1.3 specifies truncation to the largest passing r, "
              "but the reference and screen are already built -- see this script's docstring: "
              "this is recorded as a caveat on the outer points, not applied as a silent "
              "re-truncation.")

    with open(os.path.join(args.out, "finite_size.json"), "w") as fh:
        json.dump(dict(stage="finite_size_gate", spec="SPEC_methane_water.md §1.3",
                       run_late=True, n_waters=N_WATERS_LARGE, box_L_nm=L,
                       volume_nm3=vmean, volume_sd_nm3=vsd, density_g_cm3=rho,
                       half_box_nm=L / 2, tolerance_kT_per_nm=TOL_KT_PER_NM, kT_kJ=kT,
                       replicas=args.replicas, equil_ps=args.equil_ps, prod_ps=args.prod_ps,
                       rows=rows, all_pass=bool(all_pass),
                       worst_r_nm=worst["r_nm"], worst_diff_kT=worst["diff_kT"],
                       wall_hours=(time.time() - t0) / 3600.0,
                       git_commit=subprocess.run(["git", "rev-parse", "HEAD"],
                                                 capture_output=True,
                                                 text=True).stdout.strip()), fh, indent=2)
    print(f"\n[done] {(time.time()-t0)/3600:.2f} h -> {args.out}/finite_size.json", flush=True)


if __name__ == "__main__":
    main()
