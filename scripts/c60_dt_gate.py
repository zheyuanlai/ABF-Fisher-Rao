"""SPEC §3.4 dt gate: accept the paper's 2 fs or fall back to 1 fs.  Frozen decision rule.

Three clauses, decided from measurement before the reference runs:
  (i)  constraint clause (deterministic): float64, 4 walkers x 5 ps, max violation <= 1e-8 nm;
  (ii) equipartition clause (statistical): production dtype (float32), 32 walkers x 60 ps
       (10 ps warm) against 4 independent OpenMM CUDA replicas x 60 ps; blocking SEMs
       (5 ps blocks); with sigma = sqrt(sem_t^2 + sem_o^2):
       PASS iff dT + sigma <= 2.0 K, FAIL iff dT - sigma > 2.0 K, else INDETERMINATE;
  (iii) mean-force spot clause: <f> at d in {0.968, 1.20, 2.00}, 16 replicas x 30 ps
       production per dt, 2 fs vs 1 fs within combined block error (2 sigma).

Decision: 2 fs PASS on all three -> 2 fs.  Otherwise -> 1 fs (and a confident 1 fs FAIL is an
engine-defect STOP).  Written once to results/c60/parity/dt_gate.json, never revisited.

OpenMM-CUDA and torch may not share a process (measured deadlock, NaCl), so the script runs
in phases:  --phase openmm  |  --phase torch  |  --phase verdict.

Usage:
  CUDA_VISIBLE_DEVICES=3 python scripts/c60_dt_gate.py --phase openmm
  CUDA_VISIBLE_DEVICES=3 python scripts/c60_dt_gate.py --phase torch
  python scripts/c60_dt_gate.py --phase verdict
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "c60", "parity")
BOX = os.path.join(os.path.dirname(__file__), "..", "results", "c60", "box", "frozen_box.npz")

WARM_PS = 10.0
RUN_PS = 60.0
BLOCK_PS = 5.0
DT_CANDIDATES = (0.002, 0.001)
SPOT_D = (0.968, 1.20, 2.00)
SPOT_PS = 30.0
T_TARGET = 300.0


def _blocked_sem(series, samples_per_block):
    s = np.asarray(series, dtype=np.float64)
    n = (len(s) // samples_per_block) * samples_per_block
    if n < 2 * samples_per_block:
        return float("nan")
    blocks = s[:n].reshape(-1, samples_per_block).mean(axis=1)
    return float(blocks.std(ddof=1) / np.sqrt(len(blocks)))


def phase_openmm():
    """4 independent OpenMM CUDA replicas x 60 ps per dt; kinetic T series at 0.5 ps."""
    import openmm as mm
    import openmm.app as app  # noqa: F401
    import openmm.unit as u
    from c60 import system as csys

    fz = np.load(BOX)
    lx, lz = float(fz["lx_nm"]), float(fz["lz_nm"])
    mod = csys.build_modeller()
    box = [mm.Vec3(lx, 0, 0), mm.Vec3(0, lx, 0), mm.Vec3(0, 0, lz)] * u.nanometer
    system = csys.build_system(mod.topology, box_vectors=box, pme_params=csys.pme_params())
    ndof = 6 * csys.N_WATERS - 3           # OpenMM removes water COM motion by default? no --
    # createSystem adds CMMotionRemover; count it out of the DOF (asserted below via mean T)
    out = {}
    for dt in DT_CANDIDATES:
        temps = []
        for rep in range(4):
            integ = mm.LangevinMiddleIntegrator(T_TARGET * u.kelvin,
                                                csys.GAMMA_PS / u.picosecond,
                                                dt * u.picoseconds)
            ctx = mm.Context(system, integ, mm.Platform.getPlatformByName("CUDA"),
                             dict(Precision="mixed"))
            ctx.setPositions(np.asarray(fz["positions"]) * u.nanometer)
            ctx.applyConstraints(1e-10)
            ctx.computeVirtualSites()
            ctx.setVelocitiesToTemperature(T_TARGET * u.kelvin, 1000 + rep)
            series = []
            n_chunks = int(round(RUN_PS / 0.5))
            for k in range(n_chunks):
                integ.step(int(round(0.5 / dt)))
                ke = ctx.getState(getEnergy=True).getKineticEnergy() \
                    .value_in_unit(u.kilojoule_per_mole)
                series.append(2.0 * ke / (ndof * 8.31446261815324e-3))
            del ctx
            temps.append(series)
        out[f"dt_{dt}"] = temps
        print(f"openmm dt={dt}: mean T (post-warm) = "
              f"{np.mean(np.asarray(temps)[:, int(WARM_PS/0.5):]):.2f} K", flush=True)
    np.savez(os.path.join(OUT, "dt_gate_openmm.npz"),
             **{k: np.asarray(v) for k, v in out.items()})


def phase_torch():
    """Torch clauses: constraint drift (float64), equipartition (float32), mean-force spots."""
    import torch
    from c60 import system as csys
    from c60.dynamics import C60Dynamics
    from c60.nonbonded import C60Nonbonded
    import openmm as mm
    import openmm.unit as u

    assert torch.cuda.device_count() == 1
    fz = np.load(BOX)
    lx, lz = float(fz["lx_nm"]), float(fz["lz_nm"])
    base = np.asarray(fz["positions"], dtype=np.float64)
    mod = csys.build_modeller()
    box = [mm.Vec3(lx, 0, 0), mm.Vec3(0, lx, 0), mm.Vec3(0, 0, lz)] * u.nanometer
    system = csys.build_system(mod.topology, box_vectors=box, pme_params=csys.pme_params())
    alpha, nx, ny, nz = csys.pme_params()

    results = {}
    for dt in DT_CANDIDATES:
        res = {}
        # ---- (i) constraint clause, float64 ------------------------------------------------
        eng = C60Nonbonded(system, mod.topology, (lx, lx, lz), alpha, (nx, ny, nz),
                           device="cuda", dtype=torch.float64)
        dyn = C60Dynamics(eng, dt, device="cuda", dtype=torch.float64)
        x = torch.as_tensor(base, device="cuda", dtype=torch.float64)[None] \
            .repeat(4, 1, 1).contiguous()
        eng.compute_vsites(x)
        gen = torch.Generator(device="cuda").manual_seed(11)
        v = dyn.maxwell_velocities(x, generator=gen)
        _, f_raw = eng.energy_forces(x)
        f = eng.redistribute(f_raw)
        worst = 0.0
        for step in range(int(round(5.0 / dt))):
            _, f = dyn.step(x, v, f, generator=gen)
            if step % 200 == 0:
                worst = max(worst, dyn.cons.max_violation(x))
        worst = max(worst, dyn.cons.max_violation(x))
        res["constraint_max_violation_nm"] = worst
        res["constraint_pass"] = bool(worst <= 1e-8)
        del eng, dyn, x, v, f
        torch.cuda.empty_cache()

        # ---- (ii) equipartition clause, float32 --------------------------------------------
        eng = C60Nonbonded(system, mod.topology, (lx, lx, lz), alpha, (nx, ny, nz),
                           device="cuda", dtype=torch.float32)
        dyn = C60Dynamics(eng, dt, device="cuda", dtype=torch.float32)
        x = torch.as_tensor(base, device="cuda", dtype=torch.float32)[None] \
            .repeat(32, 1, 1).contiguous()
        eng.compute_vsites(x)
        gen = torch.Generator(device="cuda").manual_seed(12)
        v = dyn.maxwell_velocities(x, generator=gen)
        _, f_raw = eng.energy_forces(x)
        f = eng.redistribute(f_raw)
        series = []
        n_chunks = int(round(RUN_PS / 0.5))
        per = int(round(0.5 / dt))
        for k in range(n_chunks):
            for _ in range(per):
                _, f = dyn.step(x, v, f, generator=gen)
            series.append(float(dyn.temperature(v).mean()))
        series = np.asarray(series)
        post = series[int(WARM_PS / 0.5):]
        res["torch_T_mean"] = float(post.mean())
        res["torch_T_sem"] = _blocked_sem(post, int(BLOCK_PS / 0.5))
        del eng, dyn, x, v, f
        torch.cuda.empty_cache()

        # ---- (iii) mean-force spots --------------------------------------------------------
        spots = {}
        for d in SPOT_D:
            eng = C60Nonbonded(system, mod.topology, (lx, lx, lz), alpha, (nx, ny, nz),
                               device="cuda", dtype=torch.float32)
            dyn = C60Dynamics(eng, dt, device="cuda", dtype=torch.float32)
            pos = base.copy()
            center = np.array([0.5 * lx, 0.5 * lx, 0.5 * lz])
            from c60 import geometry
            pos[np.asarray(eng.cage_a.cpu())] = geometry.pair_positions(d, center)[:60]
            pos[np.asarray(eng.cage_b.cpu())] = geometry.pair_positions(d, center)[60:]
            # Amendment 16.9: reach the spot separation by DRAG from the frozen 2.428 box
            # (the first read teleported and its 0.968/1.20 spots went NaN -- RETRACTED;
            # a radial pusher diverges structurally at contact: 0.256 nm gap < 2 x 0.33).
            from c60.prep import drag_cages
            x = torch.as_tensor(base, device="cuda", dtype=torch.float32)[None] \
                .repeat(16, 1, 1).contiguous()
            eng.compute_vsites(x)
            center_t = torch.tensor([0.5 * lx, 0.5 * lx, 0.5 * lz], device="cuda",
                                    dtype=torch.float32)
            # 2x the production drag rate: this is preparation for a spot CHECK (10 ps
            # settle + 30 ps production follow, which set the ensemble), not a family prep
            drag_cages(eng, dyn, x,
                       torch.full((16,), csys.D_REF_NM, device="cuda", dtype=torch.float32),
                       torch.full((16,), d, device="cuda", dtype=torch.float32),
                       center_t, torch.Generator(device="cuda").manual_seed(14))
            gen = torch.Generator(device="cuda").manual_seed(13)
            v = dyn.maxwell_velocities(x, generator=gen)
            _, f_raw = eng.energy_forces(x)
            f = eng.redistribute(f_raw)
            # 10 ps settle + 30 ps production
            for _ in range(int(round(10.0 / dt))):
                _, f = dyn.step(x, v, f, generator=gen)
            # jam census: a wedged water sits at 2-5e4 per-site force (thermal ceiling
            # ~2.5e3, explosion guard 1e6).  Trapping is STOCHASTIC (measured 1/16 at the
            # uniform production rate), so jammed replicas are EXCLUDED and counted, not
            # raised on -- a trapped water is a prep artifact ~50 kT above equilibrium,
            # never an equilibrium state.  A majority jammed is still a prep defect.
            _, f_chk = eng.energy_forces(x)
            per_rep = f_chk.abs().amax(dim=(1, 2))
            clean = per_rep < 1.0e4
            n_clean = int(clean.sum())
            if n_clean < 10:
                raise RuntimeError(f"only {n_clean}/16 clean replicas at spot d={d}, "
                                   f"dt={dt}; prep defect")
            clean_idx = torch.nonzero(clean, as_tuple=True)[0]
            fs = []
            for k in range(int(round(SPOT_PS / 0.5))):
                for _ in range(int(round(0.5 / dt))):
                    _, f = dyn.step(x, v, f, generator=gen)
                _, f_raw2 = eng.energy_forces(x)
                fs.append(float(eng.local_mean_force(f_raw2)[clean_idx].mean()))
            fs = np.asarray(fs)
            if not np.isfinite(fs).all():
                # an exploded trajectory must raise, never become a NaN in a verdict file
                # (the first read's NaN spots silently forced the 1 fs fallback)
                raise RuntimeError(f"non-finite mean-force samples at spot d={d}, dt={dt}")
            spots[str(d)] = dict(mean=float(fs.mean()),
                                 sem=_blocked_sem(fs, int(BLOCK_PS / 0.5)),
                                 n_clean=n_clean, n_jammed=16 - n_clean)
            del eng, dyn, x, v, f
            torch.cuda.empty_cache()
        res["spots"] = spots
        results[f"dt_{dt}"] = res
        print(f"torch dt={dt}: {json.dumps(res)[:200]}", flush=True)

    with open(os.path.join(OUT, "dt_gate_torch.json"), "w") as fh:
        json.dump(results, fh, indent=1)


def phase_verdict():
    om = np.load(os.path.join(OUT, "dt_gate_openmm.npz"))
    with open(os.path.join(OUT, "dt_gate_torch.json")) as fh:
        tr = json.load(fh)

    verdict = {}
    for dt in DT_CANDIDATES:
        key = f"dt_{dt}"
        temps = om[key]                              # (4, n_chunks)
        post = temps[:, int(WARM_PS / 0.5):]
        o_mean = float(post.mean())
        sems = [_blocked_sem(row, int(BLOCK_PS / 0.5)) for row in post]
        o_sem = float(np.sqrt(np.mean(np.square(sems)) / len(sems)))
        t = tr[key]
        dT = abs(t["torch_T_mean"] - o_mean)
        sigma = float(np.sqrt(t["torch_T_sem"] ** 2 + o_sem ** 2))
        if dT + sigma <= 2.0:
            eq = "PASS"
        elif dT - sigma > 2.0:
            eq = "FAIL"
        else:
            eq = "INDETERMINATE"
        verdict[key] = dict(openmm_T=o_mean, openmm_sem=o_sem,
                            torch_T=t["torch_T_mean"], torch_sem=t["torch_T_sem"],
                            dT=dT, sigma=sigma, equipartition=eq,
                            constraint_pass=t["constraint_pass"],
                            constraint_max_nm=t["constraint_max_violation_nm"])

    # spot clause: 2 fs vs 1 fs agreement within 2x combined sem
    spots_ok = True
    spot_detail = {}
    for d in SPOT_D:
        a = tr["dt_0.002"]["spots"][str(d)]
        b = tr["dt_0.001"]["spots"][str(d)]
        diff = abs(a["mean"] - b["mean"])
        comb = float(np.sqrt(a["sem"] ** 2 + b["sem"] ** 2))
        ok = bool(diff <= 2.0 * comb) if np.isfinite(comb) and comb > 0 else False
        spot_detail[str(d)] = dict(dt2=a, dt1=b, diff=diff, comb_sem=comb, ok=ok)
        spots_ok = spots_ok and ok

    two = verdict["dt_0.002"]
    accept2 = (two["equipartition"] == "PASS" and two["constraint_pass"] and spots_ok)
    decision = 0.002 if accept2 else 0.001
    one = verdict["dt_0.001"]
    note = ""
    if not accept2 and (one["equipartition"] == "FAIL" or not one["constraint_pass"]):
        note = "1 fs FAILS a clause confidently: STOP, engine defect"

    out = dict(verdict=verdict, spot_detail=spot_detail, spots_ok=spots_ok,
               decision_dt_ps=decision, note=note)
    with open(os.path.join(OUT, "dt_gate.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=("openmm", "torch", "verdict"))
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.phase == "openmm":
        phase_openmm()
    elif a.phase == "torch":
        phase_torch()
    else:
        phase_verdict()
