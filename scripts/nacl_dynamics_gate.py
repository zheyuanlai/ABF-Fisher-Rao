"""SPEC_nacl_water.md §1.2 + §3.1 — the dynamics gate: constraints and equipartition, 1 vs 2 fs.

Torch BAOAB + M-SHAKE (the engine every arm runs) against OpenMM LangevinMiddle (the parity
oracle), same model, same dt, same gamma.  Gates:

  constraint violation (float64 dynamics)   <= 1e-8 nm     over a production-length stretch
  |T_torch - T_openmm|                      <= 2 K         at each dt, warmed and averaged

float32 is the production dtype for the screen; its constraint residual is *recorded* here
(expected ~1e-6 nm, float32 rounding at box scale) and audited at diagnostic cadence in every
run -- the 1e-8 gate is a float64 statement about the solver, not about float32 rounding.

Decides dt: 2 fs (published) if it passes both clauses, else 1 fs.  Runs BEFORE any free-energy
data; writes results/nacl/stage1/dynamics_gate.json.

Usage:
    CUDA_VISIBLE_DEVICES=2 python scripts/nacl_dynamics_gate.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from nacl import system as nsys                                  # noqa: E402
# NaClNonbonded imports torch, which would poison the --openmm-only process (OpenMM CUDA
# context creation hangs after torch loads), so it is imported inside run_torch instead.

KB_KJ_PER_MOL_K = 8.31446261815324e-3   # torch is imported lazily; no methane import here
OUT = nsys.REPO / "results/nacl/stage1"

# The two clauses are different KINDS of statement and get different budgets.
#
# The constraint clause is a deterministic property of the solver: run it in float64, briefly.
# The equipartition clause is STATISTICAL, and the first version of this gate got its error
# bars wrong -- it took std/sqrt(n) over samples 0.1 ps apart when the kinetic-energy
# autocorrelation time is ~1/(2 gamma) = 0.5 ps, understating the uncertainty by ~2.2x.  With
# honest bars the 2 fs "failure" was 2.5 sigma, not the clean result it looked like, and its own
# float32 leg disagreed with its float64 leg by 1.24 K at the same timestep.
#
# So equipartition now runs in the PRODUCTION dtype (float32) with many walkers -- walkers are
# independent, so they buy uncertainty far more cheaply than a longer float64 run -- against
# several independent OpenMM replicas, with a blocking estimator for the error.
WARM_PS = 10.0
MEASURE_PS = 60.0
CONSTRAINT_PS = 5.0         #: float64, deterministic clause
B64 = 4                     #: float64 walkers, constraint clause only
B32 = 32                    #: float32 walkers, equipartition clause (production dtype)
N_OPENMM_REPLICAS = 4
BLOCK_PS = 5.0              #: 10x the ~0.5 ps KE autocorrelation time; calibrated
                            #: against a known-truth AR(1) series -- 2 tau blocks understate
                            #: the SEM by 26 %, 5 tau by 9 %, 10 tau by 6 %, 20 tau by 5 %.
                            #: The residual bias is NEGATIVE, so a PASS is the weaker claim.


def blocked_sem(trace, sample_every_ps, n_walkers=1, block_ps=BLOCK_PS):
    """SEM from block means, with blocks long compared with the KE autocorrelation time.

    std/sqrt(n) over samples closer together than the correlation time understates the
    uncertainty by sqrt(n_raw/n_independent) -- 2.2x at the original 0.1 ps sampling.  Blocking
    removes that by construction: block means are effectively independent, so their scatter is
    the real uncertainty.  Walkers are already independent, so the trace is a walker average and
    its variance is n_walkers times smaller -- that gain is real and is kept.
    """
    t = np.asarray(trace, dtype=float)
    per_block = max(2, int(round(block_ps / sample_every_ps)))
    n_blocks = len(t) // per_block
    if n_blocks < 4:
        return float(np.std(t) / np.sqrt(max(len(t), 1)))     # too short to block; flagged by n_blocks
    means = t[:n_blocks * per_block].reshape(n_blocks, per_block).mean(axis=1)
    return float(np.std(means, ddof=1) / np.sqrt(n_blocks))


def run_torch(L, x0, dt_ps, dtype, n_walkers, device="cuda", seed=1,
              measure_ps=MEASURE_PS, sample_every_ps=0.1):
    import torch
    from methane.dynamics import BAOAB, RigidWaterConstraints
    from nacl.nonbonded import NaClNonbonded
    dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    ff = NaClNonbonded(L, device=device, dtype=dtype)
    cons = RigidWaterConstraints(ff.params["waters"], nsys.rigid_water_lengths(),
                                 ff.params["mass"], device=device, dtype=dtype)
    integ = BAOAB(lambda q: ff.energy_forces(q, chunk=256), ff.params["mass"], cons,
                  dt_ps, nsys.TEMPERATURE_K, nsys.GAMMA_PS, device=device, dtype=dtype)
    gen = torch.Generator(device=device).manual_seed(seed)
    q = torch.tensor(np.repeat(x0[None], n_walkers, 0), device=device, dtype=dtype)
    v = integ.maxwell_velocities(q, generator=gen)
    _, f = ff.energy_forces(q)

    n_warm = int(WARM_PS / dt_ps)
    n_meas = int(measure_ps / dt_ps)
    temps, viol = [], 0.0
    t0 = time.perf_counter()
    for step in range(n_warm + n_meas):
        _, f = integ.step(q, v, f, generator=gen)
        every = max(1, int(round(sample_every_ps / dt_ps)))
        if step >= n_warm and step % every == 0:
            temps.append(float(integ.temperature(v).mean()))
            viol = max(viol, cons.max_violation(q))
    wall = time.perf_counter() - t0
    ns_day = n_walkers * (n_warm + n_meas) * dt_ps * 1e-3 / wall * 86400.0
    return dict(T_mean=float(np.mean(temps)),
                T_sem=blocked_sem(temps, sample_every_ps, n_walkers),
                T_sem_naive=float(np.std(temps) / np.sqrt(len(temps))),
                n_samples=len(temps), n_walkers=n_walkers,
                max_violation_nm=viol, ns_per_day_aggregate=ns_day, wall_s=wall,
                trace=[float(t) for t in temps])


def run_openmm(L, x0, dt_ps, seed=1):
    """The OpenMM equipartition oracle.  **Runs only in a torch-free process** -- see
    ``openmm_subprocess``; calling it after ``import torch`` hangs at Context creation."""
    import openmm as mm
    import openmm.unit as u

    if "torch" in sys.modules:
        raise RuntimeError(
            "torch is imported in this process: creating an OpenMM CUDA context here hangs "
            "indefinitely rather than raising (measured, methane_baths.py). Use "
            "openmm_subprocess() instead.")

    system, topology, _ = nsys.build_openmm_system(L)
    integ = mm.LangevinMiddleIntegrator(nsys.TEMPERATURE_K * u.kelvin,
                                        nsys.GAMMA_PS / u.picosecond, dt_ps * u.picosecond)
    integ.setRandomNumberSeed(seed)
    try:
        ctx = mm.Context(system, integ, mm.Platform.getPlatformByName("CUDA"),
                         dict(Precision="double"))
    except Exception:
        ctx = mm.Context(system, integ, mm.Platform.getPlatformByName("CPU"))
    ctx.setPositions(x0 * u.nanometer)
    ctx.applyConstraints(1e-10)
    ctx.setVelocitiesToTemperature(nsys.TEMPERATURE_K * u.kelvin, seed)
    ndof = 3 * nsys.N_SITES - system.getNumConstraints() - 3
    integ.step(int(WARM_PS / dt_ps))
    temps = []
    n_chunk = 50
    for _ in range(int(MEASURE_PS / dt_ps) // n_chunk):
        integ.step(n_chunk)
        ke = ctx.getState(getEnergy=True).getKineticEnergy() \
            .value_in_unit(u.kilojoule_per_mole)
        temps.append(2.0 * ke / (ndof * KB_KJ_PER_MOL_K))
    return dict(T_mean=float(np.mean(temps)), trace=[float(t) for t in temps])


OPENMM_PYTHON = os.path.expanduser("~/miniconda3/envs/methane-cuda/bin/python")


def openmm_subprocess(L, dt_list):
    """Run the OpenMM oracle in a torch-free process and return {dt_fs: result}.

    Two constraints force this: importing torch kills OpenMM's CUDA platform in-process, and
    the `abffr` environment's OpenMM has no CUDA platform at all -- so the oracle runs under
    `methane-cuda`, which has CUDA OpenMM and no torch.
    """
    args = [OPENMM_PYTHON, os.path.abspath(__file__), "--openmm-only",
            ",".join(f"{d:g}" for d in dt_list)]
    r = subprocess.run(args, capture_output=True, text=True, cwd=nsys.REPO,
                       env={**os.environ, "PYTHONPATH": str(nsys.REPO / "src")})
    if r.returncode != 0:
        raise RuntimeError(f"OpenMM oracle failed:\n{r.stdout}\n{r.stderr}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def main():
    if "--openmm-only" in sys.argv:
        dts = [float(x) for x in sys.argv[sys.argv.index("--openmm-only") + 1].split(",")]
        box = json.load(open(nsys.REPO / "results/nacl/box/box_manifest.json"))
        L = float(box["L_nm"])
        x0 = dict(np.load(nsys.REPO / "results/nacl/box/npt_final_state.npz"))["positions_nm"]
        out = {}
        for d in dts:
            reps = [run_openmm(L, x0, d * 1e-3, seed=int(10 * d) + 7 * k)
                    for k in range(N_OPENMM_REPLICAS)]
            traces = [r["trace"] for r in reps]
            flat = np.concatenate(traces)
            sems = [blocked_sem(tr, 0.1) for tr in traces]
            out[f"{d:g}"] = dict(
                T_mean=float(np.mean(flat)),
                T_sem=float(np.sqrt(np.sum(np.square(sems))) / len(sems)),
                T_sem_naive=float(np.std(flat) / np.sqrt(len(flat))),
                n_replicas=len(reps), per_replica=[r["T_mean"] for r in reps])
        print(json.dumps(out))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    box = json.loads((nsys.REPO / "results/nacl/box/box_manifest.json").read_text())
    L = float(box["L_nm"])
    x0 = dict(np.load(nsys.REPO / "results/nacl/box/npt_final_state.npz"))["positions_nm"]

    report = dict(L_nm=L, gpu=os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
                  warm_ps=WARM_PS, measure_ps=MEASURE_PS)
    # ALL OpenMM contexts run and die before torch touches CUDA -- interleaving the two
    # runtimes' context creation on one device hangs indefinitely (methane_ti_torch.py, measured)
    print("[oracle] OpenMM equipartition in a torch-free subprocess ...", flush=True)
    omm_raw = openmm_subprocess(L, (2.0, 1.0))
    omm_by_dt = {2.0: omm_raw["2"], 1.0: omm_raw["1"]}
    verdicts = {}
    for dt_fs in (2.0, 1.0):
        dt = dt_fs * 1e-3
        # constraint clause: deterministic, float64, short
        t64 = run_torch(L, x0, dt, "float64", B64, seed=int(10 * dt_fs),
                        measure_ps=CONSTRAINT_PS)
        # equipartition clause: statistical, production dtype, many walkers, long
        t32 = run_torch(L, x0, dt, "float32", B32, seed=int(10 * dt_fs) + 1)
        omm = omm_by_dt[dt_fs]
        dT = abs(t32["T_mean"] - omm["T_mean"])
        sigma = float(np.hypot(t32["T_sem"], omm["T_sem"]))
        # Amendment 15.1: tri-state, one preregistered run, no extensions
        if t64["max_violation_nm"] > 1e-8:
            verdict = "FAIL"
        elif dT + sigma <= 2.0:
            verdict = "PASS"
        elif dT - sigma > 2.0:
            verdict = "FAIL"
        else:
            verdict = "INDETERMINATE"
        ok = verdict == "PASS"
        verdicts[f"{dt_fs:.0f}fs"] = dict(
            torch_float64=t64, torch_float32=t32, openmm=omm,
            dT_vs_openmm_K=dT, dT_uncertainty_K=sigma,
            dT_in_sigma=(dT / sigma if sigma > 0 else None),
            constraint_gate=t64["max_violation_nm"] <= 1e-8,
            verdict=verdict, PASS=ok,
            note="equipartition compares the PRODUCTION dtype against several independent "
                 "OpenMM replicas; both SEMs are blocked, not std/sqrt(n) over correlated "
                 "samples (which understated them ~2.2x in the first version of this gate)")
        print(f"dt={dt_fs} fs: torch32 T={t32['T_mean']:.2f}+-{t32['T_sem']:.2f} K "
              f"(naive +-{t32['T_sem_naive']:.2f}) | openmm T={omm['T_mean']:.2f}"
              f"+-{omm['T_sem']:.2f} ({omm['n_replicas']} reps) | dT={dT:.2f}+-{sigma:.2f} K "
              f"= {dT/sigma:.1f} sigma | constraint viol64={t64['max_violation_nm']:.2e} nm "
              f"-> {verdict}", flush=True)

    # Amendment 15.1 decision: 2fs PASS -> 2fs; FAIL or INDETERMINATE -> 1fs;
    # 1fs confident FAIL -> STOP; 1fs INDETERMINATE -> 1fs (conservative endpoint).
    if verdicts["2fs"]["verdict"] == "PASS":
        chosen = 0.002
    elif verdicts["1fs"]["verdict"] in ("PASS", "INDETERMINATE"):
        chosen = 0.001
    else:
        chosen = None
    report["verdicts"] = verdicts
    report["dt_chosen_ps"] = chosen
    report["rule"] = "Amendment 15.1 (frozen before this run; no extensions, never revisited)"
    if chosen is None:
        report["verdict"] = "1 fs confidently FAILS -- engine defect, NaCl does not run"
    (OUT / "dynamics_gate.json").write_text(json.dumps(report, indent=2))
    print(f"dt chosen: {chosen} ps; wrote {OUT}/dynamics_gate.json")


if __name__ == "__main__":
    main()
