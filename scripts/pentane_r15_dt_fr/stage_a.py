#!/usr/bin/env python
"""Stage A of the pentane R15 timestep campaign: does the production integrator sample the intended
Gibbs law?  Frozen design: configs/pentane_r15_dt_fr/prereg_stage_a.json.

    python scripts/pentane_r15_dt_fr/stage_a.py init                     # exact initial ensembles + exact accumulators
    python scripts/pentane_r15_dt_fr/stage_a.py run --ens flat --dt 5e-4 --clip 200 --group 0
    python scripts/pentane_r15_dt_fr/stage_a.py launch --dts 5e-4 2.5e-4 1.25e-4 --max-procs 24

One process per (ensemble, dt, clip, group); one atomic .npz each (resume-safe: existing files are skipped).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
torch.set_default_dtype(torch.float64)
from alkanes import potentials as pot                                       # noqa: E402
from alkanes import dt_validation as dv                                     # noqa: E402

OUT = os.path.join(ROOT, "results", "pentane_r15_dt_fr", "stage_a")
REF_V2 = os.path.join(ROOT, "cache", "alkanes_cv", "ref_pentane_b2_R15_v2_meanforce.npz")
N_GROUPS = 8
N_WALK = 1024
T_A = 10.0
T_BURN = 0.25
SEED_INIT = {"gibbs": 20261018, "flat": 20261019}
WALLS = (1.45, 3.65, 200.0)


def params(clip):
    return pot.AlkaneParams(n_atoms=5, beta=2.0, sigma=2.3, epsilon=1.0, decouple=False, force_clip=float(clip))


def flat_bias_mf(ref):
    """+F'_ref on the window, edge values held constant outside it (continuous bias force)."""
    win = ref["window_mask"]; idx = np.nonzero(win)[0]
    mf = np.array(ref["Fprime"], float)
    mf[: idx[0]] = mf[idx[0]]; mf[idx[-1] + 1:] = mf[idx[-1]]
    return mf


def cmd_init(a):
    os.makedirs(OUT, exist_ok=True)
    torch.set_num_threads(4)
    ref = np.load(REF_V2)
    p = params(200.0)
    t0 = time.time()
    if not os.path.exists(os.path.join(OUT, "init_gibbs.npz")):
        q, info = dv.sample_exact(N_GROUPS * N_WALK, p, SEED_INIT["gibbs"], target="gibbs", n_pool=20_000_000, chunk=1_000_000)
        np.savez(os.path.join(OUT, "init_gibbs.npz"), q=q.numpy(), info=json.dumps(info))
        print("gibbs init", info, f"{time.time() - t0:.0f}s", flush=True)
    if not os.path.exists(os.path.join(OUT, "init_flat.npz")):
        q, info = dv.sample_exact(N_GROUPS * N_WALK, p, SEED_INIT["flat"], target="flat", n_pool=200_000_000, chunk=1_000_000,
                                  grid=ref["grid"], bias=ref["F"], window_mask=ref["window_mask"])
        np.savez(os.path.join(OUT, "init_flat.npz"), q=q.numpy(), info=json.dumps(info))
        print("flat init", info, f"{time.time() - t0:.0f}s", flush=True)
    # exact accumulators (bond / bend histograms, 9-basin populations) on a weighted pool, both ensembles
    if not os.path.exists(os.path.join(OUT, "exact_accumulators.npz")):
        gen = torch.Generator().manual_seed(SEED_INIT["gibbs"] + 1)
        grid = ref["grid"]; dz = float(ref["dz"]); lo = float(grid[0]) - dz / 2
        Fref = torch.as_tensor(ref["F"]); win = torch.as_tensor(ref["window_mask"])
        tot = {}
        for _ in range(40):
            q, logw = dv.exact_gibbs_pool(1_000_000, p, gen)
            R = torch.linalg.norm(q[:, 4] - q[:, 0], dim=-1)
            b = ((R - lo) / dz).floor().long().clamp(0, grid.size - 1)
            inwin = win[b] & (R >= lo) & (R < lo + dz * grid.size)
            wg = torch.exp(logw)
            wf = torch.where(inwin, torch.exp(logw + p.beta * Fref[b]), torch.zeros_like(logw))
            phi1 = dv.geom.signed_dihedral(q, 0, 1, 2, 3); phi2 = dv.geom.signed_dihedral(q, 1, 2, 3, 4)
            b9 = dv.basin9_of(phi1, phi2)
            for tag, w in (("gibbs", wg), ("flat", wf)):
                ea = dv.exact_accumulators(q, p, grid, weights=w)
                ea["b9"] = torch.zeros(9, dtype=torch.float64).scatter_add_(0, b9, w).numpy()
                ea["fam"] = torch.zeros(4, dtype=torch.float64).scatter_add_(0, dv.family_of(phi1, phi2), w).numpy()
                ea["bond_m1"] = np.array([float((w * torch.linalg.norm(q[:, k + 1] - q[:, k], dim=-1)).sum()) for k in range(4)])
                ea["bond_m2"] = np.array([float((w * torch.linalg.norm(q[:, k + 1] - q[:, k], dim=-1) ** 2).sum()) for k in range(4)])
                ea["wsum"] = float(w.sum())
                for k in ("c", "fs", "bond", "bend", "b9", "fam", "bond_m1", "bond_m2", "wsum"):
                    tot[f"{tag}_{k}"] = tot.get(f"{tag}_{k}", 0) + np.asarray(ea[k])
        np.savez(os.path.join(OUT, "exact_accumulators.npz"), **tot)
        print("exact accumulators done", f"{time.time() - t0:.0f}s", flush=True)


def run_path(ens, dt, clip, group):
    return os.path.join(OUT, "raw", f"{ens}__dt{dt:g}__clip{clip:g}__g{group}.npz")


def cmd_run(a):
    torch.set_num_threads(1)
    path = run_path(a.ens, a.dt, a.clip, a.group)
    if os.path.exists(path):
        print("exists", path); return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ref = np.load(REF_V2)
    q0 = torch.as_tensor(np.load(os.path.join(OUT, f"init_{a.ens}.npz"))["q"])[a.group * N_WALK:(a.group + 1) * N_WALK].clone()
    p = params(a.clip)
    n_steps = int(round(T_A / a.dt)); n_burn = int(round(T_BURN / a.dt))
    assert abs(n_steps * a.dt - T_A) < 1e-9 and abs(n_burn * a.dt - T_BURN) < 1e-9
    kw = {}
    if a.ens == "flat":
        kw = dict(bias_mf=flat_bias_mf(ref), wall_lo=WALLS[0], wall_hi=WALLS[1], k_wall=WALLS[2])
    seed = 1000003 * (a.group + 1) + (0 if a.ens == "gibbs" else 77)          # same noise for every dt? no: dt-specific streams below
    seed = seed + int(round(a.dt * 1e7))                                         # clip / no-clip share the stream (paired)
    out = dv.em_validation_run(q0, p, a.dt, n_steps, seed, grid=ref["grid"], n_burn=n_burn, n_snap=40, **kw)
    out.update(dict(ens=a.ens, dt=a.dt, clip=a.clip, group=a.group, n_steps=n_steps, n_burn=n_burn, seed=seed, T_A=T_A,
                    torch_version=torch.__version__, numpy_version=np.__version__))
    tmp = path + ".tmp.npz"
    np.savez(tmp, **out); os.replace(tmp, path)
    print(f"done {path} wall {out['wall_s']:.0f}s clip_frac {out['clip_frac']:.2e} max_disp {out['max_disp']:.3f} nonfinite {out['n_nonfinite']}", flush=True)


def cmd_launch(a):
    jobs = [(e, dt, c, g) for dt in a.dts for e in ("flat", "gibbs") for c in (200.0, 0.0) for g in range(N_GROUPS)]
    jobs = [j for j in jobs if not os.path.exists(run_path(*j))]
    jobs.sort(key=lambda j: -1.0 / j[1])                                         # longest (smallest dt) first
    print(f"{len(jobs)} jobs, max {a.max_procs} concurrent", flush=True)
    os.makedirs(os.path.join(OUT, "logs"), exist_ok=True)
    running = []
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
    while jobs or running:
        while jobs and len(running) < a.max_procs:
            e, dt, c, g = jobs.pop(0)
            log = open(os.path.join(OUT, "logs", f"{e}__dt{dt:g}__clip{c:g}__g{g}.log"), "w")
            pr = subprocess.Popen([sys.executable, "-u", os.path.abspath(__file__), "run", "--ens", e, "--dt", repr(dt), "--clip", repr(c), "--group", str(g)],
                                  stdout=log, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
            running.append((pr, log, (e, dt, c, g)))
        time.sleep(5)
        still = []
        for pr, log, j in running:
            if pr.poll() is None:
                still.append((pr, log, j))
            else:
                log.close()
                print(f"[{time.strftime('%H:%M:%S')}] finished {j} rc={pr.returncode}", flush=True)
        running = still
    print("all done", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    r = sub.add_parser("run"); r.add_argument("--ens", choices=["gibbs", "flat"], required=True); r.add_argument("--dt", type=float, required=True)
    r.add_argument("--clip", type=float, default=200.0); r.add_argument("--group", type=int, required=True)
    l = sub.add_parser("launch"); l.add_argument("--dts", type=float, nargs="+", default=[5e-4, 2.5e-4, 1.25e-4]); l.add_argument("--max-procs", type=int, default=24)
    a = ap.parse_args()
    {"init": cmd_init, "run": cmd_run, "launch": cmd_launch}[a.cmd](a)


if __name__ == "__main__":
    main()
