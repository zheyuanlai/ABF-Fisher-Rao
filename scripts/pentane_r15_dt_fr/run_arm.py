#!/usr/bin/env python
"""Run ONE arm (ABF or ABF + uniform-marginal Fisher--Rao) of the pentane R15 timestep campaign
(docs/PENTANE_R15_DT_FR.md) on CPU: ``--n-seeds`` matched seeds in one single-threaded process, one .npz.

The frozen production cell (configs/alkanes_cv_extension/r15_methods.yaml, stage production) with every
time-dependent knob rescaled to physical time by :func:`alkanes.dt_schedule.physical_schedule` -- the only
things that change with ``--dt`` are the step counts.  ABF and FR arms with the same ``--rng-seed`` share the
initial configurations and the dynamics noise stream; FR draws come from its own stream.  The reference free
energy is never loaded here (evaluation only, in the analysis script).

    python scripts/pentane_r15_dt_fr/run_arm.py --arm fr --fr-rate 0.02 --dt 1.25e-4 --T 40 \
        --rng-seed 20261101 --n-seeds 4 --out results/pentane_r15_dt_fr/stage_cd_pilot/raw
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
torch.set_default_dtype(torch.float64)
from alkanes import potentials as pot, core_dist as cd                     # noqa: E402
from alkanes.distance_cv import DistanceCV                                 # noqa: E402
from alkanes.dt_schedule import physical_schedule                          # noqa: E402

# frozen production cell (r15_methods.yaml stage production); fr_rate is the arm's
CELL = dict(beta=2.0, sigma=2.3, epsilon=1.0, force_clip=200.0, R_lo=1.4, R_hi=3.7, wall_lo=1.45, wall_hi=3.65, k_wall=200.0,
            n_grid=256, abf_bandwidth=0.04, kde_bandwidth=0.06, abf_force_clip=60.0, score_clip=2.0, target_ema_rate=0.005,
            max_event_fraction=0.01, n_grid2=48, n_rbins=12, n_replicas=1024, init="all-trans (phi1, phi2) = (0, 0) + 1e-3 noise")


class ClipStatsForces:
    """The production force (arithmetic identical to potentials.forces) + per-seed clip activity."""

    def __init__(self, R, N):
        self.R, self.N = R, N
        self.n_clip = np.zeros(R, dtype=np.int64); self.n_eval = np.zeros(R, dtype=np.int64)
        self.excess = np.zeros(R); self.max_raw = np.zeros(R)

    def __call__(self, q, p):
        q = q.detach().requires_grad_(True)
        e = pot.total_energy(q, p).sum()
        (g,) = torch.autograd.grad(e, q, create_graph=False)
        f = -g
        if p.force_clip and p.force_clip > 0:
            norm = torch.linalg.norm(f, dim=-1, keepdim=True)
            scale = torch.clamp(p.force_clip / norm.clamp_min(pot.EPS), max=1.0)
            f = f * scale
            nr = norm[..., 0].reshape(self.R, -1)
            self.n_clip += (nr > p.force_clip).sum(-1).numpy()
            self.excess += (nr - p.force_clip).clamp_min(0.0).sum(-1).numpy()
            self.max_raw = np.maximum(self.max_raw, nr.amax(-1).detach().numpy())
        self.n_eval += q.shape[0] // self.R * q.shape[1]
        return f.detach()


def git_commit():
    try:
        c = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.call(["git", "-C", ROOT, "diff", "--quiet", "HEAD", "--", "src", "scripts/pentane_r15_dt_fr"]) != 0
        return c + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def build(args):
    sch = physical_schedule(args.dt, T=args.T, deposit_interval_t=args.deposit_interval)
    method = "abf" if args.arm == "abf" else "fr_uniform"
    p = pot.AlkaneParams(n_atoms=5, beta=CELL["beta"], sigma=CELL["sigma"], epsilon=CELL["epsilon"], decouple=False,
                         force_clip=CELL["force_clip"])
    sim = cd.DistSimConfig(dt=args.dt, n_steps=sch["n_steps"], n_replicas=CELL["n_replicas"], save_every=sch["save_every"],
                           rng_seed=args.rng_seed, R_lo=CELL["R_lo"], R_hi=CELL["R_hi"], wall_lo=CELL["wall_lo"], wall_hi=CELL["wall_hi"],
                           k_wall=CELL["k_wall"], n_grid=CELL["n_grid"], abf_bandwidth=CELL["abf_bandwidth"],
                           kde_bandwidth=CELL["kde_bandwidth"], abf_warmup_steps=sch["abf_warmup_steps"],
                           abf_force_clip=CELL["abf_force_clip"], estimator_burn_in_steps=sch["estimator_burn_in_steps"],
                           fr_rate=(args.fr_rate if method != "abf" else 0.0), score_clip=CELL["score_clip"],
                           fr_start_steps=sch["fr_start_steps"], fr_every=sch["fr_every"], target_ema_rate=CELL["target_ema_rate"],
                           max_event_fraction=CELL["max_event_fraction"], fr_domain=None, n_grid2=CELL["n_grid2"], n_rbins=CELL["n_rbins"],
                           deposit_stride=sch["deposit_stride"], numerics_diag=True, track_families=True)
    return p, sim, method, sch


def run_name(args):
    nm = f"{args.arm}" + (f"_r{args.fr_rate:g}" if args.arm == "fr" else "") + f"__dt{args.dt:g}__T{args.T:g}"
    if args.deposit_interval is not None:
        nm += f"__dep{args.deposit_interval:g}"
    return nm + f"__ns{args.n_seeds}__rng{args.rng_seed}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["abf", "fr"])
    ap.add_argument("--fr-rate", type=float, default=0.02)
    ap.add_argument("--dt", type=float, required=True)
    ap.add_argument("--T", type=float, default=40.0)
    ap.add_argument("--deposit-interval", type=float, default=None, help="physical deposit interval (default: every step)")
    ap.add_argument("--n-seeds", type=int, default=4)
    ap.add_argument("--rng-seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    p, sim, method, sch = build(args)
    name = run_name(args)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, name + ".npz")
    if os.path.exists(path):
        print(f"exists, skipping: {path}"); return
    print(f"[{name}] method {method} schedule {sch} rate {sim.fr_rate} host {platform.node()}", flush=True)
    ff = ClipStatsForces(args.n_seeds, sim.n_replicas)
    t0 = time.time()
    out = cd.run_sampler_dist(method, p, sim, list(range(args.n_seeds)), DistanceCV(0, 4), "cpu", initial_dihedrals=[0.0, 0.0],
                              oracle_free_energy=None, collect_conditional=True, verbose=True, force_fn=ff)
    wall = time.time() - t0
    save = {}
    for k, v in out.items():
        if k == "first_discovery":
            for kk, vv in v.items():
                save[f"first_discovery_{kk}"] = np.asarray(vv)
        elif v is not None:
            save[k] = np.asarray(v)
    meta = dict(arm=args.arm, method=method, fr_rate=sim.fr_rate, dt=args.dt, T=args.T, deposit_interval=args.deposit_interval,
                n_seeds=args.n_seeds, rng_seed=args.rng_seed, schedule=sch, sim=asdict(sim), cell=CELL, git_commit=git_commit(),
                torch=torch.__version__, numpy=np.__version__, python=platform.python_version(), host=platform.node(),
                threads=torch.get_num_threads(), device="cpu", wall_seconds=wall,
                force_evals_per_seed=int(sch["n_steps"] + 1) * sim.n_replicas)
    save.update(dict(meta_json=json.dumps(meta), arm=args.arm, name=name, wall_seconds=wall, n_steps=sch["n_steps"], n_replicas=sim.n_replicas,
                     clip_n=ff.n_clip, clip_evals=ff.n_eval, clip_excess=ff.excess, clip_max_raw=ff.max_raw,
                     had_nan=bool(not np.isfinite(out["pmf"][-1]).all())))
    tmp = path + ".tmp.npz"
    np.savez(tmp, **save); os.replace(tmp, path)
    with open(path + ".sha256", "w") as fh:
        fh.write(hashlib.sha256(open(path, "rb").read()).hexdigest() + "\n")
    print(f"wrote {path} wall {wall / 60:.1f} min clip_frac {ff.n_clip.sum() / max(ff.n_eval.sum(), 1):.2e} "
          f"max_disp {np.max(out['max_step_displacement']):.3f} nonfinite {int(np.sum(out['n_nonfinite_walker_steps']))}", flush=True)


if __name__ == "__main__":
    main()
