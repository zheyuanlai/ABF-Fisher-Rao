#!/usr/bin/env python
"""Run ONE arm of the pentane R15 OT + repair campaign (docs/PENTANE_R15_OT_REPAIR.md) -- all
seeds of the arm in one GPU process, one .npz.

Arms (the WCA M3 six-arm decomposition on the frozen beta 2 R15 production cell):
  A   abf      plain ABF                                 (OT alpha 0, no repair)
  F   fr       ABF + uniform Fisher-Rao (rate 0.02, cap 1 %/opportunity) on the OT domain
  T   ot       ABF + capped Wasserstein reallocation (alpha*, |dR| <= 2 bins), no repair
  R   abf_r    ABF + projected constrained rejuvenation (m_repair steps, every walker, every opportunity)
  F+R fr_r     F + the same rejuvenation
  T+R ot_r     T + the same rejuvenation

The arms share initial conditions and the outer noise stream per seed (OT consumes no RNG; FR
and the repair have their own streams).  The uniform target of F and T lives on the same
reference-free domain [R_LJ(thermal_delta), wall_hi].  Refuses to run on any GPU but 1.

    CUDA_VISIBLE_DEVICES=1 python -u scripts/run_pentane_r15_ot.py --arm ot_r --alpha 0.03 \
        --n-seeds 8 --rng-seed 20260719 --out results/ot_repair_campaign/pentane_r15/pilot/raw
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
torch.set_default_dtype(torch.float64)
from alkanes import potentials as pot, core_dist as cd                     # noqa: E402
from alkanes.distance_cv import DistanceCV                                 # noqa: E402
from alkanes.ot_repair_dist import DistOTConfig, lj_forbidden_radius, compiled_forces, eager_forces   # noqa: E402

ARMS = {"abf": ("abf", False), "fr": ("fr_uniform", False), "ot": ("abf", False),
        "abf_r": ("abf", True), "fr_r": ("fr_uniform", True), "ot_r": ("abf", True)}
# frozen production cell: configs/alkanes_cv_extension/r15_methods.yaml (stage production)
CELL = dict(beta=2.0, sigma=2.3, epsilon=1.0, force_clip=200.0, dt=5.0e-4, R_lo=1.4, R_hi=3.7, wall_lo=1.45, wall_hi=3.65,
            k_wall=200.0, n_grid=256, abf_bandwidth=0.04, kde_bandwidth=0.06, abf_warmup_steps=5000, abf_force_clip=60.0,
            estimator_burn_in_steps=6000, fr_rate=0.02, score_clip=2.0, fr_start_steps=12000, fr_every=5,
            target_ema_rate=0.005, max_event_fraction=0.01, n_grid2=48, n_rbins=12, n_replicas=1024, thermal_delta=10.0)
DZ = (CELL["R_hi"] - CELL["R_lo"]) / CELL["n_grid"]


def build(args):
    method, repaired = ARMS[args.arm]
    p = pot.AlkaneParams(n_atoms=5, beta=CELL["beta"], sigma=CELL["sigma"], epsilon=CELL["epsilon"], decouple=False, force_clip=CELL["force_clip"])
    lo = args.domain_lo if args.domain_lo is not None else lj_forbidden_radius(p, CELL["thermal_delta"])
    domain = (float(lo), float(args.domain_hi))
    sim = cd.DistSimConfig(dt=CELL["dt"], n_steps=args.n_steps, n_replicas=args.n_replicas, save_every=args.save_every, rng_seed=args.rng_seed,
                           R_lo=CELL["R_lo"], R_hi=CELL["R_hi"], wall_lo=CELL["wall_lo"], wall_hi=CELL["wall_hi"], k_wall=CELL["k_wall"],
                           n_grid=CELL["n_grid"], abf_bandwidth=CELL["abf_bandwidth"], kde_bandwidth=CELL["kde_bandwidth"],
                           abf_warmup_steps=CELL["abf_warmup_steps"], abf_force_clip=CELL["abf_force_clip"],
                           estimator_burn_in_steps=CELL["estimator_burn_in_steps"], fr_rate=CELL["fr_rate"], score_clip=CELL["score_clip"],
                           fr_start_steps=CELL["fr_start_steps"], fr_every=CELL["fr_every"], target_ema_rate=CELL["target_ema_rate"],
                           max_event_fraction=CELL["max_event_fraction"], fr_domain=(domain if method == "fr_uniform" else None),
                           n_grid2=CELL["n_grid2"], n_rbins=CELL["n_rbins"])
    alpha = float(args.alpha) if args.arm in ("ot", "ot_r") else 0.0
    ot = DistOTConfig(alpha=alpha, dR_max=float(args.dR_max), m_repair=(int(args.m_repair) if repaired else 0), domain=domain)
    return p, sim, ot, method


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--alpha", type=float, default=0.0, help="OT displacement fraction (T, T+R arms)")
    ap.add_argument("--dR-max", type=float, default=2 * DZ, help="per-event cap on |dR| (default 2 grid bins)")
    ap.add_argument("--m-repair", type=int, default=5, help="projected inner steps per opportunity (repaired arms)")
    ap.add_argument("--domain-lo", type=float, default=None, help="uniform-target lower edge (default: LJ rule at thermal_delta)")
    ap.add_argument("--domain-hi", type=float, default=CELL["wall_hi"])
    ap.add_argument("--n-seeds", type=int, default=8)
    ap.add_argument("--rng-seed", type=int, default=20260719)
    ap.add_argument("--n-steps", type=int, default=80000)
    ap.add_argument("--save-every", type=int, default=4000)
    ap.add_argument("--n-replicas", type=int, default=CELL["n_replicas"])
    ap.add_argument("--out", required=True, help="raw directory")
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--allow-any-gpu", action="store_true")
    args = ap.parse_args()
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if torch.cuda.is_available() and cvd != "1" and not args.allow_any_gpu:
        raise SystemExit(f"GPU policy: this campaign runs on GPU 1 only (CUDA_VISIBLE_DEVICES={cvd!r})")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    p, sim, ot, method = build(args)
    force_fn = eager_forces if (args.no_compile or dev == "cpu") else compiled_forces()
    name = f"{args.arm}" + (f"_a{args.alpha:g}" if args.arm in ("ot", "ot_r") else "") + (f"_m{args.m_repair}" if ot.m_repair > 0 else "") \
        + f"__ns{args.n_seeds}__rng{args.rng_seed}__T{args.n_steps}" + (f"__{args.tag}" if args.tag else "")
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, name + ".npz")
    if os.path.exists(path):
        print(f"exists, skipping: {path}"); return
    print(f"[{args.arm}] method {method} alpha {ot.alpha} cap {ot.dR_max:.4f} m_repair {ot.m_repair} domain {ot.domain} "
          f"seeds {args.n_seeds} rng {args.rng_seed} steps {args.n_steps} device {dev} CVD {cvd!r} force {'eager' if force_fn is eager_forces else 'compiled'}", flush=True)
    t0 = time.time()
    out = cd.run_sampler_dist(method, p, sim, list(range(args.n_seeds)), DistanceCV(0, 4), dev, initial_dihedrals=[0.0, 0.0],
                              collect_conditional=True, verbose=True, ot=ot, force_fn=force_fn)
    wall = time.time() - t0
    save = {}
    for k, v in out.items():
        if k == "first_discovery":
            for kk, vv in v.items():
                save[f"first_discovery_{kk}"] = np.asarray(vv)
        elif v is None:
            continue
        elif isinstance(v, (str, int, float, np.integer, np.floating, bool)):
            save[k] = np.asarray(v)
        else:
            save[k] = np.asarray(v)
    save.update(dict(arm=args.arm, name=name, method=method, alpha=ot.alpha, dR_max=ot.dR_max, m_repair=ot.m_repair, domain=np.asarray(ot.domain),
                     n_seeds=args.n_seeds, rng_seed=args.rng_seed, n_steps=args.n_steps, save_every=args.save_every, n_replicas=args.n_replicas,
                     wall_seconds=wall, device=dev, cuda_visible_devices=cvd, force_impl=("eager" if force_fn is eager_forces else "compiled"),
                     sim_json=json.dumps(asdict(sim)), ot_json=json.dumps(asdict(ot)), cell_json=json.dumps(CELL),
                     had_nan=bool(not np.isfinite(out["pmf"][-1]).all())))
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **save); os.replace(tmp, path)
    print(f"[{args.arm}] done {wall / 60:.1f} min; inner steps/seed {int(out.get('inner_steps_total', 0))}; "
          f"moved frac {float(np.mean(out.get('ot_moved_frac', np.zeros(1)))):.3f} capped {float(np.mean(out.get('ot_capped_frac', np.zeros(1)))):.3f}; "
          f"NaN {save['had_nan']}; -> {os.path.relpath(path, ROOT)}", flush=True)


if __name__ == "__main__":
    main()
