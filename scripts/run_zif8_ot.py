#!/usr/bin/env python
"""Run ONE arm of the ZIF-8 Z5 six-arm block (docs/ZIF8_OT_Z4Z5.md) -- all seeds in one process.

    CUDA_VISIBLE_DEVICES=1 python -u scripts/run_zif8_ot.py --arm ot_r --alpha 0.1 --n-replicas 128 \
        --n-seeds 8 --rng-seed 20260971 --n-steps 300000 --out results/ot_repair_campaign/zif8/Z5/pilot/raw
Arms: abf (A), fr (F), ot (T), abf_r (R), fr_r (F+R), ot_r (T+R).  Corrected baseline (h_bias 0.10 A),
unwrapped gate band, determinism flags ON (paired arms).  Refuses to run on any GPU but 1.
"""
from __future__ import annotations
import argparse, json, os, socket, subprocess, sys, time
from dataclasses import asdict
import numpy as np, torch
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path.insert(0, os.path.join(ROOT, "src"))
from zif8.core_zif8 import ZIF8OTConfig, ZIF8SimConfig, ZIF8System, engine_kwargs, run_sampler
PRE = json.load(open(os.path.join(ROOT, "configs/uniform_campaign/zif8_prereg.json")))
CORR = json.load(open(os.path.join(ROOT, "configs/information_campaign/corrected_baseline_prereg.json")))
ARMS = {"abf": ("abf", False), "fr": ("fr_uniform", False), "ot": ("abf", False), "abf_r": ("abf", True), "fr_r": ("fr_uniform", True), "ot_r": ("abf", True)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS)); ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--cap-bins", type=float, default=2.0); ap.add_argument("--every", type=int, default=100); ap.add_argument("--m-repair", type=int, default=100)
    ap.add_argument("--n-replicas", type=int, required=True); ap.add_argument("--n-seeds", type=int, default=8); ap.add_argument("--rng-seed", type=int, default=20260971)
    ap.add_argument("--n-steps", type=int, default=300_000); ap.add_argument("--save-every", type=int, default=2000); ap.add_argument("--fr-rate", type=float, default=0.05)
    ap.add_argument("--out", required=True); ap.add_argument("--tag", default=""); ap.add_argument("--allow-any-gpu", action="store_true"); ap.add_argument("--dry-run", action="store_true", help="short warm-up/schedule for pipeline tests")
    a = ap.parse_args()
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cvd != "1" and not a.allow_any_gpu:
        raise SystemExit(f"GPU policy: GPU 1 only (CUDA_VISIBLE_DEVICES={cvd!r})")
    method, repaired = ARMS[a.arm]
    s = {k: v for k, v in PRE["sampler"].items() if not k.startswith("_")}
    s.update(abf_bandwidth_A=CORR["corrected_baseline"]["h_bias_A"], n_replicas=a.n_replicas, n_steps=a.n_steps, save_every=a.save_every,
             gate_band_unwrapped=True, fr_rate=(a.fr_rate if method == "fr_uniform" else 0.0))
    if a.dry_run:
        s.update(abf_warmup_steps=600, estimator_burn_in_steps=600, fr_start_steps=600, save_every=300)
    sim = ZIF8SimConfig(**s, rng_seed=a.rng_seed)
    ot = ZIF8OTConfig(alpha=(float(a.alpha) if a.arm in ("ot", "ot_r") else 0.0), cap_bins=a.cap_bins, every=a.every, m_repair=(a.m_repair if repaired else 0))
    name = f"{a.arm}" + (f"_a{ot.alpha:g}" if ot.alpha > 0 else "") + (f"_m{ot.m_repair}e{ot.every}" if ot.m_repair else "") + f"__N{a.n_replicas}__ns{a.n_seeds}__rng{a.rng_seed}__T{a.n_steps}" + (f"__{a.tag}" if a.tag else "")
    os.makedirs(a.out, exist_ok=True); path = os.path.join(a.out, name + ".npz")
    if os.path.exists(path):
        print(f"exists, skipping {path}"); return
    dev = torch.device("cuda"); system = ZIF8System(300.0, dev, root=ROOT, **engine_kwargs(PRE))
    seeds = list(range(a.n_seeds))
    print(f"[{a.arm}] method {method} alpha {ot.alpha} cap {ot.cap_bins} bins every {ot.every} m_repair {ot.m_repair} | N {a.n_replicas} seeds {a.n_seeds} rng {a.rng_seed} steps {a.n_steps} h_bias {sim.abf_bandwidth_A}", flush=True)
    t0 = time.time()
    out = run_sampler(method, system, sim, seeds=seeds, init_pool=os.path.join(ROOT, "cache/zif8/init_pool_T300.npz"), verbose=True, progress_every=25, ot=ot)
    payload = {k: v for k, v in out.items() if isinstance(v, (np.ndarray, np.generic, int, float, str))}
    try:
        rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        rev = "unknown"
    payload["meta"] = json.dumps(dict(arm=a.arm, method=method, ot=asdict(ot), sim=asdict(sim), seeds=seeds, n_replicas=a.n_replicas, n_steps=a.n_steps, h_bias_A=sim.abf_bandwidth_A,
                                      h_read_A=CORR["corrected_baseline"]["h_read_A"], wall_seconds=time.time() - t0, git_rev=rev, host=socket.gethostname(), cuda_visible_devices=cvd))
    tmp = path + ".tmp.npz"; np.savez_compressed(tmp, **payload); os.replace(tmp, path)
    print(f"[{a.arm}] done {(time.time() - t0) / 60:.1f} min; inner steps/seed {int(out.get('inner_steps_total', 0))}; NaN {not np.isfinite(np.asarray(out['pmf'])[-1]).all()} -> {os.path.relpath(path, ROOT)}", flush=True)


if __name__ == "__main__":
    main()
