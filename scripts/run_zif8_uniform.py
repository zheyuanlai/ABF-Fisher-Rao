#!/usr/bin/env python
"""Two-arm production for the ethane/ZIF-8 cell: abf vs fr_uniform, ONE
process, paired by the shared rng_seed streams and the shared init pool.
Refuses to start unless the cell was licensed by the ABF-only screen and its
FR rate frozen by the safety-only calibration.

Only two arms exist in this campaign: plain ABF, and ABF + Fisher-Rao
reallocation toward the UNIFORM marginal.  There is no sham arm and no oracle
target -- in a real molecular application nobody knows the landscape, so the
uniform target is the only one that is actually deployable.

    CUDA_VISIBLE_DEVICES=3 python -u scripts/run_zif8_uniform.py --temperature 300
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from zif8.core_zif8 import (ZIF8SimConfig, ZIF8System, engine_kwargs,  # noqa: E402
                            run_sampler)

PREREG = os.path.join(ROOT, "configs/uniform_campaign/zif8_prereg.json")


def git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--temperature", type=float, default=300.0)
    ap.add_argument("--chunk", type=int, default=None)
    ap.add_argument("--allow-verdict", default=None,
                    help="comma list of screen verdicts that license this run; "
                         "default = the preregistered selection rule")
    ap.add_argument("--only", default=None, choices=["abf", "fr_uniform"])
    a = ap.parse_args()
    pre = json.load(open(PREREG))
    tag = f"T{a.temperature:g}"
    scr = json.load(open(os.path.join(
        ROOT, f"results/uniform_campaign/zif8/screen/screen_{tag}.json")))
    allowed = (a.allow_verdict.split(",") if a.allow_verdict
               else pre["screen"]["licensed_verdicts"])
    assert scr["verdict"] in allowed, \
        f"screen verdict {scr['verdict']!r} does not license a two-arm run ({allowed})"
    sel = json.load(open(os.path.join(
        ROOT, f"results/uniform_campaign/zif8/calibration/fr_rate_selection_{tag}.json")))
    rate = sel["selected"]
    assert rate is not None, "no safe FR rate was found in the calibration"

    s = {k: v for k, v in pre["sampler"].items() if not k.startswith("_")}
    assert int(sel["fr_start_steps"]) == int(s["fr_start_steps"])
    assert int(sel["n_steps"]) == int(s["n_steps"]), \
        "the calibration horizon differs from production -- ESS would be over-certified"
    seeds = list(range(pre["production"]["seed_first"],
                       pre["production"]["seed_first"]
                       + pre["production"]["n_seeds"]))
    rng_seed = int(pre["production"]["rng_seed"])
    sim = ZIF8SimConfig(**s, rng_seed=rng_seed, fr_rate=float(rate))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        assert torch.cuda.device_count() == 1, "pin exactly one GPU"
    ek = engine_kwargs(pre)
    if a.chunk:
        ek["chunk"] = a.chunk
    system = ZIF8System(a.temperature, device, root=ROOT, **ek)
    print(f"  engine: dtype {ek['dtype']}, force kernel "
          f"{ek['force_dtype'] or ek['dtype']}, chunk {ek['chunk']}")
    pool = os.path.join(ROOT, f"cache/zif8/init_pool_{tag}.npz")
    out_dir = os.path.join(ROOT, f"results/uniform_campaign/zif8/production_{tag}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"ZIF-8 production {tag}: screen verdict {scr['verdict']}, "
          f"fr_rate={rate}, {len(seeds)} labels {seeds[0]}-{seeds[-1]}, "
          f"N={sim.n_replicas}, {sim.n_steps} steps "
          f"({sim.n_steps*sim.dt:.0f} ps), rng_seed={rng_seed}", flush=True)

    t0 = time.time()
    for method in (("abf", "fr_uniform") if a.only is None else (a.only,)):
        path = os.path.join(out_dir, f"{method}.npz")
        if os.path.exists(path):
            print(f"  skip {method} (exists)", flush=True)
            continue
        out = run_sampler(method, system, sim, seeds=seeds, init_pool=pool,
                          verbose=True, progress_every=10)
        out["seeds"] = np.asarray(seeds)
        payload = {k: v for k, v in out.items()
                   if isinstance(v, (np.ndarray, np.generic, int, float, str))}
        payload["meta"] = json.dumps(dict(
            method=method, cell=tag, seeds=seeds, rng_seed=rng_seed, fr_rate=rate,
            screen_verdict=scr["verdict"], config_hash=sim.config_hash(),
            n_replicas=sim.n_replicas, n_steps=sim.n_steps, dt=sim.dt,
            prereg=os.path.relpath(PREREG, ROOT), git_rev=git_rev(),
            host=socket.gethostname(),
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", "")))
        tmp = path + ".tmp.npz"
        np.savez_compressed(tmp, **payload)
        os.replace(tmp, path)
        print(f"  wrote {path}", flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {out_dir}")


if __name__ == "__main__":
    main()
