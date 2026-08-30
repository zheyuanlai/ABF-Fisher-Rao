#!/usr/bin/env python
"""Two-arm production for one olefin/CHA cell: abf vs fr_uniform, one process,
paired by the shared rng_seed streams.  Refuses to start unless the cell was
licensed by the screen and its rate frozen by the safety calibration.

    CUDA_VISIBLE_DEVICES=3 python -u scripts/run_cha_uniform.py --guest ethene --temperature 450
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
from cha.core_cha import CHASimConfig, CHASystem, run_sampler  # noqa: E402

PREREG = os.path.join(ROOT, "configs/uniform_campaign/cha_prereg.json")


def git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--guest", required=True)
    ap.add_argument("--temperature", type=float, required=True)
    ap.add_argument("--allow-verdict", default="establishment_limited,abf_sufficient",
                    help="screen verdicts that license this two-arm run")
    a = ap.parse_args()
    pre = json.load(open(PREREG))
    tag = f"{a.guest}_{a.temperature:g}"
    scr = json.load(open(os.path.join(ROOT,
                    f"results/uniform_campaign/cha/screen/screen_{tag}.json")))
    allowed = a.allow_verdict.split(",")
    assert scr["verdict"] in allowed, \
        f"screen verdict {scr['verdict']!r} does not license a two-arm run ({allowed})"
    sel = json.load(open(os.path.join(ROOT,
                    f"results/uniform_campaign/cha/calibration/fr_rate_selection_{tag}.json")))
    rate = sel["selected"]
    assert rate is not None, "no safe rate found in calibration"

    s = {k: v for k, v in pre["sampler"].items() if not k.startswith("_")}
    assert int(sel["fr_start_steps"]) == int(s["fr_start_steps"])
    seeds = list(range(pre["production"]["seeds_first"][tag],
                       pre["production"]["seeds_first"][tag]
                       + pre["production"]["seeds_count"]))
    rng_seed = int(pre["production"]["rng_seed"][tag])
    sim = CHASimConfig(**s, rng_seed=rng_seed, fr_rate=float(rate))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        assert torch.cuda.device_count() == 1, "pin exactly one GPU"
    system = CHASystem(a.guest, a.temperature, device, root=ROOT)
    out_dir = os.path.join(ROOT, f"results/uniform_campaign/cha/production_{tag}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"CHA production {tag}: screen verdict {scr['verdict']}, fr_rate={rate}, "
          f"{len(seeds)} labels {seeds[0]}-{seeds[-1]}, N={sim.n_replicas}, "
          f"{sim.n_steps} steps, rng_seed={rng_seed}", flush=True)

    t0 = time.time()
    for method in ("abf", "fr_uniform"):
        path = os.path.join(out_dir, f"{method}.npz")
        if os.path.exists(path):
            print(f"  skip {method} (exists)", flush=True)
            continue
        out = run_sampler(method, system, sim, seeds=seeds, verbose=True)
        out["seeds"] = np.asarray(seeds)
        payload = {k: v for k, v in out.items()
                   if isinstance(v, (np.ndarray, np.generic, int, float, str))}
        payload["meta"] = json.dumps(dict(
            method=method, cell=tag, seeds=seeds, rng_seed=rng_seed, fr_rate=rate,
            screen_verdict=scr["verdict"], config_hash=sim.config_hash(),
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
