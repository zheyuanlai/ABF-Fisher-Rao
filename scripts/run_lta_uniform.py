#!/usr/bin/env python
"""Uniform-FR campaign, Stage 4 production: ethane/LTA, abf vs fr_uniform, 300 K.

Both arms run in ONE process from the same rng_seed, so they share initial
conditions and the dynamics noise stream (the alkanes pairing convention), and
the cross-process determinism trap cannot enter.  The FR rate must already be
frozen into the prereg by the safety-only calibration; this runner refuses to
start otherwise.  Neither arm ever receives the reference.

    CUDA_VISIBLE_DEVICES=3 python -u scripts/run_lta_uniform.py
"""
from __future__ import annotations

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
from lta.core_lta import LTAParams, LTASimConfig, LTASystem, run_sampler  # noqa: E402

PREREG = os.path.join(ROOT, "configs/uniform_campaign/lta_prereg.json")
SELECTION = os.path.join(ROOT, "results/uniform_campaign/lta/calibration/fr_rate_selection.json")
OUT = os.path.join(ROOT, "results/uniform_campaign/lta/production")
RNG_SEED = 20260831


def git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:
        return "unknown"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-prereg", default=None,
                    help="temperature-sweep prereg; requires --temperature")
    ap.add_argument("--temperature", type=float, default=None)
    a = ap.parse_args()

    if a.sweep_prereg is not None:
        assert a.temperature is not None, "--temperature required with --sweep-prereg"
        pre = json.load(open(a.sweep_prereg))
        tkey = f"{a.temperature:g}"
        pt = pre["per_T"][tkey]
        sel_path = os.path.join(ROOT, "results/uniform_campaign/lta/calibration",
                                f"fr_rate_selection_T{tkey}.json")
        sel = json.load(open(sel_path))
        rate = pt["fr_rate"]
        assert rate is not None, f"fr_rate not frozen for T={tkey}: fill per_T first"
        assert rate == sel["selected"], \
            f"prereg rate {rate} != calibration selection {sel['selected']} (T={tkey})"
        assert int(sel["fr_start_steps"]) == int(pre["sampler"]["fr_start_steps"]), \
            "calibration ran under a different fr_start than this prereg"
        seeds = list(range(pt["seeds_first"], pt["seeds_first"] + pre["seeds_count"]))
        rng_seed = int(pt["rng_seed"])
        temperature = float(a.temperature)
        out_dir = os.path.join(ROOT, "results/uniform_campaign/lta",
                               f"production_T{tkey}")
        prereg_path = a.sweep_prereg
    else:
        pre = json.load(open(PREREG))
        sel = json.load(open(SELECTION))
        rate = pre["fr_rate"]["value"]
        assert rate is not None, "fr_rate not frozen: run calibrate_lta_fr.py first"
        assert rate == sel["selected"], \
            f"prereg rate {rate} != calibration selection {sel['selected']}"
        seeds = list(range(pre["seeds"]["first"],
                           pre["seeds"]["first"] + pre["seeds"]["count"]))
        rng_seed = RNG_SEED
        temperature = pre["system"]["temperature_K"]
        out_dir = OUT
        prereg_path = PREREG
    assert pre["arms"] == ["abf", "fr_uniform"], "two arms exactly"
    s = pre["sampler"]
    sim = LTASimConfig(n_steps=s["n_steps"], n_replicas=s["n_replicas"],
                       dt=s["dt"], save_every=s["save_every"], n_grid=s["n_grid"],
                       abf_bandwidth=s["abf_bandwidth"],
                       kde_bandwidth=s["kde_bandwidth"],
                       abf_warmup_steps=s["abf_warmup_steps"],
                       abf_force_clip=s["abf_force_clip"],
                       estimator_burn_in_steps=s["estimator_burn_in_steps"],
                       fr_start_steps=s["fr_start_steps"], fr_every=s["fr_every"],
                       score_clip=s["score_clip"],
                       max_event_fraction=s["max_event_fraction"],
                       target_ema_rate=s["target_ema_rate"],
                       fr_rate=float(rate), rng_seed=rng_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        assert torch.cuda.device_count() == 1, "pin exactly one GPU"
    params = LTAParams(temperature=temperature)
    system = LTASystem(params, device, root=ROOT)
    os.makedirs(out_dir, exist_ok=True)

    print(f"UNIFORM-FR campaign, LTA production (ethane/LTA {temperature:g} K)")
    print(f"  {len(seeds)} seed labels {seeds[0]}-{seeds[-1]}, N={sim.n_replicas}, "
          f"{sim.n_steps} steps, fr_start={s['fr_start_steps']}, fr_rate={rate} "
          f"(safety-frozen), rng_seed={rng_seed}")
    print(f"  device={device} CUDA_VISIBLE_DEVICES="
          f"{os.environ.get('CUDA_VISIBLE_DEVICES', '')!r}\n", flush=True)

    t0 = time.time()
    for method in ("abf", "fr_uniform"):
        path = os.path.join(out_dir, f"{method}.npz")
        if os.path.exists(path):
            print(f"  skip {method} (exists)", flush=True)
            continue
        out = run_sampler(method, system, sim, seeds=seeds, verbose=True)
        out["seeds"] = np.asarray(seeds)
        out["config_hash"] = sim.config_hash()
        payload = {k: v for k, v in out.items()
                   if isinstance(v, (np.ndarray, np.generic, int, float, str))}
        payload["meta"] = json.dumps(dict(
            method=method, seeds=seeds, rng_seed=rng_seed, fr_rate=rate,
            config_hash=sim.config_hash(), prereg=os.path.relpath(prereg_path, ROOT),
            git_rev=git_rev(), host=socket.gethostname(),
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            temperature_K=temperature))
        tmp = path + ".tmp.npz"
        np.savez_compressed(tmp, **payload)
        os.replace(tmp, path)
        print(f"  wrote {path}", flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {out_dir}")


if __name__ == "__main__":
    main()
