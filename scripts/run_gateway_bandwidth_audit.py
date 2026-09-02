#!/usr/bin/env python
"""Entropic gateway corrected-baseline audit, step 1: ABF-only ONLINE bandwidth matrix.

Preregistration: configs/information_campaign/gateway_baseline_audit_prereg.json.
Three ABF arms differing ONLY in the online bandwidth h (legacy 0.07, 1/2, 1/4); the
sampler/cell block is asserted against the frozen confirmatory prereg.  One batch per
arm with the SAME (init, seed) rows and the SAME batch_seed -> identical Langevin noise
across arms (no FR draws), so the arms are paired exactly.  Raw accumulators Sf, C are
recorded at every save so analyze_gateway_bandwidth_audit.py sweeps the read-out
offline and exactly.

Prints wall time only -- no error metric (prereg prohibition).

    CUDA_VISIBLE_DEVICES=3 python -u scripts/run_gateway_bandwidth_audit.py   # GPU 3 ONLY
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
import gateway_core as gw  # noqa: E402

PREREG = os.path.join(ROOT, "configs/information_campaign/gateway_baseline_audit_prereg.json")
BASE_PREREG = os.path.join(ROOT, "results/gateway_anchor/CONFIRMATORY_PREREGISTRATION.json")
OUT_DIR = os.path.join(ROOT, "results", "information_campaign", "gateway_baseline_audit")
BATCH_SEED = 31_000


def git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def build_config(base_sampler, cell, init, h):
    s, c = base_sampler, cell
    return gw.GatewayConfig(
        beta=c["beta"], H=c["beta_H_kT"] / c["beta"], omega_out=1.0, r=c["r"], s=c["s"],
        N=s["N"], dt=s["dt"], n_steps=s["n_steps"], save_every=s["save_every"], init=init,
        h=float(h), min_count=s["min_count"], gamma=float("nan"), eta=s["eta"],
        fr_every=s["fr_every"], fr_burnin=s["fr_burnin"], ramp_fraction=s["ramp_fraction"],
        target_ema_rate=s["target_ema_rate"], score_clip=s["score_clip"],
        max_event_fraction=s["max_event_fraction"], ess_window_steps=s["ess_window_steps"])


def save_arm(path, recs, h):
    keys = ["t", "P_regions", "Q_regions", "l2_f_t", "l2_fp_t", "ess_t", "wmax_t",
            "x_grid", "F_hat", "Fp_hat", "F_ref", "Fp_ref",
            "F_prof_t", "Fp_prof_t", "phat_t", "kl_uniform_t", "Sf_t", "C_t"]
    npz = {k: np.stack([r[k] for r in recs]) for k in keys}          # float64 throughout
    for k in recs[0]:
        if k not in keys and k != "config":
            npz[k] = np.array([r[k] for r in recs])
    npz["config_json"] = np.array([json.dumps(r["config"], sort_keys=True) for r in recs])
    npz["h_bias"] = np.array(float(h))
    np.savez_compressed(path, **npz)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    pre = json.load(open(PREREG))
    base = json.load(open(BASE_PREREG))
    sampler, cell = base["sampler"], base["cell"]
    assert pre["cell"] == cell, "cell block drifted from the frozen confirmatory prereg"
    assert pre["inits"] == base["inits"]
    assert float(pre["arms_h_bias"][0]) == float(sampler["h"]) == 0.07, "first arm must be the legacy h"
    assert float(sampler["min_count"]) == 1.0
    seeds = list(range(pre["seed_first"], pre["seed_first"] + pre["n_seeds"]))
    assert not (set(seeds) & (set(range(16)) | set(range(100, 132)))), "reused labels"

    if torch.cuda.is_available():
        assert torch.cuda.device_count() == 1, "pin exactly one GPU"
    elif not a.dry_run:
        print("WARNING: no CUDA device visible; this will be slow", flush=True)
    os.makedirs(a.out, exist_ok=True)
    rows = [(init, sd) for init in pre["inits"] for sd in seeds]
    print("Gateway corrected-baseline audit, step 1: ABF-only online bandwidth matrix")
    print(f"  arms h_bias {pre['arms_h_bias']}; min_count {sampler['min_count']} fixed; "
          f"{len(rows)} (init, seed) rows per arm, batch_seed {BATCH_SEED} shared")
    print(f"  seeds {seeds[0]}-{seeds[-1]}, inits {pre['inits']}; no error metric is printed here (prereg)")
    if a.dry_run:
        for h in pre["arms_h_bias"]:
            p = os.path.join(a.out, f"raw_hb{h:g}.npz")
            print(f"  {'EXISTS' if os.path.exists(p) else 'run   '}  {os.path.relpath(p, ROOT)}")
        return

    t_start, done = time.time(), []
    for h in pre["arms_h_bias"]:
        path = os.path.join(a.out, f"raw_hb{h:g}.npz")
        if os.path.exists(path):
            print(f"  skip {os.path.relpath(path, ROOT)}", flush=True)
            done.append(path)
            continue
        cfgs = [build_config(sampler, cell, init, h) for init, _ in rows]
        spec = gw.BatchSpec(configs=cfgs, seeds=[sd for _, sd in rows], methods=[gw.ABF],
                            batch_seed=BATCH_SEED)
        t0 = time.time()
        recs = gw.simulate_batch(spec, store_profiles=True, store_accumulators=True)
        assert all(abs(float(json.loads(json.dumps(r["config"]))["h"]) - h) < 1e-12 for r in recs)
        save_arm(path, recs, h)
        done.append(path)
        print(f"  arm h_bias={h:g}: {len(recs)} rows saved ({time.time() - t0:.0f}s)", flush=True)

    prov = dict(script=os.path.basename(__file__), git_rev=git_rev(), host=socket.gethostname(),
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                device=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
                prereg=os.path.relpath(PREREG, ROOT), base_prereg=os.path.relpath(BASE_PREREG, ROOT),
                arms_h_bias=pre["arms_h_bias"], seeds=seeds, inits=pre["inits"], batch_seed=BATCH_SEED,
                files=[os.path.relpath(p, ROOT) for p in done], wall_seconds=time.time() - t_start)
    with open(os.path.join(a.out, "provenance.json"), "w") as fh:
        json.dump(prov, fh, indent=2, default=float)
    print(f"\ndone: {len(done)} arms in {time.time() - t_start:.0f}s -> {os.path.relpath(a.out, ROOT)}")


if __name__ == "__main__":
    main()
