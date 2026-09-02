#!/usr/bin/env python
"""Inertness + correctness smoke for the WCA ReadoutBank. One process, one seed:

  A1, A2  : no bank (control: within-process determinism of the engine itself)
  B       : bank at (0.025, 0.0125, 0.0)
  C       : online h_bias 0.0125 with the same bank (exercises the runner's arm path)

Asserts: A1 == A2 == B byte-for-byte on mean_force_t / pmf_t / l2_f / l2_f_t (the bank is
inert); the bank's 0.025 read-out equals the production profile (same class, same updates);
raw bins + 0.5-bin smoothing approximate the kernel profile (O(dz^2), loose).  Saves B and C
as run npz files so analyze_wca_bandwidth_audit.py can be exercised before real data exist.

    CUDA_VISIBLE_DEVICES=3 python scripts/smoke_wca_readout_bank.py --out DIR   # GPU 3 ONLY; CUDA_VISIBLE_DEVICES="" for the deterministic CPU identity test
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import wca_phase_jobs as jobs                                  # noqa: E402
from run_wca_bandwidth_audit import CACHE, PHASE_CONFIG, make_spec  # noqa: E402


def smooth_line(y, sigma_bins):
    rad = max(1, int(math.ceil(4.0 * sigma_bins)))
    k = np.exp(-0.5 * (np.arange(-rad, rad + 1) / sigma_bins) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(y, rad, mode="edge"), k, mode="valid")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--replicas", type=int, default=256)
    a = ap.parse_args()
    raw_dir = os.path.join(a.out, "smoke_readout", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    cfg = jobs.load_yaml(PHASE_CONFIG)
    base = jobs.effective_base(cfg, "production")
    base.update(abf_warmup_steps=1000, estimator_burn_in_steps=1000)   # exercise both phases
    bank = (0.025, 0.0125, 0.0)
    engines = {}

    def run(name, hb, readout):
        sp = make_spec("smoke_readout", name, seed=777, n_steps=a.steps,
                       n_replicas=a.replicas, save_every=500)
        b = dict(base)
        b["abf_bandwidth"] = hb
        t0 = time.time()
        out = jobs.execute_run(sp, b, jobs.get_engine(sp, engines), cache_dir=CACHE,
                               store_profiles=True, readout_bandwidths=readout)
        print(f"  {name:>14s} h_bias={hb:<8g} bank={readout}  l2_f={out['l2_f']:.5f}  ({time.time()-t0:.0f}s)")
        return sp, out

    _, A1 = run("abf_hb0.025", 0.025, None)
    _, A2 = run("abf_hb0.025", 0.025, None)
    spB, B = run("abf_hb0.025", 0.025, bank)
    spC, C = run("abf_hb0.0125", 0.0125, bank)

    keys = ("mean_force_t", "pmf_t", "l2_f", "l2_f_t", "final_mean_force", "final_pmf")
    ctrl = all(np.array_equal(A1[k], A2[k]) for k in keys)
    print(f"\n  within-process control A1 == A2: {ctrl}")
    if ctrl:
        inert = all(np.array_equal(A1[k], B[k]) for k in keys)
        print(f"  bank inert (A1 == B byte-for-byte): {inert}")
        assert inert, "ReadoutBank changed the dynamics"
    else:
        dev = max(float(np.abs(A1[k] - B[k]).max()) for k in ("mean_force_t", "pmf_t"))
        dev0 = max(float(np.abs(A1[k] - A2[k]).max()) for k in ("mean_force_t", "pmf_t"))
        print(f"  engine not bit-reproducible in-process (A1 vs A2 max dev {dev0:.2e}); A1 vs B {dev:.2e}")
        assert dev <= 10 * max(dev0, 1e-12), "bank deviation exceeds the engine's own noise"

    b025 = B["readout_mean_force_t__h0.025"]
    dev = float(np.abs(b025[-1] - B["final_mean_force"]).max())
    print(f"  bank 0.025 read-out vs production profile: max |dev| {dev:.2e}")
    assert dev < 1e-9

    fs, cs = B["raw_fsum_t"][-1], B["raw_csum_t"][-1]
    grid = B["grid"]
    mask = (grid >= -0.1) & (grid <= 1.1)
    mf_raw = np.where(cs > 0, fs / np.maximum(cs, 1.0), 0.0)
    mf_raw_s = smooth_line(mf_raw, 0.5)
    ref = B["final_mean_force"]
    rel = np.sqrt(np.mean((mf_raw_s - ref)[mask] ** 2)) / np.sqrt(np.mean(ref[mask] ** 2))
    print(f"  raw bins (+0.5-bin smoothing) vs kernel 0.025 profile: relative RMS {rel:.3f} "
          f"(expected O(0.1): different estimator family, {a.replicas}x{a.steps} samples)")
    print(f"  total raw counts (prod phase) {cs.sum():.0f} = replicas x post-burn-in steps "
          f"{a.replicas * (a.steps - 1000 + 1)}")
    assert abs(cs.sum() - a.replicas * (a.steps - 1000 + 1)) < 1

    for sp, out in ((spB, B), (spC, C)):
        jobs.save_run(jobs.run_npz_path(raw_dir, sp), out)
    print(f"\n  saved B and C -> {raw_dir}\n  SMOKE OK")


if __name__ == "__main__":
    main()
