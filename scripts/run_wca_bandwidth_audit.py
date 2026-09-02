#!/usr/bin/env python
"""WCA Case IX corrected-baseline audit, step 1: ABF-only ONLINE bandwidth matrix.

Preregistration: configs/information_campaign/wca_baseline_audit_prereg.json.
Three ABF arms differing ONLY in the online bandwidth h_bias (legacy 0.025, 1/2,
1/4); every other value inherited verbatim from the accepted Case IX cell
(run_uniform_wca.py: corrected TI reference cache/phase_hp_v3, N=1024, 120k
steps, abf_smooth_sigma 0.5 held fixed).  Every run also carries a read-out bank
(report-only estimators at 0.0125 and 0.00625 plus raw binned sums) so the
read-out bandwidth is swept OFFLINE by analyze_wca_bandwidth_audit.py -- the
same instrument used on ZIF-8 and LTA.  No FR arm.

All three arms of one seed run in the SAME process (shared lattice init + noise
stream; process-level GPU nondeterminism is why cross-process pairing is not
claimed).

    CUDA_VISIBLE_DEVICES=3 python -u scripts/run_wca_bandwidth_audit.py --seeds 600-615   # GPU 3 ONLY
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import wca_phase_jobs as jobs      # noqa: E402

PREREG = os.path.join(ROOT, "configs/information_campaign/wca_baseline_audit_prereg.json")
PHASE_CONFIG = os.path.join(ROOT, "configs/wca_phase_diagram_production.yaml")
OUT_ROOT = os.path.join(ROOT, "results", "information_campaign", "wca_baseline_audit")
CACHE = os.path.join(ROOT, "cache", "phase_hp_v3")
REFERENCE_NPZ = os.path.join(CACHE, "wca_ti_b1_h2_w2_n10_a1.5_g160.npz")

# The accepted Case IX cell, frozen (asserted against the YAML, as in run_uniform_wca.py).
CELL = dict(beta=1.0, h=2.0, w=2.0, n_dim=10, a=1.5, sigma=1.0, epsilon=1.0)
FR_KNOBS = dict(fr_rate=0.10, target_ema_rate=0.005, max_event_fraction=0.02,
                fr_every=5, fr_start_steps=20000, score_clip=2.0)   # inert for method=abf
N_STEPS, N_REPLICAS, SAVE_EVERY = 120_000, 1024, 2500


def git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def make_spec(stage, name, seed, n_steps=N_STEPS, n_replicas=N_REPLICAS, save_every=SAVE_EVERY):
    return jobs.PhaseRunSpec(stage=stage, name=name, method="abf", seed=int(seed),
                             n_steps=n_steps, n_replicas=n_replicas, save_every=save_every,
                             **CELL, **FR_KNOBS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=None, help="inclusive range a-b or comma list; default: prereg")
    ap.add_argument("--out", default=OUT_ROOT)
    ap.add_argument("--stage", default="bandwidth_audit")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    pre = json.load(open(PREREG))
    arms = [(f"abf_hb{h:g}", float(h)) for h in pre["arms_h_bias"]]
    readout = tuple(float(h) for h in pre["readout_bank_bandwidths"])
    assert float(pre["arms_h_bias"][0]) == 0.025, "first arm must be the legacy bandwidth"
    assert pre["n_steps"] == N_STEPS and pre["n_replicas"] == N_REPLICAS

    cfg = jobs.load_yaml(PHASE_CONFIG)
    base = jobs.effective_base(cfg, "production")
    assert float(base["abf_bandwidth"]) == 0.025 and float(base["abf_smooth_sigma"]) == 0.5, \
        "YAML base block moved; the legacy arm would not be the legacy"
    for k, v in CELL.items():
        assert float(cfg["system_defaults"][k]) == float(v), f"cell {k} moved in the YAML"
    assert os.path.exists(REFERENCE_NPZ), f"corrected TI reference missing: {REFERENCE_NPZ}"

    seeds_arg = a.seeds or f"{pre['seed_first']}-{pre['seed_first'] + pre['n_seeds'] - 1}"
    if "-" in seeds_arg:
        lo, hi = seeds_arg.split("-")
        seeds = list(range(int(lo), int(hi) + 1))
    else:
        seeds = [int(x) for x in seeds_arg.split(",")]

    if os.environ.get("CUDA_VISIBLE_DEVICES", "") == "" and not a.dry_run:
        print("WARNING: CUDA_VISIBLE_DEVICES is unset; pin exactly one GPU", flush=True)
    raw_dir = os.path.join(a.out, a.stage, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    print("WCA Case IX corrected-baseline audit: ABF-only online bandwidth matrix")
    print(f"  arms {[n for n, _ in arms]}; read-out bank {readout}; smooth_sigma {base['abf_smooth_sigma']} fixed")
    print(f"  {N_STEPS} steps, N = {N_REPLICAS}, seeds {seeds[0]}-{seeds[-1]} ({len(seeds)})")
    total = len(seeds) * len(arms)
    if a.dry_run:
        for sd in seeds:
            for name, _ in arms:
                sp = make_spec(a.stage, name, sd)
                print(f"  {'EXISTS' if jobs.run_is_valid(jobs.run_npz_path(raw_dir, sp)) else 'run   '}  {sp.run_id()}")
        return

    engines, done, t_start = {}, 0, time.time()
    for sd in seeds:
        for name, hb in arms:
            sp = make_spec(a.stage, name, sd)
            path = jobs.run_npz_path(raw_dir, sp)
            if not a.overwrite and jobs.run_is_valid(path):
                print(f"  skip {sp.run_id()}", flush=True)
                done += 1
                continue
            base_arm = dict(base)
            base_arm["abf_bandwidth"] = hb          # the ONE thing that differs between arms
            eng = jobs.get_engine(sp, engines)
            t0 = time.time()
            out = jobs.execute_run(sp, base_arm, eng, cache_dir=CACHE, verbose=a.verbose,
                                   store_profiles=True, readout_bandwidths=readout)
            assert "v2" in str(out.get("reference_label", "")), "unexpected reference"
            assert float(out["abf_bandwidth_online"]) == hb
            jobs.save_run(path, out)
            done += 1
            print(f"  [{done}/{total}] {name:>14s} seed{sd}: L2(F)={out['l2_f']:.4f} "
                  f"intF={out['integrated_l2_f']:.3f} ({time.time() - t0:.0f}s)", flush=True)

    prov = dict(script=os.path.basename(__file__), git_rev=git_rev(), host=socket.gethostname(),
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                prereg=os.path.relpath(PREREG, ROOT), config=os.path.relpath(PHASE_CONFIG, ROOT),
                cache_dir=os.path.relpath(CACHE, ROOT), cell=CELL, arms=dict(arms),
                readout_bank=list(readout), n_steps=N_STEPS, n_replicas=N_REPLICAS,
                save_every=SAVE_EVERY, seeds=seeds, wall_seconds=time.time() - t_start, n_runs=done)
    with open(os.path.join(a.out, a.stage, f"provenance_{seeds[0]}-{seeds[-1]}.json"), "w") as fh:
        json.dump(prov, fh, indent=2, default=float)
    print(f"\ndone: {done} runs in {(time.time() - t_start) / 3600:.2f} h -> {raw_dir}")


if __name__ == "__main__":
    main()
