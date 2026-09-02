#!/usr/bin/env python
"""WCA Case IX corrected-baseline CONFIRMATION: abf vs fr_uniform, fresh paired seeds.

Preregistration: configs/information_campaign/wca_corrected_confirmation_prereg.json.
Everything is the accepted Case IX protocol (run_uniform_wca.py) at the legacy online
bandwidth 0.025 -- step 1 resolved no online arm -- plus the read-out bank (0.0125,
0.00625, raw) so both arms can be scored at the corrected read-out h_read* = 0.0125 by
analyze_wca_corrected_confirmation.py.  FR knobs inherited verbatim (h_bias unchanged,
so the earned rate stands).  Both arms of one seed run in the SAME process.

This runner prints NO error metric (prereg prohibition): wall time and the FR arm's
safety counters only.  The committed analyzer is the first thing to read e_F.

    CUDA_VISIBLE_DEVICES=3 python -u scripts/run_wca_corrected_confirmation.py   # GPU 3 ONLY
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

PREREG = os.path.join(ROOT, "configs/information_campaign/wca_corrected_confirmation_prereg.json")
PHASE_CONFIG = os.path.join(ROOT, "configs/wca_phase_diagram_production.yaml")
OUT_ROOT = os.path.join(ROOT, "results", "information_campaign", "wca_corrected_confirmation")
CACHE = os.path.join(ROOT, "cache", "phase_hp_v3")
REFERENCE_NPZ = os.path.join(CACHE, "wca_ti_b1_h2_w2_n10_a1.5_g160.npz")

CELL = dict(beta=1.0, h=2.0, w=2.0, n_dim=10, a=1.5, sigma=1.0, epsilon=1.0)
N_STEPS, N_REPLICAS, SAVE_EVERY = 120_000, 1024, 2500
ARMS = [("abf", "abf"), ("fr_uniform", "fr_uniform")]


def git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=None, help="inclusive range a-b or comma list; default: prereg")
    ap.add_argument("--out", default=OUT_ROOT)
    ap.add_argument("--stage", default="confirmation")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    pre = json.load(open(PREREG))
    cb = pre["corrected_baseline"]
    fr_knobs = {k: v for k, v in pre["fr_knobs"].items() if not k.startswith("_")}
    readout = tuple(float(h) for h in cb["readout_bank_bandwidths"])
    assert pre["n_steps"] == N_STEPS and pre["n_replicas"] == N_REPLICAS
    assert [n for n, _ in ARMS] == pre["arms"]

    cfg = jobs.load_yaml(PHASE_CONFIG)
    base = jobs.effective_base(cfg, "production")
    assert float(base["abf_bandwidth"]) == float(cb["h_bias"]) == 0.025, "legacy online bandwidth moved"
    assert float(base["abf_smooth_sigma"]) == float(cb["abf_smooth_sigma"]) == 0.5
    for k, v in CELL.items():
        assert float(cfg["system_defaults"][k]) == float(v), f"cell {k} moved in the YAML"
    for block in ("fr_uniform", "fr_estimated"):     # inherited, identical across the two blocks
        m = cfg["methods"][block]
        for k, v in fr_knobs.items():
            got = m.get(k, base.get(k))
            assert float(got) == float(v), f"FR knob {k} in {block}: config {got}, frozen {v}"
    assert os.path.exists(REFERENCE_NPZ), f"corrected TI reference missing: {REFERENCE_NPZ}"

    seeds_arg = a.seeds or f"{pre['seed_first']}-{pre['seed_first'] + pre['n_seeds'] - 1}"
    if "-" in seeds_arg:
        lo, hi = seeds_arg.split("-")
        seeds = list(range(int(lo), int(hi) + 1))
    else:
        seeds = [int(x) for x in seeds_arg.split(",")]
    assert not any(400 <= s <= 415 or 600 <= s <= 615 for s in seeds), "reused labels"

    if os.environ.get("CUDA_VISIBLE_DEVICES", "") == "" and not a.dry_run:
        print("WARNING: CUDA_VISIBLE_DEVICES is unset; pin exactly one GPU", flush=True)
    raw_dir = os.path.join(a.out, a.stage, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    def make_spec(name, method, seed):
        return jobs.PhaseRunSpec(stage=a.stage, name=name, method=method, seed=int(seed),
                                 n_steps=N_STEPS, n_replicas=N_REPLICAS, save_every=SAVE_EVERY,
                                 **CELL, **fr_knobs)

    print("WCA Case IX corrected-baseline CONFIRMATION: abf vs fr_uniform (fresh paired seeds)")
    print(f"  h_bias {base['abf_bandwidth']} (legacy), smooth_sigma {base['abf_smooth_sigma']}, read-out bank {readout}")
    print(f"  {N_STEPS} steps, N = {N_REPLICAS}, FR knobs {fr_knobs}")
    print(f"  seeds {seeds[0]}-{seeds[-1]} ({len(seeds)}); no error metric is printed here (prereg)")
    total = len(seeds) * len(ARMS)
    if a.dry_run:
        for sd in seeds:
            for name, m in ARMS:
                sp = make_spec(name, m, sd)
                print(f"  {'EXISTS' if jobs.run_is_valid(jobs.run_npz_path(raw_dir, sp)) else 'run   '}  {sp.run_id()}")
        return

    engines, done, t_start = {}, 0, time.time()
    for sd in seeds:
        for name, m in ARMS:
            sp = make_spec(name, m, sd)
            path = jobs.run_npz_path(raw_dir, sp)
            if not a.overwrite and jobs.run_is_valid(path):
                print(f"  skip {sp.run_id()}", flush=True)
                done += 1
                continue
            eng = jobs.get_engine(sp, engines)
            t0 = time.time()
            out = jobs.execute_run(sp, base, eng, cache_dir=CACHE, verbose=a.verbose,
                                   store_profiles=True, readout_bandwidths=readout)
            assert "v2" in str(out.get("reference_label", "")), "unexpected reference"
            assert float(out["abf_bandwidth_online"]) == 0.025
            jobs.save_run(path, out)
            done += 1
            safety = (f" repl={out['total_replacement_events']} essA={out['final_ancestor_ess']:.0f}"
                      if m != "abf" else "")
            print(f"  [{done}/{total}] {name:>11s} seed{sd}: saved{safety} ({time.time() - t0:.0f}s)", flush=True)

    prov = dict(script=os.path.basename(__file__), git_rev=git_rev(), host=socket.gethostname(),
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                prereg=os.path.relpath(PREREG, ROOT), config=os.path.relpath(PHASE_CONFIG, ROOT),
                cache_dir=os.path.relpath(CACHE, ROOT), cell=CELL, fr_knobs=fr_knobs,
                h_bias=float(base["abf_bandwidth"]), readout_bank=list(readout),
                n_steps=N_STEPS, n_replicas=N_REPLICAS, save_every=SAVE_EVERY, seeds=seeds,
                arms=[n for n, _ in ARMS], wall_seconds=time.time() - t_start, n_runs=done)
    with open(os.path.join(a.out, a.stage, f"provenance_{seeds[0]}-{seeds[-1]}.json"), "w") as fh:
        json.dump(prov, fh, indent=2, default=float)
    print(f"\ndone: {done} runs in {(time.time() - t_start) / 3600:.2f} h -> {raw_dir}")


if __name__ == "__main__":
    main()
