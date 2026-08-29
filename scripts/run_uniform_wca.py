#!/usr/bin/env python
"""Uniform-FR campaign, Stage 2: WCA dimer at the corrected-reference Case IX cell.

Two arms only -- abf and fr_uniform -- with every physical value and every FR knob
inherited verbatim from the accepted Case IX v2 protocol (run_wca_sham.py at the
b1_h2 cell, corrected TI reference cache/phase_hp_v3, --store-profiles).  The
uniform target and its YAML method block already exist in the accepted config;
nothing is tuned here.  Preregistration: docs/UNIFORM_FR_CAMPAIGN.md.

Both arms of one seed run fresh in the SAME process (pairing = shared initial
conditions; process-level GPU nondeterminism is the reason the old abf npz files
are not reused -- see docs/V2_PREREGISTRATION.md on cross-process levels).

    CUDA_VISIBLE_DEVICES=3 python -u scripts/run_uniform_wca.py --seeds 400-403
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

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import wca_phase_jobs as jobs      # noqa: E402

PHASE_CONFIG = os.path.join(ROOT, "configs/wca_phase_diagram_production.yaml")
OUT_ROOT = os.path.join(ROOT, "results", "uniform_campaign", "wca")
CACHE = os.path.join(ROOT, "cache", "phase_hp_v3")
REFERENCE_NPZ = os.path.join(CACHE, "wca_ti_b1_h2_w2_n10_a1.5_g160.npz")

# The accepted Case IX cell, frozen.  Read back from the YAML and asserted, so a config
# edit cannot silently move the experiment.
CELL = dict(beta=1.0, h=2.0, w=2.0, n_dim=10, a=1.5, sigma=1.0, epsilon=1.0)
FR_KNOBS = dict(fr_rate=0.10, target_ema_rate=0.005, max_event_fraction=0.02,
                fr_every=5, fr_start_steps=20000, score_clip=2.0)
N_STEPS, N_REPLICAS, SAVE_EVERY = 120_000, 1024, 2500

ARMS = [("abf", "abf"), ("fr_uniform", "fr_uniform")]


def git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:
        return "unknown"


def make_spec(stage, name, method, seed):
    return jobs.PhaseRunSpec(stage=stage, name=name, method=method, seed=int(seed),
                             n_steps=N_STEPS, n_replicas=N_REPLICAS,
                             save_every=SAVE_EVERY, **CELL, **FR_KNOBS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=PHASE_CONFIG)
    ap.add_argument("--stage", default="uniform")
    ap.add_argument("--seeds", default="400-415",
                    help="inclusive range or comma list (Case IX seeds)")
    ap.add_argument("--out", default=OUT_ROOT)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--cache-dir", default=CACHE)
    a = ap.parse_args()

    cfg = jobs.load_yaml(a.config)
    base = jobs.effective_base(cfg, "production")
    # The uniform arm's knobs must match BOTH the YAML's fr_uniform block and the frozen
    # Case IX fr_estimated block -- inherited, not tuned, and identical across the two.
    for block in ("fr_uniform", "fr_estimated"):
        m = cfg["methods"][block]
        for k, v in FR_KNOBS.items():
            got = m.get(k, base.get(k))
            assert float(got) == float(v), \
                f"FR knob {k} in {block}: config has {got}, frozen value is {v}"

    # The corrected reference must already EXIST: load_or_compute_ti_reference silently
    # computes a missing file, which would mix reference provenance mid-campaign.
    assert os.path.exists(REFERENCE_NPZ), \
        f"corrected TI reference missing: {REFERENCE_NPZ}"

    if "-" in a.seeds:
        lo, hi = a.seeds.split("-")
        seeds = list(range(int(lo), int(hi) + 1))
    else:
        seeds = [int(x) for x in a.seeds.split(",")]

    if os.environ.get("CUDA_VISIBLE_DEVICES", "") == "" and not a.dry_run:
        print("WARNING: CUDA_VISIBLE_DEVICES is unset; pin exactly one GPU", flush=True)
    raw_dir = os.path.join(a.out, a.stage, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    print("UNIFORM-FR campaign, Stage 2 (WCA Case IX cell, corrected reference)")
    print(f"  cell b{CELL['beta']:g}_h{CELL['h']:g}_w{CELL['w']:g}_n{CELL['n_dim']}"
          f"_a{CELL['a']:g}   M = {CELL['n_dim'] ** 2}")
    print(f"  {N_STEPS} steps, N = {N_REPLICAS}, FR knobs {FR_KNOBS}")
    print(f"  seeds {seeds[0]}-{seeds[-1]} ({len(seeds)}), arms {[n for n, _ in ARMS]}")
    total = len(seeds) * len(ARMS)
    print(f"  {total} runs; at ~230-800 s each\n")
    if a.dry_run:
        for sd in seeds:
            for name, m in ARMS:
                sp = make_spec(a.stage, name, m, sd)
                print(f"  {'EXISTS' if jobs.run_is_valid(jobs.run_npz_path(raw_dir, sp)) else 'run   '}"
                      f"  {sp.run_id()}")
        return

    engines = {}
    t_start = time.time()
    done = 0
    for sd in seeds:
        for name, m in ARMS:
            sp = make_spec(a.stage, name, m, sd)
            path = jobs.run_npz_path(raw_dir, sp)
            if not a.overwrite and jobs.run_is_valid(path):
                print(f"  skip {sp.run_id()}", flush=True)
                done += 1
                continue
            eng = jobs.get_engine(sp, engines)
            t0 = time.time()
            out = jobs.execute_run(sp, base, eng, cache_dir=a.cache_dir, verbose=a.verbose,
                                   store_profiles=True)
            ref_label = str(out.get("reference_label", ""))
            assert "v2" in ref_label, \
                f"run scored against an unexpected reference: {ref_label!r}"
            jobs.save_run(path, out)
            done += 1
            print(f"  [{done}/{total}] {name:>12s} seed{sd}: L2(F)={out['l2_f']:.4f} "
                  f"intF={out['integrated_l2_f']:.3f} repl={out['total_replacement_events']} "
                  f"essA={out['final_ancestor_ess']:.0f} ({time.time() - t0:.0f}s)",
                  flush=True)

    prov = dict(script=os.path.basename(__file__), git_rev=git_rev(),
                host=socket.gethostname(),
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                config=os.path.relpath(a.config, ROOT), stage=a.stage,
                cache_dir=os.path.relpath(a.cache_dir, ROOT),
                cell=CELL, fr_knobs=FR_KNOBS, n_steps=N_STEPS, n_replicas=N_REPLICAS,
                save_every=SAVE_EVERY, seeds=seeds, arms=[n for n, _ in ARMS],
                wall_seconds=time.time() - t_start, n_runs=done)
    # one provenance file per seed block, so concurrent shards cannot overwrite each other
    tag = f"{seeds[0]}-{seeds[-1]}"
    with open(os.path.join(a.out, a.stage, f"provenance_{tag}.json"), "w") as fh:
        json.dump(prov, fh, indent=2, default=float)
    print(f"\ndone: {done} runs in {(time.time() - t_start) / 3600:.2f} h -> {raw_dir}")


if __name__ == "__main__":
    main()
