#!/usr/bin/env python
"""Matched-sham control at the frozen WCA-positive cell.  No hyperparameter search.

The gateway result is sham-controlled; the molecular positive is not.  Until it is, the
claim "directional Fisher-Rao selection helps a many-body molecular system" rests on a
comparison that generic turnover of the same intensity was never tested against.  This run
supplies that control at the cell the study already accepted as positive, with nothing
retuned.

Everything physical and every FR knob is taken from the accepted phase-diagram production
config's ``b1_h2`` cell (beta=1, h=2, w=2, n_dim=10, a=1.5; fr_rate 0.10, target EMA 0.005,
event cap 0.02, fr_every 5, fr_start 20000, score clip 2.0; 120k steps, N=1024).  Only the
seeds are new.

Because the WCA sampler runs one method per process, a sham cannot watch its partner online.
It replays a per-opportunity count sequence recorded from the partner's own run on the same
seed, so the pass order is forced:

    pass 1   abf, fr_estimated, fr_oracle
    pass 2   sham_practical  (replays fr_estimated's schedule)
             sham_oracle     (replays fr_oracle's schedule)

Resumable: a valid npz is never recomputed, so an interrupted run continues where it left
off.

    CUDA_VISIBLE_DEVICES=2 python -u scripts/run_wca_sham.py
    CUDA_VISIBLE_DEVICES=2 python -u scripts/run_wca_sham.py --dry-run
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
import wca_abffr_core as core      # noqa: E402
import wca_phase_jobs as jobs      # noqa: E402

PHASE_CONFIG = os.path.join(ROOT, "configs/wca_phase_diagram_production.yaml")
OUT_ROOT = os.path.join(ROOT, "results", "wca_sham")
CACHE = os.path.join(ROOT, "cache", "phase")

# The accepted positive cell.  These are read back from the YAML and asserted, so a change
# to the config cannot silently move this experiment.
CELL = dict(beta=1.0, h=2.0, w=2.0, n_dim=10, a=1.5, sigma=1.0, epsilon=1.0)
FR_KNOBS = dict(fr_rate=0.10, target_ema_rate=0.005, max_event_fraction=0.02,
                fr_every=5, fr_start_steps=20000, score_clip=2.0)
N_STEPS, N_REPLICAS, SAVE_EVERY = 120_000, 1024, 2500

PASS1 = [("abf", "abf"), ("fr_estimated", "fr_estimated"), ("fr_oracle", "fr_oracle")]
PASS2 = [("sham_practical", "sham_practical"), ("sham_oracle", "sham_oracle")]


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
    ap.add_argument("--stage", default="sham")
    ap.add_argument("--seeds", default="400-415",
                    help="inclusive range or comma list; must be unused elsewhere")
    ap.add_argument("--out", default=OUT_ROOT)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    cfg = jobs.load_yaml(a.config)
    base = jobs.effective_base(cfg, "production")
    # Assert the frozen knobs still match the accepted config: this experiment must not
    # drift because someone edited the phase-diagram YAML.
    prod = cfg["methods"]["fr_estimated"]
    for k, v in FR_KNOBS.items():
        got = prod.get(k, base.get(k))
        assert float(got) == float(v), f"FR knob {k}: config has {got}, frozen value is {v}"

    if "-" in a.seeds:
        lo, hi = a.seeds.split("-")
        seeds = list(range(int(lo), int(hi) + 1))
    else:
        seeds = [int(x) for x in a.seeds.split(",")]

    if os.environ.get("CUDA_VISIBLE_DEVICES", "") == "" and not a.dry_run:
        print("WARNING: CUDA_VISIBLE_DEVICES is unset; pin exactly one GPU", flush=True)
    raw_dir = os.path.join(a.out, a.stage, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    print(f"WCA matched-sham control at the accepted positive cell")
    print(f"  cell b{CELL['beta']:g}_h{CELL['h']:g}_w{CELL['w']:g}_n{CELL['n_dim']}"
          f"_a{CELL['a']:g}   M = {CELL['n_dim'] ** 2}")
    print(f"  {N_STEPS} steps, N = {N_REPLICAS}, FR knobs {FR_KNOBS}")
    print(f"  seeds {seeds[0]}-{seeds[-1]} ({len(seeds)}), arms "
          f"{[n for n, _ in PASS1 + PASS2]}")
    total = len(seeds) * (len(PASS1) + len(PASS2))
    print(f"  {total} runs; at ~230 s each that is ~{total * 230 / 3600:.1f} GPU-hours\n")
    if a.dry_run:
        for sd in seeds:
            for name, m in PASS1 + PASS2:
                sp = make_spec(a.stage, name, m, sd)
                print(f"  {'EXISTS' if jobs.run_is_valid(jobs.run_npz_path(raw_dir, sp)) else 'run   '}"
                      f"  {sp.run_id()}")
        return

    engines = {}
    t_start = time.time()
    done = 0
    for sd in seeds:
        counts = {}
        for name, m in PASS1:
            sp = make_spec(a.stage, name, m, sd)
            path = jobs.run_npz_path(raw_dir, sp)
            if not a.overwrite and jobs.run_is_valid(path):
                with np.load(path, allow_pickle=True) as z:
                    if "fr_event_counts" in z.files:
                        counts[m] = np.asarray(z["fr_event_counts"])
                print(f"  skip {sp.run_id()}", flush=True)
                done += 1
                continue
            eng = jobs.get_engine(sp, engines)
            t0 = time.time()
            out = jobs.execute_run(sp, base, eng, cache_dir=CACHE, verbose=a.verbose)
            jobs.save_run(path, out)
            counts[m] = np.asarray(out["fr_event_counts"])
            done += 1
            print(f"  [{done}/{total}] {name:>14s} seed{sd}: L2(F)={out['l2_f']:.4f} "
                  f"intF={out['integrated_l2_f']:.3f} repl={out['total_replacement_events']} "
                  f"essA={out['final_ancestor_ess']:.0f} ({time.time() - t0:.0f}s)",
                  flush=True)

        for name, m in PASS2:
            partner = core.SHAM_PARTNER[m]
            sp = make_spec(a.stage, name, m, sd)
            path = jobs.run_npz_path(raw_dir, sp)
            if not a.overwrite and jobs.run_is_valid(path):
                print(f"  skip {sp.run_id()}", flush=True)
                done += 1
                continue
            if partner not in counts:
                # Its partner's schedule is the whole point; running without it would
                # produce an arm that is not intensity-matched and looks identical in the
                # artifact.
                raise SystemExit(
                    f"cannot run {m} for seed {sd}: {partner} produced no fr_event_counts. "
                    f"Re-run pass 1 for this seed with --overwrite.")
            eng = jobs.get_engine(sp, engines)
            t0 = time.time()
            out = jobs.execute_run(sp, base, eng, cache_dir=CACHE, verbose=a.verbose,
                                   replay_counts=counts[partner])
            assert int(out["total_replacement_events"]) == int(counts[partner].sum()), (
                "sham did not reproduce its partner's total replacement count")
            jobs.save_run(path, out)
            done += 1
            print(f"  [{done}/{total}] {name:>14s} seed{sd}: L2(F)={out['l2_f']:.4f} "
                  f"intF={out['integrated_l2_f']:.3f} repl={out['total_replacement_events']} "
                  f"(matched to {partner}) essA={out['final_ancestor_ess']:.0f} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    prov = dict(script=os.path.basename(__file__), git_rev=git_rev(),
                host=socket.gethostname(),
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                config=os.path.relpath(a.config, ROOT), stage=a.stage,
                cell=CELL, fr_knobs=FR_KNOBS, n_steps=N_STEPS, n_replicas=N_REPLICAS,
                save_every=SAVE_EVERY, seeds=seeds,
                arms=[n for n, _ in PASS1 + PASS2],
                sham_partners={m: core.SHAM_PARTNER[m] for _, m in PASS2},
                wall_seconds=time.time() - t_start, n_runs=done)
    with open(os.path.join(a.out, a.stage, "provenance.json"), "w") as fh:
        json.dump(prov, fh, indent=2, default=float)
    print(f"\ndone: {done} runs in {(time.time() - t_start) / 3600:.2f} h -> {raw_dir}")


if __name__ == "__main__":
    main()
