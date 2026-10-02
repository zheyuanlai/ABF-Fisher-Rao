#!/usr/bin/env python
"""WCA FR-start ladder under the histogram estimator (docs/WCA_FR_START.md).

    CUDA_VISIBLE_DEVICES=3 python -u scripts/run_wca_fr_start_ladder.py [--seeds 3300-3307] [--starts 0,2500,10000,20000]
    python scripts/run_wca_fr_start_ladder.py --analyze

Every arm of a seed runs in one process from the same lattice init and noise stream (the campaign's
convention).  Per run it saves the engine's own read-out series and scalars to results/wca_fr_start/raw/.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS); sys.path.insert(0, os.path.join(SCRIPTS, "..", "src"))
ROOT = os.path.join(SCRIPTS, "..")
OUT = os.path.join(ROOT, "results", "wca_fr_start")
KEYS = ("l2_f", "l2_fp", "integrated_l2_f", "total_replacement_events", "min_ancestor_ess_window", "min_ancestor_ess",
        "max_ancestor_frac_over_time", "n_round_trips", "n_barrier_crossings", "runtime_seconds", "fr_event_fraction",
        "bias_absmax", "bias_clip_fraction")


def parse_seeds(s):
    if "-" in s:
        a, b = s.split("-"); return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def simulate(a):
    import dataclasses
    import torch
    import wca_phase_jobs as jobs
    from run_histogram_abf_wca import load_frozen, make_spec, CACHE, SELECTED
    c, base, fr = load_frozen()
    n_bins = int(json.load(open(SELECTED))["wca"]["selected_n_bins"])
    if torch.cuda.is_available():
        assert torch.cuda.device_count() == 1, "pin exactly one GPU"
    starts = [int(x) for x in a.starts.split(",")]
    raw = os.path.join(OUT, "raw"); os.makedirs(raw, exist_ok=True)
    for seed in parse_seeds(a.seeds):
        engines = {}
        arms = [("hist_abf", "abf", None)] + [(f"hist_fr_s{s}", "fr_uniform", s) for s in starts]
        for name, method, start in arms:
            path = os.path.join(raw, f"{name}__seed{seed}.npz")
            if os.path.exists(path) and not a.overwrite:
                print(f"  skip {name} seed {seed}", flush=True); continue
            sp = make_spec("frstart", name, method, seed, c, fr, "histogram", n_bins)
            if start is not None:
                sp = dataclasses.replace(sp, fr_start_steps=int(start))
            eng = jobs.get_engine(sp, engines)
            t0 = time.time()
            r = jobs.execute_run(sp, base, eng, cache_dir=CACHE, verbose=False, store_profiles=True)
            assert "v2" in str(r.get("reference_label", "")) and r["abf_estimator"] == "histogram" and not r["had_nan"]
            np.savez_compressed(path, times=np.asarray(r["profile_times"], float), l2_f_t=np.asarray(r["l2_f_t"], float),
                                l2_fp_t=np.asarray(r["l2_fp_t"], float), seed=seed, name=name, method=method,
                                fr_start_steps=int(sp.fr_start_steps), spec_json=str(r["spec_json"]),
                                **{k: float(r[k]) for k in KEYS})
            print(f"  seed {seed} {name:16s} fr_start {sp.fr_start_steps:6d}: e_F(T) {float(r['l2_f']):.4f}  I_F {float(r['integrated_l2_f']):.2f}  "
                  f"repl {int(r['total_replacement_events'])}  ESSw/N {float(r['min_ancestor_ess_window']) / sp.n_replicas:.3f}  ({time.time() - t0:.0f}s)", flush=True)


def analyze():
    from analyze_fr_start_timing import paired_stats, tau_eps
    runs = {}
    for f in sorted(glob.glob(os.path.join(OUT, "raw", "*.npz"))):
        d = np.load(f, allow_pickle=True)
        runs.setdefault(str(d["name"]), {})[int(d["seed"])] = d
    if "hist_abf" not in runs:
        print("no abf runs"); return
    abf = runs["hist_abf"]; seeds = sorted(abf)
    t = np.asarray(abf[seeds[0]]["times"], float); T = float(t[-1])
    L = ["# WCA FR-start ladder (histogram estimator, 160 bins, frozen cell): FR vs the seed's ABF arm", "",
         f"Seeds {seeds[0]}-{seeds[-1]} (n = {len(seeds)}); own read-out; paired relative change (median, BCa 95 % CI, wins).", "",
         "| arm | FR start (step / t) | e_F(T) median ABF / FR | d e_F(T) | I_F median ABF / FR | d I_F | replacements | windowed ESS/N min | max lineage share | time to ABF final accuracy, FR / ABF |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    summary = {}
    for name in sorted(k for k in runs if k != "hist_abf"):
        common = [s for s in seeds if s in runs[name]]
        if not common:
            continue
        ea = np.array([float(abf[s]["l2_f"]) for s in common]); ef = np.array([float(runs[name][s]["l2_f"]) for s in common])
        ia = np.array([float(abf[s]["integrated_l2_f"]) for s in common]); i_f = np.array([float(runs[name][s]["integrated_l2_f"]) for s in common])
        sf, si = paired_stats(ea, ef), paired_stats(ia, i_f)
        E_a = np.stack([np.asarray(abf[s]["l2_f_t"], float) for s in common], 1); E_f = np.stack([np.asarray(runs[name][s]["l2_f_t"], float) for s in common], 1)
        eps = np.median(E_a[-1]); ta = tau_eps(t, E_a, eps, T); tf = tau_eps(t, E_f, eps, T)
        rep = np.median([float(runs[name][s]["total_replacement_events"]) for s in common])
        ess = np.median([float(runs[name][s]["min_ancestor_ess_window"]) for s in common]) / 1024.0
        share = np.median([float(runs[name][s]["max_ancestor_frac_over_time"]) for s in common])
        st = int(runs[name][common[0]]["fr_start_steps"])
        pct = lambda r: f"{100 * r['median']:+.1f} % [{100 * r['lo']:+.1f}, {100 * r['hi']:+.1f}] {int(round(r['win_rate'] * r['n']))}/{r['n']}"
        L.append(f"| {name} | {st} / t = {st * 0.002:g} | {np.median(ea):.4f} / {np.median(ef):.4f} | {pct(sf)} | {np.median(ia):.2f} / {np.median(i_f):.2f} | {pct(si)} | {rep:.0f} | {ess:.3f} | {share:.3f} | {np.nanmedian(tf):.0f} / {np.nanmedian(ta):.0f} |")
        summary[name] = dict(fr_start_steps=st, n=len(common), final=sf, integrated=si, replacements=rep, ess_window=ess, max_share=share,
                             tau_fr=float(np.nanmedian(tf)), tau_abf=float(np.nanmedian(ta)))
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "scoreboard.md"), "w").write("\n".join(L) + "\n")
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=2, default=float)
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="3300-3307"); ap.add_argument("--starts", default="0,2500,10000,20000")
    ap.add_argument("--overwrite", action="store_true"); ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    if a.analyze:
        analyze()
    else:
        simulate(a); analyze()


if __name__ == "__main__":
    main()
