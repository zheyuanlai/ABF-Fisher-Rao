#!/usr/bin/env python
"""WCA capped-OT confirmatory campaign M3 runner (docs/WCA_OT_CONFIRMATORY_M3.md).

    calibration  seeds 880-883: A, F, T(alpha) for alpha in {0.03, 0.05, 0.10, 0.20}   (M3-A, blind)
    core         seeds 900-915: A, F, T(alpha*)                                        (M3-B)
    repair       seeds 900-915: R, F+R, T(alpha*)+R  (5 projected inner steps, every walker, every event)

alpha* is read from <campaign>/M3/calibration/alpha_star.json, written by
scripts/analyze_wca_ot_m3.py --stage calibration from the marginal action only.  Arms of one seed
run in one process (same initial conditions and outer noise stream).  Prints wall time and
safety / accounting counters only -- never a free-energy error.

    CUDA_VISIBLE_DEVICES=1 python -u scripts/run_wca_ot_m3.py --stage calibration
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import socket
import sys
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import wca_phase_jobs as jobs      # noqa: E402
from wca_ot_repair import OTConfig  # noqa: E402
from run_wca_targeted_relax import (CACHE, READOUT, TAKEN, base_config, git_rev,  # noqa: E402
                                    load_tau_map, make_spec, parse_seeds)

CAMPAIGN = os.path.join(ROOT, "results", "ot_repair_campaign", "wca", "M3")
PREREG = os.path.join(ROOT, "docs", "WCA_OT_CONFIRMATORY_M3.md")
ALPHAS_CAL = (0.03, 0.05, 0.10, 0.20)
SEEDS = dict(calibration="880-883", core="900-915", repair="900-915")
C_REPAIR = 0.5                      # 5 inner steps with the frozen 10-dt tau map
USED_ELSEWHERE = set(range(800, 808)) | set(range(820, 824))


def campaign_seeds_on_disk():
    seen = set()
    for f in glob.glob(os.path.join(ROOT, "results", "ot_repair_campaign", "wca", "*", "raw", "*.npz")) + \
             glob.glob(os.path.join(ROOT, "results", "ot_repair_campaign", "wca", "M3", "*", "raw", "*.npz")):
        for tok in os.path.basename(f).split("__"):
            if tok.startswith("seed"):
                seen.add(int(tok[4:]))
    return seen


def arms_for(stage, tau, dz_max, alpha_star=None):
    tg = tuple(tau.tolist())
    T = lambda al, c=0.0, rep=False: OTConfig(alpha=al, dz_max=dz_max, c_repair=c, tau_grid=tg, scheme="projected", repair_all=rep)  # noqa: E731
    if stage == "calibration":
        return [("abf", "abf", None), ("fr_uniform", "fr_uniform", None)] + [(f"ot_a{al:g}", "abf", T(al)) for al in ALPHAS_CAL]
    assert alpha_star is not None, "alpha* is required for core/repair"
    if stage == "core":
        return [("abf", "abf", None), ("fr_uniform", "fr_uniform", None), (f"ot_a{alpha_star:g}", "abf", T(alpha_star))]
    if stage == "repair":
        return [("abf_rej", "abf", T(0.0, C_REPAIR, True)), ("fr_rej", "fr_uniform", T(0.0, C_REPAIR, True)),
                (f"ot_a{alpha_star:g}_rej", "abf", T(alpha_star, C_REPAIR, True))]
    raise ValueError(stage)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("calibration", "core", "repair"))
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--alpha-star", type=float, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--gpu-override", action="store_true", help="bypass the GPU-1-only check (do not use)")
    a = ap.parse_args()
    if not a.gpu_override and os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() != "1":
        sys.exit("this round may use GPU 1 only: set CUDA_VISIBLE_DEVICES=1")

    base = base_config()
    tau, tmap = load_tau_map()
    n_grid = int(base["n_grid"])
    dz_max = 2.0 * (float(base["z_max"]) - float(base["z_min"])) / max(n_grid - 1, 1)
    seeds = parse_seeds(a.seeds or SEEDS[a.stage])
    bad = set(seeds) & (TAKEN | USED_ELSEWHERE)
    assert not bad, f"seeds {sorted(bad)} collide with an accepted study"
    if a.stage == "core":
        onq = campaign_seeds_on_disk() - set(parse_seeds(SEEDS["core"]))
        assert not (set(seeds) & onq), "core seeds must be fresh to this campaign"
    alpha_star = a.alpha_star
    if a.stage in ("core", "repair") and alpha_star is None:
        js = os.path.join(CAMPAIGN, "calibration", "alpha_star.json")
        assert os.path.exists(js), f"{js} missing: run the calibration analyzer first"
        d = json.load(open(js)); alpha_star = float(d["alpha_star"])
        print(f"alpha* = {alpha_star:g} from {os.path.relpath(js, ROOT)} (ratio {d.get('ratio'):.3f}, capped {d.get('capped_frac'):.3f}, blind={d.get('blind')})")
    arms = arms_for(a.stage, tau, dz_max, alpha_star)
    print(f"M3-{a.stage}: {len(arms)} arms x {len(seeds)} seeds ({seeds[0]}-{seeds[-1]}); cap {dz_max:.4f}; tau map {tmap['sha256'][:12]}; "
          f"arms {[n for n, _, _ in arms]}; GPU {os.environ.get('CUDA_VISIBLE_DEVICES')}; no error metric printed", flush=True)
    if a.dry_run:
        return
    raw_dir = os.path.join(CAMPAIGN, a.stage, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    engines, done, t_start, log = {}, 0, time.time(), []
    total = len(seeds) * len(arms)
    for sd in seeds:
        for name, method, ot in arms:
            sp = make_spec(f"M3_{a.stage}", name, method, sd)
            path = jobs.run_npz_path(raw_dir, sp)
            if not a.overwrite and jobs.run_is_valid(path):
                print(f"  skip {sp.run_id()}", flush=True); done += 1; continue
            eng = jobs.get_engine(sp, engines)
            t0 = time.time()
            out = jobs.execute_run(sp, base, eng, cache_dir=CACHE, verbose=a.verbose, store_profiles=True,
                                   readout_bandwidths=READOUT, ot=ot)
            assert "v2" in str(out.get("reference_label", "")), "unexpected reference"
            assert not bool(out["had_nan"]), f"NaN in {sp.run_id()} -> UNSAFE"
            jobs.save_run(path, out)
            done += 1
            info = ""
            if method != "abf":
                info += f" repl={int(out['total_replacement_events'])} essW={float(out['min_ancestor_ess_window']) / int(out['n_replicas']):.3f}"
            if ot is not None:
                info += (f" |dz| mean={float(out['ot_absdz_mean']):.4f} max={float(out['ot_absdz_max']):.4f} capped={float(out['ot_capped_frac']):.3f}"
                         f" inner={int(out['relax_steps_total'])} ({float(out['relax_cost_ratio']):.3f}x)")
            print(f"  [{done}/{total}] {name:>14s} seed{sd}: saved{info} ({time.time() - t0:.0f}s)", flush=True)
            log.append(dict(name=name, seed=sd, wall=time.time() - t0))
    prov = dict(script=os.path.basename(__file__), stage=a.stage, git_rev=git_rev(), host=socket.gethostname(),
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""), prereg=os.path.relpath(PREREG, ROOT),
                seeds=seeds, arms=[n for n, _, _ in arms], alpha_star=alpha_star, dz_max=dz_max, c_repair=C_REPAIR,
                tau_map_sha256=tmap["sha256"], wall_seconds=time.time() - t_start, n_runs=done, runs=log)
    json.dump(prov, open(os.path.join(CAMPAIGN, a.stage, f"provenance_{seeds[0]}-{seeds[-1]}.json"), "w"), indent=2, default=float)
    print(f"\ndone M3-{a.stage}: {done} runs in {(time.time() - t_start) / 3600:.2f} h")


if __name__ == "__main__":
    main()
