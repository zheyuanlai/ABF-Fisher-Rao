#!/usr/bin/env python
"""WCA OT + repair campaign runner, stage M2: in-sampler arms on the accepted Case IX cell.

Arms (all plain-ABF backbone + Wasserstein reallocation on the FR schedule):
    ot_a{alpha}_c0     T0 -- OT lift, NO repair (the injected conditional error is deposited)
    ot_a{alpha}_c{c}   TR -- OT lift + projected constrained repair of c * tau_f(z') per moved walker
paired by seed against the targeted-relax campaign's W1 `abf` / `fr_uniform` and W1b `abf_ptarg1` /
`fr_ptarg1` runs (same seeds, same initial conditions and outer noise stream; OT consumes no RNG and
repair uses its own generator).  Prints wall time and safety / accounting counters only -- never a
free-energy error.  Preregistration: docs/WCA_OT_REPAIR_MECHANISM.md.

    CUDA_VISIBLE_DEVICES=3 python -u scripts/run_wca_ot_repair.py --seeds 820-823 --alpha 0.05 --c 0,1
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import wca_abffr_core as core      # noqa: E402
import wca_phase_jobs as jobs      # noqa: E402
from wca_ot_repair import OTConfig  # noqa: E402
from run_wca_targeted_relax import (CACHE, N_REPLICAS, READOUT, TAKEN, base_config, git_rev,  # noqa: E402
                                    load_tau_map, make_spec, parse_seeds)

CAMPAIGN = os.path.join(ROOT, "results", "ot_repair_campaign", "wca")
PREREG = os.path.join(ROOT, "docs", "WCA_OT_REPAIR_MECHANISM.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="M2")
    ap.add_argument("--seeds", default="820-823")
    ap.add_argument("--alpha", default="0.05", help="comma list of OT strengths")
    ap.add_argument("--c", default="0,1", help="comma list of repair multipliers c (0 = T0)")
    ap.add_argument("--dz-max", type=float, default=None, help="per-event cap in z units (default 2 grid bins)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    base = base_config()
    tau, tmap = load_tau_map()
    seeds = parse_seeds(a.seeds)
    assert not (set(seeds) & TAKEN), "seeds collide with an accepted study"
    n_grid = int(base["n_grid"])
    dz_grid = (float(base["z_max"]) - float(base["z_min"])) / max(n_grid - 1, 1)
    dz_max = float(a.dz_max) if a.dz_max is not None else 2.0 * dz_grid
    alphas = [float(x) for x in a.alpha.split(",")]
    cs = [float(x) for x in a.c.split(",")]
    arms = [(f"ot_a{al:g}_c{c:g}", OTConfig(alpha=al, dz_max=dz_max, c_repair=c, tau_grid=tuple(tau.tolist()), scheme="projected"))
            for al in alphas for c in cs]
    print(f"{a.stage}: {len(arms)} arms x {len(seeds)} seeds ({seeds[0]}-{seeds[-1]}); dz_max {dz_max:.4f} (= {dz_max / dz_grid:.1f} bins); "
          f"tau map {tmap['sha256'][:12]} (min {tau.min():.4f} max {tau.max():.4f}); arms {[n for n, _ in arms]}; no error metric printed", flush=True)
    if a.dry_run:
        return
    raw_dir = os.path.join(CAMPAIGN, a.stage, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    engines, done, t_start = {}, 0, time.time()
    total = len(seeds) * len(arms)
    log = []
    for sd in seeds:
        for name, ot in arms:
            sp = make_spec(a.stage, name, "abf", sd)
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
            info = (f" events={int(out['ot_n_opportunities'])} moved={float(out['ot_moved_frac']):.3f} |dz| mean={float(out['ot_absdz_mean']):.4f} "
                    f"max={float(out['ot_absdz_max']):.4f} capped={float(out['ot_capped_frac']):.3f} inner={int(out['relax_steps_total'])} "
                    f"({float(out['relax_cost_ratio']):.3f}x) innerwall={float(out['relax_inner_wall_seconds']):.0f}s")
            print(f"  [{done}/{total}] {name:>16s} seed{sd}: saved{info} ({time.time() - t0:.0f}s)", flush=True)
            log.append(dict(name=name, seed=sd, wall=time.time() - t0, events=int(out["ot_n_opportunities"]),
                            moved=float(out["ot_moved_frac"]), absdz_mean=float(out["ot_absdz_mean"]), absdz_max=float(out["ot_absdz_max"]),
                            capped=float(out["ot_capped_frac"]), inner=int(out["relax_steps_total"]), cost=float(out["relax_cost_ratio"])))
    prov = dict(script=os.path.basename(__file__), stage=a.stage, git_rev=git_rev(), host=socket.gethostname(),
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""), prereg=os.path.relpath(PREREG, ROOT),
                seeds=seeds, arms=[n for n, _ in arms], alpha=alphas, c=cs, dz_max=dz_max, tau_map_sha256=tmap["sha256"],
                wall_seconds=time.time() - t_start, n_runs=done, runs=log)
    os.makedirs(os.path.join(CAMPAIGN, a.stage), exist_ok=True)
    json.dump(prov, open(os.path.join(CAMPAIGN, a.stage, f"provenance_{seeds[0]}-{seeds[-1]}.json"), "w"), indent=2, default=float)
    print(f"\ndone {a.stage}: {done} runs in {(time.time() - t_start) / 3600:.2f} h")


if __name__ == "__main__":
    main()
