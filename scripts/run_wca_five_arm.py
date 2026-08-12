"""Q1 — is the Fisher-Rao direction better than the directed selection ABF already had?

    python scripts/run_wca_five_arm.py --stage calibrate      # pick c for the prior-art arms
    python scripts/run_wca_five_arm.py --stage confirm        # the five-arm comparison

The question v1 never asked. v1 measured mFR against ABF and against matched **random**
turnover; it never measured it against a **directed** alternative, and Chapter 6 of
Lelievre-Rousset-Stoltz applies selection to this very WCA dimer.

Five arms, matched population, compute, seeds and noise stream:

    abf                baseline
    fr_estimated       the proposed method, frozen exactly as Case IX ran it
    sham_practical     matched-turnover random-direction control
    book_laplacian     S = c * d2p/dz2 / p            (Ch. 6)
    count_balancing    S = c * (1 - p/p_bar)          (Remark 6.10 / NAMD)

**Turnover matching is the point of the calibration stage.** An arm must not win by selecting
harder, so `c` for each prior-art rule is chosen on **held-out** seeds (500-503, disjoint from
the 400-415 confirmatory block) to bring its replacement count closest to `fr_estimated`'s on
the same cell. The mFR configuration is **not** retuned -- §3 of the preregistration forbids it,
and retuning the proposed method against its own baselines would be the obvious way to
manufacture a win.

Amendment 8 applies to all three selection rules equally: each depends on the CV alone, so
`d/dt p(y|xi)|_sel = 0` and none can repair a conditional-equilibration failure. WCA passed
Gate 0, so that shared limitation is inactive here and this is a clean test of how the rules
redistribute an already-valid conditional population.

Scored against `cache/phase_hp_v3` -- the corrected reference. Running this on the cached one
could have reordered the arms, since its error peaks at `z ~ 0.26`, inside the transition region
where the selection rules differ most.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import wca_abffr_core as core      # noqa: E402
import wca_phase_jobs as jobs      # noqa: E402

PHASE_CONFIG = os.path.join(ROOT, "configs/wca_phase_diagram_production.yaml")
CELL = dict(beta=1.0, h=2.0, w=2.0, n_dim=10, a=1.5, sigma=1.0, epsilon=1.0)
FR_KNOBS = dict(fr_rate=0.10, target_ema_rate=0.005, max_event_fraction=0.02,
                fr_every=5, fr_start_steps=20000, score_clip=2.0)
import os as _os
N_STEPS = int(_os.environ.get('FIVEARM_STEPS', 120_000))
N_REPLICAS = int(_os.environ.get('FIVEARM_N', 1024))
SAVE_EVERY = 2500
PRIOR = ("book_laplacian", "count_balancing")
C_LADDER = (0.1, 0.3, 1.0, 3.0)
CAL_SEEDS = tuple(int(x) for x in _os.environ.get('FIVEARM_CAL','500,501').split(','))                      # held out from the 400-415 confirmatory block
CONFIRM_SEEDS = tuple(range(400, 416))


def spec_for(method, seed, name, stage, n_steps, prior_c=1.0):
    return jobs.PhaseRunSpec(stage=stage, name=name, method=method, seed=int(seed),
                             n_steps=int(n_steps), n_replicas=N_REPLICAS,
                             save_every=SAVE_EVERY, prior_c=float(prior_c), **CELL,
                             **FR_KNOBS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("calibrate", "confirm"), required=True)
    ap.add_argument("--out", default="results/wca_five_arm")
    ap.add_argument("--cache-dir", default="cache/phase_hp_v3")
    ap.add_argument("--cal-steps", type=int, default=40_000,
                    help="short runs: calibration only needs the replacement RATE")
    ap.add_argument("--seeds", default=None, help="override, e.g. 400-403")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    base = jobs.effective_base(jobs.load_yaml(PHASE_CONFIG), "production")
    os.makedirs(args.out, exist_ok=True)
    raw_dir = os.path.join(args.out, args.stage, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    engines = {}

    def run(sp):
        path = jobs.run_npz_path(raw_dir, sp)
        if jobs.run_is_valid(path):
            return jobs.load_run(path)
        eng = jobs.get_engine(sp, engines)
        out = jobs.execute_run(sp, base, eng, cache_dir=args.cache_dir,
                               verbose=args.verbose, store_profiles=True)
        jobs.save_run(path, out)
        return out

    if args.stage == "calibrate":
        print(f"CALIBRATION: matching prior-art turnover to fr_estimated")
        print(f"  held-out seeds {CAL_SEEDS}, c ladder {C_LADDER}, {args.cal_steps} steps\n",
              flush=True)
        target = []
        for s in CAL_SEEDS:
            o = run(spec_for("fr_estimated", s, "cal_fr", "calibrate", args.cal_steps))
            target.append(float(o["total_replacement_events"]))
            print(f"  fr_estimated seed{s}: {target[-1]:.0f} replacements", flush=True)
        tgt = float(np.median(target))
        print(f"\n  target turnover = {tgt:.0f}\n", flush=True)

        chosen = {}
        table = {}
        for m in PRIOR:
            rows = []
            for c in C_LADDER:
                reps = []
                for s in CAL_SEEDS:
                    o = run(spec_for(m, s, f"cal_{m}_c{c}", "calibrate", args.cal_steps,
                                     prior_c=c))
                    reps.append(float(o["total_replacement_events"]))
                r = float(np.median(reps))
                active = r >= 0.5 * N_REPLICAS          # §3.2 activity requirement
                rows.append(dict(c=c, replacements=r, active=active,
                                 rel_to_target=r / max(tgt, 1.0)))
                print(f"  {m:16s} c={c:5.2f}: {r:8.0f} replacements  "
                      f"({r/max(tgt,1.0):5.2f}x target)  active={active}", flush=True)
            ok = [r for r in rows if r["active"]]
            pick = min(ok or rows, key=lambda r: abs(np.log(max(r["replacements"], 1) / max(tgt, 1))))
            chosen[m] = pick["c"]
            table[m] = rows
            print(f"  -> {m}: c = {pick['c']}  "
                  f"({pick['replacements']:.0f} vs target {tgt:.0f})\n", flush=True)

        with open(os.path.join(args.out, "calibration.json"), "w") as fh:
            json.dump(dict(target_turnover=tgt, cal_seeds=list(CAL_SEEDS),
                           c_ladder=list(C_LADDER), chosen=chosen, table=table,
                           activity_rule="median replacements >= 0.5 N",
                           note="mFR is NOT retuned; only the prior-art intensities are chosen"),
                      fh, indent=2)
        print(f"wrote {args.out}/calibration.json   chosen: {chosen}")
        return

    # ---------------- confirmatory ----------------
    cal_path = os.path.join(args.out, "calibration.json")
    if not os.path.exists(cal_path):
        raise SystemExit("run --stage calibrate first")
    chosen = json.load(open(cal_path))["chosen"]
    seeds = CONFIRM_SEEDS
    if args.seeds:
        lo, hi = args.seeds.split("-")
        seeds = tuple(range(int(lo), int(hi) + 1))
    print(f"CONFIRMATORY five-arm: seeds {seeds[0]}-{seeds[-1]}, c = {chosen}", flush=True)

    t0 = time.perf_counter()
    order = [("abf", 1.0), ("fr_estimated", 1.0)] + [(m, chosen[m]) for m in PRIOR]
    for s in seeds:
        partner = None
        for m, c in order:
            o = run(spec_for(m, s, f"five_{m}", "confirm", N_STEPS, prior_c=c))
            if m == "fr_estimated":
                partner = o
            print(f"  seed{s} {m:16s} intF={float(o['integrated_l2_f']):8.3f} "
                  f"repl={int(o['total_replacement_events']):6d} "
                  f"({(time.perf_counter()-t0)/60:.1f} min)", flush=True)
        # sham replays fr_estimated's own schedule on the same seed
        sp = spec_for("sham_practical", s, "five_sham_practical", "confirm", N_STEPS)
        path = jobs.run_npz_path(raw_dir, sp)
        if not jobs.run_is_valid(path):
            eng = jobs.get_engine(sp, engines)
            out = jobs.execute_run(sp, base, eng, cache_dir=args.cache_dir,
                                   verbose=args.verbose, store_profiles=True,
                                   replay_counts=partner["fr_event_counts"])
            jobs.save_run(path, out)
        else:
            out = jobs.load_run(path)
        print(f"  seed{s} {'sham_practical':16s} intF={float(out['integrated_l2_f']):8.3f} "
              f"repl={int(out['total_replacement_events']):6d}", flush=True)
    print(f"\nDONE in {(time.perf_counter()-t0)/60:.1f} min -> {raw_dir}")


if __name__ == "__main__":
    main()
