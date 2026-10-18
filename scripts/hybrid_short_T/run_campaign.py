#!/usr/bin/env python
"""Stage III: production (exploration seeds) and the fresh-seed confirmation of the hybrid short-T study.

    python scripts/hybrid_short_T/run_campaign.py prod [--systems ...] [--max-seeds 2] [--dry-run] [--max-ccds 12]
    python scripts/hybrid_short_T/run_campaign.py confirm [--dry-run]       # needs analysis/confirm_selection.json
    python scripts/hybrid_short_T/run_campaign.py status                    # complete / pending counts

Production (prereg section 5): per system every N of the ladder, ABF and (N >= 2) FR, the prod_seeds of
configs/hybrid_short_T/campaign.json, the full shortened budget B, on the FROZEN common route of
results/hybrid_short_T/routing/routing.json (the same backend for both arms at a given N).  Dispatch order:
seed-major blocks, the three systems interleaved within a block (gateway, lta300, lta150 at seed index b), and within
one (system, seed) the arms in the order default_rng(20261018 + seed).permutation.  --max-seeds k runs only the first
k seed blocks (the smoke stage: k = 2).

Confirmation: analysis/confirm_selection.json (written by analyze.py from the exploration seeds) names per system the
best ABF and the best FR configuration (N, backend); each is run on the 32 confirm_seeds on the same backend.

Refusals: no routing.json; an engine source (serial or parallel) differs from the hashes frozen in routing.json; a
complete record of a different job definition.  Cap: configs guards.cpu_core_hours_cap used core-hours for the whole
study (ledger), projected from the frozen routes' benchmark c_ss (+ 15 s per run).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

import hst_common as H  # noqa: E402
import expB_common as C  # noqa: E402
import benchmark as BM  # noqa: E402

ORDER_SEED_BASE = 20261018
RUN_OVERHEAD_S = 15.0
SELECTION = os.path.join(H.OUT_ROOT, "analysis", "confirm_selection.json")


def engine_check(routing, cfg=None):
    probs = []
    for s in H.SYSTEMS:
        spec = H.system_spec(s, cfg)
        for impl in ("serial", "parallel"):
            cur = C.engine_sources_sha(spec, impl, spec["engine_module"])
            want = routing["engine_sources"][s][impl]
            if cur != want:
                diff = sorted(k for k in set(cur) | set(want) if cur.get(k) != want.get(k))
                probs.append(f"{s} {impl}: engine sources changed since the routing freeze: {diff}")
    return probs


def route_of(routing, system, N):
    return routing["routes"][system][str(int(N))]["common"]


def prod_jobs(routing, systems, max_seeds=None, cfg=None):
    specs = {s: H.system_spec(s, cfg) for s in systems}
    nb = max(len(specs[s]["prod_seeds"]) for s in systems)
    if max_seeds is not None:
        nb = min(nb, int(max_seeds))
    jobs = []
    for b in range(nb):
        for s in systems:
            sp = specs[s]
            if b >= len(sp["prod_seeds"]):
                continue
            seed = sp["prod_seeds"][b]
            arms = [(m, N) for N in sp["N_ladder"] for m in H.methods_for(N)]
            perm = np.random.default_rng(ORDER_SEED_BASE + seed).permutation(len(arms))
            for i in perm:
                m, N = arms[int(i)]
                r = route_of(routing, s, N)
                jobs.append(H.make_job("prod", sp, m, N, seed, r["impl"], r["threads"]))
    return jobs


def confirm_jobs(routing, selection, cfg=None):
    jobs = []
    for s in H.SYSTEMS:
        sel = (selection.get("systems") or {}).get(s)
        if not sel:
            continue
        sp = H.system_spec(s, cfg)
        arms = [(a["method"], int(a["N"]), a["impl"], int(a["threads"])) for a in sel.get("arms", [])]
        for b, seed in enumerate(sp["confirm_seeds"]):
            perm = np.random.default_rng(ORDER_SEED_BASE + seed).permutation(len(arms))
            for i in perm:
                m, N, impl, t = arms[int(i)]
                r = route_of(routing, s, N)
                if (impl, t) != (r["impl"], r["threads"]):
                    raise H.HSTError(f"{s} {m}{N}: selection backend {impl}/{t} != frozen route {r['backend']}")
                jobs.append(H.make_job("confirm", sp, m, N, seed, impl, t))
    return jobs


def cost_fn(routing, summary):
    summ = summary["configs"]

    def cost(job):
        key = f"{job['system']}|{H.config_name(job['method'], job['N'], job['impl'], job['threads'])}"
        c = summ[key]["c_ss_median"]
        return job["N"] * job["n_steps"] * c + RUN_OVERHEAD_S
    return cost


def status(routing, cfg=None):
    import hst_sched as S
    out = {}
    for kind, jobs in (("prod", prod_jobs(routing, H.SYSTEMS, cfg=cfg)),
                       ("confirm", confirm_jobs(routing, C.read_json(SELECTION), cfg) if os.path.exists(SELECTION)
                        else [])):
        n = dict(complete=0, ran=0, missing=0, mismatch=0, contended_all=0)
        for j in jobs:
            sig = H.job_signature(j, H.system_spec(j["system"], cfg))
            st = H.record_status(H.job_paths(j), sig)
            n["complete" if st == "complete" else "ran" if st == "ran" else "mismatch" if st == "mismatch"
              else "missing"] += 1
            if st == "complete":
                recs = S.replicate_records(j, H.OUT_ROOT)
                if recs and not any(not r.get("contended", True) for r in recs):
                    n["contended_all"] += 1
        out[kind] = dict(n_jobs=len(jobs), **n)
    out["used_core_h"] = H.ledger_used_core_h()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("prod", "confirm"):
        p = sub.add_parser(name)
        p.add_argument("--systems", nargs="*", default=list(H.SYSTEMS))
        p.add_argument("--max-seeds", type=int, default=None)
        p.add_argument("--max-ccds", type=int, default=12)
        p.add_argument("--max-cores", type=int, default=96)
        p.add_argument("--cap", type=float, default=None)
        p.add_argument("--dry-run", action="store_true")
    sub.add_parser("status")
    a = ap.parse_args(argv)
    X = H.load_config()
    routing = BM.load_routing()
    if a.cmd == "status":
        print(json.dumps(C.json_safe(status(routing)), indent=1))
        return 0
    probs = engine_check(routing)
    if probs:
        raise H.HSTError("refusing: " + "; ".join(probs))
    summary = C.read_json(os.path.join(BM.BENCH_DIR, "summary.json"))
    if a.cmd == "prod":
        jobs = prod_jobs(routing, a.systems, a.max_seeds)
    else:
        if not os.path.exists(SELECTION):
            raise H.HSTError(f"{SELECTION} missing: run analyze.py on the exploration seeds first")
        jobs = confirm_jobs(routing, C.read_json(SELECTION))
    import hst_sched as S
    cap = a.cap if a.cap is not None else float(X["guards"]["cpu_core_hours_cap"])
    sch = S.Scheduler(jobs, a.cmd, cost_fn(routing, summary), cap_core_h=cap, max_ccds=a.max_ccds,
                      max_cores=a.max_cores, run_timeout_s=float(X["guards"]["run_timeout_s"]))
    print(json.dumps(C.json_safe(sch.run(dry_run=a.dry_run))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
