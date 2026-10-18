#!/usr/bin/env python
"""Stage III: production (exploration seeds) and the fresh-seed confirmation of the hybrid short-T study.

    python scripts/hybrid_short_T/run_campaign.py prod [--systems ...] [--max-seeds 2] [--dry-run] [--max-ccds 6]
    python scripts/hybrid_short_T/run_campaign.py confirm [--dry-run]       # needs analysis/confirm_selection.json
    python scripts/hybrid_short_T/run_campaign.py status                    # complete / pending counts

Production (prereg section 5, Amendment A2): per system every N of the ladder, ABF and (N >= 2) FR, the prod_seeds
of configs/hybrid_short_T/campaign.json, the full shortened budget B, on the FROZEN common route of
results/hybrid_short_T/routing/routing.json (the same backend for both arms at a given N).  Dispatch order:
seed-major blocks, the three systems interleaved within a block (gateway, lta300, lta150 at seed index b), within one
(system, seed) the arms in the order default_rng(20261018 + seed).permutation (the scheduler's A2.4 look-ahead may
start a later job of the queue while the head waits for an idle CCD set).  --max-seeds k runs only the first k seed
blocks (the smoke stage: k = 2).  A2.6: the job list is trimmed to the largest number of WHOLE seed blocks whose
projected used core-hours (benchmark c_ss on the route x B + 15 s, x threads, x the expected attempts at the
benchmark's contended fraction of that thread count) fit the remaining cap; the rest is reported NOT RUN.

Confirmation (A2.7): analysis/confirm_selection.json lists per system the top-3 configurations of each method; the
design (arms per method k, seeds per arm n) is the first of (3, 32), (2, 32), (2, 16), (1, 16) whose projected cost
fits the remaining cap (else NOT RUN); every selected arm runs on its frozen route.

Concurrency (A2.4): at most 6 CCD sets of this campaign at a time (the benchmark's).  Refusals: no routing.json; an
engine source or a run-time harness file differs from the hashes frozen in routing.json (unless
--allow-harness-change '<amendment>'); a complete record of a different job definition.
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
CONFIRM_DESIGNS = ((3, 32), (2, 32), (2, 16), (1, 16))
PLAN_DIR = os.path.join(H.OUT_ROOT, "plans")


def source_check(routing, cfg=None):
    """(engine problems, harness problems) against routing.json's frozen hashes."""
    eng, har = [], []
    for s in H.SYSTEMS:
        spec = H.system_spec(s, cfg)
        for impl in ("serial", "parallel"):
            cur = C.engine_sources_sha(spec, impl, spec["engine_module"])
            want = routing["engine_sources"][s][impl]
            if cur != want:
                diff = sorted(k for k in set(cur) | set(want) if cur.get(k) != want.get(k))
                eng.append(f"{s} {impl}: engine sources changed since the routing freeze: {diff}")
    want = routing.get("harness_sources") or {}
    if want:
        cur = C.files_sha([os.path.join(H.ROOT, k) for k in want])
        diff = sorted(k for k in want if cur.get(k) != want[k])
        if diff:
            har.append(f"run-time harness changed since the routing freeze: {diff}")
    return eng, har


def route_of(routing, system, N):
    return routing["routes"][system][str(int(N))]["common"]


def prod_jobs(routing, systems, max_seeds=None, cfg=None):
    """[(block index, job)] in dispatch order."""
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
                jobs.append((b, H.make_job("prod", sp, m, N, seed, r["impl"], r["threads"])))
    return jobs


def projected_core_h(job, summary):
    """Projected used core-hours of one job incl. expected repeats (A2.6)."""
    r = dict(impl=job["impl"], threads=job["threads"])
    return BM.run_cost_core_s(summary, r, job["system"], job["method"], job["N"],
                              job["N"] * job["n_steps"]) / 3600.0


def trim_blocks(block_jobs, summary, remaining, done_fn):
    """A2.6: the largest number of whole seed blocks whose projected cost of the NOT-yet-complete jobs fits."""
    cost = {}
    for b, j in block_jobs:
        if not done_fn(j):
            cost[b] = cost.get(b, 0.0) + projected_core_h(j, summary)
    blocks = sorted({b for b, _ in block_jobs})
    keep, acc = [], 0.0
    for b in blocks:
        c = cost.get(b, 0.0)
        if acc + c > remaining:
            break
        keep.append(b)
        acc += c
    return keep, acc, cost


def confirm_plan(routing, selection, summary, remaining, cfg=None):
    """A2.7: the first design (k arms per method, n seeds per arm) that fits; [] if none."""
    for k, n in CONFIRM_DESIGNS:
        jobs = confirm_jobs(routing, selection, k, n, cfg)
        need = sum(projected_core_h(j, summary) for j in jobs)
        if need <= remaining:
            return dict(k=k, n_seeds=n, projected_core_h=need), jobs
    return dict(k=0, n_seeds=0, projected_core_h=0.0, not_run="no design fits the remaining cap"), []


def confirm_jobs(routing, selection, k=3, n_seeds=32, cfg=None):
    jobs = []
    for s in H.SYSTEMS:
        sel = (selection.get("systems") or {}).get(s)
        if not sel:
            continue
        sp = H.system_spec(s, cfg)
        arms = [(a["method"], int(a["N"]), a["impl"], int(a["threads"])) for a in sel.get("arms", [])
                if int(a["rank"]) <= k]
        for seed in sp["confirm_seeds"][:n_seeds]:
            perm = np.random.default_rng(ORDER_SEED_BASE + seed).permutation(len(arms))
            for i in perm:
                m, N, impl, t = arms[int(i)]
                r = route_of(routing, s, N)
                if (impl, t) != (r["impl"], r["threads"]):
                    raise H.HSTError(f"{s} {m}{N}: selection backend {impl}/{t} != frozen route {r['backend']}")
                jobs.append(H.make_job("confirm", sp, m, N, seed, impl, t))
    return jobs


def cost_fn(summary):
    def cost(job):
        e = BM.entry(summary["configs"], job["system"], job["method"], job["N"], job["impl"], job["threads"])
        return job["N"] * job["n_steps"] * e["c_ss_median"] + RUN_OVERHEAD_S
    return cost


def is_complete(job, cfg=None):
    sig = H.job_signature(job, H.system_spec(job["system"], cfg))
    return H.record_status(H.job_paths(job), sig) == "complete"


def status(routing, cfg=None):
    import hst_sched as S
    out = {}
    sel = C.read_json(SELECTION) if os.path.exists(SELECTION) else None
    plan_c = os.path.join(PLAN_DIR, "confirm_plan.json")
    cdesign = C.read_json(plan_c) if os.path.exists(plan_c) else None
    for kind, jobs in (("prod", [j for _, j in prod_jobs(routing, H.SYSTEMS, cfg=cfg)]),
                       ("confirm", confirm_jobs(routing, sel, cdesign["k"], cdesign["n_seeds"], cfg)
                        if (sel and cdesign and cdesign.get("k")) else [])):
        n = dict(complete=0, ran=0, missing=0, mismatch=0, no_uncontended_replicate=0)
        for j in jobs:
            sig = H.job_signature(j, H.system_spec(j["system"], cfg))
            st = H.record_status(H.job_paths(j), sig)
            n["complete" if st == "complete" else "ran" if st == "ran" else "mismatch" if st == "mismatch"
              else "missing"] += 1
            if st == "complete":
                recs = S.replicate_records(j, H.OUT_ROOT, BM.PROD_MAX_REPEATS)
                if recs and not any(not r.get("contended", True) for r in recs):
                    n["no_uncontended_replicate"] += 1
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
        p.add_argument("--max-ccds", type=int, default=6)
        p.add_argument("--max-cores", type=int, default=48)
        p.add_argument("--cap", type=float, default=None)
        p.add_argument("--dry-run", action="store_true")
        p.add_argument("--allow-harness-change", default=None, help="amendment reference that licenses it")
    sub.add_parser("status")
    a = ap.parse_args(argv)
    X = H.load_config()
    routing = BM.load_routing()
    if a.cmd == "status":
        print(json.dumps(C.json_safe(status(routing)), indent=1))
        return 0
    eng, har = source_check(routing)
    if eng:
        raise H.HSTError("refusing: " + "; ".join(eng))
    if har and not a.allow_harness_change:
        raise H.HSTError("refusing (pass --allow-harness-change '<amendment>' if licensed): " + "; ".join(har))
    summary = C.read_json(os.path.join(BM.BENCH_DIR, "summary.json"))
    cap = a.cap if a.cap is not None else float(X["guards"]["cpu_core_hours_cap"])
    used = H.ledger_used_core_h()
    os.makedirs(PLAN_DIR, exist_ok=True)
    if a.cmd == "prod":
        bj = prod_jobs(routing, a.systems, a.max_seeds)
        keep, need, cost = trim_blocks(bj, summary, cap - used, is_complete)
        nb = len({b for b, _ in bj})
        plan = dict(utc=C.json_safe(__import__("time").strftime("%Y-%m-%dT%H:%M:%SZ")), used_core_h=used, cap=cap,
                    blocks_requested=nb, blocks_kept=len(keep), projected_core_h_kept=need,
                    per_block_projected_core_h={str(b): c for b, c in sorted(cost.items())},
                    not_run_blocks=[b for b in range(nb) if b not in keep], harness_change=a.allow_harness_change)
        C.write_json(os.path.join(PLAN_DIR, "prod_plan.json" if a.max_seeds is None else
                                  f"prod_plan_max{a.max_seeds}.json"), plan)
        print(json.dumps(C.json_safe(plan), indent=1))
        jobs = [j for b, j in bj if b in keep]
    else:
        if not os.path.exists(SELECTION):
            raise H.HSTError(f"{SELECTION} missing: run analyze.py --select on the exploration seeds first")
        pfile = os.path.join(PLAN_DIR, "confirm_plan.json")
        if os.path.exists(pfile):
            design = C.read_json(pfile)
            jobs = confirm_jobs(routing, C.read_json(SELECTION), design["k"], design["n_seeds"]) if design["k"] \
                else []
        else:
            design, jobs = confirm_plan(routing, C.read_json(SELECTION), summary, cap - used)
            design.update(used_core_h=used, cap=cap)
            if not a.dry_run:
                C.write_json(pfile, design)
        print(json.dumps(C.json_safe(design), indent=1))
    import hst_sched as S
    sch = S.Scheduler(jobs, a.cmd, cost_fn(summary), cap_core_h=cap, max_ccds=a.max_ccds,
                      max_cores=a.max_cores, run_timeout_s=float(X["guards"]["run_timeout_s"]),
                      max_repeats=BM.PROD_MAX_REPEATS)
    print(json.dumps(C.json_safe(sch.run(dry_run=a.dry_run))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
