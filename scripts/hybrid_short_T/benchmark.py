#!/usr/bin/env python
"""Stage II: CPU throughput benchmark (serial S1 vs walker-parallel P_t) and the frozen routing rule.

    python scripts/hybrid_short_T/benchmark.py bench [--dry-run] [--max-ccds 6] [--systems gateway lta300 lta150]
    python scripts/hybrid_short_T/benchmark.py summary          # results/hybrid_short_T/bench/summary.json
    python scripts/hybrid_short_T/benchmark.py freeze           # results/hybrid_short_T/routing/routing.json (ONCE)

Protocol (prereg section 4; configs/hybrid_short_T/campaign.json 'benchmark').  Configurations: every N of each
ladder, ABF at every N and FR at N >= 2, backends S1 and P_t for t in {1, 2, 4, 8, 16} with t <= N.  Each run is the
production configuration truncated to the bench window n_b(N) (hst_common.bench_window), a fresh process on an
exclusive CCD set with the 1 Hz telemetry monitor (hst_sched.Scheduler, kind 'bench'); a run contended over its whole
timed window is repeated (at most 3 times).  Order: the configuration list of all three systems in a fixed shuffled
order (default_rng(20261018)); repeat r is that list rotated by r L / 3 and uses bench_seeds[r].

Summary per (system, method, N, backend): the steady-state cost c_ss of every repeat (the first uncontended replicate
of that repeat; a repeat with none is excluded and counted), median / min / max over repeats, throughput 1 / c_ss,
c_total (whole timed run), c_e2e ((start overhead + timed run) / walker-steps), warm-up seconds, peak RSS, the
parallel engine's phase split; FR overhead = c_ss(FR) / c_ss(ABF) - 1 per repeat at the same (N, backend).

Routing (frozen rule, prereg section 4): per system and N from the ABF configurations only: P* = the fastest parallel
thread count by median c_ss (within 3 % = tied -> fewer threads); route to P* iff median c_ss(P*) <= median c_ss(S1)
/ 1.05 AND max over repeats of c_ss(P*) < min over repeats of c_ss(S1); else S1.  Secondary (deployment-optimised):
the same rule on each method's own configurations.  ``freeze`` writes routing.json once (refuses to overwrite), with
the engine source hashes, the production cost projection and the validation verdicts; run_campaign.py refuses to
start without it or if an engine source changed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

import hst_common as H  # noqa: E402
import expB_common as C  # noqa: E402

BENCH_DIR = os.path.join(H.OUT_ROOT, "bench")
ROUTING_DIR = os.path.join(H.OUT_ROOT, "routing")
ROUTING = os.path.join(ROUTING_DIR, "routing.json")
ORDER_SEED = 20261018
THREADS = (1, 2, 4, 8, 16)
RUN_OVERHEAD_S = 15.0
GUESS_NS = {"gateway": 80.0, "lta300": 900.0, "lta150": 900.0}


# ================================================================================================ plan
def configs(spec):
    out = []
    for N in spec["N_ladder"]:
        for m in H.methods_for(N):
            out.append((spec["system"], m, N, "serial", 1))
            for t in THREADS:
                if t <= N:
                    out.append((spec["system"], m, N, "parallel", t))
    return out


def bench_jobs(systems, cfg=None, repeats=3):
    specs = {s: H.system_spec(s, cfg) for s in systems}
    allc = [c for s in systems for c in configs(specs[s])]
    perm = np.random.default_rng(ORDER_SEED).permutation(len(allc))
    base = [allc[i] for i in perm]
    L = len(base)
    jobs = []
    for r in range(repeats):
        k = (r * L) // repeats
        for (s, m, N, impl, t) in base[k:] + base[:k]:
            jobs.append(H.make_job("bench", specs[s], m, N, specs[s]["bench_seeds"][r], impl, t, r=r))
    return jobs


def bench_cost_s(job):
    ws = job["N"] * job["n_steps"]
    c = GUESS_NS[job["system"]] * 1e-9
    if job["impl"] == "parallel":
        c = c / max(1.0, 0.6 * job["threads"])
    return ws * c + RUN_OVERHEAD_S


# ================================================================================================ summary
def load_bench(out_root=H.OUT_ROOT):
    """{(system, config, r): [complete records of all replicates]}."""
    out = {}
    root = os.path.join(out_root, "bench")
    if not os.path.isdir(root):
        return out
    for sysn in sorted(os.listdir(root)):
        d = os.path.join(root, sysn)
        if not os.path.isdir(d):
            continue
        for cfgn in sorted(os.listdir(d)):
            cd = os.path.join(d, cfgn)
            if not os.path.isdir(cd):
                continue
            for f in sorted(os.listdir(cd)):
                if not f.endswith(".json") or ".failed" in f:
                    continue
                rec = C.read_json(os.path.join(cd, f))
                if rec.get("status") != "complete":
                    continue
                j = rec["job"]
                out.setdefault((sysn, cfgn, int(j["r"])), []).append(rec)
    for v in out.values():
        v.sort(key=lambda r: r["job"]["replicate"])
    return out


def pick(recs):
    """The first uncontended replicate of one repeat (None if every replicate is contended)."""
    for r in recs:
        if not r.get("contended", True):
            return r
    return None


def summarize(out_root=H.OUT_ROOT, cfg=None, write=True):
    B = load_bench(out_root)
    rows = {}
    for (sysn, cfgn, r), recs in B.items():
        rows.setdefault((sysn, cfgn), {})[r] = recs
    summ = {}
    for (sysn, cfgn), byr in sorted(rows.items()):
        j0 = next(iter(byr.values()))[0]["job"]
        vals, tot, e2e, warm, rss, flagged, n_contended_reps = [], [], [], [], [], 0, 0
        timers = []
        per_r = {}
        for r, recs in sorted(byr.items()):
            p = pick(recs)
            if p is None:
                n_contended_reps += 1
                continue
            b = p["bench"]
            vals.append(b["c_ss"])
            per_r[int(r)] = b["c_ss"]
            tot.append(b["c_total"])
            so = p.get("start_overhead_s", math.nan)
            e2e.append((so + p["wall_total_s"]) / b["walker_steps"] if math.isfinite(so) else math.nan)
            warm.append(p.get("warmup_s", math.nan))
            rss.append(p.get("peak_rss_mb", math.nan))
            flagged += int(bool((p.get("allocation") or {}).get("flagged")))
            if p.get("timers") and all(math.isfinite(v) for v in p["timers"].values()):
                timers.append({k: v / p["wall_total_s"] for k, v in p["timers"].items()})
        key = f"{sysn}|{cfgn}"
        if not vals:
            summ[key] = dict(system=sysn, config=cfgn, method=j0["method"], N=j0["N"], impl=j0["impl"],
                             threads=j0["threads"], n=0, n_contended_repeats=n_contended_reps)
            continue
        v = np.asarray(vals)
        summ[key] = dict(
            system=sysn, config=cfgn, method=j0["method"], N=int(j0["N"]), impl=j0["impl"],
            threads=int(j0["threads"]), backend=H.backend_label(j0["impl"], j0["threads"]), n=int(v.size),
            n_contended_repeats=n_contended_reps, n_flagged_alloc=flagged, c_ss=v.tolist(), c_ss_by_r=per_r,
            c_ss_median=float(np.median(v)), c_ss_min=float(v.min()), c_ss_max=float(v.max()),
            throughput_median=float(1.0 / np.median(v)), c_total_median=float(np.median(tot)),
            c_e2e_median=float(np.nanmedian(e2e)) if np.any(np.isfinite(e2e)) else math.nan,
            warmup_s_median=float(np.nanmedian(warm)), peak_rss_mb_max=float(np.nanmax(rss)),
            timer_fractions=({k: float(np.median([t[k] for t in timers])) for k in timers[0]} if timers else None),
            n_b=int(j0["n_steps"]))
    # FR overhead per (system, N, backend) and repeat
    fro = {}
    for key, s in summ.items():
        if s["method"] != "fr" or not s["n"]:
            continue
        ka = key.replace("|fr", "|abf", 1)
        a = summ.get(ka)
        if not a or not a["n"]:
            continue
        rr = sorted(set(s["c_ss_by_r"]) & set(a["c_ss_by_r"]))
        ov = [s["c_ss_by_r"][r] / a["c_ss_by_r"][r] - 1.0 for r in rr]
        if ov:
            fro[key] = dict(system=s["system"], N=s["N"], backend=s["backend"], overhead=ov,
                            overhead_median=float(np.median(ov)), overhead_min=float(min(ov)),
                            overhead_max=float(max(ov)))
    out = dict(version=H.VERSION, configs=summ, fr_overhead=fro, n_records=sum(len(v) for v in B.values()),
               utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    if write:
        C.write_json(os.path.join(BENCH_DIR, "summary.json"), out)
    return out


# ================================================================================================ routing
def route_one(summ, system, method, N, min_gain=1.05, tie_frac=0.03):
    """The frozen rule for one (system, method, N) from the benchmark summary."""
    s1 = summ.get(f"{system}|{H.config_name(method, N, 'serial', 1)}")
    par = [summ.get(f"{system}|{H.config_name(method, N, 'parallel', t)}") for t in THREADS if t <= N]
    par = [p for p in par if p and p["n"]]
    if not s1 or not s1["n"]:
        raise H.HSTError(f"{system} {method} N {N}: no uncontended S1 benchmark")
    out = dict(system=system, method=method, N=int(N), S1=dict(c_ss_median=s1["c_ss_median"],
                                                                c_ss_min=s1["c_ss_min"], c_ss_max=s1["c_ss_max"],
                                                                n=s1["n"]))
    if not par:
        out.update(impl="serial", threads=1, backend="S1", reason="no parallel benchmark")
        return out
    best = min(par, key=lambda p: p["c_ss_median"])
    tied = [p for p in par if p["c_ss_median"] <= (1.0 + tie_frac) * best["c_ss_median"]]
    pstar = min(tied, key=lambda p: p["threads"])
    out["P_star"] = dict(threads=pstar["threads"], c_ss_median=pstar["c_ss_median"], c_ss_min=pstar["c_ss_min"],
                         c_ss_max=pstar["c_ss_max"], n=pstar["n"], fastest_threads=best["threads"],
                         tied_threads=sorted(p["threads"] for p in tied))
    gain = s1["c_ss_median"] / pstar["c_ss_median"]
    reliable = pstar["c_ss_max"] < s1["c_ss_min"]
    out["gain_vs_S1"] = gain
    out["reliably_faster"] = bool(reliable)
    if gain >= min_gain and reliable:
        out.update(impl="parallel", threads=int(pstar["threads"]), backend=f"P{pstar['threads']}",
                   c_ss_median=pstar["c_ss_median"], reason=f"P{pstar['threads']} {gain:.2f}x S1, every repeat faster")
    else:
        out.update(impl="serial", threads=1, backend="S1", c_ss_median=s1["c_ss_median"],
                   reason=(f"best parallel P{pstar['threads']} only {gain:.3f}x S1" if gain < min_gain else
                           f"P{pstar['threads']} {gain:.2f}x S1 but repeats overlap"))
    return out


def routing(summary, cfg=None):
    X = H.load_config() if cfg is None else cfg
    R = X["routing"]
    summ = summary["configs"]
    out = {}
    for s in H.SYSTEMS:
        spec = H.system_spec(s, cfg)
        rows = {}
        for N in spec["N_ladder"]:
            common = route_one(summ, s, "abf", N, R["min_gain"], R["tie_frac"])
            own = {m: route_one(summ, s, m, N, R["min_gain"], R["tie_frac"]) for m in H.methods_for(N)}
            costs = {}
            for m in H.methods_for(N):
                for impl, t in (("common", None), ("own", None)):
                    rr = common if impl == "common" else own[m]
                    key = f"{s}|{H.config_name(m, N, rr['impl'], rr['threads'])}"
                    costs[f"{m}_{impl}"] = summ[key]["c_ss_median"] if key in summ and summ[key]["n"] else math.nan
            rows[str(N)] = dict(common=common, own=own, c_ss_on_route=costs)
        out[s] = rows
    return out


def projection(route, summary, cfg=None):
    """Projected used core-hours and wall of production + confirmation on the frozen routes."""
    summ = summary["configs"]
    tot = dict(prod_core_h=0.0, prod_wall_core_s=0.0, confirm_core_h_upper=0.0)
    per = {}
    for s in H.SYSTEMS:
        spec = H.system_spec(s, cfg)
        ch, mx = 0.0, 0.0
        for N in spec["N_ladder"]:
            rr = route[s][str(N)]["common"]
            for m in H.methods_for(N):
                key = f"{s}|{H.config_name(m, N, rr['impl'], rr['threads'])}"
                c = summ[key]["c_ss_median"]
                w = spec["B"] * c + RUN_OVERHEAD_S
                ch += len(spec["prod_seeds"]) * rr["threads"] * w / 3600.0
                mx = max(mx, rr["threads"] * w)
        conf = 2 * len(spec["confirm_seeds"]) * mx / 3600.0
        per[s] = dict(prod_core_h=ch, confirm_core_h_upper=conf)
        tot["prod_core_h"] += ch
        tot["confirm_core_h_upper"] += conf
    return dict(per_system=per, **tot)


def freeze(out_root=H.OUT_ROOT, cfg=None, force=False):
    if os.path.exists(ROUTING) and not force:
        raise H.HSTError(f"{ROUTING} exists: the routing is frozen once (an amendment is needed to change it)")
    summary = summarize(out_root, cfg)
    route = routing(summary, cfg)
    val = os.path.join(H.OUT_ROOT, "validation", "validation.json")
    vrec = C.read_json(val) if os.path.exists(val) else None
    blocked = (vrec or {}).get("blocked") or {}
    for s, why in blocked.items():          # a failed gate blocks the parallel backend for that system
        for N, row in route.get(s, {}).items():
            for d in [row["common"]] + list(row["own"].values()):
                if d["impl"] == "parallel":
                    d.update(impl="serial", threads=1, backend="S1", reason=f"BLOCKED by validation: {why}")
    srcs = {}
    for s in H.SYSTEMS:
        spec = H.system_spec(s, cfg)
        srcs[s] = {impl: C.engine_sources_sha(spec, impl, spec["engine_module"]) for impl in ("serial", "parallel")}
    used = H.ledger_used_core_h()
    rec = dict(version=H.VERSION, frozen_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               git_commit=C.git_commit()[0], rule=(cfg or H.load_config())["routing"], routes=route,
               engine_sources=srcs, projection=projection(route, summary, cfg), used_core_h_at_freeze=used,
               validation=vrec, bench_records=summary["n_records"])
    C.write_json(ROUTING, rec)
    return rec


def load_routing(path=ROUTING):
    if not os.path.exists(path):
        raise H.HSTError(f"{path} missing: run benchmark.py freeze first (prereg section 4)")
    return C.read_json(path)


# ================================================================================================ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bench")
    b.add_argument("--systems", nargs="*", default=list(H.SYSTEMS))
    b.add_argument("--max-ccds", type=int, default=6)
    b.add_argument("--dry-run", action="store_true")
    b.add_argument("--cap", type=float, default=None)
    b.add_argument("--out-root", default=H.OUT_ROOT, help="tests / smoke only")
    b.add_argument("--max-jobs", type=int, default=None, help="tests / smoke only: the first k jobs")
    b.add_argument("--only", nargs="*", default=None, help="tests / smoke only: config names")
    sub.add_parser("summary")
    sub.add_parser("freeze")
    a = ap.parse_args(argv)
    if a.cmd == "bench":
        import hst_sched as S
        X = H.load_config()
        cap = a.cap if a.cap is not None else float(X["guards"]["cpu_core_hours_cap"])
        jobs = bench_jobs(a.systems)
        if a.only:
            jobs = [j for j in jobs if j["config"] in a.only]
        if a.max_jobs is not None:
            jobs = jobs[:a.max_jobs]
        sch = S.Scheduler(jobs, "bench", bench_cost_s, out_root=a.out_root, cap_core_h=cap, max_ccds=a.max_ccds,
                          max_cores=a.max_ccds * 8, run_timeout_s=1800.0)
        print(json.dumps(C.json_safe(sch.run(dry_run=a.dry_run))))
        if not a.dry_run and a.out_root == H.OUT_ROOT:
            summarize()
        return 0
    if a.cmd == "summary":
        s = summarize()
        for k, v in sorted(s["configs"].items()):
            if v["n"]:
                print(f"{k:32s} n {v['n']} c_ss {v['c_ss_median'] * 1e9:9.2f} ns/ws "
                      f"[{v['c_ss_min'] * 1e9:.2f}, {v['c_ss_max'] * 1e9:.2f}]  thr {v['throughput_median']:.3g}/s")
            else:
                print(f"{k:32s} NO UNCONTENDED REPEAT")
        return 0
    if a.cmd == "freeze":
        rec = freeze()
        for s, rows in rec["routes"].items():
            for N, row in rows.items():
                c = row["common"]
                print(f"{s:8s} N {N:>5s} -> {c['backend']:4s} ({c['reason']})")
        print(json.dumps(C.json_safe(rec["projection"]), indent=1))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
