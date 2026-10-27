"""Benchmark summary and the frozen routing of the NSCC A100 study (campaign.json 'routing').

    python scripts/nscc_a100/route.py summarize      # results/nscc_a100/bench/summary.json + tables.md
    python scripts/nscc_a100/route.py freeze         # results/nscc_a100/routing/routing.json (refuses to overwrite)
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nscc_common as NC  # noqa: E402

H = NC.H
BENCH = os.path.join(NC.OUT_ROOT, "bench")
ROUTING = os.path.join(NC.OUT_ROOT, "routing", "routing.json")


def load():
    """{(system, method, N, backend): [c_ss of each repeat (ns)]}"""
    out = {}
    for p in glob.glob(os.path.join(BENCH, "*", "*", "s*_r*.json")):
        rec = json.load(open(p))
        if rec.get("status") != "complete" or "bench" not in rec:
            continue
        j = rec["job"]
        out.setdefault((j["system"], j["method"], j["N"], j["backend"]), []).append(
            (int(j["r"]), rec["bench"]["c_ss"] * 1e9, rec["bench"]["c_total"] * 1e9, rec.get("start_overhead_s")))
    return {k: sorted(v) for k, v in out.items()}


def summarize(write=True):
    D = load()
    S = {}
    for (system, method, N, be), v in D.items():
        c = np.array([x[1] for x in v])
        S.setdefault(system, {}).setdefault(method, {}).setdefault(str(N), {})[be] = dict(
            repeats=len(v), c_ss_ns=c.tolist(), median=float(np.median(c)), min=float(c.min()), max=float(c.max()),
            us_per_step=float(np.median(c)) * N * 1e-3,
            c_total_median=float(np.median([x[2] for x in v])),
            start_overhead_s=[x[3] for x in v])
    if write:
        os.makedirs(BENCH, exist_ok=True)
        with open(os.path.join(BENCH, "summary.json"), "w") as f:
            json.dump(dict(summary=S, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())), f, indent=1)
    return S


def faster_all(a, b):
    return max(a) < min(b)


def cpu_route(ent, min_gain=1.05, tie_frac=0.03):
    """The hybrid short-T rule: S1 vs the fastest walker-parallel thread count (ties within 3 % to fewer threads)."""
    s1 = ent.get("S1")
    par = {be: e for be, e in ent.items() if be.startswith("P")}
    if not par:
        return "S1", "only S1"
    best = min(e["median"] for e in par.values())
    tied = sorted((int(be[1:]), be) for be, e in par.items() if e["median"] <= best * (1 + tie_frac))
    pstar = tied[0][1]
    if s1 is None:
        return pstar, "no S1 entry"
    ok = par[pstar]["median"] <= s1["median"] / min_gain and faster_all(par[pstar]["c_ss_ns"], s1["c_ss_ns"])
    return (pstar, f"{pstar} >= 5 % faster than S1 in every repeat") if ok else ("S1", f"{pstar} not >= 5 % faster")


def route_one(ent, min_gain=1.05, tie_frac=0.03):
    cpu, why = cpu_route(ent, min_gain, tie_frac)
    g = ent.get("G1")
    if g is None:
        return dict(backend=cpu, cpu_route=cpu, why=why + "; no G1 entry")
    c = ent[cpu]
    ok = g["median"] <= c["median"] / min_gain and faster_all(g["c_ss_ns"], c["c_ss_ns"])
    gain = c["median"] / g["median"]
    return dict(backend="G1" if ok else cpu, cpu_route=cpu, gpu_speedup_vs_cpu_route=gain,
                why=why + (f"; G1 {gain:.2f}x faster than {cpu} in every repeat" if ok
                           else f"; G1 {gain:.2f}x vs {cpu}: not routed"))


def routing(S):
    cfg = NC.load_config()
    rr = cfg["routing"]
    out = dict(route={}, deployment_optimised={})
    for system in NC.SYSTEMS:
        spec = H.system_spec(system)
        out["route"][system] = {}
        out["deployment_optimised"][system] = {}
        for N in sorted(spec["N_ladder"]):
            ent = S[system]["abf"][str(N)]
            out["route"][system][str(N)] = route_one(ent, rr["min_gain"], rr["tie_frac"])
            out["deployment_optimised"][system][str(N)] = {
                m: route_one(S[system][m][str(N)], rr["min_gain"], rr["tie_frac"])["backend"]
                for m in H.methods_for(N) if str(N) in S[system].get(m, {})}
    return out


def freeze():
    if os.path.exists(ROUTING):
        raise SystemExit(f"{ROUTING} exists (frozen); refusing to overwrite")
    S = summarize()
    R = routing(S)
    srcs = {}
    for system in NC.SYSTEMS:
        spec = H.system_spec(system)
        for impl in ("serial", "parallel", "gpu"):
            srcs[f"{system}:{impl}"] = NC.C.files_sha(NC.engine_sources(spec, impl))
    R.update(frozen_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), engine_sources=srcs,
             launch_table=json.load(open(os.path.join(NC.ROOT, "configs", "nscc_a100", "gpu_launch.json"))),
             rule=NC.load_config()["routing"]["rule"])
    os.makedirs(os.path.dirname(ROUTING), exist_ok=True)
    with open(ROUTING, "w") as f:
        json.dump(R, f, indent=1)
    print(ROUTING)
    tables(S, R)


def tables(S, R=None):
    lines = ["# NSCC A100 study: throughput benchmark", "",
             "c_ss = steady-state wall seconds per walker-step (ns), median of 3 repeats on the hybrid short-T "
             "benchmark windows (FR active over the timed window). CPU: AMD EPYC 7713 (exclusive CCDs); "
             "G1: one A100-SXM4-40GB. us/step = c_ss N.", ""]
    for system in NC.SYSTEMS:
        if system not in S:
            continue
        lines += [f"## {system}", "", "| N | method | S1 | best P_t | G1 | G1 us/step | G1 / best CPU | route |",
                  "|---|---|---|---|---|---|---|---|"]
        for N in sorted(int(k) for k in S[system].get("abf", {})):
            for m in ("abf", "fr"):
                ent = S[system].get(m, {}).get(str(N))
                if not ent:
                    continue
                par = {be: e for be, e in ent.items() if be.startswith("P")}
                bp = min(par.items(), key=lambda kv: kv[1]["median"]) if par else None
                cpu_best = min([e["median"] for be, e in ent.items() if be != "G1"] or [math.nan])
                g = ent.get("G1")
                rt = R["route"][system][str(N)]["backend"] if R else ""
                lines.append(f"| {N} | {m} | {ent['S1']['median']:.1f} | "
                             f"{(bp[0] + ' ' + format(bp[1]['median'], '.1f')) if bp else '--'} | "
                             f"{format(g['median'], '.1f') if g else '--'} | {format(g['us_per_step'], '.2f') if g else '--'} | "
                             f"{format(cpu_best / g['median'], '.2f') + 'x' if g else '--'} | {rt} |")
        lines.append("")
    with open(os.path.join(BENCH, "tables.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(os.path.join(BENCH, "tables.md"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("summarize", "freeze"))
    a = ap.parse_args(argv)
    if a.cmd == "summarize":
        S = summarize()
        R = routing(S) if all(s in S for s in NC.SYSTEMS) else None
        tables(S, R)
    else:
        freeze()


if __name__ == "__main__":
    main()
