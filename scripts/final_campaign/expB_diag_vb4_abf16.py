"""Diagnostic for the gateway V-B4 ABF N=16 failure (investigation, NOT a preregistered gate or extension).

V-B4 (PREREGISTRATION.md §2.3) found the counter-mode median Ibar_F of gateway ABF N=16 at 1.30x [1.07, 1.46] the
validated engine's (32 production seeds 8100-8131); against all 96 validated seeds (production + the equal-budget
fresh-seed confirmation 8200-8263) 1.20x [1.05, 1.34], Mann-Whitney p 0.0014.  The kernel replays the validated
arithmetic bitwise (V-B1), the counter normals are i.i.d. N(0,1) to 1e-4 over the full step range, and the Exp B
scorer equals the equal-budget scorer exactly.  This script separates "counter mode differs in law" from sampling:

  A. validated serial engine (gateway_ladder_numba, diagnostics off) on the counter runs' own seeds 8500-8531;
  B. counter mode on a fresh, otherwise unused block 8740-8771 (outside every preregistered block, so the
     preregistered extension seeds 8532-8563 / 8700-8731 stay untouched).

Both are scored with expB_common.Scorer / ibar_uniform exactly like V-B4.  One single-threaded job per process.
Output: results/final_campaign/expB/diagnostics/vb4_abf16/{A_validated,B_counter}/s<seed>.json (+ compressed npz).
Usage: taskset -c <cpus> python scripts/final_campaign/expB_diag_vb4_abf16.py --cpus <same cpus> [--which A|B|both] [--analyze]
(the pool runs one single-threaded process per cpu under the taskset; no per-job pinning)
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT = os.path.join(ROOT, "results/final_campaign/expB/diagnostics/vb4_abf16")
SEEDS_A = list(range(8500, 8532))
SEEDS_B = list(range(8740, 8772))
SEEDS_C = list(range(8500, 8504))   # C: parallel engine in REPLAY mode at full length (long-run bitwise identity vs A)
N = 16


def _job(args):
    which, seed = args
    os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", NUMBA_NUM_THREADS="1")
    sys.path[:0] = [os.path.join(ROOT, p) for p in ("src", "scripts/final_campaign", "scripts/equal_budget", "scripts")]
    import expB_common as C
    spec = C.system_spec("gateway")
    n_steps = C.n_steps_for(spec["B"], N)
    save_steps = C.eqb_production_save_grid(spec, n_steps)
    d = os.path.join(OUT, {"A": "A_validated", "B": "B_counter", "C": "C_replay"}[which])
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"s{seed}.json")
    if os.path.exists(path):
        return json.load(open(path))
    if which == "A":
        res = C.run_serial(spec, "abf", N, seed, n_steps, save_steps)
    elif which == "B":
        res = C.run_parallel(spec, "abf", N, seed, n_steps, save_steps, threads=1, noise="counter")
    else:
        res = C.run_parallel(spec, "abf", N, seed, n_steps, save_steps, threads=1, noise="replay")
    sc = C.Scorer(spec)
    e = sc.e_F(res)
    rec = dict(which=which, seed=seed, N=N, n_steps=int(n_steps), engine=str(res.get("engine", "")),
               Ibar_F=C.ibar_uniform(e, res["save_step"], n_steps), final_e_F=float(e[-1]),
               wall_total_s=float(res.get("wall_total_s", np.nan)), affinity=sorted(os.sched_getaffinity(0)),
               git=C.git_commit())
    np.savez_compressed(os.path.join(d, f"s{seed}.npz"), M_all=np.asarray(res["M_all"]), C_all=np.asarray(res["C_all"]),
                        save_step=np.asarray(res["save_step"]))
    with open(path, "w") as fh:
        json.dump(rec, fh, indent=1)
    return rec


def analyze():
    sys.path[:0] = [os.path.join(ROOT, p) for p in ("scripts/final_campaign", "scripts/equal_budget", "scripts")]
    from scipy.stats import mannwhitneyu

    def load(sub, seeds):
        v = []
        for s in seeds:
            p = os.path.join(OUT, sub, f"s{s}.json")
            if os.path.exists(p):
                v.append(json.load(open(p))["Ibar_F"])
        return np.asarray(v, float)

    def summ(path, N_):
        ps = json.load(open(path))["per_N"][str(N_)]["abf"]["Ibar_F"]["per_seed"]
        return np.asarray(list(ps.values()) if isinstance(ps, dict) else ps, float)

    vb4 = json.load(open(os.path.join(ROOT, "results/final_campaign/expB/equiv/gateway/vb4.json")))
    ps = vb4["per_seed"]["abf16"]["counter"]
    counter_vb4 = np.asarray(list(ps.values()) if isinstance(ps, dict) else ps, float)
    sets = {
        "validated production 8100-8131": summ(os.path.join(ROOT, "results/equal_budget_v2/gateway/analysis/summary.json"), N),
        "validated confirmation 8200-8263": summ(os.path.join(ROOT, "results/equal_budget_v2/confirm_best_gateway/analysis/summary.json"), N),
        "A validated on counter seeds 8500-8531": load("A_validated", SEEDS_A),
        "counter V-B4 8500-8531": counter_vb4,
        "B counter fresh 8740-8771": load("B_counter", SEEDS_B),
    }
    out = {}
    for k, v in sets.items():
        out[k] = dict(n=int(v.size), median=float(np.median(v)) if v.size else None,
                      mean=float(v.mean()) if v.size else None,
                      sdlog=float(np.std(np.log(v))) if v.size else None)
    val = np.concatenate([sets["validated production 8100-8131"], sets["validated confirmation 8200-8263"],
                          sets["A validated on counter seeds 8500-8531"]])
    cnt = np.concatenate([sets["counter V-B4 8500-8531"], sets["B counter fresh 8740-8771"]])
    rng = np.random.default_rng(20261010)
    r = [np.median(cnt[rng.integers(0, cnt.size, cnt.size)]) / np.median(val[rng.integers(0, val.size, val.size)])
         for _ in range(10000)]
    out["pooled"] = dict(n_validated=int(val.size), n_counter=int(cnt.size),
                         ratio=float(np.median(cnt) / np.median(val)), ci95=[float(x) for x in np.percentile(r, [2.5, 97.5])],
                         mannwhitney_p=float(mannwhitneyu(cnt, val).pvalue))
    a, c = sets["A validated on counter seeds 8500-8531"], counter_vb4
    if a.size == c.size and a.size:
        out["same_seed_A_vs_counter"] = dict(ratio_of_medians=float(np.median(c) / np.median(a)),
                                             mannwhitney_p=float(mannwhitneyu(c, a).pvalue))
    bit = {}
    for sd in SEEDS_C:
        pa, pc = os.path.join(OUT, "A_validated", f"s{sd}.npz"), os.path.join(OUT, "C_replay", f"s{sd}.npz")
        if os.path.exists(pa) and os.path.exists(pc):
            za, zc = np.load(pa), np.load(pc)
            bit[str(sd)] = bool(all(np.array_equal(za[k], zc[k]) for k in ("M_all", "C_all", "save_step")))
    out["C_long_run_replay_bitwise_vs_A"] = bit
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "diagnostic_summary.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpus", default="0-31")
    ap.add_argument("--which", default="all", choices=["A", "B", "C", "all"])
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    if a.analyze:
        analyze()
        return
    cpus = []
    for part in a.cpus.split(","):
        lo, _, hi = part.partition("-")
        cpus += list(range(int(lo), int(hi or lo) + 1))
    jobs = []
    if a.which in ("C", "all"):
        jobs += [("C", s) for s in SEEDS_C]
    if a.which in ("A", "all"):
        jobs += [("A", s) for s in SEEDS_A]
    if a.which in ("B", "all"):
        jobs += [("B", s) for s in SEEDS_B]
    ctx = mp.get_context("spawn")
    with ctx.Pool(len(cpus), maxtasksperchild=1) as pool:
        for rec in pool.imap_unordered(_job, jobs):
            print(rec["which"], rec["seed"], round(rec["Ibar_F"], 7), flush=True)
    analyze()


if __name__ == "__main__":
    main()
