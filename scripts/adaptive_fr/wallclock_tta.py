"""Experiment E.4 (+ E-A2): wall-clock time to accuracy = measured seconds per step at P numba threads x the
configuration's per-seed sustained time to accuracy in steps (geometric mean over the five kT rungs, restricted at the
common budget; from equal_budget_frontier.py).  Combines measured throughput with measured tau; it is not a re-run at
every P.  Seeds whose tau is censored at the budget enter at the budget (a lower bound), as in E.3.

Throughput source: benchmark run 2 (E_wallclock_bench_run2.json, all K in one session) is primary; run 1
(E_wallclock_bench.json, K >= 64, before E-A2) is reported as a replicate (run2 / run1 per row).

Statistics per P: median wall-clock per configuration; best configuration per algorithm by the smaller median;
best FR / best ABF and same-K FR / ABF ratios of medians with an unpaired independent bootstrap of seeds (10 000).
The throughput is treated as a constant (its run-to-run spread is reported separately, not propagated).

    python scripts/adaptive_fr/wallclock_tta.py
"""
import json
import os

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
AN = os.path.join(ROOT, "results", "adaptive_fr_campaign", "analysis")


def ratio_ci(a, b, n=10000, seed=2026):
    rng = np.random.default_rng(seed)
    ma = np.median(a[rng.integers(0, len(a), (n, len(a)))], 1)
    mb = np.median(b[rng.integers(0, len(b), (n, len(b)))], 1)
    return float(np.median(a) / np.median(b)), [float(np.quantile(ma / mb, 0.025)), float(np.quantile(ma / mb, 0.975))]


def median_ci(a, n=10000, seed=2026):
    rng = np.random.default_rng(seed)
    m = np.median(a[rng.integers(0, len(a), (n, len(a)))], 1)
    return [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def bench(name):
    p = os.path.join(AN, name)
    if not os.path.exists(p):
        return None, {}
    b = json.load(open(p))
    return b, {(r["K"], r["method"], r["threads"]): r["s_per_step"] for r in b["rows"]}


def main():
    fr = json.load(open(os.path.join(AN, "E_frontier.json")))
    b1, s1 = bench("E_wallclock_bench.json")
    b2, s2 = bench("E_wallclock_bench_run2.json")
    sps = s2 or s1
    src = "run2" if s2 else "run1"
    out = dict(definition=__doc__, throughput_source=src,
               bench_meta={k: ({kk: vv for kk, vv in b.items() if kk != "rows"} if b else None) for k, b in (("run1", b1), ("run2", b2))})
    if s1 and s2:
        out["replicate_run2_over_run1"] = {f"K{K}_{m}_P{P}": s2[(K, m, P)] / v for (K, m, P), v in s1.items() if (K, m, P) in s2}
    cfgs = fr["configs"]
    Ps = sorted({P for (_, _, P) in sps})
    for mode in ("production", "all_in_common"):
        geo = fr[mode]["geo"]
        res = dict(per_P={}, best_thread_count={})
        for P in Ps + ["best"]:
            row = dict(configs={})
            wall = {}
            for k, c in cfgs.items():
                K, m = c["K"], ("abf" if c["alg"] == "ABF" else "fr")
                if P == "best":
                    cand = {pp: sps[(K, m, pp)] for pp in Ps if (K, m, pp) in sps}
                    if not cand:
                        continue
                    pbest = min(cand, key=cand.get)
                    s = cand[pbest]
                else:
                    if (K, m, P) not in sps:
                        continue
                    s, pbest = sps[(K, m, P)], P
                tau_steps = np.array(geo[k]["per_seed"]) * 1e8 / K
                wall[k] = tau_steps * s
                row["configs"][k] = dict(K=K, alg=c["alg"], threads=pbest, s_per_step=s, median_tau_steps=float(np.median(tau_steps)),
                                         median_wall_s=float(np.median(wall[k])), ci95=median_ci(wall[k]))
            best = {}
            for alg in ("ABF", "FR-stop"):
                ks = [k for k in wall if cfgs[k]["alg"] == alg]
                best[alg] = min(ks, key=lambda k: np.median(wall[k]))
            row["best_allocation"] = best
            row["ratios"] = {}
            for K in sorted({c["K"] for c in cfgs.values()}, reverse=True):
                a_, b_ = f"FR_K{K}", f"ABF_K{K}"
                if a_ in wall and b_ in wall:
                    m_, ci = ratio_ci(wall[a_], wall[b_])
                    row["ratios"][f"{a_} / {b_}"] = dict(ratio=m_, ci95=ci)
            m_, ci = ratio_ci(wall[best["FR-stop"]], wall[best["ABF"]])
            row["ratios"][f"best FR ({best['FR-stop']}) / best ABF ({best['ABF']})"] = dict(ratio=m_, ci95=ci)
            row["verdict"] = ("best FR BEATS best ABF in wall-clock" if ci[1] < 1 else
                              "best ABF BEATS best FR in wall-clock" if ci[0] > 1 else "UNRESOLVED")
            res["per_P"][str(P)] = row
        out[mode] = res
        print(f"[{mode}] wall-clock time to accuracy (s), throughput {src}")
        for P in Ps + ["best"]:
            r = res["per_P"][str(P)]
            q = [v for k, v in r["ratios"].items() if k.startswith("best")][0]
            print(f"  P {str(P):>4s}: " + "  ".join(f"{k} {v['median_wall_s']:.1f}" for k, v in r["configs"].items()) +
                  f" | best ABF {r['best_allocation']['ABF']}, best FR {r['best_allocation']['FR-stop']}: "
                  f"{q['ratio']:.3f} [{q['ci95'][0]:.3f}, {q['ci95'][1]:.3f}] {r['verdict']}")
    if "replicate_run2_over_run1" in out:
        v = np.array(list(out["replicate_run2_over_run1"].values()))
        print(f"throughput replicate run2/run1: median {np.median(v):.3f}, range [{v.min():.3f}, {v.max():.3f}] over {len(v)} rows")
    p = os.path.join(AN, "E_wallclock_tta.json")
    json.dump(out, open(p, "w"), indent=1)
    print("wrote", p)


if __name__ == "__main__":
    main()
