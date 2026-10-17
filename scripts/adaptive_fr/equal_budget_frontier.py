"""PREREGISTRATION sections E.3 + E-A2 (frontier, online rule R1): two algorithms x four allocations at an equal
force-evaluation budget B = 5.12 x 10^8, ITE 300 K (same arithmetic as section A.5, equal_budget4.py).

    ABF / FR-R1  K1024 x 50 t.u.    block C1    (results/adaptive_fr_campaign/C1_ite300_K1024_R1, rate 2.0)
    ABF          K256  x 200 t.u.   EB          (results/convergence_fr_campaign/EB_ite300_abf_K256_T200)
    FR-R1        K256  x 200 t.u.   EA_R1       (results/adaptive_fr_campaign/EA_ite300_K256_T200_R1, rate 1.0)
    ABF / FR-R1  K64   x 800 t.u.   EB64        (results/adaptive_fr_campaign/EB64_ite300_K64_T800_R1, rate 0.2)
    ABF / FR-R1  K16   x 3200 t.u.  EB16        (results/adaptive_fr_campaign/EB16_ite300_K16_T3200_R1, amendment E-A2)

Time to accuracy in molecular force evaluations, B = K x steps (production) or K x (equil + steps) (all-in);
sustained criterion evaluated up to the COMMON budget (curves truncated there); restricted at that budget.
Per-seed summary = geometric mean over the five kT rungs of the restricted tau_B.  Cross-allocation statistics are
UNPAIRED: ratio of medians, 95 % CI from independent bootstrap of seeds within each configuration (10 000).
"Best tested allocation" of an algorithm = its allocation with the smaller median per-seed geometric-mean tau_B.

    ABFFR_DATA_ROOT=/home/zheyuanlai/ABF-Fisher-Rao python scripts/adaptive_fr/equal_budget_frontier.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "convergence_fr"))
import atlas_lib as L  # noqa: E402
import systems as S  # noqa: E402
from eri import metrics as mt  # noqa: E402

LADDER_KT = [0.080, 0.064, 0.048, 0.040, 0.032]
CONF = [
    dict(key="ABF_K1024", alg="ABF", K=1024, dir="results/adaptive_fr_campaign/C1_ite300_K1024_R1", pat="abf_N1024_s*.npz"),
    dict(key="FR_K1024", alg="FR-stop", K=1024, dir="results/adaptive_fr_campaign/C1_ite300_K1024_R1", pat="fr_N1024_s*.npz"),
    dict(key="ABF_K256", alg="ABF", K=256, dir="results/convergence_fr_campaign/EB_ite300_abf_K256_T200", pat="abf_N256_s*.npz"),
    dict(key="FR_K256", alg="FR-stop", K=256, dir="results/adaptive_fr_campaign/EA_ite300_K256_T200_R1", pat="fr_N256_s*.npz"),
    dict(key="ABF_K64", alg="ABF", K=64, dir="results/adaptive_fr_campaign/EB64_ite300_K64_T800_R1", pat="abf_N64_s*.npz"),
    dict(key="FR_K64", alg="FR-stop", K=64, dir="results/adaptive_fr_campaign/EB64_ite300_K64_T800_R1", pat="fr_N64_s*.npz"),
    dict(key="ABF_K16", alg="ABF", K=16, dir="results/adaptive_fr_campaign/EB16_ite300_K16_T3200_R1", pat="abf_N16_s*.npz"),
    dict(key="FR_K16", alg="FR-stop", K=16, dir="results/adaptive_fr_campaign/EB16_ite300_K16_T3200_R1", pat="fr_N16_s*.npz"),
]


def load(cf, Fr, Gr):
    files = S.glob_both(cf["dir"], cf["pat"])
    E, seeds = [], []
    for f in files:
        r = mt.load_run(f)
        c = r["cfg"]
        assert int(c["N"]) == cf["K"] and int(r["save_steps"][-1]) == int(c["n_steps"])
        E.append(mt.errors(r, Fr, Gr)["eF"])
        seeds.append(int(c["seed"]))
        steps = r["save_steps"].astype(float)
        eq = int(c["equil_steps"])
    return dict(eF=np.array(E).T, steps=steps, eq=eq, seeds=seeds, K=cf["K"])


def tau_B(d, eps, budget, allin):
    """Sustained tau in force evaluations, evaluated on the curve truncated at `budget` (inf if not sustained)."""
    off = d["K"] * d["eq"] if allin else 0.0
    B = d["K"] * d["steps"] + off
    sel = B <= budget * (1 + 1e-12)
    return L.tau(d["eF"][sel], B[sel], eps, "sustained"), float(B[sel][-1])


def ratio_ci(a, b, n=10000, seed=2026):
    rng = np.random.default_rng(seed)
    ma = np.median(a[rng.integers(0, len(a), (n, len(a)))], 1)
    mb = np.median(b[rng.integers(0, len(b), (n, len(b)))], 1)
    return float(np.median(a) / np.median(b)), [float(np.quantile(ma / mb, 0.025)), float(np.quantile(ma / mb, 0.975))]


def main():
    mt.set_reference_base(S.dpath("results/overnight/ite/reference"))
    Fr, Gr = mt.reference_for_grid(180, "ite_itq3")
    kT = S.KB * 300.0
    data = {cf["key"]: load(cf, Fr, Gr) for cf in CONF}
    for k, d in data.items():
        assert len(d["seeds"]) == 16, (k, len(d["seeds"]))
    out = dict(definition=__doc__, configs={cf["key"]: dict(alg=cf["alg"], K=cf["K"], seeds=data[cf["key"]]["seeds"]) for cf in CONF})
    for mode, allin in (("production", False), ("all_in_common", True)):
        budget = min((d["K"] * (d["steps"][-1] + (d["eq"] if allin else 0))) for d in data.values())
        res = dict(budget=budget, rungs=[], geo={})
        geo = {k: np.zeros(16) for k in data}
        for ek in LADDER_KT:
            eps = ek * kT
            row = dict(eps_kJmol=eps)
            for k, d in data.items():
                t, Bend = tau_B(d, eps, budget, allin)
                tr = np.minimum(t, Bend) / 1e8
                geo[k] += np.log(tr) / len(LADDER_KT)
                row[k] = dict(reached=int(np.isfinite(t).sum()), median_restricted_1e8=float(np.median(tr)),
                              restricted_mean_1e8=float(np.mean(tr)), per_seed=tr.tolist())
            res["rungs"].append(row)
        for k in data:
            res["geo"][k] = dict(median_1e8=float(np.exp(np.median(geo[k]))), per_seed=np.exp(geo[k]).tolist())
        best = {}
        for alg in ("ABF", "FR-stop"):
            ks = [cf["key"] for cf in CONF if cf["alg"] == alg]
            best[alg] = min(ks, key=lambda k: np.median(geo[k]))
        res["best_allocation"] = best
        pairs = [("FR_K16", "ABF_K16"), ("FR_K64", "ABF_K64"), ("FR_K256", "ABF_K256"), ("FR_K1024", "ABF_K1024"),
                 ("ABF_K16", "ABF_K64"), ("ABF_K64", "ABF_K256"), ("ABF_K256", "ABF_K1024"),
                 ("FR_K16", "FR_K64"), ("FR_K64", "FR_K256"), ("FR_K256", "FR_K1024"), (best["FR-stop"], best["ABF"])]
        pairs = list(dict.fromkeys(pairs))
        res["ratios_geo_tauB"] = {}
        for a_, b_ in pairs:
            m, ci = ratio_ci(np.exp(geo[a_]), np.exp(geo[b_]))
            res["ratios_geo_tauB"][f"{a_} / {b_}"] = dict(ratio=m, ci95=ci,
                                                          note="unpaired independent bootstrap" if a_.split('_')[1] != b_.split('_')[1] else "same K: paired data, shown unpaired here (the paired analysis is analyze_block)")
        fin = {}
        for k, d in data.items():
            off = d["K"] * d["eq"] if allin else 0.0
            B = d["K"] * d["steps"] + off
            i = int(np.searchsorted(B, budget * (1 + 1e-12), side="right")) - 1
            fin[k] = d["eF"][i]
            res.setdefault("eF_at_budget", {})[k] = dict(median=float(np.median(d["eF"][i])), B_at=float(B[i]))
        m, ci = ratio_ci(fin[best["FR-stop"]], fin[best["ABF"]])
        res["eF_at_budget_best_FR_over_best_ABF"] = dict(ratio=m, ci95=ci)
        q = res["ratios_geo_tauB"][f"{best['FR-stop']} / {best['ABF']}"]
        res["question_B_verdict"] = ("best FR allocation BEATS best ABF allocation" if q["ci95"][1] < 1 else
                                     "best ABF allocation BEATS best FR allocation" if q["ci95"][0] > 1 else "UNRESOLVED")
        out[mode] = res
        print(f"[{mode}] budget {budget:.4g}; best ABF {best['ABF']}, best FR {best['FR-stop']}")
        for k in data:
            print(f"   {k:10s} median geo tau_B {res['geo'][k]['median_1e8']:.3f} e8 | " +
                  " ".join(f"{r_['eps_kJmol']:.3f}:{r_[k]['median_restricted_1e8']:.2f}({r_[k]['reached']})" for r_ in res["rungs"]))
        for kk, v in res["ratios_geo_tauB"].items():
            print(f"   {kk:24s} {v['ratio']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]")
        print("   e_F at budget:", {k: round(v['median'], 4) for k, v in res['eF_at_budget'].items()}, "| best FR / best ABF",
              round(res['eF_at_budget_best_FR_over_best_ABF']['ratio'], 3), [round(x, 3) for x in res['eF_at_budget_best_FR_over_best_ABF']['ci95']])
        print("   QUESTION B:", res["question_B_verdict"])
    p = os.path.join(S.ROOT, "results", "adaptive_fr_campaign", "analysis", "E_frontier.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(out, open(p, "w"), indent=1)
    np.savez(p.replace(".json", "_curves.npz"), **{f"eF_{k}": d["eF"] for k, d in data.items()},
             **{f"steps_{k}": d["steps"] for k, d in data.items()})
    print("wrote", p)


if __name__ == "__main__":
    main()
