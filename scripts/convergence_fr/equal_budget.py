"""Amendment A2, block EB (descriptive): time to accuracy in MOLECULAR FORCE EVALUATIONS, B = K x steps, for
    ABF   K 256  x 2 000 000 steps (EB, 16 seeds)
    ABF   K 1024 x   500 000 steps (E1 ABF arm, 16 seeds)
    ABF + FR stopped at 10 t.u., K 1024 x 500 000 steps (E1 FR arm, the same 16 seeds as E1 ABF)
All three spend the same total budget (5.12e8 force evaluations; equilibration excluded and reported separately).
Allocations are UNPAIRED (different seeds / K): medians of restricted sustained tau_B per allocation and the ratio of
medians with an independent bootstrap (10 000) over seeds within each allocation.

    python scripts/convergence_fr/equal_budget.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import atlas_lib as L  # noqa: E402
import systems as S  # noqa: E402

LADDER_KT = [0.080, 0.064, 0.048, 0.040, 0.032]


def ratio_ci(a, b, n=10000, seed=2026):
    rng = np.random.default_rng(seed)
    ma = np.median(a[rng.integers(0, len(a), (n, len(a)))], 1)
    mb = np.median(b[rng.integers(0, len(b), (n, len(b)))], 1)
    r = ma / mb
    return float(np.median(a) / np.median(b)), [float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975))]


def main():
    ref = "results/overnight/ite/reference"
    eb = S.load_eri_cell("results/convergence_fr_campaign/EB_ite300_abf_K256_T200", 256, ref, "ite_itq3", arms=("abf",))
    e1 = S.load_eri_cell("results/convergence_fr_campaign/E1_ite300_stop10", 1024, ref, "ite_itq3")
    kT = e1["kT"]
    alloc = {
        "ABF K256 x T200 (EB)": (eb["t"] / eb["dt"] * eb["K"], eb["arms"]["abf"]["eF"]),
        "ABF K1024 x T50 (E1 ABF arm)": (e1["t"] / e1["dt"] * e1["K"], e1["arms"]["abf"]["eF"]),
        "ABF+FR stop 10, K1024 x T50 (E1 FR arm)": (e1["t"] / e1["dt"] * e1["K"], e1["arms"]["fr"]["eF"]),
    }
    Bmax = min(v[0][-1] for v in alloc.values())
    out = dict(budget_force_evals=float(Bmax), note="tau in units of 1e8 molecular force evaluations; censored at the budget",
               equilibration_force_evals={"EB": 256 * 50000, "E1": 1024 * 50000}, rungs=[])
    keys = list(alloc)
    for ek in LADDER_KT:
        eps = ek * kT
        row = dict(eps_kJmol=eps, eps_kT=ek)
        taus = {}
        for k in keys:
            B, E = alloc[k]
            tau = L.tau(E, B, eps, "sustained") / 1e8
            taus[k] = np.minimum(tau, Bmax / 1e8)
            row[k] = dict(reached=int(np.isfinite(tau).sum()), n=int(len(tau)),
                          median_restricted=float(np.median(taus[k])))
        for a, b in ((keys[0], keys[2]), (keys[1], keys[2]), (keys[0], keys[1])):
            m, ci = ratio_ci(taus[a], taus[b])
            row[f"median ratio [{a}] / [{b}]"] = dict(ratio=m, ci95=ci)
        out["rungs"].append(row)
        print(f"eps {eps:.3f}: " + "  ".join(f"{k.split(' (')[0]}: {row[k]['median_restricted']:.2f} ({row[k]['reached']}/{row[k]['n']})" for k in keys)
              + f"  | EB/FRstop {row[f'median ratio [{keys[0]}] / [{keys[2]}]']['ratio']:.2f} {[round(x, 2) for x in row[f'median ratio [{keys[0]}] / [{keys[2]}]']['ci95']]}"
              + f"  EB/E1ABF {row[f'median ratio [{keys[0]}] / [{keys[1]}]']['ratio']:.2f} {[round(x, 2) for x in row[f'median ratio [{keys[0]}] / [{keys[1]}]']['ci95']]}")
    fin = {}
    for k in keys:
        B, E = alloc[k]
        i = int(np.searchsorted(B, Bmax - 1e-6))
        fin[k] = E[i]
        out.setdefault("eF_at_budget", {})[k] = dict(median=float(np.median(E[i])), q25=float(np.quantile(E[i], 0.25)),
                                                    q75=float(np.quantile(E[i], 0.75)))
    for a, b in ((keys[0], keys[2]), (keys[0], keys[1])):
        m, ci = ratio_ci(fin[a], fin[b])
        out[f"eF_at_budget ratio [{a}] / [{b}]"] = dict(ratio=m, ci95=ci)
    print("e_F at the full budget:", {k.split(' (')[0]: round(v['median'], 4) for k, v in out["eF_at_budget"].items()})
    print({k: (round(v["ratio"], 3), [round(x, 3) for x in v["ci95"]]) for k, v in out.items() if k.startswith("eF_at_budget ratio")})
    p = os.path.join(S.ROOT, "results", "convergence_fr_campaign", "analysis", "EB_equal_budget.json")
    json.dump(out, open(p, "w"), indent=1)
    np.savez(p.replace(".json", "_curves.npz"), **{f"B_{i}": alloc[k][0] for i, k in enumerate(keys)},
             **{f"eF_{i}": alloc[k][1] for i, k in enumerate(keys)}, labels=np.array(keys))
    print("wrote", p)


if __name__ == "__main__":
    main()
