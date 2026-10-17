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

Selection-aware ("nested") bootstrap (added 2026-10-17 after the reviewer's point that the best K is selected on the
data that scores it while the fixed-selection CI holds the chosen configurations fixed): in each of the 10 000
replicates the seeds are resampled within EVERY configuration (independently, as above), the best K of each algorithm
is RE-SELECTED on the resampled medians, and best FR / best ABF is taken.  Reported next to the fixed-selection CI
under the new keys `nested_bootstrap` (ratio CI, the selection frequencies per algorithm, the fraction of replicates
that re-select the fixed choice) and `question_B_verdict_nested`.  Every pre-existing key is computed as before.

Amendment E-A3 (frontier bracket below K16): when the blocks EB8 (K8 x T6400, both arms), EB4 (K4 x T12800, ABF) and
EB1 (K1 x T51200, ABF) are present, the same analysis is ALSO run on the extended configuration set (K1..K1024) and
written to E_frontier_ext.json; E_frontier.json keeps the eight-configuration set of E.3 + E-A2.  A post-hoc
sensitivity variant, E_frontier_ext_matched.json, reads every configuration on the 1.28 x 10^6-force-evaluation
save grid of the earlier cells (the E-A3 cells, saved every 1 t.u., are subsampled), because the sustained criterion
is stricter on a finer grid while tau is rounded up to the next save on a coarser one.

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
# amendment E-A3: the bracket below K16 (K8 both arms; K4 and K1 ABF only, see PREREGISTRATION.md E-A3)
CONF_EXT = CONF + [
    dict(key="ABF_K8", alg="ABF", K=8, dir="results/adaptive_fr_campaign/EB8_ite300_K8_T6400_R1", pat="abf_N8_s*.npz"),
    dict(key="FR_K8", alg="FR-stop", K=8, dir="results/adaptive_fr_campaign/EB8_ite300_K8_T6400_R1", pat="fr_N8_s*.npz"),
    dict(key="ABF_K4", alg="ABF", K=4, dir="results/adaptive_fr_campaign/EB4_ite300_K4_T12800_abf", pat="abf_N4_s*.npz"),
    dict(key="ABF_K1", alg="ABF", K=1, dir="results/adaptive_fr_campaign/EB1_ite300_K1_T51200_abf", pat="abf_N1_s*.npz"),
]
N_BOOT, BOOT_SEED = 10000, 2026
# the E.3 + E-A2 definition text exactly as published (6716aefc): E_frontier.json keeps it under `definition`; the
# amendment paragraphs of this docstring go under `definition_amendments` (E_frontier_ext*.json carry the full docstring)
DEFINITION_E3 = 'PREREGISTRATION sections E.3 + E-A2 (frontier, online rule R1): two algorithms x four allocations at an equal\nforce-evaluation budget B = 5.12 x 10^8, ITE 300 K (same arithmetic as section A.5, equal_budget4.py).\n\n    ABF / FR-R1  K1024 x 50 t.u.    block C1    (results/adaptive_fr_campaign/C1_ite300_K1024_R1, rate 2.0)\n    ABF          K256  x 200 t.u.   EB          (results/convergence_fr_campaign/EB_ite300_abf_K256_T200)\n    FR-R1        K256  x 200 t.u.   EA_R1       (results/adaptive_fr_campaign/EA_ite300_K256_T200_R1, rate 1.0)\n    ABF / FR-R1  K64   x 800 t.u.   EB64        (results/adaptive_fr_campaign/EB64_ite300_K64_T800_R1, rate 0.2)\n    ABF / FR-R1  K16   x 3200 t.u.  EB16        (results/adaptive_fr_campaign/EB16_ite300_K16_T3200_R1, amendment E-A2)\n\nTime to accuracy in molecular force evaluations, B = K x steps (production) or K x (equil + steps) (all-in);\nsustained criterion evaluated up to the COMMON budget (curves truncated there); restricted at that budget.\nPer-seed summary = geometric mean over the five kT rungs of the restricted tau_B.  Cross-allocation statistics are\nUNPAIRED: ratio of medians, 95 % CI from independent bootstrap of seeds within each configuration (10 000).\n"Best tested allocation" of an algorithm = its allocation with the smaller median per-seed geometric-mean tau_B.\n\n    ABFFR_DATA_ROOT=/home/zheyuanlai/ABF-Fisher-Rao python scripts/adaptive_fr/equal_budget_frontier.py\n'


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


def ratio_ci(a, b, n=N_BOOT, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    ma = np.median(a[rng.integers(0, len(a), (n, len(a)))], 1)
    mb = np.median(b[rng.integers(0, len(b), (n, len(b)))], 1)
    return float(np.median(a) / np.median(b)), [float(np.quantile(ma / mb, 0.025)), float(np.quantile(ma / mb, 0.975))]


def nested_best(conf, geo, fin, best_fixed, n=N_BOOT, seed=BOOT_SEED):
    """Selection-aware bootstrap of best FR / best ABF.  Per replicate: resample the 16 seeds of every configuration
    independently (one index draw per configuration, shared by its tau and e_F arrays), re-select the best K per
    algorithm by the smaller resampled median geometric-mean tau_B, and take the ratio of the selected medians
    (geo tau_B; and e_F at the budget of the same selection)."""
    rng = np.random.default_rng(seed)
    keys = [cf["key"] for cf in conf]
    med_tau, med_fin = {}, {}
    for k in keys:
        x = np.exp(geo[k])
        idx = rng.integers(0, len(x), (n, len(x)))
        med_tau[k] = np.median(x[idx], 1)
        med_fin[k] = np.median(np.asarray(fin[k])[idx], 1)
    out = dict(n_replicates=n, seed=seed, selection_frequency={}, selected_median_1e8={})
    sel, best_tau, best_fin = {}, {}, {}
    for alg in ("ABF", "FR-stop"):
        ks = [cf["key"] for cf in conf if cf["alg"] == alg]
        M = np.stack([med_tau[k] for k in ks], 1)
        i = np.argmin(M, 1)
        sel[alg] = np.array(ks)[i]
        best_tau[alg] = M[np.arange(n), i]
        best_fin[alg] = np.stack([med_fin[k] for k in ks], 1)[np.arange(n), i]
        out["selection_frequency"][alg] = {k: float(np.mean(sel[alg] == k)) for k in ks}
    r = best_tau["FR-stop"] / best_tau["ABF"]
    rf = best_fin["FR-stop"] / best_fin["ABF"]
    same = (sel["ABF"] == best_fixed["ABF"]) & (sel["FR-stop"] == best_fixed["FR-stop"])
    out.update(
        geo_tauB_best_FR_over_best_ABF=dict(
            fixed_selection_ratio=float(np.median(np.exp(geo[best_fixed["FR-stop"]])) / np.median(np.exp(geo[best_fixed["ABF"]]))),
            bootstrap_median_ratio=float(np.median(r)), ci95=[float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975))],
            ci99=[float(np.quantile(r, 0.005)), float(np.quantile(r, 0.995))],
            p_ratio_below_1=float(np.mean(r < 1.0))),
        eF_at_budget_best_FR_over_best_ABF=dict(
            bootstrap_median_ratio=float(np.median(rf)), ci95=[float(np.quantile(rf, 0.025)), float(np.quantile(rf, 0.975))]),
        fraction_replicates_reselecting_fixed_choice=float(np.mean(same)),
        selection_pair_frequency={f"{a} / {b}": float(np.mean((sel["FR-stop"] == a) & (sel["ABF"] == b)))
                                  for a in dict.fromkeys(sel["FR-stop"].tolist()) for b in dict.fromkeys(sel["ABF"].tolist())
                                  if np.any((sel["FR-stop"] == a) & (sel["ABF"] == b))},
        note="unpaired independent bootstrap within every configuration; the best K per algorithm is re-selected in each "
             "replicate, so the CI includes the uncertainty of the selection that the fixed-selection CI omits")
    return out


def pair_list(conf, best):
    """Same-K FR / ABF (ascending K), adjacent-K ratios within each algorithm (ascending K), best FR / best ABF."""
    Ks = sorted({cf["K"] for cf in conf})
    have = {cf["key"] for cf in conf}
    pairs = [(f"FR_K{K}", f"ABF_K{K}") for K in Ks if f"FR_K{K}" in have and f"ABF_K{K}" in have]
    for alg in ("ABF", "FR"):
        ks = [K for K in Ks if f"{alg}_K{K}" in have]
        pairs += [(f"{alg}_K{a}", f"{alg}_K{b}") for a, b in zip(ks[:-1], ks[1:])]
    for alg in ("ABF", "FR"):                                   # E-A3 verdict rule: every K < 16 against K16
        if f"{alg}_K16" in have:
            pairs += [(f"{alg}_K{K}", f"{alg}_K16") for K in Ks if K < 16 and f"{alg}_K{K}" in have]
    pairs.append((best["FR-stop"], best["ABF"]))
    return list(dict.fromkeys(pairs))


def coarsen(d, fe_per_save):
    """Keep only the saves that fall on a grid of `fe_per_save` production force evaluations (plus step 0), so a cell
    saved more often is READ at the resolution of the cells saved every 1.28 x 10^6 (E-A3 sensitivity check:
    the sustained criterion is stricter on a finer grid, and tau is rounded up to the next save on a coarser one)."""
    B = d["K"] * d["steps"]
    keep = np.isclose(np.mod(B, fe_per_save), 0) | np.isclose(np.mod(B, fe_per_save), fe_per_save)
    return dict(d, eF=d["eF"][keep], steps=d["steps"][keep])


def analyse(conf, name, Fr, Gr, kT, coarsen_to=None):
    data = {cf["key"]: load(cf, Fr, Gr) for cf in conf}
    if coarsen_to:
        data = {k: coarsen(d, coarsen_to) for k, d in data.items()}
    for k, d in data.items():
        assert len(d["seeds"]) == 16, (k, len(d["seeds"]))
    full = name != "E_frontier"
    out = dict(definition=__doc__ if full else DEFINITION_E3,
               configs={cf["key"]: dict(alg=cf["alg"], K=cf["K"], seeds=data[cf["key"]]["seeds"]) for cf in conf})
    if coarsen_to:
        out["matched_save_grid_force_evals"] = float(coarsen_to)
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
            ks = [cf["key"] for cf in conf if cf["alg"] == alg]
            best[alg] = min(ks, key=lambda k: np.median(geo[k]))
        res["best_allocation"] = best
        pairs = pair_list(conf, best)
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
        # --- selection-aware (nested) bootstrap: new keys only ---
        nb = nested_best(conf, geo, fin, best)
        res["nested_bootstrap"] = nb
        qn = nb["geo_tauB_best_FR_over_best_ABF"]
        res["question_B_verdict_nested"] = ("best FR allocation BEATS best ABF allocation" if qn["ci95"][1] < 1 else
                                            "best ABF allocation BEATS best FR allocation" if qn["ci95"][0] > 1 else "UNRESOLVED")
        res["save_resolution_force_evals"] = {k: float(d["K"] * np.median(np.diff(d["steps"]))) for k, d in data.items()}
        out[mode] = res
        if not full:
            out["definition_amendments"] = __doc__[len(DEFINITION_E3.rstrip()):].strip()
        print(f"[{name} {mode}] budget {budget:.4g}; best ABF {best['ABF']}, best FR {best['FR-stop']}")
        for k in data:
            print(f"   {k:10s} median geo tau_B {res['geo'][k]['median_1e8']:.3f} e8 | " +
                  " ".join(f"{r_['eps_kJmol']:.3f}:{r_[k]['median_restricted_1e8']:.2f}({r_[k]['reached']})" for r_ in res["rungs"]))
        for kk, v in res["ratios_geo_tauB"].items():
            print(f"   {kk:24s} {v['ratio']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]")
        print("   e_F at budget:", {k: round(v['median'], 4) for k, v in res['eF_at_budget'].items()}, "| best FR / best ABF",
              round(res['eF_at_budget_best_FR_over_best_ABF']['ratio'], 3), [round(x, 3) for x in res['eF_at_budget_best_FR_over_best_ABF']['ci95']])
        print("   QUESTION B (fixed selection):", res["question_B_verdict"])
        print(f"   nested bootstrap: best FR / best ABF {qn['fixed_selection_ratio']:.3f} (boot median {qn['bootstrap_median_ratio']:.3f}) "
              f"95 % [{qn['ci95'][0]:.3f}, {qn['ci95'][1]:.3f}] 99 % [{qn['ci99'][0]:.3f}, {qn['ci99'][1]:.3f}]; "
              f"P(ratio<1) {qn['p_ratio_below_1']:.3f}; fixed choice re-selected in {nb['fraction_replicates_reselecting_fixed_choice']:.3f}")
        for alg, fq in nb["selection_frequency"].items():
            print(f"      selected {alg:8s}: " + ", ".join(f"{k} {v:.3f}" for k, v in fq.items()))
        print("   QUESTION B (nested):", res["question_B_verdict_nested"])
    p = os.path.join(S.ROOT, "results", "adaptive_fr_campaign", "analysis", f"{name}.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(out, open(p, "w"), indent=1)
    np.savez(p.replace(".json", "_curves.npz"), **{f"eF_{k}": d["eF"] for k, d in data.items()},
             **{f"steps_{k}": d["steps"] for k, d in data.items()})
    print("wrote", p)


def ext_available():
    return all(len(S.glob_both(cf["dir"], cf["pat"])) == 16 for cf in CONF_EXT[len(CONF):])


def main():
    mt.set_reference_base(S.dpath("results/overnight/ite/reference"))
    Fr, Gr = mt.reference_for_grid(180, "ite_itq3")
    kT = S.KB * 300.0
    analyse(CONF, "E_frontier", Fr, Gr, kT)
    if ext_available():
        analyse(CONF_EXT, "E_frontier_ext", Fr, Gr, kT)
        # sensitivity (post hoc, descriptive): every configuration read on the 1.28 x 10^6-force-evaluation save grid
        # of the E.3 / E-A2 cells (the E-A3 cells are subsampled: every 16th save at K8, 32nd at K4, 128th at K1)
        analyse(CONF_EXT, "E_frontier_ext_matched", Fr, Gr, kT, coarsen_to=1.28e6)
    else:
        print("E-A3 blocks (EB8 / EB4 / EB1) not complete: E_frontier_ext.json not written")


if __name__ == "__main__":
    main()
