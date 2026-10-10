import sys, glob, numpy as np
from scipy import stats
sys.path.insert(0, "/home/zheyuanlai/ABF-Fisher-Rao/src")
import lta_ladder_numba as E
ROOT = "/home/zheyuanlai/ABF-Fisher-Rao"

def score(r, F):
    use_prod = r["C_prod"].sum(-1) > 0
    mf = np.where(use_prod[:, None], E.mean_force(r["M_prod"], r["C_prod"]), E.mean_force(r["M_all"], r["C_all"]))
    d = E.histogram_pmf(mf) - F[None]; d -= d.mean(-1, keepdims=True)
    e = np.sqrt((d ** 2).mean(-1))
    return e[-1], np.trapezoid(e, r["save_t"])

for T in (300, 150):
    F = np.load(f"{ROOT}/results/uniform_campaign/lta/reference/reference_T{T}.npz", allow_pickle=True)["F"]
    prod = {}
    for m in ("abf", "fr_uniform"):
        P = np.load(f"{ROOT}/results/lta_histogram/production_T{T}/{m}.npz", allow_pickle=True)
        d = P["pmf"] - F[None, None]; d -= d.mean(-1, keepdims=True)
        e = np.sqrt((d ** 2).mean(-1))
        prod[m] = dict(eT=e[-1], IF=np.trapezoid(e, P["times"], axis=0), ev=P["total_replacement_events"].astype(float),
                       cx=P["n_cage_crossings"] / 1024.0,
                       ess=(np.nanmin(P["ancestor_ess"][P["steps"] >= 20000], axis=0) / 1024 if m != "abf" else None))
    seeds = sorted(int(p.split("_s")[1].split(".")[0]) for p in glob.glob(f"T{T}_abf_s*.npz") if "ckpt" not in p)
    seeds = [s for s in seeds if glob.glob(f"T{T}_fr_s{s}.npz")]
    mine = {m: dict(eT=[], IF=[], ev=[], cx=[], ess=[]) for m in ("abf", "fr")}
    for s in seeds:
        for m in ("abf", "fr"):
            r = E.load_result(f"T{T}_{m}_s{s}.npz")
            eT, IF = score(r, F)
            mine[m]["eT"].append(eT); mine[m]["IF"].append(IF); mine[m]["ev"].append(float(r["total_replacement_events"]))
            mine[m]["cx"].append(r["n_cage_crossings_lineage"][-1] / 1024.0)
            if m == "fr":
                mine[m]["ess"].append(np.nanmin(r["gen_ess"][r["save_step"] >= 20000]))
    print(f"=== T {T} K: numba seeds {seeds} (n={len(seeds)}) vs production 16 seeds  [median (min..max); Mann-Whitney p]")
    for m, pm in (("abf", "abf"), ("fr", "fr_uniform")):
        for k in ("eT", "IF", "ev", "cx", "ess"):
            a = np.array(mine[m][k]); b = prod[pm][k]
            if b is None or a.size == 0: continue
            p = stats.mannwhitneyu(a, b).pvalue
            print(f"  {m:3s} {k:3s}: numba {np.median(a):8.4f} ({a.min():.4f}..{a.max():.4f}) | prod {np.median(b):8.4f} ({b.min():.4f}..{b.max():.4f}) | p {p:.3f}")
    ca = 100 * (np.array(mine["fr"]["IF"]) - np.array(mine["abf"]["IF"])) / np.array(mine["abf"]["IF"])
    cp = 100 * (prod["fr_uniform"]["IF"] - prod["abf"]["IF"]) / prod["abf"]["IF"]
    print(f"  paired dI_F %: numba median {np.median(ca):.1f} {np.round(np.sort(ca), 1)} | prod median {np.median(cp):.1f} "
          f"({cp.min():.1f}..{cp.max():.1f}) | MW p {stats.mannwhitneyu(ca, cp).pvalue:.3f}")
