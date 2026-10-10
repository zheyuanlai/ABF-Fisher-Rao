import sys, glob, numpy as np
from scipy import stats
sys.path.insert(0, "/home/zheyuanlai/ABF-Fisher-Rao/src")
import lta_ladder_numba as E
ROOT = "/home/zheyuanlai/ABF-Fisher-Rao"
g = -np.pi + (np.arange(180) + 0.5) * 2 * np.pi / 180
cage = np.abs(g) * 11.919 / (2 * np.pi) > 4

def stats_of_numba(r, F, m):
    use_prod = r["C_prod"].sum(-1) > 0
    mf = np.where(use_prod[:, None], E.mean_force(r["M_prod"], r["C_prod"]), E.mean_force(r["M_all"], r["C_all"]))
    d = E.histogram_pmf(mf) - F[None]; d -= d.mean(-1, keepdims=True)
    e = np.sqrt((d ** 2).mean(-1))
    o = dict(eT=e[-1], IF=np.trapezoid(e, r["save_t"]), cx=r["n_cage_crossings_lineage"][-1] / 1024.0,
             trans=r["n_transitions"][-1] / 1024.0, fwin=r["region_frac"][7:, 2].mean())
    if m == "fr":
        o.update(ev=float(r["total_replacement_events"]), ess=np.nanmin(r["gen_ess"][r["save_step"] >= 20000]),
                 dcage=r["death_hist"][cage].sum() / r["death_hist"].sum())
    return o

for T in (300, 150):
    F = np.load(f"{ROOT}/results/uniform_campaign/lta/reference/reference_T{T}.npz", allow_pickle=True)["F"]
    seeds = [s for s in range(1, 33) if glob.glob(f"T{T}_abf_s{s}.npz") and glob.glob(f"T{T}_fr_s{s}.npz")]
    print(f"=== T {T} K: numba n={len(seeds)} vs production n=16; mean +- SE; Welch p")
    IFs = {}
    for m, pm in (("abf", "abf"), ("fr", "fr_uniform")):
        P = np.load(f"{ROOT}/results/lta_histogram/production_T{T}/{pm}.npz", allow_pickle=True)
        d = P["pmf"] - F[None, None]; d -= d.mean(-1, keepdims=True)
        e = np.sqrt((d ** 2).mean(-1))
        prod = dict(eT=e[-1], IF=np.trapezoid(e, P["times"], axis=0), cx=P["n_cage_crossings"] / 1024.0,
                    trans=P["n_transitions"] / 1024.0, fwin=P["frac_window"][7:].mean(0))
        if m == "fr":
            prod.update(ev=P["total_replacement_events"].astype(float),
                        ess=np.nanmin(P["ancestor_ess"][P["steps"] >= 20000], axis=0) / 1024,
                        dcage=P["death_hist"][:, cage].sum(1) / P["death_hist"].sum(1))
        rows = [stats_of_numba(E.load_result(f"T{T}_{m}_s{s}.npz"), F, m) for s in seeds]
        IFs[m] = (np.array([r["IF"] for r in rows]), prod["IF"])
        for k in prod:
            a = np.array([r[k] for r in rows]); b = np.asarray(prod[k], float)
            p = stats.ttest_ind(a, b, equal_var=False).pvalue
            a1, a2 = a[:16], a[16:]
            print(f"  {m:3s} {k:5s}: numba {a.mean():9.4f}+-{a.std(ddof=1)/np.sqrt(a.size):.4f} (s1-16 {a1.mean():.4f}, s17-32 {a2.mean() if a2.size else float('nan'):.4f}) | prod {b.mean():9.4f}+-{b.std(ddof=1)/4:.4f} | p {p:.3f}")
    ca = 100 * (IFs["fr"][0] - IFs["abf"][0]) / IFs["abf"][0]; cp = 100 * (IFs["fr"][1] - IFs["abf"][1]) / IFs["abf"][1]
    print(f"  paired dI_F %: numba median {np.median(ca):.2f} (s1-16 {np.median(ca[:16]):.2f}, s17-32 {np.median(ca[16:]) if ca.size > 16 else float('nan'):.2f}) | prod median {np.median(cp):.2f} | MW p {stats.mannwhitneyu(ca, cp).pvalue:.3f}")
