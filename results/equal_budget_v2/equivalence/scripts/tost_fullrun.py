"""Audit 0A section 8C equivalence gate (TOST on bootstrap CI of the median ratio numba/production) applied to the
implementer's 32-seed full-knob numba runs vs the 16-seed CUDA production."""
import sys, glob, numpy as np
sys.path.insert(0, "/home/zheyuanlai/ABF-Fisher-Rao/src")
import lta_ladder_numba as E
ROOT = "/home/zheyuanlai/ABF-Fisher-Rao"
D = "/tmp/claude-1008/-home-zheyuanlai-ABF-Fisher-Rao/b25e8a63-d0d8-4935-af2b-7d50d5f3ea3b/scratchpad/engines/fullrun"
rng = np.random.default_rng(20261010)
def boot_ratio(a, b, B=10000, stat=np.median):
    ia = rng.integers(0, a.size, (B, a.size)); ib = rng.integers(0, b.size, (B, b.size))
    r = stat(a[ia], axis=1) / stat(b[ib], axis=1)
    return stat(a) / stat(b), np.quantile(r, [0.05, 0.95])
MARG = dict(IF=0.05, ev=0.05, cx=0.05, eT=0.15)
for T in (300, 150):
    F = np.load(f"{ROOT}/results/uniform_campaign/lta/reference/reference_T{T}.npz", allow_pickle=True)["F"]
    IFs = {}
    for m, pm in (("abf", "abf"), ("fr", "fr_uniform")):
        P = np.load(f"{ROOT}/results/lta_histogram/production_T{T}/{pm}.npz", allow_pickle=True)
        d = P["pmf"] - F[None, None]; d -= d.mean(-1, keepdims=True)
        e = np.sqrt((d ** 2).mean(-1))
        prod = dict(eT=e[-1], IF=np.trapezoid(e, P["times"], axis=0), cx=P["n_cage_crossings"] / 1024.0,
                    ev=P["total_replacement_events"].astype(float))
        rows = {k: [] for k in prod}
        for s in range(1, 33):
            r = E.load_result(f"{D}/T{T}_{m}_s{s}.npz")
            use_prod = r["C_prod"].sum(-1) > 0
            mf = np.where(use_prod[:, None], E.mean_force(r["M_prod"], r["C_prod"]), E.mean_force(r["M_all"], r["C_all"]))
            dd = E.histogram_pmf(mf) - F[None]; dd -= dd.mean(-1, keepdims=True)
            ee = np.sqrt((dd ** 2).mean(-1))
            rows["eT"].append(ee[-1]); rows["IF"].append(np.trapezoid(ee, r["save_t"]))
            rows["cx"].append(r["n_cage_crossings_lineage"][-1] / 1024.0); rows["ev"].append(float(r["total_replacement_events"]))
        IFs[m] = (np.array(rows["IF"]), prod["IF"])
        for k in ("IF", "eT", "cx", "ev"):
            if k == "ev" and m == "abf":
                continue
            a = np.array(rows[k]); b = np.asarray(prod[k], float)
            est, (lo, hi) = boot_ratio(a, b)
            mg = MARG[k]
            ok = (lo > 1 - mg) and (hi < 1 + mg)
            print(f"T {T} {m:3s} {k:3s}: median ratio {est:.4f} 90% CI [{lo:.4f}, {hi:.4f}] margin +-{mg:.0%} -> {'EQUIVALENT' if ok else 'NOT SHOWN'}")
    ca = 100 * (IFs["fr"][0] - IFs["abf"][0]) / IFs["abf"][0]
    print(f"T {T} paired dI_F numba median {np.median(ca):.2f} %")
