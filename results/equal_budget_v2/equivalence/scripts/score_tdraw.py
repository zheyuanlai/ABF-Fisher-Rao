import sys, numpy as np
from scipy import stats
sys.path.insert(0, "/home/zheyuanlai/ABF-Fisher-Rao/src")
import lta_ladder_numba as E
ROOT = "/home/zheyuanlai/ABF-Fisher-Rao"
FULL = "/tmp/claude-1008/-home-zheyuanlai-ABF-Fisher-Rao/b25e8a63-d0d8-4935-af2b-7d50d5f3ea3b/scratchpad/engines/fullrun"
F = np.load(f"{ROOT}/results/uniform_campaign/lta/reference/reference_T300.npz", allow_pickle=True)["F"]
def numba_stats(r):
    use_prod = r["C_prod"].sum(-1) > 0
    mf = np.where(use_prod[:, None], E.mean_force(r["M_prod"], r["C_prod"]), E.mean_force(r["M_all"], r["C_all"]))
    d = E.histogram_pmf(mf) - F[None]; d -= d.mean(-1, keepdims=True)
    e = np.sqrt((d ** 2).mean(-1))
    assert r["save_step"][7] == 21000
    return dict(fwin=r["region_frac"][7:, 2].mean(), IF=np.trapezoid(e, r["save_t"]), eT=e[-1],
                cx=r["n_cage_crossings_lineage"][-1] / 1024.0, trans=r["n_transitions"][-1] / 1024.0)
P = np.load(f"{ROOT}/results/lta_histogram/production_T300/abf.npz", allow_pickle=True)
d = P["pmf"] - F[None, None]; d -= d.mean(-1, keepdims=True); e = np.sqrt((d ** 2).mean(-1))
assert P["steps"][7] == 21000
prod = dict(fwin=P["frac_window"][7:].mean(0), IF=np.trapezoid(e, P["times"], axis=0), eT=e[-1],
            cx=P["n_cage_crossings"] / 1024.0, trans=P["n_transitions"] / 1024.0)
tdr = [numba_stats(E.load_result(f"tdraw/T300_abf_s{s}.npz")) for s in range(1, 49) if __import__("os").path.exists(f"tdraw/T300_abf_s{s}.npz")]
nat = [numba_stats(E.load_result(f"{FULL}/T300_abf_s{s}.npz")) for s in range(1, 33)]
C = np.load(f"{ROOT}/results/lta_histogram/calibration/width_ladder_T300/n180.npz", allow_pickle=True)
cal = None
for k in ("frac_window",):
    if k in C.files:
        cal = C[k][7:].mean(0)
print("window occupancy, saves step >= 21000 (mean +- SE, n):")
groups = dict(numba_PCG64=np.array([r["fwin"] for r in nat]), numba_torchCPU_MT=np.array([r["fwin"] for r in tdr]),
              prod_CUDA=np.asarray(prod["fwin"], float))
if cal is not None:
    groups["calib_CUDA"] = np.asarray(cal, float)
for k, v in groups.items():
    print(f"  {k:18s} {v.mean():.4f} +- {v.std(ddof=1)/np.sqrt(v.size):.4f}  sd {v.std(ddof=1):.4f}  n {v.size}")
def cmp(a, b, name):
    print(f"  {name}: Welch p {stats.ttest_ind(a, b, equal_var=False).pvalue:.3g}, MW p {stats.mannwhitneyu(a, b).pvalue:.3g}")
cmp(groups["numba_torchCPU_MT"], groups["numba_PCG64"], "torchCPU-draws vs numba-native")
cmp(groups["numba_torchCPU_MT"], groups["prod_CUDA"], "torchCPU-draws vs prod CUDA  ")
cmp(groups["numba_PCG64"], groups["prod_CUDA"], "numba-native vs prod CUDA    ")
cpu = np.concatenate([groups["numba_torchCPU_MT"], groups["numba_PCG64"]])
cuda = np.concatenate([groups["prod_CUDA"]] + ([groups["calib_CUDA"]] if cal is not None else []))
cmp(cpu, cuda, f"all CPU-engine (n {cpu.size}) vs all CUDA (n {cuda.size})")
rng = np.random.default_rng(1)
for key in ("IF", "eT", "cx", "trans"):
    a = np.array([r[key] for r in tdr]); b = np.array([r[key] for r in nat]); c = np.asarray(prod[key], float)
    print(f"{key:5s}: torchCPU {a.mean():.4f}+-{a.std(ddof=1)/np.sqrt(a.size):.4f} | native {b.mean():.4f}+-{b.std(ddof=1)/np.sqrt(32):.4f} | prod {c.mean():.4f}+-{c.std(ddof=1)/4:.4f} | p(tc vs nat) {stats.ttest_ind(a, b, equal_var=False).pvalue:.3f} p(tc vs prod) {stats.ttest_ind(a, c, equal_var=False).pvalue:.3f}")
