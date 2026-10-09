#!/usr/bin/env python
"""Analysis of the conditional Phase-V WCA confirmatory run (configs/numerical_validation/wca_confirmatory_prereg.json).

    python scripts/numerical_validation/analyze_wca_confirmatory.py --dt 0.000125 --N 256 --T 240

Scores every arm with the accepted read-out against the EXACT (MC) reference; paired per-seed contrasts with
10000-resample bootstrap CIs of the median; the frozen decision rule; pooled long-run limits (FR tilt); genealogy,
coverage and overhead secondaries.  Writes <run dir>/summary.json, scoreboard.md and figures.
"""
import argparse
import glob
import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "numerical_validation"))
import analyze_wca_validation as A  # noqa: E402
import rescore_wca_history as R  # noqa: E402
import wca_numba as wn  # noqa: E402

ABF, FR, SH = "hist_abf", "hist_fr_uniform", "hist_fr_sham"


def boot(x, seed):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    m = np.median(x[rng.integers(0, len(x), (10000, len(x)))], 1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def contrast(a, b, seed):
    d = 100.0 * (np.asarray(a) / np.asarray(b) - 1.0)
    return dict(median=float(np.median(d)), ci95=boot(d, seed), wins=int((d < 0).sum()), n=len(d), per_seed=d.tolist())


class Pool:
    def __init__(self, label, M, C):
        self.label, self.n = label, len(M)
        self.Ms, self.Cs = np.asarray(M), np.asarray(C)
        self.kind = "mf"
        self.F = self.boot(np.arange(self.n))
        loo = np.array([self.boot(np.delete(np.arange(self.n), i)) for i in range(self.n)])
        self.se = np.sqrt((self.n - 1) / self.n * ((loo - loo.mean(0)) ** 2).sum(0))

    def boot(self, idx):
        return A.F_mf(self.Ms[idx].sum(0), self.Cs[idx].sum(0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt", type=float, required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--T", type=float, required=True)
    a = ap.parse_args()
    d = os.path.join(ROOT, "results", "numerical_validation", "wca_confirmatory", f"dt{a.dt:g}_N{a.N}_T{a.T:g}")
    seeds = sorted(int(os.path.basename(f)[1:].split("_")[0]) for f in glob.glob(os.path.join(d, "raw", "s*_abf.npz")))
    seeds = [s for s in seeds if os.path.exists(os.path.join(d, "raw", f"s{s}_frsham.npz"))]
    assert seeds, "no complete seed"
    ref, nmc = R.mc_reference()
    sc = {ABF: [], FR: [], SH: []}
    pooled = {ABF: ([], []), FR: ([], []), SH: ([], [])}
    sec = {k: dict(repl=[], min_ess_w=[], wmax=[], rt=[], frac_stretched=[], wall=[]) for k in (ABF, FR, SH)}
    times = None
    for s in seeds:
        for part in ("abf", "frsham"):
            res = wn.load_result(os.path.join(d, "raw", f"s{s}_{part}.npz"))
            meta = json.loads(str(res["meta_json"]))
            for i, arm in enumerate(res["arms"]):
                o = wn.score_ladder_result(res, arm, reference=ref)
                times = o["times"]
                sc[arm].append(dict(I_F=float(o["integrated_l2_f"]), e_F=float(o["l2_f"]), e_Fp=float(o["l2_fp"]), e_t=np.asarray(o["l2_f_t"])))
                pooled[arm][0].append(np.asarray(res["M_rep"])[i, -1, :A.NB])
                pooled[arm][1].append(np.asarray(res["C_rep"])[i, -1, :A.NB])
                sec[arm]["repl"].append(float(np.asarray(res["repl_cumulative"])[i, -1]))
                sec[arm]["min_ess_w"].append(float(np.asarray(res["min_ancestor_ess_window"])[i]) / a.N)
                sec[arm]["wmax"].append(float(np.nanmax(np.asarray(res["max_ancestor_frac"])[i])) if arm != ABF else float("nan"))
                sec[arm]["rt"].append(float(np.asarray(res["n_round_trips"])[i, -1]) / a.N)
                sec[arm]["frac_stretched"].append(float(np.mean(np.asarray(res["frac_regions"])[i, :, 2])))
                sec[arm]["wall"].append(meta["wall_s"] / len(res["arms"]))
    S = dict(prereg="configs/numerical_validation/wca_confirmatory_prereg.json", dt=a.dt, N=a.N, T=a.T, seeds=seeds,
             reference=ref["label"], n_mc_chains=nmc)
    S["medians"] = {arm: {k: float(np.median([r[k] for r in v])) for k in ("I_F", "e_F", "e_Fp")} for arm, v in sc.items()}
    C = {}
    for (x, y), sd in (((FR, ABF), 1), ((SH, ABF), 2), ((FR, SH), 3)):
        for k, off in (("I_F", 0), ("e_F", 10), ("e_Fp", 20)):
            C[f"{x}_vs_{y}_{k}"] = contrast([r[k] for r in sc[x]], [r[k] for r in sc[y]], 20261009 + sd + off)
    S["contrasts"] = C
    c, cs = C[f"{FR}_vs_{ABF}_I_F"], C[f"{FR}_vs_{SH}_I_F"]
    if c["ci95"][1] < 0 and cs["ci95"][1] < 0:
        verdict = "FR_BENEFICIAL"
    elif c["ci95"][0] > 0:
        verdict = "FR_HARMFUL"
    elif c["ci95"][0] >= -10 and c["ci95"][1] <= 10:
        verdict = "FR_NEUTRAL"
    else:
        verdict = "INCONCLUSIVE"
    S["verdict"] = verdict
    # pooled limits and the FR tilt
    mc = A.load_group("mc", "mc")
    E_mc = A.Ens("MC", mc, "density")
    P = {arm: Pool(arm, *pooled[arm]) for arm in pooled}
    lim = {}
    for arm, E in P.items():
        cm = A.compare(E, E_mc, nboot=1000)
        lim[arm] = dict(D_vs_MC=cm["D"], up=cm["D_upper95"], noise=cm["noise"])
    ct = A.compare(P[FR], P[ABF], nboot=1000)
    lim["FR_tilt_vs_ABF"] = dict(D=ct["D"], up=ct["D_upper95"], noise=ct["noise"])
    cs_ = A.compare(P[SH], P[ABF], nboot=1000)
    lim["sham_vs_ABF"] = dict(D=cs_["D"], up=cs_["D_upper95"], noise=cs_["noise"])
    S["pooled_limits"] = lim
    S["secondary"] = {arm: {k: (float(np.nanmedian(v)) if len(v) else None) for k, v in d_.items()} for arm, d_ in sec.items()}
    S["secondary"]["note"] = "min_ess_w = min windowed ancestor ESS / N (median over seeds); wmax = max ancestor family share over saves; rt = round trips per replica; wall = process wall-clock / arms in process (s)"
    w_abf = S["secondary"][ABF]["wall"]
    S["overhead_frsham_process_per_arm_vs_abf"] = (S["secondary"][FR]["wall"] / w_abf - 1.0) if w_abf else None
    S["median_curves"] = {arm: np.median(np.array([r["e_t"] for r in v]), 0).tolist() for arm, v in sc.items()}
    S["times"] = np.asarray(times).tolist()
    json.dump(S, open(os.path.join(d, "summary.json"), "w"), indent=1, default=float)
    L = [f"# WCA confirmatory (dt {a.dt:g}, N {a.N}, T {a.T:g}): ABF vs ABF+FR vs matched sham, exact (MC) reference", "",
         f"seeds {seeds[0]}-{seeds[-1]} ({len(seeds)}); verdict (frozen rule): **{verdict}**", "",
         "| arm | median I_F | median e_F(T) | median own e_F' | replacements | min windowed ESS/N | max family share | round trips / replica |",
         "|---|---|---|---|---|---|---|---|"]
    for arm in (ABF, FR, SH):
        m, q = S["medians"][arm], S["secondary"][arm]
        L.append(f"| {arm} | {m['I_F']:.4f} | {m['e_F']:.4f} | {m['e_Fp']:.3f} | {q['repl']:.0f} | {q['min_ess_w']:.3f} | {q['wmax']:.4f} | {q['rt']:.2f} |")
    L += ["", "| contrast | I_F | e_F(T) | e_F' |", "|---|---|---|---|"]
    for x, y in ((FR, ABF), (SH, ABF), (FR, SH)):
        cells = []
        for k in ("I_F", "e_F", "e_Fp"):
            q = C[f"{x}_vs_{y}_{k}"]
            cells.append(f"{q['median']:+.1f} % [{q['ci95'][0]:+.1f}, {q['ci95'][1]:+.1f}] ({q['wins']}/{q['n']})")
        L.append(f"| {x} vs {y} | " + " | ".join(cells) + " |")
    L += ["", "pooled long-run limits (D vs exact, kT): " + ", ".join(f"{k} {v['D_vs_MC']:.4f}" for k, v in lim.items() if "D_vs_MC" in v),
          f"FR tilt D(FR limit, ABF limit) = {lim['FR_tilt_vs_ABF']['D']:.4f} [up {lim['FR_tilt_vs_ABF']['up']:.4f}]; sham vs ABF {lim['sham_vs_ABF']['D']:.4f}",
          f"FR+sham process wall per arm vs ABF process: {100 * S['overhead_frsham_process_per_arm_vs_abf']:+.1f} %"]
    open(os.path.join(d, "scoreboard.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    figure(S, P, E_mc, d)


def figure(S, P, E_mc, d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.3))
    cols = {ABF: "#2a78d6", FR: "#eb6834", SH: "#8a8984"}
    t = np.array(S["times"])
    for arm, c in cols.items():
        axs[0].semilogy(t[1:], np.array(S["median_curves"][arm])[1:], color=c, lw=1.6, label=arm)
    axs[0].set_xlabel("t (t.u.)"); axs[0].set_ylabel("median e_F(t) vs exact Gibbs (kT)")
    axs[0].set_title(f"(a) dt {S['dt']:g}, N {S['N']}: {S['verdict']}", loc="left"); axs[0].legend()
    W = A.WIN
    axs[1].fill_between(A.CEN[W], -2 * E_mc.se[W], 2 * E_mc.se[W], color="#dddddd", label="MC +-2 se")
    for arm, c in cols.items():
        axs[1].plot(A.CEN[W], (A.centre(P[arm].F - E_mc.F))[W], color=c, lw=1.4, label=f"{arm} pooled limit")
    axs[1].axhline(0, color="k", lw=0.6)
    axs[1].set_xlabel("z"); axs[1].set_ylabel("F - F_MC (kT)"); axs[1].set_title("(b) pooled long-run limits minus exact", loc="left")
    axs[1].legend(fontsize=7)
    fig.tight_layout()
    os.makedirs(os.path.join(d, "figures"), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(d, "figures", f"fig_wca_confirmatory.{ext}"), dpi=160)


if __name__ == "__main__":
    main()
