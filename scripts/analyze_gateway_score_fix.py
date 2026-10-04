#!/usr/bin/env python
"""EXPLORATORY score-fix arms of the gateway replica ladder (configs/gateway_replica_ladder/exploratory_score_fix.json)."""
import glob
import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from analyze_gateway_replica_ladder import Scorer, boot_median, DESIGN, OUT, C_ABF, C_FR, C_INK, C_MUTED  # noqa: E402


def main():
    design = json.load(open(DESIGN))
    res = {}
    lines = ["# Gateway replica ladder -- EXPLORATORY score-fix arms (post-hoc mechanism test)", "",
             "Arms paired on shared noise within this stage; effects = per-seed (arm - ABF)/ABF, median [bootstrap 95 % CI] (wins/32).", ""]
    for ref in ("em", "analytic"):
        sc = Scorer(design["cell"], ref)
        recs = {}
        for f in sorted(glob.glob(os.path.join(OUT, "score_fix", "raw", "N*_s*.npz"))):
            z = np.load(f, allow_pickle=True)
            m = json.loads(str(z["meta_json"]))
            lin = np.isclose(z["u"] * 200, np.round(z["u"] * 200)) & (z["u"] > 0)
            for a, name in enumerate(m["arms"]):
                eF, _ = sc.score(z["M"][a, :, :180], z["C"][a, :, :180])
                ev = float(z["die"][a, -1] + z["clone"][a, -1])
                recs[(m["N"], m["seed"], name)] = dict(Ibar=float(np.mean(eF[lin])), fin=float(eF[-1]), ev=ev,
                                                       death=float(z["die"][a, -1]) / max(ev, 1.0), ess=float(z["ess"][a].min()))
        Ns = sorted({k[0] for k in recs}, reverse=True)
        arms = ["fr_h180", "fr_emp_h180", "fr_loo_h180"]
        lines += [f"## reference: {ref}{' (dt-consistent, primary for this test)' if ref == 'em' else ' (preregistered ladder reference)'}", "",
                  "| N | ABF Ibar_F | arm | dIbar_F | d e_F(B) | events | death fraction | min ESS/N |", "|---|---|---|---|---|---|---|---|"]
        res[ref] = {}
        for N in Ns:
            seeds = sorted({k[1] for k in recs if k[0] == N})
            abI = np.median([recs[(N, s, "abf_h180")]["Ibar"] for s in seeds])
            for arm in arms:
                dI = [100 * (recs[(N, s, arm)]["Ibar"] / recs[(N, s, "abf_h180")]["Ibar"] - 1) for s in seeds]
                dF = [100 * (recs[(N, s, arm)]["fin"] / recs[(N, s, "abf_h180")]["fin"] - 1) for s in seeds]
                bI, bF = boot_median(dI), boot_median(dF)
                ev = np.median([recs[(N, s, arm)]["ev"] for s in seeds])
                dfr = np.median([recs[(N, s, arm)]["death"] for s in seeds])
                ess = np.median([recs[(N, s, arm)]["ess"] for s in seeds])
                res[ref].setdefault(str(N), {})[arm] = dict(dIbar=bI, dfin=bF, wins_I=int(np.sum(np.array(dI) < 0)),
                                                             wins_F=int(np.sum(np.array(dF) < 0)), events=ev, death_frac=dfr, min_ess=ess)
                lines.append(f"| {N} | {abI:.5f} | {arm} | {bI[0]:+.1f} % [{bI[1]:+.1f}, {bI[2]:+.1f}] ({int(np.sum(np.array(dI) < 0))}/32) | "
                             f"{bF[0]:+.1f} % [{bF[1]:+.1f}, {bF[2]:+.1f}] ({int(np.sum(np.array(dF) < 0))}/32) | {ev:.0f} | {dfr:.2f} | {ess:.2f} |")
        lines.append("")
    open(os.path.join(OUT, "score_fix", "scoreboard.md"), "w").write("\n".join(lines))
    json.dump(res, open(os.path.join(OUT, "score_fix", "summary.json"), "w"), indent=2, default=float)
    print("\n".join(lines))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": C_MUTED,
                         "xtick.color": C_MUTED, "ytick.color": C_MUTED, "legend.frameon": False, "axes.titleweight": "bold",
                         "grid.color": "#e6e5e0"})
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    sty = {"fr_h180": (C_FR, "o", "accepted score (KL-centred)"), "fr_emp_h180": ("#1baf7a", "s", "empirical centring"),
           "fr_loo_h180": ("#4a3aa7", "D", "leave-one-out + empirical centring")}
    for ax, key, ttl in ((axs[0], "dIbar", "integrated error  (1/B) ∫ e_F db"), (axs[1], "dfin", "final error  e_F(B)")):
        for k, arm in enumerate(sty):
            xs = np.array(sorted([int(n) for n in res["em"]]), float)
            v = np.array([res["em"][str(int(n))][arm][key] for n in xs])
            off = [0.9, 1.0, 1.1][k]
            col, mk, lab = sty[arm]
            ax.errorbar(xs * off, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt=mk, color=col, ms=4.5, capsize=2, lw=1, label=lab)
            ax.plot(xs * off, v[:, 0], color=col, lw=1, alpha=0.6)
        ax.axhline(0, color=C_MUTED, lw=0.8)
        ax.set_xscale("log", base=2); ax.set_xticks([4, 16, 64, 256, 1024]); ax.set_xticklabels(["4", "16", "64", "256", "1k"])
        ax.set_xlabel("replicas N  (fixed budget B)"); ax.set_ylabel("vs ABF, paired median change (%)"); ax.grid(True, axis="y")
        ax.set_title(ttl, loc="left")
    axs[0].legend(fontsize=7.5, loc="lower left")
    fig.suptitle("Gateway, EXPLORATORY: does fixing the small-N score bias rescue FR at few replicas? (dt-consistent reference, 32 seeds)",
                 fontsize=9.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, "figures", f"fig_score_fix_emref.{ext}"), dpi=170)


if __name__ == "__main__":
    main()
