"""Phase J, re-scored against the honest baseline: plain SHUS.

The frozen primary was the matched-turnover weighted sham.  That reference is only
meaningful if the sham is inert, which it is in every equal-weight experiment of this
campaign -- and is NOT here.  So every ratio is recomputed against plain SHUS, and the
sham's own degradation is reported as the finding it is.
"""
import glob, json, math, os, sys
import numpy as np
sys.path.insert(0, "src")
from abpfr.io import load_run
from abpfr.metrics import paired_bootstrap_ci

OUT = "results/appmap_phaseJ_variance"
SEEDS = list(range(920, 936))
ARMS = ["shus", "fr_cond", "sham_cond", "wfr_cond", "wfr_cond_hot", "wcnt_cond_hot",
        "wstate_hot", "wstate_eq", "wsham_cond", "wsham_eq"]
PI = math.pi

def load(cell, arm):
    out = []
    for s in SEEDS:
        a, m = load_run(os.path.join(OUT, f"hp{cell[0]:g}_d{cell[1]:g}_{arm}_seed{s}"))
        out.append((a, m))
    return out

def stats(rows):
    F = np.array([a["pmf_t"][-1] for a, _ in rows])
    F = F - F.mean(axis=1, keepdims=True)
    Fref = rows[0][0]["F_ref"]; Fref = Fref - Fref.mean()
    phi = rows[0][0]["x_grid"]
    w = np.abs(np.abs(phi) - PI / 2) < PI / 4
    r = {}
    for tag, sel in (("all", np.ones_like(w)), ("B", w)):
        d = F[:, sel] - Fref[sel]
        r[tag] = dict(mse=float((d ** 2).mean()), bias2=float((d.mean(0) ** 2).mean()),
                      var=float(F[:, sel].var(axis=0).mean()))
    r["IF"] = np.array([m["int_l2_f"] for _, m in rows])
    r["essw"] = np.median([m["min_ess_w"] for _, m in rows])
    r["turn"] = np.median([m["total_turnover"] for _, m in rows])
    return r

for cell in ((1.5, 1.0), (2.0, 1.0)):
    print(f"\n=== cell Hperp={cell[0]} Delta={cell[1]} — everything vs PLAIN SHUS ===")
    S = {a: stats(load(cell, a)) for a in ARMS}
    b = S["shus"]
    print(f"{'arm':>15}{'dI_F %':>22}{'var(B)':>10}{'var(all)':>10}"
          f"{'bias2(B)':>11}{'MSE(B)':>10}{'ESSw':>7}{'turn':>7}")
    for a in ARMS:
        d, lo, hi = paired_bootstrap_ci(100.0 * (S[a]["IF"] - b["IF"]) / b["IF"])
        print(f"{a:>15}{d:9.1f} [{lo:6.1f},{hi:6.1f}]"
              f"{100*(S[a]['B']['var']/b['B']['var']-1):+10.1f}"
              f"{100*(S[a]['all']['var']/b['all']['var']-1):+10.1f}"
              f"{100*(S[a]['B']['bias2']/b['B']['bias2']-1):+11.1f}"
              f"{100*(S[a]['B']['mse']/b['B']['mse']-1):+10.1f}"
              f"{S[a]['essw']:7.3f}{S[a]['turn']:7.0f}")
    print("   (percentages are vs plain SHUS; ESSw = median min weight-ESS)")
