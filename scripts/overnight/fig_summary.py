"""Cross-cell summary (ABF only): barrier decomposition, early-phase error share, ABF e_F(T) vs K, classification."""
import json, os, sys
import numpy as np
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "eri_campaign"))
from figstyle import setup, SERIES, INK2, MUTED
plt = setup()
KB = 0.008314462618
cells = [("ERI 600 K", 600, "results/eri_campaign/reference/gard_tait/gate1.json", "results/eri_campaign/screen_dilute/classification.json"),
         ("ITE 300 K", 300, "results/overnight/ite/reference/ite_itq3/gate1.json", "results/overnight/ite/screen/classification.json"),
         ("ITE 150 K", 150, "results/overnight/ite150/reference/ite_itq3/gate1.json", "results/overnight/ite150/screen/classification.json")]
dps = json.load(open(os.path.join(ROOT, "results/overnight/deficit_phase_share.json")))
fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
x = np.arange(len(cells))
dU = []; mTS = []
for name, T, g1, cl in cells:
    b = json.load(open(os.path.join(ROOT, g1)))["barrier"]
    kT = KB * T
    dU.append(b["U_barrier"] / kT); mTS.append(b["mTS_barrier"] / kT)
ax[0].bar(x, dU, color=SERIES[1], width=0.55, label="energetic  dU / kT", edgecolor="white", linewidth=2)
ax[0].bar(x, mTS, bottom=dU, color=SERIES[2], width=0.55, label="entropic  -T dS / kT", edgecolor="white", linewidth=2)
for i in range(len(cells)):
    tot = dU[i] + mTS[i]
    ax[0].text(i, tot + 0.4, f"{tot:.1f} kT\n{100 * mTS[i] / tot:.0f} % entropic", ha="center", fontsize=8, color=INK2)
ax[0].set_xticks(x, [c[0] for c in cells]); ax[0].set_ylabel("barrier on the CV (kT)"); ax[0].set_ylim(0, 30)
ax[0].set_title("Exact-MC barrier decomposition", fontsize=9); ax[0].legend(fontsize=7, loc="upper left")
for j, K in enumerate((64, 256, 1024)):
    sh = [100 * dps[f"{c[0]} K{K}"]["IF_share"] for c in cells]
    ax[1].bar(x + (j - 1) * 0.25, sh, width=0.23, color=SERIES[j], label=f"K = {K}", edgecolor="white", linewidth=1)
for i, c in enumerate(cells):
    ax[1].text(i, 62, f"T_reach {dps[c[0] + ' K1024']['T_reach']:.0f} t.u.\n({100 * dps[c[0] + ' K1024']['frac_T']:.1f} % of T)", ha="center", fontsize=7, color=INK2)
ax[1].set_xticks(x, [c[0] for c in cells]); ax[1].set_ylabel("% of ABF's I_F accrued before the window\npopulation reaches 0.75 Q* (for good)")
ax[1].set_ylim(0, 88); ax[1].set_title("Early (bias build-up) phase: short but costly", fontsize=9); ax[1].legend(fontsize=7, loc="upper center", ncol=3)
for i, (name, T, g1, cl) in enumerate(cells):
    d = json.load(open(os.path.join(ROOT, cl)))["cells"]
    Ks = sorted(int(k) for k in d)
    ax[2].plot(Ks, [d[str(k)]["secondary"]["median_eF_T"] for k in Ks], "o-", color=SERIES[i], ms=6, label=f"{name}: " + ", ".join(sorted(set(d[str(k)]["label"].split(" ")[0] for k in Ks))))
ax[2].set_xscale("log", base=2); ax[2].set_xlabel("replicas K"); ax[2].set_ylabel("ABF median e_F(T) vs exact MC (kJ/mol)")
ax[2].set_title("ABF-only final error; frozen classification", fontsize=9); ax[2].legend(fontsize=7)
fig.tight_layout(); out = os.path.join(ROOT, "figures/overnight/summary"); os.makedirs(out, exist_ok=True)
fig.savefig(os.path.join(out, "fig_overnight_summary.png"), bbox_inches="tight"); plt.close(fig); print("->", out)
