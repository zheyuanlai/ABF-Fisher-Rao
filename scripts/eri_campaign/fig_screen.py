"""ABF-only screen figure: e_F(t), window occupancy vs bias-aware target, translocations, early-transient share."""
import glob, os, re, sys, math
import numpy as np
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import setup, SERIES, INK2, MUTED
from eri import metrics as mt
from eri.model import KB
plt = setup()
d = os.path.join(ROOT, sys.argv[1] if len(sys.argv) > 1 else "results/eri_campaign/screen_dilute")
out = os.path.join(ROOT, sys.argv[2] if len(sys.argv) > 2 else "figures/eri_campaign/screen"); os.makedirs(out, exist_ok=True)
Fr, Gr, _ = mt.reference_on_grid(180)
fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
for i, K in enumerate((64, 256, 1024)):
    E, R, TR = [], [], []
    for p in sorted(glob.glob(os.path.join(d, f"abf_N{K}_s*.npz"))):
        r = mt.load_run(p); t = r["times"]
        e = mt.errors(r, Fr, Gr); E.append(e["eF"])
        m = mt.marginal(r, Fr, KB * r["cfg"]["T_K"]); R.append(m["frac_win"] / np.maximum(m["Q_win"], 1e-12))
        TR.append(r["sv_i"][:, 1] / K)
    E, R, TR = map(np.array, (E, R, TR))
    c = SERIES[i]
    ax[0].plot(t, np.median(E, 0), color=c, label=f"K = {K}")
    ax[0].fill_between(t, np.quantile(E, 0.25, 0), np.quantile(E, 0.75, 0), color=c, alpha=0.2, lw=0)
    # 2 t.u. block averages of the occupancy ratio
    blk = np.arange(0, 40.01, 2.0)
    Rb = np.array([[np.mean(rr[(t >= a) & (t < b)]) for a, b in zip(blk[:-1], blk[1:])] for rr in R])
    xb = 0.5 * (blk[:-1] + blk[1:])
    ax[1].plot(xb, np.median(Rb, 0), "o-", color=c, ms=3, label=f"K = {K}")
    ax[1].fill_between(xb, Rb.min(0), Rb.max(0), color=c, alpha=0.15, lw=0)
    ax[2].plot(t, np.median(TR, 0), color=c, label=f"K = {K}")
ax[0].set_yscale("log"); ax[0].set_xlabel("t (t.u.)"); ax[0].set_ylabel("e_F(t) vs exact MC (kJ/mol)")
ax[0].set_title("ABF-only: free-energy error (median, IQR; 8 seeds)"); ax[0].legend(fontsize=8)
ax[1].axhline(1.0, color=INK2, lw=1); ax[1].axhline(0.75, color=MUTED, ls="--", lw=1)
ax[1].text(20, 0.71, "deficit threshold 0.75 (Amendment 2)", color=INK2, fontsize=7, ha="center")
ax[1].set_xlabel("t (t.u.), 2 t.u. blocks"); ax[1].set_ylabel("window occupancy / bias-aware target Q*_W")
ax[1].set_title("Establishment: never deficient"); ax[1].set_ylim(0.4, 1.6); ax[1].legend(fontsize=8)
ax[2].set_xlabel("t (t.u.)"); ax[2].set_ylabel("true cage-to-cage translocations per walker")
ax[2].set_title("Physical transitions (core-set, clones excluded)"); ax[2].legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(out, "fig_screen_abf_only.png")); plt.close(fig)
print("->", out)
