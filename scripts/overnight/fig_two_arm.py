"""Two-arm figure (ABF vs ABF + uniform FR only) for one paired block: e_F(t), e_F'(t), window occupancy / uniform,
TV(p, uniform), windowed ancestor ESS (FR arm), and per-seed relative change of Ibar_F."""
import argparse, glob, json, os, re, sys
import numpy as np
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "scripts", "eri_campaign"))
from figstyle import setup, ABF, FR, INK2, MUTED
from eri import metrics as mt
plt = setup()
ap = argparse.ArgumentParser()
ap.add_argument("dir"); ap.add_argument("--K", type=int, required=True); ap.add_argument("--refbase", required=True)
ap.add_argument("--variant", default="ite_itq3"); ap.add_argument("--title", default=""); ap.add_argument("--out", required=True)
a = ap.parse_args()
mt.set_reference_base(a.refbase)
runs = {"abf": {}, "fr": {}}
for m in runs:
    for f in sorted(glob.glob(os.path.join(ROOT, a.dir, f"{m}_N{a.K}_s*.npz"))):
        if f.endswith((".ckpt.npz", ".tmp.npz")):
            continue
        s = int(re.search(r"_s(\d+)\.npz$", f).group(1))
        r = mt.load_run(f); Fr, Gr = mt.reference_for_grid(r["sv_C"].shape[1], a.variant); e = mt.errors(r, Fr, Gr)
        h = r["sv_h"]; p = h / np.maximum(h.sum(1, keepdims=True), 1); ng = h.shape[1]
        grid = (np.arange(ng) + 0.5) / ng * 2 * np.pi - np.pi; W = np.abs(grid) < r["cfg"]["window_phi"]
        pc = p.reshape(p.shape[0], 18, -1).sum(2)
        runs[m][s] = dict(t=e["t"], eF=e["eF"], eG=np.sqrt(((mt.mean_force(r["sv_M"], r["sv_C"]) - Gr[None, :]) ** 2).mean(-1)),
                          win=p[:, W].sum(1) / W.mean(), tv=0.5 * np.abs(pc - 1 / 18).sum(1), ess=r["sv_u"][:, 2], Ibar=e["Ibar_F"])
seeds = sorted(set(runs["abf"]) & set(runs["fr"]))
t = runs["abf"][seeds[0]]["t"]
fig, ax = plt.subplots(2, 3, figsize=(12, 6.4))
def band(axx, key, m, col, lab, log=False):
    X = np.array([runs[m][s][key] for s in seeds])
    axx.plot(t, np.median(X, 0), color=col, label=lab)
    axx.fill_between(t, np.quantile(X, 0.25, 0), np.quantile(X, 0.75, 0), color=col, alpha=0.2, lw=0)
    if log: axx.set_yscale("log")
for m, col, lab in (("abf", ABF, "ABF"), ("fr", FR, "ABF + FR")):
    band(ax[0, 0], "eF", m, col, lab, log=True); band(ax[0, 1], "eG", m, col, lab, log=True)
    band(ax[1, 0], "win", m, col, lab); band(ax[1, 1], "tv", m, col, lab)
band(ax[1, 2], "ess", "fr", FR, "ABF + FR: windowed ancestor ESS / K")
ax[0, 0].set_ylabel("e_F(t) vs exact MC (kJ/mol)"); ax[0, 0].set_title("free-energy error (median, IQR)", fontsize=9)
ax[0, 1].set_ylabel("e_F'(t) (kJ/mol/rad)"); ax[0, 1].set_title("mean-force error", fontsize=9)
ax[1, 0].set_ylabel("window occupancy / uniform share"); ax[1, 0].axhline(1, color=MUTED, lw=0.8); ax[1, 0].set_title("window population", fontsize=9)
ax[1, 1].set_ylabel("TV(p, uniform), 18 coarse bins"); ax[1, 1].set_title("marginal vs uniform", fontsize=9)
ax[1, 2].set_ylabel("ESS / K (FR arm)"); ax[1, 2].set_ylim(0, 1.05); ax[1, 2].set_title("genealogical diversity", fontsize=9)
d = np.array([runs["fr"][s]["Ibar"] / runs["abf"][s]["Ibar"] - 1 for s in seeds]) * 100
ax[0, 2].axhline(0, color=MUTED, lw=0.8); ax[0, 2].axhspan(-10, 10, color=MUTED, alpha=0.12, lw=0)
ax[0, 2].plot(np.arange(len(d)), np.sort(d), "o", color=FR, ms=6)
ax[0, 2].set_xlabel("seed (sorted)"); ax[0, 2].set_ylabel("Ibar_F(FR)/Ibar_F(ABF) - 1 (%)")
ax[0, 2].set_title(f"paired per-seed change; median {np.median(d):+.1f} %, FR better {int((d < 0).sum())}/{len(d)}", fontsize=9)
for axx in ax.flat:
    if axx is not ax[0, 2]:
        axx.set_xlabel("t (t.u.)")
ax[0, 0].legend(fontsize=8); ax[1, 2].legend(fontsize=7)
fig.suptitle(a.title, fontsize=10); fig.tight_layout()
os.makedirs(os.path.join(ROOT, a.out), exist_ok=True)
fig.savefig(os.path.join(ROOT, a.out, f"fig_two_arm_K{a.K}.png"), bbox_inches="tight"); print("->", a.out)
