"""ITE (candidate A') reference figure: F / U / -TS along the CV, conditional hidden variables, channel geometry."""
import math, os, sys
import numpy as np
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "scripts", "eri_campaign"))
from figstyle import setup, SERIES, MUTED, INK2
from eri.model import Model
plt = setup()
FF = dict(T_K=300.0, eps_O_K=93.0, sig_O=3.48, eps_Si_K=0.0, sig_Si=1.0, eps_CC_K=108.0, sig_CC=3.76, rc=12.0, shift=True, k_bond_K=96500.0, r0=1.54)
m = Model("ite_itq3", params=FF); fw = m.fw
z = np.load(os.path.join(ROOT, "results/overnight/ite/reference/ite_itq3/reference_profile.npz"))
kT = float(z["kT"]); g = z["grid"]; X = g * fw.P / (2 * math.pi)
F = z["F"] - z["F"].min(); jm = int(np.argmin(z["F"])); U = z["U"] - z["U"][jm]; mTS = F - U
out = os.path.join(ROOT, "figures/overnight/ite"); os.makedirs(out, exist_ok=True)
fig, ax = plt.subplots(1, 3, figsize=(12, 3.5), gridspec_kw=dict(width_ratios=[1.2, 1, 1.3]))
ax[0].fill_between(X, F - 2 * z["F_se"], F + 2 * z["F_se"], color=SERIES[0], alpha=0.25, lw=0)
ax[0].plot(X, F, color=SERIES[0], label="F (exact MC umbrella/WHAM)")
ax[0].plot(X, U, color=SERIES[1], label="U = <V | phi>")
ax[0].plot(X, mTS, color=SERIES[2], label="-TS = F - U")
ax[0].axvline(0, color=MUTED, lw=0.8, ls=":")
ax[0].set_xlabel("tagged COM X - window plane (A), X along crystal b"); ax[0].set_ylabel("kJ/mol, relative to the F minimum")
ax[0].set_title("ethane / rigid Si-ITE, 300 K: 15.9 kBT, 48 % entropic", fontsize=9); ax[0].legend(fontsize=7, loc="upper left")
ax[1].plot(X, z["ux2"], color=SERIES[0], label="<u_x^2 | phi> (alignment with channel)")
ax[1].plot(X, z["eta"], color=SERIES[1], label="<eta | phi>, eta = cos(2 pi Z / a)")
ax[1].set_xlabel("tagged COM X - window plane (A)"); ax[1].set_ylabel("conditional mean"); ax[1].set_ylim(0, 1.05)
ax[1].set_title("Hidden coordinates (MC reference)", fontsize=9); ax[1].legend(fontsize=7, loc="lower left")
# side view: O atoms with |Y - Y_axis| < 2.5 A around the family-A channel at Y = c/4, Z = 0; X-Z plane
P = fw.pos[fw.types == 0]; yax = fw.cages[0][1]
d = P[:, 1] - yax; d -= fw.box[1] * np.round(d / fw.box[1])
zz = P[:, 2].copy(); zz[zz > fw.box[2] / 2] -= fw.box[2]
sel = (np.abs(d) < 2.5) & (np.abs(zz) < 9)
ax[2].scatter(P[sel, 0], zz[sel], s=7, color="#b5b4ae", label="framework O (|Y - Y_axis| < 2.5 A)")
wx = (fw.x_window + np.arange(3) * fw.P)
for i, x in enumerate(wx):
    ax[2].axvline(x, color=SERIES[1], lw=1, ls="--", label="phi = 0 (8-ring window plane)" if i == 0 else None)
for i, x in enumerate(wx + fw.P / 2):
    ax[2].axvline(x, color=SERIES[0], lw=1, ls=":", label="phi = +-pi (cage)" if i == 0 else None)
ax[2].set_xlim(0, fw.box[0]); ax[2].set_ylim(-9, 9); ax[2].set_aspect("equal")
ax[2].set_xlabel("X (A) along crystal b: the 1-D channel"); ax[2].set_ylabel("Z (A) along crystal a")
ax[2].set_title("Channel side view (one family-A channel)", fontsize=9); ax[2].legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
fig.tight_layout(); fig.savefig(os.path.join(out, "fig_ite_reference.png"), bbox_inches="tight"); plt.close(fig)
print("->", out)
