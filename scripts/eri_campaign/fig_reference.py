"""Figures of the Gate 1 / Gate 2 reference: F, U, -TS; single-bead literature check; timestep ladder; geometry."""
import json, math, os, sys
import numpy as np
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import setup, SERIES, INK2, MUTED
from eri import refanalysis as ra
from eri.model import KB, Model
plt = setup()
OUT = os.path.join(ROOT, "figures", "eri_campaign", "reference"); os.makedirs(OUT, exist_ok=True)
base = os.path.join(ROOT, "results", "eri_campaign", "reference", "gard_tait")
z = np.load(os.path.join(base, "reference_profile.npz"))
kT = float(z["kT"]); x = z["grid"]
a = 13.27; P = a / 2
F = z["F"] - z["F"].min()
U = z["U"]; jm = int(np.argmin(z["F"]))
Ur = U - U[jm]
mTS = F - Ur
fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.4))
X = x * P / (2 * math.pi)
ax[0].fill_between(X, F - 2 * z["F_se"], F + 2 * z["F_se"], color=SERIES[0], alpha=0.25, lw=0)
ax[0].plot(X, F, color=SERIES[0], label="F (MC umbrella/WHAM)")
ax[0].plot(X, Ur, color=SERIES[1], label="U = <V | phi>")
ax[0].plot(X, mTS, color=SERIES[2], label="-TS = F - U")
ax[0].set_xlabel("tagged COM X - window plane (A)"); ax[0].set_ylabel("kJ/mol (relative to the cage minimum)")
ax[0].set_title("Ethane / Gard & Tait ERI, 600 K: 6.67 kBT barrier, 84 % entropic")
ax[0].legend(loc="upper left", fontsize=8)
ax[0].axvline(0, color=MUTED, lw=0.8, ls=":")
# literature check: bead profiles for both frameworks
sel = json.load(open(os.path.join(ROOT, "results", "eri_campaign", "reference", "framework_selection.json")))
for i, v in enumerate(("gard_tait", "schlenker")):
    bF = np.array(sel[v]["bead_s_profile"]); s = (np.arange(len(bF)) + 0.5) / len(bF) * 1.05 * (a if v == "gard_tait" else 13.252) / (2 * math.sqrt(3))
    ax[1].plot(s, bF, color=SERIES[i], label=f"{'Gard & Tait 1973' if v == 'gard_tait' else 'Schlenker 1977'}: barrier {bF[0]:.2f} kBT")
ax[1].axhline(6.13, color=INK2, lw=1.0, ls="--"); ax[1].text(2.2, 6.25, "PNAS 2005 infinite dilution: 6.13 kBT", color=INK2, fontsize=8)
ax[1].set_xlabel("single-bead |s| from the nearest 8-ring window (A)"); ax[1].set_ylabel("beta F (single bead)")
ax[1].set_title("Literature check (frozen selection rule, band +-0.35)"); ax[1].legend(loc="center right", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_reference_profiles.png")); plt.close(fig)
# Gate 2 ladder
g = {t: json.load(open(os.path.join(base, f"gate2_{t}.json"))) for t in ("EM_2e-04", "EM_1e-04", "EM_5e-05", "LM_2e-04")}
fig, ax = plt.subplots(1, 3, figsize=(10, 3.0))
dts = [2e-4, 1e-4, 5e-5]
for j, key in enumerate(("D_density", "D_MF")):
    ax[0].plot(dts, [g[f"EM_{d:.0e}"][key]["D"] for d in dts], "o-", color=SERIES[j], ms=5, label=f"EM {key}")
    ax[0].errorbar(dts, [g[f"EM_{d:.0e}"][key]["D"] for d in dts], yerr=[[0] * 3, [g[f"EM_{d:.0e}"][key]["D_upper95"] - g[f"EM_{d:.0e}"][key]["D"] for d in dts]], fmt="none", ecolor=SERIES[j], lw=1)
ax[0].axhline(0.05, color=INK2, ls="--", lw=1); ax[0].text(5.2e-5, 0.052, "tolerance 0.05 kJ/mol", color=INK2, fontsize=8)
ax[0].set_xscale("log"); ax[0].set_xlabel("dt (t.u.)"); ax[0].set_ylabel("D vs exact MC (kJ/mol)"); ax[0].legend(fontsize=8); ax[0].set_title("F(phi): debiased RMS vs MC")
ax[1].plot(dts, [100 * g[f"EM_{d:.0e}"]["bond_var_rel"] for d in dts], "o-", color=SERIES[0], ms=5, label="EM")
ax[1].plot([2e-4], [100 * g["LM_2e-04"]["bond_var_rel"]], "s", color=SERIES[1], ms=6, label="LM 2e-4")
ax[1].set_xscale("log"); ax[1].set_xlabel("dt (t.u.)"); ax[1].set_ylabel("bond variance error vs MC (%)"); ax[1].legend(fontsize=8); ax[1].set_title("stiffest mode: first order in dt")
mcb = json.load(open(os.path.join(base, "gate1.json")))["barrier"]["F_barrier"]
ax[2].plot(dts, [g[f"EM_{d:.0e}"]["barrier"]["F_barrier"] - mcb for d in dts], "o-", color=SERIES[0], ms=5, label="EM")
ax[2].plot([2e-4], [g["LM_2e-04"]["barrier"]["F_barrier"] - mcb], "s", color=SERIES[1], ms=6, label="LM 2e-4")
ax[2].axhline(0, color=MUTED, lw=0.8); ax[2].set_xscale("log"); ax[2].set_xlabel("dt (t.u.)"); ax[2].set_ylabel("barrier - MC barrier (kJ/mol)"); ax[2].legend(fontsize=8); ax[2].set_title("barrier shift")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_gate2_timestep.png")); plt.close(fig)
# geometry: xy projection of a slab at window height, cages, windows, CV planes
m = Model("gard_tait"); fw = m.fw
fig, ax = plt.subplots(1, 1, figsize=(6.4, 5.6))
pos, ty = fw.pos, fw.types
zc = 0.0
for zw in (0.0, fw.c / 2):
    pass
slab = np.abs(((pos[:, 2] - 0.0 + fw.box[2] / 2) % fw.box[2]) - fw.box[2] / 2) < 1.2
ax.scatter(pos[slab & (ty == 0), 0], pos[slab & (ty == 0), 1], s=5, color="#c4c3bd", label="framework O (|z - z_window| < 1.2 A)")
ax.scatter(pos[slab & (ty == 1), 0], pos[slab & (ty == 1), 1], s=6, color="#8a8984", label="Si")
cg = fw.cages; ax.scatter(cg[:, 0], cg[:, 1], s=40, facecolor="none", edgecolor=SERIES[0], lw=1.4, label="ERI cage axes")
win = fw.windows[np.abs(((fw.windows[:, 2] + fw.box[2] / 2) % fw.box[2]) - fw.box[2] / 2) < 0.1]
ph = 2 * math.pi * (win[:, 0] - fw.x_window) / fw.P; ph = (ph + math.pi) % (2 * math.pi) - math.pi
ax.scatter(win[np.abs(ph) < 1e-6, 0], win[np.abs(ph) < 1e-6, 1], s=28, marker="D", color=SERIES[1], label="8-ring windows crossed by phi (phi = 0)")
ax.scatter(win[np.abs(ph) > 1e-6, 0], win[np.abs(ph) > 1e-6, 1], s=28, marker="D", facecolor="none", edgecolor=SERIES[1], label="windows within a cage row (phi = +-pi)")
for k in range(7):
    xv = fw.x_window + k * fw.P
    if xv < fw.box[0]:
        ax.axvline(xv, color=SERIES[1], lw=0.6, ls=":")
ax.set_xlim(0, fw.box[0]); ax.set_ylim(0, fw.box[1]); ax.set_aspect("equal")
ax.set_xlabel("X along hexagonal a1 (A): the CV direction"); ax.set_ylabel("Y (A)")
ax.set_title("Cage-to-cage geometry (xy projection at a window height)\nCV phi = 2 pi (X_com - a/4)/(a/2); dotted lines = phi = 0 planes", fontsize=9)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2, fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_geometry.png")); plt.close(fig)
print("figures ->", OUT)
