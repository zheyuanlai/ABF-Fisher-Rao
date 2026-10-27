"""Figures of the finite-loading LTA study -> figures/lta_loading/.

  reference   the loaded tagged-guest F(z) vs the single-guest exact-MC F(z)
  ladder      median e_F vs walker-steps per N (small multiples), ABF vs FR, IQR bands, mid threshold
  confirm     b_mid distributions of the two confirmed arms (if the confirmation exists)
"""
from __future__ import annotations

import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

import analyze as A  # noqa: E402
from lta_loading import engine as LE  # noqa: E402

OUT = os.path.join(ROOT, "figures", "lta_loading")
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
COL = {"abf": "#2a78d6", "fr": "#eb6834", "single": "#52514e"}
NAME = {"abf": "ABF", "fr": "ABF + FR"}

plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 2.0,
                     "legend.frameon": False, "savefig.dpi": 200})


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)


def fig_reference():
    z = np.load(os.path.join(A.RES, "reference", "reference.npz"))
    single = np.load(os.path.join(ROOT, "results/equal_budget_v2/references/lta_T150_reference.npz"))
    a = 11.919
    x = z["grid_phi"] * a / (2 * np.pi)
    kT = 0.008314462618 * 150.0
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for F, se, key, lab in ((single["F_ref"], None, "single", "1 guest per box (exact MC)"),
                            (z["F_ref"], z["F_ref_se"], "fr", "16 guests per box (this study)")):
        F = (F - F.min()) / kT
        ax.plot(x, F, color=COL[key], lw=2, label=lab)
        if se is not None:
            ax.fill_between(x, F - 2 * se / kT, F + 2 * se / kT, color=COL[key], alpha=0.25, lw=0)
    ax.set_xlabel("tagged-guest position along the cage–window–cage axis, z (Å)")
    ax.set_ylabel("F(z) − min F  (kT, 150 K)")
    ax.legend(loc="upper left")
    ax.set_title("Ethane in LTA at 150 K: loading barely moves the window barrier", color=INK, fontsize=9.5)
    save(fig, "fig_reference")


def ladder_curves():
    F_ref, _ = A.reference()
    out = {}
    for p in sorted(glob.glob(os.path.join(A.RES, "ladder", "N*", "*.npz"))):
        res, meta = LE.load_npz(p)
        e = A.errors(res, F_ref)
        out[(meta["method"], int(meta["N"]))] = (res["save_step"] * float(meta["N"]), e)
    return out


def fig_ladder():
    cur = ladder_curves()
    Ns = sorted({k[1] for k in cur})
    if not Ns:
        return
    fig, axes = plt.subplots(1, len(Ns), figsize=(2.3 * len(Ns), 2.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, N in zip(axes, Ns):
        for m in ("abf", "fr"):
            if (m, N) not in cur:
                continue
            b, e = cur[(m, N)]
            sel = b > 0
            med = np.median(e, axis=1)
            q1, q3 = np.percentile(e, [25, 75], axis=1)
            ax.fill_between(b[sel] / 1e6, q1[sel], q3[sel], color=COL[m], alpha=0.2, lw=0)
            ax.plot(b[sel] / 1e6, med[sel], color=COL[m], lw=1.8, label=NAME[m])
        ax.axhline(A.THR["mid"], color=INK2, lw=0.8, ls="--")
        ax.set_yscale("log")
        ax.set_title(f"N = {N}", color=INK, fontsize=9)
        ax.set_xlabel("walker-steps (10⁶)")
    axes[0].set_ylabel("e_F (kJ/mol), median and IQR of 16 seeds")
    axes[0].text(0.45, A.THR["mid"] * 1.12, "mid accuracy 0.13", color=INK2, fontsize=7.5,
                 transform=axes[0].get_yaxis_transform())
    axes[-1].legend(loc="upper right")
    fig.suptitle("Finite-loading LTA, 150 K: free-energy error vs equal walker-step budget", color=INK, fontsize=10,
                 y=1.04)
    save(fig, "fig_ladder")


def fig_confirm():
    path = os.path.join(A.RES, "confirm", "confirm.json")
    if not os.path.exists(path):
        return
    d = json.load(open(path))
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    for i, m in enumerate(("abf", "fr")):
        rows = [r for r in d["rows"] if r["method"] == m]
        if not rows:
            continue
        v = np.array([r["b_mid"] for r in rows], float) / 1e6
        fin = np.sort(v[np.isfinite(v)])
        ecdf = np.arange(1, len(fin) + 1) / len(v)
        ax.step(fin, ecdf, where="post", color=COL[m], lw=2, label=f"{NAME[m]}, N = {rows[0]['N']}")
    ax.set_xlabel("walker-steps to sustained mid accuracy, b_mid (10⁶)")
    ax.set_ylabel("fraction of seeds")
    R = d["R_B"]
    ax.set_title(f"Confirmation, 64 seeds per arm: R_B = {R[0]:.2f} [{R[1]:.2f}, {R[2]:.2f}]", color=INK,
                 fontsize=9.5)
    ax.legend(loc="lower right")
    save(fig, "fig_confirm")


if __name__ == "__main__":
    fig_reference()
    fig_ladder()
    fig_confirm()
    print("figures ->", OUT)
