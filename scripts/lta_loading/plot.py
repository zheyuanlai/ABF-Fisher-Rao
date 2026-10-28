"""Figures of the finite-loading LTA study -> figures/lta_loading/.

  reference   the loaded tagged-guest F(z) vs the single-guest exact-MC F(z)
  ladder      median e_F vs walker-steps per N (small multiples), ABF vs FR, IQR bands, mid threshold
  confirm     b_mid distributions of the two confirmed arms (if the confirmation exists)
  fixedN      e_F at equal N, ABF vs FR, against physical time per walker and modelled A100 wall-clock
              (ladder, every N; confirmation, N = 64); median curves also in fixedN_convergence.csv
"""
from __future__ import annotations

import csv
import glob
import json
import math
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
    """Two panels: the primary walker-step confirmation (frozen arms) and the A3 modelled wall-clock confirmation."""
    p1 = os.path.join(A.RES, "confirm", "confirm.json")
    p2 = os.path.join(A.RES, "confirm_wall", "confirm_wall.json")
    if not (os.path.exists(p1) and os.path.exists(p2)):
        return
    d1, d2 = json.load(open(p1)), json.load(open(p2))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.0))
    for ax, d, key, scale, xl, Rk in ((axes[0], d1, "b_mid", 1e6, "walker-steps to sustained mid accuracy (10⁶)", "R_B"),
                                      (axes[1], d2, "W_mid", 1.0, "modelled A100 seconds to sustained mid accuracy",
                                       "R_W")):
        for m in ("abf", "fr"):
            rows = [r for r in d["rows"] if r["method"] == m]
            v = np.array([r[key] for r in rows], float) / scale
            fin = np.sort(v[np.isfinite(v)])
            ax.step(fin, np.arange(1, len(fin) + 1) / len(v), where="post", color=COL[m], lw=2,
                    label=f"{NAME[m]}, N = {rows[0]['N']}")
        R = d[Rk]
        ax.set_title(f"{Rk} = {R[0]:.2f} [{R[1]:.2f}, {R[2]:.2f}] — {d['decision']}", color=INK, fontsize=9)
        ax.set_xlabel(xl)
        ax.legend(loc="lower right")
    axes[0].set_ylabel("fraction of 64 seeds")
    fig.suptitle("Finite-loading LTA, 150 K: confirmation on fresh seeds", color=INK, fontsize=10, y=1.04)
    save(fig, "fig_confirm")


AXIS = {"t": "physical time per walker, t = k h (t.u.)",
        "W": "modelled A100 wall-clock, k × s/step (s)"}


def fixed_n_curves(pattern):
    """{(method, N): curves} of the batch files matching pattern, against step k, physical time t = k h and modelled
    wall-clock W = k c_ss(method, N) (median s/step of one seed on one A100, bench.json; the Amendment A3 model)."""
    F_ref, _ = A.reference()
    c_ss = {k: v["median_c_ss"] for k, v in
            json.load(open(os.path.join(A.RES, "bench", "bench.json")))["summary"].items()}
    out = {}
    for p in sorted(glob.glob(pattern)):
        res, meta = LE.load_npz(p)
        m, N = meta["method"], int(meta["N"])
        k = res["save_step"].astype(float)
        e = A.errors(res, F_ref)
        first = [A.persistent_first(e[:, r] <= A.THR["mid"]) for r in range(e.shape[1])]
        out[(m, N)] = dict(k=k, e=e, k_mid=np.array([math.inf if i is None else k[i] for i in first]),
                           scale={"t": float(meta["cfg"]["h"]), "W": c_ss[f"{m}_N{N}"]})
    return out


def fixed_n_panel(ax, cur, N, axis):
    """Median and IQR of e_F for ABF and FR at one N; dot + dotted line at the median time to persistent mid
    accuracy (which can exceed the median curve's first crossing); the box gives those medians."""
    notes = []
    for m in ("abf", "fr"):
        if (m, N) not in cur:
            continue
        d = cur[(m, N)]
        x = d["k"] * d["scale"][axis]
        sel = x > 0
        q1, med, q3 = np.percentile(d["e"], [25, 50, 75], axis=1)
        ax.fill_between(x[sel], q1[sel], q3[sel], color=COL[m], alpha=0.2, lw=0)
        ax.plot(x[sel], med[sel], color=COL[m], lw=1.6, label=NAME[m])
        xm = A.median_inf(d["k_mid"]) * d["scale"][axis]
        n_c, n = int(np.sum(~np.isfinite(d["k_mid"]))), len(d["k_mid"])
        if math.isfinite(xm):
            ax.axvline(xm, color=COL[m], lw=0.9, ls=":")
            ax.plot([xm], [A.THR["mid"]], "o", color=COL[m], ms=5, mec="white", mew=1, zorder=5)
            txt = f"{xm:.3g}" + (" t.u." if axis == "t" else " s")
        else:
            txt = "not reached"
        notes.append(f"{NAME[m]}: {txt}" + (f" ({n_c}/{n} cens.)" if n_c else ""))
    ax.axhline(A.THR["mid"], color=INK2, lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.text(0.03, 0.03, "median to mid accuracy\n" + "\n".join(notes), transform=ax.transAxes, ha="left",
            va="bottom", fontsize=6.5, color=INK, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRID, lw=0.6))


def write_fixed_n_csv(sets):
    with open(os.path.join(OUT, "fixedN_convergence.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["set", "method", "N", "n_seeds", "step", "t_tu", "W_s", "eF_q25", "eF_median", "eF_q75"])
        for name, cur in sets:
            for (m, N), d in sorted(cur.items(), key=lambda kv: (kv[0][1], kv[0][0])):
                q1, med, q3 = np.percentile(d["e"], [25, 50, 75], axis=1)
                for i, k in enumerate(d["k"]):
                    w.writerow([name, m, N, d["e"].shape[1], int(k), f"{k * d['scale']['t']:.6g}",
                                f"{k * d['scale']['W']:.6g}", f"{q1[i]:.6g}", f"{med[i]:.6g}", f"{q3[i]:.6g}"])


def fig_fixed_n():
    """Equal N, ABF vs FR: e_F against physical time (top) and modelled A100 wall-clock (bottom). At fixed N the
    two x axes differ only by FR's extra cost per step (+19-25 %), which shifts its curves right in wall-clock."""
    lad = fixed_n_curves(os.path.join(A.RES, "ladder", "N*", "*.npz"))
    Ns = sorted({k[1] for k in lad})
    if not Ns:
        return
    fig, axes = plt.subplots(2, len(Ns), figsize=(2.6 * len(Ns), 5.6), sharey=True)
    for c, N in enumerate(Ns):
        for r, axis in enumerate(("t", "W")):
            fixed_n_panel(axes[r, c], lad, N, axis)
            axes[r, c].set_xlabel(AXIS[axis], fontsize=7.5)
        axes[0, c].set_title(f"N = {N}", color=INK, fontsize=9)
    for r in range(2):
        axes[r, 0].set_ylabel("e_F (kJ/mol)")
    axes[0, 0].legend(loc="upper right", frameon=True, facecolor="white", edgecolor=GRID, framealpha=1)
    fig.suptitle("Finite-loading LTA, 150 K, equal number of walkers: e_F vs physical time (top) and modelled "
                 "A100 wall-clock (bottom)\nmedian and IQR of 16 seeds; dashed: mid accuracy 0.13 kJ/mol; "
                 "wall-clock = steps × measured s/step of one seed (start-up excluded)", color=INK, fontsize=9.5,
                 y=1.02)
    fig.tight_layout()
    save(fig, "fig_fixedN_convergence")

    con = fixed_n_curves(os.path.join(A.RES, "confirm", "*.npz"))
    sets = [("ladder", lad)]
    if con:
        sets.append(("confirm", con))
        N = sorted({k[1] for k in con})[0]
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.2), sharey=True)
        for ax, axis in zip(axes, ("t", "W")):
            fixed_n_panel(ax, con, N, axis)
            ax.set_xlabel(AXIS[axis])
        axes[0].set_ylabel("e_F (kJ/mol)")
        axes[0].legend(loc="upper right", frameon=True, facecolor="white", edgecolor=GRID, framealpha=1)
        n = con[("abf", N)]["e"].shape[1]
        fig.suptitle(f"Confirmation seeds, both methods at N = {N}: e_F vs physical time and modelled A100 "
                     f"wall-clock\nmedian and IQR of {n} seeds; dashed: mid accuracy 0.13 kJ/mol", color=INK,
                     fontsize=9.5, y=1.07)
        save(fig, "fig_fixedN_confirm")
    write_fixed_n_csv(sets)


if __name__ == "__main__":
    fig_reference()
    fig_ladder()
    fig_confirm()
    fig_fixed_n()
    print("figures ->", OUT)
