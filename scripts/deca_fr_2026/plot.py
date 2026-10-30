"""Figures of the 2026-10 deca-alanine study, from analyze.py outputs (no new numbers here).

    python scripts/deca_fr_2026/plot.py --config configs/deca_fr_2026/<stage>.yaml \
        [--reference results/deca_fr_2026/reference/reference.npz]

Two series at most per panel (ABF = blue solid, ABF+FR = vermillion dashed; Okabe-Ito, legend on
every multi-series panel), medians with inter-quartile bands over seeds, recessive grids.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                             # noqa: E402
import numpy as np                                                          # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C                                                          # noqa: E402

KB = 0.008314462618
STYLE = {"abf": dict(color="#0072B2", ls="-", label="ABF"),
         "fr_uniform": dict(color="#D55E00", ls="--", label="ABF + FR")}
plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 9, "axes.titlesize": 9, "legend.fontsize": 7.5,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "lines.linewidth": 1.6,
                     "figure.dpi": 150, "savefig.dpi": 200})


def _band(ax, t, c, style):
    ax.fill_between(t, c["q25"], c["q75"], color=style["color"], alpha=0.18, lw=0)
    ax.plot(t, c["median"], color=style["color"], ls=style["ls"], label=style["label"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--reference", default="results/deca_fr_2026/reference/reference.npz")
    ap.add_argument("--tag", default="", help="read analysis_<tag>/ and write figures_<tag>/")
    args = ap.parse_args()
    cfg = C.load_yaml(args.config)
    stage = cfg.get("stage")
    out_dir = args.out or cfg["out"]
    sfx = f"_{args.tag}" if args.tag else ""
    an = os.path.join(out_dir, "analysis" + sfx)
    fig_dir = os.path.join(out_dir, "figures" + sfx)
    os.makedirs(fig_dir, exist_ok=True)
    with open(os.path.join(an, f"{stage}_summary.json")) as fh:
        S = json.load(fh)
    cz = np.load(os.path.join(an, f"{stage}_curves.npz"))
    curves = {k: cz[k] for k in cz.files}
    ref = np.load(args.reference) if os.path.exists(args.reference) else None
    kT = KB * 300.0
    cells = list(S["cells"])

    def cv(cell, m, k):
        return {q: curves[f"{cell}/{m}/{k}/{q}"] for q in ("median", "q25", "q75")}

    # ---- Figure 1: e_F(t) and e_F'(t)
    if ref is not None:
        fig, axes = plt.subplots(2, len(cells), figsize=(3.2 * len(cells), 4.6), squeeze=False)
        for j, cell in enumerate(cells):
            for m in S["cells"][cell]["arms"]:
                t = curves[f"{cell}/{m}/times"] / 1000.0
                _band(axes[0, j], t, cv(cell, m, "e_F"), STYLE[m])
                _band(axes[1, j], t, cv(cell, m, "e_Fp"), STYLE[m])
            axes[0, j].set_title(cell); axes[0, j].set_ylabel("$e_F$ (kJ/mol)"); axes[0, j].set_yscale("log")
            axes[1, j].set_ylabel("$e_{F'}$ (kJ/mol/nm)"); axes[1, j].set_xlabel("t (ns)"); axes[1, j].set_yscale("log")
            if len(S["cells"][cell]["arms"]) > 1:
                axes[0, j].legend(frameon=False)
        fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "fig1_convergence.png")); plt.close(fig)

    # ---- Figure 2: profiles at the end (reference, both arms), and the error profiles
    if ref is not None:
        rgrid, F_ref, Fp_ref = ref["grid"], ref["F_ref"], ref["Fprime_ref"]
        rmask = (rgrid >= 1.30) & (rgrid <= 3.50)
        fig, axes = plt.subplots(2, len(cells), figsize=(3.2 * len(cells), 4.6), squeeze=False)
        for j, cell in enumerate(cells):
            axes[0, j].plot(rgrid, (F_ref - F_ref[rmask].mean()) / kT, color="#222222", lw=1.2, label="reference")
            axes[1, j].plot(rgrid, Fp_ref, color="#222222", lw=1.2, label="reference")
            for m in S["cells"][cell]["arms"]:
                d = np.load(os.path.join(out_dir, cell, f"{m}.npz"))
                grid = d["grid"]; mask = (grid >= 1.30) & (grid <= 3.50)   # the arm's own grid
                pmf = d["pmf"][-1]
                pmf = pmf - pmf[:, mask].mean(-1, keepdims=True)
                axes[0, j].plot(grid, np.median(pmf, 0) / kT, **{k: v for k, v in STYLE[m].items()})
                axes[1, j].plot(grid, np.median(d["mean_force"][-1], 0), **{k: v for k, v in STYLE[m].items()})
            axes[0, j].set_title(cell); axes[0, j].set_ylabel("F (kT)"); axes[0, j].legend(frameon=False)
            axes[1, j].set_ylabel("F' (kJ/mol/nm)"); axes[1, j].set_xlabel("R (nm)")
            for a in axes[:, j]:
                a.axvspan(1.2, 1.3, color="#dddddd", alpha=0.5, lw=0); a.axvspan(3.5, 3.6, color="#dddddd", alpha=0.5, lw=0)
        fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "fig2_profiles.png")); plt.close(fig)

    # ---- Figure 3: marginal TV(t) and conditional TV per interval (second half)
    fig, axes = plt.subplots(2, len(cells), figsize=(3.2 * len(cells), 4.6), squeeze=False)
    for j, cell in enumerate(cells):
        arms = S["cells"][cell]["arms"]
        for m, r in arms.items():
            t = curves[f"{cell}/{m}/times"] / 1000.0
            _band(axes[0, j], t, cv(cell, m, "marginal_tv"), STYLE[m])
            if "cond_tv_per_interval_second_half" in r:
                pi = np.array(r["cond_tv_per_interval_second_half"], float)
                x = 1.4 + 0.2 * np.arange(pi.shape[1])
                axes[1, j].errorbar(x + (0.03 if m == "fr_uniform" else -0.03), np.nanmedian(pi, 0),
                                    yerr=[np.nanmedian(pi, 0) - np.nanpercentile(pi, 25, 0),
                                          np.nanpercentile(pi, 75, 0) - np.nanmedian(pi, 0)],
                                    fmt="o" if m == "abf" else "s", ms=4, color=STYLE[m]["color"],
                                    label=STYLE[m]["label"], capsize=2, lw=1)
        axes[0, j].set_title(cell); axes[0, j].set_ylabel("TV(p, q_wall)"); axes[0, j].set_xlabel("t (ns)")
        axes[1, j].set_ylabel("TV p(Y|R) vs ref, 2nd half"); axes[1, j].set_xlabel("R interval centre (nm)")
        if len(arms) > 1:
            axes[0, j].legend(frameon=False); axes[1, j].legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "fig3_marginal_conditional.png")); plt.close(fig)

    # ---- Figure 4: genealogy
    fig, axes = plt.subplots(1, len(cells), figsize=(3.2 * len(cells), 2.5), squeeze=False)
    for j, cell in enumerate(cells):
        arms = S["cells"][cell]["arms"]
        for m, r in arms.items():
            t = curves[f"{cell}/{m}/times"] / 1000.0
            _band(axes[0, j], t, cv(cell, m, "ess"), STYLE[m])
            if m == "fr_uniform":
                for s in r["fr_stop_ps"]:
                    axes[0, j].axvline(s / 1000.0, color=STYLE[m]["color"], alpha=0.25, lw=0.8)
        axes[0, j].set_title(cell); axes[0, j].set_ylabel("ancestor ESS / K"); axes[0, j].set_xlabel("t (ns)")
        axes[0, j].set_ylim(0, 1.02)
        if len(arms) > 1:
            axes[0, j].legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "fig4_genealogy.png")); plt.close(fig)

    # ---- Figure 5: equal budget
    eb = S.get("equal_budget", {})
    if eb:
        fig, axes = plt.subplots(1, len(eb), figsize=(3.2 * len(eb), 2.6), squeeze=False)
        for j, (B, per_m) in enumerate(eb.items()):
            for m, rows in per_m.items():
                Ks = sorted(rows.values(), key=lambda r: r["K"])
                axes[0, j].errorbar([r["K"] for r in Ks], [r["e_F_median"] for r in Ks],
                                    yerr=[[r["e_F_median"] - np.percentile(r["e_F"], 25) for r in Ks],
                                          [np.percentile(r["e_F"], 75) - r["e_F_median"] for r in Ks]],
                                    fmt="o-" if m == "abf" else "s--", color=STYLE[m]["color"],
                                    label=STYLE[m]["label"], ms=4, capsize=2)
            axes[0, j].set_xscale("log", base=2); axes[0, j].set_title(f"budget {B} walker-steps")
            axes[0, j].set_xlabel("K"); axes[0, j].set_ylabel("$e_F$ (kJ/mol)"); axes[0, j].legend(frameon=False)
        fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "fig5_equal_budget.png")); plt.close(fig)
    print(f"-> {fig_dir}")


if __name__ == "__main__":
    main()
