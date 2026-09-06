#!/usr/bin/env python
"""Mechanism figure for the pentane R15 OT confirmatory block (frozen raw outputs only, no reruns).

(a) final walker marginal p(R) for A, F, T against the uniform target on the OT domain;
(b) where OT reallocates sampling effort: deposit-density redistribution T - A from the raw
    production accumulators (csum_prod), with the final-walker Δp(R) as a thin cross-check;
(c) where free-energy information improves: per-bin RMS over 16 seeds of |F'_hat - F'_v2| for A and T
    from the estimator's final smoothed profile, with the preregistered regional RMS summaries.

    python scripts/plot_pentane_r15_ot_mechanism.py
-> results/ot_repair_campaign/pentane_r15/confirmatory/figures/pentane_r15_ot_mechanism.{png,pdf,svg}
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib as mpl                       # noqa: E402
import matplotlib.pyplot as plt                # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
STAGE = os.path.join(ROOT, "results", "ot_repair_campaign", "pentane_r15", "confirmatory")
REF_V2 = os.path.join(ROOT, "cache", "alkanes_cv", "ref_pentane_b2_R15_v2_meanforce.npz")
# validated palette (dataviz reference instance): baseline ink, categorical slot 1 (blue), slot 2 (orange)
COL = {"A": "#52514e", "F": "#2a78d6", "T": "#eb6834"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dddcd7"
R_MARKS = (2.17, 2.75, 3.17)


def load():
    runs = {}
    for f in sorted(glob.glob(os.path.join(STAGE, "raw", "*.npz"))):
        d = np.load(f, allow_pickle=True)
        runs[str(d["arm"])] = d
    return runs


def main():
    runs = load(); A, F, T = runs["abf"], runs["fr"], runs["ot"]
    v2 = np.load(REF_V2, allow_pickle=True); grid = np.asarray(v2["grid"]); dz = float(v2["dz"]); win = np.asarray(v2["window_mask"]); Fp2 = np.asarray(v2["Fprime"])
    w_lo, w_hi = float(v2["window_lo"]), float(v2["window_hi"])
    lo, hi = np.asarray(T["ot_domain"]).tolist(); U = 1.0 / (hi - lo)
    mech = json.load(open(os.path.join(STAGE, "mechanism_extras.json")))
    summ = json.load(open(os.path.join(STAGE, "summary.json")))
    c = summ["contrasts"]["T vs A"]
    n_seeds = int(A["n_seeds"])

    # (a) final walker marginal: mean over seeds of the per-seed reflected-KDE marginal (each normalised)
    p_fin = {k: np.asarray(d["p_hat"])[-1].mean(0) for k, d in (("A", A), ("F", F), ("T", T))}
    # (b) deposit density from the raw production accumulators, pooled over seeds, normalised on the grid
    dep = {}
    for k, d in (("A", A), ("F", F), ("T", T)):
        cs = np.asarray(d["csum_prod"]).sum(0); dep[k] = cs / (cs.sum() * dz)
    d_dep = dep["T"] - dep["A"]; d_fin = p_fin["T"] - p_fin["A"]
    # (c) per-bin RMS over seeds of the final smoothed mean-force error
    rms = {}
    for k, d in (("A", A), ("T", T)):
        mf = np.asarray(d["mean_force"])[-1]                          # (seeds, G)
        rms[k] = np.sqrt(np.mean((mf - Fp2[None, :]) ** 2, axis=0))
    mA, mT = mech["A"], mech["T"]

    mpl.rcParams.update({"font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 10, "legend.fontsize": 8,
                         "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.edgecolor": INK2, "axes.linewidth": 0.8,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK, "text.color": INK,
                         "font.family": "DejaVu Sans", "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 5.05), layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.878), w_pad=0.08)

    def dress(ax, ylabel, title, letter):
        ax.axvspan(w_lo, w_hi, color="#f3f2ee", zorder=0, lw=0)
        for x in (lo, hi):
            ax.axvline(x, color=INK2, lw=0.7, ls=(0, (2, 3)), zorder=1)
        ax.set_xlim(1.9, 3.72); ax.set_xlabel("end-to-end distance  $R_{15}$")
        ax.set_ylabel(ylabel); ax.set_title(title, loc="left", fontweight="bold")
        ax.text(-0.12, 1.03, letter, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom")
        ax.grid(True, axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    def legend_below(ax, ncol, **kw):
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=ncol, handlelength=2.4,
                  columnspacing=1.2, **kw)

    # ---------------- (a) ----------------
    ax = axes[0]
    dress(ax, "walker density  $p(R)$  at the end of the run", "Marginal allocation along $R_{15}$", "a")
    ax.hlines(U, lo, hi, color=INK2, lw=1.2, ls="--", zorder=2, label="uniform target on the OT domain")
    ax.plot(grid, p_fin["A"], color=COL["A"], lw=2.2, label="A  ABF", zorder=3)
    ax.plot(grid, p_fin["F"], color=COL["F"], lw=1.4, ls=(0, (3, 2)), label="F  uniform FR (≈ A)", zorder=4)
    ax.plot(grid, p_fin["T"], color=COL["T"], lw=2.2, label="T  OT (capped, α 0.01)", zorder=5)
    ymax_a = 1.12 * max(p_fin[k].max() for k in p_fin); ax.set_ylim(0, ymax_a)
    ax.annotate("over-populated\ncompact edge", xy=(2.19, p_fin["A"][np.argmin(np.abs(grid - 2.19))]), xytext=(2.36, 0.93 * ymax_a),
                fontsize=8, color=INK2, ha="left", va="top", arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
    ax.annotate("extended region\nunder-sampled by A,\nrepopulated by T", xy=(3.3, p_fin["T"][np.argmin(np.abs(grid - 3.3))]), xytext=(3.12, 0.50 * ymax_a),
                fontsize=8, color=INK2, ha="left", va="top", arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
    ax.text(0.5 * (w_lo + w_hi), 0.012 * ymax_a, f"evaluation window [{w_lo:.2f}, {w_hi:.2f}]", ha="center", va="bottom", fontsize=7.3, color=INK2)
    legend_below(ax, 2)

    # ---------------- (b) ----------------
    ax = axes[1]
    dress(ax, "$\\Delta$ density of deposited samples,  T − A", "Where OT reallocates sampling effort", "b")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.fill_between(grid, 0, d_dep, where=d_dep > 0, color=COL["T"], alpha=0.35, lw=0, label="more sampling under T")
    ax.fill_between(grid, 0, d_dep, where=d_dep < 0, color="#9c9a93", alpha=0.45, lw=0, label="less sampling under T")
    ax.plot(grid, d_dep, color=COL["T"], lw=1.9, label="deposits (raw accumulators, 16 seeds pooled)")
    ax.plot(grid, d_fin, color=INK2, lw=1.0, ls=":", label="final walkers, $p_T - p_A$")
    ymax = 1.25 * np.nanmax(np.abs(d_dep[win])); ax.set_ylim(-ymax, ymax)
    for x in R_MARKS:
        ax.axvline(x, color=INK, lw=0.7, ls=(0, (4, 3)), zorder=1)
        ax.text(x + 0.015, 0.97 * ymax, f"R = {x}", rotation=90, ha="left", va="top", fontsize=7.3, color=INK)
    moved = float(np.sum(d_dep[grid >= 2.75]) * dz)
    txt = (f"A → T, medians (n = {n_seeds})\n"
           f"low-support fraction  {mA['low_support_fraction']:.3f} → {mT['low_support_fraction']:.3f}\n"
           f"walkers R < 2.17   {100 * mA['frac_compact_end']:.1f}% → {100 * mT['frac_compact_end']:.1f}%\n"
           f"walkers R > 3.17    {100 * mA['frac_extended_end']:.1f}% → {100 * mT['frac_extended_end']:.1f}%\n"
           f"KL(p ‖ uniform)    {mA['kl_final']:.2f} → {mT['kl_final']:.2f}\n"
           f"{100 * moved:.1f}% of deposits moved to R ≥ 2.75")
    ax.text(0.975, 0.04, txt, transform=ax.transAxes, va="bottom", ha="right", fontsize=7.2, color=INK,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=GRID, lw=0.8, alpha=0.95), zorder=6)
    legend_below(ax, 2, fontsize=7.4)

    # ---------------- (c) ----------------
    ax = axes[2]
    dress(ax, "RMS over 16 seeds of  $|\\hat F'(R) - F'_{\\rm ref}(R)|$", "Where free-energy information improves", "c")
    ax.fill_between(grid, rms["T"], rms["A"], where=(rms["T"] < rms["A"]) & win, color=COL["T"], alpha=0.30, lw=0, label="T more accurate")
    ax.fill_between(grid, rms["T"], rms["A"], where=(rms["T"] >= rms["A"]) & win, color="#9c9a93", alpha=0.40, lw=0, label="A more accurate")
    ax.plot(grid[win], rms["A"][win], color=COL["A"], lw=2.0, label="A  ABF")
    ax.plot(grid[win], rms["T"][win], color=COL["T"], lw=2.0, label="T  OT")
    ymax_c = 1.75 * max(rms["A"][win].max(), rms["T"][win].max()); ax.set_ylim(0, ymax_c)
    ax.axvline(2.75, color=INK, lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.text(2.72, 0.985 * ymax_c, "R < 2.75: compact half,\ntorsional bottleneck\nregional RMS\nA %.2f → T %.2f\nunchanged" % (mA["mf_rms_err_R_lt_275"], mT["mf_rms_err_R_lt_275"]),
            ha="right", va="top", fontsize=7.6, color=INK, linespacing=1.25)
    ax.text(2.78, 0.985 * ymax_c, "R ≥ 2.75: mid / extended,\nrepopulated by OT\nregional RMS\nA %.2f → T %.2f\n−%.0f%%" % (mA["mf_rms_err_R_ge_275"], mT["mf_rms_err_R_ge_275"], 100 * (1 - mT["mf_rms_err_R_ge_275"] / mA["mf_rms_err_R_ge_275"])),
            ha="left", va="top", fontsize=7.6, color=INK, linespacing=1.25)
    legend_below(ax, 4, fontsize=7.4)

    fig.text(0.01, 0.975, "Pentane $R_{15}$, β = 2, 16 fresh seeds — gentle Wasserstein reallocation re-balances the marginal; the free-energy gain sits where sampling was added",
             fontsize=10.8, fontweight="bold", ha="left", va="top")
    fig.text(0.01, 0.938, (f"T vs A:  ΔI_F^(C) = {100 * c['I_F']['median']:+.1f}% [{100 * c['I_F']['ci95'][0]:+.1f}, {100 * c['I_F']['ci95'][1]:+.1f}], "
                           f"{c['I_F']['wins']}/{c['I_F']['n']};  D_cond unchanged ({100 * c['D_star']['median']:+.1f}%).   "
                           "Torsional transitions and round trips unchanged; T contains no repair step — the mechanism is marginal sample reallocation."),
             fontsize=8.3, color=INK2, ha="left", va="top")
    fig.text(0.01, 0.912, "Corrected mean-force reference v2; evaluation window [2.08, 3.54] shaded; dotted verticals = OT target domain [2.02, 3.65]; "
             "the same frozen confirmatory raw outputs as the paired analysis, no reruns.", fontsize=7.6, color=INK2, ha="left", va="top")
    out = os.path.join(STAGE, "figures", "pentane_r15_ot_mechanism")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{out}.{ext}", dpi=300 if ext == "png" else None)
    print("wrote", out + ".{png,pdf,svg}")
    print(f"regional RMS A/T: R<2.75 {mA['mf_rms_err_R_lt_275']:.2f}/{mT['mf_rms_err_R_lt_275']:.2f}; R>=2.75 {mA['mf_rms_err_R_ge_275']:.2f}/{mT['mf_rms_err_R_ge_275']:.2f}; "
          f"deposit mass T-A below 2.75: {np.sum(d_dep[grid < 2.75]) * dz:+.3f}, above: {np.sum(d_dep[grid >= 2.75]) * dz:+.3f}")


if __name__ == "__main__":
    main()
