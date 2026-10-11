#!/usr/bin/env python
"""Experiment A synthesis figures A1, A2, A3, A5, A6, A7, A8 (docs/final_campaign/PREREGISTRATION.md section 1.8).

    python scripts/final_campaign/expA_figures.py [--cells-dir configs/final_campaign/cells/expA]
        [--synthesis-dir results/final_campaign/expA/synthesis] [--fig-dir figures/final_campaign/expA]
        [--dpi 160] [--strict-legibility] [--no-verify-runs] [--provisional] [--force] [--recompute] [--dry-run]

Reads ONLY outputs of the analysis tools (nothing is recomputed except plotting positions): <synthesis-dir>/
cross_cell.json (expA_cross_cell.py), <synthesis-dir>/mech.json (expA_mech.py) and each cell's
<analysis_dir>/median_curves.npz (scripts/equal_budget/analyze_ladder.py).  A4 / A4b (MC1 / MC2) are drawn by the
validation tool; the standard per-cell figures by scripts/equal_budget/plot_config.py.

  A1_Ibar_F_vs_lambda          seed-median Ibar_F of ABF and ABF+FR vs lambda, one panel per N, bootstrap 95 % CIs
                               (log y: the arms differ by a few per cent while lambda may move both)
  A2_final_eF_vs_lambda        the same for the final e_F
  A3_ratio_and_contrasts       top: FR/ABF ratio exp(median L) vs lambda with CIs (log-ratio axis, 1 = no FR
                               effect); bottom: the E1 / E2 (95 % and Holm-consistent 97.5 % CI) and T1-T3 contrasts
                               R on a log axis with the [0.90, 1.11] equivalence band, Holm / unadjusted p in the labels
  A5_young_clone_bias          per N: FR B_c by clone-age class and ABF's B vs lambda (debiased RMS, reflected CI); the
                               young-clone ([0.01, 0.1) t.u.) matched excess over ABF vs lambda; MC3 = its change from
                               lambda 1 with CI (filled marker = MC3 passes: Holm p over N < 0.05 and estimate < 0)
  A6_marginal_TV_half          per N: seed-median Ibar_TV_half of both arms vs lambda and the paired G(Ibar_TV_half)
                               with its CI (HA3: upper CI < 0)
  A7_eF_convergence            seed-median e_F(u) with the interquartile band, ABF vs ABF+FR, one panel per (lambda, N),
                               common y range (log), budget fraction u on a log axis
  A8_genealogy                 FR's time-averaged windowed ESS/N, maximum windowed lineage fraction and realised deaths
                               per walker per t.u. vs lambda (seed medians, CIs), one series per N
Each figure is written as PNG and PDF and checked by scripts/equal_budget/fig_legibility.check_figure_safe (axis
labels, legends, overlapping or cut-off text, legends over data); the result is stored in <fig-dir>/
MANIFEST_expA_synthesis.json with the sha256 of every input and the input_signature.  --strict-legibility exits 1 on
any failure (after writing everything).  Fixture inputs are stamped on every figure.  Output policy as
expA_analysis_common.decide_write (decided on the manifest).  cross_cell.json and mech.json must have been made from
the CURRENT cell configs and summaries (sha256; otherwise the script stops: re-run cross / mech first), and the
summaries must be current and fresh (cross_cell.check_summary / check_freshness; --no-verify-runs skips freshness and
forbids complete).  Writing order: a placeholder manifest (state 'writing', no signature, complete false), then every
figure (each written atomically after the disk-space check), then the final manifest with every file's sha256.
complete = cross_cell.json complete AND mech.json complete AND the CURRENT expA_analysis_common.completeness() (every
cell's completeness audit PASS on the current summary / config, frozen engine only, ...; the audit status is part of
the input signature) AND every figure passes the legibility check AND no --provisional (expA_analyze.py passes it
when the synthesis runs over a failed audit with --allow-audit-fail).
"""
from __future__ import annotations

import argparse
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import expA_analysis_common as C  # noqa: E402
import numpy as np  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import fig_legibility as LG  # noqa: E402
import mech_analysis as MA  # noqa: E402  (style helpers only: setup_style, new_fig, title, fig_legend, lam_axis ...)

SCHEMA = "final_expA_figures/1"
MANIFEST = "MANIFEST_expA_synthesis.json"
C_ABF, C_FR = MA.C_ABF, MA.C_FR
N_PALETTE = ("#08306b", "#6a51a3", "#4292c6", "#9e9ac8")
N_MARKERS = ("s", "^", "o", "D")
CODE_FILES = [os.path.abspath(__file__), os.path.join(HERE, "expA_analysis_common.py"),
              os.path.join(C.ROOT, "scripts", "mechanism", "mech_analysis.py"),
              os.path.join(C.ROOT, "scripts", "equal_budget", "fig_legibility.py")]
OFF = 1.13                                    # multiplicative lambda offset between the arms on the log axis


class Writer:
    def __init__(self, out_dir, dpi):
        self.out_dir, self.dpi = out_dir, dpi
        self.entries = []

    def save(self, fig, name, description, waivers=()):
        C.guard_output(self.out_dir)
        os.makedirs(self.out_dir, exist_ok=True)
        files = []
        for ext in ("pdf", "png"):
            p = os.path.join(self.out_dir, f"{name}.{ext}")
            C.save_figure(fig, p, dpi=self.dpi if ext == "png" else None)     # atomic, after the disk check
            files.append(os.path.basename(p))
        res = LG.check_figure_safe(fig, waivers=waivers)
        plt.close(fig)
        if res["status"] != "pass":
            print("  " + LG.one_line(name, res), file=sys.stderr, flush=True)
        self.entries.append(dict(id=name.split("_")[0], name=name, description=description, files=files,
                                 file_sha256={f: C.sha256_file(os.path.join(self.out_dir, f)) for f in files},
                                 legibility=res))


# ============================================================================================ helpers
def lams_present(blk):
    return sorted(float(k) for k in blk)


def Ns_of(doc):
    return [str(n) for n in doc["N"]]


def n_style(i):
    return N_PALETTE[i % len(N_PALETTE)], N_MARKERS[i % len(N_MARKERS)]


def banners(fixture):
    return (["FIXTURE: synthetic tiny-budget data, pipeline check only, never production evidence"]
            if fixture else [])


def no_data(ax, why="no data"):
    ax.text(0.5, 0.5, why, transform=ax.transAxes, ha="center", va="center", fontsize=8.5, color=MA.MUTED)
    ax.set_xticks([])
    ax.set_yticks([])


def set_log_y(ax, vals, pad=1.4):
    v = np.asarray([x for x in vals if x is not None], float)
    v = v[np.isfinite(v) & (v > 0)]
    if v.size == 0:
        return False
    ax.set_yscale("log")
    lo, hi = v.min() / pad, v.max() * pad
    if hi / lo < 1.5:
        mid = math.sqrt(lo * hi)
        lo, hi = mid / 1.25, mid * 1.25
    ax.set_ylim(lo, hi)
    MA.log_ticks(ax, "y")
    return True


def set_ratio_y(ax, vals, extra=(1.0,)):
    v = np.asarray([x for x in list(vals) + list(extra) if x is not None], float)
    v = v[np.isfinite(v) & (v > 0)]
    lo, hi = (v.min(), v.max()) if v.size else (0.9, 1.11)
    lo, hi = min(lo, 0.9) / 1.08, max(hi, 1.11) * 1.08
    ax.set_yscale("log")
    ax.set_ylim(lo, hi)
    MA.ratio_ticks(ax)


def errbar(ax, x, mid, ci, color, marker="o", filled=True, ms=6):
    lo, hi = (C.fnum(ci[0]), C.fnum(ci[1])) if ci else (math.nan, math.nan)
    if math.isinf(hi):
        hi = math.nan
    MA.errpt(ax, x, C.fnum(mid), lo, hi, color, marker=marker, ms=ms, filled=filled)


def finite_vals(*xs):
    out = []
    for x in xs:
        if isinstance(x, (list, tuple)):
            out += [C.fnum(v) for v in x]
        else:
            out.append(C.fnum(x))
    return [v for v in out if math.isfinite(v)]


# ============================================================================================ A1 / A2 / A6 (top)
def fig_arm_medians(W, cc, metric, name, ylabel, desc, title_line):
    Ns = Ns_of(cc)
    fig, axs = MA.new_fig(4.2 * len(Ns) + 0.6, 4.0, 1, len(Ns))
    lamsall = [float(l) for l in cc["lams"]]
    for j, N in enumerate(Ns):
        ax = axs[0][j]
        vals = []
        any_ = False
        for lk, byN in cc["per_cell"].items():
            b = (byN.get(N) or {}).get(metric)
            if not b:
                continue
            lam = float(lk)
            for arm, col, f in (("abf", C_ABF, 1 / OFF), ("fr", C_FR, OFF)):
                e = b[arm]
                if math.isfinite(C.fnum(e["median"])):
                    errbar(ax, lam * f, e["median"], e["ci95"], col)
                    vals += finite_vals(e["median"], e["ci95"])
                    any_ = True
        if not any_:
            no_data(ax)
            continue
        MA.lam_axis(ax, lamsall)
        set_log_y(ax, vals)
        ax.set_xlabel("transverse mobility λ")
        ax.set_ylabel(ylabel)
        ax.set_title(f"N = {N}")
    MA.fig_legend(fig, [Line2D([], [], color=C_ABF, marker="o", lw=0, label="ABF (seed median, 95 % CI)"),
                        Line2D([], [], color=C_FR, marker="o", lw=0, label="ABF + FR (seed median, 95 % CI)")], 2)
    MA.title(fig, [title_line, "seed bootstrap of the median (10 000 resamples); the two arms are offset in λ for "
                               "readability"], fig.get_figwidth(), banners(cc.get("fixture")))
    W.save(fig, name, desc)


def fig_A6(W, cc):
    Ns = Ns_of(cc)
    fig, axs = MA.new_fig(4.2 * len(Ns) + 0.6, 7.2, 2, len(Ns))
    lamsall = [float(l) for l in cc["lams"]]
    for j, N in enumerate(Ns):
        ax, axg = axs[0][j], axs[1][j]
        vals, gv, any_ = [], [], False
        for lk, byN in cc["per_cell"].items():
            b = (byN.get(N) or {}).get("Ibar_TV_half")
            if not b:
                continue
            lam = float(lk)
            for arm, col, f in (("abf", C_ABF, 1 / OFF), ("fr", C_FR, OFF)):
                e = b[arm]
                if math.isfinite(C.fnum(e["median"])):
                    errbar(ax, lam * f, e["median"], e["ci95"], col)
                    vals += finite_vals(e["median"], e["ci95"])
                    any_ = True
            g = b["G"]
            if math.isfinite(C.fnum(g["median"])):
                hi = C.fnum(g["ci95"][1])
                errbar(axg, lam, 100 * C.fnum(g["median"]), [100 * C.fnum(g["ci95"][0]), 100 * hi], MA.INK,
                       filled=bool(hi < 0))
                gv += finite_vals(100 * C.fnum(g["median"]), [100 * C.fnum(x) for x in g["ci95"]])
        if not any_:
            no_data(ax)
            no_data(axg)
            continue
        MA.lam_axis(ax, lamsall)
        set_log_y(ax, vals)
        ax.set_xlabel("transverse mobility λ")
        ax.set_ylabel("Ibar_TV_half (marginal)")
        ax.set_title(f"N = {N}: arm medians")
        MA.lam_axis(axg, lamsall)
        axg.axhline(0.0, color=MA.MUTED, lw=0.9, ls="--", zorder=1)
        if gv:
            lo, hi = min(gv + [0.0]), max(gv + [0.0])
            pad = 0.08 * (hi - lo) if hi > lo else 1.0
            axg.set_ylim(lo - pad, hi + pad)
        axg.set_xlabel("transverse mobility λ")
        axg.set_ylabel("G(Ibar_TV_half) = (FR - ABF)/ABF  [%]")
        axg.set_title(f"N = {N}: paired G (filled: upper CI < 0, HA3 holds)")
    MA.fig_legend(fig, [Line2D([], [], color=C_ABF, marker="o", lw=0, label="ABF"),
                        Line2D([], [], color=C_FR, marker="o", lw=0, label="ABF + FR"),
                        Line2D([], [], color=MA.INK, marker="o", lw=0, label="G median, 95 % CI")], 3)
    MA.title(fig, ["A6: reaction-coordinate marginal Ibar_TV_half vs λ and its FR gain (HA3)",
                   "seed medians with bootstrap 95 % CIs; G: paired per seed (analysis summary)"],
             fig.get_figwidth(), banners(cc.get("fixture")))
    W.save(fig, "A6_marginal_TV_half", "Ibar_TV_half arm medians and G(Ibar_TV_half) vs lambda per N (HA3)")


# ============================================================================================ A3
def fig_A3(W, cc):
    Ns = Ns_of(cc)
    fig, axs = MA.new_fig(5.0 * len(Ns) + 0.6, 8.4, 2, len(Ns), height_ratios=[1.0, 1.15])
    lamsall = [float(l) for l in cc["lams"]]
    rows = cc["contrasts"]["Ibar_F"]["rows"]
    for j, N in enumerate(Ns):
        ax, axf = axs[0][j], axs[1][j]
        vals = []
        for lk, byN in cc["per_cell"].items():
            b = (byN.get(N) or {}).get("Ibar_F")
            if b and math.isfinite(C.fnum(b["ratio"])):
                errbar(ax, float(lk), b["ratio"], b["ratio_ci95"], C_FR)
                vals += finite_vals(b["ratio"], b["ratio_ci95"])
        if not vals:
            no_data(ax)
        else:
            MA.lam_axis(ax, lamsall)
            ax.axhline(1.0, color=MA.MUTED, lw=0.9, ls="--", zorder=1)
            set_ratio_y(ax, vals)
            ax.set_xlabel("transverse mobility λ")
            ax.set_ylabel("Ibar_F^FR / Ibar_F^ABF  (exp median L)")
            ax.set_title(f"N = {N}: FR/ABF ratio per cell (< 1: FR better)")
        sel = [r for r in rows if str(r["N"]) == N]
        ylab, xs = [], []
        for i, r in enumerate(sel):
            y = len(sel) - 1 - i
            ok = math.isfinite(C.fnum(r["ratio"]))
            ptxt = (f"Holm p {C.fmt(r.get('p_holm'), 2)}" if r["kind"] == "family" else f"p {C.fmt(r['p'], 2)}")
            ylab.append((y, f"{r['id']}  λ{r['a']} vs λ{r['b']}\n{ptxt}, n {r['n']}"))
            if not ok:
                continue
            mid = C.fnum(r["ratio"])
            lo, hi = (C.fnum(v) for v in r["ratio_ci95"])
            col = MA.INK if r["kind"] == "family" else MA.MUTED
            if r.get("ratio_ci975") and r["kind"] == "family":
                l2, h2 = (C.fnum(v) for v in r["ratio_ci975"])
                if math.isfinite(l2) and math.isfinite(h2):
                    axf.plot([l2, h2], [y, y], color=col, lw=1.0, zorder=3)
                    xs += [l2, h2]
            axf.plot([lo, hi], [y, y], color=col, lw=3.0, solid_capstyle="butt", zorder=4)
            axf.plot([mid], [y], marker="o", color=col, ms=6, zorder=5)
            xs += [lo, hi, mid]
        if not xs:
            no_data(axf)
            continue
        axf.axvspan(C.EQUIV[0], C.EQUIV[1], color="#d9d9d9", alpha=0.55, lw=0, zorder=1)
        axf.axvline(1.0, color=MA.MUTED, lw=0.9, ls="--", zorder=2)
        v = np.array(xs + [C.EQUIV[0], C.EQUIV[1]], float)
        v = v[np.isfinite(v) & (v > 0)]
        axf.set_xscale("log")
        axf.set_xlim(v.min() / 1.06, v.max() * 1.06)
        axf.set_yticks([y for y, _ in ylab])
        axf.set_yticklabels([t for _, t in ylab], fontsize=8)
        axf.set_ylim(-0.6, len(sel) - 0.4)
        axf.grid(axis="y", visible=False)
        lo_, hi_ = axf.get_xlim()
        cand = np.array([0.5, 0.67, 0.8, 0.9, 1.0, 1.11, 1.25, 1.5, 2.0])
        t = cand[(cand >= lo_) & (cand <= hi_)]
        if t.size >= 2:
            axf.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(t))
            axf.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda val, _: f"{val:g}"))
            axf.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        axf.set_xlabel("R  (< 1: FR relatively better at λ_a)")
        axf.set_ylabel("contrast")
        axf.set_title(f"N = {N}: E1 / E2 (Holm over N) and trends T1-T3")
    MA.fig_legend(fig, [Line2D([], [], color=C_FR, marker="o", lw=0, label="exp(median L), 95 % CI"),
                        Line2D([], [], color=MA.INK, lw=3, label="E1 / E2: 95 % CI (thick), 97.5 % CI (thin)"),
                        Line2D([], [], color=MA.MUTED, lw=3, label="T1-T3: 95 % CI (unadjusted)"),
                        Patch(color="#d9d9d9", label="equivalence band [0.90, 1.11]")], 2)
    MA.title(fig, ["A3: FR/ABF Ibar_F ratio vs λ and the preregistered contrasts",
                   "L = log(Ibar_F^FR / Ibar_F^ABF) per seed; seeds coupled across λ; paired seed bootstrap"],
             fig.get_figwidth(), banners(cc.get("fixture")))
    W.save(fig, "A3_ratio_and_contrasts", "FR/ABF Ibar_F ratio vs lambda and the E1/E2/T1-T3 contrasts with CIs")


# ============================================================================================ A5
def fig_A5(W, mech):
    Ns = Ns_of(mech)
    fig, axs = MA.new_fig(13.0, 3.9 * len(Ns) + 0.8, len(Ns), 3)
    lamsall = [float(l) for l in mech["lams"]]
    names = list(MA.AGE_DEFAULT)
    for i, N in enumerate(Ns):
        a1, a2, a3 = axs[i]
        v1, v2, v3 = [], [], []
        for lk, byN in mech["d2"].items():
            b = byN.get(N) or {}
            if "fr" not in b:
                continue
            lam = float(lk)
            for c in range(MA.N_AGE + 1):
                col = MA.CLASS_COLOR[c] if c < MA.N_AGE else C_FR
                f = lam * OFF ** ((c - 2.5) / 2.5)
                mid = C.fnum(b["fr"]["B"][c])
                ci = [b["fr"]["B_ci95"][0][c], b["fr"]["B_ci95"][1][c]]
                if math.isfinite(mid):
                    errbar(a1, f, mid, ci, col, ms=4.5, marker="o" if c < MA.N_AGE else "D")
                    v1 += finite_vals(mid, ci)
            ab = b["abf"]
            if math.isfinite(C.fnum(ab["B"])):
                errbar(a1, lam * OFF ** 1.6, ab["B"], ab["B_ci95"], C_ABF, marker="s", ms=4.5)
                v1 += finite_vals(ab["B"], ab["B_ci95"])
            y = b["young_clone"]
            if math.isfinite(C.fnum(y["B_excess"])):
                errbar(a2, lam, y["B_excess"], y["B_excess_ci95"], MA.CLASS_COLOR[1])
                v2 += finite_vals(y["B_excess"], y["B_excess_ci95"])
        for lk, blk in mech["mc3"].items():
            r = (blk.get("per_N") or {}).get(N) or {}
            if "B_excess_did" in r and math.isfinite(C.fnum(r["B_excess_did"])):
                errbar(a3, float(lk), r["B_excess_did"], r["B_excess_did_ci95"], MA.INK,
                       filled=bool(r.get("passes")))
                v3 += finite_vals(r["B_excess_did"], r["B_excess_did_ci95"])
        for ax, v, yl, tt in ((a1, v1, "debiased RMS bias B_c of deposits", f"N = {N}: FR B_c by clone age; ABF B"),
                              (a2, v2, "young-clone matched excess over ABF",
                               f"N = {N}: young clones [0.01, 0.1) t.u."),
                              (a3, v3, "MC3: excess(λ) - excess(λ = 1)", f"N = {N}: MC3 change from λ = 1")):
            if not v:
                no_data(ax)
                continue
            MA.lam_axis(ax, lamsall)
            lo, hi = min(v + [0.0]), max(v + [0.0])
            pad = 0.08 * (hi - lo) if hi > lo else max(abs(hi), 1e-6) * 0.2
            ax.set_ylim(lo - pad, hi + pad)
            if ax is not a1:
                ax.axhline(0.0, color=MA.MUTED, lw=0.9, ls="--", zorder=1)
            ax.set_xlabel("transverse mobility λ")
            ax.set_ylabel(yl)
            ax.set_title(tt)
    hs = [Line2D([], [], color=MA.CLASS_COLOR[c], marker="o", lw=0,
                 label=(f"FR, clone age {names[c]} t.u." if c < 4 else "FR, never cloned")) for c in range(MA.N_AGE)]
    hs += [Line2D([], [], color=C_FR, marker="D", lw=0, label="FR, all deposits"),
           Line2D([], [], color=C_ABF, marker="s", lw=0, label="ABF, all deposits"),
           Line2D([], [], color=MA.INK, marker="o", lw=0, label="MC3 passes (Holm p < 0.05, < 0)"),
           Line2D([], [], color=MA.INK, marker="o", mfc="white", lw=0, label="MC3 does not pass")]
    MA.fig_legend(fig, hs, 4)
    MA.title(fig, ["A5: age-dependent deposit bias (D2) vs λ and the young-clone manipulation check MC3",
                   "end snapshot; debiased RMS vs the bin-averaged F*' (reflected seed-bootstrap CI); matched excess "
                   "= class error minus ABF error at the class's positions"],
             fig.get_figwidth(), banners(mech.get("fixture")))
    W.save(fig, "A5_young_clone_bias", "D2 B_c by clone-age class, young-clone matched excess and MC3 vs lambda")


# ============================================================================================ A7
def fig_A7(W, cells, Ns, fixture):
    lams = sorted(cells)
    fig, axs = MA.new_fig(3.2 * len(lams) + 0.6, 3.1 * len(Ns) + 0.9, len(Ns), len(lams), sharey=True, sharex=True)
    allv = []
    curves = {}
    for lam in lams:
        p = os.path.join(C.MF.cell_analysis_dir(cells[lam]["P"]), "median_curves.npz")
        if os.path.exists(p):
            with np.load(p, allow_pickle=False) as z:
                curves[lam] = {k: np.asarray(z[k]) for k in z.files}
    for i, N in enumerate(Ns):
        for j, lam in enumerate(lams):
            ax = axs[i][j]
            z = curves.get(lam)
            if z is None or f"N{N}_abf_e_F_median" not in z:
                no_data(ax, "no median curves")
                continue
            u = np.asarray(z["u"], float)
            for arm, col in (("abf", C_ABF), ("fr", C_FR)):
                k = f"N{N}_{arm}_e_F"
                if f"{k}_median" not in z:
                    continue
                med, lo, hi = (MA.pos(z[f"{k}_{q}"]) for q in ("median", "q25", "q75"))
                ax.fill_between(u, lo, hi, color=col, alpha=0.18, lw=0)
                ax.plot(u, med, color=col, lw=1.5)
                allv += [x for x in np.concatenate([lo, hi]) if np.isfinite(x)]
            ax.set_xscale("log")
            ax.set_title(f"λ = {lam:g}, N = {N}")
            if i == len(Ns) - 1:
                ax.set_xlabel("budget fraction u")
            if j == 0:
                ax.set_ylabel("e_F (seed median, IQR)")
    if allv:
        for row in axs:
            for ax in row:
                if ax.lines:
                    set_log_y(ax, allv, pad=1.2)
                    break
    MA.fig_legend(fig, [Line2D([], [], color=C_ABF, lw=2, label="ABF (median, IQR band)"),
                        Line2D([], [], color=C_FR, lw=2, label="ABF + FR (median, IQR band)")], 2)
    MA.title(fig, ["A7: free-energy error e_F(u) convergence, ABF vs ABF + FR, per (λ, N)",
                   "seed median and interquartile band over the 200 uniform budget fractions; common y range"],
             fig.get_figwidth(), banners(fixture))
    W.save(fig, "A7_eF_convergence", "median e_F(u) curves, ABF vs FR, one panel per (lambda, N)")


# ============================================================================================ A8
def fig_A8(W, mech):
    Ns = Ns_of(mech)
    fig, axs = MA.new_fig(13.0, 4.2, 1, 3)
    lamsall = [float(l) for l in mech["lams"]]
    specs = (("ess_win_tavg", "time-averaged windowed ESS/N", "FR ancestor ESS/N (t >= window)"),
             ("maxfam_win_max", "maximum windowed lineage fraction", "FR largest lineage share"),
             ("fr_deaths_per_walker_per_t", "realised deaths per walker per t.u.", "FR death rate"))
    hs = []
    for k, (key, yl, tt) in enumerate(specs):
        ax = axs[0][k]
        v = []
        for i, N in enumerate(Ns):
            col, mk = n_style(i)
            f = OFF ** (i - (len(Ns) - 1) / 2)
            for lk, byN in mech["genealogy"].items():
                e = (byN.get(N) or {}).get(key)
                if e and math.isfinite(C.fnum(e["median"])):
                    errbar(ax, float(lk) * f, e["median"], e["ci95"], col, marker=mk)
                    v += finite_vals(e["median"], e["ci95"])
            if k == 0:
                hs.append(Line2D([], [], color=col, marker=mk, lw=0, label=f"N = {N} (seed median, 95 % CI)"))
        if not v:
            no_data(ax)
            continue
        MA.lam_axis(ax, lamsall)
        lo, hi = min(v), max(v)
        pad = 0.08 * (hi - lo) if hi > lo else max(abs(hi), 1e-3) * 0.1
        ax.set_ylim(max(0.0, lo - pad) if lo >= 0 else lo - pad, hi + pad)
        ax.set_xlabel("transverse mobility λ")
        ax.set_ylabel(yl)
        ax.set_title(tt)
    MA.fig_legend(fig, hs, len(hs))
    MA.title(fig, ["A8: FR genealogy vs λ (guard g3 compares λ with λ = 1 at each N)",
                   "ESS/N averaged over the uniform-u saves after the first genealogy window; lineage fraction = max "
                   "over all saves"], fig.get_figwidth(), banners(mech.get("fixture")))
    W.save(fig, "A8_genealogy", "FR genealogy (ESS/N time average, max lineage fraction, deaths/walker/t.u.) vs lambda")


# ============================================================================================ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells-dir", default=C.CELLS_DIR)
    ap.add_argument("--synthesis-dir", default=C.SYN_DIR)
    ap.add_argument("--fig-dir", default=C.FIG_DIR)
    ap.add_argument("--dpi", type=int, default=160)
    ap.add_argument("--strict-legibility", action="store_true")
    ap.add_argument("--no-verify-runs", action="store_true",
                    help="skip the summaries' run-tree freshness check (forbids complete)")
    ap.add_argument("--provisional", action="store_true",
                    help="force complete = false (the synthesis runs over a failed completeness audit)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    sd, fd = os.path.abspath(a.synthesis_dir), os.path.abspath(a.fig_dir)
    cp, mp = os.path.join(sd, "cross_cell.json"), os.path.join(sd, "mech.json")
    cc, mech = C.load_signed(cp, "cross_cell.json"), C.load_signed(mp, "mech.json")
    cells = C.load_cells(a.cells_dir)
    if not cells:
        raise C.ExpAError(f"no cell config under {a.cells_dir}")
    verify = not a.no_verify_runs
    diff = C.synthesis_input_problems(cells, cross_cell=cc, mech=mech)
    curves = {C.lam_key(l): os.path.join(C.MF.cell_analysis_dir(r["P"]), "median_curves.npz")
              for l, r in sorted(cells.items())}
    ins = dict(cross_cell=C.sha256_file(cp), mech=C.sha256_file(mp),
               median_curves={k: (C.sha256_file(p) if os.path.exists(p) else None) for k, p in curves.items()})
    # complete needs the inputs complete AND the CURRENT completeness basis (an audit may have changed since)
    extra = [f"{nm} is not complete" for nm, d in (("cross_cell.json", cc), ("mech.json", mech)) if not d.get("complete")]
    comp = C.completeness(cells, None, verify, a.provisional, extra_reasons=extra)
    sig = C.digest(dict(schema=SCHEMA, inputs=ins, code=C.code_digests(CODE_FILES), dpi=a.dpi, verify_runs=verify,
                        audits=C.audit_signature(comp["audits"]), provisional=bool(a.provisional)))
    mpath = os.path.join(fd, MANIFEST)
    if a.dry_run:
        try:
            decision = C.decide_write(mpath, sig, a.force, a.recompute)
        except C.ExpAError as e:
            decision = f"REFUSE ({e})"
        sp = C.summary_problems(cells, verify)
        print(f"complete-eligible {comp['complete']}; would {decision} figures A1 A2 A3 A5 A6 A7 A8 in {fd} (signature "
              f"{sig[:16]})" + (f"; inputs not current: would stop ({diff[:2]})" if diff else "")
              + (f"; summaries not current for lambda {[f'{l:g}' for l in sp]}: would stop" if sp else ""))
        return 0
    if diff:
        raise C.ExpAError("cross_cell.json / mech.json were not made from the current cell configs and summaries: "
                          + "; ".join(diff[:6]) + " (re-run expA_cross_cell.py and expA_mech.py)")
    C.require_current_summaries(cells, verify)          # also before skipping
    decision = C.decide_write(mpath, sig, a.force, a.recompute)
    if decision == "skip":
        print(f"{mpath} is up to date (signature {sig[:16]})")
        return 0
    prev = C.previous_signature(mpath)
    C.begin_output(mpath, SCHEMA, prev)
    MA.setup_style()
    W = Writer(fd, a.dpi)
    fig_arm_medians(W, cc, "Ibar_F", "A1_Ibar_F_vs_lambda", "Ibar_F (integrated free-energy error)",
                    "seed-median Ibar_F of both arms vs lambda per N with bootstrap CIs",
                    "A1: integrated free-energy error Ibar_F vs transverse mobility λ")
    fig_arm_medians(W, cc, "final_e_F", "A2_final_eF_vs_lambda", "final e_F (u = 1)",
                    "seed-median final e_F of both arms vs lambda per N with bootstrap CIs",
                    "A2: final free-energy error e_F(u = 1) vs transverse mobility λ")
    fig_A3(W, cc)
    fig_A5(W, mech)
    fig_A6(W, cc)
    fig_A7(W, cells, [str(n) for n in cc["N"]], cc.get("fixture"))
    fig_A8(W, mech)
    leg = LG.summarize({e["name"]: e["legibility"] for e in W.entries})
    bad = [e["name"] for e in W.entries if e["legibility"]["status"] != "pass"]
    if bad:                                            # PREREGISTRATION 1.8: every figure is checked for legibility
        comp = C.completeness(cells, None, verify, a.provisional,
                              extra_reasons=extra + [f"legibility check failed for {bad}"])
    man = dict(schema=SCHEMA, input_signature=sig, provenance=C.provenance(CODE_FILES), inputs=ins,
               input_paths=dict(cross_cell=C.rel(cp), mech=C.rel(mp), median_curves={k: C.rel(p) for k, p in curves.items()}),
               complete=comp["complete"], completeness=comp, provisional=bool(a.provisional),
               freshness_verified=verify, inputs_complete=dict(cross_cell=cc.get("complete"), mech=mech.get("complete")),
               fixture=bool(cc.get("fixture") or mech.get("fixture")), figures=W.entries, legibility_summary=leg,
               not_drawn_here=dict(A4="MC1 conditional relaxation: validation tool", A4b="MC2 gate transport: validation tool"))
    if prev and prev != sig:
        man["replaced_signature"] = prev
    C.write_json(mpath, man)
    print(f"wrote {len(W.entries)} figures to {fd}; legibility failures: {bad or 'none'} (complete {man['complete']})")
    return 1 if (bad and a.strict_legibility) else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except C.ExpAError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)
