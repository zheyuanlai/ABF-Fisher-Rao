"""Render the convergence atlas: L2(F) against time, one convention, every study.

    python scripts/make_convergence_atlas_figures.py

Writes into report/figures/:

    fig_conv_01_atlas.png       every study with both arms, same axes convention
    fig_conv_02_mechanism.png   the two things the atlas alone does not show
    fig_conv_03_speedup.png     time-to-accuracy speedups with censoring made visible
    fig_conv_04_toy_selection.png   why the 2-D toy panel is not evidence

Colour is assigned by ENTITY and fixed across every panel -- an arm keeps its hue
wherever it appears. The palette is Okabe-Ito, validated for CVD separation; the one
adjacent pair inside the 6-8 dE floor band (oracle vs book_laplacian) is additionally
separated by line style and by a direct label at the right edge of every curve, which is
the secondary encoding that band requires.

ABF is deliberately NOT a hue: it is the baseline every panel is read against, so it is
drawn in neutral ink, heavier than the arms. The shams are drawn in light grey because a
control that behaves like the baseline SHOULD recede.

Log y throughout. The question the advisor is asking -- does mFR decay faster, or does it
merely settle to a lower floor -- is a question about the SHAPE of these curves, and on a
linear axis both look the same.
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ATLAS = os.path.join(ROOT, "results", "convergence_atlas")
FIGDIR = os.path.join(ROOT, "report", "figures")

INK, MUTED, FAINT = "#1a1a1a", "#666666", "#b0b0b0"
STYLE = {                                  # entity -> (colour, linestyle, z, label)
    "abf":             (INK,       "-",   5, "ABF"),
    "mfr":             ("#0072B2", "-",   6, "mFR"),
    "count_balancing": ("#D55E00", "-",   6, "count-balancing"),
    "mfr_oracle":      ("#009E73", "-.",  4, "mFR (oracle)"),
    "book_laplacian":  ("#CC79A7", ":",   3, "book Laplacian"),
    "fr_uniform":      ("#E69F00", "--",  3, "FR (uniform)"),
    "mfr_active":      ("#56B4E9", "--",  3, "mFR (aggressive)"),
    "opes":            ("#882255", (0, (3, 1, 1, 1)), 2, "OPES"),
    "sham":            (FAINT,     "--",  2, "matched sham"),
    "sham_oracle":     ("#d5d5d5", ":",   1, "sham (oracle)"),
    "mfr_oracle_r015": ("#56B4E9", "--",  3, "mFR oracle, rate 0.15"),
    "mfr_oracle_r045": ("#E69F00", "--",  3, "mFR oracle, rate 0.45"),
}

MAIN = ["wca_starved", "wca_five_arm", "gateway_left",
        "eb_beta8", "eb_beta12", "eb_beta4",
        "eb_beta2", "toy2d", "butane_phi1",
        "pentane_phi1", "pentane_r15", "alanine"]
DROP_FROM_MAIN = {"pentane_2d", "gateway_one_right"}


def style():
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 200, "font.size": 9,
        "axes.titlesize": 9.5, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
        "axes.grid": True, "grid.alpha": 0.18, "grid.linewidth": 0.6,
        "lines.linewidth": 1.5, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#999999", "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.labelcolor": INK, "text.color": INK,
    })


def verdict(entry):
    """Regime label from the PRIMARY endpoint only, with its own CI -- not from the eye."""
    m, (lo, hi) = entry["rel_integrated_pct"], entry["rel_integrated_ci95"]
    if m <= -5.0 and hi < 0:
        return "mFR FASTER", "#0072B2"
    if m >= +5.0 and lo > 0:
        return "mFR SLOWER", "#D55E00"
    return "no difference", MUTED


def band(ax, t, curves, arm, label_right=True, lw=None):
    c, ls, z, lab = STYLE.get(arm, (MUTED, "-", 1, arm))
    med = np.nanmedian(curves, axis=0)
    lo = np.nanpercentile(curves, 25, axis=0)
    hi = np.nanpercentile(curves, 75, axis=0)
    ax.fill_between(t, lo, hi, color=c, alpha=0.13, linewidth=0, zorder=z - 0.5)
    ax.plot(t, med, color=c, linestyle=ls, zorder=z,
            linewidth=lw if lw else (2.1 if arm == "abf" else 1.5), label=lab)
    return med


def panel(ax, name, d, meta, spd, arms=None, legend=False):
    p = meta[name]
    t = d[f"{name}::times"]
    show = arms or [a for a in p["arms"] if "__alt" not in a]
    ends = []
    for a in show:
        key = f"{name}::{a}"
        if key not in d.files:
            continue
        med = band(ax, t, d[key], a)
        ends.append((med[-1], a))
    ax.set_yscale("log")
    ax.set_xlim(t[0], t[-1])
    if p["t_fr"]:
        ax.axvline(p["t_fr"], color=MUTED, linestyle=(0, (2, 3)), linewidth=0.9, zorder=0.5)
        ax.text(p["t_fr"], ax.get_ylim()[1], " FR on", fontsize=6.5, color=MUTED,
                va="top", ha="left")
    ax.set_title(p["label"], loc="left", pad=13, fontsize=9)
    e = spd["panels"][name]["arms"].get("mfr") or spd["panels"][name]["arms"].get("mfr_oracle")
    if e:
        txt, col = verdict(e)
        ax.text(0, 1.012, f"{txt}   {e['rel_integrated_pct']:+.1f} %  "
                          f"(int. $L_2$, {e['wins']}/{e['n_paired']} seeds)",
                transform=ax.transAxes, fontsize=7, color=col, va="bottom", ha="left")
    ax.set_xlabel(p["x_label"], labelpad=1)
    direct_labels(ax, t[-1], ends)
    if legend:
        ax.legend(frameon=False, loc="lower left", ncol=2, fontsize=7,
                  handlelength=1.8, columnspacing=1.0, borderpad=0.2)


def direct_labels(ax, x_end, ends, min_sep=0.052):
    """Label every curve at its right edge, pushed apart so no two labels overlap.

    Direct labels are not decoration here: they are the secondary encoding that the
    palette's one 6-8 dE adjacent pair and its sub-3:1 contrast slots both require. A
    label that lands on top of another label supplies none of that, so collisions are
    resolved rather than tolerated.
    """
    if not ends:
        return
    lo, hi = ax.get_ylim()
    def to_frac(y):
        if ax.get_yscale() == "log":
            return (np.log10(max(y, 1e-300)) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))
        return (y - lo) / (hi - lo)
    items = sorted(((to_frac(y), a) for y, a in ends), reverse=True)
    sep = min(min_sep, 0.96 / max(len(items), 1))     # always room for the whole stack
    placed = []
    for f, a in items:                       # greedy downward sweep from the top curve
        f = min(f, 0.99)
        if placed and f > placed[-1][0] - sep:
            f = placed[-1][0] - sep
        placed.append((f, a))
    overflow = 0.01 - placed[-1][0]          # stack ran off the bottom: lift it as a block
    if overflow > 0:
        lift = min(overflow, 0.99 - placed[0][0])
        placed = [(f + lift, a) for f, a in placed]
    for f, a in placed:
        c, _, _, lab = STYLE.get(a, (MUTED, "-", 1, a))
        ax.annotate(lab, xy=(1.0, f), xycoords="axes fraction",
                    xytext=(4, 0), textcoords="offset points", color=c,
                    fontsize=6.4, va="center", ha="left", clip_on=False)


def fig_atlas(d, meta, spd):
    style()
    names = [n for n in MAIN if n in meta]
    fig, axes = plt.subplots(4, 3, figsize=(13.2, 15.0))
    seen = []
    for ax, name in zip(axes.ravel(), names):
        panel(ax, name, d, meta, spd)
        seen += [a for a in meta[name]["arms"] if "__alt" not in a]
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$L_2(F)$  [$k_BT$],  log scale")
    order = [a for a in STYLE if a in set(seen)]
    handles = [plt.Line2D([], [], color=STYLE[a][0], linestyle=STYLE[a][1],
                          linewidth=2.1 if a == "abf" else 1.6) for a in order]
    fig.legend(handles, [STYLE[a][3] for a in order], frameon=False, ncol=len(order),
               loc="lower center", bbox_to_anchor=(0.5, -0.004), fontsize=8.5,
               handlelength=2.2, columnspacing=1.6)
    fig.suptitle("Convergence atlas: free-energy error against simulation time, "
                 "every study with both an ABF and an mFR arm",
                 fontsize=13, y=0.998, x=0.008, ha="left")
    fig.text(0.008, 0.9805,
             "Median over seeds, band = IQR. Additive constant removed on each system's "
             "evaluation window, then interior-window RMS -- one convention throughout. "
             "Vertical scales are NOT comparable between panels.",
             fontsize=8, color=MUTED, ha="left")
    fig.tight_layout(rect=[0, 0.017, 1, 0.977])
    out = os.path.join(FIGDIR, "fig_conv_01_atlas.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_mechanism(d, meta, spd):
    """The two things the small multiples hide."""
    style()
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1))

    # (a) five-arm: is the acceleration Fisher-Rao-specific?
    panel(axes[0], "wca_five_arm", d, meta, spd)
    axes[0].set_ylabel(r"$L_2(F)$  [$k_BT$]")
    axes[0].set_title("(a) Is the gain Fisher-Rao-specific?", loc="left", pad=13)

    # (b) gateway: the advantage is EARLY and it reverses by the end
    ax = axes[1]
    t = d["gateway_left::times"]
    for a in ("abf", "mfr", "sham"):
        band(ax, t, d[f"gateway_left::{a}"], a)
    ax.set_yscale("log")
    ax.set_xlim(t[0], t[-1])
    r = d["gateway_left::mfr"] / d["gateway_left::abf"][:len(d["gateway_left::mfr"])]
    ax.set_ylabel(r"$L_2(F)$  [$k_BT$]")
    ax.set_xlabel(meta["gateway_left"]["x_label"], labelpad=1)
    ax.set_title("(b) Entropic gateway: the advantage is early", loc="left", pad=13)
    ax2 = ax.inset_axes([0.44, 0.60, 0.54, 0.36])
    ax2.axhline(1.0, color=MUTED, linewidth=0.8)
    ax2.plot(t, np.median(r, axis=0), color="#0072B2", linewidth=1.4)
    ax2.fill_between(t, np.percentile(r, 25, axis=0), np.percentile(r, 75, axis=0),
                     color="#0072B2", alpha=0.15, linewidth=0)
    ax2.set_ylim(0.3, 1.35)
    ax2.set_title("mFR / ABF error ratio", fontsize=6.5, pad=2, color=MUTED)
    ax2.tick_params(labelsize=6)
    ax2.grid(alpha=0.15)

    # (c) the same contrast on WCA, where it does NOT reverse
    ax = axes[2]
    t = d["wca_starved::times"]
    r = d["wca_starved::mfr"] / d["wca_starved::abf"]
    ax.axhline(1.0, color=MUTED, linewidth=0.9)
    ax.plot(t, np.median(r, axis=0), color="#0072B2", linewidth=1.8, label="WCA (starved)")
    ax.fill_between(t, np.percentile(r, 25, axis=0), np.percentile(r, 75, axis=0),
                    color="#0072B2", alpha=0.15, linewidth=0)
    # Both curves are the SAME entity -- the mFR arm -- so both keep mFR's hue; the system
    # is encoded by line style and by the label, never by borrowing another arm's colour.
    tg = d["gateway_left::times"]
    rg = d["gateway_left::mfr"] / d["gateway_left::abf"]
    ax.plot(tg / tg[-1] * t[-1], np.median(rg, axis=0), color="#0072B2",
            linewidth=1.8, linestyle="--", label="gateway (time rescaled)")
    ax.set_ylim(0.35, 1.35)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylabel("mFR error / ABF error")
    ax.set_xlabel("time (WCA units; gateway rescaled to match)", labelpad=1)
    ax.set_title("(c) Growing advantage vs early-only advantage", loc="left", pad=13)
    ax.legend(frameon=False, loc="lower left", fontsize=7.5)
    ax.annotate("below 1 = mFR ahead", xy=(0.98, 0.96), xycoords="axes fraction",
                ha="right", va="top", fontsize=7, color=MUTED)

    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig_conv_02_mechanism.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_speedup(spd):
    """Time-to-accuracy. Censored arms are DRAWN, at the edge, not omitted."""
    style()
    rows = []
    for name, rec in spd["panels"].items():
        if name in DROP_FROM_MAIN:
            continue
        e = rec["arms"].get("mfr") or rec["arms"].get("mfr_oracle")
        if not e:
            continue
        rows.append((name, rec["label"], e))
    rows.sort(key=lambda r: r[2]["rel_integrated_pct"])
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 0.42 * len(rows) + 2.4),
                             gridspec_kw={"width_ratios": [1.15, 1]})
    y = np.arange(len(rows))

    ax = axes[0]
    for i, (_, _, e) in enumerate(rows):
        lo, hi = e["rel_integrated_ci95"]
        m = e["rel_integrated_pct"]
        c = "#0072B2" if (m <= -5 and hi < 0) else ("#D55E00" if (m >= 5 and lo > 0) else MUTED)
        ax.plot([lo, hi], [i, i], color=c, linewidth=2.4, solid_capstyle="round", alpha=0.55)
        ax.plot([m], [i], "o", color=c, markersize=7, markeredgecolor="white", markeredgewidth=1.1)
        ax.annotate(f"{m:+.1f} %", xy=(m, i), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=6.8, color=c)
    # The comparator that decides whether the acceleration is Fisher-Rao-SPECIFIC. It exists
    # in exactly one study, and omitting it from the summary would let the summary read as
    # stronger evidence for FR than the campaign actually has.
    for i, (name, _, _) in enumerate(rows):
        cb = spd["panels"][name]["arms"].get("count_balancing")
        if not cb:
            continue
        ax.plot([cb["rel_integrated_pct"]], [i], "D", color="#D55E00", markersize=6,
                markeredgecolor="white", markeredgewidth=1.0, zorder=5,
                label="count-balancing (non-FR rule)")
        ax.annotate(f"{cb['rel_integrated_pct']:+.1f} %", xy=(cb["rel_integrated_pct"], i),
                    xytext=(0, -13), textcoords="offset points", ha="center",
                    fontsize=6.8, color="#D55E00")
        ax.legend(frameon=False, loc="lower left", fontsize=7.5)
    ax.axvline(0, color=INK, linewidth=1.0)
    ax.axvspan(-5, 5, color=MUTED, alpha=0.07, linewidth=0)
    ax.set_yticks(y, [r[1] for r in rows], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("change in time-integrated $L_2(F)$ vs ABF  [%]   "
                  "(negative = mFR converges faster)")
    ax.set_title("Primary endpoint, paired by seed, 95 % bootstrap CI", loc="left", pad=8)
    ax.grid(axis="y", alpha=0)

    ax = axes[1]
    keys = ["e0/2", "e0/4", "e0/8", "abf_final"]
    marks = ["o", "s", "^", "D"]
    labs = [r"$\varepsilon=e_0/2$", r"$e_0/4$", r"$e_0/8$", r"ABF's final error"]
    # The thresholds are ORDERED (each is stricter than the last), so this is a sequential
    # encoding -- one hue, light to dark -- not a categorical one. Reusing the arm hues here
    # would make "orange" mean count-balancing in one panel and a threshold in the next.
    ramp = ["#9ecae1", "#6baed6", "#3182bd", "#08306b"]
    for k, mk, lb, col in zip(keys, marks, labs, ramp):
        xs, ys, cens_y = [], [], []
        for i, (_, _, e) in enumerate(rows):
            sp = e["speedup"][k]
            if sp["status"] == "ok" and np.isfinite(sp["median_curve"]) and sp["median_curve"] > 0:
                xs.append(sp["median_curve"]); ys.append(i)
            elif sp["status"] in ("arm_never", "neither_reaches"):
                cens_y.append(i)
        ax.plot(xs, ys, mk, markersize=5.8, label=lb, color=col,
                markeredgecolor="white", markeredgewidth=0.7)
        ax.plot([0.30] * len(cens_y), cens_y, "x", markersize=4.5, color=FAINT)
    ax.axvline(1.0, color=INK, linewidth=1.0)
    ax.set_xscale("log")
    ax.set_xlim(0.25, 22)
    ax.set_xticks([0.5, 1, 2, 5, 10, 20], ["0.5x", "1x", "2x", "5x", "10x", "20x"])
    ax.set_yticks(y, [""] * len(rows))
    ax.invert_yaxis()
    ax.set_xlabel(r"speedup  $S_\varepsilon=\tau_\varepsilon(\mathrm{ABF})/\tau_\varepsilon(\mathrm{mFR})$"
                  "   (>1 = mFR gets there sooner)")
    ax.set_title("Time to a prescribed accuracy", loc="left", pad=8)
    ax.legend(frameon=False, loc="lower right", fontsize=7)
    ax.grid(axis="y", alpha=0)

    fig.suptitle("Does the Fisher-Rao correction make the free energy converge faster?",
                 fontsize=12.5, x=0.008, ha="left", y=1.0)
    fig.text(0.008, -0.012,
             "e0 is the error at t = 0, where every arm carries an identically zero bias -- "
             "so the thresholds are a property of the system, not of the method. "
             "✕ in the left margin of the right panel = mFR never reaches that threshold "
             "at all; those cases are drawn, not dropped.",
             fontsize=7.4, color=MUTED, ha="left", va="top")
    fig.tight_layout(rect=[0, 0.012, 1, 0.97])
    out = os.path.join(FIGDIR, "fig_conv_03_speedup.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_toy_selection(d, meta):
    """The 2-D toy's mFR arm was chosen from 36 configs; ABF had one. Draw all 36."""
    style()
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    t = d["toy2d::times"]
    alts = [k for k in d.files if k.startswith("toy2d::mfr__alt")]
    for i, k in enumerate(alts):
        ax.plot(t, np.median(d[k], axis=0), color="#0072B2", alpha=0.16, linewidth=0.8,
                zorder=1, label="mFR, the 35 configs not selected" if i == 0 else None)
    band(ax, t, d["toy2d::abf"], "abf")
    band(ax, t, d["toy2d::mfr"], "mfr", lw=2.1)
    ax.set_yscale("log")
    ax.set_xlim(t[0], t[-1])
    ax.set_ylabel(r"$L_2(F)$,  log scale")
    ax.set_xlabel(meta["toy2d"]["x_label"])
    ax.set_title("2-D metastability toy: the selection the panel hides", loc="left", pad=14)
    ax.text(0, 1.015, "Each FR arm was tuned over 36 configurations; ABF was run at one. "
                      "The spread below is the tuning budget, not a method effect.",
            transform=ax.transAxes, fontsize=7.5, color=MUTED, va="bottom")
    ax.legend(frameon=False, loc="lower left", fontsize=7.5)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig_conv_04_toy_selection.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    d = np.load(os.path.join(ATLAS, "atlas.npz"))
    meta = {p["panel"]: p for p in json.load(open(os.path.join(ATLAS, "atlas.json")))["panels"]}
    spd = json.load(open(os.path.join(ATLAS, "speedup.json")))
    os.makedirs(FIGDIR, exist_ok=True)
    for f in (fig_atlas(d, meta, spd), fig_mechanism(d, meta, spd),
              fig_speedup(spd), fig_toy_selection(d, meta)):
        print("  wrote", os.path.relpath(f, ROOT))


if __name__ == "__main__":
    main()
