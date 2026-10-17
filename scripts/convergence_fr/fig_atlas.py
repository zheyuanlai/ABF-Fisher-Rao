"""Experiment 1 figures from results/convergence_fr_campaign/atlas/atlas_summary.json (ABF vs ABF + FR only).

    python scripts/convergence_fr/fig_atlas.py   -> figures/convergence_fr/{convergence_eF, convergence_mean_force,
        running_integrated_error, time_to_accuracy, speedup_vs_tolerance, early_population_establishment}.{pdf,png}

Bands: light = 25-75 % quantiles over seeds; dark = 95 % bootstrap CI of the median (2000 resamples of seeds).
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "eri_campaign"))
from figstyle import ABF, FR, INK2, MUTED, setup  # noqa: E402

plt = setup()
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
FIG = os.path.join(ROOT, "figures", "convergence_fr")
ORDER = ["LTA150", "LTA300", "ITE300", "ITE150"]
ARMS = (("abf", "ABF", ABF), ("fr", "ABF + FR", FR))


def title(ax, r, extra=""):
    ax.set_title(f"{r['label']}  ·  K {r['K']}  ·  T {r['T']:g} t.u.  ·  {r['bins']} bins", fontsize=9.5, loc="left")
    ax.text(0.0, 1.005, "", transform=ax.transAxes)


def vlines(ax, r, ymax=None):
    ax.axvline(r["t_fr"], color=MUTED, ls=":", lw=1.0)
    if r.get("t_stop"):
        ax.axvline(r["t_stop"], color=FR, ls="--", lw=1.0)


def band(ax, t, c, color, label, log=True):
    m = np.array(c["median"], float)
    ax.fill_between(t, c["q25"], c["q75"], color=color, alpha=0.13, lw=0)
    ax.fill_between(t, c["ci_lo"], c["ci_hi"], color=color, alpha=0.35, lw=0)
    ax.plot(t, m, color=color, lw=1.6, label=label)


def footer(fig, S):
    lines = []
    for k in ORDER:
        r = S[k]
        lines.append(f"{r['label']}: {r['model']}; {r['status']}; FR starts at {r['t_fr']:g} t.u. (dotted).")
    fig.text(0.01, -0.02, "\n".join(lines) + "\nBands: light = 25–75 % of seeds, dark = 95 % bootstrap CI of the "
             "median. Errors vs the exact MC reference; additive constant removed.", fontsize=6.6, color=INK2,
             va="top")


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def conv(S, key, name, ylabel):
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.6))
    for ax, k in zip(axs, ORDER):
        r = S[k]
        t = np.array(r["curves"]["t"])
        for arm, lab, col in ARMS:
            band(ax, t, r["curves"][f"{key}_{arm}"], col, lab)
        ax.set_yscale("log")
        vlines(ax, r)
        title(ax, r)
        ax.set_xlabel("t (t.u.)")
        if k == "ITE150":
            ax.axvline(50, color=INK2, lw=0.8, ls="-.")
            ax.text(51, ax.get_ylim()[1] * 0.7, "T = 50", fontsize=7, color=INK2)
    axs[0].set_ylabel(ylabel)
    axs[0].legend(loc="upper right")
    footer(fig, S)
    fig.tight_layout()
    save(fig, name)


def running(S):
    fig, axs = plt.subplots(2, 4, figsize=(15, 6.2), sharex="col")
    for j, k in enumerate(ORDER):
        r = S[k]
        t = np.array(r["curves"]["t"])
        ax = axs[0, j]
        for arm, lab, col in ARMS:
            band(ax, t, r["curves"][f"ibar_{arm}"], col, lab)
        ax.set_yscale("log")
        vlines(ax, r)
        title(ax, r)
        ax = axs[1, j]
        for nm, ls, lab in (("rel_ibar_full", "-", "full horizon  Ī_F(t)"), ("rel_ibar_post", "--", "post-ramp  Ī_F(t)")):
            c = r["curves"][nm]
            m = 100 * np.array(c["median"], float)
            ax.fill_between(t, 100 * np.array(c["ci_lo"], float), 100 * np.array(c["ci_hi"], float), color=FR,
                            alpha=0.18 if ls == "-" else 0.10, lw=0)
            ax.plot(t, m, color=FR, ls=ls, lw=1.5, label=lab)
        ax.axhline(0, color=INK2, lw=0.8)
        vlines(ax, r)
        ax.set_xlabel("t (t.u.)")
        e = r["endpoints"]
        ax.text(0.98, 0.04, f"T: full {100 * e['Ibar_F_full']['median_rel']:+.1f} %  "
                f"post-ramp {100 * e['Ibar_F_post_ramp']['median_rel']:+.1f} %", transform=ax.transAxes,
                ha="right", fontsize=7.5, color=INK2)
    axs[0, 0].set_ylabel("running Ī_F(t) (kJ/mol)")
    axs[1, 0].set_ylabel("per-seed Ī_F(FR)/Ī_F(ABF) − 1 (%)\nmedian over seeds")
    axs[0, 0].legend(loc="upper right")
    axs[1, 0].legend(loc="upper right", fontsize=7)
    footer(fig, S)
    fig.tight_layout()
    save(fig, "running_integrated_error")


def tta(S, ladder):
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.9))
    for ax, k in zip(axs, ORDER):
        r = S[k]
        T = r["T"]
        rungs = list(r["time_to_accuracy"]["sustained"].items())
        for i, (ek, st) in enumerate(rungs):
            for off, arm, col in ((-0.17, "abf", ABF), (0.17, "fr", FR)):
                tau = np.array(st[f"tau_{arm}"], float)
                fin = np.isfinite(tau)
                x = i + off + np.linspace(-0.07, 0.07, len(tau))
                ax.scatter(x[fin], tau[fin], s=10, color=col, lw=0, alpha=0.85)
                ax.scatter(x[~fin], np.full((~fin).sum(), T * 1.04), s=12, facecolor="white", edgecolor=col, lw=0.8)
                if fin.sum() * 2 > len(tau):
                    ax.hlines(np.median(np.minimum(tau, T)), i + off - 0.12, i + off + 0.12, color=col, lw=2.2)
            ax.text(i, T * 1.12, f"S {st['S_median']:.2f}", ha="center", fontsize=6.8, color=INK2)
        ax.axhline(T, color=MUTED, lw=0.8)
        ax.set_xticks(range(len(rungs)))
        ax.set_xticklabels([f"{st['eps_kJmol']:.3g}\n({float(ek[:-2]):.3f} kT)" for ek, st in rungs], fontsize=6.8)
        ax.set_ylim(0, T * 1.18)
        ax.set_xlabel("tolerance ε, kJ/mol (k_BT)")
        title(ax, r)
    axs[0].set_ylabel("sustained time to accuracy τ_ε (t.u.)")
    h = [axs[0].scatter([], [], s=10, color=ABF, label="ABF"), axs[0].scatter([], [], s=10, color=FR, label="ABF + FR"),
         axs[0].scatter([], [], s=12, facecolor="white", edgecolor=INK2, label="censored (never sustained by T)")]
    fig.legend(handles=h, loc="upper center", ncol=3, fontsize=7.5, bbox_to_anchor=(0.5, 1.04))
    fig.text(0.01, -0.02, "Each dot is one seed. Sustained: e_F ≤ ε at every later save up to T. Bars: median "
             "(drawn only when > half the seeds reach). S: median per-seed restricted speedup τ_ABF/τ_FR "
             "(censored at T; both censored = 1).", fontsize=6.8, color=INK2, va="top")
    fig.tight_layout()
    save(fig, "time_to_accuracy")


def speedup(S):
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
    for ax, k in zip(axs, ORDER):
        r = S[k]
        n = r["n_seeds"]
        for kind, ls, lab in (("sustained", "-", "sustained"), ("first", "--", "first hit")):
            rows = r["speedup_dense"][kind]
            x = np.array([q["eps_kT"] for q in rows])
            y = np.array([q["S"] for q in rows])
            lo = np.array([q["lo"] for q in rows])
            hi = np.array([q["hi"] for q in rows])
            if kind == "sustained":
                ax.fill_between(x, lo, hi, color=FR, alpha=0.2, lw=0)
                cens = np.array([max(q["cens_abf"], q["cens_fr"]) > 0.25 * n for q in rows])
                for xi, ci in zip(x, cens):
                    if ci:
                        ax.axvspan(xi / 1.04, xi * 1.04, color="#dddcd8", alpha=0.6, lw=0)
            ax.plot(x, y, color=FR, ls=ls, lw=1.5, label=lab)
        ax.axhline(1, color=INK2, lw=0.8)
        for ek in r["time_to_accuracy"]["sustained"]:
            ax.axvline(float(ek[:-2]), color=MUTED, lw=0.5, ls=":")
        ax.set_xscale("log")
        ax.set_yscale("log")
        from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator, FuncFormatter
        ax.yaxis.set_major_locator(FixedLocator([0.5, 0.7, 1.0, 1.5, 2.0, 3.0]))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_major_locator(FixedLocator([0.02, 0.03, 0.05, 0.1, 0.2, 0.3]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xlabel("tolerance ε / k_BT")
        title(ax, r)
    axs[0].set_ylabel("speedup S(ε) = τ_ABF / τ_FR\n(median per-seed, restricted)")
    axs[0].legend(loc="upper left", fontsize=7)
    fig.text(0.01, -0.02, "Band: 95 % bootstrap CI (sustained). Grey: > 25 % of seeds censored in an arm (S is "
             "then a bound). Dotted: the frozen tolerance ladder.", fontsize=6.8, color=INK2, va="top")
    fig.tight_layout()
    save(fig, "speedup_vs_tolerance")


def establishment(S):
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.6))
    for ax, k in zip(axs, ORDER):
        r = S[k]
        t = np.array(r["curves"]["t"])
        tmax = min(r["T"], 30.0)
        sel = t <= tmax
        for arm, lab, col in ARMS:
            c = r["curves"][f"win_{arm}"]
            ax.fill_between(t[sel], np.array(c["q25"])[sel], np.array(c["q75"])[sel], color=col, alpha=0.15, lw=0)
            ax.plot(t[sel], np.array(c["median"])[sel], color=col, lw=1.5, label=lab)
            te = r["establishment"][f"{arm}_median"]
            ax.axvline(te, color=col, lw=1.0, ls="--")
        sh = r["establishment"]["win_share"]
        ax.axhline(sh, color=INK2, lw=0.8)
        ax.axhline(0.75 * sh, color=MUTED, lw=0.8, ls=":")
        vlines(ax, r)
        pdiff = r["establishment"]["paired_difference"]
        ax.text(0.98, 0.05, f"t_est ABF {r['establishment']['abf_median']:.1f}, FR {r['establishment']['fr_median']:.1f}"
                + (f"\npaired Δ {pdiff[0]:+.1f} [{pdiff[1]:+.1f}, {pdiff[2]:+.1f}] t.u." if pdiff else ""),
                transform=ax.transAxes, ha="right", fontsize=7.2, color=INK2)
        ax.set_xlabel("t (t.u.)")
        title(ax, r)
    axs[0].set_ylabel("window occupancy (fraction of walkers)")
    axs[0].legend(loc="center right", fontsize=7)
    fig.text(0.01, -0.02, "Window: |φ| < 0.5 rad (ITE) or |z| < 1.5 Å (LTA). Solid grey: flat-bias (uniform "
             "marginal) share; dotted: 0.75 × that. Dashed verticals: median establishment time (trailing 1-t.u. "
             "mean reaches 0.75 × share). Bands: 25–75 % of seeds.", fontsize=6.8, color=INK2, va="top")
    fig.tight_layout()
    save(fig, "early_population_establishment")


def main():
    S = json.load(open(os.path.join(ROOT, "results", "convergence_fr_campaign", "atlas", "atlas_summary.json")))["systems"]
    conv(S, "eF", "convergence_eF", "e_F(t) = ‖F̂_t − F_ref‖ (kJ/mol, RMS)")
    conv(S, "eG", "convergence_mean_force", "e_F′(t) = ‖Γ̂_t − F′_ref‖ (kJ/mol/rad, RMS)")
    running(S)
    tta(S, None)
    speedup(S)
    establishment(S)


if __name__ == "__main__":
    main()
