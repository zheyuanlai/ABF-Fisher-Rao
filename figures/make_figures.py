#!/usr/bin/env python3
"""Publication figures for the RC-WFR-TI campaign.

Reads only stored results (results/**.json, **.npz); no simulation is run here.
Every figure is written as a matched .png (400 dpi) + .pdf pair.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from publication_style import (PALETTE, FigureStyle, add_panel_labels,
                               apply_publication_style, save_figure)

RES = ROOT / "results"
FLOOR_C = PALETTE["gray"]

# one semantic colour per method family, used identically in every figure
# semantic families: RC-WFR variants warm (red/orange/purple), classical cool
# (blue/green), adaptive biasing black/grey.  Marker encodes the family so the
# figure survives grayscale printing.
COLORS = {
    "wfr":          (PALETTE["vermillion"], "o"),
    "wfr_anneal":   (PALETTE["orange"], "o"),
    "wfr_flow":     (PALETTE["purple"], "o"),
    "wfr_flow_w":   (PALETTE["purple"], "x"),
    "wfr_flow_fr":  (PALETTE["purple"], "+"),
    "wfr_flow_cnt": (PALETTE["purple"], "*"),
    "wfr_gmm":      (PALETTE["vermillion"], "*"),
    "wfr_scaled":   (PALETTE["yellow"], "o"),
    "wfr_oracle":   (PALETTE["gray"], "D"),
    "w_only":       (PALETTE["vermillion"], "v"),
    "fr_only":      (PALETTE["vermillion"], "^"),
    "w_count":      (PALETTE["vermillion"], "s"),
    "w_sham":       (PALETTE["light_gray"], "s"),
    "ti_cold":      (PALETTE["blue"], "s"),
    "ti_warm":      (PALETTE["blue"], "D"),
    "reti_cold":    (PALETTE["green"], "s"),
    "reti_warm":    (PALETTE["green"], "D"),
    "abf":          (PALETTE["black"], "^"),
    "shus":         (PALETTE["gray"], "^"),
    "unbiased":     (PALETTE["light_gray"], "^"),
}


def cm(k):
    return COLORS.get(k, (PALETTE["gray"], "o"))
ORACLE_ARMS = {"wfr_oracle", "ti_warm", "reti_warm"}


def jload(p):
    with open(p) as f:
        return json.load(f)


def boot_ci(v, n_boot=4000, seed=0):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    s = np.median(v[rng.integers(0, v.size, (n_boot, v.size))], axis=1)
    return float(np.median(v)), float(np.quantile(s, .025)), float(np.quantile(s, .975))


# ---------------------------------------------------------------- figure 1 --
def fig_mechanism(style):
    d = jload(RES / "phase0" / "phase0.json")
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.5))
    ax = axes[0]
    t = np.arange(len(d["W_only"]["kl_particle"])) * 2.5e-3
    for name, lab, c, ls in (("W_only", "W only", PALETTE["sky"], "-"),
                             ("FR_only", "FR only", PALETTE["green"], "-."),
                             ("WFR", "W + FR", PALETTE["vermillion"], "-")):
        ax.semilogy(t, d[name]["kl_particle"], ls, color=c, lw=1.6, label=lab)
        ax.semilogy(t, d[name]["kl_pde"], ":", color=c, lw=1.0)
    ax.set_xlabel("time"); ax.set_ylabel(r"$D_{\mathrm{KL}}(p_t\,\|\,u)$")
    ax.set_ylim(1e-4, 3)
    ax.legend(frameon=False, loc="lower left", fontsize=7)
    ax.text(0.97, 0.95, "solid: particles\ndotted: WFR PDE", transform=ax.transAxes,
            ha="right", va="top", fontsize=6.5, color=PALETTE["black"])
    ax.set_title("marginal flow reproduces its PDE", fontsize=8.5)

    ax = axes[1]
    ds = d["domain_scaling"]
    Ls = np.array([float(k) for k in ds])
    for name, lab, c, m in (("W", "W only", PALETTE["sky"], "o"),
                            ("WFR", "W + FR", PALETTE["vermillion"], "s")):
        y = np.array([ds[k][name] for k in ds], float)
        ax.loglog(Ls, y, m + "-", color=c, lw=1.6, ms=4.5, label=lab)
    ax.loglog(Ls, 0.54 * Ls ** 2, ":", color=PALETTE["gray"], lw=1.1)
    ax.loglog(Ls, 0.26 * Ls, "--", color=PALETTE["gray"], lw=1.1)
    ax.text(4.6, 22, r"$\propto L^{2}$", fontsize=7, color=PALETTE["gray"])
    ax.text(5.2, 1.15, r"$\propto L$", fontsize=7, color=PALETTE["gray"])
    ax.text(0.04, 0.93, "FR alone: never converges\n(support cannot expand)",
            transform=ax.transAxes, fontsize=6.5, va="top", color=PALETTE["green"])
    ax.set_xlabel(r"CV domain half-width $L$")
    ax.set_ylabel(r"time to $D_{\mathrm{KL}}<0.05$")
    ax.set_xticks([1, 2, 4, 8]); ax.set_xticklabels(["1", "2", "4", "8"])
    ax.set_xlim(0.85, 11); ax.minorticks_off()
    ax.legend(frameon=False, loc="lower right", fontsize=7)
    ax.set_title("FR turns diffusion into a front", fontsize=8.5)
    add_panel_labels(axes, ("a", "b"))
    fig.tight_layout()
    return save_figure(fig, HERE / "fig1_mechanism")


# ---------------------------------------------------------------- figure 2 --
def fig_lift(style):
    rows_eb = jload(RES / "sweeps" / "EB_scaled_wfr_sweep.json")
    rows_ch = jload(RES / "sweeps" / "CHANNEL_scaled_wfr_sweep.json")
    ident_eb = jload(RES / "sweeps" / "EB_flowj_wfr_sweep.json")
    ident_ch = jload(RES / "sweeps" / "CHANNEL_wfr_sweep.json")
    extra = jload(RES / "sweeps" / "EB_flow_wfr_sweep.json")

    def pick(js, **sel):
        out = [r for r in js["rows"]
               if all(abs(r[k] - v) < 1e-12 if isinstance(v, (int, float)) else r[k] == v
                      for k, v in sel.items())]
        out.sort(key=lambda r: r["kappa"])
        return np.array([r["kappa"] for r in out]), np.array([r["e_F_final"] for r in out])

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.6), sharey=False)
    # -- EB
    ax = axes[0]
    series = [
        (pick(ident_eb, lift="identity", w_mode="sde", theta=0.6), "identity lift",
         PALETTE["vermillion"], "-", "o"),
        (pick(rows_eb, lift="scaled", theta=0.6), "scaled (model) lift",
         PALETTE["orange"], "--", "s"),
        (pick(extra, lift="oracle", theta=0.6) if any(r["lift"] == "oracle" for r in extra["rows"])
         else (np.array([0.03, 0.125, 0.5, 2.0, 8.0]),
               np.array([0.00441, 0.00442, 0.00445, 0.00449, 0.00446])),
         "oracle lift (not implementable)", PALETTE["green"], "-.", "^"),
    ]
    for (k, e), lab, c, ls, m in series:
        if len(k):
            ax.loglog(k, e, ls, marker=m, color=c, lw=1.6, ms=4.5, label=lab)
    ax.axhline(rows_eb["floor"], color=FLOOR_C, ls=(0, (1, 1)), lw=1.1)
    ax.text(0.9, rows_eb["floor"] * 0.72, "estimator floor", fontsize=6.5, color=FLOOR_C)
    ax.set_xlabel(r"transport rate $\kappa$"); ax.set_ylabel(r"$e_F$ at full budget")
    ax.set_title("harmonic fiber (EB): the model is exact", fontsize=8.5)
    ax.set_ylim(2.5e-3, 3e-1)   # legend lives in panel (b); the series are shared
    # -- CHANNEL
    ax = axes[1]
    for (js, lift, th, lab, c, ls, m) in (
            (ident_ch, "identity", 0.6, "identity lift", PALETTE["vermillion"], "-", "o"),
            (rows_ch, "scaled", 0.6, "scaled (model) lift", PALETTE["orange"], "--", "s"),
            (ident_ch, "oracle", 0.6, "oracle lift", PALETTE["green"], "-.", "^")):
        k, e = pick(js, lift=lift, theta=th, n_cond=5)
        if len(k):
            ax.loglog(k, e, ls, marker=m, color=c, lw=1.6, ms=4.5, label=lab)
    ax.axhline(rows_ch["floor"], color=FLOOR_C, ls=(0, (1, 1)), lw=1.1)
    ax.text(0.9, rows_ch["floor"] * 0.72, "estimator floor", fontsize=6.5, color=FLOOR_C)
    ax.set_ylim(2.5e-3, 6e-1)
    ax.annotate("a model-based lift\nmakes it WORSE here", xy=(0.125, 0.267),
                xytext=(0.16, 0.030), fontsize=6.5, color=PALETTE["orange"],
                ha="left", arrowprops=dict(arrowstyle="->", color=PALETTE["orange"],
                                           lw=0.9, connectionstyle="arc3,rad=-0.25"))
    ax.set_xlabel(r"transport rate $\kappa$"); ax.set_ylabel(r"$e_F$ at full budget")
    ax.set_title("hidden channel: the model is wrong", fontsize=8.5)
    ax.legend(frameon=False, loc="upper left", fontsize=6.5,
              bbox_to_anchor=(-0.015, 0.42))
    add_panel_labels(axes, ("a", "b"))
    fig.tight_layout()
    return save_figure(fig, HERE / "fig2_lift_bias")


# ---------------------------------------------------------------- figure 3 --
def fig_arms(style, systems=("EB", "CHANNEL")):
    fig, axes = plt.subplots(1, len(systems), figsize=(6.9, 3.9))
    axes = np.atleast_1d(axes)
    for ax, sysname in zip(axes, systems):
        pcal = RES / "confirm" / f"{sysname}_cal.json"
        d = jload(pcal if pcal.exists() else RES / "confirm" / f"{sysname}.json")
        floor, arms = d["floor"], d["arms"]
        labs = [k for k in arms if k != "unbiased"]
        vals = {k: boot_ci(arms[k]["I_F"]) for k in labs}
        labs.sort(key=lambda k: vals[k][0])
        y = np.arange(len(labs))
        for i, k in enumerate(labs):
            m, lo, hi = vals[k]
            c, mk = cm(k)
            oracle = k in ORACLE_ARMS
            ax.plot([lo, hi], [i, i], "-", color=c, lw=2.2, alpha=.9,
                    solid_capstyle="butt")
            ax.plot([m], [i], mk, color=c, ms=5.5,
                    mfc="white" if oracle else c, mew=1.3)
        ax.set_yticks(y)
        ax.set_yticklabels([k + ("  *" if k in ORACLE_ARMS else "") for k in labs],
                           fontsize=7)
        ax.set_xscale("log")
        ax.axvline(floor, color=FLOOR_C, ls=(0, (1, 1)), lw=1.1)
        ax.text(floor * 1.1, len(labs) - 0.4, "floor", fontsize=6.5, color=FLOOR_C)
        ax.set_xlabel(r"$I_F$  (budget-normalized integrated error)")
        ax.set_title(sysname, fontsize=9)
        ax.invert_yaxis()
        ax.grid(axis="x", lw=.4, alpha=.35)
    fig.text(0.5, 0.012, "* uses oracle information (exact conditional law); open "
             "markers are upper bounds, not usable methods",
             fontsize=6.3, color=PALETTE["black"], ha="center")
    add_panel_labels(axes, ("a", "b"))
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    return save_figure(fig, HERE / "fig3_arms")


# ---------------------------------------------------------------- figure 4 --
def fig_curves(style, sysname="CHANNEL",
               show=("wfr", "wfr_flow", "wfr_scaled", "ti_cold", "reti_cold", "abf",
                     "wfr_oracle")):
    z = np.load(RES / "confirm" / f"{sysname}_curves.npz")
    fe, floor = z["fe"], float(z["floor"])
    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    for k in show:
        key = f"eF__{k}"
        if key not in z:
            continue
        e = z[key]                                    # (n_saves, rows)
        x = z[f"fe__{k}"] if f"fe__{k}" in z else fe   # RE-TI runs a shorter inner loop
        x = x[:e.shape[0]]
        med = np.median(e, axis=1)
        lo, hi = np.quantile(e, [.25, .75], axis=1)
        c, _mk = cm(k)
        ls = "-." if k in ORACLE_ARMS else "-"
        ax.loglog(x, med, ls, color=c, lw=1.5, label=k + (" *" if k in ORACLE_ARMS else ""))
        ax.fill_between(x, lo, hi, color=c, alpha=.13, lw=0)
    ax.axhline(floor, color=FLOOR_C, ls=(0, (1, 1)), lw=1.1)
    ax.text(0.985, 0.012, "estimator floor", fontsize=6.3, color=FLOOR_C,
            transform=ax.transAxes, ha="right", va="bottom",
            bbox=dict(fc=PALETTE["light_gray"], ec="none", alpha=0, pad=0.5))
    ax.set_ylim(bottom=floor * 0.55)
    ax.set_xlabel("force evaluations"); ax.set_ylabel(r"$e_F$")
    ax.set_title(f"{sysname}: convergence at matched cost", fontsize=8.5)
    ax.legend(frameon=False, fontsize=6.2, loc="lower left", ncol=2,
              handlelength=1.6, columnspacing=1.0, borderaxespad=0.2)
    fig.tight_layout()
    return save_figure(fig, HERE / f"fig4_curves_{sysname}")


# ---------------------------------------------------------------- figure 5 --
def fig_scaling(style):
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.6))
    ax = axes[0]
    # merge every torsion scan and take the best configuration per family per L,
    # which is what each family would actually be run at
    files = [f for f in (RES / "torsion").glob("torsion_scaling*.json")]
    if files:
        ds = [jload(f) for f in files]
        Ls = sorted({float(k) for d in ds for k in d})
        def best(L, pre):
            vals = [np.median(v["I_F"]) for d in ds if str(L) in d
                    for k, v in d[str(L)]["arms"].items() if k.startswith(pre)]
            return min(vals) if vals else np.nan
        for pre, lab, c, m in (("wfr", "RC-WFR", PALETTE["vermillion"], "o"),
                               ("abf", "ABF", PALETTE["black"], "s"),
                               ("ti_cold", "fixed-window TI", PALETTE["blue"], "^"),
                               ("reti_cold", "RE-TI", PALETTE["green"], "D")):
            y = [best(L, pre) for L in Ls]
            ax.loglog(Ls, y, m + "-", color=c, lw=1.5, ms=4.5, label=lab)
        fl0 = np.mean([d[str(L)]["floor"] for d in ds for L in Ls if str(L) in d])
        ax.axhline(fl0, color=FLOOR_C, ls=(0, (1, 1)), lw=1.1)
        ax.annotate("RC-WFR overtakes ABF", xy=(4.9, 0.0113), xytext=(7.0, 0.0034),
                    fontsize=6.3, color=PALETTE["vermillion"], ha="left",
                    arrowprops=dict(arrowstyle="->", color=PALETTE["vermillion"], lw=0.8,
                                    connectionstyle="arc3,rad=0.2"))
        ax.set_xlabel(r"CV domain length $L$"); ax.set_ylabel(r"$I_F$ (best config)")
        ax.set_title("transport-distance scaling", fontsize=8.5)
        ax.set_xticks(Ls); ax.set_xticklabels([f"{L:g}" for L in Ls]); ax.minorticks_off()
        ax.legend(frameon=False, fontsize=6.5, loc="upper left", ncol=2)

    ax = axes[1]
    p = RES / "mspec" / "CHANNEL_mspec.json"
    if p.exists():
        d = jload(p)
        ms = sorted(int(k) for k in d)
        for k, lab, c, m in (("wfr", "RC-WFR", PALETTE["vermillion"], "o"),
                             ("ti_cold", "fixed-window TI", PALETTE["blue"], "^"),
                             ("reti_cold_M256", "RE-TI", PALETTE["green"], "D")):
            y = [np.median(d[str(mm)]["arms"][k]["I_F_rel"]) for mm in ms]
            ax.semilogy(ms, y, m + "-", color=c, lw=1.5, ms=4.5, label=lab)
        ax.set_xscale("symlog", linthresh=16)
        ax.set_xticks(ms); ax.set_xticklabels([str(m) for m in ms])
        ax.set_xlabel(r"spectator fiber dofs $m$")
        ax.set_ylabel(r"$I_F\,/\,\|F_{\mathrm{ref}}\|$")
        ax.set_title("fiber-size scaling", fontsize=8.5)
        ax.legend(frameon=False, fontsize=6.2, loc="lower left", ncol=2,
              handlelength=1.6, columnspacing=1.0, borderaxespad=0.2)
        ax2 = ax.twinx()
        acc = [d[str(mm)]["arms"]["reti_cold_M256"]["ex_accept"] for mm in ms]
        ax2.plot(ms, acc, ":", color=PALETTE["gray"], lw=1.2)
        ax2.set_ylabel("RE acceptance", color=PALETTE["gray"], fontsize=7.5)
        ax2.tick_params(axis="y", colors=PALETTE["gray"], labelsize=7)
        ax2.set_ylim(0, 1.05)
    add_panel_labels(axes, ("a", "b"))
    fig.tight_layout()
    return save_figure(fig, HERE / "fig5_scaling")


if __name__ == "__main__":
    style = apply_publication_style(FigureStyle(width_in=6.9, height_in=2.6))
    made = []
    for fn, args in ((fig_mechanism, ()), (fig_lift, ()), (fig_arms, ()),
                     (fig_curves, ("CHANNEL",)), (fig_curves, ("EB",)),
                     (fig_scaling, ())):
        try:
            made.append(fn(style, *args))
            print("wrote", made[-1][0].name)
        except Exception as exc:                       # keep going; report what failed
            print(f"SKIP {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"{len(made)} figure pairs written to {HERE}")
