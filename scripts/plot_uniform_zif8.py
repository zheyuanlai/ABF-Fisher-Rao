#!/usr/bin/env python
"""Figures for the ethane/ZIF-8 flexible-framework stage of the uniform-FR
campaign (abf vs fr_uniform), campaign style.

Under results/uniform_campaign/zif8/figures/ (png + pdf):

  fig_zif8_convergence_T{T}  THE headline: three stacked, shared-x panels
                             KL(p^xi_t || uniform) -> e_F'(t) -> e_F(t)
  fig_zif8_speedup_T{T}      e_F(t) with the tau thresholds/crossings drawn,
                             plus the paired per-seed I_F scatter
  fig_zif8_gate_T{T}         the HIDDEN-coordinate mechanism: A_gate at the
                             window vs the umbrella reference, J_gate(t),
                             and the gate aperture at completed transits
  fig_zif8_genealogy_T{T}    ancestor ESS/N, max lineage share, replacements
  fig_zif8_reference_T{T}    F / U / -TS and the reference xi-marginal

    python scripts/plot_uniform_zif8.py --temperature 300 [--root .]

This script PLOTS; it does not adjudicate.  Verdicts and statistics belong to
scripts/analyze_uniform_zif8.py -- they are read out of
results/uniform_campaign/zif8/summary_T{T}.json when that file exists and
simply omitted from the annotations when it does not.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(SCRIPTS, ".."))
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, os.path.join(REPO, "src"))
from publication_style import PALETTE, apply_publication_style, save_figure  # noqa: E402
# error_series / tau / PERSIST / FRACTIONS are the CAMPAIGN definitions: the
# gauge-aligned L2 over the scoring mask, and the persistence-qualified
# time-to-accuracy.  They are imported, never re-derived here.
from analyze_uniform_cha import FRACTIONS, PERSIST, error_series, tau  # noqa: E402
from zif8.core_zif8 import ZIF8SimConfig, js_divergence  # noqa: E402

# ---- the two arm colours, fixed for every panel in this script -------------
C_ABF = PALETTE["blue"]            # ABF
C_FR = PALETTE["vermillion"]       # ABF + uniform mFR
C_REF = PALETTE["black"]           # umbrella/WHAM reference and guide lines
C_AUX = PALETTE["gray"]            # FR-start marker, paired-seed hairlines
ARMS = (("abf", C_ABF, "ABF"), ("fr_uniform", C_FR, "ABF + uniform mFR"))

MASK_PAD = 1.0                     # scoring mask: [xi_A - pad, xi_B + pad]
LOG_SPAN = 20.0                    # dynamic range above which a panel goes log-y


# --------------------------------------------------------------------- utils
def med_iqr(a, axis):
    return (np.median(a, axis), np.percentile(a, 25, axis), np.percentile(a, 75, axis))


def nan_med_iqr(a, axis):
    """med/IQR ignoring NaN; an all-NaN slice (e.g. a save block with no replica
    in the gate band) legitimately yields NaN and is simply not drawn."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return (np.nanmedian(a, axis), np.nanpercentile(a, 25, axis),
                np.nanpercentile(a, 75, axis))


def rms_series(prof, ref, mask):
    """Gauge-FREE L2 of (prof - ref) over the mask, for (T, R, G) profiles.

    The mean force has no additive gauge freedom, so -- unlike ``error_series``
    used for the free-energy panel -- nothing is re-centred here.
    """
    d = (np.asarray(prof, dtype=float) - np.asarray(ref, dtype=float)[None, None, :])[:, :, mask]
    return np.sqrt((d * d).mean(axis=-1))


def wants_log(*arrays, span=LOG_SPAN):
    """True when the median curves span enough orders of magnitude for log-y."""
    vals = []
    for a in arrays:
        v = np.median(np.asarray(a, dtype=float), axis=1).ravel()
        vals.append(v[np.isfinite(v) & (v > 0)])
    v = np.concatenate(vals) if vals else np.zeros(0)
    return bool(v.size >= 2 and v.max() / v.min() >= span)


def die(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


def load_meta(npz):
    try:
        return json.loads(str(npz["meta"]))
    except Exception:
        return {}


# Logical knob -> the ZIF8SimConfig / meta field names that have carried it.
# The engine renamed several fields when the CV went periodic, so every lookup
# tries the aliases in order rather than assuming one spelling.
CFG_ALIASES = {
    "dt": ("dt",),
    "fr_start_steps": ("fr_start_steps",),
    "gate_lo": ("gate_lo",),
    "gate_hi": ("gate_hi",),
    "n_gate_bins": ("n_gate_bins",),
    "gate_band": ("gate_band_A", "gate_band"),
    "n_replicas": ("n_replicas",),
    "fr_rate": ("fr_rate",),
}
CFG_LAST_RESORT = dict(dt=0.0005, fr_start_steps=0, gate_lo=0.0, gate_hi=1.0,
                       n_gate_bins=40, gate_band=1.0, n_replicas=256, fr_rate=None)


def resolve_config(metas):
    """ZIF8SimConfig defaults, overridden by anything the run metas declare."""
    base = ZIF8SimConfig()
    vals = {}
    for key, names in CFG_ALIASES.items():
        vals[key] = next((getattr(base, n) for n in names if hasattr(base, n)),
                         CFG_LAST_RESORT[key])
    for meta in metas:
        for src in (meta, meta.get("sim"), meta.get("config"), meta.get("sim_config")):
            if not isinstance(src, dict):
                continue
            for key, names in CFG_ALIASES.items():
                for n in names:
                    if src.get(n) is not None:
                        vals[key] = src[n]
    return vals


def gate_bin_edges(sources, cfg, n_bins):
    """A_gate histogram edges: whichever npz declares them, else gate_lo/hi."""
    for src in sources:
        if src is not None and "gate_edges" in src:
            e = np.asarray(src["gate_edges"], dtype=float).ravel()
            if e.size == n_bins + 1:
                return 0.5 * (e[:-1] + e[1:]), e
    e = np.linspace(float(cfg["gate_lo"]), float(cfg["gate_hi"]), n_bins + 1)
    return 0.5 * (e[:-1] + e[1:]), e


def norm_density(h, edges):
    h = np.asarray(h, dtype=float).ravel()
    w = np.diff(edges)
    s = float(h.sum())
    return h / (s * w) if s > 0 else np.zeros_like(h)


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--temperature", type=float, required=True)
    ap.add_argument("--root", default=None,
                    help="repository root holding results/ (default: this checkout)")
    a_cli = ap.parse_args()

    root = os.path.abspath(a_cli.root) if a_cli.root else REPO
    tkey = f"{a_cli.temperature:g}"
    zif8 = os.path.join(root, "results/uniform_campaign/zif8")
    ref_path = os.path.join(zif8, "reference", f"reference_T{tkey}.npz")
    prod = os.path.join(zif8, f"production_T{tkey}")
    run_paths = {m: os.path.join(prod, f"{m}.npz") for m in ("abf", "fr_uniform")}
    summary_path = os.path.join(zif8, f"summary_T{tkey}.json")
    out = os.path.join(zif8, "figures")

    missing = [p for p in [ref_path, *run_paths.values()] if not os.path.exists(p)]
    if missing:
        for p in missing:
            print(f"ERROR: missing required input: {p}", file=sys.stderr)
        die(f"cannot plot T={tkey} without the reference and both production arms")

    ref = np.load(ref_path, allow_pickle=True)
    runs = {m: np.load(p, allow_pickle=True) for m, p in run_paths.items()}
    summary = {}
    if os.path.exists(summary_path):
        try:
            summary = json.load(open(summary_path))
        except Exception as exc:                      # a broken summary is not fatal
            print(f"warning: could not read {summary_path}: {exc}")
    else:
        print(f"note: no summary json at {summary_path}; annotations omitted")

    cfg = resolve_config([load_meta(runs[m]) for m in runs])
    os.makedirs(out, exist_ok=True)
    apply_publication_style()

    # ---- shapes, grids, clocks -------------------------------------------
    shapes = {m: np.asarray(runs[m]["pmf"]).shape for m in runs}
    if shapes["abf"] != shapes["fr_uniform"]:
        die("arm shape mismatch (T, R, G): "
            f"abf {shapes['abf']} vs fr_uniform {shapes['fr_uniform']} -- "
            "the arms are not a paired-seed block")
    grid = np.asarray(runs["abf"]["grid"], dtype=float)
    ref_grid = np.asarray(ref["grid"], dtype=float)
    if grid.shape != ref_grid.shape or not np.allclose(grid, ref_grid):
        die(f"reference and engine grids differ: reference {ref_grid.shape}, "
            f"runs {grid.shape}")
    if not np.allclose(np.asarray(runs["abf"]["times"], dtype=float),
                       np.asarray(runs["fr_uniform"]["times"], dtype=float)):
        die("arm save clocks differ; the two arms are not comparable point-by-point")

    # plotting/scoring axis: the physical xi in Angstrom when the engine exports
    # one (periodic-CV convention), otherwise the CV grid itself.
    axis, axis_is_A = grid, False
    for src in (runs["abf"], ref):
        if "xi_grid" in src:
            axis, axis_is_A = np.asarray(src["xi_grid"], dtype=float), True
            break
    xlab = r"$\xi$ ($\mathrm{\AA}$)" if axis_is_A else r"$\xi$ (CV units)"
    if "xi_A" in ref and "xi_B" in ref:
        xi_A, xi_B = float(ref["xi_A"]), float(ref["xi_B"])
        mask = (axis >= xi_A - MASK_PAD) & (axis <= xi_B + MASK_PAD)
    else:
        xi_A = xi_B = None
        mask = np.ones(axis.shape, dtype=bool)
        print("note: reference declares no xi_A/xi_B; scoring over the full grid")
    if not mask.any():
        die(f"empty scoring mask: xi_A={xi_A}, xi_B={xi_B}, axis "
            f"[{axis.min():g}, {axis.max():g}]")
    F_ref = np.asarray(ref["F"], dtype=float)
    # the engine reports mean_force as dF/d(CV), so differentiate on ``grid``
    # (the CV axis) even when the figures are drawn against the physical axis
    dF_ref = np.gradient(F_ref, grid)
    mf_unit = r"kJ/mol/$\mathrm{\AA}$" if not axis_is_A else "kJ/mol per CV unit"
    kT = float(ref["kT"])
    t = np.asarray(runs["abf"]["times"], dtype=float)
    steps = np.asarray(runs["abf"]["steps"], dtype=float)
    dt = float(t[-1] / steps[-1]) if steps[-1] > 0 else float(cfg["dt"])
    fr_start_t = float(cfg["fr_start_steps"]) * dt
    R = shapes["abf"][1]

    err = {m: error_series(np.asarray(runs[m]["pmf"], dtype=float), F_ref, mask)
           for m in runs}                                    # gauge-ALIGNED
    errp = {m: rms_series(runs[m]["mean_force"], dF_ref, mask) for m in runs}  # gauge-free
    kl = {m: np.asarray(runs[m]["kl_uniform"], dtype=float) for m in runs}

    d_int = summary.get("d_int_pct", {}) if isinstance(summary, dict) else {}
    fr_rate = summary.get("fr_rate", cfg["fr_rate"])
    rate_txt = f", FR rate {float(fr_rate):g}" if fr_rate is not None else ""
    ttl = f"Ethane/ZIF-8 {a_cli.temperature:g} K, {R} paired seed labels{rate_txt}"
    ttl_narrow = (f"Ethane/ZIF-8 {a_cli.temperature:g} K\n"
                  f"{R} seeds{rate_txt}; median and IQR")
    written = []

    def emit(fig, basename):
        png, pdf = save_figure(fig, os.path.join(out, basename))
        plt.close(fig)
        written.append(basename)
        print(f"wrote {pdf} (+ {png.name})")

    def mark_fr(ax, label=None):
        ax.axvline(fr_start_t, color=C_AUX, lw=0.8, ls="--", label=label)

    # ===================================================================
    # 1. THE headline: marginal -> mean force -> free energy, one x axis
    # ===================================================================
    fig, axes = plt.subplots(3, 1, figsize=(3.4, 7.0), sharex=True, layout="constrained")
    panels = (
        (kl, r"$D_{\mathrm{KL}}(\hat p^{\,\xi}_t \,\|\, u)$",
         wants_log(kl["abf"], kl["fr_uniform"])),
        (errp, r"$e_{F'}(t)$ (" + mf_unit + ")", True),
        (err, r"$e_F(t)$ (kJ/mol)", True),
    )
    for ax, (series, ylab, logy) in zip(axes, panels):
        for m, c, lab in ARMS:
            md, lo, hi = med_iqr(series[m], 1)
            ax.plot(t, md, color=c, lw=1.4, label=lab)
            ax.fill_between(t, lo, hi, color=c, alpha=0.18, lw=0)
        mark_fr(ax)
        ax.set_ylabel(ylab)
        if logy:
            ax.set_yscale("log")
    axes[0].plot([], [], color=C_AUX, lw=0.8, ls="--", label=r"$t_{\rm FR}$ start")
    axes[0].legend(frameon=False, fontsize=7.5, loc="best")
    axes[0].set_title(ttl_narrow, fontsize=8.5)
    if d_int.get("median") is not None:
        ci = d_int.get("ci95") or [float("nan"), float("nan")]
        note = (r"$\Delta I_F$ " + f"{float(d_int['median']):+.2f}%"
                + f"\nCI95 [{float(ci[0]):+.2f}, {float(ci[1]):+.2f}]")
        if d_int.get("wins") is not None:
            note += f"\nwins {int(d_int['wins'])}/{R}"
        if summary.get("verdict"):
            note += f"\n{summary['verdict']}"
        axes[2].text(0.97, 0.95, note, transform=axes[2].transAxes, ha="right",
                     va="top", fontsize=7, color=C_REF)
    axes[2].set_xlabel("t (ps)")
    emit(fig, f"fig_zif8_convergence_T{tkey}")

    # ===================================================================
    # 2. time-to-accuracy, drawn; plus the paired I_F scatter
    # ===================================================================
    curves = {m: np.median(err[m], axis=1) for m in err}
    e0 = float(curves["abf"][0])
    eps_list = {f"e0/{int(round(1 / f))}": e0 * f for f in FRACTIONS}
    eps_list["abf_final"] = float(curves["abf"][-1])
    taus = {name: (tau(t, curves["abf"], eps, PERSIST),
                   tau(t, curves["fr_uniform"], eps, PERSIST))
            for name, eps in eps_list.items()}
    speed = summary.get("time_to_accuracy") or {}
    for name, d in speed.items():                    # the analysis script wins
        try:
            eps_list[name] = float(d["eps"])
            taus[name] = (float(d["tau_abf"]), float(d["tau_uni"]))
        except (KeyError, TypeError, ValueError):
            continue

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.0), layout="constrained")
    ax = axes[0]
    for m, c, lab in ARMS:
        ax.plot(t, curves[m], color=c, lw=1.5, label=lab)
    for name, eps in eps_list.items():
        ax.axhline(eps, color=C_REF, lw=0.7, ls=":")
        ax.text(t[-1], eps, f" {name}", fontsize=6.5, color=C_REF, va="bottom",
                ha="right")
        for (m, c, _), tv in zip(ARMS, taus[name]):
            if np.isfinite(tv):
                ax.axvline(tv, color=c, lw=0.8, ls="-", alpha=0.45)
                ax.plot([tv], [eps], marker="o", ms=3.5, color=c)
    mark_fr(ax)
    ax.set_yscale("log")
    ax.set_xlabel("t (ps)")
    ax.set_ylabel(r"median $e_F(t)$ (kJ/mol)")
    sp = []
    for name in eps_list:
        ta, tu = taus[name]
        sp.append(f"{name}: {ta / tu:.2f}x" if np.isfinite(ta) and np.isfinite(tu)
                  and tu > 0 else f"{name}: n/a")
    ax.set_title("time to accuracy  " + ", ".join(sp), fontsize=7.5)
    ax.legend(frameon=False, fontsize=7.5)

    ax = axes[1]
    I = {m: np.trapezoid(err[m], t, axis=0) for m in err}
    aa, uu = I["abf"], I["fr_uniform"]
    lims = [float(min(aa.min(), uu.min())), float(max(aa.max(), uu.max()))]
    pad = 0.05 * (lims[1] - lims[0] or 1.0)
    lims = [lims[0] - pad, lims[1] + pad]
    ax.plot(lims, lims, color=C_REF, lw=0.9, ls=":", label="y = x")
    ax.plot(aa, uu, ls="none", marker="o", ms=4, color=C_FR, alpha=0.85)
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$I_F$, ABF")
    ax.set_ylabel(r"$I_F$, ABF + uniform mFR")
    wins = int((uu < aa).sum())
    ax.set_title(f"below the line = uniform mFR ahead ({wins}/{R})", fontsize=8)
    ax.legend(frameon=False, fontsize=7.5)
    emit(fig, f"fig_zif8_speedup_T{tkey}")

    # ===================================================================
    # 3. the hidden gate coordinate: distribution, J_gate(t), transits
    # ===================================================================
    blocks = {m: np.asarray(runs[m]["gate_hist_block"], dtype=float) for m in runs}
    n_bins = blocks["abf"].shape[-1]
    centers, edges = gate_bin_edges([ref, runs["abf"], runs["fr_uniform"]], cfg, n_bins)
    g_ref = np.asarray(ref["gate_hist_window"], dtype=float).ravel() if \
        "gate_hist_window" in ref else np.zeros(0)
    ref_ok = g_ref.size == n_bins
    if not ref_ok:
        print(f"warning: reference gate histogram has {g_ref.size} bins but the runs "
              f"have {n_bins}; the reference overlay and J_gate are omitted")

    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.9), layout="constrained")
    ax = axes[0]
    for m, c, lab in ARMS:
        h = np.asarray(runs[m]["gate_hist_cumulative"], dtype=float).sum(axis=0)
        ax.plot(centers, norm_density(h, edges), color=c, lw=1.4,
                drawstyle="steps-mid", label=lab)
    if ref_ok:
        ax.plot(centers, norm_density(g_ref, edges), color=C_REF, lw=1.0, ls=":",
                drawstyle="steps-mid", label="umbrella ref")
    ax.set_xlabel(r"$A_{\rm gate}$ ($\mathrm{\AA}$)")
    ax.set_ylabel(r"density at $|\xi| < \delta_w$")
    ax.set_title("gate aperture at the window", fontsize=8.5)
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1]
    if ref_ok:
        for m, c, lab in ARMS:
            b = blocks[m]                                  # (T, R, Gg)
            j = js_divergence(b, g_ref[None, None, :])
            j = np.where(b.sum(axis=-1) > 0, j, np.nan)     # empty blocks carry no info
            md, lo, hi = nan_med_iqr(j, 1)
            ax.plot(t, md, color=c, lw=1.4, label=lab)
            ax.fill_between(t, lo, hi, color=c, alpha=0.18, lw=0)
        ax.set_ylabel(r"$J_{\rm gate}(t)$ (JS to umbrella ref)")
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3g"))
        ax.legend(frameon=False, fontsize=7)
    else:
        ax.text(0.5, 0.5, "reference gate histogram\nunavailable", ha="center",
                va="center", transform=ax.transAxes, fontsize=8, color=C_AUX)
        ax.set_ylabel(r"$J_{\rm gate}(t)$")
    mark_fr(ax)
    ax.set_xlabel("t (ps)")
    ax.set_title("conditional gate equilibration", fontsize=8.5)

    ax = axes[2]
    any_cross = False
    for m, c, lab in ARMS:
        s = np.asarray(runs[m]["cross_gate_samples"], dtype=float).ravel()
        s = s[np.isfinite(s)]
        if s.size == 0:
            continue
        any_cross = True
        ax.hist(s, bins=edges, density=True, histtype="step", color=c, lw=1.3,
                label=f"{lab} (n={s.size})")
    if any_cross:
        ax.legend(frameon=False, fontsize=7)
    else:
        ax.text(0.5, 0.5, "no completed transits", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color=C_AUX)
    ax.set_xlabel(r"$A_{\rm gate}$ at transit ($\mathrm{\AA}$)")
    ax.set_ylabel("density")
    ax.set_title("aperture at transit", fontsize=8.5)
    fig.suptitle(ttl + ": the hidden gate coordinate (never biased)", fontsize=9)
    emit(fig, f"fig_zif8_gate_T{tkey}")

    # ===================================================================
    # 4. genealogy health of the FR arm
    # ===================================================================
    nuq = np.asarray(runs["fr_uniform"]["n_unique_ancestor"], dtype=float)
    N = int(np.nanmax(nuq[0])) if nuq.size else int(cfg["n_replicas"])
    if N <= 0:
        N = int(cfg["n_replicas"])
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.9), layout="constrained")
    ess = np.asarray(runs["fr_uniform"]["ancestor_ess"], dtype=float) / N
    md, lo, hi = nan_med_iqr(ess, 1)
    axes[0].plot(t, md, color=C_FR, lw=1.4)
    axes[0].fill_between(t, lo, hi, color=C_FR, alpha=0.18, lw=0)
    axes[0].axhline(0.30, color=C_REF, lw=0.8, ls=":", label="floor 0.30")
    axes[0].set_ylabel("ancestor ESS / N")
    axes[0].legend(frameon=False, fontsize=7)
    wmax = np.asarray(runs["fr_uniform"]["max_ancestor_frac"], dtype=float)
    md, lo, hi = nan_med_iqr(wmax, 1)
    axes[1].plot(t, md, color=C_FR, lw=1.4)
    axes[1].fill_between(t, lo, hi, color=C_FR, alpha=0.18, lw=0)
    axes[1].axhline(0.05, color=C_REF, lw=0.8, ls=":", label="cap 0.05")
    axes[1].set_ylabel("max lineage share")
    axes[1].legend(frameon=False, fontsize=7)
    if "repl_cumulative" in runs["fr_uniform"]:
        ev = np.asarray(runs["fr_uniform"]["repl_cumulative"], dtype=float) / N
        md, lo, hi = med_iqr(ev, 1)
        axes[2].plot(t, md, color=C_FR, lw=1.4)
        axes[2].fill_between(t, lo, hi, color=C_FR, alpha=0.18, lw=0)
        axes[2].set_ylabel("cumulative events / N")
    else:
        tot = np.asarray(runs["fr_uniform"]["total_replacement_events"], dtype=float) / N
        axes[2].plot(np.arange(len(tot)), tot, ls="none", marker="o", ms=4, color=C_FR)
        axes[2].set_ylabel("total events / N")
        axes[2].set_xlabel("seed label")
    for ax in axes[:2]:
        mark_fr(ax)
        ax.set_xlabel("t (ps)")
    if "repl_cumulative" in runs["fr_uniform"]:
        mark_fr(axes[2])
        axes[2].set_xlabel("t (ps)")
    fig.suptitle(ttl + ": genealogy of the uniform-mFR arm", fontsize=9)
    emit(fig, f"fig_zif8_genealogy_T{tkey}")

    # ===================================================================
    # 5. the umbrella reference itself
    # ===================================================================
    U_ref = np.asarray(ref["U"], dtype=float)
    fig, ax = plt.subplots(figsize=(4.6, 3.0), layout="constrained")
    ax.plot(axis, F_ref / kT, color=C_REF, lw=1.5, label=r"$F(\xi)$")
    ax.plot(axis, U_ref / kT, color=C_ABF, lw=1.3, ls="--", label=r"$U(\xi)$")
    ax.plot(axis, (F_ref - U_ref) / kT, color=C_FR, lw=1.3, ls="-.",
            label=r"$-TS(\xi)$")
    band = float(cfg["gate_band"])
    ax.axvspan(-band, band, color=PALETTE["light_gray"], alpha=0.5, lw=0, zorder=0)
    for x in (xi_A, xi_B):
        if x is not None:
            ax.axvline(x, color=C_AUX, lw=0.8, ls=":")
    ax.set_xlabel(xlab)
    ax.set_ylabel(r"energy ($k_BT$)")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax2 = ax.twinx()
    ax2.plot(axis, np.asarray(ref["p"], dtype=float), color=PALETTE["green"], lw=1.1)
    ax2.set_ylabel(r"reference $p(\xi)$", color=PALETTE["green"])
    ax2.tick_params(axis="y", labelcolor=PALETTE["green"])
    ax2.spines["right"].set_visible(True)
    cages = r"; dotted: cages $\xi_A$, $\xi_B$" if xi_A is not None else ""
    ax.set_title(f"Ethane/ZIF-8 {a_cli.temperature:g} K umbrella reference\n"
                 r"(shaded: $|\xi| < \delta_w$" + cages + ")", fontsize=8.5)
    emit(fig, f"fig_zif8_reference_T{tkey}")

    print(f"{len(written)} figures -> {out}")


if __name__ == "__main__":
    main()
