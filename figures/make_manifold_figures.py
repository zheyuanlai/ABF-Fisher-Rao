#!/usr/bin/env python3
"""Figures for the manifold (nonlinear reaction coordinate) reformulation.

Reads only stored results under results/manifold/; no simulation is run here.
Every figure is written as a matched .png (400 dpi) + .pdf pair.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from publication_style import (PALETTE, FigureStyle, add_panel_labels,
                               apply_publication_style, save_figure)

RES = ROOT / "results" / "manifold"
LIFT_C = {"cartesian": (PALETTE["vermillion"], "o", "cartesian  (w = 0)"),
          "minnorm": (PALETTE["orange"], "s", "min-norm  (w = c/G)"),
          "adiabatic": (PALETTE["blue"], "D", "adiabatic  (w = w*)")}
ARM_C = {"ti_cold": (PALETTE["blue"], "s", "fixed TI, cold"),
         "ti_warm": (PALETTE["sky"], "D", "fixed TI, warm"),
         "fr_only": (PALETTE["gray"], "^", "FR only"),
         "wfr_cart": (PALETTE["vermillion"], "o", "WFR, cartesian lift"),
         "wfr_minnorm": (PALETTE["orange"], "s", "WFR, min-norm lift"),
         "wfr_adiab": (PALETTE["green"], "D", "WFR, adiabatic lift"),
         "wfr_oracle": (PALETTE["black"], "*", "WFR, oracle refresh")}
FLOOR = 0.004


def _load(p):
    q = RES / p
    return json.load(open(q)) if q.exists() else None


# ---------------------------------------------------------------- fig M0 ---
def fig_mechanism(style):
    """What the three lifts actually do, drawn on the real system."""
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "src"))
    import torch
    from rcwfr.grid import DEVICE, DTYPE
    from rcwfr.systems.graph import build_mfib

    s = build_mfib(omega=1.0, a=0.6, k=1.4)
    z0, z1 = -0.9, -0.5
    y = torch.linspace(-2.2, 2.2, 1200, device=DEVICE, dtype=DTYPE)

    def nu(zv):
        zz = torch.full_like(y, zv)
        _, Psi, _, _ = s._psi_parts(y, zz)
        w = -s.p.beta * Psi
        e = torch.exp(w - w.max())
        return (e / torch.trapezoid(e, x=y)).cpu().numpy()

    yy = y.cpu().numpy()
    n0, n1 = nu(z0), nu(z1)
    sy = s.cv.s(y).cpu().numpy()
    x0, x1 = z0 - sy, z1 - sy                     # the two level sets, as graphs

    # min-norm lift: integrate dy/dz = c/G from z0 to z1 on the same grid
    ym = y.clone()
    nsub = 400
    h = (z1 - z0) / nsub
    for _ in range(nsub):
        c = s.cv.c(ym)
        ym = ym + h * c / (1.0 + c * c)
    J = torch.gradient(ym, spacing=(y,))[0].cpu().numpy()
    ym_np = ym.cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(style.width_in * 2.05,
                                            style.height_in * 1.12))

    # ---- (a) the two fibers in the ambient plane --------------------------
    ax = axes[0]
    peak = max(n0.max(), n1.max())
    for xg, nn, c in ((x0, n0, PALETTE["gray"]), (x1, n1, PALETTE["black"])):
        ax.plot(xg, yy, color=c, lw=1.1, zorder=2)
        w = 0.30 * nn / peak                      # conditional density, as a ribbon
        ax.fill_betweenx(yy, xg, xg - w, color=c, alpha=0.22, lw=0, zorder=1)
        i = int(np.argmax(nn))
        ax.plot([xg[i]], [yy[i]], "|", color=c, ms=9, mew=1.4, zorder=3)
    i0m, i1m = int(np.argmax(n0)), int(np.argmax(n1))
    ax.annotate(r"$\nu(\cdot|z_0)$", xy=(x0[i0m] - 0.16, yy[i0m]),
                xytext=(-1.92, -0.62), fontsize=6.6, color=PALETTE["gray"],
                arrowprops=dict(arrowstyle="-", lw=.6, color=PALETTE["gray"]))
    ax.annotate(r"$\nu(\cdot|z_1)$", xy=(x1[i1m] - 0.14, yy[i1m]),
                xytext=(-1.92, 1.42), fontsize=6.6, color=PALETTE["black"],
                arrowprops=dict(arrowstyle="-", lw=.6, color=PALETTE["black"]))
    j = 41
    ax.text(x0[j] - 0.05, yy[j], r"$\Sigma(z_0)$", fontsize=7,
            color=PALETTE["gray"], ha="right", va="center")
    ax.text(x1[j] + 0.06, yy[j], r"$\Sigma(z_1)$", fontsize=7,
            color=PALETTE["black"], va="center")

    for yq in (-1.3, -0.3, 0.55, 1.4):
        i = int(np.argmin(np.abs(yy - yq)))
        px, py = x0[i], yy[i]
        ax.plot([px], [py], "o", color=PALETTE["black"], ms=3.2, zorder=5)
        # cartesian: move x only
        ax.annotate("", xy=(px + (z1 - z0), py), xytext=(px, py), zorder=4,
                    arrowprops=dict(arrowstyle="-|>", lw=1.3, mutation_scale=7,
                                    color=PALETTE["vermillion"]))
        # min-norm: along grad xi
        cval = float(s.cv.c(torch.tensor(py, device=DEVICE, dtype=DTYPE)))
        G = 1 + cval * cval
        ax.annotate("", xy=(px + (z1 - z0) / G, py + (z1 - z0) * cval / G),
                    xytext=(px, py), zorder=4,
                    arrowprops=dict(arrowstyle="-|>", lw=1.3, mutation_scale=7,
                                    color=PALETTE["orange"]))
        # adiabatic: the CDF-matching map
        yt = float(s.lift_cdf(torch.tensor([z0], device=DEVICE, dtype=DTYPE),
                              torch.tensor([py], device=DEVICE, dtype=DTYPE),
                              torch.tensor([z1], device=DEVICE, dtype=DTYPE)))
        ax.annotate("", xy=(z1 - float(s.cv.s(torch.tensor(yt, device=DEVICE,
                                                           dtype=DTYPE))), yt),
                    xytext=(px, py), zorder=4,
                    arrowprops=dict(arrowstyle="-|>", lw=1.3, mutation_scale=7,
                                    color=PALETTE["blue"],
                                    connectionstyle="arc3,rad=-0.18"))
    ax.set_xlabel(r"ambient coordinate  $x$")
    ax.set_ylabel(r"fiber coordinate  $y$")
    ax.set_title("three lifts from the same point", fontsize=style.font_size)
    ax.set_ylim(-2.3, 2.3)
    ax.set_xlim(-2.15, 0.42)
    hs = [plt.Line2D([], [], color=PALETTE[c], lw=1.4, label=l) for c, l in
          (("vermillion", "cartesian"), ("orange", "min-norm"), ("blue", "adiabatic"))]
    ax.legend(handles=hs, fontsize=style.legend_size - 0.6, frameon=False,
              loc="lower left")

    # ---- (b) what arrives on the new fiber -------------------------------
    ax = axes[1]
    ax.fill_between(yy, n1, color=PALETTE["black"], alpha=0.13, lw=0,
                    label=r"target  $\nu(\cdot\,|\,z_1)$")
    ax.plot(yy, n1, color=PALETTE["black"], lw=1.2)
    ax.plot(yy, n0, color=PALETTE["vermillion"], lw=1.5, ls="-",
            label="cartesian arrives")
    ax.plot(ym_np, n0 / np.clip(J, 1e-9, None), color=PALETTE["orange"], lw=1.5,
            ls="--", label="min-norm arrives")
    ax.plot(yy, n1, color=PALETTE["blue"], lw=1.5, ls=":", label="adiabatic arrives")
    ax.set_xlim(-2.2, 2.2)
    ax.set_xlabel(r"fiber coordinate  $y$")
    ax.set_ylabel(r"density on  $\Sigma(z_1)$")
    ax.set_title(r"the lift is what decides this", fontsize=style.font_size)
    ax.legend(fontsize=style.legend_size - 0.8, frameon=False)
    add_panel_labels(axes)
    fig.tight_layout()
    save_figure(fig, HERE / "figM0_mechanism")


# ---------------------------------------------------------------- fig M1 ---
def fig_lift(style):
    d = _load("lift_a_MFIB_a0.6_k1.4_om1.0.json")
    if d is None:
        return
    rows = d["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(style.width_in * 2.05, style.height_in))

    ax = axes[0]
    z0 = -0.9
    rs = sorted([r for r in rows if r["z0"] == z0], key=lambda r: r["dz"])
    dz = np.array([r["dz"] for r in rs])
    for m, (c, mk, lab) in LIFT_C.items():
        if m == "adiabatic":
            # exactly zero by construction; plotted at the sampled value so the
            # histogram noise floor is visible rather than an empty axis
            y = np.maximum(np.abs([r["klmc_" + m] for r in rs]), 1e-5)
            ax.loglog(dz, y, marker=mk, color=c, label=lab + "  (= 0)",
                      ms=style.marker_size, lw=1.4)
            continue
        y = np.array([max(r[f"kl_{m}"], 1e-12) for r in rs])
        ax.loglog(dz, y, marker=mk, color=c, label=lab, ms=style.marker_size, lw=1.4)
        pred = np.array([r[f"pred_{m}"] for r in rs])
        ax.loglog(dz, pred, color=c, ls=":", lw=1.0)
    ax.axhline(1e-4, color=PALETTE["light_gray"], lw=0.9, ls="--")
    ax.text(dz[0] * 1.02, 1.35e-4, "PIT histogram floor", fontsize=6.0,
            color=PALETTE["gray"])
    ax.set_xlabel(r"transport step  $\Delta z$")
    ax.set_ylabel(r"conditional lag  $D_{\rm cond}$")
    ax.set_title(rf"frozen fiber, $z_0={z0}$" + "\n" + r"(dotted: $C\,\Delta z^2/2$)",
                 fontsize=style.font_size)
    ax.set_ylim(5e-6, 30)
    ax.set_xticks([0.0125, 0.025, 0.05, 0.1, 0.2, 0.4])
    ax.set_xticklabels(["0.0125", "0.025", "0.05", "0.1", "0.2", "0.4"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.legend(fontsize=style.legend_size - 0.4, frameon=False, loc="lower right")

    ax = axes[1]
    zs = sorted(set(r["z0"] for r in rows))
    for m in ("cartesian", "minnorm"):
        c, mk, lab = LIFT_C[m]
        C = [[r for r in rows if r["z0"] == z][0]["C"][m] for z in zs]
        ax.semilogy(zs, C, marker=mk, color=c, label=lab, ms=style.marker_size, lw=1.4)
    ax.set_xlabel(r"reaction coordinate  $z$")
    ax.set_ylabel(r"lag coefficient  $C(z)$")
    ax.set_title("the geometric lift stops helping\nexactly at the barrier top",
                 fontsize=style.font_size)
    ax.axvline(0.0, color=PALETTE["light_gray"], lw=0.9, ls="--")
    ax.legend(fontsize=style.legend_size - 0.4, frameon=False, loc="lower right")
    add_panel_labels(axes)
    fig.tight_layout()
    save_figure(fig, HERE / "figM1_lift")


# ---------------------------------------------------------------- fig M2 ---
def fig_timescale(style):
    d = _load("timescale.json")
    if d is None:
        return
    # the LEFT WELL only: at the barrier top the conditional is multimodal, C_eff
    # diverges and linear response is not the right description (see Finding M9)
    rows = [r for r in d["rows"] if -1.15 < r["z"] < -0.55 and r["v"] > 0]
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(style.width_in * 2.05, style.height_in))
    omegas = sorted(set(r["omega"] for r in rows))
    cmap = [PALETTE["blue"], PALETTE["green"], PALETTE["orange"], PALETTE["vermillion"]]

    ax = axes[0]
    for i, om in enumerate(omegas):
        vs = sorted(set(r["v"] for r in rows if r["omega"] == om))
        y = [np.mean([r["D_cartesian"] for r in rows
                      if r["omega"] == om and r["v"] == v]) for v in vs]
        p = [np.mean([r["pred_cartesian"] for r in rows
                      if r["omega"] == om and r["v"] == v]) for v in vs]
        a = [np.mean([r["D_adiabatic"] for r in rows
                      if r["omega"] == om and r["v"] == v]) for v in vs]
        ax.loglog(vs, y, "o", color=cmap[i % 4], ms=style.marker_size,
                  label=rf"$\omega={om}$")
        ax.loglog(vs, p, "-", color=cmap[i % 4], lw=1.2)
        if i == 0:
            ax.loglog(vs, np.maximum(np.abs(a), 1e-5), "D--", color=PALETTE["black"],
                      ms=style.marker_size - 1, lw=1.0, label="adiabatic lift")
    ax.set_ylim(3e-6, 3)
    ax.set_xlabel(r"transport speed  $v = \dot z$")
    ax.set_ylabel(r"steady-state  $D_{\rm cond}$")
    ax.set_title("lines: closed-form prediction,\nno fitted constant",
                 fontsize=style.font_size)
    ax.set_xticks(vs)
    ax.set_xticklabels([str(v) for v in vs])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.legend(fontsize=style.legend_size - 1.2, frameon=False, loc="upper left",
              ncol=2, columnspacing=0.8, handletextpad=0.4)

    ax = axes[1]
    for m in ("cartesian", "minnorm"):
        c, mk, lab = LIFT_C[m]
        x = np.array([r[f"pred_{m}"] for r in rows])
        y = np.array([r[f"D_{m}"] for r in rows])
        ok = (x > 3e-4) & (y > 3e-4) & (x < 0.1)
        ax.loglog(x[ok], y[ok], mk, color=c, ms=style.marker_size - 1.2,
                  alpha=0.55, mew=0, label=lab)
    lim = [3e-4, 2e-1]
    ax.plot(lim, lim, color=PALETTE["black"], lw=1.0, ls="--")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(r"predicted  $C_{\rm eff}\,v^2/2$")
    ax.set_ylabel(r"measured  $D_{\rm cond}$")
    ax.set_title("parity, linear-response regime\n(predicted $D<0.1$)",
                 fontsize=style.font_size)
    ax.legend(fontsize=style.legend_size - 0.6, frameon=False, loc="upper left")
    add_panel_labels(axes)
    fig.tight_layout()
    save_figure(fig, HERE / "figM2_timescale")


# ---------------------------------------------------------------- fig M3 ---
def fig_arms(style):
    d = _load("arms/CHANNEL_a0.6_k1.4_nc5_neq0.json")
    if d is None:
        return
    floor = d["floor"]
    fig, axes = plt.subplots(1, 3, figsize=(style.width_in * 3.0, style.height_in))

    ax = axes[0]
    for arm, (c, mk, lab) in ARM_C.items():
        if arm not in d["arms"]:
            continue
        a = d["arms"][arm]
        fe = np.array(a["fe"]); err = np.median(np.array(a["err"]), axis=1)
        ax.loglog(fe, err, color=c, lw=1.4, label=lab)
    ax.axhline(floor, color=PALETTE["light_gray"], ls="--", lw=1.0)
    ax.text(0.97, 0.03, "estimator floor", fontsize=6.2, color=PALETTE["gray"],
            ha="right", transform=ax.transAxes)
    ax.set_xlabel("force evaluations")
    ax.set_ylabel(r"$\|\hat F - F\|_{L^2}$")
    ax.set_title("nonlinear CV, hidden-channel fiber", fontsize=style.font_size)
    ax.legend(fontsize=style.legend_size - 1.4, frameon=False, loc="lower left",
              bbox_to_anchor=(0.0, 0.06))

    ax = axes[1]
    names = [a for a in ARM_C if a in d["arms"]]
    vals = [np.median(d["arms"][a]["final"]) for a in names]
    lo = [np.percentile(d["arms"][a]["final"], 25) for a in names]
    hi = [np.percentile(d["arms"][a]["final"], 75) for a in names]
    cols = [ARM_C[a][0] for a in names]
    ax.barh(range(len(names)), vals, color=cols,
            xerr=[np.array(vals) - lo, np.array(hi) - vals],
            error_kw=dict(lw=0.8, ecolor=PALETTE["black"]))
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([ARM_C[a][2] for a in names], fontsize=6.4)
    ax.invert_yaxis()
    ax.axvline(floor, color=PALETTE["light_gray"], ls="--", lw=1.0)
    ax.set_xscale("log")
    ax.set_xlim(floor * 0.6, max(hi) * 1.6)
    ax.set_xlabel(r"final $\|\hat F - F\|_{L^2}$")
    ax.set_title("median of 16 seeds, IQR", fontsize=style.font_size)

    ax = axes[2]
    for arm in ("wfr_cart", "wfr_minnorm", "wfr_adiab", "wfr_oracle", "ti_cold"):
        if arm not in d["arms"]:
            continue
        c, mk, lab = ARM_C[arm]
        a = d["arms"][arm]
        ax.semilogx(a["fe"], np.maximum(a["dcond"], 1e-4), color=c, lw=1.4, label=lab)
    ax.set_xlabel("force evaluations")
    ax.set_ylabel(r"$D_{\rm cond}$")
    ax.set_title("the lag that causes it", fontsize=style.font_size)
    ax.legend(fontsize=style.legend_size - 1.4, frameon=False)
    add_panel_labels(axes)
    fig.tight_layout()
    save_figure(fig, HERE / "figM3_arms")


# ---------------------------------------------------------------- fig M4 ---
def fig_fixman(style):
    d = _load("fixman.json")
    if d is None:
        return
    fig, ax = plt.subplots(figsize=(style.width_in, style.height_in))
    for i, name in enumerate(("EB", "SLOWFIB", "CHANNEL")):
        rs = [r for r in d["rows"] if r["system"] == name and r["a"] > 0]
        rs.sort(key=lambda r: r["ak"])
        ax.loglog([r["ak"] for r in rs], [r["rmse_F_minus_Frgd"] for r in rs],
                  marker="osD"[i], ls="none", ms=style.marker_size,
                  color=[PALETTE["blue"], PALETTE["green"], PALETTE["vermillion"]][i],
                  label=name)
    xs = np.array([0.1, 3.0])
    ax.loglog(xs, 0.011 * xs ** 2, color=PALETTE["black"], ls=":", lw=1.0)
    ax.text(0.9, 0.0022, r"$\propto (ak)^2$", fontsize=6.5)
    ax.axhline(d["floor"], color=PALETTE["light_gray"], ls="--", lw=1.0)
    ax.text(0.12, d["floor"] * 1.2, "estimator floor", fontsize=6.2,
            color=PALETTE["gray"])
    ax.set_xlabel(r"reaction-coordinate nonlinearity  $a k$")
    ax.set_ylabel(r"RMSE$(F - F_{\rm rgd})$")
    ax.set_title("cost of omitting the Fixman factor", fontsize=style.font_size)
    ax.legend(fontsize=style.legend_size, frameon=False, loc="upper left")
    fig.tight_layout()
    save_figure(fig, HERE / "figM4_fixman")


# ---------------------------------------------------------------- fig M5 ---
def fig_kappa(style):
    """The transport-rate trade-off, and whether a correct lift escapes it."""
    import glob, re
    files = sorted(glob.glob(str(RES / "arms" / "*_kap*.json")),
                   key=lambda f: float(re.search(r"_kap([\d.]+)\.json", f).group(1)))
    if len(files) < 3:
        return
    ks, series, floor = [], {}, None
    for f in files:
        ks.append(float(re.search(r"_kap([\d.]+)\.json", f).group(1)))
        d = json.load(open(f)); floor = d["floor"]
        for a, v in d["arms"].items():
            series.setdefault(a, []).append(float(np.median(v["final"])))
    fig, ax = plt.subplots(figsize=(style.width_in * 1.25, style.height_in))
    for a in ("wfr_cart", "wfr_minnorm", "wfr_adiab"):
        if a not in series or len(series[a]) != len(ks):
            continue
        c, mk, lab = ARM_C[a]
        ax.loglog(ks, series[a], marker=mk, color=c, ms=style.marker_size, lw=1.4,
                  label=lab)
    ax.axhline(floor, color=PALETTE["light_gray"], ls="--", lw=1.0)
    ax.text(0.97, 0.03, "estimator floor", fontsize=6.2, color=PALETTE["gray"],
            ha="right", transform=ax.transAxes)
    ax.set_xlabel(r"transport rate  $\kappa_W$")
    ax.set_ylabel(r"$\|\hat F - F\|_{L^2}$")
    # the birth-death-only arm is the kappa -> 0 limit: the naive lifts' best point
    fr = _load("arms/CHANNEL_a0.6_k1.4_nc5_neq0.json")
    if fr and "fr_only" in fr["arms"]:
        v = float(np.median(fr["arms"]["fr_only"]["final"]))
        ax.axhline(v, color=PALETTE["gray"], ls=":", lw=1.0)
        ax.text(0.98, v * 0.78, "no transport at all (birth-death only)", fontsize=6.2,
                color=PALETTE["gray"], ha="right",
                transform=ax.get_yaxis_transform())
    ax.set_title("with a bias term the transport can only hurt;\nwithout one, faster "
                 "is strictly better", fontsize=style.font_size)
    ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.legend(fontsize=style.legend_size - 0.6, frameon=False)
    fig.tight_layout()
    save_figure(fig, HERE / "figM5_kappa")


if __name__ == "__main__":
    style = apply_publication_style()
    fig_mechanism(style)
    fig_fixman(style)
    fig_lift(style)
    fig_timescale(style)
    fig_arms(style)
    fig_kappa(style)
    print("figures written to", HERE)
