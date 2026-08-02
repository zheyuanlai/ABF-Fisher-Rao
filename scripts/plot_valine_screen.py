#!/usr/bin/env python
"""Figures for the Val screening phase: the state map, the pilot FES, and the two corrections.

Four panels, and each exists to make one claim checkable by eye rather than only by table.

  A  S1 states projected onto the SELECTED CV.  If the seven 3-D states did not separate here,
     the distinguishability gate would be arguing against the picture.
  B  F_pilot(phi, chi1) with its watershed regions.  The regions are what V3 measures.
  C  chi1 rotamer changes per walker per ns against time, split by whether the walker was seeded
     in a well or on a barrier.  Flat and coincident is the whole of CORRECTION 1: an equilibrium
     rate, not a seeding transient.
  D  the dt sweep.  The kinetic temperature collapses as O(dt^2) while the equipartition
     estimators do not move -- CORRECTION 2.

Usage
-----
    python scripts/plot_valine_screen.py --out results/valine/figures
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import matplotlib                                                            # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                              # noqa: E402
from matplotlib.colors import ListedColormap                                 # noqa: E402

from alanine.basins import BasinMap, grid_deg                                # noqa: E402

KB = 0.008314462618


def panel_states(ax, state_map):
    st = np.load(os.path.join(state_map, "states.npz"), allow_pickle=True)
    ex = np.load(os.path.join(state_map, "explore.npz"), allow_pickle=True)
    th = ex["theta"][:, ::20]                       # thin for plotting only
    lab = st["frame_labels"][:, ::20]
    c3 = np.degrees(st["centres"])
    K = c3.shape[0]
    m = lab >= 0
    phi = np.degrees(th[:, :, 0][m])
    chi = np.degrees(th[:, :, 2][m])
    lb = lab[m]
    cmap = plt.get_cmap("tab10")
    idx = np.random.default_rng(0).permutation(phi.size)[:120_000]
    ax.scatter(phi[idx], chi[idx], c=[cmap(int(k) % 10) for k in lb[idx]], s=0.4, alpha=0.35,
               linewidths=0, rasterized=True)
    for k in range(K):
        ax.plot(c3[k, 0], c3[k, 2], "k*", ms=13, mfc=cmap(k % 10), mew=1.2)
        ax.annotate(f"B{k}", (c3[k, 0], c3[k, 2]), textcoords="offset points",
                    xytext=(7, 6), fontsize=9, fontweight="bold")
    ax.axvline(0, color="0.4", lw=0.8, ls=":")
    ax.set_title("A  S1 states projected on the selected CV\n"
                 "(7 states; balanced accuracy 0.973 from these two axes)", fontsize=10)


def panel_pilot(ax, pilot, temperature=300.0):
    kT = KB * temperature
    pf = np.load(os.path.join(pilot, "pilot_reference.npz"), allow_pickle=True)
    F = pf["F"]
    n = F.shape[0]
    g = grid_deg(n)
    mask = np.isfinite(F) & (F < 8.0 * kT)
    bm = BasinMap(F, mask, kT, ceiling_kT=8.0, min_prominence_kT=1.0, max_basins=8,
                  name_hints=())
    Fk = np.where(np.isfinite(F), F / kT, np.nan)
    im = ax.pcolormesh(g, g, np.clip(Fk, 0, 12).T, cmap="viridis_r", shading="nearest",
                       vmin=0, vmax=12)
    lab = np.where(bm.label >= 0, bm.label, np.nan).astype(float)
    ax.contour(g, g, lab.T, levels=np.arange(-0.5, len(bm.names)), colors="w", linewidths=0.7)
    for k, nm in enumerate(bm.names):
        c = bm.centres_deg[k]
        ax.annotate(nm, c, color="w", fontsize=9, fontweight="bold", ha="center")
    ax.set_title("B  pilot $F(\\phi,\\chi_1)$, $\\psi$ free, with its watershed regions\n"
                 "(screening quality; the regions are what V3 measures)", fontsize=10)
    return im


def panel_rate(ax, state_map):
    ex = np.load(os.path.join(state_map, "explore.npz"), allow_pickle=True)
    th = ex["theta"]
    tgt, origin = ex["lattice_targets"], ex["origin"]
    chi = th[:, :, 2].astype(np.float64)

    def rot(c):
        r = np.zeros(c.shape, np.int8)
        r[(c > 0) & (c <= np.radians(120))] = 1
        r[(c < 0) & (c >= np.radians(-120))] = 2
        return r

    R = rot(chi)
    ch = R[:, 1:] != R[:, :-1]
    T = ch.shape[1]
    chi0 = np.degrees(tgt[origin][:, 2])
    d = np.min(np.abs((chi0[:, None] - np.array([-180., -60., 60., 180.]) + 180) % 360 - 180), 1)
    well = d <= 20
    nb = 10
    ts = (np.arange(nb) + 0.5) * (T * 0.2 / nb)
    for m, lb, st in ((well, "seeded in a chi1 well", "-o"),
                      (~well, "seeded on a chi1 barrier", "-s")):
        y = [ch[m][:, b * T // nb:(b + 1) * T // nb].sum() / m.sum() / (T // nb * 0.2) * 1000
             for b in range(nb)]
        ax.plot(ts, y, st, ms=4, label=f"{lb}  (n={int(m.sum())})")
    ax.axhline(2.70, color="0.3", ls="--", lw=1,
               label="2.70 / walker / ns  $\\Rightarrow$ 6-8 kT, backbone free")
    ax.set_ylim(0, None)
    ax.set_xlabel("time in production (ps)")
    ax.set_ylabel("$\\chi_1$ rotamer changes / walker / ns")
    ax.legend(fontsize=7.5, loc="upper right")
    ax.set_title("C  CORRECTION 1: flat in time and independent of the seed\n"
                 "= an equilibrium rate, not a transient (Stage 0 said 11.3-17.9 kT)",
                 fontsize=10)


def panel_dt(ax, dt_bias):
    d = json.load(open(os.path.join(dt_bias, "dt_bias.json")))
    rows = d["rows"]
    dts = sorted({r["dt_fs"] for r in rows})
    styles = {"unrestrained": ("o-", "tab:blue"), "stage2_clamp": ("s--", "tab:red"),
              "pilot_clamp": ("^:", "tab:orange")}
    x2 = [t * t for t in dts]
    for grp, (st, col) in styles.items():
        y = [next(r["T_kin"] for r in rows if r["dt_fs"] == t and r["group"] == grp) - 300
             for t in dts]
        ax.plot(x2, y, st, color=col, ms=5, label=f"$T_{{kin}}$  {grp}")
    for key, col, lb in (("T_bond", "tab:green", "$T_{bond}$"),
                         ("T_angle", "tab:purple", "$T_{angle}$")):
        y = [next(r[key] for r in rows if r["dt_fs"] == t and r["group"] == "unrestrained") - 300
             for t in dts]
        y0 = y[dts.index(min(dts))]
        ax.plot(x2, [v - y0 for v in y], "d-.", color=col, ms=5,
                label=f"{lb} $-$ its own 0.25 fs value")
    ax.axhline(0, color="0.5", lw=0.8)
    # dt^2 on the x-axis, so an O(dt^2) quantity is a STRAIGHT LINE THROUGH THE ORIGIN and the
    # claim is checkable by eye rather than from the ratio in the caption.
    ax.set_xlabel("$dt^2$ (fs$^2$)")
    ax.set_ylabel("deviation from 300 K (K)")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(-8.6, 2.2)
    ax.legend(fontsize=7, loc="lower left", ncol=1, framealpha=0.9)
    ax.set_title("D  CORRECTION 2: $T_{kin}$ collapses as $O(dt^2)$ and does not\n"
                 "depend on the restraint; the configurational estimators do not move",
                 fontsize=10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-map", default="results/valine/state_map")
    ap.add_argument("--pilot", default="results/valine/pilot_reference")
    ap.add_argument("--dt-bias", default="results/valine/dt_bias")
    ap.add_argument("--out", default="results/valine/figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 11))
    panel_states(axes[0, 0], a.state_map)
    have_pilot = os.path.exists(os.path.join(a.pilot, "pilot_reference.npz"))
    if have_pilot:
        im = panel_pilot(axes[0, 1], a.pilot)
        fig.colorbar(im, ax=axes[0, 1], label="$F$ (kT)", fraction=0.046)
    else:
        axes[0, 1].text(0.5, 0.5, "pilot reference not built yet", ha="center", va="center")
    panel_rate(axes[1, 0], a.state_map)
    panel_dt(axes[1, 1], a.dt_bias)
    for ax in (axes[0, 0], axes[0, 1]):
        ax.set_xlabel("$\\phi$ (deg)")
        ax.set_ylabel("$\\chi_1$ (deg)")
        ax.set_xlim(-180, 180)
        ax.set_ylim(-180, 180)
        ax.set_xticks(range(-180, 181, 90))
        ax.set_yticks(range(-180, 181, 90))
    fig.suptitle("Ace-Val-Nme screening: state map, pilot reference, and two Stage-0 corrections",
                 fontsize=12.5, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    p = os.path.join(a.out, "valine_screen.png")
    fig.savefig(p, dpi=150)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
