#!/usr/bin/env python
"""Ethane/LTA: where the entropy goes -- the configuration cloud in the cage vs in the window.

From a movie record (results/lta_movie/T{T}/movie_data.npz, the ABF arm by default) pool the replica
configurations over a time window after the ABF ramp and draw
  (a) the transverse cross-section (y, z mod a) of the ethane COM for molecules in the cage
      (|z_CV| > 4 A) and in the window (|z_CV| < 1.5 A), over the 8-ring O atoms,
  (b) the distribution of |cos theta| = |u_x| (molecular axis vs the channel axis) in the two regions,
  (c) the transverse radius r_perp = sqrt((y - a/2)^2 + (z - a/2)^2) distribution in the two regions,
and print the numbers (transverse rms, mean |u_x|) -- the "hundreds of times more configurations in the
cage" picture of Schuring et al. (2002), measured on our sampled configurations.

    python scripts/plot_lta_configuration_cloud.py --data results/lta_movie/T300/movie_data.npz
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from publication_style import PALETTE, apply_publication_style, save_figure  # noqa: E402

PI = math.pi
C_CAGE, C_WIN = PALETTE["blue"], PALETTE["vermillion"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--arm", default="abf")
    ap.add_argument("--t-from", type=float, default=None, help="pool snapshots with t >= this (default: ramp end)")
    ap.add_argument("--t-to", type=float, default=None)
    ap.add_argument("--out", default=None, help="figure stem (default next to the data)")
    a = ap.parse_args()
    d = np.load(a.data, allow_pickle=True)
    ap_ = float(d["a_pseudo"]); T = float(d["temperature"])
    t = np.asarray(d["snap_t"], float)
    t_from = float(d["t_warm"]) if a.t_from is None else a.t_from
    t_to = t[-1] if a.t_to is None else a.t_to
    sel = (t >= t_from) & (t <= t_to)
    q = np.asarray(d[f"{a.arm}/snap_q"], np.float64)[sel]           # (S, N, 2, 3)
    phi = np.asarray(d[f"{a.arm}/snap_phi"], np.float64)[sel]        # (S, N)
    com = q.mean(axis=2)                                             # (S, N, 3)
    u = q[:, :, 0, :] - q[:, :, 1, :]; u /= np.linalg.norm(u, axis=-1, keepdims=True)
    zcv = np.abs(phi) * ap_ / (2 * PI)
    cage = zcv > 4.0; win = zcv < 1.5
    yz = np.mod(com[..., 1:], ap_)                                   # transverse, folded into one cell
    ux = np.abs(u[..., 0])
    rperp = np.hypot(yz[..., 0] - ap_ / 2, yz[..., 1] - ap_ / 2)
    o = np.asarray(d["o_pos"], float)
    ring = o[np.abs(np.mod(o[:, 0] + ap_ / 2, ap_) - ap_ / 2) < 1.0]  # O atoms within 1 A of a window plane
    ring_yz = np.mod(ring[:, 1:], ap_)
    n_c, n_w = int(cage.sum()), int(win.sum())
    print(f"T = {T:g} K, arm {a.arm}, pooled {sel.sum()} snapshots t in [{t_from:g}, {t_to:g}]: "
          f"{n_c} cage samples, {n_w} window samples")
    stats = {}
    for name, m in (("cage", cage), ("window", win)):
        if m.sum() == 0:
            print(f"  {name}: no samples"); continue
        stats[name] = dict(rperp_rms=float(np.sqrt((rperp[m] ** 2).mean())), ux_mean=float(ux[m].mean()),
                           ux_frac_above_0_9=float((ux[m] > 0.9).mean()), n=int(m.sum()))
        print(f"  {name:6s}: transverse rms {stats[name]['rperp_rms']:.2f} A, <|cos theta|> {stats[name]['ux_mean']:.3f}, "
              f"P(|cos theta| > 0.9) {stats[name]['ux_frac_above_0_9']:.2f}")
    apply_publication_style()
    fig = plt.figure(figsize=(7.1, 2.6), layout="constrained")
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1.1, 1.1])
    rng = np.random.default_rng(0)
    for k, (name, m, c) in enumerate((("α-cage  (|z| > 4 Å)", cage, C_CAGE), ("8-ring window  (|z| < 1.5 Å)", win, C_WIN))):
        ax = fig.add_subplot(gs[0, k])
        ax.scatter(ring_yz[:, 0], ring_yz[:, 1], s=26, color=PALETTE["light_gray"], edgecolor=PALETTE["gray"], lw=0.4, zorder=1)
        pts = yz[m]
        if len(pts):
            if len(pts) > 6000:
                pts = pts[rng.choice(len(pts), 6000, replace=False)]
            ax.scatter(pts[:, 0], pts[:, 1], s=1.2, color=c, alpha=0.35, lw=0, rasterized=True, zorder=2)
        ax.set_xlim(0, ap_); ax.set_ylim(0, ap_); ax.set_aspect("equal")
        ax.set_xlabel("y (Å)"); ax.set_title(name, fontsize=8, color=c)
        if k == 0:
            ax.set_ylabel("z (Å)")
        else:
            ax.set_yticklabels([])
        ax.text(0.03, 0.97, f"n = {int(m.sum()):,}", transform=ax.transAxes, fontsize=6.5, va="top", color=PALETTE["gray"])
    ax = fig.add_subplot(gs[0, 2])
    bins = np.linspace(0, 1, 21)
    for name, m, c in (("cage", cage, C_CAGE), ("window", win, C_WIN)):
        if m.sum():
            ax.hist(ux[m], bins=bins, density=True, histtype="step", color=c, lw=1.4, label=name)
    ax.axhline(1.0, color=PALETTE["light_gray"], lw=0.8, zorder=0)
    ax.text(0.02, 1.03, "isotropic", fontsize=6.5, color=PALETTE["gray"], va="bottom")
    ax.set_xlabel(r"$|\cos\theta|$, axis vs channel"); ax.set_ylabel("density")
    ax.set_title("orientation", fontsize=8); ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    ax = fig.add_subplot(gs[0, 3])
    rb = np.linspace(0, 6, 31)
    for name, m, c in (("cage", cage, C_CAGE), ("window", win, C_WIN)):
        if m.sum():
            ax.hist(rperp[m], bins=rb, density=True, histtype="step", color=c, lw=1.4, label=name)
    ax.set_xlabel(r"$r_\perp$ off the axis (Å)"); ax.set_ylabel("density")
    ax.set_title("position", fontsize=8); ax.legend(frameon=False, fontsize=6.5)
    fig.suptitle(f"Ethane in LTA, {T:g} K: COM cross-section (y, z) and orientation in the cage vs in the window "
                 f"({a.arm} arm, t ∈ [{t_from:g}, {t_to:g}])", fontsize=8)
    stem = a.out or os.path.join(os.path.dirname(os.path.abspath(a.data)), f"fig_lta_configuration_cloud_T{T:g}")
    save_figure(fig, stem)
    import json
    with open(stem + ".json", "w") as fh:
        json.dump(dict(temperature=T, arm=a.arm, t_from=t_from, t_to=t_to, stats=stats), fh, indent=2)
    print("wrote", os.path.relpath(stem, ROOT))


if __name__ == "__main__":
    main()
