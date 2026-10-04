#!/usr/bin/env python
"""Ensemble-level view of an LTA movie record: space-time (kymograph) plots of the REPLICA DENSITY
along the CV for ABF and ABF + FR, their paired difference, the error field of the learned free energy,
with every Fisher-Rao death / birth overlaid where and when it happened.  Prototype for an
"ensemble movie" (the movie version is the same picture with a moving time cursor).

    python scripts/plot_lta_ensemble_kymograph.py --data results/lta_movie/T150/movie_data.npz
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from publication_style import PALETTE, apply_publication_style, save_figure  # noqa: E402

PI = math.pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--nz", type=int, default=60)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = np.load(a.data, allow_pickle=True)
    ap_ = float(d["a_pseudo"]); T = float(d["temperature"]); kT = float(d["kT"]); N = int(d["N"])
    t = np.asarray(d["snap_t"], float)
    edges = np.linspace(-ap_ / 2, ap_ / 2, a.nz + 1); zc = 0.5 * (edges[1:] + edges[:-1])
    uni = N / a.nz
    dens, err = {}, {}
    F_ref = np.asarray(d["F_ref"], float); zg = np.asarray(d["z_grid"], float)
    for m in ("abf", "fr_uniform"):
        z = np.asarray(d[f"{m}/snap_phi"], float) * ap_ / (2 * PI)             # (S, N)
        H = np.stack([np.histogram(z[s], bins=edges)[0] for s in range(len(t))])  # (S, nz)
        dens[m] = H / uni
        F = np.asarray(d[f"{m}/snap_F"], float)
        e = F - F_ref[None, :]; e -= e.mean(axis=1, keepdims=True)
        err[m] = e / kT                                                           # (S, 180) kT
    apply_publication_style()
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 6.0), layout="constrained",
                             gridspec_kw=dict(width_ratios=[1, 1, 1.0]))
    ext = (t[0], t[-1], edges[0], edges[-1])
    norm = TwoSlopeNorm(vmin=-1.5, vcenter=0.0, vmax=1.5)      # log2(density / uniform)
    for j, (m, lab) in enumerate((("abf", "ABF"), ("fr_uniform", "ABF + Fisher–Rao"))):
        ax = axes[0, j]
        im = ax.imshow(np.log2(np.clip(dens[m], 1 / 32, 32)).T, origin="lower", aspect="auto", extent=ext,
                       cmap="RdBu_r", norm=norm, interpolation="nearest")
        ax.set_title(f"{lab}: replica density / uniform", fontsize=8)
        if m == "fr_uniform":
            die = d["fr_uniform/snap_ev_die"]; birth = d["fr_uniform/snap_ev_birth"]
            td = t[np.clip(die[:, 0].astype(int) - 1, 0, len(t) - 1)]
            ax.scatter(td, die[:, 1] * ap_ / (2 * PI), s=1.2, color="#7f0000", lw=0, alpha=0.6, label="death")
            ax.scatter(td, birth[:, 1] * ap_ / (2 * PI), s=1.2, color="#004d1a", lw=0, alpha=0.6, label="birth (source)")
            ax.legend(loc="upper right", fontsize=6, frameon=True, framealpha=0.8, markerscale=4)
        ax.axvline(float(d["t_fr"]), color=PALETTE["black"], lw=0.6, ls=":")
        ax.set_ylabel("z (Å)  window 0, cages ±a/2")
    ax = axes[0, 2]
    im2 = ax.imshow((dens["fr_uniform"] - dens["abf"]).T, origin="lower", aspect="auto", extent=ext, cmap="PuOr_r",
                    norm=TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0), interpolation="nearest")
    ax.set_title("difference FR − ABF (uniform units)", fontsize=8)
    fig.colorbar(im, ax=axes[0, :2].tolist(), fraction=0.03, pad=0.01, label=r"$\log_2$(density / uniform)")
    fig.colorbar(im2, ax=axes[0, 2], fraction=0.06, pad=0.02)
    enorm = TwoSlopeNorm(vmin=-2.0, vcenter=0.0, vmax=2.0)
    for j, m in enumerate(("abf", "fr_uniform")):
        ax = axes[1, j]
        im3 = ax.imshow(err[m].T, origin="lower", aspect="auto", extent=(t[0], t[-1], zg[0], zg[-1]), cmap="RdBu_r",
                        norm=enorm, interpolation="nearest")
        ax.set_title(r"learned $\hat F_t(z) - F_{\rm ref}(z)$  [$k_BT$]", fontsize=8)
        ax.axvline(float(d["t_fr"]), color=PALETTE["black"], lw=0.6, ls=":")
        ax.set_ylabel("z (Å)")
    fig.colorbar(im3, ax=axes[1, :2].tolist(), fraction=0.03, pad=0.01, label=r"error [$k_BT$]")
    ax = axes[1, 2]
    for m, c, lab in (("abf", PALETTE["blue"], "ABF"), ("fr_uniform", PALETTE["vermillion"], "ABF + FR")):
        ax.plot(t, np.asarray(d[f"{m}/snap_eF"], float) / kT, color=c, lw=1.2, label=lab)
    ax.set_yscale("log"); ax.set_title(r"$e_F(t)$  [$k_BT$]", fontsize=8); ax.legend(frameon=False, fontsize=6.5)
    ax.axvline(float(d["t_fr"]), color=PALETTE["black"], lw=0.6, ls=":")
    # bottom: occupancy of the three regions over time, both arms; and the window band of the density
    ax = axes[2, 0]
    for m, c, lab in (("abf", PALETTE["blue"], "ABF"), ("fr_uniform", PALETTE["vermillion"], "ABF + FR")):
        ax.plot(t, np.asarray(d[f"{m}/frac_window"], float), color=c, lw=1.3, label=f"{lab}: window")
        ax.plot(t, np.asarray(d[f"{m}/frac_cage"], float), color=c, lw=1.0, ls="--", label=f"{lab}: cage")
    ax.axhline(3.0 / ap_, color=PALETTE["gray"], lw=0.8, ls=":"); ax.axhline(2 * (ap_ / 2 - 4.0) / ap_, color=PALETTE["gray"], lw=0.8, ls=":")
    ax.set_title("occupancy of window (|z|<1.5 Å) and cages (|z|>4 Å); dotted = uniform", fontsize=7.5)
    ax.set_xlabel("t"); ax.legend(frameon=False, fontsize=6, ncol=2)
    ax = axes[2, 1]
    kl = {}
    for m, c, lab in (("abf", PALETTE["blue"], "ABF"), ("fr_uniform", PALETTE["vermillion"], "ABF + FR")):
        p = np.clip(dens[m] / a.nz, 1e-12, None)                      # probability per bin
        kl[m] = (p * np.log(p * a.nz)).sum(axis=1)
        ax.plot(t, kl[m], color=c, lw=1.2, label=lab)
    ax.set_yscale("log"); ax.set_title(r"$D_{KL}(\hat p_t\,\|\,{\rm uniform})$ from the 60-bin histogram", fontsize=7.5)
    ax.set_xlabel("t"); ax.legend(frameon=False, fontsize=6.5)
    ax = axes[2, 2]
    cA = np.asarray(d["abf/snap_crossings"], float); cF = np.asarray(d["fr_uniform/snap_crossings"], float)
    ax.plot(t, cA / N, color=PALETTE["blue"], lw=1.2, label="ABF"); ax.plot(t, cF / N, color=PALETTE["vermillion"], lw=1.2, label="ABF + FR")
    ax.set_title("cumulative cage→cage crossings per replica", fontsize=7.5); ax.set_xlabel("t"); ax.legend(frameon=False, fontsize=6.5)
    for ax in axes[0, :].tolist() + axes[1, :2].tolist():
        ax.set_xlabel("t")
    fig.suptitle(f"Ethane/LTA {T:g} K, one paired seed: the ENSEMBLE over time (replica density and learned-F error as space–time fields)", fontsize=8.5)
    stem = a.out or os.path.join(os.path.dirname(os.path.abspath(a.data)), f"fig_lta_ensemble_kymograph_T{T:g}")
    save_figure(fig, stem); print("wrote", os.path.relpath(stem, ROOT))
    i10 = np.searchsorted(t, 10.0); i20 = np.searchsorted(t, 20.0)
    print(f"window-band density/uniform at t=10: ABF {dens['abf'][i10, a.nz//2-3:a.nz//2+3].mean():.2f}  FR {dens['fr_uniform'][i10, a.nz//2-3:a.nz//2+3].mean():.2f}; "
          f"t=20: {dens['abf'][i20, a.nz//2-3:a.nz//2+3].mean():.2f} / {dens['fr_uniform'][i20, a.nz//2-3:a.nz//2+3].mean():.2f}")


if __name__ == "__main__":
    main()
