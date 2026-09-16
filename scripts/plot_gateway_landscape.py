#!/usr/bin/env python
"""Presentation heatmap: the entropic-gateway energy landscape V(x, y) in kT.

    V(x, y) = H (x^2 - 1)^2 + 1/2 omega(x)^2 y^2,
    omega(x) = omega_out + (omega_in - omega_out) exp(-x^2 / (2 s^2)),  omega_in = r omega_out,

with the parameters of the closed gateway runs (H 0.5, beta 16, r 32, s 0.1, omega_out 1), read
from the confirmation file so the figure cannot drift from the data.  Analytic free energy
F(x) = H (x^2 - 1)^2 + beta^-1 log omega(x); barrier beta*H + log r = 11.47 kT.

    python scripts/plot_gateway_landscape.py
"""
from __future__ import annotations

import json
import os

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RAW = os.path.join(ROOT, "results/information_campaign/gateway_corrected_confirmation/raw.npz")
OUT = os.path.join(ROOT, "results/information_campaign/gateway_corrected_confirmation/figures")

C_INK = "#0b0b0b"
C_INK2 = "#52514e"
XMIN, XMAX = -1.8, 1.8
YMIN, YMAX = -0.8, 0.8
V_MAX_KT = 20.0


def main():
    z = np.load(RAW, allow_pickle=True)
    cfg = json.loads(str(z["config_json"][0]))
    H, beta, r, s, w_out = (float(cfg[k]) for k in ("H", "beta", "r", "s", "omega_out"))
    w_in = r * w_out

    x = np.linspace(XMIN, XMAX, 1801)
    y = np.linspace(YMIN, YMAX, 1601)
    X, Y = np.meshgrid(x, y)
    omega = w_out + (w_in - w_out) * np.exp(-X * X / (2 * s * s))
    V_kT = beta * (H * (X * X - 1) ** 2 + 0.5 * omega ** 2 * Y * Y)

    barrier = beta * H + np.log(r)
    print(f"H {H:g}, beta {beta:g}, r {r:g}, s {s:g}: barrier {barrier:.2f} kT = "
          f"{beta*H:.2f} energetic + {np.log(r):.2f} entropic; sigma_y basin {1/np.sqrt(beta)/w_out:.3f}, "
          f"gateway {1/np.sqrt(beta)/w_in:.4f}")

    plt.rcParams.update({
        "font.size": 14, "axes.labelsize": 17, "xtick.labelsize": 13, "ytick.labelsize": 13,
        "axes.edgecolor": C_INK2, "axes.linewidth": 0.8, "xtick.color": C_INK2, "ytick.color": C_INK2,
        "axes.labelcolor": C_INK, "text.color": C_INK, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(10.5, 5.2), constrained_layout=True)
    im = ax.imshow(np.clip(V_kT, 0, V_MAX_KT), origin="lower", extent=(XMIN, XMAX, YMIN, YMAX),
                   cmap="Blues", vmin=0, vmax=V_MAX_KT, aspect="auto", interpolation="nearest",
                   rasterized=True)
    cs = ax.contour(X, Y, V_kT, levels=[1, 2, 4, 8, 16], colors="white", linewidths=0.7, alpha=0.85)
    ax.clabel(cs, fmt=lambda v: f"{v:g} kT", fontsize=9, colors="white", inline=True, inline_spacing=4)

    # region labels
    ax.text(-1.0, 0.62, "basin  $B_-$", ha="center", va="center", fontsize=15, color=C_INK)
    ax.text(1.0, 0.62, "basin  $B_+$", ha="center", va="center", fontsize=15, color=C_INK)
    ax.annotate("entropic gateway", xy=(0.0, 0.03), xytext=(0.0, 0.45), ha="center", va="bottom",
                fontsize=15, color="white",
                arrowprops=dict(arrowstyle="-|>", color="white", lw=1.2, shrinkA=0, shrinkB=2))

    ax.set_xlabel(r"collective variable  $\xi = x$")
    ax.set_ylabel(r"fast coordinate  $y$")
    ax.set_xticks(np.arange(-1.5, 1.6, 0.5))
    cb = fig.colorbar(im, ax=ax, pad=0.02, extend="max")
    cb.set_label(r"energy  $\beta V(x, y)$  in $k_BT$")
    cb.outline.set_edgecolor(C_INK2)

    os.makedirs(OUT, exist_ok=True)
    base = os.path.join(OUT, "fig_gateway_landscape")
    fig.savefig(base + ".png", dpi=200)
    fig.savefig(base + ".pdf")
    print("saved", base + ".{png,pdf}")


if __name__ == "__main__":
    main()
