#!/usr/bin/env python
"""Presentation figure: entropic gateway convergence, ABF vs ABF+FR, at the corrected read-out.

Data: results/information_campaign/gateway_corrected_confirmation/raw.npz (32 fresh pairs,
seeds 400-415 x {left, one_right}).  The engine's saved error series are at the legacy
read-out h = 0.07; the project's primary read-out is h_read* = 0.0175 (frozen plateau rule,
docs/GATEWAY_CORRECTED_BASELINE.md), so both error series are recomputed offline from the
saved force sums / counts with the analyzer's own helpers, and self-checked against the engine
at h = 0.07 and against summary.json at h_read*.

One row, two panels: free-energy error (left) and mean-force error (right), median over the
32 paired seeds with the interquartile band, log-y.

    python scripts/plot_gateway_presentation_convergence.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(SCRIPTS, "..")
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, os.path.join(ROOT, "src"))
from analyze_gateway_bandwidth_audit import mean_force_at, e_f  # noqa: E402
from eb_abffr_core import EVAL_LO, EVAL_HI  # noqa: E402

DIR = os.path.join(ROOT, "results/information_campaign/gateway_corrected_confirmation")
RAW = os.path.join(DIR, "raw.npz")
SUMMARY = os.path.join(DIR, "summary.json")
OUT = os.path.join(DIR, "figures")

C_ABF = "#2a78d6"   # blue
C_FR = "#eb6834"    # orange
C_INK = "#0b0b0b"
C_INK2 = "#52514e"
C_GRID = "#e4e3df"


def e_fp(Fp_t, Fp_ref, dx, mask):
    """engine convention for the mean force: RMS over the eval window, no centring."""
    d = (Fp_t - Fp_ref[..., None, :] if Fp_t.ndim == Fp_ref.ndim + 1 else Fp_t - Fp_ref)[..., mask]
    return np.sqrt((d * d).mean(-1))


def med_iqr(a):
    a = np.asarray(a)
    return np.median(a, 0), np.percentile(a, 25, 0), np.percentile(a, 75, 0)


def main():
    z = np.load(RAW, allow_pickle=True)
    summ = json.load(open(SUMMARY))
    h_star = float(summ["h_read_star"])
    method = np.array([str(m) for m in z["method"]])
    init = np.array([str(i) for i in z["init"]])
    seed = z["seed"].astype(int)
    rows = {}
    for i in range(len(method)):
        rows.setdefault((init[i], seed[i]), {})[method[i]] = i
    pairs = sorted(k for k, v in rows.items() if set(v) >= {"abf", "fr_uniform"})
    ia = np.array([rows[k]["abf"] for k in pairs]); iu = np.array([rows[k]["fr_uniform"] for k in pairs])

    x = np.asarray(z["x_grid"][0], float); dx = float(x[1] - x[0]); mask = (x >= EVAL_LO) & (x <= EVAL_HI)
    t = np.asarray(z["t"][0], float)
    cfg0 = json.loads(str(z["config_json"][0]))
    h_bias, min_count = float(cfg0["h"]), float(cfg0["min_count"])
    Sf, C = np.asarray(z["Sf_t"], float), np.asarray(z["C_t"], float)
    F_ref, Fp_ref = np.asarray(z["F_ref"], float), np.asarray(z["Fp_ref"], float)

    # --- self-check 1: the offline read-out at h_bias reproduces the engine's own series ---
    Fp_leg = mean_force_at(Sf, C, h_bias, dx, min_count)
    dev_f = np.abs(e_f(Fp_leg, F_ref, dx, mask) - np.asarray(z["l2_f_t"], float)).max()
    dev_fp = np.abs(e_fp(Fp_leg, Fp_ref, dx, mask) - np.asarray(z["l2_fp_t"], float)).max()
    print(f"self-check vs engine at h_bias={h_bias:g}: max|dev| e_F {dev_f:.2e}, e_F' {dev_fp:.2e}")
    assert dev_f < 1e-9, "free-energy read-out does not reproduce the engine"
    assert dev_fp < 1e-9, "mean-force read-out does not reproduce the engine"

    # --- the primary read-out ---
    Fp_star = mean_force_at(Sf, C, h_star, dx, min_count)
    eF = e_f(Fp_star, F_ref, dx, mask)
    eFp = e_fp(Fp_star, Fp_ref, dx, mask)

    # --- self-check 2: reproduce summary.json's primary contrasts ---
    I = np.trapezoid(eF, t, axis=1)
    d_int = np.median(100 * (I[iu] - I[ia]) / I[ia]); d_fin = np.median(100 * (eF[iu, -1] - eF[ia, -1]) / eF[ia, -1])
    ref = summ["per_readout"][f"{h_star:g}"]
    print(f"self-check vs summary at h_read*={h_star:g}: dI_F {d_int:+.2f} (summary {ref['d_int']['median']:+.2f}), "
          f"d e_F(T) {d_fin:+.2f} (summary {ref['d_fin']['median']:+.2f})")
    assert abs(d_int - ref["d_int"]["median"]) < 1e-6 and abs(d_fin - ref["d_fin"]["median"]) < 1e-6

    keep = t > 0
    plt.rcParams.update({
        "font.size": 14, "axes.labelsize": 16, "axes.titlesize": 17,
        "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 14,
        "axes.edgecolor": C_INK2, "axes.linewidth": 0.8,
        "xtick.color": C_INK2, "ytick.color": C_INK2,
        "axes.labelcolor": C_INK, "text.color": C_INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6), constrained_layout=True)
    panels = [
        (eF, r"free energy error  $\|\hat F_t - F\|_{L^2}$", "Free energy"),
        (eFp, r"mean force error  $\|\hat F'_t - F'\|_{L^2}$", "Mean force"),
    ]
    arms = [("abf", ia, C_ABF, "ABF"), ("fr_uniform", iu, C_FR, "ABF + FR")]
    for ax, (E, ylab, title) in zip(axes, panels):
        ends = {}
        for key, idx, c, lab in arms:
            md, lo, hi = med_iqr(E[idx])
            ax.fill_between(t[keep], lo[keep], hi[keep], color=c, alpha=0.18, lw=0)
            ax.plot(t[keep], md[keep], color=c, lw=2.4, label=lab)
            ends[key] = (c, lab, md[-1])
        gap = np.log10(ends["abf"][2] / ends["fr_uniform"][2])
        shift = max(0.0, (0.16 - gap) / 2)
        pct = 100 * (ends["fr_uniform"][2] / ends["abf"][2] - 1)
        for key, sgn in (("abf", +1), ("fr_uniform", -1)):
            c, lab, y = ends[key]
            text = lab if key == "abf" else f"{lab}  {pct:+.0f}%"
            ax.annotate(text, xy=(t[-1], y * 10 ** (sgn * shift)), xytext=(6, 0),
                        textcoords="offset points", va="center", ha="left",
                        color=c, fontsize=13, fontweight="bold")
        ax.set_yscale("log")
        ax.set_xlim(0, t[-1] * 1.30)
        ax.set_xticks(np.arange(0, t[-1] + 1, 10))
        ax.set_xlabel("simulation time  $t$")
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(True, axis="y", color=C_GRID, lw=0.7)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, loc="upper right")

    os.makedirs(OUT, exist_ok=True)
    base = os.path.join(OUT, "fig_gateway_presentation_convergence")
    fig.savefig(base + ".png", dpi=200)
    fig.savefig(base + ".pdf")
    print("saved", base + ".{png,pdf}")
    for E, name in ((eF, "F"), (eFp, "F'")):
        a, u = np.median(E[ia, -1]), np.median(E[iu, -1])
        print(f"final {name} at h_read*: ABF {a:.4f}  ABF+FR {u:.4f}  ({100*(u/a-1):+.1f}%)")


if __name__ == "__main__":
    main()
