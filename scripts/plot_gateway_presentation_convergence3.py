#!/usr/bin/env python
"""Presentation figure, three panels: entropic gateway convergence of F, F' and the xi-marginal.

Same data and read-out as plot_gateway_presentation_convergence.py (32 fresh pairs, h_read* =
0.0175, recomputed from the raw accumulators with self-checks).  Third panel: RMS distance of
the walker marginal p_hat_t(xi) (the engine's per-save KDE, eta = 0.1) from the uniform density
u = 1/(XMAX - XMIN) over the eval window, both arms.

    python scripts/plot_gateway_presentation_convergence3.py
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
from eb_abffr_core import EVAL_LO, EVAL_HI, XMIN, XMAX  # noqa: E402

DIR = os.path.join(ROOT, "results/information_campaign/gateway_corrected_confirmation")
RAW = os.path.join(DIR, "raw.npz")
SUMMARY = os.path.join(DIR, "summary.json")
OUT = os.path.join(DIR, "figures")

C_ABF = "#2a78d6"
C_FR = "#eb6834"
C_INK = "#0b0b0b"
C_INK2 = "#52514e"
C_GRID = "#e4e3df"


def rms_window(A, ref, mask):
    d = (A - ref[..., None, :] if A.ndim == ref.ndim + 1 else A - ref)[..., mask]
    return np.sqrt((d * d).mean(-1))


def med_iqr(a):
    a = np.asarray(a)
    return np.median(a, 0), np.percentile(a, 25, 0), np.percentile(a, 75, 0)


def main():
    z = np.load(RAW, allow_pickle=True)
    summ = json.load(open(SUMMARY))
    h_star = float(summ["h_read_star"])
    method = np.array([str(m) for m in z["method"]]); init = np.array([str(i) for i in z["init"]])
    seed = z["seed"].astype(int)
    rows = {}
    for i in range(len(method)):
        rows.setdefault((init[i], seed[i]), {})[method[i]] = i
    pairs = sorted(k for k, v in rows.items() if set(v) >= {"abf", "fr_uniform"})
    idx = {"abf": np.array([rows[k]["abf"] for k in pairs]), "fr_uniform": np.array([rows[k]["fr_uniform"] for k in pairs])}

    x = np.asarray(z["x_grid"][0], float); dx = float(x[1] - x[0]); mask = (x >= EVAL_LO) & (x <= EVAL_HI)
    t = np.asarray(z["t"][0], float)
    cfg0 = json.loads(str(z["config_json"][0])); h_bias, min_count = float(cfg0["h"]), float(cfg0["min_count"])
    Sf, C = np.asarray(z["Sf_t"], float), np.asarray(z["C_t"], float)
    F_ref, Fp_ref = np.asarray(z["F_ref"], float), np.asarray(z["Fp_ref"], float)

    Fp_leg = mean_force_at(Sf, C, h_bias, dx, min_count)
    dev_f = np.abs(e_f(Fp_leg, F_ref, dx, mask) - np.asarray(z["l2_f_t"], float)).max()
    dev_fp = np.abs(rms_window(Fp_leg, Fp_ref, mask) - np.asarray(z["l2_fp_t"], float)).max()
    print(f"self-check vs engine at h_bias={h_bias:g}: max|dev| e_F {dev_f:.2e}, e_F' {dev_fp:.2e}")
    assert dev_f < 1e-9 and dev_fp < 1e-9

    Fp_star = mean_force_at(Sf, C, h_star, dx, min_count)
    eF, eFp = e_f(Fp_star, F_ref, dx, mask), rms_window(Fp_star, Fp_ref, mask)
    phat = np.asarray(z["phat_t"], float)
    u = np.full_like(x, 1.0 / (XMAX - XMIN))
    eP = rms_window(phat, u, mask)
    dev_n = np.abs(np.trapezoid(phat, dx=dx, axis=-1) - 1).max()
    print(f"self-check phat_t normalisation: max|trapz - 1| {dev_n:.1e}")
    assert dev_n < 1e-6

    ia, iu = idx["abf"], idx["fr_uniform"]
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
        "axes.edgecolor": C_INK2, "axes.linewidth": 0.8, "xtick.color": C_INK2, "ytick.color": C_INK2,
        "axes.labelcolor": C_INK, "text.color": C_INK, "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 4.8), constrained_layout=True)
    panels = [
        (eF, r"free energy error  $\|\hat F_t - F\|_{L^2}$", "Free energy"),
        (eFp, r"mean force error  $\|\hat F'_t - F'\|_{L^2}$", "Mean force"),
        (eP, r"marginal error  $\|\hat p_t(\xi) - u\|_{L^2}$", r"Marginal of $\xi$"),
    ]
    arms = [("abf", C_ABF, "ABF"), ("fr_uniform", C_FR, "ABF + FR")]
    for ax, (E, ylab, title) in zip(axes, panels):
        ends = {}
        for m, c, lab in arms:
            md, lo, hi = med_iqr(E[idx[m]])
            ax.fill_between(t[keep], lo[keep], hi[keep], color=c, alpha=0.18, lw=0)
            ax.plot(t[keep], md[keep], color=c, lw=2.4, label=lab)
            ends[m] = (c, lab, md[-1])
        gap = np.log10(ends["abf"][2] / ends["fr_uniform"][2])
        shift = max(0.0, (0.16 - abs(gap)) / 2)
        # paired per-seed median, the campaign endpoint convention (NOT the ratio of medians)
        fr_fin, abf_fin = E[idx["fr_uniform"]][:, -1], E[idx["abf"]][:, -1]
        pct = float(np.nanmedian(100.0 * (fr_fin - abf_fin) / abf_fin))
        up = "abf" if gap >= 0 else "fr_uniform"
        for m in ("abf", "fr_uniform"):
            c, lab, y = ends[m]
            sgn = +1 if m == up else -1
            text = lab if m == "abf" else f"{lab}  {pct:+.0f}%"
            ax.annotate(text, xy=(t[-1], y * 10 ** (sgn * shift)), xytext=(6, 0), textcoords="offset points",
                        va="center", ha="left", color=c, fontsize=13, fontweight="bold")
        ax.set_yscale("log")
        ax.set_xlim(0, t[-1] * 1.32)
        ax.set_xticks(np.arange(0, t[-1] + 1, 10))
        ax.set_xlabel("simulation time  $t$")
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(True, axis="y", color=C_GRID, lw=0.7)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, loc="upper right")

    os.makedirs(OUT, exist_ok=True)
    base = os.path.join(OUT, "fig_gateway_presentation_convergence3")
    fig.savefig(base + ".png", dpi=200)
    fig.savefig(base + ".pdf")
    print("saved", base + ".{png,pdf}")
    for E, name in ((eF, "F"), (eFp, "F'"), (eP, "p(xi)")):
        a, uu = np.median(E[ia, -1]), np.median(E[iu, -1])
        pct = np.median(100.0 * (E[iu, -1] - E[ia, -1]) / E[ia, -1])
        print(f"final {name}: ABF {a:.4f}  ABF+FR {uu:.4f}  (paired median {pct:+.1f}%)")


if __name__ == "__main__":
    main()
