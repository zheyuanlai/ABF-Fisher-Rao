#!/usr/bin/env python
"""Presentation figures, three panels per temperature: ethane/LTA convergence of F, F' and the
xi-marginal, ABF vs ABF+FR, for the v2 temperature sweep (80 / 150 / 225 / 300 K).

Data: results/uniform_campaign/lta/production_T{T}/{abf,fr_uniform}.npz -- the v2 sweep
(fr_start_steps 20000 = end of the ABF warm-up), 16 paired seeds per temperature.  NOT the
closed v1 300 K stage, whose null was FR lateness.

  panel 1  e_F(t)  = full-circle aligned RMS of F_hat - F_ref, the frozen campaign convention
                     (scripts/analyze_uniform_lta.py); reproduced here and self-checked against
                     each summary_T{T}.json
  panel 2  e_F'(t) = full-circle RMS of F'_hat - F'_ref.  The umbrella/WHAM reference stores no
                     mean force, so F'_ref is derived: the reference F is first smoothed with the
                     SAME wrapped-Gaussian kernel the engine's estimator uses (abf_bandwidth
                     0.05 rad) and then differentiated.  Kernel-matching matters -- on the raw
                     reference the arm contrast moves by a factor of ~2 between a finite-difference
                     and a spectral derivative, and after matching the two agree to <= 1 point.
                     Both schemes are computed and their agreement is asserted.
  panel 3  e_p(t)  = full-circle RMS of p_hat_t(phi) - u, u = 1/(2 pi) the uniform density on the
                     circle, which is exactly the FR arm's target (asserted against q_target).

Median over the 16 seeds with the interquartile band, log-y, FR onset marked.
One figure per temperature.

    python scripts/plot_lta_presentation_convergence3.py
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
LTA = os.path.join(ROOT, "results/uniform_campaign/lta")
OUT = os.path.join(LTA, "figures")
PREREG = os.path.join(ROOT, "configs/uniform_campaign/lta_sweep_prereg.json")

TEMPS = (80, 150, 225, 300)
C_ABF = "#2a78d6"
C_FR = "#eb6834"
C_INK = "#0b0b0b"
C_INK2 = "#52514e"
C_GRID = "#e4e3df"
TWO_PI = 2.0 * math.pi


def circular_interp_ref(F_ref, grid_ref, grid_eng):
    """analyze_uniform_lta.circular_interp_ref."""
    order = np.argsort(grid_ref)
    gr, fr = grid_ref[order], F_ref[order]
    gx = np.concatenate([gr - TWO_PI, gr, gr + TWO_PI])
    fx = np.concatenate([fr, fr, fr])
    return np.interp(grid_eng, gx, fx)


def error_series(pmf, F_ref_on_grid):
    """analyze_uniform_lta.error_series: aligned full-circle RMS per (save, seed)."""
    d = pmf - F_ref_on_grid[None, None, :]
    d = d - d.mean(axis=-1, keepdims=True)
    return np.sqrt((d * d).mean(axis=-1))


def rms_series(A, ref):
    return np.sqrt(((A - ref[None, None, :]) ** 2).mean(axis=-1))


def smoothing_matrix(grid, bw):
    """Row-normalised wrapped Gaussian, the kernel of alkanes.periodic at the engine bandwidth."""
    d = np.abs(grid[:, None] - grid[None, :])
    d = np.minimum(d, TWO_PI - d)
    K = np.exp(-0.5 * (d / bw) ** 2)
    return K / K.sum(axis=1, keepdims=True)


def grad_fd(F, dphi):
    Fc = np.concatenate([F[-3:], F, F[:3]])
    return np.gradient(Fc, dphi)[3:-3]


def grad_spectral(F, dphi):
    k = np.fft.fftfreq(len(F), d=dphi / TWO_PI)
    return np.real(np.fft.ifft(1j * k * np.fft.fft(F)))


def med_iqr(a):
    a = np.asarray(a)
    return np.median(a, 1), np.percentile(a, 25, 1), np.percentile(a, 75, 1)


def panel(ax, E, ylab, title, t, t_fr):
    """Draws one panel; returns the PAIRED per-seed median percentage change at T, which is the
    campaign's endpoint convention (median of per-seed ratios, NOT the ratio of the medians)."""
    ends = {}
    for m, c, lab in (("abf", C_ABF, "ABF"), ("fr_uniform", C_FR, "ABF + FR")):
        md, lo, hi = med_iqr(E[m])
        keep = t > 0
        ax.fill_between(t[keep], lo[keep], hi[keep], color=c, alpha=0.18, lw=0)
        ax.plot(t[keep], md[keep], color=c, lw=2.4, label=lab)
        ends[m] = (c, lab, md[-1])
    pct = float(np.median(100.0 * (E["fr_uniform"][-1] - E["abf"][-1]) / E["abf"][-1]))
    gap = np.log10(ends["abf"][2] / ends["fr_uniform"][2])
    shift = max(0.0, (0.16 - abs(gap)) / 2)
    up = "abf" if gap >= 0 else "fr_uniform"
    for m in ("abf", "fr_uniform"):
        c, lab, y = ends[m]
        sgn = +1 if m == up else -1
        text = lab if m == "abf" else f"{lab}  {pct:+.0f}%"
        ax.annotate(text, xy=(t[-1], y * 10 ** (sgn * shift)), xytext=(6, 0),
                    textcoords="offset points", va="center", ha="left",
                    color=c, fontsize=13, fontweight="bold")
    ax.axvline(t_fr, color=C_INK2, lw=1.0, ls=":")
    ax.text(t_fr, 1.0, " FR on", transform=ax.get_xaxis_transform(),
            va="top", ha="left", color=C_INK2, fontsize=12)
    ax.set_yscale("log")
    ax.set_xlim(0, t[-1] * 1.32)
    ax.set_xticks(np.arange(0, t[-1] + 1, 10))
    ax.set_xlabel("simulation time  $t$")
    ax.set_ylabel(ylab)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(True, axis="y", color=C_GRID, lw=0.7)
    ax.set_axisbelow(True)
    return pct


def main():
    pre = json.load(open(PREREG))
    bw = float(pre["sampler"]["abf_bandwidth"])
    fr_start = float(pre["sampler"]["fr_start_steps"])
    n_steps = float(pre["sampler"]["n_steps"])

    plt.rcParams.update({
        "font.size": 14, "axes.labelsize": 16, "axes.titlesize": 17,
        "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 14,
        "axes.edgecolor": C_INK2, "axes.linewidth": 0.8,
        "xtick.color": C_INK2, "ytick.color": C_INK2,
        "axes.labelcolor": C_INK, "text.color": C_INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    os.makedirs(OUT, exist_ok=True)
    print(f"{'T':>5} {'dI_F check':>22} {'d e_F(T) check':>24}   final deltas  F / F' / p")
    for T in TEMPS:
        ref = np.load(os.path.join(LTA, f"reference/reference_T{T}.npz"), allow_pickle=True)
        runs = {m: np.load(os.path.join(LTA, f"production_T{T}/{m}.npz"), allow_pickle=True)
                for m in ("abf", "fr_uniform")}
        summ = json.load(open(os.path.join(LTA, f"summary_T{T}.json")))
        grid = np.asarray(runs["abf"]["grid"], float)
        dphi = float(runs["abf"]["dphi"])
        t = np.asarray(runs["abf"]["times"], float)
        assert np.allclose(t, np.asarray(runs["fr_uniform"]["times"], float))
        t_fr = t[-1] * fr_start / n_steps

        F_ref = circular_interp_ref(ref["F"], ref["grid_phi"], grid)
        # kernel-matched reference mean force; two derivative schemes must agree
        F_ref_k = smoothing_matrix(grid, bw) @ F_ref
        Fp_fd, Fp_sp = grad_fd(F_ref_k, dphi), grad_spectral(F_ref_k, dphi)
        u = np.full_like(grid, 1.0 / TWO_PI)

        eF, eFp, eFp_alt, eP = {}, {}, {}, {}
        for m, r in runs.items():
            eF[m] = error_series(np.asarray(r["pmf"], float), F_ref)
            mf = np.asarray(r["mean_force"], float)
            eFp[m] = rms_series(mf, Fp_fd)
            eFp_alt[m] = rms_series(mf, Fp_sp)
            eP[m] = rms_series(np.asarray(r["p_hat"], float), u)

        # self-check 1: the FR arm's target really is the uniform density
        q = np.asarray(runs["fr_uniform"]["q_target"], float)[-1]
        assert np.abs(q - 1.0 / TWO_PI).max() < 1e-6, "FR target is not uniform"
        # self-check 2: p_hat is a normalised density
        dev_n = np.abs(np.asarray(runs["abf"]["p_hat"], float).sum(-1) * dphi - 1).max()
        assert dev_n < 1e-6, f"p_hat not normalised ({dev_n:.1e})"
        # self-check 3: the free-energy contrasts reproduce the frozen summary
        I = {m: np.trapezoid(eF[m], t, axis=0) for m in eF}
        d_int = np.median(100 * (I["fr_uniform"] - I["abf"]) / I["abf"])
        d_fin = np.median(100 * (eF["fr_uniform"][-1] - eF["abf"][-1]) / eF["abf"][-1])
        s_int, s_fin = summ["d_int_pct"]["median"], summ["d_final_pct"]["median"]
        assert abs(d_int - s_int) < 1e-6 and abs(d_fin - s_fin) < 1e-6
        # self-check 4: the mean-force contrast does not depend on the derivative scheme
        c_fd = 100 * (np.median(eFp["fr_uniform"][-1]) / np.median(eFp["abf"][-1]) - 1)
        c_sp = 100 * (np.median(eFp_alt["fr_uniform"][-1]) / np.median(eFp_alt["abf"][-1]) - 1)
        assert abs(c_fd - c_sp) < 2.0, f"derivative scheme moves the F' contrast ({c_fd:.1f} vs {c_sp:.1f})"

        fig, axes = plt.subplots(1, 3, figsize=(18.5, 4.8), constrained_layout=True)
        pcts = [
            panel(axes[0], eF, r"free energy error  $\|\hat F_t - F\|_{L^2}$", "Free energy", t, t_fr),
            panel(axes[1], eFp, r"mean force error  $\|\hat F'_t - F'\|_{L^2}$", "Mean force", t, t_fr),
            panel(axes[2], eP, r"marginal error  $\|\hat p_t(\phi) - u\|_{L^2}$", r"Marginal of $\xi$", t, t_fr),
        ]
        axes[0].legend(frameon=False, loc="upper right")
        base = os.path.join(OUT, f"fig_lta_presentation_convergence3_T{T}")
        fig.savefig(base + ".png", dpi=200)
        fig.savefig(base + ".pdf")
        plt.close(fig)
        print(f"{T:>5} {d_int:+10.2f} (frozen {s_int:+7.2f}) {d_fin:+10.2f} (frozen {s_fin:+7.2f})   "
              f"{pcts[0]:+6.1f}% {pcts[1]:+6.1f}% {pcts[2]:+6.1f}%   -> {os.path.basename(base)}.{{png,pdf}}")


if __name__ == "__main__":
    main()
