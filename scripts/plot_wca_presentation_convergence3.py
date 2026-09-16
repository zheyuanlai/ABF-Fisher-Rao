#!/usr/bin/env python
"""Presentation figure, three panels: WCA dimer convergence of F, F' and the xi-marginal.

Data: results/information_campaign/wca_corrected_confirmation/confirmation/raw (16 fresh paired
seeds 700-715, the project's primary WCA result).  These runs -- unlike the uniform-campaign runs
400-415 -- saved the raw per-bin count accumulators at every save, so the walker marginal along
xi is available for BOTH arms at every time.  F and F' are read out at the primary bandwidth
h_read* = 0.0125 with the analyzer's own helpers and self-checked against summary.json.

  panel 1  e_F(t)   = aligned RMS of F_hat - F_ref over the eval window
  panel 2  e_F'(t)  = RMS of F'_hat - F'_ref over the eval window
  panel 3  e_p(t)   = RMS of p_hat_t(xi) - u over the eval window, u = uniform on the xi domain,
                      p_hat_t = time-averaged walker marginal over the save interval ending at t
                      (interval difference of the cumulative raw counts; the warm-up reset
                      interval is dropped)

Median over seeds with the interquartile band, log-y, FR onset marked.

    python scripts/plot_wca_presentation_convergence3.py
"""
from __future__ import annotations

import glob
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
from analyze_wca_bandwidth_audit import readouts, Z_LO, Z_HI  # noqa: E402
from wca_abffr_core import profile_l2_error_np  # noqa: E402

DIR = os.path.join(ROOT, "results/information_campaign/wca_corrected_confirmation")
RAW = os.path.join(DIR, "confirmation/raw")
SUMMARY = os.path.join(DIR, "summary.json")
OUT = os.path.join(DIR, "figures")

C_ABF = "#2a78d6"
C_FR = "#eb6834"
C_INK = "#0b0b0b"
C_INK2 = "#52514e"
C_GRID = "#e4e3df"


def load():
    runs = {}
    for f in sorted(glob.glob(os.path.join(RAW, "confirmation__*.npz"))):
        d = np.load(f, allow_pickle=True)
        runs[(str(d["name"]), int(d["seed"]))] = {k: d[k] for k in d.files}
    seeds = sorted({s for (m, s) in runs if m == "abf" and ("fr_uniform", s) in runs})
    return runs, seeds


def med_iqr(a):
    a = np.asarray(a)
    return (np.nanmedian(a, 0), np.nanpercentile(a, 25, 0), np.nanpercentile(a, 75, 0))


def marginal_error_series(run, grid, mask):
    """RMS distance of the interval-averaged walker marginal from uniform, per save (NaN at t=0 and
    at the warm-up reset interval)."""
    C = np.asarray(run["raw_csum_t"], float)
    dC = np.diff(C, axis=0)
    u = np.full_like(grid, 1.0 / (grid[-1] - grid[0]))
    out = np.full(C.shape[0], np.nan)
    for k in range(1, C.shape[0]):
        n = dC[k - 1].sum()
        if n <= 0:              # accumulator reset inside this interval
            continue
        p = dC[k - 1] / (n * (grid[1] - grid[0]))
        out[k] = profile_l2_error_np(p, u, grid, mask=mask)
    return out


def main():
    runs, seeds = load()
    summ = json.load(open(SUMMARY))
    h_star = str(summ["h_read_star"])
    r0 = runs[("abf", seeds[0])]
    grid = np.asarray(r0["grid"], float)
    mask = (grid >= Z_LO) & (grid <= Z_HI)
    ref_F = np.asarray(r0["reference_free_energy"], float)
    ref_Fp = np.asarray(r0["reference_mean_force"], float)
    t = np.asarray(r0["profile_times"], float)
    sigma = float(r0.get("abf_smooth_sigma", 0.5))
    spec = json.loads(str(r0["spec_json"]))
    t_fr = float(spec["fr_start_steps"]) * float(r0["profile_times"][1] / r0["profile_steps"][1]) \
        if "profile_steps" in r0 else 40.0
    key = f"readout_mean_force_t__h{h_star}"

    eF, eFp, eP = {}, {}, {}
    for m in ("abf", "fr_uniform"):
        eF[m] = np.stack([readouts(runs[(m, s)], grid, mask, ref_F, sigma)[h_star] for s in seeds])
        eFp[m] = np.stack([[profile_l2_error_np(mf, ref_Fp, grid, mask=mask) for mf in runs[(m, s)][key]]
                           for s in seeds])
        eP[m] = np.stack([marginal_error_series(runs[(m, s)], grid, mask) for s in seeds])

    # self-check 1: mean-force error convention reproduces the engine's l2_fp_t at the legacy read-out
    dev = max(np.abs(np.array([profile_l2_error_np(mf, ref_Fp, grid, mask=mask) for mf in runs[(m, s)]["mean_force_t"]])
                     - np.asarray(runs[(m, s)]["l2_fp_t"], float)).max() for m in ("abf", "fr_uniform") for s in seeds)
    print(f"self-check e_F' convention vs engine l2_fp_t (legacy read-out): max|dev| {dev:.2e}")
    assert dev < 1e-5
    # self-check 2: reproduce summary.json's primary contrasts at h_read*
    I = {m: np.trapezoid(eF[m], t, axis=1) for m in eF}
    d_int = np.median(100 * (I["fr_uniform"] - I["abf"]) / I["abf"])
    d_fin = np.median(100 * (eF["fr_uniform"][:, -1] - eF["abf"][:, -1]) / eF["abf"][:, -1])
    ref = summ["per_readout"][h_star]
    print(f"self-check vs summary at h_read*={h_star}: dI_F {d_int:+.2f} (summary {ref['d_int']['median']:+.2f}), "
          f"d e_F(T) {d_fin:+.2f} (summary {ref['d_fin']['median']:+.2f})")
    assert abs(d_int - ref["d_int"]["median"]) < 1e-6 and abs(d_fin - ref["d_fin"]["median"]) < 1e-6

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
            md, lo, hi = med_iqr(E[m])
            ok = (t > 0) & np.isfinite(md)
            ax.fill_between(t[ok], lo[ok], hi[ok], color=c, alpha=0.18, lw=0)
            ax.plot(t[ok], md[ok], color=c, lw=2.4, label=lab)
            ends[m] = (c, lab, md[-1])
        gap = np.log10(ends["abf"][2] / ends["fr_uniform"][2])
        shift = max(0.0, (0.16 - abs(gap)) / 2)
        # paired per-seed median, the campaign endpoint convention (NOT the ratio of medians)
        fr_fin, abf_fin = E["fr_uniform"][:, -1], E["abf"][:, -1]
        pct = float(np.nanmedian(100.0 * (fr_fin - abf_fin) / abf_fin))
        up = "abf" if gap >= 0 else "fr_uniform"
        for m in ("abf", "fr_uniform"):
            c, lab, y = ends[m]
            sgn = +1 if m == up else -1
            text = lab if m == "abf" else f"{lab}  {pct:+.0f}%"
            ax.annotate(text, xy=(t[-1], y * 10 ** (sgn * shift)), xytext=(6, 0), textcoords="offset points",
                        va="center", ha="left", color=c, fontsize=13, fontweight="bold")
        ax.axvline(t_fr, color=C_INK2, lw=1.0, ls=":")
        ax.text(t_fr, 1.0, " FR on", transform=ax.get_xaxis_transform(), va="top", ha="left", color=C_INK2, fontsize=12)
        ax.set_yscale("log")
        ax.set_xlim(0, t[-1] * 1.32)
        ax.set_xticks(np.arange(0, t[-1] + 1, 50))
        ax.set_xlabel("simulation time  $t$")
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(True, axis="y", color=C_GRID, lw=0.7)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, loc="upper right")

    os.makedirs(OUT, exist_ok=True)
    base = os.path.join(OUT, "fig_wca_presentation_convergence3")
    fig.savefig(base + ".png", dpi=200)
    fig.savefig(base + ".pdf")
    print("saved", base + ".{png,pdf}")
    for E, name in ((eF, "F"), (eFp, "F'"), (eP, "p(xi)")):
        a, u = np.nanmedian(E["abf"][:, -1]), np.nanmedian(E["fr_uniform"][:, -1])
        pct = np.nanmedian(100.0 * (E["fr_uniform"][:, -1] - E["abf"][:, -1]) / E["abf"][:, -1])
        print(f"final {name}: ABF {a:.4f}  ABF+FR {u:.4f}  (paired median {pct:+.1f}%)")


if __name__ == "__main__":
    main()
