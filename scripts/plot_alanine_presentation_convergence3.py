#!/usr/bin/env python
"""Presentation figure, three panels: alanine dipeptide convergence of F, grad F and the
(phi, psi) marginal, ABF vs ABF+FR (uniform target).

Data: results/uniform_campaign/alanine/N2048_uniform/raw -- the uniform-FR campaign stage,
16 paired seeds, 2048 replicas, 100 ps, FR from 20 ps.  Analysis follows the campaign's
REPAIRED metrics (2026-09-02): the kernel-matched reference uses the row-NORMALISED
wrapped-Gaussian (the unnormalised one scaled the reference ~9.6x and made the endpoint a
constant common to both arms) and the mean-force error uses the periodic central difference
(the spectral derivative rang off the reference's unvisited-cell fill).  Both repaired
endpoints are recomputed here and asserted against analysis/pilot_decision_N2048_uniform.json.

  panel 1  e_F(t)     = equilibrium-weighted aligned L2 of F_hat - K*F_ref, the campaign primary
  panel 2  e_gradF(t) = equilibrium-weighted L2 of grad F_hat - grad F_ref, its co-primary
  panel 3  e_p(t)     = L2 over the torus of the walker marginal minus the uniform density,
                        u = 1/(2 pi)^2, which is exactly this arm's FR target

This system is the project's atomistic NEUTRALITY CONTROL: the expected reading is that the
two arms coincide in all three panels.  The uniform target is also physically unreachable here
-- only 2239 of the 9409 (phi, psi) cells lie below 8 kT -- so panel 3 has a floor neither arm
can cross, and that floor is annotated.

    python scripts/plot_alanine_presentation_convergence3.py
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alanine.metrics_ala import build_masks, smooth_reference, aligned_l2, grad_errors  # noqa: E402
from presentation_endlabels import draw_end_labels  # noqa: E402

STAGE = os.path.join(ROOT, "results/uniform_campaign/alanine/N2048_uniform")
ANA = os.path.join(ROOT, "results/uniform_campaign/alanine/analysis")
REF = os.path.join(ROOT, "results/alanine/reference/reference.npz")
OUT = os.path.join(ROOT, "results/uniform_campaign/alanine/figures")

C_ABF = "#2a78d6"
C_FR = "#eb6834"
C_INK = "#0b0b0b"
C_INK2 = "#52514e"
C_GRID = "#e4e3df"
TWO_PI = 2.0 * math.pi
T_FR = 20.0          # fr_start_steps 20000 x 0.001 ps
WINDOW = (20.0, 100.0)
KM_BANDWIDTH = 0.08  # the estimator's own kernel, configs/uniform_campaign/alanine_uniform.yaml


def med_iqr(a):
    a = np.asarray(a)
    return np.median(a, 1), np.percentile(a, 25, 1), np.percentile(a, 75, 1)


def panel(ax, E, ylab, title, t, floor=None):
    """Draws one panel; returns (paired per-seed median % change at T, end-label spec)."""
    ends = {}
    for m, c, lab in (("abf", C_ABF, "ABF"), ("fr_uniform", C_FR, "ABF + FR")):
        md, lo, hi = med_iqr(E[m])
        keep = t > 0
        ax.fill_between(t[keep], lo[keep], hi[keep], color=c, alpha=0.18, lw=0)
        ax.plot(t[keep], md[keep], color=c, lw=2.4, label=lab)
        ends[m] = (c, lab, md[-1])
    pct = float(np.median(100.0 * (E["fr_uniform"][-1] - E["abf"][-1]) / E["abf"][-1]))
    labels = [(ends["abf"][2], C_ABF, "ABF"),
              (ends["fr_uniform"][2], C_FR, f"ABF + FR  {pct:+.1f}%")]
    if floor is not None:
        ax.axhline(floor, color=C_INK2, lw=1.0, ls="--")
        ax.text(t[-1] * 0.02, floor, "unreachable-target floor ", va="bottom", ha="left",
                color=C_INK2, fontsize=11)
    ax.axvline(T_FR, color=C_INK2, lw=1.0, ls=":")
    ax.text(T_FR, 1.0, " FR on", transform=ax.get_xaxis_transform(),
            va="top", ha="left", color=C_INK2, fontsize=12)
    ax.set_yscale("log")
    ax.set_xlim(0, t[-1] * 1.32)
    ax.set_xticks(np.arange(0, t[-1] + 1, 20))
    ax.set_xlabel("simulation time  $t$  (ps)")
    ax.set_ylabel(ylab)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(True, axis="y", color=C_GRID, lw=0.7)
    ax.set_axisbelow(True)
    return pct, (ax, float(t[-1]), labels)


def main():
    prov = json.load(open(os.path.join(ANA, "reference_provenance_N2048_uniform.json")))
    dec = json.load(open(os.path.join(ANA, "pilot_decision_N2048_uniform.json")))
    kT, n_grid = float(prov["kT_kJ"]), int(prov["n_grid"])
    F_ref = np.load(REF, allow_pickle=True)["F"]
    pack = build_masks(F_ref, kT)
    F_sm = smooth_reference(F_ref, KM_BANDWIDTH, n_grid)
    w = pack["weights"]["equilibrium"]

    runs = {m: np.load(sorted(glob.glob(os.path.join(STAGE, f"raw/*__{m}__*.npz")))[0], allow_pickle=True)
            for m in ("abf", "fr_uniform")}
    t = np.asarray(runs["abf"]["times"], float)
    assert np.allclose(t, np.asarray(runs["fr_uniform"]["times"], float))
    dz = TWO_PI / n_grid
    u_dens = 1.0 / TWO_PI ** 2

    eF, eG, eP = {}, {}, {}
    for m, d in runs.items():
        pmf = np.asarray(d["pmf"], float)
        S, R = pmf.shape[:2]
        eF[m] = np.array([[aligned_l2(pmf[i, r], F_sm, w) for r in range(R)] for i in range(S)])
        eG[m] = np.array([[grad_errors(pmf[i, r], pack["F"], w, n_grid) for r in range(R)] for i in range(S)])
        mh = np.asarray(d["marg_hist"], float) / (dz * dz)          # bin mass -> density
        eP[m] = np.sqrt(((mh - u_dens) ** 2).mean(axis=(-1, -2)))

    # self-check 1: marg_hist is a probability mass function
    dev = np.abs(np.asarray(runs["abf"]["marg_hist"], float).sum(axis=(-1, -2)) - 1).max()
    assert dev < 1e-5, f"marg_hist not normalised ({dev:.1e})"
    # self-check 2: the repaired primary reproduces the frozen decision
    sel = (t >= WINDOW[0]) & (t <= WINDOW[1])
    I = {m: np.trapezoid(eF[m][sel], t[sel], axis=0) for m in eF}
    d_int = float(np.median(100.0 * (I["fr_uniform"] - I["abf"]) / I["abf"]))
    frozen_int = 100.0 * dec["primary_kernel_matched_integrated_FES"]["equilibrium"]["median"]
    print(f"self-check primary (kernel-matched integrated FES, {WINDOW[0]:g}-{WINDOW[1]:g} ps): "
          f"{d_int:+.4f}% (frozen {frozen_int:+.4f}%)")
    assert abs(d_int - frozen_int) < 1e-6
    # self-check 3: the repaired co-primary reproduces the frozen decision
    d_g = float(np.median(100.0 * (eG["fr_uniform"][-1] - eG["abf"][-1]) / eG["abf"][-1]))
    frozen_g = 100.0 * dec["endpoint_mean_force"]["median"]
    print(f"self-check endpoint mean force: {d_g:+.4f}% (frozen {frozen_g:+.4f}%)")
    assert abs(d_g - frozen_g) < 1e-6

    # the floor panel 3 cannot cross: the target puts mass where the molecule never goes
    reach = pack["mask8"]
    floor = math.sqrt(((0.0 - u_dens) ** 2 * (~reach)).sum() / reach.size)
    print(f"reachable cells (F <= 8 kT): {int(reach.sum())} of {reach.size}; "
          f"unreachable-target floor {floor:.5f}")

    plt.rcParams.update({
        "font.size": 14, "axes.labelsize": 16, "axes.titlesize": 17,
        "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 14,
        "axes.edgecolor": C_INK2, "axes.linewidth": 0.8,
        "xtick.color": C_INK2, "ytick.color": C_INK2,
        "axes.labelcolor": C_INK, "text.color": C_INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 4.8), constrained_layout=True)
    results = [
        panel(axes[0], eF, r"free energy error  $\|\hat F_t - F\|_{w}$", "Free energy", t),
        panel(axes[1], eG, r"mean force error  $\|\nabla\hat F_t - \nabla F\|_{w}$", "Mean force", t),
        panel(axes[2], eP, r"marginal error  $\|\hat p_t(\phi,\psi) - u\|_{L^2}$",
              r"Marginal of $\xi$", t, floor=floor),
    ]
    pcts = [r[0] for r in results]
    axes[0].legend(frameon=False, loc="upper right")
    draw_end_labels(fig, [r[1] for r in results])

    os.makedirs(OUT, exist_ok=True)
    base = os.path.join(OUT, "fig_alanine_presentation_convergence3")
    fig.savefig(base + ".png", dpi=200)
    fig.savefig(base + ".pdf")
    print("saved", base + ".{png,pdf}")
    for E, name, pct in ((eF, "F", pcts[0]), (eG, "grad F", pcts[1]), (eP, "p(phi,psi)", pcts[2])):
        print(f"final {name}: ABF {np.median(E['abf'][-1]):.4f}  ABF+FR {np.median(E['fr_uniform'][-1]):.4f}"
              f"  (paired median {pct:+.2f}%)")
    print(f"campaign classification: {dec['classification']}")


if __name__ == "__main__":
    main()
