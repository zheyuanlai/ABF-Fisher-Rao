#!/usr/bin/env python
"""Preview PDFs from the exported CSVs and meta/export.json only (the CSVs are the deliverable).

  figs/previews/convergence_2x3.pdf        rows: phi alone, psi alone; columns: e_F, e_F', e_xi; ABF vs FR 0.02 (primary),
                                           median + IQR, log y; FR first acts at 0.5 ps (dotted), ABF ramp ends at 5 ps
  figs/previews/convergence_2x3_rate015.pdf the same with FR 0.15 (supplementary)
  figs/previews/psi_hidden_coordinate.pdf  F(psi) vs references; phi > 0 walker mass vs time; D_cond by window
  figs/previews/mean_force_diagnostics.pdf  e_F' with its reference-uncertainty band, seed dispersion, debiased
"""
import argparse
import csv
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common_ala as CA  # noqa: E402

D = os.path.join(CA.PKG, "figs", "data")
S = os.path.join(CA.PKG, "figs", "data_supplementary")
P = os.path.join(CA.PKG, "figs", "previews")
BLUE, ORANGE, RED = "#1f77b4", "#ff7f0e", "#d62728"
MD = {"CreationDate": None}
YL = {"eF": r"$e_F(t)$ [kJ/mol]", "eFp": r"$e_{F'}(t)$ [kJ/mol/rad]", "exi": r"$e_\xi(t)$ [rad$^{-1/2}$]"}


def read(path):
    with open(path) as fh:
        r = list(csv.reader(fh))
    return r[0], np.array([[float(v) for v in row] for row in r[1:]])


def band(ax, a, c0, color, ls, label):
    ax.fill_between(a[:, 0], a[:, c0 + 1], a[:, c0 + 2], color=color, alpha=0.22, lw=0)
    ax.plot(a[:, 0], a[:, c0], color=color, ls=ls, lw=1.4, label=label)


def conv(rate015, png):
    M = json.load(open(os.path.join(CA.PKG, "meta", "export.json")))
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2))
    frlab = "ABF + uniform FR, rate 0.15" if rate015 else "ABF + uniform FR, rate 0.02"
    for i, cv in enumerate(("phi", "psi")):
        for j, k in enumerate(("eF", "eFp", "exi")):
            ax = axes[i, j]
            path = os.path.join(S, f"{cv}_{k}_rate015.csv") if rate015 else os.path.join(D, f"{cv}_{k}.csv")
            _, a = read(path)
            band(ax, a, 1, BLUE, "-", "histogram ABF")
            band(ax, a, 4, RED if rate015 else ORANGE, "--", frlab)
            ax.set_yscale("log")
            ax.set_xlim(0, 100)
            ax.axvline(0.5, color="0.4", lw=0.8, ls=":")
            ax.axvline(5.0, color="0.7", lw=0.8, ls="-.")
            if k == "eFp":
                rel = M["mean_force_reliability"][cv]
                ax.axhline(rel["reference_uncertainty_total"], color="0.3", lw=0.9, ls=(0, (1, 2)),
                           label=f"reference uncertainty {rel['reference_uncertainty_total']:.2f} ({rel['verdict'].split(' ')[0]})")
            if k == "eF":
                ax.axhline(M["mean_force_reliability"][cv]["eF_reference_noise_floor"], color="0.3", lw=0.9, ls=(0, (1, 2)),
                           label=f"reference noise floor {M['mean_force_reliability'][cv]['eF_reference_noise_floor']:.2f}")
            if k == "exi":
                ax.axhline(M["floors"]["exi_multinomial_uniform"], color="0.3", lw=0.9, ls=(0, (1, 2)),
                           label=f"finite-N floor {M['floors']['exi_multinomial_uniform']:.3f}")
            ax.set_xlabel("t [ps]")
            ax.set_ylabel(YL[k])
            ax.grid(alpha=0.25, which="both", lw=0.4)
            ax.legend(fontsize=6.5, loc="upper right")
            ax.set_title(f"{'phi' if cv == 'phi' else 'psi'} alone: " + {"eF": "free energy", "eFp": "mean force", "exi": "CV marginal"}[k], fontsize=9.5)
    fig.suptitle("Alanine dipeptide, incomplete 1-D CV: median and IQR over 16 paired seeds (N = 2048); dotted: first FR "
                 "opportunity 0.5 ps; dash-dot: end of the 5 ps ABF ramp; errors vs the FULL 1-D marginal of the 2-D MBAR reference",
                 fontsize=8.5)
    fig.tight_layout()
    save(fig, "convergence_2x3_rate015" if rate015 else "convergence_2x3", png)


def hidden(png):
    M = json.load(open(os.path.join(CA.PKG, "meta", "export.json")))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    h, a = read(os.path.join(S, "psi_hidden_pmf.csv"))
    ax = axes[0]
    ax.plot(a[:, 0], a[:, 1], "k-", lw=1.8, label="reference: full marginal F(psi)")
    ax.plot(a[:, 0], a[:, 2], "k--", lw=1.2, label="reference: phi < 0 conditional")
    for c0, col, lab in ((3, BLUE, "ABF"), (6, ORANGE, "FR 0.02"), (9, RED, "FR 0.15")):
        ax.fill_between(a[:, 0], a[:, c0 + 1], a[:, c0 + 2], color=col, alpha=0.2, lw=0)
        ax.plot(a[:, 0], a[:, c0], color=col, lw=1.1, label=f"{lab}: learned F(psi) at 100 ps")
    ax.set_xlabel("psi [deg]")
    ax.set_ylabel("F [kJ/mol]")
    ax.set_ylim(-2, 40)
    ax.set_xlim(-180, 180)
    ax.legend(fontsize=7)
    ax.set_title("(a) learned F(psi): median, IQR (aligned per seed)", fontsize=9.5)
    ax = axes[1]
    _, m = read(os.path.join(S, "psi_hidden_mass_phipos_instantaneous.csv"))
    for c0, col, lab in ((1, BLUE, "ABF"), (4, ORANGE, "FR 0.02"), (7, RED, "FR 0.15")):
        ax.fill_between(m[:, 0], m[:, c0 + 1], m[:, c0 + 2], color=col, alpha=0.2, lw=0)
        ax.plot(m[:, 0], m[:, c0], color=col, lw=1.3, label=lab)
    tgt, eq = M["ref_mass_phi_pos_uniform_psi"], M["ref_mass_phi_pos_equilibrium"]
    ax.axhline(tgt, color="k", lw=1.2, label=f"target under a uniform-psi bias {tgt:.3f}")
    ax.axhline(eq, color="0.5", lw=0.9, ls=":", label=f"unbiased Gibbs value {eq:.3f} (not the target)")
    ax.set_xlabel("t [ps]")
    ax.set_ylabel("fraction of walkers with phi > 0")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 0.36)
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title("(b) hidden phi: instantaneous mass on the C7ax side", fontsize=9.5)
    ax = axes[2]
    _, w = read(os.path.join(S, "psi_hidden_dcond_windows.csv"))
    for c0, col, lab, off in ((3, BLUE, "ABF", -0.8), (6, ORANGE, "FR 0.02", 0.0), (9, RED, "FR 0.15", 0.8)):
        x = w[:, 2] + off
        ax.errorbar(x, w[:, c0], yerr=[w[:, c0] - w[:, c0 + 1], w[:, c0 + 2] - w[:, c0]], fmt="o-", color=col, ms=4, capsize=3, lw=1.2, label=lab)
    ax.plot(w[:, 2], w[:, 12], ":", color="0.4", lw=1.0, label="multinomial sampling floor (ABF)")
    for i_, (lo, hi) in enumerate(w[:, :2]):
        ax.axvspan(lo, hi, color="0.90" if i_ % 2 == 0 else "0.97", zorder=0)
    ax.set_xlabel("time window [ps]: [1,5], [5,20], [20,50], [50,100]")
    ax.set_ylabel("D_cond = mean TV of p(phi | psi) vs reference")
    ax.set_ylim(0, None)
    ax.set_xlim(0, 100)
    ax.legend(fontsize=7)
    ax.set_title("(c) hidden conditional discrepancy (window-pooled, not instantaneous)", fontsize=9.5)
    fig.suptitle("psi-only ABF: the hidden phi equilibrates on a > 100 ps timescale and uniform FR slows it (16 paired seeds)", fontsize=9)
    fig.tight_layout()
    save(fig, "psi_hidden_coordinate", png)


def mfdiag(png):
    M = json.load(open(os.path.join(CA.PKG, "meta", "export.json")))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4))
    for ax, cv in zip(axes, ("phi", "psi")):
        rel = M["mean_force_reliability"][cv]
        for name, ls, lab in (("eFp", "-", "e_F' vs central-difference reference"), ("eFp_debiased", "--", "noise-debiased (indicative)"),
                              ("eFp_dispersion", ":", "seed dispersion (reference-free precision)")):
            path = os.path.join(D, f"{cv}_{name}.csv") if name == "eFp" else os.path.join(S, f"{cv}_{name}.csv")
            _, a = read(path)
            pos = a[:, 0] > 0          # t = 0: every seed holds the same empty estimator (dispersion exactly 0)
            ax.plot(a[pos, 0], a[pos, 1], color=BLUE, ls=ls, lw=1.3, label=f"ABF: {lab}")
            ax.plot(a[pos, 0], a[pos, 4], color=ORANGE, ls=ls, lw=1.1, label=f"FR 0.02: {lab}")
        ax.axhspan(0, rel["reference_uncertainty_total"], color="0.85", zorder=0, label=f"reference uncertainty {rel['reference_uncertainty_total']:.2f}")
        ax.set_yscale("log")
        ax.set_xlim(0, 100)
        ax.set_ylim(0.03, None)
        ax.set_xlabel("t [ps]")
        ax.set_ylabel("kJ/mol/rad")
        ax.set_title(f"{cv} alone: mean-force reference {rel['verdict'].split(' ')[0]} (ratio {rel['ratio']:.2f})", fontsize=9.5)
        ax.grid(alpha=0.25, which="both", lw=0.4)
        ax.legend(fontsize=6.3)
    fig.tight_layout()
    save(fig, "mean_force_diagnostics", png)


def save(fig, name, png):
    os.makedirs(P, exist_ok=True)
    fig.savefig(os.path.join(P, f"{name}.pdf"), metadata=MD)
    if png:
        os.makedirs(png, exist_ok=True)
        fig.savefig(os.path.join(png, f"{name}.png"), dpi=105)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--png", default=None)
    a = ap.parse_args()
    conv(False, a.png)
    conv(True, a.png)
    hidden(a.png)
    mfdiag(a.png)
    print("previews written to", P)


if __name__ == "__main__":
    main()
