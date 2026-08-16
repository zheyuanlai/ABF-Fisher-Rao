"""Figures for the NaCl reference stage: F/W against the published curve, family spread, RDFs.

Publication-shaped, but the point is diagnostic: the family-spread panel is the Gate 0 evidence
and the external-check panel is the only place the published 100 ns ABF curve appears.

Usage:
    python scripts/plot_nacl_reference.py --ref results/nacl/reference --out fig/nacl
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
import numpy as np                                               # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from nacl import system as nsys                                  # noqa: E402

FAM = {0: "CIP-derived", 1: "SSIP-derived", 2: "dissoc-derived", 3: "local-equil"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="results/nacl/reference")
    ap.add_argument("--rdf", default="results/nacl/stage0/reference_rdfs.npz")
    ap.add_argument("--out", default="fig/nacl")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    d = np.load(os.path.join(args.ref, "reference.npz"))
    rep = json.load(open(os.path.join(args.ref, "reference_report.json")))
    r, F, W = d["r_nm"], d["F_ref"], d["W_ref"]
    kT = nsys.kT_kJ()

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))

    # --- F and W, with the basins -----------------------------------------------------------
    a = ax[0, 0]
    a.plot(r, (F - F.min()) / kT, lw=2, label=r"$F_{\rm ref}$ (reaction-coordinate)")
    a.plot(r, (W - W.min()) / kT, lw=2, ls="--", label=r"$W_{\rm ref}$ (radial PMF)")
    for b in rep["basins"]:
        a.axvspan(b["r_lo_nm"], b["r_hi_nm"], alpha=0.08,
                  color="C2" if b["label"] == "CIP" else "C3")
        if b.get("r_min_nm"):
            a.axvline(b["r_min_nm"], color="0.4", lw=0.7, ls=":")
    w = rep["endpoint_window"]
    a.axvspan(w["r_lo_nm"], w["r_hi_nm"], alpha=0.05, color="C0")
    a.set_xlabel(r"$r_{\rm NaCl}$ (nm)"); a.set_ylabel(r"free energy ($k_BT$)")
    a.set_title(f"reference (accepted, ratio = {rep['acceptance']['ratio']:.3f})")
    a.legend(frameon=False, fontsize=9, loc="upper right")
    # The 0.20-0.24 nm wall rises ~150 kT and, left alone, compresses the entire physical
    # landscape -- a 5.3 kT barrier and a 2.5 kT well -- into the thickness of the axis line.
    # Clip to the frozen scoring window instead, and mark what the reader is meant to read off.
    phys = (r >= w["r_lo_nm"] - 0.01)
    top = float(np.max((F[phys] - F.min()) / kT))
    a.set_ylim(-0.6, top * 1.35)
    ph = rep["physical"]
    a.annotate("", xy=(0.35, ph["dW_barrier_kJ"] / kT), xytext=(0.35, 0.0),
               arrowprops=dict(arrowstyle="<->", lw=1.0, color="0.25"))
    a.text(0.365, ph["dW_barrier_kJ"] / kT * 0.55,
           f"barrier\n{ph['dW_barrier_kJ']/kT:.2f} $k_BT$", fontsize=8, color="0.25", va="center")
    a.text(0.26, -0.45, "CIP", fontsize=8, ha="center", color="C2")
    a.text(0.9, 0.35, "merged outer basin (min at the domain edge)",
           fontsize=7.5, ha="center", color="C3")

    # --- per-family mean force: the Gate 0 evidence ------------------------------------------
    # Plotted as RESIDUALS from the consensus, not as raw curves. On raw axes the wall drives
    # f_loc to -2.5e4 kJ/mol/nm and the four families collapse onto one visually identical
    # line -- which is the gate's conclusion but not evidence for it, since families that
    # disagreed by 10 % would also look identical there. The residual shows the disagreement
    # itself. The gate statistic is an AGGREGATE ratio over the window, so a pointwise
    # normalised spread is a different quantity and is deliberately not drawn on top of it.
    a = ax[0, 1]
    g0 = rep["gate0"]
    lo = w["r_lo_nm"]
    seg = r >= lo
    for fi in range(d["f_fam"].shape[0]):
        a.plot(r[seg], (d["f_fam"][fi] - d["f_cons"])[seg], lw=1.3, label=FAM.get(fi, str(fi)))
    a.fill_between(r[seg], -d["f_sem"][seg], d["f_sem"][seg], alpha=0.30, color="0.4",
                   label="consensus s.e.m.", zorder=0)
    a.axhline(0.0, color="0.2", lw=0.8)
    a.set_xlim(lo - 0.02, r[-1] + 0.02)
    a.set_xlabel(r"$r_{\rm NaCl}$ (nm)")
    a.set_ylabel(r"family $\langle f_{\rm loc}\rangle$ $-$ consensus (kJ/mol/nm)")
    a.set_title(f"Gate 0: family spread {g0['global_spread_ratio']:.4f} global / "
                f"{g0['barrier_region_ratio']:.4f} barrier")
    a.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper right")
    a.text(0.02, 0.03,
           "independently prepared solvent families held at the same $r$;\n"
           "deca-alanine FAILS the same statistic at 0.61",
           transform=a.transAxes, fontsize=7.5, color="0.25", va="bottom")

    # --- external check against the published 100 ns ABF PMF --------------------------------
    a = ax[1, 0]
    pub = np.loadtxt(nsys.SRC_TUTORIAL / "output/abf.pmf")
    pub_r, pub_F = pub[:, 0] * 0.1, pub[:, 1] * 4.184
    m = (pub_r >= r[0]) & (pub_r <= r[-1])
    ours = np.interp(pub_r[m], r, F)
    tail = pub_r[m] >= max(r[-1] - 0.2, pub_r[m][0])
    shift = float(np.mean(ours[tail] - pub_F[m][tail]))
    a.plot(pub_r[m], pub_F[m] / kT, lw=2, label="Talmazan 2025, 100 ns ABF")
    a.plot(pub_r[m], (ours - shift) / kT, lw=2, ls="--", label="this work, constrained TI")
    a.set_xlabel(r"$r_{\rm NaCl}$ (nm)"); a.set_ylabel(r"$F$ ($k_BT$, aligned at the tail)")
    a.set_title(f"external check (rms {rep['external_check']['rms_kJ']:.2f} kJ/mol) "
                "-- reported, not a gate")
    a.legend(frameon=False, fontsize=9)
    a.set_ylim(-2, 20)

    # --- the reference RDFs and the frozen R0 values -----------------------------------------
    a = ax[1, 1]
    if os.path.exists(args.rdf):
        g = np.load(args.rdf)
        frozen = json.load(open(nsys.STAGE0 / "descriptor_freeze.json"))
        for key, lab, col in (("g_NaO", r"$g_{\rm NaO}$", "C0"),
                              ("g_ClO", r"$g_{\rm ClO}$", "C1"),
                              ("g_ClH", r"$g_{\rm ClH}$", "C2")):
            a.plot(g["r_nm"], g[key], lw=1.5, label=lab, color=col)
        for key, col in (("R0_NaO_nm", "C0"), ("R0_ClO_nm", "C1"), ("R0_ClH_nm", "C2")):
            a.axvline(frozen[key], color=col, ls=":", lw=1.0)
        a.set_xlim(0, 0.8)
        a.set_xlabel(r"$r$ (nm)"); a.set_ylabel("g(r)")
        a.set_title("reference RDFs; dotted = frozen switch radii")
        a.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(args.out, f"nacl_reference.{ext}"), dpi=180)
    print(f"-> {args.out}/nacl_reference.png / .pdf")


if __name__ == "__main__":
    main()
