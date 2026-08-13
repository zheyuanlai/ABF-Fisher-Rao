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
    a.legend(frameon=False, fontsize=9)

    # --- per-family mean force: the Gate 0 evidence ------------------------------------------
    a = ax[0, 1]
    for fi in range(d["f_fam"].shape[0]):
        a.plot(r, d["f_fam"][fi], lw=1.4, label=FAM.get(fi, str(fi)))
    a.fill_between(r, d["f_cons"] - d["f_sem"], d["f_cons"] + d["f_sem"], alpha=0.25,
                   color="0.4", label="consensus $\\pm$ s.e.m.")
    a.set_xlabel(r"$r_{\rm NaCl}$ (nm)"); a.set_ylabel(r"$\langle f_{\rm loc}\rangle$ (kJ/mol/nm)")
    g0 = rep["gate0"]
    a.set_title(f"Gate 0: family spread {g0['global_spread_ratio']:.3f} global / "
                f"{g0['barrier_region_ratio']:.3f} barrier")
    a.legend(frameon=False, fontsize=8)

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
