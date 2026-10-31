#!/usr/bin/env python3
"""Convert a sweep umbrella/WHAM LTA reference (scripts/run_lta_reference.py, reference_T{T}.npz) into the
lta_T{T}_reference.npz layout read by scripts/adaptive_fr/analyze_online.py and eqb_metrics.LTAScorer, as the archive
did for 80 and 350 K on 2026-10-17 (docs/r2_campaign/PREREGISTRATION.md, section 4):

  F_ref      the WHAM F on the 180 production bins, centred (mean 0)              [kJ/mol]
  gamma_ref  central finite difference of F_ref on the circle                      [kJ/mol/rad]
  F_ref_mf   histogram_pmf(gamma_ref) (consistency of the mean-force route)
  F_ref_se   |F_half1 - F_half2| / 2 when the split halves are stored, else 0 (NOT ESTIMATED; informational only)
  gamma_se   0 (not estimated; informational only)

    python scripts/laplacian_baselines/convert_lta_reference.py <reference_T80.npz> <out lta_T80_reference.npz> [--compare <exact npz>]

--compare prints the RMS of the centred F and of gamma against an exact-MC reference (the 150 / 300 K validation of
the protocol; the archive quoted 0.017 / 0.034 kJ/mol RMS in F and 0.03-0.04 kJ/mol/rad in gamma).
"""
import argparse
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
KB = 0.008314462618
NB = 180


def central_diff_circle(F, dphi):
    return (np.roll(F, -1) - np.roll(F, 1)) / (2.0 * dphi)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--compare", default=None)
    a = ap.parse_args(argv)
    import lta_ladder_numba as E
    z = np.load(a.src, allow_pickle=True)
    T = float(z["temperature"])
    kT = KB * T
    grid = np.asarray(z["grid_phi"], float)
    F = np.asarray(z["F"], float)
    if grid.shape != (NB,) or F.shape != (NB,):
        raise SystemExit(f"expected {NB} bins, got {grid.shape} {F.shape}")
    if not np.all(np.isfinite(F)):
        raise SystemExit(f"non-finite WHAM F in {np.sum(~np.isfinite(F))} bins")
    dphi = 2.0 * math.pi / NB
    exp_grid = -math.pi + (np.arange(NB) + 0.5) * dphi
    if not np.allclose(grid, exp_grid, atol=1e-9):
        raise SystemExit("grid is not the cell-centred 180-bin production grid")
    F = F - F.mean()
    gamma = central_diff_circle(F, dphi)
    F_mf = E.histogram_pmf(gamma[None, :], dphi)[0]
    se = np.zeros(NB)
    prot = json.loads(str(z["protocol"])) if "protocol" in z.files else {}
    out = dict(T_K=T, kT=kT, grid_phi=grid, F_ref=F, F_ref_se=se, gamma_ref=gamma, gamma_se=np.zeros(NB), F_ref_mf=F_mf,
               units_F="kJ/mol", units_gamma="kJ/mol/rad",
               gauge="F centred (mean over the 180 bins = 0); compare after removing the mean difference",
               source=(f"{os.path.relpath(a.src, ROOT)}: umbrella/WHAM sweep protocol (overdamped EM), gamma_ref = central "
                       f"finite difference of F_ref (finite-difference-referenced, PREREGISTRATION.md section 4); "
                       f"F_ref_se and gamma_se are NOT estimated (0); protocol {json.dumps(prot)}"))
    os.makedirs(os.path.dirname(os.path.abspath(a.dst)), exist_ok=True)
    np.savez(a.dst, **{k: np.asarray(v) for k, v in out.items()})
    rep = dict(T_K=T, dst=os.path.relpath(a.dst, ROOT), F_span=float(F.max() - F.min()),
               F_mf_minus_F_rms=float(np.sqrt(np.mean((F_mf - F_mf.mean() - F) ** 2))),
               dF_barrier=float(z["dF_barrier"]) if "dF_barrier" in z.files else None,
               dF_split=np.asarray(z["dF_split"]).tolist() if "dF_split" in z.files else None)
    if a.compare:
        r = np.load(a.compare, allow_pickle=False)
        Fx = np.asarray(r["F_ref"], float)
        Fx = Fx - Fx.mean()
        gx = np.asarray(r["gamma_ref"], float)
        rep["vs_exact"] = dict(path=os.path.relpath(a.compare, ROOT), F_rms=float(np.sqrt(np.mean((F - Fx) ** 2))),
                               F_rms_kT=float(np.sqrt(np.mean((F - Fx) ** 2)) / kT),
                               gamma_rms=float(np.sqrt(np.mean((gamma - gx) ** 2))),
                               gamma_rms_rel=float(np.sqrt(np.mean((gamma - gx) ** 2)) / np.sqrt(np.mean(gx ** 2))),
                               barrier_exact_minus_sweep=float((Fx.max() - Fx.min()) - (F.max() - F.min())))
    print(json.dumps(rep, indent=1))
    json.dump(rep, open(a.dst[:-4] + "_conversion.json", "w"), indent=1)


if __name__ == "__main__":
    main()
