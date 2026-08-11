"""Post-hoc audit of the deca-alanine reference: drift, acceptance, and Gate A.

    python scripts/analyze_deca_reference.py --dir results/deca/reference

Why a separate drift check exists
---------------------------------
§4.5 acceptance is a statement about the spread **between** independent builds. That measures
statistical reproducibility and it is blind to a systematic error that moves all three builds
together -- an equilibration uniformly too short, say. The between-build ratio can be superb
while the consensus is still sliding.

Two independent things are therefore checked here:

1. **Time drift.** ``F`` from the first half of production against ``F`` from the second half.
   If the reference is converged these agree; if the hidden conformational degrees of freedom
   inside each window are still relaxing, they do not. The sample layout preserves time order
   inside each window block, so this is recoverable from the saved artifact.

2. **Initial-condition independence.** The builds are seeded from *deliberately different*
   conformational pools (helical, extended, PPII, bridge, mixed, left-handed). Tight agreement
   between them is therefore not merely statistical -- three runs that started from different
   regions of conformation space and landed on the same PMF is evidence that the slow degrees
   of freedom equilibrated. This is reported explicitly because it is the stronger of the two
   arguments and would otherwise go unstated.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from deca.labels import N_GATE_A_STATES, conditional_tv                    # noqa: E402
from deca.umbrella import UmbrellaConfig, mbar_weights, pmf_from_weights, window_centers  # noqa: E402

GATE_A_THRESHOLD = 0.30
EFFECT_SIZE_PCT = 10.0


def _blocks(xi_all, keep, n_w, n_rep, n_sample):
    """Recover ``(n_sample, n_kept_w)`` per window from the flattened window-major array."""
    keep = np.asarray(keep, bool).reshape(n_w, n_rep)
    out, off = [], 0
    for w in range(n_w):
        nk = int(keep[w].sum())
        n = n_sample * nk
        out.append(xi_all[off:off + n].reshape(n_sample, nk))
        off += n
    if off != xi_all.size:
        raise AssertionError(f"layout mismatch: consumed {off} of {xi_all.size}")
    return out, keep


def _F_from_blocks(blocks, keep, cfg, centers):
    """Rebuild the (n_sample, n_w*n_rep) array a slice implies, then MBAR + PMF."""
    n_w, n_rep = keep.shape
    n_sample = blocks[0].shape[0]
    xi = np.zeros((n_sample, n_w * n_rep), dtype=np.float64)
    for w in range(n_w):
        idx = np.flatnonzero(keep[w])
        xi[:, w * n_rep + idx] = blocks[w]
        dead = np.flatnonzero(~keep[w])
        if dead.size:                      # fill excluded slots with in-window values; the
            xi[:, w * n_rep + dead] = blocks[w][:, :1]   # keep mask drops them again anyway
    xa, w_, info, _ = mbar_weights(xi, centers, n_rep, cfg.k_umbrella, cfg.beta,
                                   keep=keep.reshape(-1))
    grid, dz, p, F, counts = pmf_from_weights(xa, w_, cfg)
    return grid, dz, F, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/deca/reference")
    args = ap.parse_args()

    with open(os.path.join(args.dir, "reference_summary.json")) as fh:
        s = json.load(fh)
    path = sorted(glob.glob(os.path.join(args.dir, "raw", "deca_umbrella__*.npz")))[-1]
    z = np.load(path, allow_pickle=True)
    grid, F_cons, F_builds = z["grid"], z["F_consensus"], z["F_builds"]
    keep_flat, xi_all, w_all, y_all = z["keep"], z["xi_all"], z["weights"], z["y_all"]

    cfg = UmbrellaConfig()
    n_w, n_rep = cfg.n_windows, cfg.n_rep
    per_build = n_w * n_rep
    n_builds = F_builds.shape[0]
    centers = window_centers(cfg)
    mask = (grid >= cfg.R_lo) & (grid <= cfg.R_hi)
    dz = float(grid[1] - grid[0])

    print(f"artifact: {path}")
    print(f"builds {n_builds}   ns/replica {s['ns_per_replica']}   "
          f"aggregate {s['aggregate_ns']:.0f} ns   runtime {s['runtime_hours']:.2f} h")

    # --------------------------------------------------------------- §4.5 acceptance (restated)
    span = float(F_cons[mask].max() - F_cons[mask].min())
    tol = EFFECT_SIZE_PCT / 100.0 * span
    print("\n--- §4.5 acceptance ---")
    print(f"  max pairwise L2 between builds : {s['pairwise_l2_max']:.4f} kJ/mol")
    print(f"  consensus span                 : {span:.2f} kJ/mol "
          f"({span / (0.008314462618*300.0):.1f} kT)")
    print(f"  a {EFFECT_SIZE_PCT:.0f}% effect is            : {tol:.2f} kJ/mol")
    print(f"  ratio (want < 1.0)             : {s['ratio']:.4f}")
    print(f"  ACCEPTED                       : {s['reference_accepted']}")

    # --------------------------------------------------------------- drift in time
    n_sample = xi_all.size // int(np.asarray(keep_flat, bool).sum())
    blocks, keep = _blocks(xi_all.astype(np.float64), keep_flat, n_w, n_rep, n_sample)
    half = n_sample // 2
    g1, _, F1, c1 = _F_from_blocks([b[:half] for b in blocks], keep, cfg, centers)
    g2, _, F2, c2 = _F_from_blocks([b[half:] for b in blocks], keep, cfg, centers)
    F1 = F1 - F1[mask].mean()
    F2 = F2 - F2[mask].mean()
    drift = float(np.sqrt(((F1 - F2)[mask] ** 2).sum() * dz))
    print("\n--- time drift (first half of production vs second) ---")
    print(f"  L2(F_first, F_second) : {drift:.4f} kJ/mol")
    print(f"  as a fraction of a {EFFECT_SIZE_PCT:.0f}% effect : {drift / tol:.4f}")
    conv = drift < tol
    print(f"  CONVERGED IN TIME (want < 1.0) : {conv}")
    if not conv:
        print("  !! the consensus is still moving. Between-build agreement CANNOT see this.")

    # --------------------------------------------------------------- initial-condition spread
    print("\n--- initial-condition independence ---")
    print("  builds were seeded from different conformational pools (helix, extended, PPII,")
    print("  bridge, mixed, left-handed), so their agreement is evidence about slow modes,")
    print("  not merely about statistics.")
    for b in range(n_builds):
        d = F_builds[b] - F_cons
        print(f"    build {b}: L2 from consensus = {np.sqrt((d[mask]**2).sum()*dz):.4f} kJ/mol")

    # --------------------------------------------------------------- Gate A (restated)
    edges = z["gate_a_edges"]
    tv, occ, p_cond = conditional_tv(xi_all, y_all.astype(int), w_all, edges,
                                     min_count=1e-3 * w_all.sum())
    with np.errstate(invalid="ignore"):
        tv_max = float(np.nanmax(tv)) if np.isfinite(tv).any() else float("nan")
    print("\n--- Gate A: CV visibility (§2.2) ---")
    print(f"  labels occupied : {int((occ > 0).sum())}/{N_GATE_A_STATES}")
    print(f"  weight share    : {np.round(occ / occ.sum(), 4)}")
    print(f"  max pairwise TV : {tv_max:.4f}  (threshold {GATE_A_THRESHOLD})")
    print(f"  GATE A          : {'PASS' if tv_max >= GATE_A_THRESHOLD else 'FAIL -- STOP'}")

    verdict = dict(reference_accepted=bool(s["reference_accepted"]), ratio=float(s["ratio"]),
                   span_kJ=span, drift_l2_kJ=drift, drift_over_effect=float(drift / tol),
                   converged_in_time=bool(conv), gate_a_max_tv=tv_max,
                   gate_a_pass=bool(tv_max >= GATE_A_THRESHOLD),
                   overall_usable=bool(s["reference_accepted"] and conv))
    with open(os.path.join(args.dir, "reference_audit.json"), "w") as fh:
        json.dump(verdict, fh, indent=2)
    print(f"\nwrote {os.path.join(args.dir, 'reference_audit.json')}")
    print(f"REFERENCE USABLE: {verdict['overall_usable']}")


if __name__ == "__main__":
    main()
