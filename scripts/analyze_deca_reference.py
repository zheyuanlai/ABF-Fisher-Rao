"""Reference acceptance for deca-alanine (v2 preregistration §4.5), and the Gate A statistic.

Answers two questions, in order, and refuses to answer the second if the first fails:

  1. **Is the reference good enough to score a 10 % effect?**  Three independent builds,
     pairwise ``L2`` discrepancy, bootstrap uncertainty, and a convergence trace against
     reference compute.  §4.5 exists because the v1 audit found a cached WCA TI reference
     sitting ~10x the arm effect away from a high-precision consensus.

  2. **Gate A — is the hidden conformational structure visible in the CV?**  Maximum pairwise
     total variation between ``p(xi | Y = a)`` across the frozen 9-state label.  Below 0.30 the
     collective variable cannot separate the states and deca-alanine STOPS.

    python scripts/analyze_deca_reference.py --dir results/deca/reference
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from deca.labels import N_GATE_A_STATES, conditional_tv                    # noqa: E402

GATE_A_THRESHOLD = 0.30          # §2.2, frozen
EFFECT_SIZE_PCT = 10.0           # the effect the reference must be able to resolve


def load_builds(d):
    out = []
    for p in sorted(glob.glob(os.path.join(d, "raw", "deca_umbrella_build*.npz"))):
        z = np.load(p, allow_pickle=True)
        out.append(dict(path=p, grid=z["grid"], F=z["F_ref"], p=z["p_ref"],
                        counts=z["bin_counts"], xi=z["xi_all"], w=z["weights"],
                        y=z["y_all"], edges=z["gate_a_edges"],
                        R_lo=float(z["cfg_R_lo"]), R_hi=float(z["cfg_R_hi"]),
                        beta=1.0 / (0.008314462618 * float(z["cfg_temperature"]))))
    return out


def _align(F, mask):
    """Free energies are defined up to a constant; compare only after removing it."""
    return F - F[mask].mean()


def pairwise_l2(builds, mask, dz):
    n = len(builds)
    M = np.full((n, n), np.nan)
    for a in range(n):
        for b in range(n):
            if a != b:
                d = _align(builds[a]["F"], mask) - _align(builds[b]["F"], mask)
                M[a, b] = float(np.sqrt((d[mask] ** 2).sum() * dz))
    return M


def bootstrap_consensus(builds, mask, dz, n_boot=2000, rng=None):
    """Bootstrap the consensus F over builds.  Returns (F_mean, F_sd, l2_sd)."""
    rng = rng or np.random.default_rng(0)
    A = np.stack([_align(b["F"], mask) for b in builds])
    n = A.shape[0]
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = A[idx].mean(1)
    boots = boots - boots[:, mask].mean(1, keepdims=True)
    Fm = A.mean(0)
    Fm = Fm - Fm[mask].mean()
    l2 = np.sqrt(((boots - Fm[None]) ** 2)[:, mask].sum(1) * dz)
    return Fm, boots.std(0), float(l2.std()), float(np.percentile(l2, 95))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/deca/reference")
    ap.add_argument("--min-builds", type=int, default=3)
    args = ap.parse_args()

    builds = load_builds(args.dir)
    print(f"loaded {len(builds)} build(s) from {args.dir}")
    if not builds:
        raise SystemExit("no builds found")

    grid = builds[0]["grid"]
    dz = float(grid[1] - grid[0])
    R_lo, R_hi = builds[0]["R_lo"], builds[0]["R_hi"]
    mask = (grid >= R_lo) & (grid <= R_hi)
    beta = builds[0]["beta"]

    # ---------------------------------------------------------------- per-build sanity
    print("\n--- per build ---")
    for i, b in enumerate(builds):
        F = _align(b["F"], mask)
        empty = int((b["counts"][mask] == 0).sum())
        print(f"  build {i}: F span {F[mask].max()-F[mask].min():8.2f} kJ/mol "
              f"({(F[mask].max()-F[mask].min())*beta:6.1f} kT)  "
              f"min at {grid[mask][F[mask].argmin()]:.3f} nm  "
              f"empty bins {empty}  min count {int(b['counts'][mask].min())}")

    verdict = {}
    if len(builds) < args.min_builds:
        print(f"\n!! only {len(builds)} build(s); §4.5 requires at least {args.min_builds}. "
              "Reference is NOT accepted and Gate A is not evaluated.")
        verdict["reference_accepted"] = False
        verdict["reason"] = f"only {len(builds)} builds"
    else:
        # ------------------------------------------------------------ §4.5 acceptance
        M = pairwise_l2(builds, mask, dz)
        Fm, Fsd, l2_sd, l2_p95 = bootstrap_consensus(builds, mask, dz)
        span = float(Fm[mask].max() - Fm[mask].min())
        tolerable = EFFECT_SIZE_PCT / 100.0 * span

        print("\n--- §4.5 reference acceptance ---")
        print(f"  pairwise L2 between builds (kJ/mol): "
              f"max {np.nanmax(M):.3f}  median {np.nanmedian(M):.3f}")
        print(f"  bootstrap consensus L2 sd {l2_sd:.3f}, 95th pct {l2_p95:.3f} kJ/mol")
        print(f"  consensus F span {span:.2f} kJ/mol; a {EFFECT_SIZE_PCT:.0f}% effect is "
              f"{tolerable:.2f} kJ/mol")
        ratio = float(np.nanmax(M) / tolerable)
        print(f"  worst pairwise discrepancy / resolvable effect = {ratio:.3f}")
        accepted = bool(ratio < 1.0)
        print(f"  ACCEPTED: {accepted}"
              + ("" if accepted else "   -- rebuild with more or longer windows"))
        verdict.update(reference_accepted=accepted,
                       pairwise_l2_max=float(np.nanmax(M)),
                       pairwise_l2_median=float(np.nanmedian(M)),
                       bootstrap_l2_sd=l2_sd, bootstrap_l2_p95=l2_p95,
                       F_span_kJ=span, resolvable_effect_kJ=float(tolerable),
                       discrepancy_over_effect=ratio)
        np.savez_compressed(os.path.join(args.dir, "consensus_reference.npz"),
                            grid=grid, F_ref=Fm, F_sd=Fsd, dz=dz, R_lo=R_lo, R_hi=R_hi,
                            n_builds=len(builds), pairwise_l2=M)

        # ------------------------------------------------------------ Gate A
        if accepted:
            xi = np.concatenate([b["xi"] for b in builds])
            w = np.concatenate([b["w"] / b["w"].sum() for b in builds])
            y = np.concatenate([b["y"] for b in builds]).astype(int)
            edges = builds[0]["edges"]
            tv, occ, p_cond = conditional_tv(xi, y, w, edges, min_count=1e-3 * w.sum())
            with np.errstate(invalid="ignore"):
                tv_max = float(np.nanmax(tv)) if np.isfinite(tv).any() else float("nan")
            live = int((occ >= 1e-3 * w.sum()).sum())
            print("\n--- Gate A: CV visibility (§2.2) ---")
            print(f"  labels with enough weight to compare: {live}/{N_GATE_A_STATES}")
            print(f"  weight share per label: {np.round(occ/occ.sum(), 4)}")
            print(f"  max pairwise TV( p(xi|Y=a), p(xi|Y=b) ) = {tv_max:.4f} "
                  f"(threshold {GATE_A_THRESHOLD})")
            passed = bool(np.isfinite(tv_max) and tv_max >= GATE_A_THRESHOLD)
            print(f"  GATE A: {'PASS -- continue to the ABF-only screen' if passed else 'FAIL -- STOP; the CV cannot separate these states'}")
            verdict.update(gate_a_max_tv=tv_max, gate_a_pass=passed,
                           gate_a_live_labels=live)
            np.savez_compressed(os.path.join(args.dir, "gate_a.npz"),
                                tv=tv, occupancy=occ, p_cond=p_cond, edges=edges)

    with open(os.path.join(args.dir, "reference_acceptance.json"), "w") as fh:
        json.dump(verdict, fh, indent=2)
    print(f"\nwrote {os.path.join(args.dir, 'reference_acceptance.json')}")


if __name__ == "__main__":
    main()
