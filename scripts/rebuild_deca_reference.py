"""Rebuild the deca-alanine reference PMF from saved samples, with edge clamping fixed.

    python scripts/rebuild_deca_reference.py --dir results/deca/reference

Why this exists
---------------
``alkanes.interval.bin_counts`` **clamps out-of-range samples into the edge bins** -- harmless in
the alkane study, where soft walls make it rare. Amendment 1 deliberately placed the umbrella
centres at ``[1.15, 3.70]``, *bracketing* the evaluation domain ``[1.20, 3.60]``, to fix a
coverage asymmetry at the top edge. Entire windows therefore sample outside the domain **by
design**, and clamping piled that mass into bin 0 and bin n-1.

The result was a spurious ~2.7 kT well at ``grid[0]`` -- 2.65 kT against neighbours at ~5.3 kT --
which the Amendment 3 basin finder read as a genuine second minimum, splitting off a 0.056 nm
"state" at the extreme low edge. That state sits almost entirely *below the screen's soft wall*
at 1.25 nm, so it can never be populated, and Gate C duly reported a persistent deficit. **The
coverage fix manufactured a fake ESTABLISHMENT-LIMITED verdict.**

No dynamics need re-running: ``xi_all`` and the MBAR weights are saved, so only the histogram
step changes. The MBAR weights themselves are unaffected and correctly use every sample --
it is only the *density on the evaluation domain* that must exclude samples from outside it.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from deca.labels import conditional_tv                                     # noqa: E402
from deca.umbrella import UmbrellaConfig, pmf_from_weights                  # noqa: E402

EFFECT_SIZE_PCT = 10.0
GATE_A_THRESHOLD = 0.30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/deca/reference")
    args = ap.parse_args()

    path = sorted(glob.glob(os.path.join(args.dir, "raw", "deca_umbrella__*.npz")))[-1]
    z = np.load(path, allow_pickle=True)
    cfg = UmbrellaConfig()
    per_build = cfg.n_windows * cfg.n_rep
    n_builds = int(z["F_builds"].shape[0])
    keep = np.asarray(z["keep"], bool)
    xi_all, w_all, y_all = z["xi_all"].astype(np.float64), z["weights"], z["y_all"]
    n_sample = xi_all.size // int(keep.sum())

    # per-build sample slices, in the same build-major order the runner concatenated them
    offs, cur = [], 0
    for b in range(n_builds):
        n = n_sample * int(keep[b * per_build:(b + 1) * per_build].sum())
        offs.append((cur, cur + n))
        cur += n
    assert cur == xi_all.size, (cur, xi_all.size)

    Fs, drops, counts0 = [], [], None
    for b, (a, e) in enumerate(offs):
        grid, dz, p, F, counts, nd = pmf_from_weights(xi_all[a:e], w_all[a:e], cfg)
        Fs.append(F)
        drops.append(nd)
        if counts0 is None:
            counts0 = counts

    mask = (grid >= cfg.R_lo) & (grid <= cfg.R_hi)
    A = np.stack([F - F[mask].mean() for F in Fs])
    M = np.full((n_builds, n_builds), np.nan)
    for a in range(n_builds):
        for b in range(n_builds):
            if a != b:
                M[a, b] = float(np.sqrt(((A[a] - A[b])[mask] ** 2).sum() * dz))
    Fm = A.mean(0)
    Fm = Fm - Fm[mask].mean()
    span = float(Fm[mask].max() - Fm[mask].min())
    tol = EFFECT_SIZE_PCT / 100.0 * span
    ratio = float(np.nanmax(M) / tol)

    kT = 0.008314462618 * 300.0
    old_F = z["F_consensus"]
    print(f"artifact: {path}")
    print(f"samples dropped as out-of-domain, per build: {drops} "
          f"({100*sum(drops)/xi_all.size:.2f} % of all samples)")
    print("\n--- the edge artifact, before and after ---")
    for i in list(range(4)) + list(range(len(grid) - 4, len(grid))):
        print(f"  grid[{i:3d}] = {grid[i]:.4f}   before {(old_F[i]-old_F.min())/kT:8.3f} kT"
              f"   after {(Fm[i]-Fm.min())/kT:8.3f} kT")

    print("\n--- corrected §4.5 ---")
    print(f"  max pairwise L2 : {np.nanmax(M):.4f} kJ/mol   (was {float(z['pairwise_l2'][~np.eye(n_builds,dtype=bool)].max()):.4f})")
    print(f"  span            : {span:.2f} kJ/mol ({span/kT:.1f} kT)")
    print(f"  ratio           : {ratio:.4f}")
    print(f"  ACCEPTED        : {ratio < 1.0}")
    print(f"  F argmin        : {grid[mask][Fm[mask].argmin()]:.4f} nm")

    edges = z["gate_a_edges"]
    tv, occ, p_cond = conditional_tv(xi_all, y_all.astype(int), w_all, edges,
                                     min_count=1e-3 * w_all.sum())
    with np.errstate(invalid="ignore"):
        tv_max = float(np.nanmax(tv)) if np.isfinite(tv).any() else float("nan")
    print(f"\n--- corrected Gate A ---")
    print(f"  max pairwise TV : {tv_max:.4f}  (was {float(z['gate_a_tv'][~np.isnan(z['gate_a_tv'])].max()):.4f})")
    print(f"  GATE A          : {'PASS' if tv_max >= GATE_A_THRESHOLD else 'FAIL'}")

    shutil.copy(path, path.replace(".npz", ".PRE_EDGEFIX.npz"))
    out = {k: z[k] for k in z.files}
    out.update(F_consensus=Fm, F_builds=A, pairwise_l2=M,
               gate_a_tv=tv, gate_a_occupancy=occ, gate_a_p_cond=p_cond)
    np.savez_compressed(path, **out)

    sp = os.path.join(args.dir, "reference_summary.json")
    with open(sp) as fh:
        s = json.load(fh)
    s.update(pairwise_l2_max=float(np.nanmax(M)), F_span_kJ=span,
             resolvable_effect_kJ=tol, ratio=ratio, reference_accepted=bool(ratio < 1.0),
             F_argmin_nm=float(grid[mask][Fm[mask].argmin()]),
             gate_a_max_pairwise_tv=tv_max,
             gate_a_pass=bool(tv_max >= GATE_A_THRESHOLD),
             edge_clamping_fixed=True,
             n_out_of_domain_samples_dropped=int(sum(drops)))
    with open(sp, "w") as fh:
        json.dump(s, fh, indent=2)
    print(f"\nrewrote {path} (previous kept as *.PRE_EDGEFIX.npz) and {sp}")


if __name__ == "__main__":
    main()
