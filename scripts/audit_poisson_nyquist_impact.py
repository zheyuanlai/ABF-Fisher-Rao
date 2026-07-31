"""Decisive impact test: does removing the Nyquist row change reported pentane 2-D conclusions?

The fix (c6a6718) zeroes the Nyquist row/column of ``Bhat`` in ``poisson_projection``.  This is
the standard and necessary remedy -- verified: the minimal "self-conjugate modes only" variant
does NOT restore ``gB == grad B`` (3.7e-1 residual), because ``fftfreq`` assigns ``k = -n/2`` at
index ``n/2`` and the conjugate partner shares that index, so ``i k`` is not antisymmetric there
(measured Hermitian defect 81.3).  The Nyquist row has no representable derivative on the grid.

But the fix therefore also removes Nyquist content from ``B`` itself.  The question that decides
whether published pentane 2-D results stand is NOT the size of that content -- it is whether the
reported error and the ABF-vs-mFR ranking move.

Method: for each saved run take ``final_pmf`` (= B as produced by the legacy code), form
``B_nyq_removed`` by zeroing its Nyquist rows, and recompute the study's own error metric against
the saved reference ``ref_joint_F`` using the saved ``eq_weight``.  Then compare the ABF-vs-mFR
relative delta under both.

Usage:  CUDA_VISIBLE_DEVICES="" python scripts/audit_poisson_nyquist_impact.py
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
torch.set_default_dtype(torch.float64)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                   "results", "poisson_nyquist_audit")


def strip_nyquist(B):
    """Zero the Nyquist row/column of a real field's spectrum (what the fix does to Bhat)."""
    n1, n2 = B.shape[-2], B.shape[-1]
    H = torch.fft.fft2(B.to(torch.complex128))
    if n1 % 2 == 0:
        H[..., n1 // 2, :] = 0.0
    if n2 % 2 == 0:
        H[..., :, n2 // 2] = 0.0
    return torch.fft.ifft2(H).real.to(B.dtype)


def weighted_l2(B, ref, w):
    """The study's metric: additive-constant-aligned, equilibrium-weighted RMS error."""
    w = w / w.sum().clamp_min(1e-30)
    shift = ((B - ref) * w).sum(dim=(-2, -1), keepdim=True)
    d = B - ref - shift
    return torch.sqrt((d * d * w).sum(dim=(-2, -1)))


def main():
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob("results/alkanes_cv_extension/2d_methods/raw/production__joint2d*.npz")
                   + glob.glob("results/alkanes_cv_extension/2d_methods/raw/control__joint2d*.npz"))
    rows = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        if "final_pmf" not in d.files or "ref_joint_F" not in d.files:
            continue
        B = torch.as_tensor(d["final_pmf"])
        ref = torch.as_tensor(d["ref_joint_F"])
        w = torch.as_tensor(d["eq_weight"]) if "eq_weight" in d.files else torch.ones_like(ref)
        ref = ref - ref.mean()
        Bs = strip_nyquist(B)
        l2_leg = weighted_l2(B, ref[None], w[None])
        l2_fix = weighted_l2(Bs, ref[None], w[None])
        rng = float(d["F_range_thermal"]) if "F_range_thermal" in d.files else 1.0
        rows.append(dict(
            stage=str(d["stage"]), method=str(d["method"]), name=str(d["name"]),
            n_seeds=int(B.shape[0]),
            l2_legacy_pct=float(l2_leg.mean() / rng * 100),
            l2_fixed_pct=float(l2_fix.mean() / rng * 100),
            rel_change_pct=float((l2_fix.mean() - l2_leg.mean()) / l2_leg.mean() * 100),
            nyq_power_frac=float(((B - Bs) ** 2).sum() / (B - B.mean()).pow(2).sum()),
        ))
        print(f"  {rows[-1]['stage']:12s} {rows[-1]['name']:14s} "
              f"L2 legacy {rows[-1]['l2_legacy_pct']:7.4f}%  fixed {rows[-1]['l2_fixed_pct']:7.4f}%  "
              f"change {rows[-1]['rel_change_pct']:+7.4f}%  nyqPow {rows[-1]['nyq_power_frac']:.2e}")

    if not rows:
        print("no production runs found")
        return

    # ABF-vs-mFR ranking under both projections
    print("\n=== ABF vs mFR ranking ===")
    lines = []
    for stage in sorted({r["stage"] for r in rows}):
        sub = {r["name"]: r for r in rows if r["stage"] == stage}
        abf = next((v for k, v in sub.items() if v["method"] == "abf"), None)
        if abf is None:
            continue
        for name, r in sorted(sub.items()):
            if r["method"] == "abf":
                continue
            d_leg = (r["l2_legacy_pct"] - abf["l2_legacy_pct"]) / abf["l2_legacy_pct"] * 100
            d_fix = (r["l2_fixed_pct"] - abf["l2_fixed_pct"]) / abf["l2_fixed_pct"] * 100
            same = (d_leg > 0) == (d_fix > 0)
            lines.append((stage, name, d_leg, d_fix, same))
            print(f"  {stage:12s} {name:14s} vs abf: legacy {d_leg:+7.3f}%  fixed {d_fix:+7.3f}%  "
                  f"sign {'SAME' if same else 'FLIPPED'}  |shift| {abs(d_fix-d_leg):.3f} pp")

    worst_change = max(abs(r["rel_change_pct"]) for r in rows)
    all_same = all(s for *_, s in lines) if lines else True
    worst_shift = max((abs(b - a) for *_, a, b, _ in lines), default=0.0)

    keys = list(rows[0].keys())
    with open(os.path.join(OUT, "impact.csv"), "w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(str(r[k]) for k in keys) + "\n")

    verdict = ("NEGLIGIBLE — retain existing conclusions" if all_same and worst_change < 5.0
               else "MATERIAL — rerun the affected 2-D arms")
    print(f"\nworst per-arm L2 change {worst_change:.4f}%  |  ranking signs all preserved: {all_same}"
          f"  |  worst ranking shift {worst_shift:.3f} pp")
    print(f"VERDICT: {verdict}")
    with open(os.path.join(OUT, "impact_verdict.txt"), "w") as fh:
        fh.write(f"worst per-arm L2 change: {worst_change:.4f}%\n"
                 f"ranking signs preserved: {all_same}\n"
                 f"worst ranking shift: {worst_shift:.4f} pp\n"
                 f"VERDICT: {verdict}\n")


if __name__ == "__main__":
    main()
