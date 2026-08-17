"""The two controls that decide whether the convergence claim survives contact with a referee.

    python scripts/audit_convergence_controls.py

Writes results/convergence_atlas/controls.json.

The atlas measures mFR against ABF while BOTH are adapting. Two questions it cannot answer:

  (1) Does the advantage survive freezing the bias?  An online endpoint is evaluated while the
      mFR population is deliberately non-Boltzmann, so part of the measured gain could be a
      property of the evaluation rather than of the learned free energy. The frozen-bias runs
      answer this -- but the WCA ones were scored against the SUPERSEDED reference. They store
      `F_recon`, so they can be rescored without re-running anything, which is what this does.

  (2) Does the TARGET matter, or would any reallocation do?  The WCA five-arm test says
      count-balancing ties mFR. The entropic bottleneck has an independent version of the same
      question in `fr_uniform` (reallocate toward a uniform occupancy -- no Fisher-Rao target
      at all) and `fr_oracle` (the exact target). Nobody has read them as a mechanism test.

Both are re-readings of stored artifacts. No dynamics are re-run.
"""
from __future__ import annotations

import glob
import json
import math
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "convergence_atlas", "controls.json")
CELL_REF = {"b1_h2": "cache/phase_hp_v3/wca_ti_b1_h2_w2_n10_a1.5_g160.npz"}


def rms_aligned(profile, ref, grid, mask):
    p = profile - np.mean((profile - ref)[mask])
    d = (p - ref)[mask]
    g = grid[mask]
    return math.sqrt(np.trapezoid(d ** 2, g) / (g[-1] - g[0]))


def paired(x, y):
    r = 100.0 * (np.asarray(x) - np.asarray(y)) / np.asarray(y)
    return dict(median_pct=float(np.median(r)), wins=int((r < 0).sum()), n=int(len(r)))


def frozen_bias():
    """WCA frozen-bias reconstruction, rescored against the corrected reference.

    The raw directory holds TWO cells (b1_h2 starved, b4_h1 easy) with overlapping seed
    numbers. Keying by (arm, seed) alone silently merges them -- it did, on the first pass,
    and produced a baseline that belonged to neither cell.
    """
    refs = {c: np.asarray(np.load(os.path.join(ROOT, p))["free_energy"], float)
            for c, p in CELL_REF.items()}
    tab = {}
    for f in glob.glob(os.path.join(ROOT, "results/wca_frozen_bias/raw/*.npz")):
        m = re.search(r"from-(.+?)__(b\d+_h\d+)_w2_n10_a1\.5__seed(\d+)", os.path.basename(f))
        if not m:
            continue
        arm, cell, s = m.group(1), m.group(2), int(m.group(3))
        z = np.load(f, allow_pickle=True)
        grid = np.asarray(z["grid"], float)
        mask = (grid >= -0.1) & (grid <= 1.1)
        rec = {"stored_superseded_ref": float(z["frozen_recon_l2_f"])}
        if cell in refs:
            rec["corrected_ref"] = rms_aligned(np.asarray(z["F_recon"], float),
                                               refs[cell], grid, mask)
        tab[(cell, arm, s)] = rec

    out = {}
    for cell in sorted({c for c, _, _ in tab}):
        for which in ("stored_superseded_ref", "corrected_ref"):
            for arm in ("fr_estimated", "fr_estimated_adaptive"):
                seeds = sorted(s for (c, a, s) in tab if c == cell and a == arm
                               and which in tab[(c, a, s)]
                               and (c, "abf", s) in tab and which in tab[(c, "abf", s)])
                if not seeds:
                    continue
                x = [tab[(cell, arm, s)][which] for s in seeds]
                y = [tab[(cell, "abf", s)][which] for s in seeds]
                rec = paired(x, y)
                rec.update(abf_median=float(np.median(y)), arm_median=float(np.median(x)))
                out[f"{cell}::{arm}::{which}"] = rec
    return out


def eb_target_ablation():
    """Entropic bottleneck: does the FR target matter, and is the gain dose-dependent?"""
    d = np.load(os.path.join(ROOT, "results/entropic_bottleneck/summaries/arrays.npz"))

    def integrated(stage, method, beta="beta8", omega="oin25", gamma="gamma15"):
        k = f"{stage}|{method}|{beta}|{omega}|{gamma}"
        return np.trapezoid(d[f"{k}::l2_f_t"], d[f"{k}::t"], axis=1)

    out = {"target_ablation_beta8": {}, "gamma_dose_response_beta8": {}}
    base = integrated("stage0_reproduce", "abf")
    for arm in ("fr_estimated", "fr_oracle", "fr_uniform"):
        v = integrated("stage0_reproduce", arm)
        n = min(len(v), len(base))
        out["target_ablation_beta8"][arm] = paired(v[:n], base[:n])
    for g in ("gamma1", "gamma3", "gamma5", "gamma10", "gamma15", "gamma25", "gamma50"):
        v = integrated("stage4_gamma", "fr_estimated", gamma=g)
        b = integrated("stage4_gamma", "abf", gamma=g)
        n = min(len(v), len(b))
        out["gamma_dose_response_beta8"][g] = paired(v[:n], b[:n])
    return out


def main():
    res = dict(
        frozen_bias_wca=frozen_bias(),
        entropic_bottleneck=eb_target_ablation(),
        gateway_frozen_bias_note=(
            "results/gateway_anchor/confirmatory_v2/raw.npz stores frozen_l2_f_kT: mFR vs ABF "
            "is -10.19 % on 24/32 seeds, while the ONLINE final-time l2_f is +11.28 % on 0/32. "
            "The two disagree in sign and both are 'the accuracy at the end of the run'."),
    )
    json.dump(res, open(OUT, "w"), indent=2)

    fb = res["frozen_bias_wca"]
    print("WCA frozen-bias reconstruction (mFR vs ABF, paired):")
    for k, v in fb.items():
        cell, arm, which = k.split("::")
        print(f"  {cell:6s} {arm:22s} {which:22s} {v['median_pct']:+7.2f} %  "
              f"{v['wins']}/{v['n']}   abf {v['abf_median']:.4f} -> arm {v['arm_median']:.4f}")
    eb = res["entropic_bottleneck"]
    print("\nEntropic bottleneck beta=8, target ablation (mFR vs ABF):")
    for a, v in eb["target_ablation_beta8"].items():
        print(f"  {a:14s} {v['median_pct']:+7.2f} %  {v['wins']}/{v['n']}")
    print("\nEntropic bottleneck beta=8, FR-rate dose response:")
    for g, v in eb["gamma_dose_response_beta8"].items():
        print(f"  {g:8s} {v['median_pct']:+7.2f} %  {v['wins']}/{v['n']}")
    print(f"\nwrote {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
