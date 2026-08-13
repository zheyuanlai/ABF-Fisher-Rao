"""Turn the NaCl constrained-TI output into F_ref/W_ref, accept it, freeze the basins, and
evaluate Gate 0 and Gate A (SPEC_nacl_water.md §5, §6).

Consumes ``results/nacl/ti_torch/ti_final.npz`` and produces

    fbar(r)     conditional mean force, per build / per family / consensus
    F_ref(r)    = C + integral fbar ds                 (cumulative trapezoid)
    W_ref(r)    = F_ref + 2 beta^-1 log r + C'         (SPEC §2)
    basins      CIP / SSIP / outer from F_ref minima with the < 2 kT merge rule
    Gate 0      cross-FAMILY spread of fbar vs |F'_ref|, against the campaign ladder
    Gate A      TV between hydration descriptor distributions across basins

Acceptance (§4.5 / SPEC §5): ratio = max pairwise build L2 / (0.10 * consensus F span) <= 0.5.
External check (reported, not a gate): the published 100 ns abf.pmf, same convention, aligned
at the dissociated plateau.

**No numerical Gate 0 threshold** (Amendment 9, binding): the verdict is argued against the
ladder -- WCA 0.040 pass, gateway 0.036 global / 0.189 local pass-marginal, deca 0.61 fail,
R15 0.564 fail -- with the error-carrying region reported separately from the global figure.

Usage:
    python scripts/nacl_ti_analyze.py --ti results/nacl/ti_torch --out results/nacl/reference
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from nacl import system as nsys                                  # noqa: E402

FAMILY_NAMES = {0: "CIP-derived", 1: "SSIP-derived", 2: "dissoc-derived", 3: "local-equil"}
LADDER = {"WCA (pass)": 0.040, "gateway global (pass)": 0.036,
          "gateway constriction (pass, marginal)": 0.189,
          "deca (FAIL)": 0.61, "R15 beta=2 (FAIL)": 0.564}


def integrate(r, f):
    F = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(r))])
    return F - F.mean()


def find_basins(r, F, kT):
    """Local minima merged while the separating barrier from the HIGHER minimum is < 2 kT."""
    mins = [i for i in range(1, len(F) - 1) if F[i] <= F[i - 1] and F[i] < F[i + 1]]
    if F[0] < F[1]:
        mins = [0] + mins
    if F[-1] < F[-2]:
        mins = mins + [len(F) - 1]
    changed = True
    while changed and len(mins) > 1:
        changed = False
        for k in range(len(mins) - 1):
            a, b = mins[k], mins[k + 1]
            barrier = F[a:b + 1].max() - max(F[a], F[b])
            if barrier < 2.0 * kT:
                mins.pop(a if F[a] > F[b] else b if False else (k if F[a] > F[b] else k + 1))
                changed = True
                break
    bounds = []
    for k in range(len(mins) - 1):
        a, b = mins[k], mins[k + 1]
        bounds.append(int(a + np.argmax(F[a:b + 1])))
    return mins, bounds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ti", default="results/nacl/ti_torch")
    ap.add_argument("--out", default="results/nacl/reference")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    d = np.load(os.path.join(args.ti, "ti_final.npz"))
    man = json.load(open(os.path.join(args.ti, "manifest.json")))
    recs, fbar = d["recs"], d["fbar"]
    ysum, ycnt = d["ysum"], d["ycnt"]
    r_grid = np.array(sorted(set(recs[:, 0])))
    builds = sorted(set(recs[:, 1].astype(int)))
    fams = sorted(set(recs[:, 2].astype(int)))
    kT = nsys.kT_kJ()
    beta = nsys.beta_per_kJ()

    # ---- per-build and consensus mean force ------------------------------------------------
    f_build = np.zeros((len(builds), len(r_grid)))
    f_fam = np.zeros((len(fams), len(r_grid)))
    f_cons = np.zeros(len(r_grid))
    f_sem = np.zeros(len(r_grid))
    for i, r in enumerate(r_grid):
        m = recs[:, 0] == r
        f_cons[i] = np.nanmean(fbar[m])
        f_sem[i] = np.nanstd(fbar[m], ddof=1) / np.sqrt(max(m.sum(), 1))
        for bi, b in enumerate(builds):
            f_build[bi, i] = np.nanmean(fbar[m & (recs[:, 1] == b)])
        for fi, fam in enumerate(fams):
            f_fam[fi, i] = np.nanmean(fbar[m & (recs[:, 2] == fam)])

    F_cons = integrate(r_grid, f_cons)
    F_builds = np.stack([integrate(r_grid, f_build[bi]) for bi in range(len(builds))])
    W_cons = F_cons + (2.0 / beta) * np.log(r_grid)
    Wp_cons = f_cons + 2.0 / (beta * r_grid)

    # ---- the frozen endpoint window (SPEC §2.1): F_ref - min <= 15 kT, contiguous -----------
    rel = F_cons - F_cons.min()
    imin = int(np.argmin(F_cons))
    ok = rel <= 15.0 * kT
    lo_i = imin
    while lo_i > 0 and ok[lo_i - 1]:
        lo_i -= 1
    hi_i = imin
    while hi_i < len(r_grid) - 1 and ok[hi_i + 1]:
        hi_i += 1
    window = np.zeros(len(r_grid), dtype=bool)
    window[lo_i:hi_i + 1] = True

    # ---- acceptance (on the frozen window) --------------------------------------------------
    span = float(F_cons[window].max() - F_cons[window].min())
    dz = float(r_grid[1] - r_grid[0])
    pair_l2 = []
    for a in range(len(builds)):
        for b in range(a + 1, len(builds)):
            da = F_builds[a][window] - F_builds[a][window].mean()
            db = F_builds[b][window] - F_builds[b][window].mean()
            pair_l2.append(float(np.sqrt(np.sum((da - db) ** 2) * dz)))
    ratio = (max(pair_l2) / (0.10 * span)) if pair_l2 and span > 0 else np.nan
    accepted = bool(ratio <= 0.5)

    # ---- basins ----------------------------------------------------------------------------
    mins, bounds = find_basins(r_grid, F_cons, kT)
    basin_edges = [0] + list(bounds) + [len(r_grid) - 1]
    basins = [dict(index=k, r_min_nm=float(r_grid[mins[k]]) if k < len(mins) else None,
                   r_lo_nm=float(r_grid[basin_edges[k]]),
                   r_hi_nm=float(r_grid[basin_edges[k + 1]]))
              for k in range(len(basin_edges) - 1)]
    labels = ["CIP", "SSIP", "outer", "outer2"][:len(basins)]
    for b_, lab in zip(basins, labels):
        b_["label"] = lab

    # ---- Gate 0: cross-family spread -------------------------------------------------------
    fam_spread = np.nanmax(f_fam, axis=0) - np.nanmin(f_fam, axis=0)
    denom_global = float(np.nanmean(np.abs(f_cons)))
    gate0_global = float(np.nanmean(fam_spread) / denom_global)
    # the error-carrying region: the barrier neighbourhood (+-0.05 nm around each basin bound)
    region = np.zeros(len(r_grid), dtype=bool)
    for bi in bounds:
        region |= np.abs(r_grid - r_grid[bi]) <= 0.05
    gate0_local = (float(np.nanmean(fam_spread[region]) /
                         np.nanmean(np.abs(f_cons[region]))) if region.any() else np.nan)

    # ---- Gate A: hydration distinguishability across basins --------------------------------
    ybar = ysum / np.maximum(ycnt, 1)[:, None]
    gateA = {}
    for comp, name in enumerate(("n_NaO", "n_ClH", "n_bridge")):
        per_basin = []
        for b_ in basins:
            m = (recs[:, 0] >= b_["r_lo_nm"]) & (recs[:, 0] <= b_["r_hi_nm"])
            per_basin.append(ybar[m, comp])
        tv_max = 0.0
        lo = min(float(np.min(v)) for v in per_basin if len(v))
        hi = max(float(np.max(v)) for v in per_basin if len(v))
        edges = np.linspace(lo, hi + 1e-9, 21)
        hists = [np.histogram(v, bins=edges, density=False)[0].astype(float) for v in per_basin]
        hists = [h / max(h.sum(), 1) for h in hists]
        for a in range(len(hists)):
            for b in range(a + 1, len(hists)):
                tv_max = max(tv_max, 0.5 * float(np.abs(hists[a] - hists[b]).sum()))
        gateA[name] = tv_max
    gateA_max = max(gateA.values())

    # ---- external literature check ---------------------------------------------------------
    pub = np.loadtxt(nsys.SRC_TUTORIAL / "output/abf.pmf")
    pub_r = pub[:, 0] * 0.1
    pub_F = pub[:, 1] * 4.184
    m = (pub_r >= r_grid[0] - 1e-9) & (pub_r <= r_grid[-1] + 1e-9)
    ours = np.interp(pub_r[m], r_grid, F_cons)
    tail = pub_r[m] >= max(r_grid[-1] - 0.2, pub_r[m][0])
    shift = float(np.mean(ours[tail] - pub_F[m][tail]))
    ext_rms = float(np.sqrt(np.mean((ours - shift - pub_F[m]) ** 2)))
    ext_max = float(np.max(np.abs(ours - shift - pub_F[m])))

    np.savez(os.path.join(args.out, "reference.npz"),
             r_nm=r_grid, f_cons=f_cons, f_sem=f_sem, f_build=f_build, f_fam=f_fam,
             F_ref=F_cons, F_builds=F_builds, W_ref=W_cons, Wp_ref=Wp_cons,
             fam_spread=fam_spread, ybar_by_traj=ybar, recs=recs,
             endpoint_window=window)

    report = dict(
        ti_manifest=man,
        n_points=len(r_grid), builds=len(builds), families=len(fams),
        endpoint_window=dict(r_lo_nm=float(r_grid[lo_i]), r_hi_nm=float(r_grid[hi_i]),
                             n_points=int(window.sum()), rule="F_ref - min <= 15 kT, "
                             "largest contiguous interval containing argmin (SPEC §2.1)"),
        acceptance=dict(pairwise_L2=pair_l2, F_span_kJ=span, ratio=ratio,
                        ACCEPTED=accepted,
                        rule="ratio <= 0.5 on the frozen window (Amendment 12.2 / §4.5)"),
        basins=basins,
        gate0=dict(global_spread_ratio=gate0_global, barrier_region_ratio=gate0_local,
                   ladder=LADDER,
                   note="no numerical threshold (Amendment 9); argued against the ladder"),
        gateA=dict(per_descriptor_TV=gateA, max_TV=gateA_max, threshold=0.30,
                   PASS=bool(gateA_max >= 0.30)),
        external_check=dict(source="Talmazan 2025 output/abf.pmf (100 ns ABF)",
                            rms_kJ=ext_rms, max_kJ=ext_max, aligned_at="dissociated tail",
                            note="reported, never a gate; arms score against our reference"),
        physical=dict(
            dW_CIP_to_SSIP_kJ=(float(W_cons[mins[1]] - W_cons[mins[0]])
                               if len(mins) > 1 else None),
            dW_barrier_kJ=(float(W_cons[bounds[0]] - W_cons[mins[0]]) if bounds else None),
            kT_kJ=kT),
    )
    with open(os.path.join(args.out, "reference_report.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=float)

    print(json.dumps({k: report[k] for k in ("acceptance", "basins", "gate0", "gateA",
                                             "external_check", "physical")},
                     indent=2, default=float))
    print(f"\n-> {args.out}/reference.npz, reference_report.json")


if __name__ == "__main__":
    main()
