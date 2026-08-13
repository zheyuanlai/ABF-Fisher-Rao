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
    # Completeness is tracked explicitly.  `nanmean` over a partly-empty group silently reports
    # a statistic computed on whatever survived, which turns "this build/family produced no
    # data here" into a number indistinguishable from a measurement.
    f_build = np.full((len(builds), len(r_grid)), np.nan)
    f_fam = np.full((len(fams), len(r_grid)), np.nan)
    f_cons = np.full(len(r_grid), np.nan)
    f_sem = np.full(len(r_grid), np.nan)
    missing = []
    for i, r in enumerate(r_grid):
        m = recs[:, 0] == r
        if np.isfinite(fbar[m]).sum() == 0:
            missing.append(dict(r_nm=float(r), what="all trajectories"))
            continue
        f_cons[i] = np.nanmean(fbar[m])
        f_sem[i] = np.nanstd(fbar[m], ddof=1) / np.sqrt(max(np.isfinite(fbar[m]).sum(), 1))
        for bi, b in enumerate(builds):
            sel = m & (recs[:, 1] == b)
            if np.isfinite(fbar[sel]).sum() == 0:
                missing.append(dict(r_nm=float(r), what=f"build {b}"))
            else:
                f_build[bi, i] = np.nanmean(fbar[sel])
        for fi, fam in enumerate(fams):
            sel = m & (recs[:, 2] == fam)
            if np.isfinite(fbar[sel]).sum() == 0:
                missing.append(dict(r_nm=float(r), what=f"family {fam}"))
            else:
                f_fam[fi, i] = np.nanmean(fbar[sel])
    complete = not missing
    if missing:
        print(f"[INCOMPLETE] {len(missing)} build/family/point groups produced no usable "
              f"samples; Gate 0 and acceptance are reported as NOT COMPLETE:", flush=True)
        for entry in missing[:20]:
            print(f"    r = {entry['r_nm']:.3f} nm: {entry['what']}", flush=True)

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
    # Only points where EVERY family reported are usable; a spread taken across a subset of the
    # families is not the cross-family spread, and averaging over such points with `nanmean`
    # would report a Gate 0 statistic that quietly omits the points most likely to be broken.
    fam_ok = np.isfinite(f_fam).all(axis=0) & np.isfinite(f_cons)
    fam_spread = np.where(fam_ok, np.nanmax(f_fam, axis=0) - np.nanmin(f_fam, axis=0), np.nan)
    denom_global = float(np.mean(np.abs(f_cons[fam_ok]))) if fam_ok.any() else np.nan
    gate0_global = (float(np.mean(fam_spread[fam_ok]) / denom_global)
                    if fam_ok.any() and np.isfinite(denom_global) else None)
    # the error-carrying region: the barrier neighbourhood (+-0.05 nm around each basin bound)
    region = np.zeros(len(r_grid), dtype=bool)
    for bi in bounds:
        region |= np.abs(r_grid - r_grid[bi]) <= 0.05
    reg_ok = region & fam_ok
    gate0_local = (float(np.mean(fam_spread[reg_ok]) / np.mean(np.abs(f_cons[reg_ok])))
                   if reg_ok.any() else None)
    gate0_coverage = dict(points_used=int(fam_ok.sum()), points_total=int(len(r_grid)),
                          barrier_points_used=int(reg_ok.sum()),
                          barrier_points_total=int(region.sum()))
    if gate0_global is None or gate0_local is None:
        print("[NOT COMPUTABLE] Gate 0 has no point where all four families reported"
              + ("" if gate0_global is None else " in the barrier region"), flush=True)

    # ---- Gate A: hydration distinguishability across basins --------------------------------
    # A basin with no descriptor samples gives an all-zero histogram, hence TV = 0 against
    # everything -- which reads as "hydration states are indistinguishable through r", a Gate A
    # FAIL that STOPS the study.  That verdict must never be manufactured by absent data, so
    # under-sampled basins make Gate A NOT COMPUTABLE instead.
    MIN_SAMPLES_PER_BASIN = 12
    ybar = np.where(ycnt[:, None] > 0, ysum / np.maximum(ycnt, 1)[:, None], np.nan)
    # half-open, so adjacent basins partition the r-grid instead of sharing their boundary
    # point (see `basin_masks` in nacl_gates.py)
    basin_masks = [((recs[:, 0] >= b_["r_lo_nm"]) &
                    ((recs[:, 0] <= b_["r_hi_nm"]) if k == len(basins) - 1
                     else (recs[:, 0] < b_["r_hi_nm"])))
                   for k, b_ in enumerate(basins)]
    basin_counts = [int(np.isfinite(ybar[m, 0]).sum()) for m in basin_masks]
    gateA_computable = all(c >= MIN_SAMPLES_PER_BASIN for c in basin_counts) and len(basins) > 1
    gateA, gateA_max = {}, None
    if gateA_computable:
        for comp, name in enumerate(("n_NaO", "n_ClH", "n_bridge")):
            per_basin = [ybar[m, comp][np.isfinite(ybar[m, comp])] for m in basin_masks]
            lo = min(float(np.min(v)) for v in per_basin)
            hi = max(float(np.max(v)) for v in per_basin)
            edges = np.linspace(lo, hi + 1e-9, 21)
            hists = [np.histogram(v, bins=edges)[0].astype(float) for v in per_basin]
            hists = [h / h.sum() for h in hists]
            tv_max = 0.0
            for a in range(len(hists)):
                for b in range(a + 1, len(hists)):
                    tv_max = max(tv_max, 0.5 * float(np.abs(hists[a] - hists[b]).sum()))
            gateA[name] = tv_max
        gateA_max = max(gateA.values())
    else:
        print(f"[NOT COMPUTABLE] Gate A: basin sample counts {basin_counts} against a floor of "
              f"{MIN_SAMPLES_PER_BASIN} (and {len(basins)} basins). Reporting NOT COMPUTABLE "
              "rather than TV = 0, which would read as a Gate A FAIL.", flush=True)

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
        completeness=dict(COMPLETE=complete, missing_groups=missing[:50],
                          n_missing=len(missing)),
        acceptance=dict(pairwise_L2=pair_l2, F_span_kJ=span, ratio=ratio,
                        ACCEPTED=bool(accepted and complete),
                        complete=complete,
                        rule="ratio <= 0.5 on the frozen window (Amendment 12.2 / §4.5), and "
                             "every build/family/point group must have reported"),
        basins=basins,
        gate0=dict(global_spread_ratio=gate0_global, barrier_region_ratio=gate0_local,
                   COMPUTABLE=bool(gate0_global is not None and gate0_local is not None),
                   coverage=gate0_coverage, ladder=LADDER,
                   note="no numerical threshold (Amendment 9); argued against the ladder"),
        gateA=dict(per_descriptor_TV=gateA, max_TV=gateA_max, threshold=0.30,
                   COMPUTABLE=bool(gateA_computable), basin_sample_counts=basin_counts,
                   min_samples_per_basin=MIN_SAMPLES_PER_BASIN,
                   PASS=(bool(gateA_max >= 0.30) if gateA_computable else None)),
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
