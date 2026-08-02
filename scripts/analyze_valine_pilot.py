#!/usr/bin/env python
"""Accept (or reject) the coarse pilot reference, and read the barriers off it.

Two jobs.

**1. Acceptance.**  The pilot is screening-only, so the bar is not the eight gates of the
alanine reference; it is the four things V3 actually needs: connected MBAR overlap, no major
state missing, qualitative split-half stability, and agreement between the two independent psi
starts.

**2. The barrier numbers, with the backbone FREE in psi.**  This is what closes the Stage-0
correction quantitatively.  Stage 0 measured F(chi1) with phi AND psi clamped at
kappa = 500 kJ/mol/rad^2 and reported 11.3-17.9 kT; the S1 exploration then measured 2.70
rotamer changes per walker per ns, implying only 6-8 kT.  The pilot restrains phi and chi1 but
leaves psi free, so it separates two distinct quantities:

  * the CONDITIONAL chi1 barrier at fixed phi -- the 1-D slice, comparable to Stage 0 except
    that psi has been allowed to relax;
  * the EFFECTIVE chi1 barrier -- the minimum-maximum path through the 2-D (phi, chi1) plane
    between two rotamer basins, which is what a walker free to move in phi actually pays, and
    therefore what the measured rate should reflect.

Reporting only the first would repeat Stage 0's error in a milder form.

Usage
-----
    python scripts/analyze_valine_pilot.py --pilot results/valine/pilot_reference
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from alanine.basins import BasinMap, grid_deg, minmax_barrier                 # noqa: E402

KB = 0.008314462618


def slice_barrier(F, kT):
    """Min-max barrier along a periodic 1-D slice, between its two deepest wells."""
    f = np.asarray(F, dtype=float)
    ok = np.isfinite(f)
    if ok.sum() < 4:
        return None
    n = len(f)
    mins = [i for i in range(n) if ok[i] and ok[(i - 1) % n] and ok[(i + 1) % n]
            and f[i] <= f[(i - 1) % n] and f[i] <= f[(i + 1) % n]]
    if len(mins) < 2:
        return None
    mins.sort(key=lambda i: f[i])
    a, b = mins[0], mins[1]
    best = np.inf
    for direction in (1, -1):
        i, peak = a, f[a]
        while i != b:
            i = (i + direction) % n
            if not ok[i]:
                peak = np.inf
                break
            peak = max(peak, f[i])
        best = min(best, peak)
    if not np.isfinite(best):
        return None
    return dict(barrier_kT=float((best - f[a]) / kT), well_kT=float(f[a] / kT),
                second_well_kT=float((f[b] - f[a]) / kT))


def psi_equilibration(pilot, n_bins=36):
    """PAIRED test of whether psi has equilibrated inside each umbrella window.

    Compare ``p(psi | window)`` ACROSS STARTS WITHIN THE SAME WINDOW, and calibrate against the
    same statistic computed between COPIES OF ONE START, which is pure sampling noise.

    The paired part is the whole point.  The first two versions of this check compared the psi
    starts *globally* -- a mean beta/PPII occupancy per start, and a per-start MBAR free energy --
    and both reported a large disagreement for a reason that has nothing to do with psi: the
    starts do not cover the same windows.  Only 61 of 315 windows survive structural validation
    for all four starts, and beta/PPII occupancy is mostly a property of WHICH WINDOW a walker is
    in.  Averaging over unmatched window sets turns a coverage difference into an apparent
    equilibration failure -- an unpaired comparison wearing a paired comparison's clothes.
    Measured here: start-memory spread 0.169 over all windows, 0.010 over the matched ones.

    Returns medians of the across-start worst pair, of the same-start noise floor, and the ratio.
    """
    s = np.load(os.path.join(pilot, "samples.npz"), allow_pickle=True)
    th, win, ps0 = s["theta"], s["window"], s["psi_start"]
    psi = th[:, th.shape[1] // 2:, 1].astype(np.float64)      # second half only
    starts = np.array(sorted(set(ps0.tolist())))
    have = {}
    for w, p in zip(win, ps0):
        have.setdefault(int(w), set()).add(round(float(p), 6))
    full = sorted(w for w, ss in have.items() if len(ss) == len(starts))
    edges = np.linspace(-math.pi, math.pi, n_bins + 1)

    worst, floor = [], []
    for w in full:
        hs = []
        for p in starts:
            h, _ = np.histogram(psi[(win == w) & np.isclose(ps0, p)].ravel(), bins=edges)
            hs.append(h / max(h.sum(), 1))
        worst.append(max(0.5 * np.abs(hs[i] - hs[j]).sum()
                         for i in range(len(hs)) for j in range(i + 1, len(hs))))
        idx = np.flatnonzero((win == w) & np.isclose(ps0, starts[0]))
        if len(idx) >= 4:
            ha, _ = np.histogram(psi[idx[:len(idx) // 2]].ravel(), bins=edges)
            hb, _ = np.histogram(psi[idx[len(idx) // 2:]].ravel(), bins=edges)
            floor.append(0.5 * np.abs(ha / max(ha.sum(), 1) - hb / max(hb.sum(), 1)).sum())
    if not worst or not floor:
        return None
    mw, mf = float(np.median(worst)), float(np.median(floor))
    return dict(n_starts=len(starts), n_matched_windows=len(full),
                across_start_tv_median=mw, same_start_noise_floor=mf,
                ratio=mw / max(mf, 1e-9),
                across_start_tv_p90=float(np.percentile(worst, 90)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", default="results/valine/pilot_reference")
    ap.add_argument("--ceiling-kT", type=float, default=8.0)
    ap.add_argument("--min-prominence-kT", type=float, default=1.0)
    ap.add_argument("--max-basins", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=300.0)
    a = ap.parse_args()
    kT = KB * a.temperature

    meta = json.load(open(os.path.join(a.pilot, "meta.json")))
    pf = np.load(os.path.join(a.pilot, "pilot_reference.npz"), allow_pickle=True)
    F = pf["F"]
    n = F.shape[0]
    g = grid_deg(n)
    finite = np.isfinite(F)

    # ---------------------------------------------------------------- acceptance
    O = pf["overlap"]
    nn = meta["nn_overlap"]
    sh = meta.get("split_half") or {}
    ps = meta.get("psi_start_agreement") or {}
    print(f"pilot: {meta['n_windows']} windows, {meta['n_seeds']} (window, psi-start) seeds, "
          f"{meta['copies_per_seed']} copies, dt {meta['dt_ps'] * 1000:.1f} fs")
    print(f"  kinetic temperature {meta['mean_temperature_K']:.2f} K "
          f"({100 * meta['temperature_deviation_frac']:.2f} % from 300)")
    print(f"  MBAR: {meta['mbar_iterations']} iterations, residual {meta['mbar_residual']:.1e}")
    # CONNECTIVITY, not the minimum pairwise overlap.  MBAR needs the window lattice to form one
    # connected component so every window's free energy is tied to every other's; a handful of
    # weak links in a high-F corner is not a defect, and gating on min(overlap) rejects a
    # perfectly usable map for it.  The first pilot had min 1e-4 yet was a single component even
    # at a threshold of 0.001.
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    n_comp, _ = connected_components(csr_matrix((O + O.T) / 2 > 0.01), directed=False)
    checks = [
        ("MBAR overlap graph is connected", n_comp == 1,
         f"{n_comp} component(s) at threshold 0.01; min pair {nn['min']:.4f}, "
         f"median {nn['median']:.3f}"),
        ("grid coverage > 25 % of cells", meta["grid_cells_filled"] / meta["grid_cells"] > 0.25,
         f"{meta['grid_cells_filled']}/{meta['grid_cells']} "
         f"({100 * meta['grid_cells_filled'] / meta['grid_cells']:.1f} %)"),
        ("split-half RMSE < 1 kT", sh.get("rmse_kT", 9e9) < 1.0,
         f"{sh.get('rmse_kT', float('nan')):.3f} kT over {sh.get('n_cells', 0)} cells"),
    ]
    pe = psi_equilibration(a.pilot)
    if pe is not None:
        checks.append(
            ("psi equilibrated in-window (paired)", pe["ratio"] < 2.0,
             f"across-start TV {pe['across_start_tv_median']:.3f} vs same-start noise floor "
             f"{pe['same_start_noise_floor']:.3f} (ratio {pe['ratio']:.2f}) over "
             f"{pe['n_matched_windows']} windows carrying all {pe['n_starts']} starts"))
    print("\nreported but NOT gated -- both compare subsets covering DIFFERENT windows, so they "
          "\nmeasure coverage as much as equilibration (see psi_equilibration's docstring):")
    print(f"  per-start FES RMSE (worst pair) {ps.get('rmse_kT', float('nan')):.2f} kT;  "
          f"unpaired start-memory spread {meta.get('psi_start_memory_spread', float('nan')):.3f}")
    print("\nacceptance (screening bar, NOT the alanine reference bar):")
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:38s} {detail}")
    accepted_ok = all(c[1] for c in checks)

    # ---------------------------------------------------------------- basins
    mask = finite & (F < a.ceiling_kT * kT)
    bm = BasinMap(F, mask, kT, ceiling_kT=a.ceiling_kT,
                  min_prominence_kT=a.min_prominence_kT, max_basins=a.max_basins,
                  name_hints=())
    pops = bm.population(F)
    print(f"\n{len(bm.names)} regions on (phi, chi1):")
    for k, nm in enumerate(bm.names):
        c = bm.centres_deg[k]
        print(f"  {nm:>4s} centre (phi {c[0]:+7.1f}, chi1 {c[1]:+7.1f})  depth "
              f"{bm.depths_kT[k]:5.2f} kT  population {pops[nm]:.4f}  "
              f"cells {int((bm.label == k).sum())}")

    # ---------------------------------------------------------------- barriers
    Fk = np.where(finite, F, np.inf)
    seeds = bm.seeds                                # (i, j, value) per region
    print("\nEFFECTIVE barriers -- min-max path through the 2-D (phi, chi1) plane, "
          "psi free.\nThis is what a walker free to move in phi actually pays.")
    print(f"  {'from':>5s} {'to':>5s} {'barrier kT':>11s}   note")
    eff = []
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            b = minmax_barrier(bm.F, (seeds[i][0], seeds[i][1]), [seeds[j]])
            if not np.isfinite(b):
                bk = float("inf")
            else:
                bk = (b - max(seeds[i][2], seeds[j][2])) / kT
            ci, cj = bm.centres_deg[i], bm.centres_deg[j]
            same_bb = (ci[0] > 0) == (cj[0] > 0)
            kind = ("chi1 rotamer, same backbone" if same_bb else "crosses the phi megabasin")
            eff.append(dict(a=bm.names[i], b=bm.names[j], barrier_kT=bk, kind=kind))
            print(f"  {bm.names[i]:>5s} {bm.names[j]:>5s} "
                  + (f"{bk:11.2f}" if np.isfinite(bk) else f"{'inf':>11s}") + f"   {kind}")

    same = [e["barrier_kT"] for e in eff
            if e["kind"].startswith("chi1") and np.isfinite(e["barrier_kT"])]
    cross = [e["barrier_kT"] for e in eff
             if e["kind"].startswith("crosses") and np.isfinite(e["barrier_kT"])]

    # conditional chi1 barrier at fixed phi, psi relaxed -- the Stage-0 comparison
    print("\nCONDITIONAL chi1 barrier at fixed phi (1-D slice, psi relaxed):")
    print(f"  {'phi':>7s} {'barrier kT':>11s} {'2nd well kT':>12s}")
    cond = []
    for i in range(n):
        if not finite[i].any():
            continue
        r = slice_barrier(F[i], kT)
        if r is None:
            continue
        r["phi_deg"] = float(g[i])
        cond.append(r)
    cond_pop = [c for c in cond if c["barrier_kT"] > 0]
    for c in cond_pop[::max(1, len(cond_pop) // 10)]:
        print(f"  {c['phi_deg']:7.1f} {c['barrier_kT']:11.2f} {c['second_well_kT']:12.2f}")
    cb = np.array([c["barrier_kT"] for c in cond_pop]) if cond_pop else np.array([np.nan])

    print(f"\nSUMMARY")
    print(f"  conditional chi1 barrier (phi fixed, psi free): median {np.nanmedian(cb):.1f} kT, "
          f"range {np.nanmin(cb):.1f}-{np.nanmax(cb):.1f} over {len(cond_pop)} phi slices")
    if same:
        print(f"  EFFECTIVE chi1 barrier (phi and psi free):      "
              f"{min(same):.1f}-{max(same):.1f} kT over {len(same)} same-backbone pairs")
    if cross:
        print(f"  backbone megabasin barrier:                    "
              f"{min(cross):.1f}-{max(cross):.1f} kT")
    print(f"  Stage 0 reported 11.3-17.9 kT with phi AND psi CLAMPED at kappa = 500.")
    print(f"  S1 measured 2.70 chi1 changes per walker per ns, implying 6-8 kT.")

    out = dict(pilot=a.pilot, accepted=bool(accepted_ok),
               checks=[dict(name=c[0], passed=bool(c[1]), detail=c[2]) for c in checks],
               psi_equilibration=pe,
               unpaired_psi_diagnostics_note=(
                   "psi_start_agreement and psi_start_memory_spread in meta.json compare subsets "
                   "covering DIFFERENT windows and are confounded by coverage; the paired "
                   "psi_equilibration test supersedes them"),
               regions=[dict(name=nm, centre_deg=list(bm.centres_deg[k]),
                             depth_kT=bm.depths_kT[k], population=pops[nm],
                             cells=int((bm.label == k).sum()))
                        for k, nm in enumerate(bm.names)],
               effective_barriers=eff,
               effective_chi1_kT=[float(x) for x in same],
               backbone_barrier_kT=[float(x) for x in cross],
               conditional_chi1_kT=dict(
                   median=float(np.nanmedian(cb)), min=float(np.nanmin(cb)),
                   max=float(np.nanmax(cb)), n_slices=len(cond_pop)),
               conditional_by_phi=cond_pop,
               stage0_clamped_kT=[11.3, 17.9],
               s1_rate_implied_kT=[6.0, 8.0])
    p = os.path.join(a.pilot, "pilot_analysis.json")
    with open(p, "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print(f"\npilot {'ACCEPTED' if accepted_ok else 'REJECTED'} for screening use")
    print(f"wrote {p}")
    # Non-zero exit on rejection, so a launch chain cannot run V3 against a pilot that failed.
    if not accepted_ok:
        sys.exit(2)


if __name__ == "__main__":
    main()
