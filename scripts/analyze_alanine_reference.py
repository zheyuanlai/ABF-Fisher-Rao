"""Acceptance gates and physics recomputation for the corrected alanine reference FES.

Recomputes, from the CORRECTED reference only (nothing is inherited from the contaminated first
attempt): basin locations, basin populations, dG(C7eq->C7ax), barrier estimates, F(psi|phi), and
the claim that psi is or is not a hidden slow coordinate.

Usage: CUDA_VISIBLE_DEVICES="" python scripts/analyze_alanine_reference.py \
           --ref results/alanine/reference/reference.npz
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

KB = 0.008314462618
TWO_PI = 2.0 * math.pi


def grid_deg(n):
    return np.degrees(-math.pi + (np.arange(n) + 0.5) * (TWO_PI / n))


def local_minima(F, mask):
    """Grid cells lower than all 8 periodic neighbours and inside ``mask``."""
    n = F.shape[0]
    out = []
    for i in range(n):
        for j in range(n):
            if not mask[i, j] or not np.isfinite(F[i, j]):
                continue
            v = F[i, j]
            nb = [F[(i + di) % n, (j + dj) % n]
                  for di in (-1, 0, 1) for dj in (-1, 0, 1) if (di, dj) != (0, 0)]
            if all((not np.isfinite(x)) or v <= x for x in nb):
                out.append((i, j, float(v)))
    return sorted(out, key=lambda t: t[2])


def basin_populations(F, beta, boxes):
    """Boltzmann populations of named (phi,psi) boxes, in degrees, periodic."""
    n = F.shape[0]
    g = grid_deg(n)
    P = np.exp(-beta * np.where(np.isfinite(F), F, np.inf))
    P = P / P.sum()
    G1, G2 = np.meshgrid(g, g, indexing="ij")

    def inbox(lo, hi, X):
        """Periodic membership on the circle, correct for a FULL-RANGE box.

        The naive ``lo % 360, hi % 360`` form maps (-180, 180) to lo == hi == 180 and selects a
        single point instead of everything, which silently emptied the phi<0 / phi>0 boxes.
        Work instead with the wrapped distance from the box centre.
        """
        if (hi - lo) >= 360.0 - 1e-9:
            return np.ones_like(X, dtype=bool)
        centre = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)
        d = np.abs((X - centre + 180.0) % 360.0 - 180.0)
        return d <= half + 1e-9

    out = {}
    for name, (p0, p1, s0, s1) in boxes.items():
        m = inbox(p0, p1, G1) & inbox(s0, s1, G2)
        out[name] = float(P[m].sum())
    return out, P


def mfep_barrier(F, a, b, n_iter=4000):
    """Crude min-max barrier between two grid cells via periodic Dijkstra on max-height paths."""
    import heapq
    n = F.shape[0]
    INF = float("inf")
    best = np.full((n, n), INF)
    src = (a[0], a[1])
    best[src] = F[src]
    pq = [(F[src], src)]
    while pq:
        h, (i, j) = heapq.heappop(pq)
        if h > best[i, j]:
            continue
        if (i, j) == (b[0], b[1]):
            return float(h)
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            k, l = (i + di) % n, (j + dj) % n
            if not np.isfinite(F[k, l]):
                continue
            nh = max(h, F[k, l])
            if nh < best[k, l]:
                best[k, l] = nh
                heapq.heappush(pq, (nh, (k, l)))
    return float("inf")


def circular_internal_barrier(Fc, kT, min_depth_kT=0.5):
    """Barrier separating the two deepest *populated* minima of a periodic 1-D profile.

    For each of the two arcs joining them, take the maximum; the accessible route is the
    SMALLER of those two maxima; the barrier is that height above the SHALLOWER minimum (the
    one that would have to be escaped).  Returns 0 when the profile has only one well.

    The earlier version returned ``max(Fc)`` over the whole circle, which is the height of the
    sterically FORBIDDEN region, not a barrier between populated states -- it scored a
    single-well profile at 6 kT instead of 0.
    """
    n = len(Fc)
    loc = [i for i in range(n)
           if Fc[i] <= Fc[(i - 1) % n] and Fc[i] <= Fc[(i + 1) % n] and np.isfinite(Fc[i])]
    if len(loc) < 2:
        return 0.0
    loc.sort(key=lambda i: Fc[i])
    a, b = loc[0], loc[1]
    if (Fc[b] - Fc[a]) > 30.0 * kT:          # second well thermally irrelevant
        return 0.0
    i, j = (a, b) if a < b else (b, a)
    arc1 = Fc[i:j + 1]
    arc2 = np.concatenate([Fc[j:], Fc[:i + 1]])
    route = min(np.nanmax(arc1), np.nanmax(arc2))
    return float(max(0.0, route - Fc[b]))    # escape the shallower of the two


def periodic_internal_barrier(Fc):
    """Barrier separating the two lowest distinct basins of a PERIODIC 1-D profile.

    Returns 0 when the profile has a single basin.  The previous implementation returned the
    global maximum of the profile -- the height of the sterically forbidden wall -- which reads
    15-22 kT for a one-basin conditional and would wrongly declare psi a hidden slow coordinate.
    """
    n = len(Fc)
    mins = [i for i in range(n)
            if Fc[i] <= Fc[(i - 1) % n] and Fc[i] <= Fc[(i + 1) % n] and np.isfinite(Fc[i])]
    if len(mins) < 2:
        return 0.0
    mins.sort(key=lambda i: Fc[i])
    a = mins[0]
    # second basin: the next-lowest minimum that is not adjacent to the first
    b = next((i for i in mins[1:] if min(abs(i - a), n - abs(i - a)) > 1), None)
    if b is None:
        return 0.0
    # min-max barrier along the two arcs, measured from the SHALLOWER basin
    def arc_max(i, j):
        out, k = -np.inf, i
        while k != j:
            k = (k + 1) % n
            out = max(out, Fc[k])
        return out
    saddle = min(arc_max(a, b), arc_max(b, a))
    return float(saddle - Fc[b])


def conditional_psi(F, beta, min_p=1e-3):
    """Internal barrier of F(psi|phi) at every phi column carrying population."""
    n = F.shape[0]
    P = np.exp(-beta * np.where(np.isfinite(F), F, np.inf))
    pphi = P.sum(1) / P.sum()
    g = grid_deg(n)
    rows = []
    for i in range(n):
        if pphi[i] <= min_p:
            continue
        col = P[i] / P[i].sum()
        Fc = -np.log(np.maximum(col, 1e-300)) / beta
        Fc = Fc - Fc.min()
        rows.append((float(g[i]), float(pphi[i]),
                     float(periodic_internal_barrier(Fc) / (KB * 300.0))))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="results/alanine/reference/reference.npz")
    ap.add_argument("--out", default="results/alanine/reference")
    ap.add_argument("--n-boot", type=int, default=200)
    a = ap.parse_args()

    d = np.load(a.ref, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    F = d["F"]
    n = F.shape[0]
    kT = meta["kT_kJ"]
    beta = 1.0 / kT
    finite = np.isfinite(F)
    F = F - F[finite].min()
    mask8 = finite & (F <= 8 * kT)
    g = grid_deg(n)

    print("=" * 78)
    print("CORRECTED ALANINE REFERENCE — acceptance gates")
    print("=" * 78)
    gates = {}

    gates["G1_seed_gate"] = (meta["seed_gate_pass"] == meta["seed_gate_total"],
                             f"{meta['seed_gate_pass']}/{meta['seed_gate_total']} seeds pass")
    gates["G2_mbar_converged"] = (meta["mbar_resid"] < 1e-8,
                                  f"resid {meta['mbar_resid']:.2e} in {meta['mbar_iters']} iters")
    ev = meta["nn_overlap_eval"]
    gates["G3_min_overlap_eval"] = (ev.get("min", 0) >= 0.03,
                                    f"min NN overlap in 8kT region {ev.get('min', float('nan')):.4f} "
                                    f"({ev.get('n_below_0p03', -1)} pairs < 0.03 of {ev.get('n_pairs', 0)})")
    al = meta["nn_overlap_all"]
    gates["G4_median_overlap"] = (al.get("median", 0) >= 0.05,
                                  f"median NN overlap (all) {al.get('median', float('nan')):.4f}")
    imin = np.unravel_index(np.argmin(np.where(finite, F, np.inf)), F.shape)
    gmin = (g[imin[0]], g[imin[1]])
    in_c7eq = (-120 <= gmin[0] <= -40) and (20 <= gmin[1] <= 100)
    gates["G5_global_min_C7eq"] = (in_c7eq, f"global min at ({gmin[0]:+.1f}, {gmin[1]:+.1f}) deg")

    for k, (ok, msg) in gates.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {k:24s} {msg}")

    # ------------------------------------------------------------------ physics
    print("\n" + "=" * 78)
    print("RECOMPUTED PHYSICS (from the corrected reference only)")
    print("=" * 78)
    mins = local_minima(F, mask8)
    print(f"local minima inside the 8 kT region ({len(mins)}):")
    for i, j, v in mins[:8]:
        print(f"   ({g[i]:+7.1f}, {g[j]:+7.1f}) deg   F = {v:6.2f} kJ/mol = {v/kT:5.2f} kT")

    boxes = {"C7eq/beta (phi<0)": (-180, 0, -180, 180),
             "C7ax (phi>0)":      (0, 180, -180, 180),
             "alphaR":            (-160, -20, -120, 30),
             "C7eq":              (-160, -40, 20, 130),
             "C7ax box":          (20, 110, -90, 10)}
    pops, P = basin_populations(F, beta, boxes)
    print("\nbasin populations:")
    for k, v in pops.items():
        print(f"   {k:20s} P = {v:.5f}")
    p_pos = pops["C7ax (phi>0)"]
    dG = -kT * math.log(max(p_pos, 1e-300) / max(1 - p_pos, 1e-300))
    print(f"\n   dG(phi>0 vs phi<=0) = {dG:+.2f} kJ/mol = {dG/kT:+.2f} kT   (P(phi>0) = {p_pos:.4f})")

    # C7eq -> C7ax barrier
    c7eq = min(mins, key=lambda t: abs(g[t[0]] + 79) + abs(g[t[1]] - 56)) if mins else None
    pos = [m for m in mins if g[m[0]] > 0]
    if c7eq and pos:
        c7ax = min(pos, key=lambda t: t[2])
        sad = mfep_barrier(np.where(finite, F, np.inf), (c7eq[0], c7eq[1]), (c7ax[0], c7ax[1]))
        print(f"   C7ax minimum at ({g[c7ax[0]]:+.1f}, {g[c7ax[1]]:+.1f}) deg, "
              f"{c7ax[2]:.2f} kJ/mol = {c7ax[2]/kT:.2f} kT above C7eq")
        print(f"   C7eq<->C7ax min-max barrier: {sad:.2f} kJ/mol = {sad/kT:.2f} kT")

    # psi as a hidden coordinate?
    rows = conditional_psi(F, beta)
    bars = np.array([r[2] for r in rows])
    print(f"\nF(psi|phi) internal barrier over {len(rows)} populated phi columns:")
    print(f"   median {np.median(bars):.2f} kT   p90 {np.percentile(bars,90):.2f} kT   "
          f"max {bars.max():.2f} kT")
    hidden = bars.max() > 3.0
    print(f"   => psi {'IS' if hidden else 'is NOT'} a hidden slow coordinate "
          f"(threshold: max internal barrier > 3 kT)")

    summary = dict(meta=meta, gates={k: [bool(v[0]), v[1]] for k, v in gates.items()},
                   global_min_deg=[float(gmin[0]), float(gmin[1])],
                   n_local_minima=len(mins),
                   local_minima=[[float(g[i]), float(g[j]), float(v)] for i, j, v in mins[:10]],
                   populations=pops, dG_phi_pos_kJ=float(dG), dG_phi_pos_kT=float(dG / kT),
                   psi_cond_barrier_kT=dict(median=float(np.median(bars)),
                                            p90=float(np.percentile(bars, 90)),
                                            max=float(bars.max())),
                   psi_is_hidden_slow=bool(hidden),
                   all_gates_pass=bool(all(v[0] for v in gates.values())))
    with open(os.path.join(a.out, "acceptance.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print(f"\nALL ACCEPTANCE GATES: {'PASS' if summary['all_gates_pass'] else 'FAIL'}")
    print(f"wrote {os.path.join(a.out, 'acceptance.json')}")


if __name__ == "__main__":
    main()
