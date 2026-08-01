"""Ramachandran basins derived from the ACCEPTED reference FES, not from literature constants.

The alkane samplers partition the CV by a single scalar threshold (``|phi| < basin_barrier``,
trans/gauche+/gauche-).  That is meaningless here: alanine's basins are 2-D regions on ``T^2``
whose boundaries follow the reference free-energy watershed.

Basins are built by flooding outward from each reference local minimum, in order of increasing
free energy, over the periodic grid, up to a free-energy ceiling.  Cells above the ceiling belong
to no basin (label ``-1``).  This is a watershed on the *accepted* reference (commit 870b3f6,
``results/alanine/reference/reference.npz``), so basin identity is fixed before any method runs
and is identical for every arm.

Reference minima (corrected reference, IUPAC degrees):
    C7eq   (-74.2, +55.7)   0.00 kT   global
    C5/beta(-152.2, +155.9) 0.45 kT
    C7ax   (+63.1, -48.2)   2.56 kT
    alphaR  region around (-80, -10)
C7eq<->C7ax min-max barrier 15.79 kT; P(C7ax box) = 0.0311; dG(phi>0) = 3.42 kT +/- 0.079
(with an additional ~0.25 kT systematic between samplers -- see ALANINE_REFERENCE_HANDOFF.md).
"""
from __future__ import annotations

import heapq
import math

import numpy as np

TWO_PI = 2.0 * math.pi


def grid_deg(n):
    """Cell-centred IUPAC grid in degrees for an ``n``-point periodic axis."""
    return np.degrees(-math.pi + (np.arange(n) + 0.5) * (TWO_PI / n))


def local_minima(F, mask):
    """Periodic 8-neighbour local minima of ``F`` inside ``mask``, sorted by depth."""
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


def minmax_barrier(F, src, targets):
    """Lowest 'water level' connecting ``src`` to ANY cell in ``targets`` (periodic Dijkstra)."""
    n = F.shape[0]
    tset = {(a, b) for a, b, *_ in targets}
    best = np.full((n, n), np.inf)
    i0, j0 = src[0], src[1]
    best[i0, j0] = F[i0, j0]
    pq = [(F[i0, j0], i0, j0)]
    while pq:
        h, i, j = heapq.heappop(pq)
        if h > best[i, j]:
            continue
        if (i, j) in tset:
            return float(h)
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = (i + di) % n, (j + dj) % n
            if not np.isfinite(F[a, b]):
                continue
            nh = max(h, F[a, b])
            if nh < best[a, b]:
                best[a, b] = nh
                heapq.heappush(pq, (nh, a, b))
    return float("inf")


def merge_by_prominence(mins, F, kT, min_prominence_kT=1.0):
    """Keep a minimum as its own basin only if a real barrier separates it from a deeper one.

    Distance-based merging is the wrong criterion here.  The FES has many shallow sub-minima
    inside the phi<0 megabasin separated by ridges well under 1 kT; ranking by depth and taking
    the deepest few keeps those and **drops C7ax** (2.56 kT), which is the one basin the study
    is about.  Prominence -- the min-max barrier to any deeper accepted minimum, measured above
    the candidate's own depth -- merges the sub-minima and keeps C7ax, whose barrier is 15.8 kT.
    """
    kept = []
    for i, j, v in mins:                       # ascending depth
        if not kept:
            kept.append((i, j, v))
            continue
        barrier = minmax_barrier(F, (i, j), kept)
        if (barrier - v) >= min_prominence_kT * kT:
            kept.append((i, j, v))
    return kept


def watershed(F, seeds, ceiling):
    """Priority-flood watershed on the periodic grid.

    Each cell is assigned to the seed whose flooding front reaches it at the lowest
    'water level' (the max free energy along the connecting path), which is the standard
    min-max-path basin definition.  Cells needing a level above ``ceiling`` stay unassigned.
    """
    n = F.shape[0]
    label = np.full((n, n), -1, dtype=np.int64)
    level = np.full((n, n), np.inf)
    pq = []
    for k, (i, j, _) in enumerate(seeds):
        level[i, j] = F[i, j]
        label[i, j] = k
        heapq.heappush(pq, (F[i, j], i, j, k))
    while pq:
        h, i, j, k = heapq.heappop(pq)
        if h > level[i, j] or label[i, j] != k:
            continue
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = (i + di) % n, (j + dj) % n
            if not np.isfinite(F[a, b]):
                continue
            nh = max(h, F[a, b])
            if nh <= ceiling and nh < level[a, b]:
                level[a, b] = nh
                label[a, b] = k
                heapq.heappush(pq, (nh, a, b, k))
    return label, level


NAME_HINTS = (
    ("C7eq", (-160.0, -40.0, 20.0, 140.0)),
    ("C5",   (-180.0, -100.0, 120.0, 180.0)),
    ("alphaR", (-160.0, -30.0, -80.0, 20.0)),
    ("C7ax", (20.0, 120.0, -100.0, 20.0)),
)


def _name_for(phi_deg, psi_deg, used):
    def inbox(v, lo, hi):
        d = abs((v - 0.5 * (lo + hi) + 180.0) % 360.0 - 180.0)
        return d <= 0.5 * (hi - lo) + 1e-9
    for nm, (p0, p1, s0, s1) in NAME_HINTS:
        if nm not in used and inbox(phi_deg, p0, p1) and inbox(psi_deg, s0, s1):
            return nm
    return None


class BasinMap:
    """Reference-derived basin labelling, shared identically by every method arm."""

    def __init__(self, F, mask, kT, ceiling_kT=8.0, min_prominence_kT=1.0, max_basins=8):
        F = np.asarray(F, dtype=float)
        self.n = F.shape[0]
        self.kT = float(kT)
        finite = np.isfinite(F)
        self.F = F - F[finite].min()
        self.mask = np.asarray(mask, dtype=bool) & finite
        mins = merge_by_prominence(local_minima(self.F, self.mask), self.F, self.kT,
                                   min_prominence_kT)[:max_basins]
        self.seeds = mins
        self.label, self.level = watershed(self.F, mins, ceiling_kT * self.kT)
        g = grid_deg(self.n)
        self.centres_deg = [(float(g[i]), float(g[j])) for i, j, _ in mins]
        self.depths_kT = [float(v / self.kT) for _, _, v in mins]
        used, names = set(), []
        for (pd, sd) in self.centres_deg:
            nm = _name_for(pd, sd, used) or f"B{len(names)}"
            used.add(nm)
            names.append(nm)
        self.names = names
        self.index = {nm: k for k, nm in enumerate(names)}

    # -- assignment -------------------------------------------------------
    def assign_np(self, phi, psi):
        """Basin label of IUPAC radians ``phi, psi`` (any shape); ``-1`` = unassigned."""
        dz = TWO_PI / self.n
        i = np.floor((np.asarray(phi) + math.pi) / dz).astype(np.int64) % self.n
        j = np.floor((np.asarray(psi) + math.pi) / dz).astype(np.int64) % self.n
        return self.label[i, j]

    def label_tensor(self, device=None, dtype=None):
        """The label grid as a torch tensor, for on-device assignment in the hot loop."""
        import torch
        return torch.as_tensor(self.label, device=device,
                               dtype=dtype or torch.long)

    def population(self, F_hat, kT=None):
        """Boltzmann populations of every basin from a reconstructed FES."""
        kT = kT or self.kT
        Fh = np.where(np.isfinite(F_hat), F_hat, np.inf)
        P = np.exp(-(Fh - np.nanmin(Fh[np.isfinite(Fh)])) / kT)
        P = P / P.sum()
        return {nm: float(P[self.label == k].sum()) for k, nm in enumerate(self.names)}

    def delta_G(self, F_hat, a, b, kT=None):
        """``dG = -kT log(P_a / P_b)`` between two named basins."""
        kT = kT or self.kT
        p = self.population(F_hat, kT)
        pa, pb = max(p[a], 1e-300), max(p[b], 1e-300)
        return -kT * math.log(pa / pb)

    def summary(self):
        return dict(n_basins=len(self.names), names=list(self.names),
                    centres_deg=list(self.centres_deg), depths_kT=list(self.depths_kT),
                    assigned_cells=int((self.label >= 0).sum()),
                    cells_per_basin={nm: int((self.label == k).sum())
                                     for k, nm in enumerate(self.names)})


def from_reference(path="results/alanine/reference/reference.npz", ceiling_kT=8.0):
    """Build the basin map from the accepted reference artifact."""
    import json
    d = np.load(path, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    return BasinMap(d["F"], d["mask8"], meta["kT_kJ"], ceiling_kT=ceiling_kT), meta
