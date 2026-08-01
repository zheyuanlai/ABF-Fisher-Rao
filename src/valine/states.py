"""Metastable-state decomposition of the full ``(phi, psi, chi1)`` torus ``T^3``.

This is the 3-D lift of :mod:`alanine.basins`, and it exists for one reason: the sec.32 screen
checked the omitted coordinate at six *anchors*, which is strong evidence but is not a global
state decomposition.  Gate V3 asks whether ABF discovers every state and then under-establishes
one, and that question cannot even be posed until the states are named.

Three properties are load-bearing and each of them has bitten this repo before.

**Periodicity is honoured everywhere.**  Distances are the shortest arc, neighbourhoods wrap,
and the watershed floods across the ``+/-pi`` seam.  Linearising the torus at ``-pi`` splits any
state straddling the seam in two -- exactly the bug `valine.umbrella.count_states` had to be
fixed for in 1-D, and in 3-D there are three seams to cross rather than one.

**Empty cells are impassable, not absent.**  A cell no walker ever visited is a region the
dynamics does not reach on this timescale.  Treating it as missing data lets the flood leak
between genuinely separated states and merges them; treating it as a wall is the honest
reading and matches `count_states`.

**The density here is NOT Boltzmann.**  Walkers are seeded on a uniform lattice and relax into
whichever state contains them, so the sampled weight of a state measures its **basin-of-
attraction volume**, not its equilibrium population.  It is the right measure for locating
state *boundaries* and for state-*conditioned* densities ``p(. | B_k)``, and it is the wrong
measure for state populations, which must come from a free-energy estimate instead.  Nothing in
this module returns anything called a population, deliberately.
"""
from __future__ import annotations

import heapq
import itertools
import math

import numpy as np

TWO_PI = 2.0 * math.pi


# --------------------------------------------------------------------------- torus geometry
def wrap(a):
    """Shortest signed arc into ``[-pi, pi)``."""
    return (np.asarray(a) + math.pi) % TWO_PI - math.pi


def torus_distance(a, b):
    """Euclidean distance on ``T^d`` using the shortest arc per coordinate.

    ``d^2(z, z') = sum_c wrap(z_c - z'_c)^2``.  This is the distance the clustering must use;
    cutting the angles at ``-pi`` and using the plain difference would split any state crossing
    the seam.
    """
    return np.sqrt((wrap(np.asarray(a) - np.asarray(b)) ** 2).sum(-1))


def cell_centres(n):
    """Cell-centred grid on ``[-pi, pi)`` for an ``n``-point periodic axis, in radians."""
    return -math.pi + (np.arange(n) + 0.5) * (TWO_PI / n)


def to_cell(angles, n):
    """Cell index of each angle, wrapping.  ``angles`` any shape, returns the same shape."""
    return np.floor((np.asarray(angles) + math.pi) / (TWO_PI / n)).astype(np.int64) % n


# --------------------------------------------------------------------------- density
def histogram_nd(angles, n, weights=None):
    """Periodic ``n^d`` histogram of ``angles`` ``(N, d)``."""
    a = np.asarray(angles)
    d = a.shape[-1]
    idx = to_cell(a.reshape(-1, d), n)
    lin = np.zeros(idx.shape[0], dtype=np.int64)
    for c in range(d):
        lin = lin * n + idx[:, c]
    h = np.bincount(lin, weights=weights, minlength=n ** d)
    return h.reshape((n,) * d)


def neg_log_density(counts, smooth_cells=1.0):
    """``G = -log rho`` in units of the sample scale, with ``+inf`` where nothing was sampled.

    Smoothing is periodic and is applied to the counts, but the ``+inf`` mask is taken from the
    UNSMOOTHED counts: a Gaussian filter would otherwise bleed weight into unvisited cells and
    quietly open a channel between two states that never actually exchange.
    """
    from scipy.ndimage import gaussian_filter

    c = np.asarray(counts, dtype=float)
    visited = c > 0
    s = gaussian_filter(c, sigma=smooth_cells, mode="wrap") if smooth_cells > 0 else c
    tot = s.sum()
    G = np.full(c.shape, np.inf)
    ok = visited & (s > 0)
    G[ok] = -np.log(s[ok] / tot)
    return G - (G[ok].min() if ok.any() else 0.0)


# --------------------------------------------------------------------------- watershed
def _face_offsets(d):
    """The ``2d`` face neighbours, used for connectivity and flooding."""
    out = []
    for c in range(d):
        for s in (-1, 1):
            o = [0] * d
            o[c] = s
            out.append(tuple(o))
    return out


def _full_offsets(d):
    """All ``3^d - 1`` neighbours, used only for local-minimum detection."""
    return [o for o in itertools.product((-1, 0, 1), repeat=d) if any(o)]


def local_minima(G):
    """Periodic full-neighbourhood local minima of ``G``, ascending in depth.

    Returns a list of ``(index_tuple, value)``.  Vectorised with ``np.roll`` -- the equivalent
    Python loop over a 3-D grid is minutes rather than milliseconds.
    """
    d = G.ndim
    finite = np.isfinite(G)
    is_min = finite.copy()
    for o in _full_offsets(d):
        sh = np.roll(G, shift=[-k for k in o], axis=tuple(range(d)))
        # an infinite neighbour is a wall, never a reason to reject a candidate
        is_min &= (G <= sh) | ~np.isfinite(sh)
    idx = np.argwhere(is_min)
    vals = G[tuple(idx.T)]
    order = np.argsort(vals)
    return [(tuple(int(v) for v in idx[k]), float(vals[k])) for k in order]


def _flood(G, sources, targets=None, ceiling=np.inf, labels=None):
    """Priority flood on the periodic grid, shared by the barrier and watershed routines.

    Each cell is reached at the lowest 'water level' -- the maximum of ``G`` along the connecting
    path -- which is the standard min-max-path definition.  If ``targets`` is given, returns the
    level at which the first target is reached (a min-max barrier).  Otherwise returns the
    ``(label, level)`` arrays of a watershed seeded at ``sources``.
    """
    shape = G.shape
    d = G.ndim
    n = shape[0]
    offs = _face_offsets(d)
    level = np.full(shape, np.inf)
    lab = np.full(shape, -1, dtype=np.int64)
    tset = None if targets is None else {tuple(t) for t in targets}
    pq = []
    for k, src in enumerate(sources):
        s = tuple(src)
        if not np.isfinite(G[s]):
            continue
        level[s] = G[s]
        lab[s] = k if labels is None else labels[k]
        heapq.heappush(pq, (G[s], s, lab[s]))
    while pq:
        h, cell, k = heapq.heappop(pq)
        if h > level[cell] or lab[cell] != k:
            continue
        if tset is not None and cell in tset:
            return float(h)
        for o in offs:
            nb = tuple((cell[c] + o[c]) % n for c in range(d))
            g = G[nb]
            if not np.isfinite(g):
                continue
            nh = h if g < h else g
            if nh <= ceiling and nh < level[nb]:
                level[nb] = nh
                lab[nb] = k
                heapq.heappush(pq, (nh, nb, k))
    if tset is not None:
        return float("inf")
    return lab, level


def minmax_barrier(G, src, targets):
    """Lowest level connecting ``src`` to any cell in ``targets`` (periodic, face-connected)."""
    return _flood(G, [src], targets=targets)


def merge_by_prominence(minima, G, min_prominence):
    """Keep a minimum only if a real barrier separates it from an already-accepted deeper one.

    Prominence -- the min-max barrier to any deeper accepted minimum, measured above the
    candidate's own depth -- is the criterion, not depth ranking.  `alanine.basins` documents
    why: ranking by depth keeps the many sub-kT sub-minima inside a megabasin and DROPS the one
    shallow-but-well-separated state the study is about.
    """
    kept = []
    for cell, v in minima:
        if not kept:
            kept.append((cell, v))
            continue
        b = minmax_barrier(G, cell, [c for c, _ in kept])
        if (b - v) >= min_prominence:
            kept.append((cell, v))
    return kept


class StateMap:
    """Metastable states of ``T^d`` from a sampled point cloud.

    ``angles``            ``(N, d)`` samples, radians, IUPAC
    ``n``                 cells per axis
    ``min_prominence_kT`` barrier a candidate minimum needs to become its own state
    ``ceiling_kT``        flood ceiling; cells needing a higher level belong to no state (-1)
    ``kT``                energy scale.  ``G`` is ``-log rho`` of a NON-Boltzmann relaxation
                          density, so ``G`` is in nats; ``kT`` converts the thresholds, which
                          are quoted in kT for continuity with the rest of the repo.  One nat of
                          relaxation density is not one kT of free energy and no claim is made
                          that it is -- these are clustering knobs, and their sensitivity is
                          reported rather than assumed away.
    """

    def __init__(self, angles, n=36, weights=None, smooth_cells=1.0,
                 min_prominence_kT=1.5, ceiling_kT=6.0, kT=1.0, max_states=24,
                 min_cells=8):
        a = np.asarray(angles)
        self.d = a.shape[-1]
        self.n = int(n)
        self.kT = float(kT)
        self.counts = histogram_nd(a, self.n, weights=weights)
        self.G = neg_log_density(self.counts, smooth_cells=smooth_cells)
        mins = local_minima(self.G)
        kept = merge_by_prominence(mins, self.G, min_prominence_kT * self.kT)[:max_states]
        lab, level = _flood(self.G, [c for c, _ in kept], ceiling=ceiling_kT * self.kT)
        # drop states too small to be meaningful, then re-flood so their cells are reassigned
        sizes = np.bincount(lab[lab >= 0].ravel(), minlength=len(kept))
        keep = [k for k in range(len(kept)) if sizes[k] >= min_cells]
        if len(keep) < len(kept):
            kept = [kept[k] for k in keep]
            lab, level = _flood(self.G, [c for c, _ in kept], ceiling=ceiling_kT * self.kT)
        self.seeds = [c for c, _ in kept]
        self.depths = [v for _, v in kept]
        self.label = lab
        self.level = level
        self.grid = cell_centres(self.n)
        self.centres = np.array([[self.grid[c] for c in cell] for cell in self.seeds])
        self.n_states = len(self.seeds)
        self.names = [f"B{k}" for k in range(self.n_states)]

    # -- assignment -------------------------------------------------------
    def assign(self, angles):
        """State label of each sample; ``-1`` for cells above the flood ceiling."""
        idx = to_cell(np.asarray(angles), self.n)
        return self.label[tuple(idx[..., c] for c in range(self.d))]

    def barrier_matrix(self):
        """Pairwise min-max barriers between state seeds, in the units of ``G``."""
        K = self.n_states
        M = np.full((K, K), np.inf)
        for i in range(K):
            M[i, i] = 0.0
            for j in range(i + 1, K):
                b = minmax_barrier(self.G, self.seeds[i], [self.seeds[j]])
                M[i, j] = M[j, i] = b - max(self.depths[i], self.depths[j])
        return M

    def cells_per_state(self):
        return np.bincount(self.label[self.label >= 0].ravel(), minlength=self.n_states)

    def summary(self):
        return dict(
            n_states=self.n_states, n_cells=self.n,
            centres_deg=np.degrees(self.centres).round(1).tolist(),
            depths=[round(v, 3) for v in self.depths],
            cells_per_state=self.cells_per_state().tolist(),
            unassigned_cells=int((self.label < 0).sum()),
            visited_cells=int((self.counts > 0).sum()),
            total_cells=int(self.counts.size))


# --------------------------------------------------------------------------- transitions
def transition_counts(labels_tk, n_states):
    """Count state-to-state moves along ``(n_walkers, n_frames)`` label trajectories.

    Frames labelled ``-1`` (above the flood ceiling) are treated as *in transit*: a move is
    credited from the last assigned state to the next assigned one.  Counting ``-1`` as its own
    state would report a spurious transition every time a walker brushed a ridge.
    """
    L = np.asarray(labels_tk)
    T = np.zeros((n_states, n_states), dtype=np.int64)
    for w in range(L.shape[0]):
        seq = L[w]
        seq = seq[seq >= 0]
        if seq.size < 2:
            continue
        ch = seq[:-1] != seq[1:]
        if ch.any():
            np.add.at(T, (seq[:-1][ch], seq[1:][ch]), 1)
    return T
