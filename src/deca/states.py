"""States, discovery and establishment for deca-alanine — Gates B and C.

Everything here is **reference-dependent**, which is exactly why none of it lives in the
sampler. The sampler saves a ``xi`` trace and a bias profile and knows nothing about the
reference; the regime is then diagnosed here, from ABF-only data, without any mFR result ever
being consulted. That separation is what made the v1 gateway classification credible and it is
reproduced deliberately.

The state definition is frozen by **Amendment 3** of ``docs/V2_PREREGISTRATION.md``, written
before ``F_ref`` existed — including the single-basin fallback, because a rule that only covers
the convenient case is not a rule.
"""
from __future__ import annotations

import numpy as np

#: Amendment 3: adjacent minima separated by a barrier below this (measured from the higher
#: minimum) are merged.  Deliberately low -- merging aggressively would manufacture a single
#: basin and force the fallback partition.
MERGE_BARRIER_KT = 2.0

#: Gate B (§2.3): a state must be reached within this fraction of the run length ...
DISCOVERY_FRACTION = 0.1
#: ... on at least this many of the screening seeds.
DISCOVERY_MIN_SEEDS = 6

#: Gate C (§2.4): under-established means below this multiple of the bias-aware target ...
ESTABLISHMENT_DEFICIT = 0.5
#: ... for a contiguous span of at least this fraction of the run.
ESTABLISHMENT_SPAN = 0.20


def find_basins(grid, F_ref, beta, merge_barrier_kT=MERGE_BARRIER_KT):
    """Amendment 3 primary rule.  Returns ``(edges, minima_idx, used_fallback)``.

    ``edges`` is a ``(K+1,)`` array of state boundaries in CV units.  ``used_fallback`` is True
    when fewer than two minima survived and the frozen tercile partition was substituted.
    """
    grid = np.asarray(grid, float)
    F = np.asarray(F_ref, float)
    kT = 1.0 / beta

    mins = [i for i in range(1, len(F) - 1) if F[i] <= F[i - 1] and F[i] < F[i + 1]]
    if F[0] < F[1]:
        mins = [0] + mins
    if F[-1] < F[-2]:
        mins = mins + [len(F) - 1]

    changed = True
    while changed and len(mins) > 1:
        changed = False
        for a in range(len(mins) - 1):
            i, j = mins[a], mins[a + 1]
            barrier = F[i:j + 1].max() - max(F[i], F[j])
            if barrier < merge_barrier_kT * kT:
                drop = a + 1 if F[j] > F[i] else a
                mins.pop(drop)
                changed = True
                break

    if len(mins) >= 2:
        edges = [grid[0]]
        for a in range(len(mins) - 1):
            i, j = mins[a], mins[a + 1]
            edges.append(float(grid[i + int(np.argmax(F[i:j + 1]))]))
        edges.append(grid[-1])
        return np.array(edges), np.array(mins), False

    # Amendment 3 fallback: equal-width terciles, declared as a partition, not as metastability.
    edges = np.linspace(grid[0], grid[-1], 4)
    return edges, np.array(mins), True


def assign_states(xi, edges):
    """Map CV values to state indices ``0..K-1``; values outside are clamped to the end states."""
    return np.clip(np.digitize(np.asarray(xi), np.asarray(edges)[1:-1]), 0, len(edges) - 2)


def bias_aware_target(grid, F_ref, B_t, beta, edges):
    """``Q*_k(t)`` — §2.1.  ``B_t`` is ``(T, n_grid)`` or ``(n_grid,)``; returns ``(T, K)``.

    A state can be rare under the unbiased ensemble and perfectly populated under the bias ABF
    has already learned, so the establishment gate must compare against this and not against the
    equilibrium mass.
    """
    grid = np.asarray(grid, float)
    F = np.asarray(F_ref, float)
    B = np.atleast_2d(np.asarray(B_t, float))
    dz = float(grid[1] - grid[0])
    sid = assign_states(grid, edges)
    K = len(edges) - 1

    logw = -beta * (F[None, :] - B)
    logw = logw - logw.max(axis=1, keepdims=True)
    w = np.exp(logw) * dz
    out = np.zeros((B.shape[0], K))
    for k in range(K):
        out[:, k] = w[:, sid == k].sum(axis=1)
    return out / out.sum(axis=1, keepdims=True).clip(1e-300)


def occupancy(xi_trace, edges):
    """Fraction of walkers in each state over time.  ``xi_trace`` ``(T, R, N)`` -> ``(T, R, K)``."""
    xi = np.asarray(xi_trace)
    K = len(edges) - 1
    sid = assign_states(xi, edges)
    T, R, N = xi.shape
    out = np.zeros((T, R, K))
    for k in range(K):
        out[..., k] = (sid == k).sum(axis=-1) / N
    return out


def hitting_times(xi_trace, steps, edges):
    """First step at which each state is occupied by any walker.  ``(R, K)``; ``-1`` = never."""
    xi = np.asarray(xi_trace)
    steps = np.asarray(steps)
    K = len(edges) - 1
    sid = assign_states(xi, edges)
    T, R, N = xi.shape
    out = np.full((R, K), -1, dtype=np.int64)
    for k in range(K):
        seen = (sid == k).any(axis=-1)                      # (T, R)
        for r in range(R):
            w = np.flatnonzero(seen[:, r])
            if w.size:
                out[r, k] = steps[w[0]]
    return out


def _longest_true_run(mask):
    """Length of the longest contiguous run of True in a 1-D boolean array."""
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def classify(xi_trace, steps, edges, Q_star, n_steps,
             discovery_fraction=DISCOVERY_FRACTION, min_seeds=DISCOVERY_MIN_SEEDS,
             deficit=ESTABLISHMENT_DEFICIT, span=ESTABLISHMENT_SPAN):
    """Apply Gates B and C.  Returns a verdict dict.

    ``Q_star`` is ``(T, K)`` aligned with ``steps``.  The regime is decided by these numbers
    alone — no mFR result is an input, and none exists when this runs on the screen.
    """
    occ = occupancy(xi_trace, edges)                        # (T, R, K)
    T, R, K = occ.shape
    hits = hitting_times(xi_trace, steps, edges)            # (R, K)

    # ---- Gate B: discovery ----
    thresh = discovery_fraction * n_steps
    discovered = (hits >= 0) & (hits < thresh)              # (R, K)
    seeds_ok = discovered.sum(axis=0)                       # (K,)
    gate_b = bool((seeds_ok >= min_seeds).all())

    # ---- Gate C: establishment, second half only ----
    half = T // 2
    steps_arr = np.asarray(steps)
    dt_frac = (steps_arr[-1] - steps_arr[0]) / max(T - 1, 1) / max(n_steps, 1)
    need = int(np.ceil(span / max(dt_frac, 1e-12)))
    Q = np.asarray(Q_star, float)
    Q = Q[None, :] if Q.ndim == 1 else Q                    # (T, K)
    Qh = Q[half:][:, None, :]                               # (T', 1, K) -> broadcasts over seeds
    under = occ[half:] < deficit * Qh
    runs = np.zeros((R, K), dtype=int)
    for r in range(R):
        for k in range(K):
            runs[r, k] = _longest_true_run(under[:, r, k])
    persistent = runs >= need
    gate_c = bool(persistent.any())

    if not gate_b:
        regime = "discovery-limited"
    elif gate_c:
        regime = "establishment-limited"
    else:
        regime = "ABF-sufficient"

    return dict(
        regime=regime, gate_b_discovery=gate_b, gate_c_establishment=gate_c,
        n_states=K, edges=edges.tolist(),
        hitting_steps=hits.tolist(), seeds_discovered_per_state=seeds_ok.tolist(),
        discovery_threshold_steps=float(thresh),
        required_contiguous_points=int(need),
        longest_deficit_run=runs.tolist(),
        worst_second_half_relative_deficit=float(
            np.nanmin(occ[half:] / np.clip(Qh, 1e-12, None))),
        mean_occupancy_second_half=occ[half:].mean(axis=0).tolist(),
        licenses_mfr=bool(regime == "establishment-limited"))
