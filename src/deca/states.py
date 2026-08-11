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


#: Amendment 6: structural labels carrying at least this share of reference weight are
#: eligible for the corroboration gate.  Frozen from the reference alone, before the corrected
#: screen was analysed.
STRUCTURAL_WEIGHT_FLOOR = 1.0e-3
#: The eligible set that floor produced on the accepted reference.  Recorded explicitly so it
#: cannot be silently recomputed against a screen result.
ELIGIBLE_LABELS = (0, 1, 2, 3, 4, 5, 6, 8)


def reference_joint(xi_ref, y_ref, w_ref, grid, n_labels=9):
    """``p_ref(xi, y)`` on the evaluation grid, from MBAR-weighted reference samples.

    Out-of-domain samples are dropped, never clamped -- the defect that carved a fake well into
    the reference PMF.  Returns ``(joint (n_labels, n_grid), label_weight (n_labels,))``.
    """
    grid = np.asarray(grid, float)
    dz = float(grid[1] - grid[0])
    lo, hi = grid[0] - 0.5 * dz, grid[-1] + 0.5 * dz
    xi = np.asarray(xi_ref, float).ravel()
    y = np.asarray(y_ref).ravel().astype(int)
    w = np.asarray(w_ref, float).ravel()
    inside = (xi >= lo) & (xi <= hi)
    xi, y, w = xi[inside], y[inside], w[inside]
    idx = np.clip(((xi - lo) / dz).astype(int), 0, grid.size - 1)

    joint = np.zeros((n_labels, grid.size))
    for a in range(n_labels):
        m = y == a
        if m.any():
            np.add.at(joint[a], idx[m], w[m])
    tot = joint.sum()
    joint = joint / max(tot, 1e-300)
    return joint, joint.sum(axis=1)


def bias_aware_structural_target(joint, B_t, beta, eligible=ELIGIBLE_LABELS):
    """``Q*_y(t)`` — Amendment 6.  ``B_t`` is ``(T, n_grid)`` or ``(n_grid,)``; returns ``(T, L)``.

    ``Q*_y  proportional to  integral p_ref(xi, y) exp(beta B_t(xi)) dxi``

    the exact structural analogue of :func:`bias_aware_target`: the bias depends only on ``xi``,
    so it reweights the joint pointwise in ``xi``.  Normalised over the **eligible** labels only,
    so an excluded sliver cannot absorb probability mass.
    """
    B = np.atleast_2d(np.asarray(B_t, float))
    e = np.asarray(eligible, int)
    logw = beta * B                                     # (T, n_grid)
    logw = logw - logw.max(axis=1, keepdims=True)
    w = np.exp(logw)
    num = w @ joint[e].T                                # (T, L)
    return num / num.sum(axis=1, keepdims=True).clip(1e-300)


def structural_occupancy(label_y, eligible=ELIGIBLE_LABELS):
    """Fraction of walkers carrying each eligible label.  ``(T, R, N)`` -> ``(T, R, L)``.

    Renormalised over the eligible labels so it is comparable with
    :func:`bias_aware_structural_target`, which is normalised the same way.
    """
    y = np.asarray(label_y).astype(int)
    T, R, N = y.shape
    out = np.zeros((T, R, len(eligible)))
    for i, a in enumerate(eligible):
        out[..., i] = (y == a).sum(axis=-1)
    tot = out.sum(axis=-1, keepdims=True)
    return np.divide(out, tot, out=np.zeros_like(out), where=tot > 0)


def structural_establishment(label_y, steps, joint, B_t, beta, n_steps,
                             eligible=ELIGIBLE_LABELS, deficit=ESTABLISHMENT_DEFICIT,
                             span=ESTABLISHMENT_SPAN):
    """Amendment 6 corroboration gate.  Returns a dict; ``any_deficit`` is the licensing bit."""
    occ = structural_occupancy(label_y, eligible)                # (T, R, L)
    Q = bias_aware_structural_target(joint, B_t, beta, eligible)  # (T_B, L)
    T, R, L = occ.shape
    if Q.shape[0] == 1:
        Q = np.repeat(Q, T, axis=0)                  # a time-constant bias applies at every t
    elif Q.shape[0] != T:
        raise ValueError(f"bias has {Q.shape[0]} time points but occupancy has {T}; "
                         "align B_t with the label trace before calling")
    half = T // 2
    steps = np.asarray(steps)
    dt_frac = (steps[-1] - steps[0]) / max(T - 1, 1) / max(n_steps, 1)
    need = int(np.ceil(span / max(dt_frac, 1e-12)))

    Qh = Q[half:][:, None, :]
    under = occ[half:] < deficit * Qh
    runs = np.zeros((R, L), dtype=int)
    for r in range(R):
        for k in range(L):
            runs[r, k] = _longest_true_run(under[:, r, k])
    persistent = runs >= need
    ratio = occ[half:] / np.clip(Qh, 1e-12, None)
    return dict(
        eligible_labels=list(map(int, eligible)),
        required_contiguous_points=int(need),
        longest_deficit_run=runs.tolist(),
        labels_with_persistent_deficit=[int(eligible[k]) for k in range(L)
                                        if bool(persistent[:, k].any())],
        seeds_with_deficit_per_label=persistent.sum(axis=0).tolist(),
        mean_second_half_occupancy=occ[half:].mean(axis=(0, 1)).tolist(),
        mean_second_half_target=Q[half:].mean(axis=0).tolist(),
        min_ratio_per_label=np.nanmin(ratio, axis=(0, 1)).tolist(),
        any_deficit=bool(persistent.any()))


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
