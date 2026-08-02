"""Reusable screening diagnostics for marginal-Fisher-Rao (mFR) gates.

Three checks that the valine screen got wrong, corrected here once so no later study
inherits them.  None of these corrections changes the valine verdict -- V3 failed on the
under-establishment condition, by a wide margin, and these touch conditions 4 and 5 and a
bookkeeping column.  They are fixed because the *machinery* is meant to be reused.

1. ``matched_cell_conditional`` -- the omitted-coordinate check.  Comparing
   ``p_run(omitted | region)`` against ``p_ref(omitted | region)`` is an **unpaired
   comparison wearing a paired comparison's clothes**: ABF deliberately flattens *within* a
   region while the reference is Boltzmann-weighted inside it, so the two weight the
   region's interior differently and the statistic is non-zero even when the conditional at
   every fixed CV cell is identical.  The fix compares cell by cell and aggregates with
   **common weights**.

2. ``corridor_aware_entries`` -- state-entry counting.  A counter that only credits
   transitions between *consecutively labelled* frames reports zero entries into any state
   reachable only across an unlabelled high-energy corridor, which is precisely the state a
   screen cares about.  The fix carries the last labelled state across the corridor.

3. ``bias_aware_region_target`` -- the establishment target.  It must be normalised on the
   reference's support, with the observed fractions conditioned the same way, and that must
   be **asserted rather than printed**.  A diagnostic that only prints is a diagnostic that
   gets skimmed; the valine screen's first target put 97 % of its mass in cells the
   reference had never sampled and printed a number saying so.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


# ---------------------------------------------------------------------------
# 1. omitted-coordinate conditional, compared at matched CV cells
# ---------------------------------------------------------------------------
def per_cell_conditional(cells, values, weights, n_cells, edges):
    """Reference conditional ``p(value | cell)`` for every CV cell.

    Returns ``(hist, counts, ess)`` with ``hist`` of shape ``(n_cells, n_bins)``, rows
    normalised where usable and zero elsewhere.  ``ess`` is the Kish effective sample size
    of the weights in each cell: with importance weights a cell can hold many samples and
    still carry almost no information, and a raw count would hide that.
    """
    n_bins = len(edges) - 1
    b = np.clip(np.digitize(values, edges) - 1, 0, n_bins - 1)
    flat = cells.astype(np.int64) * n_bins + b
    hist = np.bincount(flat, weights=weights, minlength=n_cells * n_bins)
    hist = hist.reshape(n_cells, n_bins)
    w1 = np.bincount(cells.astype(np.int64), weights=weights, minlength=n_cells)
    w2 = np.bincount(cells.astype(np.int64), weights=weights ** 2, minlength=n_cells)
    counts = np.bincount(cells.astype(np.int64), minlength=n_cells).astype(float)
    ess = np.where(w2 > 0, w1 ** 2 / np.maximum(w2, EPS), 0.0)
    row = hist.sum(axis=1, keepdims=True)
    hist = np.where(row > 0, hist / np.maximum(row, EPS), 0.0)
    return hist, counts, ess


def matched_cell_conditional(cell_hist, cell_ess, cell_region, cell_weight,
                             obs_hist_by_region, n_regions, min_cell_ess=20.0):
    """Compare an omitted coordinate region by region using **common cell weights**.

    Parameters
    ----------
    cell_hist : (n_cells, n_bins)
        Reference conditional ``p_ref(omitted | cell)``, from :func:`per_cell_conditional`.
    cell_ess : (n_cells,)
        Reference effective sample size per cell; cells below ``min_cell_ess`` are unusable
        and their weight is reported as dropped rather than silently redistributed.
    cell_region : (n_cells,)
        Region label per cell, ``-1`` outside every region.
    cell_weight : (n_cells,)
        The **common** weight applied to both sides.  Supply the run's own empirical cell
        occupancy when it was recorded; otherwise the bias-aware occupancy
        ``propto exp(-beta (F_ref - B_t))``, which is what an ABF run's interior weighting
        converges to.  Either way both sides are aggregated with the *same* numbers, which
        is the entire point.
    obs_hist_by_region : (n_regions, n_bins)
        Observed (run) histogram of the omitted coordinate per region, unnormalised.

    Returns a dict with, per region, the matched TV, the *unmatched* region-aggregated TV
    that the old check computed, and the weight that had to be dropped for want of a usable
    reference conditional.  A large ``dropped_weight`` invalidates that region's number and
    is returned so a caller can gate on it.
    """
    n_bins = cell_hist.shape[1]
    usable = cell_ess >= min_cell_ess
    out = {"per_region": [], "min_cell_ess": float(min_cell_ess)}
    worst_matched, worst_unmatched = None, None
    for k in range(n_regions):
        in_k = cell_region == k
        w = np.where(in_k, cell_weight, 0.0)
        tot = w.sum()
        obs = obs_hist_by_region[k].astype(float)
        rec = {"region": int(k), "total_weight": float(tot),
               "n_obs": float(obs.sum()), "tv_matched": None, "tv_unmatched": None,
               "dropped_weight": None, "n_cells_used": 0}
        if tot <= 0 or obs.sum() <= 0:
            out["per_region"].append(rec)
            continue
        wk = np.where(usable, w, 0.0)
        rec["dropped_weight"] = float((tot - wk.sum()) / tot)
        rec["n_cells_used"] = int((wk > 0).sum())
        obs_p = obs / obs.sum()
        if wk.sum() > 0:
            ref_matched = (cell_hist * wk[:, None]).sum(axis=0) / wk.sum()
            tv = 0.5 * np.abs(ref_matched - obs_p).sum()
            rec["tv_matched"] = float(tv)
            worst_matched = tv if worst_matched is None else max(worst_matched, tv)
        # the OLD statistic: the reference's own (Boltzmann) interior weighting, which is
        # not the run's.  Kept so the size of the confound is a measured number.
        wu = np.where(in_k & usable, cell_ess, 0.0)
        if wu.sum() > 0:
            ref_unmatched = (cell_hist * wu[:, None]).sum(axis=0) / wu.sum()
            tvu = 0.5 * np.abs(ref_unmatched - obs_p).sum()
            rec["tv_unmatched"] = float(tvu)
            worst_unmatched = tvu if worst_unmatched is None else max(worst_unmatched, tvu)
        out["per_region"].append(rec)
    out["worst_tv_matched"] = worst_matched
    out["worst_tv_unmatched"] = worst_unmatched
    out["n_bins"] = int(n_bins)
    return out


# ---------------------------------------------------------------------------
# 2. corridor-aware state entries
# ---------------------------------------------------------------------------
def corridor_aware_entries(labels, n_states, min_dwell=2):
    """Count persistent state entries across unlabelled corridors.

    ``labels`` is ``(T, R, N)`` with ``-1`` for "above the region ceiling" -- the
    high-energy corridor a walker must cross to reach a far state.  The naive counter
    credits a transition only when two *consecutive* frames carry different labels, so a
    walker that leaves state A, spends any time unlabelled, and arrives in state B
    contributes nothing.  Every state behind a real barrier is then reported as never
    entered, which is exactly backwards: the higher the barrier, the more certainly the
    counter reads zero.

    Here the last *labelled* state is carried across the corridor, and an entry is credited
    when a walker becomes persistently labelled (``min_dwell`` consecutive frames) in a
    different state.  The dwell requirement is what stops a walker brushing a boundary for
    one save from counting as an entry.

    Returns ``entries (R, n_states)``, ``trans (R, n_states, n_states)`` (last labelled
    state -> new state, first arrivals excluded), and ``first_entry (R, n_states)`` in save
    indices, ``-1`` where never entered.
    """
    labels = np.asarray(labels)
    T, R, N = labels.shape
    flat = labels.reshape(T, R * N)
    conf = flat >= 0
    for d in range(1, min_dwell):
        conf[:T - d] &= (flat[d:] == flat[:T - d])
    if min_dwell > 1:
        conf[T - min_dwell + 1:] = False          # not enough look-ahead to confirm

    run_idx = np.repeat(np.arange(R), N)
    last = np.full(R * N, -1, dtype=np.int64)
    entries = np.zeros((R, n_states), dtype=np.int64)
    trans = np.zeros((R, n_states, n_states), dtype=np.int64)
    first = np.full((R, n_states), -1, dtype=np.int64)
    for t in range(T):
        c = conf[t]
        lab = flat[t]
        new = c & (lab != last)
        if new.any():
            ri, lb = run_idx[new], lab[new].astype(np.int64)
            np.add.at(entries, (ri, lb), 1)
            fresh = first[ri, lb] < 0
            if fresh.any():
                first[ri[fresh], lb[fresh]] = t
            prev = last[new]
            real = prev >= 0
            if real.any():
                np.add.at(trans, (ri[real], prev[real], lb[real]), 1)
        last = np.where(c, lab, last)
    return entries, trans, first


# ---------------------------------------------------------------------------
# 3. bias-aware establishment target, on the reference's support
# ---------------------------------------------------------------------------
def bias_aware_region_target(F_ref, B_t, label, beta, n_regions=None, atol=1e-9):
    """``Q*_k``: the ideal biased population of each region under the current bias.

    ``q*(z) propto exp(-beta (F_ref(z) - B_t(z)))``, normalised over the **reference's
    labelled support** and integrated over each region.  Scoring against the *unbiased*
    population would flag a state as starved exactly when ABF has correctly flattened it,
    manufacturing the signal mFR is meant to remove.

    ``F_ref`` may be non-finite where the reference has no support; those cells are
    excluded, and :func:`assert_supported_target` then requires that the exclusion left no
    mass behind.  ``B_t`` is ``(..., *grid)``; ``label`` is the grid-shaped region map with
    ``-1`` outside every region.
    """
    label = np.asarray(label)
    K = int(label.max()) + 1 if n_regions is None else int(n_regions)
    inside = (label >= 0) & np.isfinite(F_ref)
    B = np.asarray(B_t, dtype=float)
    lead = B.shape[:B.ndim - label.ndim]
    Bf = B.reshape(-1, *label.shape)
    Q = np.zeros((Bf.shape[0], K))
    for i in range(Bf.shape[0]):
        lg = np.where(inside, -beta * (np.where(inside, F_ref, 0.0) - Bf[i]), -np.inf)
        lg = lg - lg[inside].max()
        q = np.where(inside, np.exp(lg), 0.0)
        q /= q.sum()
        for k in range(K):
            Q[i, k] = q[label == k].sum()
    assert_supported_target(Q, atol=atol)
    return Q.reshape(*lead, K)


def assert_supported_target(Q, atol=1e-9):
    """HARD guard: after conditioning, reference-unsupported target mass must be **zero**.

    Not "small", and not "printed".  The failure this exists to stop is silent and
    confident: cap the unsampled cells at some large free energy, normalise over the whole
    domain, and the target concentrates wherever the reference is least trustworthy --
    because ABF flattens, so the bias grows large exactly there and ``exp(-beta(F - B))``
    explodes.  The valine screen shipped that version, printed a diagnostic saying 97 % of
    the target mass had landed in those cells, and it was caught by a human reading the
    log.  An assertion does not depend on someone reading the log.
    """
    tot = np.asarray(Q).sum(axis=-1)
    worst = float(np.max(np.abs(tot - 1.0)))
    if not worst < atol:
        raise AssertionError(
            f"bias-aware region target does not sum to 1 on the reference-supported "
            f"domain (worst |sum_k Q_k - 1| = {worst:.3e} > {atol:g}). Either a region is "
            f"missing from the decomposition or the target was normalised over cells the "
            f"reference does not support.")
    return worst


def assert_matched_conditioning(P, Q, atol=1e-9):
    """Observed fractions and target must live on the same support.

    ``P`` is compared against ``Q`` region by region, so if ``P`` is conditioned on the
    whole domain while ``Q`` is conditioned on the labelled support, every region reads
    deficient by the same factor and the deficit is an artifact of the conditioning.
    """
    sp = float(np.max(np.abs(np.asarray(P).sum(axis=-1) - 1.0)))
    sq = float(np.max(np.abs(np.asarray(Q).sum(axis=-1) - 1.0)))
    if not (sp < atol and sq < atol):
        raise AssertionError(
            f"observed and target fractions are not conditioned on the same support "
            f"(worst |sum P - 1| = {sp:.3e}, |sum Q - 1| = {sq:.3e}). Condition the "
            f"observed fractions on the reference's labelled support before comparing.")
    return sp, sq
