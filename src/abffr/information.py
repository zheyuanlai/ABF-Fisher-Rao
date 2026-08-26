"""Online local difficulty ``Gamma_hat(z)``: the only landscape input to ``r``.

Frozen protocol: ``docs/QR_DECOUPLING_PREREGISTRATION.md``.

``Gamma(z) = 2 int_0^inf Cov(f(X_0), f(X_s) | xi = z) ds`` is the asymptotic
variance of the local mean-force observation: conditional force variance times
integrated autocorrelation.  It is what makes one cell statistically expensive
and another cheap, and unlike the free energy it is *not* recoverable from the
landscape -- the campaign's kappa-family varies it 16x at fixed ``F``.

It is estimated by batch means over the same force observations that feed the
ABF accumulator, which fixes two things at once.

**The eligibility rule closes a feedback loop.**  Sibling replicas are correlated,
so they inflate the measured variance of a cell; if that inflation fed the
allocation, a cell that received clones would measure as harder, receive more
clones, and measure as harder still.  That is the shape of the v3 oracle-flip
failure -- an estimate steering the allocation that produces it.  Because clones
are held out of the accumulator until rejuvenated, and this estimator reads the
accumulator's own eligible stream, the loop is bounded by the same
``eps_gene`` that sets the hold.

**The block length is a measurement, not a guess.**  Batch means are biased low
when the block is not long enough relative to ``tau_int``, and they are biased
low *hardest exactly where difficulty is highest* -- the anti-detection failure
mode for this campaign.  :func:`block_length_adequacy` reports the ratio the
Stage-1 validation must clear before any allocation arm runs.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

#: Shrinkage toward the pooled difficulty.  With B blocks the relative error of a
#: variance is ~sqrt(2/(B-1)) -- 47% at B = 10 -- so an unshrunk Gamma_hat would
#: hand the allocator noise shaped like signal.  Frozen, not tuned per system.
SHRINK_WEIGHT = 0.3


@dataclass
class BlockAccumulator:
    """Per-cell block means of the eligible mean-force observations."""

    n_cells: int
    n_blocks: int = 10
    _sum: np.ndarray = field(init=False)
    _cnt: np.ndarray = field(init=False)
    _means: List[np.ndarray] = field(init=False, default_factory=list)
    _counts: List[np.ndarray] = field(init=False, default_factory=list)

    def __post_init__(self):
        self._sum = np.zeros(self.n_cells, dtype=float)
        self._cnt = np.zeros(self.n_cells, dtype=float)

    def observe(self, cell_index: np.ndarray, force: np.ndarray,
                eligible: Optional[np.ndarray] = None) -> None:
        """Add one step's observations.  ``eligible`` excludes held-out clones."""
        cell_index = np.asarray(cell_index, dtype=int)
        force = np.asarray(force, dtype=float)
        if eligible is not None:
            keep = np.asarray(eligible, dtype=bool)
            cell_index, force = cell_index[keep], force[keep]
        np.add.at(self._sum, cell_index, force)
        np.add.at(self._cnt, cell_index, 1.0)

    def close_block(self) -> None:
        """Seal the current block and start the next; keeps the last ``n_blocks``."""
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(self._cnt > 0, self._sum / np.maximum(self._cnt, 1e-300),
                            np.nan)
        self._means.append(mean)
        self._counts.append(self._cnt.copy())
        if len(self._means) > self.n_blocks:
            self._means.pop(0)
            self._counts.pop(0)
        self._sum[:] = 0.0
        self._cnt[:] = 0.0

    @property
    def n_closed(self) -> int:
        return len(self._means)


def gamma_hat(acc: BlockAccumulator, min_blocks: int = 4,
              shrink: float = SHRINK_WEIGHT) -> np.ndarray:
    """Batch-means asymptotic variance per cell, shrunk toward the pooled value.

    ``Gamma_j = mean_b(N_{j,b}) * Var_b(fbar_{j,b})``.  Cells with too few sealed
    blocks fall back to the pooled estimate rather than to zero: zero difficulty
    would read as "allocate nothing here", which is the opposite of what "we have
    not measured this cell yet" should mean.
    """
    if acc.n_closed < 2:
        return np.ones(acc.n_cells, dtype=float)
    M = np.vstack(acc._means)                      # (B, J)
    N = np.vstack(acc._counts)
    valid = np.isfinite(M)
    n_valid = valid.sum(axis=0)

    # A cell with fewer than two sealed blocks has no variance to report; it is
    # masked out below, so the empty-slice warning here is expected, not a symptom.
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        var = np.nanvar(np.where(valid, M, np.nan), axis=0, ddof=1)
        nbar = np.nansum(np.where(valid, N, np.nan), axis=0) / np.maximum(n_valid, 1)
    raw = np.where(n_valid >= min_blocks, var * nbar, np.nan)

    ok = np.isfinite(raw) & (raw > 0)
    pooled = float(np.exp(np.mean(np.log(raw[ok])))) if ok.any() else 1.0
    out = np.where(ok, raw, pooled)
    s = float(np.clip(shrink, 0.0, 1.0))
    return (1.0 - s) * out + s * pooled


def tau_hat(acc: BlockAccumulator, gamma: np.ndarray,
            var_within: np.ndarray) -> np.ndarray:
    """``tau_j = Gamma_j / sigma_j^2`` -- integrated autocorrelation, in time units.

    Feeds :func:`abffr.balanced_representation.rejuvenation_steps`, so a slow cell
    holds its clones longer than a fast one.
    """
    gamma = np.asarray(gamma, dtype=float)
    var_within = np.asarray(var_within, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        tau = np.where(var_within > 0, gamma / var_within, np.nan)
    finite = np.isfinite(tau) & (tau > 0)
    fallback = float(np.median(tau[finite])) if finite.any() else 1.0
    return np.where(finite, tau, fallback)


def block_length_adequacy(block_steps: int, dt: float,
                          tau: np.ndarray) -> Dict[str, float]:
    """``block_length / tau_j`` per cell -- the Stage-1B gate on the estimator.

    Batch means need the block to be long relative to the correlation time.  This
    campaign builds cells whose ``tau`` differs by 16x by construction, so a block
    length chosen once for the fast cells silently under-measures the slow ones --
    and under-measuring difficulty where difficulty is greatest would make the
    allocator blind to precisely the signal it exists to follow.
    """
    tau = np.asarray(tau, dtype=float)
    length = float(block_steps) * float(dt)
    ratio = length / np.maximum(tau, 1e-300)
    return {
        "block_time": length,
        "ratio_min": float(np.min(ratio)),
        "ratio_median": float(np.median(ratio)),
        "tau_max": float(np.max(tau)),
        "n_cells_below_10": int(np.sum(ratio < 10.0)),
    }
