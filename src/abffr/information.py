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


# --------------------------------------------------------------------------- #
# The sample-efficient estimator that the measured tau forced
# --------------------------------------------------------------------------- #
#
# Measured on the campaign's own potential (fixed x, hidden-coordinate channel
# isolated): tau at the slow end of a 16x kappa cell is ~4.7 time units at fixed
# x, and ~1.2 after x-motion turns the cell population over.  Batch means need a
# block much longer than tau AND B blocks of it, so B = 10 blocks of 10 tau is
# 60000 steps against a 50000-step run.  The frozen W = 5000 / B = 10 budget is
# short by more than an order of magnitude, and short in the direction that
# hides difficulty rather than inventing it.
#
# The decomposition Gamma = sigma^2 tau is far cheaper than the product is,
# because the two factors need different amounts of data:
#
#   sigma^2  is an *instantaneous* spread across the replicas in a cell.  It
#            needs no window at all, and its error does not grow with tau.
#   tau      is a *shape* parameter of the autocorrelation.  Fitting a decay
#            needs a few tau of series; summing one, as batch means implicitly
#            does, needs tens of tau times B.
#
# So this estimator buys roughly an order of magnitude in window at the cost of
# one assumption -- that the slow mode is approximately a single exponential,
# which is what a relaxing hidden coordinate is.  Stage 1B validates it against
# the long-run Gamma_ref exactly as it would validate batch means; if the acf is
# not close to exponential the fit reports poor quality and the cell falls back.


def conditional_force_variance(cell_index: np.ndarray, force: np.ndarray,
                               mean_force_at: np.ndarray, n_cells: int,
                               eligible: Optional[np.ndarray] = None) -> np.ndarray:
    """``sigma^2_j`` from the residual spread of eligible replicas in each cell.

    The residual is taken against the current ``Fhat'`` *at each replica's own
    position*, not against the cell mean: a cell is 0.1875 wide and the mean
    force varies across it, so a raw within-cell variance would charge that
    systematic variation to the noise and inflate difficulty wherever ``F'`` is
    steep -- which is a landscape feature, not a statistical one.
    """
    cell_index = np.asarray(cell_index, dtype=int)
    resid = np.asarray(force, dtype=float) - np.asarray(mean_force_at, dtype=float)
    if eligible is not None:
        keep = np.asarray(eligible, dtype=bool)
        cell_index, resid = cell_index[keep], resid[keep]
    s2 = np.zeros(int(n_cells))
    cnt = np.zeros(int(n_cells))
    np.add.at(s2, cell_index, resid ** 2)
    np.add.at(cnt, cell_index, 1.0)
    out = np.where(cnt > 1, s2 / np.maximum(cnt - 1.0, 1.0), np.nan)
    ok = np.isfinite(out) & (out > 0)
    return np.where(ok, out, float(np.median(out[ok])) if ok.any() else 1.0)


@dataclass
class MeanForceHistory:
    """Ring buffer of per-cell mean force, for the autocorrelation fit."""

    n_cells: int
    capacity: int = 400

    def __post_init__(self):
        self._buf: List[np.ndarray] = []

    def push(self, cell_index: np.ndarray, force: np.ndarray,
             eligible: Optional[np.ndarray] = None) -> None:
        cell_index = np.asarray(cell_index, dtype=int)
        force = np.asarray(force, dtype=float)
        if eligible is not None:
            keep = np.asarray(eligible, dtype=bool)
            cell_index, force = cell_index[keep], force[keep]
        s = np.zeros(self.n_cells)
        c = np.zeros(self.n_cells)
        np.add.at(s, cell_index, force)
        np.add.at(c, cell_index, 1.0)
        self._buf.append(np.where(c > 0, s / np.maximum(c, 1e-300), np.nan))
        if len(self._buf) > self.capacity:
            self._buf.pop(0)

    @property
    def n_samples(self) -> int:
        return len(self._buf)

    def series(self) -> np.ndarray:
        return np.vstack(self._buf) if self._buf else np.zeros((0, self.n_cells))


def tau_from_lag1(hist: MeanForceHistory, obs_interval: float,
                  min_samples: int = 40) -> np.ndarray:
    """Integrated autocorrelation time per cell, by bias-corrected AR(1) fit.

    The obvious estimator -- fit ``log rho_k`` against lag -- fails at this
    campaign's budget, and it was worth measuring why.  The sample
    autocorrelation carries a downward bias of about ``2 tau / n`` that is
    roughly *constant in the lag*, so for a slow cell (``tau = 48`` observations
    from ``n = 300``) it subtracts ~0.32 from every ``rho_k``.  That does not
    merely add noise: it steepens the fitted decay, and the estimator reports
    the hard cell as easy.  A 16x spread came back as 5.8x.

    For a relaxing hidden coordinate the slow mode is a single exponential, and
    for that model the lag-1 regression is the maximum-likelihood estimator --
    it uses every sample once instead of spending the series on high lags where
    the bias dominates the signal.  Kendall's correction
    ``phi + (1 + 3 phi) / n`` removes the known small-sample bias of the OLS
    coefficient, which is the same order as the effect being measured.

    Returns NaN where the fit is not usable, never a small number: "we could not
    measure this cell" and "this cell is easy" must not look alike to the
    allocator.
    """
    S = hist.series()
    if S.shape[0] < min_samples:
        return np.full(hist.n_cells, np.nan)
    out = np.full(hist.n_cells, np.nan)

    for j in range(hist.n_cells):
        v = S[:, j]
        v = v[np.isfinite(v)]
        n = v.size
        if n < min_samples:
            continue
        v = v - v.mean()
        denom = float(v[:-1] @ v[:-1])
        if denom <= 0:
            continue
        phi = float(v[1:] @ v[:-1]) / denom
        phi = phi + (1.0 + 3.0 * phi) / n          # Kendall small-sample bias
        if not (0.0 < phi < 1.0):
            continue                                # white noise, or unresolved
        out[j] = (-1.0 / np.log(phi)) * float(obs_interval)
    return out


def gamma_hat_decomposed(sigma2: np.ndarray, tau: np.ndarray,
                         shrink: float = SHRINK_WEIGHT) -> np.ndarray:
    """``Gamma_j = sigma^2_j tau_j``, with unmeasured ``tau`` filled by median.

    Only *ratios* across cells reach the allocator (``r ∝ sqrt(a Gamma)`` is
    scale-free), so the factor of 2 in the definition of the asymptotic variance
    is deliberately not carried here -- a global constant that cannot change any
    allocation is a constant that should not be able to introduce a units bug.
    """
    sigma2 = np.asarray(sigma2, dtype=float)
    tau = np.asarray(tau, dtype=float)
    ok = np.isfinite(tau) & (tau > 0)
    tau = np.where(ok, tau, float(np.median(tau[ok])) if ok.any() else 1.0)
    raw = np.maximum(sigma2, 0.0) * tau
    good = np.isfinite(raw) & (raw > 0)
    pooled = float(np.exp(np.mean(np.log(raw[good])))) if good.any() else 1.0
    out = np.where(good, raw, pooled)
    s = float(np.clip(shrink, 0.0, 1.0))
    return (1.0 - s) * out + s * pooled
