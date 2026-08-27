"""IO-ABF: information-optimal replica allocation, held by the bias.

Frozen protocol: ``docs/IO_ABF_OVERNIGHT_PREREGISTRATION.md``.

This module is the *transfer layer* for the q-r campaign's result.  Stage 2 of
that campaign measured, on the kappa family, that ``r ∝ sqrt(a Gamma)`` held by
the bias accelerates plain ABF by 1.55-1.87x, and that no birth--death arm comes
close.  Nothing here re-derives that; the estimator, the floor and the
constrained solve are imported unchanged from :mod:`abffr.information`,
:mod:`abffr.allocation` and :mod:`abffr.cell_mass`.  What is new is that the
allocator no longer lives inside one 2-D engine: it takes a reaction coordinate,
a grid, an evaluation mask and a stream of mean-force observations, and returns a
force to add to the ABF bias.

Three arms, and only ``r`` differs::

    A0    no allocation at all -- the applied bias force is exactly ABF's
    A6b   r ∝ sqrt(a Gamma_hat),                     floored, bias-held
    A6c   r ∝ sqrt(a Gamma_hat + lam M^2),  ESS_M/K >= rho, floored, bias-held

The realisation is the bias, never birth--death::

    B_t(z) = A_hat_t(z) + beta^-1 log r*_t(z)
      =>    p_t(z) ∝ exp(-beta (F - A_hat)) r*_t(z)  ->  r*_t(z)

so the applied force gains one term, ``beta^-1 d/dz log r*``, and *nothing else
in the engine changes*.  No clone, no kill, no resampling, no weight.  The
mean-force accumulator is untouched because the extra term is a function of the
reaction coordinate alone, which is the same fibre-conditional invariance that
licenses ABF itself: the conditional law of the orthogonal degrees of freedom at
fixed ``z`` is unchanged, so ``E[dV/dz | z]`` is still ``F'(z)``.

Batched over runs
-----------------
The two streams that are touched every observation -- the residual second moment
and the per-cell mean-force history -- are accumulated in torch over all ``R``
runs at once.  Everything that runs only at an *opportunity* (the AR(1) fit, the
Fisher--Rao mass step, the constrained solve) calls the frozen numpy code
row-by-row, so the arithmetic is literally the code Stage 2 validated rather
than a re-implementation of it.

What this module may not read
-----------------------------
``a`` comes from grid geometry and the evaluation mask; ``Gamma_hat`` from
observed forces; ``q`` from the *running* ABF free energy.  No reference free
energy, no barrier location, no landscape parameter reaches any of them.
:func:`assert_no_reference_leakage` is the gate, and it is a source-level check
rather than a promise.
"""
from __future__ import annotations

import inspect
import io
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from . import allocation as al
from . import cell_mass as cm
from . import information as inf

#: The arms this module implements.  ``A0`` is "allocation absent" and is here so
#: a batch can carry it as a row rather than as a separate call: an A0 row must
#: come out of the same engine, the same noise and the same accumulator as its
#: candidates, or the pairing the endpoint relies on is not a pairing.
ARMS = ("A0", "A6b", "A6c")

#: Arms whose applied bias force gains the allocation term.
ALLOCATING = ("A6b", "A6c")

EPS = 1e-300


@dataclass(frozen=True)
class IOConfig:
    """Everything frozen before a run.  No field here is swept.

    ``n_cells`` follows the frozen rule ``J = min(32, floor(K / 8))`` so a cell
    holds ~8 replicas under uniform occupancy: ``sigma^2`` and ``tau`` cannot be
    guessed from one or two samples, and when the walker count is small it is
    the *cell count* that gives way, never the estimator.
    """

    n_cells: int
    obs_every: int
    opportunity_every: int
    burnin_fraction: float = 0.20
    stop_fraction: float = 0.80
    history_capacity: int = 600
    floor_fraction: float = al.FLOOR_FRACTION      # 0.25, shared by every arm
    rho: float = 0.5                                # A6c mass-ESS floor
    theta: float = 1.0                              # exact FR projection
    shrink: float = inf.SHRINK_WEIGHT               # 0.3
    tau_min_samples: int = 40

    def __post_init__(self):
        if self.n_cells < 2:
            raise ValueError("need at least two allocation cells")
        if not 0.0 <= self.burnin_fraction < self.stop_fraction <= 1.0:
            raise ValueError("need 0 <= burnin < stop <= 1")
        if self.stop_fraction >= 1.0:
            raise ValueError(
                "the allocation window must close strictly before the end of the "
                "run: the long-time limit belongs to ABF by construction")


def cells_for_walkers(n_particles: int) -> int:
    """``J = min(32, floor(K / 8))`` -- the frozen cell-count rule."""
    return int(max(2, min(32, int(n_particles) // 8)))


def cadence_for_run(n_steps: int, history_capacity: int = 600,
                    n_opportunities: int = 48,
                    window_fraction: float = 0.60,
                    history_span_fraction: float = 0.15,
                    obs_every_override: Optional[int] = None) -> Dict[str, int]:
    """Observation and opportunity cadences, as a pure function of ``n_steps``.

    Both are structural, so neither can become a per-system knob:

    * ``opportunity_every`` puts exactly ``n_opportunities`` refreshes inside the
      ``[burnin, stop]`` window whatever the run length is;
    * ``obs_every`` makes the mean-force history span ``history_span_fraction``
      of the run with ``history_capacity`` samples in it.

    ``obs_every_override`` is the one place a system may differ, and only through
    rule R-OBS: a *pre-run* A0-only probe measures the per-cell mean-force
    autocorrelation and sets the sampling interval near ``tau_med / 2``, because
    the AR(1) fit fails in opposite ways on both sides -- ``phi -> 0`` when the
    interval is long against ``tau`` (the cell reads as unresolved) and
    ``phi -> 1`` when it is short (the fitted ``phi`` crosses 1 on noise).  The
    override never reads a candidate arm and never reads a reference.
    """
    n_steps = int(n_steps)
    opp = max(1, int(round(window_fraction * n_steps / float(n_opportunities))))
    obs = (int(obs_every_override) if obs_every_override is not None
           else max(1, int(round(history_span_fraction * n_steps
                                 / float(history_capacity)))))
    return {"opportunity_every": opp, "obs_every": max(1, obs)}


def firing_steps(n_steps: int, cfg: IOConfig) -> np.ndarray:
    """Opportunity steps.  Same three-phase shape the q-r campaign froze."""
    burn = int(round(cfg.burnin_fraction * n_steps))
    stop = int(round(cfg.stop_fraction * n_steps))
    every = int(cfg.opportunity_every)
    return np.array([s for s in range(1, int(n_steps) + 1)
                     if burn <= s < stop and (s - burn) % every == 0], dtype=int)


# --------------------------------------------------------------------------- #
# G0.2: the allocation may not read the answer
# --------------------------------------------------------------------------- #
#: Names that would mean the allocator had been handed the thing it exists to
#: estimate.  Checked against the *source* of every function the allocation path
#: runs, so the gate cannot be satisfied by a promise in a docstring.
FORBIDDEN_TOKENS = ("F_ref", "Fp_ref", "q_ref", "barrier", "oracle", "x_barrier",
                    "omega_in", "kappa", "reference_profiles")


def _executable_source(obj) -> str:
    """Source with every comment and string literal removed.

    Prose may discuss the reference; code may not touch it.  A naive line filter
    keeps multi-line docstrings and so reports the module's own explanation of
    why it does not read ``F_ref`` as evidence that it does -- the gate must read
    what runs, which means tokenising rather than grepping.
    """
    import tokenize
    src = inspect.getsource(obj)
    out = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            out.append(tok.string)
    except (tokenize.TokenError, IndentationError):       # pragma: no cover
        return src
    return " ".join(out)


def assert_no_reference_leakage() -> None:
    """Fail if any module on the allocation path *executes* a reference name.

    G0.2.  Checked against tokenised source rather than against a promise in a
    docstring, and over the whole path -- the estimator, the allocator, the mass
    and the orchestration -- because a leak one module down is still a leak.
    """
    bad = []
    for mod in (al, inf, cm):
        code = _executable_source(mod)
        for tok in FORBIDDEN_TOKENS:
            if re.search(r"\b" + re.escape(tok) + r"\b", code):
                bad.append(f"{mod.__name__}: {tok}")
    for obj, label in ((IOAllocator, "IOAllocator"), (IOBatch, "IOBatch"),
                       (tau_from_series, "tau_from_series")):
        code = _executable_source(obj)
        for tok in FORBIDDEN_TOKENS:
            if re.search(r"\b" + re.escape(tok) + r"\b", code):
                bad.append(f"{label}: {tok}")
    if bad:
        raise AssertionError(
            "reference quantity reachable from the allocation path: "
            + ", ".join(sorted(set(bad))))


def assert_no_birth_death() -> None:
    """Fail if the IO path could move a replica.  G0.3.

    Two checks, because either alone is weak.  The token check rules out a
    resampler being *called*; the interface check rules out one being *served* --
    every public method is annotated to return a field, so there is no integer
    index for an engine to apply even if it wanted one.  ``torch.gather`` does
    appear, in :meth:`IOAllocator.bias_force_at`, where it reads the force
    *grid* at replica positions; that is an interpolation, and the pair of
    checks is what separates it from a permutation of the population.
    """
    src = " ".join(_executable_source(o) for o in
                   (IOAllocator, IOBatch, IOAllocator.refresh,
                    IOAllocator.observe, IOAllocator.bias_force_at))
    for tok in ("resample", "clone", "birth", "death", "multinomial", "kill",
                "choice", "permutation", "scatter_"):
        if re.search(r"\b" + re.escape(tok) + r"\b",
                     src.replace("scatter_add_", "scatteradd")):
            raise AssertionError(
                f"IO-ABF path names {tok!r}: it must not move a replica")
    for name in ("refresh", "bias_force_at"):
        ann = inspect.signature(getattr(IOAllocator, name)).return_annotation
        if ann not in (torch.Tensor, "torch.Tensor"):
            raise AssertionError(
                f"IOAllocator.{name} must return a field, not indices")


def assert_returns_field(allocator: "IOAllocator", X: torch.Tensor,
                         A_grid: torch.Tensor) -> None:
    """Runtime half of G0.3: what comes back is a float field, not an index set."""
    f = allocator.refresh(0, X, A_grid, record=False)
    if not torch.is_floating_point(f) or tuple(f.shape) != (allocator.R, allocator.G):
        raise AssertionError("refresh returned something that is not an (R, G) field")
    b = allocator.bias_force_at(X)
    if not torch.is_floating_point(b) or b.shape != X.shape:
        raise AssertionError("bias_force_at returned something that is not a force")


# --------------------------------------------------------------------------- #
# the batched observation streams
# --------------------------------------------------------------------------- #
def _numpy_median_rowwise(v: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Row-wise median over ``valid`` entries, with numpy's even-length convention.

    ``torch.nanmedian`` returns the *lower* of the two middle values where
    ``numpy.median`` averages them.  The difference is one fallback value in one
    unmeasured cell, which is exactly the size of thing that makes a batched
    re-implementation quietly stop being the frozen estimator.  This is not a
    second estimator; it is the same one, evaluated the same way.
    """
    big = torch.finfo(v.dtype).max
    filled = torch.where(valid, v, torch.full_like(v, big))
    srt, _ = torch.sort(filled, dim=1)
    n = valid.sum(dim=1)                                      # (R,)
    n_safe = torch.clamp(n, min=1)
    lo = torch.clamp((n_safe - 1) // 2, min=0).unsqueeze(1)
    hi = torch.clamp(n_safe // 2, min=0).unsqueeze(1)
    med = 0.5 * (torch.gather(srt, 1, lo) + torch.gather(srt, 1, hi))
    return torch.where((n > 0).unsqueeze(1), med, torch.ones_like(med))


def _row_median_fill(v: torch.Tensor, positive_only: bool = True) -> torch.Tensor:
    """Replace non-finite (and, optionally, non-positive) entries by the row median.

    Reproduces the numpy fallback in
    :func:`abffr.information.conditional_force_variance` exactly: a cell we could
    not measure must look like a *typical* cell, never like an easy one, because
    "unmeasured" and "cheap" would then be the same instruction to the allocator.
    """
    ok = torch.isfinite(v)
    if positive_only:
        ok = ok & (v > 0)
    med = _numpy_median_rowwise(v, ok)
    return torch.where(ok, v, med.expand_as(v))


class IOBatch:
    """Torch-side accumulation of the two streams the estimator consumes."""

    def __init__(self, R: int, J: int, capacity: int, device, dtype):
        self.R, self.J = int(R), int(J)
        self.capacity = int(capacity)
        self.device, self.dtype = device, dtype
        self.s2_sum = torch.zeros((R, J), device=device, dtype=dtype)
        self.s2_n = 0
        self._buf: List[torch.Tensor] = []

    def observe(self, cell: torch.Tensor, force: torch.Tensor,
                fp_at: torch.Tensor) -> None:
        """One observation batch.  ``cell`` (R,N) long; ``force``/``fp_at`` (R,N).

        The residual is taken against ``F'_hat`` *at each replica's own position*,
        not against the cell mean.  A cell is wide enough that ``F'`` varies
        across it, and charging that systematic variation to the noise would make
        every steep region read as statistically hard -- a landscape feature
        wearing a difficulty costume.
        """
        R, J = self.R, self.J
        ones = torch.ones_like(force)
        cnt = torch.zeros((R, J), device=self.device, dtype=self.dtype)
        cnt.scatter_add_(1, cell, ones)
        resid = force - fp_at
        ssq = torch.zeros((R, J), device=self.device, dtype=self.dtype)
        ssq.scatter_add_(1, cell, resid * resid)
        s2 = torch.where(cnt > 1, ssq / torch.clamp(cnt - 1.0, min=1.0),
                         torch.full_like(ssq, float("nan")))
        self.s2_sum = self.s2_sum + _row_median_fill(s2, positive_only=True)
        self.s2_n += 1

        fsum = torch.zeros((R, J), device=self.device, dtype=self.dtype)
        fsum.scatter_add_(1, cell, force)
        fmean = torch.where(cnt > 0, fsum / torch.clamp(cnt, min=EPS),
                            torch.full_like(fsum, float("nan")))
        self._buf.append(fmean)
        if len(self._buf) > self.capacity:
            self._buf.pop(0)

    @property
    def n_samples(self) -> int:
        return len(self._buf)

    def sigma2(self) -> np.ndarray:
        """``(R, J)`` mean residual variance, or ones before any observation."""
        if self.s2_n == 0:
            return np.ones((self.R, self.J))
        return (self.s2_sum / float(self.s2_n)).detach().cpu().numpy()

    def series(self) -> np.ndarray:
        """``(R, n, J)`` mean-force history, oldest first."""
        if not self._buf:
            return np.zeros((self.R, 0, self.J))
        return torch.stack(self._buf, dim=1).detach().cpu().numpy()


def tau_from_series(series: np.ndarray, obs_interval: float,
                    min_samples: int = 40) -> np.ndarray:
    """Batched twin of :func:`abffr.information.tau_from_lag1`.

    Same arithmetic per cell, cell by cell, including the Kendall small-sample
    correction and the "NaN, never a small number" convention: an unresolved cell
    and an easy cell must not look alike to the allocator.  A regression test
    asserts equality against the original on the same input, so this stays a
    batching of the frozen estimator rather than a second estimator.
    """
    series = np.asarray(series, dtype=float)
    if series.ndim != 3:
        raise ValueError("series must be (R, n, J)")
    R, n_obs, J = series.shape
    out = np.full((R, J), np.nan)
    if n_obs < int(min_samples):
        return out
    for r in range(R):
        S = series[r]
        for j in range(J):
            v = S[:, j]
            v = v[np.isfinite(v)]
            n = v.size
            if n < int(min_samples):
                continue
            v = v - v.mean()
            denom = float(v[:-1] @ v[:-1])
            if denom <= 0:
                continue
            phi = float(v[1:] @ v[:-1]) / denom
            phi = phi + (1.0 + 3.0 * phi) / n
            if not (0.0 < phi < 1.0):
                continue
            out[r, j] = (-1.0 / np.log(phi)) * float(obs_interval)
    return out


# --------------------------------------------------------------------------- #
# the allocator
# --------------------------------------------------------------------------- #
class IOAllocator:
    """One allocator for a whole batch of runs; ``arms[r]`` selects each row's rule.

    Holds, per row: the Fisher--Rao cell mass, and (through :class:`IOBatch`,
    shared across rows) the two observation streams.  Emits a grid-level force
    and a diagnostic row per opportunity.  It never returns indices, because
    there is nothing to resample.
    """

    def __init__(self, arms: Sequence[str], x_grid: torch.Tensor,
                 eval_mask: torch.Tensor, beta: np.ndarray, dt: float,
                 cfg: IOConfig, device=None, dtype=None):
        arms = [str(a) for a in arms]
        for a in arms:
            if a not in ARMS:
                raise ValueError(f"unknown IO arm {a!r}; have {list(ARMS)}")
        self.arms = arms
        self.R = len(arms)
        self.cfg = cfg
        self.dt = float(dt)
        self.device = device if device is not None else x_grid.device
        self.dtype = dtype if dtype is not None else x_grid.dtype

        xg = x_grid.detach().cpu().numpy().astype(float)
        mask = eval_mask.detach().cpu().numpy().astype(bool)
        self.x_grid_np = xg
        d = np.diff(xg)
        self._uniform_grid = bool(
            np.allclose(d, d[0], rtol=1e-9, atol=0.0) and d[0] > 0)
        self.x0, self.x1 = float(xg[0]), float(xg[-1])
        self.G = xg.size
        J = int(cfg.n_cells)
        self.J = J
        self.edges = np.linspace(self.x0, self.x1, J + 1)
        self.cell_of_grid = np.clip(np.digitize(xg, self.edges) - 1, 0, J - 1)

        # Static leverage a_j: how much a unit of mean-force variance at j costs
        # the endpoint, through the cumulative trapezoid and the centring.  Pure
        # grid geometry and the mask -- which is why the mask has to be a fixed
        # geometric window and not a thermal one.
        self.a_cell = al.cell_reduce(al.leverage(xg, mask), self.cell_of_grid, J)

        self.beta = np.asarray(beta, dtype=float).reshape(-1)
        if self.beta.size != self.R:
            raise ValueError("beta must have one entry per run")

        self.masses = [cm.CellMass(n_cells=J, theta=float(cfg.theta))
                       for _ in range(self.R)]
        self.stream = IOBatch(self.R, J, cfg.history_capacity,
                              self.device, self.dtype)

        self.edges_t = torch.as_tensor(self.edges, device=self.device,
                                       dtype=self.dtype)
        self.cell_of_grid_t = torch.as_tensor(self.cell_of_grid,
                                              device=self.device,
                                              dtype=torch.long)
        self.allocating = torch.as_tensor(
            [a in ALLOCATING for a in arms], device=self.device)
        # Zero until the first opportunity: before the allocation window opens
        # every arm is plain ABF, which is what makes the burn-in shared.
        self.force_grid = torch.zeros((self.R, self.G), device=self.device,
                                      dtype=self.dtype)
        self.rows: List[Dict] = []
        self._obs_interval_steps = int(cfg.obs_every)

    # -- geometry ----------------------------------------------------------
    def cell_of(self, X: torch.Tensor) -> torch.Tensor:
        """``(R, N)`` cell index of each replica.  Clamped, never wrapped."""
        return torch.clamp(torch.bucketize(X.contiguous(), self.edges_t) - 1,
                           0, self.J - 1)

    # -- the observation stream -------------------------------------------
    def observe(self, X: torch.Tensor, force: torch.Tensor,
                fp_at: torch.Tensor) -> None:
        self.stream.observe(self.cell_of(X), force, fp_at)

    # -- the estimator ------------------------------------------------------
    def gamma_hat(self) -> Dict[str, np.ndarray]:
        """``(R, J)`` sigma^2, tau and Gamma = sigma^2 tau, frozen decomposition."""
        sigma2 = self.stream.sigma2()
        tau_raw = tau_from_series(
            self.stream.series(),
            obs_interval=float(self._obs_interval_steps) * self.dt,
            min_samples=int(self.cfg.tau_min_samples))
        gamma = np.empty_like(sigma2)
        for r in range(self.R):
            gamma[r] = inf.gamma_hat_decomposed(sigma2[r], tau_raw[r],
                                                shrink=float(self.cfg.shrink))
        return {"sigma2": sigma2, "tau": tau_raw, "gamma": gamma}

    # -- the target ---------------------------------------------------------
    def _r_star_row(self, r_idx: int, g: np.ndarray, q: np.ndarray):
        arm = self.arms[r_idx]
        if arm == "A6b":
            return al.apply_floor(al.r_neyman(g), self.cfg.floor_fraction), 0.0, float("nan")
        if arm == "A6c":
            # The floor is inside the solve, so the ESS reported is the ESS of
            # the target the run applies -- see allocation.r_ess_constrained.
            out = al.r_ess_constrained(g, q, rho=float(self.cfg.rho),
                                       floor_fraction=float(self.cfg.floor_fraction))
            return out.r, float(out.lam), float(out.ess_fraction)
        return al.r_uniform(self.J), 0.0, float("nan")

    def _force_from_r(self, r_target: np.ndarray, beta: float) -> np.ndarray:
        """``d/dz [beta^-1 log r*(z)]`` on the profile grid.

        ``log r*`` is carried to the grid by linear interpolation between cell
        centres, not as a piecewise constant: differentiating a step function
        would place the whole increment on the cell boundaries as spikes, which
        is a discretisation artefact and not the intended force.
        """
        centres = 0.5 * (self.edges[1:] + self.edges[:-1])
        log_r = np.log(np.maximum(r_target, EPS))
        log_r_grid = np.interp(self.x_grid_np, centres, log_r)
        # Scalar spacing on a uniform grid, coordinates otherwise.  np.gradient's
        # non-uniform branch does not cancel exactly in floating point, so a
        # *constant* log r* differentiates to ~1e-15 rather than to 0 -- small,
        # but it would mean a uniform target still nudged the dynamics, and the
        # identity gate G0.1 is only worth having if it is an identity.
        step = float(self.x_grid_np[1] - self.x_grid_np[0])
        spacing = (step if self._uniform_grid else self.x_grid_np)
        return np.gradient(log_r_grid, spacing) / float(beta)

    # -- one opportunity ----------------------------------------------------
    def refresh(self, step: int, X: torch.Tensor, A_grid: torch.Tensor,
                record: bool = True) -> torch.Tensor:
        """Update the mass and the target; return the ``(R, G)`` bias force.

        ``A_grid`` is the *running* ABF free energy, never a reference: it feeds
        the Fisher--Rao target ``q ∝ exp(-beta A_hat)`` and nothing else.
        """
        est = self.gamma_hat()
        A_np = A_grid.detach().cpu().numpy()
        cell = self.cell_of(X)
        counts_t = torch.zeros((self.R, self.J), device=self.device,
                               dtype=self.dtype)
        counts_t.scatter_add_(1, cell, torch.ones_like(X, dtype=self.dtype))
        counts = counts_t.detach().cpu().numpy()
        K = float(X.shape[1])

        new_force = np.zeros((self.R, self.G))
        for r in range(self.R):
            A_cell = np.array([A_np[r][self.cell_of_grid == j].mean()
                               for j in range(self.J)])
            self.masses[r].fr_step(
                cm.log_target_from_free_energy(A_cell, self.beta[r]))
            q = self.masses[r].mass
            g = self.a_cell * est["gamma"][r]
            r_star, lam, ess_pred = self._r_star_row(r, g, q)
            if self.arms[r] in ALLOCATING:
                new_force[r] = self._force_from_r(r_star, self.beta[r])
            if not record:
                continue
            row = dict(step=int(step), run=int(r), arm=self.arms[r],
                       lam=float(lam), ess_predicted=float(ess_pred),
                       mass_ess_at_occupancy=al.mass_ess_fraction(q, counts[r] / K),
                       n_occupied=int((counts[r] > 0).sum()),
                       valid_tau_fraction=float(
                           np.mean(np.isfinite(est["tau"][r]) & (est["tau"][r] > 0))),
                       r_star=r_star.tolist(),
                       occupancy=(counts[r] / K).tolist(),
                       sigma2=est["sigma2"][r].tolist(),
                       tau=est["tau"][r].tolist(),
                       gamma=est["gamma"][r].tolist(),
                       q=q.tolist())
            if self.arms[r] == "A6c":
                free = al.apply_floor(al.r_neyman(g), self.cfg.floor_fraction)
                row["fr_active"] = bool(lam > 0.0)
                row["tv_to_unconstrained"] = float(0.5 * np.abs(r_star - free).sum())
                row["mass_ess_unconstrained"] = al.mass_ess_fraction(q, free)
                row["r_star_unconstrained"] = free.tolist()
            self.rows.append(row)

        self.force_grid = torch.as_tensor(new_force, device=self.device,
                                          dtype=self.dtype)
        return self.force_grid

    # -- what the engine applies -------------------------------------------
    def bias_force_at(self, X: torch.Tensor) -> torch.Tensor:
        """Linear interpolation of the allocation force onto ``(R, N)`` positions.

        Zero for every A0 row at every step, and zero for every row before the
        first opportunity, so the arms share their burn-in exactly.
        """
        G = self.G
        dx = (self.x1 - self.x0) / float(G - 1)
        pos = torch.clamp((X - self.x0) / dx, 0.0, float(G - 1) - 1e-9)
        i0 = pos.floor().long()
        w = (pos - i0.to(pos.dtype))
        f0 = torch.gather(self.force_grid, 1, i0)
        f1 = torch.gather(self.force_grid, 1, torch.clamp(i0 + 1, max=G - 1))
        return f0 * (1.0 - w) + f1 * w

    # -- diagnostics --------------------------------------------------------
    def final_state(self, with_series: bool = False) -> Dict[str, np.ndarray]:
        est = self.gamma_hat()
        out = {"a_cell": self.a_cell,
               "cell_edges": self.edges,
               "sigma2": est["sigma2"], "tau": est["tau"],
               "gamma": est["gamma"],
               "q": np.vstack([m.mass for m in self.masses])}
        if with_series:
            out["series"] = self.stream.series()
            out["obs_every"] = np.array([self._obs_interval_steps])
        return out


# --------------------------------------------------------------------------- #
# Rule R-OBS: choosing the observation interval, once per system, from A0 alone
# --------------------------------------------------------------------------- #
#: Target lag-1 autocorrelation for the chosen interval.  exp(-1/2) puts the
#: interval at ``tau / 2``, in the middle of the band where the AR(1) fit works:
#: ``phi -> 0`` (interval long against tau) makes a cell read as unresolved, and
#: ``phi -> 1`` (interval short) makes the fitted phi cross 1 on sampling noise.
#: Both failures are silent, and they point in opposite directions, so there is
#: no safe default -- the interval has to be measured.
R_OBS_TARGET_RHO = float(np.exp(-0.5))


def probe_obs_every(series: np.ndarray, dense_obs_every: int,
                    target_rho: float = R_OBS_TARGET_RHO,
                    max_lag: Optional[int] = None) -> Dict[str, float]:
    """Observation interval, in engine steps, from a dense A0-only probe.

    For each cell, the smallest lag whose mean-force autocorrelation has fallen
    to ``target_rho``; the interval is the median of those lags.  This is a *lag
    search*, not a second estimator of ``Gamma``: what it returns is the sampling
    cadence, and ``Gamma_hat`` is still ``sigma_hat^2 tau_hat`` with the frozen
    lag-1 AR(1) fit evaluated at that cadence.

    Reads only A0 trajectories.  No reference free energy, no candidate arm.
    """
    S = np.asarray(series, dtype=float)
    if S.ndim == 2:
        S = S[None]
    R, n, J = S.shape
    max_lag = int(max_lag if max_lag is not None else max(2, n // 4))
    lags = []
    for r in range(R):
        for j in range(J):
            v = S[r, :, j]
            v = v[np.isfinite(v)]
            if v.size < 50:
                continue
            v = v - v.mean()
            denom = float(v @ v)
            if denom <= 0:
                continue
            hit = None
            for k in range(1, min(max_lag, v.size - 2)):
                rho = float(v[k:] @ v[:-k]) / denom * (v.size / (v.size - k))
                if rho <= target_rho:
                    hit = k
                    break
            if hit is not None:
                lags.append(hit)
    if not lags:
        return {"obs_every": int(dense_obs_every), "n_cells_resolved": 0,
                "lag_median": float("nan")}
    lag_med = float(np.median(lags))
    return {"obs_every": int(max(1, round(lag_med * dense_obs_every))),
            "n_cells_resolved": len(lags),
            "lag_median": lag_med,
            "lag_q10": float(np.quantile(lags, 0.1)),
            "lag_q90": float(np.quantile(lags, 0.9))}
