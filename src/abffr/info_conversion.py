"""Information-conversion audit: one bd_standard pulse toward the finite-horizon
information-optimal allocation, then a decorrelation-length cooldown.

Frozen protocol: ``docs/INFORMATION_CONVERSION_AUDIT_PREREGISTRATION.md``.

This module is deliberately *not* a general engine.  It implements exactly one
experiment: plain ABF to a burn-in checkpoint, a fork into arms, at most ONE
standard Fisher--Rao birth--death opportunity per arm, and plain ABF afterwards.
Particle reallocation can only happen through :func:`abffr.fr_v3.bd_standard`;
no resampler, no bias-held allocation, no clipping, no caps, no post-clone
perturbation and no held-out clones exist on any code path here, and a unit
test asserts the source stays that way.

The dynamics/estimator arithmetic mirrors ``simulation_torch.run_batch`` on its
``post_propagation`` / ``binned_smooth`` path (the validated q-r configuration)
operation for operation; ``tests/test_info_conversion.py`` holds an end-to-end
parity gate against the production engine.  Deposits happen every physical step
-- ``abf.update_every`` is only the grid-refresh cadence -- which is what makes
``M = K x H`` the correct count of future deposition opportunities.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from . import allocation as al
from . import fr_v3
from . import kappa_family as kfam
from . import potentials
from . import torch_utils as tu
from .io_utils import make_rng_streams
from .simulation import _init_positions

EPS = 1e-12


# --------------------------------------------------------------------------- #
# Stage 0C: the constrained finite-horizon information target
# --------------------------------------------------------------------------- #
def solve_finite_horizon_target(av: np.ndarray, C: np.ndarray, M: float,
                                K: int, n_iter: int = 400) -> Dict:
    """``min_pi sum_j av_j/(C_j + M pi_j)`` s.t. ``sum pi = 1``, ``pi_j >= 1/K``.

    ``av = a_j V_j`` is the leverage-weighted oracle difficulty.  Solved by
    monotone bisection on the KKT multiplier ``lam``: the unconstrained
    stationarity condition on a free cell is ``av_j M/(C_j+M pi_j)^2 = lam``,
    so ``pi_j(lam) = max(1/K, (sqrt(av_j M/lam) - C_j)/M)`` and
    ``sum_j pi_j(lam)`` is continuous and non-increasing -- the root is unique.

    The ``1/K`` floor is a coverage constraint (at least one expected replica
    per cell) and guarantees the FR target is strictly positive.  Cells with
    ``av_j = 0`` (outside the evaluation mask) sit on the floor at any ``lam``.
    """
    av = np.maximum(np.asarray(av, dtype=float), 0.0)
    C = np.asarray(C, dtype=float)
    if av.shape != C.shape:
        raise ValueError("av and C must have the same number of cells")
    if np.any(C < 0):
        raise ValueError("counts must be non-negative")
    J = av.size
    M = float(M)
    if M <= 0:
        raise ValueError("M must be positive")
    floor = 1.0 / float(K)
    if J * floor > 1.0 + 1e-12:
        raise ValueError("infeasible: J/K > 1 leaves no mass for the floor")
    if not (av > 0).any():
        raise ValueError("all-zero a_j V_j: the objective has no optimum")

    def pi_of(lam: float) -> np.ndarray:
        with np.errstate(divide="ignore"):
            free = (np.sqrt(av * M / lam) - C) / M
        return np.maximum(floor, free)

    lo, hi = 1.0, 1.0
    for _ in range(600):                       # bracket: sum decreasing in lam
        if pi_of(lo).sum() > 1.0:
            break
        lo /= 8.0
    for _ in range(600):
        if pi_of(hi).sum() < 1.0:
            break
        hi *= 8.0
    for _ in range(n_iter):
        mid = math.sqrt(lo * hi)
        if pi_of(mid).sum() > 1.0:
            lo = mid
        else:
            hi = mid
    lam = math.sqrt(lo * hi)
    pi = pi_of(lam)

    # Exact renormalisation of the bisection residual over the free cells,
    # proportional to their headroom above the floor: preserves the floor and
    # the free-cell KKT ratios to first order; the residual itself is ~1e-13.
    residual = 1.0 - pi.sum()
    head = pi - floor
    if head.sum() > 0:
        pi = pi + residual * head / head.sum()
    if abs(pi.sum() - 1.0) > 1e-9:
        raise RuntimeError(f"water-filling failed to normalise: sum={pi.sum()}")
    return {"pi": pi, "lam": float(lam),
            "floor_bound": (pi <= floor + 1e-12),
            "risk": predicted_finite_risk(av, C, M, pi)}


def predicted_finite_risk(av: np.ndarray, C: np.ndarray, M: float,
                          pi: np.ndarray) -> float:
    """``sum_j av_j / (C_j + M pi_j)`` -- the finite-horizon risk model."""
    av = np.asarray(av, dtype=float)
    live = av > 0
    denom = np.asarray(C, dtype=float) + float(M) * np.asarray(pi, dtype=float)
    if np.any(denom[live] <= 0):
        return float("inf")
    return float(np.sum(av[live] / denom[live]))


# --------------------------------------------------------------------------- #
# Geometry: cells, leverage, reference cell forces
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CellGeometry:
    edges: np.ndarray            # (J+1,)
    cell_of_grid: np.ndarray     # (G,)
    a_cell: np.ndarray           # (J,)
    f_ref_cell: np.ndarray       # (J,) node-average of Fprime_ref per cell
    n_cells: int


def build_cells(x_grid: np.ndarray, eval_mask: np.ndarray,
                n_cells: int, Fprime_ref: np.ndarray) -> CellGeometry:
    """Same construction as the q-r campaign's arm layer: digitize, equal cells."""
    x = np.asarray(x_grid, dtype=float)
    J = int(n_cells)
    edges = np.linspace(float(x[0]), float(x[-1]), J + 1)
    cog = np.clip(np.digitize(x, edges) - 1, 0, J - 1)
    a_cell = al.cell_reduce(al.leverage(x, eval_mask), cog, J)
    fr = np.asarray(Fprime_ref, dtype=float)
    f_ref = np.array([fr[cog == j].mean() for j in range(J)])
    return CellGeometry(edges=edges, cell_of_grid=cog, a_cell=a_cell,
                        f_ref_cell=f_ref, n_cells=J)


def cell_index_torch(X: torch.Tensor, edges_t: torch.Tensor) -> torch.Tensor:
    """Torch twin of ``np.clip(np.digitize(x, edges) - 1, 0, J - 1)``.

    ``bucketize(right=True)`` returns i with ``edges[i-1] <= x < edges[i]``,
    which is exactly ``digitize(right=False)``.
    """
    J = edges_t.numel() - 1
    return (torch.bucketize(X, edges_t, right=True) - 1).clamp_(0, J - 1)


def target_density_grid(pi: np.ndarray, geom: CellGeometry,
                        x_grid: np.ndarray) -> np.ndarray:
    """Piecewise-constant target density ``q*(x) = pi_cell(x) / cell_width``."""
    width = geom.edges[1] - geom.edges[0]
    q = np.asarray(pi, dtype=float)[geom.cell_of_grid] / width
    return q


# --------------------------------------------------------------------------- #
# Noise: chunk-keyed (production) and sequential-bank (parity vs run_batch)
# --------------------------------------------------------------------------- #
class ChunkKeyedNoise:
    """Langevin variates keyed by (base_seed, seed, chunk, step, slot).

    Independent of batch composition, arm structure, and resume point: chunk
    ``c`` for seed ``s`` is always drawn from a fresh generator seeded
    ``stable_seed('langevin-chunk', base_seed, s, c)``.  All arms forked from
    one seed therefore share identical noise per (step, slot).
    """

    def __init__(self, seeds: Sequence[int], n_particles: int, device,
                 dtype, base_seed: int = 0, chunk_steps: int = 500):
        self.seeds = [int(s) for s in seeds]
        self.uniq = sorted(set(self.seeds))
        self.row_index = torch.as_tensor(
            [self.uniq.index(s) for s in self.seeds], device=device,
            dtype=torch.long)
        self.N = int(n_particles)
        self.device, self.dtype = device, dtype
        self.base_seed = int(base_seed)
        self.chunk_steps = int(chunk_steps)
        self._chunk = None
        self._chunk_idx = -1

    def rebind(self, seeds: Sequence[int]) -> None:
        """Change the row -> seed mapping (the fork); chunks stay valid."""
        self.seeds = [int(s) for s in seeds]
        new_uniq = sorted(set(self.seeds))
        if new_uniq != self.uniq:
            self.uniq = new_uniq
            self._chunk, self._chunk_idx = None, -1
        self.row_index = torch.as_tensor(
            [self.uniq.index(s) for s in self.seeds], device=self.device,
            dtype=torch.long)

    def at(self, step: int) -> Tuple[torch.Tensor, torch.Tensor]:
        c = int(step) // self.chunk_steps
        if c != self._chunk_idx:
            chunks = []
            for s in self.uniq:
                g = tu.make_generator(
                    tu.stable_seed("langevin-chunk", self.base_seed, s, c),
                    self.device)
                chunks.append(torch.randn(
                    (self.chunk_steps, 2, self.N), generator=g,
                    device=self.device, dtype=self.dtype))
            self._chunk = torch.stack(chunks, dim=0)
            self._chunk_idx = c
        noise = self._chunk.index_select(
            0, self.row_index)[:, int(step) - c * self.chunk_steps]
        return noise[:, 0], noise[:, 1]


class SequentialBankNoise:
    """Verbatim twin of ``simulation_torch._MatchedNoiseBank`` (parity only)."""

    def __init__(self, seeds, n_particles, n_steps, device, dtype, base_seed,
                 chunk_steps=1024):
        self.n_particles = int(n_particles)
        self.n_steps = int(n_steps)
        self.device, self.dtype = device, dtype
        self.chunk_steps = max(int(chunk_steps), 1)
        uniq = sorted({int(s) for s in seeds})
        seed_to_row = {s: i for i, s in enumerate(uniq)}
        self.row_index = torch.as_tensor(
            [seed_to_row[int(s)] for s in seeds], device=device,
            dtype=torch.long)
        self.generators = [
            tu.make_generator(tu.stable_seed("langevin", base_seed, s), device)
            for s in uniq]
        self.chunk = None
        self.chunk_start = -1

    def at(self, step):
        if (self.chunk is None or step < self.chunk_start
                or step >= self.chunk_start + self.chunk.shape[1]):
            length = min(self.chunk_steps, self.n_steps - int(step))
            self.chunk = torch.stack([
                torch.randn((length, 2, self.n_particles), generator=g,
                            device=self.device, dtype=self.dtype)
                for g in self.generators], dim=0)
            self.chunk_start = int(step)
        noise = self.chunk.index_select(
            0, self.row_index)[:, step - self.chunk_start]
        return noise[:, 0], noise[:, 1]


# --------------------------------------------------------------------------- #
# Engine state
# --------------------------------------------------------------------------- #
@dataclass
class EngineState:
    X: torch.Tensor              # (B, N)
    Y: torch.Tensor
    C_acc: torch.Tensor          # (B, G) grid count accumulator
    S_acc: torch.Tensor          # (B, G) grid force accumulator
    Fprime_hat: torch.Tensor     # (B, G)
    F_hat: torch.Tensor          # (B, G)
    ancestors: torch.Tensor      # (B, N) long
    cell_cnt: torch.Tensor       # (B, J) diagnostic hard-cell counts
    cell_sum: torch.Tensor       # (B, J) diagnostic hard-cell force sums
    step: int
    seeds: List[int]             # per row (repeats after the fork)
    n_pulses: np.ndarray         # per row; asserted <= 1

    def clone_rows(self, reps: int) -> "EngineState":
        """Fork: repeat every row ``reps`` times (seed-major order)."""
        def rep(t):
            return t.repeat_interleave(reps, dim=0).contiguous()
        return EngineState(
            X=rep(self.X), Y=rep(self.Y), C_acc=rep(self.C_acc),
            S_acc=rep(self.S_acc), Fprime_hat=rep(self.Fprime_hat),
            F_hat=rep(self.F_hat), ancestors=rep(self.ancestors),
            cell_cnt=rep(self.cell_cnt), cell_sum=rep(self.cell_sum),
            step=self.step,
            seeds=[s for s in self.seeds for _ in range(reps)],
            n_pulses=np.repeat(self.n_pulses, reps))


@dataclass
class PairTracker:
    """Sibling (clone, continuation) force products, per tracked row."""

    clone_idx: torch.Tensor      # (P,) long
    cont_idx: torch.Tensor       # (P,) long
    sums: List[List[float]] = field(default_factory=list)

    def record(self, forces_row: torch.Tensor, step_since: int) -> None:
        fa = forces_row[self.clone_idx]
        fb = forces_row[self.cont_idx]
        self.sums.append([
            float(step_since), float(fa.numel()), float(fa.sum()),
            float(fb.sum()), float((fa * fb).sum()),
            float((fa * fa).sum()), float((fb * fb).sum())])


class InfoConversionEngine:
    """The single-pulse experiment engine.  See the module docstring."""

    def __init__(self, cfg: Dict, kappa_cell: str, x_grid: np.ndarray,
                 F_ref: np.ndarray, Fprime_ref: np.ndarray,
                 geom: CellGeometry, device: torch.device,
                 dtype: torch.dtype = torch.float64, base_seed: int = 0,
                 noise_mode: str = "chunk"):
        sim, abf, dom = cfg["simulation"], cfg["abf"], cfg["domain"]
        self.beta = float(sim["beta"])
        self.dt = float(sim["dt"])
        self.N = int(sim["n_particles"])
        self.update_every = max(1, int(abf["update_every"]))
        self.min_count = float(abf.get("min_count", 1.0))
        self.h = float(abf["h"])
        self.eta = float(cfg.get("kde", {}).get("eta", 0.10))
        self.xmin, self.xmax = float(dom["x_min"]), float(dom["x_max"])
        self.ymin, self.ymax = float(dom["y_min"]), float(dom["y_max"])
        self.x_tilt = float(cfg.get("potential", {}).get("x_tilt", 0.0))
        self.kappa_a, self.kappa_shift = kfam.KAPPA_CELLS[kappa_cell]
        self.device, self.dtype = device, dtype
        self.base_seed = int(base_seed)
        self.noise_mode = noise_mode
        self.chunk_steps = int(sim.get("noise_chunk_steps", 500))

        x = np.asarray(x_grid, dtype=float)
        self.x_grid = x
        self.G = x.size
        self.x_grid_t = torch.as_tensor(x, device=device, dtype=dtype)
        self.x0 = float(x[0])
        self.dx = tu.grid_spacing(self.x_grid_t)
        self.idx0 = int(np.argmin(np.abs(x)))
        self.F_ref_t = torch.as_tensor(np.asarray(F_ref), device=device,
                                       dtype=dtype).view(1, self.G)
        self.geom = geom
        self.edges_t = torch.as_tensor(geom.edges, device=device, dtype=dtype)
        self.k_h, self.r_h = tu.gaussian_kernel1d(self.h, self.dx, device, dtype)
        self.k_eta, self.r_eta = tu.gaussian_kernel1d(self.eta, self.dx,
                                                      device, dtype)
        self.noise_scale = float(np.sqrt(2.0 * self.dt / self.beta))
        self.noise = None

    # -- state ------------------------------------------------------------- #
    def init_state(self, seeds: Sequence[int],
                   n_steps_hint: int = 50_000) -> EngineState:
        seeds = [int(s) for s in seeds]
        B, N = len(seeds), self.N
        X = np.empty((B, N)); Y = np.empty((B, N))
        for b, s in enumerate(seeds):
            rng_init, _, _ = make_rng_streams(s)
            X[b] = _init_positions(rng_init, N, self.xmin, self.xmax, "uniform")
            Y[b] = _init_positions(rng_init, N, self.ymin, self.ymax, "uniform")
        dev, dt_ = self.device, self.dtype
        if self.noise_mode == "chunk":
            self.noise = ChunkKeyedNoise(seeds, N, dev, dt_, self.base_seed,
                                         self.chunk_steps)
        else:
            self.noise = SequentialBankNoise(seeds, N, n_steps_hint, dev, dt_,
                                             self.base_seed)
        J = self.geom.n_cells
        return EngineState(
            X=torch.as_tensor(X, device=dev, dtype=dt_),
            Y=torch.as_tensor(Y, device=dev, dtype=dt_),
            C_acc=torch.zeros((B, self.G), device=dev, dtype=dt_),
            S_acc=torch.zeros((B, self.G), device=dev, dtype=dt_),
            Fprime_hat=torch.zeros((B, self.G), device=dev, dtype=dt_),
            F_hat=torch.zeros((B, self.G), device=dev, dtype=dt_),
            ancestors=torch.arange(N, device=dev).expand(B, N).contiguous(),
            cell_cnt=torch.zeros((B, J), device=dev, dtype=dt_),
            cell_sum=torch.zeros((B, J), device=dev, dtype=dt_),
            step=0, seeds=seeds, n_pulses=np.zeros(B, dtype=int))

    def fork(self, state: EngineState, n_arms: int) -> EngineState:
        forked = state.clone_rows(n_arms)
        if isinstance(self.noise, ChunkKeyedNoise):
            self.noise.rebind(forked.seeds)
        else:
            raise RuntimeError("fork requires the chunk-keyed noise mode")
        return forked

    # -- estimator arithmetic (mirrors run_batch verbatim) ------------------ #
    def recompute_grid(self, st: EngineState) -> None:
        num_s = tu.smooth_grid(st.S_acc, self.k_h, self.r_h, self.dx)
        den_s = tu.smooth_grid(st.C_acc, self.k_h, self.r_h, self.dx)
        st.Fprime_hat = num_s / (den_s + self.min_count + EPS)
        st.F_hat = tu.center_at_index(
            tu.cumulative_trapezoid(st.Fprime_hat, self.dx), self.idx0)

    def p_hat(self, st: EngineState) -> torch.Tensor:
        hist = tu.scatter_grid(
            tu.nearest_index(st.X, self.x0, self.dx, self.G), self.G)
        p = tu.smooth_grid(hist, self.k_eta, self.r_eta, self.dx) / self.N
        return tu.normalize_density(p, self.dx)

    # -- propagation ------------------------------------------------------- #
    def run(self, st: EngineState, n_steps: int,
            profile_cb=None, eval_every: int = 500,
            pair_trackers: Optional[Dict[int, PairTracker]] = None,
            pulse_step: Optional[int] = None) -> None:
        """Advance ``n_steps`` of plain ABF.  No FR happens here, ever."""
        end = st.step + int(n_steps)
        B = st.X.shape[0]
        J = self.geom.n_cells
        row_off_c = (torch.arange(B, device=self.device) * J).view(B, 1)
        while st.step < end:
            step = st.step
            X, Y = st.X, st.Y
            dvdx = potentials.dVdx_xy_torch(X, Y) + self.x_tilt
            dvdy = potentials.dVdy_xy_torch(X, Y)
            abf_at_X = tu.interp1d(st.Fprime_hat, X, self.x0, self.dx)
            noise_x, noise_y = self.noise.at(step)
            X_prop = tu.reflect_into(
                X + (-dvdx + abf_at_X) * self.dt + self.noise_scale * noise_x,
                self.xmin, self.xmax)
            kap = kfam.kappa_at_torch(X, self.kappa_a, self.kappa_shift)
            if kap is None:
                Y_prop = tu.reflect_into(
                    Y + (-dvdy) * self.dt + self.noise_scale * noise_y,
                    self.ymin, self.ymax)
            else:
                Y_prop = tu.reflect_into(
                    Y + (-kap * dvdy) * self.dt
                    + torch.sqrt(kap) * self.noise_scale * noise_y,
                    self.ymin, self.ymax)
            tu.assert_finite("X_prop", X_prop)
            tu.assert_finite("Y_prop", Y_prop)

            dvdx_prop = potentials.dVdx_xy_torch(X_prop, Y_prop) + self.x_tilt
            idx = tu.nearest_index(X_prop, self.x0, self.dx, self.G)
            # Deposits use tu.scatter_grid exactly as run_batch does, so the
            # accumulation arithmetic is the engine's, not a re-derivation.
            st.C_acc += tu.scatter_grid(idx, self.G)
            st.S_acc += tu.scatter_grid(idx, self.G, dvdx_prop)
            cidx = cell_index_torch(X_prop, self.edges_t)
            st.cell_cnt.view(-1).index_add_(
                0, (cidx + row_off_c).view(-1),
                torch.ones_like(dvdx_prop).view(-1))
            st.cell_sum.view(-1).index_add_(
                0, (cidx + row_off_c).view(-1), dvdx_prop.view(-1))

            next_step = step + 1
            if next_step % self.update_every == 0:
                # order matters and mirrors run_batch: deposit, then refresh
                st.X, st.Y = X_prop, Y_prop
                self.recompute_grid(st)
            else:
                st.X, st.Y = X_prop, Y_prop
            st.step = next_step

            if pair_trackers and pulse_step is not None:
                since = next_step - pulse_step
                if since <= 500 or since % 10 == 0:
                    for b, trk in pair_trackers.items():
                        if trk.clone_idx.numel():
                            trk.record(dvdx_prop[b], since)

            if profile_cb is not None and (
                    next_step % eval_every == 0 or next_step == end):
                profile_cb(st)

    # -- the one FR opportunity -------------------------------------------- #
    def pulse(self, st: EngineState, q_grid: torch.Tensor, p90_by_row: Dict[int, float],
              generators: Dict[int, torch.Generator]) -> List[Dict]:
        """At most ONE ``bd_standard`` opportunity per listed row.

        ``q_grid`` is (B, G), normalised.  Rows absent from ``p90_by_row`` are
        untouched (plain-ABF arms).  Executed strictly *after* the current
        step's deposit -- the caller forks after ``run``, so a clone created
        here has deposited nothing and first speaks after its next propagation.
        """
        p_pre = self.p_hat(st)
        log_p_part = torch.log(
            tu.interp1d(p_pre, st.X, self.x0, self.dx).clamp_min(EPS))
        log_q_part = torch.log(
            tu.interp1d(q_grid, st.X, self.x0, self.dx).clamp_min(EPS))
        kl_pre = _kl_grid(p_pre, q_grid, self.dx)
        tv_pre = _tv_grid(p_pre, q_grid, self.dx)

        rows = []
        for b, p90 in sorted(p90_by_row.items()):
            if st.n_pulses[b] >= 1:
                raise RuntimeError(
                    f"row {b} already pulsed: the protocol allows exactly one")
            score = fr_v3.FRScore(log_p=log_p_part[b], log_q=log_q_part[b])
            s90 = float(torch.quantile(score.S.abs(), 0.90))
            occ_pre = torch.bincount(
                cell_index_torch(st.X[b], self.edges_t),
                minlength=self.geom.n_cells).double().cpu().numpy() / self.N
            if s90 <= 0.0:
                rows.append(dict(row=b, p90=float(p90), s90=s90, dtau=0.0,
                                 n_events=0, n_replacements=0,
                                 degenerate=True))
                st.n_pulses[b] += 1
                continue
            dtau = fr_v3.bd_timestep(score, float(p90))
            src, n_events = fr_v3.bd_standard(score, dtau, generators[b])
            is_clone = fr_v3.clone_mask(src)
            st.X[b] = st.X[b][src]
            st.Y[b] = st.Y[b][src]
            st.ancestors[b] = st.ancestors[b][src]
            st.n_pulses[b] += 1
            rows.append(dict(
                row=b, p90=float(p90), s90=s90, dtau=float(dtau),
                n_events=int(n_events),
                n_replacements=fr_v3.replacement_count(src, self.N),
                degenerate=False,
                occ_pre=occ_pre, src=src.detach().cpu().numpy().copy(),
                is_clone=is_clone.detach().cpu().numpy().copy()))

        p_post = self.p_hat(st)
        kl_post = _kl_grid(p_post, q_grid, self.dx)
        tv_post = _tv_grid(p_post, q_grid, self.dx)
        for r in rows:
            b = r["row"]
            r["kl_pre"] = float(kl_pre[b]); r["kl_post"] = float(kl_post[b])
            r["tv_pre"] = float(tv_pre[b]); r["tv_post"] = float(tv_post[b])
            ess, wmax = _anc_stats_row(st.ancestors[r["row"]], self.N)
            r["ess_anc"] = ess; r["wmax_family"] = wmax
        return rows


def _kl_grid(p_hat, q_grid, dx):
    p = p_hat.clamp_min(EPS)
    q = q_grid.clamp_min(EPS)
    return tu.trapezoid(p * (torch.log(p) - torch.log(q)), dx)


def _tv_grid(p_hat, q_grid, dx):
    return 0.5 * tu.trapezoid((p_hat - q_grid).abs(), dx)


def _anc_stats_row(anc: torch.Tensor, n_particles: int) -> Tuple[float, float]:
    c = torch.bincount(anc, minlength=n_particles).double()
    w = c / float(n_particles)
    ess = float(1.0 / (w * w).sum().clamp_min(EPS))
    return ess / float(n_particles), float(w.max())


def local_ancestor_ess(anc_row: np.ndarray, cell_row: np.ndarray,
                       gaining: np.ndarray, n_particles: int) -> Dict:
    """Ancestor ESS within each pi*-gaining cell, post-pulse."""
    vals = []
    for j in np.flatnonzero(gaining):
        sel = cell_row == j
        n = int(sel.sum())
        if n == 0:
            continue
        c = np.bincount(anc_row[sel], minlength=n_particles).astype(float)
        w = c / n
        vals.append(float(1.0 / max((w ** 2).sum(), EPS)) / n)
    if not vals:
        return {"local_ess_min": float("nan"), "local_ess_median": float("nan"),
                "n_gaining_occupied": 0}
    return {"local_ess_min": float(np.min(vals)),
            "local_ess_median": float(np.median(vals)),
            "n_gaining_occupied": len(vals)}


def clone_pairs(src: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(clone_slot, continuation_slot) pairs from a BD ``src`` map."""
    seen: Dict[int, int] = {}
    a, b = [], []
    for slot, parent in enumerate(np.asarray(src, dtype=int)):
        if parent in seen:
            a.append(slot); b.append(seen[parent])
        else:
            seen[parent] = slot
    return np.asarray(a, dtype=int), np.asarray(b, dtype=int)
