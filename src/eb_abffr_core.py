"""Batched single-GPU ABF / ABF+Fisher-Rao engine for the entropic-bottleneck study.

Faithful port of ABF-FR-Entropic-Bottleneck.ipynb to vectorized PyTorch.

Model
-----
    V(x,y) = H (x^2-1)^2 + 1/2 omega(x)^2 y^2,
    omega(x) = omega_out + (omega_in - omega_out) exp(-x^2 / 2 s^2),
    xi(x,y) = x.

Analytic reference (up to an additive constant):
    F_ref(x)  = H (x^2-1)^2 + beta^{-1} log omega(x) + C,
    F'_ref(x) = 4 H x (x^2-1) + beta^{-1} omega'(x)/omega(x),
    Y | X=x   ~ N(0, 1/(beta omega(x)^2)).

Batching
--------
A *run* is a (config, seed, method) triple.  We stack runs as a (B, M, N)
tensor where:
  * B  indexes (config, seed) groups that SHARE initial conditions and Langevin
        noise (matched-seed comparison);
  * M  indexes methods (e.g. abf, fr_estimated) within a (config, seed) group;
  * N  is the particle count.
Per-config physical parameters broadcast as (B, 1, 1); per-method flags as
(1, M, 1).  The whole study fits comfortably on one H200; the only serial cost
is the Python step loop, which is shared across the entire batch.

NO-LEAKAGE INVARIANT
--------------------
The estimated-target method (`fr_estimated`) NEVER reads F_ref.  Its FR target
q_n(x) is built from an online EMA of the ABF bias B_n only.  F_ref is used
solely for (a) post-hoc L2 evaluation and (b) the optional, explicitly
non-deployable `fr_oracle` diagnostic.  `assert_no_oracle_leakage` enforces this.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

EPS = 1e-30

# -----------------------------------------------------------------------------
# fixed domain / grid (shared across all configs: domain does not depend on
# physical parameters, so kernels and the eval window are global)
# -----------------------------------------------------------------------------
XMIN, XMAX = -1.8, 1.8
YMIN, YMAX = -2.0, 2.0  # informational; Y is unbounded in the dynamics
N_GRID = 181
EVAL_LO, EVAL_HI = -1.5, 1.5  # interior window for the L2 errors

# conditional-fidelity probe locations and half-width
COND_CENTERS = (-1.0, -0.5, 0.0, 0.5, 1.0)
COND_HALFWIDTH = 0.05  # bin = center +/- this


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = choose_device()
DTYPE = torch.float64  # double precision: cheap here and keeps L2 errors clean
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False


# -----------------------------------------------------------------------------
# configuration dataclasses
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class PhysConfig:
    """Per-config physical + numerical parameters (one B-row)."""
    beta: float = 8.0
    H: float = 2.5
    omega_out: float = 1.0
    omega_in: float = 25.0
    s: float = 0.25
    gamma: float = 15.0          # FR birth-death rate
    N: int = 256
    dt: float = 1e-3
    n_steps: int = 40000
    save_every: int = 400
    # ABF mean-force smoothing
    h: float = 0.07
    min_count: float = 1.0
    # Online ABF mean-force estimator (object A; docs/HISTOGRAM_ABF_REPLICATION.md):
    #   'kernel'    -- the accepted engine: nearest-node accumulators on the 181-node grid,
    #                  Gaussian smoothing at bandwidth h, bias force linearly interpolated;
    #   'histogram' -- textbook P0 ABF: abf_n_bins equal bins on [XMIN, XMAX],
    #                  Gamma_j = M_j / (C_j + min_count), every walker feels its OWN bin's
    #                  value (piecewise constant, no interpolation, no smoothing).
    # Structural (uniform within a batch).  The FR marginal KDE (eta) is untouched by either.
    abf_estimator: str = "kernel"
    abf_n_bins: int = 0
    # FR knobs
    eta: float = 0.10            # marginal-density bandwidth
    fr_every: int = 10
    fr_burnin: int = 0
    ramp_fraction: float = 0.10
    target_ema_rate: float = 0.005
    score_clip: float = 3.0
    max_event_fraction: float = 0.08
    # windowed ancestor-ESS: reset lineage labels every this many steps so ESS
    # reflects RECENT diversity (ESS-from-origin coalesces to ~1 on any long run
    # regardless of gamma, so it cannot discriminate the FR-rate failure boundary).
    ess_window_steps: int = 4000


@dataclass(frozen=True)
class MethodSpec:
    """Per-method flags (one M-column)."""
    name: str
    use_fr: bool
    target_mode: str  # 'none' (abf), 'estimated', 'uniform', 'oracle'


# canonical method registry
ABF = MethodSpec("abf", use_fr=False, target_mode="none")
FR_ESTIMATED = MethodSpec("fr_estimated", use_fr=True, target_mode="estimated")
FR_UNIFORM = MethodSpec("fr_uniform", use_fr=True, target_mode="uniform")
FR_ORACLE = MethodSpec("fr_oracle", use_fr=True, target_mode="oracle")

METHOD_REGISTRY = {m.name: m for m in (ABF, FR_ESTIMATED, FR_UNIFORM, FR_ORACLE)}


def assert_no_oracle_leakage(methods: Sequence[MethodSpec]) -> None:
    """Only `fr_oracle` may consult F_ref; everything else must not."""
    for m in methods:
        if m.target_mode == "oracle":
            assert m.name == "fr_oracle", f"oracle target on non-oracle method {m.name}"
        else:
            assert m.target_mode in ("none", "estimated", "uniform"), (
                f"method {m.name} has unexpected target_mode {m.target_mode}"
            )


# -----------------------------------------------------------------------------
# grid + analytic reference (vectorized over configs)
# -----------------------------------------------------------------------------
def build_grid(device=DEVICE, dtype=DTYPE):
    x_grid = torch.linspace(XMIN, XMAX, N_GRID, device=device, dtype=dtype)
    dx = float(x_grid[1] - x_grid[0])
    eval_mask = (x_grid >= EVAL_LO) & (x_grid <= EVAL_HI)
    idx0 = int(torch.argmin(torch.abs(x_grid)).item())  # x nearest 0 (F gauge)
    return x_grid, dx, eval_mask, idx0


def omega_of(x, omega_out, omega_in, s):
    return omega_out + (omega_in - omega_out) * torch.exp(-x * x / (2.0 * s * s))


def domega_of(x, omega_out, omega_in, s):
    return -(omega_in - omega_out) * (x / (s * s)) * torch.exp(-x * x / (2.0 * s * s))


def U_of(x, Hc):
    return Hc * (x * x - 1.0) ** 2


def dU_of(x, Hc):
    return 4.0 * Hc * x * (x * x - 1.0)


def reference_profiles(x_grid, eval_mask, beta, Hc, omega_out, omega_in, s):
    """F_ref (centered on eval window) and F'_ref on the grid.

    beta, Hc, omega_* may be (B,1) tensors -> returns (B, N_GRID).
    """
    xg = x_grid.unsqueeze(0)  # (1, G)
    om = omega_of(xg, omega_out, omega_in, s)
    dom = domega_of(xg, omega_out, omega_in, s)
    F_ref = U_of(xg, Hc) + torch.log(om) / beta
    F_ref = F_ref - F_ref[:, eval_mask].mean(dim=1, keepdim=True)
    Fp_ref = dU_of(xg, Hc) + dom / (om * beta)
    return F_ref, Fp_ref


# -----------------------------------------------------------------------------
# batched numerics (all operate on a flattened run axis R = B*M)
# -----------------------------------------------------------------------------
def gaussian_kernel(bw, dx, device, dtype):
    """Match the notebook's gaussian_kernel: radius 4*bw/dx, normalized by sum*dx."""
    r = max(1, int(round(4.0 * bw / dx)))
    t = torch.arange(-r, r + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (t * dx / bw) ** 2)
    k = k / (k.sum() * dx)
    return k, r


def smooth(v, kernel, r, dx):
    """Reflect-pad + valid convolution along the grid axis, matching the
    notebook's np.convolve(pad(v,'reflect'), k_sliced, 'valid') EXACTLY.

    v: (R, G) -> (R, G).  No dx scaling: np.convolve is a bare sum-of-products,
    and the kernel is already 1/(sum*dx)-normalized.  This matters because the
    mean force adds an UNSCALED min_count to the smoothed counts, so any extra
    dx factor would change the low-count regularization.
    """
    R, G = v.shape
    pad = min(r, G - 1)
    vp = F.pad(v.unsqueeze(1), (pad, pad), mode="reflect")  # (R,1,G+2pad)
    # central (2*pad+1) slice of the kernel; flip for conv1d (cross-correlation)
    k = kernel[r - pad: kernel.numel() - (r - pad)].flip(0).view(1, 1, -1)
    out = F.conv1d(vp, k)  # (R,1,G)
    return out.squeeze(1)


def cumtrapz(y, dx):
    """Cumulative trapezoid with leading 0, along grid axis. y:(R,G)->(R,G)."""
    seg = 0.5 * (y[:, 1:] + y[:, :-1]) * dx
    out = torch.zeros_like(y)
    out[:, 1:] = torch.cumsum(seg, dim=1)
    return out


def trapz(y, dx):
    return torch.sum(0.5 * (y[:, 1:] + y[:, :-1]) * dx, dim=1)


def binned_density(X, kernel, r, dx):
    """Histogram particles onto the grid, smooth, normalize. X:(R,N)->p:(R,G)."""
    R, N = X.shape
    idx = torch.clamp(torch.round((X - XMIN) / dx).long(), 0, N_GRID - 1)
    hist = torch.zeros((R, N_GRID), device=X.device, dtype=X.dtype)
    hist.scatter_add_(1, idx, torch.ones_like(X))
    p = smooth(hist, kernel, r, dx) / float(N)
    mass = torch.clamp(trapz(p, dx), min=EPS).unsqueeze(1)
    return torch.clamp(p / mass, min=EPS)


def interp1d(X, grid_vals, dx):
    """Linear interpolation of per-row profiles at per-particle locations.

    X:(R,N) particle positions; grid_vals:(R,G) per-row profile. ->(R,N).
    Clamps to grid edges (np.interp behavior).
    """
    pos = torch.clamp((X - XMIN) / dx, 0.0, N_GRID - 1.0)
    i0 = torch.clamp(torch.floor(pos).long(), 0, N_GRID - 2)
    frac = pos - i0.to(X.dtype)
    v0 = torch.gather(grid_vals, 1, i0)
    v1 = torch.gather(grid_vals, 1, i0 + 1)
    return v0 + frac * (v1 - v0)


def reflect_into(q, lo, hi):
    """Reflect coordinates into [lo, hi] (matches the notebook's reflect())."""
    span = hi - lo
    qm = torch.remainder(q - lo, 2.0 * span)
    return torch.where(qm > span, 2.0 * span - qm, qm) + lo


def fr_target_from(F_target, B, beta, dx):
    """q(x) ~ exp(-beta (F_target - B)), centered + normalized. (R,G)->(R,G)."""
    e = -beta * (F_target - B)
    e = e - e.max(dim=1, keepdim=True).values
    q = torch.exp(e)
    mass = torch.clamp(trapz(q, dx), min=EPS).unsqueeze(1)
    return torch.clamp(q / mass, min=EPS)


def l2_error(a, b, eval_mask):
    """Interior-window RMS error. a,b:(R,G)->(R,)."""
    d = (a - b)[:, eval_mask]
    return torch.sqrt(torch.mean(d * d, dim=1))


def reference_mean_force(x, beta, Hc, omega_out, omega_in, s):
    """F'_ref at arbitrary positions (broadcasts x:(1,Q) against (R,1) parameters)."""
    om = omega_of(x, omega_out, omega_in, s)
    dom = domega_of(x, omega_out, omega_in, s)
    return dU_of(x, Hc) + dom / (om * beta)


# -----------------------------------------------------------------------------
# online ABF mean-force estimators (object A of the algorithm).  The FR marginal
# density (object B) is `binned_density` above; neither estimator touches it.
# -----------------------------------------------------------------------------
TRUST_MIN_COUNT = 10.0   # reporting threshold for "untrusted bin" diagnostics; affects no dynamics


class KernelABFEstimator:
    """The accepted estimator: nearest-node accumulators C / Sf on the 181-node grid,
    Gaussian smoothing at bandwidth h with the notebook's unscaled min_count
    regularisation, bias force linearly interpolated from the grid profile.

    Op for op the code that lived inline in ``simulate_batch`` before the histogram
    estimator existed (tests/test_histogram_abf.py pins the outputs to a fixture
    generated at that commit)."""

    def __init__(self, R, h, min_count, dx, device, dtype):
        self.dx, self.min_count = dx, min_count
        self.k_h, self.r_h = gaussian_kernel(h, dx, device, dtype)
        self.C = torch.zeros((R, N_GRID), device=device, dtype=dtype)
        self.Sf = torch.zeros((R, N_GRID), device=device, dtype=dtype)

    def update(self, X, fx):
        idx = torch.clamp(torch.round((X - XMIN) / self.dx).long(), 0, N_GRID - 1)
        self.C.scatter_add_(1, idx, torch.ones_like(X))
        self.Sf.scatter_add_(1, idx, fx)

    def mean_force_profile(self):
        return smooth(self.Sf, self.k_h, self.r_h, self.dx) / (
            smooth(self.C, self.k_h, self.r_h, self.dx) + self.min_count + EPS)

    def pmf_profile(self, Fp, idx0):
        B = cumtrapz(Fp, self.dx)
        return B - B[:, idx0:idx0 + 1]

    def evaluate(self, X, Fp):
        return interp1d(X, Fp, self.dx)


def histogram_bin_index(x, lo, delta, n_bins):
    """Bin of ``x``: bin j = [e_j, e_{j+1}) with e_j = lo + j*delta, the last bin closed at
    the right end; values outside [lo, hi] land in the boundary bins.  The 1e-9 nudge makes a
    point sitting exactly on an edge (every grid node, on this grid) belong to the bin to its
    RIGHT as the [left, right) convention says, independent of floating-point rounding."""
    return torch.clamp(torch.floor((x - lo) / delta + 1e-9).long(), 0, n_bins - 1)


class HistogramABFEstimator:
    """Textbook (Darve-Pohorille) ABF, batched over the run axis R.

    ``n_bins`` equal bins on [XMIN, XMAX]; per bin the cumulative count ``C_j`` and force
    sum ``M_j`` (online: identify bin, C += 1, M += f; no sample is ever revisited);
    ``Gamma_j = M_j / (C_j + min_count)`` -- the engine's existing soft ramp (min_count = 1 in
    the accepted configuration; min_count = 0 gives exactly M_j / C_j on populated bins and
    0 on empty ones).  ``evaluate`` returns the value of the walker's OWN bin: piecewise
    constant, no interpolation, no smoothing.

    Own read-out: ``pmf_profile`` is the EXACT integral of the piecewise-constant force
    (piecewise linear, continuous) at the grid nodes, gauged at ``idx0`` like the kernel arm;
    ``fp_error`` is the function-space RMS against a reference sampled on an ``n_sub``-point
    midpoint sub-grid per bin (never on an edge) -- the primary F' error;
    ``mean_force_profile`` is the P0 value at the nodes with the [left, right) rule (on this
    grid every node is a bin edge, so that is the secondary read-out only)."""

    def __init__(self, R, n_bins, min_count, x_grid, device, dtype, n_sub=8):
        assert int(n_bins) > 0, "histogram estimator needs abf_n_bins > 0"
        self.n_bins, self.min_count = int(n_bins), float(min_count)
        self.lo, self.hi = XMIN, XMAX
        self.delta = (XMAX - XMIN) / self.n_bins
        self.edges = XMIN + self.delta * torch.arange(self.n_bins + 1, device=device, dtype=dtype)
        self.centres = 0.5 * (self.edges[1:] + self.edges[:-1])
        self.C = torch.zeros((R, self.n_bins), device=device, dtype=dtype)
        self.M = torch.zeros((R, self.n_bins), device=device, dtype=dtype)
        # grid node -> bin, and the node's offset from its bin's left edge (exact integral)
        self.g2h = histogram_bin_index(x_grid, XMIN, self.delta, self.n_bins)
        self.node_offset = x_grid - self.edges[self.g2h]
        # Gauss-Legendre sub-grid (n_sub nodes per bin, never on an edge) for the function-space
        # F' error; an 8-point midpoint rule would under-read the RMS of a locally linear
        # residual by 1/64 in MSE, the composite Gauss rule is exact for it
        self.n_sub = int(n_sub)
        gl_x, gl_w = np.polynomial.legendre.leggauss(self.n_sub)
        k = torch.as_tensor(0.5 * (gl_x + 1.0), device=device, dtype=dtype)
        self.sub_x = (self.edges[:-1, None] + self.delta * k[None, :]).reshape(-1)
        self.sub_w = torch.as_tensor(0.5 * gl_w, device=device, dtype=dtype).repeat(self.n_bins)
        self.sub2h = torch.arange(self.n_bins, device=device).repeat_interleave(self.n_sub)

    def update(self, X, fx):
        idx = histogram_bin_index(X, self.lo, self.delta, self.n_bins)
        self.C.scatter_add_(1, idx, torch.ones_like(X))
        self.M.scatter_add_(1, idx, fx)

    def bin_mean_force(self):
        den = self.C + self.min_count
        safe = torch.where(den > 0, den, torch.ones_like(den))
        return torch.where(den > 0, self.M / safe, torch.zeros_like(den))

    def evaluate(self, X, Fp_bins=None):
        Fp_bins = self.bin_mean_force() if Fp_bins is None else Fp_bins
        return torch.gather(Fp_bins, 1, histogram_bin_index(X, self.lo, self.delta, self.n_bins))

    def mean_force_profile(self, Fp_bins=None):
        Fp_bins = self.bin_mean_force() if Fp_bins is None else Fp_bins
        return Fp_bins[:, self.g2h]

    def pmf_profile(self, idx0, Fp_bins=None):
        Fp_bins = self.bin_mean_force() if Fp_bins is None else Fp_bins
        F_edges = torch.zeros((Fp_bins.shape[0], self.n_bins + 1), device=Fp_bins.device,
                              dtype=Fp_bins.dtype)
        F_edges[:, 1:] = torch.cumsum(Fp_bins * self.delta, dim=1)
        F_nodes = F_edges[:, self.g2h] + Fp_bins[:, self.g2h] * self.node_offset
        return F_nodes - F_nodes[:, idx0:idx0 + 1]

    def fp_error(self, Fp_ref_sub, sub_mask, Fp_bins=None):
        """RMS of Gamma - F'_ref over the sub-points selected by ``sub_mask``.
        Fp_ref_sub: (R, Q) or (1, Q) reference values on ``self.sub_x``."""
        Fp_bins = self.bin_mean_force() if Fp_bins is None else Fp_bins
        d = (Fp_bins[:, self.sub2h] - Fp_ref_sub)[:, sub_mask]
        w = self.sub_w[sub_mask]
        return torch.sqrt(torch.sum(d * d * w, dim=1) / w.sum())

    def window_bins(self, lo, hi):
        return (self.centres >= lo) & (self.centres <= hi)


def p0_projection_floor(n_bins, x_grid, eval_mask, idx0, beta, Hc, omega_out, omega_in, s,
                        device=DEVICE, dtype=DTYPE, n_avg=64):
    """Deterministic discretisation floor of the P0 estimator: bin averages of the analytic
    F'_ref (``n_avg``-point midpoint rule per bin), the exact P0 profile, its exact integral,
    and the own-read-out e_F / e_F' conventions.  No simulation."""
    est = HistogramABFEstimator(1, n_bins, 0.0, x_grid, device, dtype)
    k = (torch.arange(n_avg, device=device, dtype=dtype) + 0.5) / n_avg
    xa = (est.edges[:-1, None] + est.delta * k[None, :])                       # (n_bins, n_avg)
    Fp_bins = reference_mean_force(xa, beta, Hc, omega_out, omega_in, s).mean(dim=1).unsqueeze(0)
    F_ref, Fp_ref = reference_profiles(x_grid, eval_mask, beta, Hc, omega_out, omega_in, s)
    Fp_ref_sub = reference_mean_force(est.sub_x.unsqueeze(0), beta, Hc, omega_out, omega_in, s)
    sub_mask = (est.sub_x >= EVAL_LO) & (est.sub_x <= EVAL_HI)
    B = est.pmf_profile(idx0, Fp_bins)
    Bc = B - B[:, eval_mask].mean(dim=1, keepdim=True)
    return dict(n_bins=int(n_bins), delta=float(est.delta),
                floor_l2_f=float(l2_error(Bc, F_ref, eval_mask)[0]),
                floor_l2_fp=float(est.fp_error(Fp_ref_sub, sub_mask, Fp_bins)[0]),
                floor_l2_fp_nodes=float(l2_error(est.mean_force_profile(Fp_bins), Fp_ref, eval_mask)[0]),
                Fp_bins=Fp_bins[0].detach().cpu().numpy(), F_nodes=Bc[0].detach().cpu().numpy())


# -----------------------------------------------------------------------------
# batched Fisher-Rao birth-death resampling
# -----------------------------------------------------------------------------
def fr_resample_indices(S, fr_mask, g, dt_fr, cap, gen):
    """Vectorized port of the notebook's fr_resample, returning a gather index.

    Faithfully reproduces, per run-row r:
      die   = (S>0) & (u < 1-exp(-g S dt_fr))        # over-represented -> kill
      clone = (S<0) & (u < 1-exp( g S dt_fr))        # under-represented -> copy
      proportional cap of (#die,#clone) to floor(max_event_fraction*N)
      new   = survivors ++ clones, then pad-from-survivors / random-subsample to N

    Args:
      S:       (R, N) clipped, kl-recentered scores.
      fr_mask: (R,)   True where the method applies FR this step.
      g:       (R, 1) ramped gamma per row.
      dt_fr:   scalar effective FR timestep (dt * fr_every).
      cap:     (R, 1) per-row event cap = floor(max_event_fraction * N).
      gen:     torch.Generator for all stochastic draws.
    Returns:
      sel: (R, N) long index such that new_state = old_state[r, sel[r]].
           For non-FR rows sel is the identity (arange).
    """
    R, N = S.shape
    dev, dt = S.device, S.dtype

    u = torch.rand((R, N), device=dev, dtype=dt, generator=gen)
    pos = S > 0
    neg = S < 0
    p_die = torch.clamp(1.0 - torch.exp(-g * S * dt_fr), 0.0, 1.0)
    p_clone = torch.clamp(1.0 - torch.exp(g * S * dt_fr), 0.0, 1.0)
    die = pos & (u < p_die)
    clone = neg & (u < p_clone)

    # --- proportional cap on total events ---------------------------------
    n_die = die.sum(dim=1, keepdim=True)      # (R,1)
    n_clone = clone.sum(dim=1, keepdim=True)
    nev = n_die + n_clone
    over = nev > cap
    # kd = min(round(cap * n_die / nev), n_die); kc = min(cap - kd, n_clone)
    kd_prop = torch.round(cap.to(dt) * n_die.to(dt) / torch.clamp(nev.to(dt), min=1.0)).long()
    kd_prop = torch.minimum(kd_prop, n_die)
    kc_prop = torch.minimum(cap - kd_prop, n_clone)
    kd = torch.where(over, kd_prop, n_die)    # uncapped rows keep all
    kc = torch.where(over, kc_prop, n_clone)

    # rank die/clone particles within their row by a random key, keep lowest kd/kc
    big = torch.finfo(dt).max
    dk = torch.where(die, torch.rand((R, N), device=dev, dtype=dt, generator=gen),
                     torch.full((R, N), big, device=dev, dtype=dt))
    ck = torch.where(clone, torch.rand((R, N), device=dev, dtype=dt, generator=gen),
                     torch.full((R, N), big, device=dev, dtype=dt))
    die_rank = dk.argsort(dim=1).argsort(dim=1)    # rank of each particle among its row
    clone_rank = ck.argsort(dim=1).argsort(dim=1)
    die = die & (die_rank < kd)
    clone = clone & (clone_rank < kc)
    surv = ~die

    # --- build candidate pool (R, 2N): survivors (slots 0..N-1) ++ clones --
    ar = torch.arange(N, device=dev).unsqueeze(0).expand(R, N)
    surv_idx = torch.where(surv, ar, torch.full_like(ar, -1))
    clone_idx = torch.where(clone, ar, torch.full_like(ar, -1))
    pool = torch.cat([surv_idx, clone_idx], dim=1)            # (R, 2N)
    valid = pool >= 0
    keys = torch.where(valid,
                       torch.rand((R, 2 * N), device=dev, dtype=dt, generator=gen),
                       torch.full((R, 2 * N), big, device=dev, dtype=dt))
    order = keys.argsort(dim=1)[:, :N]                        # first N by key
    sel = torch.gather(pool, 1, order)                       # (R, N), valids first, -1 tail
    valid_count = valid.sum(dim=1, keepdim=True)             # = (N - n_die) + n_clone

    # --- pad the (deficit) -1 tail by sampling survivors WITH replacement, to
    # match the notebook's rng.choice(surv, deficit, replace=True). surv_perm
    # lists survivors (random order) in its first n_surv slots; drawing a uniform
    # rank in [0, n_surv) per slot is a with-replacement survivor draw. ---
    sk = torch.where(surv, torch.rand((R, N), device=dev, dtype=dt, generator=gen),
                     torch.full((R, N), big, device=dev, dtype=dt))
    surv_perm = sk.argsort(dim=1)                            # survivors first, random order
    n_surv = surv.sum(dim=1, keepdim=True).to(dt)           # (R,1) >= 0.92*N > 0
    rand_rank = torch.clamp((torch.rand((R, N), device=dev, dtype=dt, generator=gen)
                             * n_surv).long(), max=N - 1)   # in [0, n_surv) per slot
    invalid_slot = ar >= valid_count                        # contiguous -1 tail
    pad = torch.gather(surv_perm, 1, rand_rank)
    sel = torch.where(invalid_slot, pad, sel)

    # non-FR rows: identity gather and zero event diagnostics.
    active = fr_mask.unsqueeze(1)
    sel = torch.where(active, sel, ar)
    die = torch.where(active, die, torch.zeros_like(die))
    clone = torch.where(active, clone, torch.zeros_like(clone))
    return sel, die, clone


# -----------------------------------------------------------------------------
# the batched simulation
# -----------------------------------------------------------------------------
@dataclass
class BatchSpec:
    """A batched call: B (config, seed) rows x M methods (flattened to R=B*M).

    The B-axis indexes arbitrary (config, seed) pairs that run together in one
    vectorized stepping loop.  Within each B-row, all M methods SHARE initial
    conditions and Langevin noise (the matched-seed comparison: ABF vs FR on the
    same trajectory realization).  Different B-rows get independent noise slices.
    `configs` and `seeds` are parallel lists of length B.
    """
    configs: Sequence[PhysConfig]
    seeds: Sequence[int]
    methods: Sequence[MethodSpec]
    batch_seed: int = 12345  # seeds the shared noise/fr generators for this call

    def __post_init__(self):
        assert len(self.configs) == len(self.seeds), "configs and seeds must align"


def _per_config_tensor(configs, attr, device, dtype):
    return torch.tensor([getattr(c, attr) for c in configs], device=device, dtype=dtype)


def init_conditions_batched(seeds, N, beta_b, omega_out_b, omega_in_b, s_b, device, dtype):
    """Per-(B-row) initial conditions, one row per (config, seed).

    Reproduces the notebook init per seed:
        rng_i = default_rng(1000+seed)
        X0 = reflect(rng_i.normal(-1, 0.05, N))      # config-independent
        Y0 = rng_i.normal(0,1,N) * sqrt(1/(beta omega(X0)^2))   # config scale
    seeds: length-B list.  beta_b etc: (B,) tensors.  Returns X0,Y0 (B,N).
    """
    B = len(seeds)
    X0 = torch.empty((B, N), device=device, dtype=dtype)
    Z0 = torch.empty((B, N), device=device, dtype=dtype)
    for b, sd in enumerate(seeds):
        rng_i = np.random.default_rng(1000 + int(sd))
        X0[b] = reflect_into(torch.as_tensor(rng_i.normal(-1.0, 0.05, N), device=device, dtype=dtype),
                             XMIN, XMAX)
        Z0[b] = torch.as_tensor(rng_i.normal(0.0, 1.0, N), device=device, dtype=dtype)
    om0 = omega_of(X0, omega_out_b.unsqueeze(1), omega_in_b.unsqueeze(1), s_b.unsqueeze(1))
    Y0 = Z0 * torch.sqrt(1.0 / (beta_b.unsqueeze(1) * om0 ** 2))
    return X0, Y0


def simulate_batch(spec: BatchSpec, device=DEVICE, dtype=DTYPE,
                   noise_seed_base=2000, fr_seed_base=3000, progress=None):
    """Run all (config, method) pairs for one seed. Returns a dict of results.

    Uniform-across-batch (asserted): N, dt, n_steps, save_every, fr_every,
    fr_burnin, ramp_fraction, h, eta, min_count, abf_estimator, abf_n_bins.
    Per-config (may vary): beta, H, omega_out, omega_in, s, gamma,
    target_ema_rate, score_clip, max_event_fraction.

    ``abf_estimator == 'histogram'`` swaps object A (the online mean-force estimator) for
    :class:`HistogramABFEstimator`; the grid profile / PMF are then rebuilt only at saves
    (and every step only when an estimated or oracle target needs them).  The FR block
    below is byte-identical under either estimator.
    """
    assert_no_oracle_leakage(spec.methods)
    cfgs, methods = list(spec.configs), list(spec.methods)
    B, M = len(cfgs), len(methods)
    R = B * M

    # uniform structural params
    c0 = cfgs[0]
    for c in cfgs:
        for a in ("N", "dt", "n_steps", "save_every", "fr_every", "fr_burnin",
                  "ramp_fraction", "h", "eta", "min_count", "abf_estimator", "abf_n_bins"):
            assert getattr(c, a) == getattr(c0, a), f"non-uniform {a} across configs"
    assert c0.abf_estimator in ("kernel", "histogram"), c0.abf_estimator
    use_hist = c0.abf_estimator == "histogram"
    N, dt, n_steps = c0.N, c0.dt, c0.n_steps
    save_every, fr_every, fr_burnin = c0.save_every, c0.fr_every, c0.fr_burnin
    ramp = int(c0.ramp_fraction * n_steps)
    dt_fr = dt * fr_every

    x_grid, dx, eval_mask, idx0 = build_grid(device, dtype)
    k_eta, r_eta = gaussian_kernel(c0.eta, dx, device, dtype)      # FR marginal KDE (object B)

    # per-config (B,) params, then broadcast to per-run (R,1) over the M axis
    def cfg_b(attr):
        return _per_config_tensor(cfgs, attr, device, dtype)
    beta_b = cfg_b("beta"); H_b = cfg_b("H")
    oout_b = cfg_b("omega_out"); oin_b = cfg_b("omega_in"); s_b = cfg_b("s")
    gamma_b = cfg_b("gamma"); ema_b = cfg_b("target_ema_rate")
    clip_b = cfg_b("score_clip"); maxfrac_b = cfg_b("max_event_fraction")

    def to_run(t_b):  # (B,) -> (R,1), repeating each config across M methods
        return t_b.repeat_interleave(M).unsqueeze(1)
    beta = to_run(beta_b); Hc = to_run(H_b)
    oout = to_run(oout_b); oin = to_run(oin_b); sw = to_run(s_b)
    gamma_r = to_run(gamma_b); ema = to_run(ema_b)
    clip_r = to_run(clip_b); maxfrac_r = to_run(maxfrac_b)
    cap_r = torch.floor(maxfrac_r * N).long()
    noise_amp = torch.sqrt(2.0 * dt / beta)            # (R,1)

    # method flags (1,M)->(R,1) by tiling over B
    use_fr_m = torch.tensor([m.use_fr for m in methods], device=device)
    fr_mask = use_fr_m.repeat(B)                        # (R,) bool
    target_mode = [m.target_mode for m in methods]      # length M
    # per-run target-mode masks (B blocks of M): repeat the M-pattern B times
    is_uniform = torch.tensor([m == "uniform" for m in target_mode], device=device).repeat(B)
    is_oracle = torch.tensor([m == "oracle" for m in target_mode], device=device).repeat(B)

    # analytic reference per run (centered F_ref + F'_ref); also used for oracle target
    F_ref_b, Fp_ref_b = reference_profiles(x_grid, eval_mask, beta_b.unsqueeze(1),
                                           H_b.unsqueeze(1), oout_b.unsqueeze(1),
                                           oin_b.unsqueeze(1), s_b.unsqueeze(1))
    F_ref = F_ref_b.repeat_interleave(M, dim=0)         # (R,G)
    Fp_ref = Fp_ref_b.repeat_interleave(M, dim=0)

    # initial conditions: one row per (config, seed) on the B-axis; methods within
    # a B-row share init (and, below, Langevin noise) -> matched-seed comparison.
    X0_b, Y0_b = init_conditions_batched(spec.seeds, N, beta_b, oout_b, oin_b, s_b, device, dtype)
    X = X0_b.repeat_interleave(M, dim=0).clone()        # (R,N)
    Y = Y0_b.repeat_interleave(M, dim=0).clone()
    anc = torch.arange(N, device=device).unsqueeze(0).expand(R, N).clone()
    ess_window = c0.ess_window_steps

    # object A: the online ABF mean-force estimator
    if use_hist:
        est = HistogramABFEstimator(R, c0.abf_n_bins, c0.min_count, x_grid, device, dtype)
        # analytic F'_ref on the estimator's midpoint sub-grid (own F' error) + window masks
        Fp_ref_sub = reference_mean_force(est.sub_x.unsqueeze(0), beta, Hc, oout, oin, sw)
        sub_mask = (est.sub_x >= EVAL_LO) & (est.sub_x <= EVAL_HI)
        win_bins = est.window_bins(EVAL_LO, EVAL_HI)
        # the grid profile / PMF are needed every step only by estimated / oracle targets
        need_profiles = any(m.target_mode in ("estimated", "oracle") for m in methods)
        Fp_bins = est.bin_mean_force()
    else:
        est = KernelABFEstimator(R, c0.h, c0.min_count, dx, device, dtype)
    Fp = torch.zeros((R, N_GRID), device=device, dtype=dtype)
    Bbias = torch.zeros((R, N_GRID), device=device, dtype=dtype)
    F_target = torch.zeros((R, N_GRID), device=device, dtype=dtype)
    max_abs_bias = torch.zeros(R, device=device, dtype=dtype)

    # generators: Langevin noise drawn per B-row then broadcast across the M
    # methods of that row (so matched methods see identical noise); FR draws use
    # a separate stream.  Seeded by batch_seed (the B-rows already differ by their
    # init; distinct batch_seed per call decorrelates repeated identical batches).
    gen_n = torch.Generator(device=device); gen_n.manual_seed(noise_seed_base + spec.batch_seed)
    gen_f = torch.Generator(device=device); gen_f.manual_seed(fr_seed_base + spec.batch_seed)

    # diagnostics buffers
    save_steps = [st for st in range(n_steps) if st % save_every == 0 or st == n_steps - 1]
    n_saves = len(save_steps)
    ts_l2f = torch.zeros((R, n_saves), device=device, dtype=dtype)
    ts_l2fp = torch.zeros((R, n_saves), device=device, dtype=dtype)
    ts_ess = torch.zeros((R, n_saves), device=device, dtype=dtype)
    if use_hist:
        ts_l2fp_nodes = torch.zeros((R, n_saves), device=device, dtype=dtype)
        ts_Ch = torch.zeros((R, n_saves, est.n_bins), device=device, dtype=dtype)
        ts_Mh = torch.zeros((R, n_saves, est.n_bins), device=device, dtype=dtype)
    save_set = set(save_steps); save_ptr = 0
    tot_die = torch.zeros(R, device=device, dtype=dtype)
    tot_clone = torch.zeros(R, device=device, dtype=dtype)
    n_fr_apply = 0

    xg = x_grid  # (G,)
    for step in range(n_steps):
        # ---- windowed ancestor-ESS: reset lineage labels at window starts ----
        if ess_window > 0 and step % ess_window == 0:
            anc = torch.arange(N, device=device).unsqueeze(0).expand(R, N).clone()
        # ---- forces ----
        om = omega_of(X, oout, oin, sw)
        dom = domega_of(X, oout, oin, sw)
        fx = dU_of(X, Hc) + om * dom * Y * Y          # dV/dx
        fy = om * om * Y                              # dV/dy

        # ---- ABF accumulation + mean force + bias ----
        est.update(X, fx)
        if use_hist:
            Fp_bins = est.bin_mean_force()
            if need_profiles or step in save_set:
                Fp = est.mean_force_profile(Fp_bins)
                Bbias = est.pmf_profile(idx0, Fp_bins)
                if need_profiles:
                    F_target = (1.0 - ema) * F_target + ema * Bbias
        else:
            Fp = est.mean_force_profile()
            Bbias = est.pmf_profile(Fp, idx0)
            F_target = (1.0 - ema) * F_target + ema * Bbias

        # ---- Langevin step (noise shared across methods via B-block broadcast) ----
        zx = torch.randn((B, N), device=device, dtype=dtype, generator=gen_n).repeat_interleave(M, dim=0)
        zy = torch.randn((B, N), device=device, dtype=dtype, generator=gen_n).repeat_interleave(M, dim=0)
        # applied ABF mean force at X: own-bin value (histogram) / interpolated grid profile (kernel)
        bias_force = est.evaluate(X, Fp_bins) if use_hist else est.evaluate(X, Fp)
        max_abs_bias = torch.maximum(max_abs_bias, bias_force.abs().max(dim=1).values)
        Xp = reflect_into(X + (-fx + bias_force) * dt + noise_amp * zx, XMIN, XMAX)
        Yp = Y + (-fy) * dt + noise_amp * zy

        # ---- Fisher-Rao birth-death ----
        do_fr = (step >= fr_burnin) and ((step - fr_burnin) % fr_every == 0)
        if do_fr and fr_mask.any():
            if ramp > 0:
                g = gamma_r * (1.0 - math.exp(-max((step - fr_burnin) / ramp, 0.0)))
            else:
                g = gamma_r
            p = binned_density(Xp, k_eta, r_eta, dx)            # (R,G)
            # build q per target mode -- vectorized, then select by row mode.
            # NO-LEAKAGE: q_oracle reads F_ref but is selected ONLY into oracle rows.
            q_est = fr_target_from(F_target, Bbias, beta, dx)
            qu = torch.ones((1, N_GRID), device=device, dtype=dtype)
            q_uni = (qu / torch.clamp(trapz(qu, dx), min=EPS)).expand(R, N_GRID)
            q = q_est  # default (estimated); also harmless filler for 'none' rows
            if any(m == "uniform" for m in target_mode):
                q = torch.where(is_uniform.unsqueeze(1), q_uni, q)
            if any(m == "oracle" for m in target_mode):
                q_orc = fr_target_from(F_ref, Bbias, beta, dx)
                q = torch.where(is_oracle.unsqueeze(1), q_orc, q)
            logp = torch.log(torch.clamp(p, min=EPS))
            logq = torch.log(torch.clamp(q, min=EPS))
            kl = trapz(p * (logp - logq), dx).unsqueeze(1)      # (R,1)
            S = (torch.log(torch.clamp(interp1d(Xp, p, dx), min=EPS))
                 - torch.log(torch.clamp(interp1d(Xp, q, dx), min=EPS)) - kl)
            S = torch.clamp(S, -clip_r, clip_r)
            sel, die, clone = fr_resample_indices(S, fr_mask, g, dt_fr, cap_r, gen_f)
            Xp = torch.gather(Xp, 1, sel)
            Yp = torch.gather(Yp, 1, sel)
            anc = torch.gather(anc, 1, sel)
            tot_die += die.sum(dim=1).to(dtype)
            tot_clone += clone.sum(dim=1).to(dtype)
            n_fr_apply += 1

        X, Y = Xp, Yp

        # ---- diagnostics ----
        if step in save_set:
            Bc = Bbias - Bbias[:, eval_mask].mean(dim=1, keepdim=True)
            ts_l2f[:, save_ptr] = l2_error(Bc, F_ref, eval_mask)
            if use_hist:
                # own read-out: function-space F' error (primary) + node-sampled (secondary)
                ts_l2fp[:, save_ptr] = est.fp_error(Fp_ref_sub, sub_mask, Fp_bins)
                ts_l2fp_nodes[:, save_ptr] = l2_error(Fp, Fp_ref, eval_mask)
                ts_Ch[:, save_ptr] = est.C
                ts_Mh[:, save_ptr] = est.M
            else:
                ts_l2fp[:, save_ptr] = l2_error(Fp, Fp_ref, eval_mask)
            ts_ess[:, save_ptr] = ancestor_ess(anc, N)
            save_ptr += 1
        if progress is not None and step % progress == 0:
            print(f"    seed {spec.seed} step {step}/{n_steps}", flush=True)

    return _finalize(locals())


# -----------------------------------------------------------------------------
# diagnostics: ancestor ESS and conditional-variance fidelity
# -----------------------------------------------------------------------------
def ancestor_ess(anc, N):
    """Effective number of distinct ancestors: (sum n_a)^2 / sum n_a^2.

    anc:(R,N) ancestor ids in [0,N). Returns (R,). ESS=N when no resampling,
    ESS->1 under diversity collapse.
    """
    R = anc.shape[0]
    counts = torch.zeros((R, N), device=anc.device, dtype=torch.float64)
    counts.scatter_add_(1, anc, torch.ones_like(anc, dtype=torch.float64))
    num = counts.sum(dim=1) ** 2
    den = torch.clamp((counts * counts).sum(dim=1), min=EPS)
    return (num / den).to(anc.dtype)


def conditional_variance_diagnostics(X, Y, beta, oout, oin, sw):
    """Empirical Var(Y | X in bin) vs analytic 1/(beta omega(x)^2) at COND_CENTERS.

    X,Y:(R,N); beta,oout,oin,sw:(R,1). Returns dict of (R, n_centers) tensors:
    emp_var, ref_var, abs_err. Bins with <2 samples give NaN emp_var.
    """
    R, N = X.shape
    centers = torch.tensor(COND_CENTERS, device=X.device, dtype=X.dtype)  # (K,)
    K = centers.numel()
    lo = centers - COND_HALFWIDTH
    hi = centers + COND_HALFWIDTH
    Xe = X.unsqueeze(2)                      # (R,N,1)
    inbin = (Xe >= lo.view(1, 1, K)) & (Xe < hi.view(1, 1, K))  # (R,N,K)
    w = inbin.to(X.dtype)
    cnt = w.sum(dim=1)                        # (R,K)
    Yexp = Y.unsqueeze(2)
    mean = (w * Yexp).sum(dim=1) / torch.clamp(cnt, min=1.0)
    var = (w * (Yexp - mean.unsqueeze(1)) ** 2).sum(dim=1) / torch.clamp(cnt - 1.0, min=1.0)
    emp_var = torch.where(cnt >= 2.0, var, torch.full_like(var, float("nan")))
    om_c = omega_of(centers.view(1, K), oout, oin, sw)          # (R,K)
    ref_var = 1.0 / (beta * om_c ** 2)
    return {
        "cond_centers": centers,
        "cond_emp_var": emp_var,
        "cond_ref_var": ref_var,
        "cond_abs_err": torch.abs(emp_var - ref_var),
        "cond_count": cnt,
    }


# -----------------------------------------------------------------------------
# finalize: pull the batched run results into per-run numpy records
# -----------------------------------------------------------------------------
def _finalize(L):
    """Assemble per-run result records from simulate_batch local scope L."""
    cfgs, methods = L["cfgs"], L["methods"]
    B, M, R, N = L["B"], L["M"], L["R"], L["N"]
    dx, eval_mask, x_grid = L["dx"], L["eval_mask"], L["x_grid"]
    X, Y, anc = L["X"], L["Y"], L["anc"]
    Fp, Bbias, F_target = L["Fp"], L["Bbias"], L["F_target"]
    F_ref, Fp_ref = L["F_ref"], L["Fp_ref"]
    beta, oout, oin, sw = L["beta"], L["oout"], L["oin"], L["sw"]
    save_steps = L["save_steps"]
    dt = L["dt"]

    # centered final F_hat
    Bc = Bbias - Bbias[:, eval_mask].mean(dim=1, keepdim=True)
    p_hat = binned_density(X, L["k_eta"], L["r_eta"], dx)
    # FR target snapshot (estimated) for record (q for non-fr rows is informational)
    q_final = fr_target_from(F_target, Bbias, beta, dx)
    cond = conditional_variance_diagnostics(X, Y, beta, oout, oin, sw)

    ts_l2f = L["ts_l2f"]; ts_l2fp = L["ts_l2fp"]; ts_ess = L["ts_ess"]
    t_axis = np.array([st * dt for st in save_steps])
    # integrated L2(F) over time via trapezoid on the save grid
    seg = 0.5 * (ts_l2f[:, 1:] + ts_l2f[:, :-1]) * torch.tensor(
        np.diff(t_axis), device=ts_l2f.device, dtype=ts_l2f.dtype)
    int_l2f = seg.sum(dim=1)

    # move to cpu numpy
    def npy(t):
        return t.detach().cpu().numpy()

    recs = []
    for b in range(B):
        for m in range(M):
            r = b * M + m
            use_fr = methods[m].use_fr
            n_die = float(L["tot_die"][r]) if use_fr else 0.0
            n_clone = float(L["tot_clone"][r]) if use_fr else 0.0
            n_fr_apply = int(L["n_fr_apply"]) if use_fr else 0
            repl_fraction = float((n_die + n_clone) / max(n_fr_apply * N, 1)) if use_fr else 0.0
            rec = {
                "config": asdict(cfgs[b]),
                "method": methods[m].name,
                "target_mode": methods[m].target_mode,
                "seed": int(L["spec"].seeds[b]),
                "final_l2_f": float(ts_l2f[r, -1]),
                "final_l2_fp": float(ts_l2fp[r, -1]),
                "int_l2_f": float(int_l2f[r]),
                "t": t_axis,
                "l2_f_t": npy(ts_l2f[r]),
                "l2_fp_t": npy(ts_l2fp[r]),
                "ess_t": npy(ts_ess[r]),
                "final_ess": float(ts_ess[r, -1]),
                "x_grid": npy(x_grid),
                "F_hat": npy(Bc[r]),
                "Fp_hat": npy(Fp[r]),
                "F_ref": npy(F_ref[r]),
                "Fp_ref": npy(Fp_ref[r]),
                "p_hat": npy(p_hat[r]),
                "q_target": npy(q_final[r]) if use_fr else None,
                "n_die": n_die,
                "n_clone": n_clone,
                "n_fr_apply": n_fr_apply,
                "repl_fraction": repl_fraction,
                "cond_centers": npy(cond["cond_centers"]),
                "cond_emp_var": npy(cond["cond_emp_var"][r]),
                "cond_ref_var": npy(cond["cond_ref_var"][r]),
                "cond_abs_err": npy(cond["cond_abs_err"][r]),
                "cond_count": npy(cond["cond_count"][r]),
                "abf_estimator": cfgs[b].abf_estimator,
                "max_abs_bias_force": float(L["max_abs_bias"][r]),
            }
            if L.get("use_hist"):
                est = L["est"]
                Fp_bins, Cb, win = est.bin_mean_force(), est.C, L["win_bins"]
                rec.update({
                    "abf_n_bins": est.n_bins,
                    "hist_delta": float(est.delta),
                    "hist_edges": npy(est.edges),
                    "Fp_bins": npy(Fp_bins[r]),
                    "C_bins": npy(Cb[r]),
                    "M_bins": npy(est.M[r]),
                    "Ch_t": npy(L["ts_Ch"][r]),
                    "Mh_t": npy(L["ts_Mh"][r]),
                    "l2_fp_nodes_t": npy(L["ts_l2fp_nodes"][r]),
                    "final_l2_fp_nodes": float(L["ts_l2fp_nodes"][r, -1]),
                    "min_count_window": float(Cb[r][win].min()),
                    "frac_untrusted_window": float((Cb[r][win] < TRUST_MIN_COUNT).to(Cb.dtype).mean()),
                    "trust_min_count": TRUST_MIN_COUNT,
                })
            recs.append(rec)
    return recs
