"""Entropic-gateway engine: an establishment-limited regime built on purpose.

Every molecular system tried so far landed in one of two regimes that say nothing about
marginal Fisher-Rao (mFR) reallocation.  Alanine and valine are *ABF-sufficient*: the slow
coordinate is in the CV, ABF flattens it, and every state is established long before the
run ends, so there is no deficit to repair.  Pentane's R15 distance CV is
*discovery-limited*: the state is never reached, and mFR cannot clone a walker that does
not exist.  The regime mFR is supposed to serve -- a state that is *found early* but
*populated slowly* -- has never been produced deliberately.

This module builds it.  The model is the smooth entropic channel already validated in
``eb_abffr_core``:

    V_{s,r}(x, y) = H (x^2 - 1)^2 + 1/2 omega_{s,r}(x)^2 y^2,
    omega_{s,r}(x) = omega_out + (omega_in - omega_out) exp(-x^2 / (2 s^2)),
    r = omega_in / omega_out > 1,      xi(x, y) = x.

with the analytic free energy

    F_{s,r}(x) = H (x^2 - 1)^2 + beta^{-1} log omega_{s,r}(x) + C.

``s`` sets the width of the constriction at ``x = 0`` and ``r`` its severity, and the two
tune discovery and establishment *independently enough to map every regime* -- which is
exactly what another peptide cannot do.  Because ``F`` is analytic there is no reference
simulation to converge, and no reference error to confound the classification.

Why this is establishment-limited rather than merely slow
---------------------------------------------------------
Under ABF the walker feels ``-dV/dx + F'_hat(x)``, so once the bias converges the barrier
is gone and transport is fast.  The bottleneck is that the bias converges *slowly at the
gateway*: the instantaneous force there carries the term ``omega omega' y^2`` whose scale
is ``(1/beta) d log omega / dx ~ log(r)/s``, with O(1) relative fluctuations from ``y^2``.
Narrow (small ``s``) and severe (large ``r``) gateways therefore have a high-variance mean
force in exactly the cells a walker passes through quickly and rarely.  A few of ``N``
walkers get across early -- discovery -- while the *rate* of independent crossings stays
low for a long time -- slow establishment.  That is the target regime, and ``(s, r)``
moves the system through it.

What this module adds over ``eb_abffr_core``
--------------------------------------------
``eb_abffr_core`` is left untouched: its artifacts are accepted and its numerics are
validated, so the primitives are imported rather than copied.  New here:

* **occupancy time series** for the three regions ``B_-``, gateway, ``B_+``, together with
  the *bias-aware* target ``Q*_k(t)`` recomputed at every save;
* ``T_hit`` / ``T_est`` on the persistence convention (a condition must hold over a
  trailing window, never at a single save);
* a **hard assertion** that the target is normalised on the reference-supported domain and
  that the observed fractions are conditioned on the same domain -- printing this number
  and moving on is how a target ends up with 97 % of its mass outside the support it was
  supposed to live on;
* initialisation modes, including the ``one_right`` mechanism control in which discovery is
  handed to the sampler for free so any remaining acceleration is *establishment*;
* the ``sham`` arm: a shadow-intensity control that fires the **same number of clone/delete
  events at the same times** as the FR arm it shadows, with the score association
  randomised.  Without it, "mFR helped" cannot be separated from "any resampling helped".
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np
import torch

import eb_abffr_core as eb
from eb_abffr_core import (  # noqa: F401  (re-exported for scripts/tests)
    DEVICE, DTYPE, EPS, N_GRID, XMIN, XMAX,
    binned_density, build_grid, cumtrapz, domega_of, dU_of, gaussian_kernel,
    interp1d, l2_error, omega_of, reflect_into, smooth, trapz, U_of,
)

# Region boundary.  |x| <= X_BASIN is the gateway corridor; outside it are the two basins,
# whose minima sit at x = -+1.  Chosen to land exactly on grid points so the grid-cell and
# per-particle region rules agree at the boundary.
X_BASIN = 0.5
REGIONS = ("minus", "gate", "plus")

# Classification thresholds.  PREDECLARED -- these are the screening plan's numbers and are
# frozen before any run; see scripts/run_gateway_phase.py.
DISCOVERY_FRAC = 0.10      # T_hit below this fraction of T is "early discovery"
EST_GAP_SUFFICIENT = 0.10  # T_est - T_hit below this fraction of T is "ABF-sufficient"
EST_GAP_LIMITED = 0.25     # ... above this fraction is "establishment-limited"
BELOW_HALF_FRAC = 0.20     # ... or below half target for at least this fraction of T
EST_BAND = (0.5, 1.5)      # established = occupancy within this multiple of the target
HOLD_FRAC = 0.05           # persistence window for T_hit / T_est
DISCOVERY_SEED_FRAC = 0.25 # "substantial fraction of seeds" for the discovery-limited call


# -----------------------------------------------------------------------------
# configuration
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class GatewayConfig:
    """One (s, r) cell.  ``omega_in`` is derived, so ``r`` is the only severity knob."""
    beta: float = 1.0
    H: float = 1.0               # energetic barrier in units of 1/beta => beta*H kT
    omega_out: float = 1.0
    r: float = 8.0               # omega_in / omega_out
    s: float = 0.15
    N: int = 2048
    dt: float = 4e-4
    n_steps: int = 250_000
    save_every: int = 1000
    init: str = "left"           # 'left' (concentrated) | 'one_right' (mechanism control)
    # ABF mean-force smoothing
    h: float = 0.07
    min_count: float = 1.0
    # FR knobs (frozen from the accepted entropic-bottleneck setup)
    gamma: float = 15.0
    eta: float = 0.10
    fr_every: int = 10
    fr_burnin: int = 0
    ramp_fraction: float = 0.10
    target_ema_rate: float = 0.005
    score_clip: float = 3.0
    max_event_fraction: float = 0.08
    ess_window_steps: int = 4000

    @property
    def omega_in(self) -> float:
        return self.omega_out * self.r

    @property
    def T_total(self) -> float:
        return self.n_steps * self.dt

    def barrier_kT(self) -> float:
        """Total free-energy barrier at the gateway, in kT: beta*H + log r."""
        return self.beta * self.H + math.log(self.r)


@dataclass(frozen=True)
class Method:
    name: str
    use_fr: bool
    target_mode: str      # 'none' | 'estimated' | 'oracle' | 'uniform'
    sham: bool = False    # randomise the score association, keep the event schedule
    shadows: str = ""     # for sham arms: the method whose realised counts are copied


ABF = Method("abf", use_fr=False, target_mode="none")
FR_ORACLE = Method("fr_oracle", use_fr=True, target_mode="oracle")
FR_ESTIMATED = Method("fr_estimated", use_fr=True, target_mode="estimated")
# One sham per FR arm.  A single sham shadowing the oracle cannot attribute the *practical*
# arm's gain: the two arms fire different numbers of events at different times (their targets
# differ), so the oracle's shadow is not a matched control for the deployable method.  Each
# sham copies its own partner's realised counts, at its own partner's event times.
SHAM_ORACLE = Method("sham_oracle", use_fr=True, target_mode="none", sham=True,
                     shadows="fr_oracle")
SHAM_PRACTICAL = Method("sham_practical", use_fr=True, target_mode="none", sham=True,
                        shadows="fr_estimated")
SHAM = SHAM_ORACLE          # backwards-compatible alias for the calibration artifacts

METHODS = {m.name: m for m in (ABF, FR_ORACLE, FR_ESTIMATED, SHAM_ORACLE, SHAM_PRACTICAL)}


def assert_no_oracle_leakage(methods: Sequence[Method]) -> None:
    """Only the explicitly non-deployable ``fr_oracle`` arm may consult ``F_ref``.

    ``sham`` computes no score at all: it copies the *counts* its partner realised and
    draws the identities uniformly, so it carries ``target_mode='none'`` and cannot leak a
    reference even by accident.
    """
    names = {m.name for m in methods}
    for m in methods:
        if m.target_mode == "oracle":
            assert m.name == "fr_oracle", f"oracle target on non-oracle method {m.name}"
        else:
            assert m.target_mode in ("none", "estimated", "uniform"), (
                f"unexpected target_mode {m.target_mode} on {m.name}")
        if m.sham:
            # A sham arm whose partner is absent would silently fall back to its own event
            # counts, which is a different (and much weaker) control than the one claimed.
            assert m.shadows in names, (
                f"sham arm {m.name!r} shadows {m.shadows!r}, which is not in this batch; "
                f"matched intensity is unobtainable without it")


# -----------------------------------------------------------------------------
# grid geometry: regions and the bias-aware target
# -----------------------------------------------------------------------------
def region_of_grid(x_grid):
    """Region index per grid point, using the SAME rule as ``region_of_particles``."""
    lab = torch.full_like(x_grid, 1, dtype=torch.long)          # gate
    lab = torch.where(x_grid < -X_BASIN, torch.zeros_like(lab), lab)
    lab = torch.where(x_grid > X_BASIN, torch.full_like(lab, 2), lab)
    return lab


def region_of_particles(X):
    lab = torch.ones_like(X, dtype=torch.long)
    lab = torch.where(X < -X_BASIN, torch.zeros_like(lab), lab)
    lab = torch.where(X > X_BASIN, torch.full_like(lab, 2), lab)
    return lab


def bias_aware_target(F_ref, B_t, glab, beta, n_regions=3):
    """``Q*_k(t)``: the ideal biased population of each region under the current bias.

    ``q*_t(x) ~ exp(-beta (F_ref(x) - B_t(x)))`` normalised over the reference-supported
    domain, then integrated over each region.  Scoring against the *unbiased* population
    instead would report a state as starved precisely when ABF has correctly flattened it,
    manufacturing the very signal mFR is meant to remove from a run in which nothing is
    wrong.

    ``F_ref`` is analytic on the whole simulation domain, so the support is the whole grid
    and the normalisation is exact.  The assertion below is not decoration: the valine
    screen shipped a version of this function that capped unsupported cells and normalised
    over them, and 97 % of the target mass landed in exactly the cells the reference knew
    nothing about.
    """
    e = -beta * (F_ref - B_t)                       # (R, G)
    e = e - e.max(dim=1, keepdim=True).values
    q = torch.exp(e)
    q = q / q.sum(dim=1, keepdim=True)              # plain Riemann sum: sums to 1 exactly
    Q = torch.stack([q[:, glab == k].sum(dim=1) for k in range(n_regions)], dim=1)
    assert_supported_target(Q, q)
    return Q


def assert_supported_target(Q, q):
    """HARD guard: the region decomposition must exhaust the supported domain.

    The valine screen printed a "capped weight" diagnostic and relied on a human noticing
    it.  A number that is only printed is a number that gets skimmed; the check that
    matters is the one that stops the run.  ``glab`` labels every grid point, so any mass
    unaccounted for is a coding error, not a physical leak.
    """
    tot = Q.sum(dim=1)
    worst = float((tot - 1.0).abs().max())
    assert worst < 1e-9, (
        f"bias-aware target does not sum to 1 on the supported domain "
        f"(worst |sum Q_k - 1| = {worst:.3e}).  Reference-unsupported target mass must be "
        f"exactly zero after conditioning, not merely small.")
    assert bool(torch.isfinite(q).all()), "non-finite bias-aware target"


# -----------------------------------------------------------------------------
# persistence-based hitting / establishment times
# -----------------------------------------------------------------------------
def first_persistent(cond, times, hold_frac=HOLD_FRAC):
    """First time at which ``cond`` holds over a whole trailing window.

    One walker brushing a basin edge for a single save interval is not a discovery, and one
    save inside the establishment band is not establishment.
    """
    n = len(times)
    hold = max(1, int(hold_frac * n))
    c = np.asarray(cond, dtype=bool)
    for i in range(n - hold + 1):
        if c[i:i + hold].all():
            return float(times[i])
    return float("nan")


def hit_and_establish(P_plus, Q_plus, times, hold_frac=HOLD_FRAC):
    """``T_hit``, ``T_est`` and the deficit summaries for the right basin."""
    t_hit = first_persistent(P_plus > 0.0, times, hold_frac)
    band = (P_plus >= EST_BAND[0] * Q_plus) & (P_plus <= EST_BAND[1] * Q_plus)
    t_est = first_persistent(band, times, hold_frac)
    T = float(times[-1])
    after = times >= (t_hit if np.isfinite(t_hit) else 0.0)
    below_half = (P_plus < 0.5 * Q_plus) & after
    dt_save = float(np.diff(times).mean()) if len(times) > 1 else 0.0
    deficit = np.clip(Q_plus - P_plus, 0.0, None)
    return dict(
        T_hit=t_hit, T_est=t_est, T_run=T,
        T_hit_frac=t_hit / T, T_est_frac=t_est / T,
        est_gap_frac=(t_est - t_hit) / T,
        below_half_frac=float(below_half.sum() / len(times)),
        # integral of the positive part of (target - observed) from discovery to the end:
        # the "how much population was missing, for how long" number.
        integrated_deficit=float(deficit[after].sum() * dt_save),
        final_occupancy=float(P_plus[-1]), final_target=float(Q_plus[-1]),
        max_rel_deficit=float(np.max(deficit[after] / np.maximum(Q_plus[after], 1e-12)))
        if after.any() else float("nan"),
    )


def classify(rows):
    """Predeclared regime label for one (s, r) cell, from its per-seed rows.

    Priority order matters and is fixed in advance: a cell whose state is not reliably
    *found* cannot be judged on how fast it fills, so discovery-limited is tested first.
    """
    n = len(rows)
    hit_frac = np.array([r["T_hit_frac"] for r in rows], dtype=float)
    gap = np.array([r["est_gap_frac"] for r in rows], dtype=float)
    below = np.array([r["below_half_frac"] for r in rows], dtype=float)
    late_or_missing = ~np.isfinite(hit_frac) | (hit_frac >= DISCOVERY_FRAC)
    if late_or_missing.mean() >= DISCOVERY_SEED_FRAC:
        return "discovery-limited"
    med_gap = float(np.nanmedian(np.where(np.isfinite(gap), gap, np.inf)))
    med_below = float(np.nanmedian(below))
    if (not np.isfinite(med_gap)) or med_gap > EST_GAP_LIMITED or med_below >= BELOW_HALF_FRAC:
        return "establishment-limited"
    if med_gap < EST_GAP_SUFFICIENT:
        return "ABF-sufficient"
    return "intermediate"


# -----------------------------------------------------------------------------
# initial conditions
# -----------------------------------------------------------------------------
def init_conditions(seeds, N, beta_b, oout_b, oin_b, s_b, inits, device, dtype):
    """One row per (config, seed).  ``init`` selects the walker placement.

    * ``left``      -- every walker in the left basin: the honest starting point for a
      discovery/establishment measurement.
    * ``one_right`` -- identical, except walker 0 starts at ``x = +1``.  Discovery is then
      guaranteed at ``t = 0``, so any acceleration measured in this arm is population
      establishment and cannot be first passage.
    """
    B = len(seeds)
    X0 = torch.empty((B, N), device=device, dtype=dtype)
    Z0 = torch.empty((B, N), device=device, dtype=dtype)
    for b, sd in enumerate(seeds):
        rng = np.random.default_rng(1000 + int(sd))
        x = rng.normal(-1.0, 0.05, N)
        if inits[b] == "one_right":
            x[0] = -x[0]                      # mirror walker 0 into the right basin
        elif inits[b] != "left":
            raise ValueError(f"unknown init {inits[b]!r}")
        X0[b] = reflect_into(torch.as_tensor(x, device=device, dtype=dtype), XMIN, XMAX)
        Z0[b] = torch.as_tensor(rng.normal(0.0, 1.0, N), device=device, dtype=dtype)
    om0 = omega_of(X0, oout_b.unsqueeze(1), oin_b.unsqueeze(1), s_b.unsqueeze(1))
    Y0 = Z0 * torch.sqrt(1.0 / (beta_b.unsqueeze(1) * om0 ** 2))
    return X0, Y0


# -----------------------------------------------------------------------------
# resampling: FR birth-death, and the sham shadow-intensity control
# -----------------------------------------------------------------------------
def resample_indices(S, fr_mask, sham_mask, partner, g, dt_fr, cap, gen):
    """Return a gather index ``sel`` with ``new = old[sel]``, plus the event masks.

    FR rows follow ``eb_abffr_core.fr_resample_indices`` exactly.  Sham rows take the event
    counts ``(kd, kc)`` **realised by their partner FR row at this same event** -- both live
    in the same batch, share initial conditions and Langevin noise, and fire on the same
    schedule -- and then draw *which* particles die and clone uniformly at random.

    Copying the partner's counts rather than re-deriving them from the sham row's own score
    matters.  A sham row's density diverges from its partner's the moment it first
    resamples, and its own score would then fire a different (in practice much larger)
    number of events; the arm would be uncontrolled in exactly the dimension it is supposed
    to control.  Here timing and intensity are matched by construction and only the
    Fisher-Rao *direction* is destroyed, which is what separates "mFR steered the
    population" from "any turnover of this magnitude would have done".
    """
    R, N = S.shape
    dev, dt = S.device, S.dtype
    ar = torch.arange(N, device=dev).unsqueeze(0).expand(R, N)

    u = torch.rand((R, N), device=dev, dtype=dt, generator=gen)
    p_die = torch.clamp(1.0 - torch.exp(-g * S * dt_fr), 0.0, 1.0)
    p_clone = torch.clamp(1.0 - torch.exp(g * S * dt_fr), 0.0, 1.0)
    die = (S > 0) & (u < p_die)
    clone = (S < 0) & (u < p_clone)

    # proportional cap on total events (identical law to the accepted engine)
    n_die = die.sum(dim=1, keepdim=True)
    n_clone = clone.sum(dim=1, keepdim=True)
    nev = n_die + n_clone
    over = nev > cap
    kd_prop = torch.round(cap.to(dt) * n_die.to(dt) / torch.clamp(nev.to(dt), min=1.0)).long()
    kd_prop = torch.minimum(kd_prop, n_die)
    kc_prop = torch.minimum(cap - kd_prop, n_clone)
    kd = torch.where(over, kd_prop, n_die)
    kc = torch.where(over, kc_prop, n_clone)

    big = torch.finfo(dt).max
    dk = torch.where(die, torch.rand((R, N), device=dev, dtype=dt, generator=gen),
                     torch.full((R, N), big, device=dev, dtype=dt))
    ck = torch.where(clone, torch.rand((R, N), device=dev, dtype=dt, generator=gen),
                     torch.full((R, N), big, device=dev, dtype=dt))
    die = die & (dk.argsort(dim=1).argsort(dim=1) < kd)
    clone = clone & (ck.argsort(dim=1).argsort(dim=1) < kc)

    # ---- sham: the PARTNER's (kd, kc), uniformly random identities ----------
    if bool(sham_mask.any()):
        kd = torch.where(sham_mask.unsqueeze(1), kd[partner], kd)
        kc = torch.where(sham_mask.unsqueeze(1), kc[partner], kc)
        rk = torch.rand((R, N), device=dev, dtype=dt, generator=gen)
        rank = rk.argsort(dim=1).argsort(dim=1)
        # the lowest-kd ranks die; the next kc are cloned, so the two sets are disjoint
        # exactly as they are in the FR arm (a particle cannot both die and be copied).
        sm = sham_mask.unsqueeze(1)
        die = torch.where(sm, rank < kd, die)
        clone = torch.where(sm, (rank >= kd) & (rank < kd + kc), clone)

    surv = ~die
    surv_idx = torch.where(surv, ar, torch.full_like(ar, -1))
    clone_idx = torch.where(clone, ar, torch.full_like(ar, -1))
    pool = torch.cat([surv_idx, clone_idx], dim=1)
    valid = pool >= 0
    keys = torch.where(valid, torch.rand((R, 2 * N), device=dev, dtype=dt, generator=gen),
                       torch.full((R, 2 * N), big, device=dev, dtype=dt))
    order = keys.argsort(dim=1)[:, :N]
    sel = torch.gather(pool, 1, order)
    valid_count = valid.sum(dim=1, keepdim=True)

    sk = torch.where(surv, torch.rand((R, N), device=dev, dtype=dt, generator=gen),
                     torch.full((R, N), big, device=dev, dtype=dt))
    surv_perm = sk.argsort(dim=1)
    n_surv = surv.sum(dim=1, keepdim=True).to(dt)
    rand_rank = torch.clamp((torch.rand((R, N), device=dev, dtype=dt, generator=gen)
                             * n_surv).long(), max=N - 1)
    sel = torch.where(ar >= valid_count, torch.gather(surv_perm, 1, rand_rank), sel)

    active = fr_mask.unsqueeze(1)
    sel = torch.where(active, sel, ar)
    die = torch.where(active, die, torch.zeros_like(die))
    clone = torch.where(active, clone, torch.zeros_like(clone))
    return sel, die, clone


def ancestor_stats(anc, N):
    """Windowed ancestor ESS and the largest lineage share ``w_max``.

    ``w_max`` is the acceptance gate that ESS alone misses: a population can retain a
    respectable ESS while one lineage quietly owns a fifth of it.
    """
    R = anc.shape[0]
    counts = torch.zeros((R, N), device=anc.device, dtype=torch.float64)
    counts.scatter_add_(1, anc, torch.ones_like(anc, dtype=torch.float64))
    ess = counts.sum(dim=1) ** 2 / torch.clamp((counts * counts).sum(dim=1), min=EPS)
    wmax = counts.max(dim=1).values / float(N)
    return ess, wmax


# -----------------------------------------------------------------------------
# the batched simulation
# -----------------------------------------------------------------------------
@dataclass
class BatchSpec:
    configs: Sequence[GatewayConfig]
    seeds: Sequence[int]
    methods: Sequence[Method]
    batch_seed: int = 12345

    def __post_init__(self):
        assert len(self.configs) == len(self.seeds), "configs and seeds must align"


def simulate_batch(spec: BatchSpec, device=DEVICE, dtype=DTYPE,
                   noise_seed_base=2000, fr_seed_base=3000, progress=None):
    """Run ``B`` (config, seed) rows x ``M`` methods, flattened to ``R = B*M``.

    Methods inside one B-row share initial conditions and Langevin noise, so an arm and its
    controls are compared on the same trajectory realisation rather than across independent
    noise.
    """
    assert_no_oracle_leakage(spec.methods)
    cfgs, methods = list(spec.configs), list(spec.methods)
    B, M = len(cfgs), len(methods)
    R = B * M

    c0 = cfgs[0]
    for c in cfgs:
        # Only the *structural* parameters must agree: they set array shapes and the step
        # loop.  beta, H, s, r and gamma are per-config and broadcast as (R,1), which is
        # what lets one batch carry a whole sweep.
        for a in ("N", "dt", "n_steps", "save_every", "fr_every", "fr_burnin",
                  "ramp_fraction", "h", "eta", "min_count"):
            assert getattr(c, a) == getattr(c0, a), f"non-uniform {a} across configs"
    N, dt, n_steps = c0.N, c0.dt, c0.n_steps
    save_every, fr_every, fr_burnin = c0.save_every, c0.fr_every, c0.fr_burnin
    ramp = int(c0.ramp_fraction * n_steps)
    dt_fr = dt * fr_every

    x_grid, dx, eval_mask, idx0 = build_grid(device, dtype)
    glab = region_of_grid(x_grid)
    k_h, r_h = gaussian_kernel(c0.h, dx, device, dtype)
    k_eta, r_eta = gaussian_kernel(c0.eta, dx, device, dtype)

    def cfg_b(fn):
        return torch.tensor([fn(c) for c in cfgs], device=device, dtype=dtype)
    beta_b = cfg_b(lambda c: c.beta); H_b = cfg_b(lambda c: c.H)
    oout_b = cfg_b(lambda c: c.omega_out); oin_b = cfg_b(lambda c: c.omega_in)
    s_b = cfg_b(lambda c: c.s); gamma_b = cfg_b(lambda c: c.gamma)
    ema_b = cfg_b(lambda c: c.target_ema_rate); clip_b = cfg_b(lambda c: c.score_clip)
    maxfrac_b = cfg_b(lambda c: c.max_event_fraction)

    def to_run(t_b):
        return t_b.repeat_interleave(M).unsqueeze(1)
    beta = to_run(beta_b); Hc = to_run(H_b)
    oout = to_run(oout_b); oin = to_run(oin_b); sw = to_run(s_b)
    gamma_r = to_run(gamma_b); ema = to_run(ema_b)
    clip_r = to_run(clip_b); maxfrac_r = to_run(maxfrac_b)
    cap_r = torch.floor(maxfrac_r * N).long()
    noise_amp = torch.sqrt(2.0 * dt / beta)

    fr_mask = torch.tensor([m.use_fr for m in methods], device=device).repeat(B)
    sham_mask = torch.tensor([m.sham for m in methods], device=device).repeat(B)
    # Row index of the arm each sham row shadows.  Methods sit inside a B-row, so the
    # partner is the same (config, seed) with a different method column -- identical
    # initial conditions and identical Langevin noise, which is what makes copying its
    # realised event counts a matched control rather than an unrelated number.
    name_col = {m.name: j for j, m in enumerate(methods)}
    partner = torch.tensor(
        [b * M + (name_col[methods[j].shadows] if methods[j].sham else j)
         for b in range(B) for j in range(M)], device=device, dtype=torch.long)
    tmode = [m.target_mode for m in methods]
    is_oracle = torch.tensor([t == "oracle" for t in tmode], device=device).repeat(B)
    is_uniform = torch.tensor([t == "uniform" for t in tmode], device=device).repeat(B)

    F_ref_b, Fp_ref_b = eb.reference_profiles(x_grid, eval_mask, beta_b.unsqueeze(1),
                                              H_b.unsqueeze(1), oout_b.unsqueeze(1),
                                              oin_b.unsqueeze(1), s_b.unsqueeze(1))
    F_ref = F_ref_b.repeat_interleave(M, dim=0)
    Fp_ref = Fp_ref_b.repeat_interleave(M, dim=0)

    inits = [c.init for c in cfgs]
    X0_b, Y0_b = init_conditions(spec.seeds, N, beta_b, oout_b, oin_b, s_b, inits,
                                 device, dtype)
    X = X0_b.repeat_interleave(M, dim=0).clone()
    Y = Y0_b.repeat_interleave(M, dim=0).clone()
    anc = torch.arange(N, device=device).unsqueeze(0).expand(R, N).clone()
    ess_window = c0.ess_window_steps

    C = torch.zeros((R, N_GRID), device=device, dtype=dtype)
    Sf = torch.zeros((R, N_GRID), device=device, dtype=dtype)
    F_target = torch.zeros((R, N_GRID), device=device, dtype=dtype)

    gen_n = torch.Generator(device=device); gen_n.manual_seed(noise_seed_base + spec.batch_seed)
    gen_f = torch.Generator(device=device); gen_f.manual_seed(fr_seed_base + spec.batch_seed)

    save_steps = [st for st in range(n_steps) if st % save_every == 0 or st == n_steps - 1]
    n_saves = len(save_steps)
    save_set, save_ptr = set(save_steps), 0
    ts_l2f = torch.zeros((R, n_saves), device=device, dtype=dtype)
    ts_l2fp = torch.zeros((R, n_saves), device=device, dtype=dtype)
    ts_ess = torch.zeros((R, n_saves), device=device, dtype=dtype)
    ts_wmax = torch.zeros((R, n_saves), device=device, dtype=dtype)
    ts_P = torch.zeros((R, n_saves, 3), device=device, dtype=dtype)
    ts_Q = torch.zeros((R, n_saves, 3), device=device, dtype=dtype)
    tot_die = torch.zeros(R, device=device, dtype=dtype)
    tot_clone = torch.zeros(R, device=device, dtype=dtype)
    n_fr_apply = 0

    for step in range(n_steps):
        if ess_window > 0 and step % ess_window == 0:
            anc = torch.arange(N, device=device).unsqueeze(0).expand(R, N).clone()

        om = omega_of(X, oout, oin, sw)
        dom = domega_of(X, oout, oin, sw)
        fx = dU_of(X, Hc) + om * dom * Y * Y
        fy = om * om * Y

        idx = torch.clamp(torch.round((X - XMIN) / dx).long(), 0, N_GRID - 1)
        C.scatter_add_(1, idx, torch.ones_like(X))
        Sf.scatter_add_(1, idx, fx)
        Fp = smooth(Sf, k_h, r_h, dx) / (smooth(C, k_h, r_h, dx) + c0.min_count + EPS)
        Bbias = cumtrapz(Fp, dx)
        Bbias = Bbias - Bbias[:, idx0:idx0 + 1]
        F_target = (1.0 - ema) * F_target + ema * Bbias

        zx = torch.randn((B, N), device=device, dtype=dtype,
                         generator=gen_n).repeat_interleave(M, dim=0)
        zy = torch.randn((B, N), device=device, dtype=dtype,
                         generator=gen_n).repeat_interleave(M, dim=0)
        bias_force = interp1d(X, Fp, dx)
        Xp = reflect_into(X + (-fx + bias_force) * dt + noise_amp * zx, XMIN, XMAX)
        Yp = Y + (-fy) * dt + noise_amp * zy

        do_fr = (step >= fr_burnin) and ((step - fr_burnin) % fr_every == 0)
        if do_fr and bool(fr_mask.any()):
            g = (gamma_r * (1.0 - math.exp(-max((step - fr_burnin) / ramp, 0.0)))
                 if ramp > 0 else gamma_r)
            p = binned_density(Xp, k_eta, r_eta, dx)
            q = eb.fr_target_from(F_target, Bbias, beta, dx)
            if bool(is_uniform.any()):
                qu = torch.ones((1, N_GRID), device=device, dtype=dtype)
                q = torch.where(is_uniform.unsqueeze(1),
                                (qu / torch.clamp(trapz(qu, dx), min=EPS)).expand(R, N_GRID), q)
            if bool(is_oracle.any()):
                q = torch.where(is_oracle.unsqueeze(1),
                                eb.fr_target_from(F_ref, Bbias, beta, dx), q)
            kl = trapz(p * (torch.log(torch.clamp(p, min=EPS))
                            - torch.log(torch.clamp(q, min=EPS))), dx).unsqueeze(1)
            S = (torch.log(torch.clamp(interp1d(Xp, p, dx), min=EPS))
                 - torch.log(torch.clamp(interp1d(Xp, q, dx), min=EPS)) - kl)
            S = torch.clamp(S, -clip_r, clip_r)
            sel, die, clone = resample_indices(S, fr_mask, sham_mask, partner, g, dt_fr,
                                               cap_r, gen_f)
            Xp = torch.gather(Xp, 1, sel)
            Yp = torch.gather(Yp, 1, sel)
            anc = torch.gather(anc, 1, sel)
            tot_die += die.sum(dim=1).to(dtype)
            tot_clone += clone.sum(dim=1).to(dtype)
            n_fr_apply += 1

        X, Y = Xp, Yp

        if step in save_set:
            Bc = Bbias - Bbias[:, eval_mask].mean(dim=1, keepdim=True)
            ts_l2f[:, save_ptr] = l2_error(Bc, F_ref, eval_mask)
            ts_l2fp[:, save_ptr] = l2_error(Fp, Fp_ref, eval_mask)
            e_, w_ = ancestor_stats(anc, N)
            ts_ess[:, save_ptr] = e_
            ts_wmax[:, save_ptr] = w_
            plab = region_of_particles(X)
            for k in range(3):
                ts_P[:, save_ptr, k] = (plab == k).to(dtype).mean(dim=1)
            ts_Q[:, save_ptr] = bias_aware_target(F_ref, Bbias, glab, beta)
            save_ptr += 1
        if progress is not None and step % progress == 0:
            print(f"    step {step}/{n_steps}", flush=True)

    # The observed fractions are conditioned on the same domain as the target: reflection
    # keeps every walker inside [XMIN, XMAX], so the three regions exhaust the population
    # exactly and P and Q are directly comparable.
    tot_P = ts_P.sum(dim=2)
    worst = float((tot_P - 1.0).abs().max())
    assert worst < 1e-9, f"observed region fractions do not sum to 1 (worst {worst:.3e})"

    return _finalize(locals())


# -----------------------------------------------------------------------------
# frozen-bias validation: an endpoint that does not reuse the online estimator
# -----------------------------------------------------------------------------
def run_frozen_bias(Fp_frozen, cfgs_per_row, group=None, n_steps=40_000, burn_frac=0.5,
                    seed=987_654, device=DEVICE, dtype=DTYPE):
    """Score a learned bias by sampling under it, with no adaptation and no birth-death.

    The online integrated error is computed from the same accumulators the FR mechanism
    perturbs, and birth-death makes replicas correlated descendants, so a gain measured that
    way could in principle be a change in the *statistics of the estimator* rather than a
    better bias.  This closes that gap.

    Each row's final mean force ``Fp_frozen`` is held fixed while a **fresh, independent**
    population -- started identically for every arm, so no arm inherits an advantage from
    where its walkers happened to end up -- runs under it with the ABF accumulators switched
    off and no resampling.

    ``group`` is an ``(R,)`` array of group ids; rows sharing a group share their fresh
    initial conditions **and their Langevin noise**, so the arms of one seed are compared on
    the same realisation and the endpoint stays paired.  Without it each row draws its own
    noise, and the arm-to-arm difference picks up a sampling variance that has nothing to do
    with the bias being scored.  Default: every row is its own group.

    At equilibrium the sampled density is
    ``p_B(x) propto exp(-beta (F(x) - B(x)))``, so

        F_hat(x) = B(x) - beta^{-1} log p_B(x) + const,     B = cumtrapz(Fp_frozen),

    and ``||F_hat - F_ref||`` is an estimate of the bias's quality that never touches the
    adaptive estimator.  A first ``burn_frac`` of the run is discarded and the density is
    time-averaged over the remainder.

    ``Fp_frozen`` is ``(R, G)``; ``cfgs_per_row`` is the length-``R`` list of configs those
    rows were run with.  Returns per-row L2 errors and the reconstructed profiles.
    """
    R = Fp_frozen.shape[0]
    assert len(cfgs_per_row) == R, "one config per frozen-bias row"
    x_grid, dx, eval_mask, idx0 = build_grid(device, dtype)
    c0 = cfgs_per_row[0]
    k_eta, r_eta = gaussian_kernel(c0.eta, dx, device, dtype)

    def col(fn):
        return torch.tensor([fn(c) for c in cfgs_per_row], device=device,
                            dtype=dtype).unsqueeze(1)
    beta = col(lambda c: c.beta); Hc = col(lambda c: c.H)
    oout = col(lambda c: c.omega_out); oin = col(lambda c: c.omega_in)
    sw = col(lambda c: c.s)
    N, dt = c0.N, c0.dt
    noise_amp = torch.sqrt(2.0 * dt / beta)

    Fp = torch.as_tensor(Fp_frozen, device=device, dtype=dtype)
    F_ref, _ = eb.reference_profiles(x_grid, eval_mask, beta, Hc, oout, oin, sw)

    # Fresh start: uniform over the domain, with the transverse channel drawn from its exact
    # conditional so it begins equilibrated.  Rows in one group share the realisation.
    if group is None:
        gidx = torch.arange(R, device=device)
    else:
        g = np.asarray(group)
        uniq = {v: i for i, v in enumerate(dict.fromkeys(g.tolist()))}
        gidx = torch.as_tensor([uniq[v] for v in g.tolist()], device=device,
                               dtype=torch.long)
    G = int(gidx.max().item()) + 1
    gen = torch.Generator(device=device); gen.manual_seed(seed)
    X = (XMIN + (XMAX - XMIN) * torch.rand((G, N), device=device, dtype=dtype,
                                           generator=gen))[gidx]
    om0 = omega_of(X, oout, oin, sw)
    Y = torch.randn((G, N), device=device, dtype=dtype, generator=gen)[gidx] * torch.sqrt(
        1.0 / (beta * om0 ** 2))

    burn = int(burn_frac * n_steps)
    acc = torch.zeros((R, N_GRID), device=device, dtype=dtype)
    n_acc = 0
    for step in range(n_steps):
        om = omega_of(X, oout, oin, sw)
        dom = domega_of(X, oout, oin, sw)
        fx = dU_of(X, Hc) + om * dom * Y * Y
        fy = om * om * Y
        zx = torch.randn((G, N), device=device, dtype=dtype, generator=gen)[gidx]
        zy = torch.randn((G, N), device=device, dtype=dtype, generator=gen)[gidx]
        # The bias is FROZEN: interpolated from the stored profile, never re-accumulated.
        X = reflect_into(X + (-fx + interp1d(X, Fp, dx)) * dt + noise_amp * zx, XMIN, XMAX)
        Y = Y + (-fy) * dt + noise_amp * zy
        if step >= burn:
            acc += binned_density(X, k_eta, r_eta, dx)
            n_acc += 1
    p_B = torch.clamp(acc / max(n_acc, 1), min=EPS)
    p_B = p_B / p_B.sum(dim=1, keepdim=True)

    B = cumtrapz(Fp, dx)
    F_hat = B - torch.log(p_B) / beta
    F_hat = F_hat - F_hat[:, eval_mask].mean(dim=1, keepdim=True)
    F_ref_c = F_ref - F_ref[:, eval_mask].mean(dim=1, keepdim=True)
    err = l2_error(F_hat, F_ref_c, eval_mask)
    # in kT, so rows at different beta are comparable
    err_kT = err * beta.squeeze(1)
    return dict(l2_f=err.detach().cpu().numpy(), l2_f_kT=err_kT.detach().cpu().numpy(),
                F_hat=F_hat.detach().cpu().numpy(), F_ref=F_ref_c.detach().cpu().numpy(),
                p_B=p_B.detach().cpu().numpy(), x_grid=x_grid.detach().cpu().numpy(),
                n_steps=n_steps, burn_frac=burn_frac, seed=seed, n_groups=G)


def _finalize(L):
    cfgs, methods = L["cfgs"], L["methods"]
    B, M, N, dt = L["B"], L["M"], L["N"], L["dt"]
    eval_mask, dx = L["eval_mask"], L["dx"]
    t_axis = np.array([st * dt for st in L["save_steps"]])

    def npy(t):
        return t.detach().cpu().numpy()

    Bc = L["Bbias"] - L["Bbias"][:, eval_mask].mean(dim=1, keepdim=True)
    ts_l2f, ts_l2fp = L["ts_l2f"], L["ts_l2fp"]
    seg_w = torch.tensor(np.diff(t_axis), device=ts_l2f.device, dtype=ts_l2f.dtype)
    int_l2f = (0.5 * (ts_l2f[:, 1:] + ts_l2f[:, :-1]) * seg_w).sum(dim=1)
    int_l2fp = (0.5 * (ts_l2fp[:, 1:] + ts_l2fp[:, :-1]) * seg_w).sum(dim=1)

    recs = []
    for b in range(B):
        for m in range(M):
            r = b * M + m
            use_fr = methods[m].use_fr
            P = npy(L["ts_P"][r]); Q = npy(L["ts_Q"][r])
            rec = dict(
                config=asdict(cfgs[b]), s=cfgs[b].s, r_ratio=cfgs[b].r, init=cfgs[b].init,
                barrier_kT=cfgs[b].barrier_kT(),
                method=methods[m].name, target_mode=methods[m].target_mode,
                sham=methods[m].sham, seed=int(L["spec"].seeds[b]),
                t=t_axis, P_regions=P, Q_regions=Q,
                l2_f_t=npy(ts_l2f[r]), l2_fp_t=npy(ts_l2fp[r]),
                ess_t=npy(L["ts_ess"][r]), wmax_t=npy(L["ts_wmax"][r]),
                final_l2_f=float(ts_l2f[r, -1]), final_l2_fp=float(ts_l2fp[r, -1]),
                int_l2_f=float(int_l2f[r]), int_l2_fp=float(int_l2fp[r]),
                final_ess=float(L["ts_ess"][r, -1]), final_wmax=float(L["ts_wmax"][r, -1]),
                min_ess_frac=float(L["ts_ess"][r].min() / N),
                max_wmax=float(L["ts_wmax"][r].max()),
                x_grid=npy(L["x_grid"]), F_hat=npy(Bc[r]), Fp_hat=npy(L["Fp"][r]),
                F_ref=npy(L["F_ref"][r]), Fp_ref=npy(L["Fp_ref"][r]),
                n_die=float(L["tot_die"][r]) if use_fr else 0.0,
                n_clone=float(L["tot_clone"][r]) if use_fr else 0.0,
                n_fr_apply=int(L["n_fr_apply"]) if use_fr else 0,
            )
            rec["repl_fraction"] = (
                (rec["n_die"] + rec["n_clone"]) / max(rec["n_fr_apply"] * N, 1)
                if use_fr else 0.0)
            rec.update(hit_and_establish(P[:, 2], Q[:, 2], t_axis))
            recs.append(rec)
    return recs
