"""Batched single-GPU ABF / ABF+Fisher-Rao engine for the ENTROPY-DOMINANT
bottleneck study.

This is an m-dimensional generalization of `eb_abffr_core.py`.  The transverse
coordinate is promoted from a scalar y to a vector y in R^m, which lets the
entropic contribution to the marginal free energy be made *larger* than the
energetic one while holding the total barrier fixed.

Model
-----
    V(x, y) = H (x^2-1)^2 + 1/2 omega(x)^2 ||y||^2,   y in R^m,
    omega(x) = omega_out + (omega_in - omega_out) exp(-x^2 / 2 s^2),
    xi(x, y) = x.

Analytic reference (up to an additive constant):
    F_ref(x)  = H (x^2-1)^2 + (m/beta) log omega(x) + C,
    F'_ref(x) = 4 H x (x^2-1) + (m/beta) omega'(x)/omega(x),
    Y | X=x   ~ N(0, 1/(beta omega(x)^2) I_m),
    omega'(x) = -(x/s^2) (omega_in - omega_out) exp(-x^2 / 2 s^2).

Instantaneous force used by ABF:
    dV/dx = 4 H x (x^2-1) + omega(x) omega'(x) ||y||^2,
    dV/dy_j = omega(x)^2 y_j.

Barrier decomposition (thermal units).  With total barrier
    B0 = beta H + m log(omega_in/omega_out)
and entropic share phi = m log(omega_in/omega_out) / B0, the design knobs are
    H        = (1-phi) B0 / beta,
    omega_in = omega_out exp(phi B0 / m),
so that DeltaF_energetic = H and DeltaF_entropic = (m/beta) log(omega_in/omega_out)
satisfy beta (DeltaF_energetic + DeltaF_entropic) = B0 and the *energetic share*
is (1-phi).  See `make_config`.

Batching
--------
Identical scheme to `eb_abffr_core.py`: a *run* is a (config, seed, method)
triple; runs stack as (B, M, N) flattened to R = B*M.  B indexes (config, seed)
rows that SHARE initial conditions and Langevin noise (matched-seed comparison);
M indexes methods within a row.  Only Y carries the extra transverse axis, so the
state is X:(R,N) and Y:(R,N,m).

NO-LEAKAGE INVARIANT
--------------------
The estimated-target method (`fr_estimated`) NEVER reads F_ref.  Its FR target
q_n(x) is an online EMA of the ABF bias only.  F_ref is used solely for (a)
post-hoc L2 evaluation and (b) the explicitly non-deployable `fr_oracle`
diagnostic.  `assert_no_oracle_leakage` enforces this.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

EPS = 1e-30

# -----------------------------------------------------------------------------
# grid / evaluation window.  The domain does not depend on physical parameters,
# so the grid is global; `configure_grid` lets the runner override it from YAML.
# -----------------------------------------------------------------------------
XMIN, XMAX = -2.0, 2.0
N_GRID = 256
EVAL_LO, EVAL_HI = -1.5, 1.5  # interior window for the L2 errors

# conditional-fidelity probe locations and half-width
COND_CENTERS = (-1.0, -0.5, 0.0, 0.5, 1.0)
COND_HALFWIDTH = 0.05  # bin = center +/- this


def configure_grid(x_min, x_max, n_grid, eval_lo, eval_hi):
    """Override the global grid / evaluation window (call once before simulating)."""
    global XMIN, XMAX, N_GRID, EVAL_LO, EVAL_HI
    XMIN, XMAX, N_GRID = float(x_min), float(x_max), int(n_grid)
    EVAL_LO, EVAL_HI = float(eval_lo), float(eval_hi)


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
    """Per-config physical + numerical parameters (one B-row).

    H and omega_in are normally derived from (B0, beta, m, omega_out, phi) via
    `make_config`; phi and B0 are kept for metadata / barrier decomposition.
    """
    beta: float = 4.0
    m: int = 2
    H: float = 2.0
    omega_out: float = 1.0
    omega_in: float = 1.0
    s: float = 0.25
    phi: float = 0.0             # entropic share (metadata)
    B0: float = 8.0             # total thermal barrier (metadata)
    gamma: float = 15.0          # FR birth-death rate
    N: int = 512
    dt: float = 5e-4
    n_steps: int = 80000
    save_every: int = 200
    # ABF mean-force smoothing
    h: float = 0.07
    min_count: float = 1.0
    # FR knobs
    eta: float = 0.10            # marginal-density bandwidth
    fr_every: int = 10
    fr_burnin: int = 0           # absolute steps
    ramp_steps: int = 2000       # absolute steps
    target_ema_rate: float = 0.005
    score_clip: float = 3.0
    max_event_fraction: float = 0.08
    # windowed ancestor-ESS reset period (recent-diversity diagnostic)
    ess_window_steps: int = 4000
    # initial-condition spread of X0 around the left well (-1)
    x_init_std: float = 0.1


def make_config(beta, m, B0, omega_out, phi, **overrides) -> PhysConfig:
    """Build a PhysConfig with H, omega_in derived to fix the total barrier B0.

        H        = (1-phi) B0 / beta,
        omega_in = omega_out exp(phi B0 / m).

    `overrides` sets any remaining field (gamma, N, dt, n_steps, ...).
    """
    H = (1.0 - phi) * B0 / beta
    omega_in = omega_out * math.exp(phi * B0 / m)
    return PhysConfig(beta=beta, m=int(m), H=H, omega_out=omega_out,
                      omega_in=omega_in, s=overrides.pop("s", 0.25),
                      phi=phi, B0=B0, **overrides)


def delta_f_energetic(cfg: PhysConfig) -> float:
    """DeltaF_energetic = H (well-to-barrier energetic rise of the double well)."""
    return float(cfg.H)


def delta_f_entropic(cfg: PhysConfig) -> float:
    """DeltaF_entropic = (m/beta) log(omega_in/omega_out)."""
    return float(cfg.m / cfg.beta * math.log(cfg.omega_in / cfg.omega_out))


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


def reference_profiles(x_grid, eval_mask, beta, Hc, omega_out, omega_in, s, mdim):
    """F_ref (centered on eval window) and F'_ref on the grid.

    beta, Hc, omega_*, mdim may be (B,1) tensors -> returns (B, N_GRID).
    """
    xg = x_grid.unsqueeze(0)  # (1, G)
    om = omega_of(xg, omega_out, omega_in, s)
    dom = domega_of(xg, omega_out, omega_in, s)
    F_ref = U_of(xg, Hc) + (mdim / beta) * torch.log(om)
    F_ref = F_ref - F_ref[:, eval_mask].mean(dim=1, keepdim=True)
    Fp_ref = dU_of(xg, Hc) + (mdim / beta) * dom / om
    return F_ref, Fp_ref


# -----------------------------------------------------------------------------
# batched numerics (operate on a flattened run axis R = B*M)
# -----------------------------------------------------------------------------
def gaussian_kernel(bw, dx, device, dtype):
    """Gaussian kernel, radius 4*bw/dx, normalized by sum*dx (matches notebook)."""
    r = max(1, int(round(4.0 * bw / dx)))
    t = torch.arange(-r, r + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (t * dx / bw) ** 2)
    k = k / (k.sum() * dx)
    return k, r


def smooth(v, kernel, r, dx):
    """Reflect-pad + valid convolution along the grid axis. v:(R,G)->(R,G)."""
    R, G = v.shape
    pad = min(r, G - 1)
    vp = F.pad(v.unsqueeze(1), (pad, pad), mode="reflect")  # (R,1,G+2pad)
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
    """
    pos = torch.clamp((X - XMIN) / dx, 0.0, N_GRID - 1.0)
    i0 = torch.clamp(torch.floor(pos).long(), 0, N_GRID - 2)
    frac = pos - i0.to(X.dtype)
    v0 = torch.gather(grid_vals, 1, i0)
    v1 = torch.gather(grid_vals, 1, i0 + 1)
    return v0 + frac * (v1 - v0)


def reflect_into(q, lo, hi):
    """Reflect coordinates into [lo, hi]."""
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


# -----------------------------------------------------------------------------
# batched Fisher-Rao birth-death resampling (identical to eb_abffr_core)
# -----------------------------------------------------------------------------
def fr_resample_indices(S, fr_mask, g, dt_fr, cap, gen):
    """Vectorized FR birth-death, returning a per-row gather index.

    See eb_abffr_core.fr_resample_indices for the full derivation; this is a
    verbatim copy (the resampler acts on the run/walker axes only and is
    independent of the transverse dimension m).
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

    # proportional cap on total events
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
    die_rank = dk.argsort(dim=1).argsort(dim=1)
    clone_rank = ck.argsort(dim=1).argsort(dim=1)
    die = die & (die_rank < kd)
    clone = clone & (clone_rank < kc)
    surv = ~die

    ar = torch.arange(N, device=dev).unsqueeze(0).expand(R, N)
    surv_idx = torch.where(surv, ar, torch.full_like(ar, -1))
    clone_idx = torch.where(clone, ar, torch.full_like(ar, -1))
    pool = torch.cat([surv_idx, clone_idx], dim=1)
    valid = pool >= 0
    keys = torch.where(valid,
                       torch.rand((R, 2 * N), device=dev, dtype=dt, generator=gen),
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
    invalid_slot = ar >= valid_count
    pad = torch.gather(surv_perm, 1, rand_rank)
    sel = torch.where(invalid_slot, pad, sel)

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

    Within each B-row all M methods SHARE initial conditions and Langevin noise
    (matched-seed ABF-vs-FR comparison).  `configs` and `seeds` are parallel
    lists of length B.
    """
    configs: Sequence[PhysConfig]
    seeds: Sequence[int]
    methods: Sequence[MethodSpec]
    batch_seed: int = 12345

    def __post_init__(self):
        assert len(self.configs) == len(self.seeds), "configs and seeds must align"


def _per_config_tensor(configs, attr, device, dtype):
    return torch.tensor([getattr(c, attr) for c in configs], device=device, dtype=dtype)


def init_conditions_batched(seeds, N, mdim, beta_b, omega_out_b, omega_in_b, s_b,
                            x_init_std, device, dtype):
    """Per-(B-row) initial conditions, one row per (config, seed).

        rng_i = default_rng(1000+seed)
        X0 = reflect(rng_i.normal(-1, x_init_std, N))         # config-independent
        Z0 = rng_i.normal(0, 1, (N, m))                       # config-independent
        Y0 = Z0 * sqrt(1/(beta omega(X0)^2))                  # config scale

    seeds: length-B list.  beta_b etc: (B,) tensors.  Returns X0:(B,N), Y0:(B,N,m).
    """
    B = len(seeds)
    X0 = torch.empty((B, N), device=device, dtype=dtype)
    Z0 = torch.empty((B, N, mdim), device=device, dtype=dtype)
    for b, sd in enumerate(seeds):
        rng_i = np.random.default_rng(1000 + int(sd))
        X0[b] = reflect_into(torch.as_tensor(rng_i.normal(-1.0, x_init_std, N),
                                             device=device, dtype=dtype), XMIN, XMAX)
        Z0[b] = torch.as_tensor(rng_i.normal(0.0, 1.0, (N, mdim)), device=device, dtype=dtype)
    om0 = omega_of(X0, omega_out_b.unsqueeze(1), omega_in_b.unsqueeze(1), s_b.unsqueeze(1))
    scale = torch.sqrt(1.0 / (beta_b.unsqueeze(1) * om0 ** 2))  # (B,N)
    Y0 = Z0 * scale.unsqueeze(-1)
    return X0, Y0


def simulate_batch(spec: BatchSpec, device=DEVICE, dtype=DTYPE,
                   noise_seed_base=2000, fr_seed_base=3000, progress=None):
    """Run all (config, method) pairs for the given (config, seed) rows.

    Uniform-across-batch (asserted): m, N, dt, n_steps, save_every, fr_every,
    fr_burnin, ramp_steps, h, eta, min_count, ess_window_steps.  Per-config (may
    vary): beta, H, omega_out, omega_in, s, gamma, target_ema_rate, score_clip,
    max_event_fraction.
    """
    assert_no_oracle_leakage(spec.methods)
    cfgs, methods = list(spec.configs), list(spec.methods)
    B, M = len(cfgs), len(methods)
    R = B * M

    c0 = cfgs[0]
    for c in cfgs:
        for a in ("m", "N", "dt", "n_steps", "save_every", "fr_every", "fr_burnin",
                  "ramp_steps", "h", "eta", "min_count", "ess_window_steps",
                  "x_init_std"):
            assert getattr(c, a) == getattr(c0, a), f"non-uniform {a} across configs"
    mdim = int(c0.m)
    N, dt, n_steps = c0.N, c0.dt, c0.n_steps
    save_every, fr_every, fr_burnin = c0.save_every, c0.fr_every, c0.fr_burnin
    ramp = int(c0.ramp_steps)
    dt_fr = dt * fr_every

    x_grid, dx, eval_mask, idx0 = build_grid(device, dtype)
    k_h, r_h = gaussian_kernel(c0.h, dx, device, dtype)
    k_eta, r_eta = gaussian_kernel(c0.eta, dx, device, dtype)

    def cfg_b(attr):
        return _per_config_tensor(cfgs, attr, device, dtype)
    beta_b = cfg_b("beta"); H_b = cfg_b("H")
    oout_b = cfg_b("omega_out"); oin_b = cfg_b("omega_in"); s_b = cfg_b("s")
    gamma_b = cfg_b("gamma"); ema_b = cfg_b("target_ema_rate")
    clip_b = cfg_b("score_clip"); maxfrac_b = cfg_b("max_event_fraction")
    mdim_b = cfg_b("m")

    def to_run(t_b):  # (B,) -> (R,1)
        return t_b.repeat_interleave(M).unsqueeze(1)
    beta = to_run(beta_b); Hc = to_run(H_b)
    oout = to_run(oout_b); oin = to_run(oin_b); sw = to_run(s_b)
    gamma_r = to_run(gamma_b); ema = to_run(ema_b)
    clip_r = to_run(clip_b); maxfrac_r = to_run(maxfrac_b)
    cap_r = torch.floor(maxfrac_r * N).long()
    noise_amp = torch.sqrt(2.0 * dt / beta)            # (R,1)

    use_fr_m = torch.tensor([m.use_fr for m in methods], device=device)
    fr_mask = use_fr_m.repeat(B)                        # (R,)
    target_mode = [m.target_mode for m in methods]
    is_uniform = torch.tensor([m == "uniform" for m in target_mode], device=device).repeat(B)
    is_oracle = torch.tensor([m == "oracle" for m in target_mode], device=device).repeat(B)

    F_ref_b, Fp_ref_b = reference_profiles(
        x_grid, eval_mask, beta_b.unsqueeze(1), H_b.unsqueeze(1),
        oout_b.unsqueeze(1), oin_b.unsqueeze(1), s_b.unsqueeze(1), mdim_b.unsqueeze(1))
    F_ref = F_ref_b.repeat_interleave(M, dim=0)         # (R,G)
    Fp_ref = Fp_ref_b.repeat_interleave(M, dim=0)

    X0_b, Y0_b = init_conditions_batched(spec.seeds, N, mdim, beta_b, oout_b, oin_b,
                                         s_b, c0.x_init_std, device, dtype)
    X = X0_b.repeat_interleave(M, dim=0).clone()        # (R,N)
    Y = Y0_b.repeat_interleave(M, dim=0).clone()        # (R,N,m)
    anc = torch.arange(N, device=device).unsqueeze(0).expand(R, N).clone()
    ess_window = c0.ess_window_steps

    C = torch.zeros((R, N_GRID), device=device, dtype=dtype)
    Sf = torch.zeros((R, N_GRID), device=device, dtype=dtype)
    F_target = torch.zeros((R, N_GRID), device=device, dtype=dtype)

    gen_n = torch.Generator(device=device); gen_n.manual_seed(noise_seed_base + spec.batch_seed)
    gen_f = torch.Generator(device=device); gen_f.manual_seed(fr_seed_base + spec.batch_seed)

    save_steps = [st for st in range(n_steps) if st % save_every == 0 or st == n_steps - 1]
    n_saves = len(save_steps)
    ts_l2f = torch.zeros((R, n_saves), device=device, dtype=dtype)
    ts_l2fp = torch.zeros((R, n_saves), device=device, dtype=dtype)
    ts_ess = torch.zeros((R, n_saves), device=device, dtype=dtype)
    ts_cross = torch.zeros((R, n_saves), device=device, dtype=dtype)
    ts_occ = torch.zeros((R, n_saves), device=device, dtype=dtype)
    ts_denom0 = torch.zeros((R, n_saves), device=device, dtype=dtype)
    save_set = set(save_steps); save_ptr = 0
    tot_die = torch.zeros(R, device=device, dtype=dtype)
    tot_clone = torch.zeros(R, device=device, dtype=dtype)
    n_fr_apply = 0

    # cumulative crossings of x=0 (sign change of X across a step)
    cum_cross = torch.zeros(R, device=device, dtype=dtype)
    # FR score-std accumulation (mean over FR applications of per-row std of S)
    sum_score_std = torch.zeros(R, device=device, dtype=dtype)
    occ_lo, occ_hi = -0.2, 0.2  # barrier-occupancy window

    for step in range(n_steps):
        if ess_window > 0 and step % ess_window == 0:
            anc = torch.arange(N, device=device).unsqueeze(0).expand(R, N).clone()
        # ---- forces ----
        om = omega_of(X, oout, oin, sw)               # (R,N)
        dom = domega_of(X, oout, oin, sw)             # (R,N)
        sqnorm = (Y * Y).sum(dim=-1)                  # (R,N) = ||y||^2
        fx = dU_of(X, Hc) + om * dom * sqnorm         # dV/dx
        fy = (om * om).unsqueeze(-1) * Y              # dV/dy_j  (R,N,m)

        # ---- ABF accumulation + mean force + bias ----
        idx = torch.clamp(torch.round((X - XMIN) / dx).long(), 0, N_GRID - 1)
        C.scatter_add_(1, idx, torch.ones_like(X))
        Sf.scatter_add_(1, idx, fx)
        Fp = smooth(Sf, k_h, r_h, dx) / (smooth(C, k_h, r_h, dx) + c0.min_count + EPS)
        Bbias = cumtrapz(Fp, dx)
        Bbias = Bbias - Bbias[:, idx0:idx0 + 1]
        F_target = (1.0 - ema) * F_target + ema * Bbias

        # ---- Langevin step (noise shared across methods via B-block broadcast) ----
        zx = torch.randn((B, N), device=device, dtype=dtype, generator=gen_n).repeat_interleave(M, dim=0)
        zy = torch.randn((B, N, mdim), device=device, dtype=dtype, generator=gen_n).repeat_interleave(M, dim=0)
        bias_force = interp1d(X, Fp, dx)
        Xp = reflect_into(X + (-fx + bias_force) * dt + noise_amp * zx, XMIN, XMAX)
        Yp = Y + (-fy) * dt + noise_amp.unsqueeze(-1) * zy

        # ---- crossing counter: genuine Langevin sign-changes of x, counted on
        # the PROPOSAL (before FR resampling, which would otherwise add spurious
        # clone-jumps across x=0 and make the FR count incomparable to ABF) ----
        cum_cross += ((X * Xp) < 0).sum(dim=1).to(dtype)

        # ---- Fisher-Rao birth-death ----
        do_fr = (step >= fr_burnin) and ((step - fr_burnin) % fr_every == 0)
        if do_fr and fr_mask.any():
            if ramp > 0:
                g = gamma_r * (1.0 - math.exp(-max((step - fr_burnin) / ramp, 0.0)))
            else:
                g = gamma_r
            p = binned_density(Xp, k_eta, r_eta, dx)
            q_est = fr_target_from(F_target, Bbias, beta, dx)
            qu = torch.ones((1, N_GRID), device=device, dtype=dtype)
            q_uni = (qu / torch.clamp(trapz(qu, dx), min=EPS)).expand(R, N_GRID)
            q = q_est
            if any(m == "uniform" for m in target_mode):
                q = torch.where(is_uniform.unsqueeze(1), q_uni, q)
            if any(m == "oracle" for m in target_mode):
                q_orc = fr_target_from(F_ref, Bbias, beta, dx)
                q = torch.where(is_oracle.unsqueeze(1), q_orc, q)
            logp = torch.log(torch.clamp(p, min=EPS))
            logq = torch.log(torch.clamp(q, min=EPS))
            kl = trapz(p * (logp - logq), dx).unsqueeze(1)
            S = (torch.log(torch.clamp(interp1d(Xp, p, dx), min=EPS))
                 - torch.log(torch.clamp(interp1d(Xp, q, dx), min=EPS)) - kl)
            S = torch.clamp(S, -clip_r, clip_r)
            # FR score-std diagnostic (per FR row, over walkers)
            sum_score_std += torch.where(fr_mask, S.std(dim=1), torch.zeros(R, device=device, dtype=dtype))
            sel, die, clone = fr_resample_indices(S, fr_mask, g, dt_fr, cap_r, gen_f)
            Xp = torch.gather(Xp, 1, sel)
            Yp = torch.gather(Yp, 1, sel.unsqueeze(-1).expand(-1, -1, mdim))
            anc = torch.gather(anc, 1, sel)
            tot_die += die.sum(dim=1).to(dtype)
            tot_clone += clone.sum(dim=1).to(dtype)
            n_fr_apply += 1

        X, Y = Xp, Yp

        # ---- diagnostics ----
        if step in save_set:
            Bc = Bbias - Bbias[:, eval_mask].mean(dim=1, keepdim=True)
            ts_l2f[:, save_ptr] = l2_error(Bc, F_ref, eval_mask)
            ts_l2fp[:, save_ptr] = l2_error(Fp, Fp_ref, eval_mask)
            ts_ess[:, save_ptr] = ancestor_ess(anc, N)
            ts_cross[:, save_ptr] = cum_cross
            ts_occ[:, save_ptr] = ((X >= occ_lo) & (X <= occ_hi)).to(dtype).mean(dim=1)
            # ABF smoothed count near x=0 (effective samples available for force)
            ts_denom0[:, save_ptr] = smooth(C, k_h, r_h, dx)[:, idx0]
            save_ptr += 1
        if progress is not None and step % progress == 0:
            print(f"    step {step}/{n_steps}", flush=True)

    return _finalize(locals())


# -----------------------------------------------------------------------------
# diagnostics: ancestor ESS and conditional-variance fidelity (m-dim)
# -----------------------------------------------------------------------------
def ancestor_ess(anc, N):
    """Effective number of distinct ancestors: (sum n_a)^2 / sum n_a^2. anc:(R,N)."""
    R = anc.shape[0]
    counts = torch.zeros((R, N), device=anc.device, dtype=torch.float64)
    counts.scatter_add_(1, anc, torch.ones_like(anc, dtype=torch.float64))
    num = counts.sum(dim=1) ** 2
    den = torch.clamp((counts * counts).sum(dim=1), min=EPS)
    return (num / den).to(anc.dtype)


def conditional_variance_diagnostics(X, Y, beta, oout, oin, sw, mdim):
    """Empirical conditional law of Y vs analytic, at COND_CENTERS.

    X:(R,N); Y:(R,N,m); beta,oout,oin,sw:(R,1). Returns dict of (R,K) tensors:
      cond_emp_var   : component-averaged empirical Var(Y_j | X in bin)
      cond_ref_var   : analytic 1/(beta omega(x)^2)
      cond_emp_sqnorm: empirical E[||Y||^2 | X in bin]
      cond_ref_sqnorm: analytic m/(beta omega(x)^2)
      cond_abs_err   : |cond_emp_var - cond_ref_var|
      cond_count     : bin sample count
    Bins with <2 samples give NaN emp_var / emp_sqnorm.
    """
    R, N = X.shape
    centers = torch.tensor(COND_CENTERS, device=X.device, dtype=X.dtype)  # (K,)
    K = centers.numel()
    lo = centers - COND_HALFWIDTH
    hi = centers + COND_HALFWIDTH
    Xe = X.unsqueeze(2)                                  # (R,N,1)
    inbin = (Xe >= lo.view(1, 1, K)) & (Xe < hi.view(1, 1, K))  # (R,N,K)
    w = inbin.to(X.dtype)                                # (R,N,K)
    cnt = w.sum(dim=1)                                   # (R,K)
    # per-component mean: (R,K,m)
    mean = torch.einsum("rnk,rnm->rkm", w, Y) / torch.clamp(cnt, min=1.0).unsqueeze(-1)
    diff2 = (Y.unsqueeze(2) - mean.unsqueeze(1)) ** 2    # (R,N,K,m)
    var = torch.einsum("rnk,rnkm->rkm", w, diff2) / torch.clamp(cnt - 1.0, min=1.0).unsqueeze(-1)
    emp_var_comp = var.mean(dim=-1)                      # (R,K) component-averaged
    emp_var_comp = torch.where(cnt >= 2.0, emp_var_comp, torch.full_like(emp_var_comp, float("nan")))
    sqnorm = (Y * Y).sum(dim=-1)                         # (R,N)
    emp_sqnorm = torch.einsum("rnk,rn->rk", w, sqnorm) / torch.clamp(cnt, min=1.0)
    emp_sqnorm = torch.where(cnt >= 2.0, emp_sqnorm, torch.full_like(emp_sqnorm, float("nan")))
    om_c = omega_of(centers.view(1, K), oout, oin, sw)   # (R,K)
    ref_var = 1.0 / (beta * om_c ** 2)
    ref_sqnorm = mdim / (beta * om_c ** 2)
    return {
        "cond_centers": centers,
        "cond_emp_var": emp_var_comp,
        "cond_ref_var": ref_var,
        "cond_emp_sqnorm": emp_sqnorm,
        "cond_ref_sqnorm": ref_sqnorm,
        "cond_abs_err": torch.abs(emp_var_comp - ref_var),
        "cond_count": cnt,
    }


# -----------------------------------------------------------------------------
# finalize: pull batched results into per-run numpy records
# -----------------------------------------------------------------------------
def _finalize(L):
    cfgs, methods = L["cfgs"], L["methods"]
    B, M, R, N = L["B"], L["M"], L["R"], L["N"]
    dx, eval_mask, x_grid = L["dx"], L["eval_mask"], L["x_grid"]
    X, Y, anc = L["X"], L["Y"], L["anc"]
    Fp, Bbias, F_target = L["Fp"], L["Bbias"], L["F_target"]
    F_ref, Fp_ref = L["F_ref"], L["Fp_ref"]
    beta, oout, oin, sw = L["beta"], L["oout"], L["oin"], L["sw"]
    mdim = L["mdim"]
    save_steps = L["save_steps"]
    dt = L["dt"]

    Bc = Bbias - Bbias[:, eval_mask].mean(dim=1, keepdim=True)
    p_hat = binned_density(X, L["k_eta"], L["r_eta"], dx)
    q_final = fr_target_from(F_target, Bbias, beta, dx)
    cond = conditional_variance_diagnostics(X, Y, beta, oout, oin, sw, mdim)

    ts_l2f = L["ts_l2f"]; ts_l2fp = L["ts_l2fp"]; ts_ess = L["ts_ess"]
    ts_cross = L["ts_cross"]; ts_occ = L["ts_occ"]; ts_denom0 = L["ts_denom0"]
    t_axis = np.array([st * dt for st in save_steps])
    seg = 0.5 * (ts_l2f[:, 1:] + ts_l2f[:, :-1]) * torch.tensor(
        np.diff(t_axis), device=ts_l2f.device, dtype=ts_l2f.dtype)
    int_l2f = seg.sum(dim=1)
    seg_fp = 0.5 * (ts_l2fp[:, 1:] + ts_l2fp[:, :-1]) * torch.tensor(
        np.diff(t_axis), device=ts_l2fp.device, dtype=ts_l2fp.dtype)
    int_l2fp = seg_fp.sum(dim=1)

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
            score_std = float(L["sum_score_std"][r] / max(n_fr_apply, 1)) if use_fr else 0.0
            rec = {
                "config": asdict(cfgs[b]),
                "method": methods[m].name,
                "target_mode": methods[m].target_mode,
                "seed": int(L["spec"].seeds[b]),
                "delta_f_energetic": delta_f_energetic(cfgs[b]),
                "delta_f_entropic": delta_f_entropic(cfgs[b]),
                "final_l2_f": float(ts_l2f[r, -1]),
                "final_l2_fp": float(ts_l2fp[r, -1]),
                "int_l2_f": float(int_l2f[r]),
                "int_l2_fp": float(int_l2fp[r]),
                "final_cross": float(ts_cross[r, -1]),
                "final_occ": float(ts_occ[r, -1]),
                "final_denom0": float(ts_denom0[r, -1]),
                "t": t_axis,
                "l2_f_t": npy(ts_l2f[r]),
                "l2_fp_t": npy(ts_l2fp[r]),
                "ess_t": npy(ts_ess[r]),
                "cross_t": npy(ts_cross[r]),
                "occ_t": npy(ts_occ[r]),
                "denom0_t": npy(ts_denom0[r]),
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
                "score_std": score_std,
                "cond_centers": npy(cond["cond_centers"]),
                "cond_emp_var": npy(cond["cond_emp_var"][r]),
                "cond_ref_var": npy(cond["cond_ref_var"][r]),
                "cond_emp_sqnorm": npy(cond["cond_emp_sqnorm"][r]),
                "cond_ref_sqnorm": npy(cond["cond_ref_sqnorm"][r]),
                "cond_abs_err": npy(cond["cond_abs_err"][r]),
                "cond_count": npy(cond["cond_count"][r]),
            }
            recs.append(rec)
    return recs


# -----------------------------------------------------------------------------
# analytic sanity checks (used by the runner's --smoke-test)
# -----------------------------------------------------------------------------
def sanity_checks(verbose=True):
    """Validate the analytic reference + conditional law.

    1. finite-difference d/dx F_ref matches F'_ref;
    2. sampled Var(Y_j|X=x) matches 1/(beta omega(x)^2) and E||Y||^2 = m/(beta omega^2);
    3. for m=1 the reference reduces to the old entropic-bottleneck formula
       F_ref = H(x^2-1)^2 + (1/beta) log omega(x).
    Returns dict of max errors; raises AssertionError on failure.
    """
    dev, dt = DEVICE, DTYPE
    out = {}

    # ---- 1. finite-difference derivative of F_ref vs F'_ref ----
    cfg = make_config(beta=4.0, m=2, B0=8.0, omega_out=1.0, phi=0.75, s=0.25)
    xs = torch.linspace(-1.5, 1.5, 4001, device=dev, dtype=dt)
    om = omega_of(xs, cfg.omega_out, cfg.omega_in, cfg.s)
    F = U_of(xs, cfg.H) + (cfg.m / cfg.beta) * torch.log(om)
    Fp_analytic = (dU_of(xs, cfg.H)
                   + (cfg.m / cfg.beta) * domega_of(xs, cfg.omega_out, cfg.omega_in, cfg.s) / om)
    dxs = float(xs[1] - xs[0])
    Fp_fd = (F[2:] - F[:-2]) / (2.0 * dxs)
    fd_err = float(torch.max(torch.abs(Fp_fd - Fp_analytic[1:-1])))
    out["fd_derivative_max_err"] = fd_err
    assert fd_err < 1e-4, f"F'_ref finite-difference mismatch: {fd_err}"

    # ---- 2. sampled conditional variance at fixed x ----
    rng = np.random.default_rng(0)
    for xq in (-1.0, -0.5, 0.0, 0.5, 1.0):
        om_x = float(omega_of(torch.tensor(xq, dtype=dt), cfg.omega_out, cfg.omega_in, cfg.s))
        ref_var = 1.0 / (cfg.beta * om_x ** 2)
        nsamp = 4_000_000
        y = rng.normal(0.0, math.sqrt(ref_var), (nsamp, cfg.m))
        emp_var = float(np.var(y, axis=0).mean())
        emp_sqnorm = float(np.mean((y ** 2).sum(axis=1)))
        rel = abs(emp_var - ref_var) / ref_var
        rel_sq = abs(emp_sqnorm - cfg.m * ref_var) / (cfg.m * ref_var)
        out[f"cond_var_relerr_x{xq:g}"] = rel
        out[f"cond_sqnorm_relerr_x{xq:g}"] = rel_sq
        assert rel < 5e-3, f"cond var mismatch at x={xq}: {rel}"
        assert rel_sq < 5e-3, f"E||Y||^2 mismatch at x={xq}: {rel_sq}"

    # ---- 3. m=1 reduces to the scalar entropic-bottleneck formula ----
    cfg1 = make_config(beta=8.0, m=1, B0=8.0, omega_out=1.0, phi=0.5, s=0.25)
    om1 = omega_of(xs, cfg1.omega_out, cfg1.omega_in, cfg1.s)
    F_general = U_of(xs, cfg1.H) + (cfg1.m / cfg1.beta) * torch.log(om1)
    F_scalar = U_of(xs, cfg1.H) + (1.0 / cfg1.beta) * torch.log(om1)
    m1_err = float(torch.max(torch.abs(F_general - F_scalar)))
    out["m1_reduction_max_err"] = m1_err
    assert m1_err < 1e-12, f"m=1 does not reduce to scalar formula: {m1_err}"

    if verbose:
        for k, v in out.items():
            print(f"  {k}: {v:.3e}")
    return out
