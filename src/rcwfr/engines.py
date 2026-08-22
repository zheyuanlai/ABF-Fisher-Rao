"""Batched engines for every arm of the campaign.

COST INVARIANT.  Every arm carries N replicas and takes `n_steps` inner steps,
each evaluating the force once per replica.  Total force evaluations are
N * n_steps for ALL arms, including the RC-WFR lift/relaxation.  W steps, FR
resampling, KDE fits and the TI quadrature cost nothing in this currency;
replica-exchange energy evaluations DO and are charged explicitly.

Arms
----
unbiased   plain overdamped Langevin                                   (control)
abf        multiple-walker ABF sharing one mean-force estimate
shus       mollified SHUS adaptive biasing potential                   (ABP/OPES family)
wfr        RC-WFR: conditional MD -> W -> lift -> FR.  Sub-arms via flags:
             w_mode='none',  fr_rule='none', init='grid_*'  -> stratified TI
             w_mode='sde',   fr_rule='none'                 -> W-only
             w_mode='none',  fr_rule='fr'                   -> FR-only
             w_mode='sde',   fr_rule='count'                -> count-balancing control
             w_mode='sde',   fr_rule='sham'                 -> matched-turnover sham
reti       stratified TI + Hamiltonian replica exchange between adjacent windows
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch

from .estimators import MeanForceAccumulator, gauge_l2
from .fisher_rao import kde_marginal, kl_to_uniform, selection_indices, tv_to_uniform
from .grid import DEVICE, DTYPE, EPS, Grid1D, interp1d
from .resampling import ancestor_stats, surviving_ancestors
from .rowspec import as_col, tile_seeds
from .shus import ShusAccumulator
from .systems.base import SepSystem
from .wasserstein import w_step_flow, w_step_sde


@dataclass
class RunConfig:
    N: int = 256
    n_seed: int = 1               # replicates per hyper-parameter configuration
    dt: float = 1e-3
    n_steps: int = 40_000
    save_every: int = 400
    bw_mf: float = 0.07
    n_min: float = 1.0
    init: str = "point"          # point | grid_cold | grid_warm | uniform_warm | uniform_cold
    x0: float = -1.0
    x0_jitter: float = 0.0        # spread of the 'point' start.  The DETERMINISTIC
                                  # probability-flow W step has zero velocity at a
                                  # delta initial condition and can never move, so it
                                  # requires a non-degenerate starting ensemble.
    # RC-WFR
    n_cond: int = 20
    n_eq: int = 0
    kappa: float = 0.125
    w_mode: str = "sde"
    w_flow_clip: float = 50.0
    theta: float = 0.3
    fr_every: int = 1
    fr_rule: str = "fr"
    bw_kde: float = 0.10
    n_bins_count: int = 45
    alpha_ess: float = 0.5
    fr_jitter: float = 0.0        # resample-move: sigma of the z-jitter applied after
                                  # an FR event.  Deterministic ('flow') transport gives
                                  # clones IDENTICAL trajectories, so without jitter the
                                  # particle ensemble collapses onto a few distinct z.
    lift: str = "identity"       # identity | oracle
    # --- annealing / burn-in (the steelman for the lift-bias tradeoff) ------
    kappa_end: float = None      # geometric anneal kappa -> kappa_end over the run
    acc_reset_at: float = None   # zero the mean-force accumulator at this budget frac
    ess_window: int = 40
    bias_n_min: float = 1.0      # ABF ramp on the APPLIED force (reporting uses n_min)
    # SHUS
    shus_gain: float = 1.0
    shus_block: int = 100
    shus_eps_bw: float = 0.07
    # replica exchange
    n_ex: int = 20               # inner steps between exchange sweeps
    n_windows: int = 0            # RE-TI: distinct windows (0 = N).  Fewer windows =
                                  # larger exchange stride = faster window-space
                                  # diffusion at coarser CV resolution.


def _init_state(sys: SepSystem, cfg: RunConfig, rows: int, gen):
    """Draw ICs for n_seed replicates and tile them across configurations."""
    g, dev, dt, N = sys.grid, sys.device, sys.dtype, cfg.N
    n_seed = cfg.n_seed if cfg.n_seed > 1 else rows
    assert rows % n_seed == 0, f"rows {rows} not divisible by n_seed {n_seed}"
    n_cfg = rows // n_seed
    X, Y = _draw_ic(sys, cfg, n_seed, gen)
    return tile_seeds(X, n_cfg), tile_seeds(Y, n_cfg)


def _draw_ic(sys: SepSystem, cfg: RunConfig, rows: int, gen):
    g, dev, dt, N = sys.grid, sys.device, sys.dtype, cfg.N
    if cfg.init == "point":
        X = torch.full((rows, N), cfg.x0, device=dev, dtype=dt)
        if cfg.x0_jitter > 0:
            X = g.enforce(X + cfg.x0_jitter * torch.randn(
                X.shape, device=dev, dtype=dt, generator=gen))
        Y = sys.sample_conditional(X, gen)
    elif cfg.init in ("grid_cold", "grid_warm"):
        M = cfg.n_windows if cfg.n_windows else N
        assert N % M == 0, f"N={N} must be a multiple of n_windows={M}"
        zs = torch.linspace(g.eval_lo, g.eval_hi, M, device=dev, dtype=dt)
        X = zs.repeat_interleave(N // M).unsqueeze(0).expand(rows, N).clone()
        src = torch.full((rows, N), cfg.x0, device=dev, dtype=dt) \
            if cfg.init == "grid_cold" else X
        Y = sys.sample_conditional(src, gen)
    elif cfg.init in ("uniform_warm", "uniform_cold"):
        X = (torch.rand((rows, N), device=dev, dtype=dt, generator=gen)
             * (g.eval_hi - g.eval_lo) + g.eval_lo)
        src = torch.full((rows, N), cfg.x0, device=dev, dtype=dt) \
            if cfg.init == "uniform_cold" else X
        Y = sys.sample_conditional(src, gen)
    else:
        raise ValueError(cfg.init)
    return X, Y


def _saver(rows, grid, n_saves, dev, dt):
    z = lambda *s: torch.zeros(s, device=dev, dtype=dt)
    return {"F": z(n_saves, rows, grid.n), "p": z(n_saves, rows, grid.n),
            "kl": z(n_saves, rows), "tv": z(n_saves, rows), "cov": z(n_saves, rows),
            "ess_anc": z(n_saves, rows), "surv_anc": z(n_saves, rows),
            "chan": z(n_saves, rows), "fe": z(n_saves)}


def _coverage(X, grid, n_bins=45):
    lo, hi = grid.eval_lo, grid.eval_hi
    w = (hi - lo) / n_bins
    raw = torch.floor((X - lo) / w)
    ok = (raw >= 0) & (raw < n_bins)
    idx = torch.clamp(raw, 0, n_bins - 1).long()
    h = torch.zeros((X.shape[0], n_bins), device=X.device, dtype=X.dtype)
    h.scatter_add_(1, idx, ok.to(X.dtype))
    return (h > 0).to(X.dtype).mean(dim=1)


def _channel_err(sys, X, Y, grid, n_bins=30):
    """L1 error of the realized P(y_1 > 0 | x) against the reference, eval window."""
    lo, hi = grid.eval_lo, grid.eval_hi
    w = (hi - lo) / n_bins
    raw = torch.floor((X - lo) / w)
    ok = ((raw >= 0) & (raw < n_bins)).to(X.dtype)
    idx = torch.clamp(raw, 0, n_bins - 1).long()
    R = X.shape[0]
    cnt = torch.zeros((R, n_bins), device=X.device, dtype=X.dtype)
    pos = torch.zeros((R, n_bins), device=X.device, dtype=X.dtype)
    cnt.scatter_add_(1, idx, ok)
    pos.scatter_add_(1, idx, ok * (Y[..., 0] > 0).to(X.dtype))
    frac = pos / torch.clamp(cnt, min=1.0)
    centers = lo + (torch.arange(n_bins, device=X.device, dtype=X.dtype) + 0.5) * w
    gi = torch.clamp(torch.round((centers - grid.xmin) / grid.dx).long(), 0, grid.n - 1)
    ref = sys.p_channel_ref[gi].unsqueeze(0)
    seen = (cnt > 0).to(X.dtype)
    return ((frac - ref).abs() * seen).sum(1) / torch.clamp(seen.sum(1), min=1.0)


def _record(out, si, sys, cfg, acc, X, Y, mask, fe, ess=None, surv=None):
    g = sys.grid
    p = kde_marginal(X, g, cfg.bw_kde)
    out["F"][si] = acc.free_energy(mask)
    out["p"][si] = p
    out["kl"][si] = kl_to_uniform(p, g)
    out["tv"][si] = tv_to_uniform(p, g)
    out["cov"][si] = _coverage(X, g)
    out["chan"][si] = _channel_err(sys, X, Y, g)
    out["ess_anc"][si] = 1.0 if ess is None else ess
    out["surv_anc"][si] = 1.0 if surv is None else surv
    out["fe"][si] = fe


# ---------------------------------------------------------------------------
def run_wfr(sys: SepSystem, cfg: RunConfig, rows: int, seed: int, sham_source=None):
    g, dev, dt = sys.grid, sys.device, sys.dtype
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    X, Y = _init_state(sys, cfg, rows, gen)
    acc = MeanForceAccumulator(rows, g, cfg.bw_mf, cfg.n_min, dev, dt)
    mask = g.eval_mask(dev, dt)
    n_outer = cfg.n_steps // cfg.n_cond
    save_outer = max(1, cfg.save_every // cfg.n_cond)
    n_saves = n_outer // save_outer
    out = _saver(rows, g, n_saves, dev, dt)
    theta0 = as_col(cfg.theta, rows, dev, dt).squeeze(1)
    kappa_col = as_col(cfg.kappa, rows, dev, dt)
    ar = torch.arange(cfg.N, device=dev).unsqueeze(0).expand(rows, cfg.N)
    anc, anc_g = ar.clone(), ar.clone()
    dtau = cfg.n_cond * cfg.dt
    anneal = cfg.kappa_end is not None
    if anneal:
        k_end = as_col(cfg.kappa_end, rows, dev, dt)
        ratio = k_end / kappa_col
    reset_it = None if cfg.acc_reset_at is None else int(cfg.acc_reset_at * n_outer)
    turnovers, si = [], 0
    for it in range(n_outer):
        if reset_it is not None and it == reset_it:
            acc.S0.zero_(); acc.S1.zero_()
        k_now = kappa_col * (ratio ** (it / max(n_outer - 1, 1))) if anneal else kappa_col
        for k in range(cfg.n_cond):
            Y = sys.step_fiber(X, Y, cfg.dt, gen)
            if k >= cfg.n_eq:
                acc.deposit(X, sys.mean_force(X, Y))
        if cfg.w_mode == "sde":
            Xn = w_step_sde(X, k_now, dtau, g, gen)
        elif cfg.w_mode == "flow":
            Xn = w_step_flow(X, cfg.kappa, dtau, g, cfg.bw_kde, cfg.w_flow_clip)
        elif cfg.w_mode == "none":
            Xn = X
        else:
            raise ValueError(cfg.w_mode)
        if cfg.lift == "oracle":
            Y = sys.sample_conditional(Xn, gen)
        elif cfg.lift == "scaled":
            Y = sys.clamp_y(sys.lift_scaled(X, Xn, Y))
        elif cfg.lift != "identity":
            raise ValueError(cfg.lift)
        X = Xn
        if cfg.fr_rule != "none" and (it % cfg.fr_every == 0):
            sham = None
            if cfg.fr_rule == "sham":
                sham = (sham_source[it] if sham_source is not None
                        else torch.zeros(rows, device=dev, dtype=torch.long))
            sel, info = selection_indices(X, g, cfg.fr_rule, theta0, gen,
                                          bw=cfg.bw_kde, n_bins=cfg.n_bins_count,
                                          alpha_ess=cfg.alpha_ess, sham_turnover=sham)
            X = torch.gather(X, 1, sel)
            Y = torch.gather(Y, 1, sel.unsqueeze(-1).expand(-1, -1, Y.shape[-1]))
            anc = torch.gather(anc, 1, sel)
            anc_g = torch.gather(anc_g, 1, sel)
            if cfg.fr_jitter > 0:
                X = g.enforce(X + cfg.fr_jitter * torch.randn(
                    X.shape, device=dev, dtype=dt, generator=gen))
            turnovers.append(info["turnover"])
        else:
            turnovers.append(torch.zeros(rows, device=dev, dtype=torch.long))
        if (it + 1) % cfg.ess_window == 0:
            anc = ar.clone()
        if (it + 1) % save_outer == 0 and si < n_saves:
            e, _ = ancestor_stats(anc, cfg.N)
            _record(out, si, sys, cfg, acc, X, Y, mask,
                    float((it + 1) * cfg.n_cond * cfg.N),
                    ess=e / cfg.N, surv=surviving_ancestors(anc_g, cfg.N) / cfg.N)
            si += 1
    out["turnover"] = torch.stack(turnovers, 0) if turnovers else None
    out["X_final"], out["Y_final"] = X, Y
    return out


# ---------------------------------------------------------------------------
def run_abf(sys: SepSystem, cfg: RunConfig, rows: int, seed: int):
    g, dev, dt = sys.grid, sys.device, sys.dtype
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    X, Y = _init_state(sys, cfg, rows, gen)
    acc = MeanForceAccumulator(rows, g, cfg.bw_mf, cfg.n_min, dev, dt)
    mask = g.eval_mask(dev, dt)
    n_saves = cfg.n_steps // cfg.save_every
    out = _saver(rows, g, n_saves, dev, dt)
    bias_col = as_col(cfg.bias_n_min, rows, dev, dt)
    si = 0
    for n in range(cfg.n_steps):
        bias = interp1d(X, acc.mean_force(bias_col), g)
        X, Y = sys.step_full(X, Y, cfg.dt, gen, bias_force_x=bias)
        acc.deposit(X, sys.mean_force(X, Y))
        if (n + 1) % cfg.save_every == 0 and si < n_saves:
            _record(out, si, sys, cfg, acc, X, Y, mask, float((n + 1) * cfg.N))
            si += 1
    out["X_final"], out["Y_final"] = X, Y
    return out


def run_shus(sys: SepSystem, cfg: RunConfig, rows: int, seed: int):
    g, dev, dt = sys.grid, sys.device, sys.dtype
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    X, Y = _init_state(sys, cfg, rows, gen)
    sh = ShusAccumulator(rows, g, sys.p.beta, cfg.shus_eps_bw, dev, dt, cfg.shus_gain)
    mask = g.eval_mask(dev, dt)
    n_saves = cfg.n_steps // cfg.save_every
    out = _saver(rows, g, n_saves, dev, dt)
    si = 0
    for n in range(cfg.n_steps):
        X, Y = sys.step_full(X, Y, cfg.dt, gen, bias_force_x=sh.bias_force_at(X))
        sh.deposit(X)
        if (n + 1) % cfg.shus_block == 0:
            sh.update(cfg.dt, cfg.N)
        if (n + 1) % cfg.save_every == 0 and si < n_saves:
            p = kde_marginal(X, g, cfg.bw_kde)
            out["F"][si] = sh.f_estimate(mask)
            out["p"][si] = p
            out["kl"][si] = kl_to_uniform(p, g)
            out["tv"][si] = tv_to_uniform(p, g)
            out["cov"][si] = _coverage(X, g)
            out["chan"][si] = _channel_err(sys, X, Y, g)
            out["ess_anc"][si] = 1.0
            out["surv_anc"][si] = 1.0
            out["fe"][si] = float((n + 1) * cfg.N)
            si += 1
    out["X_final"], out["Y_final"] = X, Y
    return out


def run_unbiased(sys: SepSystem, cfg: RunConfig, rows: int, seed: int):
    g, dev, dt = sys.grid, sys.device, sys.dtype
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    X, Y = _init_state(sys, cfg, rows, gen)
    acc = MeanForceAccumulator(rows, g, cfg.bw_mf, cfg.n_min, dev, dt)
    mask = g.eval_mask(dev, dt)
    n_saves = cfg.n_steps // cfg.save_every
    out = _saver(rows, g, n_saves, dev, dt)
    si = 0
    for n in range(cfg.n_steps):
        X, Y = sys.step_full(X, Y, cfg.dt, gen)
        acc.deposit(X, sys.mean_force(X, Y))
        if (n + 1) % cfg.save_every == 0 and si < n_saves:
            _record(out, si, sys, cfg, acc, X, Y, mask, float((n + 1) * cfg.N))
            si += 1
    out["X_final"], out["Y_final"] = X, Y
    return out


# ---------------------------------------------------------------------------
def run_reti(sys: SepSystem, cfg: RunConfig, rows: int, seed: int):
    """Stratified TI + Hamiltonian replica exchange between adjacent windows.

    Replica i is pinned at window centre z_i (a hard constraint, as in blue-moon
    TI); every `n_ex` steps an alternating even/odd sweep proposes swapping the
    fiber configurations of adjacent windows with the Metropolis rate
        min(1, exp(-beta [V(z_i,Y_j) + V(z_j,Y_i) - V(z_i,Y_i) - V(z_j,Y_j)])).
    The two NEW energies per pair are charged to the force budget (1 per replica
    per sweep), so the reported force-evaluation axis stays honest.
    """
    g, dev, dt = sys.grid, sys.device, sys.dtype
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    assert cfg.init in ("grid_cold", "grid_warm"), "reti needs window initialization"
    X, Y = _init_state(sys, cfg, rows, gen)
    acc = MeanForceAccumulator(rows, g, cfg.bw_mf, cfg.n_min, dev, dt)
    mask = g.eval_mask(dev, dt)
    # COST MATCHING: an exchange sweep costs 2 new energies per pair = ~N per sweep,
    # i.e. one extra force-equivalent per replica every n_ex steps.  Shorten the inner
    # loop so the TOTAL charge equals N * cfg.n_steps, the budget every other arm uses.
    n_inner = int(round(cfg.n_steps / (1.0 + 1.0 / cfg.n_ex)))
    n_saves = max(1, n_inner // cfg.save_every)
    out = _saver(rows, g, n_saves, dev, dt)
    N = cfg.N
    M = cfg.n_windows if cfg.n_windows else N
    rep = N // M                       # replicas per window
    acc_num = torch.zeros((), device=dev, dtype=dt)
    acc_den = torch.zeros((), device=dev, dtype=dt)
    fe = 0.0
    si, parity = 0, 0
    for n in range(n_inner):
        Y = sys.step_fiber(X, Y, cfg.dt, gen)
        acc.deposit(X, sys.mean_force(X, Y))
        fe += N
        if (n + 1) % cfg.n_ex == 0:
            i0 = parity
            parity ^= 1
            # partner of replica i is i + rep (the same slot in the next window)
            base_idx = torch.arange(0, (M - 1) * rep, device=dev)
            wnd = base_idx // rep
            idx = base_idx[(wnd % 2) == i0]
            if idx.numel():
                jdx = idx + rep
                Xi, Xj = X[:, idx], X[:, jdx]
                Yi, Yj = Y[:, idx], Y[:, jdx]
                Enew = sys.energy(Xi, Yj) + sys.energy(Xj, Yi)
                Eold = sys.energy(Xi, Yi) + sys.energy(Xj, Yj)
                pacc = torch.exp(torch.clamp(-sys.p.beta * (Enew - Eold), max=0.0))
                u = torch.rand(pacc.shape, device=dev, dtype=dt, generator=gen)
                sw = (u < pacc).unsqueeze(-1)
                Y[:, idx] = torch.where(sw, Yj, Yi)
                Y[:, jdx] = torch.where(sw, Yi, Yj)
                acc_num += (u < pacc).to(dt).sum()
                acc_den += float(pacc.numel())
                fe += 2.0 * idx.numel() * rows / rows      # 2 new energies per pair
        if (n + 1) % cfg.save_every == 0 and si < n_saves:
            _record(out, si, sys, cfg, acc, X, Y, mask, fe)
            si += 1
    out["X_final"], out["Y_final"] = X, Y
    out["ex_accept"] = float(acc_num / torch.clamp(acc_den, min=1.0))
    return out


ARMS = {"wfr": run_wfr, "abf": run_abf, "shus": run_shus,
        "unbiased": run_unbiased, "reti": run_reti}
