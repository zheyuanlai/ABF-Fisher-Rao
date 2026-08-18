"""Analytic 2D periodic surrogate: mollified SHUS (+ optional FR) on the torus.

Purpose (docs/PREREGISTRATION_APPLICATION_MAP.md, Phase D): a technical validation
target for the 2D periodic engine BEFORE any atomistic (phi, psi) system — exact
reference, known basins, cheap large-K testing, bandwidth and count-vs-FR
resolution studies. Not an application result.

System: overdamped Langevin directly on xi = (phi, psi) in T^2 (identity CV, so the
reference free energy IS the potential, exactly):

    V(phi, psi) = H1 (1 - cos 2 phi)/2 + H2 (1 - cos 2 psi)/2
                  + Hc cos(phi) cos(psi),

four basins near (0,0), (0,pi), (pi,0), (pi,pi); H1/H2 set the barriers along each
axis and Hc splits the basin depths by 2*Hc (a controllable establishment
challenge: the deep pair must flood the shallow pair through saddles).

Batching mirrors the gateway engine: B (config, seed) rows x M methods share
initial conditions and Langevin noise (paired arms); FR events gather walker
arrays only (estimator protection). Full pmf/marginal profiles are stored at a
thinned cadence (profile_every) — 2D frames are ~50x larger than 1D ones.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch

from ..events2d import fr_event2
from ..grid import DEVICE, DTYPE, EPS
from ..grid2d import (GridT2, binned_density2, integral2, kl_to_uniform2,
                      periodic_gaussian_kernel, tv_to_uniform2, wrap_periodic)
from ..resampling import ancestor_stats, surviving_ancestors
from ..shus2d import ShusAccumulator2, mollified_fixed_point2
from .gateway import Method, _fires_at_block, _schedule_source  # shared arm logic

REFERENCE_ID = "torus2d-analytic-v1"
PI = math.pi
GRID2 = GridT2(x1min=-PI, L1=2 * PI, n1=72, x2min=-PI, L2=2 * PI, n2=72)

# four basin quadrants, labeled by the nearest well center
BASINS = ((0.0, 0.0), (0.0, PI), (PI, 0.0), (PI, PI))
REGIONS = ("b00", "b0p", "bp0", "bpp")


# -----------------------------------------------------------------------------
# physics
# -----------------------------------------------------------------------------
def V_of(phi, psi, H1, H2, Hc):
    return (0.5 * H1 * (1.0 - torch.cos(2.0 * phi))
            + 0.5 * H2 * (1.0 - torch.cos(2.0 * psi))
            + Hc * torch.cos(phi) * torch.cos(psi))


def gradV_of(phi, psi, H1, H2, Hc):
    dphi = H1 * torch.sin(2.0 * phi) - Hc * torch.sin(phi) * torch.cos(psi)
    dpsi = H2 * torch.sin(2.0 * psi) - Hc * torch.cos(phi) * torch.sin(psi)
    return dphi, dpsi


def reference_surface(H1, H2, Hc, device=DEVICE, dtype=DTYPE):
    """Exact F_ref on the grid (identity CV), zero-mean centered.  -> (n1, n2)."""
    P1, P2 = GRID2.mesh(device, dtype)
    F = V_of(P1, P2, H1, H2, Hc)
    return F - F.mean()


def region_of(phi, psi):
    """Quadrant label 0..3 by nearest well center (cos > 0 <-> nearer 0 than pi)."""
    near_pi_1 = (torch.cos(phi) < 0).long()
    near_pi_2 = (torch.cos(psi) < 0).long()
    return near_pi_1 * 2 + near_pi_2


# -----------------------------------------------------------------------------
# configuration
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Torus2DConfig:
    beta: float = 4.0
    H1: float = 2.0
    H2: float = 2.0
    Hc: float = 0.5
    K: int = 1024
    dt: float = 1e-3
    n_steps: int = 200_000
    block: int = 20
    eps_bw: float = 0.10          # mollifier bandwidth (radians)
    eta_bw: float = 0.25          # KDE bandwidth (marginal / FR score)
    n_saves: int = 400
    profile_every: int = 8        # store pmf/marginal every k-th save
    ess_window_steps: int = 4000
    init: str = "b00"             # all walkers in the (0,0) basin

    @property
    def T_total(self) -> float:
        return self.n_steps * self.dt

    def barrier_kT(self) -> float:
        """Lowest saddle out of the deep basin, in kT."""
        return self.beta * (min(self.H1, self.H2) - self.Hc)


def analytic_floors(cfg: Torus2DConfig, device="cpu", dtype=DTYPE):
    """(e*, KL*) of the mollified fixed point on this cell (pre-run computable)."""
    F_ref = reference_surface(cfg.H1, cfg.H2, cfg.Hc, device, dtype)
    return mollified_fixed_point2(F_ref, cfg.beta, cfg.eps_bw, GRID2, device, dtype)


# -----------------------------------------------------------------------------
# the batched simulation
# -----------------------------------------------------------------------------
def simulate_batch(configs, seeds, methods, batch_seed=12345, device=DEVICE,
                   dtype=DTYPE, progress=None):
    cfgs, methods = list(configs), list(methods)
    assert len(cfgs) == len(seeds)
    B, M = len(cfgs), len(methods)
    R = B * M

    c0 = cfgs[0]
    for c in cfgs:
        for a in ("K", "dt", "n_steps", "block", "eps_bw", "eta_bw", "n_saves",
                  "profile_every", "ess_window_steps"):
            assert getattr(c, a) == getattr(c0, a), f"non-uniform {a} across configs"
    K, dt, n_steps, block = c0.K, c0.dt, c0.n_steps, c0.block
    assert n_steps % block == 0
    n_blocks = n_steps // block

    by_name = {m.name: m for m in methods}
    scheds = [_schedule_source(m, by_name) for m in methods]
    name_col = {m.name: j for j, m in enumerate(methods)}
    partner_col = [name_col[m.shadows] if m.sham else j for j, m in enumerate(methods)]
    partner = torch.tensor([b * M + partner_col[j] for b in range(B) for j in range(M)],
                           device=device, dtype=torch.long)
    event_blocks = [k for k in range(1, n_blocks + 1)
                    if any(_fires_at_block(s, k, n_blocks) for s in scheds)]
    n_events = len(event_blocks)
    fires = torch.tensor(
        [[_fires_at_block(scheds[j], k, n_blocks) for b in range(B) for j in range(M)]
         for k in event_blocks], device=device, dtype=torch.bool
    ).reshape(n_events, R) if n_events else torch.zeros((0, R), device=device,
                                                        dtype=torch.bool)
    is_fr_row = torch.tensor([m.use_fr and not m.sham for m in methods],
                             device=device).repeat(B)
    is_sham_row = torch.tensor([m.sham for m in methods], device=device).repeat(B)
    is_coarse_row = torch.tensor([m.coarse_bins > 0 for m in methods],
                                 device=device).repeat(B)
    coarse_nb = max((m.coarse_bins for m in methods), default=0)
    assert all(m.coarse_bins in (0, coarse_nb) for m in methods), \
        "all count-balancing arms in one batch must share coarse_bins"
    theta0 = torch.tensor([m.theta if (m.use_fr and not m.sham) else 0.0
                           for m in methods], device=device, dtype=dtype).repeat(B)
    alpha_ess = torch.tensor([m.alpha_ess for m in methods], device=device,
                             dtype=dtype).repeat(B)
    assert all(m.g_shus > 0 for m in methods), "g_shus must be positive"
    gain = torch.tensor([m.g_shus for m in methods], device=device,
                        dtype=dtype).repeat(B)

    k1e, r1e = periodic_gaussian_kernel(c0.eta_bw, GRID2.dx1, GRID2.n1, device, dtype)
    k2e, r2e = periodic_gaussian_kernel(c0.eta_bw, GRID2.dx2, GRID2.n2, device, dtype)

    def cfg_b(fn):
        return torch.tensor([fn(c) for c in cfgs], device=device, dtype=dtype)
    beta_b = cfg_b(lambda c: c.beta)
    H1_b, H2_b, Hc_b = (cfg_b(lambda c: c.H1), cfg_b(lambda c: c.H2),
                        cfg_b(lambda c: c.Hc))

    def to_run(t_b):
        return t_b.repeat_interleave(M).reshape(R, 1)
    beta = to_run(beta_b)
    H1, H2, Hc = to_run(H1_b), to_run(H2_b), to_run(Hc_b)
    noise_amp = torch.sqrt(2.0 * dt / beta)

    F_ref = torch.stack([reference_surface(float(c.H1), float(c.H2), float(c.Hc),
                                           device, dtype)
                         for c in cfgs]).repeat_interleave(M, dim=0)  # (R, n1, n2)
    rho_ref = torch.exp(-beta.reshape(R, 1, 1) * F_ref)
    rho_ref = rho_ref / integral2(rho_ref, GRID2).reshape(R, 1, 1)

    # initial conditions: all walkers jittered in the (0,0) well; paired across arms
    X1 = torch.empty((B, K), device=device, dtype=dtype)
    X2 = torch.empty((B, K), device=device, dtype=dtype)
    for b, sd in enumerate(seeds):
        rng = np.random.default_rng(1000 + int(sd))
        assert cfgs[b].init == "b00", f"unknown init {cfgs[b].init!r}"
        X1[b] = torch.as_tensor(rng.normal(0.0, 0.15, K), device=device, dtype=dtype)
        X2[b] = torch.as_tensor(rng.normal(0.0, 0.15, K), device=device, dtype=dtype)
    X1 = wrap_periodic(X1.repeat_interleave(M, dim=0), GRID2.x1min, GRID2.L1)
    X2 = wrap_periodic(X2.repeat_interleave(M, dim=0), GRID2.x2min, GRID2.L2)
    anc = torch.arange(K, device=device).unsqueeze(0).expand(R, K).clone()
    anc_g = anc.clone()
    shus = ShusAccumulator2(R, GRID2, beta, c0.eps_bw, device, dtype, gain=gain)
    dep_ref_cur = torch.full((R,), float("nan"), device=device, dtype=dtype)
    dep_self_cur = torch.full((R,), float("nan"), device=device, dtype=dtype)

    gen_n = torch.Generator(device=device)
    gen_n.manual_seed(2000 + batch_seed)
    gen_f = torch.Generator(device=device)
    gen_f.manual_seed(3000 + batch_seed)

    save_steps = sorted({*range(0, n_steps, max(1, n_steps // c0.n_saves)),
                         n_steps - 1})
    n_saves = len(save_steps)
    save_set = set(save_steps)
    prof_steps = save_steps[:: c0.profile_every]
    if save_steps[-1] not in prof_steps:
        prof_steps = prof_steps + [save_steps[-1]]
    n_prof = len(prof_steps)
    prof_set = set(prof_steps)

    ts = {k: torch.zeros((R, n_saves), device=device, dtype=dtype) for k in
          ("l2_f", "kl_u", "tv_u", "ess_anc", "wmax", "ess_anc_glob", "wmax_glob",
           "n_anc", "dep_ref", "dep_self")}
    ts["P"] = torch.zeros((R, n_saves, 4), device=device, dtype=dtype)
    prof = {"pmf": torch.zeros((R, n_prof, GRID2.n1, GRID2.n2), device=device,
                               dtype=dtype),
            "marg": torch.zeros((R, n_prof, GRID2.n1, GRID2.n2), device=device,
                                dtype=dtype)}
    ev = {k: torch.zeros((R, max(n_events, 1)), device=device, dtype=dtype)
          for k in ("theta", "ess_fr", "turnover")}
    tot_turn = torch.zeros(R, device=device, dtype=dtype)
    save_ptr, prof_ptr, event_ptr = 0, 0, 0

    for step in range(n_steps):
        if c0.ess_window_steps > 0 and step % c0.ess_window_steps == 0:
            anc = torch.arange(K, device=device).unsqueeze(0).expand(R, K).clone()

        # ---- physical propagation (overdamped EM on the torus) -----------------
        g1, g2 = gradV_of(X1, X2, H1, H2, Hc)
        b1, b2 = shus.bias_force_at(X1, X2)
        z1 = torch.randn((B, K), device=device, dtype=dtype,
                         generator=gen_n).repeat_interleave(M, dim=0)
        z2 = torch.randn((B, K), device=device, dtype=dtype,
                         generator=gen_n).repeat_interleave(M, dim=0)
        X1 = wrap_periodic(X1 + (-g1 + b1) * dt + noise_amp * z1,
                           GRID2.x1min, GRID2.L1)
        X2 = wrap_periodic(X2 + (-g2 + b2) * dt + noise_amp * z2,
                           GRID2.x2min, GRID2.L2)

        # ---- SHUS deposit -------------------------------------------------------
        shus.deposit(X1, X2)

        # ---- block boundary: update, then (maybe) an FR event -------------------
        if (step + 1) % block == 0:
            r_n = shus.R / integral2(shus.R, GRID2).reshape(R, 1, 1)
            inc = shus.update(dt, K)
            d_n = inc / torch.clamp(integral2(inc, GRID2), min=EPS).reshape(R, 1, 1)
            dep_ref_cur = torch.sqrt(((d_n - rho_ref) ** 2).mean(dim=(1, 2)))
            dep_self_cur = torch.sqrt(((d_n - r_n) ** 2).mean(dim=(1, 2)))
            blk = (step + 1) // block
            if event_ptr < n_events and event_blocks[event_ptr] == blk:
                active = fires[event_ptr]
                sel, turn, theta_used, essf = fr_event2(
                    X1, X2, active & is_fr_row, active & is_sham_row, is_coarse_row,
                    coarse_nb, partner, theta0, alpha_ess, k1e, r1e, k2e, r2e,
                    GRID2, gen_f)
                ev["theta"][:, event_ptr] = theta_used
                ev["ess_fr"][:, event_ptr] = essf
                ev["turnover"][:, event_ptr] = turn.to(dtype)
                tot_turn += turn.to(dtype)
                # ESTIMATOR PROTECTION: walker arrays only
                X1 = torch.gather(X1, 1, sel)
                X2 = torch.gather(X2, 1, sel)
                anc = torch.gather(anc, 1, sel)
                anc_g = torch.gather(anc_g, 1, sel)
                event_ptr += 1

        # ---- checkpoints ---------------------------------------------------------
        if step in save_set:
            F_hat = shus.f_estimate()
            d = F_hat - F_ref
            d = d - d.mean(dim=(1, 2), keepdim=True)
            ts["l2_f"][:, save_ptr] = torch.sqrt((d * d).mean(dim=(1, 2)))
            p_hat = binned_density2(X1, X2, k1e, r1e, k2e, r2e, GRID2)
            ts["kl_u"][:, save_ptr] = kl_to_uniform2(p_hat, GRID2)
            ts["tv_u"][:, save_ptr] = tv_to_uniform2(p_hat, GRID2)
            e_, w_ = ancestor_stats(anc, K)
            ts["ess_anc"][:, save_ptr] = e_
            ts["wmax"][:, save_ptr] = w_
            eg_, wg_ = ancestor_stats(anc_g, K)
            ts["ess_anc_glob"][:, save_ptr] = eg_
            ts["wmax_glob"][:, save_ptr] = wg_
            ts["n_anc"][:, save_ptr] = surviving_ancestors(anc_g, K)
            ts["dep_ref"][:, save_ptr] = dep_ref_cur
            ts["dep_self"][:, save_ptr] = dep_self_cur
            lab = region_of(X1, X2)
            for k in range(4):
                ts["P"][:, save_ptr, k] = (lab == k).to(dtype).mean(dim=1)
            if step in prof_set:
                prof["pmf"][:, prof_ptr] = F_hat
                prof["marg"][:, prof_ptr] = p_hat
                prof_ptr += 1
            save_ptr += 1
        if progress is not None and step % progress == 0:
            print(f"    step {step}/{n_steps}", flush=True)

    totP = ts["P"].sum(dim=2)
    worstP = float((totP - 1.0).abs().max())
    assert worstP < 1e-9, f"region fractions do not sum to 1 (worst {worstP:.3e})"

    # ---- finalize ---------------------------------------------------------------
    t_axis = np.array([s * dt for s in save_steps])
    prof_t = np.array([s * dt for s in prof_steps])
    ev_t = np.array([k * block * dt for k in event_blocks])

    def npy(t):
        return t.detach().cpu().numpy()

    recs = []
    for b in range(B):
        for m in range(M):
            r = b * M + m
            l2 = npy(ts["l2_f"][r])
            recs.append(dict(
                config=asdict(cfgs[b]), seed=int(seeds[b]),
                method=asdict(methods[m]), batch_seed=batch_seed,
                reference_id=REFERENCE_ID,
                eval_window=(GRID2.x1min, GRID2.x1min + GRID2.L1,
                             GRID2.x2min, GRID2.x2min + GRID2.L2),
                time=t_axis, profile_time=prof_t,
                x1_grid=npy(GRID2.x1(device, dtype)),
                x2_grid=npy(GRID2.x2(device, dtype)),
                F_ref=npy(F_ref[r]),
                pmf_t=npy(prof["pmf"][r]), marginal_t=npy(prof["marg"][r]),
                l2_f_t=l2, kl_u_t=npy(ts["kl_u"][r]), tv_u_t=npy(ts["tv_u"][r]),
                ess_anc_t=npy(ts["ess_anc"][r]), wmax_t=npy(ts["wmax"][r]),
                ess_anc_glob_t=npy(ts["ess_anc_glob"][r]),
                wmax_glob_t=npy(ts["wmax_glob"][r]), n_anc_t=npy(ts["n_anc"][r]),
                dep_ref_l2_t=npy(ts["dep_ref"][r]),
                dep_self_l2_t=npy(ts["dep_self"][r]),
                P_regions=npy(ts["P"][r]),
                event_time=ev_t, event_theta=npy(ev["theta"][r]),
                event_ess_fr=npy(ev["ess_fr"][r]),
                event_turnover=npy(ev["turnover"][r]),
                final_l2_f=float(l2[-1]),
                int_l2_f=float(np.trapezoid(l2, t_axis)),
                total_turnover=float(tot_turn[r]),
            ))
    return recs
