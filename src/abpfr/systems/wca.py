"""WCA dimer in explicit 2D solvent: mollified SHUS (+ optional FR) on xi = dimer length.

System (physics ported from ABF-Fisher-Rao src/wca_abffr_core.py, docs/PROVENANCE.md):
100 particles (a 2-particle dimer + 98 WCA solvent) in a periodic 2D box, dimer
double-well of height h and width w, overdamped Euler-Maruyama with clipped forces,

    xi(q) = (|q0 - q1| - r0) / (2 w)  in  [-0.2, 1.2],   r0 = 2^{1/6} sigma,

soft walls outside the xi window. One replica = one full box: an FR clone copies the
ENTIRE solvent environment, which is precisely the conditional-relaxation risk the
theory branch quantifies.

Cells are named b{beta}h{h} (e.g. b1h2: beta=1, h=2). Only b1h2 has the corrected
high-precision reference (hp_v3); load_reference() HARD-refuses anything else, and
runs on other cells simply carry no e_F series (gates T_hit/T_est are reference-free;
full pmf_t is stored so cells can be rescored if they earn an hp reference later).

Dynamics run in float32 (pair forces dominate; the old campaign's validated choice);
the SHUS estimator and all reaction-coordinate statistics run in float64.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

import numpy as np
import torch

from ..events import fr_event
from ..fisher_rao import kl_to_uniform, tv_to_uniform
from ..grid import (DEVICE, EPS, Grid1D, binned_density, gaussian_kernel,
                    interp1d, trapz)
from ..resampling import ancestor_stats, surviving_ancestors
from ..shus import ShusAccumulator
from .gateway import Method  # one Method dataclass for every system

REFERENCE_ID = "wca-hp_v3-b1h2"
REFERENCE_LABEL_PREFIX = "HP reference v2"      # the corrected, unsmoothed build
GRID = Grid1D(xmin=-0.2, xmax=1.2, n=160, eval_lo=0.0, eval_hi=1.0)

# regions along xi: compact / transition / stretched
Z_COMPACT, Z_STRETCHED = 0.25, 0.75
REGIONS = ("compact", "transition", "stretched")

DYN_DTYPE = torch.float32
RC_DTYPE = torch.float64


@dataclass(frozen=True)
class WCAConfig:
    """One (cell, seed) row: physics + SHUS numerics."""
    beta: float = 1.0
    h: float = 2.0               # dimer barrier height
    w: float = 2.0
    n_dim: int = 10              # sqrt(number of particles)
    a: float = 1.5               # lattice constant; box L = n_dim * a
    sigma: float = 1.0
    epsilon: float = 1.0
    min_r: float = 0.65
    force_clip: float = 250.0
    wall_strength: float = 80.0
    K: int = 1024                # replicas (whole boxes)
    dt: float = 2e-3
    n_steps: int = 250_000
    block: int = 20              # SHUS adaptation block, in MD steps
    eps_bw: float = 0.025        # mollifier bandwidth (deposits)
    eta_bw: float = 0.07         # KDE bandwidth (marginal / FR score)
    n_saves: int = 400
    ess_window_steps: int = 4000

    @property
    def n_particles(self) -> int:
        return self.n_dim * self.n_dim

    @property
    def box_length(self) -> float:
        return self.n_dim * self.a

    @property
    def r0(self) -> float:
        return 2.0 ** (1.0 / 6.0) * self.sigma

    @property
    def T_total(self) -> float:
        return self.n_steps * self.dt

    def cell_name(self) -> str:
        return f"b{self.beta:g}h{self.h:g}"


# -----------------------------------------------------------------------------
# reference (hp_v3 only; hard guard)
# -----------------------------------------------------------------------------
def load_reference(cfg: WCAConfig, device=DEVICE):
    """The corrected high-precision reference for b1h2, or a refusal.

    Returns (F_ref (G,), meta) with F_ref centered on the eval window, float64.
    Refuses: wrong cell, wrong grid, wrong label, missing metadata. No silent
    fallback to superseded references — that mistake was already paid for once.
    """
    base = os.path.join(os.path.dirname(__file__), "..", "references")
    path = os.path.join(base, "wca_hp_v3_b1h2.npz")
    meta_path = os.path.join(base, "wca_hp_v3_b1h2.meta.json")
    assert os.path.exists(path) and os.path.exists(meta_path), \
        "hp_v3 reference artifact or metadata missing from src/abpfr/references/"
    with open(meta_path) as f:
        meta = json.load(f)
    assert meta.get("reference_version", "").startswith("hp"), \
        f"reference metadata version {meta.get('reference_version')!r} is not an hp build"
    cell = meta["cell"]
    for k in ("beta", "h", "w", "n_dim", "a", "sigma", "epsilon"):
        assert abs(float(cell[k]) - float(getattr(cfg, k))) < 1e-12, (
            f"reference cell mismatch on {k}: reference {cell[k]} vs config "
            f"{getattr(cfg, k)} — refusing to score this cell against b1h2's reference")
    with np.load(path) as z:
        label = str(z["label"])
        assert label.startswith(REFERENCE_LABEL_PREFIX), \
            f"reference label {label!r} is not the corrected HP build"
        grid = z["grid"]
        F = z["free_energy"].astype(np.float64)
    assert len(grid) == GRID.n and abs(grid[0] - GRID.xmin) < 1e-9 \
        and abs(grid[-1] - GRID.xmax) < 1e-9, "reference grid != evaluation grid"
    F_ref = torch.as_tensor(F, device=device, dtype=RC_DTYPE)
    mask = GRID.eval_mask(device, RC_DTYPE)
    F_ref = F_ref - F_ref[mask].mean()
    return F_ref, meta


def load_gate_proxy(cell_name: str, device=DEVICE):
    """Superseded phase-tier TI profile, permitted ONLY to set gate tolerances
    (the KL* term of D_tol needs the rough amplitude of the mollified fixed
    point, nothing more). NEVER use these to score e_F -- that is the exact
    mistake the old campaign paid for. Returns raw F (G,) float64."""
    assert cell_name in ("b2h6", "b4h1"), \
        f"no gate proxy for {cell_name}; b1h2 uses the hp_v3 reference"
    path = os.path.join(os.path.dirname(__file__), "..", "references",
                        f"wca_phase_proxy_{cell_name}.npz")
    with np.load(path) as z:
        F = z["free_energy"].astype(np.float64)
        grid = z["grid"]
    assert len(grid) == GRID.n and abs(grid[0] - GRID.xmin) < 1e-9
    return torch.as_tensor(F, device=device, dtype=RC_DTYPE)


def mollified_marginal_floor(F_ref, beta, eps_bw, device=DEVICE):
    """Analytic floors of mollified SHUS for a NUMERIC profile: (e*, KL*).

    Same identity as gateway.mollified_fixed_point (R* = K_eps * e^{-beta F}),
    evaluated on the WCA grid from a reference profile tensor (G,)."""
    from ..grid import smooth
    F = F_ref.reshape(1, -1).to(device=device, dtype=RC_DTYPE)
    mask = GRID.eval_mask(device, RC_DTYPE)
    rho = torch.exp(-beta * (F - F[:, mask].mean()))
    k, r = gaussian_kernel(eps_bw, GRID.dx, device, RC_DTYPE)
    rho_m = smooth(rho, k, r, GRID.dx)
    F_star = -torch.log(torch.clamp(rho_m, min=EPS)) / beta
    d = (F_star - F)[:, mask]
    d = d - d.mean(dim=1, keepdim=True)
    e_star = float(torch.sqrt((d * d).mean()))
    p_star = rho / torch.clamp(rho_m, min=EPS)
    p_star = p_star / trapz(p_star, GRID.dx).unsqueeze(1)
    import math as _m
    kl_star = float(trapz(p_star * (torch.log(torch.clamp(p_star, min=EPS))
                                    - _m.log(1.0 / GRID.volume)), GRID.dx))
    return e_star, kl_star


# -----------------------------------------------------------------------------
# physics (batched over flattened boxes)
# -----------------------------------------------------------------------------
def wrap(q, L):
    return torch.remainder(q, L)


def minimum_image(delta, L):
    return delta - L * torch.round(delta / L)


class WCAEngine:
    """Pair-list WCA + dimer forces for (Bx, N, 2) boxes; h and beta per box."""

    def __init__(self, cfg: WCAConfig, device, dtype=DYN_DTYPE):
        self.N = cfg.n_particles
        self.L = float(cfg.box_length)
        self.sigma = float(cfg.sigma)
        self.epsilon = float(cfg.epsilon)
        self.w = float(cfg.w)
        self.r0 = float(cfg.r0)
        self.min_r = float(cfg.min_r)
        self.force_clip = float(cfg.force_clip)
        pair_i, pair_j = torch.triu_indices(self.N, self.N, offset=1, device=device)
        keep = ~((pair_i == 0) & (pair_j == 1))     # dimer pair handled analytically
        self.pair_i = pair_i[keep].long()
        self.pair_j = pair_j[keep].long()

    def force(self, q, h_box):
        """q: (Bx, N, 2); h_box: (Bx, 1) dimer heights.  Returns clipped forces."""
        Bx = q.shape[0]
        qi = q.index_select(1, self.pair_i)
        qj = q.index_select(1, self.pair_j)
        delta = minimum_image(qi - qj, self.L)
        r = torch.linalg.norm(delta, dim=-1)
        r_safe = torch.clamp(r, min=self.min_r * self.sigma)
        active = r <= self.r0
        inv6 = (self.sigma / r_safe) ** 6
        inv12 = inv6 * inv6
        dVdr = 4.0 * self.epsilon * (-12.0 * inv12 + 6.0 * inv6) / r_safe
        dVdr = torch.where(active, dVdr, torch.zeros_like(dVdr))
        f_pair = (-dVdr / r_safe).unsqueeze(-1) * delta
        forces = torch.zeros_like(q)
        idx_i = self.pair_i.view(1, -1, 1).expand(Bx, -1, 2)
        idx_j = self.pair_j.view(1, -1, 1).expand(Bx, -1, 2)
        forces.scatter_add_(1, idx_i, f_pair)
        forces.scatter_add_(1, idx_j, -f_pair)

        d01 = minimum_image(q[:, 0, :] - q[:, 1, :], self.L)
        r01 = torch.linalg.norm(d01, dim=1).clamp_min(EPS)
        u = (r01 - self.r0 - self.w) / self.w
        dVdr_dim = -4.0 * h_box.squeeze(1) * u * (1.0 - u * u) / self.w
        f01 = (-dVdr_dim / r01).unsqueeze(-1) * d01
        forces[:, 0, :] += f01
        forces[:, 1, :] -= f01

        norm = torch.linalg.norm(forces, dim=-1, keepdim=True)
        scale = torch.clamp(self.force_clip / torch.clamp(norm, min=EPS), max=1.0)
        return forces * scale

    def energy(self, q, h_box):
        qi = q.index_select(1, self.pair_i)
        qj = q.index_select(1, self.pair_j)
        delta = minimum_image(qi - qj, self.L)
        r = torch.linalg.norm(delta, dim=-1)
        r_safe = torch.clamp(r, min=self.min_r * self.sigma)
        active = r <= self.r0
        inv6 = (self.sigma / r_safe) ** 6
        V = 4.0 * self.epsilon * (inv6 * inv6 - inv6) + self.epsilon
        V = torch.where(active, V, torch.zeros_like(V)).sum(dim=1)
        d01 = minimum_image(q[:, 0, :] - q[:, 1, :], self.L)
        r01 = torch.linalg.norm(d01, dim=1).clamp_min(EPS)
        u = (r01 - self.r0 - self.w) / self.w
        return V + h_box.squeeze(1) * (1.0 - u * u) ** 2


def reaction_coordinate(q, cfg: WCAConfig):
    d01 = minimum_image(q[:, 0, :] - q[:, 1, :], cfg.box_length)
    r01 = torch.linalg.norm(d01, dim=1).clamp_min(EPS)
    return (r01 - cfg.r0) / (2.0 * cfg.w)


def add_rc_force(q, forces, scalar, cfg: WCAConfig):
    """Add scalar(z) * grad(xi) to the dimer particles.  scalar: (Bx,)."""
    d01 = minimum_image(q[:, 0, :] - q[:, 1, :], cfg.box_length)
    r01 = torch.linalg.norm(d01, dim=1).clamp_min(EPS)
    g0 = d01 / (2.0 * cfg.w * r01[:, None])
    forces[:, 0, :] += scalar[:, None] * g0
    forces[:, 1, :] -= scalar[:, None] * g0
    return forces


def lattice_init(cfg: WCAConfig, K, seed, device, dtype=DYN_DTYPE, jitter=0.015):
    """K boxes on a jittered lattice, dimer at compact separation r0 (xi = 0)."""
    g = torch.Generator(device=device)
    g.manual_seed(1000 + int(seed))
    coords = [((0.5 + i) * cfg.a, (0.5 + j) * cfg.a)
              for i in range(cfg.n_dim) for j in range(cfg.n_dim)]
    base = torch.tensor(coords, device=device, dtype=dtype)
    q = base.unsqueeze(0).repeat(K, 1, 1)
    shift = torch.rand((K, 1, 2), device=device, dtype=dtype, generator=g) * cfg.box_length
    q = wrap(q + shift, cfg.box_length)
    if jitter > 0:
        q[:, 2:, :] = wrap(q[:, 2:, :] + jitter * torch.randn(
            q[:, 2:, :].shape, device=device, dtype=dtype, generator=g), cfg.box_length)
    q[:, 1, :] = wrap(q[:, 0, :] + torch.tensor([0.0, cfg.r0], device=device,
                                                dtype=dtype), cfg.box_length)
    return q


def region_fractions(z):
    """(R, K) xi values -> (R, 3) compact/transition/stretched fractions."""
    compact = (z < Z_COMPACT).to(z.dtype).mean(dim=1)
    stretched = (z > Z_STRETCHED).to(z.dtype).mean(dim=1)
    return torch.stack([compact, 1.0 - compact - stretched, stretched], dim=1)


# -----------------------------------------------------------------------------
# the batched simulation (mirrors gateway.simulate_batch; boxes instead of points)
# -----------------------------------------------------------------------------
def simulate_batch(configs, seeds, methods, batch_seed=12345, device=DEVICE,
                   progress=None, score_b1h2=True):
    """B (config, seed) rows x M methods, each row K whole boxes.

    e_F is computed ONLY for rows whose cell is b1h2 (against the guarded hp_v3
    reference); other rows carry nan e_F and rely on reference-free gates. All
    rows store full pmf_t/marginal_t for later rescoring.
    """
    cfgs, methods = list(configs), list(methods)
    assert len(cfgs) == len(seeds)
    B, M = len(cfgs), len(methods)
    R = B * M

    c0 = cfgs[0]
    for c in cfgs:
        for a in ("K", "dt", "n_steps", "block", "eps_bw", "eta_bw", "n_saves",
                  "ess_window_steps", "n_dim", "a", "sigma", "epsilon", "w",
                  "min_r", "force_clip", "wall_strength"):
            assert getattr(c, a) == getattr(c0, a), f"non-uniform {a} across configs"
    K, dt, n_steps, block = c0.K, c0.dt, c0.n_steps, c0.block
    assert n_steps % block == 0
    n_blocks = n_steps // block
    N, L = c0.n_particles, c0.box_length

    from .gateway import _fires_at_block, _schedule_source
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
    theta0 = torch.tensor([m.theta if (m.use_fr and not m.sham) else 0.0
                           for m in methods], device=device, dtype=RC_DTYPE).repeat(B)
    alpha_ess = torch.tensor([m.alpha_ess for m in methods], device=device,
                             dtype=RC_DTYPE).repeat(B)

    eval_mask = GRID.eval_mask(device, RC_DTYPE)
    k_eta, r_eta = gaussian_kernel(c0.eta_bw, GRID.dx, device, RC_DTYPE)

    beta_row = torch.tensor([c.beta for c in cfgs], device=device,
                            dtype=RC_DTYPE).repeat_interleave(M)
    h_row = torch.tensor([c.h for c in cfgs], device=device,
                         dtype=DYN_DTYPE).repeat_interleave(M)
    is_b1h2 = torch.tensor([abs(c.beta - 1.0) < 1e-12 and abs(c.h - 2.0) < 1e-12
                            for c in cfgs], device=device).repeat_interleave(M)
    F_ref = None
    rho_ref = None
    if score_b1h2 and bool(is_b1h2.any()):
        F_ref_1, _meta = load_reference(
            next(c for c in cfgs if abs(c.beta - 1.0) < 1e-12 and abs(c.h - 2.0) < 1e-12),
            device=device)
        F_ref = F_ref_1                                      # (G,), b1h2 only
        rho = torch.exp(-1.0 * F_ref)                        # beta = 1 on b1h2
        rho_ref = rho / trapz(rho.unsqueeze(0), GRID.dx)

    engine = WCAEngine(c0, device)
    noise_amp_box = torch.sqrt(2.0 * dt / beta_row.to(DYN_DTYPE)).repeat_interleave(
        K).view(R * K, 1, 1)
    h_box = h_row.repeat_interleave(K).view(R * K, 1)

    q0 = torch.cat([lattice_init(c, K, sd, device) for c, sd in zip(cfgs, seeds)])
    # rows of one B-row share initial conditions across methods (paired arms)
    q = q0.view(B, K, N, 2).repeat_interleave(M, dim=0).reshape(R * K, N, 2).clone()
    anc = torch.arange(K, device=device).unsqueeze(0).expand(R, K).clone()
    anc_g = anc.clone()
    shus = ShusAccumulator(R, GRID, beta_row.reshape(R, 1), c0.eps_bw, device, RC_DTYPE)
    dep_ref_cur = torch.full((R,), float("nan"), device=device, dtype=RC_DTYPE)
    dep_self_cur = torch.full((R,), float("nan"), device=device, dtype=RC_DTYPE)

    gen_n = torch.Generator(device=device)
    gen_n.manual_seed(2000 + batch_seed)
    gen_f = torch.Generator(device=device)
    gen_f.manual_seed(3000 + batch_seed)

    save_steps = sorted({*range(0, n_steps, max(1, n_steps // c0.n_saves)),
                         n_steps - 1})
    n_saves = len(save_steps)
    save_set = set(save_steps)
    ts = {k: torch.zeros((R, n_saves), device=device, dtype=RC_DTYPE) for k in
          ("l2_f", "kl_u", "tv_u", "ess_anc", "wmax", "ess_anc_glob", "wmax_glob",
           "n_anc", "dep_ref", "dep_self")}
    ts["P"] = torch.zeros((R, n_saves, 3), device=device, dtype=RC_DTYPE)
    ts["pmf"] = torch.zeros((R, n_saves, GRID.n), device=device, dtype=RC_DTYPE)
    ts["marg"] = torch.zeros((R, n_saves, GRID.n), device=device, dtype=RC_DTYPE)
    ev = {k: torch.zeros((R, max(n_events, 1)), device=device, dtype=RC_DTYPE)
          for k in ("theta", "ess_fr", "turnover")}
    tot_turn = torch.zeros(R, device=device, dtype=RC_DTYPE)
    save_ptr, event_ptr = 0, 0

    for step in range(n_steps):
        if c0.ess_window_steps > 0 and step % c0.ess_window_steps == 0:
            anc = torch.arange(K, device=device).unsqueeze(0).expand(R, K).clone()

        # ---- physical propagation ------------------------------------------------
        forces = engine.force(q, h_box)
        z_box = reaction_coordinate(q, c0)                       # (R*K,) float32
        # SHUS bias force + soft walls, both along grad(xi)
        bias = interp1d(z_box.view(R, K).to(RC_DTYPE), shus.Fp, GRID)
        wall = -c0.wall_strength * (torch.clamp(z_box - GRID.xmax, min=0.0)
                                    + torch.clamp(z_box - GRID.xmin, max=0.0))
        scalar = bias.to(DYN_DTYPE).view(R * K) + wall
        forces = add_rc_force(q, forces, scalar, c0)
        noise = torch.randn((B, K, N, 2), device=device, dtype=DYN_DTYPE,
                            generator=gen_n).repeat_interleave(M, dim=0)
        q = wrap(q + forces * dt + noise_amp_box * noise.reshape(R * K, N, 2), L)

        # ---- SHUS deposit (post-step positions, float64 RC) -----------------------
        z = reaction_coordinate(q, c0).view(R, K).to(RC_DTYPE)
        shus.deposit(z)

        # ---- block boundary: update, then (maybe) an FR event ---------------------
        if (step + 1) % block == 0:
            r_n = shus.R / trapz(shus.R, GRID.dx).unsqueeze(1)
            inc = shus.update(dt, K)
            d_n = inc / torch.clamp(trapz(inc, GRID.dx), min=EPS).unsqueeze(1)
            dd = (d_n - r_n)[:, eval_mask]
            dep_self_cur = torch.sqrt((dd * dd).mean(dim=1))
            if rho_ref is not None:
                dd = (d_n - rho_ref)[:, eval_mask]
                dep_ref_cur = torch.where(
                    is_b1h2, torch.sqrt((dd * dd).mean(dim=1)), dep_ref_cur)
            blk = (step + 1) // block
            if event_ptr < n_events and event_blocks[event_ptr] == blk:
                active = fires[event_ptr]
                sel, turn, theta_used, essf = fr_event(
                    z, active & is_fr_row, active & is_sham_row, is_coarse_row,
                    coarse_nb, partner, theta0, alpha_ess, k_eta, r_eta, GRID, gen_f)
                ev["theta"][:, event_ptr] = theta_used
                ev["ess_fr"][:, event_ptr] = essf
                ev["turnover"][:, event_ptr] = turn.to(RC_DTYPE)
                tot_turn += turn.to(RC_DTYPE)
                # gather ENTIRE boxes; the SHUS accumulator is untouched
                q = q.view(R, K, N, 2)[
                    torch.arange(R, device=device).unsqueeze(1), sel].reshape(
                    R * K, N, 2)
                anc = torch.gather(anc, 1, sel)
                anc_g = torch.gather(anc_g, 1, sel)
                event_ptr += 1

        # ---- checkpoints -----------------------------------------------------------
        if step in save_set:
            F_hat = shus.f_estimate(eval_mask)
            if F_ref is not None:
                d = (F_hat - F_ref.unsqueeze(0))[:, eval_mask]
                d = d - d.mean(dim=1, keepdim=True)
                l2 = torch.sqrt((d * d).mean(dim=1))
                ts["l2_f"][:, save_ptr] = torch.where(
                    is_b1h2, l2, torch.full_like(l2, float("nan")))
            else:
                ts["l2_f"][:, save_ptr] = float("nan")
            z_now = reaction_coordinate(q, c0).view(R, K).to(RC_DTYPE)
            p_hat = binned_density(z_now, k_eta, r_eta, GRID)
            ts["kl_u"][:, save_ptr] = kl_to_uniform(p_hat, GRID)
            ts["tv_u"][:, save_ptr] = tv_to_uniform(p_hat, GRID)
            e_, w_ = ancestor_stats(anc, K)
            ts["ess_anc"][:, save_ptr] = e_
            ts["wmax"][:, save_ptr] = w_
            eg_, wg_ = ancestor_stats(anc_g, K)
            ts["ess_anc_glob"][:, save_ptr] = eg_
            ts["wmax_glob"][:, save_ptr] = wg_
            ts["n_anc"][:, save_ptr] = surviving_ancestors(anc_g, K)
            ts["dep_ref"][:, save_ptr] = dep_ref_cur
            ts["dep_self"][:, save_ptr] = dep_self_cur
            ts["P"][:, save_ptr] = region_fractions(z_now)
            ts["pmf"][:, save_ptr] = F_hat
            ts["marg"][:, save_ptr] = p_hat
            save_ptr += 1
        if progress is not None and step % progress == 0:
            print(f"    step {step}/{n_steps}", flush=True)

    # ---- finalize ---------------------------------------------------------------
    t_axis = np.array([s * dt for s in save_steps])
    ev_t = np.array([k * block * dt for k in event_blocks])

    def npy(t):
        return t.detach().cpu().numpy()

    F_ref_np = npy(F_ref) if F_ref is not None else np.full(GRID.n, np.nan)
    recs = []
    for b in range(B):
        for m in range(M):
            r = b * M + m
            l2 = npy(ts["l2_f"][r])
            recs.append(dict(
                config=asdict(cfgs[b]), seed=int(seeds[b]),
                method=asdict(methods[m]), batch_seed=batch_seed,
                cell=cfgs[b].cell_name(),
                reference_id=REFERENCE_ID if bool(is_b1h2[r]) else "none",
                eval_window=(GRID.eval_lo, GRID.eval_hi),
                time=t_axis, x_grid=npy(GRID.x(device, RC_DTYPE)),
                F_ref=F_ref_np if bool(is_b1h2[r]) else np.full(GRID.n, np.nan),
                pmf_t=npy(ts["pmf"][r]), marginal_t=npy(ts["marg"][r]),
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
                int_l2_f=float(np.trapezoid(l2, t_axis)) if np.isfinite(l2).all()
                else float("nan"),
                total_turnover=float(tot_turn[r]),
            ))
    return recs
