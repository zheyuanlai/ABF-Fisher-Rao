"""Entropic gateway: mollified SHUS (+ optional temporary Fisher-Rao) on the 2D channel.

System (ported physics, see docs/PROVENANCE.md):

    V(x, y)  = H (x^2 - 1)^2 + 1/2 omega(x)^2 y^2,
    omega(x) = omega_out + (omega_in - omega_out) exp(-x^2 / (2 s^2)),
    xi(x, y) = x,       F_ref(x) = H (x^2 - 1)^2 + beta^{-1} log omega(x) + C.

Overdamped Euler-Maruyama, reflecting walls in x on [-1.8, 1.8], y unbounded.  s sets
the width of the constriction and r = omega_in/omega_out its severity; the analytic
F_ref means there is no reference-simulation error to confound convergence claims.

One batch carries B (config, seed) rows x M methods flattened to R = B*M rows; methods
inside a B-row share initial conditions and Langevin noise, so every arm is compared
against its baseline on the same realisation (paired comparison).  All state lives on
one device (production: a single H200); nothing syncs to host inside the step loop
except the rare theta-backoff reduction at FR events.

Cycle ordering (frozen): propagate block -> SHUS update -> FR resampling -> propagate.
An FR event only gathers (X, Y, anc); it can not touch the SHUS accumulator.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch

from ..fisher_rao import kl_to_uniform, theta_backoff, tv_to_uniform, uniform_log_ratio
from ..grid import (DEVICE, DTYPE, EPS, Grid1D, binned_density, central_diff,
                    gaussian_kernel, interp1d, reflect_into)
from ..resampling import (ancestor_stats, matched_turnover_indices,
                          systematic_resample, turnover_counts)
from ..shus import ShusAccumulator

REFERENCE_ID = "gateway-analytic-v1"
GRID = Grid1D(xmin=-1.8, xmax=1.8, n=181, eval_lo=-1.5, eval_hi=1.5)

# Region geometry: |x| <= X_BASIN is the gateway corridor, outside are the basins.
X_BASIN = 0.5
REGIONS = ("minus", "gate", "plus")


# -----------------------------------------------------------------------------
# physics
# -----------------------------------------------------------------------------
def omega_of(x, omega_out, omega_in, s):
    return omega_out + (omega_in - omega_out) * torch.exp(-x * x / (2.0 * s * s))


def domega_of(x, omega_out, omega_in, s):
    return -(omega_in - omega_out) * (x / (s * s)) * torch.exp(-x * x / (2.0 * s * s))


def U_of(x, Hc):
    return Hc * (x * x - 1.0) ** 2


def dU_of(x, Hc):
    return 4.0 * Hc * x * (x * x - 1.0)


def reference_profiles(x_grid, eval_mask, beta, Hc, omega_out, omega_in, s):
    """F_ref (centered on the eval window) and F'_ref.  Params: (B,1) -> (B,G)."""
    xg = x_grid.unsqueeze(0)
    om = omega_of(xg, omega_out, omega_in, s)
    dom = domega_of(xg, omega_out, omega_in, s)
    F_ref = U_of(xg, Hc) + torch.log(om) / beta
    F_ref = F_ref - F_ref[:, eval_mask].mean(dim=1, keepdim=True)
    Fp_ref = dU_of(xg, Hc) + dom / (om * beta)
    return F_ref, Fp_ref


# -----------------------------------------------------------------------------
# configuration
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class GatewayConfig:
    """One (config, seed) row.  omega_in is derived: r is the only severity knob."""
    beta: float = 1.0
    H: float = 1.0
    omega_out: float = 1.0
    r: float = 8.0
    s: float = 0.15
    K: int = 1024                # walkers
    dt: float = 4e-4
    n_steps: int = 250_000
    block: int = 20              # SHUS adaptation block, in MD steps
    eps_bw: float = 0.07         # mollifier bandwidth (deposits)
    eta_bw: float = 0.10         # KDE bandwidth (marginal / FR score)
    n_saves: int = 400
    ess_window_steps: int = 4000
    init: str = "left"           # 'left' | 'one_right' (mechanism control)

    @property
    def omega_in(self) -> float:
        return self.omega_out * self.r

    @property
    def T_total(self) -> float:
        return self.n_steps * self.dt

    def barrier_kT(self) -> float:
        return self.beta * self.H + math.log(self.r)


@dataclass(frozen=True)
class Method:
    """One arm.  A sham copies its partner's event schedule and realized turnover."""
    name: str
    use_fr: bool = False
    sham: bool = False
    shadows: str = ""            # sham arms: name of the FR arm whose events are copied
    theta: float = 0.0           # finite FR step size per event
    t_on_frac: float = 0.0       # FR window (fractions of the run); persistent: off=1
    t_off_frac: float = 0.0
    fr_every_blocks: int = 5     # FR event stride, in adaptation blocks
    alpha_ess: float = 0.5       # ESS_FR floor as a fraction of K


SHUS = Method("shus")


def _schedule_source(m: Method, by_name):
    """The method whose (window, stride) a row fires on: itself, or its sham partner."""
    if m.sham:
        assert m.shadows in by_name, (
            f"sham arm {m.name!r} shadows {m.shadows!r}, which is not in this batch; "
            f"matched intensity is unobtainable without it")
        return by_name[m.shadows]
    return m


def _fires_at_block(sched: Method, block_idx: int, n_blocks: int) -> bool:
    if sched.theta <= 0.0 or sched.t_off_frac <= sched.t_on_frac:
        return False
    frac = block_idx / n_blocks
    return (sched.t_on_frac <= frac < sched.t_off_frac
            and block_idx % sched.fr_every_blocks == 0)


# -----------------------------------------------------------------------------
# regions and the bias-aware occupancy target (diagnostics only -- never fed back)
# -----------------------------------------------------------------------------
def region_of(x):
    lab = torch.ones_like(x, dtype=torch.long)
    lab = torch.where(x < -X_BASIN, torch.zeros_like(lab), lab)
    lab = torch.where(x > X_BASIN, torch.full_like(lab, 2), lab)
    return lab


def bias_aware_target(F_ref, F_bias, glab, beta):
    """Q*_k(t): ideal biased population per region under the CURRENT bias.

    q* ~ exp(-beta (F_ref - F_bias)) normalized by plain Riemann sum (sums to 1
    exactly), then summed per region.  Purely diagnostic: consulting F_ref inside the
    sampler would be oracle leakage, so this is only ever computed for saved series.
    """
    e = -beta * (F_ref - F_bias)
    e = e - e.max(dim=1, keepdim=True).values
    q = torch.exp(e)
    q = q / q.sum(dim=1, keepdim=True)
    return torch.stack([q[:, glab == k].sum(dim=1) for k in range(3)], dim=1)


# -----------------------------------------------------------------------------
# initial conditions
# -----------------------------------------------------------------------------
def init_conditions(seeds, K, beta_b, oout_b, oin_b, s_b, inits, device, dtype):
    """x ~ N(-1, 0.05) (left basin); y drawn from its exact conditional so the
    transverse channel starts equilibrated.  'one_right' mirrors walker 0."""
    B = len(seeds)
    X0 = torch.empty((B, K), device=device, dtype=dtype)
    Z0 = torch.empty((B, K), device=device, dtype=dtype)
    for b, sd in enumerate(seeds):
        rng = np.random.default_rng(1000 + int(sd))
        x = rng.normal(-1.0, 0.05, K)
        if inits[b] == "one_right":
            x[0] = -x[0]
        elif inits[b] != "left":
            raise ValueError(f"unknown init {inits[b]!r}")
        X0[b] = reflect_into(torch.as_tensor(x, device=device, dtype=dtype),
                             GRID.xmin, GRID.xmax)
        Z0[b] = torch.as_tensor(rng.normal(0.0, 1.0, K), device=device, dtype=dtype)
    om0 = omega_of(X0, oout_b.unsqueeze(1), oin_b.unsqueeze(1), s_b.unsqueeze(1))
    Y0 = Z0 * torch.sqrt(1.0 / (beta_b.unsqueeze(1) * om0 ** 2))
    return X0, Y0


# -----------------------------------------------------------------------------
# the batched simulation
# -----------------------------------------------------------------------------
def simulate_batch(configs, seeds, methods, batch_seed=12345, device=DEVICE,
                   dtype=DTYPE, progress=None, stop_at=None, start_state=None):
    """Run B (config, seed) rows x M methods.  Returns per-row records, or, when
    stop_at is given, a resumable state dict (checkpoint/resume equivalence is a
    Stage-0 test, so the state carries everything including generator states).

    stop_at must be a multiple of the adaptation block: between blocks the deposit
    buffer is empty and the SHUS state is exactly two tensors.
    """
    cfgs, methods = list(configs), list(methods)
    assert len(cfgs) == len(seeds), "configs and seeds must align"
    B, M = len(cfgs), len(methods)
    R = B * M

    c0 = cfgs[0]
    for c in cfgs:
        for a in ("K", "dt", "n_steps", "block", "eps_bw", "eta_bw", "n_saves",
                  "ess_window_steps"):
            assert getattr(c, a) == getattr(c0, a), f"non-uniform {a} across configs"
        assert c.omega_in ** 2 * c.dt < 2.0, (
            f"transverse OU channel unstable: omega_in^2*dt = {c.omega_in**2*c.dt:.3f}")
    K, dt, n_steps, block = c0.K, c0.dt, c0.n_steps, c0.block
    assert n_steps % block == 0, "n_steps must be a whole number of adaptation blocks"
    n_blocks = n_steps // block

    by_name = {m.name: m for m in methods}
    scheds = [_schedule_source(m, by_name) for m in methods]
    name_col = {m.name: j for j, m in enumerate(methods)}
    partner_col = [name_col[m.shadows] if m.sham else j for j, m in enumerate(methods)]
    partner = torch.tensor([b * M + partner_col[j] for b in range(B) for j in range(M)],
                           device=device, dtype=torch.long)

    # global FR event slots: block indices at which ANY row fires
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
    theta0 = torch.tensor([m.theta if (m.use_fr and not m.sham) else 0.0
                           for m in methods], device=device, dtype=dtype).repeat(B)
    alpha_ess = torch.tensor([m.alpha_ess for m in methods], device=device,
                             dtype=dtype).repeat(B)

    x_grid = GRID.x(device, dtype)
    eval_mask = GRID.eval_mask(device, dtype)
    glab = region_of(x_grid)
    k_eta, r_eta = gaussian_kernel(c0.eta_bw, GRID.dx, device, dtype)

    def cfg_b(fn):
        return torch.tensor([fn(c) for c in cfgs], device=device, dtype=dtype)
    beta_b = cfg_b(lambda c: c.beta)
    H_b = cfg_b(lambda c: c.H)
    oout_b = cfg_b(lambda c: c.omega_out)
    oin_b = cfg_b(lambda c: c.omega_in)
    s_b = cfg_b(lambda c: c.s)

    def to_run(t_b):
        return t_b.repeat_interleave(M).unsqueeze(1)
    beta = to_run(beta_b)
    Hc = to_run(H_b)
    oout = to_run(oout_b)
    oin = to_run(oin_b)
    sw = to_run(s_b)
    noise_amp = torch.sqrt(2.0 * dt / beta)

    F_ref_b, Fp_ref_b = reference_profiles(x_grid, eval_mask, beta_b.unsqueeze(1),
                                           H_b.unsqueeze(1), oout_b.unsqueeze(1),
                                           oin_b.unsqueeze(1), s_b.unsqueeze(1))
    F_ref = F_ref_b.repeat_interleave(M, dim=0)
    Fp_ref = Fp_ref_b.repeat_interleave(M, dim=0)

    save_steps = sorted({*range(0, n_steps, max(1, n_steps // c0.n_saves)),
                         n_steps - 1})
    n_saves = len(save_steps)
    save_set = set(save_steps)

    if start_state is None:
        X0_b, Y0_b = init_conditions(seeds, K, beta_b, oout_b, oin_b, s_b,
                                     [c.init for c in cfgs], device, dtype)
        X = X0_b.repeat_interleave(M, dim=0).clone()
        Y = Y0_b.repeat_interleave(M, dim=0).clone()
        anc = torch.arange(K, device=device).unsqueeze(0).expand(R, K).clone()
        shus = ShusAccumulator(R, GRID, beta, c0.eps_bw, device, dtype)
        gen_n = torch.Generator(device=device)
        gen_n.manual_seed(2000 + batch_seed)
        gen_f = torch.Generator(device=device)
        gen_f.manual_seed(3000 + batch_seed)
        step0, save_ptr, event_ptr = 0, 0, 0
        ts = {k: torch.zeros((R, n_saves), device=device, dtype=dtype) for k in
              ("l2_f", "l2_fp", "kl_u", "tv_u", "ess_anc", "wmax")}
        ts["P"] = torch.zeros((R, n_saves, 3), device=device, dtype=dtype)
        ts["Q"] = torch.zeros((R, n_saves, 3), device=device, dtype=dtype)
        ts["pmf"] = torch.zeros((R, n_saves, GRID.n), device=device, dtype=dtype)
        ts["marg"] = torch.zeros((R, n_saves, GRID.n), device=device, dtype=dtype)
        ev = {k: torch.zeros((R, max(n_events, 1)), device=device, dtype=dtype)
              for k in ("theta", "ess_fr", "turnover")}
        tot_die = torch.zeros(R, device=device, dtype=dtype)
    else:
        st = start_state
        X, Y, anc = st["X"], st["Y"], st["anc"]
        shus = ShusAccumulator(R, GRID, beta, c0.eps_bw, device, dtype)
        shus.load_state_dict(st["shus"])
        gen_n = torch.Generator(device=device)
        gen_n.set_state(st["gen_n"])
        gen_f = torch.Generator(device=device)
        gen_f.set_state(st["gen_f"])
        step0, save_ptr, event_ptr = st["step"], st["save_ptr"], st["event_ptr"]
        ts, ev, tot_die = st["ts"], st["ev"], st["tot_die"]

    stop_step = n_steps if stop_at is None else int(stop_at)
    assert stop_step % block == 0 or stop_step == n_steps, \
        "stop_at must land on an adaptation-block boundary"

    for step in range(step0, stop_step):
        if c0.ess_window_steps > 0 and step % c0.ess_window_steps == 0:
            anc = torch.arange(K, device=device).unsqueeze(0).expand(R, K).clone()

        # ---- physical propagation (one Euler-Maruyama step) --------------------
        om = omega_of(X, oout, oin, sw)
        dom = domega_of(X, oout, oin, sw)
        gVx = dU_of(X, Hc) + om * dom * Y * Y
        gVy = om * om * Y
        zx = torch.randn((B, K), device=device, dtype=dtype,
                         generator=gen_n).repeat_interleave(M, dim=0)
        zy = torch.randn((B, K), device=device, dtype=dtype,
                         generator=gen_n).repeat_interleave(M, dim=0)
        X = reflect_into(X + (-gVx + shus.bias_force_at(X)) * dt + noise_amp * zx,
                         GRID.xmin, GRID.xmax)
        Y = Y + (-gVy) * dt + noise_amp * zy

        # ---- SHUS deposit: physically propagated samples only ------------------
        shus.deposit(X)

        # ---- block boundary: SHUS update, then (maybe) an FR event -------------
        if (step + 1) % block == 0:
            shus.update(dt, K)
            blk = (step + 1) // block
            if event_ptr < n_events and event_blocks[event_ptr] == blk:
                active = fires[event_ptr]
                fr_act = active & is_fr_row
                sham_act = active & is_sham_row
                sel = torch.arange(K, device=device).unsqueeze(0).expand(R, K).clone()
                turn = torch.zeros(R, device=device, dtype=torch.long)
                if bool(fr_act.any()):
                    p_hat = binned_density(X, k_eta, r_eta, GRID)
                    logr = uniform_log_ratio(X, p_hat, GRID)
                    w, theta_used, essf = theta_backoff(logr, theta0, alpha_ess)
                    sel_fr = systematic_resample(w, gen_f)
                    turn_fr = turnover_counts(sel_fr, K)
                    sel = torch.where(fr_act.unsqueeze(1), sel_fr, sel)
                    turn = torch.where(fr_act, turn_fr, turn)
                    ev["theta"][:, event_ptr] = torch.where(fr_act, theta_used,
                                                            torch.zeros_like(theta_used))
                    ev["ess_fr"][:, event_ptr] = torch.where(
                        fr_act, essf, torch.full_like(essf, float("nan")))
                if bool(sham_act.any()):
                    m_sham = turn[partner]
                    sel_sham = matched_turnover_indices(m_sham, K, gen_f, device, dtype)
                    sel = torch.where(sham_act.unsqueeze(1), sel_sham, sel)
                    turn = torch.where(sham_act, m_sham, turn)
                ev["turnover"][:, event_ptr] = turn.to(dtype)
                tot_die += turn.to(dtype)
                # ESTIMATOR PROTECTION: only walker arrays are gathered.  The SHUS
                # accumulator and its buffer are not touched at an FR event.
                X = torch.gather(X, 1, sel)
                Y = torch.gather(Y, 1, sel)
                anc = torch.gather(anc, 1, sel)
                event_ptr += 1

        # ---- checkpoints ---------------------------------------------------------
        if step in save_set:
            F_hat = shus.f_estimate(eval_mask)
            d = (F_hat - F_ref)[:, eval_mask]
            d = d - d.mean(dim=1, keepdim=True)
            ts["l2_f"][:, save_ptr] = torch.sqrt((d * d).mean(dim=1))
            dp = (shus.Fp - Fp_ref)[:, eval_mask]
            ts["l2_fp"][:, save_ptr] = torch.sqrt((dp * dp).mean(dim=1))
            p_hat = binned_density(X, k_eta, r_eta, GRID)
            ts["kl_u"][:, save_ptr] = kl_to_uniform(p_hat, GRID)
            ts["tv_u"][:, save_ptr] = tv_to_uniform(p_hat, GRID)
            e_, w_ = ancestor_stats(anc, K)
            ts["ess_anc"][:, save_ptr] = e_
            ts["wmax"][:, save_ptr] = w_
            plab = region_of(X)
            for k in range(3):
                ts["P"][:, save_ptr, k] = (plab == k).to(dtype).mean(dim=1)
            ts["Q"][:, save_ptr] = bias_aware_target(F_ref, F_hat, glab, beta)
            ts["pmf"][:, save_ptr] = F_hat
            ts["marg"][:, save_ptr] = p_hat
            save_ptr += 1
        if progress is not None and step % progress == 0:
            print(f"    step {step}/{n_steps}", flush=True)

    if stop_at is not None and stop_step < n_steps:
        return {"X": X, "Y": Y, "anc": anc, "shus": shus.state_dict(),
                "gen_n": gen_n.get_state(), "gen_f": gen_f.get_state(),
                "step": stop_step, "save_ptr": save_ptr, "event_ptr": event_ptr,
                "ts": ts, "ev": ev, "tot_die": tot_die}

    # region-target sanity (diagnostic mass must be exactly accounted for)
    totQ = ts["Q"].sum(dim=2)
    worstQ = float((totQ - 1.0).abs().max())
    assert worstQ < 1e-9, f"bias-aware target does not sum to 1 (worst {worstQ:.3e})"
    totP = ts["P"].sum(dim=2)
    worstP = float((totP - 1.0).abs().max())
    assert worstP < 1e-9, f"region fractions do not sum to 1 (worst {worstP:.3e})"

    return _finalize(cfgs, seeds, methods, save_steps, dt, x_grid, eval_mask,
                     F_ref, Fp_ref, ts, ev, event_blocks, block, tot_die, batch_seed)


def _finalize(cfgs, seeds, methods, save_steps, dt, x_grid, eval_mask, F_ref, Fp_ref,
              ts, ev, event_blocks, blk_steps, tot_die, batch_seed):
    B, M = len(cfgs), len(methods)
    t_axis = np.array([s * dt for s in save_steps])
    ev_t = np.array([k * blk_steps * dt for k in event_blocks])

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
                eval_window=(GRID.eval_lo, GRID.eval_hi),
                time=t_axis, x_grid=npy(x_grid),
                F_ref=npy(F_ref[r]), Fp_ref=npy(Fp_ref[r]),
                pmf_t=npy(ts["pmf"][r]), marginal_t=npy(ts["marg"][r]),
                l2_f_t=l2, l2_fp_t=npy(ts["l2_fp"][r]),
                kl_u_t=npy(ts["kl_u"][r]), tv_u_t=npy(ts["tv_u"][r]),
                ess_anc_t=npy(ts["ess_anc"][r]), wmax_t=npy(ts["wmax"][r]),
                P_regions=npy(ts["P"][r]), Q_regions=npy(ts["Q"][r]),
                event_time=ev_t, event_theta=npy(ev["theta"][r]),
                event_ess_fr=npy(ev["ess_fr"][r]),
                event_turnover=npy(ev["turnover"][r]),
                final_l2_f=float(l2[-1]),
                int_l2_f=float(np.trapezoid(l2, t_axis)),
                total_turnover=float(tot_die[r]),
            ))
    return recs


# -----------------------------------------------------------------------------
# frozen-bias validation: an endpoint that does not reuse the online estimator
# -----------------------------------------------------------------------------
def run_frozen_bias(F_frozen, cfgs_per_row, group=None, n_steps=40_000, burn_frac=0.5,
                    seed=987_654, device=DEVICE, dtype=DTYPE):
    """Score a learned bias potential by sampling under it, frozen, with no FR.

    A fresh independent population runs under V - F_frozen(xi); at equilibrium
    p_B ~ exp(-beta (F - F_frozen)), so F_hat = F_frozen + beta^{-1}*(-log p_B) + C.
    Rows sharing a `group` id share initial conditions and Langevin noise, keeping the
    endpoint paired across arms of one seed.
    """
    F_frozen = torch.as_tensor(F_frozen, device=device, dtype=dtype)
    R = F_frozen.shape[0]
    assert len(cfgs_per_row) == R, "one config per frozen-bias row"
    x_grid = GRID.x(device, dtype)
    eval_mask = GRID.eval_mask(device, dtype)
    c0 = cfgs_per_row[0]
    k_eta, r_eta = gaussian_kernel(c0.eta_bw, GRID.dx, device, dtype)

    def col(fn):
        return torch.tensor([fn(c) for c in cfgs_per_row], device=device,
                            dtype=dtype).unsqueeze(1)
    beta = col(lambda c: c.beta)
    Hc = col(lambda c: c.H)
    oout = col(lambda c: c.omega_out)
    oin = col(lambda c: c.omega_in)
    sw = col(lambda c: c.s)
    K, dt = c0.K, c0.dt
    noise_amp = torch.sqrt(2.0 * dt / beta)
    Fp_frozen = central_diff(F_frozen, GRID.dx)
    F_ref, _ = reference_profiles(x_grid, eval_mask, beta, Hc, oout, oin, sw)

    if group is None:
        gidx = torch.arange(R, device=device)
    else:
        g = list(group)
        uniq = {v: i for i, v in enumerate(dict.fromkeys(g))}
        gidx = torch.tensor([uniq[v] for v in g], device=device, dtype=torch.long)
    G = int(gidx.max().item()) + 1
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    X = (GRID.xmin + (GRID.xmax - GRID.xmin) * torch.rand(
        (G, K), device=device, dtype=dtype, generator=gen))[gidx]
    om0 = omega_of(X, oout, oin, sw)
    Y = torch.randn((G, K), device=device, dtype=dtype, generator=gen)[gidx] * \
        torch.sqrt(1.0 / (beta * om0 ** 2))

    burn = int(burn_frac * n_steps)
    acc = torch.zeros((R, GRID.n), device=device, dtype=dtype)
    n_acc = 0
    for step in range(n_steps):
        om = omega_of(X, oout, oin, sw)
        dom = domega_of(X, oout, oin, sw)
        gVx = dU_of(X, Hc) + om * dom * Y * Y
        gVy = om * om * Y
        zx = torch.randn((G, K), device=device, dtype=dtype, generator=gen)[gidx]
        zy = torch.randn((G, K), device=device, dtype=dtype, generator=gen)[gidx]
        X = reflect_into(X + (-gVx + interp1d(X, Fp_frozen, GRID)) * dt
                         + noise_amp * zx, GRID.xmin, GRID.xmax)
        Y = Y + (-gVy) * dt + noise_amp * zy
        if step >= burn:
            acc += binned_density(X, k_eta, r_eta, GRID)
            n_acc += 1
    p_B = torch.clamp(acc / max(n_acc, 1), min=EPS)
    p_B = p_B / p_B.sum(dim=1, keepdim=True)

    F_hat = F_frozen - torch.log(p_B) / beta
    F_hat = F_hat - F_hat[:, eval_mask].mean(dim=1, keepdim=True)
    d = (F_hat - F_ref)[:, eval_mask]
    d = d - d.mean(dim=1, keepdim=True)
    err = torch.sqrt((d * d).mean(dim=1))
    return dict(l2_f=err.detach().cpu().numpy(),
                F_hat=F_hat.detach().cpu().numpy(),
                F_ref=F_ref.detach().cpu().numpy(),
                p_B=p_B.detach().cpu().numpy(),
                x_grid=x_grid.detach().cpu().numpy(),
                n_steps=n_steps, burn_frac=burn_frac, seed=seed)
