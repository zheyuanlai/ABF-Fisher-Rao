"""Application-map Q4a: first clone-decorrelation measurement (WCA solvent).

Design frozen in docs/PREREGISTRATION_APPLICATION_MAP.md (commit fde0c9c). WCA
stays FR-free (the Q2 closure stands): this is measurement, not intervention.

Protocol per (cell, seed) row, all rows in one batch:
  1. plain SHUS, K = 256, to t0 = 100 (Stage-4 numerics);
  2. freeze the learned bias;
  3. select 64 parents stratified over xi in [0,1] (8 bins x 8 closest);
  4. duplicate each parent into two children; evolve all 128 children under the
     frozen bias with independent noise to lag 100;
  5. record xi and the orthogonal solvent coordination
       n_coord = sum_j 0.5 (1 - tanh((min_bead |q_j - q_bead|_MI - 1.6)/0.1));
  6. decorrelation m(tau) = 1 - d_sib(tau)/d_ind(tau) (RMS pair differences;
     independent baseline: child-A pairs of DIFFERENT same-bin parents);
     tau_clone = first recorded lag with m <= 1/e (5-record persistence against
     single-record noise crossings).

Usage: python scripts/run_appmap_q4a_clone_decorr.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch

from abpfr.grid import interp1d
from abpfr.shus import ShusAccumulator
from abpfr.systems import wca

CELLS = {"b1h2": dict(beta=1.0, h=2.0), "b2h6": dict(beta=2.0, h=6.0)}
SEEDS = [0, 1, 2, 3]
K = 256
T0_STEPS = 50_000            # t0 = 100 at dt = 2e-3
LAG_STEPS = 50_000           # lag horizon 100
REC_STRIDE = 100             # record every 0.2 t -> 501 lags
N_BINS, PER_BIN = 8, 8       # 64 parents stratified over xi in [0, 1]
RC, WS = 1.6, 0.1            # coordination switching function (frozen)
BATCH_SEED = 20260830
OUT = "results/appmap_q4a_clone_decorr"

COMMON = dict(K=K, dt=2e-3, n_steps=T0_STEPS, block=20)   # Stage-4 numerics


def n_coord(q, cfg):
    """Smooth first-shell solvent count around the dimer.  q: (B, N, 2) -> (B,)."""
    L = cfg.box_length
    d0 = torch.linalg.norm(wca.minimum_image(q[:, 2:] - q[:, 0:1], L), dim=-1)
    d1 = torch.linalg.norm(wca.minimum_image(q[:, 2:] - q[:, 1:2], L), dim=-1)
    r = torch.minimum(d0, d1)
    return (0.5 * (1.0 - torch.tanh((r - RC) / WS))).sum(dim=1)


def main():
    device = wca.DEVICE
    torch.manual_seed(0)
    rows = [(cn, sd) for cn in CELLS for sd in SEEDS]
    R = len(rows)
    cfgs = [wca.WCAConfig(**CELLS[cn], **COMMON) for cn, _ in rows]
    c0 = cfgs[0]
    N, L, dt, block = c0.n_particles, c0.box_length, c0.dt, c0.block
    engine = wca.WCAEngine(c0, device)
    beta_row = torch.tensor([c.beta for c in cfgs], device=device,
                            dtype=wca.RC_DTYPE)
    h_box = torch.tensor([c.h for c in cfgs], device=device,
                         dtype=wca.DYN_DTYPE).repeat_interleave(K).view(R * K, 1)
    amp = torch.sqrt(2.0 * dt / beta_row.to(wca.DYN_DTYPE)).repeat_interleave(
        K).view(R * K, 1, 1)
    gen = torch.Generator(device=device)
    gen.manual_seed(BATCH_SEED)

    q = torch.cat([wca.lattice_init(c, K, sd, device)
                   for c, (_, sd) in zip(cfgs, rows)])
    shus = ShusAccumulator(R, wca.GRID, beta_row.reshape(R, 1), c0.eps_bw, device,
                           wca.RC_DTYPE)

    print(f"Q4a phase 1: {R} rows x {K} boxes, plain SHUS to t0="
          f"{T0_STEPS*dt:.0f}")
    t0 = time.time()
    for step in range(T0_STEPS):
        forces = engine.force(q, h_box)
        z_box = wca.reaction_coordinate(q, c0)
        bias = interp1d(z_box.view(R, K).to(wca.RC_DTYPE), shus.Fp, wca.GRID)
        wall = -c0.wall_strength * (torch.clamp(z_box - wca.GRID.xmax, min=0.0)
                                    + torch.clamp(z_box - wca.GRID.xmin, max=0.0))
        scalar = bias.to(wca.DYN_DTYPE).view(R * K) + wall
        forces = wca.add_rc_force(q, forces, scalar, c0)
        noise = torch.randn((R * K, N, 2), device=device, dtype=wca.DYN_DTYPE,
                            generator=gen)
        q = wca.wrap(q + forces * dt + amp * noise, L)
        shus.deposit(wca.reaction_coordinate(q, c0).view(R, K).to(wca.RC_DTYPE))
        if (step + 1) % block == 0:
            shus.update(dt, K)
        if step % 10_000 == 0:
            print(f"    step {step}/{T0_STEPS}", flush=True)
    print(f"phase 1 wall {time.time()-t0:.0f}s")
    Fp_frozen = shus.Fp.clone()                     # (R, G): bias frozen here

    # ---- parent selection: 8 bins x 8 closest walkers, deterministic ----------
    z = wca.reaction_coordinate(q, c0).view(R, K).to(wca.RC_DTYPE)
    centers = torch.linspace(0.0625, 0.9375, N_BINS, device=device,
                             dtype=wca.RC_DTYPE)
    P = N_BINS * PER_BIN
    parent_idx = torch.empty((R, P), device=device, dtype=torch.long)
    taken = torch.zeros((R, K), device=device, dtype=torch.bool)
    for b in range(N_BINS):
        d = (z - centers[b]).abs() + taken.to(wca.RC_DTYPE) * 1e6
        sel = torch.topk(-d, PER_BIN, dim=1).indices
        parent_idx[:, b * PER_BIN:(b + 1) * PER_BIN] = sel
        taken.scatter_(1, sel, True)
    q_par = q.view(R, K, N, 2)[torch.arange(R, device=device).unsqueeze(1),
                               parent_idx]                       # (R, P, N, 2)
    z_par = torch.gather(z, 1, parent_idx)
    spread = float((torch.gather(z, 1, parent_idx)
                    - centers.repeat_interleave(PER_BIN)).abs().max())
    print(f"parents selected: worst |xi - bin center| = {spread:.3f}")

    # ---- phase 2: duplicated children under the frozen bias -------------------
    C = 2 * P
    q_ch = q_par.repeat_interleave(2, dim=1).reshape(R * C, N, 2).clone()
    h_ch = torch.tensor([c.h for c in cfgs], device=device,
                        dtype=wca.DYN_DTYPE).repeat_interleave(C).view(R * C, 1)
    amp_ch = torch.sqrt(2.0 * dt / beta_row.to(wca.DYN_DTYPE)).repeat_interleave(
        C).view(R * C, 1, 1)
    n_lags = LAG_STEPS // REC_STRIDE + 1
    xi_t = torch.zeros((n_lags, R, C), device=device, dtype=wca.RC_DTYPE)
    nc_t = torch.zeros((n_lags, R, C), device=device, dtype=wca.RC_DTYPE)

    def record(ptr):
        xi_t[ptr] = wca.reaction_coordinate(q_ch, c0).view(R, C).to(wca.RC_DTYPE)
        nc_t[ptr] = n_coord(q_ch, c0).view(R, C).to(wca.RC_DTYPE)

    record(0)
    print(f"Q4a phase 2: {R} rows x {C} children, frozen bias, lag "
          f"{LAG_STEPS*dt:.0f}")
    t0 = time.time()
    ptr = 1
    for step in range(LAG_STEPS):
        forces = engine.force(q_ch, h_ch)
        z_box = wca.reaction_coordinate(q_ch, c0)
        bias = interp1d(z_box.view(R, C).to(wca.RC_DTYPE), Fp_frozen, wca.GRID)
        wall = -c0.wall_strength * (torch.clamp(z_box - wca.GRID.xmax, min=0.0)
                                    + torch.clamp(z_box - wca.GRID.xmin, max=0.0))
        scalar = bias.to(wca.DYN_DTYPE).view(R * C) + wall
        forces = wca.add_rc_force(q_ch, forces, scalar, c0)
        noise = torch.randn((R * C, N, 2), device=device, dtype=wca.DYN_DTYPE,
                            generator=gen)
        q_ch = wca.wrap(q_ch + forces * dt + amp_ch * noise, L)
        if (step + 1) % REC_STRIDE == 0:
            record(ptr)
            ptr += 1
        if step % 10_000 == 0:
            print(f"    step {step}/{LAG_STEPS}", flush=True)
    print(f"phase 2 wall {time.time()-t0:.0f}s")

    # ---- analysis --------------------------------------------------------------
    xi_np = xi_t.cpu().numpy()                       # (n_lags, R, C)
    nc_np = nc_t.cpu().numpy()
    tau_axis = np.arange(n_lags) * REC_STRIDE * dt
    # sibling pairs: children (2i, 2i+1); independent pairs: child A of parents
    # (2j, 2j+1) within each bin (same-bin by construction of parent ordering)
    sib_a = np.arange(P) * 2
    sib_b = sib_a + 1
    ind_pairs = []
    for b in range(N_BINS):
        ps = np.arange(b * PER_BIN, (b + 1) * PER_BIN)
        for j in range(0, PER_BIN - 1, 2):
            ind_pairs.append((2 * ps[j], 2 * ps[j + 1]))     # child A vs child A
    ind_a = np.array([p[0] for p in ind_pairs])
    ind_b = np.array([p[1] for p in ind_pairs])

    def m_curve(obs):
        d_sib = np.sqrt(((obs[:, :, sib_a] - obs[:, :, sib_b]) ** 2).mean(axis=2))
        d_ind = np.sqrt(((obs[:, :, ind_a] - obs[:, :, ind_b]) ** 2).mean(axis=2))
        return 1.0 - d_sib / np.clip(d_ind, 1e-12, None)     # (n_lags, R)

    def tau_clone(m):                                        # (n_lags,) -> float
        thr, hold = 1.0 / np.e, 5
        below = m <= thr
        for i in range(len(m) - hold + 1):
            if below[i:i + hold].all():
                return float(tau_axis[i])
        return float("nan")

    m_xi = m_curve(xi_np)
    m_nc = m_curve(nc_np)
    summary = {"protocol": dict(K=K, t0=T0_STEPS * dt, lag=LAG_STEPS * dt,
                                rec_stride_t=REC_STRIDE * dt, n_bins=N_BINS,
                                per_bin=PER_BIN, rc=RC, ws=WS,
                                batch_seed=BATCH_SEED),
               "rows": []}
    print(f"\n{'cell':<6s} {'seed':>4s} {'tau_clone(xi)':>14s} "
          f"{'tau_clone(n_coord)':>19s}")
    for r, (cn, sd) in enumerate(rows):
        tx, tn = tau_clone(m_xi[:, r]), tau_clone(m_nc[:, r])
        print(f"{cn:<6s} {sd:>4d} {tx:>14.1f} {tn:>19.1f}")
        summary["rows"].append(dict(cell=cn, seed=sd, tau_xi=tx, tau_nc=tn))
    for cn in CELLS:
        tx = np.array([r["tau_xi"] for r in summary["rows"] if r["cell"] == cn])
        tn = np.array([r["tau_nc"] for r in summary["rows"] if r["cell"] == cn])
        summary[cn] = dict(tau_xi_median=float(np.nanmedian(tx)),
                           tau_nc_median=float(np.nanmedian(tn)),
                           tau_xi_censored=int(np.isnan(tx).sum()),
                           tau_nc_censored=int(np.isnan(tn).sum()))
        print(f"{cn}: median tau_clone(xi) = {summary[cn]['tau_xi_median']:.1f}, "
              f"median tau_clone(n_coord) = {summary[cn]['tau_nc_median']:.1f} "
              f"(censored {summary[cn]['tau_xi_censored']}/"
              f"{summary[cn]['tau_nc_censored']})")

    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, "curves.npz"), tau=tau_axis,
                        m_xi=m_xi, m_nc=m_nc, xi_t=xi_np.astype(np.float32),
                        nc_t=nc_np.astype(np.float32),
                        rows=np.array([f"{cn}_s{sd}" for cn, sd in rows]))
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"summary -> {OUT}/summary.json")
    make_figure(tau_axis, m_xi, m_nc, rows)


def make_figure(tau, m_xi, m_nc, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    cmap = {"b1h2": "C0", "b2h6": "C3"}
    for r, (cn, sd) in enumerate(rows):
        axes[0].plot(tau, m_xi[:, r], color=cmap[cn], alpha=0.6,
                     label=cn if sd == 0 else None)
        axes[1].plot(tau, m_nc[:, r], color=cmap[cn], alpha=0.6)
    for ax, ttl in zip(axes, (r"$\xi$ (dimer extension)",
                              r"$n_{\rm coord}$ (solvent shell)")):
        ax.axhline(1 / np.e, ls="--", c="gray", lw=0.8)
        ax.set_xlabel(r"lag $\tau$")
        ax.set_title(ttl)
        ax.set_xscale("symlog", linthresh=1.0)
    axes[0].set_ylabel(r"sibling excess correlation $m(\tau)$")
    axes[0].legend()
    fig.suptitle("Q4a: clone decorrelation under frozen bias (WCA, K=256 parents"
                 " stratified over $\\xi$)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "q4a_clone_decorr.png"), dpi=130)
    print(f"figure -> {OUT}/q4a_clone_decorr.png")


if __name__ == "__main__":
    main()
