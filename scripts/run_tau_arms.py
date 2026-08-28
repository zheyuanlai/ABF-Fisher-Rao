"""Phase 5 arm comparison: A0 / A6a / A6b on the validated tau benchmark.

Frozen protocol: ``docs/MECHANISM_CAMPAIGN_PREREGISTRATION.md`` (Phase 5).

Adaptive ABF on ``tau_bench_core`` with the engines' own estimator form
(smooth(S)/(smooth(C)+m), h = 0.07, m = 1), plus the frozen ``io_abf`` allocator
for the bias-held arms.  Arms are columns of one batch and share the Langevin
noise stream per seed (eb-style repeat_interleave), so the comparison is paired.

The verdict is NOT time-to-accuracy.  Per the preregistration the gate and the
prediction both live in the decomposition: first ``eta_bias < 0.1`` on A0 at T
(else the regime is not variance-dominated and no Neyman claim may be made),
then whether ``tr(Q Sigma)`` falls from A6a to A6b as ``r ∝ sqrt(a Gamma_hat)``
predicts.  Profiles are stored at every save, per the instrumentation rule.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import tau_bench_core as tb                                       # noqa: E402
import eb_abffr_core as eb                                        # noqa: E402
from abffr import io_abf                                          # noqa: E402

OUT = os.path.join(ROOT, "results", "qr_mechanism", "phase5_arms")
ARMS = ("A0", "A6a", "A6b")
SEEDS = list(range(16))
T_TOTAL = 160.0
H_ABF, MIN_COUNT = 0.07, 1.0


def main():
    device = torch.device("cuda")
    dtype = torch.float64
    cfg = tb.TauConfig()
    n_steps = int(round(T_TOTAL / cfg.dt))
    save_every = n_steps // 100
    xg = torch.linspace(tb.XMIN, tb.XMAX, tb.N_GRID, device=device, dtype=dtype)
    dx = float(xg[1] - xg[0])
    emask = (xg >= tb.EVAL_LO) & (xg <= tb.EVAL_HI)
    A_ref, Ap_ref = tb.reference(xg, cfg, emask)
    k_h, r_h = eb.gaussian_kernel(H_ABF, dx, device, dtype)

    B, M = len(SEEDS), len(ARMS)
    R = B * M
    io_cfg = io_abf.IOConfig(
        n_cells=16, obs_every=10,
        opportunity_every=max(1, int(round(0.60 * n_steps / 48.0))),
        history_capacity=20_000)
    arms_R = [ARMS[m] for _ in range(B) for m in range(M)]
    # A6a is r ∝ sqrt(a): the allocator's A6b rule with Gamma forced flat.  The
    # cleanest way to freeze that without touching the frozen module is a
    # subclass that overrides gamma_hat for the A6a rows only.
    class Alloc(io_abf.IOAllocator):
        def gamma_hat(self):
            est = super().gamma_hat()
            flat = np.ones_like(est["gamma"])
            for r_i, arm in enumerate(self.arms_true):
                if arm == "A6a":
                    est["gamma"][r_i] = flat[r_i]
            return est
    alloc = Alloc(["A0" if a == "A0" else "A6b" for a in arms_R],
                  xg, emask, np.full(R, cfg.beta), cfg.dt, io_cfg,
                  device=device, dtype=dtype)
    alloc.arms_true = arms_R

    gen_n = torch.Generator(device=device); gen_n.manual_seed(20260829)
    X0 = torch.empty((B, cfg.N), device=device, dtype=dtype)
    Y0 = torch.empty((B, cfg.N), device=device, dtype=dtype)
    for b, sd in enumerate(SEEDS):
        x, y = tb.init_conditions(cfg, 1, seed=sd, device=device, dtype=dtype)
        X0[b], Y0[b] = x[0], y[0]
    X = X0.repeat_interleave(M, dim=0).clone()
    Y = Y0.repeat_interleave(M, dim=0).clone()

    C = torch.zeros((R, tb.N_GRID), device=device, dtype=dtype)
    S = torch.zeros((R, tb.N_GRID), device=device, dtype=dtype)
    firing = set(int(v) for v in io_abf.firing_steps(n_steps, io_cfg))
    noise_x = math.sqrt(2.0 * cfg.mu_x * cfg.dt / cfg.beta)

    saves, out_Fp, out_C, out_t = [], [], [], []
    t0 = time.time()
    Fp_hat = torch.zeros((R, tb.N_GRID), device=device, dtype=dtype)
    for step in range(n_steps):
        fx = tb.local_force(X, Y, cfg)
        ix = torch.clamp(torch.round((X - tb.XMIN) / dx).long(), 0, tb.N_GRID - 1)
        C.scatter_add_(1, ix, torch.ones_like(X))
        S.scatter_add_(1, ix, fx)
        if step % 10 == 0:
            Fp_hat = eb.smooth(S, k_h, r_h, dx) / (
                eb.smooth(C, k_h, r_h, dx) + MIN_COUNT + eb.EPS)
        if step % io_cfg.obs_every == 0:
            alloc.observe(X, fx, torch.gather(Fp_hat, 1, ix))
        if step in firing:
            F_hat = torch.cumulative_trapezoid(Fp_hat, dx=dx, dim=1)
            F_hat = torch.cat([torch.zeros((R, 1), device=device, dtype=dtype),
                               F_hat], dim=1)
            alloc.refresh(step, X, F_hat)
        bias = torch.gather(Fp_hat, 1, ix) + alloc.bias_force_at(X)
        # y-noise for the fibre lives inside step_xy; pair the x-noise per seed
        kap = tb.kappa_of(X, cfg)
        zx = torch.randn((B, cfg.N), generator=gen_n, device=device,
                         dtype=dtype).repeat_interleave(M, dim=0)
        zy = torch.randn((B, cfg.N), generator=gen_n, device=device,
                         dtype=dtype).repeat_interleave(M, dim=0)
        fy = kap * cfg.k * (Y - cfg.c * X)
        Y = Y - fy * cfg.dt + torch.sqrt(2.0 * kap / cfg.beta * cfg.dt) * zy
        Xn = X + cfg.mu_x * (-fx + bias) * cfg.dt + noise_x * zx
        span = tb.XMAX - tb.XMIN
        Xw = torch.remainder(Xn - tb.XMIN, 2.0 * span)
        X = torch.where(Xw > span, 2.0 * span - Xw, Xw) + tb.XMIN

        if step % save_every == 0 or step == n_steps - 1:
            out_Fp.append(Fp_hat.detach().cpu().numpy().copy())
            out_C.append(C.detach().cpu().numpy().copy())
            out_t.append(step * cfg.dt)
            if len(out_t) % 10 == 1:
                print(f"  step {step}/{n_steps} ({time.time()-t0:.0f}s)",
                      flush=True)

    os.makedirs(OUT, exist_ok=True)
    Fpt = np.array(out_Fp); Ct = np.array(out_C)
    for b, sd in enumerate(SEEDS):
        for m, arm in enumerate(ARMS):
            r = b * M + m
            np.savez_compressed(
                os.path.join(OUT, f"{arm}__seed{sd:02d}.npz"),
                t=np.array(out_t), Fp_hat_t=Fpt[:, r], C_t=Ct[:, r],
                x_grid=xg.cpu().numpy(), A_ref=A_ref.cpu().numpy(),
                Ap_ref=Ap_ref.cpu().numpy(), eval_mask=emask.cpu().numpy(),
                io_gamma=alloc.gamma_hat()["gamma"][r],
                io_a_cell=alloc.a_cell, io_cell_edges=alloc.edges,
                seed=sd, arm=arm)
    rows = [x for x in alloc.rows]
    with open(os.path.join(OUT, "meta.json"), "w") as fh:
        json.dump(dict(config=dict(beta=cfg.beta, H=cfg.H, k=cfg.k, c=cfg.c,
                                   mu_x=cfg.mu_x, a_kappa=cfg.a_kappa,
                                   dt=cfg.dt, N=cfg.N),
                       T=T_TOTAL, n_steps=n_steps, seeds=SEEDS, arms=ARMS,
                       io_cfg=dict(n_cells=io_cfg.n_cells,
                                   obs_every=io_cfg.obs_every,
                                   opportunity_every=io_cfg.opportunity_every),
                       n_opportunities=len(rows) // max(R, 1),
                       wall_seconds=time.time() - t0), fh, indent=2)
    print(f"done in {time.time()-t0:.0f}s -> {os.path.relpath(OUT, ROOT)}",
          flush=True)


if __name__ == "__main__":
    main()
