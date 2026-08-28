"""Phase 1/2/4: prescribed-r passive-estimator experiments on the EB potential.

Frozen protocol: ``docs/MECHANISM_CAMPAIGN_PREREGISTRATION.md``.

Nothing here adapts.  The bias force is the exact static

    F_bias(z) = A'_ref(z) + beta^-1 d/dz log r(z)

so the stationary marginal is exactly ``r``; X0 is drawn from ``r`` by inverse
CDF and Y0 from the exact conditional, so there is no burn-in ambiguity.  The
ABF estimator runs PASSIVELY -- it accumulates counts and force sums through
``eb_abffr_core``'s own ``smooth`` / kernel code path and steers nothing.

What is saved is what the bias model consumes: cumulative counts C_t(z), force
sums S_t(z), and the estimator built from them, at every save.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import eb_abffr_core as eb                                        # noqa: E402

OUT = os.path.join(ROOT, "results", "qr_mechanism")

#: Preregistered target family r_{alpha,k}(z) prop exp[alpha cos(k pi z / L)].
L = eb.XMAX
PHASE1_TARGETS = [(0.0, 1), (1.0, 1), (-1.0, 1), (2.0, 1), (-2.0, 1),
                  (1.0, 3), (-1.0, 3)]
SEEDS = list(range(16))


def target_on_grid(xg, alpha, k):
    logr = alpha * torch.cos(k * math.pi * xg / L)
    r = torch.exp(logr - logr.max())
    return r / (r.sum() * float(xg[1] - xg[0]))


def dlogr(x, alpha, k):
    return -alpha * (k * math.pi / L) * torch.sin(k * math.pi * x / L)


def run_block(alphaks, seeds, beta=8.0, h=0.07, min_count=1.0, n_steps=40_000,
              dt=1e-3, N=4096, save_every=400, device=None, tag="phase1"):
    device = device or torch.device("cuda")
    dtype = torch.float64
    cfg = eb.PhysConfig(beta=beta)
    xg, dx, emask, idx0 = eb.build_grid(device, dtype)
    G = xg.numel()
    k_h, r_h = eb.gaussian_kernel(h, dx, device, dtype)
    bcol = torch.tensor([beta], device=device, dtype=dtype).view(1, 1)
    F_ref, Fp_ref = eb.reference_profiles(
        xg, emask, bcol, torch.tensor([[cfg.H]], device=device, dtype=dtype),
        torch.tensor([[cfg.omega_out]], device=device, dtype=dtype),
        torch.tensor([[cfg.omega_in]], device=device, dtype=dtype),
        torch.tensor([[cfg.s]], device=device, dtype=dtype))
    F_ref, Fp_ref = F_ref[0], Fp_ref[0]

    rows = [(a, k, s) for (a, k) in alphaks for s in seeds]
    R = len(rows)
    alpha_r = torch.tensor([a for a, _, _ in rows], device=device,
                           dtype=dtype).view(R, 1)
    k_r = torch.tensor([float(k) for _, k, _ in rows], device=device,
                       dtype=dtype).view(R, 1)

    # exact static bias force per row, on the grid: A'_ref + beta^-1 dlog r
    bias_grid = Fp_ref.view(1, G) + (1.0 / beta) * (
        -alpha_r * (k_r * math.pi / L) * torch.sin(k_r * math.pi * xg.view(1, G) / L))

    # init X ~ r by inverse CDF (per row), Y ~ exact conditional
    X = torch.empty((R, N), device=device, dtype=dtype)
    for i, (a, k, s) in enumerate(rows):
        r = target_on_grid(xg, a, k)
        cdf = torch.cumsum(r, 0); cdf = cdf / cdf[-1]
        rng = np.random.default_rng(10_000 + s)
        u = torch.as_tensor(rng.random(N), device=device, dtype=dtype)
        j = torch.searchsorted(cdf, u).clamp_max(G - 1)
        X[i] = xg[j]
    om0 = eb.omega_of(X, cfg.omega_out, cfg.omega_in, cfg.s)
    gen0 = torch.Generator(device=device); gen0.manual_seed(4242)
    Y = torch.randn((R, N), generator=gen0, device=device, dtype=dtype) \
        / (math.sqrt(beta) * om0)

    gen = torch.Generator(device=device); gen.manual_seed(20260828)
    noise = math.sqrt(2.0 * dt / beta)
    C = torch.zeros((R, G), device=device, dtype=dtype)
    Sf = torch.zeros((R, G), device=device, dtype=dtype)

    saves = [st for st in range(n_steps) if st % save_every == 0 or st == n_steps - 1]
    save_set = set(saves)
    out_C, out_S, out_Fp, out_t = [], [], [], []
    t0 = time.time()
    for step in range(n_steps):
        om = eb.omega_of(X, cfg.omega_out, cfg.omega_in, cfg.s)
        dom = eb.domega_of(X, cfg.omega_out, cfg.omega_in, cfg.s)
        fx = eb.dU_of(X, cfg.H) + om * dom * Y * Y
        fy = om * om * Y
        idx = torch.clamp(torch.round((X - eb.XMIN) / dx).long(), 0, G - 1)
        C.scatter_add_(1, idx, torch.ones_like(X))
        Sf.scatter_add_(1, idx, fx)
        if step in save_set:
            # the ENGINE's estimator line, verbatim (passive: nothing consumes it)
            Fp_hat = eb.smooth(Sf, k_h, r_h, dx) / (
                eb.smooth(C, k_h, r_h, dx) + min_count + eb.EPS)
            out_C.append(C.detach().cpu().numpy().copy())
            out_S.append(Sf.detach().cpu().numpy().copy())
            out_Fp.append(Fp_hat.detach().cpu().numpy().copy())
            out_t.append(step * dt)
        zx = torch.randn((R, N), generator=gen, device=device, dtype=dtype)
        zy = torch.randn((R, N), generator=gen, device=device, dtype=dtype)
        bias = eb.interp1d(X, bias_grid.expand(R, G), dx)
        X = eb.reflect_into(X + (-fx + bias) * dt + noise * zx, eb.XMIN, eb.XMAX)
        Y = Y + (-fy) * dt + noise * zy

    d = os.path.join(OUT, tag)
    os.makedirs(d, exist_ok=True)
    Ct = np.array(out_C); St = np.array(out_S); Fpt = np.array(out_Fp)
    for i, (a, k, s) in enumerate(rows):
        np.savez_compressed(
            os.path.join(d, f"a{a:+.2f}_k{k}_seed{s:02d}.npz"),
            t=np.array(out_t), C_t=Ct[:, i], S_t=St[:, i], Fp_hat_t=Fpt[:, i],
            x_grid=xg.cpu().numpy(), F_ref=F_ref.cpu().numpy(),
            Fp_ref=Fp_ref.cpu().numpy(), eval_mask=emask.cpu().numpy(),
            alpha=a, k=k, seed=s, beta=beta, h=h, min_count=min_count,
            n_steps=n_steps, dt=dt, N=N)
    meta = dict(tag=tag, targets=[list(x) for x in alphaks], seeds=seeds,
                beta=beta, h=h, min_count=min_count, n_steps=n_steps, dt=dt,
                N=N, wall_seconds=time.time() - t0)
    with open(os.path.join(d, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[{tag}] {R} rows in {time.time() - t0:.0f}s "
          f"-> {os.path.relpath(d, ROOT)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["1", "2h", "2m", "4"])
    a = ap.parse_args()
    if a.phase == "1":
        run_block(PHASE1_TARGETS, SEEDS, tag="phase1")
    elif a.phase == "2h":
        for h in (0.035, 0.14):        # 0.07 already covered by phase 1
            run_block([(2.0, 1), (0.0, 1)], SEEDS, h=h, tag=f"phase2_h{h:g}")
    elif a.phase == "2m":
        for m in (0.1, 10.0):          # 1.0 already covered by phase 1
            run_block([(2.0, 1), (0.0, 1)], SEEDS, min_count=m,
                      tag=f"phase2_m{m:g}")
    else:
        for beta in (1.0, 2.0, 4.0, 16.0):   # 8.0 covered by phase 1
            run_block([(2.0, 1)], SEEDS, beta=beta, tag=f"phase4_beta{beta:g}")


if __name__ == "__main__":
    main()
