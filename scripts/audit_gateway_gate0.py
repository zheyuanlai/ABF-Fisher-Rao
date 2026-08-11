"""Gate 0 backfill for the entropic gateway — is its conditional ensemble equilibrated?

    python scripts/audit_gateway_gate0.py --out results/v2_validity_audits/gateway_gate0

The gateway is the project's cleanest positive (mFR -12.5 % on 31/32 seeds, beating its own
sham by -14.9 %). It was classified **establishment-limited before Gate 0 existed**, so under
Amendment 10 that reading is *provisional* until backfilled. This backfills it.

**Why the gateway can be audited exactly.** With

    V(x, y) = H (x^2 - 1)^2 + 1/2 omega(x)^2 y^2,     xi = x

the conditional law is Gaussian and known in closed form,

    p(y | x) = N(0, 1 / (beta omega(x)^2)),

the transverse dynamics at fixed `x` is exactly Ornstein-Uhlenbeck with rate `omega(x)^2` (the
sampler is overdamped with unit mobility), so

    tau_perp(x) = 1 / omega(x)^2,        and  <y^2> relaxes at rate 2 omega(x)^2,

and the conditional mean force is analytic:

    <f_loc>_x = 4 H x (x^2 - 1) + omega omega' <y^2>_x
              = 4 H x (x^2 - 1) + omega'/(beta omega)  =  F'(x).

So unlike deca-alanine (umbrella reference, 8.4 % floor) and R15 (importance-sampling
reference), here there is **no reference error at all** to confound the verdict.

**The sharp risk this tests.** `tau_perp` being small on paper is not enough. The gateway is a
narrow constriction (`s = 0.1`) where `omega` is large, and a walker that *crosses it faster
than `tau_perp`* carries an unequilibrated `y` through exactly the bins where the mFR effect
lives. That is deca's failure mode. Three things are therefore measured, not assumed:

  1. `tau_perp(x)` empirically, by starting `y` from a deliberately WRONG width and watching
     `<y^2>` relax -- the gateway analogue of deca's torsional pools;
  2. the gateway **residence time** of walkers in the actual adaptive run;
  3. the Gate 0 statistic itself: per-`x`-bin `<y^2>` against `1/(beta omega^2)` and
     `<f_loc>` against the exact `F'(x)`, in a live ABF run.

Deca gave 61 % relative mean-force error here; R15 beta=2 gave 0.564/0.593.
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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import eb_abffr_core as eb                                                  # noqa: E402
from eb_abffr_core import N_GRID, XMAX, XMIN, dU_of, domega_of, omega_of    # noqa: E402

# ---- the accepted confirmatory cell (results/gateway_anchor/CONFIRMATORY_PREREGISTRATION) ----
BETA, S, R_SEV, BETA_H = 16.0, 0.1, 32.0, 8.0
OOUT = 1.0
OIN = OOUT * R_SEV
HC = BETA_H / BETA
N_WALK, DT, N_STEPS = 2048, 4.0e-4, 100_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/v2_validity_audits/gateway_gate0")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--relax-steps", type=int, default=20_000)
    ap.add_argument("--n-relax", type=int, default=8192)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev, dt_ = args.device, torch.float64
    torch.manual_seed(20260811)

    x_probe = np.array([0.0, 0.05, 0.10, 0.20, 0.50, 1.0])
    om_probe = np.asarray(omega_of(torch.as_tensor(x_probe), OOUT, OIN, S))
    tau_analytic = 1.0 / om_probe ** 2

    print(f"gateway Gate 0: beta={BETA}, s={S}, r={R_SEV}, omega_in={OIN}, omega_out={OOUT}")
    print(f"\n--- 1. tau_perp(x), analytic vs measured ---")
    print(f"{'x':>6} {'omega':>8} {'tau=1/om^2':>12} {'measured':>10} {'ratio':>7}")

    # ---- 1. measured tau_perp: hold x fixed, start y at the WRONG width, watch <y^2> relax ----
    meas = []
    for xv, om in zip(x_probe, om_probe):
        X = torch.full((args.n_relax,), float(xv), device=dev, dtype=dt_)
        # start from the BULK width, which at the gateway is r^2 = 1024x too wide
        Y = torch.randn(args.n_relax, device=dev, dtype=dt_) * math.sqrt(1.0 / (BETA * OOUT ** 2))
        eq = 1.0 / (BETA * float(om) ** 2)
        y2, ts = [], []
        namp = math.sqrt(2.0 * DT / BETA)
        for s in range(args.relax_steps):
            omx = omega_of(X, OOUT, OIN, S)
            Y = Y - DT * (omx * omx * Y) + namp * torch.randn(Y.shape, device=dev, dtype=dt_)
            if s % 5 == 0:
                y2.append(float((Y * Y).mean()))
                ts.append(s * DT)
        y2 = np.array(y2); ts = np.array(ts)
        d = np.abs(y2 - eq) / abs(y2[0] - eq)
        k = np.argmax(d <= math.exp(-1.0)) if (d <= math.exp(-1.0)).any() else len(d) - 1
        tm = float(ts[k])
        meas.append(tm)
        print(f"{xv:6.2f} {float(om):8.3f} {1.0/float(om)**2:12.3e} {tm:10.3e} "
              f"{tm/(1.0/float(om)**2):7.2f}")

    # ---- 2 + 3. live ABF run: gateway residence, and the Gate 0 statistic ----
    print(f"\n--- 2/3. live ABF run ({N_WALK} walkers, {N_STEPS} steps, T={N_STEPS*DT}) ---")
    x_grid = torch.linspace(XMIN, XMAX, N_GRID, device=dev, dtype=dt_)
    dx = float(x_grid[1] - x_grid[0])
    eval_mask = torch.ones(N_GRID, dtype=torch.bool, device=dev)
    F_ref, Fp_ref = eb.reference_profiles(x_grid, eval_mask, BETA, HC, OOUT, OIN, S)
    Fp_ref = torch.as_tensor(Fp_ref, device=dev, dtype=dt_).reshape(-1)

    X = torch.full((N_WALK,), -1.0, device=dev, dtype=dt_)                 # 'left' init
    Y = torch.randn(N_WALK, device=dev, dtype=dt_) * torch.sqrt(
        1.0 / (BETA * omega_of(X, OOUT, OIN, S) ** 2))
    fsum = torch.zeros(N_GRID, device=dev, dtype=dt_)
    csum = torch.zeros(N_GRID, device=dev, dtype=dt_)
    y2sum = torch.zeros(N_GRID, device=dev, dtype=dt_)
    namp = math.sqrt(2.0 * DT / BETA)
    in_gate = torch.zeros(N_WALK, device=dev, dtype=dt_)
    gate_runs = []
    t0 = time.perf_counter()
    for s in range(N_STEPS):
        om = omega_of(X, OOUT, OIN, S)
        dom = domega_of(X, OOUT, OIN, S)
        f_loc = dU_of(X, HC) + om * dom * Y * Y                 # = dV/dx, the ABF estimator
        idx = ((X - XMIN) / dx).long().clamp(0, N_GRID - 1)
        fsum.scatter_add_(0, idx, f_loc)
        csum.scatter_add_(0, idx, torch.ones_like(f_loc))
        y2sum.scatter_add_(0, idx, Y * Y)
        # ABF bias: apply the running mean force estimate
        mf = fsum / csum.clamp_min(1.0)
        bias = mf.gather(0, idx) * (csum.gather(0, idx) >= 20).to(dt_)
        gate = (X.abs() <= S).to(dt_)
        ended = (in_gate > 0) & (gate == 0)
        if bool(ended.any()):
            gate_runs.append(in_gate[ended].cpu().numpy() * DT)
        in_gate = torch.where(gate > 0, in_gate + 1.0, torch.zeros_like(in_gate))
        fx = dU_of(X, HC) + om * dom * Y * Y - bias
        fy = om * om * Y
        X = X - DT * fx + namp * torch.randn(X.shape, device=dev, dtype=dt_)
        Y = Y - DT * fy + namp * torch.randn(Y.shape, device=dev, dtype=dt_)
        X = eb.reflect_into(X, XMIN, XMAX) if hasattr(eb, "reflect_into") else X.clamp(XMIN, XMAX)
    print(f"  run: {time.perf_counter()-t0:.0f} s")

    res = np.concatenate(gate_runs) if gate_runs else np.array([0.0])
    tau_gate = 1.0 / OIN ** 2
    print(f"\n  gateway residence time (|x| <= {S}): median {np.median(res):.4f}, "
          f"mean {res.mean():.4f}, n={res.size}")
    print(f"  tau_perp at the gateway              : {tau_gate:.3e}")
    print(f"  RESIDENCE / TAU_PERP                 : {np.median(res)/tau_gate:.1f}x")

    occ = csum > 200
    mf_obs = (fsum / csum.clamp_min(1.0))[occ]
    y2_obs = (y2sum / csum.clamp_min(1.0))[occ]
    xg = x_grid[occ]
    y2_eq = 1.0 / (BETA * omega_of(xg, OOUT, OIN, S) ** 2)
    ref = Fp_ref[occ]
    rel_mf = float((mf_obs - ref).abs().mean() / ref.abs().mean())
    rel_y2 = float(((y2_obs - y2_eq).abs() / y2_eq).mean())

    gate_bins = occ & (x_grid.abs() <= S)
    if bool(gate_bins.any()):
        g_mf = (fsum / csum.clamp_min(1.0))[gate_bins]
        g_ref = Fp_ref[gate_bins]
        g_y2 = (y2sum / csum.clamp_min(1.0))[gate_bins]
        g_eq = 1.0 / (BETA * omega_of(x_grid[gate_bins], OOUT, OIN, S) ** 2)
        rel_mf_gate = float((g_mf - g_ref).abs().mean() / g_ref.abs().mean())
        rel_y2_gate = float(((g_y2 - g_eq).abs() / g_eq).mean())
    else:
        rel_mf_gate = rel_y2_gate = float("nan")

    print(f"\n  GATE 0 STATISTIC (bins with >200 samples: {int(occ.sum())}/{N_GRID})")
    print(f"    relative |<f_loc> - F'_ref| / |F'_ref|   all bins : {rel_mf:.3f}")
    print(f"                                             gateway  : {rel_mf_gate:.3f}")
    print(f"    relative |<y^2> - 1/(beta om^2)| / eq    all bins : {rel_y2:.3f}")
    print(f"                                             gateway  : {rel_y2_gate:.3f}")
    print(f"    for comparison: deca 0.61 | R15 beta=2 0.564/0.593")

    passed = bool(rel_mf < 0.25 and rel_mf_gate < 0.25)
    verdict = ("GATE 0 PASSES: the gateway's conditional ensemble is equilibrated, including "
               "inside the constriction. Its establishment-limited classification -- and the "
               "mFR positive resting on it -- survive the backfill.") if passed else (
               "GATE 0 FAILS: the gateway's conditional ensemble is NOT equilibrated. Its "
               "establishment-limited classification must be revisited.")
    print(f"\n  VERDICT: {verdict}")

    np.savez_compressed(os.path.join(args.out, "gateway_gate0.npz"),
                        x_probe=x_probe, omega_probe=om_probe, tau_analytic=tau_analytic,
                        tau_measured=np.array(meas), residence=res,
                        x_grid=x_grid.cpu().numpy(), counts=csum.cpu().numpy(),
                        mf_obs=(fsum / csum.clamp_min(1.0)).cpu().numpy(),
                        Fp_ref=Fp_ref.cpu().numpy(),
                        y2_obs=(y2sum / csum.clamp_min(1.0)).cpu().numpy())
    with open(os.path.join(args.out, "verdict.json"), "w") as fh:
        json.dump(dict(beta=BETA, s=S, r=R_SEV, tau_perp_gateway=tau_gate,
                       residence_median=float(np.median(res)),
                       residence_over_tau=float(np.median(res) / tau_gate),
                       rel_mean_force_all=rel_mf, rel_mean_force_gateway=rel_mf_gate,
                       rel_y2_all=rel_y2, rel_y2_gateway=rel_y2_gate,
                       deca_reference=0.61, r15_b2_reference=[0.564, 0.593],
                       gate0_pass=passed, verdict=verdict), fh, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
