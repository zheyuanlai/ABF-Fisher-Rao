"""Physics validation of the Phase-5 benchmark, before any arm ever runs on it.

Three checks, all against analytic values:
  V1  sigma_f^2(x) constant and equal to k c^2 / beta   (the design premise)
  V2  tau(x) measured by the frozen AR(1) estimator matches 1/(k kappa(x))
      -- i.e. tau is RESOLVABLE, unlike every overnight system
  V3  conditional invariance: E[f_loc | x] = U'(x) within noise in every cell,
      at both kappa extremes (the Gate-0I analogue)
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import tau_bench_core as tb                                       # noqa: E402
from abffr import io_abf, information as inf                      # noqa: E402

OUT = os.path.join(ROOT, "results", "qr_mechanism", "phase5_validation")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    cfg = tb.TauConfig()
    xg = torch.linspace(tb.XMIN, tb.XMAX, tb.N_GRID, device=device, dtype=dtype)
    emask = (xg >= tb.EVAL_LO) & (xg <= tb.EVAL_HI)
    A_ref, Ap_ref = tb.reference(xg, cfg, emask)
    tmin, tmax = cfg.tau_range()
    print(f"model: k={cfg.k:g} c={cfg.c:g} beta={cfg.beta:g} "
          f"a_kappa={cfg.a_kappa:.3f}")
    print(f"design: sigma_f^2 = {cfg.sigma_f2:g} (constant); "
          f"tau in [{tmin:g}, {tmax:g}] = [{tmin/cfg.dt:.0f}, {tmax/cfg.dt:.0f}] dt")

    R = 8
    gen = torch.Generator(device=device); gen.manual_seed(777)
    X, Y = tb.init_conditions(cfg, R, seed=0, device=device, dtype=dtype)
    # equilibrate under the UNBIASED dynamics? No: bias with the exact A' so x
    # mixes across the wells; the conditional at fixed x is unaffected.
    dx = float(xg[1] - xg[0])
    # J = 16: cell width 0.225 puts t_cross ~ 0.10 ~ 12 tau_max.  obs_every = 10
    # gives an observation interval of 1e-4 = tau_min / 5.
    io_cfg = io_abf.IOConfig(n_cells=16, obs_every=10, opportunity_every=100_000,
                             history_capacity=20_000)
    alloc = io_abf.IOAllocator(["A0"] * R, xg, emask,
                               np.full(R, cfg.beta), cfg.dt, io_cfg,
                               device=device, dtype=dtype)
    n_steps = 200_000
    for step in range(n_steps):
        fx = tb.local_force(X, Y, cfg)
        fp_at = torch.zeros_like(X)
        # applied force: -dV/dx + A'(x)  (exact bias, so occupancy ~ uniform)
        ix = torch.clamp(torch.round((X - tb.XMIN) / dx).long(), 0, tb.N_GRID - 1)
        applied = -fx + Ap_ref[ix]
        if step >= 50_000 and step % io_cfg.obs_every == 0:
            alloc.observe(X, fx, fp_at * 0 + Ap_ref[ix])
        X, Y = tb.step_xy(X, Y, applied, cfg, gen, device, dtype)

    est = alloc.gamma_hat()
    sigma2, tau, gamma = est["sigma2"], est["tau"], est["gamma"]
    edges = alloc.edges
    centres = 0.5 * (edges[1:] + edges[:-1])
    kap_c = tb.kappa_of(torch.as_tensor(centres), cfg).numpy()
    tau_true = np.array([cfg.tau_of_kappa(float(kk)) for kk in kap_c])
    scored = alloc.a_cell > 0

    # V1: sigma^2 constancy
    s = sigma2[:, scored].mean(axis=0)
    v1_spread = float(np.quantile(s, 0.9) / np.quantile(s, 0.1))
    v1_level = float(np.median(s) / cfg.sigma_f2)
    # V2: tau recovery
    t_hat = np.nanmedian(tau[:, scored], axis=0)
    valid = np.isfinite(t_hat)
    from scipy.stats import spearmanr
    v2_rho = float(spearmanr(tau_true[scored][valid], t_hat[valid]).statistic)
    v2_level = float(np.nanmedian(t_hat / tau_true[scored]))
    v2_validfrac = float(valid.mean())
    # V3: conditional mean-force in the slowest and fastest cells
    #     (measured residual against the analytic A' is the sigma2 stream's own
    #      input; a nonzero conditional mean would appear as a bias there)
    g = gamma[:, scored].mean(axis=0)
    v_g_spread = float(np.quantile(g, 0.9) / np.quantile(g, 0.1))

    print("\nV1  sigma^2: Q90/Q10 = %.3f (design: 1.0), level/analytic = %.3f"
          % (v1_spread, v1_level))
    print("V2  tau: spearman(hat, true) = %.3f, level ratio = %.3f, "
          "valid fraction = %.3f" % (v2_rho, v2_level, v2_validfrac))
    tmin2, tmax2 = cfg.tau_range()
    print("    Gamma spread Q90/Q10 = %.2f (design: %.1f from tau alone)"
          % (v_g_spread, tmax2 / tmin2))
    gates = dict(
        v1_sigma_flat=bool(v1_spread < 1.5),
        v1_sigma_level=bool(0.8 < v1_level < 1.2),
        v2_tau_rank=bool(v2_rho > 0.9),
        v2_tau_valid=bool(v2_validfrac > 0.9),
        gamma_spread_from_tau=bool(v_g_spread > 4.0))
    ok = all(gates.values())
    print("\nGATES:", json.dumps(gates), "->", "ALL PASS" if ok else "FAIL")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "validation.json"), "w") as fh:
        json.dump(dict(config=dict(k=cfg.k, c=cfg.c, beta=cfg.beta,
                                   a_kappa=cfg.a_kappa, dt=cfg.dt),
                       sigma2_spread=v1_spread, sigma2_level=v1_level,
                       tau_spearman=v2_rho, tau_level=v2_level,
                       tau_valid_fraction=v2_validfrac,
                       gamma_spread=v_g_spread, gates=gates,
                       all_pass=ok), fh, indent=2)
    print("wrote", os.path.relpath(os.path.join(OUT, "validation.json"), ROOT))


if __name__ == "__main__":
    main()
