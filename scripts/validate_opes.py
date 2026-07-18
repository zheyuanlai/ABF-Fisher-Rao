"""Self-consistency + correctness validation for the engine-agnostic OPES core.

Runs a self-contained 1-D overdamped-Langevin double well (analytic F known), so no
physical engine is needed.  Checks the OPES_METAD fixed point:

  1. reweighted marginal recovers the true Boltzmann marginal  => F_hat ~ F  (L2 small);
  2. the biased (sampled) marginal approaches the well-tempered target ~ P^(1/gamma);
  3. gamma=inf (flat target) drives the biased marginal toward uniform;
  4. the applied bias A_n ~ (1 - 1/gamma) * F  (up to an additive constant);
  5. no-reference-leakage guard fires.

Usage:  CUDA_VISIBLE_DEVICES="" python -u scripts/validate_opes.py   (CPU, seconds)
        python -u scripts/validate_opes.py --device cuda
"""
from __future__ import annotations
import argparse, math, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import opes_core as oc  # noqa: E402

# --- analytic model: F(z) double well on [zmin,zmax], barrier ~ DE ------------
Z_MIN, Z_MAX, N_GRID = 0.0, 1.0, 201
BETA = 1.0


F_AMP = 3.0  # => barrier height 2*F_AMP = 6 kT between the two minima


def true_free_energy(z):
    # double well: minima at z=0.25 and 0.75, barrier at z=0.5, high walls at edges
    return F_AMP * np.cos(4.0 * math.pi * z)


def true_force(z):  # -dF/dz  (physical mean force along z)
    return F_AMP * 4.0 * math.pi * np.sin(4.0 * math.pi * z)


def _center(p, grid):
    return p - np.trapezoid(p, grid) / (grid[-1] - grid[0])


def run_langevin_opes(cfg, n_walkers=2048, n_steps=40000, dt=1e-4, device="cpu", seed=0):
    """Overdamped Langevin on the analytic F with an OPES bias applied along z.

    dz = (F_phys(z) + F_bias(z)) dt + sqrt(2 dt/beta) xi , reflected at [Z_MIN,Z_MAX].
    Deposits every cfg.pace steps. Returns (state, biased_hist, grid_np).
    """
    dev = torch.device(device)
    g = torch.Generator(device=dev); g.manual_seed(seed)
    st = oc.OPESState(cfg, dev)
    grid_np = st.grid.detach().cpu().numpy()
    z = Z_MIN + (Z_MAX - Z_MIN) * torch.rand(n_walkers, generator=g, device=dev)
    noise = math.sqrt(2.0 * dt / cfg.beta)
    edges = np.linspace(Z_MIN, Z_MAX, cfg.n_grid + 1)
    biased_hist = np.zeros(cfg.n_grid)
    burn = n_steps // 4
    for step in range(n_steps):
        zc = z.detach().cpu().numpy()
        f_phys = torch.as_tensor(true_force(zc), device=dev, dtype=z.dtype)
        f_bias = st.bias_force_at(z, step=step)
        z = z + (f_phys + f_bias) * dt + noise * torch.randn(n_walkers, generator=g, device=dev)
        # reflect into [Z_MIN, Z_MAX]
        z = torch.abs(z - Z_MIN) + Z_MIN
        z = Z_MAX - torch.abs(Z_MAX - z)
        z = z.clamp(Z_MIN, Z_MAX)
        if (step + 1) % cfg.pace == 0:
            st.deposit(z)
        if step >= burn:
            biased_hist += np.histogram(z.detach().cpu().numpy(), bins=edges)[0]
    biased_hist = biased_hist / max(biased_hist.sum(), 1) / (grid_np[1] - grid_np[0])
    return st, biased_hist, grid_np


def l2(a, b, grid):
    w = grid[-1] - grid[0]
    return float(np.sqrt(np.trapezoid((a - b) ** 2, grid) / w))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-walkers", type=int, default=2048)
    ap.add_argument("--n-steps", type=int, default=40000)
    ap.add_argument("--gamma", type=float, default=6.0, help="well-tempered bias factor")
    args = ap.parse_args(argv)

    grid = np.linspace(Z_MIN, Z_MAX, N_GRID)
    F_true = _center(true_free_energy(grid), grid)
    # true Boltzmann marginal P ~ exp(-beta F)
    P = np.exp(-BETA * (true_free_energy(grid) - true_free_energy(grid).min()))
    P = P / np.trapezoid(P, grid)

    fails = []

    # ---- 5. no-leakage guard ----
    try:
        oc.assert_no_reference_leakage(True)
        fails.append("no-leakage guard did NOT fire")
    except ValueError:
        print("[ok] no-reference-leakage guard fires")

    # ---- well-tempered run ----
    cfg = oc.OPESConfig(z_min=Z_MIN, z_max=Z_MAX, n_grid=N_GRID, beta=BETA,
                        barrier=2.0 * F_AMP, pace=100, sigma=0.03, sigma_mode="fixed",
                        gamma=args.gamma, gamma_from_barrier=False, bias_force_clip=200.0,
                        warmup_steps=2000)
    st, biased, g_np = run_langevin_opes(cfg, args.n_walkers, args.n_steps, device=args.device)

    # 1. reweighted (native) F_hat ~ F_true
    F_hat = _center(st.free_energy().detach().cpu().numpy(), g_np)
    e_F = l2(F_hat, F_true, g_np)
    print(f"[check1] L2(F_hat, F_true) = {e_F:.4f}  (barrier {2*F_AMP:.1f} kT)")
    if e_F > 0.6:
        fails.append(f"native F error too large: {e_F:.3f}")

    # 2. biased marginal ~ well-tempered target P^(1/gamma)
    wt = P ** (1.0 / args.gamma); wt = wt / np.trapezoid(wt, grid)
    wt_i = np.interp(g_np, grid, wt)
    e_wt = l2(biased, wt_i, g_np)
    e_boltz = l2(biased, np.interp(g_np, grid, P), g_np)
    print(f"[check2] L2(biased, WT-target)={e_wt:.4f}  vs L2(biased, Boltzmann)={e_boltz:.4f}")
    if e_wt >= e_boltz:
        fails.append("biased marginal not closer to WT target than to Boltzmann")

    # 4. applied bias A_n ~ (1-1/gamma) F  (compare shapes, centred)
    A = _center(st.applied_bias().detach().cpu().numpy(), g_np)
    # fixed point: applied bias potential V = -(1-1/gamma) F  (force -V' = +(1-1/g)F'
    # combines with physical -F' to leave residual -F'/gamma => samples P^(1/gamma)).
    A_pred = _center(-(1.0 - 1.0 / args.gamma) * true_free_energy(grid), grid)
    A_pred_i = np.interp(g_np, grid, A_pred)
    e_A = l2(A, A_pred_i, g_np)
    print(f"[check4] L2(A_n, (1-1/gamma)F) = {e_A:.4f}")
    if e_A > 0.8:
        fails.append(f"applied bias shape off: {e_A:.3f}")

    # 3. flat-target ablation: gamma=inf drives biased marginal toward uniform
    cfg_flat = oc.OPESConfig(z_min=Z_MIN, z_max=Z_MAX, n_grid=N_GRID, beta=BETA,
                             barrier=2.0 * F_AMP, pace=100, sigma=0.03,
                             gamma=float("inf"), gamma_from_barrier=False,
                             bias_force_clip=200.0, warmup_steps=2000)
    _, biased_flat, gf = run_langevin_opes(cfg_flat, args.n_walkers, args.n_steps, device=args.device)
    uni = np.full_like(gf, 1.0 / (gf[-1] - gf[0]))
    e_flat_uni = l2(biased_flat, uni, gf)
    e_flat_boltz = l2(biased_flat, np.interp(gf, grid, P), gf)
    print(f"[check3] flat: L2(biased,uniform)={e_flat_uni:.4f} vs L2(biased,Boltzmann)={e_flat_boltz:.4f}")
    if e_flat_uni >= e_flat_boltz:
        fails.append("flat-target biased marginal not closer to uniform than Boltzmann")

    print("\n" + ("FAILED: " + "; ".join(fails) if fails else "ALL OPES VALIDATION CHECKS PASSED"))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
