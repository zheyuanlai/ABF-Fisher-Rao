"""Phase 0: validate the marginal WFR implementation against the PDE.

No free energy here.  Purely: do the particle W / FR / W+FR operators reproduce

    d_t p = kappa * Lap(p) - lambda * p * (log p - E_p log p)     (reflecting BC)

and do they show the textbook complementarity (W expands support at a rate that
degrades like 1/L^2; FR is L-independent but cannot expand support at all)?
"""
from __future__ import annotations

import argparse, json, math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch

from rcwfr.grid import DEVICE, DTYPE, EPS, Grid1D, trapz
from rcwfr.fisher_rao import kde_marginal, kl_to_uniform, selection_indices
from rcwfr.wasserstein import w_step_sde, w_step_flow


def pde_reference(grid: Grid1D, p0, kappa, lam, T, n_sub=200_000):
    """Explicit-Euler solve of the WFR PDE with Neumann BC on the grid."""
    dx = grid.dx
    dt = T / n_sub
    assert kappa * dt / dx**2 < 0.5, f"unstable: {kappa*dt/dx**2}"
    p = p0.clone()
    u = 1.0 / grid.volume
    out = []
    for n in range(n_sub):
        lap = torch.zeros_like(p)
        lap[:, 1:-1] = (p[:, 2:] - 2 * p[:, 1:-1] + p[:, :-2]) / dx**2
        lap[:, 0] = 2 * (p[:, 1] - p[:, 0]) / dx**2          # Neumann
        lap[:, -1] = 2 * (p[:, -2] - p[:, -1]) / dx**2
        logr = torch.log(torch.clamp(p, min=1e-300)) - math.log(u)
        mean = trapz(p * logr, dx).unsqueeze(1)
        p = p + dt * (kappa * lap - lam * p * (logr - mean))
        p = torch.clamp(p, min=1e-300)
        p = p / trapz(p, dx).unsqueeze(1)
        out.append(p.clone())
    return out, dt


def run_particles(grid, N, kappa, lam, dtau, n_iter, bw, seed, w_mode, fr_rule,
                  x0=0.0, sig0=0.15, rows=8, alpha_ess=0.0, n_bins=45):
    gen = torch.Generator(device=DEVICE); gen.manual_seed(seed)
    X = x0 + sig0 * torch.randn((rows, N), device=DEVICE, dtype=DTYPE, generator=gen)
    X = grid.enforce(X)
    theta = torch.full((rows,), 1.0 - math.exp(-lam * dtau), device=DEVICE, dtype=DTYPE)
    kl, tv, supp = [], [], []
    for it in range(n_iter):
        if w_mode == "sde":
            X = w_step_sde(X, kappa, dtau, grid, gen)
        elif w_mode == "flow":
            X = w_step_flow(X, kappa, dtau, grid, bw)
        if fr_rule != "none":
            sel, _ = selection_indices(X, grid, fr_rule, theta, gen, bw=bw,
                                       n_bins=n_bins, alpha_ess=alpha_ess)
            X = torch.gather(X, 1, sel)
        p = kde_marginal(X, grid, bw)
        kl.append(kl_to_uniform(p, grid))
        supp.append(X.max(dim=1).values - X.min(dim=1).values)
    return (torch.stack(kl).cpu().numpy(), torch.stack(supp).cpu().numpy(), X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/phase0")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res = {}

    # ---------------- (1) operator fidelity vs the PDE ----------------------
    # small dtau + large N + small bandwidth => particle law -> PDE law
    grid = Grid1D(-1.0, 1.0, 401, -1.0, 1.0)
    N, rows = 65536, 4
    kappa, lam = 0.05, 5.0
    dtau, n_iter = 2.5e-3, 400          # T = 1.0
    bw = 0.03
    xg = grid.x()
    for name, (kp, lm, wm, fr) in {
        "W_only":  (kappa, 0.0, "sde", "none"),
        "FR_only": (0.0, lam, "none", "fr"),
        "WFR":     (kappa, lam, "sde", "fr"),
    }.items():
        kl, supp, Xf = run_particles(grid, N, kp, lm, dtau, n_iter, bw, 7,
                                     wm, fr, rows=rows, sig0=0.15)
        p0 = torch.exp(-0.5 * (xg / 0.15) ** 2).unsqueeze(0)
        p0 = p0 / trapz(p0, grid.dx).unsqueeze(1)
        # the particle KDE convolves the law with bandwidth bw, so compare the PDE
        # solution smoothed by the SAME kernel
        traj, dtp = pde_reference(grid, p0, kp, lm, dtau * n_iter, n_sub=40_000)
        from rcwfr.grid import gaussian_kernel, smooth
        k, r = gaussian_kernel(bw, grid.dx, DEVICE, DTYPE)
        kl_pde = []
        for i in range(0, len(traj), len(traj) // n_iter):
            ps = smooth(traj[i], k, r, grid.dx, grid.bc)
            ps = ps / trapz(ps, grid.dx).unsqueeze(1)
            kl_pde.append(float(kl_to_uniform(ps, grid)[0]))
        kl_pde = np.array(kl_pde[:n_iter])
        kl_par = kl.mean(axis=1)
        n = min(len(kl_pde), len(kl_par))
        rel = np.abs(kl_par[:n] - kl_pde[:n]) / np.maximum(kl_pde[:n], 1e-12)
        res[name] = {"kl_particle": kl_par[:n].tolist(), "kl_pde": kl_pde[:n].tolist(),
                     "max_rel_dev_after_10": float(rel[10:].max()),
                     "median_rel_dev": float(np.median(rel[10:]))}
        print(f"{name:8s}  KL(0)={kl_par[0]:.4f} -> KL(T) particle {kl_par[n-1]:.5f} "
              f"PDE {kl_pde[n-1]:.5f}   median rel dev {np.median(rel[10:]):.4f}")

    # ---------------- (2) FR cannot expand support --------------------------
    # bandwidth -> 0 removes the KDE leak; the particle support must be frozen.
    grid2 = Grid1D(-3.0, 3.0, 601, -3.0, 3.0)
    for bw2 in (0.20, 0.02):
        kl, supp, Xf = run_particles(grid2, 4096, 0.0, 5.0, 0.02, 500, bw2, 11,
                                     "none", "fr", rows=4, sig0=0.15)
        print(f"FR-only bw={bw2:4.2f}: support width {supp[0].mean():.4f} -> "
              f"{supp[-1].mean():.4f}  (KL {kl[0].mean():.3f} -> {kl[-1].mean():.3f})")
        res[f"fr_support_bw{bw2}"] = {"w0": float(supp[0].mean()),
                                      "wT": float(supp[-1].mean()),
                                      "kl0": float(kl[0].mean()),
                                      "klT": float(kl[-1].mean())}

    # ---------------- (3) domain-size scaling of the two rates --------------
    print("\ndomain-size scaling (time to KL < 0.05, dtau=0.02):")
    scal = {}
    for L in (1.0, 2.0, 4.0, 8.0):
        gL = Grid1D(-L, L, int(200 * L) + 1, -L, L)
        row = {}
        for name, (kp, lm, wm, fr) in {
            "W": (0.25, 0.0, "sde", "none"),
            "FR": (0.0, 5.0, "none", "fr"),
            "WFR": (0.25, 5.0, "sde", "fr"),
        }.items():
            kl, supp, _ = run_particles(gL, 8192, kp, lm, 0.02, 3000, 0.10, 3,
                                        wm, fr, rows=2, sig0=0.15)
            klm = kl.mean(axis=1)
            hit = np.where(klm < 0.05)[0]
            t = float(hit[0] * 0.02) if len(hit) else float("nan")
            row[name] = t
        scal[L] = row
        print(f"  L={L:4.1f} half-domain:  W {row['W']!s:>8}  FR {row['FR']!s:>8}  "
              f"WFR {row['WFR']!s:>8}")
    res["domain_scaling"] = scal

    with open(os.path.join(a.out, "phase0.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("\nwrote", os.path.join(a.out, "phase0.json"))


if __name__ == "__main__":
    main()
