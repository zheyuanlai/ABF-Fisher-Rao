"""OPES_METAD adapter for the 2-D metastability toy (xi = x).

Standalone runner over the same reflected-Langevin dynamics the ABF/mFR study uses
(``potentials.{dVdx,dVdy}_xy_torch``), with the ABF mean force swapped for an
:class:`opes_core.OPESState` bias along x.  The reference is the exact
y-quadrature free energy (``abffr.reference.compute_reference``) -- so this is the
cheap CORRECTNESS GATE for the whole OPES stack: OPES F must match quadrature.

F is reconstructed the same way as ABF/mFR (integrate the local mean force
accumulated under the biased dynamics) for a fair comparison; the native reweight
estimate is also returned. No reference leakage: the quadrature F is used only for
post-hoc L2 evaluation.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch

from abffr import potentials
from abffr import reference as ref_mod
from abffr import torch_utils as tu
import opes_core as oc

EPS = 1e-12


def _reference(x_grid_np, beta, y_min=-2.5, y_max=3.5, ny=801):
    y_grid = np.linspace(y_min, y_max, ny)
    r = ref_mod.compute_reference(x_grid_np, y_grid, beta)
    return r["F_ref"], r["Fprime_ref"], r["p_ref"]


def run_opes_meta(beta=4.0, dt=0.002, n_particles=1000, n_steps=100000, seed=0,
                  x_min=-3.0, x_max=3.0, n_grid=401, opes_cfg=None,
                  barrier=8.0, pace=100, sigma=0.12, gamma=float("inf"),
                  gamma_from_barrier=True, warmup_steps=None, abf_h=0.05,
                  estimator="meanforce", device=None, dtype=None, verbose=False):
    """One OPES run on the 2-D metastability toy. Returns a flat result dict."""
    device = tu.resolve_device(None) if device is None else device
    dtype = torch.float32 if dtype is None else dtype
    oc.assert_no_reference_leakage(False, "opes")
    if opes_cfg is None:
        opes_cfg = oc.OPESConfig(
            z_min=x_min, z_max=x_max, n_grid=n_grid, beta=beta, barrier=barrier,
            pace=pace, sigma=sigma, gamma=gamma, gamma_from_barrier=gamma_from_barrier,
            bias_force_clip=60.0, warmup_steps=(n_steps // 10 if warmup_steps is None else warmup_steps),
            fill_edges=True)

    x_grid_np = np.linspace(x_min, x_max, n_grid)
    F_ref, Fp_ref, p_ref = _reference(x_grid_np, beta)
    x_grid = torch.as_tensor(x_grid_np, device=device, dtype=dtype)
    dx = float(x_grid[1] - x_grid[0]); x0 = float(x_min); G = n_grid
    idx0 = int(np.argmin(np.abs(x_grid_np)))
    y_min, y_max = -2.5, 3.5
    noise_scale = math.sqrt(2.0 * dt / beta)

    gen = torch.Generator(device=device); gen.manual_seed(1000 + seed)
    X = (x_min + (x_max - x_min) * torch.rand(n_particles, generator=gen, device=device, dtype=dtype))
    Y = (y_min + (y_max - y_min) * torch.rand(n_particles, generator=gen, device=device, dtype=dtype))

    opes = oc.OPESState(opes_cfg, device, dtype)
    # ABF-style mean-force accumulators (readout only)
    kernel, radius = tu.gaussian_kernel1d(abf_h, dx, device, dtype)
    C_acc = torch.zeros(G, device=device, dtype=dtype)
    S_acc = torch.zeros(G, device=device, dtype=dtype)
    burn = n_steps // 5

    t0 = time.time()
    for step in range(n_steps):
        dvdx = potentials.dVdx_xy_torch(X, Y)
        dvdy = potentials.dVdy_xy_torch(X, Y)
        if step >= burn:
            idx = tu.nearest_index(X.reshape(1, -1), x0, dx, G)
            C_acc += tu.scatter_grid(idx, G).squeeze(0)
            S_acc += tu.scatter_grid(idx, G, dvdx.reshape(1, -1)).squeeze(0)
        bias = opes.bias_force_at(X, step=step)
        nx = torch.randn(X.shape, generator=gen, device=device, dtype=dtype)
        ny = torch.randn(Y.shape, generator=gen, device=device, dtype=dtype)
        X = tu.reflect_into(X + (-dvdx + bias) * dt + noise_scale * nx, x_min, x_max)
        Y = tu.reflect_into(Y + (-dvdy) * dt + noise_scale * ny, y_min, y_max)
        if (step + 1) % max(int(opes_cfg.pace), 1) == 0:
            opes.deposit(X)

    # reconstruct F
    if estimator == "reweight":
        F_hat = tu.to_numpy(opes.free_energy()) if hasattr(tu, "to_numpy") else opes.free_energy().cpu().numpy()
        Fp_hat = opes.mean_force().cpu().numpy()
    else:
        num_s = tu.smooth_grid(S_acc.reshape(1, -1), kernel, radius, dx).squeeze(0)
        den_s = tu.smooth_grid(C_acc.reshape(1, -1), kernel, radius, dx).squeeze(0)
        Fp_hat_t = num_s / (den_s + 1.0 + EPS)
        F_hat_t = tu.center_at_index(tu.cumulative_trapezoid(Fp_hat_t.reshape(1, -1), dx), idx0).squeeze(0)
        F_hat = F_hat_t.cpu().numpy(); Fp_hat = Fp_hat_t.cpu().numpy()

    # interior evaluation window matching the ABF/mFR metastability study
    # (EvalConfig.from_domain([-3,3], margin=0.5) -> [-2.5, 2.5]); the full grid
    # is dominated by the steep reflecting walls (F_ref ~ 15 kT at |x|=3) and is
    # NOT what the baseline reports.
    eval_lo, eval_hi = -2.5, 2.5
    mask = (x_grid_np >= eval_lo) & (x_grid_np <= eval_hi)
    F_hat = F_hat - F_hat[mask].mean(); F_ref_c = F_ref - F_ref[mask].mean()
    def _rms(a, b):
        w = x_grid_np[mask][-1] - x_grid_np[mask][0]
        return float(np.sqrt(np.trapezoid((a[mask] - b[mask]) ** 2, x_grid_np[mask]) / w))
    l2_f = _rms(F_hat, F_ref_c); l2_fp = _rms(Fp_hat, Fp_ref)
    d = opes.diagnostics()
    if verbose:
        print(f"meta opes seed{seed} b{barrier}: L2F={l2_f:.4f} L2Fp={l2_fp:.4f} "
              f"neff={d['neff_frac']:.2f} ({time.time()-t0:.0f}s)")
    return dict(method="opes", seed=int(seed), l2_f=l2_f, l2_fp=l2_fp, beta=beta,
                opes_barrier=barrier, opes_pace=pace, opes_sigma=sigma,
                opes_gamma=("inf" if math.isinf(opes.gamma) else opes.gamma),
                opes_neff_frac=d["neff_frac"], opes_n_kernels=d["n_kernels"],
                opes_bias_range=d["bias_range"], estimator=estimator,
                F_hat=F_hat, F_ref=F_ref_c, x_grid=x_grid_np,
                runtime_seconds=time.time() - t0, had_nan=bool(not np.isfinite(l2_f)))
