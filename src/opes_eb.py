"""OPES_METAD adapter for the entropic-bottleneck engine.

Standalone per-(config, seed) OPES runner that reuses the eb engine's own force
(``dU_of`` + omega geometry), reflected integrator and grid, swapping the ABF mean
force for an :class:`opes_core.OPESState` bias.  Reconstructs F two ways so the
choice is a flag, not a rewrite:
  * "meanforce": integrate the local mean force accumulated under the OPES-biased
    dynamics (matches how ABF/mFR reconstruct F -> fair comparison);
  * "reweight": native OPES estimate -beta^{-1} log rho(z).

Emits the same flat npz record shape as ``eb_abffr_core.simulate_batch`` rows so the
existing analysis consumes OPES runs unchanged. No F_ref leakage.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch

import eb_abffr_core as eb
import opes_core as oc


def run_opes_eb(config: "eb.PhysConfig", seed: int, opes_cfg: oc.OPESConfig,
                device=None, dtype=None, estimator: str = "meanforce", verbose=False):
    """One OPES run on one entropic-bottleneck config. Returns a flat result dict."""
    device = eb.DEVICE if device is None else device
    dtype = eb.DTYPE if dtype is None else dtype
    oc.assert_no_reference_leakage(False, "opes")

    x_grid, dx, eval_mask, idx0 = eb.build_grid(device, dtype)
    G = x_grid.numel()
    beta = float(config.beta); Hc = config.H
    oout, oin, sw = config.omega_out, config.omega_in, config.s
    N = int(config.N); dt = float(config.dt); n_steps = int(config.n_steps)
    noise_amp = math.sqrt(2.0 * dt / beta)

    F_ref, Fp_ref = eb.reference_profiles(
        x_grid, eval_mask, torch.tensor(beta, device=device, dtype=dtype),
        torch.tensor(Hc, device=device, dtype=dtype),
        torch.tensor(oout, device=device, dtype=dtype),
        torch.tensor(oin, device=device, dtype=dtype),
        torch.tensor(sw, device=device, dtype=dtype))
    F_ref = F_ref.squeeze(0); Fp_ref = Fp_ref.squeeze(0)

    # init conditions (reuse eb's batched init with a single config)
    X0, Y0 = eb.init_conditions_batched(
        [seed], N, torch.tensor([beta], device=device, dtype=dtype),
        torch.tensor([oout], device=device, dtype=dtype),
        torch.tensor([oin], device=device, dtype=dtype),
        torch.tensor([sw], device=device, dtype=dtype), device, dtype)
    X = X0.reshape(1, N).clone(); Y = Y0.reshape(1, N).clone()
    gen = torch.Generator(device=device); gen.manual_seed(3000 + seed)

    opes = oc.OPESState(opes_cfg, device, dtype)
    # ABF-style mean-force accumulators (readout only; not fed to dynamics)
    C = torch.zeros(1, G, device=device, dtype=dtype)
    Sf = torch.zeros(1, G, device=device, dtype=dtype)
    k_h, r_h = eb.gaussian_kernel(config.h, dx, device, dtype)
    burn = n_steps // 5
    t0 = time.perf_counter()
    for step in range(n_steps):
        om = eb.omega_of(X, oout, oin, sw); dom = eb.domega_of(X, oout, oin, sw)
        fx = eb.dU_of(X, Hc) + om * dom * Y * Y
        fy = om * om * Y
        if step >= burn:
            idx = torch.clamp(torch.round((X - eb.XMIN) / dx).long(), 0, G - 1)
            C.scatter_add_(1, idx, torch.ones_like(X)); Sf.scatter_add_(1, idx, fx)
        z = X.reshape(-1)
        bias = opes.bias_force_at(z, step=step).reshape(1, N)
        zx = torch.randn((1, N), device=device, dtype=dtype, generator=gen)
        zy = torch.randn((1, N), device=device, dtype=dtype, generator=gen)
        X = eb.reflect_into(X + (-fx + bias) * dt + noise_amp * zx, eb.XMIN, eb.XMAX)
        Y = Y + (-fy) * dt + noise_amp * zy
        if (step + 1) % max(int(opes_cfg.pace), 1) == 0:
            opes.deposit(X.reshape(-1))

    # F reconstruction
    if estimator == "reweight":
        F_hat = opes.free_energy()
        Fp_hat = opes.mean_force()
    else:  # meanforce (default): integrate the OPES-sampled local mean force
        Fp_hat = eb.smooth(Sf, k_h, r_h, dx) / (eb.smooth(C, k_h, r_h, dx) + config.min_count + eb.EPS)
        B = eb.cumtrapz(Fp_hat, dx)          # keep 2-D (B, G) for cumtrapz
        F_hat = (B - B[:, idx0:idx0 + 1]).squeeze(0)
        Fp_hat = Fp_hat.squeeze(0)
    F_hat = F_hat - F_hat[eval_mask].mean()
    F_ref_c = F_ref - F_ref[eval_mask].mean()

    def _rms1d(a, b):  # interior-window RMS on 1-D profiles
        d = (a - b)[eval_mask]
        return float(torch.sqrt(torch.mean(d * d)))
    l2_f = _rms1d(F_hat, F_ref_c)
    l2_fp = _rms1d(Fp_hat, Fp_ref)
    d = opes.diagnostics()
    return dict(
        method="opes", seed=int(seed), l2_f=l2_f, l2_fp=l2_fp,
        beta=beta, H=Hc, omega_out=oout, omega_in=oin, s=sw,
        opes_barrier=opes_cfg.barrier, opes_pace=opes_cfg.pace, opes_sigma=opes_cfg.sigma,
        opes_gamma=("inf" if math.isinf(opes.gamma) else opes.gamma),
        opes_neff_frac=d["neff_frac"], opes_n_kernels=d["n_kernels"],
        opes_bias_range=d["bias_range"], estimator=estimator,
        F_hat=eb.npy_(F_hat) if hasattr(eb, "npy_") else F_hat.detach().cpu().numpy(),
        F_ref=F_ref_c.detach().cpu().numpy(), x_grid=x_grid.detach().cpu().numpy(),
        runtime_seconds=time.perf_counter() - t0, had_nan=bool(not np.isfinite(l2_f)))
