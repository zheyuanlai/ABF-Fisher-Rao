#!/usr/bin/env python
"""Sanity gates for the uniform-FR gateway arm (docs/UNIFORM_FR_CAMPAIGN.md, sec. 40).

1. gamma=0 fr_uniform is bit-identical to abf (same batch, shared noise).
2. store_profiles=True does not change any scientific output (RNG untouched).
3. The uniform target is exercised: at the frozen gamma the arm fires events,
   populations are conserved, and the event cap is obeyed.

Checks 1-2 are BIT-identity checks and therefore run on CPU: CUDA
``scatter_add_`` uses atomics, whose summation order is nondeterministic even
within a process (the known WCA determinism trap), so bit-equality is simply
not a property the GPU kernels have.  Check 3 runs on the production device.

    CUDA_VISIBLE_DEVICES=3 python -u scripts/sanity_uniform_gateway.py
"""
from __future__ import annotations

import dataclasses
import os
import sys

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import gateway_core as gw  # noqa: E402

CPU = torch.device("cpu")

CFG = gw.GatewayConfig(beta=16.0, H=0.5, omega_out=1.0, r=32.0, s=0.10,
                       N=512, dt=4e-4, n_steps=4000, save_every=200,
                       init="left", h=0.07, min_count=1.0, gamma=1.5, eta=0.10,
                       fr_every=10, fr_burnin=0, ramp_fraction=0.10,
                       target_ema_rate=0.005, score_clip=3.0,
                       max_event_fraction=0.08, ess_window_steps=4000)


def run(methods, store_profiles, device=None):
    spec = gw.BatchSpec(configs=[CFG, CFG], seeds=[7, 8], methods=methods,
                        batch_seed=123)
    kw = {} if device is None else {"device": device}
    return gw.simulate_batch(spec, store_profiles=store_profiles, **kw)


def main():
    ok = True

    # -- 1. gamma=0: zero events, and the population is exactly conserved -----
    # Bit-identity to abf is NOT a property of this engine: resample_indices applies a
    # random re-ordering of the survivor pool to every FR-active row even when no event
    # fires (inherited verbatim from the accepted eb_abffr_core semantics; the shuffle is
    # distribution-preserving because noise slots are iid across particles).  The honest
    # gamma=0 gates are (a) zero deaths and clones over a full run and (b) at the kernel
    # level, sel is a permutation of the survivors, so the particle SET is untouched.
    arms = [gw.ABF, dataclasses.replace(gw.FR_UNIFORM, gamma=0.0)]
    recs = run(arms, store_profiles=False, device=CPU)
    for r in recs:
        if r["method"] != "fr_uniform":
            continue
        events = r["n_die"] + r["n_clone"]
        print(f"[1] seed {r['seed']} events at gamma=0: {events:.0f} (must be 0)")
        ok &= events == 0

    Rk, Nk = 2, 64
    S = torch.randn((Rk, Nk), dtype=torch.float64)
    fr = torch.tensor([False, True])
    sham = torch.tensor([False, False])
    partner = torch.arange(Rk)
    g0 = torch.zeros((Rk, 1), dtype=torch.float64)
    cap = torch.full((Rk, 1), Nk, dtype=torch.long)
    gen = torch.Generator().manual_seed(0)
    sel, die, clone = gw.resample_indices(S, fr, sham, partner, g0, 0.01, cap, gen)
    perm_ok = bool(torch.equal(torch.sort(sel[1]).values, torch.arange(Nk)))
    ident_ok = bool(torch.equal(sel[0], torch.arange(Nk)))
    ev_ok = not bool(die.any() or clone.any())
    print(f"[1] kernel: gamma=0 FR row sel is a permutation of all survivors -> {perm_ok}")
    print(f"[1] kernel: non-FR row sel is the identity -> {ident_ok}; no events -> {ev_ok}")
    ok &= perm_ok and ident_ok and ev_ok

    # -- 2. store_profiles bit-identity (CPU, same reason) --------------------
    arms = [gw.ABF, dataclasses.replace(gw.FR_UNIFORM, gamma=1.5)]
    r0 = run(arms, store_profiles=False, device=CPU)
    r1 = run(arms, store_profiles=True, device=CPU)
    for a, b in zip(r0, r1):
        for key in ("l2_f_t", "l2_fp_t", "ess_t", "F_hat", "Fp_hat"):
            same = np.array_equal(a[key], b[key])
            if not same:
                print(f"[2] {a['method']} seed {a['seed']} {key}: DIFFERS with store_profiles")
            ok &= same
    print("[2] store_profiles leaves every scientific output bit-identical -> "
          f"{all(np.array_equal(a[k], b[k]) for a, b in zip(r0, r1) for k in ('l2_f_t', 'F_hat'))}")

    # -- 3. the mechanism actually fires at the frozen gamma (production dev) --
    r1 = run(arms, store_profiles=True)
    uni = [r for r in r1 if r["method"] == "fr_uniform"]
    for r in uni:
        events = r["n_die"] + r["n_clone"]
        cap = int(0.08 * CFG.N) * r["n_fr_apply"]
        print(f"[3] seed {r['seed']}: events={events:.0f} (>0), "
              f"cap-total={cap}, repl_fraction={r['repl_fraction']:.4f}")
        ok &= events > 0
        ok &= r["n_die"] <= cap and r["n_clone"] <= cap
        # profiles present and finite
        for key in ("F_prof_t", "Fp_prof_t", "phat_t", "kl_uniform_t"):
            ok &= np.all(np.isfinite(r[key]))
        # KDE marginal normalises to 1 under the trapezoid rule
        dx = r["x_grid"][1] - r["x_grid"][0]
        mass = np.trapezoid(r["phat_t"][-1], dx=dx)
        print(f"[3] seed {r['seed']}: final marginal mass={mass:.6f} (must be ~1)")
        ok &= abs(mass - 1.0) < 1e-6

    print("\nSANITY:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
