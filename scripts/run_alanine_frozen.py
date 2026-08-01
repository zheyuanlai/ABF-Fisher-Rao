"""Frozen-bias validation for the alanine arms.

Separates a genuinely better LEARNED BIAS from an online estimator that merely counted cloned
samples repeatedly.  For each adaptive endpoint:

  1. take the final saved potential ``B``;
  2. discard the adaptive particle population entirely;
  3. start fresh, **method-independent** configurations (the same ensemble for every arm);
  4. run with the bias frozen -- no ABF update, no birth-death;
  5. reconstruct ``F = B - beta^-1 log p_B + C``.

The applied field is ``spectral_gradient(B)`` of the *saved* potential, which the online sampler
asserts equals the field it applied, so online and frozen runs feel the identical bias.

Preregistered: the frozen-bias improvement must retain >= 2/3 of the online improvement.
Below that, an apparent online gain is an accumulator/cloning artifact, not a better bias.

Usage:
  CUDA_VISIBLE_DEVICES=7 python -u scripts/run_alanine_frozen.py \
      --root results/alanine_oracle/pilot --stage N2048 --ps 50
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from alanine.basins import from_reference                                     # noqa: E402
from alanine.cv2d import BackboneCV2D                                         # noqa: E402
from alanine.dynamics import BAOAB, KB                                        # noqa: E402
from alanine.forcefield import TorchFF, extract_parameters, parameter_hash    # noqa: E402
from alanine.metrics_ala import aligned_l2, build_masks, smooth_reference     # noqa: E402
from alanine.system import PHI_ATOMS, PSI_ATOMS, reference_minimum            # noqa: E402
from alkanes import density2d as d2                                           # noqa: E402
from alkanes import poisson2d as ps                                           # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_alanine_study import ALLOWED_GPUS, enforce_gpu_policy, init_c7eq     # noqa: E402


def frozen_run(B_saved, tff, cv, X0, N, ps_len, dt, gamma, T, n_grid, clip, device, dtype, seed):
    """Fresh dynamics under a frozen bias; returns the biased marginal ``p_B``."""
    beta = 1.0 / (KB * T)
    g1, g2, dz1, dz2 = d2.torus_grid(n_grid, n_grid, device=device, dtype=dtype)
    Bt = torch.as_tensor(B_saved, device=device, dtype=dtype)[None]
    gB1, gB2 = ps.spectral_gradient(Bt, dz1, dz2)
    x = init_c7eq(X0, tff, 1, N, 5.0, dt, gamma, T, device, dtype, seed).reshape(N, -1, 3)
    integ = BAOAB(tff.masses.cpu().numpy(), dt, gamma, T, lambda z: tff.forces(z),
                  device=device, dtype=dtype)
    gen = torch.Generator(device=device).manual_seed(int(seed) + 31)
    v = integ.maxwell((N, x.shape[1], 3), gen, device, dtype)

    def total(z):
        phi, gfull = cv.grad_only(z)
        a1 = phi[:, 0][None]
        a2 = phi[:, 1][None]
        c1 = d2.bilinear_interp2(gB1, g1, g2, dz1, dz2, a1, a2)[0]
        c2 = d2.bilinear_interp2(gB2, g1, g2, dz1, dz2, a1, a2)[0]
        mag = torch.sqrt(c1 * c1 + c2 * c2).clamp_min(1e-30)
        s = torch.clamp(clip / mag, max=1.0)
        c1, c2 = c1 * s, c2 * s
        return tff.forces(z) + (c1[:, None, None] * gfull[:, 0] + c2[:, None, None] * gfull[:, 1])

    f = total(x)
    n_steps = int(ps_len / dt)
    burn = n_steps // 5
    acc = torch.zeros(1, n_grid, n_grid, device=device, dtype=dtype)
    K1, K2 = d2.kernels(g1, g2, 0.15, 0.15)
    n_acc = 0
    for s in range(n_steps):
        x, v, f = integ.step(x, v, f, gen)
        f = total(x)
        if s >= burn and (s + 1) % 20 == 0:
            phi = cv.values(x)
            acc += d2.kde2(phi[0][None], phi[1][None], K1, K2, n_grid, n_grid, dz1, dz2)
            n_acc += 1
    p_B = d2.normalize2(acc / max(n_acc, 1), dz1, dz2)[0]
    F_rec = Bt[0] - (1.0 / beta) * torch.log(p_B.clamp_min(1e-300))
    return F_rec.cpu().numpy(), p_B.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--ps", type=float, default=50.0)
    ap.add_argument("--replicas", type=int, default=2048)
    ap.add_argument("--reference", default="results/alanine/reference/reference.npz")
    a = ap.parse_args()

    vis, free = enforce_gpu_policy(a.replicas * 1.35e-3 * 2)
    device, dtype = "cuda", torch.float64
    bm, ref_meta = from_reference(a.reference)
    F_ref = np.load(a.reference, allow_pickle=True)["F"]
    kT = ref_meta["kT_kJ"]
    n_grid = int(ref_meta["n_grid"])
    pack = build_masks(F_ref, kT)
    F_sm = smooth_reference(F_ref, 0.08, n_grid)
    w = pack["weights"]["equilibrium"]

    system, X0 = reference_minimum()
    P = extract_parameters(system)
    assert parameter_hash(P) == ref_meta["param_hash"]
    tff = TorchFF(P, device=device, dtype=dtype)
    cv = BackboneCV2D(PHI_ATOMS, PSI_ATOMS, n_atoms=22)

    out_dir = os.path.join(a.root, "frozen")
    os.makedirs(out_dir, exist_ok=True)
    res = {}
    for f in sorted(glob.glob(os.path.join(a.root, a.stage, "raw", "*.npz"))):
        d = np.load(f, allow_pickle=True)
        meta = json.loads(str(d["meta"]))
        m = meta["method"]
        online, frozen = [], []
        for r in range(d["final_pmf"].shape[0]):
            B = d["final_pmf"][r]
            online.append(aligned_l2(B, F_sm, w))
            t0 = time.perf_counter()
            F_rec, p_B = frozen_run(B, tff, cv, X0, a.replicas, a.ps, 0.001, 1.0, 300.0,
                                    n_grid, 200.0, device, dtype, seed=5000 + r)
            frozen.append(aligned_l2(F_rec, F_sm, w))
            print(f"  {m} seed{r}: online {online[-1]:.4f}  frozen {frozen[-1]:.4f}  "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)
        res[m] = dict(online=online, frozen=frozen)

    a_on = np.array(res["abf"]["online"]); a_fr = np.array(res["abf"]["frozen"])
    o_on = np.array(res["fr_oracle"]["online"]); o_fr = np.array(res["fr_oracle"]["frozen"])
    imp_on = float(np.median((o_on - a_on) / a_on))
    imp_fr = float(np.median((o_fr - a_fr) / a_fr))
    retention = float(imp_fr / imp_on) if imp_on < 0 else float("nan")
    summary = dict(stage=a.stage, ps=a.ps, replicas=a.replicas,
                   online_improvement=imp_on, frozen_improvement=imp_fr,
                   retention=retention, per_method=res,
                   cuda_visible_devices=vis,
                   note=("retention = frozen improvement / online improvement; >= 2/3 required. "
                         "Below that, an online gain is an accumulator/cloning artifact."))
    json.dump(summary, open(os.path.join(out_dir, f"frozen_{a.stage}.json"), "w"),
              indent=2, default=float)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_method"}, indent=2))


if __name__ == "__main__":
    main()
