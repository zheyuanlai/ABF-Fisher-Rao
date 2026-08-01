"""Gate 6: block-bootstrap uncertainty and effective sample size for the alanine reference.

Resamples **whole copies within each window** with replacement (each window has 16 independently
thermalised copies), re-solves MBAR on every replicate, and propagates to SE on the FES and on
the reported observables.

Why copy-level and not frame-level: a bootstrap that resamples individual frames treats
correlated samples as independent and understates the error. Resampling whole copies keeps each
copy's internal correlation intact and exposes the copy-to-copy component -- which is exactly why
the 16 copies were independently thermalised rather than cloned from one structure.

Known limitation, stated rather than hidden: this bootstrap is still blind to error COMMON to all
windows and all copies (a shared systematic in the force field, the integrator, or the CV
definition). It bounds statistical error, not systematic error. The independent-sampler
cross-check is what probes the latter.

Usage: CUDA_VISIBLE_DEVICES=7 python -u scripts/bootstrap_alanine_reference.py
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

from alanine import reference as ref                                    # noqa: E402
from alanine.dynamics import KB                                         # noqa: E402

TWO_PI = 2.0 * math.pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="results/alanine/reference/reference.npz")
    ap.add_argument("--out", default="results/alanine/reference")
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--per-window", type=int, default=300)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda" and os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        raise SystemExit("CUDA_VISIBLE_DEVICES must be set explicitly")
    dtype = torch.float64
    d = np.load(a.ref, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    kT = meta["kT_kJ"]
    beta = 1.0 / kT
    kappa = meta["kappa"]
    ng = meta["n_grid"]
    nw = meta["windows"]
    ncop = meta["copies"]
    traj = d["traj"]                     # (K*copies, frames, 2)
    centers = torch.as_tensor(d["centers"], device=dev, dtype=dtype)
    K = centers.shape[0]
    frames = traj.shape[1]
    tw = traj.reshape(K, ncop, frames, 2)
    gdeg = np.degrees(-math.pi + (np.arange(ng) + 0.5) * (TWO_PI / ng))
    pos = gdeg > 0

    def solve(sel_traj):
        """sel_traj: (K, ncop, frames, 2) -> (F grid, dG, P_pos, ess_frac)."""
        flat = sel_traj.reshape(K, ncop * frames, 2)
        stride = max(1, (ncop * frames) // a.per_window)
        sub = flat[:, ::stride][:, :a.per_window]
        phi = torch.as_tensor(sub[..., 0].reshape(-1), device=dev, dtype=dtype)
        psi = torch.as_tensor(sub[..., 1].reshape(-1), device=dev, dtype=dtype)
        N_k = torch.full((K,), sub.shape[1], device=dev, dtype=torch.long)
        f, _, _ = ref.mbar_solve(phi, psi, centers, kappa, beta, N_k, tol=1e-8, max_iter=5000)
        logw = ref.mbar_log_weights(phi, psi, centers, kappa, beta, N_k, f)
        F, _, p = ref.fes_from_weights(phi, psi, logw, ng, beta)
        Pn = p.cpu().numpy()
        Pn = Pn / Pn.sum()
        Ppos = float(Pn[pos, :].sum())
        dG = -kT * math.log(max(Ppos, 1e-300) / max(1 - Ppos, 1e-300))
        w = torch.exp(logw - logw.max())
        ess = float((w.sum() ** 2 / (w ** 2).sum()) / w.numel())
        return F.cpu().numpy(), dG, Ppos, ess

    t0 = time.perf_counter()
    F0, dG0, P0, ess0 = solve(tw)
    print(f"point estimate: dG = {dG0:.4f} kJ/mol = {dG0/kT:.4f} kT | P(phi>0) = {P0:.5f} | "
          f"MBAR weight ESS fraction = {ess0:.4f}  ({time.perf_counter()-t0:.1f}s)", flush=True)

    rng = np.random.default_rng(20260805)
    dGs, Ps, Fs = [], [], []
    for b in range(a.n_boot):
        idx = rng.integers(0, ncop, size=(K, ncop))          # resample COPIES within each window
        sel = np.take_along_axis(tw, idx[:, :, None, None], axis=1)
        try:
            Fb, dGb, Pb, _ = solve(sel)
        except RuntimeError as e:
            print(f"  replicate {b}: MBAR failed ({e}); skipped", flush=True)
            continue
        dGs.append(dGb); Ps.append(Pb); Fs.append(Fb)
        if (b + 1) % 25 == 0:
            print(f"  {b+1}/{a.n_boot} replicates, dG sd so far {np.std(dGs)/kT:.4f} kT "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)

    dGs = np.array(dGs); Ps = np.array(Ps)
    Fs = np.array(Fs)
    finite = np.isfinite(Fs).all(0) & np.isfinite(F0)
    F_se = np.full_like(F0, np.nan)
    F_se[finite] = Fs[:, finite].std(0)
    m8 = finite & (F0 - np.nanmin(F0[finite]) <= 8 * kT)

    out = dict(n_boot=int(len(dGs)), per_window=a.per_window,
               dG_kJ=float(dG0), dG_kT=float(dG0 / kT),
               dG_se_kJ=float(dGs.std()), dG_se_kT=float(dGs.std() / kT),
               P_phi_pos=float(P0), P_se=float(Ps.std()),
               mbar_weight_ess_frac=float(ess0),
               F_se_median_kJ=float(np.nanmedian(F_se[m8])),
               F_se_p90_kJ=float(np.nanpercentile(F_se[m8], 90)),
               F_se_max_kJ=float(np.nanmax(F_se[m8])),
               F_se_median_kT=float(np.nanmedian(F_se[m8]) / kT),
               eval_cells=int(m8.sum()), wall_seconds=time.perf_counter() - t0)
    np.savez_compressed(os.path.join(a.out, "bootstrap.npz"), F_se=F_se, dGs=dGs, mask8=m8,
                        meta=json.dumps(out))
    with open(os.path.join(a.out, "bootstrap.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
