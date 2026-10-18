#!/usr/bin/env python
"""Code-level numerical audit of the pentane R15 cell (docs/PENTANE_R15_DT_FR.md, Section 3).

  1. potentials.forces vs central finite differences of total_energy (unclipped);
  2. analytic DistanceCV geometry / local mean force vs the autodiff path, and the closed form
     f_R = 1/2 e.(grad_5 V - grad_1 V) - 2/(beta R);
  3. exact Hessian of V at the energy minima of every torsional family and on exact Gibbs samples:
     largest eigenvalues, EM stability indicator lambda*dt, harmonic EM variance ratio 1/(1 - lambda dt/2);
  4. clip activity of the three clips on exact Gibbs samples (physical per-atom |F| <= 200, local
     mean-force sample |f_R| <= 480, ABF bias |A'| <= 60 -- the last checked against the reference F');
  5. noise / beta consistency of the production update (stationary variance of a harmonic test mode).

    python scripts/pentane_r15_dt_fr/audit_numerics.py --out results/pentane_r15_dt_fr/audit
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

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
torch.set_default_dtype(torch.float64)
from alkanes import potentials as pot, geometry as geom                    # noqa: E402
from alkanes.distance_cv import DistanceCV                                  # noqa: E402
from alkanes import dt_validation as dv                                     # noqa: E402

REF_V2 = os.path.join(ROOT, "cache", "alkanes_cv", "ref_pentane_b2_R15_v2_meanforce.npz")
DTS = [5.0e-4, 2.5e-4, 1.25e-4, 6.25e-5]
GAUCHE = math.radians(116.57)


def params(clip=200.0):
    return pot.AlkaneParams(n_atoms=5, beta=2.0, sigma=2.3, epsilon=1.0, decouple=False, force_clip=clip)


def fd_check(n=64, h=1e-6, seed=0):
    p0 = params(0.0)
    q, _ = dv.exact_gibbs_pool(n, p0, torch.Generator().manual_seed(seed))
    f = pot.forces(q, p0)
    fd = torch.zeros_like(q)
    for a in range(5):
        for c in range(3):
            e = torch.zeros_like(q); e[:, a, c] = h
            fd[:, a, c] = -(pot.total_energy(q + e, p0) - pot.total_energy(q - e, p0)) / (2 * h)
    rel = float((f - fd).norm() / f.norm())
    return dict(n=n, h=h, max_abs_err=float((f - fd).abs().max()), rel_err=rel, max_force=float(f.norm(dim=-1).max()))


def mean_force_formula_check(n=4096, seed=1):
    p = params(200.0); cv = DistanceCV(0, 4)
    q, _ = dv.exact_gibbs_pool(n, p, torch.Generator().manual_seed(seed))
    R, g, dv_a = cv.geometry(q); R2, g2, dv_b = cv.geometry_autodiff(q)
    F = pot.forces(q, params(0.0))
    f_loc, _, _ = cv.local_mean_force(q, F, p.beta)
    e = (q[:, 4] - q[:, 0]) / R[:, None]
    closed = 0.5 * (e * (-F[:, 4] + F[:, 0])).sum(-1) - 2.0 / (p.beta * R)
    return dict(max_grad_err=float((g - g2).abs().max()), max_div_err=float((dv_a - dv_b).abs().max()),
                max_closed_form_err=float((f_loc - closed).abs().max()), grad_norm2=float((g * g).sum((1, 2)).mean()))


def family_minima():
    p0 = params(0.0)
    starts = {"TT": [0.0, 0.0], "TG+": [0.0, GAUCHE], "G+G+": [GAUCHE, GAUCHE], "G+G-": [GAUCHE, -GAUCHE]}
    out = {}
    for k, d in starts.items():
        q = geom.place_chain(torch.tensor([d]), 5)
        q = dv.minimise(q, p0, n_iter=40000, lr=2e-4)
        H = dv.hessian_batch(q, p0)[0]
        ev = torch.linalg.eigvalsh(H).numpy()
        R = float(DistanceCV(0, 4).value(q)[0])
        fres = float(pot.forces(q, p0).norm())
        out[k] = dict(R15=R, residual_force=fres, eig_top5=ev[-5:].tolist(), eig_n_near_zero=int((np.abs(ev) < 1e-3).sum()),
                      lam_max=float(ev[-1]),
                      em={f"{dt:g}": dict(lam_dt=float(ev[-1] * dt), var_ratio_top=(1.0 / (1.0 - ev[-1] * dt / 2) if ev[-1] * dt < 2 else float("inf")),
                                          var_ratio_modes=[(1.0 / (1.0 - l * dt / 2) if l * dt < 2 else float("inf")) for l in ev[-9:]])
                          for dt in DTS},
                      eig_all=ev.tolist())
    return out


def gibbs_hessian_stats(n=2000, seed=2):
    p0 = params(0.0)
    q, _ = dv.sample_exact(n, p0, seed, target="gibbs", n_pool=400_000, chunk=200_000)
    H = dv.hessian_batch(q, p0)
    ev = torch.linalg.eigvalsh(H)
    lmax = ev[:, -1].numpy()
    res = dict(n=n, lam_max_quantiles={str(qq): float(np.quantile(lmax, qq)) for qq in (0.01, 0.5, 0.99, 1.0)},
               lam_min_quantiles={str(qq): float(np.quantile(ev[:, 0].numpy(), qq)) for qq in (0.0, 0.01, 0.5)})
    for dt in DTS:
        res[f"frac_lam_dt_gt2_dt{dt:g}"] = float(np.mean(lmax * dt > 2.0))
        res[f"frac_lam_dt_gt1.5_dt{dt:g}"] = float(np.mean(lmax * dt > 1.5))
        res[f"median_lam_dt_dt{dt:g}"] = float(np.median(lmax) * dt)
    return res


def gibbs_clip_stats(n=2_000_000, seed=3):
    """Clip activity of the three production clips on exact Gibbs samples (weighted pool)."""
    p = params(200.0); cv = DistanceCV(0, 4)
    gen = torch.Generator().manual_seed(seed)
    tot_w = 0.0; clip_w = 0.0; fclip_w = 0.0; n_atoms = 0; maxraw = 0.0
    with torch.no_grad():
        pass
    for _ in range(n // 200_000):
        q, logw = dv.exact_gibbs_pool(200_000, p, gen)
        w = torch.exp(logw)
        F, nc, mr, _ = dv.clipped_forces_with_stats(q, p)
        p0 = params(0.0)
        Fr = pot.forces(q, p0)
        norm = Fr.norm(dim=-1)                                  # (M, 5)
        clip_w += float((w[:, None] * (norm > 200.0)).sum()); tot_w += float(w.sum()) * 5
        f_loc, R, _ = cv.local_mean_force(q, F, p.beta)
        fclip_w += float((w * (f_loc.abs() > 480.0)).sum())
        maxraw = max(maxraw, float(norm.max()))
    ref = np.load(REF_V2)
    win = ref["window_mask"]
    return dict(n=n, phys_clip_frac_atoms=clip_w / tot_w, fsample_clip_frac=fclip_w / (tot_w / 5), max_raw_force=maxraw,
                ref_absFp_max_window=float(np.abs(ref["Fprime"][win]).max()),
                ref_frac_window_bins_absFp_gt_60=float(np.mean(np.abs(ref["Fprime"][win]) > 60.0)))


def harmonic_noise_check(dt=5e-4, lam=1000.0, beta=2.0, n=200_000, steps=4000, seed=4):
    """1-D overdamped EM x <- x - dt lam x + sqrt(2 dt / beta) xi: stationary Var = 1/(beta lam (1 - lam dt/2))."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.zeros(n)
    for _ in range(steps):
        x = x - dt * lam * x + math.sqrt(2 * dt / beta) * torch.randn(n, generator=gen)
    v = float(x.var()); exact = 1.0 / (beta * lam * (1 - lam * dt / 2))
    return dict(dt=dt, lam=lam, var_em=v, var_em_theory=exact, var_gibbs=1.0 / (beta * lam), ratio=v * beta * lam)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "pentane_r15_dt_fr", "audit"))
    a = ap.parse_args()
    torch.set_num_threads(4)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time(); res = {}
    res["fd_forces"] = fd_check(); print("FD:", res["fd_forces"], flush=True)
    res["mean_force_formula"] = mean_force_formula_check(); print("MF formula:", res["mean_force_formula"], flush=True)
    res["harmonic_noise"] = harmonic_noise_check(); print("harmonic:", res["harmonic_noise"], flush=True)
    res["family_minima"] = family_minima()
    for k, v in res["family_minima"].items():
        print(f"  {k}: R15 {v['R15']:.3f} |F| {v['residual_force']:.1e} lam_max {v['lam_max']:.1f} top5 {np.round(v['eig_top5'], 1).tolist()} "
              + " ".join(f"dt{d}: lam*dt {e['lam_dt']:.3f} var x{e['var_ratio_top']:.2f}" for d, e in v["em"].items()), flush=True)
    res["gibbs_hessian"] = gibbs_hessian_stats(); print("Gibbs Hessian:", res["gibbs_hessian"], flush=True)
    res["gibbs_clip"] = gibbs_clip_stats(); print("Gibbs clips:", res["gibbs_clip"], flush=True)
    res["wall_s"] = time.time() - t0
    json.dump(res, open(os.path.join(a.out, "audit_numerics.json"), "w"), indent=1)
    print(f"wrote {a.out}/audit_numerics.json ({res['wall_s']:.0f}s)")


if __name__ == "__main__":
    main()
