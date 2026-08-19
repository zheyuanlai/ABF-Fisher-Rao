"""Application-map Phase ALA-1: joint (K x g) plain-SHUS screen on alanine (NO FR).

Design frozen in docs/PREREGISTRATION_APPLICATION_MAP.md before any run: for each
cv in {phi, phipsi} and K in {32, 128, 512}, one batch with five noise-paired gain
arms g in {0.5, 1, 2, 4, 8}, seeds 0..7, 0.5 ns. The tune-first rule is built into
the screen; classification uses the frozen vocabulary + mechanism taxonomy.

Usage:
    python scripts/run_appmap_ala1_screen.py --cv phipsi --K 128    # one batch
    python scripts/run_appmap_ala1_screen.py --analyze              # full map
"""
import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch

from abpfr.diagnostics import establishment_time_median, first_persistent
from abpfr.grid1p import binned_density1p, integral1p, kl_to_uniform1p, smooth1p
from abpfr.grid2d import periodic_gaussian_kernel
from abpfr.io import save_run
from abpfr.metrics import paired_bootstrap_ci
from abpfr.systems import alanine as ala
from abpfr.systems.gateway import Method

SEEDS = list(range(8))
GAINS = (0.5, 1.0, 2.0, 4.0, 8.0)
KS = (32, 128, 512)
CVS = ("phipsi", "phi")
COMMON = dict(n_steps=500_000, block=20, eps_bw=0.08, eta_bw=0.25, n_saves=400,
              profile_every=8, ess_window_steps=4000)
OUT = "results/appmap_ala1_screen"


def batch_seed(cv, K):
    return 20260900 + K + (7 if cv == "phi" else 0)


def gname(g):
    return f"shus_g{g:g}"


def d_tol(cv, K, device="cpu", dtype=torch.float64):
    """1.5 x (analytic KL* of the mollified fixed point + finite-K KDE noise)."""
    ref = ala.load_reference(device, dtype)
    kT = ref["kT"]
    gen = torch.Generator(device=device)
    gen.manual_seed(777)
    if cv == "phipsi":
        from abpfr.diagnostics import kde_noise_floor2
        from abpfr.shus2d import mollified_fixed_point2
        F2 = torch.where(torch.isfinite(ref["F2"]), ref["F2"],
                         torch.full_like(ref["F2"], 12 * kT))
        kl_star = mollified_fixed_point2(F2, 1.0 / kT, COMMON["eps_bw"],
                                         ala.ALA_GRID2)["kl_star"]
        noise = kde_noise_floor2(K, COMMON["eta_bw"], ala.ALA_GRID2, n_rep=256,
                                 seed=777)
    else:
        g1 = ala.ALA_GRID1
        k, r = periodic_gaussian_kernel(COMMON["eps_bw"], g1.dx, g1.n, device,
                                        dtype)
        rho = torch.exp(-(1.0 / kT) * ref["F1"]).unsqueeze(0)
        rho_m = smooth1p(rho, k, r)
        p_star = rho / torch.clamp(rho_m, min=1e-300)
        p_star = p_star / integral1p(p_star, g1).unsqueeze(1)
        kl_star = float(kl_to_uniform1p(p_star, g1))
        ke, re_ = periodic_gaussian_kernel(COMMON["eta_bw"], g1.dx, g1.n, device,
                                           dtype)
        X = g1.xmin + g1.L * torch.rand((256, K), device=device, dtype=dtype,
                                        generator=gen)
        noise = np.sort(kl_to_uniform1p(
            binned_density1p(X, ke, re_, g1), g1).cpu().numpy())
    n95 = float(np.quantile(noise, 0.95))
    return 1.5 * (float(kl_star) + n95), float(kl_star), n95


def run_batch(cv, K):
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    cfgs = [ala.AlaConfig(K=K, cv=cv, **COMMON) for _ in SEEDS]
    arms = [Method(gname(g), g_shus=g) for g in GAINS]
    print(f"ALA-1: cv={cv}, K={K}: {len(SEEDS)} seeds x {len(GAINS)} gains "
          f"= {len(SEEDS)*len(GAINS)} rows, T={cfgs[0].T_total:.0f} ps")
    t0 = time.time()
    recs = ala.simulate_batch(cfgs, SEEDS, arms, batch_seed=batch_seed(cv, K),
                              device=device, progress=25_000)
    print(f"wall {time.time()-t0:.0f}s")
    os.makedirs(OUT, exist_ok=True)
    keys = ["time", "profile_time", "pmf_t", "marginal_t", "x1_grid", "x2_grid",
            "F_ref", "eval_mask", "basin_labels", "l2_f_t", "kl_u_t", "tv_u_t",
            "e_cond_t", "temp_kin_t", "ess_anc_t", "wmax_t", "n_anc_t",
            "dep_ref_l2_t", "dep_self_l2_t", "P_regions"]
    for rec in recs:
        arrays = {k: rec[k] for k in keys}
        if cv == "phi":
            arrays["marginal2_t"] = rec["marginal2_t"]
        meta = {"reference_id": rec["reference_id"], "eval_window": rec["eval_window"],
                "config": rec["config"], "method": rec["method"], "seed": rec["seed"],
                "cv": cv, "K": K, "stage": "appmap_ala1_screen"}
        save_run(os.path.join(OUT, f"{cv}_K{K}_{rec['method']['name']}"
                                   f"_seed{rec['seed']}"), arrays, meta)
    print(f"records -> {OUT}/{cv}_K{K}_*.npz")


def classify(rows):
    hit = np.array([r["hit_frac"] for r in rows])
    est = np.array([r["est_frac"] for r in rows])
    late = ~np.isfinite(hit) | (hit > 0.2)
    if late.mean() >= 0.25:
        return "discovery-limited"
    gap = np.where(np.isfinite(est), est, np.inf) - hit
    med_gap = float(np.nanmedian(gap))
    if med_gap >= 0.25:
        return "establishment-limited"
    med_est = float(np.nanmedian(np.where(np.isfinite(est), est, np.inf)))
    if med_est <= 0.2:
        return "SHUS-sufficient"
    return "intermediate"


def analyze():
    """Basin occupancies are RESCORED here from the stored joint KDEs with the
    corrected prominence-based basins (the engine's online P_regions in the
    first-screen records used the buggy depth-sorted seeds; labels are
    diagnostics-only, so no rerun is needed). Occupied = basin KDE mass
    >= 0.5/K (a KDE-smeared one-walker equivalent, generous side)."""
    import torch
    ref = ala.load_reference("cpu", torch.float64)
    lab, _seeds = ala.basin_labels(ref["F2"], ref["mask8"])
    lab = lab.numpy()
    basin_masks = np.stack([(lab == k) for k in range(4)])       # (4, 97, 97)
    dA = ala.DZ * ala.DZ

    summary = {"common": COMMON, "seeds": SEEDS, "map": {}}
    for cv in CVS:
        for K in KS:
            if not glob.glob(f"{OUT}/{cv}_K{K}_{gname(1.0)}_seed0.npz"):
                continue
            tol, kl_star, n95 = d_tol(cv, K)
            by = {}
            for g in GAINS:
                rows = {}
                for sd in SEEDS:
                    p = f"{OUT}/{cv}_K{K}_{gname(g)}_seed{sd}.npz"
                    if not os.path.exists(p):
                        break
                    with np.load(p) as z:
                        t, T = z["time"], float(z["time"][-1])
                        # basin occupancy from the stored joint KDE (corrected
                        # basins), at profile cadence
                        p2 = z["marginal_t"] if cv == "phipsi" else z["marginal2_t"]
                        occ = np.stack([(p2 * bm[None]).sum(axis=(1, 2)) * dA
                                        for bm in basin_masks], axis=1)  # (np, 4)
                        occupied_all = (occ >= 0.5 / K).all(axis=1)
                        th = first_persistent(occupied_all, z["profile_time"],
                                              hold_frac=0.05)
                        te = establishment_time_median(z["kl_u_t"], t, tol,
                                                       hold_frac=0.10)
                        rows[sd] = dict(
                            seed=sd, T_hit=th, T_est=te,
                            hit_frac=th / T if np.isfinite(th) else np.nan,
                            est_frac=te / T if np.isfinite(te) else np.nan,
                            I_F=float(np.trapezoid(z["l2_f_t"], t)),
                            eT=float(z["l2_f_t"][-1]),
                            D_T=float(z["kl_u_t"][-1]),
                            e_cond_T=float(z["e_cond_t"][-1]),
                            n_basins_final=int((occ[-1] >= 0.5 / K).sum()))
                if len(rows) == len(SEEDS):
                    by[g] = rows
            if 1.0 not in by:
                continue
            base = by[1.0]
            print(f"\n=== cv={cv} K={K}  (D_tol={tol:.3f} = 1.5*(KL*={kl_star:.3f}"
                  f" + n95={n95:.3f}))")
            print(f"{'gain':>5s} {'hit/T':>7s} {'est/T':>9s} {'I_F':>8s} "
                  f"{'e_F(T)':>8s} {'dI_F%':>18s} {'eTr':>6s} {'D_T':>7s} "
                  f"{'Econd':>7s} {'nb':>3s}  class")
            cell = {}
            qualified = {}
            for g in sorted(by):
                rows = by[g]
                lst = list(rows.values())
                dI = np.array([(rows[sd]["I_F"] - base[sd]["I_F"])
                               / base[sd]["I_F"] for sd in SEEDS])
                rT = np.array([rows[sd]["eT"] / base[sd]["eT"] for sd in SEEDS])
                m, lo, hi = paired_bootstrap_ci(dI)
                m_rT = float(np.median(rT))
                med = lambda k: float(np.nanmedian([r[k] for r in lst]))
                n_c = sum(not np.isfinite(r["est_frac"]) for r in lst)
                cls = classify(lst)
                if m_rT <= 1.05:
                    qualified[g] = m
                print(f"{g:>5g} {med('hit_frac'):>7.3f} {med('est_frac'):>7.3f}"
                      f"({n_c}c) {med('I_F'):>8.1f} {med('eT'):>8.3f} "
                      f"{100*m:>6.1f} [{100*lo:>4.1f},{100*hi:>4.1f}] {m_rT:>6.2f} "
                      f"{med('D_T'):>7.3f} {med('e_cond_T'):>7.3f} "
                      f"{med('n_basins_final'):>3.0f}  {cls}")
                cell[f"{g:g}"] = dict(
                    per_seed=lst, classification=cls,
                    paired=dict(dI_F=m, ci=[lo, hi], eT_ratio=m_rT),
                    median={k: med(k) for k in
                            ("hit_frac", "est_frac", "I_F", "eT", "D_T",
                             "e_cond_T", "n_basins_final")})
            if qualified:
                best = min(qualified.values())
                cands = [g for g, v in qualified.items() if v <= best + 0.02]
                g_star = min(cands)
                cell["g_star"] = g_star
                print(f"g* = {g_star:g} (qualified {sorted(qualified)})")
            summary["map"][f"{cv}_K{K}"] = {"D_tol": tol, "kl_star": kl_star,
                                            "noise95": n95, "gains": cell}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nsummary -> {OUT}/summary.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", choices=CVS)
    ap.add_argument("--K", type=int, choices=KS)
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    if a.analyze:
        analyze()
    elif a.cv and a.K:
        run_batch(a.cv, a.K)
    else:
        ap.error("pass --cv CV --K K, or --analyze")
