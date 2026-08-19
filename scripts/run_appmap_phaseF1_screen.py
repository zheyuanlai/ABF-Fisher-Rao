"""Phase F1: plain-SHUS Type-C screen on the bi-channel torus (NO reallocation).

Design frozen in docs/PREREGISTRATION_APPLICATION_MAP.md (Phase F) before this run.
Six cells x 8 seeds x 4 adaptation gains in ONE noise-paired batch; the screen
decides, from plain-SHUS rows only, whether any cell carries a Type-C deficit that
adaptation-rate tuning does not repair.  F2 (the reallocation experiment) is gated on
this outcome and gets its own dated freeze.
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from abpfr.diagnostics import (establishment_time_median, first_persistent,
                               kde_noise_floor)
from abpfr.grid1p import binned_density1p, kl_to_uniform1p
from abpfr.grid2d import periodic_gaussian_kernel
from abpfr.io import save_run
from abpfr.systems import bichannel as bc
from abpfr.systems.gateway import Method

SEEDS = list(range(8))
GAINS = (1.0, 2.0, 4.0, 8.0)
CELLS = ((1.0, 0.0), (1.5, 0.0), (2.0, 0.0), (2.5, 0.0), (1.5, 0.5), (2.0, 0.5))
BATCH_SEED = 20260950
COMMON = dict(K=1024, dt=1e-3, n_steps=800_000, block=20, eps_bw=0.06, eta_bw=0.25,
              n_saves=400, profile_every=8, joint_every=40, ess_window_steps=4000,
              n_strata=32)
OUT = "results/appmap_phaseF1_screen"
SAVE_KEYS = ["time", "profile_time", "joint_time", "pmf_t", "marginal_t", "pB_phi_t",
             "joint_t", "x_grid", "psi_grid", "F_ref", "F2_ref", "p_cond_ref",
             "pB_phi_ref", "l2_f_t", "kl_u_t", "tv_u_t", "e_cond_t", "e_chan_t",
             "ess_anc_t", "wmax_t", "ess_anc_glob_t", "wmax_glob_t", "n_anc_t",
             "dep_ref_l2_t", "dep_self_l2_t", "P_regions", "event_time",
             "event_theta", "event_ess_fr", "event_turnover"]


def d_tol(cfg, device="cpu", dtype=torch.float64):
    """1.5 x (analytic KL* of the mollified fixed point + finite-K KDE noise), the
    same construction every closed campaign used, on the periodic phi axis."""
    kl_star = bc.analytic_floors(cfg, device, dtype)["kl_star"]
    gen = torch.Generator(device=device)
    gen.manual_seed(777)
    k, r = periodic_gaussian_kernel(cfg.eta_bw, bc.GRID1.dx, bc.GRID1.n, device, dtype)
    X = bc.GRID1.xmin + bc.GRID1.L * torch.rand((256, cfg.K), device=device,
                                                dtype=dtype, generator=gen)
    noise = np.sort(kl_to_uniform1p(binned_density1p(X, k, r, bc.GRID1),
                                    bc.GRID1).cpu().numpy())
    n95 = float(np.quantile(noise, 0.95))
    return 1.5 * (kl_star + n95), kl_star, n95


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    cfgs, seeds = [], []
    for (hp, dl) in CELLS:
        for sd in SEEDS:
            cfgs.append(bc.BiChannelConfig(beta=4.0, Hperp=hp, Delta=dl, **COMMON))
            seeds.append(sd)
    arms = [Method(f"shus_g{g:g}", g_shus=g) for g in GAINS]
    print(f"F1: {len(CELLS)} cells x {len(SEEDS)} seeds x {len(GAINS)} gains = "
          f"{len(cfgs)*len(arms)} rows, T={cfgs[0].T_total:.0f}, K={COMMON['K']}")
    t0 = time.time()
    recs = bc.simulate_batch(cfgs, seeds, arms, batch_seed=BATCH_SEED, device=device,
                             progress=50_000)
    print(f"wall {time.time()-t0:.0f}s")

    os.makedirs(OUT, exist_ok=True)
    for rec in recs:
        cfg = rec["config"]
        tag = (f"hp{cfg['Hperp']:g}_d{cfg['Delta']:g}_{rec['method']['name']}"
               f"_seed{rec['seed']}")
        save_run(os.path.join(OUT, tag), {k: rec[k] for k in SAVE_KEYS},
                 {k: rec[k] for k in ("config", "method", "seed", "batch_seed",
                                      "reference_id", "eval_window", "p_B_ref",
                                      "p_B_ref_biased", "final_l2_f", "int_l2_f",
                                      "final_e_cond", "final_e_chan", "int_e_chan",
                                      "final_p_B", "total_turnover")})
    analyze(recs)


def analyze(recs):
    print("\n" + "=" * 100)
    print("F1 OUTCOME — plain-SHUS Type-C screen (frozen gate: "
          "e_F(T) >= 10 e*, P_B(T/4) > 0.01 all seeds, gain does not repair, "
          "E_chan(T) >= 2 x floor, P_B(last 10%) < 0.8 p_B_ref_biased)")
    print("=" * 100)
    for (hp, dl) in CELLS:
        cfg = bc.BiChannelConfig(beta=4.0, Hperp=hp, Delta=dl, **COMMON)
        e_star = bc.analytic_floors(cfg, "cpu")["e_star"]
        amp = bc.type_c_amplitude(cfg, "cpu")
        fl = bc.conditional_floors(cfg, COMMON["K"], n_rep=64, device="cpu")
        ech_floor = float(np.quantile(fl["e_chan"], 0.95))
        tol, kl_star, n95 = d_tol(cfg)
        sub = [r for r in recs if r["config"]["Hperp"] == hp
               and r["config"]["Delta"] == dl]
        pBb = sub[0]["p_B_ref_biased"]
        print(f"\n--- cell Hperp={hp} Delta={dl} | e*={e_star:.4f} "
              f"typeC_amp={amp:.3f} E_chan floor={ech_floor:.4f} "
              f"D_tol={tol:.4f} p_B_ref(biased)={pBb:.4f} ---")
        print(f"{'arm':>10} {'e_F(T)':>8} {'/e*':>6} {'I_F':>9} {'E_chan':>7} "
              f"{'/floor':>7} {'P_B(T)':>7} {'T_hit^B':>8} {'T_est(marg)':>11} "
              f"{'cens':>5}")
        stats = {}
        for g in GAINS:
            rr = [r for r in sub if r["method"]["name"] == f"shus_g{g:g}"]
            t = rr[0]["time"]
            eF = np.median([r["final_l2_f"] for r in rr])
            IF = np.median([r["int_l2_f"] for r in rr])
            ech = np.median([r["final_e_chan"] for r in rr])
            pB = np.median([r["final_p_B"] for r in rr])
            hits = [first_persistent(r["P_regions"][:, 1] > 0.0, t, 0.05) for r in rr]
            ests = [establishment_time_median(r["kl_u_t"], t, tol, 0.10) for r in rr]
            cens = int(np.sum(np.isnan(ests)))
            stats[g] = dict(eF=eF, IF=IF, ech=ech, pB=pB)
            print(f"{'g=%g'%g:>10} {eF:8.4f} {eF/e_star:6.1f} {IF:9.2f} {ech:7.4f} "
                  f"{ech/ech_floor:7.2f} {pB:7.4f} {np.nanmedian(hits):8.2f} "
                  f"{np.nanmedian(ests):11.2f} {cens:5d}")
        # frozen eligibility gate, evaluated on the best (lowest median I_F) gain
        g_best = min(GAINS, key=lambda g: stats[g]["IF"])
        rr = [r for r in sub if r["method"]["name"] == f"shus_g{g_best:g}"]
        t = rr[0]["time"]
        q = int(0.25 * len(t))
        c1 = stats[g_best]["eF"] >= 10 * e_star
        c2 = min(float(r["P_regions"][q, 1]) for r in rr) > 0.01
        c3 = min(stats[g]["eF"] for g in GAINS) >= 10 * e_star
        c4 = stats[g_best]["ech"] >= 2 * ech_floor
        tail = np.median([np.median(r["P_regions"][int(0.9*len(t)):, 1]) for r in rr])
        c5 = tail < 0.8 * pBb
        print(f"  g_best (min median I_F) = {g_best:g};  gate: "
              f"e_F>=10e* {c1}, channel reached {c2}, gain-immune {c3}, "
              f"E_chan>=2x floor {c4}, still live {c5}  "
              f"=> {'ELIGIBLE' if all([c1,c2,c3,c4,c5]) else 'not eligible'}")


if __name__ == "__main__":
    main()
