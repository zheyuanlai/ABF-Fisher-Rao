"""Phase F3a: the augmented CV (SHUS on (phi,psi)) against 1D SHUS + conditional
reallocation, at equal computational budget.

Frozen in docs/PREREGISTRATION_APPLICATION_MAP.md before this run.  Same cells, same
seeds, same batch_seed and same B as F2, so every row here is EXACTLY noise-paired
with its stored F2 counterpart.  Scored on the same deliverable F(phi).
"""
import glob
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from abpfr.io import load_run, save_run
from abpfr.metrics import paired_bootstrap_ci
from abpfr.systems import bichannel as bc
from abpfr.systems.gateway import Method

CELLS = ((2.0, 0.0), (2.5, 0.0), (2.0, 0.5))
SEEDS = list(range(600, 616))
GAINS = (1.0, 2.0, 4.0, 8.0)
BATCH_SEED = 20260960
COMMON = dict(K=1024, dt=1e-3, n_steps=800_000, block=20, eps_bw=0.06, eta_bw=0.25,
              n_saves=400, profile_every=8, joint_every=40, ess_window_steps=4000,
              n_strata=32)
OUT = "results/appmap_phaseF3a_augcv"
F2 = "results/appmap_phaseF2_realloc"
SAVE_KEYS = ["time", "profile_time", "joint_time", "pmf_t", "pmf2_t", "marginal_t",
             "pB_phi_t", "joint_t", "x_grid", "psi_grid", "F_ref", "F2_ref",
             "p_cond_ref", "pB_phi_ref", "l2_f_t", "kl_u_t", "tv_u_t", "e_cond_t",
             "e_chan_t", "ess_anc_t", "wmax_t", "n_anc_t", "dep_ref_l2_t",
             "dep_self_l2_t", "P_regions"]


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    cfgs, seeds = [], []
    for (hp, dl) in CELLS:
        for sd in SEEDS:
            cfgs.append(bc.BiChannelConfig(beta=4.0, Hperp=hp, Delta=dl,
                                           cv="phipsi", **COMMON))
            seeds.append(sd)
    arms = [Method(f"aug_g{g:g}", g_shus=g) for g in GAINS]
    print(f"F3a: augmented CV, {len(cfgs)*len(arms)} rows, T={cfgs[0].T_total:.0f}, "
          f"noise-paired with F2 (B={len(cfgs)}, batch_seed={BATCH_SEED})")
    t0 = time.time()
    recs = bc.simulate_batch(cfgs, seeds, arms, batch_seed=BATCH_SEED, device=device,
                             progress=100_000)
    print(f"wall {time.time()-t0:.0f}s")
    os.makedirs(OUT, exist_ok=True)
    for rec in recs:
        c = rec["config"]
        tag = (f"hp{c['Hperp']:g}_d{c['Delta']:g}_{rec['method']['name']}"
               f"_seed{rec['seed']}")
        save_run(os.path.join(OUT, tag), {k: rec[k] for k in SAVE_KEYS},
                 {k: rec[k] for k in ("config", "method", "seed", "batch_seed",
                                      "reference_id", "eval_window", "p_B_ref",
                                      "p_B_ref_biased", "final_l2_f", "int_l2_f",
                                      "final_e_chan", "final_p_B")})
    analyze(recs)


def _f2(cell, arm, key):
    out = []
    for sd in SEEDS:
        a, m = load_run(f"{F2}/hp{cell[0]:g}_d{cell[1]:g}_{arm}_seed{sd}")
        out.append(m[key])
    return np.array(out)


def _here(recs, cell, arm, key):
    d = {r["seed"]: r for r in recs if r["config"]["Hperp"] == cell[0]
         and r["config"]["Delta"] == cell[1] and r["method"]["name"] == arm}
    return np.array([d[s][key] for s in SEEDS])


def analyze(recs):
    print("\n" + "=" * 100)
    print("F3a OUTCOME — augmented CV vs 1D CV + conditional reallocation "
          "(equal K, equal steps, same noise, same deliverable F(phi))")
    print("=" * 100)
    for cell in CELLS:
        c2 = bc.BiChannelConfig(beta=4.0, Hperp=cell[0], Delta=cell[1],
                                cv="phipsi", **COMMON)
        e_star = bc.analytic_floors_2d(c2, "cpu")["e_star"]
        base_IF, base_eF = _f2(cell, "shus", "int_l2_f"), _f2(cell, "shus", "final_l2_f")
        print(f"\n--- cell Hperp={cell[0]} Delta={cell[1]} | e*={e_star:.5f} "
              f"| 1D baseline I_F={np.median(base_IF):.2f} "
              f"e_F(T)={np.median(base_eF):.4f} ---")
        print(f"{'arm':>13} {'dI_F % vs 1D shus':>21} {'e_F(T)':>8} {'/e*':>7} "
              f"{'eT ratio':>9}")
        stats = {}
        for arm in ("fr_cond", "cnt_cond"):
            IF, eF = _f2(cell, arm, "int_l2_f"), _f2(cell, arm, "final_l2_f")
            m, lo, hi = paired_bootstrap_ci(100.0 * (IF - base_IF) / base_IF)
            print(f"{arm+' (F2)':>13} {m:9.2f} [{lo:6.2f},{hi:6.2f}] "
                  f"{np.median(eF):8.4f} {np.median(eF)/e_star:7.1f} "
                  f"{np.median(eF)/np.median(base_eF):9.3f}")
            stats[arm] = IF
        for g in GAINS:
            arm = f"aug_g{g:g}"
            IF, eF = _here(recs, cell, arm, "int_l2_f"), _here(recs, cell, arm, "final_l2_f")
            m, lo, hi = paired_bootstrap_ci(100.0 * (IF - base_IF) / base_IF)
            print(f"{arm:>13} {m:9.2f} [{lo:6.2f},{hi:6.2f}] {np.median(eF):8.4f} "
                  f"{np.median(eF)/e_star:7.1f} "
                  f"{np.median(eF)/np.median(base_eF):9.3f}")
            stats[arm] = IF
        # frozen Pareto rule for g*
        eT1 = np.median(_here(recs, cell, "aug_g1", "final_l2_f"))
        qual = [g for g in GAINS
                if np.median(_here(recs, cell, f"aug_g{g:g}", "final_l2_f")) / eT1 <= 1.05]
        gstar = min(qual, key=lambda g: np.median(stats[f"aug_g{g:g}"]))
        best = np.median(stats[f"aug_g{gstar:g}"])
        for g in sorted(qual):
            if abs(np.median(stats[f"aug_g{g:g}"]) - best) / abs(best) < 0.02:
                gstar = g
                break
        print(f"  g* (frozen Pareto rule, qualifying {qual}) = {gstar:g}")
        # THE primary contrast
        va, vb = stats[f"aug_g{gstar:g}"], stats["fr_cond"]
        m, lo, hi = paired_bootstrap_ci(100.0 * (va - vb) / vb)
        verdict = ("augmented CV WINS" if hi < 0 else
                   "conditional reallocation WINS" if lo > 0 else "TIE")
        print(f"  PRIMARY  aug_g{gstar:g} vs fr_cond: {m:7.2f}% "
              f"[{lo:6.2f}, {hi:6.2f}]  => {verdict}")


if __name__ == "__main__":
    main()
