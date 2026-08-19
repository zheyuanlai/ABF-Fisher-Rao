"""Phase F2: the reallocation experiment on the F1-eligible Type-C cells.

Design frozen in docs/PREREGISTRATION_APPLICATION_MAP.md (Phase F) before this run:
cells, g* = 1, window [20, 215] (t_on = ceil(Q90(T_hit^B)) pooled; censored-T_est^chan
fallback t_off = t_on + 0.25 (T - t_on)), dose-transferred stride 49 (199 events,
matching the frozen winner's dose) with a raw-stride-10 secondary, arms and decision
rules all fixed in advance.
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from abpfr.io import save_run
from abpfr.metrics import paired_bootstrap_ci
from abpfr.systems import bichannel as bc
from abpfr.systems.gateway import Method

CELLS = ((2.0, 0.0), (2.5, 0.0), (2.0, 0.5))
SEEDS = list(range(600, 616))
BATCH_SEED = 20260960
COMMON = dict(K=1024, dt=1e-3, n_steps=800_000, block=20, eps_bw=0.06, eta_bw=0.25,
              n_saves=400, profile_every=8, joint_every=40, ess_window_steps=4000,
              n_strata=32)
T_ON_FRAC, T_OFF_FRAC = 20.0 / 800.0, 215.0 / 800.0
THETA, STRIDE, STRIDE_HI, ALPHA = 0.01, 49, 10, 0.5
WIN = dict(theta=THETA, t_on_frac=T_ON_FRAC, t_off_frac=T_OFF_FRAC, alpha_ess=ALPHA)
ARMS = [
    Method("shus", g_shus=1.0),
    Method("fr_cond", use_fr=True, cond_fr=True, fr_every_blocks=STRIDE, **WIN),
    Method("fr_cond_hi", use_fr=True, cond_fr=True, fr_every_blocks=STRIDE_HI, **WIN),
    Method("cnt_cond", use_fr=True, cond_fr=True, cond_bins1=32, cond_bins2=9,
           fr_every_blocks=STRIDE, **WIN),
    Method("fr_marg", use_fr=True, fr_every_blocks=STRIDE, **WIN),
    Method("sham_cond", sham=True, shadows="fr_cond", fr_every_blocks=STRIDE, **WIN),
]
OUT = "results/appmap_phaseF2_realloc"
SAVE_KEYS = ["time", "profile_time", "joint_time", "pmf_t", "marginal_t", "pB_phi_t",
             "joint_t", "x_grid", "psi_grid", "F_ref", "F2_ref", "p_cond_ref",
             "pB_phi_ref", "l2_f_t", "kl_u_t", "tv_u_t", "e_cond_t", "e_chan_t",
             "ess_anc_t", "wmax_t", "ess_anc_glob_t", "wmax_glob_t", "n_anc_t",
             "dep_ref_l2_t", "dep_self_l2_t", "P_regions", "event_time",
             "event_theta", "event_ess_fr", "event_turnover"]


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    cfgs, seeds = [], []
    for (hp, dl) in CELLS:
        for sd in SEEDS:
            cfgs.append(bc.BiChannelConfig(beta=4.0, Hperp=hp, Delta=dl, **COMMON))
            seeds.append(sd)
    print(f"F2: {len(CELLS)} cells x {len(SEEDS)} seeds x {len(ARMS)} arms = "
          f"{len(cfgs)*len(ARMS)} rows, T={cfgs[0].T_total:.0f}, "
          f"window [{T_ON_FRAC*800:.0f}, {T_OFF_FRAC*800:.0f}]")
    t0 = time.time()
    recs = bc.simulate_batch(cfgs, seeds, ARMS, batch_seed=BATCH_SEED, device=device,
                             progress=50_000)
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
                                      "final_e_cond", "final_e_chan", "int_e_chan",
                                      "final_p_B", "total_turnover")})
    analyze(recs)


def _paired(recs, cell, arm, key):
    sub = {r["seed"]: r for r in recs if r["config"]["Hperp"] == cell[0]
           and r["config"]["Delta"] == cell[1] and r["method"]["name"] == arm}
    return np.array([sub[s][key] for s in SEEDS])


def analyze(recs):
    print("\n" + "=" * 104)
    print("F2 OUTCOME — conditional reallocation on Type-C cells (frozen rules: "
          "fr_cond vs shus <= -5% with CI<0 AND beating sham => independent value; "
          ">= -2% or CI straddling 0 => adds nothing)")
    print("=" * 104)
    for cell in CELLS:
        cfg = bc.BiChannelConfig(beta=4.0, Hperp=cell[0], Delta=cell[1], **COMMON)
        e_star = bc.analytic_floors(cfg, "cpu")["e_star"]
        base_IF = _paired(recs, cell, "shus", "int_l2_f")
        base_eF = _paired(recs, cell, "shus", "final_l2_f")
        pBb = [r for r in recs if r["config"]["Hperp"] == cell[0]
               and r["config"]["Delta"] == cell[1]][0]["p_B_ref_biased"]
        print(f"\n--- cell Hperp={cell[0]} Delta={cell[1]} | e*={e_star:.4f} "
              f"p_B_ref(biased)={pBb:.4f} | baseline I_F={np.median(base_IF):.2f} "
              f"e_F(T)={np.median(base_eF):.4f} ---")
        print(f"{'arm':>11} {'dI_F %':>19} {'e_F(T)':>8} {'/e*':>6} {'E_chan':>7} "
              f"{'P_B(T)':>7} {'turnover':>9} {'ESSanc':>7} {'n_anc/K':>8}")
        for arm in [m.name for m in ARMS]:
            IF = _paired(recs, cell, arm, "int_l2_f")
            rel = 100.0 * (IF - base_IF) / base_IF
            med, lo, hi = paired_bootstrap_ci(rel)
            eF = np.median(_paired(recs, cell, arm, "final_l2_f"))
            ech = np.median(_paired(recs, cell, arm, "final_e_chan"))
            pB = np.median(_paired(recs, cell, arm, "final_p_B"))
            tv = np.median(_paired(recs, cell, arm, "total_turnover"))
            sub = [r for r in recs if r["config"]["Hperp"] == cell[0]
                   and r["config"]["Delta"] == cell[1]
                   and r["method"]["name"] == arm]
            ess = np.median([np.min(r["ess_anc_t"]) / COMMON["K"] for r in sub])
            nanc = np.median([r["n_anc_t"][-1] / COMMON["K"] for r in sub])
            print(f"{arm:>11} {med:8.2f} [{lo:6.2f},{hi:6.2f}] {eF:8.4f} "
                  f"{eF/e_star:6.1f} {ech:7.4f} {pB:7.4f} {tv:9.0f} {ess:7.3f} "
                  f"{nanc:8.3f}")
        # frozen head-to-head contrasts
        for a, b in (("fr_cond", "cnt_cond"), ("fr_cond", "fr_marg"),
                     ("fr_cond", "sham_cond"), ("fr_cond_hi", "fr_cond")):
            va, vb = _paired(recs, cell, a, "int_l2_f"), _paired(recs, cell, b, "int_l2_f")
            m, lo, hi = paired_bootstrap_ci(100.0 * (va - vb) / vb)
            print(f"   contrast {a:>11} vs {b:<11}: {m:7.2f}% [{lo:6.2f}, {hi:6.2f}]")


if __name__ == "__main__":
    main()
