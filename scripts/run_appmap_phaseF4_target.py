"""Phase F4: how much does the conditional TARGET's parametrization matter?

Frozen in docs/PREREGISTRATION_APPLICATION_MAP.md before this run.  "Uniform in z" is
not a canonical target -- it moves under reparametrization of the descriptor -- so
this measures the cost of getting it deliberately, arbitrarily wrong, and bounds what
a perfect target could add (oracle arm, diagnosis only).
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

CELLS = ((2.0, 0.0), (2.0, 0.5))
SEEDS = list(range(720, 736))
BATCH_SEED = 20260980
COMMON = dict(K=1024, dt=1e-3, n_steps=800_000, block=20, eps_bw=0.06, eta_bw=0.25,
              n_saves=400, profile_every=8, joint_every=40, ess_window_steps=4000,
              n_strata=32)
WIN = dict(theta=0.01, t_on_frac=20.0 / 800.0, t_off_frac=215.0 / 800.0,
           alpha_ess=0.5, fr_every_blocks=49)
ARMS = [
    Method("shus", g_shus=1.0),
    Method("fr_cond", use_fr=True, cond_fr=True, **WIN),
    Method("fr_cond_rp+", use_fr=True, cond_fr=True, cond_target="reparam",
           cond_target_a=0.8, **WIN),
    Method("fr_cond_rp-", use_fr=True, cond_fr=True, cond_target="reparam",
           cond_target_a=-0.8, **WIN),
    Method("fr_cond_oracle", use_fr=True, cond_fr=True, cond_target="oracle", **WIN),
    Method("sham_cond", sham=True, shadows="fr_cond", **WIN),
]
OUT = "results/appmap_phaseF4_target"
SAVE_KEYS = ["time", "profile_time", "joint_time", "pmf_t", "marginal_t", "pB_phi_t",
             "joint_t", "x_grid", "psi_grid", "F_ref", "F2_ref", "p_cond_ref",
             "pB_phi_ref", "l2_f_t", "kl_u_t", "tv_u_t", "e_cond_t", "e_chan_t",
             "ess_anc_t", "wmax_t", "n_anc_t", "dep_ref_l2_t", "dep_self_l2_t",
             "P_regions", "event_time", "event_theta", "event_ess_fr",
             "event_turnover"]


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    cfgs, seeds = [], []
    for (hp, dl) in CELLS:
        for sd in SEEDS:
            cfgs.append(bc.BiChannelConfig(beta=4.0, Hperp=hp, Delta=dl, **COMMON))
            seeds.append(sd)
    print(f"F4: {len(cfgs)*len(ARMS)} rows, T={cfgs[0].T_total:.0f}")
    t0 = time.time()
    recs = bc.simulate_batch(cfgs, seeds, ARMS, batch_seed=BATCH_SEED, device=device,
                             progress=100_000)
    print(f"wall {time.time()-t0:.0f}s")
    os.makedirs(OUT, exist_ok=True)
    for rec in recs:
        c = rec["config"]
        tag = (f"hp{c['Hperp']:g}_d{c['Delta']:g}_{rec['method']['name']}"
               f"_seed{rec['seed']}").replace("+", "p").replace("-", "m", 1)
        save_run(os.path.join(OUT, tag), {k: rec[k] for k in SAVE_KEYS},
                 {k: rec[k] for k in ("config", "method", "seed", "batch_seed",
                                      "reference_id", "eval_window", "p_B_ref",
                                      "p_B_ref_biased", "final_l2_f", "int_l2_f",
                                      "final_e_chan", "final_p_B", "total_turnover")})
    analyze(recs)


def _v(recs, cell, arm, key):
    d = {r["seed"]: r for r in recs if r["config"]["Hperp"] == cell[0]
         and r["config"]["Delta"] == cell[1] and r["method"]["name"] == arm}
    return np.array([d[s][key] for s in SEEDS])


def analyze(recs):
    print("\n" + "=" * 96)
    print("F4 OUTCOME — conditional target sensitivity (rp+/rp- are uniform in "
          "psi' = psi +- 0.8 sin psi: a 9:1 arbitrary mis-specification)")
    print("=" * 96)
    for cell in CELLS:
        cfg = bc.BiChannelConfig(beta=4.0, Hperp=cell[0], Delta=cell[1], **COMMON)
        e_star = bc.analytic_floors(cfg, "cpu")["e_star"]
        base = _v(recs, cell, "shus", "int_l2_f")
        pBb = [r for r in recs if r["config"]["Hperp"] == cell[0]
               and r["config"]["Delta"] == cell[1]][0]["p_B_ref_biased"]
        print(f"\n--- cell Hperp={cell[0]} Delta={cell[1]} | e*={e_star:.5f} "
              f"| p_B_ref(biased)={pBb:.4f} | baseline I_F={np.median(base):.2f} ---")
        print(f"{'arm':>15} {'dI_F % vs shus':>20} {'e_F(T)':>8} {'/e*':>6} "
              f"{'P_B(T)':>7} {'turnover':>9}")
        for m in ARMS:
            IF = _v(recs, cell, m.name, "int_l2_f")
            md, lo, hi = paired_bootstrap_ci(100.0 * (IF - base) / base)
            eF = np.median(_v(recs, cell, m.name, "final_l2_f"))
            print(f"{m.name:>15} {md:9.2f} [{lo:6.2f},{hi:6.2f}] {eF:8.4f} "
                  f"{eF/e_star:6.1f} {np.median(_v(recs, cell, m.name, 'final_p_B')):7.4f} "
                  f"{np.median(_v(recs, cell, m.name, 'total_turnover')):9.0f}")
        for a in ("fr_cond_rp+", "fr_cond_rp-", "fr_cond_oracle"):
            va, vb = _v(recs, cell, a, "int_l2_f"), _v(recs, cell, "fr_cond", "int_l2_f")
            md, lo, hi = paired_bootstrap_ci(100.0 * (va - vb) / vb)
            print(f"   contrast {a:>15} vs fr_cond: {md:7.2f}% [{lo:6.2f}, {hi:6.2f}]")


if __name__ == "__main__":
    main()
