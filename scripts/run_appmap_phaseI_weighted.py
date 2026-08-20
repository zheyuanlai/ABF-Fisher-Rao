"""Phase I: does WEIGHTED fiber-wise selection make conditional reallocation
target-safe -- and is anything left of it once it is?

Frozen in docs/PREREGISTRATION_APPLICATION_MAP.md before this run.  F4 showed that in
the equal-weight step the selection IS the represented distribution, so an arbitrary
reparametrization of the hidden descriptor flips the sign of the benefit.  Weighted
selection fires the IDENTICAL event (same score, same theta, same draw) and lets the
descendants carry compensating weights, so the score allocates computational effort
while the ensemble keeps representing the same law.

Eleven arms in one paired batch: the F4 target ladder with equal weights, the same
ladder weighted, and both shams.  Primary endpoint is the TARGET-INDUCED SPREAD of
dI_F within each family (P8); the oracle arms are the decisive diagnostic (P10) --
they separate "reallocation reduces variance" from "reallocation borrows the answer
from its target".
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
SEEDS = list(range(800, 816))
BATCH_SEED = 20261000
COMMON = dict(K=1024, dt=1e-3, n_steps=800_000, block=20, eps_bw=0.06, eta_bw=0.25,
              n_saves=400, profile_every=8, joint_every=40, ess_window_steps=4000,
              n_strata=32)
WIN = dict(theta=0.01, t_on_frac=20.0 / 800.0, t_off_frac=215.0 / 800.0,
           alpha_ess=0.5, fr_every_blocks=49)
WT = dict(cond_weighted=True)
RP = dict(cond_target="reparam")
ARMS = [
    Method("shus", g_shus=1.0),
    Method("fr_cond", use_fr=True, cond_fr=True, **WIN),
    Method("fr_cond_rp+", use_fr=True, cond_fr=True, cond_target_a=0.8, **RP, **WIN),
    Method("fr_cond_rp-", use_fr=True, cond_fr=True, cond_target_a=-0.8, **RP, **WIN),
    Method("fr_cond_oracle", use_fr=True, cond_fr=True, cond_target="oracle", **WIN),
    Method("wfr_cond", use_fr=True, cond_fr=True, **WT, **WIN),
    Method("wfr_cond_rp+", use_fr=True, cond_fr=True, cond_target_a=0.8, **RP, **WT,
           **WIN),
    Method("wfr_cond_rp-", use_fr=True, cond_fr=True, cond_target_a=-0.8, **RP, **WT,
           **WIN),
    Method("wfr_cond_oracle", use_fr=True, cond_fr=True, cond_target="oracle", **WT,
           **WIN),
    Method("sham_cond", sham=True, shadows="fr_cond", **WIN),
    Method("wsham_cond", sham=True, shadows="wfr_cond", **WT, **WIN),
    # EXPLORATORY dose ladder (P11): weighting makes hard allocation safe, so these
    # ask whether ten times the dose buys the variance reduction the frozen dose may
    # be too weak to show.  No dose-matched sham -> a positive here is a lead, not a
    # claim; a null from the hot ORACLE arm closes the hypothesis on this system.
    Method("wfr_cond_hot", use_fr=True, cond_fr=True, **WT,
           **{**WIN, "theta": 0.1}),
    Method("wfr_cond_hot_oracle", use_fr=True, cond_fr=True, cond_target="oracle",
           **WT, **{**WIN, "theta": 0.1}),
]
# the three targets a modeller could actually choose; the oracle is not one of them
CHOOSABLE = ("fr_cond", "fr_cond_rp+", "fr_cond_rp-")
OUT = "results/appmap_phaseI_weighted"
SAVE_KEYS = ["time", "profile_time", "joint_time", "pmf_t", "marginal_t", "pB_phi_t",
             "joint_t", "x_grid", "psi_grid", "F_ref", "F2_ref", "p_cond_ref",
             "pB_phi_ref", "l2_f_t", "kl_u_t", "tv_u_t", "e_cond_t", "e_chan_t",
             "ess_anc_t", "wmax_t", "n_anc_t", "dep_ref_l2_t", "dep_self_l2_t",
             "P_regions", "P_regions_n", "ess_w_t", "wmax_w_t", "w_sum_t",
             "event_time", "event_theta", "event_ess_fr", "event_turnover"]
META_KEYS = ("config", "method", "seed", "batch_seed", "reference_id", "eval_window",
             "p_B_ref", "p_B_ref_biased", "final_l2_f", "int_l2_f", "final_e_chan",
             "final_p_B", "final_p_B_n", "final_ess_w", "min_ess_w", "total_turnover")


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    cfgs, seeds = [], []
    for (hp, dl) in CELLS:
        for sd in SEEDS:
            cfgs.append(bc.BiChannelConfig(beta=4.0, Hperp=hp, Delta=dl, **COMMON))
            seeds.append(sd)
    print(f"I1: {len(cfgs)*len(ARMS)} rows, T={cfgs[0].T_total:.0f}", flush=True)
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
                 {k: rec[k] for k in META_KEYS})
    analyze(recs)


def _v(recs, cell, arm, key):
    d = {r["seed"]: r for r in recs if r["config"]["Hperp"] == cell[0]
         and r["config"]["Delta"] == cell[1] and r["method"]["name"] == arm}
    return np.array([d[s][key] for s in SEEDS])


def analyze(recs):
    print("\n" + "=" * 100)
    print("I1 OUTCOME — weighted vs equal-weight fiber-wise selection under three "
          "choosable targets + an oracle")
    print("=" * 100)
    for cell in CELLS:
        cfg = bc.BiChannelConfig(beta=4.0, Hperp=cell[0], Delta=cell[1], **COMMON)
        e_star = bc.analytic_floors(cfg, "cpu")["e_star"]
        base = _v(recs, cell, "shus", "int_l2_f")
        print(f"\n--- cell Hperp={cell[0]} Delta={cell[1]} | e*={e_star:.5f} "
              f"| baseline I_F={np.median(base):.2f} ---")
        print(f"{'arm':>17} {'dI_F % vs shus':>22} {'eF/e*':>7} {'P_B':>7} "
              f"{'P_B^n':>7} {'min ESSw':>9} {'turnover':>9}")
        med = {}
        for m in ARMS:
            IF = _v(recs, cell, m.name, "int_l2_f")
            d, lo, hi = paired_bootstrap_ci(100.0 * (IF - base) / base)
            med[m.name] = (d, lo, hi)
            eF = np.median(_v(recs, cell, m.name, "final_l2_f"))
            print(f"{m.name:>17} {d:9.2f} [{lo:7.2f},{hi:7.2f}] {eF/e_star:7.1f} "
                  f"{np.median(_v(recs, cell, m.name, 'final_p_B')):7.4f} "
                  f"{np.median(_v(recs, cell, m.name, 'final_p_B_n')):7.4f} "
                  f"{np.median(_v(recs, cell, m.name, 'min_ess_w')):9.4f} "
                  f"{np.median(_v(recs, cell, m.name, 'total_turnover')):9.0f}")
        s_eq = max(med[a][0] for a in CHOOSABLE) - min(med[a][0] for a in CHOOSABLE)
        s_wt = (max(med["w" + a][0] for a in CHOOSABLE)
                - min(med["w" + a][0] for a in CHOOSABLE))
        print(f"\n  P8  target-induced spread over {CHOOSABLE}:")
        print(f"      equal weight S = {s_eq:6.2f} points | weighted S = {s_wt:6.2f} "
              f"points | ratio = {s_wt / max(s_eq, 1e-9):.3f}  "
              f"(P8 asks for <= 0.5)")
        for a in CHOOSABLE + ("fr_cond_oracle",):
            va, vb = _v(recs, cell, "w" + a, "int_l2_f"), _v(recs, cell, a, "int_l2_f")
            d, lo, hi = paired_bootstrap_ci(100.0 * (va - vb) / vb)
            print(f"      weighted vs equal-weight, {a:>15}: {d:7.2f}% "
                  f"[{lo:7.2f}, {hi:7.2f}]")
        for a, b in (("wfr_cond", "wsham_cond"), ("fr_cond", "sham_cond")):
            va, vb = _v(recs, cell, a, "int_l2_f"), _v(recs, cell, b, "int_l2_f")
            d, lo, hi = paired_bootstrap_ci(100.0 * (va - vb) / vb)
            print(f"      {a:>15} vs its sham: {d:7.2f}% [{lo:7.2f}, {hi:7.2f}]")
        kl = {a: np.median(_v(recs, cell, a, "kl_u_t")[:, -1])
              for a in ("shus", "fr_cond", "wfr_cond")}
        print(f"      CV-marginal invariance  KL(p_phi||u) at T: " +
              "  ".join(f"{k}={v:.4f}" for k, v in kl.items()))


if __name__ == "__main__":
    main()
