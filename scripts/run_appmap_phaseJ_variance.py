"""Phase J: can a MEASURE-PRESERVING conditional selection reduce variance when the
represented conditional is already correct?

Frozen in docs/PREREGISTRATION_APPLICATION_MAP.md before any J2 row.  Phase I showed
that equal-weight reallocation repairs a Type-C deficit only by writing its target into
the represented law, and that the weighted version therefore buys nothing on a deficit
that IS a bias.  This script builds the complementary regime -- start the ensemble at
the exact stationary law of the converged bias and warm-start the accumulator at its
fixed point, so the conditional is right in expectation and only finite-K noise
remains -- and asks whether allocation can cut the variance.

  --scan    the analytic trade-off recorded in the freeze (rarity vs relevance)
  --screen  J1: plain SHUS only, the eligibility gate, over a K ladder
  (default)  J2: the ten-arm experiment on the anchor cell
"""
import itertools
import math
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

PI = math.pi
CELLS = ((1.5, 1.0), (2.0, 1.0))          # exchange active / exchange ~ T
SEEDS = list(range(920, 936))
BATCH_SEED = 20261020
COMMON = dict(K=256, dt=1e-3, n_steps=800_000, block=20, eps_bw=0.06, eta_bw=0.25,
              n_saves=400, profile_every=8, joint_every=40, ess_window_steps=4000,
              n_strata=8, init="stationary", warm_start=True)
WIN = dict(t_on_frac=20.0 / 800.0, t_off_frac=215.0 / 800.0, alpha_ess=0.5,
           fr_every_blocks=49)
WT = dict(cond_weighted=True)
ARMS = [
    Method("shus"),
    Method("fr_cond", use_fr=True, cond_fr=True, theta=0.01, **WIN),
    Method("sham_cond", sham=True, shadows="fr_cond", theta=0.01, **WIN),
    Method("wfr_cond", use_fr=True, cond_fr=True, theta=0.01, **WT, **WIN),
    Method("wfr_cond_hot", use_fr=True, cond_fr=True, theta=0.1, **WT, **WIN),
    Method("wcnt_cond_hot", use_fr=True, cond_fr=True, cond_bins1=8, cond_bins2=8,
           theta=0.1, **WT, **WIN),
    Method("wstate_hot", use_fr=True, cond_fr=True, cond_state=True, theta=0.1,
           **WT, **WIN),
    Method("wstate_eq", use_fr=True, cond_fr=True, cond_state=True, theta=1.0,
           **WT, **WIN),
    Method("wsham_cond", sham=True, shadows="wfr_cond_hot", theta=0.1, **WT, **WIN),
    Method("wsham_eq", sham=True, shadows="wstate_eq", theta=1.0, **WT, **WIN),
]
PAIRS = (("wfr_cond", "wsham_cond"), ("wfr_cond_hot", "wsham_cond"),
         ("wcnt_cond_hot", "wsham_cond"), ("wstate_hot", "wsham_cond"),
         ("wstate_eq", "wsham_eq"), ("fr_cond", "sham_cond"))
OUT = "results/appmap_phaseJ_variance"
SAVE_KEYS = ["time", "profile_time", "joint_time", "pmf_t", "marginal_t", "pB_phi_t",
             "joint_t", "x_grid", "psi_grid", "F_ref", "F2_ref", "p_cond_ref",
             "pB_phi_ref", "l2_f_t", "kl_u_t", "tv_u_t", "e_cond_t", "e_chan_t",
             "ess_anc_t", "wmax_t", "n_anc_t", "dep_ref_l2_t", "dep_self_l2_t",
             "P_regions", "P_regions_n", "ess_w_t", "wmax_w_t", "w_sum_t",
             "event_time", "event_theta", "event_ess_fr", "event_turnover"]
META_KEYS = ("config", "method", "seed", "batch_seed", "reference_id", "eval_window",
             "p_B_ref", "p_B_ref_biased", "final_l2_f", "int_l2_f", "final_e_chan",
             "final_p_B", "final_p_B_n", "final_ess_w", "min_ess_w", "total_turnover")


# -----------------------------------------------------------------------------
# the structural trade-off recorded in the freeze
# -----------------------------------------------------------------------------
def scan():
    print(f"{'Hperp':>6}{'Delta':>7}{'Ha=Hb':>7} | {'p_B(Bolt)':>10}{'p_B(bias)':>10}"
          f" | {'tauA->B':>9}{'tauB->A':>9} | {'|dF from B|':>12}{'e*':>8}{'ratio':>7}")
    for hp, dl, ha in itertools.product((1.25, 1.5, 1.75, 2.0), (0.5, 1.0, 1.5, 2.0),
                                        (1.0, 2.0)):
        cfg = bc.BiChannelConfig(beta=4.0, Hperp=hp, Delta=dl, Ha=ha, Hb=ha)
        ref = bc.reference_objects(cfg.beta, hp, dl, ha, ha, "cpu")
        tAB, tBA, pB = cfg.channel_times()
        P1, P2 = bc.GRID2.mesh("cpu")
        rho = torch.exp(-cfg.beta * (bc.V_of(P1, P2, hp, dl, ha, ha)
                                     - bc.V_of(P1, P2, hp, dl, ha, ha).min()))
        ZA = (rho * (torch.cos(P2) >= 0).double()).sum(dim=1) * bc.GRID2.dx2
        F_A = -torch.log(torch.clamp(ZA, min=1e-300)) / cfg.beta
        d = (ref["F1"] - ref["F1"].mean()) - (F_A - F_A.mean())
        dF = float(torch.sqrt(((d - d.mean()) ** 2).mean()))
        e_star = bc.analytic_floors(cfg, "cpu")["e_star"]
        print(f"{hp:6.2f}{dl:7.2f}{ha:7.1f} | {pB:10.4f}"
              f"{ref['p_B_ref_biased']:10.4f} | {tAB:9.0f}{tBA:9.0f} | "
              f"{dF:12.4f}{e_star:8.4f}{dF/e_star:7.1f}")


# -----------------------------------------------------------------------------
# metrics: the MSE of F_hat(T) split into what selection can and cannot move
# -----------------------------------------------------------------------------
def _window(phi):
    """The phi-window channel B carries: ||phi| - pi/2| < pi/4."""
    return np.abs(np.abs(phi) - PI / 2) < PI / 4


def decompose(recs):
    """(mse, bias2, var) globally and in the B window, from the seed ensemble."""
    F = np.array([r["pmf_t"][-1] for r in recs])
    F = F - F.mean(axis=1, keepdims=True)
    Fref = recs[0]["F_ref"]
    Fref = Fref - Fref.mean()
    m = _window(recs[0]["x_grid"])
    out = {}
    for tag, sel in (("all", np.ones_like(m)), ("B", m)):
        d = F[:, sel] - Fref[sel]
        out[tag] = (float((d ** 2).mean()), float((d.mean(axis=0) ** 2).mean()),
                    float(F[:, sel].var(axis=0).mean()))
    return out


def screen():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    seeds = list(range(900, 908))
    for K in (1024, 256, 64):
        cfgs, sds = [], []
        for hp, dl in ((1.5, 1.0), (2.0, 1.0), (1.5, 1.5), (2.0, 1.5)):
            for sd in seeds:
                cfgs.append(bc.BiChannelConfig(beta=4.0, Hperp=hp, Delta=dl,
                                               **{**COMMON, "K": K}))
                sds.append(sd)
        recs = bc.simulate_batch(cfgs, sds, [Method("shus")], batch_seed=31415,
                                 device=device)
        print(f"\nK={K}: {'cell':>14}{'eF/e*':>8}{'drift/sd':>10}{'var/MSE':>9}"
              f"{'var/MSE (B)':>13}")
        for hp, dl in ((1.5, 1.0), (2.0, 1.0), (1.5, 1.5), (2.0, 1.5)):
            rr = [r for r in recs if r["config"]["Hperp"] == hp
                  and r["config"]["Delta"] == dl]
            cfg = bc.BiChannelConfig(beta=4.0, Hperp=hp, Delta=dl,
                                     **{**COMMON, "K": K})
            e_star = bc.analytic_floors(cfg, "cpu")["e_star"]
            pBr = rr[0]["p_B_ref_biased"]
            drift = (np.median([r["final_p_B"] for r in rr]) - pBr)
            d = decompose(rr)
            print(f"      hp{hp:g}_d{dl:g}"
                  f"{np.median([r['final_l2_f'] for r in rr])/e_star:8.1f}"
                  f"{drift/math.sqrt(pBr*(1-pBr)/K):10.2f}"
                  f"{d['all'][2]/d['all'][0]:9.3f}{d['B'][2]/d['B'][0]:13.3f}")


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    cfgs, seeds = [], []
    for hp, dl in CELLS:
        for sd in SEEDS:
            cfgs.append(bc.BiChannelConfig(beta=4.0, Hperp=hp, Delta=dl, **COMMON))
            seeds.append(sd)
    print(f"J2: {len(cfgs)*len(ARMS)} rows, T={cfgs[0].T_total:.0f}", flush=True)
    t0 = time.time()
    recs = bc.simulate_batch(cfgs, seeds, ARMS, batch_seed=BATCH_SEED, device=device,
                             progress=200_000)
    print(f"wall {time.time()-t0:.0f}s")
    os.makedirs(OUT, exist_ok=True)
    for rec in recs:
        c = rec["config"]
        tag = (f"hp{c['Hperp']:g}_d{c['Delta']:g}_{rec['method']['name']}"
               f"_seed{rec['seed']}")
        save_run(os.path.join(OUT, tag), {k: rec[k] for k in SAVE_KEYS},
                 {k: rec[k] for k in META_KEYS})
    analyze(recs)


def _rows(recs, cell, arm):
    d = {r["seed"]: r for r in recs if r["config"]["Hperp"] == cell[0]
         and r["config"]["Delta"] == cell[1] and r["method"]["name"] == arm}
    return [d[s] for s in SEEDS]


def _v(recs, cell, arm, key):
    return np.array([r[key] for r in _rows(recs, cell, arm)])


def analyze(recs):
    print("\n" + "=" * 104)
    print("J2 OUTCOME — measure-preserving allocation on a VARIANCE-limited "
          "conditional (stationary start, warm-started bias)")
    print("=" * 104)
    for cell in CELLS:
        cfg = bc.BiChannelConfig(beta=4.0, Hperp=cell[0], Delta=cell[1], **COMMON)
        e_star = bc.analytic_floors(cfg, "cpu")["e_star"]
        tAB, tBA, _ = cfg.channel_times()
        base = _v(recs, cell, "shus", "int_l2_f")
        pBr = _rows(recs, cell, "shus")[0]["p_B_ref_biased"]
        sd_b = math.sqrt(pBr * (1 - pBr) / COMMON["K"])
        print(f"\n--- cell Hperp={cell[0]} Delta={cell[1]} | e*={e_star:.5f} | "
              f"tau_B->A={tBA:.0f} (T=800) | p_B_ref={pBr:.4f} (+-{sd_b:.4f} at "
              f"K={COMMON['K']}) ---")
        print(f"{'arm':>15}{'dI_F %':>20}{'eF/e*':>7}{'var(B)':>10}{'bias2(B)':>10}"
              f"{'MSE(B)':>10}{'P_B drift/sd':>13}{'ESSw':>7}{'turn':>7}")
        dec = {}
        for m in ARMS:
            rr = _rows(recs, cell, m.name)
            dec[m.name] = decompose(rr)
            IF = _v(recs, cell, m.name, "int_l2_f")
            d, lo, hi = paired_bootstrap_ci(100.0 * (IF - base) / base)
            drift = np.median(_v(recs, cell, m.name, "final_p_B")) - pBr
            print(f"{m.name:>15}{d:8.2f} [{lo:6.2f},{hi:6.2f}]"
                  f"{np.median(_v(recs, cell, m.name, 'final_l2_f'))/e_star:7.1f}"
                  f"{dec[m.name]['B'][2]:10.2e}{dec[m.name]['B'][1]:10.2e}"
                  f"{dec[m.name]['B'][0]:10.2e}{drift/sd_b:13.2f}"
                  f"{np.median(_v(recs, cell, m.name, 'min_ess_w')):7.3f}"
                  f"{np.median(_v(recs, cell, m.name, 'total_turnover')):7.0f}")
        print("  P13 (primary) — B-window seed VARIANCE vs matched sham, and MSE:")
        for a, sh in PAIRS:
            va, vs = dec[a]["B"][2], dec[sh]["B"][2]
            ma, ms = dec[a]["B"][0], dec[sh]["B"][0]
            print(f"      {a:>15} vs {sh:>12}: var {100*(va/vs-1):+7.1f}%   "
                  f"MSE {100*(ma/ms-1):+7.1f}%")
        print("  global (all phi):")
        for a, sh in PAIRS:
            va, vs = dec[a]["all"][2], dec[sh]["all"][2]
            print(f"      {a:>15} vs {sh:>12}: var {100*(va/vs-1):+7.1f}%")


if __name__ == "__main__":
    if "--scan" in sys.argv:
        scan()
    elif "--screen" in sys.argv:
        screen()
    else:
        main()
