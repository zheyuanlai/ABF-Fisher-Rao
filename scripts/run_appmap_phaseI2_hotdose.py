"""Phase I2: is the hot-dose weighted lead variance reduction, or the residual
O(1/walkers-per-stratum) bias of the weight bookkeeping?

Frozen in docs/PREREGISTRATION_APPLICATION_MAP.md before this run.  I1 found that
weighted selection removes essentially all of conditional reallocation's benefit --
and its target sensitivity with it -- except at ten times the dose, where the arm
allocating hard toward the ORACLE conditional gained -28% on the anchor cell.  The
weight rule is a ratio estimator, so it carries a small bias toward the target that
scales as 1/(walkers per stratum) and points the same way as that gain.  This script
measures the bias directly (no dynamics), then runs the same five arms at 32 and at
128 walkers per stratum with matched-turnover weighted shams.
"""
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from abpfr.fisher_rao_cond import (conditional_log_ratio,
                                   stratified_systematic_resample, stratum_of,
                                   child_weights, theta_backoff_cond, weight_ess)
from abpfr.grid2d import GridT2, binned_density2, periodic_gaussian_kernel
from abpfr.io import save_run
from abpfr.metrics import paired_bootstrap_ci
from abpfr.systems import bichannel as bc
from abpfr.systems.gateway import Method

PI = math.pi
CELL = (2.0, 0.0)
SEEDS = list(range(800, 816))
BATCH_SEED = 20261010
STRATA = (32, 8)                       # 32 and 128 walkers per stratum at K = 1024
COMMON = dict(K=1024, dt=1e-3, n_steps=800_000, block=20, eps_bw=0.06, eta_bw=0.25,
              n_saves=400, profile_every=8, joint_every=40, ess_window_steps=4000)
HOT = dict(theta=0.1, t_on_frac=20.0 / 800.0, t_off_frac=215.0 / 800.0,
           alpha_ess=0.5, fr_every_blocks=49)
WT = dict(cond_weighted=True)
ARMS = [
    Method("shus"),
    Method("wfr_cond_hot", use_fr=True, cond_fr=True, **WT, **HOT),
    Method("wfr_cond_hot_oracle", use_fr=True, cond_fr=True, cond_target="oracle",
           **WT, **HOT),
    Method("wsham_hot", sham=True, shadows="wfr_cond_hot", **WT, **HOT),
    Method("wsham_hot_oracle", sham=True, shadows="wfr_cond_hot_oracle", **WT, **HOT),
]
OUT = "results/appmap_phaseI2_hotdose"
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
# the bias measurement: the same weight rule, driven by the same score, NO dynamics
# -----------------------------------------------------------------------------
def static_drift(n_strata, theta, n_events=200, K=1024, R=8, p_rare=0.15, seed=11,
                 device="cpu", dtype=torch.float64):
    """Movement of the REPRESENTED fiber population under repeated events alone.

    With the population frozen between events, every change in the weighted fiber
    fraction is bookkeeping: the importance rule is unbiased in expectation, but the
    per-stratum renormalization that holds the xi-marginal exactly is a ratio
    estimator and is biased at O(1/cnt_j).  This is the size the I1 hot-dose lead has
    to stand clear of, measured on the configuration that produced it.
    """
    G2 = bc.GRID2
    g = torch.Generator(device=device).manual_seed(seed)
    z1 = (torch.rand(R, K, generator=g, device=device, dtype=dtype) * 2 - 1) * PI
    rare = torch.rand(R, K, generator=g, device=device) < p_rare
    z2 = torch.where(rare, torch.full((R, K), PI, device=device, dtype=dtype),
                     torch.zeros(R, K, device=device, dtype=dtype))
    z2 = torch.remainder(z2 + 0.3 * torch.randn(R, K, generator=g, device=device,
                                                dtype=dtype) + PI, 2 * PI) - PI
    k1, r1 = periodic_gaussian_kernel(0.25, G2.dx1, G2.n1, device, dtype)
    W = torch.ones(R, K, device=device, dtype=dtype)
    frac = lambda z, w: ((z.abs() > PI / 2).to(dtype) * w).sum(1) / w.sum(1)
    before = frac(z2, W)
    gen = torch.Generator(device=device).manual_seed(12)
    for _ in range(n_events):
        st = stratum_of(z1, G2, n_strata)
        p2 = binned_density2(z1, z2, k1, r1, k1, r1, G2)
        lr = conditional_log_ratio(z1, z2, p2, G2)
        w, cnt, _, _ = theta_backoff_cond(lr, st, n_strata,
                                          torch.full((R,), theta, device=device,
                                                     dtype=dtype), 0.5)
        sel = stratified_systematic_resample(w, st, cnt, n_strata, gen)
        W = child_weights(W, sel, st, n_strata, w, cnt)
        z1, z2 = torch.gather(z1, 1, sel), torch.gather(z2, 1, sel)
    return dict(before=float(before.mean()), after=float(frac(z2, W).mean()),
                particle=float((z2.abs() > PI / 2).to(dtype).mean()),
                ess_w=float(weight_ess(W).mean()),
                w_sum=float(W.sum(1).mean()) / K)


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("static bias of the weight bookkeeping (no dynamics, 200 events):")
    for S in STRATA:
        for th in (0.01, 0.1):
            d = static_drift(S, th)
            print(f"  n_strata={S:3d} ({COMMON['K']//S:4d} walkers/stratum) "
                  f"theta={th:4}: represented {d['before']:.4f} -> {d['after']:.4f} "
                  f"(drift {d['after']-d['before']:+.4f}) | particles "
                  f"{d['particle']:.4f} | ess_w {d['ess_w']:.3f} "
                  f"| sum W/K {d['w_sum']:.9f}", flush=True)
    all_recs = {}
    for S in STRATA:
        cfgs = [bc.BiChannelConfig(beta=4.0, Hperp=CELL[0], Delta=CELL[1],
                                   n_strata=S, **COMMON) for _ in SEEDS]
        print(f"\nI2: n_strata={S}, {len(cfgs)*len(ARMS)} rows", flush=True)
        t0 = time.time()
        recs = bc.simulate_batch(cfgs, SEEDS, ARMS, batch_seed=BATCH_SEED,
                                 device=device, progress=200_000)
        print(f"wall {time.time()-t0:.0f}s")
        os.makedirs(OUT, exist_ok=True)
        for rec in recs:
            tag = (f"S{S}_{rec['method']['name']}_seed{rec['seed']}")
            save_run(os.path.join(OUT, tag), {k: rec[k] for k in SAVE_KEYS},
                     {k: rec[k] for k in META_KEYS})
        all_recs[S] = recs
    analyze(all_recs)


def _v(recs, arm, key):
    d = {r["seed"]: r for r in recs if r["method"]["name"] == arm}
    return np.array([d[s][key] for s in SEEDS])


def analyze(all_recs):
    print("\n" + "=" * 100)
    print("I2 OUTCOME — hot-dose weighted selection at 32 vs 128 walkers per stratum")
    print("=" * 100)
    for S, recs in all_recs.items():
        base = _v(recs, "shus", "int_l2_f")
        pB0 = np.median(_v(recs, "shus", "final_p_B"))
        print(f"\n--- n_strata={S} ({COMMON['K']//S} walkers/stratum) | "
              f"baseline I_F={np.median(base):.2f} | plain-SHUS P_B={pB0:.4f} ---")
        print(f"{'arm':>21} {'dI_F % vs shus':>22} {'P_B':>7} {'dP_B':>8} "
              f"{'P_B^n':>7} {'min ESSw':>9} {'KL(p|u)':>9} {'turnover':>9}")
        for m in ARMS:
            IF = _v(recs, m.name, "int_l2_f")
            d, lo, hi = paired_bootstrap_ci(100.0 * (IF - base) / base)
            pB = np.median(_v(recs, m.name, "final_p_B"))
            print(f"{m.name:>21} {d:9.2f} [{lo:7.2f},{hi:7.2f}] {pB:7.4f} "
                  f"{pB-pB0:+8.4f} "
                  f"{np.median(_v(recs, m.name, 'final_p_B_n')):7.4f} "
                  f"{np.median(_v(recs, m.name, 'min_ess_w')):9.4f} "
                  f"{np.median(_v(recs, m.name, 'kl_u_t')[:, -1]):9.5f} "
                  f"{np.median(_v(recs, m.name, 'total_turnover')):9.0f}")
        for a, sh in (("wfr_cond_hot", "wsham_hot"),
                      ("wfr_cond_hot_oracle", "wsham_hot_oracle")):
            va, vb = _v(recs, a, "int_l2_f"), _v(recs, sh, "int_l2_f")
            d, lo, hi = paired_bootstrap_ci(100.0 * (va - vb) / vb)
            print(f"   P12  {a:>21} vs {sh:>17}: {d:7.2f}% [{lo:7.2f}, {hi:7.2f}]")


if __name__ == "__main__":
    main()
