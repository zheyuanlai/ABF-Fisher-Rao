"""Phase F3b: where does the augmented CV start to cost?

Frozen in docs/PREREGISTRATION_APPLICATION_MAP.md before this run.  A 96x96
accumulator needs walkers to fill it; conditional reallocation needs only enough
walkers per stratum.  n_strata = max(4, K // 32) is frozen (holds ~32 walkers per
stratum), which makes the conditional method K-dependent -- disclosed, not hidden.
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from abpfr.metrics import paired_bootstrap_ci
from abpfr.systems import bichannel as bc
from abpfr.systems.gateway import Method

CELL = (2.0, 0.0)
SEEDS = list(range(700, 708))
KS = (64, 256)
GAINS = (1.0, 2.0, 4.0, 8.0)
BASE = dict(dt=1e-3, n_steps=800_000, block=20, eps_bw=0.06, eta_bw=0.25,
            n_saves=400, profile_every=8, joint_every=80, ess_window_steps=4000)
WIN = dict(theta=0.01, t_on_frac=20.0 / 800.0, t_off_frac=215.0 / 800.0,
           alpha_ess=0.5, fr_every_blocks=49)
ARMS_1D = [Method("shus"), Method("fr_cond", use_fr=True, cond_fr=True, **WIN),
           Method("cnt_cond", use_fr=True, cond_fr=True, cond_bins1=32,
                  cond_bins2=9, **WIN),
           Method("sham_cond", sham=True, shadows="fr_cond", **WIN)]
ARMS_2D = [Method(f"aug_g{g:g}", g_shus=g) for g in GAINS]


def run(K, cv, arms, device):
    n_str = max(4, K // 32)
    cfgs = [bc.BiChannelConfig(beta=4.0, Hperp=CELL[0], Delta=CELL[1], K=K,
                               cv=cv, n_strata=n_str, **BASE) for _ in SEEDS]
    t0 = time.time()
    recs = bc.simulate_batch(cfgs, SEEDS, arms, batch_seed=20260970 + K,
                             device=device, progress=200_000)
    print(f"  K={K} cv={cv}: {len(cfgs)*len(arms)} rows, n_strata={n_str}, "
          f"wall {time.time()-t0:.0f}s")
    return recs


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    out = {}
    for K in KS:
        out[(K, "phi")] = run(K, "phi", ARMS_1D, device)
        out[(K, "phipsi")] = run(K, "phipsi", ARMS_2D, device)
    print("\n" + "=" * 92)
    print("F3b OUTCOME — sample-complexity ladder on the anchor cell "
          "(P6: the augmented CV's margin shrinks as K falls)")
    print("=" * 92)
    cfg1 = bc.BiChannelConfig(beta=4.0, Hperp=CELL[0], Delta=CELL[1], K=1024, **BASE)
    e_star = bc.analytic_floors(cfg1, "cpu")["e_star"]
    print(f"e* = {e_star:.5f} (identical for both CV choices)")
    for K in KS:
        r1, r2 = out[(K, "phi")], out[(K, "phipsi")]

        def v(recs, arm, key):
            d = {r["seed"]: r for r in recs if r["method"]["name"] == arm}
            return np.array([d[s][key] for s in SEEDS])
        base = v(r1, "shus", "int_l2_f")
        print(f"\n--- K = {K} (n_strata = {max(4, K//32)}) | 1D baseline "
              f"I_F={np.median(base):.2f} e_F(T)={np.median(v(r1,'shus','final_l2_f')):.4f} ---")
        print(f"{'arm':>11} {'dI_F % vs 1D shus':>21} {'e_F(T)':>8} {'/e*':>7}")
        cols = {}
        for recs, names in ((r1, [m.name for m in ARMS_1D]),
                            (r2, [m.name for m in ARMS_2D])):
            for a in names:
                IF, eF = v(recs, a, "int_l2_f"), v(recs, a, "final_l2_f")
                m, lo, hi = paired_bootstrap_ci(100.0 * (IF - base) / base)
                print(f"{a:>11} {m:9.2f} [{lo:6.2f},{hi:6.2f}] {np.median(eF):8.4f} "
                      f"{np.median(eF)/e_star:7.1f}")
                cols[a] = IF
        eT1 = np.median(v(r2, "aug_g1", "final_l2_f"))
        qual = [g for g in GAINS
                if np.median(v(r2, f"aug_g{g:g}", "final_l2_f")) / eT1 <= 1.05]
        gs = min(qual, key=lambda g: np.median(cols[f"aug_g{g:g}"]))
        m, lo, hi = paired_bootstrap_ci(
            100.0 * (cols[f"aug_g{gs:g}"] - cols["fr_cond"]) / cols["fr_cond"])
        print(f"  g*={gs:g};  aug_g{gs:g} vs fr_cond: {m:7.2f}% [{lo:6.2f}, {hi:6.2f}]"
              f"  => {'augmented CV wins' if hi < 0 else 'conditional wins' if lo > 0 else 'TIE'}")


if __name__ == "__main__":
    main()
