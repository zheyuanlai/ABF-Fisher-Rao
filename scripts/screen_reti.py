"""Screen the RE-TI baseline's own knobs (window count, exchange period) on a system.

The baselines must be tuned at least as hard as the new arm; RE-TI's window-space
mobility is set by (window spacing)^2 x (exchange rate) x acceptance, and window
spacing trades against CV resolution, so this is its real hyper-parameter.
"""
from __future__ import annotations
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, torch
from rcwfr.campaign import estimator_floor, run_arm, save_json, score
from rcwfr.engines import RunConfig
from rcwfr.registry import build, torsion

ap = argparse.ArgumentParser()
ap.add_argument("--system", default="CHANNEL")
ap.add_argument("--N", type=int, default=256)
ap.add_argument("--steps", type=int, default=100_000)
ap.add_argument("--seeds", type=int, default=8)
ap.add_argument("--seed", type=int, default=3000)
ap.add_argument("--Ms", type=int, nargs="*", default=[256, 128, 64, 32])
ap.add_argument("--nexs", type=int, nargs="*", default=[1, 5, 20])
ap.add_argument("--arms", nargs="*", default=["reti_cold"])
a = ap.parse_args()
S = torsion(float(a.system[len("TORSION_L"):])) if a.system.startswith("TORSION_L") else build(a.system)
base = RunConfig(N=a.N, n_seed=a.seeds, n_steps=a.steps, save_every=max(500, a.steps//60),
                 bw_mf=0.02, n_min=1.0, bw_kde=0.10, x0=-1.0, x0_jitter=0.05)
fl = float(estimator_floor(S, base, [2**22], rows=4)[2**22].mean())
print(f"=== RE-TI screen {a.system}  fe={a.N*a.steps:.3g} floor={fl:.5f} ===", flush=True)
rec = []
for arm in a.arms:
    for M in a.Ms:
        for nex in a.nexs:
            t0 = time.time()
            run, _ = run_arm(S, arm, base, a.seeds, a.seed,
                             overrides=dict(n_windows=M, n_ex=nex))
            sc = score(run, S)
            row = dict(arm=arm, M=M, n_ex=nex, I_F=float(np.median(sc["I_F"])),
                       e_F=float(np.median(sc["e_F_final"])),
                       chan=float(np.median(sc["chan"][-1])),
                       acc=run.get("ex_accept"), fe=float(sc["fe"][-1]))
            rec.append(row)
            print(f"  {arm:10s} M={M:4d} n_ex={nex:3d}: I_F={row['I_F']:.5f} "
                  f"e_F={row['e_F']:.5f} (/fl {row['e_F']/fl:5.1f}) chan={row['chan']:.4f} "
                  f"acc={row['acc']:.3f} fe={row['fe']:.3g} [{time.time()-t0:.0f}s]", flush=True)
            del run; torch.cuda.empty_cache()
save_json(f"results/sweeps/{a.system}_reti_screen.json", {"floor": fl, "rows": rec})
best = min(rec, key=lambda r: r["I_F"])
print("\nBEST", best)
