"""P2: does RC-WFR overtake RE-TI as the fiber (system size) grows?

Hamiltonian-exchange acceptance decays as the fiber measure changes more between
neighbouring windows; RC-WFR's unconditional move never rejects.  If RC-WFR has a
regime at all, this is where it must appear.  Spectator dofs are added to the
CHANNEL fiber with an x-dependent stiffness, so they both (i) enlarge the RE energy
gap and (ii) enlarge the free energy - errors are therefore ALSO reported relative
to the RMS of F_ref so the axis stays comparable across m_spec.
"""
from __future__ import annotations
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, torch

from rcwfr.campaign import (estimator_floor, paired_bootstrap, rel_change, run_arm,
                            save_json, score)
from rcwfr.engines import RunConfig
from rcwfr.registry import build

ap = argparse.ArgumentParser()
ap.add_argument("--system", default="CHANNEL")
ap.add_argument("--ms", type=int, nargs="*", default=[0, 32, 128, 512])
ap.add_argument("--N", type=int, default=256)
ap.add_argument("--steps", type=int, default=60_000)
ap.add_argument("--seeds", type=int, default=8)
ap.add_argument("--seed", type=int, default=5000)
ap.add_argument("--kappa", type=float, default=0.125)
ap.add_argument("--theta", type=float, default=0.6)
ap.add_argument("--oms_out", type=float, default=0.6)
ap.add_argument("--oms_in", type=float, default=2.4)
ap.add_argument("--out", default="results/mspec")
a = ap.parse_args()

ENTRIES = [("wfr", "wfr", {}), ("ti_cold", "ti_cold", {}),
           ("reti_cold_M256", "reti_cold", {"n_ex": 5, "n_windows": 256}),
           ("reti_cold_M64", "reti_cold", {"n_ex": 5, "n_windows": 64}),
           ("abf", "abf", {})]
allres = {}
for m in a.ms:
    S = build(a.system, m_spec=m, oms_out=a.oms_out, oms_in=a.oms_in)
    base = RunConfig(N=a.N, n_seed=a.seeds, n_steps=a.steps,
                     save_every=max(500, a.steps // 60), bw_mf=0.02, n_min=1.0,
                     bw_kde=0.10, x0=-1.0, kappa=a.kappa, theta=a.theta, n_cond=5)
    fl = float(estimator_floor(S, base, [2 ** 22], rows=4)[2 ** 22].mean())
    Frms = float(torch.sqrt(((S.F_ref[0][S.grid.eval_mask()] -
                              S.F_ref[0][S.grid.eval_mask()].mean()) ** 2).mean()))
    print(f"\n=== m_spec={m}  floor={fl:.5f}  |F|_rms={Frms:.3f} ===", flush=True)
    res = {}
    for label, arm, ov in ENTRIES:
        t0 = time.time()
        run, _ = run_arm(S, arm, base, a.seeds, a.seed, overrides=ov)
        sc = score(run, S); sc["wall"] = time.time() - t0
        if "ex_accept" in run:
            sc["ex_accept"] = run["ex_accept"]
        res[label] = sc
        ex = f" acc={run['ex_accept']:.3f}" if "ex_accept" in run else ""
        print(f"  {label:16s} I_F={np.median(sc['I_F']):.5f} "
              f"I_F_rel={np.median(sc['I_F_rel']):.5f} "
              f"e_F={np.median(sc['e_F_final']):.5f} (/fl {np.median(sc['e_F_final'])/fl:5.1f}) "
              f"chan={np.median(sc['chan'][-1]):.4f} [{sc['wall']:.0f}s]{ex}", flush=True)
        del run; torch.cuda.empty_cache()
    out = {}
    for b in ("reti_cold_M256", "reti_cold_M64", "ti_cold", "abf"):
        d = rel_change(res["wfr"]["I_F"], res[b]["I_F"])
        out[f"wfr_vs_{b}"] = list(paired_bootstrap(d))
        mm, lo, hi = out[f"wfr_vs_{b}"]
        print(f"  wfr vs {b:16s}: {100*mm:+7.1f}% [{100*lo:+6.1f},{100*hi:+6.1f}]", flush=True)
    allres[m] = {"floor": fl, "F_rms": Frms, "cmp": out,
                 "arms": {k: {"I_F": v["I_F"].tolist(),
                              "I_F_rel": v["I_F_rel"].tolist(),
                              "e_F_final": v["e_F_final"].tolist(),
                              "chan": v["chan"][-1].tolist(),
                              "ex_accept": v.get("ex_accept")} for k, v in res.items()}}
save_json(os.path.join(a.out, f"{a.system}_mspec.json"), allres)
print("\nSUMMARY: RE acceptance and wfr-vs-RE-TI as the fiber grows")
print(f"{'m_spec':>7} {'acc M256':>9} {'acc M64':>8} | {'wfr vs reti(M256)':>22} {'wfr chan':>9} {'reti chan':>10}")
for m, r in allres.items():
    a256 = r["arms"]["reti_cold_M256"]["ex_accept"]; a64 = r["arms"]["reti_cold_M64"]["ex_accept"]
    mm, lo, hi = r["cmp"]["wfr_vs_reti_cold_M256"]
    print(f"{m:>7} {a256:>9.3f} {a64:>8.3f} | {100*mm:+8.1f}% [{100*lo:+6.1f},{100*hi:+6.1f}] "
          f"{np.median(r['arms']['wfr']['chan']):>9.4f} "
          f"{np.median(r['arms']['reti_cold_M256']['chan']):>10.4f}")
