"""P1: does RC-WFR's advantage over ABF grow with the CV domain length L?

The local landscape is identical for every L (periodic wells at fixed spacing), so
L is a pure transport-distance knob.  Every arm gets the SAME total force budget
`budget = N * n_steps`; N itself is an arm knob (a stratified arm may prefer many
short windows, an adaptive arm may prefer few long walkers), and n_steps is derived.
"""
from __future__ import annotations
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, torch

from rcwfr.campaign import estimator_floor, paired_bootstrap, rel_change, run_arm, save_json, score
from rcwfr.engines import RunConfig
from rcwfr.registry import torsion

ap = argparse.ArgumentParser()
ap.add_argument("--Ls", type=float, nargs="*", default=[3.0, 6.0, 12.0, 24.0])
ap.add_argument("--budget", type=float, default=2.56e7)
ap.add_argument("--seeds", type=int, default=8)
ap.add_argument("--seed", type=int, default=4000)
ap.add_argument("--kappa", type=float, default=0.125)
ap.add_argument("--theta", type=float, default=0.6)
ap.add_argument("--n_cond", type=int, default=5)
ap.add_argument("--H", type=float, default=1.0)
ap.add_argument("--bw_mf", type=float, default=0.02)
ap.add_argument("--Ns", type=int, nargs="*", default=None)
ap.add_argument("--tag", default="")
ap.add_argument("--flow", action="store_true",
                help="use the calibrated probability-flow RC-WFR arm")
ap.add_argument("--flow_jitter", type=float, default=0.01)
ap.add_argument("--out", default="results/torsion")
a = ap.parse_args()

# (label, arm, N, extra overrides).  N is the arm's own knob at fixed total budget.
ENTRIES = [
    ("wfr_N256",     "wfr",       256,  {}),
    ("wfr_N1024",    "wfr",       1024, {}),
    ("w_only_N256",  "w_only",    256,  {}),
    ("ti_cold_N256", "ti_cold",   256,  {}),
    ("ti_cold_N1024","ti_cold",   1024, {}),
    ("ti_warm_N1024","ti_warm",   1024, {}),
    ("reti_cold_N1024","reti_cold",1024,{"n_ex": 5}),
    ("abf_N256",     "abf",       256,  {}),
    ("abf_N1024",    "abf",       1024, {}),
]
if a.Ns:                       # let every family choose its own replica count at
    ENTRIES = []               # FIXED total budget: many short windows vs few long ones
    _wov = dict(w_mode="flow", fr_jitter=a.flow_jitter) if a.flow else {}
    for _n in a.Ns:
        ENTRIES += [(f"wfr_N{_n}", "wfr", _n, dict(_wov)),
                    (f"ti_cold_N{_n}", "ti_cold", _n, {}),
                    (f"reti_cold_N{_n}", "reti_cold", _n, {"n_ex": 5}),
                    (f"abf_N{_n}", "abf", _n, {})]

allres = {}
for L in a.Ls:
    S = torsion(L, H=a.H)
    n_steps0 = int(a.budget / 256)
    probe = RunConfig(N=256, n_steps=n_steps0, bw_mf=a.bw_mf, n_min=1.0)
    fl = float(estimator_floor(S, probe, [2 ** 22], rows=4)[2 ** 22].mean())
    print(f"\n=== TORSION L={L}  wells={S.p.n_wells}  budget={a.budget:.3g} "
          f"floor={fl:.5f}  beta*dF={S.p.beta*float(S.F_ref.max()-S.F_ref.min()):.1f} ===",
          flush=True)
    res = {}
    for label, arm, N, ov in ENTRIES:
        n_steps = int(a.budget / N)
        cfg = RunConfig(N=N, n_seed=a.seeds, n_steps=n_steps,
                        save_every=max(200, n_steps // 60), bw_mf=a.bw_mf, n_min=1.0,
                        bw_kde=max(0.10, L / 60), n_bins_count=45, x0=-L / 4,
                        kappa=a.kappa, theta=a.theta, n_cond=a.n_cond, ess_window=40)
        t0 = time.time()
        run, _ = run_arm(S, arm, cfg, a.seeds, a.seed, overrides=ov)
        sc = score(run, S); sc["wall"] = time.time() - t0
        res[label] = sc
        print(f"  {label:16s} N={N:5d} steps={n_steps:7d} "
              f"I_F={np.median(sc['I_F']):.5f} e_F={np.median(sc['e_F_final']):.5f} "
              f"(/fl {np.median(sc['e_F_final'])/fl:5.1f}) cov={np.median(sc['cov'][-1]):.3f} "
              f"[{sc['wall']:.0f}s]", flush=True)
        del run; torch.cuda.empty_cache()
    best = lambda pre: min([k for k in res if k.startswith(pre)],
                           key=lambda k: np.median(res[k]["I_F"]))
    bw, ba, bt = best("wfr"), best("abf"), best("ti_cold")
    d1 = rel_change(res[bw]["I_F"], res[ba]["I_F"])
    d2 = rel_change(res[bw]["I_F"], res[bt]["I_F"])
    m1, lo1, hi1 = paired_bootstrap(d1); m2, lo2, hi2 = paired_bootstrap(d2)
    print(f"  --> best wfr={bw} vs best abf={ba}: {100*m1:+.1f}% [{100*lo1:+.1f},{100*hi1:+.1f}]")
    print(f"  --> best wfr={bw} vs best ti_cold={bt}: {100*m2:+.1f}% [{100*lo2:+.1f},{100*hi2:+.1f}]",
          flush=True)
    allres[L] = {"floor": fl, "n_wells": S.p.n_wells,
                 "arms": {k: {"I_F": v["I_F"].tolist(),
                              "e_F_final": v["e_F_final"].tolist(),
                              "cov": v["cov"][-1].tolist(),
                              "chan": v["chan"][-1].tolist()} for k, v in res.items()},
                 "wfr_vs_abf": [m1, lo1, hi1, bw, ba],
                 "wfr_vs_ti": [m2, lo2, hi2, bw, bt]}
save_json(os.path.join(a.out, f"torsion_scaling{a.tag}.json"), allres)
print("\nSUMMARY  (paired median rel change in I_F, negative = RC-WFR better)")
print(f"{'L':>6} {'wells':>6} | {'vs ABF':>26} | {'vs fixed TI':>26}")
for L, r in allres.items():
    m1, lo1, hi1, *_ = r["wfr_vs_abf"]; m2, lo2, hi2, *_ = r["wfr_vs_ti"]
    print(f"{L:>6} {r['n_wells']:>6} | {100*m1:+7.1f}% [{100*lo1:+6.1f},{100*hi1:+6.1f}] "
          f"| {100*m2:+7.1f}% [{100*lo2:+6.1f},{100*hi2:+6.1f}]")
