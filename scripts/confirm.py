"""Stage 2: frozen-hyperparameter confirmation on FRESH seeds, all systems.

Hyper-parameters come from the Stage-1 screens recorded in docs/RESULTS_LOG.md and
are not touched here.  Saves full e_F(fe) curves so every endpoint can be rescored
without re-simulating.
"""
from __future__ import annotations
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, torch

from rcwfr.campaign import (estimator_floor, paired_bootstrap, rel_change, run_arm,
                            save_json, save_npz, score)
from rcwfr.engines import RunConfig
from rcwfr.registry import build, torsion

def frozen(a):
    """Stage-1 winners, frozen.  Each entry: label, arm, overrides."""
    K, TH = a.kappa, a.theta
    FK, FT, FJ = a.flow_kappa, a.flow_theta, a.flow_jitter
    return [
    ("wfr",         "wfr",       dict(kappa=K, theta=TH, n_cond=5, lift="identity")),
    ("wfr_anneal",  "wfr",       dict(kappa=0.5, kappa_end=0.003, acc_reset_at=0.3,
                                      theta=TH, n_cond=5, lift="identity")),
    ("wfr_flow",    "wfr_flow",  dict(kappa=FK, theta=FT, n_cond=5, lift="identity",
                                      fr_jitter=FJ)),
    ("wfr_flow_w",  "wfr_flow",  dict(kappa=FK, theta=0.0, n_cond=5, lift="identity",
                                      fr_jitter=FJ)),
    ("wfr_flow_cnt","wfr",       dict(kappa=FK, theta=FT, n_cond=5, lift="identity",
                                      w_mode="flow", fr_rule="count", fr_jitter=FJ)),
    ("wfr_scaled",  "wfr",       dict(kappa=K, theta=TH, n_cond=5, lift="scaled")),
    ("wfr_oracle",  "wfr_oracle", dict(kappa=0.5, theta=TH, n_cond=5)),
    ("w_only",      "w_only",    dict(kappa=K, n_cond=5)),
    ("fr_only",     "fr_only",   dict(theta=TH, n_cond=5)),
    ("w_count",     "w_count",   dict(kappa=K, theta=TH, n_cond=5)),
    ("w_sham",      "w_sham",    dict(kappa=K, n_cond=5)),
    ("ti_cold",     "ti_cold",   dict(n_cond=5)),
    ("ti_warm",     "ti_warm",   dict(n_cond=5)),
    ("reti_cold",   "reti_cold", dict(n_ex=a.reti_nex, n_windows=a.reti_M)),
    ("reti_warm",   "reti_warm", dict(n_ex=a.reti_nex, n_windows=a.reti_M)),
    ("abf",         "abf",       dict(bias_n_min=1.0)),
    ("shus",        "shus",      dict(shus_gain=1000.0)),
    ("unbiased",    "unbiased",  {}),
]



ap = argparse.ArgumentParser()
ap.add_argument("--system", default="EB")
ap.add_argument("--N", type=int, default=256)
ap.add_argument("--steps", type=int, default=100_000)
ap.add_argument("--seeds", type=int, default=32)
ap.add_argument("--seed", type=int, default=9000)
ap.add_argument("--out", default="results/confirm")
# calibrated per-system RC-WFR settings (Stage-1 screens, docs/RESULTS_LOG.md)
ap.add_argument("--kappa", type=float, default=0.125)
ap.add_argument("--theta", type=float, default=0.6)
ap.add_argument("--flow_kappa", type=float, default=2.0)
ap.add_argument("--flow_theta", type=float, default=0.3)
ap.add_argument("--flow_jitter", type=float, default=0.01)
ap.add_argument("--reti_M", type=int, default=256)
ap.add_argument("--reti_nex", type=int, default=5)
a = ap.parse_args()

S = torsion(float(a.system[len("TORSION_L"):])) if a.system.startswith("TORSION_L") \
    else build(a.system)
base = RunConfig(N=a.N, n_seed=a.seeds, n_steps=a.steps,
                 save_every=max(500, a.steps // 100), bw_mf=0.02, n_min=1.0,
                 bw_kde=0.10, n_bins_count=45, x0=-1.0, ess_window=40,
                 # every arm shares this small start jitter so the comparison stays
                 # paired; the deterministic 'flow' W step has zero velocity at a delta
                 # ensemble, so a degenerate start would silently disable that arm.
                 x0_jitter=0.05)
fl = float(estimator_floor(S, base, [2 ** 23], rows=4)[2 ** 23].mean())
print(f"=== CONFIRM {a.system}  N={a.N} steps={a.steps} fe={a.N*a.steps:.3g} "
      f"seeds={a.seeds} floor={fl:.5f} ===", flush=True)

res, curves = {}, {}
for label, arm, ov in frozen(a):
    # a sham arm needs its FR partner's per-event turnover: run wfr first and reuse it
    sham_src = res.get("_turnover") if arm == "w_sham" else None
    t0 = time.time()
    run, _ = run_arm(S, arm, base, a.seeds, a.seed, overrides=ov, sham_source=sham_src)
    if label == "wfr" and run.get("turnover") is not None:
        res["_turnover"] = run["turnover"]
    sc = score(run, S); sc["wall"] = time.time() - t0
    if "ex_accept" in run:
        sc["ex_accept"] = run["ex_accept"]
    curves[label] = sc["e_F"]
    res[label] = sc
    ex = f" acc={run['ex_accept']:.3f}" if "ex_accept" in run else ""
    print(f"  {label:12s} I_F={np.median(sc['I_F']):.5f} e_F={np.median(sc['e_F_final']):.5f} "
          f"(/fl {np.median(sc['e_F_final'])/fl:6.1f}) chan={np.median(sc['chan'][-1]):.4f} "
          f"cov={np.median(sc['cov'][-1]):.3f} ess={np.median(sc['ess_anc'][-1]):.2f} "
          f"[{sc['wall']:.0f}s]{ex}", flush=True)
    del run; torch.cuda.empty_cache()
res.pop("_turnover", None)

fe = res["wfr"]["fe"]
save_npz(os.path.join(a.out, f"{a.system}_curves.npz"), floor=np.array(fl), fe=fe,
         **{f"eF__{k}": v for k, v in curves.items()})
save_json(os.path.join(a.out, f"{a.system}.json"),
          {"floor": fl, "arms": {k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                                     for kk, vv in v.items() if kk != "e_F"}
                                 for k, v in res.items()}})

print("\npaired median relative change in I_F (negative = row better than column)")
cols = ["ti_cold", "reti_cold", "abf"]
print(f"{'arm':12s} " + " ".join(f"{c:>26s}" for c in cols))
for lab in res:
    cells = []
    for c in cols:
        m, lo, hi = paired_bootstrap(rel_change(res[lab]["I_F"], res[c]["I_F"]))
        cells.append(f"{100*m:+7.1f}% [{100*lo:+6.1f},{100*hi:+6.1f}]")
    print(f"{lab:12s} " + " ".join(f"{c:>26s}" for c in cells))
