"""Row-batched RC-WFR hyper-parameter sweep on one system (steelman screen)."""
from __future__ import annotations
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, torch

from rcwfr.campaign import estimator_floor, score, save_json
from rcwfr.engines import RunConfig, run_wfr
from rcwfr.registry import build
from rcwfr.rowspec import expand_grid, row_column
from rcwfr.grid import DEVICE, DTYPE

ap = argparse.ArgumentParser()
ap.add_argument("--system", default="CHANNEL")
ap.add_argument("--steps", type=int, default=100_000)
ap.add_argument("--N", type=int, default=256)
ap.add_argument("--seeds", type=int, default=8)
ap.add_argument("--seed", type=int, default=3000)
ap.add_argument("--m_spec", type=int, default=None)
ap.add_argument("--kappas", type=float, nargs="*", default=[0.03, 0.125, 0.5, 2.0, 8.0])
ap.add_argument("--thetas", type=float, nargs="*", default=[0.0, 0.3, 0.6])
ap.add_argument("--nconds", type=int, nargs="*", default=[5, 25])
ap.add_argument("--anneal", type=float, nargs="*", default=[None])
ap.add_argument("--reset", type=float, nargs="*", default=[None])
ap.add_argument("--lift", nargs="*", default=["identity"])
ap.add_argument("--w_mode", nargs="*", default=["sde"])
ap.add_argument("--bw_kde", type=float, nargs="*", default=[0.10])
ap.add_argument("--jitter", type=float, default=0.0)
ap.add_argument("--fr_jitter", type=float, nargs="*", default=[0.0])
ap.add_argument("--tag", default="")
a = ap.parse_args()

kw = {} if a.m_spec is None else {"m_spec": a.m_spec}
S = build(a.system, **kw)
base = RunConfig(N=a.N, n_steps=a.steps, dt=1e-3, save_every=max(500, a.steps // 100),
                 bw_mf=0.02, n_min=1.0, bw_kde=0.10, x0=-1.0,
                 x0_jitter=a.jitter)
fl = float(estimator_floor(S, base, [2 ** 22], rows=4)[2 ** 22].mean())
print(f"=== sweep {a.system}{a.tag}  fe={a.N*a.steps:.3g}  floor={fl:.5f} ===", flush=True)
print(f"{'lift':>9} {'n_cond':>6} {'anneal':>7} {'reset':>6} {'kappa':>7} {'theta':>6} "
      f"| {'I_F':>9} {'e_F_fin':>9} {'/fl':>6} {'chan':>7} {'cov':>5} {'ess':>5}", flush=True)
cfgs = expand_grid({"kappa": a.kappas, "theta": a.thetas})
n_cfg = len(cfgs)
kap = row_column([c["kappa"] for c in cfgs], a.seeds, DEVICE, DTYPE)
th = row_column([c["theta"] for c in cfgs], a.seeds, DEVICE, DTYPE)
rec = []
import itertools as _it
for lift, wmode, bwk, frj, n_cond, ann, rst in _it.product(
        a.lift, a.w_mode, a.bw_kde, a.fr_jitter, a.nconds, a.anneal, a.reset):
        cfg = RunConfig(**{**base.__dict__, "n_seed": a.seeds, "n_cond": n_cond,
                           "kappa": kap, "theta": th, "w_mode": wmode,
                           "bw_kde": bwk, "fr_rule": "fr", "lift": lift, "fr_jitter": frj,
                           "kappa_end": ann, "acc_reset_at": rst})
        t0 = time.time()
        r = run_wfr(S, cfg, rows=n_cfg * a.seeds, seed=a.seed)
        sc = score(r, S)
        sh = lambda k: sc[k].reshape(n_cfg, a.seeds)
        IF, EF = sh("I_F"), sh("e_F_final")
        CH = sc["chan"][-1].reshape(n_cfg, a.seeds)
        CV = sc["cov"][-1].reshape(n_cfg, a.seeds)
        ES = sc["ess_anc"][-1].reshape(n_cfg, a.seeds)
        for i, c in enumerate(cfgs):
            row = dict(lift=lift, w_mode=wmode, bw_kde=bwk, n_cond=n_cond,
                       fr_jitter=frj, anneal=ann, reset=rst,
                       **c, I_F=float(np.median(IF[i])),
                       e_F_final=float(np.median(EF[i])),
                       chan=float(np.median(CH[i])), cov=float(np.median(CV[i])),
                       ess=float(np.median(ES[i])))
            rec.append(row)
            print(f"{lift[:4]+'/'+wmode[:4]:>9} {n_cond:>6} {str(frj):>7} {str(rst):>6} "
                  f"{c['kappa']:>7} {c['theta']:>6} | {row['I_F']:>9.5f} "
                  f"{row['e_F_final']:>9.5f} {row['e_F_final']/fl:>6.1f} "
                  f"{row['chan']:>7.4f} {row['cov']:>5.2f} {row['ess']:>5.2f}",
                  flush=True)
        del r; torch.cuda.empty_cache()
        print(f"   [{time.time()-t0:.0f}s]", flush=True)

save_json(f"results/sweeps/{a.system}{a.tag}_wfr_sweep.json", {"floor": fl, "rows": rec})
best = min(rec, key=lambda r: r["I_F"])
print("\nBEST", best)
