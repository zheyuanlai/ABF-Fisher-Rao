"""Run a labelled suite of arms on one system at matched force-evaluation budget.

Every entry is (label, arm, overrides).  All arms share N, dt, n_steps and the
estimator, and all replicates share initial conditions, so every pairwise
comparison is paired by seed.  Results are stored as npz + json for rescoring.
"""
from __future__ import annotations

import argparse, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch

from rcwfr.campaign import (ARM_LIBRARY, estimator_floor, paired_bootstrap,
                            rel_change, run_arm, save_json, save_npz, score)
from rcwfr.engines import RunConfig
from rcwfr.registry import build


def run_suite(sysm, entries, base_cfg, n_seed, seed, verbose=True):
    res = {}
    for label, arm, ov in entries:
        t0 = time.time()
        cfg = RunConfig(**{**base_cfg.__dict__, "n_seed": n_seed})
        run, used = run_arm(sysm, arm, cfg, n_seed, seed, overrides=ov)
        sc = score(run, sysm)
        sc["wall"] = time.time() - t0
        sc["arm"], sc["overrides"] = arm, ov
        if "ex_accept" in run:
            sc["ex_accept"] = run["ex_accept"]
        res[label] = sc
        if verbose:
            ex = f" acc={run['ex_accept']:.3f}" if "ex_accept" in run else ""
            print(f"  {label:16s} I_F={np.median(sc['I_F']):.5f} "
                  f"e_F_fin={np.median(sc['e_F_final']):.5f} "
                  f"chan={np.median(sc['chan'][-1]):.4f} "
                  f"cov={np.median(sc['cov'][-1]):.3f} "
                  f"({sc['wall']:.0f}s){ex}", flush=True)
        del run
        torch.cuda.empty_cache()
    return res


def report(res, baselines, out_json=None, floor=None):
    lines = []
    labels = list(res)
    hdr = (f"{'arm':16s} {'I_F':>9s} {'e_F_fin':>9s} {'/floor':>7s} {'chan':>7s} "
           f"{'ess_anc':>8s} {'wall':>6s}")
    lines.append(hdr); lines.append("-" * len(hdr))
    for lab in labels:
        s = res[lab]
        f = np.median(s["e_F_final"]) / floor if floor else float("nan")
        lines.append(f"{lab:16s} {np.median(s['I_F']):9.5f} "
                     f"{np.median(s['e_F_final']):9.5f} {f:7.1f} "
                     f"{np.median(s['chan'][-1]):7.4f} "
                     f"{np.median(s['ess_anc'][-1]):8.3f} {s['wall']:6.0f}")
    lines.append("")
    for b in baselines:
        if b not in res:
            continue
        lines.append(f"paired median relative change in I_F vs {b}  "
                     f"(negative = arm better; 95% bootstrap CI):")
        for lab in labels:
            if lab == b:
                continue
            d = rel_change(res[lab]["I_F"], res[b]["I_F"])
            m, lo, hi = paired_bootstrap(d)
            flag = "  <== better" if hi < 0 else ("  (worse)" if lo > 0 else "")
            lines.append(f"   {lab:16s} {100*m:+7.1f}%  [{100*lo:+7.1f}, {100*hi:+7.1f}]{flag}")
        lines.append("")
    txt = "\n".join(lines)
    print(txt)
    if out_json:
        save_json(out_json, {k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                                 for kk, vv in v.items() if kk != "e_F"}
                             for k, v in res.items()})
    return txt


DEFAULT_ENTRIES = [
    ("wfr",        "wfr",       {}),
    ("w_only",     "w_only",    {}),
    ("fr_only",    "fr_only",   {}),
    ("w_count",    "w_count",   {}),
    ("wfr_oracle", "wfr_oracle", {}),
    ("ti_cold",    "ti_cold",   {}),
    ("ti_warm",    "ti_warm",   {}),
    ("reti_cold",  "reti_cold", {}),
    ("reti_warm",  "reti_warm", {}),
    ("abf",        "abf",       {}),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="CHANNEL")
    ap.add_argument("--N", type=int, default=256)
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--seed", type=int, default=2000)
    ap.add_argument("--kappa", type=float, default=0.125)
    ap.add_argument("--theta", type=float, default=0.6)
    ap.add_argument("--n_cond", type=int, default=5)
    ap.add_argument("--n_ex", type=int, default=5)
    ap.add_argument("--m_spec", type=int, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="results/compare")
    a = ap.parse_args()

    kw = {} if a.m_spec is None else {"m_spec": a.m_spec}
    sysm = build(a.system, **kw)
    base = RunConfig(N=a.N, n_steps=a.steps, dt=1e-3,
                     save_every=max(500, a.steps // 100), bw_mf=0.02, n_min=1.0,
                     bw_kde=0.10, n_bins_count=45, x0=-1.0, ess_window=40,
                     kappa=a.kappa, theta=a.theta, n_cond=a.n_cond, n_ex=a.n_ex)
    fl = estimator_floor(sysm, base, [2 ** 23], rows=4)[2 ** 23].mean()
    name = f"{a.system}{a.tag}"
    print(f"=== {name}  N={a.N} steps={a.steps} fe={a.N*a.steps:.3g} "
          f"seeds={a.seeds}  floor={fl:.5f} ===", flush=True)
    res = run_suite(sysm, DEFAULT_ENTRIES, base, a.seeds, a.seed)
    txt = report(res, ["ti_cold", "reti_cold", "abf"],
                 out_json=os.path.join(a.out, f"{name}.json"), floor=float(fl))
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"{name}.txt"), "w") as f:
        f.write(f"floor={fl:.6f}\n" + txt)


if __name__ == "__main__":
    main()
