"""Experiment B DEVELOPMENT (exploratory seeds, ITE only): run ABF + CONTINUOUS FR in chunks and record, every
`--every` t.u., the ONLINE observables a stopping controller could use.  Pure host-side reads of the run state
between chunks; the dynamics are exactly engine.run's.  Nothing here reads a reference or an ABF trajectory.

    NUMBA_NUM_THREADS=1 python scripts/adaptive_fr/dev_monitor.py --temp 300 --K 1024 --rate 2.0 --T 20 --seed 1200000

Recorded per check (cumulative arrays; window statistics are formed offline from differences):
  BH, DH (fine-bin births / deaths), C (deposit counts), M (mean-force sums), occupancy histogram of the current
  positions, replacements, opportunities, whole-run and windowed ancestor ESS / max lineage, true translocations.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from eri import engine as E  # noqa: E402
from eri.model import Model  # noqa: E402


def base_cfg(temp):
    j = json.load(open(os.path.join(ROOT, "configs", "overnight", "ite", "jobs_prod.json")))["jobs"][0]["cfg"]
    c = {k: v for k, v in j.items() if k != "seed"}
    if temp == 150:
        c150 = json.load(open(os.path.join(ROOT, "configs", "overnight", "ite150", "jobs_prod.json")))["jobs"][0]["cfg"]
        c["T_K"], c["ff"] = c150["T_K"], c150["ff"]
    return c


def fam(lab, N):
    cnt = np.bincount(lab, minlength=N).astype(float)
    return float(cnt.sum() ** 2 / (cnt ** 2).sum() / N), float(cnt.max() / N)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--temp", type=int, required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--rate", type=float, required=True)
    ap.add_argument("--T", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--every", type=float, default=0.25)
    ap.add_argument("--variant", default="ite_itq3")
    a = ap.parse_args()
    c = dict(base_cfg(a.temp), N=a.K, fr_rate=a.rate, seed=a.seed, n_steps=int(round(a.T / 1e-4)),
             save_every=1250, genealogy_window_steps=25000)
    c = E.make_cfg(**c)
    m = Model(c["variant"], T_K=c["T_K"], params=c["ff"])
    r = E.Run(c, "fr", model=m)
    every = int(round(a.every / c["dt"]))
    rec = {k: [] for k in ("t", "BH", "DH", "C", "M", "occ", "repl", "nopp", "ess", "maxf", "essw", "maxfw", "transloc")}
    P, xw = m.fw.P, m.fw.x_window
    ng = c["n_grid"]
    t0 = time.time()
    k = 0
    while k <= c["n_steps"]:
        if k > 0:
            r.advance(k)
        X = 0.5 * (r.q[:, 0, 0, 0] + r.q[:, 0, 1, 0])
        ph = ((2 * math.pi * (X - xw) / P + math.pi) % (2 * math.pi)) - math.pi
        occ = np.bincount((np.floor((ph + math.pi) / (2 * math.pi / ng)).astype(int)) % ng, minlength=ng)
        e1, m1 = fam(r.anc, c["N"])
        e2, m2 = fam(r.ancw, c["N"])
        for key, val in (("t", k * c["dt"]), ("BH", r.BH.copy()), ("DH", r.DH.copy()), ("C", r.C.copy()),
                         ("M", r.Msum.copy()), ("occ", occ), ("repl", int(r.ist[E.S_REPL])), ("nopp", int(r.ist[E.S_NOPP])),
                         ("ess", e1), ("maxf", m1), ("essw", e2), ("maxfw", m2), ("transloc", int(r.ist[E.S_TRANSLOC]))):
            rec[key].append(val)
        k += every
    out = os.path.join(ROOT, "results", "adaptive_fr_campaign", "dev", f"dev_T{a.temp}_K{a.K}_rate{a.rate:g}_s{a.seed}.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, cfg=json.dumps(c), wall=time.time() - t0, **{k: np.array(v) for k, v in rec.items()})
    print("wrote", out, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
