"""Pre-registration scouting for Experiment A (docs/final_campaign/PREREGISTRATION.md §1.1): what does transverse
mobility lambda change on the shifted fibre, apart from the Gibbs law (which it does not change)?

Flat-bias Euler-Maruyama chains of the frozen validation kernel (src/gateway_family_validation.chain; NO ABF, NO FR,
so no performance quantity of the experiment is read), shift kappa = 1, h = 2.5e-5, 512 walkers started from the
exact flat-bias law, 1 t.u. discarded, 30 t.u. recorded, at lambda in {1, 4, 16, 128}:

* gate transport: hysteresis side changes between x < -0.5 and x > 0.5 per walker per t.u. (traces every 0.01 t.u.);
* coupled conditional relaxation: 1/e time and integrals to fixed cuts of the conditional ACF of z_y at
  x0 in {-1, -0.21, 0} (the harness's own online ACF recorders, acf_plan with its default horizon);
* predictions: frozen-x 1/(lam kappa^2) and coupled 1/(kappa^2 (lam + m'(x0)^2)).

Seeds 440000 + index (validation-type raw seeds, checked by check_seeds.py). One process per lambda.
Usage: python scripts/final_campaign/expA_scout_transport.py --lam 16   (writes results/final_campaign/expA/scout/)
       python scripts/final_campaign/expA_scout_transport.py --collect
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
import gateway_family_validation as GF  # noqa: E402

OUT = os.path.join(ROOT, "results/final_campaign/expA/scout")
LAMS = (1.0, 4.0, 16.0, 128.0)
H = 2.5e-5
NW = 512
T_BURN, T_REC = 1.0, 30.0
TRACE_DT = 0.01


def side_changes(tr):
    """Hysteresis side changes (|x| > 0.5) per walker, from traces (n_samples, n_walkers)."""
    lab = np.zeros(tr.shape[1], dtype=np.int64)
    n = np.zeros(tr.shape[1], dtype=np.int64)
    for row in tr:
        new = np.where(row > 0.5, 1, np.where(row < -0.5, -1, 0))
        flip = (new != 0) & (lab != 0) & (new != lab)
        n += flip
        lab = np.where(new != 0, new, lab)
    return n


def run(lam):
    m = GF.make_model("shift", kappa=1.0, lam=lam)
    idx = LAMS.index(lam)
    seed = 440000 + idx
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-1.5, 1.5, NW)
    y0 = GF.initial_y(x0, rng.standard_normal(NW), m)
    plan = GF.acf_plan(m, H)
    n_burn = int(round(T_BURN / H))
    n_steps = int(round(T_REC / H))
    tr_every = int(round(TRACE_DT / H))
    t0 = time.time()
    r = GF.chain(m, 0, x0, y0, n_steps, n_burn, H, 1, seed, trace_every=tr_every, n_trace=NW, n_acf=64, acf=plan)
    wall = time.time() - t0
    trx = np.asarray(r["traces_x"], dtype=np.float64)
    sc = side_changes(trx)
    rate = sc / T_REC
    # jackknife over 16 walker groups
    g = rate.reshape(16, -1).mean(1)
    jk = np.array([np.delete(g, i).mean() for i in range(16)])
    se = math.sqrt(15.0 / 16.0 * ((jk - jk.mean()) ** 2).sum())
    acc0, acc1 = r["acf0"], r["acf1"]
    out = dict(lam=lam, seed=seed, h=H, walkers=NW, T_rec=T_REC, wall_s=wall, nonfinite=int(r["nonfinite"]),
               side_change_rate_per_walker_per_tu=float(rate.mean()), side_change_rate_se=se, acf={})
    st = np.asarray(r["acf_stride"])
    for wi, x0w in enumerate(GF.ACF_WINDOWS):
        mp2 = float(GF.dm_centre(np.array([x0w]), m)[0] ** 2)
        rec = {}
        for lev, acc, lags in ((0, acc0, r["lags0"]), (1, acc1, r["lags1"])):
            t = np.asarray(lags, dtype=np.float64) * st[lev] * H
            ry = np.asarray(GF.acf_from_sums(acc[wi])["rho_y"], dtype=np.float64)
            rec[f"level{lev}"] = dict(t=t.tolist(), rho_y=ry.tolist(),
                                       tau_e=float(GF.tau_efold(t, ry)),
                                       tau_int_cut={str(c): float(GF.tau_integrated(t, ry, c)) for c in (0.3125, 1.25, 5.0)
                                                    if t[-1] >= c})
        rec["mprime2"] = mp2
        rec["pred_frozen_tau"] = 1.0 / lam
        rec["pred_coupled_tau"] = 1.0 / (lam + mp2)
        out["acf"][str(x0w)] = rec
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"scout_lam{lam:g}.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: out[k] for k in ("lam", "wall_s", "nonfinite", "side_change_rate_per_walker_per_tu",
                                          "side_change_rate_se")}))


def collect():
    rows = []
    for lam in LAMS:
        p = os.path.join(OUT, f"scout_lam{lam:g}.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        row = dict(lam=lam, rate=d["side_change_rate_per_walker_per_tu"], rate_se=d["side_change_rate_se"])
        for x0w, rec in d["acf"].items():
            lev = rec.get("level0") or rec.get("level1")
            row[f"tau_e@{x0w}"] = lev["tau_e"]
            row[f"pred_coupled@{x0w}"] = rec["pred_coupled_tau"]
        rows.append(row)
    with open(os.path.join(OUT, "scout_summary.json"), "w") as fh:
        json.dump(rows, fh, indent=1)
    for r in rows:
        print(r)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float)
    ap.add_argument("--collect", action="store_true")
    a = ap.parse_args()
    if a.collect:
        collect()
    else:
        run(a.lam)
