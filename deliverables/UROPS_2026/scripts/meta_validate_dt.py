#!/usr/bin/env python
"""Stage 1 of deliverables/UROPS_2026/metastability/PREREGISTRATION.md: Euler-Maruyama timestep gate.

Flat-bias law pi(x, y) ~ exp(-beta (V(x, y) - F_ref(x))) on the box (exact x-marginal uniform, exact y-conditional),
EM with bias force +F'_ref(x) at h in {2e-3, 1e-3, 5e-4, 2.5e-4} and exact MALA at h = 1e-3, every chain started from
exact i.i.d. draws.  64 groups x 1000 walkers, burn-in 20 t.u., 200 t.u. of accumulation on 120 bins; jackknife over
groups.  Writes <out>/validation.json and FROZEN.json['timestep'] (the largest passing h).

    python meta_validate_dt.py --out $ABFFR_DATA_ROOT/results/urops_2026/metastability/validation --procs 64
"""
import argparse
import datetime
import json
import math
import multiprocessing as mp
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

C.setup_paths()
import meta_engine as E  # noqa: E402

BETA = 4.0
NBV = 120
G, NW = 64, 1000
T_BURN, T_ACC = 20.0, 200.0
CANDIDATES = [2e-3, 1e-3, 5e-4, 2.5e-4]
MALA_H = 1e-3
GATES = dict(V1_var_rel_rms=0.02, V2_F_mf=0.0137 / 3.0, V3_Fp=0.0718 / 3.0)
FROZEN = os.path.join(C.PKG, "metastability", "FROZEN.json")
_REF = {}


def ref_fine():
    if "r" not in _REF:
        _REF["r"] = E.reference(BETA, nx=6001, ny=2001)
    return _REF["r"]


def exact_init(seed, n):
    from abffr import potentials as P
    xg, F, Fp, p = ref_fine()
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([int(seed), 99])))
    x = rng.uniform(E.XMIN, E.XMAX, n)
    u = rng.random(n)
    yg = np.linspace(E.YMIN, E.YMAX, 4001)
    y = np.empty(n)
    for s in range(0, n, 250):
        xx = x[s:s + 250, None]
        ph = BETA * P.potential_xy(xx, yg[None, :])
        w = np.exp(-(ph - ph.min(1, keepdims=True)))
        cdf = np.concatenate([np.zeros((w.shape[0], 1)), np.cumsum(0.5 * (w[:, 1:] + w[:, :-1]) * np.diff(yg), 1)], 1)
        for i in range(w.shape[0]):
            y[s + i] = np.interp(u[s + i] * cdf[i, -1], cdf[i], yg)
    return x, y


def worker(job):
    kind, h, g = job
    xg, F, Fp, p = ref_fine()
    x, y = exact_init(1_000_000 + 1000 * g + int(round(1e6 * h)), NW)
    E.seed_numba(int((7_000_003 * g + int(round(1e7 * h)) + (1 if kind == "mala" else 0)) % (2 ** 31 - 1)))
    Cv, Mv, Yv, Y2v = (np.zeros(NBV) for _ in range(4))
    zx, zy = np.zeros(NW), np.zeros(NW)
    nb_, na = int(round(T_BURN / h)), int(round(T_ACC / h))
    acc = np.zeros(2)
    if kind == "em":
        E.advance_fixed_bias(x, y, nb_, h, BETA, Fp, xg[0], xg[1] - xg[0], NBV, Cv, Mv, Yv, Y2v, False, zx, zy)
        E.advance_fixed_bias(x, y, na, h, BETA, Fp, xg[0], xg[1] - xg[0], NBV, Cv, Mv, Yv, Y2v, True, zx, zy)
    else:
        E.mala(x, y, nb_, h, BETA, Fp, xg[0], xg[1] - xg[0], F, NBV, Cv, Mv, Yv, Y2v, False, acc)
        acc[:] = 0
        E.mala(x, y, na, h, BETA, Fp, xg[0], xg[1] - xg[0], F, NBV, Cv, Mv, Yv, Y2v, True, acc)
    return kind, h, g, Cv, Mv, Yv, Y2v, acc


def exact_bins():
    """Exact bin quantities of the flat-bias law: <dV/dx>, <y>, <y^2> per bin (uniform x-weight inside a bin)."""
    from abffr import potentials as P
    xg = np.linspace(E.XMIN, E.XMAX, 6001)
    yg = np.linspace(E.YMIN, E.YMAX, 2001)
    Ey, Ey2, Ef = np.zeros(xg.size), np.zeros(xg.size), np.zeros(xg.size)
    for s in range(0, xg.size, 500):
        xx = xg[s:s + 500, None]
        ph = BETA * P.potential_xy(xx, yg[None, :])
        w = np.exp(-(ph - ph.min(1, keepdims=True)))
        Z = np.trapezoid(w, yg, axis=1)
        Ey[s:s + 500] = np.trapezoid(w * yg[None, :], yg, axis=1) / Z
        Ey2[s:s + 500] = np.trapezoid(w * yg[None, :] ** 2, yg, axis=1) / Z
        Ef[s:s + 500] = np.trapezoid(w * P.dVdx_xy(xx, yg[None, :]), yg, axis=1) / Z
    d = (E.XMAX - E.XMIN) / NBV
    j = np.clip(np.floor((xg - E.XMIN) / d + 1e-9).astype(int), 0, NBV - 1)
    wt = np.full(xg.size, 1.0)
    wt[[0, -1]] = 0.5                                  # trapezoid in x within the box
    agg = lambda v: np.bincount(j, weights=wt * v, minlength=NBV) / np.bincount(j, weights=wt, minlength=NBV)
    fbar, ybar, y2bar = agg(Ef), agg(Ey), agg(Ey2)
    return fbar, y2bar - ybar ** 2


def stats_of(Cs, Ms, Ys, Y2s, fbar, vbar, eval_mask, kT):
    C_, M_, Y_, Y2_ = Cs.sum(0), Ms.sum(0), Ys.sum(0), Y2s.sum(0)
    d = (E.XMAX - E.XMIN) / NBV
    mf = M_ / C_
    dmf = (mf - fbar)[eval_mask]
    Fmf = np.cumsum(np.where(eval_mask, mf - fbar, 0.0)) * d
    Fmf = Fmf[eval_mask] - Fmf[eval_mask].mean()
    Fd = -kT * np.log(C_ / C_.sum())
    Fd = Fd[eval_mask] - Fd[eval_mask].mean()
    var = Y2_ / C_ - (Y_ / C_) ** 2
    vrel = (var / vbar - 1.0)[eval_mask]
    return dict(D_Fp=float(np.sqrt(np.mean(dmf ** 2))), D_F_mf=float(np.sqrt(np.mean(Fmf ** 2))),
                D_F_density=float(np.sqrt(np.mean(Fd ** 2))), var_rel_rms=float(np.sqrt(np.mean(vrel ** 2))),
                var_rel_max=float(np.max(np.abs(vrel))))


def jackknife(Cs, Ms, Ys, Y2s, fbar, vbar, mask, kT):
    full = stats_of(Cs, Ms, Ys, Y2s, fbar, vbar, mask, kT)
    n = Cs.shape[0]
    loo = [stats_of(np.delete(Cs, i, 0), np.delete(Ms, i, 0), np.delete(Ys, i, 0), np.delete(Y2s, i, 0), fbar, vbar, mask, kT)
           for i in range(n)]
    out = {}
    for k, v in full.items():
        a = np.array([r[k] for r in loo])
        se = float(np.sqrt((n - 1) / n * np.sum((a - a.mean()) ** 2)))
        out[k] = dict(value=v, se=se, upper95=v + 1.645 * se)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=64)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    jobs = [("em", h, g) for h in CANDIDATES for g in range(G)] + [("mala", MALA_H, g) for g in range(G)]
    t0 = datetime.datetime.now(datetime.timezone.utc)
    with mp.get_context("spawn").Pool(a.procs) as pool:
        res = pool.map(worker, jobs, chunksize=1)
    fbar, vbar = exact_bins()
    cen = E.XMIN + (np.arange(NBV) + 0.5) * (E.XMAX - E.XMIN) / NBV
    mask = (cen >= -2.5) & (cen <= 2.5)
    kT = 1.0 / BETA
    S = dict(prereg="deliverables/UROPS_2026/metastability/PREREGISTRATION.md (Stage 1)", gates=GATES, groups=G,
             walkers_per_group=NW, T_burn=T_BURN, T_acc=T_ACC, nbv=NBV, started_utc=t0.isoformat(), per_h={})
    raw = {}
    for kind, h in [("em", h) for h in CANDIDATES] + [("mala", MALA_H)]:
        rr = sorted([r for r in res if r[0] == kind and r[1] == h], key=lambda r: r[2])
        Cs, Ms, Ys, Y2s = (np.array([r[i] for r in rr]) for i in (3, 4, 5, 6))
        raw[f"{kind}_{h:g}"] = dict(C=Cs, M=Ms, Y=Ys, Y2=Y2s)
        st = jackknife(Cs, Ms, Ys, Y2s, fbar, vbar, mask, kT)
        rec = dict(stats=st, V1=st["var_rel_rms"]["value"] <= GATES["V1_var_rel_rms"],
                   V2=st["D_F_mf"]["upper95"] <= GATES["V2_F_mf"], V3=st["D_Fp"]["upper95"] <= GATES["V3_Fp"])
        rec["pass"] = bool(rec["V1"] and rec["V2"] and rec["V3"])
        if kind == "mala":
            acc = np.sum([r[7] for r in rr], 0)
            rec["acceptance"] = float(acc[0] / acc[1])
            rec["V4_pass"] = bool(rec["V2"] and rec["V3"])
        S["per_h"][f"{kind}_{h:g}"] = rec
        print(f"{kind} h {h:g}: D_Fp {st['D_Fp']['value']:.4f} [up {st['D_Fp']['upper95']:.4f}] | D_F_mf "
              f"{st['D_F_mf']['value']:.5f} [up {st['D_F_mf']['upper95']:.5f}] | D_F_dens {st['D_F_density']['value']:.4f} | "
              f"var rel rms {100 * st['var_rel_rms']['value']:.2f} % max {100 * st['var_rel_max']['value']:.2f} % | "
              f"pass {rec['pass']}" + (f" | acc {rec['acceptance']:.4f}" if kind == 'mala' else ""), flush=True)
    passing = [h for h in CANDIDATES if S["per_h"][f"em_{h:g}"]["pass"]]
    S["V4_mala_harness"] = S["per_h"][f"mala_{MALA_H:g}"]["V4_pass"]
    S["selected_h"] = max(passing) if passing else None
    S["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    S["fbar"], S["var_exact"], S["centres"] = fbar.tolist(), vbar.tolist(), cen.tolist()
    np.savez_compressed(os.path.join(a.out, "validation_raw.npz"), **{f"{k}_{q}": v for k, d in raw.items() for q, v in d.items()})
    json.dump(C.json_safe(S), open(os.path.join(a.out, "validation.json"), "w"), indent=1)
    fz = json.load(open(FROZEN)) if os.path.exists(FROZEN) else {}
    fz["timestep"] = dict(h=S["selected_h"], passing=passing, mala_harness_pass=S["V4_mala_harness"],
                          frozen_utc=S["finished_utc"], source="meta_validate_dt.py -> validation.json")
    json.dump(fz, open(FROZEN, "w"), indent=1)
    print("selected h:", S["selected_h"], "| MALA harness:", S["V4_mala_harness"])


if __name__ == "__main__":
    main()
