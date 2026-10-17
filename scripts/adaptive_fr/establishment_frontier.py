"""PREREGISTRATION amendment E-A3 (analysis item 4): window establishment at every frontier allocation of ITE 300 K,
on each cell's own save grid (1 t.u. for the E-A3 cells EB8 / EB4 / EB1; 1.28 x 10^6 force evaluations = 0.125, 0.5,
2 and 8 t.u. for K1024, K256, K64 and K16).  DESCRIPTIVE; feeds no verdict.

Two measures per seed, median per configuration:
  inst  the frozen measure of every cell (atlas_lib.establishment_time): the first save at which the trailing 1-t.u.
        mean of the INSTANTANEOUS window fraction (walkers with |phi| < window_phi at the save, / K) reaches 0.75 x
        the uniform window share  2 window_phi / 2 pi.  It takes only K + 1 values per save and is weak at K <= 8.
  cum   the same first-passage rule applied to the window share of the DEPOSITS collected during the trailing
        interval max(1 t.u., save interval): difference of the cumulative window-bin counts C over the interval
        divided by the difference of all counts (10^4 K samples per t.u.).
For the same-K two-arm cells the paired per-seed difference FR - ABF is given with a bootstrap CI (both measures).

    ABFFR_DATA_ROOT=/home/zheyuanlai/ABF-Fisher-Rao python scripts/adaptive_fr/establishment_frontier.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "convergence_fr"))
import atlas_lib as L  # noqa: E402
import systems as S  # noqa: E402
from equal_budget_frontier import CONF_EXT  # noqa: E402
from eri import metrics as mt  # noqa: E402

LEVEL = 0.75


def first_passage(x, t, thr):
    """First save time at which x (n_save, R) >= thr, inf if never."""
    out = np.full(x.shape[1], np.inf)
    for r in range(x.shape[1]):
        w = np.where(x[:, r] >= thr)[0]
        if len(w):
            out[r] = t[w[0]]
    return out


def load(cf):
    files = S.glob_both(cf["dir"], cf["pat"])
    inst, cum, seeds, stops = [], [], [], []
    for f in files:
        r = mt.load_run(f)
        c = r["cfg"]
        N, ng = int(c["N"]), r["sv_C"].shape[1]
        assert N == cf["K"]
        t = r["times"]
        share = 2.0 * float(c["window_phi"]) / (2 * np.pi)
        inst.append(r["sv_i"][:, 3] / N)
        grid = (np.arange(ng) + 0.5) / ng * 2 * np.pi - np.pi
        win = np.abs(grid) < float(c["window_phi"])
        Cw, Ct = r["sv_C"][:, win].sum(1), r["sv_C"].sum(1)
        ds = float(np.median(np.diff(t)))
        w = max(1, int(round(1.0 / ds)))
        s = np.arange(w, len(t))
        sh = np.full(len(t), np.nan)
        sh[s] = (Cw[s] - Cw[s - w]) / np.maximum(Ct[s] - Ct[s - w], 1.0)
        cum.append(sh)
        seeds.append(int(c["seed"]))
        stops.append(float(r["fr_controller_t_stop"]) if "fr_controller_t_stop" in r else None)
    inst, cum = np.array(inst).T, np.array(cum).T
    T_inst = L.establishment_time(inst, t, share, level=LEVEL)
    T_cum = first_passage(np.nan_to_num(cum, nan=-1.0), t, LEVEL * share)
    return dict(K=cf["K"], alg=cf["alg"], seeds=seeds, save_interval_tu=ds, trailing_window_tu=w * ds, share=share,
                inst=T_inst, cum=T_cum, fr_stop_tu=stops, n_seeds=len(seeds))


def summ(x):
    x = np.asarray(x, float)
    fin = np.isfinite(x)
    return dict(median=float(np.median(x)) if fin.all() else None, n_never=int((~fin).sum()),
                median_finite=float(np.median(x[fin])) if fin.any() else None, per_seed=[None if not np.isfinite(v) else float(v) for v in x])


def main():
    out = dict(definition=__doc__, configs={})
    data = {}
    for cf in CONF_EXT:
        if len(S.glob_both(cf["dir"], cf["pat"])) != 16:
            print("skip (incomplete):", cf["key"])
            continue
        d = load(cf)
        data[cf["key"]] = d
        out["configs"][cf["key"]] = dict(K=d["K"], alg=d["alg"], n_seeds=d["n_seeds"], seeds=d["seeds"],
                                         save_interval_tu=d["save_interval_tu"], trailing_window_tu=d["trailing_window_tu"],
                                         uniform_window_share=d["share"], level=LEVEL,
                                         inst=summ(d["inst"]), cum=summ(d["cum"]),
                                         fr_stop_tu_median=(float(np.median([s for s in d["fr_stop_tu"] if s is not None]))
                                                            if any(s is not None for s in d["fr_stop_tu"]) else None))
        print(f"{cf['key']:10s} grid {d['save_interval_tu']:6.3f} t.u. | inst median {np.median(d['inst']):7.3f} (never {int((~np.isfinite(d['inst'])).sum())}) "
              f"| cum median {np.median(d['cum']):7.3f} (never {int((~np.isfinite(d['cum'])).sum())})")
    out["paired_same_K"] = {}
    for K in sorted({d["K"] for d in data.values()}):
        a, b = f"ABF_K{K}", f"FR_K{K}"
        if a in data and b in data and data[a]["seeds"] == data[b]["seeds"]:
            res = {}
            for m in ("inst", "cum"):
                da, db = data[a][m], data[b][m]
                ok = np.isfinite(da).all() and np.isfinite(db).all()
                res[m] = dict(abf_median=float(np.median(da)), fr_median=float(np.median(db)),
                              paired_difference_fr_minus_abf=L.boot_median(db - da) if ok else None,
                              fr_shorter_seeds=int((db < da).sum()), equal_seeds=int((db == da).sum()))
            out["paired_same_K"][f"K{K}"] = res
            print(f"K{K} paired: inst ABF {res['inst']['abf_median']:.3f} -> FR {res['inst']['fr_median']:.3f}; "
                  f"cum ABF {res['cum']['abf_median']:.3f} -> FR {res['cum']['fr_median']:.3f} "
                  f"(FR shorter in {res['cum']['fr_shorter_seeds']}/16, equal {res['cum']['equal_seeds']})")
    p = os.path.join(S.ROOT, "results", "adaptive_fr_campaign", "analysis", "E_establishment.json")
    json.dump(out, open(p, "w"), indent=1)
    print("wrote", p)


if __name__ == "__main__":
    main()
