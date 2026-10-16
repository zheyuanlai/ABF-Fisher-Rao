"""SAFETY-ONLY FR rate calibration (overnight PREREGISTRATION section 6).  BLIND BY CONSTRUCTION: reads only the
genealogy and event counters of FR-arm runs (sv_u windowed ESS / max lineage share, sv_i replacement counts); it
never loads a reference, never computes a free-energy, mean-force or marginal error.

    python scripts/overnight/fr_calibrate.py results/overnight/<cell>/frcal --K 1024

Selected rate = the LARGEST rate whose median over seeds satisfies at every save after fr_start:
    windowed ancestor ESS/K >= 0.30;  windowed max lineage share <= max(0.05, 3/K);
    total replacements <= 2 K per t.u. (run average after fr_start).
"""
import argparse
import glob
import json
import os
import re

import numpy as np

U_ESSW, U_MAXFW = 2, 3
V_REPL = 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--K", type=int, required=True)
    a = ap.parse_args()
    files = sorted(f for f in glob.glob(os.path.join(a.dir, f"fr_N{a.K}_rate*_s*.npz"))
                   if not f.endswith((".ckpt.npz", ".tmp.npz")))
    by = {}
    for f in files:
        rate = float(re.search(r"rate([0-9.e-]+)_s", os.path.basename(f)).group(1))
        z = np.load(f, allow_pickle=False)
        cfg = json.loads(str(z["cfg"]))
        steps = z["save_steps"]
        after = steps > cfg["fr_start_steps"]
        essw = z["sv_u"][after, U_ESSW]
        maxw = z["sv_u"][after, U_MAXFW]
        repl = z["sv_i"][:, V_REPL]
        dur = (steps[-1] - cfg["fr_start_steps"]) * cfg["dt"]
        by.setdefault(rate, []).append(dict(min_essw=float(essw.min()), max_maxw=float(maxw.max()),
                                            repl_per_tu=float(repl[-1] / dur), essw=essw, maxw=maxw,
                                            total_repl=int(repl[-1])))
    K = a.K
    lim_max = max(0.05, 3.0 / K)
    table = {}
    for rate in sorted(by):
        runs = by[rate]
        med_essw = np.median(np.array([r["essw"] for r in runs]), 0)
        med_maxw = np.median(np.array([r["maxw"] for r in runs]), 0)
        med_repl = float(np.median([r["repl_per_tu"] for r in runs]))
        ok = bool(med_essw.min() >= 0.30 and med_maxw.max() <= lim_max and med_repl <= 2 * K)
        table[rate] = dict(n_seeds=len(runs), min_median_essw=float(med_essw.min()),
                           max_median_maxw=float(med_maxw.max()), median_repl_per_tu=med_repl,
                           median_total_repl=float(np.median([r["total_repl"] for r in runs])), safe=ok)
        print(f"rate {rate:6g}: seeds {len(runs)}  min median ESSw/K {med_essw.min():.3f}  max median lineage "
              f"{med_maxw.max():.4f} (lim {lim_max:.4f})  repl/t.u. {med_repl:.1f} (lim {2 * K})  -> "
              f"{'SAFE' if ok else 'unsafe'}")
    safe = [r for r in table if table[r]["safe"]]
    choice = max(safe) if safe else None
    print("SELECTED RATE:", choice)
    json.dump(dict(K=K, criteria=dict(essw_min=0.30, maxw_max=lim_max, repl_per_tu_max=2 * K), table=table,
                   selected_rate=choice), open(os.path.join(a.dir, f"frcal_K{K}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
