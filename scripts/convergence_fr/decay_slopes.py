"""Late-time error decay exponent: per seed, least-squares slope of log e_F vs log t on [0.4 T, T], median and IQR
over seeds (descriptive; a slope near -1 = dilution of a fixed early bias, near -1/2 = variance, near 0 = floor).

    python scripts/convergence_fr/decay_slopes.py E1 X1 X2 M300_256 M150_256 ERI256 ERI1024 E2
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import atlas_lib as L  # noqa: E402

ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
AN = os.path.join(ROOT, "results", "convergence_fr_campaign", "analysis")
out = {}
for b in sys.argv[1:]:
    p = os.path.join(AN, f"{b}_curves.npz")
    if not os.path.exists(p):
        continue
    z = np.load(p)
    t = z["t"]
    T = t[-1]
    sel = (t >= 0.4 * T) & (t <= T)
    row = {}
    for k in ("eF_abf", "eF_fr"):
        sl = np.array([np.polyfit(np.log(t[sel]), np.log(z[k][sel, r]), 1)[0] for r in range(z[k].shape[1])])
        row[k] = dict(median=float(np.median(sl)), iqr=[float(np.quantile(sl, 0.25)), float(np.quantile(sl, 0.75))],
                      ci95_median=list(L.boot_median(sl)[1:]), per_seed=sl.tolist())
    out[b] = row
    print(f"{b:10s} ABF {row['eF_abf']['median']:+.2f} [{row['eF_abf']['ci95_median'][0]:+.2f}, {row['eF_abf']['ci95_median'][1]:+.2f}]"
          f"   FR {row['eF_fr']['median']:+.2f} [{row['eF_fr']['ci95_median'][0]:+.2f}, {row['eF_fr']['ci95_median'][1]:+.2f}]")
json.dump(out, open(os.path.join(ROOT, "results", "convergence_fr_campaign", "mechanism", "decay_slopes.json"), "w"), indent=1)
