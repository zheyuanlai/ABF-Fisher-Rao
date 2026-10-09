#!/usr/bin/env python
"""Sum the recorded wall-clock of every job of the 2026-10-09 numerical-validation campaign (CPU core-hours;
every job is single-threaded).  -> results/numerical_validation/compute_accounting.json"""
import glob, json, os
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
R = os.path.join(ROOT, "results", "numerical_validation")
out = {}
def add(name, files, key="meta_json", wall="wall_s"):
    tot, n = 0.0, 0
    for f in files:
        if ".tmp." in f:
            continue
        try:
            m = json.loads(str(np.load(f)[key]))
            tot += float(m.get(wall, 0.0)); n += 1
        except Exception:
            pass
    out[name] = dict(jobs=n, core_hours=tot / 3600.0)
for g in ("em_impl", "mc", "mala", "lm", "baoab", "diag"):
    add(f"wca/{g}", glob.glob(os.path.join(R, "wca", g, "*.npz")))
for T in (300, 150):
    add(f"lta/T{T}", glob.glob(os.path.join(R, "lta", f"T{T}", "*", "*.npz")))
add("gateway_dt", glob.glob(os.path.join(R, "gateway_dt", "dt*", "*.npz")))
add("wca_confirmatory", glob.glob(os.path.join(R, "wca_confirmatory", "*", "raw", "*.npz")))
out["total_core_hours"] = sum(v["core_hours"] for v in out.values() if isinstance(v, dict))
out["gpu_hours"] = 0.0
out["note"] = "single-threaded numba/numpy jobs on a shared 2x EPYC 9554 node; wall-clock under co-tenancy (load 40-260); tests and analyses not counted (< 1 core-h)"
json.dump(out, open(os.path.join(R, "compute_accounting.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
