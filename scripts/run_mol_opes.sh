#!/usr/bin/env bash
# OPES / ABP: the adaptive-biasing-POTENTIAL family, so the comparison covers
# both major adaptive approaches rather than only adaptive-biasing-force.
set -e
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=3
python scripts/mol_run_many.py --spec results/mol/opes_screen_spec.json
python - <<'PY'
import json, glob, os, math
import numpy as np
Z0 = round(-30.0*math.pi/180.0, 4)
spec = []
for sysname, seeds, z0 in (("PEN", 32, 0.0), ("ALA", 16, Z0)):
    best, bg = None, None
    for p in sorted(glob.glob(f"results/mol/campaign/{sysname}_opes_screen_g*.npz")):
        v = float(np.median(np.load(p)["I_F"]))
        if best is None or v < best:
            best, bg = v, float(os.path.basename(p).split("_g")[1][:-4])
    print(sysname, "best gain", bg, "I_F", best)
    spec.append(dict(system=sysname, arm="opes", seeds=seeds, seed0=90000, N=256,
                     steps=400_000, bw_mf=0.05, save_every=10_000, z0=z0,
                     shus_gain=bg, tag="confirm"))
spec.append(dict(system="PEN", arm="opes", seeds=16, seed0=90000, N=256,
                 steps=1_600_000, bw_mf=0.05, save_every=40_000,
                 shus_gain=spec[0]["shus_gain"], tag="long"))
json.dump(spec, open("results/mol/opes_confirm_spec.json", "w"), indent=1)
PY
python scripts/mol_run_many.py --spec results/mol/opes_confirm_spec.json
echo OPES_DONE
