#!/usr/bin/env bash
# Re-run the adaptive-biasing-potential baselines after removing a Fixman weight
# that never belonged there.  These arms sample UNCONSTRAINED, so conditioning on
# z already gives nu^xi; the (det G)^{-1/2} weight belongs to constrained
# sampling.  Applying it was worth 0.15 kcal/mol and was their entire apparent
# bias floor.
set -e
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=3
python scripts/mol_run_many.py --spec results/mol/baseline_screen.json
python - <<'PY'
import json, glob, os, math
import numpy as np
Z0 = round(-30.0*math.pi/180.0, 4)
spec = []
for arm in ("opes", "abp"):
    for sysname, seeds, z0 in (("PEN", 32, 0.0), ("ALA", 16, Z0)):
        best, bt = None, None
        for p in sorted(glob.glob(f"results/mol/campaign/{sysname}_{arm}_screen_*.npz")):
            v = float(np.median(np.load(p)["I_F"]))
            if best is None or v < best:
                best, bt = v, os.path.basename(p)[:-4].split("_screen_")[1]
        if bt is None:
            continue
        kw = dict(system=sysname, arm=arm, seeds=seeds, seed0=90000, N=256,
                  steps=400_000, bw_mf=0.05, save_every=10_000, z0=z0,
                  tag="confirm")
        if arm == "opes":
            sg, bar = bt[1:].split("b")
            kw.update(opes_sigma=float(sg), opes_barrier=float(bar))
        else:
            kw.update(shus_gain=float(bt[1:]))
        print(sysname, arm, "best", bt, "I_F", best)
        spec.append(kw)
        if sysname == "PEN":
            spec.append({**kw, "seeds": 16, "steps": 1_600_000,
                         "save_every": 40_000, "tag": "long"})
json.dump(spec, open("results/mol/baseline_confirm.json", "w"), indent=1)
PY
python scripts/mol_run_many.py --spec results/mol/baseline_confirm.json
echo BASELINES_DONE
