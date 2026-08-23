#!/usr/bin/env bash
# What phase 2 did not reach: the count-balancing control, the slow-mode ranking
# diagnostic, and the hexane experiment.
set -e
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=3
python scripts/mol_mode_diagnostic.py --system PEN
python scripts/mol_mode_diagnostic.py --system HEX
python - <<'PY'
import json
f = json.load(open("results/mol/frozen.json"))
k, t = str(f["wfr_rot"]["kappa"]), str(f["wfr_rot"]["theta"])
json.dump([dict(system="PEN", arm="w_count", seeds=16, seed0=90000, N=256,
                steps=400000, bw_mf=0.05, save_every=10000, tag="confirm",
                kappa=k, theta=t)],
          open("results/mol/wcount_spec.json", "w"), indent=1)
PY
python scripts/mol_run_many.py --spec results/mol/wcount_spec.json
python scripts/mol_run_many.py --spec results/mol/hexane_spec.json
echo PHASE2B_DONE
