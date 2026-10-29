#!/usr/bin/env bash
# Stage II-A estimator pilot, cell hist129: K=64, 1 ns, 8 seeds, ABF only.
set -euo pipefail
CFG=configs/deca_fr_2026/estimator_pilot.yaml
python scripts/deca_fr_2026/run_cell.py --config $CFG --cell hist129 --prepare-only
python scripts/deca_fr_2026/run_cell.py --config $CFG --cell hist129 --method abf
echo PILOT_hist129_OK
