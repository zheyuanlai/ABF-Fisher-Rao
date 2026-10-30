#!/usr/bin/env bash
# Stage III, reduced intensity (rule: no tested rate met the lineage criterion): rate001.
set -euo pipefail
CFG=configs/deca_fr_2026/fr_calibration.yaml
python scripts/deca_fr_2026/run_cell.py --config $CFG --cell rate001 --prepare-only
python scripts/deca_fr_2026/run_cell.py --config $CFG --cell rate001 --method fr_uniform
echo CALIB_rate001_OK
