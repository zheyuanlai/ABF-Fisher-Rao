#!/usr/bin/env bash
# Stage III FR calibration, GPU 1: rate 0.02.
set -euo pipefail
CFG=configs/deca_fr_2026/fr_calibration.yaml
python scripts/deca_fr_2026/run_cell.py --config $CFG --cell rate020 --prepare-only
python scripts/deca_fr_2026/run_cell.py --config $CFG --cell rate020 --method fr_uniform
echo CALIB_T1_OK
