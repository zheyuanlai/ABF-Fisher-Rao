#!/usr/bin/env bash
# Stage III FR calibration, GPU 2: rate 0.05.
set -euo pipefail
CFG=configs/deca_fr_2026/fr_calibration.yaml
python scripts/deca_fr_2026/run_cell.py --config $CFG --cell rate050 --prepare-only
python scripts/deca_fr_2026/run_cell.py --config $CFG --cell rate050 --method fr_uniform
echo CALIB_T2_OK
