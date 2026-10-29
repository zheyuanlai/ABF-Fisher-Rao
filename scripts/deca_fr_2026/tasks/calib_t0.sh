#!/usr/bin/env bash
# Stage III FR calibration, GPU 0: ABF arm (shared by every rate: same init, same rng_seed, ABF
# never reads fr_rate) then the rate 0.005 FR arm.
set -euo pipefail
CFG=configs/deca_fr_2026/fr_calibration.yaml
python scripts/deca_fr_2026/run_cell.py --config $CFG --cell rate005 --prepare-only
python scripts/deca_fr_2026/run_cell.py --config $CFG --cell rate005 --method abf
python scripts/deca_fr_2026/run_cell.py --config $CFG --cell rate005 --method fr_uniform
echo CALIB_T0_OK
