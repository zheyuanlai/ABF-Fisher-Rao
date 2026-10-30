#!/usr/bin/env bash
# Reduced-intensity calibration on one GPU: rate 0.002 then 0.001 (~45 min each at B = 1024).
set -euo pipefail
bash scripts/deca_fr_2026/tasks/calib_rate002.sh
bash scripts/deca_fr_2026/tasks/calib_rate001.sh
echo CALIB_LO_OK
