#!/usr/bin/env bash
# g2 job, GPU 1: calibration rates 0.02 then 0.05 (sequential, ~1 h each at B = 1024).
set -euo pipefail
bash scripts/deca_fr_2026/tasks/calib_t1.sh
bash scripts/deca_fr_2026/tasks/calib_t2.sh
echo CALIB_G2B_OK
