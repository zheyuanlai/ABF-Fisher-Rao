#!/usr/bin/env bash
# g3 calibration job, GPU 3: the production K=64 x 2 ns ABF arm (ABF never reads the FR settings,
# so it can run before the rate is frozen; its fr_uniform partner reuses the same init.npz later).
set -euo pipefail
bash scripts/deca_fr_2026/tasks/prod_K64_T2ns_abf.sh
echo CALIB_T3_OK
