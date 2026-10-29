#!/usr/bin/env bash
# g3 job, GPU 2: Stage II-A estimator pilots, kernel h=0.02 then histogram 257.
set -euo pipefail
bash scripts/deca_fr_2026/tasks/pilot_kernel_h02.sh
bash scripts/deca_fr_2026/tasks/pilot_hist257.sh
echo STAGE1_T2_OK
