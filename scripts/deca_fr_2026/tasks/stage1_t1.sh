#!/usr/bin/env bash
# g3 job, GPU 1: Stage II-A pilots, kernel h=0.04 then histogram 129 (~2 x 55 min at B = 512).
set -euo pipefail
bash scripts/deca_fr_2026/tasks/pilot_kernel_h04.sh
bash scripts/deca_fr_2026/tasks/pilot_hist129.sh
echo STAGE1_T1_OK
