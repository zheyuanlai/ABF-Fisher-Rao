#!/usr/bin/env bash
# g3 job, GPU 2: Stage II-A pilot kernel h=0.02, then the Stage I-B timestep check.
set -euo pipefail
bash scripts/deca_fr_2026/tasks/pilot_kernel_h02.sh
bash scripts/deca_fr_2026/tasks/validation_dt.sh
echo STAGE1_T2_OK
