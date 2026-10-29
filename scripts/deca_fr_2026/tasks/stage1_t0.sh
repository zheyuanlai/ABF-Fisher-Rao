#!/usr/bin/env bash
# g3 job, GPU 0: Stage I-A replay then Stage I-B timestep check.
set -euo pipefail
bash scripts/deca_fr_2026/tasks/validation_replay.sh
bash scripts/deca_fr_2026/tasks/validation_dt.sh
echo STAGE1_T0_OK
