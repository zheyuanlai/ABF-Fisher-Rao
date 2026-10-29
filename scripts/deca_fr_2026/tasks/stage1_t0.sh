#!/usr/bin/env bash
# g3 job, GPU 0: Stage I-A replay (legacy defect, then fixed).  ~2 x 35 min at B = 128.
set -euo pipefail
bash scripts/deca_fr_2026/tasks/validation_replay.sh
echo STAGE1_T0_OK
