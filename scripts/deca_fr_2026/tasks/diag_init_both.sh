#!/usr/bin/env bash
# Both diagnostic init cells on one GPU, sequentially (~2 x 110 min).
set -euo pipefail
bash scripts/deca_fr_2026/tasks/diag_init_ppii.sh
bash scripts/deca_fr_2026/tasks/diag_init_mixed.sh
echo DIAG_INIT_BOTH_OK
