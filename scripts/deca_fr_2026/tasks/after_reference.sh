#!/usr/bin/env bash
# CPU job after the three reference builds: merge (MBAR, F'_ref, drift, Gate A), then the frozen
# Stage I/II analysis and figures against the merged reference.
set -euo pipefail
bash scripts/deca_fr_2026/tasks/reference_merge.sh
bash scripts/deca_fr_2026/tasks/stage1_finish.sh
echo AFTER_REFERENCE_OK
