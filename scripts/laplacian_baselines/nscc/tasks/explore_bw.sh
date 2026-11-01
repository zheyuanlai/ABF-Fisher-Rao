#!/bin/bash
# EXPLORATORY: LTA 150 K Laplacian arm with wider KDE bandwidths (scripts/laplacian_baselines/make_explore.py). Idempotent.
set -uo pipefail
cd "$LB_ROOT"
python scripts/laplacian_baselines/run_all.py "${EXPLORE_CFG:-configs/laplacian_baselines/explore_R2C3_bw.json}" --procs "$NPROCS"
