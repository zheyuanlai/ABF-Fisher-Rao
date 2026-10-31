#!/bin/bash
# Laplacian arms of the 8 positive cells at the FROZEN per-cell scale (make_jobs.py apply after the pilot decision):
# archived seeds, archived FR configuration otherwise, one pool of NPROCS single-threaded processes (idempotent).
set -uo pipefail
cd "$LB_ROOT"
grep -h '"calibration_status"' configs/laplacian_baselines/jobs_R2*.json configs/laplacian_baselines/gateway.json | sort | uniq -c
if grep -q UNVALIDATED configs/laplacian_baselines/jobs_R2C1.json; then echo "scale not frozen (make_jobs.py apply not run)"; exit 2; fi
python scripts/laplacian_baselines/run_all.py configs/laplacian_baselines/jobs_R2C1.json configs/laplacian_baselines/jobs_R2C2.json \
    configs/laplacian_baselines/jobs_R2C3.json configs/laplacian_baselines/jobs_R2C4.json configs/laplacian_baselines/jobs_R2H80.json \
    configs/laplacian_baselines/jobs_R2H350.json configs/laplacian_baselines/prod_gateway.json --procs "$NPROCS"
