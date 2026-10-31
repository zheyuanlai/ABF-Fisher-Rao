#!/bin/bash
# ABF + FR re-runs of the 8 positive cells (archived seeds/configs) and the fresh-seed laplacian_scale pilot ladder,
# in one pool of NPROCS single-threaded processes (idempotent; resubmission resumes).
set -uo pipefail
cd "$LB_ROOT"
python scripts/laplacian_baselines/run_all.py configs/laplacian_baselines/rerun_*.json configs/laplacian_baselines/pilot_*.json --procs "$NPROCS"
