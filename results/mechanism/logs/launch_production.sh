#!/bin/bash
# Mechanism campaign production: Experiment I (all variants incl. the alpha1 rerun, Amendment A4) then Experiment II
# (lam1 aliases alpha1 N 2048/512). docs/mechanism/SCIENTIFIC_PLAN.md section 5; gate PASS at h 2.5e-5.
cd /home/zheyuanlai/ABF-Fisher-Rao
source /home/zheyuanlai/miniconda3/etc/profile.d/conda.sh && conda activate abffr
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 NUMBA_CACHE_DIR=/home/zheyuanlai/.cache/numba_mech
taskset -c 128-255 python -u scripts/mechanism/run_cells.py --experiment matched_free_energy conditional_relaxation --include-reuse --workers 120 > results/mechanism/logs/production.log 2>&1
echo "exit $?" >> results/mechanism/logs/production.log
echo "[$(date -u +%FT%TZ)] production done" > results/mechanism/logs/PRODUCTION_DONE
