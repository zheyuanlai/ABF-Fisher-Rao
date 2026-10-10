#!/bin/bash
# Sequential production chain of the equal-budget ladders (docs/equal_budget/SCIENTIFIC_PLAN.md section 6).
cd /home/zheyuanlai/ABF-Fisher-Rao
source /home/zheyuanlai/miniconda3/etc/profile.d/conda.sh && conda activate abffr
export CUDA_VISIBLE_DEVICES="" NUMBA_CACHE_DIR=/home/zheyuanlai/.cache/numba_eqb
W=${WORKERS:-110}
echo "[$(date -u +%FT%TZ)] start lta300" ; python -u scripts/equal_budget/run_ladder.py --system lta300 --order coarse_first --workers $W
echo "[$(date -u +%FT%TZ)] start lta150" ; python -u scripts/equal_budget/run_ladder.py --system lta150 --order coarse_first --workers $W
echo "[$(date -u +%FT%TZ)] start gateway"; python -u scripts/equal_budget/run_ladder.py --system gateway --order ladder --workers $W
echo "[$(date -u +%FT%TZ)] chain done"
