#!/bin/bash
cd /home/zheyuanlai/ABF-Fisher-Rao
source /home/zheyuanlai/miniconda3/etc/profile.d/conda.sh && conda activate abffr
export CUDA_VISIBLE_DEVICES="" NUMBA_CACHE_DIR=/home/zheyuanlai/.cache/numba_eqb
python -u scripts/equal_budget/run_ladder.py --system lta300 --config configs/equal_budget_v2/confirm_best_lta300.json --workers 40 > results/equal_budget_v2/logs/confirm_lta300.log 2>&1 &
python -u scripts/equal_budget/run_ladder.py --system lta150 --config configs/equal_budget_v2/confirm_best_lta150.json --workers 40 > results/equal_budget_v2/logs/confirm_lta150.log 2>&1 &
python -u scripts/equal_budget/run_ladder.py --system gateway --config configs/equal_budget_v2/confirm_best_gateway.json --workers 60 > results/equal_budget_v2/logs/confirm_gateway.log 2>&1 &
wait
echo "[$(date -u +%FT%TZ)] confirm chain done" >> results/equal_budget_v2/logs/confirm_chain.log
