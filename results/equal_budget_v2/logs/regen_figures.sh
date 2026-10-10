#!/bin/bash
# Regenerate every production figure with the legibility-checked plotting scripts (2026-10-10 ~09:30 UTC).
cd /home/zheyuanlai/ABF-Fisher-Rao
source /home/zheyuanlai/miniconda3/etc/profile.d/conda.sh && conda activate abffr
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 NUMBA_CACHE_DIR=/home/zheyuanlai/.cache/numba_eqb
L=results/equal_budget_v2/logs/regen
mkdir -p $L
for N in 2048 1024 512 256 128 64 32 16 8 4 2 1; do python scripts/equal_budget/plot_config.py --system gateway --only-N $N > $L/gateway_N$N.log 2>&1 & done
for s in lta300 lta150; do for N in 1024 512 256 128 64 32 16 8 4 2 1; do python scripts/equal_budget/plot_config.py --system $s --only-N $N > $L/${s}_N$N.log 2>&1 & done; done
wait
python scripts/equal_budget/plot_synthesis.py --system all > $L/synthesis_all.log 2>&1
echo "synthesis exit $?" >> $L/synthesis_all.log
python scripts/equal_budget/audit_completeness.py --system all > $L/audit_all.log 2>&1
echo "audit exit $?" >> $L/audit_all.log
echo "[$(date -u +%FT%TZ)] regen done" > $L/DONE
