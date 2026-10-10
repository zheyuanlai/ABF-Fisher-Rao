#!/bin/bash
# Experiment I/II timestep-validation gate runs (docs/mechanism/SCIENTIFIC_PLAN.md section 3 + Amendment 1).
cd /home/zheyuanlai/ABF-Fisher-Rao
source /home/zheyuanlai/miniconda3/etc/profile.d/conda.sh && conda activate abffr
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 NUMBA_CACHE_DIR=/home/zheyuanlai/.cache/numba_mech
python -u scripts/mechanism/run_validation.py --workers 120 --cpus 136-255 --h 2.5e-5 > results/mechanism/logs/validation_h2.5e-5.log 2>&1
echo "exit $?" >> results/mechanism/logs/validation_h2.5e-5.log
echo "[$(date -u +%FT%TZ)] validation done" > results/mechanism/logs/VALIDATION_DONE
