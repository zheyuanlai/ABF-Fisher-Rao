#!/bin/bash
# Experiment III GPU campaign (Amendment 2): LTA 300 K and 150 K on GPU 3 only, one run at a time.
cd /home/zheyuanlai/ABF-Fisher-Rao
source /home/zheyuanlai/miniconda3/etc/profile.d/conda.sh && conda activate abffr
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 NUMBA_CACHE_DIR=/home/zheyuanlai/.cache/numba_mech CUDA_VISIBLE_DEVICES=3
python -u scripts/mechanism/parallel_benchmark.py campaign --backend gpu_torch --systems lta300 lta150 --seeds-per-arm 8 --cpus 20 --max-device-hours 8 > results/mechanism/logs/bench_gpu.log 2>&1
echo "exit $?" >> results/mechanism/logs/bench_gpu.log
date -u +%FT%TZ > results/mechanism/logs/BENCH_GPU_DONE
