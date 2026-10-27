#!/bin/bash
# after the performance changes: LTA step split, quick Stage I of all systems, then the launch-tuning sweeps
set -u
bash scripts/nscc_a100/tasks/build_gpu.sh
python scripts/nscc_a100/diag_lta_perf.py
python scripts/nscc_a100/validate_gpu.py --system gateway --quick
python scripts/nscc_a100/validate_gpu.py --system lta150 --quick
rm -f results/nscc_a100/bench/tune_gateway.jsonl results/nscc_a100/bench/tune_lta150.jsonl
python scripts/nscc_a100/bench_gpu.py --system lta150 --N 1,4,16,64,256,1024 --methods abf --sweep --scale 0.25 --out results/nscc_a100/bench/tune_lta150.jsonl
python scripts/nscc_a100/bench_gpu.py --system gateway --N 16,64,128,256,512,1024,2048 --methods abf --sweep --scale 0.25 --out results/nscc_a100/bench/tune_gateway.jsonl
