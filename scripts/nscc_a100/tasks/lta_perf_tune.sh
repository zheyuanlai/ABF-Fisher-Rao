#!/bin/bash
# LTA after the unrolled group reduction: step split, quick Stage I, then the LTA launch sweep
set -u
bash scripts/nscc_a100/tasks/build_gpu.sh
python scripts/nscc_a100/diag_lta_perf.py
python scripts/nscc_a100/validate_gpu.py --system lta150 --quick
rm -f results/nscc_a100/bench/tune_lta150.jsonl
python scripts/nscc_a100/bench_gpu.py --system lta150 --N 1,4,16,64,256,1024 --methods abf --sweep --scale 0.25 --out results/nscc_a100/bench/tune_lta150.jsonl
