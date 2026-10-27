#!/bin/bash
set -u
bash scripts/nscc_a100/tasks/build_gpu.sh
for f in gw_cuda lta_cuda; do
  nvcc -O3 -arch=sm_80 --fmad=false -std=c++17 -Xptxas -v -c src/gpu/$f.cu -o /tmp/${f}_ptxas.o 2>&1 | grep -E -A2 "Compiling entry|Function properties for _Z1[0-9](gw|lt)_(step|fr)" | grep -E "entry|stack|registers" | head -20
done
python scripts/nscc_a100/diag_lta.py
python scripts/nscc_a100/diag_lta_perf.py
