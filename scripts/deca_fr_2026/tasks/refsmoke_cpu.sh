#!/usr/bin/env bash
# CPU-only end-to-end check of run_reference.py + merge_reference.py on a toy umbrella set
# (4 windows x 3 replicas, production-length pull so the structural screen passes).
set -euo pipefail
OUT=results/deca_fr_2026/smoke/reference_cpu
rm -rf $OUT
for b in 0 1; do
  python scripts/deca_fr_2026/run_reference.py --build-index $b --out $OUT --device cpu --n-windows 4 --n-rep 3 \
      --win-lo 1.5 --win-hi 2.4 --n-pull-steps 20000 --n-equil-steps 500 --n-prod-steps 3000 --checkpoint-every 1500 --sample-every 50
done
python scripts/deca_fr_2026/merge_reference.py --dir $OUT
echo REFSMOKE_CPU_OK
