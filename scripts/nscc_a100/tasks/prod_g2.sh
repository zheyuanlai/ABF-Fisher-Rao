#!/bin/bash
# Two A100s: Amendment A1 bitwise check, then production of the GPU-routed arms (GPU A: gateway + LTA 150 K, then the
# re-measured G1 FR benchmark rows; GPU B: LTA 300 K).  One runner per GPU, children pinned to separate cores.
set -u
python scripts/nscc_a100/freeze.py check --stage amendment1 || { echo "FREEZE CHECK FAILED"; exit 3; }
bash scripts/nscc_a100/tasks/build_gpu.sh
CUDA_VISIBLE_DEVICES=$GPU_A python scripts/nscc_a100/check_a1.py --out-root /scratch/users/nus/e1297740/abffr_a1check || { echo "A1 CHECK FAILED"; exit 4; }
python scripts/nscc_a100/plan.py prod
python scripts/nscc_a100/plan.py bench
J=${PBS_JOBID%%.*}
( CUDA_VISIBLE_DEVICES=$GPU_A python scripts/nscc_a100/runner.py --jobs results/nscc_a100/jobs/prod_gpu.jsonl --mode gpu --systems gateway,lta150 --pin-offset 0
  CUDA_VISIBLE_DEVICES=$GPU_A python scripts/nscc_a100/runner.py --jobs results/nscc_a100/jobs/bench_gpu.jsonl --mode gpu --configs fr --pin-offset 0 ) > $ABFFR_LOGS/prod_g2_A_$J.log 2>&1 &
CUDA_VISIBLE_DEVICES=$GPU_B python scripts/nscc_a100/runner.py --jobs results/nscc_a100/jobs/prod_gpu.jsonl --mode gpu --systems lta300 --pin-offset 4 > $ABFFR_LOGS/prod_g2_B_$J.log 2>&1 &
wait
grep -h RUNNER_DONE $ABFFR_LOGS/prod_g2_[AB]_$J.log
