#!/bin/bash
# Confirmation, GPU-routed selected arms, on two A100s (one runner per GPU, seeds split by parity)
set -u
python scripts/nscc_a100/freeze.py check --stage selection || { echo "FREEZE CHECK FAILED"; exit 3; }
bash scripts/nscc_a100/tasks/build_gpu.sh
python scripts/nscc_a100/plan.py confirm
J=${PBS_JOBID%%.*}
CUDA_VISIBLE_DEVICES=$GPU_A python scripts/nscc_a100/runner.py --jobs results/nscc_a100/jobs/confirm_gpu.jsonl --mode gpu --shard 0/2 --pin-offset 0 > $ABFFR_LOGS/confirm_g2_A_$J.log 2>&1 &
CUDA_VISIBLE_DEVICES=$GPU_B python scripts/nscc_a100/runner.py --jobs results/nscc_a100/jobs/confirm_gpu.jsonl --mode gpu --shard 1/2 --pin-offset 4 > $ABFFR_LOGS/confirm_g2_B_$J.log 2>&1 &
wait
grep -h RUNNER_DONE $ABFFR_LOGS/confirm_g2_[AB]_$J.log
