#!/bin/bash
# Frozen design: Stage I record of all three systems, the A100 benchmark (3 repeats, every N, ABF and FR), then the
# GPU half of the V4 law check (full-budget runs on the validation seeds)
set -u
python scripts/nscc_a100/freeze.py check --stage design || { echo "FREEZE CHECK FAILED"; exit 3; }
bash scripts/nscc_a100/tasks/build_gpu.sh
python scripts/nscc_a100/validate_gpu.py --system gateway
python scripts/nscc_a100/validate_gpu.py --system lta150
python scripts/nscc_a100/validate_gpu.py --system lta300
python scripts/nscc_a100/plan.py bench
python scripts/nscc_a100/runner.py --jobs results/nscc_a100/jobs/bench_gpu.jsonl --mode gpu
python scripts/nscc_a100/plan.py law
python scripts/nscc_a100/runner.py --jobs results/nscc_a100/jobs/law_gpu.jsonl --mode gpu
