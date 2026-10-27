#!/bin/bash
# Frozen design: the CPU benchmark (S1 and P_t, every N, ABF and FR, 3 repeats) on exclusive CCDs of one node, then
# the CPU half of the V4 law check
set -u
python scripts/nscc_a100/freeze.py check --stage design || { echo "FREEZE CHECK FAILED"; exit 3; }
python scripts/nscc_a100/plan.py bench
python scripts/nscc_a100/runner.py --jobs results/nscc_a100/jobs/bench_cpu.jsonl --mode cpu --max-ccds 8
python scripts/nscc_a100/plan.py law
python scripts/nscc_a100/runner.py --jobs results/nscc_a100/jobs/law_cpu.jsonl --mode cpu --max-ccds 8
