#!/bin/bash
# Confirmation, CPU-routed selected arms (exclusive CCDs of one EPYC 7713 socket)
set -u
python scripts/nscc_a100/freeze.py check --stage selection || { echo "FREEZE CHECK FAILED"; exit 3; }
python scripts/nscc_a100/plan.py confirm
python scripts/nscc_a100/runner.py --jobs results/nscc_a100/jobs/confirm_cpu.jsonl --mode cpu --max-ccds 8
