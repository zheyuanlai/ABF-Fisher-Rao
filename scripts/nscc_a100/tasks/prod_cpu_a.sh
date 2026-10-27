#!/bin/bash
# production, CPU-routed arms of all systems (exclusive CCDs of one EPYC 7713 socket)
set -u
python scripts/nscc_a100/freeze.py check --stage amendment1 || { echo "FREEZE CHECK FAILED"; exit 3; }
python scripts/nscc_a100/plan.py prod
python scripts/nscc_a100/runner.py --jobs results/nscc_a100/jobs/prod_cpu.jsonl --mode cpu --max-ccds 8 --systems lta300
