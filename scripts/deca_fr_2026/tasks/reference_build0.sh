#!/usr/bin/env bash
# Reference build 0 (96 windows x 32 replicas, 2 ns/replica): one A100, > 2 h -> queue g1.
set -euo pipefail
python scripts/deca_fr_2026/run_reference.py --build-index 0 --out results/deca_fr_2026/reference
echo REF_BUILD_0_OK
