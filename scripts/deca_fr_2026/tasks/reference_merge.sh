#!/usr/bin/env bash
# CPU job: MBAR per build, consensus, F' reference, drift, Gate A, acceptance.
set -euo pipefail
python scripts/deca_fr_2026/merge_reference.py --dir results/deca_fr_2026/reference
echo REF_MERGE_OK
