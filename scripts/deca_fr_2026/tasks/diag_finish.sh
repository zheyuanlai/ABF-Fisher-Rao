#!/usr/bin/env bash
# Analysis + figures of the diagnostic init cells (CPU job).
set -euo pipefail
python scripts/deca_fr_2026/analyze.py --config configs/deca_fr_2026/diagnostics.yaml
python scripts/deca_fr_2026/plot.py --config configs/deca_fr_2026/diagnostics.yaml
echo DIAG_FINISH_OK
