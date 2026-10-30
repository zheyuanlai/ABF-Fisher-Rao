#!/usr/bin/env bash
# Frozen analysis and figures of the production stage (CPU job).
set -euo pipefail
python scripts/deca_fr_2026/analyze.py --config configs/deca_fr_2026/production.yaml
python scripts/deca_fr_2026/plot.py --config configs/deca_fr_2026/production.yaml
echo PROD_FINISH_OK
