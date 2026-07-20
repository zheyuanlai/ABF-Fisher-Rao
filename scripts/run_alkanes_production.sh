#!/usr/bin/env bash
# Sequential production launcher for the alkane study (headline-first order so the
# most important results land first; resume-safe -- valid runs are skipped).
# Usage: CUDA_VISIBLE_DEVICES=5 bash scripts/run_alkanes_production.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh && conda activate abffr
CFG=configs/alkanes/production.yaml
for STAGE in b1 p1 b2; do
  echo "=== PRODUCTION STAGE $STAGE  $(date) ==="
  python -u scripts/run_alkanes.py --config "$CFG" --stage "$STAGE" --require-single-gpu
done
echo "=== FROZEN-BIAS VALIDATION $(date) ==="
python -u scripts/run_alkanes_frozen.py --config "$CFG" --stage b1 --cells butane
python -u scripts/run_alkanes_frozen.py --config "$CFG" --stage p1 --cells pentane || true
echo "=== ALL PRODUCTION DONE $(date) ==="
