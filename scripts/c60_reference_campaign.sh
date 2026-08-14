#!/usr/bin/env bash
# C60 reference campaign: builds 1-3 sequentially, then the OpenMM spot-check, then analysis.
# One process at a time on GPU 3 (SPEC §11).  Reviewed launch action recorded in
# docs/C60_EXECUTION_STATE.md; run only after scripts/c60_launch_ladder.sh completed.
#
# Usage:  setsid nohup bash scripts/c60_reference_campaign.sh > results/c60/reference/campaign.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."

NVLIB=$HOME/miniconda3/envs/abffr/lib/python3.14/site-packages/nvidia/cu13/lib
export LD_LIBRARY_PATH="$NVLIB:${LD_LIBRARY_PATH:-}"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate abffr

PINNED=$(cat results/c60/PINNED_COMMIT)
HEAD=$(git rev-parse HEAD)
if [ "$PINNED" != "$HEAD" ]; then
  echo "FAIL: HEAD $HEAD != pinned $PINNED"; exit 1
fi

for b in 1 2 3; do
  echo "=== reference build $b ($(date -u)) ==="
  CUDA_VISIBLE_DEVICES=3 python -u scripts/c60_reference.py --build "$b"
done

echo "=== OpenMM spot-check ($(date -u)) ==="
CUDA_VISIBLE_DEVICES=3 python -u scripts/c60_reference_spotcheck.py

echo "=== analysis ($(date -u)) ==="
python -u scripts/c60_reference_analyze.py

echo "=== CAMPAIGN COMPLETE ($(date -u)) ==="
