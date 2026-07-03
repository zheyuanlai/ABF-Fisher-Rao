#!/usr/bin/env bash
# Safe launcher for the WCA serial one-walker ABF control (Part H) on the shared H200 box.
#
# Compute policy (HARD): use ONLY GPUs from {4,5,6,7}; never 0-3; never more than TWO GPUs;
# one process per GPU. This script refuses any GPU outside 4-7 and any list longer than 2,
# and prints nvidia-smi + the selected GPU list before launching anything. Resumable: valid
# per-seed results are skipped and interrupted jobs resume from their checkpoints, so re-running
# the same command continues an interrupted sweep.
#
# Usage:
#   bash scripts/run_wca_serial_abf_h200.sh <config> <stage> <gpus_csv> [extra args...]
#
# Examples:
#   bash scripts/run_wca_serial_abf_h200.sh configs/wca_serial_abf_equal_budget.yaml benchmark 4 --benchmark
#   bash scripts/run_wca_serial_abf_h200.sh configs/wca_serial_abf_equal_budget.yaml smoke 4 --checkpoint-every 5000
#   bash scripts/run_wca_serial_abf_h200.sh configs/wca_serial_abf_equal_budget.yaml production 4,5
set -euo pipefail

CONFIG="${1:?need config path}"
STAGE="${2:?need stage name}"
GPUS_CSV="${3:?need GPU id(s) from 4-7, e.g. 4 or 4,5}"
shift 3
EXTRA=("$@")

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

IFS=',' read -r -a GPUS <<< "$GPUS_CSV"
NGPU="${#GPUS[@]}"
if (( NGPU < 1 || NGPU > 2 )); then
  echo "ERROR: use 1 or 2 GPUs (got '$GPUS_CSV'). Never more than 2." >&2
  exit 1
fi
for g in "${GPUS[@]}"; do
  if ! [[ "$g" =~ ^[4-7]$ ]]; then
    echo "ERROR: GPU '$g' is not allowed. Use ONLY GPUs 4,5,6,7 (never 0-3)." >&2
    exit 1
  fi
done

echo "[h200] ================= nvidia-smi ================="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv || true
echo "[h200] =============================================="
echo "[h200] config=$CONFIG stage=$STAGE selected GPUs=(${GPUS[*]}) processes=$NGPU"

LOGDIR="$(python - "$CONFIG" <<'PY'
import sys, yaml, os
cfg = yaml.safe_load(open(sys.argv[1]))
print(os.path.join(cfg["output_root"], "logs"))
PY
)"
mkdir -p "$LOGDIR"
STAMP="$(date +%Y%m%d_%H%M%S)"

# 1) Precompute / verify TI references (sharded by distinct physics; refs are already cached).
echo "[h200] verifying TI references across ${NGPU} GPU(s) ..."
ref_pids=()
for i in "${!GPUS[@]}"; do
  gpu="${GPUS[$i]}"
  CUDA_VISIBLE_DEVICES="$gpu" python -u scripts/run_wca_serial_abf.py \
    --config "$CONFIG" --stage "$STAGE" --precompute-references \
    --shard "$i" --num-shards "$NGPU" "${EXTRA[@]}" \
    > "$LOGDIR/${STAGE}_${STAMP}_refs_shard${i}.log" 2>&1 &
  ref_pids+=("$!")
done
for p in "${ref_pids[@]}"; do wait "$p" || { echo "[h200] reference verify failed" >&2; exit 1; }; done
echo "[h200] references ready."

# 2) Fan the batched jobs out, one process per GPU, disjoint shards.
pids=()
for i in "${!GPUS[@]}"; do
  gpu="${GPUS[$i]}"
  log="$LOGDIR/${STAGE}_${STAMP}_shard${i}_gpu${gpu}.log"
  echo "[h200] launching shard $i/$NGPU on GPU $gpu -> $log"
  CUDA_VISIBLE_DEVICES="$gpu" python -u scripts/run_wca_serial_abf.py \
    --config "$CONFIG" --stage "$STAGE" \
    --shard "$i" --num-shards "$NGPU" "${EXTRA[@]}" \
    > "$log" 2>&1 &
  pids+=("$!")
done

rc=0
for p in "${pids[@]}"; do
  if ! wait "$p"; then rc=1; fi
done
echo "[h200] all shards finished (rc=$rc). Logs in $LOGDIR"
exit "$rc"
