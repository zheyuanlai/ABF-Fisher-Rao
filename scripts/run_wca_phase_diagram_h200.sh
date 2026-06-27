#!/usr/bin/env bash
# Safe launcher for the WCA phase-diagram study on the shared H200 box.
#
# Runs ONE process per GPU (the H200 benchmark showed packing multiple processes
# onto one GPU gives no speedup -- the kernels serialise). Uses at most TWO GPUs.
#
# Usage:
#   bash scripts/run_wca_phase_diagram_h200.sh <config> <stage> <gpus_csv> [extra args...]
#
# Examples:
#   bash scripts/run_wca_phase_diagram_h200.sh configs/wca_phase_diagram_smoke.yaml   smoke 4
#   bash scripts/run_wca_phase_diagram_h200.sh configs/wca_phase_diagram_pilot.yaml   pilot 4,7
#   bash scripts/run_wca_phase_diagram_h200.sh configs/wca_phase_diagram_production.yaml production 4,7 --overwrite
#
# The first step precomputes TI references sharded by distinct physics across the
# selected GPU(s), so parallel workers never race on the cache.
set -euo pipefail

CONFIG="${1:?need config path}"
STAGE="${2:?need stage name}"
GPUS_CSV="${3:?need GPU id(s), e.g. 4 or 4,7}"
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

LOGDIR="$(python - "$CONFIG" <<'PY'
import sys, yaml, os
cfg = yaml.safe_load(open(sys.argv[1]))
print(os.path.join(cfg["output_root"], "logs"))
PY
)"
mkdir -p "$LOGDIR"
STAMP="$(date +%Y%m%d_%H%M%S)"

echo "[h200] config=$CONFIG stage=$STAGE gpus=(${GPUS[*]}) logs=$LOGDIR"

# 1) Precompute TI references first, sharded by DISTINCT PHYSICS across the GPUs
#    (disjoint cache files -> no write race), then barrier before the runs.
echo "[h200] precomputing TI references across ${NGPU} GPU(s) ..."
ref_pids=()
for i in "${!GPUS[@]}"; do
  gpu="${GPUS[$i]}"
  CUDA_VISIBLE_DEVICES="$gpu" python -u scripts/run_wca_phase_diagram.py \
    --config "$CONFIG" --stage "$STAGE" --precompute-references \
    --shard "$i" --num-shards "$NGPU" \
    > "$LOGDIR/${STAGE}_${STAMP}_refs_shard${i}.log" 2>&1 &
  ref_pids+=("$!")
done
for p in "${ref_pids[@]}"; do wait "$p" || { echo "[h200] reference precompute failed" >&2; exit 1; }; done
echo "[h200] references ready."

# 2) Fan the runs out, one process per GPU, disjoint shards.
pids=()
for i in "${!GPUS[@]}"; do
  gpu="${GPUS[$i]}"
  log="$LOGDIR/${STAGE}_${STAMP}_shard${i}_gpu${gpu}.log"
  echo "[h200] launching shard $i/$NGPU on GPU $gpu -> $log"
  CUDA_VISIBLE_DEVICES="$gpu" python -u scripts/run_wca_phase_diagram.py \
    --config "$CONFIG" --stage "$STAGE" --num-gpus "$NGPU" \
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
