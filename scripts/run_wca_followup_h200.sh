#!/usr/bin/env bash
# Safe launcher for the WCA follow-up studies on the shared H200 box.
#
# Runs ONE process per GPU (kernels serialise, so packing gives no speedup) and
# uses at most TWO GPUs. Resumable: valid existing runs are skipped, so re-running
# the same command continues an interrupted sweep.
#
# Usage:
#   bash scripts/run_wca_followup_h200.sh <config> <stage> <gpus_csv> [extra args...]
#
# Examples:
#   bash scripts/run_wca_followup_h200.sh configs/wca_representative.yaml representative 0
#   bash scripts/run_wca_followup_h200.sh configs/wca_representative.yaml representative 0,4
#   bash scripts/run_wca_followup_h200.sh configs/wca_equal_compute.yaml equal_compute 0,4
#   bash scripts/run_wca_followup_h200.sh configs/wca_frozen_bias.yaml frozen_bias 0
#
# For frozen_bias, the source study (results/wca_representative) must already have
# runs; the launcher does NOT precompute TI references for frozen (they are shared
# with the representative cells and already cached).
set -euo pipefail

CONFIG="${1:?need config path}"
STAGE="${2:?need stage name}"
GPUS_CSV="${3:?need GPU id(s), e.g. 0 or 0,4}"
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

read -r LOGDIR MODE <<< "$(python - "$CONFIG" "$STAGE" <<'PY'
import sys, yaml, os
cfg = yaml.safe_load(open(sys.argv[1])); stage = sys.argv[2]
st = cfg.get("stages", {}).get(stage, {})
mode = st.get("mode", cfg.get("mode", "sample"))
print(os.path.join(cfg["output_root"], "logs"), mode)
PY
)"
mkdir -p "$LOGDIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
echo "[h200] config=$CONFIG stage=$STAGE mode=$MODE gpus=(${GPUS[*]}) logs=$LOGDIR"

# 1) Precompute TI references (sample mode only; frozen reuses cached refs).
if [[ "$MODE" != "frozen" ]]; then
  echo "[h200] precomputing TI references across ${NGPU} GPU(s) ..."
  ref_pids=()
  for i in "${!GPUS[@]}"; do
    gpu="${GPUS[$i]}"
    CUDA_VISIBLE_DEVICES="$gpu" python -u scripts/run_wca_followup.py \
      --config "$CONFIG" --stage "$STAGE" --precompute-references \
      --shard "$i" --num-shards "$NGPU" \
      > "$LOGDIR/${STAGE}_${STAMP}_refs_shard${i}.log" 2>&1 &
    ref_pids+=("$!")
  done
  for p in "${ref_pids[@]}"; do wait "$p" || { echo "[h200] reference precompute failed" >&2; exit 1; }; done
  echo "[h200] references ready."
fi

# 2) Fan the runs out, one process per GPU, disjoint shards.
pids=()
for i in "${!GPUS[@]}"; do
  gpu="${GPUS[$i]}"
  log="$LOGDIR/${STAGE}_${STAMP}_shard${i}_gpu${gpu}.log"
  echo "[h200] launching shard $i/$NGPU on GPU $gpu -> $log"
  CUDA_VISIBLE_DEVICES="$gpu" python -u scripts/run_wca_followup.py \
    --config "$CONFIG" --stage "$STAGE" --max-gpus "$NGPU" \
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
