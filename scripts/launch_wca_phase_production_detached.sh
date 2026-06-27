#!/usr/bin/env bash
# Detached WCA phase-diagram production launcher.
# Safe to run over SSH, then close your laptop: it starts a tmux session.
#
# Usage examples:
#   bash scripts/launch_wca_phase_production_detached.sh
#   GPUS=4 bash scripts/launch_wca_phase_production_detached.sh
#   GPUS=4,7 SESSION=wca_phase_prod bash scripts/launch_wca_phase_production_detached.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ENV_NAME="${ENV_NAME:-abffr}"
GPUS="${GPUS:-4,7}"
SESSION="${SESSION:-wca_phase_production}"
CONFIG="configs/wca_phase_diagram_production.yaml"
STAGE="production"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="results/wca_phase_diagram/production/logs"
OUTLOG="results/wca_phase_diagram/production_detached_${STAMP}.log"
RUN_SCRIPT="results/wca_phase_diagram/production_tmux_${STAMP}.sh"

mkdir -p "$LOGDIR" "$(dirname "$OUTLOG")"

if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: missing $CONFIG. Run this from the ABF-Fisher-Rao repo." >&2
  exit 1
fi

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
if (( ${#GPU_ARR[@]} < 1 || ${#GPU_ARR[@]} > 2 )); then
  echo "ERROR: GPUS must contain one or two GPU IDs, e.g. GPUS=4 or GPUS=4,7" >&2
  exit 1
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "ERROR: tmux is not installed/available. Try running inside an existing tmux session instead." >&2
  exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "ERROR: tmux session '$SESSION' already exists. Attach with: tmux attach -t $SESSION" >&2
  exit 1
fi

echo "[launcher] repo:    $ROOT"
echo "[launcher] env:     $ENV_NAME"
echo "[launcher] GPUs:    $GPUS"
echo "[launcher] session: $SESSION"
echo "[launcher] log:     $OUTLOG"
echo

echo "[launcher] Dry-run summary:"
conda run -n "$ENV_NAME" python scripts/run_wca_phase_diagram.py \
  --config "$CONFIG" \
  --stage "$STAGE" \
  --dry-run \
  --device cuda \
  --num-gpus "${#GPU_ARR[@]}" \
  --batch-size-configs 1

echo
read -r -p "Start detached production run now? Type YES to continue: " CONFIRM
if [[ "$CONFIRM" != "YES" ]]; then
  echo "Aborted."
  exit 0
fi

cat > "$RUN_SCRIPT" <<RUNEOF
#!/usr/bin/env bash
set -uo pipefail
cd "$ROOT"
echo "[tmux] started at \$(date)" | tee -a "$OUTLOG"
conda run -n "$ENV_NAME" bash scripts/run_wca_phase_diagram_h200.sh "$CONFIG" "$STAGE" "$GPUS" --batch-size-configs 1 2>&1 | tee -a "$OUTLOG"
rc=\${PIPESTATUS[0]}
if [[ \$rc -eq 0 ]]; then
  echo "[tmux] production finished; running analysis/figures/report at \$(date)" | tee -a "$OUTLOG"
  conda run -n "$ENV_NAME" python scripts/analyze_wca_phase_diagram.py --config "$CONFIG" --stages "$STAGE" 2>&1 | tee -a "$OUTLOG" || rc=\$?
  if [[ \$rc -eq 0 ]]; then
    conda run -n "$ENV_NAME" python scripts/plot_wca_phase_diagram.py --config "$CONFIG" --stage "$STAGE" --report-figdir report/figures 2>&1 | tee -a "$OUTLOG" || rc=\$?
  fi
  if [[ \$rc -eq 0 ]]; then
    conda run -n "$ENV_NAME" python scripts/make_phase_report_assets.py --summaries results/wca_phase_diagram/production/summaries --tabledir report/tables 2>&1 | tee -a "$OUTLOG" || rc=\$?
  fi
  if [[ \$rc -eq 0 ]]; then
    (cd report && tectonic -X compile main.tex) 2>&1 | tee -a "$OUTLOG" || rc=\$?
  fi
fi
echo "[tmux] finished with rc=\$rc at \$(date)" | tee -a "$OUTLOG"
exit \$rc
RUNEOF
chmod +x "$RUN_SCRIPT"

tmux new-session -d -s "$SESSION" "bash '$ROOT/$RUN_SCRIPT'"

echo
echo "Started detached tmux session: $SESSION"
echo "Attach:    tmux attach -t $SESSION"
echo "Detach:    Ctrl-b then d"
echo "Watch log: tail -f $OUTLOG"
echo "GPU use:   watch -n 5 nvidia-smi"
echo "Runner:    $RUN_SCRIPT"
