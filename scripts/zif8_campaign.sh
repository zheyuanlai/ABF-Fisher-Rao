#!/usr/bin/env bash
# Unattended ethane/ZIF-8 campaign chain, in the preregistered order.
#
#   reference  ->  ABF-only screen (classifies)  ->  FR safety calibration
#              ->  two-arm production            ->  analysis + figures
#
# Every stage asserts its own preregistered gates, so the chain STOPS rather
# than proceeding on a failed gate: a discovery-limited screen makes
# run_zif8_uniform refuse to start, and a calibration that finds no safe rate
# does the same.  That is the point -- this script adds no judgement of its
# own, it only removes the waiting.
#
#   CUDA_VISIBLE_DEVICES=3 bash scripts/zif8_campaign.sh 300 [logdir]
set -u
T="${1:-300}"
LOG="${2:-results/uniform_campaign/zif8/logs}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
mkdir -p "$LOG"
export PYTHONUNBUFFERED=1

run () {  # run <name> <script> [args...]
  local name="$1"; shift
  local log="$LOG/${name}_T${T}.log"
  echo "[$(date +%H:%M:%S)] START $name  -> $log"
  if python -u "$@" > "$log" 2>&1; then
    echo "[$(date +%H:%M:%S)] OK    $name"
    tail -n 12 "$log" | grep -vE "^W0|torch._dynamo|recompil|To (log|diagnose)"
  else
    echo "[$(date +%H:%M:%S)] FAIL  $name  (chain stops; see $log)"
    tail -n 25 "$log"
    exit 1
  fi
}

run reference   scripts/run_zif8_reference.py  --temperature "$T"
run screen      scripts/run_zif8_screen.py     --temperature "$T"
run calibration scripts/calibrate_zif8_fr.py   --temperature "$T"
run production  scripts/run_zif8_uniform.py    --temperature "$T"
run analysis    scripts/analyze_uniform_zif8.py --temperature "$T"
run figures     scripts/plot_uniform_zif8.py   --temperature "$T"
echo "[$(date +%H:%M:%S)] campaign complete for T=$T"
