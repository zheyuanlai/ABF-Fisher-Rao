#!/usr/bin/env bash
# Resume the ethane/ZIF-8 chain after the ABF arm was started EARLY, concurrently
# with the FR safety calibration.
#
# That overlap is sound, not a shortcut: the ABF arm never calls the birth-death
# step, so its trajectory is bit-identical for any fr_rate (verified directly, in
# two separate processes, before this was used).  The FR arm still cannot start
# until the safety ladder has frozen the rate -- the gate is intact.
#
#   CUDA_VISIBLE_DEVICES=3 nohup bash scripts/zif8_resume.sh 300 &
set -u
T="${1:-300}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT" || exit 1
LOG="results/uniform_campaign/zif8/logs"; mkdir -p "$LOG"
SEL="results/uniform_campaign/zif8/calibration/fr_rate_selection_T${T}.json"
export PYTHONUNBUFFERED=1

say () { echo "[$(date +%H:%M:%S)] $*"; }

say "waiting for the FR safety calibration to freeze a rate"
while pgrep -f "calibrate_zif8_fr.py --temperature $T" >/dev/null; do sleep 60; done
[ -f "$SEL" ] || { say "FAIL calibration produced no selection file"; exit 1; }
RATE=$(python -c "import json;print(json.load(open('$SEL'))['selected'])")
say "calibration done, selected fr_rate = $RATE"
[ "$RATE" = "None" ] && { say "FAIL no safe FR rate -- the two-arm run is refused"; exit 1; }

say "START fr_uniform arm (the ABF arm runs concurrently or is already done)"
if python -u scripts/run_zif8_uniform.py --temperature "$T" --only fr_uniform \
      > "$LOG/production_fr_T${T}.log" 2>&1; then
  say "OK    fr_uniform arm"
else
  say "FAIL  fr_uniform arm (see $LOG/production_fr_T${T}.log)"; tail -20 "$LOG/production_fr_T${T}.log"; exit 1
fi

say "waiting for the ABF arm to finish"
while pgrep -f "run_zif8_uniform.py --temperature $T --only abf" >/dev/null; do sleep 60; done
for m in abf fr_uniform; do
  [ -f "results/uniform_campaign/zif8/production_T${T}/${m}.npz" ] || {
    say "FAIL missing ${m}.npz"; exit 1; }
done

for step in analyze_uniform_zif8 plot_uniform_zif8; do
  say "START $step"
  if python -u "scripts/$step.py" --temperature "$T" > "$LOG/${step}_T${T}.log" 2>&1; then
    say "OK    $step"; grep -vE "^W0|dynamo|recompil" "$LOG/${step}_T${T}.log" | tail -22
  else
    say "FAIL  $step"; tail -25 "$LOG/${step}_T${T}.log"; exit 1
  fi
done
say "ethane/ZIF-8 T=$T complete"
