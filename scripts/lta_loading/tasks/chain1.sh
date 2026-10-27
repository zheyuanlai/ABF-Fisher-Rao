# Smoke gates first (fail fast), then the full stage 1.
bash scripts/lta_loading/tasks/smoke.sh || exit 1
grep -q ALL_PASSED <(python -c "
import json; v = json.load(open('results/lta_loading/validation_smoke/validation.json'))
print('ALL_PASSED' if all(x.get('passed') for x in v.values()) else 'FAILED')") || { echo SMOKE_FAILED; exit 2; }
bash scripts/lta_loading/tasks/stage1.sh
