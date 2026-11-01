#!/bin/bash
# Validation of the Laplacian selection rule on NSCC: unit tests, ITE framework rebuild, three-arm smokes of every engine.
set -uo pipefail
cd "$LB_ROOT"
echo "== pytest tests/test_lrs_laplacian.py =="
python -m pytest -q tests/test_lrs_laplacian.py 2>&1 | tail -15
rc1=${PIPESTATUS[0]}
echo "== ITE framework (cache/eri/framework_ite_itq3.npz is gitignored; rebuild from the tracked CIF) =="
if [ ! -f cache/eri/framework_ite_itq3.npz ]; then
  cp cache/eri/framework_ite_itq3.json "$LB_LOGS/framework_ite_itq3.json.tracked"
  python scripts/overnight/save_ite.py
  # save_ite.py rewrites the JSON (npz sha differs by zip timestamps); keep the tracked JSON, the smoke compares geometry
  cp "$LB_LOGS/framework_ite_itq3.json.tracked" cache/eri/framework_ite_itq3.json
fi
ls -la cache/eri/framework_ite_itq3.npz
echo "== smoke =="
python scripts/laplacian_baselines/smoke.py
rc2=$?
echo "VALIDATE_EXIT pytest=$rc1 smoke=$rc2"
[ "$rc1" -eq 0 ] && [ "$rc2" -eq 0 ]
