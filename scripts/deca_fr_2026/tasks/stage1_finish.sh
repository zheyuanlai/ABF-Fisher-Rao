#!/usr/bin/env bash
# Analysis + figures of Stage I (validation) and Stage II-A (estimator pilot), once the reference exists.
set -euo pipefail
python scripts/deca_fr_2026/analyze.py --config configs/deca_fr_2026/validation.yaml
python scripts/deca_fr_2026/plot.py --config configs/deca_fr_2026/validation.yaml
python scripts/deca_fr_2026/analyze.py --config configs/deca_fr_2026/estimator_pilot.yaml
python scripts/deca_fr_2026/plot.py --config configs/deca_fr_2026/estimator_pilot.yaml
echo STAGE1_FINISH_OK
