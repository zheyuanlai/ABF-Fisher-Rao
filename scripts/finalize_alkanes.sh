#!/usr/bin/env bash
# Regenerate summaries, figures, tables/macros and recompile the report from whatever
# production data exists (safe to run incrementally as stages complete). No GPU except
# the pentane reference build (already cached) and figure rendering (CPU).
set -uo pipefail
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh && conda activate abffr
CVD="${CUDA_VISIBLE_DEVICES:-}"
export CUDA_VISIBLE_DEVICES=""   # analysis/plots are CPU-only
python scripts/analyze_alkanes.py --config configs/alkanes/production.yaml
python scripts/plot_alkanes.py    --config configs/alkanes/production.yaml --stage production \
    --tuning-root results/alkanes/tuning --report-figdir report/figures
python scripts/make_alkanes_report_assets.py
cd report && tectonic -X compile main.tex 2>&1 | grep -iE "error|undefined|fatal|! " | head
echo "report pages: $(pdfinfo main.pdf 2>/dev/null | awk '/Pages/{print $2}')"
