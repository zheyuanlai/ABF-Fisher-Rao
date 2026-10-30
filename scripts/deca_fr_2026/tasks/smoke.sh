#!/usr/bin/env bash
# gdev smoke (<= 2 h): CPU regression tests, A100 throughput, a tiny GPU cell (both arms), a tiny
# reference build + merge.  Everything here is a code-validity check, not a result.
set -euo pipefail
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
echo "=== CPU tests"
CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_deca_fr_2026.py tests/test_deca_correctness_closure.py tests/test_deca_sampler.py tests/test_deca_stage0.py -q -x -p no:cacheprovider
echo "=== throughput"
python scripts/deca_fr_2026/bench.py --sizes 128 512 1024 2048 4096 --steps 200
echo "=== tiny cell"
OUT=results/deca_fr_2026/smoke
rm -rf $OUT
python scripts/deca_fr_2026/run_cell.py --config configs/deca_fr_2026/smoke.yaml --cell tiny --prepare-only --out $OUT
python scripts/deca_fr_2026/run_cell.py --config configs/deca_fr_2026/smoke.yaml --cell tiny --method abf --out $OUT
python scripts/deca_fr_2026/run_cell.py --config configs/deca_fr_2026/smoke.yaml --cell tiny --method fr_uniform --out $OUT
echo "=== tiny reference"
for b in 0 1; do
  python scripts/deca_fr_2026/run_reference.py --build-index $b --out $OUT/reference --n-windows 12 --n-rep 4 --win-lo 1.4 --win-hi 2.6 \
      --n-pull-steps 20000 --n-equil-steps 2000 --n-prod-steps 20000 --checkpoint-every 10000 --sample-every 100
done
python scripts/deca_fr_2026/merge_reference.py --dir $OUT/reference
python scripts/deca_fr_2026/analyze.py --stage smoke --config configs/deca_fr_2026/smoke.yaml --out $OUT --reference $OUT/reference/reference.npz || echo "analysis smoke FAILED (non-fatal in smoke)"
echo SMOKE_OK
