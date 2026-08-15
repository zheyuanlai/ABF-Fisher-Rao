#!/usr/bin/env bash
# C60 launch ladder -- Amendment 15.3 discipline, adapted from the NaCl ladder.
#
# Preflight -> throughput -> dt gate -> reference smoke -> checkpoint-resume check -> STOP.
# The ladder never launches the reference itself; that is a separate reviewed action
# recorded in docs/C60_EXECUTION_STATE.md.  A failed step stops the ladder; a defect fix
# means a new pinned commit and a restart from the top.
#
# Usage:  bash scripts/c60_launch_ladder.sh
set -euo pipefail
cd "$(dirname "$0")/.."

RESULTS=results/c60
NVLIB=$HOME/miniconda3/envs/abffr/lib/python3.14/site-packages/nvidia/cu13/lib
export LD_LIBRARY_PATH="$NVLIB:${LD_LIBRARY_PATH:-}"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate abffr

echo "== [1/6] preflight =="
PINNED=$(cat "$RESULTS/PINNED_COMMIT")
HEAD=$(git rev-parse HEAD)
if [ "$PINNED" != "$HEAD" ]; then
  echo "FAIL: HEAD $HEAD != pinned $PINNED"; exit 1
fi
if ! git diff --quiet -- src scripts tests docs; then
  echo "FAIL: code tree dirty (src/scripts/tests/docs)"; exit 1
fi
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3)
if [ "$USED" -gt 500 ]; then
  echo "FAIL: GPU 3 not idle (${USED} MiB)"; exit 1
fi
echo "preflight OK: commit $HEAD, GPU 3 idle (${USED} MiB)"

echo "== [2/6] engine gate (test suite) =="
python -m pytest tests/test_c60_engine.py -q 2>&1 | tail -2

echo "== [3/6] idle-device throughput =="
CUDA_VISIBLE_DEVICES=3 python scripts/c60_throughput.py

echo "== [4/6] dt gate =="
CUDA_VISIBLE_DEVICES=3 python scripts/c60_dt_gate.py --phase openmm
CUDA_VISIBLE_DEVICES=3 python scripts/c60_dt_gate.py --phase torch
python scripts/c60_dt_gate.py --phase verdict

echo "== [5/6] reference smoke (1/50th durations) =="
CUDA_VISIBLE_DEVICES=3 python scripts/c60_reference.py --build 1 --smoke

echo "== [6/6] checkpoint-resume verification (smoke, killed and resumed) =="
rm -rf "$RESULTS/reference/build1_smoke"
# generous timeout: torch.compile alone takes minutes; the kill must land mid-run.
# Resumable artifacts, any of which proves the resume path: anchors.npz (phase A done),
# starts_dragged.npz (phase B drags done), checkpoint.pt (main loop), windows.npz (complete).
CUDA_VISIBLE_DEVICES=3 timeout 900 python scripts/c60_reference.py --build 1 --smoke || true
if [ ! -f "$RESULTS/reference/build1_smoke/checkpoint.pt" ] && \
   [ ! -f "$RESULTS/reference/build1_smoke/starts_dragged.npz" ] && \
   [ ! -f "$RESULTS/reference/build1_smoke/anchors.npz" ] && \
   [ ! -f "$RESULTS/reference/build1_smoke/windows.npz" ]; then
  echo "FAIL: no resumable artifact written before the kill"; exit 1
fi
CUDA_VISIBLE_DEVICES=3 python scripts/c60_reference.py --build 1 --smoke
echo "resume completed"

echo "== LADDER COMPLETE: STOP.  Reference launch is a separate reviewed action. =="
