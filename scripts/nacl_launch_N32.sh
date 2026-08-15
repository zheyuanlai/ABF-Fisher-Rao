#!/usr/bin/env bash
# Launch the N=32 screen cell -- the ONLY remaining NaCl cell that can return a verdict.
#
# N=16 and N=8 are excluded by arithmetic, not by measurement: targets partition so Q*_k <= 1,
# hence lambda_k = N Q*_k <= N, so no cell with N <= 16 reaches the frozen lambda >= 16 power
# threshold for any state, whatever the sampling does. See results/nacl/RESULT_N64.md.
#
# Runs from the PINNED WORKTREE at 53dfb30 -- the same sampler that produced N=64, so the two
# cells share a data-generating process. Do NOT re-pin to HEAD: the delta is one diagnostic
# print (verified by diff), and re-pinning would buy nothing and cost ladder homogeneity.
# Analysis is separate and runs at HEAD, where the Gate C power guard lives.
#
# Refuses to launch unless nacl_preflight.py passes, which includes re-reading the governing
# compute clause out of the preregistration. Requires GPU 2 to be idle: tau_perp must have
# exited (one process per GPU).
#
#   bash scripts/nacl_launch_N32.sh
set -euo pipefail

MAIN=/home/zheyuanlai/ABF-Fisher-Rao
WORKTREE=/home/zheyuanlai/ABF-Fisher-Rao-nacl-run
PY=/home/zheyuanlai/miniconda3/envs/abffr/bin/python
DEVICE=2
OUT="$MAIN/results/nacl/screen_N32"
LOG="$MAIN/results/nacl/screen_N32_run.log"

"$PY" "$MAIN/scripts/nacl_preflight.py" --device "$DEVICE" --stage screen_N32 --out "$OUT"

cd "$WORKTREE"
setsid nohup env CUDA_VISIBLE_DEVICES="$DEVICE" \
  bash -c "exec '$PY' -u scripts/nacl_screen.py --cells 32 --out '$OUT'" \
  > "$LOG" 2>&1 < /dev/null &

sleep 5
echo "launched PID $! -> $LOG"
echo "seeds 4000-4007, N=32, T=3.125 ns, 1562500 steps, 256 walkers"
echo "resume after an interruption with:  --resume  (checkpoints ~every 20 min)"
