#!/usr/bin/env bash
# Launch screen half B (seeds 4004-4007) on whichever GPU frees first.
#
# GPU 3 is methane's (2 seeds left, frees ~09:00-13:00 UTC); GPU 2 is running half A
# (frees ~13:36). Whichever comes first gets half B, so no device idles waiting for the other.
#
# Idle test matches PYTHON processes on that device only -- `pgrep -f` alone also matches
# harness wrapper shells whose argv contains the command, which is how a guard becomes a
# permanent hang (measured, 2026-08-13).
set -u
MAIN=/home/zheyuanlai/ABF-Fisher-Rao
WT=/home/zheyuanlai/ABF-Fisher-Rao-nacl-run
PY=~/miniconda3/envs/abffr/bin/python
LOG=$MAIN/results/nacl/screen_B_launch.log
exec >> "$LOG" 2>&1

gpu_busy_mib () {   # $1 = index
  local uuid
  uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v i="$1" '$1==i{print $2}')
  nvidia-smi --query-compute-apps=gpu_uuid,used_memory --format=csv,noheader,nounits \
    | awk -F', ' -v u="$uuid" '$1==u{s+=$2} END{print s+0}'
}

echo "[watch] waiting for GPU 2 or 3 to free ($(date -u))"
TARGET=""
while [ -z "$TARGET" ]; do
  for g in 3 2; do
    if [ "$(gpu_busy_mib "$g")" -lt 500 ]; then
      sleep 45
      [ "$(gpu_busy_mib "$g")" -lt 500 ] && { TARGET=$g; break; }
    fi
  done
  [ -z "$TARGET" ] && sleep 120
done
echo "[watch] GPU $TARGET free at $(date -u); launching half B there"

cd "$WT" || exit 1
pinned=$(cat "$MAIN/results/nacl/PINNED_COMMIT")
commit=$(git -C "$WT" rev-parse HEAD)
[ "$commit" = "$pinned" ] || { echo "[STOP] worktree $commit != pin $pinned"; exit 1; }

CUDA_VISIBLE_DEVICES=$TARGET nohup $PY scripts/nacl_screen.py \
  --out "$MAIN/results/nacl/screen_B" --cells 64 --seeds 4004,4005,4006,4007 --prep-ps 50 \
  >> "$MAIN/results/nacl/screen_B_run.log" 2>&1
echo "[watch] half B exited $? at $(date -u)"
