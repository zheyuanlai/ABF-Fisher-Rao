#!/usr/bin/env bash
# Serial driver: one GPU job at a time (the exact-conditional interpolation is
# memory-hungry and this box shares its GPU with other users' jobs).
set -u
cd /home/zheyuanlai/reaction-coordinate-wfr
mkdir -p results/logs results/manifold
for cfg in "0.6 1.4" "0.3 1.4" "0.6 0.7"; do
  set -- $cfg
  echo "=== E2a a=$1 k=$2 ==="
  python scripts/exp_lift.py a --system MFIB --omega 1.0 --a $1 --k $2 \
      > results/logs/E2a_MFIB_a$1_k$2.log 2>&1
done
echo "=== E3 timescale ==="
python scripts/exp_timescale.py > results/logs/E3_timescale.log 2>&1
echo "=== DONE ==="
