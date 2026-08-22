#!/usr/bin/env bash
set -u
cd /home/zheyuanlai/reaction-coordinate-wfr
while pgrep -f "scripts/exp_arms.py" > /dev/null; do sleep 30; done
echo "=== E11: secondary-CV lift across four spectator timescales ==="
python scripts/exp_secondary_cv.py --steps 100000 --seeds 16 \
    >> results/logs/E11_secondary_cv.log 2>&1
echo "=== E11 DONE ==="
