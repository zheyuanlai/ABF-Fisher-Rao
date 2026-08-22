#!/usr/bin/env bash
set -u
cd /home/zheyuanlai/reaction-coordinate-wfr
while pgrep -f "scripts/exp_arms.py" > /dev/null; do sleep 30; done
# Step 2: the two knobs that set the learned lift's floor.  Warm-up discard is ON
# throughout (reset at 0.5) because E8 showed it is worth -30% on its own.
for D in 1.0 0.9995 0.999 0.997 0.99; do
  for B in 0.015 0.03 0.06; do
    python scripts/exp_arms.py --system CHANNEL --a 0.6 --k 1.4 --steps 100000 \
        --seeds 16 --arms wfr_fit_decay --fit_decay $D --fit_bw_z $B \
        --acc_reset_at 0.5 --tag _scr_d${D}_b${B} \
        >> results/logs/E10_lift_screen.log 2>&1
  done
done
echo "=== E10 DONE ==="
