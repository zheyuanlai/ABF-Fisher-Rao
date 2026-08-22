#!/usr/bin/env bash
set -u
cd /home/zheyuanlai/reaction-coordinate-wfr
while pgrep -f "scripts/exp_secondary_cv.py" > /dev/null; do sleep 20; done
echo "=== E13: design rule, properly stressed (arms batched) ==="
python scripts/exp_secondary_cv.py --A 1.0 --oms_out 0.25 1.0 4.0 \
    --arms ti_cold wfr_naive wfr_promote wfr_both wfr_oracle \
    --steps 100000 --seeds 16 --out results/manifold/design_rule.json \
    >> results/logs/E13_design_rule.log 2>&1
echo "=== E13 DONE ==="
echo "=== E12: bandwidth past the edge of the E10 grid ==="
for B in 0.10 0.16 0.24 0.36; do
  python scripts/exp_arms.py --system CHANNEL --a 0.6 --k 1.4 --steps 100000 \
      --seeds 16 --arms wfr_fit_decay --fit_decay 0.999 --fit_bw_z $B \
      --acc_reset_at 0.5 --tag _scr_d0.999_b${B} \
      >> results/logs/E12_bw_extend.log 2>&1
done
echo "=== E12 DONE ==="
