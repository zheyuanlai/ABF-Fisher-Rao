#!/usr/bin/env bash
set -u
cd /home/zheyuanlai/reaction-coordinate-wfr
echo "=== E5: fitted (self-built) adiabatic lift, nonlinear CV ==="
python scripts/exp_arms.py --system CHANNEL --a 0.6 --k 1.4 --steps 100000 --seeds 16 \
    --arms wfr_cart wfr_fit wfr_fit_decay wfr_adiab --tag _fit \
    >> results/logs/E5_fitlift.log 2>&1
echo "=== E5-long: 4x the budget, to test whether the estimate self-corrects ==="
python scripts/exp_arms.py --system CHANNEL --a 0.6 --k 1.4 --steps 400000 --seeds 8 \
    --arms wfr_cart wfr_fit wfr_fit_decay wfr_adiab ti_cold --tag _fitlong \
    >> results/logs/E5_fitlift.log 2>&1
echo "=== E5 DONE ==="
