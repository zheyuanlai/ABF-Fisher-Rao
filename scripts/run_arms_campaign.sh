#!/usr/bin/env bash
set -u
cd /home/zheyuanlai/reaction-coordinate-wfr
echo "=== E4a: full arm comparison, nonlinear CV ==="
python scripts/exp_arms.py --system CHANNEL --a 0.6 --k 1.4 --steps 100000 --seeds 16 \
    >> results/logs/E4_arms.log 2>&1
echo "=== E4a-control: SAME potential, LINEAR CV (a=0) ==="
python scripts/exp_arms.py --system CHANNEL --a 0.0 --k 1.4 --steps 100000 --seeds 16 \
    >> results/logs/E4_arms.log 2>&1
echo "=== E4b: burn-in sweep (Version I trade-off) ==="
for NEQ in 0 5 10 15 19; do
  python scripts/exp_arms.py --system CHANNEL --a 0.6 --k 1.4 --steps 100000 --seeds 16 \
      --n_cond 20 --n_eq $NEQ --arms ti_cold wfr_cart wfr_minnorm wfr_adiab \
      >> results/logs/E4_arms.log 2>&1
done
echo "=== E4 DONE ==="
