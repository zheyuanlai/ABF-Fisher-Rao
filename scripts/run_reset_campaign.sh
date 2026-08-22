#!/usr/bin/env bash
set -u
cd /home/zheyuanlai/reaction-coordinate-wfr
while pgrep -f "scripts/exp_arms.py" > /dev/null; do sleep 30; done
echo "=== E8: does discarding the self-built lift's warm-up recover the benefit? ==="
for R in 0.5 0.75; do
python scripts/exp_arms.py --system CHANNEL --a 0.6 --k 1.4 --steps 100000 --seeds 16 \
    --arms wfr_cart wfr_fit_decay wfr_adiab --acc_reset_at $R --tag _reset$R \
    >> results/logs/E8_reset.log 2>&1
done
echo "=== E8 DONE ==="
