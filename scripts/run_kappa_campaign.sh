#!/usr/bin/env bash
set -u
cd /home/zheyuanlai/reaction-coordinate-wfr
while pgrep -f "scripts/exp_arms.py" > /dev/null; do sleep 30; done
# E9: the transport-rate trade-off, and whether a correct lift escapes it.
# kappa sets how fast WFR moves z per epoch, i.e. how little fiber relaxation
# happens per unit of transport.  A naive lift should trace a U: too slow wastes
# budget on coverage, too fast pays lift bias.  A correct lift has no bias term,
# so it should have no U.
for K in 0.25 0.5 1.0 2.0 4.0 8.0; do
python scripts/exp_arms.py --system CHANNEL --a 0.6 --k 1.4 --steps 100000 --seeds 16 \
    --arms wfr_cart wfr_minnorm wfr_adiab --kappa $K --tag _kap$K \
    >> results/logs/E9_kappa.log 2>&1
done
echo "=== E9 DONE ==="
