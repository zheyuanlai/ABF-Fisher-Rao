# Confirmation (PREREGISTRATION §7): the frozen selection, fresh seeds 32000-32063.
set -u
read B NA NF < <(python -c "
import json
b = json.load(open('results/lta_loading/pilot/budget.json'))['B']
s = json.load(open('results/lta_loading/ladder/ladder.json'))['selection']
print(b, s['best_abf_N'], s['best_fr_N'])")
echo "B=$B best ABF N=$NA best FR N=$NF"
python scripts/lta_loading/run_batch.py --method abf --N $NA --seeds 32000-32063 --n-steps $((B / NA)) \
    --out results/lta_loading/confirm/abf_N$NA.npz || exit 1
python scripts/lta_loading/run_batch.py --method fr --N $NF --seeds 32000-32063 --n-steps $((B / NF)) \
    --out results/lta_loading/confirm/fr_N$NF.npz || exit 2
python scripts/lta_loading/analyze.py confirm || exit 3
echo CONFIRM_DONE
