# Amendment A3: the W_mid-optimal arms of the ladder (analyze.py ladder, selection.wall_*), confirmation seeds.
set -u
read B NA NF < <(python -c "
import json
b = json.load(open('results/lta_loading/pilot/budget.json'))['B']
s = json.load(open('results/lta_loading/ladder/ladder.json'))['selection']
print(b, s['wall_abf_N'], s['wall_fr_N'])")
echo "A3: B=$B wall-optimal ABF N=$NA FR N=$NF"
python scripts/lta_loading/run_batch.py --method abf --N $NA --seeds 32000-32063 --n-steps $((B / NA)) \
    --out results/lta_loading/confirm_wall/abf_N$NA.npz || exit 1
python scripts/lta_loading/run_batch.py --method fr --N $NF --seeds 32000-32063 --n-steps $((B / NF)) \
    --out results/lta_loading/confirm_wall/fr_N$NF.npz || exit 2
python scripts/lta_loading/analyze.py confirm_wall || exit 3
echo CONFIRM_WALL_DONE
