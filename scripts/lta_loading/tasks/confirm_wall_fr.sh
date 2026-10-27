# Amendment A3: W_mid-optimal FR arm on the confirmation seeds.
B=$(python -c "import json; print(json.load(open('results/lta_loading/pilot/budget.json'))['B'])")
python scripts/lta_loading/run_batch.py --method fr --N 256 --seeds 32000-32063 --n-steps $((B / 256)) \
    --out results/lta_loading/confirm_wall/fr_N256.npz
