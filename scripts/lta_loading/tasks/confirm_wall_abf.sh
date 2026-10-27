# Amendment A3: W_mid-optimal ABF arm on the confirmation seeds.
B=$(python -c "import json; print(json.load(open('results/lta_loading/pilot/budget.json'))['B'])")
python scripts/lta_loading/run_batch.py --method abf --N 128 --seeds 32000-32063 --n-steps $((B / 128)) \
    --out results/lta_loading/confirm_wall/abf_N128.npz
