# Exploration ladder (PREREGISTRATION §6): N in {64..1024}, both arms, seeds 31000-31015, n_steps = B / N.
set -u
B=$(python -c "import json; print(json.load(open('results/lta_loading/pilot/budget.json'))['B'])")
echo "budget B = $B walker-steps"
for N in 256 1024 64 512 128; do
  for M in abf fr; do
    python scripts/lta_loading/run_batch.py --method $M --N $N --seeds 31000-31015 --n-steps $((B / N)) \
        --out results/lta_loading/ladder/N$N/$M.npz || exit 1
  done
done
python scripts/lta_loading/analyze.py ladder || exit 2
echo LADDER_DONE
