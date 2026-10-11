#!/usr/bin/env bash
# Formal V-B1/V-B2/V-B3/V-B5 correctness gates of Experiment B (PREREGISTRATION.md §2.3, Amendments 1-2).
# Correctness only (no timing): runs on one 16-core allocation, sequentially. Usage: expB_run_vb_gates.sh CPUS
set -u
CPUS=${1:-64-79}
cd "$(dirname "$0")/../.."
export PYTHONPATH=src CUDA_VISIBLE_DEVICES="" NUMBA_CACHE_DIR=/home/zheyuanlai/.cache/numba_final_par NUMBA_NUM_THREADS=16
OUT=results/final_campaign/expB/validation
mkdir -p $OUT
PY="taskset -c $CPUS conda run --no-capture-output -n abffr python -m"
t0=$(date +%s)
for N in 16 64 512; do for M in abf fr; do
  $PY parallel.gateway_parallel --vb1 --N $N --method $M --threads 1,4,16 --out $OUT/gw_vb1_N${N}_${M}.json; echo "gw vb1 N$N $M exit $?"
  $PY parallel.gateway_parallel --vb2 --N $N --method $M --threads 1,2,4,8,16 --out $OUT/gw_vb2_N${N}_${M}.json; echo "gw vb2 N$N $M exit $?"
done; done
for N in 16 64 512; do
  $PY parallel.gateway_parallel --vb5 --N $N --threads 16 --vb5-criterion event --amendment EXPA-CHECK6-EVENT-RULE --out $OUT/gw_vb5_N${N}.json; echo "gw vb5 N$N exit $?"
done
$PY parallel.lta_parallel --vb1 --out $OUT/lta; echo "lta vb1 exit $?"
$PY parallel.lta_parallel --vb2 --out $OUT/lta; echo "lta vb2+vb5 exit $?"
$PY parallel.lta_parallel --vb3 --out $OUT/lta; echo "lta vb3 exit $?"
t1=$(date +%s); echo "total wall $((t1-t0)) s on 16 cores"
