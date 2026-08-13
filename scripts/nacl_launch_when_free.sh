#!/usr/bin/env bash
# Wait for a GPU to be genuinely free, then run the NaCl stages that need an idle device.
#
# The methane session hands over GPU 2 at its seed-5004 boundary (Amendment 14.4).  Rather than
# guess the moment, poll the device's own compute-app list and start when it is empty --
# and record the idle state at launch, because a contended device reads ~28x slow and is
# indistinguishable from a code defect (the campaign's recorded trap).
#
# Usage:  bash scripts/nacl_launch_when_free.sh 2
set -u
GPU="${1:-2}"
PY=~/miniconda3/envs/abffr/bin/python
LOG_DIR=results/nacl/stage1
mkdir -p "$LOG_DIR"

uuid_of_index () { nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v i="$1" '$1==i {print $2}'; }
mem_on_gpu () {
  local uuid; uuid="$(uuid_of_index "$1")"
  nvidia-smi --query-compute-apps=gpu_uuid,used_memory --format=csv,noheader,nounits \
    | awk -F', ' -v u="$uuid" '$1==u {s+=$2} END {print s+0}'
}

echo "[watch] waiting for GPU $GPU to go idle ($(date -u))"
while :; do
  used="$(mem_on_gpu "$GPU")"
  if [ "${used:-1}" -lt 500 ]; then
    # require it to stay idle for two consecutive checks, so we do not race a restart
    sleep 45
    used2="$(mem_on_gpu "$GPU")"
    if [ "${used2:-1}" -lt 500 ]; then
      echo "[watch] GPU $GPU idle (${used2} MiB) at $(date -u)"
      break
    fi
  fi
  sleep 60
done

export CUDA_VISIBLE_DEVICES="$GPU"

echo "[stage] Triton correctness + throughput on an IDLE device"
$PY scripts/nacl_benchmark.py --correctness --timing > "$LOG_DIR/benchmark.log" 2>&1
echo "[stage] benchmark exit=$? ($(date -u))"
tail -20 "$LOG_DIR/benchmark.log"

# The TI driver has compiled but never executed.  Prove it end to end on 2 r-points before the
# real reference commits hours to it -- the expensive failure is a broken driver discovered
# with an idle GPU waiting on it.  --dt is a declared override; this run is not a reference.
echo "[stage] TI driver smoke run (2 r-points, 1 replica, ps blocks)"
$PY scripts/nacl_ti_torch.py --smoke --dt 0.002 --out results/nacl/ti_smoke \
    > "$LOG_DIR/ti_smoke.log" 2>&1
SMOKE=$?
echo "[stage] smoke exit=$SMOKE ($(date -u))"
tail -15 "$LOG_DIR/ti_smoke.log"
if [ "$SMOKE" -ne 0 ]; then
  echo "[stop] TI driver failed its smoke run; not starting the reference"
  exit 1
fi

echo "[stage] dynamics gate (constraints + equipartition, 2 fs vs 1 fs)"
$PY scripts/nacl_dynamics_gate.py > "$LOG_DIR/dynamics_gate.log" 2>&1
echo "[stage] dynamics gate exit=$? ($(date -u))"
tail -12 "$LOG_DIR/dynamics_gate.log"

if ! grep -qE "\"dt_chosen_ps\": 0\.[0-9]" "$LOG_DIR/dynamics_gate.json" 2>/dev/null; then
  echo "[stop] no timestep passed the gate -- NaCl does not run; see dynamics_gate.json"
  exit 1
fi

TRITON=""
if grep -q '"triton_correctness_pass": true' "$LOG_DIR/benchmark.json" 2>/dev/null; then
  TRITON="--triton"
fi
echo "[stage] constrained-TI reference ${TRITON:-(tensor path)} ($(date -u))"
$PY scripts/nacl_ti_torch.py --out results/nacl/ti_torch $TRITON \
    > results/nacl/ti_torch_run.log 2>&1
echo "[stage] TI exit=$? ($(date -u))"
tail -30 results/nacl/ti_torch_run.log
