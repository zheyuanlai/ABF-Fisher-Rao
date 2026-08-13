#!/usr/bin/env bash
# The NaCl launch ladder (Amendment 15.3). Mechanical, gated, and it STOPS after verification:
# it never launches the TI reference — that is a separate, reviewed action recorded in
# docs/NACL_EXECUTION_STATE.md.
#
#   preflight -> Triton correctness + idle throughput -> TI smoke -> dt gate (15.1)
#             -> checkpoint-resume verification on the real driver -> STOP
#
# A failed step stops the ladder. The fix path is: patch -> test -> commit -> new pinned
# worktree -> restart the ladder from the top. No patch-and-continue.
#
# Run FROM THE PINNED WORKTREE:   bash scripts/nacl_launch_ladder.sh 2
set -u
GPU="${1:-2}"
WT="$(cd "$(dirname "$0")/.." && pwd)"
MAIN=/home/zheyuanlai/ABF-Fisher-Rao
PY=~/miniconda3/envs/abffr/bin/python
LOG="$MAIN/results/nacl/stage1"
mkdir -p "$LOG"
exec > >(tee -a "$LOG/ladder.log") 2>&1

fail () { echo "[LADDER STOP] $1  ($(date -u))"; exit 1; }
step () { echo; echo "==== [$1] $(date -u)"; }

# ---------------- preflight -------------------------------------------------------------
step "preflight"
[ -f "$MAIN/results/nacl/PINNED_COMMIT" ] || fail "no PINNED_COMMIT file; cut the worktree first"
pinned=$(cat "$MAIN/results/nacl/PINNED_COMMIT")
commit=$(git -C "$WT" rev-parse HEAD)
[ "$commit" = "$pinned" ] || fail "worktree at $commit, pin is $pinned"
git -C "$WT" diff --quiet HEAD -- src scripts tests docs \
  || fail "code paths dirty in the pinned worktree"
echo "pinned commit OK: $commit"

( cd "$WT" && $PY -m pytest tests -q -k nacl ) || fail "NaCl test suite"

uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v i="$GPU" '$1==i{print $2}')
wait_idle () {
  while :; do
    used=$(nvidia-smi --query-compute-apps=gpu_uuid,used_memory --format=csv,noheader,nounits \
           | awk -F', ' -v u="$uuid" '$1==u{s+=$2} END{print s+0}')
    if [ "$used" -lt 500 ]; then
      sleep 45
      used2=$(nvidia-smi --query-compute-apps=gpu_uuid,used_memory --format=csv,noheader,nounits \
              | awk -F', ' -v u="$uuid" '$1==u{s+=$2} END{print s+0}')
      [ "$used2" -lt 500 ] && break
    fi
    sleep 60
  done
}
echo "waiting for GPU $GPU ($uuid) to go idle ..."
wait_idle
echo "GPU $GPU idle at $(date -u)"

$PY - "$LOG/launch_manifest.json" "$commit" "$GPU" <<'PYEOF'
import hashlib, json, subprocess, sys, os
out, commit, gpu = sys.argv[1:4]
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
main = "/home/zheyuanlai/ABF-Fisher-Rao"
drv = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip().splitlines()[0]
json.dump(dict(commit=commit, gpu=gpu, driver=drv,
               site_params_sha256=sha(f"{main}/results/nacl/stage0/site_params.npz"),
               baths_sha256=sha(f"{main}/results/nacl/baths/baths.npz"),
               box_manifest=json.load(open(f"{main}/results/nacl/box/box_manifest.json"))["L_nm"],
               dt_rule="Amendment 15.1"), open(out, "w"), indent=2)
print("launch manifest ->", out)
PYEOF
[ $? -eq 0 ] || fail "launch manifest"

export CUDA_VISIBLE_DEVICES="$GPU"
cd "$WT"

# ---------------- 1: Triton correctness + idle throughput -------------------------------
step "benchmark (correctness + timing)"
$PY scripts/nacl_benchmark.py --correctness --timing --out "$LOG" || fail "benchmark"
grep -q '"triton_correctness_pass": true' "$LOG/benchmark.json" \
  && echo "Triton: PASS" || echo "Triton: FAIL (reference will use the tensor path)"

# ---------------- 2: TI smoke (driver proven end to end) --------------------------------
step "TI smoke"
$PY scripts/nacl_ti_torch.py --smoke --dt 0.002 --out "$MAIN/results/nacl/ti_smoke" \
  || fail "TI smoke run"

# ---------------- 3: dt gate under the frozen 15.1 rule ---------------------------------
step "dt gate (Amendment 15.1)"
if grep -qE '"dt_chosen_ps": 0\.[0-9]+' "$LOG/dynamics_gate.json" 2>/dev/null; then
  echo "dt gate already DECIDED -- Amendment 15.1 permits exactly one run, ever; reusing it."
else
  $PY scripts/nacl_dynamics_gate.py || fail "dt gate crashed"
fi
grep -qE '"dt_chosen_ps": 0\.[0-9]+' "$LOG/dynamics_gate.json" \
  || fail "no timestep chosen -- engine defect; NaCl does not run"
echo "dt decision:"; grep -E '"dt_chosen_ps"|"verdict"' "$LOG/dynamics_gate.json"

# ---------------- 4: resume verification on the real driver -----------------------------
step "checkpoint-resume verification"
bash scripts/nacl_verify_resume.sh || fail "resume verification (bit-identity)"

step "LADDER COMPLETE — stopping here by design"
echo "Next permitted action (after review): launch the TI reference; update"
echo "docs/NACL_EXECUTION_STATE.md first. The ladder does not do this."
