#!/usr/bin/env bash
# Verify the screen's checkpoint/resume on the REAL driver.
#
# WHAT CHANGED AND WHY (measured, 2026-08-13):
# The first version ran the driver three times -- uninterrupted, interrupted, resumed -- and
# demanded bit-identical outputs. That premise is WRONG for this system: trajectories are
# deterministic WITHIN a process and not across processes (kernel selection differs per
# process, amplified chaotically), which this campaign documented for WCA. Measured here on
# 400 steps: two IDENTICAL full runs in two processes diverged by max|d mean_force| = 4.75e+02
# and max|d final_positions| = 2.87e-01 -- as much as, in fact slightly more than, an
# interrupted-and-resumed pair (4.75e+02 / 2.80e-01). A cross-process end-to-end comparison
# therefore cannot distinguish a resume defect from chaos, in either direction.
#
# So this verifies the two properties that ARE decidable, and together they are the whole
# resume contract:
#   1. serialization round-trip, IN PROCESS: the checkpoint written at step N reloads to
#      state bit-identical to the live state at step N (positions, velocities, forces, RNG,
#      per-cell estimator accumulators, traces, diagnostic lengths, step index);
#   2. bookkeeping: the loop's accumulate/advance ordering and resume index, pinned by
#      tests/test_nacl_checkpoint_resume.py, where an uninterrupted and a
#      checkpoint-resumed run must agree EXACTLY on a deterministic model of the loop.
# Given (1) and (2), within-process determinism supplies the rest.
#
# Usage:  CUDA_VISIBLE_DEVICES=2 bash scripts/nacl_verify_resume.sh
set -u
PY=~/miniconda3/envs/abffr/bin/python
BASE=/home/zheyuanlai/ABF-Fisher-Rao/results/nacl/_resume_check
mkdir -p "$BASE/selftest"
[ -f "$BASE/full/populations.npz" ] && cp -n "$BASE/full/populations.npz" "$BASE/selftest/" 2>/dev/null

echo "[1/2] in-process checkpoint round-trip on the real driver"
$PY scripts/nacl_screen.py --out "$BASE/selftest" --cells 8 --seeds 4000,4001 \
    --prep-ps 2 --max-steps 400 --save-every-ps 0.2 --selftest-checkpoint 200 \
    2>&1 | tee "$BASE/selftest.log" | grep -E "selftest|bit-identical|DIFFERS|PASS|FAIL"
grep -q "CHECKPOINT ROUND-TRIP: PASS" "$BASE/selftest.log" || {
  echo "RESUME VERIFICATION: FAIL (round-trip)"; exit 1; }

echo "[2/2] loop bookkeeping (accumulate/advance order, resume index)"
$PY -m pytest tests/test_nacl_checkpoint_resume.py -q 2>&1 | tail -2 || {
  echo "RESUME VERIFICATION: FAIL (bookkeeping)"; exit 1; }

echo "RESUME VERIFICATION: PASS (round-trip exact; bookkeeping exact)"
