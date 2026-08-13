#!/usr/bin/env bash
# End-to-end verification that the screen's checkpoint/resume is exact -- on the REAL driver,
# not on a model of it.
#
# A unit harness can prove the bookkeeping is right and still miss the driver: the thing that
# has to be true is that an interrupted-and-resumed run reproduces an uninterrupted one through
# the actual sampler, estimators and packed-batch retirement logic.  Determinism holds WITHIN a
# process and not across (the WCA finding), so both runs are compared on the same machine, same
# process count, same order -- the paired difference is what is meaningful, and here it must be
# exactly zero.
#
# Usage:  CUDA_VISIBLE_DEVICES=2 bash scripts/nacl_verify_resume.sh
set -u
PY=~/miniconda3/envs/abffr/bin/python
BASE=results/nacl/_resume_check
STEPS=400
STOP=200
rm -rf "$BASE"; mkdir -p "$BASE"

COMMON="--cells 8 --seeds 4000,4001 --prep-ps 2 --max-steps $STEPS --save-every-ps 0.2"

echo "[1/3] uninterrupted reference run ($STEPS steps)"
$PY scripts/nacl_screen.py --out "$BASE/full" $COMMON > "$BASE/full.log" 2>&1 || {
  echo "FAILED: uninterrupted run"; tail -20 "$BASE/full.log"; exit 1; }

echo "[2/3] interrupted run (checkpoint at $STOP, exit)"
cp -r "$BASE/full/populations.npz" "$BASE/" 2>/dev/null || true
mkdir -p "$BASE/split"
cp "$BASE/full/populations.npz" "$BASE/split/" 2>/dev/null || true
$PY scripts/nacl_screen.py --out "$BASE/split" $COMMON --stop-after $STOP \
    --checkpoint-every-steps $STOP > "$BASE/split_a.log" 2>&1 || {
  echo "FAILED: interrupted run"; tail -20 "$BASE/split_a.log"; exit 1; }

echo "[3/3] resumed run (continue to $STEPS)"
$PY scripts/nacl_screen.py --out "$BASE/split" $COMMON --resume \
    > "$BASE/split_b.log" 2>&1 || {
  echo "FAILED: resumed run"; tail -20 "$BASE/split_b.log"; exit 1; }

$PY - "$BASE" <<'EOF'
import sys, numpy as np, glob, os
base = sys.argv[1]
ok = True
for path in sorted(glob.glob(os.path.join(base, "full", "cell_N*.npz"))):
    name = os.path.basename(path)
    a = np.load(path); b = np.load(os.path.join(base, "split", name))
    for key in ("mean_force", "pmf", "eff_counts", "xi_trace", "y_trace",
                "diag_occupancy", "diag_pmf", "final_positions"):
        if key not in a:
            continue
        x, y = np.asarray(a[key]), np.asarray(b[key])
        if x.shape != y.shape:
            print(f"  {name}:{key}  SHAPE {x.shape} vs {y.shape}   FAIL"); ok = False; continue
        d = float(np.abs(x - y).max()) if x.size else 0.0
        flag = "ok" if d == 0.0 else "FAIL"
        if d != 0.0:
            ok = False
        print(f"  {name}:{key:16s} max|diff| = {d:.3e}   {flag}")
print("\nRESUME VERIFICATION:", "PASS (bit-identical)" if ok else "FAIL")
sys.exit(0 if ok else 1)
EOF
