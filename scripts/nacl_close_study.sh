#!/usr/bin/env bash
# Close the NaCl study: run every gate and every audit over the COMPLETE map, in one command.
#
# One command on purpose. The closure needs four things run and read together, and the failure
# mode this repository keeps producing is a step that quietly does not happen -- an audit not
# run reads exactly like an audit that passed. Anything that must be looked at before the
# verdict is written belongs in here, not in a person's memory at 23:40.
#
# ANALYSIS RUNS AT HEAD, NOT AT THE SAMPLER PIN. 53dfb30 is not an ancestor of f88e434, so the
# pinned worktree's nacl_gates.py has no power guard, no cell-map guard and Gate A transposed.
# Run this from the main tree only; the report carries analysis_provenance so a report from the
# superseded tree is detectable by the ABSENCE of that block.
#
#   bash scripts/nacl_close_study.sh
set -euo pipefail

MAIN=/home/zheyuanlai/ABF-Fisher-Rao
PY=/home/zheyuanlai/miniconda3/envs/abffr/bin/python
ALL="$MAIN/results/nacl/screen_all"
N32="$MAIN/results/nacl/screen_N32/cell_N32.npz"
N64="$MAIN/results/nacl/screen_merged/cell_N64.npz"

cd "$MAIN"
[ -f "$N32" ] || { echo "REFUSING: $N32 does not exist -- the N=32 cell has not completed."; exit 1; }
[ -f "$N64" ] || { echo "REFUSING: $N64 missing."; exit 1; }

# The cell must be COMPLETE, not merely present: a partial npz written by an interrupted run
# would be analysed as though it were the preregistered cell.
"$PY" - "$N32" <<'PY'
import sys, numpy as np
d = np.load(sys.argv[1])
n_steps, dt, T_ns = int(d["n_steps"]), float(d["dt_ps"]), float(d["T_ns"])
want = int(round(T_ns * 1000.0 / dt))
S = len(np.asarray(d["seed_labels"]).ravel())
if n_steps < want:
    raise SystemExit(f"REFUSING: cell_N32 has {n_steps} of {want} steps -- run did not finish.")
if S != 8:
    raise SystemExit(f"REFUSING: cell_N32 has {S} seeds, expected the full block of 8.")
print(f"[ok] cell_N32 complete: N={int(d['N'])}, {S} seeds, {n_steps} steps, T={T_ns:.4f} ns")
PY

mkdir -p "$ALL"
ln -sf "$N64" "$ALL/cell_N64.npz"
ln -sf "$N32" "$ALL/cell_N32.npz"

echo
echo "======================= GATES over the COMPLETE map ======================="
"$PY" scripts/nacl_gates.py --screen "$ALL" --ref results/nacl/reference --out "$ALL"

for cell in N32 N64; do
  echo
  echo "=========== CIP/SSIP windowed POWER audit -- $cell ==========="
  echo "(Gate C is unpowered at CIP; this is the statistic that clears or fails to clear it.)"
  "$PY" scripts/nacl_audit_cip_power.py --screen "$ALL" --cell "$cell" \
      --out "results/nacl/screen_all/cip_power_audit_$cell.json"
  echo
  echo "=========== WITHIN-BASIN shape audit -- $cell ==========="
  echo "(Gate C is basin-integrated; a jam preserving the integral is invisible to it.)"
  "$PY" scripts/nacl_audit_within_basin.py --screen "$ALL" --cell "$cell" \
      --out "results/nacl/screen_all/within_basin_audit_$cell.json"
done

echo
echo "======================= READ AGAINST THE PRE-COMMITMENT ======================="
echo "results/nacl/CLOSURE_PRECOMMIT.md fixed all four branches BEFORE the number existed."
echo "Do not decide what the outcome means now -- look up which branch it is."
grep -n "^\*\*(" results/nacl/CLOSURE_PRECOMMIT.md || true
echo
echo "Projected at N=32 (recorded in advance): SSIP lambda ~30.7 powered, CIP ~0.78 unpowered."
echo "A MATERIAL departure from that is itself a finding about the bias-aware target."
