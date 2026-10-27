# Stage I gates -> reference -> ABF-only budget pilot -> c_ss benchmark, gated in order (PREREGISTRATION §3-5, §8).
set -u
python scripts/lta_loading/validate.py || exit 1
python - <<'PY' || { echo "STAGE1_GATES_FAILED"; exit 2; }
import json, sys
v = json.load(open("results/lta_loading/validation/validation.json"))
bad = [k for k, x in v.items() if not x.get("passed")]
print("gates:", {k: x.get("passed") for k, x in v.items()})
sys.exit(1 if bad else 0)
PY
python scripts/lta_loading/reference.py run || exit 3
python - <<'PY' || { echo "REFERENCE_NOT_ACCEPTED"; exit 4; }
import json, sys
s = json.load(open("results/lta_loading/reference/reference.json"))
ok = s["F_ref_se_rms"] <= 0.03 and s["mf_vs_density_rms"] <= 0.05
print("reference acceptance:", ok, s["F_ref_se_rms"], s["mf_vs_density_rms"])
sys.exit(0 if ok else 1)
PY
python scripts/lta_loading/run_batch.py --method abf --N 256 --seeds 30200-30207 --n-steps 1200000 \
    --out results/lta_loading/pilot/abf_N256.npz || exit 5
python scripts/lta_loading/analyze.py pilot || exit 6
python scripts/lta_loading/bench.py || exit 7
echo STAGE1_DONE
