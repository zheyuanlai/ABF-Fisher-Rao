#!/bin/bash
# runner / CPU-node smoke test (not study data): a few short bench windows of every CPU backend
set -u
python - <<'PY'
import json, sys
sys.path.insert(0, "scripts/nscc_a100")
import nscc_common as NC
jobs = []
for system in ("gateway", "lta150"):
    for N, be in ((1, "S1"), (16, "P16"), (64, "P8"), (64, "S1"), (256, "P4")):
        jobs.append(NC.make_job("bench", system, "fr" if N > 1 else "abf", N, 1, be, r=0))
with open("/scratch/users/nus/e1297740/abffr_smoke_jobs.jsonl", "w") as f:
    for j in jobs:
        f.write(json.dumps(j) + "\n")
PY
python scripts/nscc_a100/runner.py --jobs /scratch/users/nus/e1297740/abffr_smoke_jobs.jsonl --mode cpu --out-root /scratch/users/nus/e1297740/abffr_smoke
