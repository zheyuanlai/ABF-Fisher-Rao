"""Freeze manifest of the NSCC A100 study: sha256 of the design documents and of every engine / harness source.

    python scripts/nscc_a100/freeze.py write  [--stage design|routing|selection]   # results/nscc_a100/freeze/<stage>.json
    python scripts/nscc_a100/freeze.py check  [--stage design]                     # exit 1 on any change

The 'design' stage is written after Stage I and the launch tuning, before any benchmark / law-check / production run;
later stages add the frozen routing and selection files.  The manifest is the tamper-evident record of what each
stage ran with (the git commit is recorded too, when available).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "results", "nscc_a100", "freeze")

DESIGN = ["docs/nscc_a100/PREREGISTRATION.md", "configs/nscc_a100/campaign.json", "configs/nscc_a100/gpu_launch.json",
          "configs/hybrid_short_T/campaign.json", "configs/equal_budget_v2/gateway_production.json",
          "configs/equal_budget_v2/lta_300K_production.json", "configs/equal_budget_v2/lta_150K_production.json",
          "cache/lta/framework.npz", "results/equal_budget_v2/references/gateway_reference.npz",
          "results/equal_budget_v2/references/lta_T150_reference.npz",
          "results/equal_budget_v2/references/lta_T300_reference.npz"]
CODE_GLOBS = ["src/gpu/*.py", "src/gpu/*.cu", "src/gpu/*.cuh", "src/parallel/*.py", "src/parallel/*.c",
              "src/gateway_ladder_numba.py", "src/lta_ladder_numba.py", "scripts/nscc_a100/nscc_common.py",
              "scripts/nscc_a100/child.py", "scripts/nscc_a100/runner.py", "scripts/nscc_a100/plan.py",
              "scripts/nscc_a100/route.py", "scripts/nscc_a100/bench_gpu.py", "scripts/nscc_a100/validate_gpu*.py",
              "scripts/hybrid_short_T/hst_common.py", "scripts/final_campaign/expB_common.py",
              "scripts/final_campaign/expB_alloc.py", "scripts/equal_budget/eqb_metrics.py",
              "scripts/mechanism/parallel_benchmark.py", "scripts/analyze_gateway_replica_ladder.py"]
STAGE_EXTRA = {"routing": ["results/nscc_a100/routing/routing.json", "results/nscc_a100/bench/summary.json"],
               "selection": ["results/nscc_a100/selection/selection.json"]}


def sha(p):
    with open(os.path.join(ROOT, p), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def files(stage):
    fs = list(DESIGN)
    for g in CODE_GLOBS:
        fs += sorted(os.path.relpath(p, ROOT) for p in glob.glob(os.path.join(ROOT, g)))
    if stage in ("routing", "amendment1", "selection"):
        fs += STAGE_EXTRA["routing"]
    if stage == "selection":
        fs += STAGE_EXTRA["selection"]
    return fs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("write", "check"))
    ap.add_argument("--stage", default="design", choices=("design", "routing", "amendment1", "selection"))
    a = ap.parse_args(argv)
    path = os.path.join(OUT, f"{a.stage}.json")
    cur = {p: sha(p) for p in files(a.stage) if os.path.exists(os.path.join(ROOT, p))}
    if a.cmd == "write":
        if os.path.exists(path):
            sys.exit(f"{path} exists (frozen)")
        os.makedirs(OUT, exist_ok=True)
        rec = dict(stage=a.stage, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), files=cur,
                   git_commit=os.environ.get("ABFFR_GIT_COMMIT"), host=os.uname().nodename,
                   note="working tree synced from the local branch nscc-a100-gpu (uncommitted study files)")
        with open(path, "w") as f:
            json.dump(rec, f, indent=1, sort_keys=True)
        print(f"wrote {path}: {len(cur)} files")
    else:
        rec = json.load(open(path))
        bad = {p: (h, cur.get(p)) for p, h in rec["files"].items() if cur.get(p) != h}
        if bad:
            for p, (h0, h1) in bad.items():
                print(f"CHANGED {p}: {h0[:12]} -> {(h1 or 'missing')[:12]}")
            sys.exit(1)
        print(f"{a.stage}: {len(rec['files'])} files unchanged since {rec['utc']}")


if __name__ == "__main__":
    main()
