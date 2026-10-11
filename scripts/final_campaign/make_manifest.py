#!/usr/bin/env python
"""Final campaign data manifest: every artifact of the campaign with size, sha256 and git status.

    python scripts/final_campaign/make_manifest.py      # -> results/final_campaign/DATA_MANIFEST.json (+ .md summary)

Scope: configs/final_campaign, scripts/final_campaign, src/parallel, tests/test_final_*.py, tests/test_parallel_*.py,
docs/final_campaign, docs/mechanism/EXPA_*, results/final_campaign, figures/final_campaign. Raw arrays (*.npz, gitignored)
are listed with their sha256 so a reader can check a re-run or a copy; the file itself stays local. Also records the
compute ledger totals per stage and the HEAD commit.
"""
from __future__ import annotations

import csv
import glob
import hashlib
import json
import os
import subprocess
import time
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
SCOPE = ["configs/final_campaign", "scripts/final_campaign", "src/parallel", "docs/final_campaign",
         "results/final_campaign", "figures/final_campaign"]
SCOPE_GLOBS = ["tests/test_final_*.py", "tests/test_parallel_*.py", "docs/mechanism/EXPA_*.md"]
SKIP_DIRS = {"__pycache__", ".pytest_cache"}


def sha256(path, bufsize=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(bufsize), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args):
    return subprocess.run(["git", "-C", ROOT, *args], capture_output=True, text=True, check=True).stdout


def main():
    tracked = set(git("ls-files", "-z").split("\0"))
    files = []
    for d in SCOPE:
        for dp, dns, fns in os.walk(os.path.join(ROOT, d)):
            dns[:] = [x for x in dns if x not in SKIP_DIRS]
            files += [os.path.join(dp, f) for f in fns]
    for g in SCOPE_GLOBS:
        files += glob.glob(os.path.join(ROOT, g))
    out, by_top = [], defaultdict(lambda: dict(n=0, bytes=0, n_tracked=0))
    for p in sorted(set(files)):
        rel = os.path.relpath(p, ROOT)
        if rel.endswith(".partial") or os.path.islink(p):
            continue
        st = os.stat(p)
        rec = dict(path=rel, bytes=st.st_size, sha256=sha256(p), tracked=rel in tracked,
                   mtime_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(st.st_mtime)))
        out.append(rec)
        top = "/".join(rel.split("/")[:3])
        by_top[top]["n"] += 1
        by_top[top]["bytes"] += st.st_size
        by_top[top]["n_tracked"] += int(rec["tracked"])
    ledger = defaultdict(float)
    with open(os.path.join(ROOT, "results/final_campaign/compute_ledger.csv")) as fh:
        for r in csv.DictReader(fh):
            try:
                ledger[r["stage"]] += float(r["used_core_h"])
            except (TypeError, ValueError):
                pass
    man = dict(version="final_campaign_manifest/1", generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               head=git("rev-parse", "HEAD").strip(), n_files=len(out), n_tracked=sum(r["tracked"] for r in out),
               bytes_total=sum(r["bytes"] for r in out),
               compute_used_core_h=dict(by_stage={k: round(v, 3) for k, v in sorted(ledger.items())},
                                        exp_a=round(sum(v for k, v in ledger.items() if k.startswith("expA")), 2),
                                        exp_b=round(sum(v for k, v in ledger.items() if k.startswith("expB")), 2),
                                        total=round(sum(ledger.values()), 2), gpu_h=0.0),
               by_directory=dict(sorted(by_top.items())), files=out)
    path = os.path.join(ROOT, "results/final_campaign/DATA_MANIFEST.json")
    with open(path, "w") as fh:
        json.dump(man, fh, indent=1)
    lines = ["# Final campaign data manifest (summary)", "",
             f"Generated {man['generated_utc']} at HEAD `{man['head'][:10]}`; full list with sha256: "
             "`results/final_campaign/DATA_MANIFEST.json`.", "",
             f"{man['n_files']} files, {man['bytes_total'] / 1e6:.1f} MB; {man['n_tracked']} tracked in git (raw `.npz` "
             "arrays are gitignored and listed with their sha256).", "",
             "| directory | files | tracked | MB |", "|---|---|---|---|"]
    lines += [f"| `{k}` | {v['n']} | {v['n_tracked']} | {v['bytes'] / 1e6:.2f} |" for k, v in man["by_directory"].items()]
    c = man["compute_used_core_h"]
    lines += ["", f"Compute (used core-h, `compute_ledger.csv`): Experiment A {c['exp_a']}, Experiment B {c['exp_b']}, "
              f"total {c['total']} of the 250 ceiling; 0 GPU-h.", ""]
    with open(os.path.join(ROOT, "results/final_campaign/DATA_MANIFEST.md"), "w") as fh:
        fh.write("\n".join(lines))
    print(f"{man['n_files']} files ({man['n_tracked']} tracked), {man['bytes_total'] / 1e6:.1f} MB; compute {c}")


if __name__ == "__main__":
    main()
