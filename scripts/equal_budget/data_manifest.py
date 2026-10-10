#!/usr/bin/env python
"""Content manifest of the equal-budget raw data (the gitignored *.npz files; docs/equal_budget/DATA_LOCATIONS.md).

    python scripts/equal_budget/data_manifest.py [--out results/equal_budget_v2/DATA_MANIFEST.json]
    python scripts/equal_budget/data_manifest.py --check [results/equal_budget_v2/DATA_MANIFEST.json]

Walks, read-only,
  results/equal_budget_v2/{gateway,lta_T300,lta_T150}/N*/s*_*.npz    raw per-run files of the production ladders
  results/equal_budget_v2/references/*.npz                           frozen references
and records per file: repo-relative path, bytes, sha256 of the file content, mtime (UTC), whether git tracks it,
and for a run file the (system, N, seed, method) parsed from its path plus engine / engine_sha256 / git_commit /
git_dirty / status read from its meta_json (a single small zip member; the arrays are not read).  A file whose
name ends in .ckpt.npz is an engine checkpoint (a run still in progress) and is listed with kind "checkpoint".
The summary gives, per system, file counts, total bytes, the engine versions and statuses seen, and the
(N, seed, method) jobs that the system's production config plans but that have no complete run file.

No other directory is visited (in particular not the confirm_best_* trees).  The manifest is written atomically.
--check re-hashes the same globs and compares them with an existing manifest: changed / missing / new files are
printed and the exit code is 1 if there is any difference, else 0.
"""
import argparse
import datetime
import glob
import hashlib
import json
import os
import re
import subprocess
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
BASE = os.path.join(ROOT, "results", "equal_budget_v2")
DEFAULT_OUT = os.path.join(BASE, "DATA_MANIFEST.json")
# out_dir -> production config (configs/equal_budget_v2/), as in run_ladder.SYSTEMS
SYSTEMS = {"gateway": "gateway_production.json", "lta_T300": "lta_300K_production.json",
           "lta_T150": "lta_150K_production.json"}
META_KEYS = ("engine", "engine_sha256", "git_commit", "git_dirty", "status")
RUN_RE = re.compile(r"^N(\d+)/s(\d+)_([a-z]+)\.npz$")


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def git_tracked():
    try:
        out = subprocess.run(["git", "-C", ROOT, "ls-files", "-z", "results/equal_budget_v2"], check=True,
                             capture_output=True).stdout
        return set(p for p in out.decode().split("\0") if p)
    except (OSError, subprocess.CalledProcessError):
        return None


def files():
    """(class, out_dir or None, absolute path) of every file the manifest covers, sorted."""
    out = []
    for d in SYSTEMS:
        for p in sorted(glob.glob(os.path.join(BASE, d, "N*", "s*_*.npz"))):
            out.append(("run", d, p))
    for p in sorted(glob.glob(os.path.join(BASE, "references", "*.npz"))):
        out.append(("reference", None, p))
    return out


def entry(cls, d, p, tracked):
    rel = os.path.relpath(p, ROOT)
    st = os.stat(p)
    e = {"path": rel, "bytes": st.st_size, "sha256": sha256(p),
         "mtime_utc": datetime.datetime.fromtimestamp(st.st_mtime, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
         "in_git": None if tracked is None else rel in tracked}
    if cls == "reference":
        e["kind"] = "reference"
        try:
            with np.load(p, allow_pickle=False) as z:
                e["arrays"] = sorted(z.files)
            e["status"] = "reference"
        except Exception as exc:  # noqa: BLE001 - record, do not stop the walk
            e["status"] = f"unreadable: {type(exc).__name__}: {exc}"
        return e
    e["system"] = d
    if p.endswith(".ckpt.npz"):
        e["kind"], e["status"] = "checkpoint", "checkpoint (run in progress or interrupted)"
        return e
    e["kind"] = "run"
    m = RUN_RE.match(os.path.relpath(p, os.path.join(BASE, d)))
    if m:
        e["N"], e["seed"], e["method"] = int(m.group(1)), int(m.group(2)), m.group(3)
    try:
        with np.load(p, allow_pickle=False) as z:
            meta = json.loads(str(z["meta_json"]))
        for k in META_KEYS:
            e[k] = meta.get(k)
        if meta.get("N") is not None and e.get("N") is not None and int(meta["N"]) != e["N"]:
            e["status"] = f"{e['status']} | PATH/META N MISMATCH ({meta['N']})"
        if meta.get("seed") is not None and e.get("seed") is not None and int(meta["seed"]) != e["seed"]:
            e["status"] = f"{e['status']} | PATH/META seed MISMATCH ({meta['seed']})"
        if meta.get("method") is not None and e.get("method") is not None and str(meta["method"]) != e["method"]:
            e["status"] = f"{e['status']} | PATH/META method MISMATCH ({meta['method']})"
    except Exception as exc:  # noqa: BLE001
        e["status"] = f"unreadable: {type(exc).__name__}: {exc}"
    return e


def planned(d):
    """(N, seed, method) planned by the production config of out_dir d (run_ladder.job_list semantics)."""
    with open(os.path.join(ROOT, "configs", "equal_budget_v2", SYSTEMS[d])) as fh:
        P = json.load(fh)
    return {(N, s, m) for N in P["N_ladder"] for s in P["seeds"] for m in (("abf",) if N == 1 else ("abf", "fr"))}


def summarise(entries):
    S = {}
    for d in SYSTEMS:
        E = [e for e in entries if e.get("system") == d]
        runs = [e for e in E if e["kind"] == "run"]
        complete = {(e["N"], e["seed"], e["method"]) for e in runs if e.get("status") == "complete" and "N" in e}
        P = planned(d)
        S[d] = {"n_files": len(E), "n_run_files": len(runs),
                "n_checkpoint_files": sum(1 for e in E if e["kind"] == "checkpoint"),
                "total_bytes": sum(e["bytes"] for e in E),
                "n_planned_by_production_config": len(P), "n_complete_planned": len(P & complete),
                "missing_planned": sorted([list(x) for x in P - complete]),
                "unplanned_complete": sorted([list(x) for x in complete - P]),
                "by_N": {str(N): sum(1 for e in runs if e.get("N") == N) for N in sorted({e["N"] for e in runs if "N" in e})},
                "engines": {f"{k[0]} sha256 {k[1]}": sum(1 for e in runs if (str(e.get("engine")), str(e.get("engine_sha256"))) == k)
                            for k in sorted({(str(e.get("engine")), str(e.get("engine_sha256"))) for e in runs})},
                "statuses": {str(k): sum(1 for e in runs if str(e.get("status")) == k) for k in sorted({str(e.get("status")) for e in runs})},
                "git_commits": {str(k): sum(1 for e in runs if str(e.get("git_commit")) == k) for k in sorted({str(e.get("git_commit")) for e in runs})},
                "n_in_git": sum(1 for e in E if e.get("in_git"))}
    R = [e for e in entries if e["kind"] == "reference"]
    S["references"] = {"n_files": len(R), "total_bytes": sum(e["bytes"] for e in R), "n_in_git": sum(1 for e in R if e.get("in_git"))}
    S["all"] = {"n_files": len(entries), "total_bytes": sum(e["bytes"] for e in entries)}
    return S


def build(out):
    tracked = git_tracked()
    F = files()
    entries = []
    for i, (cls, d, p) in enumerate(F):
        entries.append(entry(cls, d, p, tracked))
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(F)} hashed", flush=True)
    try:
        head = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        head = None
    M = {"created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
         "generator": "scripts/equal_budget/data_manifest.py", "repo_head": head,
         "globs": [f"results/equal_budget_v2/{d}/N*/s*_*.npz" for d in SYSTEMS] + ["results/equal_budget_v2/references/*.npz"],
         "hash": "sha256 of the whole file", "summary": summarise(entries), "files": entries}
    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(M, f, indent=1)
        f.write("\n")
    os.replace(tmp, out)
    for k, v in M["summary"].items():
        extra = "" if k in ("references", "all") else (f", complete {v['n_complete_planned']}/{v['n_planned_by_production_config']} planned, "
                                                        f"checkpoints {v['n_checkpoint_files']}, engines {v['engines']}")
        print(f"{k:10s}: {v['n_files']} files, {v['total_bytes'] / 1e6:.1f} MB{extra}")
    print(f"wrote {os.path.relpath(out, ROOT)}")


def check(path):
    M = json.load(open(path))
    old = {e["path"]: e for e in M["files"]}
    cur = {os.path.relpath(p, ROOT): p for _, _, p in files()}
    changed = [r for r in sorted(set(old) & set(cur))
               if os.path.getsize(cur[r]) != old[r]["bytes"] or sha256(cur[r]) != old[r]["sha256"]]
    missing, new = sorted(set(old) - set(cur)), sorted(set(cur) - set(old))
    for tag, L in (("CHANGED", changed), ("MISSING", missing), ("NEW", new)):
        for r in L:
            print(f"{tag:8s} {r}")
    print(f"checked {len(set(old) & set(cur))} files against {os.path.relpath(path, ROOT)} ({M.get('created_utc')}): "
          f"{len(changed)} changed, {len(missing)} missing, {len(new)} new")
    return 1 if (changed or missing or new) else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--check", nargs="?", const=DEFAULT_OUT, default=None, metavar="MANIFEST")
    a = ap.parse_args()
    if a.check:
        sys.exit(check(a.check))
    build(a.out)


if __name__ == "__main__":
    main()
