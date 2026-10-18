#!/usr/bin/env python
"""Data manifest of the hybrid short-T study: sha256 and size of every file under results/hybrid_short_T/ (run
records, science npz, telemetry sidecars, benchmark, validation, routing, analysis), written to
results/hybrid_short_T/SHA256SUMS.txt (sha256sum format) + MANIFEST.json (counts, completeness).

    python scripts/hybrid_short_T/manifest.py [--check]      # --check: verify the files against SHA256SUMS.txt

Completeness: every production / confirmation record with status 'complete' must have its npz (replicate 0) whose
sha256 equals the npz_sha256 recorded in its JSON; violations are listed and make the exit status non-zero.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hst_common as H  # noqa: E402

SKIP_SUFFIX = (".log", ".ledger_logged", ".telemetry.jsonl")
SUMS = os.path.join(H.OUT_ROOT, "SHA256SUMS.txt")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def files():
    out = []
    for d, _, fs in os.walk(H.OUT_ROOT):
        for f in fs:
            p = os.path.join(d, f)
            rel = os.path.relpath(p, H.ROOT)
            if f.endswith(SKIP_SUFFIX) or f.startswith(".") or p == SUMS or f == "MANIFEST.json":
                continue
            out.append(rel)
    return sorted(out)


def completeness():
    probs, n = [], 0
    for kind in ("prod", "confirm"):
        root = os.path.join(H.OUT_ROOT, kind)
        for d, _, fs in os.walk(root):
            for f in fs:
                if not f.endswith(".json") or ".failed" in f:
                    continue
                rec = json.load(open(os.path.join(d, f)))
                if rec.get("status") != "complete":
                    continue
                n += 1
                npz = os.path.join(d, f[:-5] + ".npz")
                if not os.path.exists(npz):
                    probs.append(f"{os.path.relpath(npz, H.ROOT)} missing")
                elif rec.get("npz_sha256") and rec["npz_sha256"] != sha(npz):
                    probs.append(f"{os.path.relpath(npz, H.ROOT)} sha256 differs from its record")
    return n, probs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    if a.check:
        bad = []
        for line in open(SUMS):
            h, rel = line.rstrip("\n").split("  ", 1)
            p = os.path.join(H.ROOT, rel)
            if not os.path.exists(p) or sha(p) != h:
                bad.append(rel)
        print(f"{len(bad)} mismatching / missing files" + (f": {bad[:10]}" if bad else ""))
        return 1 if bad else 0
    fl = files()
    with open(SUMS, "w") as fh:
        for rel in fl:
            fh.write(f"{sha(os.path.join(H.ROOT, rel))}  {rel}\n")
    n, probs = completeness()
    by = {}
    for rel in fl:
        k = rel.split(os.sep)[2] if len(rel.split(os.sep)) > 2 else "."
        by[k] = by.get(k, 0) + 1
    man = dict(n_files=len(fl), by_top_dir=by, n_complete_run_records=n, completeness_problems=probs,
               bytes=sum(os.path.getsize(os.path.join(H.ROOT, r)) for r in fl))
    with open(os.path.join(H.OUT_ROOT, "MANIFEST.json"), "w") as fh:
        json.dump(man, fh, indent=1)
    print(json.dumps(man, indent=1))
    return 1 if probs else 0


if __name__ == "__main__":
    sys.exit(main())
