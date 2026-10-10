#!/usr/bin/env python
"""Amendment 1 (A4) of docs/mechanism/SCIENTIFIC_PLAN.md: every re-run alpha = 1 / lambda = 1 job of the new engine
must equal its equal-budget counterpart bitwise on all shared arrays.

    python scripts/mechanism/compare_alpha1_bitwise.py [--new-root results/mechanism/matched_free_energy/alpha1]
        [--old-root results/equal_budget_v2/gateway] [--out results/mechanism/reuse_gate/alpha1_bitwise.json]

Compares every s<seed>_<method>.npz under <new-root>/N<N>/ with <old-root>/N<N>/ (np.array_equal, equal_nan, same
dtype and shape) over the keys of the OLD file, except the session bookkeeping (cfg_json, meta_json, wall_s,
peak_rss_mb, n_checkpoints_written, resumed_from).  Keys only in the new file (the family engine's extra diagnostics)
are listed, not compared.  Verdict PASS iff every planned job (N 2048 / 512 / 128 x seeds 8100-8131 x abf / fr) exists
in both trees and is bitwise equal.  Exit 1 on any mismatch or missing job.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
EXCLUDED = {"cfg_json", "meta_json", "wall_s", "peak_rss_mb", "n_checkpoints_written", "resumed_from"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def compare(new, old):
    with np.load(new, allow_pickle=False) as zn, np.load(old, allow_pickle=False) as zo:
        kn, ko = set(zn.files), set(zo.files)
        mism, missing = [], sorted(k for k in ko - kn if k not in EXCLUDED)
        compared = 0
        for k in sorted(ko & kn):
            if k in EXCLUDED:
                continue
            a, b = zn[k], zo[k]
            compared += 1
            if a.dtype != b.dtype or a.shape != b.shape or not np.array_equal(a, b, equal_nan=a.dtype.kind in "fc"):
                mism.append(k)
        return dict(compared=compared, mismatched=mism, missing_in_new=missing, extra_in_new=sorted(kn - ko))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-root", default=os.path.join(ROOT, "results", "mechanism", "matched_free_energy", "alpha1"))
    ap.add_argument("--old-root", default=os.path.join(ROOT, "results", "equal_budget_v2", "gateway"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "mechanism", "reuse_gate", "alpha1_bitwise.json"))
    ap.add_argument("--N", type=int, nargs="*", default=[2048, 512, 128])
    ap.add_argument("--seeds", type=int, nargs="*", default=list(range(8100, 8132)))
    a = ap.parse_args()
    jobs, bad = [], 0
    for N in a.N:
        for s in a.seeds:
            for m in ("abf", "fr"):
                pn = os.path.join(a.new_root, f"N{N}", f"s{s}_{m}.npz")
                po = os.path.join(a.old_root, f"N{N}", f"s{s}_{m}.npz")
                rec = dict(N=N, seed=s, method=m, new=os.path.relpath(pn, ROOT), old=os.path.relpath(po, ROOT))
                if not (os.path.exists(pn) and os.path.exists(po)):
                    rec.update(status="MISSING", new_exists=os.path.exists(pn), old_exists=os.path.exists(po))
                    bad += 1
                else:
                    r = compare(pn, po)
                    ok = not r["mismatched"] and not r["missing_in_new"]
                    rec.update(status="EQUAL" if ok else "MISMATCH", old_sha256=sha(po), new_sha256=sha(pn), **r)
                    bad += 0 if ok else 1
                jobs.append(rec)
    out = dict(rule="docs/mechanism/SCIENTIFIC_PLAN.md Amendment 1 (A4): every re-run alpha=1/lambda=1 job bitwise equal "
                    "to its equal-budget counterpart on all shared arrays (session bookkeeping excluded)",
               generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), n_jobs=len(jobs),
               n_equal=sum(j["status"] == "EQUAL" for j in jobs), verdict="PASS" if bad == 0 else "FAIL",
               excluded_keys=sorted(EXCLUDED), jobs=jobs)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"{out['verdict']}: {out['n_equal']}/{out['n_jobs']} jobs bitwise equal -> {a.out}")
    for j in jobs:
        if j["status"] != "EQUAL":
            print("  ", j["N"], j["seed"], j["method"], j["status"], j.get("mismatched", ""), j.get("missing_in_new", ""))
    sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
