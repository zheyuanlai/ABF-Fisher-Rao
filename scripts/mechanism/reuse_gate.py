#!/usr/bin/env python
"""The bitwise-reuse gate of the alpha1 / lam1 cells (docs/mechanism/SCIENTIFIC_PLAN.md section 5).

    taskset -c 200-203 python scripts/mechanism/reuse_gate.py [--cell configs/mechanism/cells/matched_free_energy/alpha1.json]
                                                              [--jobs N:SEED:METHOD ...] [--out-dir results/mechanism/reuse_gate]
                                                              [--workers 4] [--dry-run]

The plan: "The alpha = 1 cells re-use results/equal_budget_v2/gateway/N{2048,512,128}/ if and only if the new engine
at alpha = 1 reproduces those files' arrays bitwise for >= 4 re-run jobs (both arms, N = 128 and 2048).  Otherwise
alpha = 1 is re-run."  lambda = 1 reuses the same files under the same condition, so ONE gate record serves both reuse
cells (their configs name it: reuse.gate_record, gate_required_N, gate_min_jobs; scripts/mechanism/make_cell_configs.py).

For every job (default: the cell's first seed at every N of reuse.gate_required_N, both arms = 4 jobs) this re-runs
src/gateway_family_numba.run_job with the reuse cell's engine_cfg (variant alpha, alpha 1, lam 1) on the reused file's
own save grid (its save_step) into <out-dir>/runs/N<N>/s<seed>_<method>.npz, then compares EVERY array of the reused
file with the new one, bitwise (np.array_equal, equal_nan; same dtype and shape), except the session bookkeeping
EXCLUDED (cfg_json / meta_json: text with engine name, timestamps, host; wall_s, peak_rss_mb, n_checkpoints_written,
resumed_from: properties of the process, not of the run).  Arrays only in the new file (the family engine's extra
diagnostics) are listed, not compared.  A reused key missing from the new file is a mismatch.

Writes <out-dir>/reuse_gate.json (schema eqb_family.REUSE_GATE_SCHEMA): verdict PASS iff every job is bitwise equal,
there are >= gate_min_jobs of them, both arms and every N of gate_required_N are covered; per job the reused file's
path and sha256 (eqb_family.reuse_gate_problems re-checks that the reused files are still those files), the new
file's sha256, the compared / excluded / extra keys and the mismatched keys.  The analysis (summary cell.reuse_gate),
the audit (REUSE_GATE) and scripts/mechanism/cross_cell.py refuse a reuse cell until this record licenses it.  If the
gate FAILS: regenerate the cell configs with make_cell_configs.py --rerun-reuse-cells and run alpha1 / lam1 with
run_cells.py --include-reuse (never edit the record by hand).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _k in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.expanduser("~/.cache/numba_mech"))

import numpy as np  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "equal_budget"))
import eqb_family as MF  # noqa: E402

DEFAULT_CELL = os.path.join(ROOT, "configs", "mechanism", "cells", "matched_free_energy", "alpha1.json")
EXCLUDED = ("cfg_json", "meta_json", "wall_s", "peak_rss_mb", "n_checkpoints_written", "resumed_from")


class GateError(RuntimeError):
    pass


def load_cell(path):
    P = json.load(open(path))
    R = P.get("reuse")
    if not (MF.is_cell(P) and R):
        raise GateError(f"{path} is not a reuse cell config (reuse block absent)")
    if MF.family_model(P["engine_cfg"]) != MF.FAMILY_MODEL_DEFAULTS:
        raise GateError(f"{path}: the reuse gate is for the original dynamics only, not {P['model']}")
    if not R.get("gate_record"):
        raise GateError(f"{path}: reuse.gate_record not set (regenerate with make_cell_configs.py)")
    return P


def default_jobs(P):
    R = P["reuse"]
    s = int(P["seeds"][0])
    return [(int(N), s, m) for N in R.get("gate_required_N", [2048, 128]) for m in ("abf", "fr")]


def parse_job(txt):
    N, s, m = txt.split(":")
    if m not in ("abf", "fr"):
        raise GateError(f"job {txt!r}: method must be abf or fr")
    return int(N), int(s), m


def compare_files(old_path, new_path):
    """(compared, extra_in_new, mismatched) keys: every array of the reused file except EXCLUDED, bitwise."""
    with np.load(old_path, allow_pickle=False) as a, np.load(new_path, allow_pickle=False) as b:
        keys = sorted(k for k in a.files if k not in EXCLUDED)
        extra = sorted(set(b.files) - set(a.files))
        bad = []
        for k in keys:
            if k not in b.files:
                bad.append(f"{k}: absent from the new file")
                continue
            x, y = a[k], b[k]
            if x.dtype != y.dtype or x.shape != y.shape:
                bad.append(f"{k}: dtype/shape {x.dtype}{x.shape} vs {y.dtype}{y.shape}")
            elif not np.array_equal(x, y, equal_nan=(x.dtype.kind in "fc")):
                bad.append(k)
    return keys, extra, bad


def run_one(args):
    """One gate job; never raises (a failure is recorded as not bitwise equal)."""
    cfg, N, seed, method, old_path, new_path = args
    rec = dict(N=N, seed=seed, method=method, reused_path=os.path.relpath(old_path, ROOT),
               new_path=os.path.relpath(new_path, ROOT) if new_path.startswith(ROOT) else new_path)
    t0 = time.time()
    try:
        import gateway_family_numba as E
        with np.load(old_path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta_json"]))
            save_steps = np.asarray(z["save_step"], dtype=np.int64)
        if meta.get("status") != "complete":
            raise GateError(f"{old_path}: status {meta.get('status')!r}")
        if (int(meta["N"]), int(meta["seed"]), meta["method"]) != (N, seed, method):
            raise GateError(f"{old_path}: meta N/seed/method {meta['N']}/{meta['seed']}/{meta['method']}")
        n_steps = int(meta["n_steps"])
        rec.update(n_steps=n_steps, reused_engine=meta.get("engine"), reused_sha256=MF.sha256_file(old_path))
        E.run_job(cfg, method, N, seed, n_steps, new_path, save_steps=[int(x) for x in save_steps],
                  checkpoint_every_s=600.0)
        keys, extra, bad = compare_files(old_path, new_path)
        with np.load(new_path, allow_pickle=False) as z:
            nm = json.loads(str(z["meta_json"]))
        rec.update(new_sha256=MF.sha256_file(new_path), new_engine=nm.get("engine"),
                   new_engine_sha256=nm.get("engine_sha256"), keys_compared=keys, keys_excluded=list(EXCLUDED),
                   extra_keys_new=extra, mismatched=bad, bitwise_equal=(not bad and len(keys) > 0))
    except Exception as e:  # noqa: BLE001
        rec.update(error=f"{type(e).__name__}: {e}", bitwise_equal=False)
    rec["wall_s"] = time.time() - t0
    return rec


def verdict_of(P, jobs):
    R = P["reuse"]
    probs = []
    ok = [j for j in jobs if j.get("bitwise_equal") is True]
    if len(ok) != len(jobs):
        probs.append(f"{len(jobs) - len(ok)} job(s) not bitwise equal")
    if len(ok) < int(R.get("gate_min_jobs", 4)):
        probs.append(f"{len(ok)} equal jobs < {R.get('gate_min_jobs', 4)}")
    if {"abf", "fr"} - {j["method"] for j in ok}:
        probs.append("both arms not covered")
    miss = sorted(set(int(n) for n in R.get("gate_required_N", [2048, 128])) - {j["N"] for j in ok})
    if miss:
        probs.append(f"N {miss} not covered")
    if any(j.get("new_engine") != MF.REUSE_GATE_ENGINE for j in ok):
        probs.append(f"new files not written by {MF.REUSE_GATE_ENGINE}")
    return ("PASS" if not probs else "FAIL"), probs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cell", default=DEFAULT_CELL, help="a reuse cell config (alpha1 or lam1: same files, same gate)")
    ap.add_argument("--jobs", nargs="*", default=None, help="N:SEED:METHOD (default: first seed x gate_required_N x both arms)")
    ap.add_argument("--out-dir", default=None, help="default: the directory of the cell's reuse.gate_record")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    P = load_cell(a.cell)
    record = MF._abs(P["reuse"]["gate_record"])
    out_dir = os.path.abspath(a.out_dir) if a.out_dir else os.path.dirname(record)
    if os.path.abspath(os.path.join(out_dir, "reuse_gate.json")) != record:
        raise GateError(f"--out-dir {out_dir} does not hold the cell's gate record {record}")
    MF.guard_output(P, out_dir)
    jobs = [parse_job(j) for j in a.jobs] if a.jobs else default_jobs(P)
    rd = MF._abs(P["results_dir"])
    cfg = dict(P["engine_cfg"])
    work = []
    for N, s, m in jobs:
        old = os.path.join(rd, f"N{N}", f"s{s}_{m}.npz")
        if not os.path.exists(old):
            raise GateError(f"reused file {old} does not exist")
        work.append((cfg, N, s, m, old, os.path.join(out_dir, "runs", f"N{N}", f"s{s}_{m}.npz")))
        print(f"gate job N {N} seed {s} {m}: {os.path.relpath(old, ROOT)} -> {work[-1][-1]}", flush=True)
    if a.dry_run:
        return 0
    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=max(1, min(a.workers, len(work)))) as ex:
        recs = list(ex.map(run_one, work))
    verdict, probs = verdict_of(P, recs)
    doc = dict(schema=MF.REUSE_GATE_SCHEMA, generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               verdict=verdict, problems=probs, engine=MF.REUSE_GATE_ENGINE,
               model=MF.family_model(P["engine_cfg"]), cell_config=os.path.relpath(os.path.abspath(a.cell), ROOT),
               results_dir=P["results_dir"], reused_engine=P["engine_version"],
               criterion=("plan section 5: >= gate_min_jobs re-run jobs bitwise equal (every array except the session "
                          "bookkeeping in keys_excluded), both arms, every N of gate_required_N"),
               gate_min_jobs=int(P["reuse"].get("gate_min_jobs", 4)),
               gate_required_N=[int(n) for n in P["reuse"].get("gate_required_N", [2048, 128])],
               jobs=recs, wall_s=time.time() - t0)
    os.makedirs(out_dir, exist_ok=True)
    with open(record + ".tmp", "w") as fh:
        json.dump(doc, fh, indent=1)
    os.replace(record + ".tmp", record)
    for r in recs:
        print(f"N {r['N']} seed {r['seed']} {r['method']}: bitwise_equal {r['bitwise_equal']} "
              f"{r.get('mismatched') or r.get('error') or ''}", flush=True)
    left = MF.reuse_gate_problems(P)
    print(f"REUSE GATE {verdict} {probs or ''}; record {record}; license check: "
          f"{'OK' if not left else left}", flush=True)
    return 0 if verdict == "PASS" and not left else 1


if __name__ == "__main__":
    sys.exit(main())
