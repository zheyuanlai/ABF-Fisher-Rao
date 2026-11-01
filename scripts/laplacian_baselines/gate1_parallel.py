#!/usr/bin/env python3
"""Run the frozen Gate 1 reference analysis (scripts/eri_campaign/analyze_reference.py) with the group-bootstrap of
eri.refanalysis.compare evaluated in parallel, bit-for-bit identical to the serial call.

    python scripts/laplacian_baselines/gate1_parallel.py --procs 32 <arguments of analyze_reference.py>
    python scripts/laplacian_baselines/gate1_parallel.py --check <dir A> <dir B>     # compare gate1.json + reference_profile.npz

Why: the serial analysis of the 1024-window ITE 150 K cell (3 x 2000 bootstrap WHAM solves) exceeded a 2 h walltime twice
(NSCC jobs 26074468, 26074906; the 640-window 300 K cell took 44 min).  Nothing numerical changes: the bootstrap index
draws come from the same seeded generator in the same order, each bootstrap profile is evaluated by a forked worker that
inherits the parent's WHAM warm-start cache (eri.refanalysis._WARM, filled when the Profile objects were built, exactly
as in the serial loop), and the per-draw statistic is the same expression.  --check verifies this on a cell whose serial
output exists (the 300 K cell on NSCC).
"""
import argparse
import math
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "eri_campaign"))
from eri import refanalysis as ra  # noqa: E402

PROCS = int(os.environ.get("NPROCS", "16"))
_A = _B = None


def _one(args):
    ia, ib = args
    db = ra.centre(_A.of(ia) - _B.of(ib))
    return float(np.mean(db ** 2))


def compare_parallel(A, B, nboot=2000, seed=12345):
    """eri.refanalysis.compare with the nboot profile evaluations spread over PROCS forked workers."""
    global _A, _B
    d = ra.centre(A.F - B.F)
    noise2 = float(np.mean(A.se ** 2 + B.se ** 2))
    rms = float(np.sqrt(np.mean(d ** 2)))
    D = math.sqrt(max(0.0, rms ** 2 - noise2))
    rng = np.random.default_rng(seed)
    GA = getattr(A, "G", 1)
    GB = getattr(B, "G", 1)
    draws = []
    for _ in range(nboot):                      # same generator, same order as the serial loop
        ia = rng.integers(0, GA, GA)
        ib = rng.integers(0, GB, GB)
        draws.append((ia, ib))
    _A, _B = A, B
    with ProcessPoolExecutor(max_workers=PROCS, mp_context=mp.get_context("fork")) as ex:
        msq = list(ex.map(_one, draws, chunksize=max(1, nboot // (8 * PROCS))))
    Db = np.array([math.sqrt(max(0.0, m - 2.0 * noise2)) for m in msq])
    return dict(A=A.label, B=B.label, D=D, D_upper95=float(np.quantile(Db, 0.95)), rms_obs=rms,
                noise=math.sqrt(noise2), max_abs=float(np.abs(d).max()))


def check(da, db):
    import json
    ok = True
    ga, gb = (json.load(open(os.path.join(d, "gate1.json"))) for d in (da, db))

    def walk(x, y, path):
        nonlocal ok
        if isinstance(x, dict):
            for k in sorted(set(x) | set(y)):
                if k not in x or k not in y:
                    ok = False
                    print(f"  gate1.json {path}/{k}: missing on one side")
                else:
                    walk(x[k], y[k], f"{path}/{k}")
        elif isinstance(x, list):
            for i, (u, v) in enumerate(zip(x, y)):
                walk(u, v, f"{path}[{i}]")
        elif x != y:
            ok = False
            print(f"  gate1.json {path}: {x!r} != {y!r}")
    walk(ga, gb, "")
    za, zb = (np.load(os.path.join(d, "reference_profile.npz")) for d in (da, db))
    for k in sorted(set(za.files) | set(zb.files)):
        if k not in za.files or k not in zb.files or not np.array_equal(za[k], zb[k]):
            ok = False
            diff = float(np.max(np.abs(za[k] - zb[k]))) if k in za.files and k in zb.files else float("nan")
            print(f"  reference_profile.npz[{k}]: differs (max abs {diff:.3e})")
    print("GATE1_PARALLEL_CHECK", "PASS (gate1.json and reference_profile.npz identical)" if ok else "FAIL")
    return 0 if ok else 3


def main():
    global PROCS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--procs", type=int, default=PROCS)
    ap.add_argument("--check", nargs=2, metavar=("DIR_A", "DIR_B"))
    a, rest = ap.parse_known_args()
    if a.check:
        sys.exit(check(*a.check))
    PROCS = a.procs
    import analyze_reference as AR
    ra.compare = compare_parallel               # AR calls ra.compare through this module object
    sys.argv = [AR.__file__] + rest
    print(f"gate1_parallel: {PROCS} workers; analyze_reference.py {' '.join(rest)}", flush=True)
    AR.main()


if __name__ == "__main__":
    main()
