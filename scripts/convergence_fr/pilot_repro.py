"""Pilot: the modified engine (fr_stop_steps added, default -1) reproduces an EXISTING run bitwise, and single-thread
throughput.  Re-runs the first --steps steps of a saved run and compares every saved array at the saves it covers.

    NUMBA_NUM_THREADS=1 python scripts/convergence_fr/pilot_repro.py results/overnight/ite/prod_confirm/fr_N1024_s0.npz --steps 20000
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from eri import engine as E  # noqa: E402
from eri.model import Model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--stop", type=int, default=None, help="optional fr_stop_steps for the re-run (must exceed --steps)")
    a = ap.parse_args()
    z = np.load(a.path if os.path.isabs(a.path) else os.path.join(ROOT, a.path), allow_pickle=False)
    cfg = json.loads(str(z["cfg"]))
    over = {k: v for k, v in cfg.items() if k in E.DEFAULT_CFG}
    if a.stop is not None:
        over["fr_stop_steps"] = a.stop
    c = E.make_cfg(**over)
    m = Model(c["variant"], T_K=c["T_K"], params=c["ff"])
    t0 = time.time()
    r = E.Run(c, str(z["method"]), model=m)
    t1 = time.time()
    r.advance(a.steps + 1)
    t2 = time.time()
    sel = z["save_steps"] <= a.steps
    ok = {k: bool(np.array_equal(getattr(r, k)[sel], z[k][sel])) for k in ("sv_C", "sv_M", "sv_h", "sv_i", "sv_u", "sv_cond")}
    nmol = c["N"] * (a.steps + 1)
    out = dict(path=a.path, method=str(z["method"]), steps=a.steps, saves_compared=int(sel.sum()), bitwise=ok,
               all_bitwise=all(ok.values()), equil_seconds=t1 - t0, run_seconds=t2 - t1,
               us_per_molecule_step=1e6 * (t2 - t1) / nmol, threads=int(os.environ.get("NUMBA_NUM_THREADS", "0")),
               note="run_seconds includes numba compilation of _run_chunk (cache=False)")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
