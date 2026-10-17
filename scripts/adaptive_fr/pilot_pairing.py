"""Experiment A pilot (Gate F): the planned FR arm (EB cfg + fr_rate 1.0 + fr_stop 10 t.u.) and an ABF re-run with
the FR arm's fr_rate are bitwise identical to the historical EB ABF file of the same seed at every save before the
first FR opportunity; the ABF re-run also matches at the first save after it (fr_rate inert under ABF).

    NUMBA_NUM_THREADS=1 python scripts/adaptive_fr/pilot_pairing.py --seed 1140000 --steps 15001
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

DATA = os.environ.get("ABFFR_DATA_ROOT", ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1140000)
    ap.add_argument("--steps", type=int, default=15001)
    a = ap.parse_args()
    z = np.load(os.path.join(DATA, f"results/convergence_fr_campaign/EB_ite300_abf_K256_T200/abf_N256_s{a.seed}.npz"))
    cfg = {k: v for k, v in json.loads(str(z["cfg"])).items() if k in E.DEFAULT_CFG}
    m = Model(cfg["variant"], T_K=cfg["T_K"], params=cfg["ff"])
    out = {}
    for tag, over, method in (("fr_stop10_rate1", dict(fr_rate=1.0, fr_stop_steps=100_000), "fr"),
                              ("abf_rate1", dict(fr_rate=1.0), "abf")):
        c = E.make_cfg(**dict(cfg, **over))
        t0 = time.time()
        r = E.Run(c, method, model=m)
        r.advance(a.steps)
        sel = r.save_steps < a.steps
        pre = sel & (r.save_steps < c["fr_start_steps"])
        chk = sel if method == "abf" else pre
        same = {k: bool(np.array_equal(getattr(r, k)[chk], z[k][chk])) for k in ("sv_C", "sv_M", "sv_h", "sv_i", "sv_u", "sv_cond")}
        out[tag] = dict(saves_compared=[int(s) for s in r.save_steps[chk]], bitwise=same, all=all(same.values()),
                        fr_events_by_step=int(r.ist[E.S_REPL]), seconds=time.time() - t0)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
