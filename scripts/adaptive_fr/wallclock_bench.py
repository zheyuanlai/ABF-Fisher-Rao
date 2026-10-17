"""Experiment E.4 (+ E-A2): measured parallel wall-clock of ABF and ABF + FR(R1) at K in {1024, 256, 64, 16} (ITE 300 K production
configurations) on this host, then wall-clock time to accuracy = measured s/step at P threads x the configuration's
median sustained tau in steps (from the confirmatory analyses).  The segment timed is 2 t.u. starting at the FR start
(FR active in the FR arm), after a JIT warm-up; equilibration is shortened for the benchmark only (it is serial and is
reported separately from the production runs' logs).

    NUMBA_NUM_THREADS=32 python scripts/adaptive_fr/wallclock_bench.py [out_name]

Run 1 (before E-A2, K >= 64) wrote E_wallclock_bench.json; run 2 (all four K) writes E_wallclock_bench_run2.json.
"""
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numba  # noqa: E402

from eri import engine as E  # noqa: E402
from eri.fr_controller import R1  # noqa: E402
from eri.model import Model  # noqa: E402

THREADS = (1, 4, 8, 16, 32)
CONFS = [(1024, 2.0, "C1"), (256, 1.0, "EA_R1"), (64, 0.2, "EB64"), (16, 1.0, "EB16")]


def cfg(K, rate, fr):
    j = json.load(open(os.path.join(ROOT, "configs", "convergence_fr", "jobs_EB.json")))["jobs"][0]["cfg"]
    c = {k: v for k, v in j.items() if k != "seed"}
    c.update(N=K, fr_rate=rate, seed=1_390_000, equil_steps=2000, n_steps=40_000, save_every=5000)
    if fr:
        c["fr_controller"] = R1
    return E.make_cfg(**c)


def main():
    m = Model("ite_itq3", T_K=300.0, params=cfg(64, 1.0, False)["ff"])
    lscpu = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
    model_name = [l.split(":", 1)[1].strip() for l in lscpu.splitlines() if l.startswith("Model name")]
    load0 = os.getloadavg()
    out = dict(utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), host=platform.node(),
               cpu=model_name[0] if model_name else "unknown", logical_cpus=os.cpu_count(),
               numba_max_threads=numba.config.NUMBA_NUM_THREADS, load_before=load0, rows=[])
    for K, rate, block in CONFS:
        for method in ("abf", "fr"):
            c = cfg(K, rate, method == "fr")
            for P in THREADS:
                if P > numba.config.NUMBA_NUM_THREADS:
                    continue
                numba.set_num_threads(P)
                r = E.Run(c, method, model=m)
                r.advance_controlled(c["fr_start_steps"])           # warm-up incl. compilation, up to the FR start
                k0 = r.k
                t0 = time.perf_counter()
                r.advance_controlled(k0 + 20_000)                    # 2 t.u. of production, FR active in the FR arm
                wall = time.perf_counter() - t0
                out["rows"].append(dict(K=K, method=method, threads=P, steps=20_000, wall_s=wall, s_per_step=wall / 20_000,
                                        core_s=wall * P, fr_events=int(r.ist[E.S_REPL]), load=os.getloadavg()[0]))
                print(f"K {K:5d} {method:3s} P {P:2d}: {wall:7.2f} s / 20000 steps  ({1e6 * wall / 20000 / K:.3f} us per molecule-step"
                      f" wall, load {os.getloadavg()[0]:.1f})", flush=True)
    out["load_after"] = os.getloadavg()
    name = sys.argv[1] if len(sys.argv) > 1 else "E_wallclock_bench_run2.json"
    p = os.path.join(ROOT, "results", "adaptive_fr_campaign", "analysis", name)
    assert not os.path.exists(p), f"refusing to overwrite {p}"
    json.dump(out, open(p, "w"), indent=1)
    print("wrote", p)


if __name__ == "__main__":
    main()
