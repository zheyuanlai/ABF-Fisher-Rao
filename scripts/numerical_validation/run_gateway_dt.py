#!/usr/bin/env python
"""Gateway time-step refinement of the accepted FR effect (configs/numerical_validation/gateway_dt_prereg.json).
    python scripts/numerical_validation/run_gateway_dt.py [--workers 64]"""
import argparse, json, os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for k in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "1"
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
OUT = os.path.join(ROOT, "results", "numerical_validation", "gateway_dt")
DTS = (4e-4, 1e-4, 2.5e-5)


def setup(dt):
    d = json.load(open(os.path.join(ROOT, "configs", "gateway_replica_ladder", "design.json")))
    cell = {k: d["cell"][k] for k in ("beta", "H", "omega_out", "omega_in", "s")}
    cell["dt"] = dt
    f = d["fr_uniform"]
    m = 4e-4 / dt
    fr = dict(gamma=f["gamma"], eta=f["eta"], fr_every=int(round(f["fr_every"] * m)), score_clip=f["score_clip"],
              max_event_fraction=f["max_event_fraction"], ramp_steps=float(f["ramp_steps"] * m), cap_min=f["cap_min"],
              min_count=d["abf"]["min_count"])
    arms = [tuple(a) for a in d["arms"]]
    return cell, fr, arms


def job(args):
    dt, seed = args
    import gateway_numba as gn
    import subprocess
    cell, fr, arms = setup(dt)
    n = int(round(40.0 / dt))
    t0 = time.perf_counter()
    res = gn.run_ladder_point(seed, 2048, n, arms, cell, fr, ess_window=int(round(4000 * 4e-4 / dt)))
    try:
        commit = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    meta = dict(dt=dt, seed=seed, N=2048, n_steps=n, T=40.0, cell=cell, fr=fr, arms=[a[0] for a in arms], commit=commit,
                wall_s=time.perf_counter() - t0, cap=int(res["cap"]))
    p = os.path.join(OUT, f"dt{dt:g}", f"s{seed}.npz")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    np.savez_compressed(p[:-4] + ".tmp.npz", meta_json=json.dumps(meta), save_at=res["save_at"], u=res["u"], M=res["M"], C=res["C"],
                        ess=res["ess"], die=res["die"], clone=res["clone"])
    os.replace(p[:-4] + ".tmp.npz", p)
    return dt, seed, meta["wall_s"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=64); a = ap.parse_args()
    J = [(dt, s) for dt in DTS for s in range(7100, 7132) if not os.path.exists(os.path.join(OUT, f"dt{dt:g}", f"s{s}.npz"))]
    J.sort(key=lambda j: -1.0 / j[0])
    print(len(J), "jobs", flush=True)
    import gateway_numba as gn
    c, f, arms = setup(4e-4)
    gn.run_ladder_point(7100, 8, 50, arms, c, f)
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(a.workers) as ex:
        for k, fu in enumerate(as_completed([ex.submit(job, j) for j in J])):
            dt, s, w = fu.result()
            print(f"[{k + 1}/{len(J)}] dt {dt:g} seed {s}: {w:.0f} s", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
