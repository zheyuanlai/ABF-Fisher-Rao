#!/usr/bin/env python
"""Gateway timestep validation runs (configs/equal_budget_v2/gateway_validation.json).
    python scripts/equal_budget/run_gateway_validation.py [--workers 64] [--h 1.25e-5]
Output: results/equal_budget_v2/gateway_validation/<tag>/g<group>.npz"""
import argparse, json, os, subprocess, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for k in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "1"
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.expanduser("~/.cache/numba_eqb"))
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
OUT = os.path.join(ROOT, "results", "equal_budget_v2", "gateway_validation")
P = dict(beta=16.0, H=0.5, omega_out=1.0, omega_in=32.0, s=0.1)
NB_FINE = 1440
T_RUN, T_BURN, NW, G = 500.0, 10.0, 256, 16


def jobs(hs):
    J = []
    for k, h in enumerate(hs):
        for g in range(G):
            J.append(("em_flat", h, g, 61000 + 100 * k + g))
    for g in range(G):
        J.append(("mala_flat", 1e-4, g, 62000 + g))
    for k, h in enumerate(hs):
        J.append(("em_unbiased", h, 0, 63000 + k))
    return J


def tag(j):
    return f"{j[0]}_h{j[1]:g}"


def run(j):
    import gateway_validation as GV
    kind, h, g, seed = j
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    if kind == "em_unbiased":
        x0 = np.array([-1.0, -1.0, -0.5, 0.0, 0.0, 0.5, 1.0, 1.0]); nw = len(x0)
        T, burn, flat, sch = 200.0, 0.0, 0, GV.EM
    else:
        nw = NW
        x0 = rng.uniform(-1.5, 1.5, nw)
        T, burn, flat, sch = T_RUN, T_BURN, 1, (GV.EM if kind == "em_flat" else GV.MALA)
    y0 = rng.standard_normal(nw) / (np.sqrt(P["beta"]) * GV.omega(x0, P))
    n = int(round((T + burn) / h)); nburn = int(round(burn / h))
    tr_every = max(1, int(round(0.05 / h)))
    C, Mf, Sy, Sy2, nacc, nprop, nref, mdx, nf, tr, X, Y = GV.run_chain(sch, x0, y0, n, nburn, h, P["beta"], P["H"], P["omega_out"],
                                                                     P["omega_in"], P["s"], flat, NB_FINE, seed, tr_every, min(nw, 16))
    try:
        commit = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    meta = dict(kind=kind, h=h, group=g, seed=seed, n_walkers=nw, T=T, burn=burn, n_steps=n, flat_bias=flat, nb_fine=NB_FINE,
                trace_every=tr_every, commit=commit, wall_s=time.perf_counter() - t0, walker_steps=float(nw) * n)
    p = os.path.join(OUT, tag(j), f"g{g:02d}.npz")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    np.savez_compressed(p[:-4] + ".tmp.npz", meta_json=json.dumps(meta), C=C, Mf=Mf, Sy=Sy, Sy2=Sy2, n_acc=nacc, n_prop=nprop,
                        n_reflect=nref, max_dx=mdx, nonfinite=nf, traces=tr.astype(np.float32))
    os.replace(p[:-4] + ".tmp.npz", p)
    return tag(j), g, meta["wall_s"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--h", type=float, nargs="*", default=[4e-4, 1e-4, 2.5e-5])
    a = ap.parse_args()
    J = [j for j in jobs(a.h) if not os.path.exists(os.path.join(OUT, tag(j), f"g{j[2]:02d}.npz"))]
    J.sort(key=lambda j: -1.0 / j[1])
    print(len(J), "jobs", flush=True)
    import gateway_validation as GV
    GV.run_chain(GV.EM, np.zeros(2), np.zeros(2), 10, 0, 1e-4, 16.0, 0.5, 1.0, 32.0, 0.1, 1, NB_FINE, 1, 5, 2)
    GV.run_chain(GV.MALA, np.zeros(2), np.zeros(2), 10, 0, 1e-4, 16.0, 0.5, 1.0, 32.0, 0.1, 1, NB_FINE, 1, 5, 2)
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(a.workers) as ex:
        for k, f in enumerate(as_completed([ex.submit(run, j) for j in J])):
            t, g, w = f.result()
            print(f"[{k + 1}/{len(J)}] {t} g{g}: {w:.0f} s", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
