#!/usr/bin/env python
"""Runs of configs/numerical_validation/wca_validation_prereg.json (CPU only, numba).

    python scripts/numerical_validation/run_wca_validation.py [--workers 120] [--only GROUP ...]

Groups (results/numerical_validation/wca/<group>/<name>_s<seed>.npz, one file per job, skipped if present):
  em_impl   production unbiased dynamics through wca_numba.run_ladder_point (abf_bias_scale 0), the same
            code path as scripts/run_wca_consistency.py, extended to dt 0.00025 / 0.000125 and to 64 seeds
  mc        Metropolis Monte Carlo on the intended potential (exact)
  mala      MALA at dt 2.5e-4 (exact)
  lm        Leimkuhler-Matthews, intended force
  baoab     BAOAB underdamped, gamma 1, intended force
  diag      the production drift in the wca_validation harness: regularisation counters and energies
Every file stores the run's metadata (git commit, numba/numpy versions, seed, dt, steps, wall time).
"""
import argparse
import json
import os
import subprocess
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for k in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k] = "1"
import numpy as np  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
OUT = os.path.join(ROOT, "results", "numerical_validation", "wca")
TPHYS = 30720.0
BURN_TU = 20.0


def _meta():
    import numba
    try:
        commit = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # pragma: no cover
        commit = "unknown"
    return dict(commit=commit, numba=numba.__version__, numpy=np.__version__, python=sys.version.split()[0],
                host=os.uname().nodename, precision="float64", device="cpu")


def job_list():
    J = []
    # EM_IMPL via the production path
    for dt in (0.002, 0.001, 0.0005):
        for s in range(3632, 3664):
            J.append(("em_impl", f"dt{dt:g}", s, dict(dt=dt)))
    for dt in (0.00025, 0.000125):
        for s in range(3600, 3664):
            J.append(("em_impl", f"dt{dt:g}", s, dict(dt=dt)))
    for s in range(9100, 9164):
        J.append(("mc", "mc", s, dict(n_steps=30_000_000, burn_in=20_000)))
    for s in range(9200, 9232):
        J.append(("mala", "dt0.00025", s, dict(dt=0.00025)))
    for dt in (0.001, 0.0005, 0.00025):
        for s in range(9300, 9332):
            J.append(("lm", f"dt{dt:g}", s, dict(dt=dt)))
    for dt in (0.005, 0.0025, 0.00125):
        for s in range(9400, 9432):
            J.append(("baoab", f"dt{dt:g}", s, dict(dt=dt)))
    for dt in (0.002, 0.001, 0.0005, 0.00025, 0.000125):
        for s in range(9500, 9516):
            J.append(("diag", f"dt{dt:g}", s, dict(dt=dt)))
    return J


def cost(j):
    g, name, s, kw = j
    per = dict(em_impl=6e-6, mc=65e-6, mala=12e-6, lm=12e-6, baoab=12e-6, diag=28e-6)[g]
    if g == "mc":
        return kw["n_steps"] * per
    T = 3072.0 if g == "diag" else TPHYS
    return T / kw["dt"] * per


def path(j):
    g, name, s, kw = j
    return os.path.join(OUT, g, f"{name}_s{s}.npz")


def run(j):
    g, name, s, kw = j
    import wca_validation as V
    t0 = time.perf_counter()
    meta = dict(_meta(), group=g, name=name, seed=s, **kw)
    if g == "em_impl":
        import wca_numba as wn
        dt = kw["dt"]
        n = int(round(TPHYS / dt))
        burn = int(round(BURN_TU / dt))
        cfg = dict(wn.ACCEPTED_CFG, abf_bias_scale=0.0, dt=dt, estimator_burn_in_steps=burn)
        res = wn.run_ladder_point(s, 1, n + burn, [("unbiased", False)], cfg=cfg, cap_min=1,
                                  save_at=np.array([burn + n // 2, burn + n]))
        meta.update(n_steps=n, burn_in_steps=burn, cfg=cfg)
        arrays = dict(C=np.asarray(res["C_prod"])[0], M=np.asarray(res["M_prod"])[0], save_at=res["save_at"],
                      bias_clip_fraction=res["bias_clip_fraction"])
    else:
        scheme = dict(mc="MC", mala="MALA", lm="LM_TRUE", baoab="BAOAB", diag="EM_IMPL")[g]
        q0 = V.lattice_q0(s)
        if g == "mc":
            n, burn = kw["n_steps"], kw["burn_in"]
            o = V.chain("MC", q0, n_steps=n + burn, seed=s, mc_delta=0.4, mc_dimer_delta=0.8, mc_dimer_moves=20,
                        z_stride=max(1, n // 100_000), burn_in=burn)
            meta.update(n_steps=n, burn_in_steps=burn, mc_delta=0.4, mc_dimer_delta=0.8, mc_dimer_moves=20)
        else:
            dt = kw["dt"]
            T = 3072.0 if g == "diag" else TPHYS
            n = int(round(T / dt))
            burn = int(round(BURN_TU / dt))
            o = V.chain(scheme, q0, dt=dt, n_steps=n + burn, seed=s, gamma=1.0, z_stride=max(1, n // 100_000),
                        burn_in=burn, diag_stride=1)
            meta.update(n_steps=n, burn_in_steps=burn, scheme=scheme, gamma=1.0 if g == "baoab" else None, T=T)
        arrays = {k: np.asarray(v) for k, v in o.items()}
    meta["wall_s"] = time.perf_counter() - t0
    p = path(j)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p[:-4] + ".tmp.npz"
    np.savez_compressed(tmp, meta_json=json.dumps(meta), **arrays)
    os.replace(tmp, p)
    return g, name, s, meta["wall_s"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=120)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    J = [j for j in job_list() if not os.path.exists(path(j)) and (a.only is None or j[0] in a.only)]
    J.sort(key=cost, reverse=True)
    print(f"{len(J)} jobs, est. {sum(map(cost, J)) / 3600:.1f} core-h", flush=True)
    # compile once in the parent (numba cache) before forking workers
    import wca_validation as V
    import wca_numba as wn
    for sc in V.SCHEMES:
        V.chain(sc, V.lattice_q0(0), n_steps=5, burn_in=1)
    wn.run_ladder_point(3600, 1, 20, [("unbiased", False)], cfg=dict(wn.ACCEPTED_CFG, abf_bias_scale=0.0),
                        save_at=np.array([20]))
    from concurrent.futures import ProcessPoolExecutor, as_completed
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(run, j) for j in J]
        for k, f in enumerate(as_completed(futs)):
            g, name, s, w = f.result()
            print(f"[{k + 1}/{len(J)}] {g} {name} seed {s}: {w / 60:.1f} min (elapsed {(time.perf_counter() - t0) / 60:.1f} min)",
                  flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
