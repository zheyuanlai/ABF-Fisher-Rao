#!/usr/bin/env python
"""Production driver of the equal-budget replica ladders (docs/equal_budget/SCIENTIFIC_PLAN.md).

    python scripts/equal_budget/run_ladder.py --system lta300 --order coarse_first [--workers 120] [--only-N 1024 256]
    python scripts/equal_budget/run_ladder.py --system gateway
    python scripts/equal_budget/run_ladder.py --system lta300 --config configs/equal_budget_v2/confirm_best_lta300.json

One process per (system, N, seed, method) via the engine's run_job (resumes from its checkpoint, skips complete
results).  Outputs results/equal_budget_v2/<dir>/N<N>/s<seed>_<method>.npz.  A ledger row per finished job is
appended to results/equal_budget_v2/<dir>/ledger.csv (wall_s, peak_rss_mb, n_force_evals, status).
"""
import argparse
import csv
import json
import math
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for k in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k] = "1"
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.expanduser("~/.cache/numba_eqb"))
import numpy as np  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
SYSTEMS = {"gateway": ("gateway_production.json", "gateway"),
           "lta300": ("lta_300K_production.json", "lta_T300"),
           "lta150": ("lta_150K_production.json", "lta_T150")}


def load_cfg(system, cfg_path=None):
    """The production config of `system`, or `cfg_path` (a derived config, e.g. the best-allocation confirmation)."""
    return json.load(open(cfg_path or os.path.join(ROOT, "configs", "equal_budget_v2", SYSTEMS[system][0])))


def engine(system):
    if system == "gateway":
        import gateway_ladder_numba as E
    else:
        import lta_ladder_numba as E
    return E


def save_grid(P, n_steps):
    """Budget grid (200 uniform + 24 log-spaced u) union the frozen physical-time checkpoints (t <= T_N),
    each at the nearest integration step."""
    E = engine(P["system_key"])
    steps, _ = E.budget_save_grid(n_steps)
    h = float(P["engine_cfg"]["h"])
    extra = [int(round(t / h)) for t in P["physical_checkpoints_t"] if 0 < round(t / h) <= n_steps]
    return np.unique(np.concatenate([steps, np.asarray(extra, dtype=np.int64)]))


def job_list(P, order, only_N=None):
    B = int(P["B"])
    Ns = list(P["N_coarse"]) + [n for n in P["N_ladder"] if n not in P["N_coarse"]] if order == "coarse_first" else list(P["N_ladder"])
    if only_N:
        Ns = [n for n in Ns if n in only_N]
    J = []
    for N in Ns:
        n_steps = B // N
        assert n_steps * N == B, (N, B)
        for seed in P["seeds"]:
            for method in (("abf",) if N == 1 else ("abf", "fr")):
                J.append((N, seed, method, n_steps))
    return J


def out_path(P, N, seed, method):
    return os.path.join(ROOT, "results", "equal_budget_v2", P["out_dir"], f"N{N}", f"s{seed}_{method}.npz")


def lta_is_complete(path):
    if not os.path.exists(path):
        return False
    try:
        with np.load(path, allow_pickle=False) as z:
            return json.loads(str(z["meta_json"])).get("status") == "complete"
    except Exception:
        return False


def lta_run_job(E, P, method, N, seed, n_steps, path):
    """LTA adapter with run_job semantics: skip complete, resume from path + '.ckpt.npz', atomic save, drop ckpt."""
    if lta_is_complete(path):
        return E.load_result(path)
    cfg = E.make_cfg(P["T_K"], N, seed, **P["engine_cfg"])
    assert cfg["n_steps"] == n_steps, (cfg["n_steps"], n_steps)
    ckpt = path + ".ckpt.npz"
    res = E.run_arm(cfg, method, save_grid(P, n_steps), checkpoint_path=ckpt,
                    checkpoint_every_s=float(P.get("checkpoint_every_s", 900.0)))
    E.save_result(path, res)
    if os.path.exists(ckpt):
        os.remove(ckpt)
    return res


def run_one(args):
    system, cfg_path, N, seed, method, n_steps = args
    P = load_cfg(system, cfg_path)
    P["system_key"] = system
    E = engine(system)
    p = out_path(P, N, seed, method)
    t0 = time.time()
    status = "complete"
    try:
        if system == "gateway":
            res = E.run_job(P["engine_cfg"], method, N, seed, n_steps, p, save_steps=save_grid(P, n_steps),
                            checkpoint_every_s=float(P.get("checkpoint_every_s", 600.0)))
        else:
            res = lta_run_job(E, P, method, N, seed, n_steps, p)
        rec = dict(wall_s=float(res.get("wall_s", time.time() - t0)), peak_rss_mb=float(res.get("peak_rss_mb", float("nan"))),
                   n_force_evals=int(res.get("n_force_evals", -1)))
    except Exception as e:  # never mark a failed run complete
        status = f"FAILED: {type(e).__name__}: {e}"
        rec = dict(wall_s=time.time() - t0, peak_rss_mb=float("nan"), n_force_evals=-1)
    return dict(system=system, N=N, seed=seed, method=method, n_steps=n_steps, status=status, **rec,
                finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=list(SYSTEMS))
    ap.add_argument("--order", default="coarse_first", choices=["coarse_first", "ladder"])
    ap.add_argument("--workers", type=int, default=120)
    ap.add_argument("--only-N", type=int, nargs="*", default=None)
    ap.add_argument("--config", default=None, help="config path overriding the production config of --system")
    a = ap.parse_args()
    cfg_path = os.path.abspath(a.config) if a.config else None
    P = load_cfg(a.system, cfg_path)
    P["system_key"] = a.system
    E = engine(a.system)
    done = E.is_complete if a.system == "gateway" else lta_is_complete
    J = [j for j in job_list(P, a.order, a.only_N) if not done(out_path(P, j[0], j[1], j[2]))]
    est = sum(j[0] * j[3] for j in J) * float(P["est_us_per_walker_step"]) * 1e-6 / 3600
    print(f"{a.system}: {len(J)} jobs to run, est. {est:.1f} core-h", flush=True)
    ledger = os.path.join(ROOT, "results", "equal_budget_v2", P["out_dir"], "ledger.csv")
    os.makedirs(os.path.dirname(ledger), exist_ok=True)
    new = not os.path.exists(ledger)
    from concurrent.futures import ProcessPoolExecutor, as_completed
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex, open(ledger, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["system", "N", "seed", "method", "n_steps", "status", "wall_s", "peak_rss_mb",
                                           "n_force_evals", "finished_utc"])
        if new:
            w.writeheader()
        futs = [ex.submit(run_one, (a.system, cfg_path) + j) for j in J]
        for k, f in enumerate(as_completed(futs)):
            r = f.result()
            w.writerow(r); fh.flush()
            print(f"[{k + 1}/{len(J)}] N {r['N']} seed {r['seed']} {r['method']}: {r['status']} {r['wall_s'] / 60:.1f} min "
                  f"(elapsed {(time.time() - t0) / 3600:.2f} h)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
