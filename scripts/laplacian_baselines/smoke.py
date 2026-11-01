#!/usr/bin/env python3
"""Short three-arm smoke of every engine with the Laplacian selection rule (validation job, a few CPU-minutes).

Checks, per engine: the three arms share the force-evaluation count; ABF, FR and Laplacian are bitwise identical before
the first selection opportunity (LTA / ERI, where selection starts after fr_start); the Laplacian arm realises
replacement events with a finite score; and prints the genealogy diagnostics (windowed ESS, largest family share) the
pilot rule reads, with their normalisation.  Also verifies the rebuilt ITE framework against the tracked geometry JSON.
"""
import json
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
FAIL = []


def check(ok, msg):
    print(("  OK   " if ok else "  FAIL ") + msg, flush=True)
    if not ok:
        FAIL.append(msg)


def gateway():
    import gateway_ladder_numba as E
    p = json.load(open(os.path.join(ROOT, "configs/laplacian_baselines/gateway.json")))
    N, seed, n_steps = 512, 8300, 160_000                      # 4 t.u. = the ramp; ~2.5 % of the production budget
    save = np.unique(np.concatenate([E.budget_save_grid(n_steps)[0], [1000, 2000, 4000]]))
    out = {}
    for method, sc in (("abf", None), ("fr", None), ("laplacian", 0.01), ("laplacian", 0.1)):
        cfg = dict(p["engine_cfg"])
        if sc is not None:
            cfg["laplacian_scale"] = sc
        t0 = time.time()
        r = E.run_arm(cfg, method, N, seed, n_steps, save, diagnostics=True)
        key = method if sc is None else f"{method}{sc:g}"
        out[key] = r
        ess = np.asarray(r["gen_ess_win"], float)
        print(f"gateway {key:14s} nfe {int(r['n_force_evals']):>12,d} deaths {int(r['fr_deaths_cum'][-1]):6d} "
              f"ess_win min/max {np.nanmin(ess):.3f}/{np.nanmax(ess):.3f} maxfam_win max {np.nanmax(r['gen_maxfam_win']):.3f} "
              f"wall {time.time() - t0:.1f}s", flush=True)
    nfe = {k: int(v["n_force_evals"]) for k, v in out.items()}
    check(len(set(nfe.values())) == 1 and nfe["abf"] == N * n_steps, f"gateway force evaluations equal {nfe}")
    check(int(out["laplacian0.01"]["fr_deaths_cum"][-1]) > 0, "gateway Laplacian 0.01 realises events")
    check(int(out["laplacian0.1"]["fr_deaths_cum"][-1]) > int(out["laplacian0.01"]["fr_deaths_cum"][-1]),
          "gateway Laplacian activity increases with the scale")
    check(not np.array_equal(out["laplacian0.01"]["M_all"][-1], out["fr"]["M_all"][-1]), "gateway Laplacian differs from FR")
    print(f"  ess normalisation: max gen_ess_win = {max(float(np.nanmax(v['gen_ess_win'])) for v in out.values()):.3f} "
          f"(<= 1 means ESS/K)")


def lta():
    import lta_ladder_numba as L
    T, N, seed, n_steps, fr_start = 150.0, 64, 1_953_000, 40_000, 20_000
    save = np.arange(0, n_steps + 1, 2000, dtype=np.int64)
    out = {}
    for method, sc in (("abf", None), ("fr", None), ("laplacian", 0.01), ("laplacian", 0.1)):
        over = dict(n_steps=n_steps, fr_start_steps=fr_start, fr_rate=0.2)
        if sc is not None:
            over["laplacian_scale"] = sc
        cfg = L.make_cfg(T, N, seed, **over)
        t0 = time.time()
        r = L.run_arm(cfg, method, save)
        key = method if sc is None else f"{method}{sc:g}"
        out[key] = r
        ess = np.asarray(r["gen_ess_win"], float)
        print(f"lta     {key:14s} nfe {int(r['n_force_evals']):>12,d} deaths {int(r['cum_deaths'][-1]):6d} "
              f"ess_win min/max {np.nanmin(ess):.3f}/{np.nanmax(ess):.3f} maxfrac_win max {np.nanmax(r['gen_max_frac_win']):.3f} "
              f"wall {time.time() - t0:.1f}s", flush=True)
    nfe = {k: int(v["n_force_evals"]) for k, v in out.items()}
    check(len(set(nfe.values())) == 1, f"lta force evaluations equal {nfe}")
    i = int(np.searchsorted(save, fr_start))
    for k in ("fr", "laplacian0.01", "laplacian0.1"):
        check(np.array_equal(out["abf"]["M_all"][: i + 1], out[k]["M_all"][: i + 1]), f"lta {k} bitwise = ABF before fr_start")
    check(int(out["laplacian0.01"]["cum_deaths"][-1]) > 0, "lta Laplacian 0.01 realises events")
    check(int(out["laplacian0.1"]["cum_deaths"][-1]) >= int(out["laplacian0.01"]["cum_deaths"][-1]), "lta Laplacian activity grows with the scale")
    check(not np.array_equal(out["laplacian0.01"]["M_all"][-1], out["fr"]["M_all"][-1]), "lta Laplacian differs from FR")


def ite_framework():
    path = os.path.join(ROOT, "cache/eri/framework_ite_itq3.npz")
    if not os.path.exists(path):
        print("  ITE framework npz missing: run scripts/overnight/save_ite.py first")
        FAIL.append("ITE framework missing")
        return False
    J = json.load(open(os.path.join(ROOT, "cache/eri/framework_ite_itq3.json")))
    z = np.load(path, allow_pickle=False)
    keys = list(z.files)
    print("  framework keys:", keys)
    ok = True
    for k, name in (("P", "P"), ("x_window", "x_window")):
        if k in z.files:
            d = abs(float(z[k]) - float(J[name]))
            ok &= d < 1e-6
            print(f"  {name}: npz {float(z[k]):.6f} json {float(J[name]):.6f}")
    for k in ("cages", "windows"):
        if k in z.files:
            d = float(np.abs(np.sort(np.asarray(z[k], float), 0) - np.sort(np.asarray(J[k], float), 0)).max())
            ok &= d < 1e-6
            print(f"  {k}: max |npz - json| = {d:.2e}")
    check(ok, "ITE framework geometry (P, x_window, cages, windows) matches the tracked JSON")
    return ok


def ite():
    from eri import engine as E
    from eri.model import Model
    spec = json.load(open(os.path.join(ROOT, "configs/adaptive_fr/jobs_R2C1.json")))
    base = [j for j in spec["jobs"] if j["method"] == "fr"][0]["cfg"]
    N, n_steps, fr_start = 64, 30_000, 10_000
    c0 = dict(base, N=N, n_steps=n_steps, save_every=1000, fr_start_steps=fr_start, seed=1_951_000,
              genealogy_window_steps=5000)
    c0.pop("fr_controller", None)
    model = Model(c0["variant"], T_K=c0["T_K"], params=c0["ff"])
    out = {}
    for method, sc in (("abf", None), ("fr", None), ("laplacian", 0.01), ("laplacian", 0.1)):
        c = E.make_cfg(**(dict(c0, laplacian_scale=sc) if sc is not None else c0))
        t0 = time.time()
        r = E.run(c, method, model=model)
        key = method if sc is None else f"{method}{sc:g}"
        out[key] = r
        print(f"ite     {key:14s} nfe {int(r['n_force_evals']):>12,d} repl {int(r['sv_i'][-1, 2]):6d} "
              f"essw min/max {np.nanmin(r['sv_u'][:, 2]):.3f}/{np.nanmax(r['sv_u'][:, 2]):.3f} maxfw max {np.nanmax(r['sv_u'][:, 3]):.3f} "
              f"sstd {np.nanmax(r['sv_u'][:, 4]):.3f} wall {time.time() - t0:.1f}s", flush=True)
    nfe = {k: int(v["n_force_evals"]) for k, v in out.items()}
    check(len(set(nfe.values())) == 1, f"ite force evaluations equal {nfe}")
    steps = np.asarray(out["abf"]["save_steps"])
    i = int(np.searchsorted(steps, fr_start))
    for k in ("fr", "laplacian0.01", "laplacian0.1"):
        check(np.array_equal(out["abf"]["sv_M"][: i + 1], out[k]["sv_M"][: i + 1]), f"ite {k} bitwise = ABF before fr_start")
    check(int(out["laplacian0.01"]["sv_i"][-1, 2]) > 0, "ite Laplacian 0.01 realises events")
    check(not np.array_equal(out["laplacian0.01"]["sv_M"][-1], out["fr"]["sv_M"][-1]), "ite Laplacian differs from FR")


if __name__ == "__main__":
    print("== gateway ==", flush=True)
    gateway()
    print("== lta ==", flush=True)
    lta()
    print("== ite framework ==", flush=True)
    if ite_framework():
        print("== ite ==", flush=True)
        ite()
    print("SMOKE_FAILURES", len(FAIL), FAIL)
    sys.exit(1 if FAIL else 0)
