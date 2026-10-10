#!/usr/bin/env python
"""Mechanism campaign: numerical-validation runs for Experiments I and II (docs/mechanism/SCIENTIFIC_PLAN.md
section 3; harness src/gateway_family_validation.py; analysis scripts/mechanism/analyze_validation.py).

    python scripts/mechanism/run_validation.py [--workers 16] [--cpus 200-215] [--h 2.5e-5 [1.25e-5 ...]]
           [--dynamics alpha1 shift ...] [--chains em_flat mala_flat exact_flat exact_unbiased em_unbiased]
           [--out results/mechanism/validation] [--dry-run] [--benchmark] [--smoke --groups 1 --out <scratch>]

Jobs, per dynamics d (index in gateway_family_validation.DYNAMICS_ORDER: alpha1, alpha0.5, alpha0, shift, lam0.5,
lam0.25, lam0.1) and timestep h (index k in the CANDIDATE list 2.5e-5, 1.25e-5 -- fixed so a seed never depends on
which subset of h is launched):
  em_flat     production EM + frozen flat bias +F*'(x); 16 groups x 256 walkers x 500/lam t.u. after 10 t.u. burn-in;
              x0 uniform on [-1.5, 1.5], y0 from the exact conditional; seeds 71000 + 1000 d + 100 k + g; conditional
              ACF recorded for the first 64 walkers of every group (two strides, gateway_family_validation.acf_plan),
              raw ACF excerpts (first 4096 samples, 16 walkers) and x/y traces (8 walkers, every 0.05/lam t.u.)
              in group 0 only.
  mala_flat   MALA on the same target, h 1e-4, y-mobility preconditioner lam (exact for any lam), 16 x 256 x 500 t.u.
              after 10 t.u.; seeds 78000 + 100 d + g.
  exact_flat  i.i.d. draws of the flat-bias target (x uniform, y exact conditional), 16 groups x 2^24; seeds
              80000 + 100 d + g (one job writes the 16 group files).
  exact_unbiased  i.i.d. draws of the unbiased Gibbs law exp(-beta F*) (inverse CDF), 16 groups x 2^22; seeds
              81000 + 100 d + g (read-out check, reported only).
  em_unbiased unbiased EM, 8 walkers from x = -1, -1, -0.5, 0, 0, 0.5, 1, 1, 200 t.u., traces every 0.05 t.u.;
              seed 79000 + 100 d + 10 k (V5 sanity).
All seeds are disjoint from the 61000-63xxx of the equal-budget gateway validation and from the production 8100-8131.
Output <out>/<dynamics>/<chain>_h<h>/g<kk>.npz (exact chains: <out>/<dynamics>/<chain>/g<kk>.npz), written atomically;
existing files are skipped (resumable at job granularity).  --smoke: tiny runs (T 5/lam t.u., burn 1, exact 2^20,
MALA 5 t.u., unbiased 5 t.u.), meta.smoke = true, --out required and must not be the production directory; --groups
other than 16 is refused without --smoke.  The analysis checks every record against this design (groups 0-15, seeds,
walkers, T, burn, h, smoke flag) and excludes non-conforming sets.
--benchmark: short chains per dynamics -> ns per walker-step and projected core-hours per job and in total.
"""
import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time

if __name__ == "__main__":          # (not when imported by the tests)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for _k in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[_k] = "1"
    os.environ.setdefault("NUMBA_CACHE_DIR", os.path.expanduser("~/.cache/numba_mech"))
import numpy as np  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
import gateway_family_validation as GF  # noqa: E402

OUT_DEFAULT = os.path.join(ROOT, "results", "mechanism", "validation")
H_CANDIDATES = (2.5e-5, 1.25e-5)          # SCIENTIFIC_PLAN section 3: candidate, refinement (index = h_index of the seed rule)
H_MALA = 1e-4
NW, G = 256, 16
T_RUN, T_BURN = 500.0, 10.0
N_ACF, ACF_REC, ACF_REC_WALKERS = 64, 4096, 16
N_TRACE = 8
N_EXACT_FLAT, N_EXACT_UNB = 1 << 24, 1 << 22
CHAINS = ("em_flat", "mala_flat", "exact_flat", "exact_unbiased", "em_unbiased")
SEED_BASE = dict(em_flat=71000, mala_flat=78000, em_unbiased=79000, exact_flat=80000, exact_unbiased=81000)


def h_index(h):
    for k, c in enumerate(H_CANDIDATES):
        if abs(h - c) <= 1e-12 * c:
            return k
    raise ValueError(f"h {h} is not a preregistered candidate {H_CANDIDATES}")


def seed_of(chain, d, h, g):
    if chain == "em_flat":
        return SEED_BASE[chain] + 1000 * d + 100 * h_index(h) + g
    if chain == "em_unbiased":
        return SEED_BASE[chain] + 100 * d + 10 * h_index(h)
    return SEED_BASE[chain] + 100 * d + g


def tag(chain, h):
    return chain if chain.startswith("exact") else f"{chain}_h{h:g}"


def out_path(out, dyn, chain, h, g):
    return os.path.join(out, dyn, tag(chain, h), f"g{g:02d}.npz")


def settings(chain, m, smoke):
    """(T, burn) in t.u. for a chain of model m."""
    lam = m["lam"]
    if chain == "em_flat":
        return ((5.0 if smoke else T_RUN) / lam, 1.0 if smoke else T_BURN)
    if chain == "mala_flat":
        return (5.0 if smoke else T_RUN, 1.0 if smoke else T_BURN)
    if chain == "em_unbiased":
        return (5.0 if smoke else 200.0, 0.0)
    return (None, None)


def jobs(dyns, hs, chains, groups, smoke):
    """(chain, dynamics_index, h, [groups]) -- one entry per output file, except the exact chains (all groups)."""
    J = []
    for d, m in enumerate(GF.load_dynamics()):
        if m["name"] not in dyns:
            continue
        for c in chains:
            if c == "em_flat":
                J += [(c, d, h, [g]) for h in hs for g in range(groups)]
            elif c == "em_unbiased":
                J += [(c, d, h, [0]) for h in hs]
            elif c == "mala_flat":
                J += [(c, d, H_MALA, [g]) for g in range(groups)]
            else:
                J.append((c, d, None, list(range(groups))))
    return J


def cost_walker_steps(j, smoke):
    c, d, h, gs = j
    m = GF.load_dynamics()[d]
    if c.startswith("exact"):
        return 0.0
    T, burn = settings(c, m, smoke)
    nw = 8 if c == "em_unbiased" else NW
    return nw * round((T + burn) / h) * len(gs)


def _commit():
    try:
        return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _save(path, meta, arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path[:-4] + ".tmp.npz"
    np.savez_compressed(tmp, meta_json=json.dumps(meta), **arrays)
    os.replace(tmp, path)


def run(j, out, smoke):
    c, d, h, gs = j
    m = GF.load_dynamics()[d]
    t0 = time.perf_counter()
    base = dict(chain=c, dynamics=m["name"], dynamics_index=d, model={k: m[k] for k in m}, smoke=bool(smoke), commit=_commit(),
                host=platform.node(), harness="src/gateway_family_validation.py", prereg="docs/mechanism/SCIENTIFIC_PLAN.md s3")
    if c.startswith("exact"):
        flat = c == "exact_flat"
        n = (1 << 20) if smoke else (N_EXACT_FLAT if flat else N_EXACT_UNB)
        for g in gs:
            t1 = time.perf_counter()
            seed = seed_of(c, d, h, g)
            A1, A2, zh = GF.exact_samples(n, m, seed, flat=flat)
            meta = dict(base, group=g, seed=seed, n_draws=n, flat=flat, wall_s=time.perf_counter() - t1, walker_steps=float(n))
            _save(out_path(out, m["name"], c, h, g), meta, dict(A1=A1, A2=A2, zhist=zh, nonfinite=np.int64(0)))
        return c, m["name"], h, gs, time.perf_counter() - t0
    g = gs[0]
    seed = seed_of(c, d, h, g)
    T, burn = settings(c, m, smoke)
    rng = np.random.default_rng(seed)
    if c == "em_unbiased":
        x0 = np.array([-1.0, -1.0, -0.5, 0.0, 0.0, 0.5, 1.0, 1.0])
        flat, scheme = 0, GF.EM
        tr_every = max(1, int(round(0.05 / h)))
        n_trace = len(x0)
    else:
        x0 = rng.uniform(-1.5, 1.5, NW)
        flat, scheme = 1, (GF.EM if c == "em_flat" else GF.MALA)
        tr_every = max(1, int(round(0.05 / (m["lam"] * h)))) if g == 0 else 10 ** 12
        n_trace = N_TRACE if g == 0 else 1
    y0 = GF.initial_y(x0, rng.standard_normal(len(x0)), m)
    n = int(round((T + burn) / h))
    nburn = int(round(burn / h))
    plan = GF.acf_plan(m, h) if c == "em_flat" else None
    n_acf = N_ACF if c == "em_flat" else 0
    rec = ACF_REC if (c == "em_flat" and g == 0) else 0
    r = GF.chain(m, scheme, x0, y0, n, nburn, h, flat, seed, trace_every=tr_every, n_trace=n_trace, n_acf=n_acf, acf=plan,
                 acf_rec=rec)
    wall = time.perf_counter() - t0
    meta = dict(base, group=g, seed=seed, h=h, T=T, burn=burn, n_steps=n, n_burn=nburn, n_walkers=len(x0), flat_bias=flat,
                trace_every=tr_every, wall_s=wall, walker_steps=float(len(x0)) * n, ns_per_walker_step=wall / (len(x0) * n) * 1e9,
                acf_windows=list(GF.ACF_WINDOWS), acf_half_width=GF.ACF_HALF_WIDTH, n_acf=n_acf,
                acf_tau_windows=({str(k): v for k, v in plan["tau_windows"].items()} if plan else None),
                zhist_lim=GF.ZH_LIM, zhist_nb=GF.ZH_NB, nb_fine=GF.NB_FINE, nb_prod=GF.NB_PROD, acc_keys=list(GF.ACC_KEYS),
                acf_keys=list(GF.ACF_KEYS))
    arrays = dict(A1=r["A1"], A2=r["A2"], n_acc=r["n_acc"], n_prop=r["n_prop"], n_reflect=r["n_reflect"], max_dx=r["max_dx"],
                  nonfinite=np.int64(r["nonfinite"]), crossings=r["crossings"].astype(np.int32), zhist=r["zhist"],
                  traces_x=r["traces_x"].astype(np.float32), traces_y=r["traces_y"].astype(np.float32),
                  X_final=r["X"], Y_final=r["Y"])
    if n_acf:
        arrays.update(acf0=r["acf0"], acf1=r["acf1"], lags0=r["lags0"], lags1=r["lags1"], acf_stride=r["acf_stride"],
                      acf_nsamples=r["acf_nsamples"])
        if rec:
            arrays.update(acf_rec0=r["acf_rec0"][:, :ACF_REC_WALKERS].astype(np.float32),
                          acf_rec1=r["acf_rec1"][:, :ACF_REC_WALKERS].astype(np.float32))
    _save(out_path(out, m["name"], c, h, g), meta, arrays)
    return c, m["name"], h, gs, wall


def done(j, out):
    c, d, h, gs = j
    name = GF.DYNAMICS_ORDER[d]
    return all(os.path.exists(out_path(out, name, c, h, g)) for g in gs)


def warmup():
    """Compile every kernel specialisation once in the parent (fork children inherit it; no cache-write races)."""
    for m in GF.load_dynamics():
        for sch in (GF.EM, GF.MALA):
            x0 = np.array([-0.5, 0.01])
            GF.chain(m, sch, x0, GF.initial_y(x0, np.zeros(2), m), 5, 0, 1e-4, 1, 1, n_acf=1, acf=GF.acf_plan(m, 1e-4), acf_rec=2)


def benchmark(dyns, hs, out, n_steps=20000):
    """ns per walker-step of em_flat (with the ACF recorder) and mala_flat per dynamics, on this core; projected
    core-hours per job and per chain/h for the FULL gate (16 groups, 500/lam t.u.)."""
    warmup()
    res = dict(when=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()), host=platform.node(), n_steps=n_steps, walkers=NW,
               cpu_affinity=sorted(os.sched_getaffinity(0)), per_dynamics={}, totals_core_h={})
    totals = {}
    for d, m in enumerate(GF.load_dynamics()):
        if m["name"] not in dyns:
            continue
        e = {}
        g = np.random.default_rng(1)
        x0 = g.uniform(-1.5, 1.5, NW)
        y0 = GF.initial_y(x0, g.standard_normal(NW), m)
        for h in hs:
            t = time.perf_counter()
            GF.chain(m, GF.EM, x0, y0, n_steps, 0, h, 1, 7, n_acf=N_ACF, acf=GF.acf_plan(m, h), acf_rec=0)
            ns = (time.perf_counter() - t) / (NW * n_steps) * 1e9
            T, burn = settings("em_flat", m, False)
            job_s = ns * 1e-9 * NW * round((T + burn) / h)
            e[f"em_flat_h{h:g}"] = dict(ns_per_walker_step=ns, core_h_per_group=job_s / 3600, core_h_16_groups=16 * job_s / 3600)
            totals[f"em_flat_h{h:g}"] = totals.get(f"em_flat_h{h:g}", 0.0) + 16 * job_s / 3600
        t = time.perf_counter()
        GF.chain(m, GF.MALA, x0, y0, n_steps // 2, 0, H_MALA, 1, 7)
        ns = (time.perf_counter() - t) / (NW * (n_steps // 2)) * 1e9
        T, burn = settings("mala_flat", m, False)
        job_s = ns * 1e-9 * NW * round((T + burn) / H_MALA)
        e["mala_flat"] = dict(ns_per_walker_step=ns, core_h_per_group=job_s / 3600, core_h_16_groups=16 * job_s / 3600)
        totals["mala_flat"] = totals.get("mala_flat", 0.0) + 16 * job_s / 3600
        t = time.perf_counter()
        GF.exact_samples(1 << 21, m, 1, flat=True)
        e["exact_flat_core_h"] = (time.perf_counter() - t) * (16 * N_EXACT_FLAT / (1 << 21)) / 3600
        totals["exact"] = totals.get("exact", 0.0) + e["exact_flat_core_h"] * (1 + N_EXACT_UNB / N_EXACT_FLAT)
        res["per_dynamics"][m["name"]] = e
        print(m["name"], json.dumps({k: (round(v, 3) if isinstance(v, float) else {kk: round(vv, 3) for kk, vv in v.items()})
                                     for k, v in e.items()}), flush=True)
    res["totals_core_h"] = totals
    res["note"] = ("single-core timings under the load at benchmark time; em_flat includes the 64-walker ACF recorder; the "
                   "unbiased 8-walker runs are negligible (< 0.01 core-h each)")
    print("TOTAL core-h:", {k: round(v, 2) for k, v in totals.items()}, flush=True)
    os.makedirs(out, exist_ok=True)
    json.dump(res, open(os.path.join(out, "benchmark.json"), "w"), indent=1)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--cpus", type=str, default=None, help="pin the pool, e.g. 200-215 (inherited by the workers)")
    ap.add_argument("--h", type=float, nargs="*", default=[H_CANDIDATES[0]])
    ap.add_argument("--dynamics", nargs="*", default=list(GF.DYNAMICS_ORDER))
    ap.add_argument("--chains", nargs="*", default=list(CHAINS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--groups", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--benchmark", action="store_true")
    a = ap.parse_args()
    if a.cpus:
        lo, _, hi = a.cpus.partition("-")
        cpus = set(range(int(lo), int(hi or lo) + 1))
        os.sched_setaffinity(0, cpus)
        a.workers = min(a.workers, len(cpus))
    for h in a.h:
        h_index(h)
    bad = [d for d in a.dynamics if d not in GF.DYNAMICS_ORDER] + [c for c in a.chains if c not in CHAINS]
    if bad:
        raise SystemExit(f"unknown dynamics/chains {bad}")
    out = os.path.abspath(a.out) if a.out else OUT_DEFAULT
    if a.smoke and (a.out is None or os.path.realpath(out).startswith(os.path.realpath(OUT_DEFAULT))):
        raise SystemExit("--smoke needs --out outside results/mechanism/validation")
    if a.benchmark:
        benchmark(a.dynamics, a.h, out)
        return
    groups = a.groups if a.groups is not None else (1 if a.smoke else G)
    if not a.smoke and groups != G:
        # review fix 2026-10-10: a partial design must not land among the gate's files (the analysis would exclude it
        # anyway: it checks every record against the frozen design)
        raise SystemExit(f"--groups {groups}: the gate design is {G} groups; other values only with --smoke")
    J = [j for j in jobs(a.dynamics, a.h, a.chains, groups, a.smoke) if not done(j, out)]
    J.sort(key=lambda j: -cost_walker_steps(j, a.smoke))
    tot = sum(cost_walker_steps(j, a.smoke) for j in J)
    print(f"{len(J)} jobs pending -> {out}; {tot:.3g} walker-steps (~{tot * 115e-9 / 3600:.2f} core-h at 115 ns)", flush=True)
    if a.dry_run:
        for j in J:
            print(j[0], GF.DYNAMICS_ORDER[j[1]], j[2], j[3], f"{cost_walker_steps(j, a.smoke):.3g}")
        return
    warmup()
    from concurrent.futures import ProcessPoolExecutor, as_completed
    t0 = time.time()
    failed = []
    with ProcessPoolExecutor(a.workers) as ex:
        fs = {ex.submit(run, j, out, a.smoke): j for j in J}
        for k, f in enumerate(as_completed(fs)):
            try:
                c, name, h, gs, w = f.result()
            except Exception as e:                       # report, keep the other jobs running; rerun resumes it
                j = fs[f]
                failed.append(j)
                print(f"[{k + 1}/{len(J)}] FAILED {GF.DYNAMICS_ORDER[j[1]]} {tag(j[0], j[2])} groups {j[3]}: {e!r}", flush=True)
                continue
            print(f"[{k + 1}/{len(J)}] {name} {tag(c, h)} g{gs[0] if len(gs) == 1 else 'all'}: {w:.0f} s "
                  f"(elapsed {time.time() - t0:.0f} s)", flush=True)
    print(f"done ({len(failed)} failed)" if failed else "done", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
