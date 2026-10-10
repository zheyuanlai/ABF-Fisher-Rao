#!/usr/bin/env python
"""Per-step wall cost at N = 512 of every implementation of the frozen production algorithms (Experiment III
planning, docs/mechanism/SCIENTIFIC_PLAN.md section 8), and the full-campaign projection.

    CUDA_VISIBLE_DEVICES=2 taskset -c 72 python scripts/mechanism/bench_step_costs.py --device cuda:0 --out F.json
    CUDA_VISIBLE_DEVICES="" taskset -c 64 python scripts/mechanism/bench_step_costs.py --device cpu --out F.json
    python scripts/mechanism/bench_step_costs.py --compare GPU.json CPU.json [--out RATIOS.json]

GPU guard (as parallel_benchmark.run_one): CUDA_VISIBLE_DEVICES must name ONE allowed benchmark GPU
(parallel_benchmark.allowed_gpus(), [2]); a device holding another compute process, or whose occupancy cannot be read,
is refused unless --allow-busy-gpu (recorded).  The measuring process must be pinned to ONE logical CPU (taskset).
--compare: GPU / CPU per-step cost ratios ONLY between rows whose host SMT-sibling state is known and equal (both
contended or both uncontended, parallel_benchmark.smt_state); other pairs are listed as UNMATCHED, never as a ratio.

Implementations, each timed (time.perf_counter; device synchronised) on a short run after one warm-up run:
  lean      scripts/mechanism/bench_torch_engines.py (the gpu_torch backend of parallel_benchmark.py)
  asis      the unmodified torch engines: gateway_core.simulate_batch (one row, estimator 'histogram', store nothing
            extra) and core_lta.run_sampler (one row, abf_estimator 'histogram'), with their per-step diagnostics
  numba     (device cpu only) the production engines' run_arm, diagnostics off (the benchmark default) and on (as in
            the equal-budget production)
Steady-state knobs: the frozen production engine_cfg, except that for LTA the bias ramp, the production-accumulator
burn-in and the FR start are moved to step 100 (runtime knobs only) so that every phase of the production loop is
active in the short run; the gateway's FR runs from step 0 as in production.  On the CPU the torch LTA engines are
skipped (about 40 ms per step at N = 512).

Projection: us/step x the full budget n_steps (gateway 6.4e6, LTA 6e5) per run, + the measured warm-up, x 8 seeds
x 2 arms x 3 systems; GPU runs are sequential on one device (GPU-hours), CPU runs one per core (core-hours).
Every measurement records the busy fraction of its pinned CPU's SMT siblings (/proc/stat) during the timed run and is
flagged smt_contended above 10 %: a busy sibling slows a numba run ~1.6x and the GPU's host (launch) thread too.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in (os.path.join(ROOT, "src"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
for _k in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.expanduser("~/.cache/numba_mech"))
import numpy as np  # noqa: E402

import parallel_benchmark as PB  # noqa: E402

N = 512
STEADY_LTA = dict(warmup_steps=100, burn_in_steps=100, fr_start_steps=100)
DEFAULT_STEPS = dict(gateway=dict(lean=16000, asis=8000, numba=64000), lta=dict(lean=1500, asis=1000, numba=3000))


def _timed(fn, sync):
    sync()
    t = time.perf_counter()
    fn()
    sync()
    return time.perf_counter() - t


def measure(system, impl, method, device, n_steps, diagnostics=False):
    P, _, _ = PB.system_spec(system)
    kind = PB.SYSTEMS[system]["kind"]
    seed = int(P["seeds"][0])
    cfg = dict(P["engine_cfg"])
    if kind == "lta":
        cfg.update(STEADY_LTA)
    if impl == "numba":
        if kind == "gateway":
            import gateway_ladder_numba as E

            def go(n):
                E.run_arm(cfg, method, N, seed, n, np.array([n]), diagnostics=diagnostics)
        else:
            import lta_ladder_numba as E

            def go(n):
                E.run_arm(E.make_cfg(P["T_K"], N, seed, **dict(cfg, n_steps=n, diagnostics=diagnostics)), method,
                          np.array([n]))
        sync = (lambda: None)
    else:
        import torch
        torch.set_num_threads(1)
        dev = torch.device(device)
        sync = (lambda: torch.cuda.synchronize(dev)) if dev.type == "cuda" else (lambda: None)
        if impl == "lean":
            import bench_torch_engines as BT

            def go(n):
                if kind == "gateway":
                    BT.GatewayTorchRun(cfg, method, N, seed, n, np.array([n]), dev, fr_timing=False).run()
                else:
                    BT.LTATorchRun(cfg, P["T_K"], method, N, seed, n, np.array([n]), dev, fr_timing=False).run()
        elif kind == "gateway":
            import gateway_core as gc
            import bench_torch_engines as BT
            kn = BT.gateway_knobs(cfg, N)

            def go(n):
                g = gc.GatewayConfig(beta=cfg["beta"], H=cfg["H"], omega_out=cfg["omega_out"],
                                     r=cfg["omega_in"] / cfg["omega_out"], s=cfg["s"], N=N, dt=cfg["h"], n_steps=n,
                                     save_every=n, estimator="histogram", n_bins=int(cfg["nb"]),
                                     min_count=cfg["min_count"], gamma=cfg["gamma"], eta=cfg["eta"],
                                     fr_every=kn["fr_every"], fr_burnin=0, ramp_fraction=kn["ramp_steps"] / n,
                                     score_clip=cfg["score_clip"], max_event_fraction=cfg["max_event_fraction"])
                assert int(g.ramp_fraction * n) == kn["ramp_steps"]
                gc.simulate_batch(gc.BatchSpec([g], [seed], [gc.ABF if method == "abf" else gc.FR_UNIFORM],
                                               batch_seed=seed), device=dev, dtype=torch.float64)
        else:
            from lta import core_lta as CL
            import bench_torch_engines as BT
            system_ = CL.LTASystem(CL.LTAParams(temperature=P["T_K"]), dev, torch.float64, root=ROOT)

            def go(n):
                sim = BT.lta_sim_config(cfg, P["T_K"], N, n, seed)
                sim.save_every = n
                CL.run_sampler("abf" if method == "abf" else "fr_uniform", system_, sim, [seed], verbose=False)
    nw = max(4, n_steps // 20) if kind == "lta" else 2 * 160
    warm = _timed(lambda: go(nw), sync)
    aff = sorted(os.sched_getaffinity(0))
    sib = PB.smt_siblings(aff)
    c0 = PB.cpu_times(set(aff) | set(sib))
    wall = _timed(lambda: go(n_steps), sync)
    c1 = PB.cpu_times(set(aff) | set(sib))
    bf = PB.busy_fraction(c0, c1)
    state, sib_busy = PB.smt_state(aff, sib, bf)
    return dict(system=system, impl=impl, method=method, device=str(device), diagnostics=bool(diagnostics),
                n_steps=int(n_steps), wall_s=wall, us_per_step=wall * 1e6 / n_steps,
                ns_per_walker_step=wall * 1e9 / (n_steps * N), warmup_s=warm, warmup_steps=nw,
                affinity=aff, smt_siblings=sib, cpu_busy_fraction=bf, smt_sibling_busy_max=sib_busy,
                smt_state=state, smt_contended=(None if state == "unknown" else state == "contended"),
                measured_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


def projection(rows, seeds_per_arm=PB.DEFAULT_SEEDS_PER_ARM):
    out = {}
    for impl in sorted({(r["impl"], r["device"], r["diagnostics"]) for r in rows}):
        tot, parts, complete = 0.0, {}, True
        for system in PB.SYSTEMS:
            P, _, _ = PB.system_spec(system)
            nf = PB.full_n_steps(P, N)
            for m in PB.METHODS:
                rr = [r for r in rows if (r["impl"], r["device"], r["diagnostics"]) == impl and r["system"] == system
                      and r["method"] == m]
                if not rr:
                    complete = False
                    continue
                s = rr[0]["us_per_step"] * 1e-6 * nf + rr[0]["warmup_s"]
                parts[f"{system}/{m}"] = dict(us_per_step=rr[0]["us_per_step"], s_per_run=s)
                tot += s * seeds_per_arm
        key = f"{impl[0]}@{impl[1]}" + ("+diag" if impl[2] else "")
        cont = sorted({f"{r['system']}/{r['method']}" for r in rows
                       if (r["impl"], r["device"], r["diagnostics"]) == impl and r.get("smt_contended")})
        out[key] = dict(total_h=tot / 3600.0, complete=complete, seeds_per_arm=seeds_per_arm, runs=parts,
                        smt_contended_measurements=cont,
                        unit="GPU-h (sequential, one device)" if impl[1].startswith("cuda") else "core-h")
    return out


def _row_state(r):
    """Host SMT-sibling state of a measurement row ('unknown' for rows without the telemetry)."""
    st = r.get("smt_state")
    if st:
        return st
    c = r.get("smt_contended")
    sb = r.get("smt_sibling_busy_max")
    if c is None or sb is None or (isinstance(sb, float) and not math.isfinite(sb)):
        return "unknown"
    return "contended" if c else "uncontended"


def matched_ratios(gpu_rows, cpu_rows, gpu_impl="lean", cpu_impl="numba"):
    """GPU / CPU per-step cost ratio per (system, method), ONLY for rows whose host sibling state is known and
    equal.  Returns {"system/method": {...}}; unmatched pairs carry status 'UNMATCHED' and no ratio."""
    out = {}
    for g in gpu_rows:
        if g["impl"] != gpu_impl or not str(g["device"]).startswith("cuda"):
            continue
        for c in cpu_rows:
            if (c["impl"] != cpu_impl or c["system"] != g["system"] or c["method"] != g["method"]
                    or c.get("diagnostics")):
                continue
            sg, sc = _row_state(g), _row_state(c)
            key = f"{g['system']}/{g['method']}"
            rec = dict(gpu_us_per_step=g["us_per_step"], cpu_us_per_step=c["us_per_step"], gpu_host_state=sg,
                       cpu_state=sc, gpu_affinity=g.get("affinity"), cpu_affinity=c.get("affinity"),
                       gpu_measured_utc=g.get("measured_utc"), cpu_measured_utc=c.get("measured_utc"))
            if sg == sc and sg != "unknown":
                rec.update(status=f"MATCHED ({sg})", ratio_gpu_over_cpu=g["us_per_step"] / c["us_per_step"])
            else:
                rec.update(status="UNMATCHED (host sibling state unknown or different): no ratio reported")
            out[key] = rec
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--systems", nargs="*", default=sorted(PB.SYSTEMS))
    ap.add_argument("--impls", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--allow-busy-gpu", action="store_true")
    ap.add_argument("--compare", nargs=2, metavar=("GPU_JSON", "CPU_JSON"), default=None)
    a = ap.parse_args(argv)
    if a.compare:
        rows = [json.load(open(f))["rows"] for f in a.compare]
        res = matched_ratios(rows[0], rows[1])
        for k, v in res.items():
            print(f"{k:16s} GPU {v['gpu_us_per_step']:8.1f} us/step [{v['gpu_host_state']}] / CPU "
                  f"{v['cpu_us_per_step']:7.1f} [{v['cpu_state']}]: " + (f"{v['ratio_gpu_over_cpu']:.2f}x"
                                                                     if "ratio_gpu_over_cpu" in v else v["status"]))
        if a.out:
            with open(a.out, "w") as fh:
                json.dump(dict(compare=list(a.compare), ratios=res), fh, indent=1)
        return 0
    if not a.out:
        raise SystemExit("--out is required for a measurement")
    aff = sorted(os.sched_getaffinity(0))
    if len(aff) != 1:
        raise SystemExit(f"pin the measuring process to ONE logical CPU (taskset -c <cpu>); affinity has {len(aff)}")
    gpu_before = None
    if not a.device.startswith("cuda"):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        try:
            _, gpu_before = PB.gpu_guard(allow_busy=a.allow_busy_gpu)
        except PB.BenchError as e:
            raise SystemExit(f"GPU guard: {e}")
    impls = a.impls or (["lean", "asis"] if a.device.startswith("cuda") else ["numba", "lean", "asis"])
    rows = []
    for system in a.systems:
        kind = PB.SYSTEMS[system]["kind"]
        for impl in impls:
            if kind == "lta" and impl in ("lean", "asis") and not a.device.startswith("cuda"):
                continue
            for diag in ((False, True) if impl == "numba" else (False,)):
                for m in PB.METHODS:
                    r = measure(system, impl, m, a.device, DEFAULT_STEPS[kind][impl], diagnostics=diag)
                    rows.append(r)
                    print(f"{system:8s} {impl:6s}{'+diag' if diag else '':6s} {m:4s} {a.device}: "
                          f"{r['us_per_step']:9.1f} us/step ({r['ns_per_walker_step']:8.1f} ns/walker-step) "
                          f"warm-up {r['warmup_s']:.2f} s, SMT sibling busy {r['smt_sibling_busy_max']:.2f}"
                          f"{' (CONTENDED)' if r['smt_contended'] else ''}", flush=True)
    meta = dict(generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), host=os.uname().nodename,
                affinity=sorted(os.sched_getaffinity(0)), smt_siblings=PB.smt_siblings(sorted(os.sched_getaffinity(0))),
                loadavg=os.getloadavg(), physical_gpu=PB.physical_gpu(), gpu_processes_before=gpu_before,
                gpu_processes_after=((PB.gpu_processes() or {}).get("processes", {}).get(PB.physical_gpu())
                                     if a.device.startswith("cuda") else None),
                N=N, steady_state_lta=STEADY_LTA,
                provenance=PB.provenance(["scripts/mechanism/bench_step_costs.py", "scripts/mechanism/bench_torch_engines.py",
                                          "src/gateway_core.py", "src/lta/core_lta.py", "src/gateway_ladder_numba.py",
                                          "src/lta_ladder_numba.py"]))
    out = dict(meta=meta, rows=rows, projection=projection(rows))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1, default=PB._json_default)
    for k, v in out["projection"].items():
        print(f"projection {k}: {v['total_h']:.2f} {v['unit']} (8 seeds x 2 arms x systems measured; complete "
              f"{v['complete']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
