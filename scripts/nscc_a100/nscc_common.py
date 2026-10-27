"""NSCC A100 study: common layer (jobs, paths, the run child for CPU and GPU backends, endpoints).

docs/nscc_a100/PREREGISTRATION.md, configs/nscc_a100/campaign.json.  The science definitions are those of the hybrid
short-T study and are imported, never re-implemented: systems / budgets / save grids / benchmark windows / endpoints
from scripts/hybrid_short_T/hst_common.py, the frozen e_F scorer and the CPU engine adapters from
scripts/final_campaign/expB_common.py.  Backends:
    S1    the validated serial numba engine (one core, its replay noise)          expB_common.run_serial
    P<t>  the validated walker-parallel CPU engine, t OpenMP threads (counter)   expB_common.run_parallel
    G1    the A100 engine of this study, one GPU (counter)                        run_gpu (src/gpu/*_gpu.py)
Jobs are JSON dicts; every run is a fresh process (scripts/nscc_a100/child.py) so that W_e2e (process start ->
persistent accuracy) includes imports, JIT / cache loads and, for G1, CUDA context creation.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in (os.path.join(ROOT, "src"), os.path.join(ROOT, "src", "gpu"), os.path.join(ROOT, "src", "parallel"),
           os.path.join(ROOT, "scripts"), os.path.join(ROOT, "scripts", "equal_budget"),
           os.path.join(ROOT, "scripts", "mechanism"), os.path.join(ROOT, "scripts", "final_campaign"),
           os.path.join(ROOT, "scripts", "hybrid_short_T"), HERE):
    if _p not in sys.path:
        sys.path.append(_p)

import expB_common as C  # noqa: E402
import hst_common as H  # noqa: E402

VERSION = "nscc_a100/1"
CONFIG = os.path.join(ROOT, "configs", "nscc_a100", "campaign.json")
OUT_ROOT = os.path.join(ROOT, "results", "nscc_a100")
SYSTEMS = H.SYSTEMS
CPU_THREADS = (1, 2, 4, 8, 16)
GPU_SOURCES = {"gateway": ("src/gpu/gateway_gpu.py", "src/gpu/gw_cuda.cu", "src/gpu/gpu_common.cuh",
                           "src/gpu/gpu_build.py"),
               "lta": ("src/gpu/lta_gpu.py", "src/gpu/lta_cuda.cu", "src/gpu/gpu_common.cuh", "src/gpu/gpu_build.py")}


class NSCCError(RuntimeError):
    pass


def load_config(path=None):
    with open(path or CONFIG) as fh:
        return json.load(fh)


def backend_parts(backend):
    if backend == "S1":
        return "serial", 1
    if backend == "G1":
        return "gpu", 1
    if backend.startswith("P") and backend[1:].isdigit():
        return "parallel", int(backend[1:])
    raise NSCCError(f"unknown backend {backend!r}")


def make_job(kind, system, method, N, seed, backend, r=None, n_steps=None):
    spec = H.system_spec(system)
    impl, threads = backend_parts(backend)
    N = int(N)
    if N == 1 and method == "fr":
        raise NSCCError("no FR arm at N = 1")
    if impl == "parallel" and threads > N:
        raise NSCCError(f"t {threads} > N {N}")
    if kind == "bench":
        n_steps = H.bench_window(spec, N)[0]
    elif n_steps is None:
        n_steps = H.n_steps_for(spec, N)
    j = dict(kind=kind, system=system, method=method, N=N, seed=int(seed), backend=backend, impl=impl,
             threads=threads, noise="replay" if impl == "serial" else "counter", n_steps=int(n_steps),
             arm=f"{method}{N}", config=f"{method}{N}_{backend}")
    if r is not None:
        j["r"] = int(r)
    return j


def job_paths(job, out_root=OUT_ROOT):
    tag = f"s{job['seed']}" + (f"_r{job['r']}" if "r" in job else "")
    base = os.path.join(out_root, job["kind"], job["system"], job["config"], tag)
    return dict(base=base, json=base + ".json", npz=base + ".npz", log=base + ".log")


def job_grid(job, spec):
    if job["kind"] == "bench":
        return H.bench_window(spec, job["N"])[1]
    return H.save_grid(spec, job["N"])[0]


def engine_sources(spec, impl):
    if impl == "gpu":
        return [os.path.join(ROOT, p) for p in GPU_SOURCES[spec["kind"]]]
    return C.engine_sources(spec, impl, spec["engine_module"])


def job_signature(job, spec):
    sig = dict(version=VERSION, kind=job["kind"], system=job["system"], method=job["method"], N=int(job["N"]),
               seed=int(job["seed"]), n_steps=int(job["n_steps"]), backend=job["backend"],
               noise=job["noise"], grid_sha=C.sha256_array(job_grid(job, spec)),
               engine_cfg=H.engine_cfg_digest(spec), B=int(spec["B"]), T_K=spec["T_K"],
               engine_sources=C.files_sha(engine_sources(spec, job["impl"])))
    if "r" in job:
        sig["r"] = int(job["r"])
    return json.loads(json.dumps(sig))


def record_status(job, out_root=OUT_ROOT, spec=None):
    p = job_paths(job, out_root)
    if not os.path.exists(p["json"]):
        return None
    try:
        rec = C.read_json(p["json"])
    except (OSError, ValueError):
        return "failed"
    if rec.get("status") != "complete":
        return rec.get("status") or "failed"
    spec = spec or H.system_spec(job["system"])
    return "complete" if rec.get("signature") == job_signature(job, spec) else "mismatch"


# ================================================================================================ the GPU adapter
def run_gpu(spec, method, N, seed, n_steps, save_steps, launch=None):
    """One run on the A100 engine through the frozen interface (expB_common.check_contract)."""
    if spec["kind"] == "gateway":
        import gateway_gpu as E
        cfg = dict(spec["engine_cfg"])
    else:
        import lta_gpu as E
        cfg = C.engine_cfg_for(spec, N, seed, n_steps)
    e0 = time.time()
    res = E.run_arm_gpu(cfg, method, int(N), int(seed), int(n_steps), np.asarray(save_steps, dtype=np.int64),
                        launch=launch)
    e1 = time.time()
    C.check_contract(res, spec["kind"], int(N), int(n_steps), save_steps)
    meta = json.loads(res["meta"])
    res = dict(res)
    res["t0_epoch"] = float(meta["t0_epoch"])
    res["t0_epoch_source"] = "engine meta (CLOCK_MONOTONIC t0 mapped to the epoch)"
    res["call_epoch"], res["return_epoch"] = e0, e1
    res["implementation"] = "gpu"
    return res


def gpu_prepare(spec):
    """Build / load the CUDA library before the timed call (part of the process start overhead)."""
    if spec["kind"] == "gateway":
        import gateway_gpu as E
    else:
        import lta_gpu as E
    E.load_lib()


# ================================================================================================ the run child
SAVE_KEYS = {"gateway": ("save_step", "M_all", "C_all", "fr_deaths_cum", "fr_opp_cum", "fr_opp_event_cum",
                         "wall_s_at_save", "n_force_evals_at_save", "X_final", "Y_final"),
             "lta": ("save_step", "M_all", "C_all", "M_prod", "C_prod", "cum_deaths", "cum_opps",
                     "cum_opps_with_event", "wall_s_at_save", "n_force_evals_at_save", "q_final")}


def run_child(job, out_root=OUT_ROOT):
    """Execute one job in this (fresh) process; write the compressed arrays (prod / confirm) and the JSON record
    (endpoints computed after the timed run)."""
    import expB_alloc as A
    try:
        proc_start = time.time() - float(A.process_age_s())
    except Exception:  # noqa: BLE001
        proc_start = math.nan
    t_imp0 = time.time()
    spec = H.system_spec(job["system"])
    paths = job_paths(job, out_root)
    sig = job_signature(job, spec)
    saves = job_grid(job, spec)
    if job["impl"] == "parallel":
        mod = C.parallel_engine(spec["system"], spec["engine_module"])
        if callable(getattr(mod, "load_kernel", None)):
            mod.load_kernel()
    elif job["impl"] == "serial":
        import parallel_benchmark  # noqa: F401
        __import__(spec["validated_module"])
    else:
        gpu_prepare(spec)
    import_s = time.time() - t_imp0
    t_call = time.time()
    if job["impl"] == "serial":
        res = C.run_serial(spec, job["method"], job["N"], job["seed"], job["n_steps"], saves)
    elif job["impl"] == "parallel":
        res = C.run_parallel(spec, job["method"], job["N"], job["seed"], job["n_steps"], saves, job["threads"],
                             job["noise"], module=spec["engine_module"])
    else:
        res = run_gpu(spec, job["method"], job["N"], job["seed"], job["n_steps"], saves, launch=job.get("launch"))
    t_ret = time.time()
    kind = spec["kind"]
    meta = C.parse_meta(res.get("meta"))
    t0 = float(res.get("t0_epoch", t_call))
    rec = dict(status="complete", version=VERSION, job=job, signature=sig, pid=os.getpid(),
               system=spec["system"], N=int(job["N"]), method=job["method"], seed=int(job["seed"]),
               backend=job["backend"], n_steps=int(job["n_steps"]), T_N=float(job["n_steps"] * spec["h"]),
               B=int(spec["B"]), h=spec["h"], T_K=spec["T_K"], engine=str(res.get("engine")),
               noise_mode=str(res.get("noise_mode")), precision="float64",
               science_sha256={k: C.sha256_array(res[k]) for k in C.SCORE_KEYS[kind] + ("save_step",)},
               wall_total_s=float(res["wall_total_s"]), warmup_s=float(res.get("warmup_s", 0.0)),
               timers={k: float(res.get(k, math.nan)) for k in C.TIMER_KEYS},
               t0_epoch=t0, proc_start_epoch=proc_start, import_s=import_s, call_epoch=t_call, return_epoch=t_ret,
               start_overhead_s=(t0 - proc_start) if math.isfinite(proc_start) else math.nan,
               n_force_evals=int(np.asarray(res["n_force_evals"])), n_saves=int(len(saves)),
               launch=meta.get("launch"), device=meta.get("device"), meta_engine=meta.get("engine"),
               host=os.uname().nodename, cpus=sorted(os.sched_getaffinity(0)),
               pbs_jobid=os.environ.get("PBS_JOBID"), cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
               git_commit=os.environ.get("ABFFR_GIT_COMMIT"),
               finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    fr_key = "fr_deaths_cum" if kind == "gateway" else "cum_deaths"
    rec["fr_deaths_total"] = int(np.asarray(res[fr_key])[-1]) if fr_key in res else None
    os.makedirs(os.path.dirname(paths["json"]), exist_ok=True)
    if job["kind"] == "bench":
        hjob = dict(N=int(job["N"]))
        rec["bench"] = H.bench_metrics(res, spec, hjob)
        rec["wall_s_at_save"] = np.asarray(res["wall_s_at_save"], float).tolist()
    else:
        scorer = H.Scorer(spec)
        hjob = dict(N=int(job["N"]), n_steps=int(job["n_steps"]), threads=int(job["threads"]),
                    impl=("serial" if job["impl"] == "serial" else "parallel"), kind=job["kind"],
                    system=job["system"], config=job["config"], seed=job["seed"], replicate=0)
        ep, e = H.run_endpoints(res, spec, scorer, hjob, rec=rec)
        ep["backend"] = job["backend"]
        rec["endpoints"] = ep
        rec["e_F"] = np.asarray(e, float).tolist()
        rec["wall_s_at_save"] = np.asarray(res["wall_s_at_save"], float).tolist()
        arrays = {k: np.asarray(res[k]) for k in SAVE_KEYS[kind] if k in res}
        arrays["e_F"] = np.asarray(e, float)
        tmp = paths["npz"] + f".tmp{os.getpid()}.npz"
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, paths["npz"])
    rec["child_wall_s"] = time.time() - proc_start if math.isfinite(proc_start) else math.nan
    C.write_json(paths["json"], rec)
    return rec


def summary_line(rec):
    j = rec["job"]
    s = f"{j['kind']} {j['system']} {j['config']} s{j['seed']}{('_r' + str(j['r'])) if 'r' in j else ''}"
    if "bench" in rec:
        b = rec["bench"]
        return f"DONE {s} c_ss {b['c_ss'] * 1e9:.2f} ns  {b['c_ss'] * j['N'] * 1e6:.3f} us/step  wall {rec['wall_total_s']:.2f}s"
    ep = rec["endpoints"]
    return (f"DONE {s} W_mid {ep['W_mid']:.3f} W_mid_e2e {ep['W_mid_e2e']:.3f} final_e_F {ep['final_e_F']:.4g} "
            f"wall {rec['wall_total_s']:.2f}s start_overhead {rec['start_overhead_s']:.2f}s")
