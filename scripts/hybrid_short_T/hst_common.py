"""Hybrid short-T wall-clock study: common layer (systems, budgets, save grids, jobs, the run child, endpoints).

docs/hybrid_short_T/PREREGISTRATION.md and configs/hybrid_short_T/campaign.json (frozen 2026-10-18, ef3b6453).
No engine code lives here.  Runs reach the engines ONLY through the Experiment B adapters
(scripts/final_campaign/expB_common.py):

    run_serial(spec, method, N, seed, n_steps, save_steps)            S1: the validated serial numba engine, its own
                                                                      replay noise, SaveClock hooks (warm-up excluded)
    run_parallel(spec, method, N, seed, n_steps, save_steps, t, noise, module)
                                                                      P_t: src/parallel (counter-mode Philox noise)

``system_spec`` builds the dict those adapters (and expB_common.Scorer) read, for the three systems of this study at
their SHORTENED budgets (the Experiment B spec only knew gateway / LTA 150 K at the original budgets).

Definitions (prereg sections 2, 4, 6)
  n_steps(N)      B / N (B = the shortened budget; integer at every rung, ExpBError otherwise)
  save grid       engine.budget_save_grid(B_orig / N) (200 uniform + 24 log-spaced fractions of the ORIGINAL run)
                  restricted to steps <= n_steps(N), union the frozen physical checkpoints t <= T_N; the last save is
                  n_steps(N) = the k*-th original uniform point (checked)
  budget grid     the 24 log points + the k* uniform points: the same fractions u / u* at every N; persistence (tau)
                  is judged on these saves only (the repository's cross-N tau_bgrid convention)
  uniform saves   the k* original uniform points (Ibar_F = mean of e_F over them, as results/equal_budget_v2/short_T)
  tau_eps         first budget-grid save after which e_F <= eps at EVERY later budget-grid save; censored -> -1
  W_eps           wall_s_at_save[tau] (s since the engine's timed start; excludes process start, imports, JIT /
                  cache load and the engine warm-up); censored = inf (never imputed)
  W_eps_e2e       W_eps + the run's start overhead (process start .. timed start, minus the monitor hold)
Bench window (prereg section 4): n_b(N) = max(bench_min_steps, ceil(bench_walker_steps / N)), 20 uniform saves (+
  fr_start for LTA); c_ss = (wall at the last save - wall at s0) / (N (n_b - s0)), s0 = the first save >= max(n_b / 2,
  fr_start (LTA)).

Jobs (dicts, JSON-serialisable):  kind in {'bench', 'prod', 'confirm', 'valid'}, system, method, N, seed, impl in
  {'serial', 'parallel'}, threads, noise ('replay' for serial, 'counter' for parallel), n_steps, replicate, and for
  bench the repeat index r.  Paths: results/hybrid_short_T/<kind>/<system>/<arm>/s<seed>[.rep<k>].{json,npz,...}
  (bench: <kind>/<system>/<config>/r<r>_s<seed>[.rep<k>]).
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
for _p in (os.path.join(ROOT, "src"), os.path.join(ROOT, "src", "parallel"), os.path.join(ROOT, "scripts"),
           os.path.join(ROOT, "scripts", "equal_budget"), os.path.join(ROOT, "scripts", "mechanism"),
           os.path.join(ROOT, "scripts", "final_campaign"), HERE):
    if _p not in sys.path:
        sys.path.append(_p)

import expB_common as C  # noqa: E402

VERSION = "hybrid_short_T/1"
CONFIG = os.path.join(ROOT, "configs", "hybrid_short_T", "campaign.json")
OUT_ROOT = os.path.join(ROOT, "results", "hybrid_short_T")
FIG_ROOT = os.path.join(ROOT, "figures", "hybrid_short_T")
DOC_ROOT = os.path.join(ROOT, "docs", "hybrid_short_T")
LEDGER = os.path.join(OUT_ROOT, "compute_ledger.csv")
REGISTRY_DIR = "/tmp/claude-1008/final_campaign_alloc"      # the shared CCD lock registry of this user's campaigns
SYSTEMS = ("gateway", "lta300", "lta150")
THR_NAMES = ("strict", "mid", "loose")
N_UNIFORM = 200
BENCH_N_SAVES = 20
CCD_CORES = 8


class HSTError(RuntimeError):
    """A violated contract, config mismatch or plan rule of this study."""


# ================================================================================================ config / systems
def load_config(path=None):
    with open(path or CONFIG) as fh:
        return json.load(fh)


def system_spec(system, cfg=None):
    """Everything the harness needs about one system at its shortened budget, in the dict layout of
    expB_common.run_serial / run_parallel / Scorer.  Cross-checks h, B = u* B_orig, the thresholds and T* against the
    frozen equal-budget production config and the short-T summary (HSTError on any mismatch)."""
    X = cfg if cfg is not None else load_config()
    if system not in X["systems"]:
        raise HSTError(f"unknown system {system!r}; expected one of {sorted(X['systems'])}")
    sx = X["systems"][system]
    P = C.load_json(os.path.join(ROOT, sx["eqb_config"]))
    ecfg = dict(P["engine_cfg"])
    thr = {n: float(sx["thresholds_e_F"][n]) for n in THR_NAMES}
    probs = []
    if float(sx["h"]) != float(ecfg["h"]):
        probs.append(f"h {sx['h']} vs production {ecfg['h']}")
    if int(sx["B_orig"]) != int(P["B"]):
        probs.append(f"B_orig {sx['B_orig']} vs production B {P['B']}")
    if [thr[n] for n in THR_NAMES] != [float(v) for v in P["thresholds"]["e_F"]]:
        probs.append(f"e_F thresholds {thr} vs production {P['thresholds']['e_F']}")
    k_star = int(sx["k_star"])
    if abs(k_star / N_UNIFORM - float(sx["u_star"])) > 1e-12:
        probs.append(f"k* {k_star} != u* {sx['u_star']} x {N_UNIFORM}")
    if int(sx["B"]) * N_UNIFORM != int(sx["B_orig"]) * k_star:
        probs.append(f"B {sx['B']} != k*/200 x B_orig")
    if list(sx["N_ladder"]) != list(P["N_ladder"]):
        probs.append("N ladder differs from production")
    if int(sx["N0"]) != int(P["anchor"]["N0"]):
        probs.append("N0 differs from production anchor")
    if abs(float(sx["T_star"]) - int(sx["B"]) // int(sx["N0"]) * float(sx["h"])) > 1e-9:
        probs.append("T* != (B / N0) h")
    if (sx["T_K"] is None) != (sx["kind"] == "gateway") or (sx["T_K"] is not None and
                                                            float(sx["T_K"]) != float(P.get("T_K"))):
        probs.append("T_K mismatch")
    if probs:
        raise HSTError(f"{system}: campaign.json and the frozen production config disagree: " + "; ".join(probs))
    if sx["kind"] == "lta":
        ecfg.pop("diagnostics", None)      # diagnostics off in every run of this study (bit-inert)
    return dict(
        system=system, kind=sx["kind"], label=sx["label"], eqb_system=sx["eqb_system"],
        T_K=(float(sx["T_K"]) if sx["T_K"] is not None else None), P=P, engine_cfg=ecfg, h=float(ecfg["h"]),
        B=int(sx["B"]), B_orig=int(sx["B_orig"]), u_star=float(sx["u_star"]), k_star=k_star, N0=int(sx["N0"]),
        T_star=float(sx["T_star"]), N_ladder=[int(n) for n in sx["N_ladder"]], thresholds=thr,
        engine_module=sx["engine_module"], validated_module=sx["validated_module"],
        prod_seeds=[int(s) for s in sx["prod_seeds"]], confirm_seeds=[int(s) for s in sx["confirm_seeds"]],
        bench_seeds=[int(s) for s in sx["bench_seeds"]], validation_seeds=[int(s) for s in sx["validation_seeds"]],
        production_reference_seeds=[int(s) for s in sx["production_reference_seeds"]],
        bench_walker_steps=int(sx["bench_walker_steps"]), bench_min_steps=int(sx["bench_min_steps"]),
        eqb_dir=os.path.join(ROOT, sx["eqb_dir"]), k_alloc=CCD_CORES,
        fr_start=(int(ecfg["fr_start_steps"]) if sx["kind"] == "lta" else 0))


def all_specs(cfg=None):
    return {s: system_spec(s, cfg) for s in SYSTEMS}


def n_steps_for(spec, N):
    B, N = int(spec["B"]), int(N)
    if N < 1 or B % N:
        raise HSTError(f"{spec['system']}: budget {B} is not divisible by N {N}")
    return B // N


def T_N(spec, N):
    return n_steps_for(spec, N) * spec["h"]


def methods_for(N):
    return ("abf",) if int(N) == 1 else ("abf", "fr")


def arm_name(method, N):
    if method not in ("abf", "fr"):
        raise HSTError(f"method must be 'abf' or 'fr', not {method!r}")
    return f"{method}{int(N)}"


def config_name(method, N, impl, threads):
    return f"{arm_name(method, N)}_{'S1' if impl == 'serial' else 'P' + str(int(threads))}"


def backend_label(impl, threads):
    return "S1" if impl == "serial" else f"P{int(threads)}"


def engine_module_for(spec):
    import expB_common as _C  # noqa: F401  (the adapters resolve 'gateway_parallel' / 'lta_parallel' themselves)
    return spec["engine_module"]


# ================================================================================================ save grids
def _engine(kind):
    if kind == "gateway":
        import gateway_ladder_numba as E
    else:
        import lta_ladder_numba as E
    return E


def save_grid(spec, N):
    """(steps int64, budget_mask bool, uniform_mask bool) of the rung's save grid (module docstring).  The
    uniform saves are exactly the k* original uniform points k (B_orig / N) / 200, k = 1..k*."""
    N = int(N)
    n_steps = n_steps_for(spec, N)
    n_orig = int(spec["B_orig"]) // N
    if n_orig * N != int(spec["B_orig"]):
        raise HSTError(f"B_orig not divisible by N {N}")
    E = _engine(spec["kind"])
    steps, _ = E.budget_save_grid(n_orig)
    steps = np.asarray(steps, dtype=np.int64)
    bud = steps[steps <= n_steps]
    h = float(spec["h"])
    phys = [int(round(t / h)) for t in spec["P"]["physical_checkpoints_t"] if 0 < round(t / h) <= n_steps]
    allst = np.unique(np.concatenate([bud, np.asarray(phys, dtype=np.int64)]))
    if allst[-1] != n_steps:
        raise HSTError(f"{spec['system']} N {N}: last save {allst[-1]} != n_steps {n_steps}")
    k = np.arange(1, spec["k_star"] + 1, dtype=np.int64)
    if (n_orig * spec["k_star"]) % N_UNIFORM or n_orig % N_UNIFORM:
        raise HSTError(f"{spec['system']} N {N}: uniform points are not integer steps")
    uni = k * (n_orig // N_UNIFORM)
    if uni[-1] != n_steps or not np.all(np.isin(uni, bud)):
        raise HSTError(f"{spec['system']} N {N}: the k* uniform points are not on the grid / do not end at n_steps")
    budget_mask = np.isin(allst, bud)
    uniform_mask = np.isin(allst, uni)
    if int(uniform_mask.sum()) != spec["k_star"]:
        raise HSTError("uniform save count != k*")
    return allst, budget_mask, uniform_mask


def bench_window(spec, N):
    """(n_b, save steps, s0): the benchmark window of a rung (prereg section 4)."""
    N = int(N)
    n_b = max(int(spec["bench_min_steps"]), int(math.ceil(spec["bench_walker_steps"] / N)))
    n_b = min(n_b, n_steps_for(spec, N))
    st = np.unique(np.clip(np.round(np.arange(1, BENCH_N_SAVES + 1) / BENCH_N_SAVES * n_b).astype(np.int64), 1,
                           n_b))
    if spec["kind"] == "lta" and 0 < spec["fr_start"] < n_b:
        st = np.unique(np.concatenate([st, [spec["fr_start"]]]))
    lo = max(n_b / 2.0, float(spec["fr_start"]))
    cand = st[st >= lo]
    s0 = int(cand[0]) if cand.size else int(st[0])
    if s0 >= n_b:
        raise HSTError(f"{spec['system']} N {N}: empty steady-state window (n_b {n_b}, s0 {s0})")
    return int(n_b), st, s0


# ================================================================================================ jobs and paths
def make_job(kind, spec, method, N, seed, impl, threads, replicate=0, r=None, n_steps=None):
    if impl not in ("serial", "parallel"):
        raise HSTError(f"impl {impl!r}")
    threads = 1 if impl == "serial" else int(threads)
    if int(N) == 1 and method == "fr":
        raise HSTError("no FR arm at N = 1")
    if impl == "parallel" and threads > int(N):
        raise HSTError(f"t {threads} > N {N}")
    if kind == "bench":
        n_steps = bench_window(spec, N)[0]
    elif n_steps is None:
        n_steps = n_steps_for(spec, N)
    j = dict(kind=kind, system=spec["system"], method=method, N=int(N), seed=int(seed), impl=impl,
             threads=int(threads), noise="replay" if impl == "serial" else "counter", n_steps=int(n_steps),
             replicate=int(replicate), arm=arm_name(method, N), config=config_name(method, N, impl, threads))
    if r is not None:
        j["r"] = int(r)
    return j


def job_key(job):
    """Identity of a job across replicates (the CCD lock carries it: never two live runs of one key)."""
    b = f"hst|{job['kind']}|{job['system']}|{job['config']}|s{job['seed']}"
    if job["kind"] == "bench":
        b += f"|r{job['r']}"
    return b + f"|rep{job['replicate']}"


def job_paths(job, out_root=OUT_ROOT):
    rep = "" if job["replicate"] == 0 else f".rep{job['replicate']}"
    if job["kind"] == "bench":
        base = os.path.join(out_root, "bench", job["system"], job["config"], f"r{job['r']}_s{job['seed']}{rep}")
    else:
        base = os.path.join(out_root, job["kind"], job["system"], job["arm"], f"s{job['seed']}{rep}")
    return dict(base=base, npz=base + ".npz", json=base + ".json", tel_jsonl=base + ".telemetry.jsonl",
                tel_npz=base + ".telemetry.npz", log=base + ".log", pause=base + ".diskpause.json",
                ledger_mark=base + ".ledger_logged")


def job_grid(job, spec):
    if job["kind"] == "bench":
        return bench_window(spec, job["N"])[1]
    return save_grid(spec, job["N"])[0]


def engine_cfg_digest(spec):
    return hashlib.sha256(json.dumps(spec["engine_cfg"], sort_keys=True).encode()).hexdigest()[:16]


def job_signature(job, spec):
    """Everything that defines the science and timing configuration of a job (not the replicate index)."""
    src = C.engine_sources(spec, job["impl"], spec["engine_module"])
    sig = dict(version=VERSION, kind=job["kind"], system=job["system"], method=job["method"], N=int(job["N"]),
               seed=int(job["seed"]), n_steps=int(job["n_steps"]), impl=job["impl"], threads=int(job["threads"]),
               noise=job["noise"], grid_sha=C.sha256_array(job_grid(job, spec)), engine_cfg=engine_cfg_digest(spec),
               B=int(spec["B"]), T_K=spec["T_K"], engine_sources=C.files_sha(src))
    if job["kind"] == "bench":
        sig["r"] = int(job["r"])
    return json.loads(json.dumps(sig))


def record_status(paths, sig):
    """'complete' (finalised, signature matches) | 'ran' (child finished, not finalised) | 'mismatch' | 'failed' |
    None (no record)."""
    if not os.path.exists(paths["json"]):
        return None
    try:
        rec = C.read_json(paths["json"])
    except (OSError, ValueError):
        return "failed"
    st = rec.get("status")
    if st not in ("complete", "ran"):
        return st or "failed"
    if rec.get("signature") != sig:
        return "mismatch"
    return "complete" if st == "complete" else "ran"


# ================================================================================================ the run (child)
def _proc_start_epoch():
    """Wall-clock epoch at which this process started (/proc start time vs /proc/uptime)."""
    import expB_alloc as A
    try:
        age = A.process_age_s()
        return time.time() - float(age)
    except Exception:  # noqa: BLE001
        return math.nan


def bench_metrics(res, spec, job):
    """Throughput numbers of one benchmark run (prereg section 4)."""
    n_b, st, s0 = bench_window(spec, job["N"])
    k = np.asarray(res["save_step"], dtype=np.int64)
    w = np.asarray(res["wall_s_at_save"], dtype=float)
    if not np.array_equal(k, st):
        raise HSTError("bench save grid mismatch")
    i0 = int(np.flatnonzero(k == s0)[0])
    N = int(job["N"])
    ws_ss = N * (int(k[-1]) - int(k[i0]))
    c_ss = (float(w[-1]) - float(w[i0])) / ws_ss
    ws = N * n_b
    return dict(n_b=n_b, s0=s0, walker_steps=ws, walker_steps_ss=ws_ss, c_ss=c_ss, throughput_ss=1.0 / c_ss,
                c_total=float(res["wall_total_s"]) / ws, throughput_total=ws / float(res["wall_total_s"]))


def run_job_child(job, out_root=OUT_ROOT, hold=False, hold_timeout=6.0, cfg=None):
    """Execute one job inside its pinned process; write the npz (prod / confirm / valid) and the JSON record with
    status 'ran' (the driver's finalisation sets 'complete')."""
    import expB_alloc as A
    proc_start = _proc_start_epoch()
    t_import0 = time.time()
    host = C.host_info()            # affinity at start (libgomp later binds the main thread to its first place)
    spec = system_spec(job["system"], cfg)
    paths = job_paths(job, out_root)
    sig = job_signature(job, spec)
    saves = job_grid(job, spec)
    holds = {}
    if job["impl"] == "parallel":
        mod = C.parallel_engine(spec["system"], spec["engine_module"])
        if callable(getattr(mod, "load_kernel", None)):      # gateway: build / load the C kernel
            mod.load_kernel()
    else:
        import parallel_benchmark  # noqa: F401
        __import__(spec["validated_module"])
    import_s = time.time() - t_import0
    hold_s = 0.0
    if hold:
        holds["start"] = A.hold_for_monitor(paths["tel_jsonl"], after=time.time(), timeout=hold_timeout,
                                            boundary="start")
        hold_s = float(holds["start"]["waited_s"])
    t_call = time.time()
    if job["impl"] == "serial":
        res = C.run_serial(spec, job["method"], job["N"], job["seed"], job["n_steps"], saves)
    else:
        res = C.run_parallel(spec, job["method"], job["N"], job["seed"], job["n_steps"], saves, job["threads"],
                             job["noise"], module=spec["engine_module"])
    t_ret = time.time()
    kind = spec["kind"]
    science = {k: C.sha256_array(res[k]) for k in C.SCORE_KEYS[kind] + ("save_step",)}
    t0 = float(res.get("t0_epoch", t_call))
    meta = C.parse_meta(res.get("meta"))
    rec = dict(
        status="ran", version=VERSION, job=job, signature=sig, science_sha256=science, pid=os.getpid(),
        system=spec["system"], label=spec["label"], N=int(job["N"]), method=job["method"], seed=int(job["seed"]),
        n_steps=int(job["n_steps"]), T_N=float(job["n_steps"] * spec["h"]), B=int(spec["B"]), h=spec["h"],
        T_K=spec["T_K"], backend=backend_label(job["impl"], job["threads"]), impl=job["impl"],
        threads=int(res.get("threads", job["threads"])), noise_mode=str(res.get("noise_mode")),
        engine=str(res.get("engine")), precision="float64", engine_cfg=spec["engine_cfg"],
        wall_total_s=float(res["wall_total_s"]), warmup_s=float(res.get("warmup_s", 0.0)),
        timers={k: float(res.get(k, math.nan)) for k in C.TIMER_KEYS},
        t0_epoch=t0, t0_epoch_source=res.get("t0_epoch_source"), proc_start_epoch=proc_start,
        import_s=import_s, hold_start_s=hold_s, call_epoch=t_call, return_epoch=t_ret,
        start_overhead_s=(t0 - proc_start - hold_s) if math.isfinite(proc_start) else math.nan,
        n_force_evals=int(np.asarray(res["n_force_evals"])), n_saves=int(len(saves)),
        peak_rss_mb=float(res.get("peak_rss_mb", math.nan)), meta=meta, host=host,
        cpus_after_run=C.host_info()["cpus"], git_commit=meta.get("git_commit") or C.git_commit()[0],
        finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    fr_key = "fr_deaths_cum" if kind == "gateway" else "cum_deaths"
    rec["fr_deaths_total"] = int(np.asarray(res[fr_key])[-1]) if fr_key in res else None
    if job["kind"] == "bench":
        rec["bench"] = bench_metrics(res, spec, job)
        rec["wall_s_at_save"] = np.asarray(res["wall_s_at_save"], float).tolist()
        rec["save_step"] = np.asarray(res["save_step"], np.int64).tolist()
    arrays = job["kind"] != "bench"
    if arrays and job["replicate"] > 0:
        first = job_paths(dict(job, replicate=0), out_root)
        try:
            f = C.read_json(first["json"])
            rec["science_identical_to_first"] = bool(f.get("science_sha256") == science)
        except (OSError, ValueError):
            rec["science_identical_to_first"] = None
        arrays = not rec["science_identical_to_first"]
    rec["process_age_s"] = A.process_age_s()
    if arrays:
        keep = {k: v for k, v in res.items() if k not in ("t0_epoch", "t0_epoch_source", "call_epoch",
                                                           "return_epoch", "implementation", "meta")}
        rec = C.save_run(paths["npz"], keep, rec)
    elif job["kind"] != "bench":
        rec = C.save_run(paths["npz"], {k: res[k] for k in ("wall_s_at_save", "n_force_evals_at_save",
                                                            "save_step")}, rec)
    else:
        C.write_json(paths["json"], rec)
    if hold:
        holds["end"] = A.hold_for_monitor(paths["tel_jsonl"], after=time.time(), timeout=hold_timeout)
        rec["holds"] = holds
        C.write_json(paths["json"], rec)
    return rec


# ================================================================================================ scoring / endpoints
class Scorer(C.Scorer):
    """expB_common.Scorer (the frozen equal-budget e_F scorer) on this study's spec."""


def load_science(job, out_root=OUT_ROOT):
    """A run's arrays: replicate 0's science + this replicate's timing arrays."""
    paths = job_paths(job, out_root)
    res = C.load_run(paths["npz"])
    if job["replicate"] > 0 and "M_all" not in res:
        first = C.load_run(job_paths(dict(job, replicate=0), out_root)["npz"])
        first.update({k: res[k] for k in ("wall_s_at_save", "n_force_evals_at_save")})
        res = first
    return res


def persistent_index(err, ok_mask=None, thr=None):
    """First index i (into err) with err[j] <= thr at every later j of the mask (budget grid); -1 if censored."""
    import eqb_metrics as M
    err = np.asarray(err, dtype=float)
    idx = np.arange(err.size) if ok_mask is None else np.flatnonzero(np.asarray(ok_mask, bool))
    with np.errstate(invalid="ignore"):
        i = M.persistent_first(err[idx] <= float(thr))
    return -1 if i is None else int(idx[i])


def run_endpoints(res, spec, scorer, job, rec=None, err=None):
    """Per-run endpoints (prereg section 6) on the budget grid."""
    N, n_steps = int(job["N"]), int(job["n_steps"])
    steps, bmask, umask = save_grid(spec, N)
    step = np.asarray(res["save_step"], dtype=np.int64)
    if not np.array_equal(step, steps):
        raise HSTError(f"{job_key(job)}: save steps differ from the frozen grid")
    wall = np.asarray(res["wall_s_at_save"], dtype=float)
    e = scorer.e_F(res) if err is None else np.asarray(err, dtype=float)
    nfe = np.asarray(res["n_force_evals_at_save"], dtype=np.int64)
    so = float((rec or {}).get("start_overhead_s", math.nan))
    out = dict(N=N, n_steps=n_steps, T_N=n_steps * spec["h"], threads=int(job["threads"]),
               backend=backend_label(job["impl"], job["threads"]), e_flat=scorer.e_flat,
               completion_wall_s=float(res["wall_total_s"]), final_e_F=float(e[-1]),
               Ibar_F=float(np.mean(e[umask])), start_overhead_s=so, n_saves=int(step.size))
    for name in THR_NAMES:
        thr = float(spec["thresholds"][name])
        i = persistent_index(e, bmask, thr)
        W = math.inf if i < 0 else float(wall[i])
        out[f"W_{name}"] = W
        out[f"W_{name}_e2e"] = W + so if (math.isfinite(W) and math.isfinite(so)) else (
            math.inf if not math.isfinite(W) else math.nan)
        out[f"idx_{name}"] = int(i)
        out[f"censored_{name}"] = bool(i < 0)
        out[f"step_{name}"] = int(step[i]) if i >= 0 else -1
        out[f"t_{name}"] = float(step[i] * spec["h"]) if i >= 0 else math.inf
        out[f"u_{name}"] = float(step[i] / n_steps) if i >= 0 else math.inf
        out[f"force_evals_{name}"] = float(nfe[i]) if i >= 0 else math.inf
        out[f"walker_steps_{name}"] = float(N * step[i]) if i >= 0 else math.inf
        out[f"used_core_h_{name}"] = float(job["threads"] * W / 3600.0) if i >= 0 else math.inf
        nccd = 1 if job["threads"] <= CCD_CORES else -(-int(job["threads"]) // CCD_CORES)
        out[f"alloc_core_h_{name}"] = float(nccd * CCD_CORES * W / 3600.0) if i >= 0 else math.inf
    return out, e


# ================================================================================================ ledger
def ledger_append(stage, item, threads, wall_s, note="", path=None):
    C.ledger_append(C.ledger_row(f"hst-{stage}", item, threads, wall_s, note=note), path or LEDGER)


def ledger_used_core_h(path=None, stage_prefix="hst-"):
    return C.ledger_used_core_h(path or LEDGER, stage_prefix=stage_prefix)
