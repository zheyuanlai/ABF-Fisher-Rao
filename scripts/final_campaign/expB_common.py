"""Experiment B common layer: systems, save grid, engine adapters, scoring and per-run endpoints.

docs/final_campaign/PREREGISTRATION.md section 2 (frozen) and docs/final_campaign/EXPB_BACKEND_DESIGN.md.  No engine
code lives here: the walker-parallel engines are reached ONLY through the FROZEN ENGINE INTERFACE

    run_arm_parallel(cfg, method, N, seed, n_steps, save_steps, *, threads, noise, timers=True) -> dict

(src/parallel/gateway_parallel.py, src/parallel/lta_parallel.py; ``parallel_engine`` imports them lazily, a test may
pass any module that implements the interface), and the validated serial engines (src/gateway_ladder_numba.py,
src/lta_ladder_numba.py, never modified) are timed exactly like scripts/mechanism/parallel_benchmark.py times them
(its SaveClock / gateway_numba_clock / lta_numba_clock hooks, imported).

Definitions (prereg section 2.5-2.6)
  save grid      u = k/200 (k = 1..200) UNION u_j = 1e-4 * 1.02^j (j = 0..465, the largest j with u_j <= 1);
                 steps = unique(clip(round(u n_steps), 1, n_steps)) -- identical in u for every arm and N.
  e_F            per save, the frozen equal-budget scorer (scripts/equal_budget/eqb_metrics.make_scorer):
                 gateway = the accepted analytic scorer on (M_all, C_all); LTA 150 K = eqb_metrics.lta_estimator
                 (production accumulators M_prod, C_prod once they hold a count, else all-steps) + lta_errors_from.
  e_flat         the error of the zero-bias flat profile (M = C = 0): the error before the first save.
  persistent     the first save i with e_F <= thr at save i and at EVERY later save of the run (NaN never meets thr);
                 none -> censored (index -1, W = inf; nothing is imputed).
  W              wall_s_at_save[i] (seconds since the engine's timed start).
  W_interp       log-linear interpolation in e_F between the last failing save (i - 1) and the first persistent save
                 i: w = w_{i-1} + (w_i - w_{i-1}) (ln e_{i-1} - ln thr) / (ln e_{i-1} - ln e_i).  For i = 0 the last
                 failing "save" is the timed start (w = 0, e = e_flat).  Censored -> inf.
  Ibar_F         mean of e_F over the 200 uniform saves u = k/200 (eqb_metrics.uniform_indices).
  Ibar_F^wall    (1/W*) int_0^W* e_F(w) dw, e_F(w) = the error of the latest save at or before w; e_flat before the
                 first save; after the last save (u = 1, the run's completion) its final error.
  completion     wall_total_s (start of the timed run to its end).
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import math
import os
import platform
import shutil
import socket
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in (os.path.join(ROOT, "src"), os.path.join(ROOT, "src", "parallel"), os.path.join(ROOT, "scripts"),
           os.path.join(ROOT, "scripts", "equal_budget"), os.path.join(ROOT, "scripts", "mechanism"), HERE):
    if _p not in sys.path:
        sys.path.append(_p)

COMMON_VERSION = "expB_common/1"
EXPB_CONFIG = os.path.join(ROOT, "configs", "final_campaign", "expB.json")
EQB_CONFIG_DIR = os.path.join(ROOT, "configs", "equal_budget_v2")
EQB_RESULTS = os.path.join(ROOT, "results", "equal_budget_v2")
OUT_ROOT = os.path.join(ROOT, "results", "final_campaign", "expB")
FIG_ROOT = os.path.join(ROOT, "figures", "final_campaign", "expB")
LEDGER = os.path.join(ROOT, "results", "final_campaign", "compute_ledger.csv")
BUILD_DIR = os.path.expanduser("~/.cache/final_campaign_build")

THR_NAMES = ("strict", "mid", "loose")
N_UNIFORM = 200
LOG_U_MIN = 1e-4
LOG_RATIO = 1.02
K_ALLOC = 16                     # allocated cores per arm (prereg 2.4: two CCDs of one NUMA node)
NB = 180
BOOT_SEED = 20261010
N_BOOT = 10_000
ORDER_SEED_BASE = 20261011       # within-seed arm order: default_rng(20261011 + seed)
SYSTEMS = {
    "gateway": dict(kind="gateway", eqb_system="gateway", eqb_config="gateway_production.json", eqb_dir="gateway",
                    T_K=None, engine_module="gateway_parallel", engine_label="gateway_parallel/1",
                    validated_module="gateway_ladder_numba"),
    "lta150": dict(kind="lta", eqb_system="lta150", eqb_config="lta_150K_production.json", eqb_dir="lta_T150",
                   T_K=150.0, engine_module="lta_parallel", engine_label="lta_parallel/1",
                   validated_module="lta_ladder_numba"),
}
# keys of the validated engines that are wall / provenance only (never compared, never required of the parallel
# engine's science part)
WALL_PROVENANCE_KEYS = ("wall_s", "peak_rss_mb", "n_checkpoints_written", "resumed_from", "sessions_json", "meta",
                        "meta_json", "cfg", "cfg_json")
TIMER_KEYS = ("t_force_move_s", "t_abf_s", "t_sync_s", "t_fr_s", "t_save_s")
CONTRACT_EXTRA = ("wall_s_at_save", "n_force_evals_at_save", "wall_total_s", "warmup_s") + TIMER_KEYS + (
    "threads", "noise_mode", "engine", "meta")
# science keys the harness itself reads (a subset of the validated engines' diagnostics-off result)
SCIENCE_KEYS = {
    "gateway": ("save_step", "save_t", "save_u", "M_all", "C_all", "fr_deaths_cum", "fr_opp_cum", "n_force_evals",
                "X_final", "Y_final", "max_abs_bias"),
    "lta": ("save_step", "save_t", "save_u", "M_all", "C_all", "M_prod", "C_prod", "cum_deaths", "cum_opps",
            "n_force_evals", "q_final"),
}
SCORE_KEYS = {"gateway": ("M_all", "C_all"), "lta": ("M_all", "C_all", "M_prod", "C_prod")}


class ExpBError(RuntimeError):
    """A violated contract, config mismatch or plan rule."""


# ================================================================================================ config / systems
def load_json(path):
    with open(path) as fh:
        return json.load(fh)


def load_expb_config(path=None):
    return load_json(path or EXPB_CONFIG)


def system_kind(system):
    if system not in SYSTEMS:
        raise ExpBError(f"unknown system {system!r}; expected one of {sorted(SYSTEMS)}")
    return SYSTEMS[system]["kind"]


def system_spec(system, expb=None, eqb_config=None, overrides=None):
    """Everything the harness needs about one system, from configs/final_campaign/expB.json and the frozen
    equal-budget production config (engine cfg, h, B, thresholds, reference).  Cross-checks h, B and the e_F
    thresholds of the two configs (ExpBError on any mismatch).  ``overrides`` (tests only) replaces top-level spec
    fields after the checks, e.g. a tiny B."""
    kind = system_kind(system)
    S = SYSTEMS[system]
    X = expb if expb is not None else load_expb_config()
    sx = X["systems"][system]
    P = eqb_config if eqb_config is not None else load_json(os.path.join(EQB_CONFIG_DIR, S["eqb_config"]))
    ecfg = dict(P["engine_cfg"])
    thr = {n: float(v) for n, v in sx["thresholds_e_F"].items()}
    probs = []
    if float(sx["h"]) != float(ecfg["h"]):
        probs.append(f"h: expB {sx['h']} vs equal-budget {ecfg['h']}")
    if int(sx["B"]) != int(P["B"]):
        probs.append(f"B: expB {sx['B']} vs equal-budget {P['B']}")
    if [thr[n] for n in THR_NAMES] != [float(v) for v in P["thresholds"]["e_F"]]:
        probs.append(f"e_F thresholds: expB {thr} vs equal-budget {P['thresholds']['e_F']}")
    if probs:
        raise ExpBError(f"{system}: expB.json and the equal-budget config disagree: " + "; ".join(probs))
    spec = dict(
        system=system, kind=kind, eqb_system=S["eqb_system"], T_K=S["T_K"], P=P, engine_cfg=ecfg,
        h=float(ecfg["h"]), B=int(P["B"]), thresholds=thr, seeds=[int(s) for s in sx["seeds"]],
        core_arms=[(str(m), int(n)) for m, n in sx["core_arms"]],
        bench_seeds=[int(s) for s in sx["bench_seeds"]],
        equivalence_seeds={k: [int(s) for s in v] for k, v in sx["equivalence_seeds"].items()},
        bench_walker_steps=int(X["feasibility"]["benchmark_walker_steps"][system]),
        bench_repeats=int(X["feasibility"]["repeats"]),
        bench_abf_N=[int(n) for n in X["feasibility"]["candidates"]["abf_N"]],
        bench_fr_N=[int(n) for n in X["feasibility"]["candidates"]["fr_N"]],
        thread_counts=[int(t) for t in X["hardware_allocation"]["thread_counts"]],
        descriptive_threads=[int(t) for t in X["hardware_allocation"]["descriptive_extra"]],
        k_alloc=int(X["hardware_allocation"]["k_physical_cores"]),
        eqb_summary=os.path.join(EQB_RESULTS, S["eqb_dir"], "analysis", "summary.json"),
        eqb_dir=os.path.join(EQB_RESULTS, S["eqb_dir"]),
        engine_module=S["engine_module"], engine_label=S["engine_label"], validated_module=S["validated_module"],
    )
    if overrides:
        spec.update(overrides)
    return spec


def arm_name(method, N):
    if method not in ("abf", "fr"):
        raise ExpBError(f"method must be 'abf' or 'fr', not {method!r}")
    return f"{method}{int(N)}"


def parse_arm(name):
    for m in ("abf", "fr"):
        if name.startswith(m) and name[len(m):].isdigit():
            return m, int(name[len(m):])
    raise ExpBError(f"not an arm name: {name!r}")


def n_steps_for(B, N):
    B, N = int(B), int(N)
    if N < 1 or B % N:
        raise ExpBError(f"budget {B} is not divisible by N {N}")
    return B // N


# ================================================================================================ save grid
def expb_u_grid():
    """The Experiment B budget fractions: 200 uniform u = k/200 UNION u_j = 1e-4 * 1.02^j, j = 0..J with J the
    largest j such that u_j <= 1 (J = 465: 466 log points)."""
    u_lin = np.arange(1, N_UNIFORM + 1, dtype=np.float64) / N_UNIFORM
    J = int(math.floor(math.log(1.0 / LOG_U_MIN) / math.log(LOG_RATIO) + 1e-9))
    u_log = LOG_U_MIN * LOG_RATIO ** np.arange(J + 1, dtype=np.float64)
    u_log = u_log[u_log <= 1.0]
    return np.union1d(u_lin, u_log), u_lin, u_log


def expb_save_grid(n_steps):
    """(steps int64, u = steps / n_steps): unique(clip(round(u n_steps), 1, n_steps)) over expb_u_grid()."""
    n_steps = int(n_steps)
    if n_steps < 1:
        raise ExpBError("n_steps must be >= 1")
    u, _, _ = expb_u_grid()
    steps = np.unique(np.clip(np.round(u * n_steps).astype(np.int64), 1, n_steps))
    return steps, steps / float(n_steps)


def uniform_save_indices(save_step, n_steps):
    """Indices of the 200 uniform saves (eqb_metrics.uniform_indices: exact integer steps when 200 | n_steps)."""
    import eqb_metrics as M
    idx, _exact = M.uniform_indices(np.asarray(save_step, dtype=np.int64), int(n_steps))
    return idx


def eqb_production_save_grid(spec, n_steps):
    """The equal-budget production grid of the system (budget grid + physical checkpoints <= T, the rule of
    scripts/equal_budget/run_ladder.save_grid) -- used for the V-B4 equivalence runs, identical to the reference."""
    import eqb_metrics as M
    E = M.engine_module(spec["kind"])
    steps, _ = E.budget_save_grid(int(n_steps))
    h = float(spec["engine_cfg"]["h"])
    extra = [int(round(t / h)) for t in spec["P"].get("physical_checkpoints_t", []) if 0 < round(t / h) <= n_steps]
    return np.unique(np.concatenate([steps, np.asarray(extra, dtype=np.int64)]))


def sha256_array(a):
    a = np.ascontiguousarray(np.asarray(a))
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ================================================================================================ provenance
def git_commit():
    try:
        out = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=20)
        commit = out.stdout.strip() or "unknown"
        st = subprocess.run(["git", "-C", ROOT, "status", "--porcelain", "--", "scripts/final_campaign", "src"],
                            capture_output=True, text=True, timeout=20)
        return commit, bool(st.stdout.strip())
    except Exception:  # noqa: BLE001
        return "unknown", None


def module_source_files(name):
    """Source files of an importable module (found without importing it); [] when not found."""
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return []
    if spec is None or not spec.origin or not os.path.exists(spec.origin):
        return []
    return [spec.origin]


BACKEND_FILES = {"gateway_parallel": ("gateway_parallel.py", "gw_kernel.c", "philox.py", "__init__.py"),
                 "lta_parallel": ("lta_parallel.py", "philox.py", "__init__.py")}


def engine_sources(spec, impl, engine_module=None):
    """Source files that define a run of ``impl`` ('parallel' | 'serial' | 'validated') of the system: the backend's
    own files (BACKEND_FILES) + the validated engine it reuses; for the serial engine the validated engine + the
    parallel_benchmark timing hooks.  A test engine module (not in BACKEND_FILES) contributes its own file."""
    validated = module_source_files(spec["validated_module"])
    if impl in ("serial", "validated"):
        pb = module_source_files("parallel_benchmark") if impl == "serial" else []
        return validated + pb
    name = engine_module if isinstance(engine_module, str) and engine_module else spec["engine_module"]
    base = name.split(".")[-1]
    if base in BACKEND_FILES:
        d = os.path.join(ROOT, "src", "parallel")
        return [os.path.join(d, f) for f in BACKEND_FILES[base]] + validated
    return module_source_files(name)[:1] + validated


def files_sha(files):
    return {os.path.relpath(p, ROOT) if p.startswith(ROOT) else p: sha256_file(p) for p in files if os.path.exists(p)}


def engine_sources_sha(spec, impl, engine_module=None):
    """sha256 of every source file that defines a run of ``impl`` (engine_sources), keyed by repo-relative path."""
    return files_sha(engine_sources(spec, impl, engine_module))


def install_exit_signals():
    """Driver processes: SIGTERM / SIGHUP raise SystemExit so that ``finally`` blocks run (they terminate this
    driver's own run processes, which live in their own sessions, and log them to the ledger)."""
    import signal

    def _exit(signum, _frame):
        raise SystemExit(128 + int(signum))
    for sig in (signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(sig, _exit)
        except (OSError, ValueError):
            pass


def host_info():
    try:
        cpus = sorted(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover
        cpus = []
    return dict(host=socket.gethostname(), platform=platform.platform(), python=platform.python_version(),
                numpy=np.__version__, cpus=cpus)


def limit_driver_threads():
    """Driver processes (benchmark / scheduler / analysis) use one thread for numpy / torch / BLAS: they sit on one
    core outside every allocation and must not spawn thread pools.  Call before torch is imported."""
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


# ================================================================================================ engines
def parallel_engine(system, module=None):
    """The walker-parallel engine module of ``system`` (lazy import).  ``module`` = an import path or a module object
    (tests: a fake engine implementing run_arm_parallel)."""
    if module is not None and not isinstance(module, str):
        return module
    name = module or os.environ.get(f"EXPB_ENGINE_MODULE_{system.upper()}") or SYSTEMS[system]["engine_module"]
    for cand in ((name,) if "." in name else (f"parallel.{name}", name)):
        try:
            mod = importlib.import_module(cand)
        except ModuleNotFoundError as e:   # only "this module does not exist" falls through; real errors raise
            parts = cand.split(".")
            if e.name not in {".".join(parts[:i + 1]) for i in range(len(parts))}:
                raise
            continue
        if not hasattr(mod, "run_arm_parallel"):
            raise ExpBError(f"engine module {cand} has no run_arm_parallel")
        return mod
    raise ExpBError(f"parallel engine module {name!r} for {system} not importable (src/parallel/ not built yet?)")


def engine_cfg_for(spec, N, seed, n_steps):
    """The cfg argument of run_arm_parallel: gateway = the equal-budget engine cfg; LTA = lta_ladder_numba.make_cfg(
    150.0, N, seed, **engine_cfg) with diagnostics=False and n_steps set (equal to B // N at full budget)."""
    if spec["kind"] == "gateway":
        return dict(spec["engine_cfg"])
    import lta_ladder_numba as L
    ecfg = dict(spec["engine_cfg"], diagnostics=False, n_steps=int(n_steps))
    return L.make_cfg(float(spec["T_K"]), int(N), int(seed), **ecfg)


def contract_problems(res, kind, N, n_steps, save_steps, require_timers=True):
    """List of violations of the frozen engine interface (empty = OK)."""
    probs = []
    need = list(SCIENCE_KEYS[kind]) + ["wall_s_at_save", "n_force_evals_at_save", "wall_total_s", "warmup_s",
                                       "threads", "noise_mode", "engine", "meta"]
    if require_timers:
        need += list(TIMER_KEYS)
    miss = [k for k in need if k not in res]
    if miss:
        return [f"missing keys {miss}"]
    k = np.asarray(res["save_step"], dtype=np.int64)
    S = k.size
    if not np.array_equal(k, np.asarray(save_steps, dtype=np.int64)):
        probs.append("save_step differs from the requested save grid")
    w = np.asarray(res["wall_s_at_save"], dtype=float)
    if w.shape != (S,) or not np.all(np.isfinite(w)) or np.any(w < 0) or np.any(np.diff(w) < 0):
        probs.append("wall_s_at_save must be finite, >= 0 and non-decreasing, one per save")
    nfe = np.asarray(res["n_force_evals_at_save"], dtype=np.int64)
    want = N * k if kind == "gateway" else N * (k + 1)
    if nfe.shape != (S,) or not np.array_equal(nfe, want):
        probs.append("n_force_evals_at_save != N k (gateway) / N (k + 1) (LTA)")
    tot = int(np.asarray(res["n_force_evals"]))
    if tot != (N * n_steps if kind == "gateway" else N * (n_steps + 1)):
        probs.append(f"n_force_evals {tot} inconsistent with N {N}, n_steps {n_steps}")
    for key in SCORE_KEYS[kind]:
        a = np.asarray(res[key])
        if a.shape != (S, NB):
            probs.append(f"{key} has shape {a.shape}, expected ({S}, {NB})")
    wt = float(res["wall_total_s"])
    if not (math.isfinite(wt) and wt >= (w[-1] if w.size and np.all(np.isfinite(w)) else 0.0) - 1e-6):
        probs.append("wall_total_s must be finite and >= the last save's wall stamp")
    if res["noise_mode"] not in ("counter", "replay"):
        probs.append(f"noise_mode {res['noise_mode']!r}")
    return probs


def check_contract(res, kind, N, n_steps, save_steps, require_timers=True):
    p = contract_problems(res, kind, N, n_steps, save_steps, require_timers)
    if p:
        raise ExpBError("engine result violates the frozen interface: " + "; ".join(p))
    return res


def run_parallel(spec, method, N, seed, n_steps, save_steps, threads, noise="counter", module=None):
    """One run on the walker-parallel engine through the frozen interface; returns the checked result dict with
    t0_epoch (wall-clock epoch of the engine's timed start: meta['t0_epoch'] if the engine records it, else
    return time - wall_total_s) and peak_rss_mb of this process."""
    E = parallel_engine(spec["system"], module)
    cfg = engine_cfg_for(spec, N, seed, n_steps)
    _reset_peak_rss()
    e0 = time.time()
    res = E.run_arm_parallel(cfg, method, int(N), int(seed), int(n_steps), np.asarray(save_steps, dtype=np.int64),
                             threads=int(threads), noise=noise, timers=True)
    e1 = time.time()
    check_contract(res, spec["kind"], int(N), int(n_steps), save_steps)
    meta = parse_meta(res.get("meta"))
    res = dict(res)
    res["t0_epoch"] = float(meta.get("t0_epoch", e1 - float(res["wall_total_s"])))
    res["t0_epoch_source"] = "engine meta" if "t0_epoch" in meta else "return epoch - wall_total_s"
    res["call_epoch"], res["return_epoch"] = e0, e1
    res["peak_rss_mb"] = _peak_rss_mb()
    res["implementation"] = "parallel"
    return res


def _proc_status_mb(field):
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith(field + ":"):
                    return int(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        pass
    return float("nan")


def _reset_peak_rss():
    try:
        with open("/proc/self/clear_refs", "w") as fh:
            fh.write("5")
        return True
    except OSError:
        return False


def _peak_rss_mb():
    return _proc_status_mb("VmHWM")


def run_serial(spec, method, N, seed, n_steps, save_steps, warm=True):
    """One run on the VALIDATED serial engine (diagnostics off, its replay noise), timed like
    scripts/mechanism/parallel_benchmark.py: a warm-up run of the same code path first (gateway 2 FR intervals; LTA
    60 steps with the FR start moved to step 10), then the timed run with the engine's chunk kernel split at the save
    steps and stamped (PB.SaveClock + PB.gateway_numba_clock / PB.lta_numba_clock).  Returns a dict in the
    parallel engine's contract (timers NaN: no clock inside the nopython kernel)."""
    import parallel_benchmark as PB          # sets OMP/NUMBA thread env to 1 (this is a 1-core run)
    kind = spec["kind"]
    save_steps = np.asarray(save_steps, dtype=np.int64)
    out = {}
    commit, dirty = git_commit()             # provenance BEFORE the timed region (git runs as a child process on
    #                                          this run's core: after the run it would land in the run's wall window)
    if kind == "gateway":
        import gateway_ladder_numba as E
        cfg = dict(spec["engine_cfg"])
        if warm:
            kn = E.derived_knobs(E.full_cfg(cfg), N)
            nw = 2 * int(kn["fr_every"])
            wclock = PB.SaveClock(np.array([nw // 2, nw]))
            t = time.perf_counter()
            with PB.gateway_numba_clock(E, wclock):
                wclock.start()
                E.run_arm(cfg, method, N, seed, nw, np.array([nw // 2, nw]), diagnostics=False)
            out["warmup_s"] = time.perf_counter() - t
        clock = PB.SaveClock(save_steps)
        with PB.gateway_numba_clock(E, clock):
            _reset_peak_rss()
            t0_epoch = time.time()
            clock.start()
            res = E.run_arm(cfg, method, N, seed, n_steps, save_steps, diagnostics=False)
            wall_total = time.perf_counter() - clock.t0
        emeta = res["meta"]
        cfg_out = res["cfg"]
        nfe_at = N * save_steps
        engine_files = [E.__file__]
    else:
        import lta_ladder_numba as E
        ecfg = dict(spec["engine_cfg"], diagnostics=False)
        if warm:
            ws_n = PB.TORCH_WARM["lta_steps"]
            frs = PB.TORCH_WARM["lta_fr_start"]
            cw = E.make_cfg(float(spec["T_K"]), N, seed, **dict(ecfg, n_steps=ws_n, fr_start_steps=frs,
                                                                 warmup_steps=frs, burn_in_steps=frs))
            ws = np.array([ws_n // 2, ws_n])
            wclock = PB.SaveClock(ws)
            t = time.perf_counter()
            with PB.lta_numba_clock(E, wclock):
                wclock.start()
                E.run_arm(cw, method, ws)
            out["warmup_s"] = time.perf_counter() - t
        c = E.make_cfg(float(spec["T_K"]), N, seed, **dict(ecfg, n_steps=int(n_steps)))
        clock = PB.SaveClock(save_steps)
        with PB.lta_numba_clock(E, clock):
            _reset_peak_rss()
            t0_epoch = time.time()
            clock.start()
            res = E.run_arm(c, method, save_steps)
            wall_total = time.perf_counter() - clock.t0
        emeta = json.loads(str(res["meta_json"]))
        cfg_out = json.loads(str(res["cfg_json"]))
        nfe_at = N * (save_steps + 1)
        engine_files = [E.__file__]
    peak = _peak_rss_mb()
    for k, v in res.items():
        if k in WALL_PROVENANCE_KEYS:
            continue
        out[k] = v
    meta = dict(engine=emeta.get("engine"), engine_sha256=emeta.get("engine_sha256"), implementation="serial",
                noise="replay (the validated engine's own stream)", diagnostics=False,
                sources=files_sha(engine_files + [PB.__file__, os.path.abspath(__file__)]),
                git_commit=commit, git_dirty=dirty, numba=_numba_version(),
                compiler="numba (validated engine)", timing="parallel_benchmark SaveClock hooks", **host_info(),
                t0_epoch=t0_epoch, engine_meta_digest=hashlib.sha256(
                    json.dumps(emeta, sort_keys=True, default=str).encode()).hexdigest())
    out.update(wall_s_at_save=np.asarray(clock.wall, dtype=float), n_force_evals_at_save=nfe_at.astype(np.int64),
               wall_total_s=float(wall_total), warmup_s=float(out.get("warmup_s", 0.0)),
               threads=1, noise_mode="replay", engine=f"{emeta.get('engine')} (validated serial)",
               cfg_json=json.dumps(cfg_out, sort_keys=True, default=str), meta=json.dumps(meta, sort_keys=True,
                                                                                         default=str))
    for k in TIMER_KEYS:
        out[k] = float("nan")
    out["t0_epoch"] = t0_epoch
    out["t0_epoch_source"] = "serial wrapper clock start"
    out["peak_rss_mb"] = peak
    out["implementation"] = "serial"
    check_contract(out, kind, int(N), int(n_steps), save_steps, require_timers=True)
    return out


def _numba_version():
    try:
        import numba
        return numba.__version__
    except Exception:  # noqa: BLE001
        return None


def parse_meta(meta):
    if meta is None:
        return {}
    if isinstance(meta, dict):
        return meta
    if isinstance(meta, np.ndarray):
        meta = meta.item() if meta.ndim == 0 else str(meta)
    try:
        return json.loads(str(meta))
    except (TypeError, ValueError):
        return {"raw": str(meta)}


# ================================================================================================ run files
def _to_array(v):
    if isinstance(v, (dict, list, tuple)) and not (isinstance(v, (list, tuple)) and v and
                                                     all(isinstance(x, (int, float, np.number)) for x in v)):
        return np.array(json.dumps(v, sort_keys=True, default=str)), True
    if isinstance(v, str):
        return np.array(v), False
    a = np.asarray(v)
    if a.dtype == object:
        return np.array(json.dumps(v, default=str)), True
    return a, False


def save_run(path, res, record, arrays=True, keep_keys=None):
    """Atomic compressed npz (no pickle) of the result arrays + a JSON sidecar ``record`` (path with .json).
    arrays=False writes only the JSON (a repeat whose science is bitwise the first replicate's)."""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    if arrays:
        out, json_keys = {}, []
        for k, v in res.items():
            if keep_keys is not None and k not in keep_keys:
                continue
            a, is_json = _to_array(v)
            out[k] = a
            if is_json:
                json_keys.append(k)
        out["__json_keys__"] = np.array(json.dumps(sorted(json_keys)))
        tmp = os.path.join(d, f".{os.path.basename(path)}.tmp{os.getpid()}")
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, **out)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        record = dict(record, npz=os.path.basename(path), npz_sha256=sha256_file(path))
    write_json(json_path(path), record)
    return record


def json_safe(obj):
    """Strict-JSON copy (eqb_metrics.json_safe: inf -> "inf", NaN -> null, numpy -> python)."""
    import eqb_metrics as M
    return M.json_safe(obj)


def engine_available(system, module=None):
    """True iff the parallel engine module of ``system`` can be found (without importing it)."""
    name = module or os.environ.get(f"EXPB_ENGINE_MODULE_{system.upper()}") or SYSTEMS[system]["engine_module"]
    for cand in ((name,) if "." in name else (f"parallel.{name}", name)):
        try:
            if importlib.util.find_spec(cand) is not None:
                return True
        except (ImportError, ValueError):
            continue
    return False


def json_path(npz_path):
    return npz_path[:-4] + ".json" if npz_path.endswith(".npz") else npz_path + ".json"


def write_json(path, obj):
    import eqb_metrics as M
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, f".{os.path.basename(path)}.tmp{os.getpid()}")
    with open(tmp, "w") as fh:
        json.dump(M.json_safe(obj), fh, indent=1, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def read_json(path):
    with open(path) as fh:
        return json.load(fh)


def load_run(path):
    out = {}
    with np.load(path, allow_pickle=False) as z:
        jk = set(json.loads(str(z["__json_keys__"]))) if "__json_keys__" in z.files else set()
        for k in z.files:
            if k == "__json_keys__":
                continue
            v = z[k]
            if k in jk:
                out[k] = json.loads(str(v))
            elif v.ndim == 0:
                out[k] = v.item()
            else:
                out[k] = v
    return out


def num(v):
    """JSON-safe convention back to float: "inf" -> inf, None -> NaN."""
    if v is None:
        return math.nan
    if isinstance(v, str):
        return float(v)
    return float(v)


# ================================================================================================ scoring
class Scorer:
    """e_F per save from the saved accumulators, with the frozen equal-budget scorer of the system."""

    def __init__(self, spec):
        import eqb_metrics as M
        self.spec = spec
        self.kind = spec["kind"]
        self._M = M
        self.s = M.make_scorer(spec["eqb_system"], spec["P"])
        self.reference_info = self.s.reference_info
        z = np.zeros((1, NB))
        self.e_flat = float(self.e_F_arrays(z, z, z, z)[0])

    def e_F_arrays(self, M_all, C_all, M_prod=None, C_prod=None):
        M_all = np.asarray(M_all, dtype=float)
        C_all = np.asarray(C_all, dtype=float)
        if self.kind == "gateway":
            eF, _ = self.s.sa.score(M_all, C_all)
            return np.asarray(eF, dtype=float)
        G, F, _ = self._M.lta_estimator(M_all, C_all, M_prod, C_prod)
        return np.asarray(self._M.lta_errors_from(G, F, self.s.F_ref, self.s.gamma_ref)[0], dtype=float)

    def e_F(self, res):
        if self.kind == "gateway":
            return self.e_F_arrays(res["M_all"], res["C_all"])
        return self.e_F_arrays(res["M_all"], res["C_all"], res["M_prod"], res["C_prod"])


# ================================================================================================ endpoints
def persistent_index(err, thr):
    """First index i with err[j] <= thr for every j >= i (NaN never meets thr); -1 if censored."""
    import eqb_metrics as M
    err = np.asarray(err, dtype=float)
    with np.errstate(invalid="ignore"):
        i = M.persistent_first(err <= float(thr))
    return -1 if i is None else int(i)


def w_at(wall, idx):
    return math.inf if idx < 0 else float(np.asarray(wall, dtype=float)[idx])


def w_interp(err, wall, thr, e_flat):
    """Log-linear (in e_F) interpolation of the crossing between the last failing save and the first persistent
    save; the timed start (w = 0, e = e_flat) is the 'save' before the first one.  inf if censored."""
    err = np.asarray(err, dtype=float)
    wall = np.asarray(wall, dtype=float)
    i = persistent_index(err, thr)
    if i < 0:
        return math.inf
    w1, e1 = float(wall[i]), float(err[i])
    if i == 0:
        w0, e0 = 0.0, float(e_flat)
    else:
        w0, e0 = float(wall[i - 1]), float(err[i - 1])
    thr = float(thr)
    if not (math.isfinite(e0) and e0 > thr):          # NaN / non-failing predecessor: no interpolation possible
        return w1 if not (math.isfinite(e0) and e0 <= thr) else w0
    if not (e1 > 0.0):
        return w0
    lt, l0, l1 = math.log(thr), math.log(e0), math.log(e1)
    if l0 == l1:
        return w1
    frac = (l0 - lt) / (l0 - l1)
    frac = min(1.0, max(0.0, frac))
    return w0 + (w1 - w0) * frac


def ibar_uniform(err, save_step, n_steps):
    idx = uniform_save_indices(save_step, n_steps)
    return float(np.mean(np.asarray(err, dtype=float)[idx]))


def ibar_wall(err, wall, W_star, e_flat):
    """(1/W*) int_0^W* e_F(w) dw for the step function e_F(w) = err of the latest save with wall <= w (e_flat before
    the first save; the final error after the last save)."""
    W_star = float(W_star)
    if not (math.isfinite(W_star) and W_star > 0):
        raise ExpBError(f"W* must be finite and > 0 (got {W_star})")
    err = np.asarray(err, dtype=float)
    wall = np.asarray(wall, dtype=float)
    if err.shape != wall.shape or err.size == 0:
        raise ExpBError("err and wall must be non-empty and of equal shape")
    if np.any(~np.isfinite(wall)) or np.any(np.diff(wall) < 0):
        raise ExpBError("wall stamps must be finite and non-decreasing")
    knots = np.concatenate([[0.0], wall])             # value on [knots[j], knots[j+1]) is vals[j]
    vals = np.concatenate([[float(e_flat)], err])
    hi = np.concatenate([wall, [math.inf]])
    lo = np.minimum(knots, W_star)
    up = np.minimum(hi, W_star)
    seg = np.maximum(up - lo, 0.0)
    with np.errstate(invalid="ignore"):
        tot = float(np.sum(np.where(seg > 0, seg * vals, 0.0)))
    return tot / W_star


def e_at_wall(err, wall, w_grid, e_flat):
    """The step function e_F(w) at the points w_grid (for median-curve figures)."""
    err = np.asarray(err, dtype=float)
    wall = np.asarray(wall, dtype=float)
    j = np.searchsorted(wall, np.asarray(w_grid, dtype=float), side="right") - 1
    return np.where(j < 0, float(e_flat), err[np.maximum(j, 0)])


def run_endpoints(res, spec, scorer, N, n_steps, threads, err=None):
    """Per-run endpoints of prereg 2.6 (thresholds of the system): W, W_interp, index, step, force evaluations and
    used / allocated core-hours to accuracy at every threshold; Ibar_F (uniform); final e_F; completion wall."""
    step = np.asarray(res["save_step"], dtype=np.int64)
    if step[-1] != int(n_steps):
        raise ExpBError(f"last save at step {step[-1]}, not at n_steps {n_steps}")
    wall = np.asarray(res["wall_s_at_save"], dtype=float)
    e = scorer.e_F(res) if err is None else np.asarray(err, dtype=float)
    nfe_at = np.asarray(res["n_force_evals_at_save"], dtype=np.int64)
    out = dict(N=int(N), n_steps=int(n_steps), threads=int(threads), e_flat=scorer.e_flat,
               completion_wall_s=float(res["wall_total_s"]), Ibar_F=ibar_uniform(e, step, n_steps),
               final_e_F=float(e[-1]), n_saves=int(step.size))
    for name in THR_NAMES:
        thr = float(spec["thresholds"][name])
        i = persistent_index(e, thr)
        W = w_at(wall, i)
        out[f"W_{name}"] = W
        out[f"W_{name}_interp"] = w_interp(e, wall, thr, scorer.e_flat)
        out[f"idx_{name}"] = i
        out[f"censored_{name}"] = bool(i < 0)
        out[f"step_{name}"] = int(step[i]) if i >= 0 else -1
        out[f"u_{name}"] = float(step[i] / n_steps) if i >= 0 else math.inf
        out[f"force_evals_{name}"] = float(N * step[i]) if i >= 0 else math.inf          # N x completed steps
        out[f"engine_force_evals_{name}"] = float(nfe_at[i]) if i >= 0 else math.inf
        out[f"used_core_h_{name}"] = float(threads * W / 3600.0) if i >= 0 else math.inf
        out[f"alloc_core_h_{name}"] = float(spec.get("k_alloc", K_ALLOC) * W / 3600.0) if i >= 0 else math.inf
    return out, e


# ================================================================================================ ledger / disk
LEDGER_HEADER = ["utc_start", "stage", "item", "threads", "wall_s", "used_core_h", "note"]


def ledger_append(row, path=None):
    """Append one row to the compute ledger (file-locked; header kept)."""
    import csv
    import fcntl
    path = path or LEDGER
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a+", newline="") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            fh.seek(0)
            empty = fh.read(1) == ""
            fh.seek(0, os.SEEK_END)
            w = csv.writer(fh)
            if empty:
                w.writerow(LEDGER_HEADER)
            w.writerow([row.get(k, "") for k in LEDGER_HEADER])
            fh.flush()
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def ledger_row(stage, item, threads, wall_s, note="", utc_start=None):
    return dict(utc_start=utc_start or time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()), stage=stage, item=item,
                threads=int(threads), wall_s=round(float(wall_s), 3),
                used_core_h=round(float(threads) * float(wall_s) / 3600.0, 6), note=note)


def ledger_used_core_h(path=None, stage_prefix=None):
    import csv
    path = path or LEDGER
    if not os.path.exists(path):
        return 0.0
    tot = 0.0
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if stage_prefix and not str(r.get("stage", "")).startswith(stage_prefix):
                continue
            try:
                tot += float(r.get("used_core_h") or 0.0)
            except ValueError:
                pass
    return tot


class DiskPauseTimeout(RuntimeError):
    """Stopping rule (iv): a disk pause that did not clear within the limit."""


class DiskGuard:
    """Pause while free space < pause_below; resume once >= resume_at (prereg section 3: 2 GB / 3 GB).  wait() blocks
    (sleeping poll_s) until running is allowed, logging every transition; raises DiskPauseTimeout after max_pause_s."""

    def __init__(self, path=ROOT, pause_below=2e9, resume_at=3e9, poll_s=60.0, max_pause_s=24 * 3600.0, log=print,
                 sleep=time.sleep, clock=time.monotonic, usage=None):
        self.path, self.pause_below, self.resume_at = path, float(pause_below), float(resume_at)
        self.poll_s, self.max_pause_s, self.log = float(poll_s), float(max_pause_s), log
        self.sleep, self.clock = sleep, clock
        self._usage = usage
        self.paused = False
        self.pause_events = []

    def free(self):
        u = self._usage(self.path) if self._usage is not None else shutil.disk_usage(self.path)
        return float(u.free)

    def ok(self):
        """Non-blocking: update the state from one measurement; True iff running is allowed."""
        f = self.free()
        if not self.paused and f < self.pause_below:
            self.paused = True
            self.pause_events.append(dict(paused_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), free=f))
            self.log(f"DISK PAUSE: {f / 1e9:.2f} GB free < {self.pause_below / 1e9:.1f} GB")
        elif self.paused and f >= self.resume_at:
            self.paused = False
            if self.pause_events:
                self.pause_events[-1].update(resumed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), free=f)
            self.log(f"DISK RESUME: {f / 1e9:.2f} GB free >= {self.resume_at / 1e9:.1f} GB")
        return not self.paused

    def wait(self, marker=None):
        """Block until running is allowed; returns the seconds spent paused (0.0 if no pause).  ``marker``: a JSON
        file holding {start, end} epochs of the pause (written when it starts and when it ends), read by the driver
        to exclude paused time from the run timeout and from the used core-hours (read_pause_marker)."""
        t0 = self.clock()
        start_epoch = None
        while not self.ok():
            if start_epoch is None:
                start_epoch = time.time()
                if marker:
                    _write_marker(marker, dict(start=start_epoch))
            if self.clock() - t0 >= self.max_pause_s:
                if marker:
                    _write_marker(marker, dict(start=start_epoch, end=time.time(), timeout=True))
                raise DiskPauseTimeout(f"disk pause did not clear within {self.max_pause_s / 3600:.1f} h")
            self.sleep(self.poll_s)
        paused = self.clock() - t0 if start_epoch is not None else 0.0
        if start_epoch is not None and marker:
            _write_marker(marker, dict(start=start_epoch, end=time.time(), paused_s=paused))
        self.paused_total_s = getattr(self, "paused_total_s", 0.0) + paused
        return paused


def _write_marker(path, obj):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, f".{os.path.basename(path)}.tmp{os.getpid()}")
    with open(tmp, "w") as fh:
        json.dump(obj, fh)
    os.replace(tmp, path)


def read_pause_marker(path, now=None):
    """Seconds of disk pause recorded in a DiskGuard marker (an ongoing pause counts up to ``now``); 0.0 if none."""
    if not path or not os.path.exists(path):
        return 0.0
    try:
        with open(path) as fh:
            d = json.load(fh)
    except (OSError, ValueError):
        return 0.0
    if d.get("paused_s") is not None:
        return float(d["paused_s"])
    st = d.get("start")
    if st is None:
        return 0.0
    end = d.get("end", time.time() if now is None else now)
    return max(0.0, float(end) - float(st))
