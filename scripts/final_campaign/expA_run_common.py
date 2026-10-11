"""Shared, importable helpers of the Experiment A production driver and pre-production checks
(docs/final_campaign/PREREGISTRATION.md sections 1.3 and 1.4).

A real module (not a script): scripts/final_campaign/run_cells.py submits ``run_timed`` to a ProcessPoolExecutor, so
the pool workers unpickle it by importing this module by name (``expA_run_common``) -- under fork AND spawn.

* the frozen identities: EXPECTED_ENGINE_SHA256 (src/gateway_family_numba.py, as recorded in every mechanism
  production file's meta.engine_sha256, e.g. results/mechanism/matched_free_energy/shift/N128/s8100_abf.npz) and
  EXPECTED_CONFIG_SHA256 (configs/final_campaign/expA.json at the freeze commit c5a2264f); ``check_engine`` /
  ``load_config`` refuse anything else;
* ``rc()``: scripts/mechanism/run_cells.py imported by FILE PATH under the module name ``mechanism_run_cells`` (the
  final-campaign driver is also called run_cells.py; a plain ``import run_cells`` would be ambiguous);
* ``expA_jobs``: the mechanism job_list of the expA config, re-ordered SEED-MAJOR (every cell x N x arm of a seed
  before the next seed; within a seed: config variant order, config N order, abf before fr);
* roots: production results/final_campaign/expA, smoke results/final_campaign/expA/smoke/T<smoke_T>;
* ``wait_for_disk``: the prereg section 3 disk pause (below 2 GB free on / no new job starts; resume at >= 3 GB; every
  pause logged; a pause longer than 24 h is the stopping rule (iv));
* ``append_csv`` (flock-protected append with header check) for the compute ledger and the run-provenance table;
* ``run_timed``: the pool worker -- re-checks the engine sha256 inside the worker, runs mechanism run_one (which calls
  gateway_family_numba.run_job: a complete result of a DIFFERENT run is refused, never overwritten), checks the saved
  state for non-finite values (the engine does not raise on NaN: a NaN x is clamped into bin 0 and the run is saved
  'complete'; such a run is re-labelled NONFINITE -- prereg stopping rule (i)), accounts the compute of earlier
  interrupted sessions of a resumed run (from the engine's own session record), and returns the ledger row plus the
  measured wall time and provenance (git commit, engine sha256, CPU set, versions);
* ``cpu_allocation_problems``: prereg section 3 -- at most 64 concurrent physical cores, SMT siblings of used cores
  idle: the affinity set must not contain two SMT siblings of one physical core, and workers <= physical cores <= 64;
* ``amendment_ref_found``: an explicit override / rule change is accepted only if its reference string occurs in the
  dated-amendment section of docs/final_campaign/PREREGISTRATION.md (below the '*Amendments (dated' line).
"""
from __future__ import annotations

import csv
import fcntl
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
for _k in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.expanduser("~/.cache/numba_final"))

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

EXPERIMENT = "expA"
CONFIG = os.path.join(ROOT, "configs", "final_campaign", "expA.json")
EXPECTED_CONFIG_SHA256 = "d1c6119d237b24136c21d7ba2d8220bd529189f033a6c03dfc94e2922f4263ea"
ENGINE_FILE = os.path.join(ROOT, "src", "gateway_family_numba.py")
EXPECTED_ENGINE_SHA256 = "69087f2c0c7edbe5e076615b9a4425ba7580a6c4537a61854ae6c633df9da4ed"
ENGINE_SHA_WITNESS = os.path.join(ROOT, "results", "mechanism", "matched_free_energy", "shift", "N128",
                                  "s8100_abf.npz")
MECH_RUN_CELLS = os.path.join(ROOT, "scripts", "mechanism", "run_cells.py")
PROD_ROOT = os.path.join(ROOT, "results", "final_campaign", "expA")
COMPUTE_LEDGER = os.path.join(ROOT, "results", "final_campaign", "compute_ledger.csv")
COMPUTE_FIELDS = ["utc_start", "stage", "item", "threads", "wall_s", "used_core_h", "note"]
PROV_FIELDS = ["experiment", "variant", "N", "seed", "method", "n_steps", "status", "start_utc", "end_utc",
               "elapsed_s", "engine_wall_s", "engine_sha256", "git_commit", "git_dirty", "config_sha256", "h", "lam",
               "precision", "device", "threads", "cpu_set", "pid", "host", "python", "numpy", "numba", "out_path",
               "out_sha256", "prior_sessions_s", "gate"]
PREREG = os.path.join(ROOT, "docs", "final_campaign", "PREREGISTRATION.md")
AMENDMENT_MARKER = "*Amendments (dated"
MAX_PHYSICAL_CORES = 64          # prereg section 3: at most 64 concurrent physical cores of this campaign
SMT_OFFSET = 128                 # prereg section 3: siblings c and c + 128 (fallback when /sys has no topology)
FINITE_KEYS = ("X_final", "Y_final", "M_all", "C_all", "max_abs_bias")
DISK_LOW = 2 * 10 ** 9            # bytes: below this no new job starts (prereg section 3)
DISK_RESUME = 3 * 10 ** 9         # bytes: resume at or above this
DISK_POLL_S = 60.0
DISK_MAX_PAUSE_S = 24 * 3600.0    # stopping rule (iv)
DISK_USAGE = shutil.disk_usage    # (patched in the tests)


class EngineMismatch(RuntimeError):
    pass


class ConfigMismatch(RuntimeError):
    pass


class DiskPauseTimeout(RuntimeError):
    pass


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def utc(t=None):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() if t is None else t))


# ------------------------------------------------------------------------------------------------- identities
def check_engine(expected=None, engine_file=None, imported=True):
    """Refuse (EngineMismatch) unless the engine file AND the imported module hash to ``expected`` (default: the
    module constant EXPECTED_ENGINE_SHA256, read at call time)."""
    expected = expected or EXPECTED_ENGINE_SHA256
    engine_file = engine_file or ENGINE_FILE
    got = sha256_file(engine_file)
    if got != expected:
        raise EngineMismatch(f"{engine_file}: sha256 {got} != the frozen engine {expected}")
    if imported:
        import gateway_family_numba as E
        imp = E._PROVENANCE["engine_sha256"]
        if imp != expected:
            raise EngineMismatch(f"imported gateway_family_numba hashes to {imp} != the frozen engine {expected}")
        if os.path.abspath(E.__file__) != os.path.abspath(engine_file):
            raise EngineMismatch(f"imported gateway_family_numba is {E.__file__}, not {engine_file}")
    return got


def witness_engine_sha(path=ENGINE_SHA_WITNESS):
    """engine_sha256 recorded in a stored mechanism production file (None if the file is absent)."""
    if not os.path.exists(path):
        return None
    import numpy as np
    with np.load(path, allow_pickle=False) as z:
        return json.loads(str(z["meta_json"])).get("engine_sha256")


def load_config(path=None, expected=None):
    """The frozen expA config; refuses (ConfigMismatch) a file whose sha256 is not ``expected`` (default: the module
    constant EXPECTED_CONFIG_SHA256, read at call time)."""
    path = path or CONFIG
    expected = expected or EXPECTED_CONFIG_SHA256
    got = sha256_file(path)
    if got != expected:
        raise ConfigMismatch(f"{path}: sha256 {got} != the frozen {expected}")
    with open(path) as fh:
        P = json.load(fh)
    if os.path.basename(P["out_root"].rstrip("/")) != EXPERIMENT:
        raise ConfigMismatch(f"{path}: out_root {P['out_root']!r} does not end in {EXPERIMENT!r}")
    return P


_RC = None


def rc():
    """scripts/mechanism/run_cells.py (unmodified), imported by file path as module ``mechanism_run_cells``."""
    global _RC
    if _RC is None:
        name = "mechanism_run_cells"
        if name in sys.modules:
            _RC = sys.modules[name]
        else:
            spec = importlib.util.spec_from_file_location(name, MECH_RUN_CELLS)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
            _RC = mod
    return _RC


def provenance():
    """Provenance block of a check record: HEAD commit, engine sha256 (file), versions, host, time."""
    import subprocess
    try:
        commit = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True,
                                timeout=20).stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        commit = "unknown"
    out = dict(git_commit=commit, engine_file_sha256=sha256_file(ENGINE_FILE), host=platform.node(),
               python=platform.python_version(), utc=utc())
    try:
        import numba
        import numpy as np
        out.update(numpy=np.__version__, numba=numba.__version__)
    except Exception:  # noqa: BLE001
        pass
    try:
        out["cpu_set"] = " ".join(str(c) for c in sorted(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        pass
    return out


def amendment_ref_found(ref, prereg=None):
    """True iff ``ref`` (>= 6 non-blank characters) occurs in the dated-amendment section of the preregistration (the
    text after the line holding AMENDMENT_MARKER).  Used to license an explicit override or rule change: a flag alone
    is never enough, the amendment must exist in the prereg first."""
    if not ref or len(str(ref).strip()) < 6:
        return False
    try:
        with open(prereg or PREREG) as fh:
            txt = fh.read()
    except OSError:
        return False
    i = txt.find(AMENDMENT_MARKER)
    if i < 0:
        return False
    tail = txt[txt.find("\n", i) + 1:] if txt.find("\n", i) >= 0 else ""
    return str(ref).strip() in tail


def smt_siblings(c, cpus_online=None):
    """The SMT sibling set of logical CPU c (from /sys; fallback: {c, c +/- SMT_OFFSET}, the prereg host layout)."""
    p = f"/sys/devices/system/cpu/cpu{int(c)}/topology/thread_siblings_list"
    try:
        with open(p) as fh:
            txt = fh.read().strip()
        out = set()
        for part in txt.split(","):
            lo, _, hi = part.partition("-")
            out.update(range(int(lo), int(hi or lo) + 1))
        if out:
            return out
    except (OSError, ValueError):
        pass
    return {int(c), int(c) + SMT_OFFSET if int(c) < SMT_OFFSET else int(c) - SMT_OFFSET}


def cpu_allocation_problems(workers, cpus=None, siblings_of=None, max_cores=None):
    """Prereg section 3 for one launcher: ``workers`` single-threaded processes on the affinity set ``cpus`` (default:
    this process's).  Problems (list of str; empty = allowed): workers < 1; workers > max_cores (64); the affinity set
    contains two SMT siblings of one physical core (the OS could then run two workers on one core and the sibling of a
    used core would not stay idle); workers > the number of physical cores in the affinity set."""
    max_cores = MAX_PHYSICAL_CORES if max_cores is None else max_cores
    if cpus is None:
        try:
            cpus = os.sched_getaffinity(0)
        except (AttributeError, OSError):
            cpus = set(range(os.cpu_count() or 1))
    cpus = {int(c) for c in cpus}
    sib = siblings_of or smt_siblings
    probs, groups, pairs = [], set(), []
    for c in sorted(cpus):
        g = frozenset(int(x) for x in sib(c)) | {c}
        groups.add(g)
        both = sorted(g & cpus)
        if len(both) > 1 and both[0] == c:
            pairs.append(both)
    if workers < 1:
        probs.append(f"--workers {workers} < 1")
    if workers > max_cores:
        probs.append(f"--workers {workers} > {max_cores} (prereg section 3: at most {max_cores} concurrent physical "
                     f"cores)")
    if pairs:
        probs.append(f"the affinity set contains SMT siblings of one physical core {pairs[:4]}"
                     + (" ..." if len(pairs) > 4 else "") + " (prereg section 3: siblings of used cores stay idle; pin "
                     "with taskset to one logical CPU per physical core)")
    if workers > len(groups):
        probs.append(f"--workers {workers} > {len(groups)} physical cores in this process's affinity set (pin with "
                     f"taskset)")
    return probs


def nonfinite_keys(path, keys=FINITE_KEYS):
    """Keys of a saved result whose arrays hold a non-finite value (or are missing)."""
    import numpy as np
    bad = []
    with np.load(path, allow_pickle=False) as z:
        for k in keys:
            if k not in z.files:
                bad.append(f"{k} (missing)")
            elif not np.all(np.isfinite(np.asarray(z[k], dtype=np.float64))):
                bad.append(k)
    return bad


def prior_sessions_s(meta, checkpoint_every_s):
    """Compute (s) of the EARLIER sessions of a resumed run, from the engine's session record: for each earlier session
    the checkpointed work (last_ckpt_epoch - run_start_epoch) plus at most one checkpoint interval of unsaved work
    after its last checkpoint (an upper bound: the session was interrupted before its next checkpoint).  0 for a run
    completed in one session.  Sessions killed before their first checkpoint left no record (< one interval each)."""
    sess = list(meta.get("sessions") or [])
    tot = 0.0
    for s in sess[:-1]:
        a, b = s.get("run_start_epoch"), s.get("last_ckpt_epoch")
        if a is not None and b is not None:
            tot += max(0.0, float(b) - float(a))
        tot += float(checkpoint_every_s)
    return tot


# ------------------------------------------------------------------------------------------------- jobs, roots
def prod_root(out_root=None):
    return os.path.abspath(out_root) if out_root else PROD_ROOT


def smoke_root(out_root=None, smoke_T=4.0):
    return os.path.join(prod_root(out_root), "smoke", f"T{float(smoke_T):g}")


def seed_major(P, jobs):
    """Seed-major order: seed (first-appearance order), then config variant order, config N order, abf before fr."""
    vorder = {v["name"]: i for i, v in enumerate(P["variants"])}
    norder = {int(n): i for i, n in enumerate(P["N"])}
    seeds = []
    for j in jobs:
        if j["seed"] not in seeds:
            seeds.append(j["seed"])
    sorder = {s: i for i, s in enumerate(seeds)}
    morder = {"abf": 0, "fr": 1}
    return sorted(jobs, key=lambda j: (sorder[j["seed"]], vorder[j["variant"]], norder[int(j["N"])],
                                       morder[j["method"]]))


def expA_jobs(P, smoke=False, smoke_T=4.0, only_cell=None, only_N=None, seeds=None):
    """The mechanism job_list of the expA config (experiment label 'expA'), seed-major.  Production seeds = the
    config's seeds (8300-8331), smoke seeds = its smoke_seeds (8390, 8391), unless ``seeds`` is given."""
    R = rc()
    if only_cell:
        names = [v["name"] for v in P["variants"]]
        bad = [c for c in only_cell if c not in names]
        if bad:
            raise ValueError(f"unknown cell(s) {bad}; cells are {names}")
    if seeds:
        seeds = [int(s) for s in seeds]
    else:
        seeds = [int(s) for s in (P["smoke_seeds"] if smoke else P["seeds"])]
    J = R.job_list(P, EXPERIMENT, smoke=smoke, smoke_T=smoke_T, only_variant=only_cell, only_N=only_N, seeds=seeds)
    for j in J:
        if int(j["N"]) < 2:
            raise ValueError("Experiment A has no N = 1 cell")
    return seed_major(P, J)


def job_signature(job):
    import gateway_family_numba as E
    return rc().job_signature(E, job)


# ------------------------------------------------------------------------------------------------- disk pause
def wait_for_disk(path="/", low=None, resume=None, poll_s=None, max_pause_s=None, log=None, usage=None,
                  sleep=time.sleep, clock=time.monotonic):
    """Return the free bytes once a new job may start.  Below ``low`` free bytes: pause, polling every ``poll_s``
    seconds (``sleep(poll_s)``), until at least ``resume`` bytes are free; every poll is logged (``log(msg)``).  A
    pause longer than ``max_pause_s`` raises DiskPauseTimeout (prereg section 3, stopping rule (iv)).  Defaults: the
    module constants DISK_LOW / DISK_RESUME / DISK_POLL_S / DISK_MAX_PAUSE_S / DISK_USAGE, read at call time."""
    low = DISK_LOW if low is None else low
    resume = DISK_RESUME if resume is None else resume
    poll_s = DISK_POLL_S if poll_s is None else poll_s
    max_pause_s = DISK_MAX_PAUSE_S if max_pause_s is None else max_pause_s
    usage = usage or DISK_USAGE
    free = usage(path).free
    if free >= low:
        return free
    t0 = clock()
    say = log or (lambda m: print(m, flush=True))
    say(f"{utc()} DISK PAUSE: {free / 1e9:.2f} GB free on {path} < {low / 1e9:.1f} GB; no new job starts until "
        f">= {resume / 1e9:.1f} GB")
    while True:
        sleep(poll_s)
        free = usage(path).free
        waited = clock() - t0
        if free >= resume:
            say(f"{utc()} DISK RESUME: {free / 1e9:.2f} GB free after a {waited:.0f} s pause")
            return free
        say(f"{utc()} disk pause continues: {free / 1e9:.2f} GB free, paused {waited:.0f} s")
        if waited > max_pause_s:
            say(f"{utc()} DISK PAUSE TIMEOUT: still {free / 1e9:.2f} GB free after {waited:.0f} s (stopping rule iv)")
            raise DiskPauseTimeout(f"disk pause did not clear within {max_pause_s:.0f} s")


# ------------------------------------------------------------------------------------------------- csv ledgers
def append_csv(path, fields, rows):
    """Append rows under an exclusive flock; a new / empty file gets the header; an existing file keeps its own
    header (rows written in its column order; a column it lacks is dropped)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a+", newline="") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            fh.seek(0)
            head = next(csv.reader(fh), None)
            fh.seek(0, os.SEEK_END)
            cols = head or list(fields)
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            if not head:
                w.writeheader()
            for r in rows:
                w.writerow(r)
            fh.flush()
            os.fsync(fh.fileno())
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def compute_row(stage, item, wall_s, start_epoch, note="", threads=1):
    return dict(utc_start=utc(start_epoch), stage=stage, item=item, threads=int(threads), wall_s=f"{wall_s:.3f}",
                used_core_h=f"{threads * wall_s / 3600.0:.6f}", note=note)


def job_item(job):
    return f"{job['variant']} N{job['N']} s{job['seed']} {job['method']} n_steps {job['n_steps']}"


# ------------------------------------------------------------------------------------------------- pool worker
def run_timed(job, root, expected_engine=None):
    """Pool worker: engine check, mechanism run_one, finiteness of the saved state (X_final, Y_final, M_all, C_all,
    max_abs_bias; a non-finite complete run is re-labelled 'NONFINITE: ...'), compute of earlier interrupted sessions
    (prior_sessions_s), measured wall time and provenance.  Never raises."""
    t0 = time.time()
    start = utc(t0)
    R = rc()
    try:
        check_engine(expected_engine)
        row = R.run_one(job, root)
    except Exception as e:  # noqa: BLE001  (never mark a failed run complete)
        row = R._row(job, f"FAILED: {type(e).__name__}: {e}", time.time() - t0, float("nan"), -1)
    out = R.out_path(root, job)
    prior = 0.0
    if row["status"].startswith("complete"):
        try:
            bad = nonfinite_keys(out)
        except Exception as e:  # noqa: BLE001
            bad = [f"unreadable result ({type(e).__name__}: {e})"]
        if bad:
            row["status"] = (f"NONFINITE: {', '.join(bad)} (file saved complete by the engine; prereg stopping rule "
                             f"(i))")
        try:
            import numpy as np
            with np.load(out, allow_pickle=False) as z:
                meta = json.loads(str(z["meta_json"]))
            prior = prior_sessions_s(meta, job.get("checkpoint_every_s", 600.0))
        except Exception:  # noqa: BLE001
            prior = 0.0
    t1 = time.time()
    prov = dict(experiment=job["experiment"], variant=job["variant"], N=job["N"], seed=job["seed"],
                method=job["method"], n_steps=job["n_steps"], status=row["status"], start_utc=start, end_utc=utc(t1),
                elapsed_s=f"{t1 - t0:.3f}", engine_wall_s=row["wall_s"], h=job["cfg"]["h"], lam=job["cfg"]["lam"],
                precision="float64", device="cpu", threads=1, pid=os.getpid(), host=platform.node(),
                python=platform.python_version(), out_path=os.path.relpath(out, ROOT) if out.startswith(ROOT) else out,
                prior_sessions_s=f"{prior:.3f}")
    try:
        prov["cpu_set"] = " ".join(str(c) for c in sorted(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        prov["cpu_set"] = ""
    try:
        import numba
        import numpy as np
        import gateway_family_numba as E
        prov.update(numpy=np.__version__, numba=numba.__version__, engine_sha256=E._PROVENANCE["engine_sha256"],
                    git_commit=E._PROVENANCE["git_commit"], git_dirty=E._PROVENANCE["git_dirty"])
    except Exception:  # noqa: BLE001
        pass
    try:
        prov["config_sha256"] = sha256_file(CONFIG)
    except OSError:
        prov["config_sha256"] = ""
    if (row["status"].startswith("complete") or row["status"].startswith("NONFINITE")) and os.path.exists(out):
        prov["out_sha256"] = sha256_file(out)
    return dict(row=row, prov=prov, start_epoch=t0, elapsed_s=t1 - t0, prior_sessions_s=prior)
