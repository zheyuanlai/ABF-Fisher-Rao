#!/usr/bin/env python
"""Final campaign, Experiment A: numerical-validation tooling (docs/final_campaign/PREREGISTRATION.md section 1.3:
timestep gate G-A1 for the three new dynamics, G-A2 for lam = 1, manipulation checks MC1 / MC2, the V5 ABF-bias smoke;
configs/final_campaign/expA_validation.json).

    python scripts/final_campaign/expA_validation.py dry-run   [run options]          # list pending jobs, cost; no run
    python scripts/final_campaign/expA_validation.py run       [--workers 8] [--cpus 64-71] [--h 2.5e-5 [1.25e-5]]
                                                               [--dynamics shift_lam4 ...] [--chains em_flat ...]
                                                               [--out results/final_campaign/expA/validation]
                                                               [--smoke --groups 1 --out <scratch>] [--dry-run]
    python scripts/final_campaign/expA_validation.py abf-smoke [--h 2.5e-5] [--workers 6] [--cpus 64-71]
                                                               [--dry-run] [--collect-only]
    python scripts/final_campaign/expA_validation.py analyze   [--res results/final_campaign/expA/validation]
                                                               [--fig figures/final_campaign/expA/validation | --no-fig]
                                                               [--abf-smoke <res>/abf_smoke.json] [--smoke]

Harness (frozen, imported -- never copied): src/gateway_family_validation.py (GF; run_chain gained ONE passive output,
side_changes, appended last: bitwise inert for the seven old dynamics, tests/test_final_expA_validation.py),
scripts/mechanism/run_validation.py (RV: the chain design, seed rule and settings of the seven old dynamics) and
scripts/mechanism/analyze_validation.py (AV: every gate clause and threshold).  Neither script is edited.

Dynamics registry (explicit; global order = GF.DYNAMICS_ORDER + the three new names, so the new dynamics take the
global indices 7, 8, 9; d' = index - 7):
  shift_lam4 / shift_lam16 / shift_lam128  shifted fibre kappa = 1, lam 4 / 16 / 128 (GF.make_model), physical
                constants from configs/final_campaign/expA_validation.json (checked equal to the frozen mechanism
                configs).  Chains exactly as RV (em_flat 16 groups x 256 walkers, 10 t.u. burn-in + T, x0 ~ U[-1.5, 1.5],
                y0 from the exact conditional, conditional ACF on the first 64 walkers, raw ACF excerpts and x/y traces
                in group 0 only; mala_flat h 1e-4, 500 t.u.; exact_flat 2^24 and exact_unbiased 2^22 i.i.d. draws per
                group; em_unbiased 8 walkers, 200 t.u.) EXCEPT, per the prereg: em_flat T = 500 / min(lam, 1) t.u.
                (= 500 for lam >= 1; RV's 500/lam would cut lam 16 to 31 t.u.), group-0 trace stride
                0.05 / (min(lam, 1) h) (prereg; applied to mala_flat too -- RV's 0.05 / (lam h) would store 1.3 M
                samples per traced walker at lam 128), and the conditional-ACF lags of BOTH levels extended (same
                strides, same lag-grid rule) to reach >= 6 t.u. absolute (ACF_HORIZON_T), so MC1 reads the same cuts
                at every lam.  Seeds (prereg section 4):
                em_flat 400000 + 1000 d' + 100 k + g, mala_flat 410000 + 100 d' + g, em_unbiased 411000 + 100 d' + 10 k,
                exact_flat 420000 + 100 d' + g, exact_unbiased 421000 + 100 d' + g (k = index of h in the FIXED list
                (2.5e-5, 1.25e-5)); checked against the config's seed_rule strings at import.
  shift         lam = 1, the OLD dynamics (index 3), RV's design and seed rule unchanged.  Its gate record G-A2 is
                results/mechanism/validation/summary.json (dynamics.shift, h 2.5e-5).  Jobs here: (i) at an h WITH a
                frozen record, a REPLAY of its 16 em_flat chains (same seeds) -- every frozen array must come out
                bitwise identical (a full-length inertness check, recorded per file in meta.replay) and the new
                side_changes output gives MC2 at lam = 1 from the G-A2 chains themselves; (ii) at an h WITHOUT a frozen
                record (the refinement 1.25e-5, G-A2: "if refinement is triggered, lam = 1 is re-gated"), em_flat and
                em_unbiased (mala/exact are h-independent and in the frozen record).
Output layout = RV's: <out>/<dynamics>/<chain>_h<h>/g<kk>.npz (exact: <out>/<dynamics>/<chain>/g<kk>.npz), arrays = RV's
plus side_changes; meta = RV's plus a run signature (design + harness sha256), provenance (git commit / dirty flag,
sha256 of harness, engine, scripts and config, library versions, host, CPU affinity) and, for replays, the bitwise
comparison.  The new dynamics are accepted ONLY with --out results/final_campaign/expA/validation (results/mechanism/
validation is refused for everything); --smoke needs an --out outside both.  Resumable at file granularity: a file
whose stored signature equals the job's is skipped; a file with a DIFFERENT (or no) signature is refused (FAILED, never
overwritten).  <out>/ledger.csv gets one row per finished job (wall, walker-steps; a FAILED job gets its
submit-to-failure wall time).
Pool and disk pause (prereg section 3), for run AND abf-smoke: jobs are submitted LAZILY (at most --workers in flight,
longest first) and, before EVERY job start, expA_run_common.wait_for_disk holds new starts while fewer than 2 GB are free
on / and resumes at >= 3 GB (every poll logged to stdout and <out>/disk_pause.log; running jobs keep being collected);
a pause longer than 24 h stops the command (exit 3, stopping rule (iv)); a failed ledger write or a broken pool stops new
starts (exit 1).  The pool workers are __main__.run_job (run) and expA_run_common.run_timed (abf-smoke), both importable
by name under Python 3.14's default forkserver start method (and under spawn / fork) -- nothing loaded with _load is
ever submitted.  summary / hook writes also wait for the disk.

abf-smoke (V5 hook; prereg section 1.3 V5, seeds 8392, 8393 from configs/final_campaign/expA.json): live ABF + FR at
N 2048, 4 t.u. (B = 2048 * 4 / h), both arms, for every new cell (and shift_lam1 at an h without a frozen record),
through scripts/mechanism/run_cells.py's run_one / job_state (gateway_family_numba.run_job; signature-checked,
resumable).  The engine file AND the imported engine must hash to expA_run_common.EXPECTED_ENGINE_SHA256 (refused,
exit 2, otherwise; run_timed re-checks it inside every worker).  Cell cfg: configs/final_campaign/cells/expA/<cell>.json
if it exists (engine_cfg + model; every key but h must equal the cfg built from configs/final_campaign/expA.json, else
refused), otherwise built from expA.json.  Output <out>/abf_smoke/h<h>/T4/<cell>/N2048/s<seed>_<method>.npz (+
run_provenance.csv there); collect = scripts/mechanism/abf_smoke.py's record() of the files that are COMPLETE results of
the expected job (gateway_family_numba.is_complete with run_cells.job_signature: cfg, method, N, seed, n_steps, save
grid, engine sha256) -> <out>/abf_smoke.json ({dynamics: {"<h>": [records]}}, the hook AV.gate_v5 reads; shift_lam1 is
filed as 'shift').  A cell is filed only with all 4 runs valid; otherwise (missing, STALE, no file) its entry for that h
is REMOVED, so V5 reads PENDING; stale / unexpected files are reported and make the command exit 1.

analyze: the frozen clauses V1, V2, V3, V3b, V4, V5 (AV functions, frozen thresholds) for the new dynamics, with the
preregistered multiplicity: the V1 shifted-fibre residual-mean 3-se clauses of the new dynamics at one h form ONE Holm
family (two-sided t_{G-1} p, FWER 5 %), and all their V4 3-se clauses (MALA and exact, central variance and residual
mean) form ONE Holm family; every tolerance clause unchanged.  lam = 1 enters with its G-A2 record (h 2.5e-5) or, at
the refinement h, with its re-gate in <res>/shift judged by the frozen AV clauses unchanged.  Selection (prereg):
AV.decide over {shift, shift_lam4, shift_lam16, shift_lam128}: h_A = 2.5e-5 iff all PASS (V1, V2, V3, V3b, V5) and V4
holds; otherwise all four move to 1.25e-5 (REFINE_REQUIRED); a failure there = UNRESOLVED (Experiment A stops).
MC1: from the em_flat lag sums, per lam and window x0 in {-1, -0.21, 0}: 1/e time and integrals of rho_y, rho_f to the
cuts 0.3125, 1.25, 5 t.u., with 16-group jackknife se and t_{G-1} 95 % CIs (lam = 1 recomputed from the frozen raw files
with the same code); z_f is not reported where the exact sd(f | x) < AV.SDF_MIN_ACF (x0 = -1: rounding noise).  MC2:
side changes per walker per t.u. (jackknife over groups); a trace-based cross-check from the group-0 traces (8 walkers,
every 0.05 t.u.) is reported for every lam.  Ratios to lam = 1 use independent-group delta-method CIs on the log scale.
lam = 1 inputs are design-checked like the new dynamics (lam1_inputs): MC1 from the frozen files (frozen design, 16
groups); MC2 from the replay only if all 16 replayed files conform to the design AND are bitwise identical to the frozen
record (inertness_replay PASS) -- otherwise MC2 at lam = 1 is not reported (INCOMPLETE / NOT_RUN: a selection note) or,
if any replayed file is NOT bitwise identical (FAIL: the harness extension is not inert), the selection is HARNESS_FAIL.
Writes <res>/summary.json, <res>/summary.md and (unless --no-fig) figures A4 (MC1) and A4b (MC2) to the --fig directory.
No ABF/FR performance quantity is read anywhere.

Cost (single-core timings on this host 2026-10-11 under load ~60: em_flat 124-132 ns, mala_flat 195 ns per walker-step):
the default h 2.5e-5 job set is ~15.9 core-h (new dynamics ~12.8 = 3 x [em_flat 2.9 + mala 1.1 + exact 0.1]; the
lam = 1 replay ~3.0); with 8 workers ~2 h wall.  The refinement h 1.25e-5 (all four cells) adds ~23 core-h.  The V5 ABF
smoke is 12 runs x 3.3e8 walker-steps (~0.1-0.2 core-h).  Disk: ~4 MB per dynamics and h.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import time
import types

if __name__ == "__main__":          # (not when imported by the tests)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for _k in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[_k] = "1"
    os.environ.setdefault("NUMBA_CACHE_DIR", os.path.expanduser("~/.cache/numba_final"))
import numpy as np  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
if HERE not in sys.path:
    sys.path.insert(1, HERE)
import gateway_family_validation as GF  # noqa: E402
import expA_run_common as C  # noqa: E402  (importable module: disk pause, engine check, pool worker run_timed, csv)


def _load(modname, relpath):
    """Import a frozen script under its own module name (registered in sys.modules).  NOTE: a module loaded this way
    is NOT importable by name in a pool worker (Python 3.14's default start method is forkserver: the workers
    re-import only __main__ and named modules), so nothing loaded with _load is ever submitted to a pool; the pool
    workers are __main__.run_job (run) and expA_run_common.run_timed (abf-smoke)."""
    if modname in sys.modules:
        return sys.modules[modname]
    spec = importlib.util.spec_from_file_location(modname, os.path.join(ROOT, relpath))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        del sys.modules[modname]
        raise
    return mod


RV = _load("mech_run_validation", os.path.join("scripts", "mechanism", "run_validation.py"))
AV = _load("mech_analyze_validation", os.path.join("scripts", "mechanism", "analyze_validation.py"))

OUT_EXPA = os.path.join(ROOT, "results", "final_campaign", "expA", "validation")
OUT_FROZEN = RV.OUT_DEFAULT                                           # results/mechanism/validation (frozen)
CFG_VAL = os.path.join(ROOT, "configs", "final_campaign", "expA_validation.json")
CFG_EXPA = os.path.join(ROOT, "configs", "final_campaign", "expA.json")
CELL_DIR = os.path.join(ROOT, "configs", "final_campaign", "cells", "expA")
FIG_DEFAULT = os.path.join(ROOT, "figures", "final_campaign", "expA", "validation")
SCOUT_SUMMARY = os.path.join(ROOT, "results", "final_campaign", "expA", "scout", "scout_summary.json")
HARNESS = os.path.join(ROOT, "src", "gateway_family_validation.py")
ENGINE = os.path.join(ROOT, "src", "gateway_family_numba.py")
NEW_INDEX0 = 7
ACF_HORIZON_T = 6.0
MC1_CUTS = (0.3125, 1.25, 5.0)
LAM1 = "shift"
H_CANDIDATES = tuple(RV.H_CANDIDATES)
H_MALA = RV.H_MALA
CHAINS = tuple(RV.CHAINS)
EM_UNBIASED_X0 = (-1.0, -1.0, -0.5, 0.0, 0.0, 0.5, 1.0, 1.0)         # RV.run's em_unbiased start (tested bitwise)
SEED_BASE_NEW = dict(em_flat=400000, mala_flat=410000, em_unbiased=411000, exact_flat=420000, exact_unbiased=421000)
SMOKE_T, SMOKE_N0, SMOKE_N = 4.0, 2048, 2048
# single-core timings on this host (2026-10-11, load ~60, 256 walkers, em_flat with the 64-walker ACF recorder): em_flat
# 124-132 ns (lam 1-128), mala_flat 195 ns per walker-step; exact draws ~0.004 core-h per 2^24 group
NS_PER_WALKER_STEP = dict(em_flat=130.0, em_unbiased=130.0, mala_flat=195.0)
EXACT_CORE_H_PER_GROUP = 0.004
LEDGER_FIELDS = ["finished_utc", "chain", "dynamics", "dynamics_index", "h", "groups", "status", "wall_s",
                 "walker_steps", "note"]


# ---------------------------------------------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------------------------------------------
def _phys(cfg):
    ec = cfg["engine_cfg"]
    return dict(beta=float(ec["beta"]), H=float(ec["H"]), omega_out=float(ec["omega_out"]),
                omega_in=float(ec["omega_in"]), s=float(ec["s"]))


def load_registry(cfg_path=CFG_VAL):
    """[lam = 1 (old 'shift', index 3), the new dynamics in config order (indices 7, 8, 9)]; refuses any drift from
    the frozen configs."""
    c = json.load(open(cfg_path))
    old = GF.load_dynamics()
    if tuple(m["name"] for m in old) != GF.DYNAMICS_ORDER:
        raise RuntimeError("GF.load_dynamics does not return DYNAMICS_ORDER")
    phys = _phys(c)
    d1 = GF.DYNAMICS_ORDER.index(LAM1)
    lam1 = old[d1]
    if (lam1["variant"] != "shift" or lam1["lam"] != 1.0 or lam1["kappa"] != 1.0
            or any(lam1[k] != v for k, v in phys.items())):
        raise RuntimeError("the frozen 'shift' dynamics is not the lam = 1 cell of Experiment A")
    reg = [dict(name=LAM1, index=d1, model=lam1, new=False)]
    for pos, v in enumerate(c["variants"]):
        idx = int(v["dynamics_index"])
        if (idx != NEW_INDEX0 + pos or v["variant"] != "shift" or float(v["kappa"]) != 1.0
                or v["name"] != f"shift_lam{float(v['lam']):g}" or float(v["lam"]) <= 1.0):
            raise RuntimeError(f"unexpected variant in {cfg_path}: {v}")
        m = GF.make_model("shift", 1.0, float(v["kappa"]), float(v["lam"]), name=v["name"], **phys)
        reg.append(dict(name=v["name"], index=idx, model=m, new=True))
    if tuple(c["h_candidates"]) != H_CANDIDATES:
        raise RuntimeError("h candidates of the config and of run_validation.py differ")
    return reg


REGISTRY = load_registry()
BY_NAME = {e["name"]: e for e in REGISTRY}
BY_INDEX = {e["index"]: e for e in REGISTRY}
NEW_NAMES = tuple(e["name"] for e in REGISTRY if e["new"])
DYNAMICS_ORDER = tuple(GF.DYNAMICS_ORDER) + NEW_NAMES                 # global order (new dynamics at 7, 8, 9)
ALL_NAMES = (LAM1,) + NEW_NAMES                                        # the four Experiment A cells (lam 1, 4, 16, 128)
for _e in REGISTRY:
    if DYNAMICS_ORDER.index(_e["name"]) != _e["index"]:
        raise RuntimeError("registry index != global DYNAMICS_ORDER index")


def is_new(m):
    return m["name"] in NEW_NAMES


def lam_of(name):
    return float(BY_NAME[name]["model"]["lam"])


# ---------------------------------------------------------------------------------------------------------------
# design: seeds, settings, traces, ACF plan (old dynamics: RV / GF unchanged)
# ---------------------------------------------------------------------------------------------------------------
def seed_of(chain, d, h, g):
    """RV.seed_of for the seven old dynamics; the prereg section 4 rule for d >= 7."""
    if d < NEW_INDEX0:
        return RV.seed_of(chain, d, h, g)
    dp = d - NEW_INDEX0
    if dp not in (0, 1, 2):
        raise ValueError(f"no seed rule for dynamics index {d}")
    if chain == "em_flat":
        return SEED_BASE_NEW[chain] + 1000 * dp + 100 * RV.h_index(h) + g
    if chain == "em_unbiased":
        return SEED_BASE_NEW[chain] + 100 * dp + 10 * RV.h_index(h)
    return SEED_BASE_NEW[chain] + 100 * dp + g


def _check_seed_rule(cfg_path=CFG_VAL):
    """seed_of == the config's seed_rule strings for every d', k, g (refuses drift)."""
    rule = json.load(open(cfg_path))["seed_rule"]
    for ch in CHAINS:
        for dp in range(3):
            for k, h in enumerate(H_CANDIDATES):
                for g in range(RV.G):
                    want = eval(rule[ch], {"__builtins__": {}}, dict(dprime=dp, k=k, g=g))  # noqa: S307 (frozen config)
                    got = seed_of(ch, NEW_INDEX0 + dp, h, g)
                    if ch == "em_unbiased" and g:
                        continue
                    if want != got:
                        raise RuntimeError(f"seed rule drift: {ch} d'={dp} k={k} g={g}: {got} != config {want}")


_check_seed_rule()


def settings(chain, m, smoke):
    """(T, burn) in t.u.: RV.settings for the old dynamics; em_flat T = 500 / min(lam, 1) for the new ones."""
    if not is_new(m):
        return RV.settings(chain, m, smoke)
    if chain == "em_flat":
        return ((5.0 if smoke else RV.T_RUN) / min(m["lam"], 1.0), 1.0 if smoke else RV.T_BURN)
    return RV.settings(chain, m, smoke)


def trace_every(chain, m, h, g):
    """RV.run's trace stride (old dynamics); 0.05 / (min(lam, 1) h) in group 0 for the new ones."""
    if chain == "em_unbiased":
        return max(1, int(round(0.05 / h)))
    if g != 0:
        return 10 ** 12
    lam = min(m["lam"], 1.0) if is_new(m) else m["lam"]
    return max(1, int(round(0.05 / (lam * h))))


def _lag_grid(L, n_lags=80, n_small=16):
    """GF.acf_plan's lag-grid rule (0..n_small, then n_lags log-spaced up to L; tested equal to GF's grids)."""
    g = np.unique(np.concatenate([np.arange(0, min(L, n_small) + 1),
                                  np.round(np.geomspace(max(1, n_small), max(L, n_small), n_lags)).astype(np.int64)]))
    return g[g <= L].astype(np.int64)


def acf_plan(m, h):
    """GF.acf_plan for the old dynamics; for the new ones the same strides with both levels' lags extended to reach
    >= ACF_HORIZON_T t.u."""
    p = GF.acf_plan(m, h)
    if not is_new(m):
        return p
    p = dict(p)
    for lev, key in ((0, "lags0"), (1, "lags1")):
        st = int(p["stride"][lev])
        if int(p[key][-1]) * st * h < ACF_HORIZON_T:
            L = int(math.ceil(ACF_HORIZON_T / (st * h) - 1e-9))
            while L * st * h < ACF_HORIZON_T:
                L += 1
            p[key] = _lag_grid(L)
    p["horizon_t"] = [float(p["lags0"][-1] * p["stride"][0] * h), float(p["lags1"][-1] * p["stride"][1] * h)]
    return p


def frozen_record_exists(name, chain, h, root=OUT_FROZEN):
    gs = [0] if chain == "em_unbiased" else range(RV.G)
    return all(os.path.exists(RV.out_path(root, name, chain, h, g)) for g in gs)


def jobs(dyns, hs, chains, groups, smoke):
    """(chain, dynamics_index, h, [groups]) as RV.jobs, for the Experiment A registry (see the module docstring for
    lam = 1)."""
    J = []
    for name in dyns:
        e = BY_NAME[name]
        d = e["index"]
        for c in chains:
            if c == "em_flat":
                J += [(c, d, h, [g]) for h in hs for g in range(groups)]
            elif c == "em_unbiased":
                J += [(c, d, h, [0]) for h in hs if e["new"] or not frozen_record_exists(name, "em_unbiased", h)]
            elif not e["new"]:
                continue                                  # mala / exact of lam = 1: h-independent, in the frozen record
            elif c == "mala_flat":
                J += [(c, d, H_MALA, [g]) for g in range(groups)]
            else:
                J.append((c, d, None, list(range(groups))))
    return J


def out_path(out, d, chain, h, g):
    return RV.out_path(out, BY_INDEX[d]["name"], chain, h, g)


def cost_walker_steps(j, smoke):
    c, d, h, gs = j
    m = BY_INDEX[d]["model"]
    if c.startswith("exact"):
        return 0.0
    T, burn = settings(c, m, smoke)
    nw = len(EM_UNBIASED_X0) if c == "em_unbiased" else RV.NW
    return nw * round((T + burn) / h) * len(gs)


# ---------------------------------------------------------------------------------------------------------------
# provenance and signatures
# ---------------------------------------------------------------------------------------------------------------
_SHA_CACHE = {}


def sha256_file(path):
    """sha256 of a file (memoised on path, size and mtime)."""
    st = os.stat(path)
    key = (os.path.abspath(path), st.st_size, st.st_mtime_ns)
    if key not in _SHA_CACHE:
        hh = hashlib.sha256()
        with open(path, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 20), b""):
                hh.update(b)
        _SHA_CACHE[key] = hh.hexdigest()
    return _SHA_CACHE[key]


def _git(*args):
    try:
        return subprocess.check_output(["git", "-C", ROOT] + list(args), text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


_PROV = {}


def provenance():
    if not _PROV:
        import numba
        import scipy
        dirty = _git("status", "--porcelain", "--", "src/gateway_family_validation.py", "src/gateway_family_numba.py",
                     "scripts/mechanism", "scripts/final_campaign", "configs/final_campaign")
        _PROV.update(
            git_commit=_git("rev-parse", "HEAD") or "unknown", git_dirty=bool(dirty) if dirty is not None else None,
            harness_sha256=sha256_file(HARNESS), engine_sha256=sha256_file(ENGINE),
            script_sha256=sha256_file(os.path.abspath(__file__)),
            run_validation_sha256=sha256_file(os.path.join(ROOT, "scripts", "mechanism", "run_validation.py")),
            analyze_validation_sha256=sha256_file(os.path.join(ROOT, "scripts", "mechanism", "analyze_validation.py")),
            config_sha256=sha256_file(CFG_VAL), python=platform.python_version(), numpy=np.__version__,
            numba=numba.__version__, scipy=scipy.__version__, host=platform.node(), precision="float64", device="CPU")
    p = dict(_PROV)
    p["cpu_affinity"] = sorted(os.sched_getaffinity(0))
    return p


def job_signature(c, d, h, g, smoke):
    """What defines the file of group g of job (c, d, h): the design and the harness (not the git commit)."""
    m = BY_INDEX[d]["model"]
    sig = dict(chain=c, dynamics=m["name"], dynamics_index=int(d), group=int(g), seed=int(seed_of(c, d, h, g)),
               smoke=bool(smoke), harness_sha256=sha256_file(HARNESS), config_sha256=sha256_file(CFG_VAL),
               model={k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in m.items()})
    if c.startswith("exact"):
        sig.update(n_draws=int((1 << 20) if smoke else (RV.N_EXACT_FLAT if c == "exact_flat" else RV.N_EXACT_UNB)),
                   flat=c == "exact_flat")
        return sig
    T, burn = settings(c, m, smoke)
    sig.update(h=float(h), T=float(T), burn=float(burn), n_steps=int(round((T + burn) / h)),
               n_burn=int(round(burn / h)), n_walkers=len(EM_UNBIASED_X0) if c == "em_unbiased" else RV.NW,
               flat_bias=0 if c == "em_unbiased" else 1, trace_every=int(trace_every(c, m, h, g)))
    if c == "em_flat":
        p = acf_plan(m, h)
        sig.update(n_acf=RV.N_ACF, acf_rec=RV.ACF_REC if g == 0 else 0, acf_stride=[int(v) for v in p["stride"]],
                   lags0=[int(v) for v in p["lags0"]], lags1=[int(v) for v in p["lags1"]])
    return json.loads(json.dumps(sig, sort_keys=True))


def file_state(path, sig):
    """'missing', 'complete' (stored signature == sig) or 'stale' (anything else: refused, never overwritten)."""
    if not os.path.exists(path):
        return "missing"
    try:
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta_json"]))
    except Exception:
        return "stale"
    return "complete" if meta.get("signature") == sig else "stale"


def job_states(j, out, smoke):
    c, d, h, gs = j
    return {g: file_state(out_path(out, d, c, h, g), job_signature(c, d, h, g, smoke)) for g in gs}


# ---------------------------------------------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------------------------------------------
def replay_compare(arrays, meta, frozen_path):
    """Bitwise comparison of a lam = 1 replay with its frozen G-A2 file: every frozen array and the design fields."""
    with np.load(frozen_path, allow_pickle=False) as z:
        keys = [k for k in z.files if k != "meta_json"]
        fm = json.loads(str(z["meta_json"]))
        bad = [k for k in keys if k not in arrays or not np.array_equal(np.asarray(z[k]), np.asarray(arrays[k]))]
    fields = ("seed", "group", "h", "T", "burn", "n_steps", "n_burn", "n_walkers", "flat_bias", "trace_every", "n_acf")
    meta_bad = [k for k in fields if fm.get(k) != meta.get(k)]
    return dict(frozen_file=os.path.relpath(frozen_path, ROOT), frozen_sha256=sha256_file(frozen_path),
                keys_compared=keys, mismatched=bad, meta_mismatched=meta_bad, bitwise_equal=not bad and not meta_bad)


def _save(path, meta, arrays):
    """RV._save (atomic: <g>.tmp.npz then os.replace); on failure (e.g. ENOSPC) the partial temp file is removed and
    the error re-raised (the job is then FAILED in the ledger and a re-run redoes it)."""
    try:
        RV._save(path, meta, arrays)
    except BaseException:
        tmp = path[:-4] + ".tmp.npz"
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        raise


def run_job(j, out, smoke):
    """One job (RV.run's computation, with the Experiment A design functions); writes its group files atomically.
    Never overwrites: a group whose file exists is skipped (the caller refuses stale files before submitting)."""
    c, d, h, gs = j
    e = BY_INDEX[d]
    m = e["model"]
    t0 = time.perf_counter()
    prov = provenance()
    base = dict(chain=c, dynamics=m["name"], dynamics_index=d, model={k: m[k] for k in m}, smoke=bool(smoke),
                commit=prov["git_commit"], host=platform.node(), harness="src/gateway_family_validation.py",
                prereg="docs/final_campaign/PREREGISTRATION.md s1.3 (G-A1 / G-A2, MC1, MC2)",
                config="configs/final_campaign/expA_validation.json", runner="scripts/final_campaign/expA_validation.py",
                provenance=prov, new_dynamics=bool(e["new"]))
    walker_steps = 0.0
    if c.startswith("exact"):
        flat = c == "exact_flat"
        n = (1 << 20) if smoke else (RV.N_EXACT_FLAT if flat else RV.N_EXACT_UNB)
        for g in gs:
            p = out_path(out, d, c, h, g)
            if os.path.exists(p):
                continue
            t1 = time.perf_counter()
            seed = seed_of(c, d, h, g)
            A1, A2, zh = GF.exact_samples(n, m, seed, flat=flat)
            meta = dict(base, group=g, seed=seed, n_draws=n, flat=flat, wall_s=time.perf_counter() - t1,
                        walker_steps=float(n), signature=job_signature(c, d, h, g, smoke))
            _save(p, meta, dict(A1=A1, A2=A2, zhist=zh, nonfinite=np.int64(0)))
        return c, m["name"], h, gs, time.perf_counter() - t0, 0.0, ""
    g = gs[0]
    p = out_path(out, d, c, h, g)
    if os.path.exists(p):
        return c, m["name"], h, gs, 0.0, 0.0, "exists (skipped)"
    seed = seed_of(c, d, h, g)
    T, burn = settings(c, m, smoke)
    rng = np.random.default_rng(seed)
    if c == "em_unbiased":
        x0 = np.array(EM_UNBIASED_X0)
        flat, scheme = 0, GF.EM
        n_trace = len(x0)
    else:
        x0 = rng.uniform(-1.5, 1.5, RV.NW)
        flat, scheme = 1, (GF.EM if c == "em_flat" else GF.MALA)
        n_trace = RV.N_TRACE if g == 0 else 1
    tr_every = trace_every(c, m, h, g)
    y0 = GF.initial_y(x0, rng.standard_normal(len(x0)), m)
    n = int(round((T + burn) / h))
    nburn = int(round(burn / h))
    plan = acf_plan(m, h) if c == "em_flat" else None
    n_acf = RV.N_ACF if c == "em_flat" else 0
    rec = RV.ACF_REC if (c == "em_flat" and g == 0) else 0
    r = GF.chain(m, scheme, x0, y0, n, nburn, h, flat, seed, trace_every=tr_every, n_trace=n_trace, n_acf=n_acf, acf=plan,
                 acf_rec=rec)
    wall = time.perf_counter() - t0
    walker_steps = float(len(x0)) * n
    meta = dict(base, group=g, seed=seed, h=h, T=T, burn=burn, n_steps=n, n_burn=nburn, n_walkers=len(x0), flat_bias=flat,
                trace_every=tr_every, wall_s=wall, walker_steps=walker_steps,
                ns_per_walker_step=wall / (len(x0) * n) * 1e9,
                acf_windows=list(GF.ACF_WINDOWS), acf_half_width=GF.ACF_HALF_WIDTH, n_acf=n_acf,
                acf_tau_windows=({str(k): v for k, v in plan["tau_windows"].items()} if plan else None),
                acf_horizon_t=(plan.get("horizon_t") if plan else None),
                zhist_lim=GF.ZH_LIM, zhist_nb=GF.ZH_NB, nb_fine=GF.NB_FINE, nb_prod=GF.NB_PROD, acc_keys=list(GF.ACC_KEYS),
                acf_keys=list(GF.ACF_KEYS), side_changes_rule=f"hysteresis x < -{GF.HYST_X} <-> x > {GF.HYST_X}, moves at "
                                                             f"steps >= burn (GF module docstring)",
                side_changes_T=float(T), signature=job_signature(c, d, h, g, smoke))
    arrays = dict(A1=r["A1"], A2=r["A2"], n_acc=r["n_acc"], n_prop=r["n_prop"], n_reflect=r["n_reflect"], max_dx=r["max_dx"],
                  nonfinite=np.int64(r["nonfinite"]), crossings=r["crossings"].astype(np.int32), zhist=r["zhist"],
                  traces_x=r["traces_x"].astype(np.float32), traces_y=r["traces_y"].astype(np.float32),
                  X_final=r["X"], Y_final=r["Y"], side_changes=r["side_changes"].astype(np.int32))
    if n_acf:
        arrays.update(acf0=r["acf0"], acf1=r["acf1"], lags0=r["lags0"], lags1=r["lags1"], acf_stride=r["acf_stride"],
                      acf_nsamples=r["acf_nsamples"])
        if rec:
            arrays.update(acf_rec0=r["acf_rec0"][:, :RV.ACF_REC_WALKERS].astype(np.float32),
                          acf_rec1=r["acf_rec1"][:, :RV.ACF_REC_WALKERS].astype(np.float32))
    note = ""
    if not e["new"] and not smoke and frozen_record_exists(m["name"], c, h):
        meta["replay"] = replay_compare(arrays, meta, RV.out_path(OUT_FROZEN, m["name"], c, h, g))
        note = "replay bitwise_equal=" + str(meta["replay"]["bitwise_equal"])
    _save(p, meta, arrays)
    return c, m["name"], h, gs, wall, walker_steps, note


def warmup():
    """Compile every kernel specialisation once in the parent; the harness kernels are @njit(cache=True), so the pool
    workers (forkserver children on Python 3.14, or fork) load them from NUMBA_CACHE_DIR instead of recompiling."""
    for e in REGISTRY:
        m = e["model"]
        for sch in (GF.EM, GF.MALA):
            x0 = np.array([-0.5, 0.01])
            GF.chain(m, sch, x0, GF.initial_y(x0, np.zeros(2), m), 5, 0, 1e-4, 1, 1, n_acf=1, acf=GF.acf_plan(m, 1e-4),
                     acf_rec=2)


def check_out(out, smoke):
    """The registry rule (prereg section 1.3): the new dynamics only with --out results/final_campaign/expA/validation;
    results/mechanism/validation never; --smoke only outside both."""
    rp = os.path.realpath(out)
    for forbidden in (OUT_FROZEN,):
        f = os.path.realpath(forbidden)
        if rp == f or rp.startswith(f + os.sep):
            raise SystemExit(f"refused: --out {out} is the frozen mechanism gate directory")
    e = os.path.realpath(OUT_EXPA)
    if smoke:
        if rp == e or rp.startswith(e + os.sep):
            raise SystemExit("refused: --smoke needs --out outside results/final_campaign/expA/validation")
    elif rp != e:
        raise SystemExit(f"refused: the Experiment A dynamics are accepted only with --out {os.path.relpath(OUT_EXPA, ROOT)}"
                         f" (got {out}); tiny test runs use --smoke --out <scratch>")


def _ledger_row(out, row):
    """One row of <out>/ledger.csv (flock-protected append, fsynced; expA_run_common.append_csv).  If the append fails
    (e.g. ENOSPC) the full row is printed to stdout as 'LEDGER WRITE FAILED ... row {json}' -- so the compute stays
    recoverable from the run log -- and the error is re-raised (drive_pool then starts no further job)."""
    try:
        C.append_csv(os.path.join(out, "ledger.csv"), LEDGER_FIELDS, [row])
    except Exception as e:
        print(f"LEDGER WRITE FAILED ({type(e).__name__}: {e}) for {os.path.join(out, 'ledger.csv')}; row "
              f"{json.dumps(row, default=str, sort_keys=True)}", flush=True)
        raise


def _pause_logger(out):
    """Disk-pause log: stdout and <out>/disk_pause.log (prereg section 3: every pause is logged)."""
    def log(msg):
        print(msg, flush=True)
        try:
            os.makedirs(out, exist_ok=True)
            with open(os.path.join(out, "disk_pause.log"), "a") as fh:
                fh.write(msg + "\n")
        except OSError as e:                                     # the stdout copy stays the record
            print(f"(disk_pause.log not writable: {e!r})", flush=True)
    return log


def drive_pool(items, submit, on_done, workers, disk_path="/", log=None, executor_factory=None):
    """Run ``items`` with at most ``workers`` jobs in flight, submitting LAZILY (a new job starts only when a slot is
    free).  Before EVERY job start: the prereg section 3 disk pause, expA_run_common.wait_for_disk(disk_path) -- no new
    job starts while fewer than 2 GB are free, resume at >= 3 GB, every poll logged through ``log``; during a pause the
    finished jobs keep being collected (ledgers stay current); a pause longer than 24 h stops the driver (stopping rule
    (iv)).  ``submit(executor, item)`` -> future; ``on_done(item, future)`` records a finished job (if it raises, e.g. an
    ENOSPC on the ledger, or the pool breaks, no further job is started; running jobs are still collected).
    Returns dict(n_items, n_started, n_done, stopped = None or the reason)."""
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
    from concurrent.futures.process import BrokenProcessPool
    log = log or (lambda m: print(m, flush=True))
    queue, pending = list(items), {}
    st = dict(n_items=len(queue), n_started=0, n_done=0, stopped=None)

    def stop(reason):
        if st["stopped"] is None:
            st["stopped"] = reason
            log(f"{C.utc()} STOP: {reason}; no further job is started")

    def collect(done):
        for f in done:
            it = pending.pop(f)
            st["n_done"] += 1
            if isinstance(f.exception(), BrokenProcessPool):
                stop(f"the worker pool broke ({f.exception()!r})")
            try:
                on_done(it, f)
            except Exception as e:  # noqa: BLE001  (logged for EVERY item, not only the first: compute stays traceable)
                log(f"{C.utc()} RECORDING FAILED for item {it!r}: {type(e).__name__}: {e}")
                stop(f"recording a finished job failed ({type(e).__name__}: {e}); item {it!r}")

    class _Stopped(Exception):
        pass

    def pause_sleep(s):
        if pending:
            done, _ = wait(list(pending), timeout=s, return_when=FIRST_COMPLETED)
            collect(done)
        else:
            time.sleep(s)
        if st["stopped"] is not None:
            raise _Stopped()                     # leave the disk pause: nothing will be started any more

    factory = executor_factory or (lambda: ProcessPoolExecutor(max_workers=workers))
    with factory() as ex:
        while queue or pending:
            while queue and len(pending) < workers and st["stopped"] is None:
                try:
                    C.wait_for_disk(disk_path, log=log, sleep=pause_sleep)
                except C.DiskPauseTimeout as e:
                    stop(f"disk pause did not clear ({e}; prereg stopping rule iv)")
                    break
                except _Stopped:
                    break
                if st["stopped"] is not None:
                    break
                if len(pending) >= workers:
                    continue
                it = queue.pop(0)
                try:
                    fut = submit(ex, it)
                except BrokenProcessPool as e:
                    queue.insert(0, it)
                    stop(f"the worker pool broke ({e!r})")
                    break
                pending[fut] = it
                st["n_started"] += 1
            if st["stopped"] is not None:
                queue = []
            if pending:
                done, _ = wait(list(pending), return_when=FIRST_COMPLETED)
                collect(done)
    return st


def _check_h(h):
    try:
        RV.h_index(h)
    except ValueError as e:
        raise SystemExit(f"refused: {e}")


def _pin(cpus):
    lo, _, hi = cpus.partition("-")
    s = set(range(int(lo), int(hi or lo) + 1))
    os.sched_setaffinity(0, s)
    return s


def cmd_run(a):
    if a.cpus:
        a.workers = min(a.workers, len(_pin(a.cpus)))
    for h in a.h:
        _check_h(h)
    bad = [d for d in a.dynamics if d not in BY_NAME] + [c for c in a.chains if c not in CHAINS]
    if bad:
        raise SystemExit(f"unknown dynamics/chains {bad} (registry: {list(BY_NAME)})")
    out = os.path.abspath(a.out) if a.out else OUT_EXPA
    check_out(out, a.smoke)
    groups = a.groups if a.groups is not None else (1 if a.smoke else RV.G)
    if not a.smoke and groups != RV.G:
        raise SystemExit(f"--groups {groups}: the gate design is {RV.G} groups; other values only with --smoke")
    allj = jobs(a.dynamics, a.h, a.chains, groups, a.smoke)
    pending, stale, n_complete = [], [], 0
    for j in allj:
        st = job_states(j, out, a.smoke)
        bad_g = [g for g, s in st.items() if s == "stale"]
        miss = [g for g, s in st.items() if s == "missing"]
        n_complete += sum(s == "complete" for s in st.values())
        if bad_g:
            stale.append((j, bad_g))
        if miss:
            pending.append((j[0], j[1], j[2], miss))
    pending.sort(key=lambda j: -cost_walker_steps(j, a.smoke))
    tot = sum(cost_walker_steps(j, a.smoke) for j in pending)
    core_h = sum(cost_walker_steps(j, a.smoke) * NS_PER_WALKER_STEP.get(j[0], 0.0) for j in pending) * 1e-9 / 3600
    core_h += sum(len(j[3]) * EXACT_CORE_H_PER_GROUP for j in pending if j[0].startswith("exact"))
    print(f"{len(allj)} jobs ({n_complete} group files complete), {len(pending)} pending -> {out}; {tot:.3g} walker-steps "
          f"(~{core_h:.2f} core-h at the 2026-10-11 single-core timings {NS_PER_WALKER_STEP} ns/walker-step)", flush=True)
    for j, gs in stale:
        print(f"STALE (refused, never overwritten): {BY_INDEX[j[1]]['name']} {RV.tag(j[0], j[2])} groups {gs}: the stored "
              f"signature differs from this job's -- move the files away to re-run", flush=True)
    if a.dry_run:
        for j in pending:
            c, d, h, gs = j
            print(c, BY_INDEX[d]["name"], f"(index {d})", h, gs, f"seeds {[seed_of(c, d, h, g) for g in gs]}",
                  f"{cost_walker_steps(j, a.smoke):.3g}")
        return 0
    if not pending:
        print("nothing to run" + (" (stale files refused)" if stale else ""), flush=True)
        return 1 if stale else 0
    warmup()
    t0 = time.time()
    failed = []
    submitted = {}

    def submit(ex, j):
        submitted[id(j)] = time.time()
        return ex.submit(run_job, j, out, a.smoke)

    def on_done(j, f):
        row = dict(finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), chain=j[0],
                   dynamics=BY_INDEX[j[1]]["name"], dynamics_index=j[1], h=j[2], groups=" ".join(map(str, j[3])))
        k = len(failed) + on_done.n_ok + 1
        try:
            c, name, h, gs, w, ws, note = f.result()
        except Exception as e:                           # report, keep the other jobs running; a rerun resumes it
            failed.append(j)
            w_used = time.time() - submitted.get(id(j), time.time())   # compute spent before the failure
            print(f"[{k}/{len(pending)}] FAILED {row['dynamics']} {RV.tag(j[0], j[2])} groups {j[3]}: {e!r} "
                  f"(~{w_used:.0f} s spent)", flush=True)
            _ledger_row(out, dict(row, status=f"FAILED: {e!r}"[:300], wall_s=round(w_used, 3), walker_steps=0.0,
                                  note="wall = submit-to-failure time (upper bound of the compute spent)"))
            return
        on_done.n_ok += 1
        print(f"[{k}/{len(pending)}] {name} {RV.tag(c, h)} g{gs[0] if len(gs) == 1 else 'all'}: {w:.0f} s {note} "
              f"(elapsed {time.time() - t0:.0f} s)", flush=True)
        _ledger_row(out, dict(row, status="complete", wall_s=round(w, 3), walker_steps=ws, note=note))
    on_done.n_ok = 0

    st = drive_pool(pending, submit, on_done, a.workers, disk_path=a.disk_path, log=_pause_logger(out))
    if st["stopped"]:
        print(f"STOPPED: {st['stopped']}; {st['n_items'] - st['n_started']} job(s) not started; re-run to resume",
              flush=True)
        return 3 if "disk pause" in st["stopped"] else 1
    print(f"done ({len(failed)} failed)" if failed else "done", flush=True)
    return 1 if (failed or stale) else 0


# ---------------------------------------------------------------------------------------------------------------
# V5 ABF-bias smoke
# ---------------------------------------------------------------------------------------------------------------
def _expA_cfg():
    return json.load(open(CFG_EXPA))


def smoke_cells(h):
    """New cells at every h; shift_lam1 too at an h where the frozen record has no ABF smoke for 'shift'."""
    frozen_hook = AV.load_abf_smoke(os.path.join(OUT_FROZEN, "abf_smoke.json"))
    cells = [v["name"] for v in _expA_cfg()["variants"] if v["name"] != "shift_lam1"]
    if not frozen_hook.get(LAM1, {}).get(f"{h:g}"):
        cells = ["shift_lam1"] + cells
    return cells


def hook_name(cell):
    return LAM1 if cell == "shift_lam1" else cell


def cell_cfg(cell, h):
    """The engine cfg of an Experiment A cell at step h: from configs/final_campaign/cells/expA/<cell>.json when it
    exists (engine_cfg + model; every key but h must equal the cfg built from expA.json), else built from expA.json."""
    import gateway_family_numba as E
    P = _expA_cfg()
    v = next(v for v in P["variants"] if v["name"] == cell)
    cfg = dict(P["engine_cfg"])
    cfg.update(h=float(h), variant=v["variant"], alpha=v["alpha"], kappa=v["kappa"], lam=v["lam"])
    src = "configs/final_campaign/expA.json"
    path = os.path.join(CELL_DIR, cell + ".json")
    if os.path.exists(path):
        cc = json.load(open(path))
        ccfg = dict(cc.get("engine_cfg") or cc.get("cfg") or {})
        if isinstance(cc.get("model"), dict):
            ccfg.update(cc["model"])
        try:
            a_, b_ = E.full_cfg(dict(ccfg, h=float(h))), E.full_cfg(cfg)
        except Exception as e:                                   # noqa: BLE001
            raise SystemExit(f"refused: {os.path.relpath(path, ROOT)} does not give a valid engine cfg: {e}")
        diff = sorted(k for k in set(a_) | set(b_) if a_.get(k) != b_.get(k))
        if diff:
            raise SystemExit(f"refused: {os.path.relpath(path, ROOT)} disagrees with configs/final_campaign/expA.json on "
                             f"{diff}")
        src = os.path.relpath(path, ROOT)
    return cfg, src


def smoke_root(out, h):
    return os.path.join(out, "abf_smoke", f"h{h:g}", f"T{SMOKE_T:g}")


def smoke_jobs(h):
    B = SMOKE_N0 * SMOKE_T / h
    Bi = int(round(B))
    if abs(B - Bi) > 1e-9 * B or Bi % SMOKE_N:
        raise SystemExit(f"smoke budget {B} is not an integer multiple of N {SMOKE_N}")
    seeds = [int(s) for s in _expA_cfg()["abf_smoke_seeds"]]
    J = []
    for cell in smoke_cells(h):
        cfg, src = cell_cfg(cell, h)
        for s in seeds:
            for method in ("abf", "fr"):
                J.append(dict(experiment="expA_abf_smoke", variant=cell, N=SMOKE_N, seed=s, method=method,
                              n_steps=Bi // SMOKE_N, cfg=cfg, cfg_source=src, checkpoint_every_s=600.0))
    return J


def collect_smoke(out, h, res=None, disk_path="/"):
    """scripts/mechanism/abf_smoke.py record() of the preregistered smoke runs -> <res>/abf_smoke.json (merged over h).

    Only a file that is a COMPLETE result of the expected job -- E.is_complete(path, run_cells.job_signature(E, job)):
    cfg, method, N, seed, n_steps, save grid AND engine sha256 -- is filed.  A cell is filed only when all its runs
    (2 seeds x 2 arms) are; otherwise its entry for this h is REMOVED from the hook (missing runs, a STALE file of a
    different run, or no file at all), so AV.gate_v5 reads PENDING, never a record of another run.  Files in a cell
    directory that are no preregistered smoke job are ignored and reported.  Returns (hook, issues): issues lists every
    stale / unexpected file (the caller exits non-zero)."""
    ABFS = _load("mech_abf_smoke", os.path.join("scripts", "mechanism", "abf_smoke.py"))
    RC = C.rc()
    import gateway_family_numba as E
    res = res or out
    hook_path = os.path.join(res, "abf_smoke.json")
    hook = json.load(open(hook_path)) if os.path.exists(hook_path) else {}
    root = smoke_root(out, h)
    by_cell = {}
    for j in smoke_jobs(h):
        by_cell.setdefault(j["variant"], []).append(j)
    issues = []
    for cell, js in by_cell.items():
        recs, stale, missing = [], [], []
        expected = set()
        for j in js:
            pth = RC.out_path(root, j)
            expected.add(os.path.abspath(pth))
            if not os.path.exists(pth):
                missing.append(f"s{j['seed']}_{j['method']}")
                continue
            if not E.is_complete(pth, RC.job_signature(E, j)):
                stale.append(pth)
                continue
            hh, r = ABFS.record(pth)
            if abs(hh - h) > 1e-12 * h or r["seed"] != j["seed"] or r["method"] != j["method"]:
                stale.append(pth)                                # (cannot happen for a signature match)
                continue
            r["file"] = os.path.relpath(pth, ROOT)
            r["sha256"] = sha256_file(pth)
            recs.append(r)
        others = sorted(p_ for p_ in glob.glob(os.path.join(root, cell, f"N{SMOKE_N}", "*.npz"))
                        if not p_.endswith(".ckpt.npz") and ".tmp" not in p_ and os.path.abspath(p_) not in expected)
        for p_ in others:
            issues.append(f"{p_}: not a preregistered smoke job of {cell} (ignored)")
            print(f"{cell:13s} h {h:g}: IGNORED {os.path.relpath(p_, ROOT)} (not a preregistered smoke job)", flush=True)
        for p_ in stale:
            issues.append(f"{p_}: STALE (not a complete result of the expected smoke job; never filed, never overwritten)")
        if len(recs) == len(js):
            hook.setdefault(hook_name(cell), {})[f"{h:g}"] = recs
            print(f"{cell:13s} h {h:g}: " + "; ".join(
                f"s{r['seed']} {r['method']}: finite {r['finite']} |G|max {r['max_abs_Gamma_visited']:.2f} vs 2x|F'| "
                f"{2 * r['max_abs_Fp_visited']:.2f} {'PASS' if r['gate_pass'] else 'FAIL'}" for r in recs), flush=True)
            continue
        # anything short of the full preregistered set must never reach AV.abf_smoke_gate (it would read FAIL instead of
        # PENDING) -- and an older entry for this h is removed
        hook.get(hook_name(cell), {}).pop(f"{h:g}", None)
        if not hook.get(hook_name(cell), True):
            hook.pop(hook_name(cell), None)
        why = []
        if missing:
            why.append(f"missing {missing}")
        if stale:
            why.append(f"STALE {[os.path.relpath(p_, ROOT) for p_ in stale]}")
        print(f"{cell:13s} h {h:g}: NOT FILED ({len(recs)}/{len(js)} valid runs; {'; '.join(why)}) -- V5 stays PENDING",
              flush=True)
    C.wait_for_disk(disk_path, log=_pause_logger(res))
    os.makedirs(res, exist_ok=True)
    tmp = hook_path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(hook, fh, indent=1)
    os.replace(tmp, hook_path)
    print("wrote", hook_path, flush=True)
    return hook, issues


def cmd_abf_smoke(a):
    if a.cpus:
        a.workers = min(a.workers, len(_pin(a.cpus)))
    _check_h(a.h)
    out = os.path.abspath(a.out) if a.out else OUT_EXPA
    check_out(out, a.smoke)
    if a.collect_only:
        _, issues = collect_smoke(out, a.h, disk_path=a.disk_path)
        return 1 if issues else 0
    try:
        sha = C.check_engine()                      # file AND imported module == the frozen engine (69087f2c...)
    except C.EngineMismatch as e:
        print(f"REFUSED: {e}", flush=True)
        return 2
    RC = C.rc()
    import gateway_family_numba as E
    root = smoke_root(out, a.h)
    J = smoke_jobs(a.h)
    states = [RC.job_state(E, RC.out_path(root, j), j) for j in J]
    todo = [j for j, s in zip(J, states) if s in ("new", "resume")]
    stale = [j for j, s in zip(J, states) if s == "stale"]
    est = sum(j["N"] * j["n_steps"] for j in todo) * 0.09e-6 / 3600
    print(f"ABF-bias smoke (V5): {len(J)} runs ({sum(s == 'complete' for s in states)} complete, {len(todo)} to run, "
          f"{len(stale)} STALE refused), est. {est:.2f} core-h -> {root}; engine sha256 {sha}", flush=True)
    for j, s in zip(J, states):
        print(f"    {j['variant']:13s} N {j['N']} seed {j['seed']} {j['method']:3s} n_steps {j['n_steps']} [{s}] cfg from "
              f"{j['cfg_source']}", flush=True)
    if a.dry_run:
        return 0
    n_fail = len(stale)
    stopped = None
    if todo:
        RC.warm_cache()
        submitted = {}

        def submit(ex, j):
            # expA_run_common.run_timed: importable by name in a forkserver / spawn / fork worker; re-checks the engine
            # sha256 inside the worker, then scripts/mechanism/run_cells.run_one (gateway_family_numba.run_job:
            # signature-checked, resumable, never overwrites a complete result of a different run); never raises
            submitted[id(j)] = time.time()
            return ex.submit(C.run_timed, j, root)

        def on_done(j, f):
            nonlocal n_fail
            try:
                o = f.result()
            except Exception as e:  # noqa: BLE001  (a dead worker / broken pool: FAILED, never complete)
                n_fail += 1
                # wall = submit-to-failure time (a finite upper bound of the compute spent; never NaN -- the production
                # driver's projection gate sums this column)
                w_used = time.time() - submitted.get(id(j), time.time())
                print(f"{j['variant']} seed {j['seed']} {j['method']}: FAILED (worker {type(e).__name__}: {e}; "
                      f"~{w_used:.0f} s spent)", flush=True)
                _ledger_row(out, dict(finished_utc=C.utc(), chain="abf_smoke", dynamics=j["variant"], dynamics_index="",
                                      h=a.h, groups=f"s{j['seed']}_{j['method']}",
                                      status=f"FAILED: worker {type(e).__name__}: {e}"[:300], wall_s=round(w_used, 3),
                                      walker_steps=0.0, note="V5 ABF-bias smoke; wall = submit-to-failure time (upper "
                                                             "bound of the compute spent)"))
                return
            r = o["row"]
            ok = r["status"].startswith("complete")
            n_fail += not ok
            print(f"{r['variant']} seed {r['seed']} {r['method']}: {r['status']} {o['elapsed_s']:.0f} s", flush=True)
            _ledger_row(out, dict(finished_utc=r["finished_utc"], chain="abf_smoke", dynamics=r["variant"],
                                  dynamics_index="", h=a.h, groups=f"s{r['seed']}_{r['method']}", status=r["status"][:300],
                                  wall_s=round(o["elapsed_s"], 3), walker_steps=max(0, r["n_force_evals"]),
                                  note=f"V5 ABF-bias smoke; engine wall {r['wall_s']:.1f} s; earlier interrupted "
                                       f"sessions {o.get('prior_sessions_s', 0.0):.0f} s (not in wall_s)"))
            C.append_csv(os.path.join(root, "run_provenance.csv"), C.PROV_FIELDS, [o["prov"]])

        st = drive_pool(todo, submit, on_done, a.workers, disk_path=a.disk_path, log=_pause_logger(out))
        stopped = st["stopped"]
        if stopped:
            print(f"STOPPED: {stopped}; {st['n_items'] - st['n_started']} run(s) not started; re-run to resume",
                  flush=True)
    _, issues = collect_smoke(out, a.h, disk_path=a.disk_path)
    if stopped:
        return 3 if "disk pause" in stopped else 1
    return 1 if (n_fail or issues) else 0


# ---------------------------------------------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------------------------------------------
def _design_issues_fn():
    """AV.design_issues with the Experiment A design (seed_of, settings) in place of RV -- the SAME code object."""
    shim = types.SimpleNamespace(G=RV.G, NW=RV.NW, N_EXACT_FLAT=RV.N_EXACT_FLAT, N_EXACT_UNB=RV.N_EXACT_UNB,
                                 seed_of=seed_of, settings=settings)
    f = AV.design_issues
    return types.FunctionType(f.__code__, dict(f.__globals__, RV=shim), "design_issues_expA", f.__defaults__,
                              f.__closure__)


design_issues = _design_issues_fn()


def extra_design_issues(recs, chain, m, h):
    """Experiment A design fields AV.design_issues does not check (new dynamics): trace stride, ACF recorder and
    horizon, presence of side_changes."""
    iss = []
    for r in recs:
        mt = r["meta"]
        g = int(mt.get("group", -1))
        pre = f"g{g:02d}: "
        if chain.startswith("exact"):
            continue
        if mt.get("trace_every") != trace_every(chain, m, h, g):
            iss.append(pre + f"trace_every {mt.get('trace_every')} != {trace_every(chain, m, h, g)}")
        if "side_changes" not in r:
            iss.append(pre + "no side_changes (pre-extension harness)")
        if chain == "em_flat":
            p = acf_plan(m, h)
            for key in ("lags0", "lags1"):
                if key not in r or not np.array_equal(r[key], p[key]):
                    iss.append(pre + f"{key} differ from the Experiment A ACF plan")
            if "acf_stride" not in r or not np.array_equal(r["acf_stride"], p["stride"]):
                iss.append(pre + "acf_stride differs from the plan")
            if mt.get("n_acf") != RV.N_ACF:
                iss.append(pre + f"n_acf {mt.get('n_acf')} != {RV.N_ACF}")
    return iss


def jk_ci(est, se, dof, level=0.95):
    from scipy.stats import t as _t
    if dof <= 0 or not np.isfinite(se):
        return [float("nan"), float("nan")]
    q = float(_t.ppf(0.5 + level / 2, dof))
    return [float(est - q * se), float(est + q * se)]


def _q(est, se, dof):
    est, se = float(est), float(se)
    return dict(est=est, se=se, ci95=jk_ci(est, se, dof))


def ratio_ci(a, b, dof):
    """a / b for independent estimates a, b = {est, se}: delta method on the log scale, t_dof 95 % CI."""
    from scipy.stats import t as _t
    ea, eb = a["est"], b["est"]
    if not (np.isfinite(ea) and np.isfinite(eb) and ea > 0 and eb > 0):
        return dict(est=float("nan"), ci95=[float("nan")] * 2, se_log=float("nan"))
    sl = math.sqrt((a["se"] / ea) ** 2 + (b["se"] / eb) ** 2)
    q = float(_t.ppf(0.975, dof)) if dof > 0 else float("nan")
    r = ea / eb
    return dict(est=r, se_log=sl, ci95=[r * math.exp(-q * sl), r * math.exp(q * sl)])


def mc1(recs, m, h):
    """MC1 coupled conditional relaxation from the em_flat lag sums (module docstring)."""
    recs = [r for r in recs if "acf0" in r]
    if not recs:
        return None
    stride = np.asarray(recs[0]["acf_stride"])
    lags = (np.asarray(recs[0]["lags0"]), np.asarray(recs[0]["lags1"]))
    for r in recs[1:]:
        if not (np.array_equal(r["acf_stride"], stride) and np.array_equal(r["lags0"], lags[0])
                and np.array_equal(r["lags1"], lags[1])):
            return dict(error="ACF plans differ between groups")
    G = len(recs)
    dof = G - 1
    lam, kappa = float(m["lam"]), float(m["kappa"])
    out = dict(h=float(h), lam=lam, n_groups=G, stride_steps=stride.tolist(), stride_t=(stride * h).tolist(),
               horizon_t=[float(lags[L][-1] * stride[L] * h) for L in (0, 1)], cuts_t=list(MC1_CUTS), windows={})
    for w, c in enumerate(GF.ACF_WINDOWS):
        lev = next((L for L in (0, 1) if lags[L][-1] * stride[L] * h >= max(MC1_CUTS) - 1e-12), None)
        xw = np.linspace(c - GF.ACF_HALF_WIDTH, c + GF.ACF_HALF_WIDTH, 401)
        mp2w = GF.dm_centre(xw, m) ** 2
        mp2 = float(GF.dm_centre(np.array([c]), m)[0] ** 2)
        e = dict(x0=c, mprime2=mp2, tau_pred_frozen=1.0 / (lam * kappa ** 2), tau_pred_coupled=1.0 / (kappa ** 2 * (lam + mp2)),
                 level=lev)
        if lev is None:
            e["note"] = f"no ACF level reaches the largest cut {max(MC1_CUTS)} t.u."
            out["windows"][f"{c:g}"] = e
            continue
        key = f"acf{lev}"
        t = lags[lev] * stride[lev] * h
        sdf_win = float(np.mean(np.sqrt(GF.cond_var_f(xw, m))))
        has_f = sdf_win > AV.SDF_MIN_ACF
        e.update(t=t.tolist(), first_lag_t=float(t[1]) if t.size > 1 else float("nan"), sd_f_window_mean=sdf_win,
                 f_reported=bool(has_f), n_origin_samples=float(AV._sum(recs, key)[w, 0, 0]),
                 rho_pred_frozen=np.mean(np.exp(-t[:, None] * lam * kappa ** 2), -1).tolist(),
                 rho_pred_coupled=np.mean(np.exp(-t[:, None] * kappa ** 2 * (lam + mp2w[None, :])), -1).tolist())
        if not has_f:
            e["f_note"] = f"exact sd(f|x) in the window {sdf_win:.3g} < {AV.SDF_MIN_ACF}: z_f is rounding noise, not reported"
        for q in (("y", "f") if has_f else ("y",)):
            def rho_fn(rs, q=q):
                return GF.acf_from_sums(AV._sum(rs, key))["rho_" + q][w]
            rho, rse = AV.jack(rho_fn, recs)
            e[f"rho_{q}"] = rho.tolist()
            e[f"rho_{q}_se"] = rse.tolist()
            with np.errstate(invalid="ignore"):
                est, se = AV.jack(lambda rs: np.array(GF.tau_efold(t, rho_fn(rs))), recs)
                e[f"tau_e_{q}"] = _q(est, se, dof)
                for cut in MC1_CUTS:
                    est, se = AV.jack(lambda rs, cut=cut: np.array(GF.tau_integrated(t, rho_fn(rs), cut)), recs)
                    e[f"int_{cut:g}_{q}"] = _q(est, se, dof)
        out["windows"][f"{c:g}"] = e
    return out


def side_changes_from_traces(tr):
    """Hysteresis side changes per walker from a (samples, walkers) trace (expA_scout_transport.side_changes' rule)."""
    lab = np.zeros(tr.shape[1], dtype=np.int64)
    n = np.zeros(tr.shape[1], dtype=np.int64)
    for row in tr:
        new = np.where(row > GF.HYST_X, 1, np.where(row < -GF.HYST_X, -1, 0))
        n += (new != 0) & (lab != 0) & (new != lab)
        lab = np.where(new != 0, new, lab)
    return n


def mc2(recs, h):
    """MC2 gate transport: side changes per walker per t.u. (jackknife over groups) + the group-0 trace cross-check."""
    out = dict(h=float(h))
    rs_ = [r for r in recs if "side_changes" in r]
    if rs_:
        G = len(rs_)
        fn = lambda rs: np.array(np.mean([r["side_changes"].mean() / r["meta"]["T"] for r in rs]))  # noqa: E731
        est, se = AV.jack(fn, rs_)
        out.update(n_groups=G, rate=_q(est, se, G - 1), walkers=int(sum(r["side_changes"].size for r in rs_)),
                   T=float(rs_[0]["meta"]["T"]), total_side_changes=int(sum(int(r["side_changes"].sum()) for r in rs_)),
                   frac_walkers_ge1=float(np.mean(np.concatenate([r["side_changes"] > 0 for r in rs_]))))
    else:
        out["rate"] = None
        out["note"] = "no side_changes in these records (pre-extension harness): per-step MC2 not available"
    g0 = [r for r in recs if int(r["meta"].get("group", -1)) == 0 and r["traces_x"].shape[0] > 2]
    if g0:
        r = g0[0]
        te = int(r["meta"]["trace_every"])
        k0 = int(math.ceil(int(r["meta"]["n_burn"]) / te))
        tr = np.asarray(r["traces_x"], float)[k0:]
        n = side_changes_from_traces(tr)
        span = (tr.shape[0] - 1) * te * h
        out["trace_crosscheck_g0"] = dict(walkers=int(tr.shape[1]), dt_t=te * h, span_t=span,
                                          rate=float(n.mean() / span) if span > 0 else float("nan"),
                                          note="descriptive: 8 walkers sampled every trace_every steps (misses "
                                               "excursions shorter than the stride)")
    return out


def v1_holm(per_h_entries):
    """The prereg V1 family at one h: the shifted-fibre residual-mean 3-se clauses of the new dynamics, two-sided
    t_dof p, Holm at FWER 5 %.  per_h_entries: list of dict(dynamics, index, z, dof).  Returns the entries with p and
    holm_rejected."""
    fam = [dict(f) for f in per_h_entries if np.isfinite(f["z"]) and f["dof"] > 0]
    pv = [AV.t_two_sided(f["z"], f["dof"]) for f in fam]
    for f, p_, r_ in zip(fam, pv, AV.holm_reject(pv, 0.05)):
        f.update(p=p_, holm_rejected=bool(r_))
    return fam


def apply_v1_holm(g, rejected_idx, zs):
    """Replace AV's per-clause |z| <= 3 shifted-fibre mean clause by the Holm decision (other V1 clauses unchanged)."""
    comp = dict(g["V1_components"])
    comp.pop("shift_mean_3se", None)
    comp["shift_mean_holm"] = [None if not np.isfinite(z) else (j not in rejected_idx) for j, z in enumerate(zs)]
    g["V1_unadjusted_3se"] = g["V1"]
    g["V1_components_holm"] = comp
    g["V1"] = AV.status(*comp.values())
    g["V1_strict"] = AV.status(*comp.values(), g["V1_consistency_3se"])
    return g


def select(table, v4):
    """Prereg selection: AV.decide over the four cells {shift, shift_lam4, shift_lam16, shift_lam128}
    (table[dyn][f'{h:g}'] = {core, full, core_strict, full_strict}; v4[dyn] = V4 status): h_A = 2.5e-5 iff all four
    PASS and V4 holds; a FAIL moves ALL four to 1.25e-5 (REFINE_REQUIRED until they are gated there); a FAIL at
    1.25e-5 = UNRESOLVED (Experiment A stops); a V4 FAIL = HARNESS_FAIL (stop and investigate)."""
    sel = AV.decide(table, list(ALL_NAMES), v4)
    sel["table"] = table
    strict = AV.decide(table, list(ALL_NAMES), v4, "core_strict", "full_strict")
    sel["strict_reading"] = dict(verdict=strict["verdict"], h_common=strict["h_common"],
                                 differs=(strict["verdict"] != sel["verdict"] or strict["h_common"] != sel["h_common"]),
                                 note="reported only (the frozen gateway V1 consistency clause |m - closed form| <= 3 se)")
    if sel["verdict"] == "UNRESOLVED":
        sel["verdict"] = "UNRESOLVED (a failure at the refinement h: Experiment A stops, prereg section 1.3)"
    sel["h_A"] = sel.get("h_common")
    if sel["h_A"]:
        B = 2048 * 40 / sel["h_A"]
        sel.update(B=B, T_N={str(N): B * sel["h_A"] / N for N in (512, 128)})
    return sel


def analyse_all(res=OUT_EXPA, abf_path=None, smoke=False, frozen_res=OUT_FROZEN):
    abf_path = abf_path or os.path.join(res, "abf_smoke.json")
    abf = AV.load_abf_smoke(abf_path)
    fsum_path = os.path.join(frozen_res, "summary.json")
    frozen = json.load(open(fsum_path)) if os.path.exists(fsum_path) else None
    S = dict(prereg=["docs/final_campaign/PREREGISTRATION.md section 1.3", "configs/final_campaign/expA_validation.json"],
             generated_utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), res=os.path.relpath(res, ROOT),
             smoke=bool(smoke), provenance=provenance(), candidates=list(H_CANDIDATES), abf_smoke_file=abf_path if abf else None,
             frozen_record=dict(file=os.path.relpath(fsum_path, ROOT), sha256=sha256_file(fsum_path) if frozen else None),
             thresholds=dict(V1=AV.TOL_V1, V1_shift_var=AV.TOL_V1_SHIFT_VAR, V1_shift_region=AV.V1_SHIFT_REGION,
                             V2_D=AV.TOL_F, V2_upper=AV.TOL_F_UP, V3_D=AV.TOL_FP, V3_upper=AV.TOL_FP_UP, V3b=AV.TOL_V3B,
                             V3b_min_exact=AV.V3B_MIN_EXACT, V4_F_upper=AV.TOL_F_UP, V5_reflections=AV.V5_REFL,
                             V5_cross_frac=AV.V5_CROSS_FRAC, V5_abf_factor=AV.ABF_SMOKE_FACTOR, z_3se=AV.Z_3SE),
             multiplicity=dict(rule="prereg section 1.3: V1 shifted-fibre residual-mean 3-se clauses of the new dynamics "
                                    "at one h = ONE Holm family; their V4 3-se clauses (MALA + exact, central variance + "
                                    "residual mean) = ONE Holm family; two-sided t_{G-1} p (16-group jackknife), FWER 5 %; "
                                    "lam = 1 keeps its frozen record (G-A2) / frozen clauses at the refinement h",
                               V1_holm={}, V4_holm=None),
             registry={e["name"]: dict(index=e["index"], lam=e["model"]["lam"], new=e["new"]) for e in REGISTRY},
             dynamics={})
    table, v4, v4_family = {}, {}, []
    v1_fam = {f"{h:g}": [] for h in H_CANDIDATES}
    for e in REGISTRY:
        if not e["new"]:
            continue
        name, d, m = e["name"], e["index"], e["model"]
        D = dict(model={k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in m.items()},
                 index=d, per_h={}, design={})

        def get(chain, h, name=name, m=m, d=d, D=D):
            tg = RV.tag(chain, h) if not chain.startswith("exact") else chain
            recs = AV.load(res, name, tg)
            if not recs:
                return recs
            iss = design_issues(recs, chain, m, d, h, smoke=smoke) + extra_design_issues(recs, chain, m, h)
            if iss:
                D["design"][tg] = iss[:20] + ([f"... {len(iss) - 20} more"] if len(iss) > 20 else [])
                if not smoke:
                    print(f"{name:13s} {tg}: EXCLUDED, does not match the design: {iss[:3]}", flush=True)
                    return []
            return recs

        o_mala = AV.analyse_flat(get("mala_flat", H_MALA), m, None)
        o_ex = AV.analyse_flat(get("exact_flat", None), m, None)
        s_mala, c_mala = AV.gate_exact_sampler(o_mala, m)
        s_ex, c_ex = AV.gate_exact_sampler(o_ex, m)
        zm, ze = AV.exact_sampler_z(o_mala, m), AV.exact_sampler_z(o_ex, m)

        def _non3se(o_, comp):
            if not o_.get("n_groups"):
                return AV.NO_DATA
            return AV.status(*[c for k, c in comp.items() if not k.endswith("_3se")])
        D["V4"] = dict(status=AV.combine(_non3se(o_mala, c_mala), _non3se(o_ex, c_ex)), mala=s_mala, exact=s_ex,
                       mala_components=c_mala, exact_components=c_ex, mala_z=zm, exact_z=ze,
                       status_unadjusted_3se=AV.combine(s_mala, s_ex))
        for samp, o_, zz in (("mala", o_mala, zm), ("exact", o_ex, ze)):
            if o_.get("n_groups"):
                for clause, vals in zz.items():
                    for j, zval in enumerate(vals):
                        v4_family.append(dict(dynamics=name, sampler=samp, clause=clause, index=j, z=float(zval),
                                              dof=int(o_["n_groups"] - 1)))
        D["mala"], D["exact_flat"] = o_mala, o_ex
        D["exact_unbiased_readout"] = AV.analyse_exact_unbiased(get("exact_unbiased", None), m)
        table[name] = {}
        for h in H_CANDIDATES:
            hk = f"{h:g}"
            if not AV.load(res, name, f"em_flat_h{hk}") and not AV.load(res, name, f"em_unbiased_h{hk}"):
                continue
            recs = get("em_flat", h)
            unb = AV.analyse_unbiased_em(get("em_unbiased", h))
            o = AV.analyse_flat(recs, m, h)
            g = AV.gates_em(o, m)
            g.update(AV.gate_v5(o, unb, abf.get(name, {}).get(hk, [])))
            if o.get("n_groups"):
                for j, z in enumerate(o["mean_centre"]["z"]):
                    v1_fam[hk].append(dict(dynamics=name, index=j, z=float(z), dof=int(o["n_groups"] - 1)))
            D["per_h"][hk] = dict(em_flat=o, em_unbiased=unb, gates=g, acf=AV.analyse_acf(recs, m, h), mc1=mc1(recs, m, h),
                                  mc2=mc2(recs, h))
        S["dynamics"][name] = D

    # V4: one Holm family over the new dynamics
    fam = [f for f in v4_family if np.isfinite(f["z"]) and f["dof"] > 0]
    pv = [AV.t_two_sided(f["z"], f["dof"]) for f in fam]
    rej = AV.holm_reject(pv, 0.05)
    for f, p_, r_ in zip(fam, pv, rej):
        f.update(p=p_, holm_rejected=bool(r_))
    rejected_dyn = sorted({f["dynamics"] for f in fam if f["holm_rejected"]})
    S["multiplicity"]["V4_holm"] = dict(n_tests=len(fam), n_rejected=int(sum(rej)), rejected_dynamics=rejected_dyn,
                                        min_p=min(pv) if pv else None, family=fam)
    for name in NEW_NAMES:
        V = S["dynamics"][name]["V4"]
        if name in rejected_dyn and V["status"] != AV.NO_DATA:
            V["status"] = AV.FAIL
        V["holm_rejected_clauses"] = [f for f in fam if f["dynamics"] == name and f["holm_rejected"]]
        v4[name] = V["status"]

    # V1: one Holm family per h over the new dynamics; per-h status
    for hk, entries in v1_fam.items():
        famh = v1_holm(entries)
        if entries:
            S["multiplicity"]["V1_holm"][hk] = dict(n_tests=len(famh), n_rejected=int(sum(f["holm_rejected"] for f in famh)),
                                                    family=famh)
        for name in NEW_NAMES:
            ph = S["dynamics"][name]["per_h"].get(hk)
            if not ph:
                continue
            g = ph["gates"]
            g["V4"] = v4[name]
            if ph["em_flat"].get("n_groups"):
                rj = {f["index"] for f in famh if f["dynamics"] == name and f["holm_rejected"]}
                apply_v1_holm(g, rj, ph["em_flat"]["mean_centre"]["z"])
            core = AV.combine(g["V1"], g["V2"], g["V3"], g["V3b"], g["V5_core"])
            full = AV.combine(core, g["V5"])
            core_s = AV.combine(g["V1_strict"], g["V2"], g["V3"], g["V3b"], g["V5_core"]) if ph["em_flat"].get("n_groups") else core
            table[name][hk] = dict(core=core, full=full, core_strict=core_s, full_strict=AV.combine(core_s, g["V5"]))
            ph.update(status_core=core, status=full, status_strict=table[name][hk]["full_strict"])
            print(f"{name:13s} h {hk}: V1 {g['V1']} (unadj. {g.get('V1_unadjusted_3se', '-')}) V2 {g['V2']} V3 {g['V3']} "
                  f"V3b {g['V3b']} V5 {g['V5_core']}/abf {g['V5_abf_smoke']} -> per-h {core}, full {full} | V4 {v4[name]}",
                  flush=True)

    # lam = 1: G-A2 record (frozen), and the re-gate at the refinement h (frozen AV clauses, unchanged)
    D1 = dict(index=BY_NAME[LAM1]["index"], per_h={}, design={})
    m1 = BY_NAME[LAM1]["model"]
    table[LAM1] = {}
    if frozen is not None and LAM1 in frozen.get("dynamics", {}):
        fr = frozen["dynamics"][LAM1]
        v4[LAM1] = fr["V4"]["status"]
        D1["G_A2"] = dict(source=os.path.relpath(fsum_path, ROOT), frozen_selection_verdict=frozen["selection"]["verdict"],
                          frozen_h_common=frozen["selection"].get("h_common"), V4=fr["V4"]["status"], per_h={})
        for hk, ph in fr.get("per_h", {}).items():
            table[LAM1][hk] = dict(core=ph["status_core"], full=ph["status"], core_strict=ph["status_core"],
                                   full_strict=ph["status_strict"])
            D1["G_A2"]["per_h"][hk] = dict(status=ph["status"], status_core=ph["status_core"],
                                          gates={k: ph["gates"][k] for k in ("V1", "V2", "V3", "V3b", "V4", "V5_core",
                                                                             "V5_abf_smoke", "V5")})
    else:
        v4[LAM1] = AV.NO_DATA
        D1["G_A2"] = dict(source=None, note=f"{fsum_path} missing")
    for h in H_CANDIDATES:
        hk = f"{h:g}"
        if hk in table[LAM1]:
            continue                                                    # the frozen record governs this h
        raw = AV.load(res, LAM1, f"em_flat_h{hk}")
        if not raw:
            continue
        recs = raw
        iss = design_issues(recs, "em_flat", m1, D1["index"], h, smoke=smoke) + extra_design_issues(recs, "em_flat", m1, h)
        if iss:
            D1["design"][f"em_flat_h{hk}"] = iss[:20]
            recs = recs if smoke else []
        unb_recs = AV.load(res, LAM1, f"em_unbiased_h{hk}")
        iss_u = (design_issues(unb_recs, "em_unbiased", m1, D1["index"], h, smoke=smoke)
                 + extra_design_issues(unb_recs, "em_unbiased", m1, h)) if unb_recs else []
        if iss_u:
            D1["design"][f"em_unbiased_h{hk}"] = iss_u[:20]
            unb_recs = unb_recs if smoke else []
        o = AV.analyse_flat(recs, m1, h)
        unb = AV.analyse_unbiased_em(unb_recs)
        g = AV.gates_em(o, m1)
        g.update(AV.gate_v5(o, unb, abf.get(LAM1, {}).get(hk, [])))
        g["V4"] = v4[LAM1]
        core = AV.combine(g["V1"], g["V2"], g["V3"], g["V3b"], g["V5_core"])
        full = AV.combine(core, g["V5"])
        core_s = AV.combine(g["V1_strict"], g["V2"], g["V3"], g["V3b"], g["V5_core"]) if o.get("n_groups") else core
        table[LAM1][hk] = dict(core=core, full=full, core_strict=core_s, full_strict=AV.combine(core_s, g["V5"]))
        D1["per_h"][hk] = dict(em_flat=o, em_unbiased=unb, gates=g, status_core=core, status=full,
                               rule="re-gate at the refinement h with the frozen AV clauses (|z| <= 3 per clause, as the "
                                    "G-A2 record)")
        print(f"{LAM1:13s} h {hk} (re-gate): V1 {g['V1']} V2 {g['V2']} V3 {g['V3']} V3b {g['V3b']} V5 {g['V5_core']}/abf "
              f"{g['V5_abf_smoke']} -> per-h {core}, full {full}", flush=True)
    S["dynamics"][LAM1] = D1

    # selection (AV.decide over the four cells)
    sel = select(table, v4)
    S["selection"] = sel

    # manipulation checks MC1 / MC2, per lam, at every h with data; the lam = 1 replay's inertness enters the selection
    S["mc"] = manipulation_checks(S, res, smoke, frozen_res)
    S["inertness_replay"] = S["mc"]["inertness_replay"]
    apply_inertness(sel, S["inertness_replay"])
    if smoke:
        sel["verdict"] = "SMOKE -- NOT A GATE RESULT (" + sel["verdict"] + ")"
    print("SELECTION:", sel["verdict"], "h_A", sel.get("h_A"), flush=True)
    for n_ in sel.get("notes", []):
        print("  note:", n_, flush=True)
    return S


def apply_inertness(sel, inert):
    """The full-length lam = 1 replay (lam1_inputs) in the selection: any replayed G-A2 chain that is NOT bitwise
    identical to its frozen file means the harness extension is not inert -> HARNESS_FAIL (stop and investigate, prereg
    section 1.3 "any failure stops Experiment A"; h_A None); INCOMPLETE / NOT_RUN -> a note (the selection itself rests
    on the frozen G-A2 record and the new dynamics; only MC2 at lam = 1 is excluded)."""
    notes = sel.setdefault("notes", [])
    st = {hk: (v or {}).get("status") for hk, v in inert.items()}
    sel["inertness_replay"] = st
    fails = sorted(hk for hk, v in st.items() if v == "FAIL")
    if fails:
        sel["verdict_before_inertness"] = sel["verdict"]
        sel["verdict"] = (f"HARNESS_FAIL (inertness_replay FAIL: the lam = 1 replay of the G-A2 em_flat chains is NOT "
                          f"bitwise identical to the frozen record at h {', '.join(fails)} -- the harness extension is "
                          f"not inert; stop and investigate; before this check: {sel['verdict']})")
        sel["h_A"] = sel["h_common"] = None
        sel.pop("B", None)
        sel.pop("T_N", None)
        notes.append(f"inertness_replay: FAIL at h {', '.join(fails)} (see mc.inertness_replay)")
    for hk, v in sorted(st.items()):
        if v in ("INCOMPLETE", "NOT_RUN"):
            notes.append(f"inertness_replay at h {hk}: {v} -- the full-length lam = 1 replay is not complete; MC2 at "
                         f"lam = 1 (and every MC2 ratio to lam = 1) is not reported; the selection rests on the frozen "
                         f"G-A2 record and the new dynamics")
        elif v == "PASS":
            notes.append(f"inertness_replay at h {hk}: PASS (16/16 replayed G-A2 em_flat chains bitwise identical)")
    return sel


def lam1_inputs(res, frozen_res, h, smoke):
    """The lam = 1 inputs of MC1 / MC2 at step h, each checked against its design before use:

    frozen  the G-A2 em_flat files in ``frozen_res`` (AV.design_issues, the frozen design, 16 groups): the MC1 source
            when they conform (no side_changes: MC2 only from their group-0 traces);
    replay  (an h WITH a frozen record) <res>/shift/em_flat_h<h>: must conform to the design (design_issues +
            extra_design_issues: groups 0-15, seeds, T, burn, n_steps, trace stride, ACF plan, side_changes) AND every
            file must carry meta.replay.bitwise_equal = True (bitwise equal to its frozen counterpart).
            inertness_replay: PASS (all 16 conform, all bitwise) -> the MC2 source; FAIL (any file NOT bitwise equal:
            the harness extension is not inert at full length -> selection HARNESS_FAIL, Experiment A stops);
            INCOMPLETE (fewer files, design deviations or a missing comparison: MC2 at lam = 1 EXCLUDED, never computed
            from a partial set); NOT_RUN (no file: MC2 at lam = 1 only from the frozen traces);
    regate  (an h WITHOUT a frozen record) <res>/shift/em_flat_h<h> (the re-gate chains): the MC1 and MC2 source when
            they conform (16 groups), otherwise excluded.
    In smoke mode deviations are recorded but the files are used (never a gate result)."""
    hk = f"{h:g}"
    m1 = BY_NAME[LAM1]["model"]
    d1 = BY_NAME[LAM1]["index"]
    out = dict(mc1_src=None, mc1_source=None, mc2_src=None, mc2_trace_only_src=None, notes=[], design={})
    frozen_recs = AV.load(frozen_res, LAM1, f"em_flat_h{hk}")
    frozen_ok = bool(frozen_recs) and not AV.design_issues(frozen_recs, "em_flat", m1, d1, h)
    if frozen_recs and not frozen_ok:
        out["notes"].append(f"frozen em_flat_h{hk} does not conform to the frozen design: not used")
    raw = AV.load(res, LAM1, f"em_flat_h{hk}")
    iss = (design_issues(raw, "em_flat", m1, d1, h, smoke=smoke) + extra_design_issues(raw, "em_flat", m1, h)) if raw else []
    if iss:
        out["design"][f"em_flat_h{hk}"] = iss[:20] + ([f"... {len(iss) - 20} more"] if len(iss) > 20 else [])
    use_raw = bool(raw) and (smoke or not iss)
    if frozen_ok:
        out["mc1_src"] = frozen_recs
        out["mc1_source"] = os.path.relpath(os.path.join(frozen_res, LAM1, f"em_flat_h{hk}"), ROOT)
        out["mc2_trace_only_src"] = frozen_recs
        reps = [r["meta"].get("replay") for r in raw]
        n_eq = sum(bool(x and x.get("bitwise_equal")) for x in reps)
        n_cmp = sum(x is not None for x in reps)
        mism = {int(r["meta"].get("group", -1)): (x.get("mismatched"), x.get("meta_mismatched"))
                for r, x in zip(raw, reps) if x and not x.get("bitwise_equal")}
        if smoke:
            status = "N/A (smoke)"
        elif not raw:
            status = "NOT_RUN"
        elif mism:
            status = "FAIL"
        elif iss or n_cmp != len(raw) or n_eq != RV.G:
            status = "INCOMPLETE"
        else:
            status = "PASS"
        out["inertness_replay"] = dict(status=status, n_files=len(raw), n_expected=RV.G, n_compared=n_cmp,
                                       n_bitwise_equal=n_eq, mismatches=mism, design_issues=iss[:20],
                                       rule="every one of the 16 replayed G-A2 em_flat chains conforms to the design and "
                                            "is bitwise identical to its frozen file (meta.replay)")
        if status == "PASS" or (smoke and raw):
            out["mc2_src"] = raw
        elif raw:
            out["notes"].append(f"MC2 at lam = 1 EXCLUDED: replay {status} ({n_eq}/{RV.G} bitwise-equal files, "
                                f"{len(iss)} design deviations); only the frozen group-0 trace cross-check is reported")
    else:
        out["inertness_replay"] = dict(status="N/A (no frozen G-A2 record at this h)")
        if use_raw:
            out["mc1_src"] = out["mc2_src"] = raw
            out["mc1_source"] = os.path.relpath(os.path.join(res, LAM1, f"em_flat_h{hk}"), ROOT)
        elif raw:
            out["notes"].append(f"lam = 1 re-gate chains at h {hk} EXCLUDED from MC1 / MC2 (design deviations: {iss[:3]})")
    return out


def manipulation_checks(S, res, smoke, frozen_res=OUT_FROZEN):
    """MC1 / MC2 tables per lam (lam = 1: lam1_inputs -- MC1 from the frozen raw files, MC2 from the replay
    side_changes, both design-checked; the replay's bitwise comparison gives inertness_replay)."""
    out = dict(cuts_t=list(MC1_CUTS), per_h={}, inertness_replay={})
    m1 = BY_NAME[LAM1]["model"]
    for h in H_CANDIDATES:
        hk = f"{h:g}"
        rows = {}
        L1 = lam1_inputs(res, frozen_res, h, smoke)
        if L1["mc1_src"] or L1["mc2_src"] or L1["mc2_trace_only_src"] or L1["notes"]:
            out["inertness_replay"][hk] = L1["inertness_replay"]
        if L1["mc1_src"] or L1["mc2_src"] or L1["mc2_trace_only_src"]:
            mc2_1 = mc2(L1["mc2_src"], h) if L1["mc2_src"] else (mc2(L1["mc2_trace_only_src"], h)
                                                                 if L1["mc2_trace_only_src"] else None)
            if mc2_1 is not None and not L1["mc2_src"]:
                mc2_1["rate"] = None
                mc2_1["note_lam1"] = ("per-step MC2 at lam = 1 not available (" + (L1["notes"][-1] if L1["notes"] else
                                      "no replay") + "): only the group-0 trace cross-check of the frozen chains")
            rows[LAM1] = dict(lam=1.0, mc1=mc1(L1["mc1_src"], m1, h) if L1["mc1_src"] else None, mc2=mc2_1,
                              mc1_source=L1["mc1_source"], replay=L1["inertness_replay"], notes=L1["notes"],
                              design=L1["design"])
        for name in NEW_NAMES:
            ph = S["dynamics"].get(name, {}).get("per_h", {}).get(hk)
            if ph:
                rows[name] = dict(lam=lam_of(name), mc1=ph.get("mc1"), mc2=ph.get("mc2"))
        if not rows:
            continue
        # ratios to lam = 1 (independent groups)
        base = rows.get(LAM1)
        for name, r in rows.items():
            if name == LAM1 or base is None:
                continue
            rat = {}
            if r["mc2"] and r["mc2"].get("rate") and base["mc2"] and base["mc2"].get("rate"):
                rat["mc2_rate"] = ratio_ci(r["mc2"]["rate"], base["mc2"]["rate"], min(r["mc2"]["n_groups"],
                                                                                    base["mc2"]["n_groups"]) - 1)
            if r["mc1"] and base["mc1"] and "windows" in r["mc1"] and "windows" in base["mc1"]:
                for wk, ew in r["mc1"]["windows"].items():
                    eb = base["mc1"]["windows"].get(wk, {})
                    for key in [k for k in ew if (k.startswith("tau_e_") or k.startswith("int_")) and isinstance(ew[k], dict)]:
                        if isinstance(eb.get(key), dict):
                            rat[f"mc1_{wk}_{key}"] = ratio_ci(ew[key], eb[key], min(r["mc1"]["n_groups"],
                                                                                   base["mc1"]["n_groups"]) - 1)
            r["ratio_to_lam1"] = rat
        out["per_h"][hk] = rows
    hA = S.get("selection", {}).get("h_A")
    out["h_shown"] = f"{hA:g}" if hA else (next(iter(out["per_h"]), None))
    return out


def _fmt(q, nd=4):
    if not isinstance(q, dict) or q.get("est") is None or not np.isfinite(q.get("est", np.nan)):
        return "-"
    ci = q.get("ci95") or [np.nan, np.nan]
    return f"{q['est']:.{nd}g} [{ci[0]:.{nd}g}, {ci[1]:.{nd}g}]"


def markdown(S):
    L = [f"# Experiment A numerical validation (G-A1 / G-A2, MC1, MC2)", "",
         f"Generated {S['generated_utc']} UTC from `{S['res']}` (commit {S['provenance']['git_commit'][:10]}"
         f"{', dirty' if S['provenance'].get('git_dirty') else ''}). Prereg: docs/final_campaign/PREREGISTRATION.md section 1.3."
         + (" **SMOKE -- NOT A GATE RESULT.**" if S.get("smoke") else ""), "",
         f"**Selection:** {S['selection']['verdict']}; h_A = {S['selection'].get('h_A')}", ""]
    L += [f"* {n_}" for n_ in S["selection"].get("notes", [])] + ([""] if S["selection"].get("notes") else [])
    L += [
         "| dynamics | lam | h | V1 | V2 | V3 | V3b | V4 | V5 core | V5 ABF smoke | status |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name in ALL_NAMES:
        D = S["dynamics"].get(name, {})
        if name == LAM1:
            for hk, ph in (D.get("G_A2", {}).get("per_h") or {}).items():
                g = ph["gates"]
                L.append(f"| shift (G-A2 record) | 1 | {hk} | {g['V1']} | {g['V2']} | {g['V3']} | {g['V3b']} | {g['V4']} | "
                         f"{g['V5_core']} | {g['V5_abf_smoke']} | {ph['status']} |")
        for hk, ph in D.get("per_h", {}).items():
            g = ph["gates"]
            L.append(f"| {name} | {lam_of(name):g} | {hk} | {g['V1']} | {g['V2']} | {g['V3']} | {g['V3b']} | {g.get('V4', '-')} | "
                     f"{g['V5_core']} | {g['V5_abf_smoke']} | {ph['status']} |")
    mu = S["multiplicity"]
    L += ["", f"V1 Holm families (per h): " + "; ".join(f"h {hk}: {v['n_rejected']}/{v['n_tests']} rejected"
                                                     for hk, v in mu["V1_holm"].items()),
          f"V4 Holm family: {mu['V4_holm']['n_rejected']}/{mu['V4_holm']['n_tests']} rejected" if mu.get("V4_holm") else ""]
    mc = S.get("mc", {})
    hk = mc.get("h_shown")
    rows = mc.get("per_h", {}).get(hk, {}) if hk else {}
    if rows:
        L += ["", f"## MC1 coupled conditional relaxation (h {hk}; jackknife 95 % CI)", "",
              "| lam | x0 | q | 1/e time | int to 0.3125 | int to 1.25 | int to 5 | pred frozen | pred coupled |",
              "|---|---|---|---|---|---|---|---|---|"]
        for name, r in sorted(rows.items(), key=lambda kv: kv[1]["lam"]):
            if not r.get("mc1") or "windows" not in r["mc1"]:
                continue
            for wk, e in r["mc1"]["windows"].items():
                for q in ("y", "f"):
                    if f"tau_e_{q}" not in e:
                        continue
                    L.append(f"| {r['lam']:g} | {wk} | {q} | {_fmt(e[f'tau_e_{q}'])} | {_fmt(e['int_0.3125_' + q])} | "
                             f"{_fmt(e['int_1.25_' + q])} | {_fmt(e['int_5_' + q])} | {e['tau_pred_frozen']:.4g} | "
                             f"{e['tau_pred_coupled']:.4g} |")
        L += ["", f"## MC2 gate transport (h {hk}): side changes x < -0.5 <-> x > 0.5 per walker per t.u.", "",
              "| lam | rate [95 % CI] | ratio to lam 1 [95 % CI] | trace cross-check (g0) |", "|---|---|---|---|"]
        for name, r in sorted(rows.items(), key=lambda kv: kv[1]["lam"]):
            m2 = r.get("mc2") or {}
            rt = (r.get("ratio_to_lam1") or {}).get("mc2_rate")
            tc = (m2.get("trace_crosscheck_g0") or {}).get("rate")
            L.append(f"| {r['lam']:g} | {_fmt(m2.get('rate'))} | {_fmt(rt) if rt else '-'} | "
                     f"{tc:.4g} |" if tc is not None else f"| {r['lam']:g} | {_fmt(m2.get('rate'))} | "
                     f"{_fmt(rt) if rt else '-'} | - |")
        rep = rows.get(LAM1, {}).get("replay")
        if rep and "n_files" in rep:
            L += ["", f"lam = 1 replay of the G-A2 em_flat chains (inertness_replay {rep['status']}): "
                      f"{rep['n_bitwise_equal']}/{rep['n_expected']} files bitwise equal to the frozen record "
                      f"({rep['n_files']} present)."]
        for n_ in rows.get(LAM1, {}).get("notes") or []:
            L.append(f"* lam = 1: {n_}")
    return "\n".join(L) + "\n"


def _json_default(v):
    return AV._json_default(v)


def figures(S, fig_dir):
    """A4 (MC1: rho_y, rho_f at x0 = -1, -0.21, 0 for every lam with the frozen-x and coupled-OU predictions; tau vs
    lam) and A4b (MC2: side-change rate vs lam and its ratio to lam = 1)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
    mc = S.get("mc", {})
    hk = mc.get("h_shown")
    rows = mc.get("per_h", {}).get(hk, {}) if hk else {}
    os.makedirs(fig_dir, exist_ok=True)
    lams = sorted(r["lam"] for r in rows.values())
    cmap = plt.get_cmap("viridis")
    col = {lam: cmap(i / max(1, len(lams) - 1)) for i, lam in enumerate(lams)}
    tag = "SMOKE -- NOT A GATE RESULT -- " if S.get("smoke") else ""
    # A4
    fig, axs = plt.subplots(2, 4, figsize=(17, 7.5))
    wins = [f"{c:g}" for c in GF.ACF_WINDOWS]
    for qi, q in enumerate(("y", "f")):
        for wi, wk in enumerate(wins):
            ax = axs[qi, wi]
            any_ = False
            for name, r in sorted(rows.items(), key=lambda kv: kv[1]["lam"]):
                e = ((r.get("mc1") or {}).get("windows") or {}).get(wk)
                if not e or f"rho_{q}" not in e:
                    continue
                t = np.array(e["t"])
                rho = np.array(e[f"rho_{q}"], float)
                se = np.nan_to_num(np.array(e[f"rho_{q}_se"], float))
                k = t > 0
                ax.plot(t[k], rho[k], color=col[r["lam"]], lw=1.1, label=f"lam {r['lam']:g}")
                ax.fill_between(t[k], (rho - 2 * se)[k], (rho + 2 * se)[k], color=col[r["lam"]], alpha=0.2, lw=0)
                ax.plot(t[k], np.array(e["rho_pred_frozen"])[k], color=col[r["lam"]], lw=0.8, ls="--")
                ax.plot(t[k], np.array(e["rho_pred_coupled"])[k], color=col[r["lam"]], lw=0.8, ls=":")
                any_ = True
            ax.set_xscale("log")
            ax.set_ylim(-0.1, 1.05)
            ax.axhline(math.exp(-1), color="#999", lw=0.5)
            ax.set_xlabel("lag t (t.u.)")
            ax.set_ylabel(f"rho_{q}")
            ax.set_title(f"({'ab'[qi]}{wi + 1}) rho_{q} at x0 = {wk} +- {GF.ACF_HALF_WIDTH:g}"
                         + ("" if any_ else "  [not reported]"), loc="left")
            if any_ and wi == 0:
                ax.legend(fontsize=6, title="solid: measured (+-2 se)\n--: frozen-x exp(-lam t)\n:: coupled exp(-(lam+m'^2) t)",
                          title_fontsize=6)
        ax = axs[qi, 3]
        mk = {"-1": "o", "-0.21": "s", "0": "^"}
        lg = np.geomspace(0.7, 200, 50)
        for wk in wins:
            xs, ys, lo, hi, pf, pc = [], [], [], [], [], []
            for name, r in rows.items():
                e = ((r.get("mc1") or {}).get("windows") or {}).get(wk)
                if not e or not isinstance(e.get(f"tau_e_{q}"), dict):
                    continue
                v = e[f"tau_e_{q}"]
                xs.append(r["lam"])
                ys.append(v["est"])
                lo.append(v["est"] - v["ci95"][0] if np.isfinite(v["ci95"][0]) else 0)
                hi.append(v["ci95"][1] - v["est"] if np.isfinite(v["ci95"][1]) else 0)
                pc.append(e["mprime2"])
            if xs:
                o = np.argsort(xs)
                ax.errorbar(np.array(xs)[o], np.array(ys)[o], yerr=[np.array(lo)[o], np.array(hi)[o]], marker=mk[wk], ms=4,
                            lw=0.8, label=f"x0 = {wk}")
                ax.plot(lg, 1.0 / (lg + pc[0]), lw=0.6, ls=":", color="#555")
        ax.plot(lg, 1.0 / lg, lw=0.8, ls="--", color="k", label="frozen-x 1/lam")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("lam")
        ax.set_ylabel(f"1/e time of rho_{q} (t.u.)")
        ax.set_title(f"({'ab'[qi]}4) 1/e time vs lam (dotted: coupled 1/(lam + m'(x0)^2))", loc="left")
        ax.legend(fontsize=6)
    fig.suptitle(f"{tag}A4 -- MC1 coupled conditional relaxation (flat-bias em_flat chains, h {hk})", fontsize=10,
                 color="#c0392b" if S.get("smoke") else "k")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(fig_dir, f"expA_A4_mc1_conditional_relaxation.{ext}"), dpi=150)
    plt.close(fig)
    # A4b
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    xs, ys, lo, hi, xr, yr, rlo, rhi, tcx, tcy = [], [], [], [], [], [], [], [], [], []
    for name, r in rows.items():
        m2 = r.get("mc2") or {}
        if m2.get("rate"):
            v = m2["rate"]
            xs.append(r["lam"])
            ys.append(v["est"])
            lo.append(v["est"] - v["ci95"][0] if np.isfinite(v["ci95"][0]) else 0)
            hi.append(v["ci95"][1] - v["est"] if np.isfinite(v["ci95"][1]) else 0)
        if (m2.get("trace_crosscheck_g0") or {}).get("rate") is not None:
            tcx.append(r["lam"])
            tcy.append(m2["trace_crosscheck_g0"]["rate"])
        rt = (r.get("ratio_to_lam1") or {}).get("mc2_rate")
        if rt and np.isfinite(rt["est"]):
            xr.append(r["lam"])
            yr.append(rt["est"])
            rlo.append(rt["est"] - rt["ci95"][0])
            rhi.append(rt["ci95"][1] - rt["est"])
    ax = axs[0]
    if xs:
        o = np.argsort(xs)
        ax.errorbar(np.array(xs)[o], np.array(ys)[o], yerr=[np.array(lo)[o], np.array(hi)[o]], marker="o", lw=1,
                    label="per-step counter (jackknife 95 % CI)")
    if tcx:
        ax.plot(tcx, tcy, "x", color="#888", label="group-0 traces (8 walkers, descriptive)")
    if os.path.exists(SCOUT_SUMMARY):
        sc = json.load(open(SCOUT_SUMMARY))
        ax.plot([r["lam"] for r in sc], [r["rate"] for r in sc], "+", color="#c0392b", ms=7, label="pre-freeze scout")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("lam")
    ax.set_ylabel("side changes per walker per t.u.")
    ax.set_title(f"(a) gate transport x<-0.5 <-> x>0.5 (h {hk})", loc="left")
    ax.legend(fontsize=6)
    ax = axs[1]
    if xr:
        o = np.argsort(xr)
        ax.errorbar(np.array(xr)[o], np.array(yr)[o], yerr=[np.array(rlo)[o], np.array(rhi)[o]], marker="s", lw=1)
    ax.axhline(1, color="k", lw=0.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("lam")
    ax.set_ylabel("rate / rate(lam = 1)")
    ax.set_title("(b) ratio to lam = 1 (log-scale delta-method 95 % CI)", loc="left")
    fig.suptitle(f"{tag}A4b -- MC2 gate transport", fontsize=10, color="#c0392b" if S.get("smoke") else "k")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(fig_dir, f"expA_A4b_mc2_gate_transport.{ext}"), dpi=150)
    plt.close(fig)


def cmd_analyze(a):
    res = os.path.abspath(a.res) if a.res else OUT_EXPA
    fig = os.path.abspath(a.fig) if a.fig else FIG_DEFAULT
    if os.path.realpath(res).startswith(os.path.realpath(OUT_FROZEN)):
        raise SystemExit("refused: --res is the frozen mechanism gate directory (it is read, never written)")
    if a.smoke and (os.path.realpath(res) == os.path.realpath(OUT_EXPA)
                    or (not a.no_fig and os.path.realpath(fig) == os.path.realpath(FIG_DEFAULT))):
        raise SystemExit("--smoke needs --res and --fig outside the production directories")
    frozen = os.path.abspath(a.frozen_res) if a.frozen_res else OUT_FROZEN
    S = analyse_all(res, a.abf_smoke, a.smoke, frozen_res=frozen)
    C.wait_for_disk(a.disk_path, log=_pause_logger(res))            # prereg section 3: no new saves below 2 GB free
    os.makedirs(res, exist_ok=True)
    for fn, txt in (("summary.json", json.dumps(S, indent=1, default=_json_default)), ("summary.md", markdown(S))):
        tmp = os.path.join(res, fn + ".tmp")
        with open(tmp, "w") as fh:
            fh.write(txt)
        os.replace(tmp, os.path.join(res, fn))
    print("wrote", os.path.join(res, "summary.json"), "and summary.md", flush=True)
    if not a.no_fig:
        figures(S, fig)
        print("figures ->", fig, flush=True)
    return 0


# ---------------------------------------------------------------------------------------------------------------
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "dry-run"):
        p = sub.add_parser(name)
        p.add_argument("--workers", type=int, default=8)
        p.add_argument("--cpus", type=str, default=None, help="pin the pool, e.g. 64-71 (inherited by the workers)")
        p.add_argument("--h", type=float, nargs="*", default=[H_CANDIDATES[0]])
        p.add_argument("--dynamics", nargs="*", default=list(ALL_NAMES))
        p.add_argument("--chains", nargs="*", default=list(CHAINS))
        p.add_argument("--out", default=None)
        p.add_argument("--groups", type=int, default=None)
        p.add_argument("--smoke", action="store_true")
        p.add_argument("--dry-run", action="store_true")
        p.add_argument("--disk-path", default="/", help=argparse.SUPPRESS)        # file system of the disk pause
    p = sub.add_parser("abf-smoke")
    p.add_argument("--h", type=float, default=H_CANDIDATES[0])
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--cpus", type=str, default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--smoke", action="store_true", help="(tests) write under a scratch --out")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--collect-only", action="store_true")
    p.add_argument("--disk-path", default="/", help=argparse.SUPPRESS)
    p = sub.add_parser("analyze")
    p.add_argument("--res", default=None)
    p.add_argument("--fig", default=None)
    p.add_argument("--no-fig", action="store_true")
    p.add_argument("--abf-smoke", default=None, help="default <res>/abf_smoke.json")
    p.add_argument("--frozen-res", default=None, help="(tests) the frozen G-A2 record directory, default "
                                                      "results/mechanism/validation (read only)")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--disk-path", default="/", help=argparse.SUPPRESS)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.cmd in ("run", "dry-run"):
        if a.cmd == "dry-run":
            a.dry_run = True
        return cmd_run(a)
    if a.cmd == "abf-smoke":
        return cmd_abf_smoke(a)
    return cmd_analyze(a)


if __name__ == "__main__":
    sys.exit(main())
