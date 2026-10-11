"""Shared helpers of the Experiment A analysis tools (docs/final_campaign/PREREGISTRATION.md section 1).

Used by scripts/final_campaign/expA_{analyze,cross_cell,mech,outcome,figures,tables,make_fixtures}.py.  Nothing here changes any
old module: the frozen analysis code (scripts/equal_budget/{eqb_metrics,eqb_family,analyze_ladder}.py,
scripts/mechanism/{cross_cell,mech_analysis}.py) is imported and called, never edited.

Cells.  One cell config per transverse mobility lambda in {1, 4, 16, 128} (shifted fibre, kappa = 1), each in the
mechanism-cell format (scripts/mechanism/make_cell_configs.py: keys cell, results_dir, analysis_dir, fig_dir, model,
engine_cfg, B, N_ladder, seeds, thresholds, ...), default directory configs/final_campaign/cells/expA/*.json.  A
cell is identified by its model (variant 'shift', kappa 1, lam), never by its file name.  A config carrying
'_fixture_of' is a FIXTURE (synthetic, tiny budgets): every output made from it is stamped and never 'complete'.

Output policy (every tool).  An output JSON records an 'input_signature' (sha256 over the inputs' content digests,
the code digests of the tool and of the modules it imports, the analysis settings and the completeness basis) and
'complete' (completeness() below).  Re-running with the SAME signature skips the work ('up to date'; --recompute
rewrites it); an existing COMPLETE output whose signature DIFFERS is never overwritten unless --force, and a forced
overwrite records the signature it replaced.  Outputs are refused inside the frozen trees (results/mechanism,
results/equal_budget_v2, figures/mechanism, figures/equal_budget_v2, configs/).  Writing order: a placeholder JSON
(state 'writing', no signature, complete false) first, then the data files, then the final JSON, so an interrupted
write never leaves a signed or COMPLETE record that does not describe the files on disk.  Every write first waits for
free disk space (PREREGISTRATION section 3: pause below 2 GB, resume above 3 GB, the pause logged; a pause longer
than 24 h stops the job) and removes its temporary file on failure.

complete (completeness()) requires ALL of: the frozen production design (every planned lambda; each cell equal to
scripts/final_campaign/make_cells.py's output from the frozen configs/final_campaign/expA.json; h, B, N ladder,
seeds), no fixture, run-tree freshness verified, every planned (N, method, seed) in every summary, n_boot = 10 000 for
the tool AND every summary, and, per cell, the completeness audit <analysis_dir>/completeness_audit.json with
verdict PASS made on the CURRENT summary and config (sha256) and listing only runs of the frozen engine (full sha256
69087f2c...), and no --provisional flag.  The audit status (verdict, audited summary / config sha256, engines) enters
the input signature, so a provisional output is recomputed once the audit passes.
"""
from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time

for _k in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in (os.path.join(ROOT, "src"), os.path.join(ROOT, "scripts", "equal_budget"),
           os.path.join(ROOT, "scripts", "mechanism"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import eqb_metrics as M  # noqa: E402
import eqb_family as MF  # noqa: E402

LAMS = (1.0, 4.0, 16.0, 128.0)
REF_LAM = 1.0
N_PROD = (512, 128)
SEEDS_PROD = tuple(range(8300, 8332))
H_CANDIDATES = (2.5e-5, 1.25e-5)
ANCHOR_N0, ANCHOR_T0 = 2048, 40.0
N_BOOT = M.N_BOOT                    # 10 000
BOOT_SEED = M.BOOT_SEED              # 20261010
ALPHA = 0.05
EQUIV = (0.90, 1.11)
SMALL_N = 10                         # PREREGISTRATION section 1.6: cells with < 10 complete seed pairs make no claim

CELLS_DIR = os.path.join(ROOT, "configs", "final_campaign", "cells", "expA")
EXPA_CONFIG = os.path.join(ROOT, "configs", "final_campaign", "expA.json")
OUT_ROOT = os.path.join(ROOT, "results", "final_campaign", "expA")
SYN_DIR = os.path.join(OUT_ROOT, "synthesis")
TABLES_DIR = os.path.join(OUT_ROOT, "tables")
LOG_DIR = os.path.join(OUT_ROOT, "logs")
FIG_DIR = os.path.join(ROOT, "figures", "final_campaign", "expA")
ENGINE_PATH = os.path.join(ROOT, "src", "gateway_family_numba.py")
ENGINE_SHA256 = "69087f2c0c7edbe5e076615b9a4425ba7580a6c4537a61854ae6c633df9da4ed"   # frozen engine (PREREG 1.1, 1.3 #8)
ENGINE_SHA256_PREFIX = ENGINE_SHA256[:8]
PREREG = os.path.join(ROOT, "docs", "final_campaign", "PREREGISTRATION.md")
AUDIT_NAME = "completeness_audit.json"
DISK_PAUSE_BELOW = 2 * 1024 ** 3      # PREREGISTRATION section 3: pause below 2 GB free ...
DISK_RESUME_ABOVE = 3 * 1024 ** 3     # ... resume above 3 GB
DISK_POLL_S = 60.0
DISK_MAX_PAUSE_S = 24 * 3600.0        # stopping rule (iv): a pause that does not clear within 24 h
DISK_PAUSE_LOG = os.path.join(LOG_DIR, "analysis", "disk_pauses.log")   # written only when a pause happens
DISK_PAUSE_LOG_ENV = "EXPA_DISK_PAUSE_LOG"   # overrides DISK_PAUSE_LOG (expA_analyze.py: <log-dir>/disk_pauses.log;
                                             # the tests: their temporary directory)
EXP1_SHIFT_SUMMARY = os.path.join(ROOT, "results", "mechanism", "matched_free_energy", "shift", "analysis",
                                  "summary.json")
PROTECTED_DIRS = tuple(os.path.join(ROOT, *p) for p in (
    ("results", "mechanism"), ("results", "equal_budget_v2"), ("figures", "mechanism"),
    ("figures", "equal_budget_v2"), ("configs",)))


class ExpAError(RuntimeError):
    pass


# =========================================================================================== small helpers
def fnum(v):
    """JSON value -> float: None -> NaN, "inf" -> inf (eqb_metrics.json_safe's convention)."""
    if v is None:
        return math.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def lam_key(lam):
    """'1', '4', '16', '128' (JSON key of a lambda)."""
    return f"{float(lam):g}"


def sha256_file(path):
    return M.sha256_file(path)


def digest(obj):
    """sha256 of the canonical JSON of obj."""
    return hashlib.sha256(json.dumps(M.json_safe(obj), sort_keys=True, default=str).encode()).hexdigest()


def rel(path):
    """Repository-relative path when inside the repository, else the absolute path."""
    p = os.path.abspath(path)
    return os.path.relpath(p, ROOT) if p.startswith(ROOT + os.sep) else p


def per_seed(S, N, method, metric):
    """{seed: float} of summary.json per_N[N][method][metric].per_seed ("inf" -> inf, null -> NaN)."""
    b = (((S or {}).get("per_N") or {}).get(str(N)) or {}).get(method) or {}
    b = b.get(metric) or {}
    return {int(k): fnum(v) for k, v in (b.get("per_seed") or {}).items()}


CONTRAST_GROUPS = ("primary", "secondary", "marginal")


def summary_contrast(S, N, metric):
    """The paired FR-vs-ABF contrast block of ``metric`` at N (contrasts[N][group][metric]) or None."""
    c = (((S or {}).get("contrasts") or {}).get(str(N))) or {}
    for g in CONTRAST_GROUPS:
        if metric in (c.get(g) or {}):
            return c[g][metric]
    return None


def complete_seeds(S, N, method):
    st = ((((S or {}).get("status") or {}).get(str(N)) or {}).get("methods") or {}).get(method) or {}
    return sorted(int(s) for s in (st.get("complete") or []))


# =========================================================================================== cells
def cell_model(P):
    m = P.get("model") or {k: (P.get("engine_cfg") or {}).get(k) for k in MF.FAMILY_MODEL_KEYS}
    return dict(variant=m.get("variant"), alpha=m.get("alpha"), kappa=m.get("kappa"), lam=m.get("lam"))


def is_fixture(P):
    return "_fixture_of" in P


def production_problems(P):
    """Differences between a cell config and the frozen production design (empty = production cell)."""
    probs = []
    if is_fixture(P):
        probs.append("fixture config ('_fixture_of')")
    h = float(P["engine_cfg"]["h"])
    if h not in H_CANDIDATES:
        probs.append(f"h {h} is not a candidate {H_CANDIDATES}")
    if int(P["B"]) != int(round(ANCHOR_N0 * ANCHOR_T0 / h)):
        probs.append(f"B {P['B']} != 2048 * 40 / h")
    if [int(n) for n in P["N_ladder"]] != list(N_PROD):
        probs.append(f"N ladder {P['N_ladder']} != {list(N_PROD)}")
    if [int(s) for s in P["seeds"]] != list(SEEDS_PROD):
        probs.append("seeds != 8300-8331")
    if not is_fixture(P):
        probs += generator_problems(P)
    return probs


def _canon(o):
    return json.loads(json.dumps(o, default=str))


def generator_problems(P):
    """Differences between a cell config and scripts/final_campaign/make_cells.py's output regenerated in memory from
    the frozen configs/final_campaign/expA.json (whose sha256 the generator pins): any changed engine knob, threshold,
    grid, location or model field is reported ([] = the cell is exactly the generator's).  The generator is imported,
    never run as a writer."""
    try:
        import make_cells as MK
    except Exception as e:  # noqa: BLE001
        return [f"cannot import the cell generator scripts/final_campaign/make_cells.py ({type(e).__name__}: {e})"]
    probs = []
    want_sha = getattr(MK, "EXPECTED_CONFIG_SHA256", None)
    if P.get("_generated_from_sha256") != want_sha:
        probs.append(f"_generated_from_sha256 {str(P.get('_generated_from_sha256'))[:12]} != the frozen experiment "
                     f"config {str(want_sha)[:12]}")
    try:
        built = MK.build_all()
    except Exception as e:  # noqa: BLE001  (e.g. configs/final_campaign/expA.json no longer the frozen file)
        return probs + [f"the cell generator refused to rebuild the cells ({type(e).__name__}: {e})"]
    want = built.get(P.get("cell"))
    if want is None:
        return probs + [f"the generator has no cell {P.get('cell')!r}"]
    W, Q = _canon(want), _canon(P)
    diff = sorted(k for k in set(W) | set(Q) if W.get(k) != Q.get(k))
    if diff:
        ec = sorted(k for k in set(W.get("engine_cfg") or {}) | set(Q.get("engine_cfg") or {})
                    if (W.get("engine_cfg") or {}).get(k) != (Q.get("engine_cfg") or {}).get(k))
        probs.append(f"cell differs from make_cells.py's output (frozen expA.json) in {diff}"
                     + (f" (engine_cfg keys {ec})" if ec else ""))
    return probs


def load_cells(cells_dir=None):
    """{lam: rec} for the Experiment A cell configs in cells_dir (default configs/final_campaign/cells/expA):
    rec = dict(config, P, lam, name, summary_path, S (or None), why).  Refuses a non-cell file, a model other than
    the shifted fibre at kappa 1, an unplanned lambda and two cells with one lambda.  Missing lambdas are simply
    absent (the callers report them)."""
    cells_dir = os.path.abspath(cells_dir or CELLS_DIR)
    out = {}
    paths = sorted(glob.glob(os.path.join(cells_dir, "*.json")))
    for p in paths:
        try:
            P = json.load(open(p))
        except Exception as e:  # noqa: BLE001
            raise ExpAError(f"unreadable cell config {p}: {e}")
        if not MF.is_cell(P):
            raise ExpAError(f"{p} is not a cell config (keys {MF.CELL_KEYS})")
        m = cell_model(P)
        if m["variant"] != "shift" or fnum(m["kappa"]) != 1.0 or m["alpha"] is not None:
            raise ExpAError(f"{p}: model {m} is not the shifted fibre at kappa = 1")
        lam = fnum(m["lam"])
        if lam not in LAMS:
            raise ExpAError(f"{p}: lambda {lam} is not one of the planned {LAMS}")
        ec = P["engine_cfg"]
        if fnum(ec.get("lam")) != lam or ec.get("variant") != "shift" or fnum(ec.get("kappa")) != 1.0:
            raise ExpAError(f"{p}: engine_cfg model fields disagree with the cell's model {m}")
        if lam in out:
            raise ExpAError(f"two cell configs with lambda {lam:g}: {out[lam]['config']} and {p}")
        sp = os.path.join(MF.cell_analysis_dir(P), "summary.json")
        rec = dict(config=os.path.abspath(p), P=P, lam=lam, name=P["cell"], summary_path=sp, S=None, why=None)
        if os.path.exists(sp):
            try:
                rec["S"] = json.load(open(sp))
            except Exception as e:  # noqa: BLE001
                rec["why"] = f"unreadable summary {sp}: {e}"
        else:
            rec["why"] = f"no summary at {sp}"
        out[lam] = rec
    return out


def common_N(cells):
    """The N ladder shared by every cell (error if they differ)."""
    Ns = None
    for lam, rec in sorted(cells.items()):
        nl = [int(n) for n in rec["P"]["N_ladder"]]
        if Ns is None:
            Ns = nl
        elif nl != Ns:
            raise ExpAError(f"cells have different N ladders ({Ns} vs {nl} at lambda {lam:g})")
    return Ns or []


def design_status(cells):
    """(production, fixture, problems): production = every planned lambda present and every cell the frozen design."""
    probs = []
    for lam in LAMS:
        if lam not in cells:
            probs.append(f"no cell config for lambda {lam:g}")
    fixture = any(is_fixture(r["P"]) for r in cells.values())
    for lam, r in sorted(cells.items()):
        probs += [f"lambda {lam:g}: {x}" for x in production_problems(r["P"])]
    hs = {float(r["P"]["engine_cfg"]["h"]) for r in cells.values()}
    if len(hs) > 1:
        probs.append(f"cells use different h {sorted(hs)} (PREREGISTRATION 1.3: one common h_A)")
    Bs = {int(r["P"]["B"]) for r in cells.values()}
    if len(Bs) > 1:
        probs.append(f"cells use different B {sorted(Bs)}")
    return (not probs), fixture, probs


def summaries_complete(cells):
    """True iff every planned lambda has a summary whose status lists every planned (N, method, seed) complete."""
    if any(lam not in cells for lam in LAMS):
        return False
    for rec in cells.values():
        S, P = rec["S"], rec["P"]
        if S is None:
            return False
        for N in P["N_ladder"]:
            for m in ("abf", "fr"):
                if complete_seeds(S, N, m) != sorted(int(s) for s in P["seeds"]):
                    return False
    return True


# =========================================================================================== summary checks
def summary_problems(cells, verify_runs=True):
    """{lam: (kind, [problems])} for the cells whose summary does not belong to the CURRENT config / scoring code /
    reference (kind 'identity', scripts/mechanism/cross_cell.check_summary) or, with verify_runs, is STALE against the
    run tree (kind 'stale', cross_cell.check_freshness: the complete files on disk are exactly the summary's and every
    per-run cache entry is valid for the file's current content).  Cells without a summary are not listed."""
    import cross_cell as X
    out = {}
    for lam, rec in sorted(cells.items()):
        if rec["S"] is None:
            continue
        p = X.check_summary(rec)
        if p:
            out[lam] = ("identity", p)
            continue
        if verify_runs:
            s = X.check_freshness(rec)
            if s:
                out[lam] = ("stale", s)
    return out


def require_current_summaries(cells, verify_runs=True):
    """Raise ExpAError naming every summary that does not belong to its config or is STALE (summary_problems)."""
    probs = summary_problems(cells, verify_runs)
    if not probs:
        return
    msgs = []
    for lam, (kind, p) in sorted(probs.items()):
        sp = cells[lam]["summary_path"]
        if kind == "identity":
            msgs.append(f"lambda {lam:g}: summary {sp} does not belong to its config: " + "; ".join(p))
        else:
            msgs.append(f"lambda {lam:g}: summary {sp} is STALE ({len(p)} issues, e.g. {p[:3]}): re-run "
                        "analyze_ladder.py on the cell (expA_analyze.py stage 'an')")
    raise ExpAError(" | ".join(msgs))


def summary_n_boot_problems(cells):
    out = []
    for lam, rec in sorted(cells.items()):
        if rec["S"] is not None and rec["S"].get("n_boot") != N_BOOT:
            out.append(f"lambda {lam:g}: summary n_boot {rec['S'].get('n_boot')!r} != {N_BOOT}")
    return out


def audit_path(rec):
    return os.path.join(MF.cell_analysis_dir(rec["P"]), AUDIT_NAME)


def engine_sha_of(key):
    """The engine sha256 of an audit 'engines' key ('<engine> <sha256>') or of a bare sha256 string (None if absent)."""
    parts = str(key or "").split()
    for t in reversed(parts):
        if len(t) == 64 and all(c in "0123456789abcdef" for c in t):
            return t
    return None


def audit_status(rec):
    """Status of one cell's completeness audit (scripts/equal_budget/audit_completeness.py --system gateway_family
    --config <cell>, written to <analysis_dir>/completeness_audit.json): passes iff present, verdict PASS, made on the
    CURRENT summary and config (sha256 equal) and listing only runs of the frozen engine.  'problems' names each
    unmet condition."""
    p = audit_path(rec)
    st = dict(path=rel(p), present=os.path.exists(p), verdict=None, generated_utc=None, n_failures=None,
              failure_codes=[], summary_sha256=None, config_sha256=None, engines=[], current=False, passes=False,
              problems=[])
    if not st["present"]:
        st["problems"].append(f"no completeness audit at {rel(p)} (run expA_analyze.py stage 'audit')")
        return st
    try:
        doc = json.load(open(p))
    except Exception as e:  # noqa: BLE001
        st["problems"].append(f"unreadable completeness audit {rel(p)}: {type(e).__name__}: {e}")
        return st
    sysd = (doc.get("systems") or {}).get(MF.FAMILY) or {}
    st.update(verdict=doc.get("verdict"), generated_utc=doc.get("generated_utc"), n_failures=sysd.get("n_failures"),
              failure_codes=sorted({str(f.get("code")) for f in (sysd.get("failures") or [])}),
              summary_sha256=(sysd.get("summary") or {}).get("sha256"), config_sha256=sysd.get("config_sha256"),
              engines=sorted(((sysd.get("run_files") or {}).get("engines") or {}).keys()))
    if doc.get("verdict") != "PASS" or sysd.get("verdict") != "PASS":
        st["problems"].append(f"completeness audit verdict {doc.get('verdict')!r} ({sysd.get('n_failures')} failure(s): "
                              f"{st['failure_codes']})")
    cur_sum = sha256_file(rec["summary_path"]) if os.path.exists(rec["summary_path"]) else None
    cur_cfg = sha256_file(rec["config"])
    ok_sum = cur_sum is not None and st["summary_sha256"] == cur_sum
    ok_cfg = st["config_sha256"] == cur_cfg
    if not ok_sum:
        st["problems"].append(f"the audit was made on summary sha256 {str(st['summary_sha256'])[:12]}, the current summary "
                              f"is {str(cur_sum)[:12]} (re-run the audit)")
    if not ok_cfg:
        st["problems"].append(f"the audit was made on config sha256 {str(st['config_sha256'])[:12]}, the current config "
                              f"is {cur_cfg[:12]}")
    bad = [e for e in st["engines"] if engine_sha_of(e) != ENGINE_SHA256]
    if bad:
        st["problems"].append(f"runs from an engine other than the frozen sha256 {ENGINE_SHA256[:12]}...: {bad}")
    st["current"] = bool(ok_sum and ok_cfg)
    st["passes"] = not st["problems"]
    return st


def audit_failures(cells):
    """{lam: [problems]} of the cells whose completeness audit does not pass (audit_status), read from the audit
    JSONs on disk (expA_analyze.py uses it when the 'audit' stage is not part of the invocation)."""
    out = {}
    for lam, rec in sorted(cells.items()):
        a = audit_status(rec)
        if not a["passes"]:
            out[lam] = a["problems"]
    return out


def audit_signature(audits):
    """The audit facts that decide 'complete' (stable across re-runs of an unchanged audit: no timestamps)."""
    return {k: dict(verdict=a.get("verdict"), summary_sha256=a.get("summary_sha256"),
                    config_sha256=a.get("config_sha256"), engines=a.get("engines"), passes=a.get("passes"))
            for k, a in sorted(audits.items())}


def completeness(cells, n_boot=None, verify_runs=True, provisional=False, extra_reasons=()):
    """dict(complete, reasons, audits): complete iff no reason (module docstring).  n_boot: the tool's own bootstrap
    size (None: the tool has no bootstrap)."""
    reasons = []
    production, fixture, probs = design_status(cells)
    if fixture:
        reasons.append("FIXTURE cells (synthetic, never production evidence)")
    if not production:
        reasons += [f"design: {p}" for p in probs if "fixture config" not in p]
    if not verify_runs:
        reasons.append("run-tree freshness NOT verified (--no-verify-runs)")
    if not summaries_complete(cells):
        reasons.append("not every planned (lambda, N, method, seed) is complete in the summaries")
    if n_boot is not None and int(n_boot) != N_BOOT:
        reasons.append(f"--n-boot {n_boot} != the preregistered {N_BOOT}")
    reasons += summary_n_boot_problems(cells)
    audits = {}
    for lam, rec in sorted(cells.items()):
        a = audit_status(rec)
        audits[lam_key(lam)] = a
        reasons += [f"lambda {lam:g}: {p}" for p in a["problems"]]
    if provisional:
        reasons.append("--provisional (a completeness audit failed and was overridden: provisional synthesis)")
    reasons += list(extra_reasons)
    return dict(complete=not reasons, reasons=reasons, audits=audits)


def synthesis_input_problems(cells, **docs):
    """Problems when a synthesis document (cross_cell.json, mech.json) was not made from the CURRENT cell configs and
    summaries (sha256 per lambda) or names a lambda that has no cell."""
    probs = []
    keys = {lam_key(l) for l in cells}
    for name, doc in docs.items():
        dc = (doc or {}).get("cells") or {}
        for lk in sorted(set(dc) - keys):
            probs.append(f"{name}: lambda {lk} has no cell config any more")
        for lam, rec in sorted(cells.items()):
            lk = lam_key(lam)
            ci = dc.get(lk)
            if ci is None:
                probs.append(f"{name}: no entry for lambda {lk}")
                continue
            cur = dict(config_sha256=sha256_file(rec["config"]),
                       summary_sha256=(sha256_file(rec["summary_path"]) if os.path.exists(rec["summary_path"]) else None))
            for k, v in cur.items():
                if ci.get(k) != v:
                    probs.append(f"{name}: lambda {lk} {k} {str(ci.get(k))[:12]} != current {str(v)[:12]}")
    return probs


# =========================================================================================== provenance
def _git(*args):
    try:
        return subprocess.run(["git", "-C", ROOT, *args], capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def code_digests(files):
    """{repo-relative path: sha256} of the given source files (missing files -> None)."""
    out = {}
    for f in files:
        f = os.path.abspath(f)
        out[rel(f)] = sha256_file(f) if os.path.exists(f) else None
    return out


def provenance(code_files=()):
    """Provenance block of an output: git commit / dirty flag, engine sha256, code digests, library versions, host."""
    import scipy
    import matplotlib
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    eng = sha256_file(ENGINE_PATH) if os.path.exists(ENGINE_PATH) else None
    return dict(git_commit=_git("rev-parse", "HEAD"), git_dirty=(bool(dirty) if dirty is not None else None),
                engine=rel(ENGINE_PATH), engine_sha256=eng,
                engine_sha256_is_frozen=bool(eng == ENGINE_SHA256),
                code=code_digests(code_files), python=platform.python_version(), numpy=np.__version__,
                scipy=scipy.__version__, matplotlib=matplotlib.__version__, host=platform.node(),
                pid=os.getpid(), cpu_affinity=(sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity")
                                               else None),
                generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


# =========================================================================================== output policy
def guard_output(path):
    """Refuse an output path inside a frozen tree (results/mechanism, results/equal_budget_v2, figures/mechanism,
    figures/equal_budget_v2, configs/)."""
    rp = os.path.realpath(os.path.abspath(path))
    for d in PROTECTED_DIRS:
        dd = os.path.realpath(d)
        if rp == dd or rp.startswith(dd + os.sep):
            raise ExpAError(f"refusing to write {path}: inside the frozen tree {d}")
    return rp


def decide_write(path, signature, force=False, recompute=False):
    """'write' or 'skip' for the output JSON ``path`` given the new input signature.  Raises ExpAError when an
    existing COMPLETE output has a different signature and force is False."""
    guard_output(path)
    if not os.path.exists(path):
        return "write"
    try:
        old = json.load(open(path))
    except Exception:  # noqa: BLE001  (an unreadable file is not a completed output)
        return "write"
    if old.get("input_signature") == signature:
        return "write" if recompute else "skip"
    if old.get("complete") and not force:
        raise ExpAError(f"{path} is a COMPLETE output of different inputs (signature {str(old.get('input_signature'))[:12]}"
                        f" != {signature[:12]}): refusing to overwrite it (use --force and record why)")
    return "write"


def previous_signature(path):
    """The signature an existing output carries (or, for an interrupted write's placeholder, the one it was replacing)."""
    try:
        d = json.load(open(path))
        return d.get("input_signature") or d.get("replacing_signature")
    except Exception:  # noqa: BLE001
        return None


def wait_for_disk(what="", free_fn=None, sleep_fn=time.sleep, log_path=None, poll_s=None, max_pause_s=None,
                  pause_below=None, resume_above=None, now_fn=time.time):
    """PREREGISTRATION section 3: while the free space of the repository's filesystem is below 2 GB, do not write --
    log the pause, poll, and resume only once it is above 3 GB; a pause that does not clear within 24 h raises
    ExpAError (stopping rule iv).  Returns the seconds paused (0.0 when no pause).  The pause / resume lines go to
    stderr and are appended (best effort) to log_path, else $EXPA_DISK_PAUSE_LOG, else DISK_PAUSE_LOG."""
    free_fn = free_fn or (lambda: shutil.disk_usage(ROOT).free)
    lo = DISK_PAUSE_BELOW if pause_below is None else pause_below
    hi = DISK_RESUME_ABOVE if resume_above is None else resume_above
    poll = DISK_POLL_S if poll_s is None else poll_s
    cap = DISK_MAX_PAUSE_S if max_pause_s is None else max_pause_s
    free = free_fn()
    if free >= lo:
        return 0.0
    lp = log_path or os.environ.get(DISK_PAUSE_LOG_ENV) or DISK_PAUSE_LOG

    def log(msg):
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} pid {os.getpid()} {msg}"
        print(line, file=sys.stderr, flush=True)
        try:
            os.makedirs(os.path.dirname(lp), exist_ok=True)
            with open(lp, "a") as fh:
                fh.write(line + "\n")
        except Exception:  # noqa: BLE001  (the disk is nearly full: logging is best effort)
            pass
    t0 = now_fn()
    log(f"DISK PAUSE before {what or 'a write'}: {free / 1024 ** 3:.2f} GB free < {lo / 1024 ** 3:.1f} GB; resuming "
        f"above {hi / 1024 ** 3:.1f} GB (PREREGISTRATION section 3)")
    while True:
        sleep_fn(poll)
        free = free_fn()
        if free > hi:
            dt = now_fn() - t0
            log(f"DISK RESUME before {what or 'a write'} after {dt:.0f} s: {free / 1024 ** 3:.2f} GB free")
            return dt
        if now_fn() - t0 > cap:
            log(f"DISK PAUSE did not clear within {cap / 3600:.0f} h: stopping (PREREGISTRATION section 3, rule iv)")
            raise ExpAError(f"disk pause longer than {cap / 3600:.0f} h before {what or 'a write'} (stopping rule iv)")


def _existing_parent(path):
    d = os.path.dirname(os.path.abspath(path))
    while d and not os.path.isdir(d):
        nd = os.path.dirname(d)
        if nd == d:
            break
        d = nd
    return d or ROOT


def _atomic(path, write_fn):
    """write_fn(tmp_path) then os.replace(tmp, path); the temporary file is removed on any failure."""
    guard_output(path)
    wait_for_disk(rel(path), free_fn=lambda: shutil.disk_usage(_existing_parent(path)).free)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    root, ext = os.path.splitext(path)
    tmp = f"{root}.tmp{os.getpid()}{ext}"
    try:
        write_fn(tmp)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def write_json(path, doc):
    """Strict JSON (inf -> "inf", NaN -> null), atomic, after the disk-space check."""
    def w(tmp):
        with open(tmp, "w") as fh:
            json.dump(M.json_safe(doc), fh, indent=1, allow_nan=False)
    _atomic(path, w)


def write_text(path, text):
    def w(tmp):
        with open(tmp, "w") as fh:
            fh.write(text)
    _atomic(path, w)


def save_figure(fig, path, **kw):
    """fig.savefig to ``path`` atomically (temporary file in the same directory, removed on failure)."""
    fmt_ = os.path.splitext(path)[1].lstrip(".")
    _atomic(path, lambda tmp: fig.savefig(tmp, format=fmt_, **kw))


def begin_output(path, schema, previous=None, note=None):
    """Placeholder written BEFORE the data files of an output: no input_signature, complete false, state 'writing'.
    An interrupted write therefore leaves a record that is neither signed (so it is never 'up to date') nor
    COMPLETE (so it never blocks the re-run); the final JSON replaces it after every data file is written."""
    write_json(path, dict(schema=schema, state="writing", input_signature=None, complete=False,
                          replacing_signature=previous, started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                          note=note or "interrupted write: the data files next to this record may be partial"))


def load_signed(path, what=None):
    """An output JSON of another tool, refusing a missing file and an interrupted write (placeholder: state 'writing'
    or no input_signature); numbers restored by reload_json."""
    what = what or os.path.basename(path)
    if not os.path.exists(path):
        raise ExpAError(f"missing input {path} ({what}: run its tool first)")
    try:
        d = json.load(open(path))
    except Exception as e:  # noqa: BLE001
        raise ExpAError(f"unreadable input {path}: {type(e).__name__}: {e}")
    if d.get("state") == "writing" or not d.get("input_signature"):
        raise ExpAError(f"{what} ({path}) is an interrupted write (no signature): re-run its tool first")
    return reload_json(d)


def reload_json(doc):
    """JSON-safe document -> numbers again ('inf' -> inf, null -> NaN where a number is expected is left None)."""
    def conv(o):
        if isinstance(o, dict):
            return {k: conv(v) for k, v in o.items()}
        if isinstance(o, list):
            return [conv(v) for v in o]
        if o == "inf":
            return math.inf
        if o == "-inf":
            return -math.inf
        return o
    return conv(doc)


# =========================================================================================== statistics
def quantile_inf(x, q):
    return M.quantile_inf(x, q)


def row_median_inf(X):
    """Row-wise median of a 2-D array that may hold +inf (eqb_metrics.quantile_inf's rule: +inf when a middle
    element is +inf)."""
    Xs = np.sort(np.asarray(X, float), axis=1)
    n = Xs.shape[1]
    lo, hi = Xs[:, (n - 1) // 2], Xs[:, n // 2]
    with np.errstate(invalid="ignore"):
        mid = 0.5 * (lo + hi)
    return np.where(np.isinf(hi), hi, np.where(np.isinf(lo), lo, mid))


def ratio_inf(a, b):
    """a / b with the censoring conventions finite / inf = 0, inf / finite = inf, inf / inf = undefined (NaN);
    b = 0: a > 0 -> inf, a = 0 -> undefined."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = a / b
    r = np.where(np.isinf(a) & np.isinf(b), np.nan, r)
    r = np.where(np.isfinite(a) & np.isinf(b), 0.0, r)
    r = np.where(np.isinf(a) & np.isfinite(b), np.inf, r)
    r = np.where((b == 0) & (a == 0), np.nan, r)
    return r


def boot_median_inf(values, n_boot=N_BOOT, seed=BOOT_SEED):
    """dict(median, ci95, n, n_inf) of a seed bootstrap of the (inf-aware) median of a {seed: value} dict (NaN
    values dropped; +inf = censored kept)."""
    seeds = sorted(s for s, v in values.items() if not math.isnan(fnum(v)))
    x = np.array([fnum(values[s]) for s in seeds], float)
    if x.size == 0:
        return dict(median=math.nan, ci95=[math.nan, math.nan], n=0, n_inf=0)
    rng = np.random.default_rng(seed)
    b = row_median_inf(x[rng.integers(0, x.size, size=(int(n_boot), x.size))])
    return dict(median=quantile_inf(x, 50), ci95=[quantile_inf(b, 2.5), quantile_inf(b, 97.5)], n=int(x.size),
                n_inf=int(np.sum(np.isinf(x))))


def ratio_of_medians(a, b, n_boot=N_BOOT, seed=BOOT_SEED):
    """Paired ratio of seed medians median_s(a) / median_s(b) over the seeds common to both {seed: value} dicts
    (seeds coupled across cells: one index draw resamples the same seeds in a and b), inf-aware medians and the
    ratio_inf conventions; percentile 95 % CI over the defined resamples; the fraction of undefined resamples is
    reported (> 2.5 %: 'unresolved')."""
    seeds = sorted(s for s in set(a) & set(b) if not (math.isnan(fnum(a[s])) or math.isnan(fnum(b[s]))))
    if not seeds:
        return dict(n=0, ratio=math.nan, ci95=[math.nan, math.nan], undefined_frac=math.nan, unresolved=True,
                    median_a=math.nan, median_b=math.nan)
    va = np.array([fnum(a[s]) for s in seeds], float)
    vb = np.array([fnum(b[s]) for s in seeds], float)
    pa, pb = quantile_inf(va, 50), quantile_inf(vb, 50)
    pt = float(ratio_inf(pa, pb))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(seeds), size=(int(n_boot), len(seeds)))
    rb = ratio_inf(row_median_inf(va[idx]), row_median_inf(vb[idx]))
    und = float(np.mean(np.isnan(rb)))
    ok = rb[~np.isnan(rb)]
    ci = [quantile_inf(ok, 2.5), quantile_inf(ok, 97.5)] if ok.size else [math.nan, math.nan]
    return dict(n=len(seeds), ratio=pt, ci95=ci, undefined_frac=und, unresolved=bool(und > 0.025),
                median_a=pa, median_b=pb)


def median_of_ratios(a, b, n_boot=N_BOOT, seed=BOOT_SEED):
    """Seed-median of the per-seed ratios a(s) / b(s) over the common seeds with both values finite and b > 0,
    with a percentile seed-bootstrap 95 % CI (default_rng(seed), n_boot resamples)."""
    seeds = sorted(s for s in set(a) & set(b) if math.isfinite(fnum(a[s])) and math.isfinite(fnum(b[s]))
                   and fnum(b[s]) > 0)
    if not seeds:
        return dict(n=0, median=math.nan, ci95=[math.nan, math.nan], per_seed={})
    r = np.array([fnum(a[s]) / fnum(b[s]) for s in seeds], float)
    rng = np.random.default_rng(seed)
    bm = np.median(r[rng.integers(0, r.size, size=(int(n_boot), r.size))], axis=1)
    return dict(n=int(r.size), median=float(np.median(r)),
                ci95=[float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))],
                per_seed={int(s): float(v) for s, v in zip(seeds, r)})


def mean_companions(abf, fr, n_boot=N_BOOT, seed=BOOT_SEED):
    """Mean-based companions of the within-cell gain (PREREGISTRATION 1.6) from {seed: value} dicts of the two arms:
    mean G = mean over seeds of (FR - ABF)/ABF (seeds with both values finite and ABF > 0) and mean L = mean of
    log(FR/ABF) (both finite and > 0), each with a percentile seed-bootstrap 95 % CI (default_rng(seed), n_boot
    resamples, one fresh generator per quantity)."""
    out = {}
    for tag, ok, fn in (("G", lambda a, f: a > 0, lambda a, f: (f - a) / a),
                        ("L", lambda a, f: a > 0 and f > 0, lambda a, f: math.log(f / a))):
        seeds = sorted(s for s in set(abf) & set(fr)
                       if math.isfinite(fnum(abf[s])) and math.isfinite(fnum(fr[s])) and ok(fnum(abf[s]), fnum(fr[s])))
        v = np.array([fn(fnum(abf[s]), fnum(fr[s])) for s in seeds], float)
        if v.size == 0:
            out[tag] = dict(n=0, mean=math.nan, ci95=[math.nan, math.nan])
            continue
        rng = np.random.default_rng(seed)
        b = v[rng.integers(0, v.size, size=(int(n_boot), v.size))].mean(axis=1)
        out[tag] = dict(n=int(v.size), mean=float(v.mean()), ci95=[float(np.percentile(b, 2.5)),
                                                                    float(np.percentile(b, 97.5))])
    out["definition"] = ("mean G = mean_s (FR - ABF)/ABF; mean L = mean_s log(FR/ABF); percentile seed-bootstrap "
                         "95 % CIs (mean-based companions of the median G, PREREGISTRATION 1.6)")
    return out


def holm(pvals, m=None):
    """Holm step-down adjusted p (scripts/mechanism/cross_cell.holm: a NaN / None p counts as 1)."""
    import cross_cell as X
    return X.holm(pvals, m)


def fmt(v, nd=3):
    v = fnum(v)
    if math.isnan(v):
        return "--"
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    return f"{v:.{nd}g}"


def fmt_ci(ci, nd=3):
    if not ci:
        return "[--, --]"
    return f"[{fmt(ci[0], nd)}, {fmt(ci[1], nd)}]"
