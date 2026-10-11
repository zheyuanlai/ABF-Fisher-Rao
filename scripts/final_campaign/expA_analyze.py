#!/usr/bin/env python
"""Experiment A analysis driver: per-cell pipeline + cross-cell synthesis (docs/final_campaign/PREREGISTRATION.md 1.5-1.8).

    taskset -c <cores> python scripts/final_campaign/expA_analyze.py [--cells-dir configs/final_campaign/cells/expA]
        [--log-dir results/final_campaign/expA/logs/analysis] [--stages an,pc,ps,audit,cross,mech,tables,figures,outcome]
        [--workers 4] [--n-boot 10000] [--synthesis-dir results/final_campaign/expA/synthesis]
        [--tables-dir results/final_campaign/expA/tables] [--fig-dir figures/final_campaign/expA]
        [--force-cell] [--force-synthesis] [--allow-audit-fail] [--compute-ledger results/final_campaign/compute_ledger.csv]
        [--dry-run]

Per cell config (configs/final_campaign/cells/expA/*.json; one per lambda), the UNCHANGED equal-budget tools with
--system gateway_family --config <cell> (each cell names its own results / analysis / figure locations):
  an     scripts/equal_budget/analyze_ladder.py [--n-boot]        -> <analysis_dir>/summary.json, tables.md, run_metrics/
  pc     scripts/equal_budget/plot_config.py --only-N <N>          -> <fig_dir>/N<N>/ (every N with a complete run)
  ps     scripts/equal_budget/plot_synthesis.py                    -> <fig_dir>/synthesis/ (required by the audit)
  audit  scripts/equal_budget/audit_completeness.py                -> <analysis_dir>/completeness_audit.json
then the Experiment A synthesis (this directory):
  cross  expA_cross_cell.py   mech  expA_mech.py   tables  expA_tables.py   figures  expA_figures.py   outcome  expA_outcome.py

Resumable and idempotent:
  an     skipped when the summary belongs to the current config / scoring code / reference
         (scripts/mechanism/cross_cell.check_summary) AND is fresh (check_freshness: the complete run files are exactly
         the summary's, every per-run cache entry valid).  A COMPLETE summary (every planned run) is NEVER overwritten
         without --force-cell, neither when it no longer matches the config, code or reference nor when it is stale
         against the run tree (a run file removed or replaced after the analysis, e.g. pruned raw arrays): the decision
         and its reason are in the run record.  analyze_ladder's per-run cache makes a re-run cheap.
  pc/ps  skipped when the state marker <log-dir>/state/<cell>_<stage>[_N<N>].json records the same signature (sha256 of
         the summary, the cell config and the plotting code) and the figure manifest exists.
  audit  always re-run (a check; writes only its own JSON).
  cross / mech / tables / figures / outcome: their own input signatures (same inputs -> skipped; a COMPLETE output of
         different inputs is refused unless --force-synthesis).
  Before 'an' the cell's frozen reference is compared with a fresh payload (scripts/mechanism/build_references.py,
  imported by path, writes nothing); a missing or different reference stops that cell with the command to build it.
  Audit gate: before the synthesis, EVERY cell's <analysis_dir>/completeness_audit.json is read (whether or not the
  audit ran in this invocation, e.g. --stages cross,mech,... or --only-lams): it must exist, have verdict PASS, have
  been made on the current summary and config (sha256) and list only runs of the frozen engine.  Otherwise the
  synthesis stages are not run, unless --allow-audit-fail (mid-production the audit fails by design: incomplete N
  without STATUS.json), in which case every synthesis tool gets --provisional and writes complete = false.  The
  synthesis tools apply the same audit test themselves (expA_analysis_common.completeness), so running them directly
  cannot produce a complete output over a failing audit either.
Disk (PREREGISTRATION section 3): no step starts (and the driver writes nothing) while the free space is below 2 GB;
resume above 3 GB; every pause is logged in <log-dir>/disk_pauses.log (the synthesis tools log theirs there too).
Every subprocess is single-threaded (CUDA_VISIBLE_DEVICES='', OMP/NUMBA/MKL/OPENBLAS threads 1) and inherits this
process's CPU affinity: pin the driver with taskset; at most --workers cells run at once.  Logs: <log-dir>/<stage>_<cell>
[_N<N>].log (appended, one header per invocation); run record <log-dir>/expA_analyze_<UTC>.json (provenance, every
step's decision, command, exit code, wall time, used core-seconds, audit status, provisional flag).  --compute-ledger
appends one row in the campaign ledger's format (utc_start, stage, item, threads, wall_s, used_core_h, note).
--dry-run prints every decision and command and runs nothing.  Exit 0 only when every requested step succeeded and
(synthesis requested) every audit passes; a provisional synthesis exits 1.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import expA_analysis_common as C  # noqa: E402
import cross_cell as X  # noqa: E402  (scripts/mechanism/cross_cell.py: summary identity + freshness checks)

SCHEMA = "final_expA_analyze/1"
EQB = os.path.join(C.ROOT, "scripts", "equal_budget")
TOOLS = dict(an=os.path.join(EQB, "analyze_ladder.py"), pc=os.path.join(EQB, "plot_config.py"),
             ps=os.path.join(EQB, "plot_synthesis.py"), audit=os.path.join(EQB, "audit_completeness.py"),
             cross=os.path.join(HERE, "expA_cross_cell.py"), mech=os.path.join(HERE, "expA_mech.py"),
             tables=os.path.join(HERE, "expA_tables.py"), figures=os.path.join(HERE, "expA_figures.py"),
             outcome=os.path.join(HERE, "expA_outcome.py"))
CELL_STAGES = ("an", "pc", "ps", "audit")
SYN_STAGES = ("cross", "mech", "tables", "figures", "outcome")
PLOT_CODE = dict(pc=[TOOLS["pc"], os.path.join(EQB, "fig_legibility.py"), os.path.join(EQB, "eqb_metrics.py"),
                     os.path.join(EQB, "eqb_family.py")],
                 ps=[TOOLS["ps"], os.path.join(EQB, "fig_legibility.py"), os.path.join(EQB, "eqb_metrics.py"),
                     os.path.join(EQB, "eqb_family.py")])
DEFAULT_LOG_DIR = os.path.join(C.LOG_DIR, "analysis")
_print_lock = threading.Lock()


def say(*a):
    with _print_lock:
        print(*a, flush=True)


def child_env(log_dir=None):
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", NUMBA_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1")
    if log_dir:                                   # the Experiment A tools log disk pauses next to the driver's logs
        env[C.DISK_PAUSE_LOG_ENV] = os.path.join(os.path.abspath(log_dir), "disk_pauses.log")
    pp = os.path.join(C.ROOT, "src")
    env["PYTHONPATH"] = pp + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def wait_disk(what, log_dir):
    """PREREGISTRATION section 3 before starting a step: no new step (and no write of the driver) while the free
    space is below 2 GB; resume above 3 GB; the pause is logged in <log-dir>/disk_pauses.log."""
    return C.wait_for_disk(what, log_path=os.path.join(os.path.abspath(log_dir), "disk_pauses.log"))


def run_cmd(cmd, log_path, dry=False):
    """Run one tool (single-threaded child) appending its output to log_path; returns (exit code, wall s).  The
    disk-space pause (wait_disk) is honoured BEFORE the child starts: the unchanged equal-budget tools do not check
    free space themselves."""
    if dry:
        return None, 0.0
    log_dir = os.path.dirname(log_path)
    wait_disk(f"{os.path.basename(cmd[1]) if len(cmd) > 1 else cmd[0]} ({os.path.basename(log_path)})", log_dir)
    os.makedirs(log_dir, exist_ok=True)
    t0 = time.time()
    with open(log_path, "a") as fh:
        fh.write(f"\n==== {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {' '.join(cmd)}\n")
        fh.flush()
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=child_env(log_dir), cwd=C.ROOT).returncode
        fh.write(f"==== exit {rc} after {time.time() - t0:.1f} s\n")
    return rc, time.time() - t0


# ============================================================================================ per-cell decisions
def reference_problems(rec):
    """[] when the cell's frozen reference exists and equals a fresh payload (build_references.compare)."""
    spec = importlib.util.spec_from_file_location(
        "mech_build_references_expA", os.path.join(C.ROOT, "scripts", "mechanism", "build_references.py"))
    BR = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(BR)
    P = rec["P"]
    path = C.MF._abs(P["reference_file"])
    if not os.path.exists(path):
        return [f"frozen reference {path} does not exist: run python scripts/mechanism/build_references.py "
                f"--cells-root {C.rel(os.path.dirname(os.path.dirname(rec['config'])))} --out-dir "
                f"{C.rel(os.path.dirname(path))}"]
    payload, _ = BR.reference_payload(P, rec["config"])
    d = BR.compare(path, payload)
    return [f"reference {path} differs from a fresh payload: {d[:4]}"] if d else []


def summary_complete(S, P):
    for N in P["N_ladder"]:
        for m in ("abf", "fr"):
            if C.complete_seeds(S, N, m) != sorted(int(s) for s in P["seeds"]):
                return False
    return True


def an_decision(rec, force=False):
    """('skip' | 'run' | 'refuse', reason).  A COMPLETE summary (every planned run) is never replaced without
    force (--force-cell): neither when it no longer matches the config / scoring code / reference
    (cross_cell.check_summary) nor when the run tree changed under it (cross_cell.check_freshness: a run file removed,
    replaced or with an invalid cache entry) -- re-analysing would silently degrade a complete, committed summary
    (e.g. after raw arrays were pruned).  An incomplete summary is simply re-analysed (mid-production)."""
    sp = rec["summary_path"]
    if not os.path.exists(sp):
        return "run", "no summary yet"
    try:
        S = json.load(open(sp))
    except Exception as e:  # noqa: BLE001
        return "run", f"unreadable summary ({e})"
    r = dict(rec, S=S)
    complete = summary_complete(S, rec["P"])
    probs = X.check_summary(r)
    if probs:
        if complete and not force:
            return "refuse", ("a COMPLETE summary no longer matches the config / scoring code / reference: "
                              + "; ".join(probs[:3]) + " (use --force-cell and record why)")
        return "run", "summary does not match the current config / code / reference: " + "; ".join(probs[:2])
    stale = X.check_freshness(r)
    if stale:
        if complete and not force:
            return "refuse", (f"a COMPLETE summary is stale against the run tree ({len(stale)} issue(s), e.g. "
                              f"{stale[:3]}): run files were removed or replaced after the analysis; refusing to "
                              "overwrite it with a degraded summary (restore the runs, or use --force-cell and record "
                              "why)")
        return "run", f"summary stale: {stale[:2]}"
    return "skip", "summary up to date (identity and freshness verified)"


def marker_path(log_dir, cell, stage, N=None):
    return os.path.join(log_dir, "state", f"{cell}_{stage}" + (f"_N{N}" if N is not None else "") + ".json")


def plot_signature(rec, stage, N=None):
    return C.digest(dict(stage=stage, N=N, summary=C.sha256_file(rec["summary_path"]),
                         config=C.sha256_file(rec["config"]), code=C.code_digests(PLOT_CODE[stage])))


def plot_decision(rec, stage, log_dir, N=None):
    mp = marker_path(log_dir, rec["name"], stage, N)
    fig_dir = C.MF.cell_fig_dir(rec["P"])
    manifest = os.path.join(fig_dir, f"N{N}" if stage == "pc" else "synthesis", "MANIFEST.json")
    if not os.path.exists(rec["summary_path"]):
        return "run", "no summary", None
    sig = plot_signature(rec, stage, N)
    try:
        old = json.load(open(mp)).get("signature")
    except Exception:  # noqa: BLE001
        old = None
    if old == sig and os.path.exists(manifest):
        return "skip", "figures up to date (state marker)", sig
    return "run", ("inputs changed since the last drawing" if old else "not drawn yet"), sig


def write_marker(log_dir, cell, stage, sig, N=None, extra=None):
    p = marker_path(log_dir, cell, stage, N)
    C.write_json(p, dict(signature=sig, stage=stage, cell=cell, N=N,
                         utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **(extra or {})))


# ============================================================================================ per-cell pipeline
def cell_pipeline(rec, a, stages, log_dir, steps):
    name, cfg = rec["name"], rec["config"]
    py = sys.executable
    base = ["--system", "gateway_family", "--config", cfg]

    def record(stage, decision, reason, cmd=None, rc=None, wall=0.0, N=None, log=None):
        st = dict(cell=name, lam=C.lam_key(rec["lam"]), stage=stage, N=N, decision=decision, reason=reason,
                  cmd=cmd, exit=rc, wall_s=wall, log=C.rel(log) if log else None)
        steps.append(st)
        say(f"[{name}] {stage}{'' if N is None else ' N' + str(N)}: {decision} ({reason})"
            + ("" if rc is None else f" -> exit {rc} in {wall:.1f} s"))
        return st

    ok = True
    if "an" in stages:
        rp = reference_problems(rec)
        if rp:
            record("an", "refuse", "; ".join(rp))
            return False
        d, why = an_decision(rec, a.force_cell)
        if d == "refuse":
            record("an", d, why)
            return False
        if d == "run":
            log = os.path.join(log_dir, f"an_{name}.log")
            cmd = [py, TOOLS["an"], *base, "--n-boot", str(a.n_boot)]
            rc, w = run_cmd(cmd, log, a.dry_run)
            record("an", "run" if not a.dry_run else "would run", why, cmd, rc, w, log=log)
            if rc not in (0, None):
                return False
        else:
            record("an", d, why)
    if not os.path.exists(rec["summary_path"]):
        if not a.dry_run:
            record("pc", "skip", "no summary: nothing to draw")
        return ok and a.dry_run
    S = json.load(open(rec["summary_path"]))
    if "pc" in stages:
        for N in [int(n) for n in rec["P"]["N_ladder"]]:
            if not (C.complete_seeds(S, N, "abf") or C.complete_seeds(S, N, "fr")):
                record("pc", "skip", "no complete run at this N", N=N)
                continue
            d, why, sig = plot_decision(rec, "pc", log_dir, N)
            if d == "skip":
                record("pc", d, why, N=N)
                continue
            log = os.path.join(log_dir, f"pc_{name}_N{N}.log")
            cmd = [py, TOOLS["pc"], *base, "--only-N", str(N)]
            rc, w = run_cmd(cmd, log, a.dry_run)
            record("pc", "run" if not a.dry_run else "would run", why, cmd, rc, w, N=N, log=log)
            if rc == 0:
                write_marker(log_dir, name, "pc", sig, N)
            ok &= rc in (0, None)
    if "ps" in stages:
        d, why, sig = plot_decision(rec, "ps", log_dir)
        if d == "skip":
            record("ps", d, why)
        else:
            log = os.path.join(log_dir, f"ps_{name}.log")
            cmd = [py, TOOLS["ps"], *base]
            rc, w = run_cmd(cmd, log, a.dry_run)
            record("ps", "run" if not a.dry_run else "would run", why, cmd, rc, w, log=log)
            if rc == 0:
                write_marker(log_dir, name, "ps", sig)
            ok &= rc in (0, None)
    if "audit" in stages:
        log = os.path.join(log_dir, f"audit_{name}.log")
        cmd = [py, TOOLS["audit"], *base]
        rc, w = run_cmd(cmd, log, a.dry_run)
        record("audit", "run" if not a.dry_run else "would run",
               "always re-run" if a.dry_run else ("PASS" if rc == 0 else "FAIL (see the log)"), cmd, rc, w, log=log)
        ok &= rc in (0, None)
    return ok


def synthesis_cmds(a, provisional=False):
    """The synthesis commands.  provisional (the synthesis runs over a failed or not-current completeness audit with
    --allow-audit-fail): every tool gets --provisional, which forces complete = false in its output."""
    py = sys.executable
    common = ["--cells-dir", a.cells_dir]
    flags = (["--force"] if a.force_synthesis else []) + (["--provisional"] if provisional else [])
    return dict(
        cross=[py, TOOLS["cross"], *common, "--out-dir", a.synthesis_dir, "--n-boot", str(a.n_boot), *flags],
        mech=[py, TOOLS["mech"], *common, "--out-dir", a.synthesis_dir, "--n-boot", str(a.n_boot), *flags],
        tables=[py, TOOLS["tables"], *common, "--out-dir", a.tables_dir, "--n-boot", str(a.n_boot), *flags],
        figures=[py, TOOLS["figures"], *common, "--synthesis-dir", a.synthesis_dir, "--fig-dir", a.fig_dir, *flags],
        outcome=[py, TOOLS["outcome"], *common, "--synthesis-dir", a.synthesis_dir, *flags])


def synthesis(a, stages, log_dir, steps, provisional=False):
    cmds = synthesis_cmds(a, provisional)
    ok = True
    for st in SYN_STAGES:
        if st not in stages:
            continue
        cmd = cmds[st] + (["--dry-run"] if a.dry_run else [])
        log = os.path.join(log_dir, f"{st}.log")
        if a.dry_run:
            r = subprocess.run(cmd, capture_output=True, text=True, env=child_env(log_dir), cwd=C.ROOT)
            msg = (r.stdout.strip().splitlines() or [r.stderr.strip()])[-1] if (r.stdout or r.stderr) else ""
            if r.returncode != 0 and st in ("figures", "outcome") and "missing input" in msg:
                msg = "would run after cross / mech (its inputs are written by those stages)"
            steps.append(dict(stage=st, decision="dry-run", reason=msg, cmd=cmds[st], exit=r.returncode, wall_s=0.0))
            say(f"[synthesis] {st}: {msg}")
            continue
        rc, w = run_cmd(cmd, log, False)
        steps.append(dict(stage=st, decision="run", reason=None, cmd=cmd, exit=rc, wall_s=w, log=C.rel(log)))
        say(f"[synthesis] {st}: exit {rc} in {w:.1f} s (log {C.rel(log)})")
        if rc != 0:
            ok = False
            break
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells-dir", default=C.CELLS_DIR)
    ap.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    ap.add_argument("--stages", default=",".join(CELL_STAGES + SYN_STAGES))
    ap.add_argument("--workers", type=int, default=4, help="cells processed at once (one single-threaded child each)")
    ap.add_argument("--n-boot", type=int, default=C.N_BOOT)
    ap.add_argument("--synthesis-dir", default=C.SYN_DIR)
    ap.add_argument("--tables-dir", default=C.TABLES_DIR)
    ap.add_argument("--fig-dir", default=C.FIG_DIR)
    ap.add_argument("--only-lams", type=float, nargs="*", default=None, help="per-cell stages for these lambdas only")
    ap.add_argument("--force-cell", action="store_true", help="re-analyse a COMPLETE summary whose identity changed or whose run tree changed (record why)")
    ap.add_argument("--force-synthesis", action="store_true", help="pass --force to the synthesis tools")
    ap.add_argument("--allow-audit-fail", action="store_true", help="run a PROVISIONAL synthesis (every tool gets --provisional: complete = false) although an audit fails")
    ap.add_argument("--compute-ledger", default=None, help="append one row to this compute ledger CSV")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    stages = [s.strip() for s in a.stages.split(",") if s.strip()]
    bad = [s for s in stages if s not in CELL_STAGES + SYN_STAGES]
    if bad:
        ap.error(f"unknown stages {bad}; choose from {CELL_STAGES + SYN_STAGES}")
    a.cells_dir = os.path.abspath(a.cells_dir)
    for d in (a.synthesis_dir, a.tables_dir, a.fig_dir, a.log_dir):
        C.guard_output(d)
    cells = C.load_cells(a.cells_dir)
    if not cells:
        raise C.ExpAError(f"no cell config under {a.cells_dir}")
    missing = [C.lam_key(l) for l in C.LAMS if l not in cells]
    if missing:
        say(f"WARNING: no cell config for lambda {missing}")
    production, fixture, probs = C.design_status(cells)
    log_dir = os.path.abspath(a.log_dir)
    t0, utc0 = time.time(), time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    steps = []
    sel = {l: r for l, r in sorted(cells.items()) if not a.only_lams or l in a.only_lams}
    say(f"Experiment A analysis: {len(sel)} cell(s) {[r['name'] for r in sel.values()]}, stages {stages}, "
        f"workers {a.workers}, production design {production}, fixture {fixture}{' (DRY RUN)' if a.dry_run else ''}")
    cell_ok = {}
    cstages = [s for s in stages if s in CELL_STAGES]
    if cstages:
        with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
            futs = {ex.submit(cell_pipeline, r, a, cstages, log_dir, steps): r["name"] for r in sel.values()}
            for f, name in futs.items():
                try:
                    cell_ok[name] = f.result()
                except Exception as e:  # noqa: BLE001
                    cell_ok[name] = False
                    steps.append(dict(cell=name, stage="?", decision="error", reason=f"{type(e).__name__}: {e}"))
                    say(f"[{name}] ERROR {type(e).__name__}: {e}")
    audit_fail = [s["cell"] for s in steps if s.get("stage") == "audit" and s.get("exit") not in (0, None)]
    other_fail = sorted(n for n, ok in cell_ok.items() if not ok and n not in audit_fail)
    # The synthesis covers EVERY cell, so the audit gate reads every cell's audit JSON on disk (also those not audited
    # in this invocation: --stages without 'audit', --only-lams): it must PASS, be made on the current summary and
    # config, and list only frozen-engine runs (expA_analysis_common.audit_status).
    audit_files = {cells[l]["name"]: p for l, p in C.audit_failures(cells).items()}
    for n, p in sorted(audit_files.items()):
        if n not in audit_fail:
            say(f"[{n}] audit on disk does not pass: {p[:2]}")
    audit_bad = sorted(set(audit_fail) | set(audit_files))
    provisional = bool(audit_bad) and a.allow_audit_fail
    syn_ok = None
    sstages = [s for s in stages if s in SYN_STAGES]
    if sstages:
        if other_fail and not a.dry_run:
            say(f"synthesis NOT run: per-cell stages failed for {other_fail}")
            syn_ok = False
        elif audit_bad and not (a.allow_audit_fail or a.dry_run):
            say(f"synthesis NOT run: completeness audit FAILED / missing / not current for {audit_bad} "
                "(--allow-audit-fail runs a PROVISIONAL synthesis, complete = false; expected while production is "
                "incomplete)")
            syn_ok = False
        else:
            if a.dry_run and audit_bad and not a.allow_audit_fail:
                say(f"[dry-run] the synthesis below runs only if every audit passes after this invocation's audit "
                    f"stage (now not passing: {audit_bad}); otherwise it is skipped (--allow-audit-fail: provisional)")
            if provisional:
                say(f"PROVISIONAL synthesis (--allow-audit-fail over the audits of {audit_bad}): every synthesis tool "
                    "gets --provisional (complete = false)")
            syn_ok = synthesis(a, sstages, log_dir, steps, provisional)
    wall = time.time() - t0
    used = sum(float(s.get("wall_s") or 0.0) for s in steps)
    rec = dict(schema=SCHEMA, started_utc=utc0, wall_s=wall, used_core_seconds=used, args=vars(a), stages=stages,
               production_design=production, fixture=fixture, design_problems=probs, cells_missing=missing,
               cell_ok=cell_ok, audit_failed=sorted(set(audit_fail)), audit_not_passing=audit_bad,
               audit_problems_on_disk=audit_files, provisional=provisional, synthesis_ok=syn_ok, steps=steps,
               provenance=C.provenance([os.path.abspath(__file__), os.path.join(HERE, "expA_analysis_common.py")]
                                       + list(TOOLS.values())))
    if not a.dry_run:
        rp = os.path.join(log_dir, f"expA_analyze_{utc0.replace(':', '').replace('-', '')}.json")
        C.write_json(rp, rec)
        say(f"run record {rp}")
        if a.compute_ledger:
            lp = os.path.abspath(a.compute_ledger)
            C.guard_output(lp)
            wait_disk("compute ledger row", log_dir)
            new = not os.path.exists(lp)
            with open(lp, "a", newline="") as fh:
                w = csv.writer(fh)
                if new:
                    w.writerow(["utc_start", "stage", "item", "threads", "wall_s", "used_core_h", "note"])
                w.writerow([utc0[:16], "expA-analysis", f"expA_analyze stages {'+'.join(stages)}",
                            min(max(1, a.workers), max(1, len(sel))), f"{wall:.0f}", f"{used / 3600:.3f}",
                            "single-threaded children; used = sum of child wall times"])
    say(f"done in {wall:.1f} s; used {used / 3600:.3f} core-h; cells ok {cell_ok}; audit not passing "
        f"{audit_bad or 'none'}; provisional {provisional}; synthesis ok {syn_ok}")
    if a.dry_run:
        return 0
    # exit 1 when a per-cell stage failed, an audit run here failed, or (synthesis requested) any cell's audit does not
    # pass -- also for a provisional synthesis, which is never final
    return 0 if (all(cell_ok.values()) and not audit_fail and not (sstages and audit_bad)
                 and syn_ok in (True, None)) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except C.ExpAError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)
