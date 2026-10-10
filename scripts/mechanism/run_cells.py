#!/usr/bin/env python
"""Production driver of the mechanism campaign, Experiments I and II (docs/mechanism/SCIENTIFIC_PLAN.md section 5).

    python scripts/mechanism/run_cells.py --dry-run
    taskset -c 200-215 python scripts/mechanism/run_cells.py --experiment matched_free_energy --workers 16
    taskset -c 200-215 python scripts/mechanism/run_cells.py --smoke --smoke-T 4 --only-N 2048 --workers 16   # V5

Reads configs/mechanism/{matched_free_energy,conditional_relaxation}.json and runs one process per (variant, N,
seed, method) through gateway_family_numba.run_job.  No FR arm at N = 1.

Job states, decided on the FULL run signature (gateway_family_numba.run_signature of the job's cfg, method, N, seed,
n_steps and default grids, engine sha256 included), never on 'status complete' alone:
  complete  a complete result of THIS run: skipped;
  STALE     a complete result of a DIFFERENT run (other n_steps, h, cfg, model, engine sha256, ...): handed to run_job,
            which refuses it -> status FAILED in the ledger, the file is never overwritten;
  resume    a checkpoint <out>.ckpt.npz exists: run_job resumes it (a checkpoint of a different run is refused, FAILED);
  new       run.

Reuse cells (alpha1 of Experiment I, lam1 of Experiment II, marked "reuse" in the config: the equal-budget gateway
production files, plan section 5).  Smoke runs never reuse anything: these cells are always run with --smoke (so the
V5 smoke covers all seven dynamics).  In a production run with --include-reuse they are run.  Otherwise each selected
reuse cell is checked (reuse_status): it is left out only when the reuse is LICENSED -- the experiment's frozen h, B,
seeds, save grids and engine knobs equal configs/equal_budget_v2/gateway_production.json (make_cell_configs.check_reuse,
the same check the cell configs are generated with), the cell config configs/mechanism/cells/<experiment>/<variant>.json
exists, was generated from the current experiment config and declares the reuse, and its gate record
(scripts/mechanism/reuse_gate.py) licenses it (eqb_family.reuse_gate_problems, the check the analysis applies).  If the
gate record does not exist yet, --defer-reuse leaves the cells out with a warning (the analysis refuses them until the
gate PASSES; if it fails, re-run with --include-reuse).  In every other case the driver REFUSES to start (exit 2) and
says why: at h_frozen = 1.25e-5, for instance, reuse is impossible and the anchor cells must be run (--include-reuse).

Identical runs are never run twice: Experiment II lam1 is the same dynamics as Experiment I alpha1 (variant alpha,
alpha 1, lam 1, same h, B, seeds) at N in {2048, 512}.  Jobs with the same (cfg, N, seed, method, n_steps) -- hence the
same run signature -- have ONE owner (the first in EXPERIMENTS / config order: matched_free_energy alpha1); the others
are aliases: the owner's complete result is hard-linked (copied if the file system refuses links) to the alias path
and the alias experiment's ledger gets a 'complete' row with note 'alias of ...' (wall_s = link time, so compute is
not double counted).  An alias whose owner was not selected pulls the owner job in (reported).  An alias path that
already holds a result of a different run is refused (FAILED, never overwritten).

Budget: h = the config's "h_frozen" if present, else engine_cfg.h; B = anchor.N0 * anchor.T0 / h (must be an integer
to 1e-9 relative, else refused); n_steps = B / N exactly (refused otherwise).  The save grid is the engine's default
prereg_save_grid(n_steps, h), the trace grid default_trace_grid, the D2/D3 snapshot grid snapshot_grid.

Outputs  <root>/<variant>/N<N>/s<seed>_<method>.npz  and one  <root>/ledger.csv  per experiment (experiment, variant,
N, seed, method, n_steps, status, wall_s, peak_rss_mb, n_force_evals, finished_utc, note), where <root> is
  production:  the config's out_root (results/mechanism/<experiment>), or <--out-root>/<experiment>;
  --smoke:     <base>/smoke/T<smoke_T>/<experiment>, base = --out-root or results/mechanism -- never a production
               path, and one directory per smoke horizon.
--smoke: the first 2 config seeds (unless --seeds) and B = N0 * smoke_T / h (T_N = smoke_T N0 / N).

Pin the CPU set with taskset (the pool workers inherit the affinity); each worker is single-threaded.  Run it as a
script (the pool workers re-import it as __mp_main__).
"""
import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _k in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.expanduser("~/.cache/numba_mech"))

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
CFG_DIR = os.path.join(ROOT, "configs", "mechanism")
EXPERIMENTS = ("matched_free_energy", "conditional_relaxation")
LEDGER_FIELDS = ["experiment", "variant", "N", "seed", "method", "n_steps", "status", "wall_s", "peak_rss_mb",
                 "n_force_evals", "finished_utc", "note"]


def load_cfg(experiment, cfg_dir=None):
    with open(os.path.join(cfg_dir or CFG_DIR, experiment + ".json")) as f:
        return json.load(f)


def frozen_h(P):
    """The config's h_frozen (written after the validation gate) if present, else engine_cfg.h."""
    v = P.get("h_frozen")
    return float(v) if v is not None else float(P["engine_cfg"]["h"])


def budget(P, T0=None):
    """B = N0 T0 / h walker-steps per arm per seed (T0 = anchor.T0 unless given), refused unless an integer."""
    h = frozen_h(P)
    N0 = int(P["anchor"]["N0"])
    T0 = float(P["anchor"]["T0"] if T0 is None else T0)
    Bf = N0 * T0 / h
    B = int(round(Bf))
    if B < 1 or abs(Bf - B) > 1e-9 * max(1.0, Bf):
        raise ValueError(f"B = N0 T0 / h = {Bf!r} is not an integer (N0 {N0}, T0 {T0}, h {h})")
    return B


def variant_cfg(P, v):
    """engine_cfg with the frozen h and the variant's model parameters."""
    c = dict(P["engine_cfg"])
    c["h"] = frozen_h(P)
    c.update(variant=v["variant"], alpha=v["alpha"], kappa=v["kappa"], lam=v["lam"])
    return c


def out_root(P, experiment, smoke=False, root_override=None, smoke_T=4.0):
    """Production: <root_override>/<experiment>, else the config's out_root.  Smoke: <base>/smoke/T<smoke_T>/<experiment>
    with base = root_override or results/mechanism (never a production path; one directory per smoke horizon)."""
    if smoke:
        base = os.path.abspath(root_override) if root_override else os.path.join(ROOT, "results", "mechanism")
        return os.path.join(base, "smoke", f"T{float(smoke_T):g}", experiment)
    if root_override:
        return os.path.join(os.path.abspath(root_override), experiment)
    return os.path.join(ROOT, P["out_root"])


def out_path(root, job):
    return os.path.join(root, job["variant"], f"N{job['N']}", f"s{job['seed']}_{job['method']}.npz")


def job_list(P, experiment, *, smoke=False, smoke_T=4.0, only_variant=None, only_N=None, seeds=None,
             include_reuse=False):
    """All (variant, N, seed, method) jobs of one experiment config, in config order.  Cells marked 'reuse' are left
    out unless include_reuse or smoke (a smoke run never reuses anything); whether leaving them out is licensed is
    main()'s decision (reuse_status)."""
    B = budget(P, smoke_T if smoke else None)
    if seeds:
        seeds = [int(s) for s in seeds]
    else:
        seeds = list(P["seeds"][:2]) if smoke else list(P["seeds"])
    J = []
    for v in P["variants"]:
        if only_variant and v["name"] not in only_variant:
            continue
        if v.get("reuse") and not (include_reuse or smoke):
            continue
        cfg = variant_cfg(P, v)
        for N in P["N"]:
            N = int(N)
            if only_N and N not in only_N:
                continue
            n_steps = B // N
            if n_steps * N != B:
                raise ValueError(f"B {B} is not divisible by N {N}")
            for seed in seeds:
                for method in (("abf",) if N == 1 else ("abf", "fr")):
                    J.append(dict(experiment=experiment, variant=v["name"], N=N, seed=int(seed), method=method,
                                  n_steps=int(n_steps), cfg=cfg,
                                  checkpoint_every_s=float(P.get("checkpoint_every_s", 600.0))))
    return J


def job_key(job):
    """Identity of the RUN (not of the cell): two jobs with the same key have the same run signature."""
    return (json.dumps(job["cfg"], sort_keys=True), int(job["N"]), int(job["seed"]), job["method"],
            int(job["n_steps"]))


def job_signature(E, job):
    """The run signature run_job compares against (the job's cfg, method, N, seed, n_steps and run_job's default
    save grid prereg_save_grid(n_steps, h); default trace / snapshot grids, init and noise seed)."""
    cfg = job["cfg"]
    save = E.prereg_save_grid(job["n_steps"], float(E.full_cfg(cfg)["h"]))
    return E.run_signature(cfg, job["method"], job["N"], job["seed"], job["n_steps"], save)


def job_state(E, path, job):
    """'complete' (a complete result of THIS run), 'stale' (a complete result of a DIFFERENT run), 'resume' (a
    checkpoint exists) or 'new'."""
    if E.is_complete(path, job_signature(E, job)):
        return "complete"
    if E.is_complete(path):
        return "stale"
    if os.path.exists(path + ".ckpt.npz"):
        return "resume"
    return "new"


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def reuse_status(P, experiment, v, cfg_dir=None, exp_cfg_path=None):
    """May the reuse cell ``v`` of ``experiment`` be left out of a production run (plan section 5)?

    Returns dict(licensed, hard, pending, cell_config):
      hard     the reuse is impossible or declined: the experiment's frozen h / B / seeds / save grids / engine knobs
               differ from the equal-budget gateway production config (make_cell_configs.check_reuse); the cell
               config is missing, stale (generated from another version of the experiment config) or declares a
               re-run (make_cell_configs.py --rerun-reuse-cells); or the gate record EXISTS and does not license it
               (verdict not PASS, reused file changed, ...: eqb_family.reuse_gate_problems);
      pending  the gate record does not exist yet;
      licensed neither."""
    for d in (os.path.join(ROOT, "scripts", "mechanism"), os.path.join(ROOT, "scripts", "equal_budget")):
        if d not in sys.path:
            sys.path.insert(0, d)
    import make_cell_configs as MC
    hard, pending = [], []
    try:
        h = MC.frozen_h(P)
        MC.check_reuse(P, h, MC.budget(P, h))
    except MC.CellConfigError as e:
        hard.append(f"reuse impossible: {e}")
    cell_path = os.path.join(cfg_dir or CFG_DIR, "cells", experiment, v["name"] + ".json")
    exp_cfg_path = exp_cfg_path or os.path.join(cfg_dir or CFG_DIR, experiment + ".json")
    cell = None
    if not os.path.exists(cell_path):
        hard.append(f"cell config {cell_path} does not exist (scripts/mechanism/make_cell_configs.py)")
    else:
        with open(cell_path) as fh:
            cell = json.load(fh)
        if cell.get("_generated_from_sha256") != _sha256_file(exp_cfg_path):
            hard.append(f"cell config {cell_path} was generated from another version of {exp_cfg_path} "
                        f"(regenerate with scripts/mechanism/make_cell_configs.py)")
        if not cell.get("reuse"):
            hard.append(f"cell config {cell_path} declares a re-run (generated with --rerun-reuse-cells)")
    if cell is not None and cell.get("reuse"):
        import eqb_family as MF
        rp = cell["reuse"].get("gate_record")
        probs = MF.reuse_gate_problems(cell)
        if probs:
            exists = bool(rp) and os.path.exists(rp if os.path.isabs(rp) else os.path.join(ROOT, rp))
            (hard if exists else pending).extend(f"reuse gate: {p}" for p in probs)
    return dict(licensed=not hard and not pending, hard=hard, pending=pending, cell_config=cell_path)


def _row(job, status, wall_s, peak_rss_mb, n_force_evals, note=""):
    return dict(experiment=job["experiment"], variant=job["variant"], N=job["N"], seed=job["seed"],
                method=job["method"], n_steps=job["n_steps"], status=status, wall_s=float(wall_s),
                peak_rss_mb=float(peak_rss_mb), n_force_evals=int(n_force_evals),
                finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), note=note)


def run_one(job, root):
    """Run (or resume, or skip) one job; never raises -- a failure is a ledger row with status FAILED (including a
    complete result of a DIFFERENT run at the output path: run_job refuses it and the file is left untouched)."""
    import gateway_family_numba as E
    p = out_path(root, job)
    t0 = time.time()
    status = "complete"
    try:
        res = E.run_job(job["cfg"], job["method"], job["N"], job["seed"], job["n_steps"], p,
                        checkpoint_every_s=job["checkpoint_every_s"])
        rec = dict(wall_s=float(res.get("wall_s", time.time() - t0)),
                   peak_rss_mb=float(res.get("peak_rss_mb", float("nan"))),
                   n_force_evals=int(res.get("n_force_evals", -1)))
        if rec["n_force_evals"] != job["N"] * job["n_steps"]:
            status = f"FAILED: n_force_evals {rec['n_force_evals']} != N n_steps {job['N'] * job['n_steps']}"
    except Exception as e:  # never mark a failed run complete
        status = f"FAILED: {type(e).__name__}: {e}"
        rec = dict(wall_s=time.time() - t0, peak_rss_mb=float("nan"), n_force_evals=-1)
    return _row(job, status, **rec)


def link_alias(E, job, src, dst):
    """Give the alias job the owner's complete result (identical run, identical signature): hard link, or an atomic
    copy where links are refused; never overwrites.  Returns the alias experiment's ledger row."""
    t0 = time.time()
    how, nfe = "none", -1
    try:
        want = job_signature(E, job)
        if not E.is_complete(src, want):
            raise RuntimeError(f"owner {src} is not a complete result of this run")
        if os.path.lexists(dst):
            raise FileExistsError(f"{dst} exists (never overwritten)")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        try:
            os.link(src, dst)
            how = "hard link"
        except OSError:
            tmp = dst + ".tmp_alias"
            shutil.copyfile(src, tmp)
            if os.path.lexists(dst):
                os.remove(tmp)
                raise FileExistsError(f"{dst} appeared while copying (never overwritten)")
            os.replace(tmp, dst)
            how = "copy"
        if not E.is_complete(dst, want):
            raise RuntimeError(f"{dst} fails the run-signature check after linking")
        import numpy as np
        with np.load(dst, allow_pickle=False) as z:
            nfe = int(z["n_force_evals"])
        status = "complete" if nfe == job["N"] * job["n_steps"] else \
            f"FAILED: n_force_evals {nfe} != N n_steps {job['N'] * job['n_steps']}"
    except Exception as e:  # noqa: BLE001
        status = f"FAILED: {type(e).__name__}: {e}"
    o = job["alias_of"]
    note = f"alias of {o['experiment']}/{o['variant']}/N{o['N']}/s{o['seed']}_{o['method']}.npz ({how}; identical run, " \
           f"not re-run)"
    return _row(job, status, time.time() - t0, float("nan"), nfe, note)


def warm_cache():
    """Compile (or load) the numba kernels once in the parent so that the pool workers load the cache."""
    import gateway_family_numba as E
    E.run_arm(dict(h=2.5e-5, variant="shift", kappa=1.0, lam=0.5), "fr", 4, 1, 200, [200])


class _Ledgers:
    """One append-only ledger.csv per experiment root, opened on first write.  An existing ledger keeps its own
    header (rows are written in its column order; a column it lacks is dropped)."""

    def __init__(self, roots):
        self.roots, self.open_ = roots, {}

    def write(self, row):
        exp = row["experiment"]
        if exp not in self.open_:
            root = self.roots[exp]
            os.makedirs(root, exist_ok=True)
            path = os.path.join(root, "ledger.csv")
            fields = LEDGER_FIELDS
            if os.path.exists(path) and os.path.getsize(path) > 0:
                with open(path, newline="") as fh:
                    fields = next(csv.reader(fh), None) or LEDGER_FIELDS
            new = not os.path.exists(path) or os.path.getsize(path) == 0
            fh = open(path, "a", newline="")
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            if new:
                w.writeheader()
            self.open_[exp] = (fh, w)
        fh, w = self.open_[exp]
        w.writerow(row)
        fh.flush()

    def close(self):
        for fh, _ in self.open_.values():
            fh.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--experiment", nargs="*", default=list(EXPERIMENTS), choices=list(EXPERIMENTS))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--only-variant", nargs="*", default=None, help="variant names (config 'name'), e.g. alpha0 shift")
    ap.add_argument("--only-N", type=int, nargs="*", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--include-reuse", action="store_true",
                    help="run the cells marked 'reuse' (alpha1 / lam1) with this engine instead of reusing")
    ap.add_argument("--defer-reuse", action="store_true",
                    help="leave the reuse cells out although their gate record does not exist yet (warning)")
    ap.add_argument("--smoke", action="store_true",
                    help="2 seeds, B = N0 * smoke_T / h, every cell incl. the reuse ones, separate output root")
    ap.add_argument("--smoke-T", type=float, default=4.0, help="smoke anchor horizon T0 (t.u. at N0 = 2048)")
    ap.add_argument("--out-root", default=None,
                    help="output base: <out-root>/<experiment>/... (--smoke: <out-root>/smoke/T<smoke_T>/<experiment>)")
    ap.add_argument("--cfg-dir", default=None, help="config directory (default configs/mechanism)")
    ap.add_argument("--est-us", type=float, default=0.085, help="us per walker-step for the core-h estimate")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if a.include_reuse and a.defer_reuse:
        ap.error("--include-reuse and --defer-reuse exclude each other")

    Ps = {exp: load_cfg(exp, a.cfg_dir) for exp in EXPERIMENTS}
    roots = {exp: out_root(Ps[exp], exp, a.smoke, a.out_root, a.smoke_T) for exp in EXPERIMENTS}

    # ---- reuse cells: run (smoke / --include-reuse), leave out (licensed / --defer-reuse while pending) or refuse
    refused = []
    for exp in a.experiment:
        for v in Ps[exp]["variants"]:
            if not v.get("reuse") or (a.only_variant and v["name"] not in a.only_variant):
                continue
            if a.smoke:
                print(f"{exp}/{v['name']}: reuse cell RUN (a smoke run never reuses)", flush=True)
                continue
            st = reuse_status(Ps[exp], exp, v, a.cfg_dir)
            if a.include_reuse:
                msg = f"{exp}/{v['name']}: reuse cell RUN (--include-reuse)"
                if st["licensed"] or (not st["hard"] and st["pending"]):
                    msg += (f"; NOTE: {st['cell_config']} still reads the equal-budget files -- regenerate the cell "
                            f"configs with make_cell_configs.py --rerun-reuse-cells for the analysis to read these runs")
                print(msg, flush=True)
            elif st["licensed"]:
                print(f"{exp}/{v['name']}: reuse cell LEFT OUT (reuse licensed: {st['cell_config']})", flush=True)
            elif not st["hard"] and a.defer_reuse:
                print(f"{exp}/{v['name']}: WARNING reuse cell LEFT OUT although the reuse is NOT yet licensed "
                      f"(--defer-reuse): {'; '.join(st['pending'])}.  The analysis refuses this cell until "
                      f"scripts/mechanism/reuse_gate.py records a PASS; if it fails, run it with --include-reuse.",
                      flush=True)
            else:
                refused.append((exp, v["name"], st))
    if refused:
        for exp, name, st in refused:
            print(f"REFUSED {exp}/{name}: the reuse cell would be left out although its reuse is not licensed:",
                  flush=True)
            for p in st["hard"] + st["pending"]:
                print(f"    - {p}", flush=True)
            print("    remedy: " + ("run it (--include-reuse)" if st["hard"] else
                                    "run scripts/mechanism/reuse_gate.py first, or --defer-reuse to run the other "
                                    "cells meanwhile, or --include-reuse to run it"), flush=True)
        return 2

    # ---- jobs; identical runs (same cfg, N, seed, method, n_steps) have ONE owner, the others are aliases
    sel = []
    for exp in a.experiment:
        sel += job_list(Ps[exp], exp, smoke=a.smoke, smoke_T=a.smoke_T, only_variant=a.only_variant,
                        only_N=a.only_N, seeds=a.seeds, include_reuse=a.include_reuse)
    owner = {}
    for exp in EXPERIMENTS:
        for j in job_list(Ps[exp], exp, smoke=a.smoke, smoke_T=a.smoke_T, only_N=a.only_N, seeds=a.seeds,
                          include_reuse=True):
            owner.setdefault(job_key(j), j)
    prim, aliases, pulled = {}, [], []
    for j in sel:
        k = job_key(j)
        o = owner[k]
        if (o["experiment"], o["variant"]) == (j["experiment"], j["variant"]):
            prim.setdefault(k, j)
        else:
            aliases.append(dict(j, alias_of=o))
    for j in aliases:
        k = job_key(j)
        if k not in prim:
            prim[k] = j["alias_of"]
            pulled.append(j["alias_of"])

    import gateway_family_numba as E
    states = {k: job_state(E, out_path(roots[j["experiment"]], j), j) for k, j in prim.items()}
    a_states = [job_state(E, out_path(roots[j["experiment"]], j), j) for j in aliases]
    if pulled:
        print(f"{len(pulled)} owner job(s) pulled in for selected aliases: "
              + ", ".join(sorted({f"{j['experiment']}/{j['variant']}" for j in pulled})), flush=True)
    for exp in EXPERIMENTS:
        P_jobs = [(j, states[k]) for k, j in prim.items() if j["experiment"] == exp]
        A_jobs = [(j, s) for j, s in zip(aliases, a_states) if j["experiment"] == exp]
        if not P_jobs and not A_jobs:
            continue
        n = lambda L, s: sum(1 for _, x in L if x == s)  # noqa: E731
        run = [j for j, s in P_jobs if s in ("new", "resume")]
        est = sum(j["N"] * j["n_steps"] for j in run) * a.est_us * 1e-6 / 3600.0
        print(f"{exp}: {len(P_jobs) + len(A_jobs)} jobs, {n(P_jobs, 'complete') + n(A_jobs, 'complete')} complete, "
              f"{len(run)} to run ({n(P_jobs, 'resume')} resume), est. {est:.1f} core-h -> {roots[exp]}", flush=True)
        n_stale = n(P_jobs, "stale") + n(A_jobs, "stale")
        if n_stale:
            print(f"    {n_stale} STALE: a complete result of a DIFFERENT run (signature mismatch) -- refused, FAILED in "
                  f"the ledger, never overwritten", flush=True)
        n_al = len(A_jobs) - n(A_jobs, "complete") - n(A_jobs, "stale")
        if n_al:
            print(f"    {n_al} alias(es) to link from their owner's identical run (not re-run): "
                  + ", ".join(sorted({f"{j['variant']} <- {j['alias_of']['experiment']}/{j['alias_of']['variant']}"
                                      for j, s in A_jobs})), flush=True)
        cells = {}
        for j, _ in P_jobs + A_jobs:
            cells.setdefault((j["variant"], j["N"], j["n_steps"]), []).append(j)
        for (v, N, ns), js in cells.items():
            print(f"    {v:<10s} N {N:<5d} n_steps {ns:<11d} T_N {ns * js[0]['cfg']['h']:.6g}  {len(js)} jobs", flush=True)
    todo = [j for k, j in prim.items() if states[k] != "complete"]
    links = [j for j, s in zip(aliases, a_states) if s != "complete"]
    if a.dry_run or not (todo or links):
        print("dry run" if a.dry_run else "nothing to run", flush=True)
        return 0

    ledgers = _Ledgers(roots)
    n_fail = 0
    by_owner = {}
    for j in links:
        by_owner.setdefault(job_key(j), []).append(j)

    def do_links(k):
        nonlocal n_fail
        for j in by_owner.pop(k, []):
            src = out_path(roots[j["alias_of"]["experiment"]], j["alias_of"])
            r = link_alias(E, j, src, out_path(roots[j["experiment"]], j))
            ledgers.write(r)
            n_fail += not r["status"].startswith("complete")
            print(f"[alias] {r['experiment']} {r['variant']} N {r['N']} seed {r['seed']} {r['method']}: {r['status']} "
                  f"({r['note']})", flush=True)

    t0 = time.time()
    try:
        for k in [k for k in by_owner if states[k] == "complete"]:
            do_links(k)
        if todo:
            warm_cache()
            from concurrent.futures import ProcessPoolExecutor, as_completed
            with ProcessPoolExecutor(max_workers=a.workers) as ex:
                futs = {ex.submit(run_one, j, roots[j["experiment"]]): job_key(j) for j in todo}
                for i, f in enumerate(as_completed(futs)):
                    r = f.result()
                    ledgers.write(r)
                    ok = r["status"].startswith("complete")
                    n_fail += not ok
                    print(f"[{i + 1}/{len(todo)}] {r['experiment']} {r['variant']} N {r['N']} seed {r['seed']} "
                          f"{r['method']}: {r['status']} {r['wall_s'] / 60:.1f} min "
                          f"(elapsed {(time.time() - t0) / 3600:.2f} h)", flush=True)
                    if ok:
                        do_links(futs[f])
        for k in list(by_owner):     # aliases whose owner did not complete: not linked, reported
            for j in by_owner.pop(k):
                r = _row(j, f"FAILED: owner {j['alias_of']['experiment']}/{j['alias_of']['variant']} did not complete",
                         0.0, float("nan"), -1, "alias, not linked")
                ledgers.write(r)
                n_fail += 1
                print(f"[alias] {r['experiment']} {r['variant']} N {r['N']} seed {r['seed']} {r['method']}: "
                      f"{r['status']}", flush=True)
    finally:
        ledgers.close()
    print(f"done ({n_fail} failed)", flush=True)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
