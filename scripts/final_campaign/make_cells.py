#!/usr/bin/env python
"""Per-cell analysis configs of Experiment A (docs/final_campaign/PREREGISTRATION.md section 1.4).

    python scripts/final_campaign/make_cells.py              # write configs/final_campaign/cells/expA/<cell>.json
    python scripts/final_campaign/make_cells.py --check      # regenerate in memory, compare with disk, write nothing
    python scripts/final_campaign/make_cells.py --validate   # run analyze_ladder's config-validation path (no data)

Reads ONLY the frozen configs/final_campaign/expA.json (never written; its sha256 is checked against the frozen value
EXPECTED_CONFIG_SHA256) and builds each cell with the mechanism campaign's own generator
(scripts/mechanism/make_cell_configs.cell_config, imported, not modified), with experiment = "expA" (the basename of
out_root), then overrides the cell's locations:

  results_dir     results/final_campaign/expA/<cell>     (scripts/final_campaign/run_cells.py writes
                                                          <results_dir>/N<N>/s<seed>_<method>.npz)
  out_dir         <cell>                                 (basename of results_dir, as eqb_family requires)
  analysis_dir    results/final_campaign/expA/<cell>/analysis
  fig_dir         figures/final_campaign/expA/<cell>
  reference_file  results/final_campaign/references/expA_<cell>_reference.npz
                  (scripts/mechanism/build_references.py --cells-root configs/final_campaign/cells
                   --out-dir results/final_campaign/references)
  ledger          results/final_campaign/expA/ledger.csv, ledger_filter {experiment: expA, variant: <cell>}

and the labels / provenance (_frozen, _generator, _wrapped_generator + its sha256, cell_label, extension).  Everything
that defines the runs (engine_cfg incl. h and the model, B, N_ladder, seeds, grids, thresholds) is cell_config's,
unchanged.

--validate runs, for every cell, the path scripts/equal_budget/analyze_ladder.py --system gateway_family takes before
it reads any data: eqb_family.check_invocation, cell_results_root / cell_analysis_dir / guard_output, analyze_ladder.plan
(n_steps = B / N exact, a multiple of 200), analyze_ladder.provenance and eqb_family.make_scorer (both need the frozen
reference: run build_references.py first), threshold_floors / tau_groups, cell_ledger, config_digest, and
eqb_family.check_plan on the meta / cfg of a tiny REAL engine run of the cell's dynamics (N 8, 200 steps) whose N and
n_steps fields are set to the production values of each N (so the engine-normalised cfg is what is compared).  Exit 1
on any problem.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
for _k in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.expanduser("~/.cache/numba_final"))

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _d in (os.path.join(ROOT, "src"), os.path.join(ROOT, "scripts", "equal_budget"),
           os.path.join(ROOT, "scripts", "mechanism")):
    if _d not in sys.path:
        sys.path.insert(0, _d)
import make_cell_configs as MC  # noqa: E402  (scripts/mechanism, unmodified)

CONFIG = os.path.join(ROOT, "configs", "final_campaign", "expA.json")
EXPECTED_CONFIG_SHA256 = "d1c6119d237b24136c21d7ba2d8220bd529189f033a6c03dfc94e2922f4263ea"   # frozen at c5a2264f
OUT_ROOT = os.path.join(ROOT, "configs", "final_campaign", "cells")
EXPERIMENT = "expA"
GENERATOR = "scripts/final_campaign/make_cells.py"
WRAPPED = "scripts/mechanism/make_cell_configs.py"
RESULTS = "results/final_campaign/expA"
FIGS = "figures/final_campaign/expA"
REFS = "results/final_campaign/references"
SYSTEM = "gateway_family"


class CellsError(RuntimeError):
    pass


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_config(path=CONFIG, check_sha=True):
    if check_sha:
        got = sha256_file(path)
        if got != EXPECTED_CONFIG_SHA256:
            raise CellsError(f"{path}: sha256 {got} != the frozen {EXPECTED_CONFIG_SHA256} (the preregistered config "
                             f"may not change; an amendment needs a new config and a reviewed generator change)")
    with open(path) as fh:
        X = json.load(fh)
    if os.path.basename(X["out_root"].rstrip("/")) != EXPERIMENT:
        raise CellsError(f"{path}: out_root {X['out_root']!r} does not end in {EXPERIMENT!r}")
    names = [v["name"] for v in X["variants"]]
    if len(set(names)) != len(names):
        raise CellsError(f"duplicate variant names {names}")
    return X


def cell_label(X, v):
    lam = float(v["lam"])
    ext = " (preregistered extension)" if v.get("extension") else ""
    return (f"{X['experiment'].split(':')[0]}: shifted fibre, kappa = {float(v['kappa']):g}, transverse mobility "
            f"lambda = {lam:g}{ext}")


def build_cell(X, v, src_path=CONFIG):
    """cell_config of the mechanism generator, with Experiment A's locations, labels and provenance."""
    P = MC.cell_config(EXPERIMENT, X, src_path, v, rerun_reuse=False)
    name = v["name"]
    if P["reuse"] is not None:
        raise CellsError(f"{name}: Experiment A cells never reuse files")
    want_results = f"{RESULTS}/{name}"
    if P["results_dir"] != want_results:
        raise CellsError(f"{name}: generator results_dir {P['results_dir']!r} != {want_results!r}")
    Q = copy.deepcopy(P)
    Q.update(
        _frozen=(f"GENERATED by {GENERATOR} (wrapping {WRAPPED} cell_config) from "
                 f"{os.path.relpath(src_path, ROOT)} (frozen experiment config); regenerate, never edit by hand"),
        _generator=GENERATOR,
        _wrapped_generator=WRAPPED,
        _wrapped_generator_sha256=sha256_file(os.path.join(ROOT, WRAPPED)),
        cell_label=cell_label(X, v),
        out_dir=name,
        results_dir=want_results,
        analysis_dir=f"{RESULTS}/{name}/analysis",
        fig_dir=f"{FIGS}/{name}",
        reference_file=f"{REFS}/{EXPERIMENT}_{name}_reference.npz",
        ledger=f"{RESULTS}/ledger.csv",
        ledger_filter=dict(experiment=EXPERIMENT, variant=name),
    )
    if v.get("extension"):
        Q["extension"] = True
    return Q


def build_all(X=None, src_path=CONFIG):
    X = X or load_config(src_path)
    return {v["name"]: build_cell(X, v, src_path) for v in X["variants"]}


def cell_path(out_root, name):
    return os.path.join(out_root, EXPERIMENT, f"{name}.json")


def dumps(P):
    return json.dumps(P, indent=1) + "\n"


# ------------------------------------------------------------------------------------------------- validation
def validate_cell(P, cfg_path, tiny_run=True, need_reference=True):
    """analyze_ladder's pre-data path for one cell; returns a list of problems ([] = accepted)."""
    import eqb_metrics as M
    import eqb_family as MF
    import analyze_ladder as AL
    probs = []

    def step(name, fn):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            probs.append(f"{name}: {type(e).__name__}: {e}")
            return None

    step("check_invocation", lambda: MF.check_invocation(SYSTEM, P, cfg_path))
    if MF.is_cell(P) is not True:
        probs.append("not a cell config (eqb_family.is_cell)")
    root = step("cell_results_root", lambda: MF.cell_results_root(P))
    if root is not None and os.path.abspath(root) != os.path.join(ROOT, RESULTS):
        probs.append(f"cell_results_root {root} != {os.path.join(ROOT, RESULTS)}")
    out = step("cell_analysis_dir", lambda: MF.cell_analysis_dir(P))
    if out is not None:
        step("guard_output", lambda: MF.guard_output(P, out, SYSTEM))
    rows = step("plan", lambda: AL.plan(P))
    if rows is not None:
        for r in rows:
            if r["n_steps"] % M.N_UNIFORM:
                probs.append(f"N {r['N']}: n_steps {r['n_steps']} is not a multiple of {M.N_UNIFORM}")
        want_T = {int(k): float(v) for k, v in P["T_N"].items()}
        for r in rows:
            if abs(r["T"] - want_T[r["N"]]) > 1e-9 * want_T[r["N"]]:
                probs.append(f"N {r['N']}: plan T {r['T']} != T_N {want_T[r['N']]}")
    step("cell_ledger", lambda: MF.cell_ledger(P))
    step("config_digest", lambda: MF.config_digest(P))
    step("tau_groups", lambda: MF.tau_groups(SYSTEM, P))
    if MF.system_kind(SYSTEM) != "gateway":
        probs.append(f"system kind of {SYSTEM} is {MF.system_kind(SYSTEM)!r}, not 'gateway'")
    ref = os.path.join(ROOT, P["reference_file"])
    if os.path.exists(ref):
        step("provenance", lambda: AL.provenance(SYSTEM, P, cfg_path))
        sc = step("make_scorer", lambda: MF.make_scorer(SYSTEM, P))
        if sc is not None:
            step("threshold_floors", lambda: MF.threshold_floors(SYSTEM, P, sc))
            step("tau_groups(floors)", lambda: MF.tau_groups(SYSTEM, P, MF.threshold_floors(SYSTEM, P, sc)))
    elif need_reference:
        probs.append(f"reference {P['reference_file']} does not exist (run scripts/mechanism/build_references.py "
                     f"--cells-root configs/final_campaign/cells --out-dir {REFS})")
    if tiny_run and rows is not None:
        import gateway_family_numba as E
        cfg = dict(P["engine_cfg"])
        res = E.run_arm(cfg, "fr", 8, 8390, 200, [100, 200])
        for r in rows:
            for method in r["methods"]:
                meta = dict(res["meta"], N=r["N"], n_steps=r["n_steps"], seed=8300, method=method)
                step(f"check_plan N {r['N']} {method}",
                     lambda meta=meta, r=r, method=method: MF.check_plan(
                         dict(meta=meta, cfg=res["cfg"]), P, f"{P['cell']}/N{r['N']}/s8300_{method}.npz",
                         dict(N=r["N"], seed=8300, method=method)))
        # and a mismatching run must be REFUSED (the validation path is live, not vacuous)
        bad = dict(res["meta"], N=rows[0]["N"], n_steps=rows[0]["n_steps"])
        other = dict(res["cfg"], lam=float(P["engine_cfg"]["lam"]) * 2.0)
        try:
            MF.check_plan(dict(meta=dict(bad, lam=other["lam"]), cfg=other), P, "negative control")
            probs.append("check_plan accepted a run with a different lambda (negative control)")
        except M.MetricsError:
            pass
    return probs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", default=OUT_ROOT)
    ap.add_argument("--check", action="store_true", help="compare with the files on disk, write nothing")
    ap.add_argument("--validate", action="store_true",
                    help="run analyze_ladder's config-validation path on the files on disk (no data)")
    ap.add_argument("--no-tiny-run", action="store_true", help="--validate without the tiny engine run")
    a = ap.parse_args(argv)
    cells = build_all()
    if a.validate:
        bad = 0
        for name, P in cells.items():
            p = cell_path(a.out_root, name)
            if not os.path.exists(p):
                print(f"{name}: MISSING {p}")
                bad += 1
                continue
            with open(p) as fh:
                txt = fh.read()
            if txt != dumps(P):
                print(f"{name}: {p} differs from the generator output (regenerate)")
                bad += 1
                continue
            probs = validate_cell(json.loads(txt), p, tiny_run=not a.no_tiny_run)
            print(f"{name}: {'ACCEPTED' if not probs else 'PROBLEMS'} by the analyze_ladder --system {SYSTEM} "
                  f"config path" + ("" if not probs else ": " + "; ".join(probs)), flush=True)
            bad += bool(probs)
        print("validation PASS" if not bad else f"validation FAIL ({bad} cell(s))")
        return 1 if bad else 0
    diff = []
    for name, P in cells.items():
        p = cell_path(a.out_root, name)
        txt = dumps(P)
        if a.check:
            if not os.path.exists(p) or open(p).read() != txt:
                diff.append(p)
            continue
        if os.path.exists(p) and open(p).read() != txt:
            print(f"{os.path.relpath(p, ROOT)}: replaced (generator output changed)", flush=True)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p + ".tmp", "w") as fh:
            fh.write(txt)
        os.replace(p + ".tmp", p)
        print(f"{os.path.relpath(p, ROOT)}: N {P['N_ladder']}, B {P['B']:,}, h {P['engine_cfg']['h']:g}, lam "
              f"{P['engine_cfg']['lam']:g}, {P['engine_version']}, results {P['results_dir']}, reference "
              f"{P['reference_file']}", flush=True)
    if a.check:
        print("cell configs up to date" if not diff else f"{len(diff)} cell config(s) differ: {diff}")
        return 1 if diff else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
