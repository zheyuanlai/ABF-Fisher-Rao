#!/usr/bin/env python
"""Per-cell analysis configs of the mechanism campaign (docs/mechanism/SCIENTIFIC_PLAN.md sections 5-6).

    python scripts/mechanism/make_cell_configs.py [--out-root configs/mechanism/cells] [--rerun-reuse-cells]
                                                  [--check]

Reads the FROZEN experiment configs configs/mechanism/{matched_free_energy,conditional_relaxation}.json (never
written) and writes one config per cell in the equal-budget config format (configs/equal_budget_v2/*_production.json
schema, so that scripts/equal_budget/{analyze_ladder,plot_config,plot_synthesis,audit_completeness}.py run on it with
``--system gateway_family --config <cell>.json``):

    configs/mechanism/cells/<experiment>/<variant>.json      experiment = basename of the config's out_root

Fields (equal-budget format + the cell's own locations and model):
  system "gateway_family", experiment, cell (= variant name), cell_label, out_dir (= basename of results_dir: the
  analysis reads <dirname(results_dir)>/<out_dir>/N<N>/s<seed>_<abf|fr>.npz), results_dir, analysis_dir, fig_dir,
  reference_file (results/mechanism/references/<experiment>_<variant>_reference.npz, built by build_references.py),
  engine, engine_version (the ONLY engine accepted for the cell's runs), reuse (null or the reuse record),
  engine_cfg (the experiment's engine_cfg + variant, alpha, kappa, lam; h = h_frozen if the experiment config has one,
  else engine_cfg.h), model, B = N0 T0 / h (anchor N0 2048, T0 40: 2048 x 40 / h), N_ladder = N_coarse = the
  experiment's N list, seeds, init, noise, physical_checkpoints_t, profile_snapshot_u, thresholds, e_Fp_floor,
  statistics, checkpoint_every_s, est_us_per_walker_step.
Thresholds: e_F (frozen gateway values); e_Fp = the experiment's e_Fp_raw_reported_only (raw e_F', the equal-budget
gateway e_F' thresholds; raw e_F' is reported, and per plan section 6 no tau is computed at a threshold below its
0.03227 floor: strict 0.012 gets none, mid / loose are secondary tau); e_Fp_stat = the tau thresholds on the
floor-free e_F'_stat (plan section 6: the cell's e_F' tau endpoint, in the primary tau group); TV_half.

Ledger: ledger = results/mechanism/<experiment>/ledger.csv (scripts/mechanism/run_cells.py writes ONE ledger per
experiment, columns experiment, variant, N, seed, method, ...) with ledger_filter {experiment, variant} (the audit
keeps only this cell's rows); a reuse cell: the equal-budget results/equal_budget_v2/gateway/ledger.csv, no filter.

Reuse cells (alpha1 of Experiment I, lam1 of Experiment II): results_dir = results/equal_budget_v2/gateway, engine
gateway_ladder_numba/2 (those files carry no model fields: the analysis reads a missing field as variant alpha,
alpha 1, kappa None, lam 1).  Allowed only when the experiment's h, B, seeds, save-grid specification and every
engine knob equal configs/equal_budget_v2/gateway_production.json (checked here; any difference is an error).  The
plan makes the reuse conditional on the new engine reproducing those files bitwise (section 5): the reuse record
names the gate record results/mechanism/reuse_gate/reuse_gate.json (scripts/mechanism/reuse_gate.py; >= 4 jobs,
both arms, N 2048 and 128), and the audit (REUSE_GATE) and cross_cell.py refuse a reuse cell until that record
licenses it (eqb_family.reuse_gate_problems).  If the gate fails, regenerate with --rerun-reuse-cells: those cells
then read results/mechanism/<experiment>/<variant> with engine gateway_family_numba/1 (and its experiment ledger).
The analysis and figure locations of a reuse cell are its own (results/mechanism/..., figures/mechanism/...), never
results/equal_budget_v2 or figures/equal_budget_v2; a summary left over from the other mode is refused by
cross_cell.py (engine_version / results_dir / reference / model / code digest / run-cache freshness).

--check   regenerate in memory and compare with the files on disk (exit 1 on any difference); writes nothing.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
EXPERIMENTS = ("matched_free_energy", "conditional_relaxation")
SRC_DIR = os.path.join(ROOT, "configs", "mechanism")
OUT_ROOT = os.path.join(SRC_DIR, "cells")
EQB_GATEWAY = os.path.join(ROOT, "configs", "equal_budget_v2", "gateway_production.json")
REUSE_RESULTS = "results/equal_budget_v2/gateway"
REUSE_LEDGER = "results/equal_budget_v2/gateway/ledger.csv"
REUSE_GATE_RECORD = "results/mechanism/reuse_gate/reuse_gate.json"
REUSE_GATE_N = [2048, 128]          # plan section 5: >= 4 re-run jobs, both arms, N = 128 and 2048
REUSE_GATE_MIN_JOBS = 4
ENGINE_REUSE = "gateway_ladder_numba/2"
ENGINE_FAMILY = "gateway_family_numba/1"
MODEL_KEYS = ("variant", "alpha", "kappa", "lam")
GENERATOR = "scripts/mechanism/make_cell_configs.py"
N_UNIFORM = 200
LABELS = {
    "alpha1": "alpha = 1 (original gateway: entropic barrier share 3.47 kT)",
    "alpha0.5": "alpha = 0.5 (half the log-omega entropy moved into the energy; Var(f|x) / 4)",
    "alpha0": "alpha = 0 (purely energetic barrier in x; deterministic force)",
    "shift": "shifted fibre, kappa = 1 (variance-matched control; constant width, curved slow fibre)",
    "lam0.1": "lambda = 0.1 (transverse mobility / 10)",
    "lam0.25": "lambda = 0.25 (transverse mobility / 4)",
    "lam0.5": "lambda = 0.5 (transverse mobility / 2)",
    "lam1": "lambda = 1 (original gateway)",
}


class CellConfigError(RuntimeError):
    pass


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def frozen_h(X):
    """The common h of the experiment: h_frozen when the experiment config has one, else engine_cfg.h."""
    return float(X["h_frozen"]) if X.get("h_frozen") is not None else float(X["engine_cfg"]["h"])


def budget(X, h):
    """B = N0 T0 / h walker-steps per arm per seed (2048 x 40 / h), exact integer; every n_steps = B / N an integer
    multiple of 200 (the 200 uniform budget fractions must be exact saves)."""
    N0, T0 = int(X["anchor"]["N0"]), float(X["anchor"]["T0"])
    Bf = N0 * T0 / h
    B = int(round(Bf))
    if abs(B - Bf) > 1e-6 * max(1.0, Bf):
        raise CellConfigError(f"B = {N0} x {T0} / {h} = {Bf} is not an integer number of walker-steps")
    for N in X["N"]:
        N = int(N)
        if B % N or (B // N) % N_UNIFORM:
            raise CellConfigError(f"B {B} / N {N} is not an integer multiple of {N_UNIFORM} steps")
        want = (X.get("T_N") or {}).get(str(N))
        if want is not None and abs((B // N) * h - float(want)) > 1e-9 * float(want):
            raise CellConfigError(f"T_N({N}) = {(B // N) * h} != the experiment's T_N {want}")
    return B


def check_reuse(X, h, B):
    """A reuse cell reads the equal-budget gateway production files: everything that defines a run must agree."""
    G = json.load(open(EQB_GATEWAY))
    probs = []
    if float(G["engine_cfg"]["h"]) != h:
        probs.append(f"h {h} != equal-budget gateway h {G['engine_cfg']['h']}")
    if int(G["B"]) != B:
        probs.append(f"B {B} != equal-budget gateway B {G['B']}")
    if [int(s) for s in G["seeds"]] != [int(s) for s in X["seeds"]]:
        probs.append("seeds differ from the equal-budget gateway seeds")
    for k in ("physical_checkpoints_t", "profile_snapshot_u"):
        if [float(v) for v in G[k]] != [float(v) for v in X[k]]:
            probs.append(f"{k} differ")
    miss = [int(N) for N in X["N"] if int(N) not in [int(n) for n in G["N_ladder"]]]
    if miss:
        probs.append(f"N {miss} not in the equal-budget gateway ladder")
    ge, xe = G["engine_cfg"], X["engine_cfg"]
    for k in sorted(set(ge) | set(xe)):
        if k in ("system", "h"):
            continue
        if ge.get(k, "<absent>") != xe.get(k, "<absent>"):
            probs.append(f"engine knob {k}: equal-budget {ge.get(k, '<absent>')!r} vs experiment {xe.get(k, '<absent>')!r}")
    if [float(v) for v in G["thresholds"]["e_F"]] != [float(v) for v in X["thresholds"]["e_F"]]:
        probs.append("e_F thresholds differ")
    if probs:
        raise CellConfigError("reuse of results/equal_budget_v2/gateway is not allowed: " + "; ".join(probs)
                              + " (regenerate with --rerun-reuse-cells)")
    return dict(eqb_config=os.path.relpath(EQB_GATEWAY, ROOT), eqb_config_sha256=sha256_file(EQB_GATEWAY),
                eqb_B=int(G["B"]), eqb_h=float(G["engine_cfg"]["h"]))


def model_of(v):
    m = {k: v.get(k) for k in MODEL_KEYS}
    if m["variant"] not in ("alpha", "shift"):
        raise CellConfigError(f"variant {m['variant']!r} is neither 'alpha' nor 'shift'")
    if m["variant"] == "alpha" and (m["alpha"] is None or m["kappa"] is not None):
        raise CellConfigError(f"alpha variant {v['name']} needs alpha and kappa null")
    if m["variant"] == "shift" and (m["kappa"] is None or m["alpha"] is not None):
        raise CellConfigError(f"shift variant {v['name']} needs kappa and alpha null")
    if not (m["lam"] is not None and float(m["lam"]) > 0):
        raise CellConfigError(f"lambda of {v['name']} must be > 0")
    out = dict(variant=m["variant"], alpha=None if m["alpha"] is None else float(m["alpha"]),
               kappa=None if m["kappa"] is None else float(m["kappa"]), lam=float(m["lam"]))
    return out


def is_original(model):
    return model["variant"] == "alpha" and model["alpha"] == 1.0 and model["lam"] == 1.0


def cell_config(experiment, X, src_path, v, rerun_reuse=False):
    h = frozen_h(X)
    B = budget(X, h)
    model = model_of(v)
    name = v["name"]
    reuse_requested = bool(v.get("reuse"))
    if reuse_requested and not is_original(model):
        raise CellConfigError(f"{experiment}/{name}: only the original dynamics (alpha 1, lam 1) may reuse files")
    reuse = None
    if reuse_requested and not rerun_reuse:
        rec = check_reuse(X, h, B)
        reuse = dict(source=REUSE_RESULTS, engine=ENGINE_REUSE, condition=v["reuse"],
                     missing_model_fields=dict(variant="alpha", alpha=1.0, kappa=None, lam=1.0), **rec,
                     gate_record=REUSE_GATE_RECORD, gate_required_N=list(REUSE_GATE_N),
                     gate_min_jobs=REUSE_GATE_MIN_JOBS,
                     note=("valid only if the new engine reproduces these files bitwise (SCIENTIFIC_PLAN section 5): "
                           "gate_record (scripts/mechanism/reuse_gate.py) must record a PASS; otherwise regenerate the "
                           "cell configs with --rerun-reuse-cells"))
        results_dir = REUSE_RESULTS
        engine_version = ENGINE_REUSE
        engine = "src/gateway_ladder_numba.py run_job (equal-budget production files, reused)"
        ledger, ledger_filter = REUSE_LEDGER, None
    else:
        # scripts/mechanism/run_cells.py writes <out_root>/<variant>/N<N>/s<seed>_<method>.npz
        results_dir = f"{X['out_root'].rstrip('/')}/{name}"
        engine_version = ENGINE_FAMILY
        engine = "src/gateway_family_numba.py run_job"
        # scripts/mechanism/run_cells.py: <out_root>/ledger.csv, ONE per experiment, rows tagged experiment + variant
        ledger, ledger_filter = f"{X['out_root'].rstrip('/')}/ledger.csv", dict(experiment=experiment, variant=name)
    ecfg = copy.deepcopy(X["engine_cfg"])
    ecfg["h"] = h
    ecfg.update(model)
    th = X["thresholds"]
    thresholds = dict(e_F=[float(x) for x in th["e_F"]], e_Fp=[float(x) for x in th["e_Fp_raw_reported_only"]],
                      e_Fp_stat=[float(x) for x in th["e_Fp_stat"]], TV_half=float(th["TV_half"]))
    Ns = [int(n) for n in X["N"]]
    P = dict(
        _frozen=(f"GENERATED by {GENERATOR} from {os.path.relpath(src_path, ROOT)} (frozen experiment config); "
                 "regenerate, never edit by hand"),
        _generated_from=os.path.relpath(src_path, ROOT), _generated_from_sha256=sha256_file(src_path),
        _generator=GENERATOR,
        system="gateway_family",
        experiment=experiment, experiment_label=X["experiment"], cell=name,
        cell_label=f"{X['experiment'].split(':')[0]}: {LABELS.get(name, name)}",
        out_dir=os.path.basename(results_dir),
        results_dir=results_dir,
        analysis_dir=f"results/mechanism/{experiment}/{name}/analysis",
        fig_dir=f"figures/mechanism/{experiment}/{name}",
        reference_file=f"results/mechanism/references/{experiment}_{name}_reference.npz",
        ledger=ledger, ledger_filter=ledger_filter,
        engine=engine, engine_version=engine_version, reuse=reuse,

        engine_cfg=ecfg, model=model,
        h_status=X.get("h_status"), h_source=("h_frozen" if X.get("h_frozen") is not None else "engine_cfg.h"),
        anchor=dict(N0=int(X["anchor"]["N0"]), T0=float(X["anchor"]["T0"])),
        B=B, B_rule=X.get("B_rule"),
        N_ladder=Ns, N_coarse=list(Ns),
        T_N={str(N): (B // N) * h for N in Ns},
        seeds=[int(s) for s in X["seeds"]],
        init=X.get("init"), noise=X.get("noise"),
        physical_checkpoints_t=list(X["physical_checkpoints_t"]),
        profile_snapshot_u=list(X["profile_snapshot_u"]),
        reference=X.get("reference"),
        thresholds=thresholds,
        thresholds_note=("e_F: frozen gateway values; e_Fp: the experiment's e_Fp_raw_reported_only (raw e_F', "
                         "reported; no tau at a threshold below the e_Fp_floor (strict), mid / loose tau secondary); "
                         "e_Fp_stat: tau thresholds on the floor-free e_F'_stat, the cell's e_F' tau endpoint (plan "
                         "section 6); TV_half"),
        e_Fp_floor=float(X["e_Fp_floor"]),
        statistics=X.get("statistics"),
        checkpoint_every_s=float(X.get("checkpoint_every_s", 600.0)),
        est_us_per_walker_step=0.09,
    )
    return P


def build_all(rerun_reuse=False):
    out = {}
    for experiment in EXPERIMENTS:
        src = os.path.join(SRC_DIR, f"{experiment}.json")
        X = json.load(open(src))
        if os.path.basename(X["out_root"].rstrip("/")) != experiment:
            raise CellConfigError(f"{src}: out_root {X['out_root']!r} does not end in {experiment!r}")
        names = [v["name"] for v in X["variants"]]
        if len(set(names)) != len(names):
            raise CellConfigError(f"{src}: duplicate variant names {names}")
        for v in X["variants"]:
            out[(experiment, v["name"])] = cell_config(experiment, X, src, v, rerun_reuse)
    return out


def cell_path(out_root, experiment, name):
    return os.path.join(out_root, experiment, f"{name}.json")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", default=OUT_ROOT)
    ap.add_argument("--rerun-reuse-cells", action="store_true",
                    help="alpha1 / lam1 read new-engine runs under results/mechanism instead of the equal-budget files")
    ap.add_argument("--check", action="store_true", help="compare with the files on disk, write nothing")
    a = ap.parse_args(argv)
    cells = build_all(a.rerun_reuse_cells)
    bad = []
    for (experiment, name), P in cells.items():
        p = cell_path(a.out_root, experiment, name)
        txt = json.dumps(P, indent=1) + "\n"
        if a.check:
            if not os.path.exists(p) or open(p).read() != txt:
                bad.append(p)
            continue
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p + ".tmp", "w") as fh:
            fh.write(txt)
        os.replace(p + ".tmp", p)
        print(f"{os.path.relpath(p, ROOT)}: N {P['N_ladder']}, B {P['B']:,}, h {P['engine_cfg']['h']:g}, "
              f"{P['engine_version']}, results {P['results_dir']}", flush=True)
    if a.check:
        print("cell configs up to date" if not bad else f"{len(bad)} cell config(s) differ: {bad}")
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
