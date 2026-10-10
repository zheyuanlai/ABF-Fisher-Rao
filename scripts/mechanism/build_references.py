#!/usr/bin/env python
"""Freeze the references of the mechanism cells (docs/mechanism/SCIENTIFIC_PLAN.md section 6).

    python scripts/mechanism/build_references.py [--cells-root configs/mechanism/cells] [--cell EXP/VARIANT ...]
                                                 [--out-dir DIR] [--force] [--check]

For every cell config configs/mechanism/cells/<experiment>/<variant>.json (scripts/mechanism/make_cell_configs.py)
writes its reference_file (default results/mechanism/references/<experiment>_<variant>_reference.npz; --out-dir puts
the files elsewhere, keeping the basenames) and references_summary.json next to them (sha256 of every file, its
model, the primary-reference digest; records of other cells already in it are kept).  Independent of every run: the arrays are functions of the cell's
engine_cfg only (eqb_family.family_reference_arrays):

  x_grid, eval_mask, F_ref, Fp_ref   PRIMARY: the accepted scorer's analytic F*, F*' (scripts/analyze_gateway_replica_
                                     ladder.Scorer(cell, 'analytic')), identical for every variant and lambda; ASSERTED
                                     bitwise equal across all cells and to results/equal_budget_v2/references/
                                     gateway_reference.npz
  F_ref_em                           SECONDARY: F from the per-dynamics frozen-x EM-consistent mean force
                                     (eqb_family.family_mean_force; alpha family 4Hx(x^2-1) + ((1-alpha)/beta) w'/w +
                                     (alpha/beta)(w'/w)/(1 - lam w^(2 alpha) h/2), shifted fibre F*'), the equal-budget
                                     EM construction; at alpha 1, lam 1 ASSERTED bitwise equal to the equal-budget
                                     F_ref_em (same h)
  Fp_bin_ref, Fp_bin_em              both mean forces averaged over the scorer's Gauss-Legendre sub-grid of the 180 bins
  h, beta, H, omega_out, omega_in, s, variant, alpha, kappa, lam (NaN = None), model_json, units, gauge,
  secondary_definition, primary_source, primary_source_sha256, primary_digest, cell_config
A reference that already exists is FROZEN: it is recomputed and compared (arrays within 1e-12, model and h exact)
and left untouched; a difference is an error unless --force (which rewrites it; record why).  --check writes nothing
and exits 1 on any missing or different reference.  eqb_family.GatewayFamilyScorer re-verifies every file at
construction, so a stale reference stops the analysis.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _k in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import numpy as np  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "equal_budget"))
import eqb_metrics as M  # noqa: E402
import eqb_family as MF  # noqa: E402

CELLS_ROOT = os.path.join(ROOT, "configs", "mechanism", "cells")
SCRIPT = "scripts/mechanism/build_references.py"


class ReferenceError_(RuntimeError):
    pass


def load_cells(cells_root, only=None):
    out = {}
    for p in sorted(glob.glob(os.path.join(cells_root, "*", "*.json"))):
        P = json.load(open(p))
        if not MF.is_cell(P):
            continue
        key = f"{P['experiment']}/{P['cell']}"
        if only and key not in only:
            continue
        out[key] = (p, P)
    if only:
        miss = sorted(set(only) - set(out))
        if miss:
            raise ReferenceError_(f"cells not found under {cells_root}: {miss}")
    return out


def nan_if_none(v):
    return math.nan if v is None else float(v)


def reference_payload(P, cfg_path):
    """(arrays to save, summary record) of one cell."""
    arr, cell, model = MF.family_reference_arrays(P["engine_cfg"])
    if model != {k: P["model"][k] for k in MF.FAMILY_MODEL_KEYS}:
        raise ReferenceError_(f"{cfg_path}: engine_cfg model {model} != config model {P['model']}")
    eqb_path = os.path.join(M.REF_DIR, "gateway_reference.npz")
    with np.load(eqb_path, allow_pickle=False) as z:
        eqb = {k: z[k] for k in z.files}
    for k in MF.PRIMARY_REF_ARRAYS:
        if not np.array_equal(arr[k], eqb[k]):
            raise ReferenceError_(f"{cfg_path}: primary {k} is not bitwise the equal-budget gateway reference")
    orig = None
    if MF.is_original_dynamics(model) and float(eqb["h"]) == float(cell["dt"]):
        orig = bool(np.array_equal(arr["F_ref_em"], eqb["F_ref_em"]))
        if not orig:
            raise ReferenceError_(f"{cfg_path}: alpha 1, lam 1 secondary is not bitwise the equal-budget F_ref_em")
    m = arr["eval_mask"].astype(bool)
    d = (arr["F_ref_em"] - arr["F_ref"])[m]
    out = dict(arr)
    out.update(h=float(cell["dt"]), beta=cell["beta"], H=cell["H"], omega_out=cell["omega_out"],
               omega_in=cell["omega_in"], s=cell["s"], variant=model["variant"], alpha=nan_if_none(model["alpha"]),
               kappa=nan_if_none(model["kappa"]), lam=float(model["lam"]),
               model_json=json.dumps(model, sort_keys=True), units="model energy units (beta = 16); F' per unit x",
               # the frozen files (2026-10-10) carry this exact text; the class now lives in eqb_family.py (moved so
               # that eqb_metrics.py stays byte-identical to the equal-budget scoring code), the arrays are unchanged
               gauge=("F_ref and F_ref_em centred on the eval window [-1.5, 1.5] (151 grid nodes); errors via "
                      "eqb_metrics.GatewayFamilyScorer"),
               secondary_definition=MF.FAMILY_SECONDARY_DEFINITION,
               primary_source=os.path.relpath(eqb_path, ROOT), primary_source_sha256=M.sha256_file(eqb_path),
               primary_digest=MF.primary_digest(arr), cell_config=os.path.relpath(cfg_path, ROOT),
               built_by=SCRIPT)
    rec = dict(cell=f"{P['experiment']}/{P['cell']}", model=model, h=float(cell["dt"]),
               primary_digest=MF.primary_digest(arr), secondary_equals_equal_budget_em=orig,
               em_floor_F_rms=float(np.sqrt(np.mean((d - d.mean()) ** 2))),
               max_abs_Fp_bin_em_minus_ref=float(np.max(np.abs(arr["Fp_bin_em"] - arr["Fp_bin_ref"]))))
    return out, rec


def compare(path, payload):
    """Differences between the frozen file and a fresh payload ([] = identical within 1e-12)."""
    with np.load(path, allow_pickle=False) as z:
        old = {k: z[k] for k in z.files}
    diffs = []
    for k, v in payload.items():
        if k not in old:
            diffs.append(f"{k} missing")
            continue
        a, b = np.asarray(old[k]), np.asarray(v)
        if a.dtype.kind in "US" or b.dtype.kind in "US":
            if str(a) != str(b) and k not in ("built_by",):
                diffs.append(f"{k} differs")
        elif a.shape != b.shape or not np.allclose(a, b, rtol=0, atol=1e-12, equal_nan=True):
            diffs.append(f"{k} differs (max {float(np.nanmax(np.abs(a - b))) if a.shape == b.shape else 'shape'})")
    return diffs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells-root", default=CELLS_ROOT)
    ap.add_argument("--cell", nargs="*", default=None, help="EXPERIMENT/VARIANT (default: every cell)")
    ap.add_argument("--out-dir", default=None, help="write here instead of each config's reference_file directory")
    ap.add_argument("--force", action="store_true", help="rewrite a frozen reference that differs")
    ap.add_argument("--check", action="store_true", help="write nothing; exit 1 on a missing / different reference")
    a = ap.parse_args(argv)
    cells = load_cells(a.cells_root, a.cell)
    if not cells:
        raise ReferenceError_(f"no cell config under {a.cells_root} (scripts/mechanism/make_cell_configs.py)")
    digests, records, bad, ref_dirs = set(), {}, [], set()
    for key, (cfg_path, P) in cells.items():
        payload, rec = reference_payload(P, cfg_path)
        digests.add(rec["primary_digest"])
        path = (os.path.join(os.path.abspath(a.out_dir), os.path.basename(P["reference_file"])) if a.out_dir
                else MF._abs(P["reference_file"]))
        ref_dirs.add(os.path.dirname(path))
        status = "new"
        if os.path.exists(path):
            diffs = compare(path, payload)
            status = "unchanged" if not diffs else "DIFFERS"
            if diffs:
                bad.append(f"{key}: {diffs[:4]}")
        if not a.check and (status == "new" or (status == "DIFFERS" and a.force)):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = path + f".tmp{os.getpid()}.npz"
            np.savez(tmp, **{k: np.asarray(v) for k, v in payload.items()})
            os.replace(tmp, path)
            status = "written" if status == "new" else "REWRITTEN (--force)"
        rec.update(path=os.path.relpath(path, ROOT) if path.startswith(ROOT + os.sep) else path,
                   sha256=M.sha256_file(path) if os.path.exists(path) else None, status=status)
        records[key] = rec
        print(f"{key}: {status} {rec['path']} sha256 {str(rec['sha256'])[:16]} em_floor_F_rms "
              f"{rec['em_floor_F_rms']:.3g} max|<F'_em> - <F'*>| {rec['max_abs_Fp_bin_em_minus_ref']:.3g}", flush=True)
    if len(digests) != 1:
        raise ReferenceError_(f"the primary reference differs between cells: {len(digests)} digests")
    if not a.check:
        if len(ref_dirs) != 1:
            raise ReferenceError_(f"the references of these cells live in several directories {sorted(ref_dirs)}")
        out_dir = ref_dirs.pop()           # the summary sits next to the files it describes
        summ_path = os.path.join(out_dir, "references_summary.json")
        old = {}
        if os.path.exists(summ_path):
            try:
                old = json.load(open(summ_path)).get("references", {})
            except Exception:  # noqa: BLE001
                old = {}
        old.update(records)
        doc = dict(schema="mechanism_references/1", script=SCRIPT, updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   primary_digest=next(iter(digests)),
                   primary_source=os.path.relpath(os.path.join(M.REF_DIR, "gateway_reference.npz"), ROOT),
                   primary_source_sha256=M.sha256_file(os.path.join(M.REF_DIR, "gateway_reference.npz")),
                   secondary_definition=MF.FAMILY_SECONDARY_DEFINITION, references=old)
        os.makedirs(out_dir, exist_ok=True)
        with open(summ_path + ".tmp", "w") as fh:
            json.dump(M.json_safe(doc), fh, indent=1, allow_nan=False)
        os.replace(summ_path + ".tmp", summ_path)
        print(f"wrote {summ_path}", flush=True)
    if bad:
        print(("CHECK FAILED" if a.check else ("rewritten with --force" if a.force else "REFUSED (frozen; use --force)"))
              + f": {bad}", file=sys.stderr, flush=True)
        return 1 if (a.check or not a.force) else 0
    if a.check:
        missing = [k for k, r in records.items() if r["status"] == "new"]
        if missing:
            print(f"CHECK FAILED: missing references {missing}", file=sys.stderr, flush=True)
            return 1
        print("references up to date", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
