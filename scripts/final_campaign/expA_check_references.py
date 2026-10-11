#!/usr/bin/env python
"""Check 2 of Experiment A (docs/final_campaign/PREREGISTRATION.md section 1.3): F*, F*' identical across lambda.

    python scripts/final_campaign/expA_check_references.py [--refs-dir results/final_campaign/references]
        [--out results/final_campaign/expA/checks/check2_references.json] [--dry-run]

Reads the four frozen cell references written by
    scripts/mechanism/build_references.py --cells-root configs/final_campaign/cells --out-dir results/final_campaign/references
(one per cell config configs/final_campaign/cells/expA/<cell>.json, reference_file) and checks, bitwise
(np.array_equal, same dtype and shape):
  * the PRIMARY arrays x_grid, eval_mask, F_ref, Fp_ref (eqb_family.PRIMARY_REF_ARRAYS) of every cell equal each other
    and equal results/equal_budget_v2/references/gateway_reference.npz   (the pass rule);
  * descriptive: the secondary arrays F_ref_em, Fp_bin_ref, Fp_bin_em across cells (the shifted fibre's EM-consistent
    mean force is F*' exactly at every lambda, so these are expected equal too), F_ref_em vs the equal-budget F_ref_em,
    the stored model / h of every file, and the single primary digest of references_summary.json.
Writes the JSON record (sha256 of every file read) and exits 1 on any failure.  Read-only otherwise; never rebuilds a
reference.  --dry-run prints the verdict without writing.  Idempotent: the record carries an input signature
(``input_signature``: sha256 over every file the check reads -- the equal-budget reference, the four cell configs,
their reference files, references_summary.json -- and this tool); an existing record with the same signature is kept
('up to date'), one with a different (or no) signature is never overwritten without --force (exit 2).  The production
gate of run_cells.py recomputes the signature (``record_is_current``), so a reference rebuilt after the check makes the
record stale.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
CELLS_DIR = os.path.join(ROOT, "configs", "final_campaign", "cells", "expA")
REFS = os.path.join(ROOT, "results", "final_campaign", "references")
EQB = os.path.join(ROOT, "results", "equal_budget_v2", "references", "gateway_reference.npz")
OUT = os.path.join(ROOT, "results", "final_campaign", "expA", "checks", "check2_references.json")
PRIMARY = ("x_grid", "eval_mask", "F_ref", "Fp_ref")
SECONDARY = ("F_ref_em", "Fp_bin_ref", "Fp_bin_em")
CELLS = ("shift_lam1", "shift_lam4", "shift_lam16", "shift_lam128")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _git_commit():
    import subprocess
    try:
        return subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=20).stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return bool(a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b))


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def input_files(refs_dir=REFS, cells_dir=CELLS_DIR, eqb_path=EQB, cells=CELLS):
    """Every file the check reads (absolute; a reference named by a missing cell config cannot be listed)."""
    out = [eqb_path, os.path.join(refs_dir, "references_summary.json")]
    for c in cells:
        cfg_path = os.path.join(cells_dir, f"{c}.json")
        out.append(cfg_path)
        try:
            with open(cfg_path) as fh:
                out.append(os.path.join(refs_dir, os.path.basename(json.load(fh)["reference_file"])))
        except (OSError, KeyError, ValueError):
            pass
    return [os.path.abspath(p) for p in out]


def input_signature(refs_dir=REFS, cells_dir=CELLS_DIR, eqb_path=EQB, cells=CELLS):
    h = hashlib.sha256()
    for p in sorted(input_files(refs_dir, cells_dir, eqb_path, cells)) + [os.path.abspath(__file__)]:
        h.update((os.path.relpath(p, ROOT) if p.startswith(ROOT + os.sep) else p).encode())
        h.update((sha256_file(p) if os.path.exists(p) else "absent").encode())
    h.update(json.dumps(list(cells)).encode())
    return h.hexdigest()


def record_is_current(rec, refs_dir=REFS, cells_dir=CELLS_DIR, eqb_path=EQB, cells=CELLS):
    if not isinstance(rec, dict) or not rec.get("input_signature"):
        return False, "record has no input signature (written by an older tool version)"
    if rec["input_signature"] != input_signature(refs_dir, cells_dir, eqb_path, cells):
        return False, "input signature differs from the current references / cell configs / tool"
    return True, "current"


def check(refs_dir=REFS, cells_dir=CELLS_DIR, eqb_path=EQB, cells=CELLS):
    rec = dict(check="2", rule=("PREREGISTRATION.md 1.3 check 2: primary arrays (x_grid, eval_mask, F_ref, Fp_ref) "
                                "bitwise equal to each other and to results/equal_budget_v2/references/"
                                "gateway_reference.npz"),
               generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), files={}, problems=[],
               git_commit=_git_commit(), tool="scripts/final_campaign/expA_check_references.py")
    eqb = load(eqb_path)
    rec["files"][os.path.relpath(eqb_path, ROOT)] = sha256_file(eqb_path)
    refs = {}
    for c in cells:
        cfg_path = os.path.join(cells_dir, f"{c}.json")
        with open(cfg_path) as fh:
            P = json.load(fh)
        p = os.path.join(refs_dir, os.path.basename(P["reference_file"]))
        if not os.path.exists(p):
            rec["problems"].append(f"{c}: reference {p} missing")
            continue
        refs[c] = load(p)
        rec["files"][os.path.relpath(p, ROOT) if p.startswith(ROOT + os.sep) else p] = sha256_file(p)
        m = json.loads(str(refs[c]["model_json"]))
        want = {k: P["model"][k] for k in ("variant", "alpha", "kappa", "lam")}
        if m != want:
            rec["problems"].append(f"{c}: stored model {m} != cell model {want}")
        if float(refs[c]["h"]) != float(P["engine_cfg"]["h"]):
            rec["problems"].append(f"{c}: stored h {float(refs[c]['h'])} != cell h {P['engine_cfg']['h']}")
    prim = {}
    for k in PRIMARY:
        row = {c: same(r[k], eqb[k]) for c, r in refs.items()}
        cross = all(same(refs[c][k], refs[cells[0]][k]) for c in refs) if cells[0] in refs else False
        prim[k] = dict(equal_to_equal_budget=row, equal_across_cells=cross)
        if not all(row.values()) or not cross or len(row) != len(cells):
            rec["problems"].append(f"primary {k}: not bitwise equal everywhere ({row}, across cells {cross})")
    sec = {}
    for k in SECONDARY:
        cross = all(same(refs[c][k], refs[cells[0]][k]) for c in refs) if cells[0] in refs else False
        sec[k] = dict(equal_across_cells=cross)
    sec["F_ref_em_equal_to_equal_budget_F_ref_em"] = {c: same(r["F_ref_em"], eqb["F_ref_em"]) for c, r in refs.items()}
    sec["max_abs_F_ref_em_minus_F_ref"] = {c: float(np.max(np.abs(r["F_ref_em"] - r["F_ref"]))) for c, r in refs.items()}
    summ = os.path.join(refs_dir, "references_summary.json")
    if os.path.exists(summ):
        S = json.load(open(summ))
        digs = {v.get("primary_digest") for k, v in S.get("references", {}).items() if k.startswith("expA/")}
        rec["summary_primary_digests"] = sorted(d for d in digs if d)
        if len(digs) != 1:
            rec["problems"].append(f"references_summary.json: {len(digs)} primary digests for the expA cells")
        rec["files"][os.path.relpath(summ, ROOT) if summ.startswith(ROOT + os.sep) else summ] = sha256_file(summ)
    else:
        rec["problems"].append(f"{summ} missing")
    rec.update(primary=prim, secondary_descriptive=sec, n_cells=len(refs),
               verdict="PASS" if not rec["problems"] and len(refs) == len(cells) else "FAIL",
               input_signature=input_signature(refs_dir, cells_dir, eqb_path, cells))
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refs-dir", default=REFS)
    ap.add_argument("--cells-dir", default=CELLS_DIR)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--dry-run", action="store_true", help="print the verdict, write nothing")
    ap.add_argument("--force", action="store_true", help="replace a record with a different (or no) input signature")
    ap.add_argument("--eqb", default=EQB, help=argparse.SUPPRESS)          # tests only
    a = ap.parse_args(argv)
    rec = check(a.refs_dir, a.cells_dir, a.eqb)
    print(f"check 2: {rec['verdict']} ({rec['n_cells']} cells)" + ("" if not rec["problems"] else
                                                                  f": {rec['problems']}"))
    for k, v in rec["primary"].items():
        print(f"  {k}: equal to equal-budget {v['equal_to_equal_budget']}, across cells {v['equal_across_cells']}")
    if a.dry_run and os.path.exists(a.out):
        try:
            old = json.load(open(a.out))
        except Exception:  # noqa: BLE001
            old = {}
        same = old.get("input_signature") == rec["input_signature"]
        print(f"dry run: existing record {a.out} (verdict {old.get('verdict')}) is "
              + ("up to date" if same else "NOT current (" + ("different" if old.get("input_signature") else "no")
                 + " input signature): a write needs --force"))
    if not a.dry_run:
        if os.path.exists(a.out):
            try:
                old = json.load(open(a.out))
            except Exception:  # noqa: BLE001
                old = {}
            if old.get("input_signature") == rec["input_signature"] and not a.force:
                print(f"record up to date (same input signature, verdict {old.get('verdict')}): {a.out}")
                return 0 if old.get("verdict") == "PASS" else 1
            if not a.force:
                print(f"REFUSED to overwrite {a.out}: an existing record with a "
                      f"{'different' if old.get('input_signature') else 'missing'} input signature (verdict "
                      f"{old.get('verdict')}) is kept; use --force to replace it", flush=True)
                return 2
            rec["supersedes"] = dict(generated_utc=old.get("generated_utc"), verdict=old.get("verdict"),
                                     input_signature=old.get("input_signature"))
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        tmp = a.out + f".tmp{os.getpid()}"
        with open(tmp, "w") as fh:
            json.dump(rec, fh, indent=1)
        os.replace(tmp, a.out)
        print(f"wrote {a.out}")
    return 0 if rec["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
