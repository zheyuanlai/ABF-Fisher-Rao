"""Shared paths and conventions of the UROPS alanine incomplete-CV export (deliverables/UROPS_2026_Alanine_CV).

Raw simulation files are git-ignored and live in the main research checkout: ABFFR_DATA_ROOT (default
/home/zheyuanlai/ABF-Fisher-Rao).  The repository's own analysis modules (src/alanine/..., scripts/analyze_*.py)
are imported from ABFFR_DATA_ROOT so that every quantity uses the code that produced the published numbers.
"""
import glob
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(PKG, "..", ".."))
DATA_ROOT = os.path.abspath(os.environ.get("ABFFR_DATA_ROOT", "/home/zheyuanlai/ABF-Fisher-Rao"))
REF = "results/alanine/reference/reference.npz"
RUN_ROOT = "results/alanine_1d"
STAGES = ("phi_abf", "phi_u02_t0", "phi_u15_t0", "psi_abf", "psi_u02_t0", "psi_u15_t0")
LABEL = {"abf": "histogram ABF", "u02": "ABF + uniform FR, rate 0.02", "u15": "ABF + uniform FR, rate 0.15"}


def setup_paths():
    for p in (os.path.join(DATA_ROOT, "src"), os.path.join(DATA_ROOT, "scripts")):
        if p not in sys.path:
            sys.path.insert(0, p)


def dpath(rel):
    return os.path.join(DATA_ROOT, rel)


def raw_file(stage):
    f = [p for p in glob.glob(dpath(f"{RUN_ROOT}/{stage}/raw/*.npz")) if "_failures" not in p]
    if len(f) != 1:
        raise FileNotFoundError(f"{stage}: expected one raw file, found {f}")
    return f[0]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git(args, cwd):
    try:
        return subprocess.run(["git"] + args, cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def provenance():
    rc = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "src", "scripts"], cwd=DATA_ROOT).returncode
    return dict(data_root=DATA_ROOT, analysis_code_commit=git(["rev-parse", "HEAD"], DATA_ROOT),
                analysis_code_tracked_files_modified=bool(rc != 0), exporter_commit=git(["rev-parse", "HEAD"], REPO))


def jdump(obj, path):
    import numpy as np

    def conv(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return str(o)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, default=conv)
