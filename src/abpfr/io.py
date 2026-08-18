"""Run-record schema and (de)serialization.

Every serious run stores the FULL PMF and marginal time series so any later change of
reference or evaluation window can be rescored without rerunning dynamics (the old WCA
campaign learned this the expensive way).  save_run refuses records missing the schema
-- a number that is only printed gets skimmed; the check that matters stops the run.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time as _time

import numpy as np

# Arrays every production record must carry.
REQUIRED_ARRAYS = (
    "time",         # (n_saves,) physical time of each checkpoint
    "pmf_t",        # (n_saves, G) or (n_prof, n1, n2) F_hat at each checkpoint
    "marginal_t",   # KDE p_hat at each checkpoint, same layout as pmf_t
    "F_ref",        # reference profile the run was scored against
)
# The grid itself: 1D records carry x_grid, 2D torus records x1_grid + x2_grid.
GRID_KEY_SETS = (("x_grid",), ("x1_grid", "x2_grid"))
# Metadata every production record must carry.
REQUIRED_META = ("reference_id", "eval_window", "config", "method", "seed")


def _check_arrays(arrays, where):
    missing = [k for k in REQUIRED_ARRAYS if k not in arrays]
    assert not missing, f"{where} missing required arrays {missing}; refusing"
    assert any(all(k in arrays for k in ks) for ks in GRID_KEY_SETS), \
        f"{where} carries no grid (need x_grid or x1_grid+x2_grid); refusing"


def save_run(path, arrays: dict, meta: dict):
    """Write <path>.npz (arrays) + <path>.json (metadata).  Hard-asserts the schema."""
    _check_arrays(arrays, "run record")
    missing = [k for k in REQUIRED_META if k not in meta]
    assert not missing, f"run record missing required metadata {missing}; refusing to save"
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    np.savez_compressed(path + ".npz", **{k: np.asarray(v) for k, v in arrays.items()})
    with open(path + ".json", "w") as f:
        json.dump({**meta, "provenance": provenance()}, f, indent=2, default=_json_default)


def load_run(path):
    with np.load(path + ".npz") as z:
        arrays = {k: z[k] for k in z.files}
    with open(path + ".json") as f:
        meta = json.load(f)
    _check_arrays(arrays, f"stored record at {path}")
    return arrays, meta


def provenance():
    """Environment stamp attached to every record."""
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, cwd=os.path.dirname(os.path.abspath(__file__)),
                             timeout=5).stdout.strip()
    except Exception:
        rev = ""
    info = {"git_rev": rev, "python": sys.version.split()[0],
            "timestamp": _time.strftime("%Y-%m-%dT%H:%M:%S")}
    try:
        import torch
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return info


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")
