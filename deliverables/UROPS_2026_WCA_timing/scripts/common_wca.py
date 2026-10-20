"""Shared definitions of the WCA Fisher-Rao timing study (deliverables/UROPS_2026_WCA_timing).

Everything that defines the experiment is read from ../frozen_config.json (written before any confirmation run);
this module only resolves paths and builds the engine configuration from it.

* REPO       : the checkout that contains this file (its src/wca_numba.py is the extended engine that runs).
* DATA_ROOT  : the main research checkout (raw data, the exact MC chains); env ABFFR_DATA_ROOT, default
               /home/zheyuanlai/ABF-Fisher-Rao.  Raw outputs go to DATA_ROOT/results/urops_2026_wca_timing/raw/.
"""
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(PKG, "..", ".."))
DATA_ROOT = os.path.abspath(os.environ.get("ABFFR_DATA_ROOT", "/home/zheyuanlai/ABF-Fisher-Rao"))
FROZEN = os.path.join(PKG, "frozen_config.json")
RAW = os.path.join(DATA_ROOT, "results", "urops_2026_wca_timing", "raw")
DT = 1.25e-4
N = 256
T = 240.0


def s(t):
    """physical time -> integration-step count (exact for every time used here)."""
    k = int(round(t / DT))
    assert abs(k * DT - t) < 1e-9, t
    return k


def engine():
    p = os.path.join(REPO, "src")
    if p not in sys.path:
        sys.path.insert(0, p)
    import wca_numba as wn
    assert os.path.abspath(wn.__file__) == engine_path(), wn.__file__
    return wn


def engine_path():
    return os.path.abspath(os.path.join(REPO, "src", "wca_numba.py"))


def reference_modules():
    """(rescore_wca_history, analyze_wca_validation) from DATA_ROOT (the exact MC chains live there)."""
    engine()
    p = os.path.join(DATA_ROOT, "scripts", "numerical_validation")
    if p not in sys.path:
        sys.path.append(p)
    import rescore_wca_history as R
    import analyze_wca_validation as A
    return R, A


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git_head(cwd=REPO):
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def git_dirty(paths, cwd=REPO):
    return subprocess.run(["git", "diff", "--quiet", "HEAD", "--"] + list(paths), cwd=cwd).returncode != 0


def validated_cfg(fr_start_t):
    """The Phase-V confirmation's engine cfg (scripts/numerical_validation/run_wca_confirmatory.cfg_for) with the
    FR start moved to ``fr_start_t``: ACCEPTED_CFG at dt 1.25e-4, ABF warm-up and estimator burn-in 20 t.u.,
    FR every 0.01 t.u., ESS window 8 t.u."""
    wn = engine()
    return dict(wn.ACCEPTED_CFG, dt=DT, abf_warmup_steps=s(20.0), estimator_burn_in_steps=s(20.0),
                fr_start_steps=s(fr_start_t), fr_every=s(0.01), ess_window_steps=s(8.0))


def save_grid():
    """Every 0.5 t.u. on [0, 20], every 1 t.u. on (20, 80], every 5 t.u. on (80, 240] (step indices)."""
    a = np.arange(0, s(20.0) + 1, s(0.5))
    b = np.arange(s(21.0), s(80.0) + 1, s(1.0))
    c = np.arange(s(85.0), s(240.0) + 1, s(5.0))
    g = np.concatenate([a, b, c]).astype(np.int64)
    assert np.all(np.diff(g) > 0)
    for t in (0.0, 5.0, 20.0, 40.0, 80.0, 240.0):
        assert s(t) in g, t
    return g


def frozen():
    return json.load(open(FROZEN))


def arm_job(arm):
    """(arms list, cfg, kwargs) of run_ladder_point for one frozen arm."""
    fz = frozen()
    A = {a["name"]: a for a in fz["arms"]}[arm]
    if A["fr"] is False:
        return [(arm, False)], validated_cfg(40.0), {}
    kw = dict(fr_stream_ids=[int(A["fr_stream_id"])])
    if A["fr_stop_t"] is not None:
        kw["fr_stop_steps"] = s(float(A["fr_stop_t"]))
    return [(arm, True)], validated_cfg(float(A["fr_start_t"])), kw


def raw_path(seed, arm):
    return os.path.join(RAW, f"s{seed}_{arm}.npz")
