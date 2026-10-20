"""Shared helpers of the UROPS 2026 figure-data exporters (deliverables/UROPS_2026/scripts/).

Every exporter turns ONE validated paired experiment (ABF vs ABF + uniform Fisher-Rao, same seeds) into
  * figs/data/<exp>_{eF,eFp,exi}.csv   seven columns t,abf_median,abf_q25,abf_q75,fr_median,fr_q25,fr_q75
                                       (median and 25th / 75th percentiles ACROSS SEEDS at every saved checkpoint;
                                       numpy's default linear-interpolation percentile; no smoothing, no time
                                       interpolation: every row is a checkpoint the simulation actually saved);
  * per_seed/<exp>_<metric>_<arm>.csv  the per-seed error at every row (rows = the same t grid, one column per seed),
                                       the unit of inference;
  * meta/<exp>.json                    configuration, provenance, definitions, units and the reproduced statistics.

Raw simulation data are git-ignored and live in the main checkout.  ABFFR_DATA_ROOT points to it (default: the
repository that contains this file).  The repository's own scorers are imported from ABFFR_DATA_ROOT/src and
ABFFR_DATA_ROOT/scripts, so every error is computed by the code that produced the validated results; the git
commit of that code and whether its tracked files were modified are recorded in each meta file.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)                                   # deliverables/UROPS_2026
REPO = os.path.abspath(os.path.join(PKG, "..", ".."))
DATA_ROOT = os.path.abspath(os.environ.get("ABFFR_DATA_ROOT", REPO))
DATA_DIR = os.path.join(PKG, "figs", "data")
PREVIEW_DIR = os.path.join(PKG, "figs", "previews")
SEED_DIR = os.path.join(PKG, "per_seed")
META_DIR = os.path.join(PKG, "meta")
COLUMNS = ("t", "abf_median", "abf_q25", "abf_q75", "fr_median", "fr_q25", "fr_q75")
METRICS = ("eF", "eFp", "exi")
EXPERIMENTS = ("gateway", "lta150", "lta300", "ite300", "meta", "alanine", "wca")


def setup_paths(*rel):
    """Put ABFFR_DATA_ROOT/src, /scripts and the given extra sub-directories on sys.path (front)."""
    for r in ("src", "scripts") + tuple(rel):
        p = os.path.join(DATA_ROOT, r)
        if p not in sys.path:
            sys.path.insert(0, p)


def dpath(rel):
    return os.path.join(DATA_ROOT, rel)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git(args, cwd):
    try:
        return subprocess.run(["git"] + args, cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"unavailable ({e.__class__.__name__})"


def code_provenance(paths=("src", "scripts", "configs")):
    """Commit of the scoring code (ABFFR_DATA_ROOT) and whether its tracked files under ``paths`` differ from it."""
    rc = subprocess.run(["git", "diff", "--quiet", "HEAD", "--"] + list(paths), cwd=DATA_ROOT).returncode
    return dict(data_root=DATA_ROOT, scoring_code_commit=git(["rev-parse", "HEAD"], DATA_ROOT),
                scoring_code_tracked_files_modified=bool(rc != 0), checked_paths=list(paths),
                exporter_commit=git(["rev-parse", "HEAD"], REPO), python=sys.version.split()[0],
                numpy=np.__version__)


def quartiles(x):
    """(q25, median, q75) across seeds (axis 1) with numpy's default linear percentile."""
    x = np.asarray(x, float)
    q25, med, q75 = np.percentile(x, [25.0, 50.0, 75.0], axis=1)
    return q25, med, q75


def l2_to_uniform(counts, widths):
    """||p - u||_{L2} per row, p = counts / (sum(counts) * width) (a density on the full histogram domain) and
    u = 1 / |domain|; counts (..., nb), widths (nb,) bin widths (all equal in every system here)."""
    c = np.asarray(counts, float)
    w = np.asarray(widths, float)
    L = float(w.sum())
    tot = c.sum(-1, keepdims=True)
    p = c / (tot * w)
    return np.sqrt((((p - 1.0 / L) ** 2) * w).sum(-1))


def l2_density_to_uniform(p, width, L):
    """||p - u||_{L2} for a density p already given per unit length on equal cells of ``width``."""
    p = np.asarray(p, float)
    return np.sqrt((((p - 1.0 / L) ** 2) * width).sum(-1))


def multinomial_floor(N, nb, width):
    """RMS of ||p_hat - u|| for N INDEPENDENT walkers drawn from the exactly uniform law onto nb equal bins of
    ``width``: E||p_hat - u||^2 = (1 - 1/nb) / (N width).  The finite-N histogram floor of panel (c)."""
    return math.sqrt((1.0 - 1.0 / nb) / (N * width))


def fmt(v):
    v = float(v)
    if not math.isfinite(v):
        return "nan"
    return f"{v:.10g}"


def write_csv(path, t, abf, fr):
    q25a, meda, q75a = quartiles(abf)
    q25f, medf, q75f = quartiles(fr)
    with open(path, "w", newline="") as fh:
        fh.write(",".join(COLUMNS) + "\n")
        for i in range(len(t)):
            fh.write(",".join(fmt(v) for v in (t[i], meda[i], q25a[i], q75a[i], medf[i], q25f[i], q75f[i])) + "\n")


def write_per_seed(path, t, x, seeds):
    x = np.asarray(x, float)
    with open(path, "w", newline="") as fh:
        fh.write("t," + ",".join(f"seed_{s}" for s in seeds) + "\n")
        for i in range(len(t)):
            fh.write(fmt(t[i]) + "," + ",".join(fmt(v) for v in x[i]) + "\n")


def read_csv(path):
    with open(path) as fh:
        r = csv.reader(fh)
        head = next(r)
        rows = [[float(v) for v in row] for row in r]
    return head, np.asarray(rows, float)


def read_per_seed(path):
    head, a = read_csv(path)
    seeds = [int(h.split("_", 1)[1]) for h in head[1:]]
    return a[:, 0], a[:, 1:], seeds


def export(exp, t, curves, seeds, meta):
    """curves: {metric: {"abf": (S, R), "fr": (S, R)}} for metric in METRICS (a missing metric is not written);
    the column order of every (S, R) array is ``seeds``."""
    for d in (DATA_DIR, SEED_DIR, META_DIR):
        os.makedirs(d, exist_ok=True)
    t = np.asarray(t, float)
    if t.ndim != 1 or np.any(np.diff(t) <= 0):
        raise ValueError(f"{exp}: time grid must be strictly increasing")
    written = {}
    for m in METRICS:
        if m not in curves:
            continue
        a, f = np.asarray(curves[m]["abf"], float), np.asarray(curves[m]["fr"], float)
        if a.shape != (t.size, len(seeds)) or f.shape != a.shape:
            raise ValueError(f"{exp} {m}: shapes {a.shape} / {f.shape} != ({t.size}, {len(seeds)})")
        p = os.path.join(DATA_DIR, f"{exp}_{m}.csv")
        write_csv(p, t, a, f)
        write_per_seed(os.path.join(SEED_DIR, f"{exp}_{m}_abf.csv"), t, a, seeds)
        write_per_seed(os.path.join(SEED_DIR, f"{exp}_{m}_fr.csv"), t, f, seeds)
        written[m] = dict(csv=os.path.relpath(p, PKG), n_rows=int(t.size),
                          n_nonfinite_abf=int((~np.isfinite(a)).sum()), n_nonfinite_fr=int((~np.isfinite(f)).sum()),
                          n_nonpositive_abf=int((np.isfinite(a) & (a <= 0)).sum()),
                          n_nonpositive_fr=int((np.isfinite(f) & (f <= 0)).sum()))
    meta = dict(meta)
    meta.update(experiment=exp, seeds=[int(s) for s in seeds], n_seeds=len(seeds), n_times=int(t.size),
                t_first=float(t[0]), t_last=float(t[-1]), files=written, provenance=code_provenance())
    with open(os.path.join(META_DIR, f"{exp}.json"), "w") as fh:
        json.dump(json_safe(meta), fh, indent=1)
    return meta


def json_safe(o):
    if isinstance(o, dict):
        return {str(k): json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_safe(v) for v in o]
    if isinstance(o, np.ndarray):
        return json_safe(o.tolist())
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if math.isfinite(f) else ("inf" if f > 0 else "-inf" if f < 0 else "nan")
    return o


# ---------------------------------------------------------------------------------------------- paired statistics
def boot_median_ci(d, n_boot=10_000, seed=0):
    """Percentile bootstrap 95 % CI of the median of per-seed values d (seeds resampled with replacement)."""
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    m = np.median(d[rng.integers(0, d.size, (n_boot, d.size))], axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def paired_rel(fr, abf, seed=0, n_boot=10_000):
    """Per-seed relative change (FR - ABF) / ABF: median, bootstrap CI of the median, FR wins (< 0) out of n."""
    fr, abf = np.asarray(fr, float), np.asarray(abf, float)
    d = (fr - abf) / abf
    return dict(median=float(np.median(d)), ci95=boot_median_ci(d, n_boot, seed), wins_fr=int((d < 0).sum()),
                n=int(d.size), abf_median=float(np.median(abf)), fr_median=float(np.median(fr)),
                per_seed=[float(v) for v in d])


def trapz_window(t, e, lo, hi):
    """Trapezoid of e (S, R) over the saves with lo <= t <= hi (per seed)."""
    t = np.asarray(t, float)
    sel = (t >= lo - 1e-9) & (t <= hi + 1e-9)
    return np.trapezoid(np.asarray(e, float)[sel], t[sel], axis=0)
