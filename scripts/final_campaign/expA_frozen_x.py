#!/usr/bin/env python
"""Final campaign, Experiment A, check 4b: frozen-x AR(1) test of the transverse integrator THROUGH THE PRODUCTION
ENGINE (docs/final_campaign/PREREGISTRATION.md section 1.3, nine-check table row 4(b);
configs/final_campaign/expA_validation.json "frozen_x_relaxation_test").

    python scripts/final_campaign/expA_frozen_x.py [--h-A 2.5e-5] [--dry-run] [--out <json>] [--cpus 64-71]

What runs.  src/gateway_family_numba.run_arm(cfg, "abf", N = 32, ..., _freeze_x_test=True, diagnostics=True) -- the
production y-update with x frozen (as tests/test_gateway_family_numba.py::test_f4; the engine is NOT modified), shifted
fibre kappa = 1, x0 = -0.21 (kappa is constant, so other x0 are replicates of the same process).  At frozen x the EM
y-update is the exact AR(1) d <- phi d + sqrt(2 lam h / beta) z for d = y - m(x0), phi = 1 - lam kappa^2 h, with EM
stationary variance var_EM = (1/(beta kappa^2)) / (1 - lam kappa^2 h / 2).
Cases (prereg): lam in {1, 4, 16, 128} at h_A, plus the resolvable-inflation case lam in {1, 16} at h = 0.1/lam (phi 0.9,
EM excess 5.3 %).  64 chains per case, started from the EM stationary law (y0 = m(x0) + sqrt(var_EM) z, z = the engine's
init_family standard normals of the case label, init_left's stream default_rng(1000 + label)), each run for
200 relaxation times = round(200 / (lam kappa^2 h)) steps.  The engine traces at most N_TRACE_MAX = 32 walkers, so the
64 chains are two engine runs of 32 traced walkers ("halves") under ONE label: half 0 uses the label's N = 32 noise seed
default_noise_seed(label, 32), half 1 the label's N = 64 noise seed default_noise_seed(label, 64) (both checked
collision-free by scripts/final_campaign/check_seeds.py, block expA_frozenx with N in {32, 64}); z[0:32] / z[32:64].
y is read from the engine's traces_y (D4) at trace_steps = every Delta steps, Delta = gcd of the lags below.
Lags.  The prereg asks for l h in {0.25, 0.5, 1, 2} / (lam kappa^2); l must be an integer number of steps, so
l_k = floor(c_k / (lam kappa^2 h) + 1/2) (round half up).  It is exact for lam 1, 4, 16 at either candidate h; at lam 128
it gives 78, 156, 313, 625 steps at h 2.5e-5 (nominal 78.125, 156.25, 312.5, 625) and 156, 313, 625, 1250 at 1.25e-5;
for the resolvable case 3, 5, 10, 20 (nominal 2.5, 5, 10, 20).  Every test compares with phi^l at the REALISED l, so the
rounding changes only which lags are probed, never the validity of the test (nominal and realised lags are recorded).
Estimators (known mean 0, normalised by the known var_EM; chains independent):
  var:    per chain  mean_t d_t^2 / var_EM                       expected 1
  acf_l:  per chain  mean_t d_t d_{t+l} / var_EM  (all origins)  expected phi^l
  se = between-chain sd / sqrt(64); T = (mean - expected) / se; two-sided p from t_63.  6 cases x 5 quantities = 30 p
  in ONE Holm family at FWER 5 %: PASS iff none is rejected.  Reported alongside (not tested): the same statistics against
  the EXACT (non-EM) variance / exp(-lam kappa^2 l h), showing whether the case can tell EM from exact.
  This verifies the y integrator (drift and noise scaling with lam); it is NOT evidence about coupled relaxation (MC1).
Labels (prereg section 4 block 8650-8661, one label per (lam, case)):
  8650-8653  lam 1, 4, 16, 128 at h 2.5e-5;   8654, 8655  resolvable lam 1 (h 0.1), lam 16 (h 0.00625);
  8656-8659  lam 1, 4, 16, 128 at h 1.25e-5 (only if the timestep is refined);   8660, 8661 spare (never used).
Output results/final_campaign/expA/checks/check4b_frozen_x.json (h_A 2.5e-5; at the refinement h_A 1.25e-5:
check4b_frozen_x_h1.25e-05.json): signature (engine sha256, design, labels, lags, estimator), provenance (git commit,
dirty flag, versions, host, CPU affinity, wall), per-case statistics, the Holm table and the verdict.  Idempotent: an
existing complete output with the same signature is reported and kept; one with a different signature is refused
(exit 2, never overwritten).  Nothing large is written (the traces stay in memory).
Refused (exit 2) unless src/gateway_family_numba.py AND the imported engine hash to the frozen
expA_run_common.EXPECTED_ENGINE_SHA256.  Disk pause (prereg section 3): before every case and before the JSON is
written, expA_run_common.wait_for_disk (no start below 2 GB free on /, resume at >= 3 GB, logged to stdout and
<dir of the output>/disk_pause.log; > 24 h = stopping rule (iv)).  Compute: every run that computes (complete OR failed)
appends one row (stage expA-check4b, threads 1, measured wall, used core-h) to <dir of the output>/ledger.csv in the
columns of results/final_campaign/compute_ledger.csv (default results/final_campaign/expA/checks/ledger.csv).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import time
from functools import reduce

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for _k in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[_k] = "1"
    os.environ.setdefault("NUMBA_CACHE_DIR", os.path.expanduser("~/.cache/numba_final"))
import numpy as np  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
if HERE not in sys.path:
    sys.path.insert(1, HERE)
import expA_run_common as C  # noqa: E402  (frozen engine identity, disk pause, compute-ledger rows)

CFG_VAL = os.path.join(ROOT, "configs", "final_campaign", "expA_validation.json")
CFG_EXPA = os.path.join(ROOT, "configs", "final_campaign", "expA.json")
ENGINE = os.path.join(ROOT, "src", "gateway_family_numba.py")
OUT_DIR = os.path.join(ROOT, "results", "final_campaign", "expA", "checks")
H_CANDIDATES = (2.5e-5, 1.25e-5)
N_CHAINS = 64
N_RELAX = 200.0
LAG_FACTORS = (0.25, 0.5, 1.0, 2.0)
FWER = 0.05
ESTIMATOR_VERSION = "expA_frozen_x/1: known mean 0, normalised by var_EM, all origins, between-chain se, t_63, Holm"
LABELS = {("h_A", 1.0, 2.5e-5): 8650, ("h_A", 4.0, 2.5e-5): 8651, ("h_A", 16.0, 2.5e-5): 8652,
          ("h_A", 128.0, 2.5e-5): 8653, ("resolvable", 1.0, None): 8654, ("resolvable", 16.0, None): 8655,
          ("h_A", 1.0, 1.25e-5): 8656, ("h_A", 4.0, 1.25e-5): 8657, ("h_A", 16.0, 1.25e-5): 8658,
          ("h_A", 128.0, 1.25e-5): 8659}
SPARE_LABELS = (8660, 8661)


def _holm_reject(pvals, alpha):
    """scripts/mechanism/analyze_validation.holm_reject (the campaign's Holm step-down, imported, not copied)."""
    if "mech_analyze_validation" in sys.modules:
        AV = sys.modules["mech_analyze_validation"]
    else:
        spec = importlib.util.spec_from_file_location("mech_analyze_validation",
                                                      os.path.join(ROOT, "scripts", "mechanism", "analyze_validation.py"))
        AV = importlib.util.module_from_spec(spec)
        sys.modules["mech_analyze_validation"] = AV
        spec.loader.exec_module(AV)
    return AV.holm_reject(list(pvals), alpha)


def holm_adjusted(pvals):
    """Holm step-down adjusted p-values (reported next to the decisions)."""
    p = np.asarray(pvals, float)
    k = p.size
    order = np.argsort(p)
    adj = np.empty(k)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (k - r) * p[i]))
        adj[i] = run
    return adj


# ---------------------------------------------------------------------------------------------------------------
# design
# ---------------------------------------------------------------------------------------------------------------
def frozen_cfg():
    c = json.load(open(CFG_VAL))["frozen_x_relaxation_test"]
    if float(c["x0"]) != -0.21 or int(c["chains"]) != N_CHAINS:
        raise RuntimeError("frozen_x_relaxation_test config drift")
    seeds = [int(s) for s in c["seeds"]]
    if sorted(set(LABELS.values()) | set(SPARE_LABELS)) != sorted(seeds):
        raise RuntimeError("label mapping does not cover exactly the preregistered block")
    return c


def lag_steps(lam, kappa, h):
    """Realised integer lags l_k = floor(c_k / (lam kappa^2 h) + 1/2) and the nominal ones."""
    nom = [f / (lam * kappa ** 2 * h) for f in LAG_FACTORS]
    return [int(math.floor(v + 0.5)) for v in nom], nom


def cases(h_A):
    """The six cases of one check-4b run: lam 1, 4, 16, 128 at h_A and the resolvable lam 1, 16 at h = 0.1/lam."""
    if h_A not in H_CANDIDATES:
        raise ValueError(f"h_A {h_A} is not a candidate {H_CANDIDATES}")
    c = frozen_cfg()
    out = []
    for lam in c["lam"]:
        out.append(dict(case="h_A", lam=float(lam), h=float(h_A), label=LABELS[("h_A", float(lam), h_A)]))
    for lam in c["resolvable_inflation_case"]["lam"]:
        out.append(dict(case="resolvable", lam=float(lam), h=0.1 / float(lam), label=LABELS[("resolvable", float(lam), None)]))
    for e in out:
        e.update(design(e["lam"], 1.0, e["h"]))
    return out


def design(lam, kappa, h, n_relax=N_RELAX, beta=16.0):
    n_steps = int(round(n_relax / (lam * kappa ** 2 * h)))
    ls, nom = lag_steps(lam, kappa, h)
    stride = reduce(math.gcd, ls)
    phi = 1.0 - lam * kappa ** 2 * h
    var_exact = 1.0 / (beta * kappa ** 2)
    var_em = var_exact / (1.0 - lam * kappa ** 2 * h / 2.0)
    return dict(kappa=kappa, n_relax=float(n_relax), n_steps=n_steps, lags_steps=ls, lags_nominal_steps=nom,
                lags_t=[l * h for l in ls], stride_steps=stride, lag_samples=[l // stride for l in ls],
                n_samples=n_steps // stride + 1, phi=phi, var_exact=var_exact, var_em=var_em,
                em_excess=var_em / var_exact - 1.0)


def engine_cfg(lam, h):
    """configs/final_campaign/expA.json engine_cfg with the shifted fibre at lam and step h; fr_interval_t = h where
    the production 0.004 t.u. is not an integer number of steps (the resolvable case; as test_f4; irrelevant for ABF)."""
    P = json.load(open(CFG_EXPA))
    cfg = dict(P["engine_cfg"])
    cfg.update(h=float(h), variant="shift", alpha=None, kappa=1.0, lam=float(lam))
    fe = cfg["fr_interval_t"] / h
    if abs(fe - round(fe)) > 1e-9 * max(1.0, fe) or round(fe) < 1:
        cfg["fr_interval_t"] = float(h)
    return cfg


# ---------------------------------------------------------------------------------------------------------------
# estimators
# ---------------------------------------------------------------------------------------------------------------
def ar1_estimates(d, var_em, phi, lag_samples, lags):
    """d (K samples, C chains): deviations from the known mean 0, sampled every stride steps from the stationary law.
    Returns [(quantity, expected, per-chain values (C,))]: 'var' = mean d^2 / var_em (expected 1) and, for each lag,
    'acf_<l>' = mean over all origins of d_t d_{t+L} / var_em (L = l / stride samples; expected phi^l)."""
    d = np.asarray(d, float)
    out = [("var", 1.0, (d * d).mean(0) / var_em)]
    for L, l in zip(lag_samples, lags):
        out.append((f"acf_{l}", float(phi) ** int(l), (d[:-L] * d[L:]).mean(0) / var_em))
    return out


def t_test(values, expected):
    """One-sample two-sided t test of the chain mean against `expected` (se from the between-chain spread)."""
    from scipy.stats import t as _t
    v = np.asarray(values, float)
    C = v.size
    mean = float(v.mean())
    se = float(v.std(ddof=1) / math.sqrt(C))
    T = (mean - expected) / se if se > 0 else float("inf") * np.sign(mean - expected)
    p = float(2.0 * _t.sf(abs(T), C - 1)) if np.isfinite(T) else 0.0
    return dict(expected=float(expected), mean=mean, se=se, t=float(T), dof=C - 1, p=p)


def evaluate(d, case):
    """The five tests of one case (EM reference) + the descriptive exact-reference statistics."""
    tests = []
    for q, expv, vals in ar1_estimates(d, case["var_em"], case["phi"], case["lag_samples"], case["lags_steps"]):
        t = dict(quantity=q, **t_test(vals, expv))
        # descriptive: the same values against the EXACT law (var 1/(beta kappa^2), rho exp(-lam kappa^2 l h))
        scale = case["var_em"] / case["var_exact"]
        if q == "var":
            ex = 1.0
        else:
            l = int(q.split("_")[1])
            ex = math.exp(-case["lam"] * case["kappa"] ** 2 * l * case["h"])
        t["vs_exact"] = t_test(np.asarray(vals) * scale, ex)
        tests.append(t)
    return tests


# ---------------------------------------------------------------------------------------------------------------
# engine runs
# ---------------------------------------------------------------------------------------------------------------
def run_case(case, x0=-0.21, n_chains=N_CHAINS):
    """Two engine runs of 32 traced walkers (halves) for one case; returns (d (K, 64), per-half records)."""
    import gateway_family_numba as E
    half = E.N_TRACE_MAX
    if n_chains % half:
        raise ValueError(f"n_chains {n_chains} must be a multiple of N_TRACE_MAX {half}")
    lam, h, label = case["lam"], case["h"], case["label"]
    cfg = engine_cfg(lam, h)
    c = E.full_cfg(cfg)
    om = c["omega_out"] + (c["omega_in"] - c["omega_out"]) * math.exp(-x0 * x0 / (2.0 * c["s"] * c["s"]))
    m0 = math.sqrt(2.0 / c["beta"]) / c["kappa"] * math.log(om)                     # the engine's msc * log(om)
    z = E._init_draws(label, n_chains)[1]                                           # init_left's standard normals
    y0_all = m0 + math.sqrt(case["var_em"]) * z
    n = case["n_steps"]
    trace = np.arange(0, n + 1, case["stride_steps"], dtype=np.int64)
    ds, recs = [], []
    for k in range(n_chains // half):
        Nn = half * (k + 1)                                    # half 0: (label, 32); half 1: (label, 64) noise seed
        ns = E.default_noise_seed(label, Nn)
        x0v = np.full(half, float(x0))
        y0 = np.ascontiguousarray(y0_all[k * half:(k + 1) * half])
        t0 = time.time()
        res = E.run_arm(cfg, "abf", half, label, n, np.array([n], np.int64), x0=x0v, y0=y0, noise_seed=ns,
                        trace_steps=trace, diagnostics=True, _freeze_x_test=True)
        wall = time.time() - t0
        ty = np.asarray(res["traces_y"], float)
        tx = np.asarray(res["traces"], float) if "traces" in res else None
        ok_x = bool(np.array_equal(res["X_final"], x0v) and (tx is None or np.all(tx == x0)))
        ok_f = bool(np.all(np.isfinite(ty)) and np.all(np.isfinite(res["Y_final"])))
        ok_end = bool(trace[-1] != n or np.array_equal(ty[-1], res["Y_final"]))
        ok_start = bool(np.array_equal(ty[0], y0))
        if ty.shape != (trace.size, half):
            raise RuntimeError(f"traces_y shape {ty.shape} != {(trace.size, half)}")
        ds.append(ty - m0)
        recs.append(dict(half=k, N=half, noise_seed=ns, noise_seed_rule=f"default_noise_seed(label, {Nn})", wall_s=wall,
                         x_frozen=ok_x, finite=ok_f, trace_end_equals_Y_final=ok_end, trace_start_equals_y0=ok_start,
                         engine_sha256=res["meta"]["engine_sha256"] if "meta" in res else None))
    return np.concatenate(ds, axis=1), recs, dict(m0=m0, cfg=cfg)


# ---------------------------------------------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------------------------------------------
def sha256_file(path):
    hh = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            hh.update(b)
    return hh.hexdigest()


def _git(*args):
    try:
        return subprocess.check_output(["git", "-C", ROOT] + list(args), text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def signature(h_A, cs, n_chains=N_CHAINS, x0=-0.21):
    return json.loads(json.dumps(dict(
        check="4b", engine_sha256=sha256_file(ENGINE), x0=x0, n_chains=n_chains, n_relax=N_RELAX, lag_factors=list(LAG_FACTORS),
        lag_rounding="floor(c / (lam kappa^2 h) + 1/2)", h_A=h_A, estimator=ESTIMATOR_VERSION, fwer=FWER,
        cases=[{k: e[k] for k in ("case", "lam", "h", "label", "n_steps", "stride_steps", "lags_steps")} for e in cs],
        engine_cfg_base={k: v for k, v in json.load(open(CFG_EXPA))["engine_cfg"].items()}), sort_keys=True))


def default_out(h_A):
    return os.path.join(OUT_DIR, "check4b_frozen_x.json" if h_A == H_CANDIDATES[0] else f"check4b_frozen_x_h{h_A:g}.json")


def ledger_path(out):
    """The compute ledger of check 4b: <dir of the output JSON>/ledger.csv (results/final_campaign/expA/checks/ledger.csv
    for the default output), expA_run_common.COMPUTE_FIELDS -- the columns of results/final_campaign/compute_ledger.csv,
    so its rows fold into the campaign ledger unchanged."""
    return os.path.join(os.path.dirname(os.path.abspath(out)), "ledger.csv")


def _pause_logger(out):
    path = os.path.join(os.path.dirname(os.path.abspath(out)), "disk_pause.log")

    def log(msg):
        print(msg, flush=True)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a") as fh:
                fh.write(msg + "\n")
        except OSError as e:
            print(f"(disk_pause.log not writable: {e!r})", flush=True)
    return log


def run_check(h_A=H_CANDIDATES[0], out=None, cs=None, n_chains=N_CHAINS, verbose=True, disk_path="/"):
    """Run every case, test, Holm; write the JSON atomically.  Returns the result dict.

    Refused (SystemExit) unless the engine file AND the imported engine hash to the frozen
    expA_run_common.EXPECTED_ENGINE_SHA256.  Before every case and before the JSON is written: the prereg section 3
    disk pause (expA_run_common.wait_for_disk; logged to stdout and <dir of out>/disk_pause.log).  Every run that
    computes (complete or failed) appends one compute row (threads 1, measured wall) to ledger_path(out)."""
    import numba
    import scipy
    cs = cs if cs is not None else cases(h_A)
    out = out or default_out(h_A)
    sig = signature(h_A, cs, n_chains)
    if os.path.exists(out):
        old = json.load(open(out))
        if old.get("signature") == sig and old.get("status") == "complete":
            if verbose:
                print(f"{out}: complete with this signature (verdict {old.get('verdict')}); kept", flush=True)
            return old
        diff = sorted(k for k in set(sig) | set(old.get("signature") or {}) if sig.get(k) != (old.get("signature") or {}).get(k))
        raise SystemExit(f"refused: {out} exists with a different signature (fields {diff}); never overwritten -- move it "
                         f"away to re-run")
    try:
        engine_sha = C.check_engine()
    except C.EngineMismatch as e:
        raise SystemExit(f"refused: {e}")
    log = _pause_logger(out)
    walker_steps = sum(n_chains * c["n_steps"] for c in cs)
    t0 = time.time()
    ledger_note = "FAILED (interrupted before the verdict)"
    try:
        S = _run_cases(h_A, out, cs, n_chains, verbose, disk_path, log, sig, engine_sha, t0, numba, scipy)
        ledger_note = f"verdict {S['verdict']}; {os.path.relpath(out, ROOT) if out.startswith(ROOT) else out}"
        return S
    except BaseException as e:
        ledger_note = f"FAILED: {type(e).__name__}: {e}"[:200]
        raise
    finally:
        wall = time.time() - t0
        try:
            C.append_csv(ledger_path(out), C.COMPUTE_FIELDS,
                         [C.compute_row("expA-check4b", f"check 4b frozen-x AR(1), h_A {h_A:g}, {len(cs)} cases, "
                                                        f"{walker_steps:.4g} walker-steps", wall, t0, note=ledger_note)])
        except OSError as e:
            print(f"COMPUTE LEDGER WRITE FAILED ({e!r}): expA-check4b wall {wall:.1f} s, 1 thread, {ledger_note}",
                  flush=True)


def _run_cases(h_A, out, cs, n_chains, verbose, disk_path, log, sig, engine_sha, t0, numba, scipy):
    results, pv, idx = [], [], []
    for ci, case in enumerate(cs):
        C.wait_for_disk(disk_path, log=log)
        d, recs, extra = run_case(case, n_chains=n_chains)
        tests = evaluate(d, case)
        r = dict(case, halves=recs, m0=extra["m0"], engine_cfg=extra["cfg"], tests=tests,
                 integrity_ok=all(h_["x_frozen"] and h_["finite"] and h_["trace_end_equals_Y_final"]
                                  and h_["trace_start_equals_y0"] for h_ in recs))
        results.append(r)
        for ti, t in enumerate(tests):
            pv.append(t["p"])
            idx.append((ci, ti))
        if verbose:
            print(f"[{case['case']:10s} lam {case['lam']:g} h {case['h']:g} label {case['label']}] "
                  + "  ".join(f"{t['quantity']}: {t['mean']:.4f} vs {t['expected']:.4f} (t {t['t']:+.2f})" for t in tests)
                  + f"  [{sum(h_['wall_s'] for h_ in recs):.0f} s]", flush=True)
    rej = _holm_reject(pv, FWER)
    adj = holm_adjusted(pv)
    for (ci, ti), r_, a_ in zip(idx, rej, adj):
        results[ci]["tests"][ti].update(holm_rejected=bool(r_), p_holm=float(a_))
    integrity = all(r["integrity_ok"] for r in results)
    n_rej = int(sum(rej))
    verdict = "PASS" if (n_rej == 0 and integrity) else "FAIL"
    S = dict(check="4b: frozen-x AR(1) test of the transverse integrator through the production engine",
             prereg="docs/final_campaign/PREREGISTRATION.md section 1.3 (nine-check table, row 4b)",
             config="configs/final_campaign/expA_validation.json frozen_x_relaxation_test",
             meaning="verifies the y integrator (drift and noise scaling with lam); NOT evidence about coupled relaxation (MC1)",
             signature=sig,
             label_mapping={f"{k[0]} lam {k[1]:g}" + (f" h {k[2]:g}" if k[2] else " h 0.1/lam"): v for k, v in LABELS.items()},
             spare_labels=list(SPARE_LABELS),
             provenance=dict(git_commit=_git("rev-parse", "HEAD"),
                             git_dirty=bool(_git("status", "--porcelain", "--", "src/gateway_family_numba.py",
                                                 "scripts/final_campaign/expA_frozen_x.py")),
                             engine_sha256=sha256_file(ENGINE), engine_sha256_checked=engine_sha,
                             engine_sha256_expected=C.EXPECTED_ENGINE_SHA256,
                             script_sha256=sha256_file(os.path.abspath(__file__)),
                             python=platform.python_version(), numpy=np.__version__, numba=numba.__version__,
                             scipy=scipy.__version__, host=platform.node(), cpu_affinity=sorted(os.sched_getaffinity(0)),
                             precision="float64", device="CPU", started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
                             wall_s=time.time() - t0),
             cases=results,
             holm=dict(n_tests=len(pv), alpha=FWER, n_rejected=n_rej,
                       rejected=[dict(case=results[ci]["case"], lam=results[ci]["lam"], h=results[ci]["h"],
                                      quantity=results[ci]["tests"][ti]["quantity"], p=results[ci]["tests"][ti]["p"])
                                 for (ci, ti), r_ in zip(idx, rej) if r_],
                       min_p=float(min(pv)) if pv else None),
             integrity_ok=integrity, verdict=verdict, status="complete",
             compute_ledger=os.path.relpath(ledger_path(out), ROOT) if ledger_path(out).startswith(ROOT) else ledger_path(out))
    C.wait_for_disk(disk_path, log=log)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(S, fh, indent=1)
    os.replace(tmp, out)
    if verbose:
        print(f"check 4b: {n_rej}/{len(pv)} Holm rejections, integrity {integrity} -> {verdict}; wrote {out}", flush=True)
    return S


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h-A", type=float, default=H_CANDIDATES[0])
    ap.add_argument("--out", default=None)
    ap.add_argument("--cpus", default=None, help="pin this process, e.g. 64 (single-threaded)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--disk-path", default="/", help=argparse.SUPPRESS)          # file system of the disk pause
    a = ap.parse_args(argv)
    if a.cpus:
        lo, _, hi = a.cpus.partition("-")
        os.sched_setaffinity(0, set(range(int(lo), int(hi or lo) + 1)))
    if a.h_A not in H_CANDIDATES:
        raise SystemExit(f"--h-A must be one of {H_CANDIDATES}")
    cs = cases(a.h_A)
    out = os.path.abspath(a.out) if a.out else default_out(a.h_A)
    tot = 0
    for e in cs:
        ws = N_CHAINS * e["n_steps"]
        tot += ws
        print(f"{e['case']:10s} lam {e['lam']:6g} h {e['h']:<9g} label {e['label']}  n_steps {e['n_steps']:>9d}  stride "
              f"{e['stride_steps']:>6d}  samples {e['n_samples']:>7d}  lags {e['lags_steps']} (nominal "
              f"{[round(v, 3) for v in e['lags_nominal_steps']]})  phi {e['phi']:.6f}  EM excess {e['em_excess']:.3g}  "
              f"{ws:.3g} walker-steps", flush=True)
    print(f"total {tot:.3g} walker-steps (2 engine runs of 32 walkers per case) -> {out}", flush=True)
    if os.path.exists(out):
        old = json.load(open(out))
        same = old.get("signature") == signature(a.h_A, cs) and old.get("status") == "complete"
        print(f"output exists: {'complete with this signature (kept)' if same else 'DIFFERENT signature -> a run is refused'}",
              flush=True)
    try:
        print(f"engine sha256 {C.check_engine()} (file and import) == the frozen engine", flush=True)
    except C.EngineMismatch as e:
        print(f"REFUSED: {e}", flush=True)
        return 2
    print(f"compute ledger: {ledger_path(out)} (fold into results/final_campaign/compute_ledger.csv); disk pause below "
          f"{C.DISK_LOW / 1e9:.0f} GB free on {a.disk_path}", flush=True)
    if a.dry_run:
        return 0
    S = run_check(a.h_A, out, cs, disk_path=a.disk_path)
    return 0 if S["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
