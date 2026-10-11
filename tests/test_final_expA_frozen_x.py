"""Final campaign, Experiment A, check 4b tooling (scripts/final_campaign/expA_frozen_x.py; docs/final_campaign/
PREREGISTRATION.md section 1.3 row 4b).

F1  the estimator on SYNTHETIC EXACT AR(1) chains (numpy, the same six-case / 30-test family layout as the check):
    calibrated -- per-test false-reject rate ~5 % at alpha 0.05, family-wise rate under Holm <= ~5 % over 300
    replications; unbiased (chain means average to phi^l); and powerful where it must be (the resolvable-inflation case
    tells EM from exact: against the exact variance it rejects in >= 80 % of replications).
F2  the design: labels (one per (lam, case), 8650-8661 with two spares), realised integer lags, trace stride, run
    lengths, EM constants.
F3  the engine path (tiny): the two resolvable cases (2000 steps each) and a shortened lam = 128 case through
    gateway_family_numba.run_arm(_freeze_x_test=True) into a scratch file: x frozen, traces aligned with y0 / Y_final,
    finite; idempotent re-run; a different signature is refused, never overwritten; exactly one compute-ledger row per
    computing run (prereg section 3: checks count against the ceiling), none for a kept or refused run.
F4  resources (prereg section 3): a non-frozen engine is refused before any compute (no output, no ledger row; --dry-run
    exit 2); the disk pause is applied before every case and before the JSON is written (logged; a pause beyond the
    maximum stops the check with a FAILED ledger row and no output); a run that fails mid-way is ledgered FAILED.
"""
import csv
import importlib.util
import json
import math
import os
import sys
import types

import numpy as np
import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


FX = _load("expA_frozen_x_under_test", os.path.join(ROOT, "scripts", "final_campaign", "expA_frozen_x.py"))


@pytest.fixture(autouse=True)
def _ample_disk(monkeypatch):
    """Hermetic disk pause (the shared root file system may be below the prereg's 2 GB, which would legitimately pause
    run_check); F4 patches DISK_USAGE itself."""
    monkeypatch.setattr(FX.C, "DISK_USAGE", lambda path: types.SimpleNamespace(free=10 ** 12))


def _ledger_rows(out):
    p = FX.ledger_path(out)
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def _ar1(rng, R, C, K, a, var):
    """R x C independent stationary AR(1) chains of K samples, coefficient a per sample, stationary variance var."""
    d = np.empty((R, K, C))
    d[:, 0] = rng.standard_normal((R, C)) * math.sqrt(var)
    s = math.sqrt(var * (1.0 - a * a))
    for k in range(1, K):
        d[:, k] = a * d[:, k - 1] + s * rng.standard_normal((R, C))
    return d


def _synthetic_cases():
    """Six cases shaped like the check: four 'h_A-like' (stride = the first lag, 801 samples, lags 1/2/4/8 samples) and
    two 'resolvable-like' (stride 1, phi 0.9, 2001 samples, lags 3/5/10/20); the EM variance is the reference."""
    out = []
    for lam, h in ((1.0, 2.5e-5), (4.0, 2.5e-5), (16.0, 2.5e-5), (128.0, 2.5e-5)):
        e = FX.design(lam, 1.0, h)
        if e["stride_steps"] == 1:                           # lam 128 at h_A: keep the synthetic case light
            e = dict(e, stride_steps=78, lags_steps=[78, 156, 312, 624], lag_samples=[1, 2, 4, 8], n_samples=801)
        out.append(dict(e, lam=lam, h=h))
    for lam in (1.0, 16.0):
        e = FX.design(lam, 1.0, 0.1 / lam)
        out.append(dict(e, lam=lam, h=0.1 / lam))
    return out


def _family_p(cs, rng, R, var_ref=None, chunk=50):
    """p-values (R, 30) of the check's estimator on exact AR(1) data generated with the EM law of each case."""
    P = np.empty((R, 0))
    for e in cs:
        a = e["phi"] ** e["stride_steps"]
        K = e["n_samples"]
        ps = []
        for r0 in range(0, R, chunk):
            r1 = min(R, r0 + chunk)
            d = _ar1(rng, r1 - r0, FX.N_CHAINS, K, a, e["var_em"])
            for rr in range(r1 - r0):
                ests = FX.ar1_estimates(d[rr], e["var_em"] if var_ref is None else var_ref(e), e["phi"],
                                        e["lag_samples"], e["lags_steps"])
                ps.append([FX.t_test(v, expv)["p"] for _, expv, v in ests])
        P = np.hstack([P, np.array(ps)])
    return P


def test_F1_calibrated_false_reject_rate_on_exact_ar1():
    rng = np.random.default_rng(20261011)
    cs = _synthetic_cases()
    R = 300
    P = _family_p(cs, rng, R)
    assert P.shape == (R, 30)
    per_test = float((P < 0.05).mean())
    assert 0.03 < per_test < 0.075, per_test                                  # nominal 5 %
    fw = float(np.mean([any(FX._holm_reject(row, 0.05)) for row in P]))
    assert fw <= 0.09, fw                                                       # nominal <= 5 % (sd 1.3 % at R 300)
    # p-values roughly uniform (no gross miscalibration in either tail)
    assert 0.07 < float((P < 0.1).mean()) < 0.13 and 0.45 < float((P < 0.5).mean()) < 0.55


def test_F1_estimator_unbiased():
    rng = np.random.default_rng(7)
    e = FX.design(16.0, 1.0, 0.1 / 16.0)
    d = _ar1(rng, 40, FX.N_CHAINS, e["n_samples"], e["phi"], e["var_em"])
    vals = [FX.ar1_estimates(d[r], e["var_em"], e["phi"], e["lag_samples"], e["lags_steps"]) for r in range(40)]
    for qi, (q, expv, _) in enumerate(vals[0]):
        allv = np.concatenate([v[qi][2] for v in vals])
        se = allv.std(ddof=1) / math.sqrt(allv.size)
        assert abs(allv.mean() - expv) < 4 * se, (q, allv.mean(), expv, se)


def test_F1_power_resolvable_case_tells_em_from_exact():
    """Data from the EM law, tested against the EXACT variance (5.3 % lower): the variance test must reject."""
    rng = np.random.default_rng(3)
    e = FX.design(1.0, 1.0, 0.1)
    R = 100
    d = _ar1(rng, R, FX.N_CHAINS, e["n_samples"], e["phi"], e["var_em"])
    rej = 0
    for r in range(R):
        ests = FX.ar1_estimates(d[r], e["var_exact"], e["phi"], e["lag_samples"], e["lags_steps"])
        rej += FX.t_test(ests[0][2], 1.0)["p"] < 0.05 / 30
    assert rej / R >= 0.8, rej / R
    assert abs(e["em_excess"] - (1 / 0.95 - 1)) < 1e-12


def test_F1_holm_adjusted_matches_decisions():
    p = [0.001, 0.02, 0.0005, 0.3, 0.012]
    adj = FX.holm_adjusted(p)
    rej = FX._holm_reject(p, 0.05)
    assert [a <= 0.05 for a in adj] == rej
    assert np.allclose(sorted(adj)[:2], [5 * 0.0005, 4 * 0.001])


# ------------------------------------------------------------------------------------------------- F2 design
def test_F2_design_labels_and_lags():
    cfg = json.load(open(FX.CFG_VAL))["frozen_x_relaxation_test"]
    assert sorted(set(FX.LABELS.values()) | set(FX.SPARE_LABELS)) == sorted(cfg["seeds"]) == list(range(8650, 8662))
    assert len(set(FX.LABELS.values())) == len(FX.LABELS)                     # one label per (lam, case)
    for hA in FX.H_CANDIDATES:
        cs = FX.cases(hA)
        assert [(c["case"], c["lam"]) for c in cs] == [("h_A", 1.0), ("h_A", 4.0), ("h_A", 16.0), ("h_A", 128.0),
                                                       ("resolvable", 1.0), ("resolvable", 16.0)]
        assert len({c["label"] for c in cs}) == 6
        for c in cs:
            assert c["n_steps"] == round(200 / (c["lam"] * c["h"]))
            assert all(l % c["stride_steps"] == 0 for l in c["lags_steps"])
            assert c["n_samples"] == c["n_steps"] // c["stride_steps"] + 1
            assert abs(c["phi"] - (1 - c["lam"] * c["h"])) < 1e-15
            assert abs(c["var_em"] - (1 / 16) / (1 - c["lam"] * c["h"] / 2)) < 1e-15
            for l, nom in zip(c["lags_steps"], c["lags_nominal_steps"]):
                assert abs(l - nom) <= 0.5
            if c["case"] == "h_A" and c["lam"] <= 16:
                assert c["lags_steps"] == [round(nom) for nom in c["lags_nominal_steps"]]
                assert all(abs(l - nom) < 1e-6 for l, nom in zip(c["lags_steps"], c["lags_nominal_steps"]))
    c128 = FX.cases(2.5e-5)[3]
    assert c128["lags_steps"] == [78, 156, 313, 625] and c128["stride_steps"] == 1
    assert FX.cases(1.25e-5)[3]["lags_steps"] == [156, 313, 625, 1250]
    res = FX.cases(2.5e-5)[4]
    assert res["lags_steps"] == [3, 5, 10, 20] and res["n_steps"] == 2000 and abs(res["em_excess"] - 0.0526316) < 1e-6
    assert FX.cases(2.5e-5)[4]["label"] == FX.cases(1.25e-5)[4]["label"] == 8654          # h-independent case
    assert {c["label"] for c in FX.cases(2.5e-5)[:4]}.isdisjoint({c["label"] for c in FX.cases(1.25e-5)[:4]})
    with pytest.raises(ValueError):
        FX.cases(3e-5)
    # engine cfg: production knobs, fr_interval_t = h only where 0.004 is not an integer number of steps
    assert FX.engine_cfg(16.0, 2.5e-5)["fr_interval_t"] == 0.004
    assert FX.engine_cfg(1.0, 0.1)["fr_interval_t"] == 0.1


# ------------------------------------------------------------------------------------------------- F3 engine (tiny)
def test_F3_engine_path_idempotent_and_refusal(tmp_path):
    pytest.importorskip("gateway_family_numba")
    out = str(tmp_path / "check4b_tiny.json")
    cs = [c for c in FX.cases(2.5e-5) if c["case"] == "resolvable"]
    c128 = dict(FX.cases(2.5e-5)[3])
    c128.update(FX.design(128.0, 1.0, 2.5e-5, n_relax=20.0))                 # shortened: 6250 steps
    cs.append(c128)
    S = FX.run_check(2.5e-5, out=out, cs=cs, verbose=False)
    assert S["status"] == "complete" and S["integrity_ok"] and S["holm"]["n_tests"] == 15
    rows = _ledger_rows(out)
    assert len(rows) == 1 and rows[0]["stage"] == "expA-check4b" and rows[0]["threads"] == "1"
    assert list(rows[0]) == FX.C.COMPUTE_FIELDS and rows[0]["note"].startswith(f"verdict {S['verdict']}")
    assert abs(float(rows[0]["used_core_h"]) - float(rows[0]["wall_s"]) / 3600) < 1e-5 and float(rows[0]["wall_s"]) > 0
    assert S["compute_ledger"] == FX.ledger_path(out)
    assert S["provenance"]["engine_sha256_checked"] == FX.C.EXPECTED_ENGINE_SHA256
    for c in S["cases"]:
        assert [h["noise_seed"] for h in c["halves"]] == [(1_000_003 * c["label"] + 7919 * N) % (2 ** 31 - 1) for N in (32, 64)]
        assert all(h["x_frozen"] and h["finite"] and h["trace_start_equals_y0"] for h in c["halves"])
        assert [t["quantity"] for t in c["tests"]] == ["var"] + [f"acf_{l}" for l in c["lags_steps"]]
        for t in c["tests"]:
            assert 0.0 <= t["p"] <= 1.0 and "p_holm" in t and "vs_exact" in t
    assert S["provenance"]["engine_sha256"] == FX.sha256_file(FX.ENGINE)
    before = open(out).read()
    S2 = FX.run_check(2.5e-5, out=out, cs=cs, verbose=False)                 # idempotent: kept, not re-run
    assert open(out).read() == before and S2["verdict"] == S["verdict"]
    with pytest.raises(SystemExit, match="different signature"):
        FX.run_check(2.5e-5, out=out, cs=cs[:2], verbose=False)
    assert open(out).read() == before
    assert len(_ledger_rows(out)) == 1                                        # kept / refused runs compute nothing


# ------------------------------------------------------------------------------------------------- F4 resources
def _tiny_cases():
    return [c for c in FX.cases(2.5e-5) if c["case"] == "resolvable"]       # 2 x 2000 steps x 64 chains


def test_F4_non_frozen_engine_refused_before_compute(tmp_path, monkeypatch, capsys):
    pytest.importorskip("gateway_family_numba")
    out = str(tmp_path / "c.json")
    monkeypatch.setattr(FX.C, "EXPECTED_ENGINE_SHA256", "0" * 64)
    with pytest.raises(SystemExit, match="refused"):
        FX.run_check(2.5e-5, out=out, cs=_tiny_cases(), verbose=False)
    assert not os.path.exists(out) and not _ledger_rows(out)
    assert FX.main(["--dry-run", "--out", out]) == 2 and "REFUSED" in capsys.readouterr().out
    monkeypatch.undo()                                                         # (also undoes _ample_disk: re-patch)
    monkeypatch.setattr(FX.C, "DISK_USAGE", lambda path: types.SimpleNamespace(free=10 ** 12))
    assert FX.main(["--dry-run", "--out", out]) == 0 and not os.path.exists(out) and not _ledger_rows(out)


class _FakeDisk:
    """shutil.disk_usage stand-in: free bytes from a list (last value repeated); counts the calls."""
    def __init__(self, frees):
        self.frees, self.n = list(frees), 0

    def __call__(self, path):
        v = self.frees[min(self.n, len(self.frees) - 1)]
        self.n += 1
        return types.SimpleNamespace(free=v)


def test_F4_disk_pause_before_every_case_and_the_json(tmp_path, monkeypatch):
    pytest.importorskip("gateway_family_numba")
    out = str(tmp_path / "c.json")
    # case 1: free (5 GB); case 2: 1.5 GB -> pause, 2.5 GB (still paused: resume needs 3 GB), 3.5 GB -> resume; JSON: free
    disk = _FakeDisk([5e9, 1.5e9, 2.5e9, 3.5e9, 5e9])
    monkeypatch.setattr(FX.C, "DISK_USAGE", disk)
    monkeypatch.setattr(FX.C, "DISK_POLL_S", 0.0)
    S = FX.run_check(2.5e-5, out=out, cs=_tiny_cases(), verbose=False)
    assert S["status"] == "complete" and disk.n == 5                          # 2 case starts + 2 polls + the JSON write
    log = open(os.path.join(str(tmp_path), "disk_pause.log")).read()
    assert log.count("DISK PAUSE:") == 1 and log.count("disk pause continues") == 1 and log.count("DISK RESUME") == 1
    assert len(_ledger_rows(out)) == 1
    # a pause that does not clear within the maximum (stopping rule iv): FAILED ledger row, no output written
    out2 = str(tmp_path / "c2.json")
    monkeypatch.setattr(FX.C, "DISK_USAGE", _FakeDisk([5e9, 5e9, 1e9]))      # both cases run; the JSON write pauses
    monkeypatch.setattr(FX.C, "DISK_MAX_PAUSE_S", 0.0)
    with pytest.raises(FX.C.DiskPauseTimeout):
        FX.run_check(2.5e-5, out=out2, cs=_tiny_cases(), verbose=False)
    assert not os.path.exists(out2)
    rows = _ledger_rows(out2)                                                  # same directory: one ledger
    assert len(rows) == 2 and rows[1]["note"].startswith("FAILED: DiskPauseTimeout") and float(rows[1]["wall_s"]) > 0


def test_F4_failed_run_is_ledgered(tmp_path, monkeypatch):
    pytest.importorskip("gateway_family_numba")
    out = str(tmp_path / "c.json")
    real, n = FX.run_case, [0]

    def boom(case, **kw):
        n[0] += 1
        if n[0] == 2:
            raise RuntimeError("engine blew up")
        return real(case, **kw)
    monkeypatch.setattr(FX, "run_case", boom)
    with pytest.raises(RuntimeError, match="blew up"):
        FX.run_check(2.5e-5, out=out, cs=_tiny_cases(), verbose=False)
    rows = _ledger_rows(out)
    assert not os.path.exists(out) and len(rows) == 1 and "FAILED: RuntimeError: engine blew up" in rows[0]["note"]
    assert float(rows[0]["wall_s"]) > 0
