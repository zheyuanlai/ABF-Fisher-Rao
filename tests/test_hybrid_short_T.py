"""Tests of the hybrid short-T wall-clock study harness (scripts/hybrid_short_T/).

Budget / grid arithmetic of every rung, the benchmark window, job identity, the persistence and endpoint
definitions, the frozen routing rule, the allocator's core choice, the analysis statistics with censoring, and the
inertness of the S1 timing hooks at a tiny size.  Engine-level identities at production sizes are Stage I
(scripts/hybrid_short_T/validate.py, results/hybrid_short_T/validation/).
"""
import copy
import json
import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "hybrid_short_T"))

import hst_common as H  # noqa: E402

SHORT_T = os.path.join(ROOT, "results", "equal_budget_v2", "short_T", "summary.json")
EXPECTED = {"gateway": (1_032_192_000, 2048, 12.6, 63), "lta300": (262_656_000, 1024, 51.3, 171),
            "lta150": (118_272_000, 1024, 23.1, 77)}


@pytest.fixture(scope="module")
def specs():
    return H.all_specs()


# ------------------------------------------------------------------------------------------------ budgets / grids
@pytest.mark.parametrize("system", H.SYSTEMS)
def test_budget_matches_task_and_short_T(specs, system):
    s = specs[system]
    B, N0, Tstar, kstar = EXPECTED[system]
    assert s["B"] == B and s["N0"] == N0 and s["k_star"] == kstar
    assert abs(s["T_star"] - Tstar) < 1e-9
    S = json.load(open(SHORT_T))
    key = {"gateway": "gateway", "lta300": "lta_T300", "lta150": "lta_T150"}[system]
    assert abs(S[key]["B_new"] - B) < 0.5 and abs(S[key]["T_star"] - Tstar) < 1e-9
    for N in s["N_ladder"]:
        n = H.n_steps_for(s, N)
        assert n * N == B
        assert abs(H.T_N(s, N) - n * s["h"]) < 1e-12


@pytest.mark.parametrize("system", H.SYSTEMS)
def test_save_grid(specs, system):
    s = specs[system]
    counts = []
    for N in s["N_ladder"]:
        st, pm, um = H.save_grid(s, N)
        n = H.n_steps_for(s, N)
        assert st[-1] == n and np.all(np.diff(st) > 0) and st[0] >= 1
        assert int(um.sum()) == s["k_star"]
        assert np.all(pm[um])                               # uniform points are persistence-grid points
        assert np.all(np.isin(H.orig_budget_steps(s, N), st[pm]))   # the original budget grid is in it (A2.1)
        assert np.all(np.isin(H.dense_steps(n), st[pm]))             # so is the dense grid
        u = st[pm] / n
        big = u[:-1] >= 1e-3
        assert np.max(u[1:][big] / u[:-1][big]) <= 1.03     # relative spacing <= ~2 % (rounding at small n_steps)
        counts.append(int(pm.sum()))
        uu = st[um] / n
        assert np.allclose(uu, np.arange(1, s["k_star"] + 1) / s["k_star"], rtol=0, atol=1e-12)
    assert max(counts) / min(counts) < 1.10                 # the same fractions up to integer-step collisions at u < 1e-3


@pytest.mark.parametrize("system", H.SYSTEMS)
def test_bench_window(specs, system):
    s = specs[system]
    for N in s["N_ladder"]:
        n_b, st, s0 = H.bench_window(s, N)
        assert st[-1] == n_b and s0 < n_b and s0 in st
        if s["kind"] == "lta" and N > 1:
            assert s0 >= s["fr_start"]                       # FR active over the steady-state window
        assert n_b <= H.n_steps_for(s, N)


def test_spec_cross_check_refuses_a_mismatch():
    X = H.load_config()
    Y = copy.deepcopy(X)
    Y["systems"]["gateway"]["B"] = 1_000_000_000
    with pytest.raises(H.HSTError):
        H.system_spec("gateway", Y)
    Y = copy.deepcopy(X)
    Y["systems"]["lta300"]["thresholds_e_F"]["mid"] = 0.1
    with pytest.raises(H.HSTError):
        H.system_spec("lta300", Y)


def test_seeds_disjoint(specs):
    seen = set()
    for s in specs.values():
        blocks = [s["prod_seeds"], s["confirm_seeds"], s["bench_seeds"], s["validation_seeds"]]
        allb = [x for b in blocks for x in b]
        assert len(allb) == len(set(allb))
        assert not set(allb) & set(s["production_reference_seeds"])
        assert not (set(allb) & seen) or s["kind"] == "lta"   # LTA streams include T in the SeedSequence
        seen |= set(allb)


# ------------------------------------------------------------------------------------------------ jobs
def test_job_identity(specs):
    s = specs["lta300"]
    j = H.make_job("prod", s, "fr", 64, 30500, "parallel", 8)
    assert j["n_steps"] == H.n_steps_for(s, 64) and j["noise"] == "counter"
    j1 = dict(j, replicate=1)
    assert H.job_key(j) != H.job_key(j1)
    assert H.job_signature(j, s) == H.job_signature(j1, s)            # the replicate is not in the signature
    p = H.job_paths(j1)
    assert p["json"].endswith(os.path.join("prod", "lta300", "fr64", "s30500.rep1.json"))
    with pytest.raises(H.HSTError):
        H.make_job("prod", s, "fr", 1, 30500, "serial", 1)
    with pytest.raises(H.HSTError):
        H.make_job("prod", s, "abf", 4, 30500, "parallel", 8)
    b = H.make_job("bench", s, "abf", 16, 30590, "serial", 1, r=0)
    assert b["n_steps"] == H.bench_window(s, 16)[0] and b["noise"] == "replay"


# ------------------------------------------------------------------------------------------------ endpoints
class _FakeScorer:
    e_flat = 1.0

    def __init__(self, e):
        self.e = np.asarray(e, float)

    def e_F(self, res):
        return self.e


def _fake_res(spec, N, e, wall):
    st, _, _ = H.save_grid(spec, N)
    return dict(save_step=st, wall_s_at_save=np.asarray(wall, float), n_force_evals_at_save=N * st,
                wall_total_s=float(wall[-1]) + 0.1)


def test_persistence_on_the_budget_grid_and_censoring(specs):
    s = specs["gateway"]
    N = 2048
    st, bm, um = H.save_grid(s, N)
    S = st.size
    wall = np.linspace(0.1, 10.0, S)
    thr = s["thresholds"]["mid"]
    e = np.full(S, 0.1)
    k = int(np.flatnonzero(bm)[40])
    e[k:] = thr * 0.5
    # a physical-checkpoint save that fails AFTER k must not break persistence (it is not on the budget grid)
    phys = np.flatnonzero(~bm)
    later = [i for i in phys if i > k]
    if later:
        e[later[0]] = 1.0
    job = H.make_job("prod", s, "fr", N, 9600, "parallel", 16)
    ep, _ = H.run_endpoints(_fake_res(s, N, e, wall), s, _FakeScorer(e), job)
    assert ep["idx_mid"] == k and ep["W_mid"] == pytest.approx(wall[k]) and not ep["censored_mid"]
    assert ep["force_evals_mid"] == N * st[k]
    pidx = np.flatnonzero(bm)
    kprev = pidx[np.searchsorted(pidx, k) - 1]
    assert ep["gap_ratio_mid"] == pytest.approx(st[k] / st[kprev])
    assert wall[kprev] <= ep["W_mid_interp"] <= ep["W_mid"]          # log-linear crossing inside the last gap
    e2 = e.copy()
    e2[-1] = 1.0                                             # fails at the last save -> censored, never imputed
    ep2, _ = H.run_endpoints(_fake_res(s, N, e2, wall), s, _FakeScorer(e2), job)
    assert ep2["censored_mid"] and math.isinf(ep2["W_mid"]) and math.isinf(ep2["W_mid_e2e"])
    assert ep2["Ibar_F"] == pytest.approx(np.mean(e2[um]))


# ------------------------------------------------------------------------------------------------ routing rule
def _summ(system, N, s1, par):
    out = {}
    out[f"{system}|{H.config_name('abf', N, 'serial', 1)}"] = dict(n=3, c_ss_median=np.median(s1), c_ss_min=min(s1),
                                                                  c_ss_max=max(s1), threads=1)
    for t, v in par.items():
        out[f"{system}|{H.config_name('abf', N, 'parallel', t)}"] = dict(n=3, c_ss_median=np.median(v),
                                                                        c_ss_min=min(v), c_ss_max=max(v), threads=t)
    return out


def test_routing_rule():
    import benchmark as BM
    # clear win of P8; P16 within 3 % -> the tie goes to fewer threads (P8)
    s = _summ("gateway", 512, [80, 81, 82], {1: [60, 61, 62], 8: [10.0, 10.1, 10.2], 16: [9.9, 10.0, 10.1]})
    r = BM.route_one(s, "gateway", "abf", 512)
    assert r["backend"] == "P8" and r["P_star"]["tied_threads"] == [8, 16]
    # gain below 5 % -> S1
    s = _summ("gateway", 4, [80, 81, 82], {1: [78, 78.5, 79], 2: [79, 80, 81], 4: [85, 86, 87]})
    assert BM.route_one(s, "gateway", "abf", 4)["backend"] == "S1"
    # gain >= 5 % in the median but overlapping repeats -> S1
    s = _summ("gateway", 8, [70, 80, 90], {1: [60, 75, 95], 2: [90, 91, 92]})
    r = BM.route_one(s, "gateway", "abf", 8)
    assert r["backend"] == "S1" and not r["reliably_faster"]
    # no parallel data -> S1
    s = _summ("lta300", 1, [800, 801, 802], {})
    assert BM.route_one(s, "lta300", "abf", 1)["backend"] == "S1"


# ------------------------------------------------------------------------------------------------ allocator
class _Reg:
    def __init__(self, held=()):
        self._held = {i: {} for i in held}

    def held(self, cleanup=True):
        return dict(self._held)


def test_allocator_choice():
    import expB_alloc as A
    import hst_sched as S
    al = S.HSTAllocator(_Reg(held=(0,)), reserved=(15,), threshold=0.10)
    busy = {c: 0.0 for c in range(256)}
    for c in A.ccd_cores(1)[:3]:                 # CCD 1: three busy cores
        busy[c] = 1.0
    for i in range(2, 15):                        # every other CCD: 5 busy cores, so any 4 of its cores include one
        for c in A.ccd_cores(i)[:5]:
            busy[c] = 0.5
    best, least, key = al.choose(4, busy)
    assert best is not None and best[0] == (1,)          # its 5 idle cores qualify ...
    assert set(best[1]).isdisjoint(A.ccd_cores(1)[:3])   # ... and the busy ones are avoided
    assert 0 not in best[0] and 15 not in best[0]        # held / reserved CCDs are never chosen
    busy16 = {c: 0.3 for c in range(256)}
    best, least, key = al.choose(16, busy16)
    assert best is None and least is not None and len(least[0]) == 2
    assert A.node_of_ccd(least[0][0]) == A.node_of_ccd(least[0][1])


# ------------------------------------------------------------------------------------------------ statistics
def test_censoring_aware_statistics():
    import analyze as AN
    rng = np.random.default_rng(0)
    a = np.array([1.0, 2.0, 3.0, np.inf, np.inf])
    assert AN.med_inf(a) == 3.0
    assert math.isinf(AN.med_inf(np.array([1.0, np.inf, np.inf])))
    p = AN.paired_W(np.array([2.0, 2.0, 2.0, np.inf]), np.array([1.0, 1.0, np.inf, np.inf]), rng)
    assert p["wins"] == 2 and p["losses"] == 1 and p["ties"] == 1
    u = AN._ratio_of_medians_boot(np.full(20, 2.0), np.full(20, 1.0), rng)
    assert u["ratio"] == 0.5 and u["hi"] < 1 and AN.decision(u["lo"], u["hi"]) == "FR faster"
    assert AN.decision(0.8, 1.2) == "unresolved" and AN.decision(1.1, 1.5) == "ABF faster"
    assert AN.joint_decision("FR faster", "FR faster") == "FR faster"
    assert AN.joint_decision("FR faster", "unresolved").startswith("unresolved")
    r = AN.ratio_arr([1.0, np.inf, np.inf, 2.0], [np.inf, 1.0, np.inf, 4.0])
    assert r[0] == 0.0 and np.isinf(r[1]) and np.isnan(r[2]) and r[3] == 0.5
    h = AN.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert h["a"]["reject"] and not h["b"]["reject"] and h["c"]["p_holm"] == pytest.approx(0.06)


def test_R_wall_ci_with_censored_abf():
    """A2.3: the review's demonstration -- ABF 17/32 censored, FR finite: R = 0 and the CI must be near 0 (it read
    [1.88, 1.92] with eqb_metrics' relative difference, which drops resamples where ABF's best is censored)."""
    import analyze as AN
    rng = np.random.default_rng(3)
    runs = {("abf", 8): {s: dict(W_mid=(np.inf if s < 17 else float(1.0 + rng.random())), flags={"mid": False})
                         for s in range(32)},
            ("fr", 8): {s: dict(W_mid=float(2.0 + rng.random()), flags={"mid": False}) for s in range(32)}}
    b = AN.best_alloc(runs, "W_mid", [8])
    assert b["R_wall"]["point"] == 0.0
    assert b["R_wall"]["ci95"][0] == 0.0 and b["R_wall"]["frac_resamples_fr_better"] > 0.5


def test_best_alloc_reselects_and_reports_R():
    import analyze as AN
    rng = np.random.default_rng(1)
    runs = {}
    for N, mu in ((4, 10.0), (64, 6.0)):
        runs[("abf", N)] = {s: dict(W_mid=float(mu * np.exp(0.1 * rng.normal())), flags={"mid": False})
                            for s in range(16)}
    for N, mu in ((4, 12.0), (64, 3.0)):
        runs[("fr", N)] = {s: dict(W_mid=float(mu * np.exp(0.1 * rng.normal())), flags={"mid": False})
                           for s in range(16)}
    b = AN.best_alloc(runs, "W_mid", [64, 4])
    assert b["abf"]["best_N"] == 64 and b["fr"]["best_N"] == 64
    assert 0.4 < b["R_wall"]["point"] < 0.6 and b["R_wall"]["ci95"][1] < 1


# ------------------------------------------------------------------------------------------------ hooks (tiny)
@pytest.mark.parametrize("system,N,n", [("gateway", 4, 4000), ("lta300", 2, 400)])
def test_serial_hooks_inert_tiny(system, N, n):
    import expB_common as C
    s = H.system_spec(system)
    s = dict(s, engine_cfg=dict(s["engine_cfg"], **({"fr_start_steps": 100, "warmup_steps": 100,
                                                       "burn_in_steps": 100} if s["kind"] == "lta" else {})))
    saves = np.array([n // 4, n // 2, n], np.int64)
    hooked = C.run_serial(s, "fr", N, 9695, n, saves)
    E = __import__(s["validated_module"])
    if s["kind"] == "gateway":
        plain = E.run_arm(dict(s["engine_cfg"]), "fr", N, 9695, n, saves, diagnostics=False)
    else:
        c = E.make_cfg(float(s["T_K"]), N, 9695, **dict(s["engine_cfg"], diagnostics=False, n_steps=n))
        plain = E.run_arm(c, "fr", saves)
    for k in C.SCORE_KEYS[s["kind"]]:
        assert np.asarray(plain[k]).tobytes() == np.asarray(hooked[k]).tobytes()
    w = np.asarray(hooked["wall_s_at_save"])
    assert np.all(np.isfinite(w)) and np.all(np.diff(w) >= 0)


# ------------------------------------------------------------------------------------------------ campaign (A2.6)
def test_trim_to_whole_seed_blocks(monkeypatch):
    import run_campaign as RC
    jobs = [(b, dict(id=(b, k))) for b in range(4) for k in range(3)]
    monkeypatch.setattr(RC, "projected_core_h", lambda j, summary: 1.0)
    keep, need, cost = RC.trim_blocks(jobs, None, remaining=7.5, done_fn=lambda j: False)
    assert keep == [0, 1] and need == 6.0                    # 3 core-h per block: 2 whole blocks fit 7.5
    keep, need, _ = RC.trim_blocks(jobs, None, remaining=7.5, done_fn=lambda j: j["id"][0] == 0)
    assert keep == [0, 1, 2] and need == 6.0                 # block 0 already done costs nothing


def test_expected_attempts():
    import benchmark as BM
    s = {"contended_fraction_by_threads": {"16": {"frac": 0.5}, "1": {"frac": 0.0}}}
    assert BM.expected_attempts(s, 1) == 1.0
    assert BM.expected_attempts(s, 16) == pytest.approx(1 + 0.5 + 0.25 + 0.125)
