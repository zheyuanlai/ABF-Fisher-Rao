"""Tests of the Experiment B harness (scripts/final_campaign/expB_{common,alloc,bench,run,equivalence,analyze}.py),
before the walker-parallel engines exist: a FAKE engine implementing the frozen interface
run_arm_parallel(cfg, method, N, seed, n_steps, save_steps, *, threads, noise, timers) returns synthetic contract
dicts whose accumulators converge to the frozen references at a seed-dependent rate (so the real equal-budget scorers
give a decreasing e_F, censored at the strict gateway threshold for small N).

Covered: the Exp B save grid; persistence / censoring; W and the log-linear interpolated W; the Ibar_F^wall
conventions (flat profile before the first save, final error after the last); the result contract; save/load round
trip; CCD topology, /proc/stat parsing and the allocation logic on a MOCKED /proc/stat (idle check incl. SMT siblings,
held CCDs, wait then least-loaded + flag, stale-lock cleanup); the monitor on mocked /proc readers; the contention
rule (foreign busy incl. own-thread subtraction, context-switch rate, endpoint windows, unknown telemetry); the
benchmark plan, gate, tie rule and competitor completion; the production order; the scheduler (repeat policy,
resumability by signature, refusal of a mismatching record, failed runs retried, projection gate, disk pause with a
mocked shutil.disk_usage); the V-B4 rules; the analysis with flagged seeds; one real pinned subprocess run.

  CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_final_expB_harness.py -q -p no:cacheprovider --basetemp <scratch>
"""
import collections
import importlib
import json
import math
import os
import sys
import time

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "final_campaign"))
import expB_common as C  # noqa: E402
import expB_alloc as A  # noqa: E402
import expB_bench as BB  # noqa: E402
import expB_run as R  # noqa: E402
import expB_equivalence as EQ  # noqa: E402

INF = math.inf
OUR_CCD = 15                      # cores 120-127: the only CCD a test may pin to
FAKE_ENGINE = r'''
"""Fake Experiment B engine (tests only): synthetic contract dicts, deterministic, thread-invariant science."""
import json, math, os
import numpy as np
ROOT = %(root)r
_REF = {}

def _ref(kind):
    if kind not in _REF:
        if kind == "gateway":
            z = np.load(os.path.join(ROOT, "results/equal_budget_v2/references/gateway_reference.npz"))
            c = -1.8 + (np.arange(180) + 0.5) * 0.02
            _REF[kind] = np.interp(c, z["x_grid"], z["Fp_ref"])
        else:
            z = np.load(os.path.join(ROOT, "results/equal_budget_v2/references/lta_T150_reference.npz"))
            _REF[kind] = np.asarray(z["gamma_ref"], float)
    return _REF[kind]

def run_arm_parallel(cfg, method, N, seed, n_steps, save_steps, *, threads, noise, timers=True):
    if os.environ.get("FAKE_FAIL_SEED") and int(os.environ["FAKE_FAIL_SEED"]) == int(seed):
        raise RuntimeError("fake engine failure")
    kind = "lta" if cfg.get("system") == "lta" else "gateway"
    k = np.asarray(save_steps, dtype=np.int64)
    rate = 0.08 if method == "fr" else 0.2
    tau = n_steps * rate * (1.0 + 0.3 * ((int(seed) * 7919) %% 13) / 13.0)
    a = 1.0 - np.exp(-k / tau)
    fp = _ref(kind)
    if kind == "gateway":
        Cc = np.outer(N * k, np.ones(180)) / 180.0
        M = (a[:, None] * fp[None, :]) * (Cc + 1.0)
        nfe_at, nfe = N * k, N * n_steps
    else:
        Cc = np.outer(N * (k + 1), np.ones(180)) / 180.0
        M = (a[:, None] * fp[None, :]) * Cc
        nfe_at, nfe = N * (k + 1), N * (n_steps + 1)
    cost = 1e-7 * N * (1.3 if method == "fr" else 1.0)
    wall = k * cost / max(1, threads) + 0.01
    out = dict(save_step=k, save_t=k * 1e-3, save_u=k / n_steps, M_all=M, C_all=Cc,
               fr_deaths_cum=np.zeros(k.size, np.int64), fr_opp_cum=np.zeros(k.size, np.int64),
               n_force_evals=np.int64(nfe), X_final=np.zeros(N), Y_final=np.zeros(N), max_abs_bias=np.float64(1.0),
               wall_s_at_save=wall, n_force_evals_at_save=nfe_at.astype(np.int64), wall_total_s=float(wall[-1] + 1e-3),
               warmup_s=0.5, t_force_move_s=0.6 * wall[-1], t_abf_s=0.2 * wall[-1], t_sync_s=0.1 * wall[-1],
               t_fr_s=(0.1 * wall[-1] if method == "fr" else 0.0), t_save_s=1e-3, threads=int(threads),
               noise_mode=noise, engine="fake/1", meta=json.dumps(dict(fake=True)), cfg=dict(cfg))
    if kind == "lta":
        out.update(M_prod=M.copy(), C_prod=Cc.copy(), cum_deaths=np.zeros(k.size, np.int64),
                   cum_opps=np.zeros(k.size, np.int64), q_final=np.zeros((N, 2, 3)))
        for key in ("fr_deaths_cum", "fr_opp_cum", "X_final", "Y_final", "max_abs_bias"):
            out.pop(key)
    return out
'''


@pytest.fixture(scope="session")
def fake_engine(tmp_path_factory):
    d = tmp_path_factory.mktemp("fake_engine")
    with open(d / "expb_fake_engine.py", "w") as fh:
        fh.write(FAKE_ENGINE % dict(root=ROOT))
    sys.path.insert(0, str(d))
    importlib.invalidate_caches()
    import expb_fake_engine
    return dict(module=expb_fake_engine, name="expb_fake_engine", dir=str(d))


@pytest.fixture(scope="session")
def gw_spec():
    return C.system_spec("gateway", overrides=dict(B=2048 * 100, seeds=[1, 2, 3, 4, 5, 6]))


@pytest.fixture(scope="session")
def gw_scorer(gw_spec):
    return C.Scorer(gw_spec)


# ============================================================================================ grid
def test_save_grid_construction():
    u, ul, ug = C.expb_u_grid()
    assert ul.size == 200 and np.array_equal(ul, np.arange(1, 201) / 200)
    assert ug.size == 466 and ug[0] == 1e-4 and ug[-1] <= 1.0 < ug[-1] * 1.02
    assert np.allclose(ug[1:] / ug[:-1], 1.02)
    assert u.size == np.unique(np.concatenate([ul, ug])).size
    for n in (6_400_000, 204_800_000, 3_276_800_000, 600_000, 153_600_000):
        steps, uu = C.expb_save_grid(n)
        assert steps.dtype == np.int64 and np.all(np.diff(steps) > 0)
        assert steps[0] >= 1 and steps[-1] == n
        assert np.array_equal(steps, np.unique(np.clip(np.round(u * n).astype(np.int64), 1, n)))
        iu = C.uniform_save_indices(steps, n)                   # the 200 uniform saves are exactly present
        assert np.array_equal(steps[iu], np.arange(1, 201) * (n // 200))
    small, _ = C.expb_save_grid(10)
    assert np.array_equal(small, np.arange(1, 11))              # clip + unique at tiny n
    a, _ = C.expb_save_grid(6_400_000)
    b, _ = C.expb_save_grid(204_800_000)
    assert a.size == b.size and np.allclose(a / 6_400_000, b / 204_800_000, rtol=0,
                                            atol=0.5 / 6_400_000 + 0.5 / 204_800_000)   # identical u up to rounding


# ============================================================================================ endpoints
def test_persistence_and_censoring():
    e = np.array([0.5, 0.1, 0.004, 0.006, 0.003, 0.002])
    assert C.persistent_index(e, 0.0055) == 4               # re-crossing at index 3 resets the window
    assert C.persistent_index(e, 0.01) == 2
    assert C.persistent_index(e, 1.0) == 0
    assert C.persistent_index(e, 0.001) == -1               # censored
    assert C.persistent_index(np.array([0.1, np.nan, 0.001]), 0.01) == 2   # NaN never meets
    assert C.persistent_index(np.array([0.001, 0.001, np.nan]), 0.01) == -1
    wall = np.arange(1, 7, dtype=float)
    assert C.w_at(wall, 4) == 5.0 and C.w_at(wall, -1) == INF


def test_interpolated_W():
    wall = np.array([1.0, 2.0, 3.0, 4.0])
    e = np.array([0.1, 0.04, 0.01, 0.005])
    thr = 0.02
    i = C.persistent_index(e, thr)
    assert i == 2
    w = C.w_interp(e, wall, thr, e_flat=0.3)
    frac = (math.log(0.04) - math.log(0.02)) / (math.log(0.04) - math.log(0.01))
    assert w == pytest.approx(2.0 + frac * 1.0) and 2.0 < w < C.w_at(wall, i)
    # first save already persistent: interpolate from the timed start with the flat-profile error
    w0 = C.w_interp(e, wall, 0.2, e_flat=0.4)
    assert w0 == pytest.approx(1.0 * (math.log(0.4) - math.log(0.2)) / (math.log(0.4) - math.log(0.1)))
    assert C.w_interp(e, wall, 0.001, 0.3) == INF          # censored
    # the threshold hit exactly at the persistent save -> w_i
    assert C.w_interp(np.array([0.1, 0.02]), np.array([1.0, 2.0]), 0.02, 0.3) == pytest.approx(2.0)


def test_ibar_wall_conventions():
    wall = np.array([1.0, 3.0, 6.0])
    e = np.array([0.5, 0.2, 0.1])
    ef = 1.0
    # [0,1): flat 1.0; [1,3): 0.5; [3,6): 0.2; >= 6 (after completion): final 0.1
    assert C.ibar_wall(e, wall, 0.5, ef) == pytest.approx(1.0)
    assert C.ibar_wall(e, wall, 2.0, ef) == pytest.approx((1.0 + 0.5) / 2.0)
    assert C.ibar_wall(e, wall, 6.0, ef) == pytest.approx((1.0 + 2 * 0.5 + 3 * 0.2) / 6.0)
    assert C.ibar_wall(e, wall, 10.0, ef) == pytest.approx((1.0 + 2 * 0.5 + 3 * 0.2 + 4 * 0.1) / 10.0)
    with pytest.raises(C.ExpBError):
        C.ibar_wall(e, wall, INF, ef)
    with pytest.raises(C.ExpBError):
        C.ibar_wall(e, np.array([1.0, 0.5, 2.0]), 1.0, ef)
    g = C.e_at_wall(e, wall, [0.0, 0.99, 1.0, 2.9, 3.0, 100.0], ef)
    assert np.array_equal(g, [1.0, 1.0, 0.5, 0.5, 0.2, 0.1])


def test_run_endpoints_with_fake_engine(fake_engine, gw_spec, gw_scorer):
    N = 16
    n = C.n_steps_for(gw_spec["B"], N)
    saves, _ = C.expb_save_grid(n)
    res = C.run_parallel(gw_spec, "abf", N, 3, n, saves, 2, "counter", module=fake_engine["module"])
    assert res["implementation"] == "parallel" and res["t0_epoch_source"] == "return epoch - wall_total_s"
    ep, e = C.run_endpoints(res, gw_spec, gw_scorer, N, n, 2)
    assert e[0] > 0.2 and e[-1] < 0.0055                     # converges below mid, not below strict
    assert ep["censored_strict"] and ep["W_strict"] == INF and ep["used_core_h_strict"] == INF
    i = ep["idx_mid"]
    assert ep["W_mid"] == res["wall_s_at_save"][i] and ep["force_evals_mid"] == N * saves[i]
    assert ep["used_core_h_mid"] == pytest.approx(2 * ep["W_mid"] / 3600)
    assert ep["alloc_core_h_mid"] == pytest.approx(16 * ep["W_mid"] / 3600)
    assert ep["W_mid_interp"] <= ep["W_mid"]
    assert ep["Ibar_F"] == pytest.approx(np.mean(e[C.uniform_save_indices(saves, n)]))


def test_contract_violations(fake_engine, gw_spec):
    n = 1000
    saves, _ = C.expb_save_grid(n)
    res = fake_engine["module"].run_arm_parallel(C.engine_cfg_for(gw_spec, 4, 1, n), "abf", 4, 1, n, saves,
                                                 threads=1, noise="counter")
    assert C.contract_problems(res, "gateway", 4, n, saves) == []
    bad = dict(res, n_force_evals_at_save=res["n_force_evals_at_save"] + 1)
    assert any("n_force_evals_at_save" in p for p in C.contract_problems(bad, "gateway", 4, n, saves))
    bad = dict(res)
    bad.pop("t_sync_s")
    assert "missing" in C.contract_problems(bad, "gateway", 4, n, saves)[0]
    bad = dict(res, wall_s_at_save=res["wall_s_at_save"][::-1].copy())
    assert any("non-decreasing" in p for p in C.contract_problems(bad, "gateway", 4, n, saves))
    with pytest.raises(C.ExpBError):
        C.check_contract(dict(res, noise_mode="other"), "gateway", 4, n, saves)


def test_save_load_roundtrip(tmp_path):
    res = dict(M_all=np.arange(6.0).reshape(2, 3), save_step=np.array([1, 2], np.int64), meta='{"a": 1}',
               cfg=dict(h=2.5e-5, nested=dict(x=[1, 2])), threads=4, engine="x")
    p = str(tmp_path / "r.npz")
    rec = C.save_run(p, res, dict(status="complete", W=INF))
    back = C.load_run(p)
    assert np.array_equal(back["M_all"], res["M_all"]) and back["cfg"] == res["cfg"] and back["threads"] == 4
    j = C.read_json(C.json_path(p))
    assert j["W"] == "inf" and j["npz_sha256"] == rec["npz_sha256"]
    C.save_run(str(tmp_path / "t.npz"), dict(wall_s_at_save=np.ones(3)), dict(status="complete"), arrays=False)
    assert not os.path.exists(tmp_path / "t.npz") and os.path.exists(tmp_path / "t.json")


# ============================================================================================ allocation
def _stat_text(busy, base=None, total=1000):
    """A /proc/stat text with per-cpu cumulative counters: busy[c] in [0, 1] of `total` jiffies since `base`."""
    lines = ["cpu  0 0 0 0 0 0 0 0 0 0"]
    for c in range(256):
        b = int(round(busy.get(c, 0.0) * total)) + (base or {}).get(c, (0, 0))[0]
        t = total + (base or {}).get(c, (0, 0))[1]
        lines.append(f"cpu{c} {b} 0 0 {t - b} 0 0 0 0 0 0")
    return "\n".join(lines) + "\nintr 0\n"


class MockStat:
    """Successive /proc/stat reads: each sample pair yields the configured busy fractions."""

    def __init__(self, busy_seq):
        self.busy_seq = list(busy_seq)
        self.k = 0
        self.state = {c: (0, 0) for c in range(256)}

    def __call__(self):
        busy = self.busy_seq[min(self.k // 2, len(self.busy_seq) - 1)]
        if self.k % 2 == 0:
            txt = _stat_text({}, self.state, total=0)
        else:
            txt = _stat_text(busy, self.state)
            self.state = A.parse_proc_stat(txt)
        self.k += 1
        return txt


class IncStat:
    """Every read advances each cpu by `total` jiffies, busy[c] of them busy (the monitor reads once per sample)."""

    def __init__(self, busy, total=100):
        self.busy, self.total, self.state = busy, total, {c: (0, 0) for c in range(256)}

    def __call__(self):
        txt = _stat_text(self.busy, self.state, self.total)
        self.state = A.parse_proc_stat(txt)
        return txt


def test_topology_and_proc_stat():
    assert A.ccd_cores(15) == list(range(120, 128)) and A.sibling(120) == 248 and A.sibling(248) == 120
    assert A.ccd_cpus(0) == list(range(8)) + list(range(128, 136))
    assert A.node_of_ccd(7) == 0 and A.node_of_ccd(8) == 1 and A.ccd_of_cpu(250) == 15
    assert len(A.candidate_sets(1)) == 16 and len(A.candidate_sets(8)) == 16
    s16 = A.candidate_sets(16)
    assert len(s16) == 2 * 28 and all(A.node_of_ccd(a) == A.node_of_ccd(b) for a, b in s16)
    assert len(A.candidate_sets(32)) == 2 * 70
    assert A.used_cores((3, 5), 16) == list(range(24, 32)) + list(range(40, 48))
    assert A.cpu_list_str([120, 121, 122, 125, 127, 126]) == "120-122,125-127"
    assert A.parse_cpu_list("120-122,125") == [120, 121, 122, 125]
    p = A.parse_proc_stat("cpu0 10 0 5 80 5 0 0 0 0 0\ncpu1 0 0 0 100 0 0 0 0 0 0\n")
    assert p[0] == (15, 100) and p[1] == (0, 100)
    env = A.omp_env([120, 121], base={})
    assert env["OMP_NUM_THREADS"] == "2" and env["OMP_PLACES"] == "{120},{121}" and env["NUMBA_NUM_THREADS"] == "2"


def test_choose_set_idle_check_siblings_and_held():
    busy = {c: 0.0 for c in range(256)}
    for c in A.ccd_cpus(0):
        busy[c] = 0.5                                        # CCD 0 busy
    busy[A.sibling(9)] = 0.2                                 # CCD 1 disqualified by an SMT sibling only
    busy[17] = 0.05                                          # CCD 2 qualifies but is more loaded than CCD 3
    best, least, loads = A.choose_set(4, busy, held=())
    assert best == (3,) and loads[(1,)][0] == 0.2
    best, _, _ = A.choose_set(4, busy, held=(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15))
    assert best == (2,)
    best, least, _ = A.choose_set(16, busy, held=())
    assert best is not None and 0 not in best and 1 not in best and A.node_of_ccd(best[0]) == A.node_of_ccd(best[1])
    allbusy = {c: 0.3 for c in range(256)}
    allbusy.update({c: 0.15 for c in A.ccd_cpus(12)})
    best, least, _ = A.choose_set(8, allbusy, held=())
    assert best is None and least == (12,)                   # nothing qualifies: least loaded is the fallback
    best, least, _ = A.choose_set(8, {}, held=())            # unknown cpus count as busy
    assert best is None


def test_allocator_with_mocked_proc_stat(tmp_path):
    reg = A.Registry(str(tmp_path / "reg"))
    busy = {c: 0.5 for c in range(256)}
    busy.update({c: 0.0 for c in A.ccd_cpus(14) + A.ccd_cpus(15)})
    sleeps = []
    al = A.Allocator(reg, reader=MockStat([busy] * 50), sleep=lambda s: sleeps.append(s), log=lambda *a: None)
    a1 = al.try_allocate(4, "run1", dict(telemetry="x"))
    assert a1["ccds"] in ([14], [15]) and len(a1["cores"]) == 4 and not a1["flagged"]
    assert set(reg.held()) == set(a1["ccds"])
    a2 = al.try_allocate(8, "run2")
    assert a2 is not None and set(a2["ccds"]).isdisjoint(a1["ccds"])
    assert al.try_allocate(1, "run3") is None                # both idle CCDs held, the rest busy
    al.release(a2)
    assert set(reg.held()) == set(a1["ccds"])
    info = reg.held()[a1["ccds"][0]]
    assert info["run_id"] == "run1" and info["telemetry"] == "x" and info["cores"] == a1["cores"]
    reg.add_pid(a1["ccds"], a1["token"], os.getpid())
    assert reg.held()[a1["ccds"][0]]["run_pid"] == os.getpid()
    al.release(a1)
    assert reg.held() == {}


def test_allocator_waits_then_least_loaded_and_flags(tmp_path):
    reg = A.Registry(str(tmp_path / "reg"))
    busy = {c: 0.4 for c in range(256)}
    busy.update({c: 0.12 for c in A.ccd_cpus(5)})
    t = [0.0]

    def sleep(s):
        t[0] += s
    al = A.Allocator(reg, reader=MockStat([busy] * 500), sleep=sleep, clock=lambda: t[0], poll_s=30.0,
                     max_wait_s=1800.0, sample_s=2.0, log=lambda *a: None)
    a = al.allocate(8, "r")
    assert a["flagged"] and a["ccds"] == [5] and a["wait_s"] >= 1800.0
    # a CCD becoming idle during the wait is taken unflagged
    reg2 = A.Registry(str(tmp_path / "reg2"))
    idle = dict(busy)
    idle.update({c: 0.0 for c in A.ccd_cpus(9)})
    t[0] = 0.0
    al2 = A.Allocator(reg2, reader=MockStat([busy, busy, busy, idle]), sleep=sleep, clock=lambda: t[0],
                      log=lambda *a: None)
    b = al2.allocate(8, "r2")
    assert b["ccds"] == [9] and not b["flagged"] and 0 < b["wait_s"] < 1800


def test_registry_stale_cleanup(tmp_path):
    alive = {111: True, 222: False}
    reg = A.Registry(str(tmp_path / "reg"), alive=lambda pid, start=None: alive.get(pid, False))
    tok = reg.acquire([3], dict(run_id="a", pids=[dict(pid=111, start=None)]))
    assert tok and 3 in reg.held()
    assert reg.acquire([3, 4], dict(run_id="b", pids=[dict(pid=111, start=None)])) is None
    assert 4 not in reg.held()                              # all-or-nothing
    alive[111] = False                                      # holder died -> stale, removed, re-acquirable
    assert reg.held() == {}
    tok2 = reg.acquire([3], dict(run_id="c", pids=[dict(pid=222, start=None), dict(pid=111, start=None)]))
    assert tok2 is None or reg.held() == {}                 # both pids dead: stale immediately
    real = A.Registry(str(tmp_path / "reg3"))               # real pid check (start time recorded)
    tk = real.acquire([2], dict(run_id="me"))
    assert tk and 2 in real.held()
    assert A.pid_alive(os.getpid()) and not A.pid_alive(os.getpid(), start=-5)


# ============================================================================================ telemetry
def _tel(cpus, cores, t0, n, busy_fn, own_fn, dv_fn=lambda i: 0, tot=100, dt=1.0):
    rows = []
    for i in range(n):
        tp = t0 + i * dt
        rows.append(dict(t=tp + dt, tp=tp, b=[busy_fn(i, c) for c in cpus], n=[tot] * len(cpus),
                         o=[own_fn(i, c) for c in cpus], dv=dv_fn(i), h=len(cores), f=[3000] * len(cores)))
    return dict(cpus=list(cpus), cores=list(cores), t=np.array([r["t"] for r in rows]),
                tp=np.array([r["tp"] for r in rows]), busy=np.array([r["b"] for r in rows]),
                tot=np.array([r["n"] for r in rows]), own=np.array([r["o"] for r in rows]),
                dv=np.array([r["dv"] for r in rows]), nthr=np.array([r["h"] for r in rows]),
                freq_mhz=np.array([r["f"] for r in rows], float), init=np.zeros(n, np.int64))


def test_contention_rule():
    cpus = A.ccd_cpus(OUR_CCD)
    cores = [120, 121, 122, 123]
    own = lambda i, c: 100 if c in cores else 0             # noqa: E731
    t0 = 1000.0
    clean = _tel(cpus, cores, t0 - 3, 40, lambda i, c: own(i, c), own)
    r = A.contention(clean, t0, 30.0, 4)
    assert r["status"] == "uncontended" and not r["contended"] and r["max_foreign_busy"] == 0.0 and r["covered"]
    # a foreign process on the SMT sibling of a used core, late in the run only
    late = _tel(cpus, cores, t0 - 3, 40, lambda i, c: own(i, c) + (50 if (c == 249 and i >= 25) else 0), own)
    assert A.contention(late, t0, 10.0, 4)["status"] == "uncontended"          # early W window: clean
    rw = A.contention(late, t0, 34.0, 4)                                        # whole run: contended
    assert rw["contended"] and rw["worst_cpu"] == 249 and rw["max_foreign_busy"] > 0.10
    # foreign work on a USED core is busy - own (own ticks are subtracted)
    shared = _tel(cpus, cores, t0 - 3, 40, lambda i, c: 100 if c in cores else 0,
                  lambda i, c: (80 if c == 120 else 100) if c in cores else 0)
    rs = A.contention(shared, t0, 30.0, 4)
    assert rs["contended"] and rs["worst_cpu"] == 120 and rs["max_foreign_busy"] == pytest.approx(0.2)
    # just below the threshold on an idle core of the CCD -> uncontended
    lowb = _tel(cpus, cores, t0 - 3, 40, lambda i, c: own(i, c) + (9 if c == 127 else 0), own)
    assert A.contention(lowb, t0, 30.0, 4)["status"] == "uncontended"
    # involuntary context switches: 4 threads x 30 s -> > 5 per thread-second contends
    ctx = _tel(cpus, cores, t0 - 3, 40, lambda i, c: own(i, c), own, dv_fn=lambda i: 24)
    rc = A.contention(ctx, t0, 30.0, 4)
    assert rc["contended"] and rc["ctx_per_thread_s"] == pytest.approx(6.0)
    ok = _tel(cpus, cores, t0 - 3, 40, lambda i, c: own(i, c), own, dv_fn=lambda i: 16)
    assert A.contention(ok, t0, 30.0, 4)["status"] == "uncontended"
    # unknown: no telemetry / no overlap / window not covered
    assert A.contention(None, t0, 10.0, 4)["status"] == "unknown"
    assert A.contention(clean, t0 + 500, 10.0, 4)["status"] == "unknown"
    part = A.contention(clean, t0, 300.0, 4)
    assert part["status"] == "unknown" and part["contended"] and not part["covered"]
    # a window shorter than one sample interval uses the overlapping interval
    assert A.contention(clean, t0 + 0.2, 0.1, 4)["n_intervals"] == 1


def test_monitor_with_mocked_proc(tmp_path):
    reg = A.Registry(str(tmp_path / "reg"))
    tel = str(tmp_path / "run.telemetry.jsonl")
    tok = reg.acquire([OUR_CCD], dict(run_id="r1", telemetry=tel, cores=[120, 121], cpus=A.ccd_cpus(OUR_CCD),
                                     threads=2))
    reg.add_pid([OUR_CCD], tok, os.getpid())
    stat = IncStat({120: 1.0, 121: 1.0, 250: 0.3})
    ticks = {"v": 0}

    def tasks(pid):
        ticks["v"] += 100
        return {1: (ticks["v"], 120, 3 * ticks["v"] // 100), 2: (ticks["v"], 121, 0)}
    t = [5000.0]

    def clock():
        t[0] += 1.0
        return t[0]
    mon = A.Monitor(reg, reader=stat, task_stats=tasks, freq=lambda cs: [3_000_000] * len(cs), clock=clock, pin=False,
                    uptime=lambda: 100.0, proc_start=lambda pid: 0)
    for _ in range(6):
        mon.sample_once()
    T = A.load_telemetry(tel)
    assert T["cpus"] == A.ccd_cpus(OUR_CCD) and T["busy"].shape[1] == 16
    assert T["t"].size == 5                                             # first sample = baseline (header + seen)
    assert np.all(T["init"] == 0) and np.all(T["x"] == 0) and np.all(T["pid"] == os.getpid())
    assert np.array_equal(T["tp"][1:], T["t"][:-1])                    # rows abut
    seen = [json.loads(x) for x in open(tel) if '"seen"' in x]
    assert seen and seen[0]["seen"] == os.getpid() and seen[0].get("baseline") == 1
    k120 = T["cpus"].index(120)
    assert np.all(T["own"][:, k120] == 100)
    r = A.contention(T, T["tp"][2], 2.0, 2, pid=os.getpid(), scope="ccd")          # pre-amendment CCD-wide rule
    assert r["contended"] and r["worst_cpu"] == 250 and r["max_foreign_busy"] == pytest.approx(0.3, abs=0.02)
    ro = A.contention(T, T["tp"][2], 2.0, 2, pid=os.getpid())                       # Amendment 2: own cores + siblings
    assert 250 - A.SMT_OFFSET not in set(T["cores"]) and ro["scope"] == "own"
    assert ro["ccd_max_foreign_busy"] == pytest.approx(0.3, abs=0.02)              # co-tenancy kept as a covariate
    assert ro["worst_cpu"] != 250 and ro["max_foreign_busy"] < 0.3
    npz = str(tmp_path / "run.telemetry.npz")
    T2 = A.finalize_telemetry(tel, npz)
    assert not os.path.exists(tel) and np.array_equal(A.load_telemetry_npz(npz)["busy"], T2["busy"])


def test_monitor_first_sight_and_exit_rows(tmp_path):
    """Own-tick accounting at both ends of a run: pid not registered yet (x = 1), first sight of a process that
    started after the previous sample (i = 2: all its ticks are own), a process seen first after a monitor restart
    (i = 1: unknown), the process gone (x = 1)."""
    reg = A.Registry(str(tmp_path / "reg"))
    tel = str(tmp_path / "run.telemetry.jsonl")
    tok = reg.acquire([OUR_CCD], dict(run_id="r1", telemetry=tel, cores=[121], cpus=A.ccd_cpus(OUR_CCD), threads=1))
    stat = IncStat({121: 1.0})
    st = dict(alive=False, ticks=0, start_ticks=0)

    def tasks(pid):
        if not st["alive"]:
            return {}
        st["ticks"] += 100
        return {7: (st["ticks"], 121, 0)}
    t = [5000.0]

    def clock():
        t[0] += 1.0
        return t[0]
    up = {"v": 1000.0}

    def uptime():
        up["v"] += 1.0
        return up["v"]
    mon = A.Monitor(reg, reader=stat, task_stats=tasks, freq=lambda cs: [0] * len(cs), clock=clock, pin=False,
                    uptime=uptime, proc_start=lambda pid: st["start_ticks"])
    mon.sample_once()                                    # baseline, no pid
    mon.sample_once()                                    # row: no pid yet -> x = 1
    reg.add_pid([OUR_CCD], tok, os.getpid())
    st["alive"], st["ticks"] = True, 30                  # 30 ticks before the next sample (started after the last)
    st["start_ticks"] = int((up["v"] + 0.3) * A.CLK_TCK)
    mon.sample_once()                                    # first sight, exact: own = 130 (30 + this read's 100)
    mon.sample_once()                                    # normal
    st["alive"] = False
    mon.sample_once()                                    # process gone -> x = 1
    T = A.load_telemetry(tel)
    k = T["cpus"].index(121)
    assert list(T["x"]) == [1, 0, 0, 1] and list(T["init"]) == [0, 2, 0, 0]
    assert T["own"][1, k] == 130 and T["own"][2, k] == 100 and T["own"][3, k] == 0
    assert any(json.loads(x).get("seen") == os.getpid() for x in open(tel) if '"seen"' in x)
    # a monitor restarted mid-run sees a process that started long before its previous sample: i = 1 (unknown)
    st["alive"] = True
    mon2 = A.Monitor(reg, reader=stat, task_stats=tasks, freq=lambda cs: [0] * len(cs), clock=clock, pin=False,
                     uptime=uptime, proc_start=lambda pid: st["start_ticks"])
    reg2 = A.Registry(str(tmp_path / "reg2"))
    tel2 = str(tmp_path / "run2.telemetry.jsonl")
    tok2 = reg2.acquire([OUR_CCD], dict(run_id="r2", telemetry=tel2, cores=[121], cpus=A.ccd_cpus(OUR_CCD), threads=1))
    mon2.registry = reg2
    mon2.sample_once()                                   # baseline without pid
    reg2.add_pid([OUR_CCD], tok2, os.getpid())
    mon2.sample_once()
    T2 = A.load_telemetry(tel2)
    assert list(T2["init"]) == [1] and T2["own"][0, T2["cpus"].index(121)] == 0


def test_monitor_samples_only_its_own_drivers_runs(tmp_path):
    """Concurrent drivers each start a monitor (--parent = driver pid): a monitor writes rows only for runs whose
    allocation names its parent as driver (or no driver), so two monitors never append to the same telemetry file."""
    reg = A.Registry(str(tmp_path / "reg"))
    tel_a, tel_b = str(tmp_path / "a.telemetry.jsonl"), str(tmp_path / "b.telemetry.jsonl")
    reg.acquire([OUR_CCD], dict(run_id="ra", telemetry=tel_a, cores=[120], cpus=A.ccd_cpus(OUR_CCD), threads=1,
                                driver=dict(pid=111, start=0)))
    reg.acquire([OUR_CCD - 1], dict(run_id="rb", telemetry=tel_b, cores=[112], cpus=A.ccd_cpus(OUR_CCD - 1),
                                    threads=1, driver=dict(pid=222, start=0)))
    stat = IncStat({120: 1.0, 112: 1.0})
    t = [5000.0]

    def clock():
        t[0] += 1.0
        return t[0]
    kw = dict(reader=stat, task_stats=lambda pid: {}, freq=lambda cs: [0] * len(cs), clock=clock, pin=False,
              uptime=lambda: 100.0, proc_start=lambda pid: 0)
    mon_a, mon_b = A.Monitor(reg, parent=111, **kw), A.Monitor(reg, parent=222, **kw)
    for _ in range(3):
        mon_a.sample_once()
        mon_b.sample_once()
    hdr = lambda p: [json.loads(x)["monitor_pid"] for x in open(p) if '"hdr"' in x]
    assert len(hdr(tel_a)) == 1 and len(hdr(tel_b)) == 1             # one header each: one monitor per file
    assert set(mon_a.runs) == {"ra"} and set(mon_b.runs) == {"rb"}
    mon_any = A.Monitor(reg, **kw)                                     # no parent (tests, ad hoc): every run
    mon_any.sample_once()
    assert set(mon_any.runs) == {"ra", "rb"}


def _tel_rows(cpus, cores, rows):
    """telemetry dict from rows (t, tp, busy{cpu: j}, own{cpu: j}, init, x, pid)."""
    return dict(cpus=list(cpus), cores=list(cores), t=np.array([r[0] for r in rows], float),
                tp=np.array([r[1] for r in rows], float),
                busy=np.array([[r[2].get(c, 0) for c in cpus] for r in rows]),
                tot=np.array([[100] * len(cpus) for _ in rows]),
                own=np.array([[r[3].get(c, 0) for c in cpus] for r in rows]),
                dv=np.zeros(len(rows), np.int64), nthr=np.ones(len(rows), np.int64),
                init=np.array([r[4] for r in rows], np.int64), x=np.array([r[5] for r in rows], np.int64),
                pid=np.array([r[6] for r in rows], np.int64), freq_mhz=np.zeros((len(rows), 0)))


def test_contention_boundary_rows_gaps_and_pids():
    """The reviewer's case: a 3.3 s single-thread run on core 121; the row containing the end of the timed region is
    the exit row (process gone at the sample: own = 0).  Old accounting: 0.175 'foreign' -> contended.  Now: bounds
    (lower 0, upper 0.18) -> 'unknown'; with the child's end hold the exit row lies after the window -> uncontended."""
    cpus, cores, P = A.ccd_cpus(OUR_CCD), [121], 4242
    full = {121: 100}
    t0 = 1000.4
    base = [(1000.0 + k, 999.0 + k, full, full, 0, 0, P) for k in range(1, 4)]         # (1000,1001] .. (1002,1003]
    exit_row = (1004.0, 1003.0, {121: 70}, {}, 0, 1, P)                                 # exits at ~1003.7
    tel = _tel_rows(cpus, cores, base + [exit_row])
    r = A.contention(tel, t0, 3.3, 1, pid=P)
    assert r["status"] == "unknown" and r["max_foreign_busy"] == pytest.approx(70 / 400) and \
        r["max_foreign_busy_lower"] == 0.0 and r["n_incomplete"] == 1
    # held: the run is alive at the sample after its end (1004), the exit row is (1004, 1005]: outside the window
    held = [(1000.0 + k, 999.0 + k, full, full, 0, 0, P) for k in range(1, 5)] + [(1005.0, 1004.0, {}, {}, 0, 1, P)]
    r = A.contention(_tel_rows(cpus, cores, held), t0, 3.3, 1, pid=P)
    assert r["status"] == "uncontended" and r["max_foreign_busy"] == 0.0 and r["n_incomplete"] == 0
    # an incomplete row with foreign work on a SIBLING: the lower bound alone exceeds 0.10 -> contended
    sib = [(1000.0 + k, 999.0 + k, full, full, 0, 0, P) for k in range(1, 4)] + [
        (1004.0, 1003.0, {121: 70, 249: 90}, {}, 0, 1, P)]
    r = A.contention(_tel_rows(cpus, cores, sib), t0, 3.3, 1, pid=P)
    assert r["status"] == "contended" and r["worst_cpu"] == 249
    # first sight with unknown history overlapping the window -> unknown; exact first sight (i = 2) is complete
    fs1 = [(1001.0, 1000.0, full, {}, 1, 0, P)] + [(1001.0 + k, 1000.0 + k, full, full, 0, 0, P) for k in range(1, 4)]
    assert A.contention(_tel_rows(cpus, cores, fs1), t0, 3.3, 1, pid=P)["status"] == "unknown"
    fs2 = [(1001.0, 1000.0, full, full, 2, 0, P)] + [(1001.0 + k, 1000.0 + k, full, full, 0, 0, P) for k in range(1, 4)]
    assert A.contention(_tel_rows(cpus, cores, fs2), t0, 3.3, 1, pid=P)["status"] == "uncontended"
    # a gap between rows (e.g. a monitor restart) -> unknown; overlapping rows / rows of another pid -> unknown
    gap = [r_ for r_ in held if r_[0] != 1002.0]
    rg = A.contention(_tel_rows(cpus, cores, gap), t0, 3.3, 1, pid=P)
    assert rg["status"] == "unknown" and rg["gap"]
    dup = held + [(1002.5, 1001.5, full, full, 0, 0, P)]
    assert A.contention(_tel_rows(cpus, cores, dup), t0, 3.3, 1, pid=P)["status"] == "unknown"
    other = [r_[:6] + (P + 1,) if r_[0] == 1002.0 else r_ for r_ in held]
    ro = A.contention(_tel_rows(cpus, cores, other), t0, 3.3, 1, pid=P)
    assert ro["status"] == "unknown" and ro["mixed_pid"]
    assert A.contention(_tel_rows(cpus, cores, other), t0, 3.3, 1)["status"] == "uncontended"   # pid not checked


def test_contention_moves_own_ticks_among_run_cores():
    """Own ticks are attributed to a thread's last processor: a main thread that ran 0.24 s on core 127 before being
    bound to 126 (the warm-up) shows own 95 / busy 76 on 126 and own 71 / busy 95 on 127 (measured on the LTA engine).
    That is not foreign work; foreign work on a sibling, or busy on a run core beyond the run's total own, still is."""
    cpus, cores, P = A.ccd_cpus(OUR_CCD), [126, 127], 77
    rows = [(1001.0, 1000.0, {126: 76, 127: 95}, {126: 95, 127: 71}, 0, 0, P),
            (1002.0, 1001.0, {126: 100, 127: 100}, {126: 100, 127: 100}, 0, 0, P)]
    r = A.contention(_tel_rows(cpus, cores, rows), 1000.5, 1.4, 2, pid=P)
    assert r["status"] == "uncontended" and r["n_rows_own_moved"] == 1
    assert r["foreign_busy"][127] == pytest.approx(5 / 200)               # 171 busy - 166 own on the run cores
    rows2 = [(1001.0, 1000.0, {126: 76, 127: 95, 254: 30}, {126: 95, 127: 71}, 0, 0, P)] + rows[1:]
    r2 = A.contention(_tel_rows(cpus, cores, rows2), 1000.5, 1.4, 2, pid=P)
    assert r2["status"] == "contended" and r2["worst_cpu"] == 254
    rows3 = [(1001.0, 1000.0, {126: 100, 127: 100}, {126: 100, 127: 60}, 0, 0, P)] + rows[1:]
    r3 = A.contention(_tel_rows(cpus, cores, rows3), 1000.5, 1.4, 2, pid=P)
    assert r3["status"] == "contended" and r3["worst_cpu"] == 127 and r3["n_rows_own_moved"] == 0


def test_hold_for_monitor(tmp_path):
    tel = str(tmp_path / "x.telemetry.jsonl")
    clock = [0.0]

    def sleep(s):
        clock[0] += s
    r = A.hold_for_monitor(tel, after=None, pid=11, timeout=1.0, sleep=sleep, clock=lambda: clock[0])
    assert not r["seen"] and r["waited_s"] >= 1.0                     # no monitor: gives up after the timeout
    with open(tel, "w") as fh:
        fh.write(json.dumps(dict(hdr=1, cpus=[1], cores=[1])) + "\n")
        fh.write(json.dumps(dict(seen=11, t=50.0, baseline=1)) + "\n")
        fh.write(json.dumps(dict(t=51.0, tp=50.0, b=[1], n=[1], o=[1], dv=0, h=2, pid=11, i=0, x=0)) + "\n")
        fh.write(json.dumps(dict(t=52.0, tp=51.0, b=[1], n=[1], o=[1], dv=0, h=0, pid=11, i=0, x=1)) + "\n")
    assert A.hold_for_monitor(tel, after=None, pid=11, timeout=1.0, sleep=sleep, clock=lambda: clock[0])["seen"]
    assert not A.hold_for_monitor(tel, after=None, pid=12, timeout=0.2, sleep=sleep, clock=lambda: clock[0])["seen"]
    e = A.hold_for_monitor(tel, after=50.5, pid=11, timeout=1.0, sleep=sleep, clock=lambda: clock[0])
    assert e["seen"] and e["row_t"] == 51.0
    # the only row after 51.5 has no live threads (process gone): not a confirmation
    assert not A.hold_for_monitor(tel, after=51.5, pid=11, timeout=0.2, sleep=sleep, clock=lambda: clock[0])["seen"]


def test_real_monitor_and_child_holds(tmp_path):
    """A real child process (one busy thread on core 123) under the hold protocol, sampled by the real Monitor
    (in a thread of this test process, 4 Hz).  Structural checks: the whole-run window contains no boundary row
    (no first-sight row with unknown history, no exit row), the rows abut, and the child confirmed both holds."""
    import subprocess
    import threading
    reg = A.Registry(str(tmp_path / "reg"))
    tel = str(tmp_path / "run.telemetry.jsonl")
    out = str(tmp_path / "child.json")
    tok = reg.acquire([OUR_CCD], dict(run_id="probe", telemetry=tel, cores=[123], cpus=A.ccd_cpus(OUR_CCD),
                                      threads=1))
    mon = A.Monitor(reg, interval=0.25, pin=False)
    stop = threading.Event()

    def loop():
        while not stop.is_set():
            mon.sample_once()
            stop.wait(0.25)
    th = threading.Thread(target=loop, daemon=True)
    th.start()
    code = (f"import sys, time, json; sys.path.insert(0, {os.path.join(ROOT, 'scripts', 'final_campaign')!r}); "
            f"import expB_alloc as A; h0 = A.hold_for_monitor({tel!r}, after=time.time(), timeout=5); "
            f"t0 = time.time(); x = 0\nwhile time.time() - t0 < 1.5: x += 1\n"
            f"w = time.time() - t0; json.dump(dict(t0=t0, wall=w, h0=h0), open({out!r}, 'w')); "
            f"h1 = A.hold_for_monitor({tel!r}, after=time.time(), timeout=5); "
            f"json.dump(dict(t0=t0, wall=w, h0=h0, h1=h1), open({out!r}, 'w'))")
    try:
        p = subprocess.Popen(["taskset", "-c", "123", sys.executable, "-c", code], start_new_session=True)
        reg.add_pid([OUR_CCD], tok, p.pid)
        assert p.wait(timeout=30) == 0
        import time as _t
        _t.sleep(0.6)                                    # the exit row
    finally:
        stop.set()
        th.join(5)
        reg.release([OUR_CCD], tok)
    rec = json.load(open(out))
    assert rec["h0"]["seen"] and rec["h1"]["seen"]
    T = A.load_telemetry(tel)
    a, b = rec["t0"], rec["t0"] + rec["wall"]
    sel = (T["t"] > a) & (T["tp"] < b)
    assert sel.sum() >= 5 and np.all(T["init"][sel] != 1) and np.all(T["x"][sel] == 0)
    assert np.all(T["pid"][sel] == p.pid) and T["x"][-1] == 1          # the exit row exists, after the window
    r = A.contention(T, a, rec["wall"], 1, pid=p.pid)
    assert r["n_incomplete"] == 0 and not r["gap"] and r["covered"]
    k = T["cpus"].index(123)
    own = T["own"][sel, k].sum()
    assert own >= 0.8 * rec["wall"] * A.CLK_TCK                         # the run's own ticks are counted as own


# ============================================================================================ benchmark
def test_bench_plan_and_rotation():
    spec = C.system_spec("gateway")
    plan = BB.bench_plan(spec)
    gate = [p for p in plan if not p["descriptive"]]
    assert len(gate) == 74 and len(plan) - len(gate) == 8      # + t = 32 for N >= 64, both methods
    assert all(p["threads"] <= p["N"] for p in plan if p["impl"] == "parallel")
    assert sum(1 for p in plan if p["impl"] == "serial") == 14
    order = BB.rotated_order(plan, 3)
    assert len(order) == 3 * len(plan)
    firsts = [order[r * len(plan)][1]["config"] for r in range(3)]
    assert len(set(firsts)) == 3
    for r in range(3):
        assert sorted(it["config"] for rr, it in order if rr == r) == sorted(p["config"] for p in plan)


def _summary(thr):
    """{(method, N, impl, t): throughput} -> bench summary dict."""
    out = {}
    for (m, N, impl, t), v in thr.items():
        cfg = BB.config_id(dict(method=m, N=N, impl=impl, threads=t))
        out[cfg] = dict(method=m, N=N, impl=impl, threads=t, descriptive=False, n=3, throughput=v,
                        wall_per_walker_step=1.0 / v, us_per_walker_step=1e6 / v)
    return out


def test_gate_tie_rule_and_completion():
    spec = C.system_spec("gateway")
    thr = {}
    for m in ("abf", "fr"):
        thr[(m, 512, "serial", 1)] = 1.0e7
        for t, sp in zip((1, 2, 4, 8, 16), (1.2, 2.3, 4.4, 8.0, 7.9)):
            thr[(m, 512, "parallel", t)] = 1.0e7 * sp
    g = BB.evaluate_gate(_summary(thr), spec)
    assert g["verdict"] == "PASS" and g["F1"]["fr"]["ratio"] == pytest.approx(8.0)
    assert g["F1b"]["ratio"] == pytest.approx(8.0 / 1.2) and g["F1b"]["best_threads"] == 8
    thr2 = dict(thr)
    thr2[("abf", 512, "parallel", 8)] = 5.0e7
    thr2[("abf", 512, "parallel", 16)] = 5.9e7                  # 5.9 x serial < 6
    assert BB.evaluate_gate(_summary(thr2), spec)["verdict"] == "FAIL"
    thr3 = dict(thr)
    thr3.pop(("fr", 512, "serial", 1))
    assert BB.evaluate_gate(_summary(thr3), spec)["verdict"] == "INCOMPLETE"
    # tie rule: within 3 % -> fewer cores; serial wins a tie with parallel t = 1
    t4 = {("abf", 16, "serial", 1): 1.0e7, ("abf", 16, "parallel", 1): 1.02e7, ("abf", 16, "parallel", 2): 1.5e7,
          ("abf", 16, "parallel", 4): 2.0e7, ("abf", 16, "parallel", 8): 2.05e7, ("abf", 16, "parallel", 16): 1.9e7}
    ch, table = BB.implementation_for(_summary(t4), "abf", 16, [1, 2, 4, 8, 16])
    assert (ch["impl"], ch["threads"]) == ("parallel", 4)       # 2.0 >= 0.97 x 2.05
    t5 = {("abf", 2, "serial", 1): 1.0e7, ("abf", 2, "parallel", 1): 1.02e7, ("abf", 2, "parallel", 2): 0.9e7}
    ch, _ = BB.implementation_for(_summary(t5), "abf", 2, [1, 2, 4, 8, 16])
    assert ch["impl"] == "serial" and ch["threads"] == 1
    t6 = {("abf", 2, "serial", 1): 1.0e7, ("abf", 2, "parallel", 1): 1.05e7}
    assert BB.implementation_for(_summary(t6), "abf", 2, [1, 2])[0]["impl"] == "parallel"
    # competitor completion: W_proj = Q50(tau_u) x B x c*, smallest per method added unless core
    summ = {}
    for N in spec["bench_abf_N"]:
        summ.update(_summary({("abf", N, "serial", 1): 1.0e7}))
    for N in spec["bench_fr_N"]:
        summ.update(_summary({("fr", N, "parallel", 1): 1.0e7, ("fr", N, "parallel", 16): 1.0e7 * min(N, 64) / 4}))
    tau = {("abf", N): ({s: 0.01 * (1 + abs(math.log2(N) - 3)) for s in range(8)} if N != 1 else
                        {s: (INF if s < 5 else 0.001) for s in range(8)}) for N in spec["bench_abf_N"]}
    tau.update({("fr", N): {s: 0.002 * N for s in range(8)} for N in spec["bench_fr_N"]})
    cc = BB.competitor_completion(spec, summ, tau_source=lambda m, n: tau[(m, n)])
    assert cc["rows"]["abf1"]["W_proj_s"] == INF                 # censoring-aware median: inf
    assert cc["added"]["abf"] == "abf8"                          # smallest projected wall, not a core arm
    assert cc["rows"]["abf8"]["W_proj_s"] == pytest.approx(0.01 * spec["B"] / 1.0e7)
    assert cc["added"]["fr"] == "fr64"                           # equal c*, tau grows with N


def test_decide_end_to_end(tmp_path):
    spec = C.system_spec("gateway")
    out = str(tmp_path)
    d = os.path.join(out, "bench", "gateway", "runs")
    os.makedirs(d)
    rng = np.random.default_rng(0)
    es = {impl: {f"src/{impl}.py": impl * 4} for impl in ("serial", "parallel")}

    def write_runs(skip=None, engine=None):
        for it in BB.bench_plan(spec):
            if skip and it["config"] in skip:
                continue
            base = 1e7 * (1.0 if it["impl"] == "serial" else 1.1 * min(it["threads"], 12) ** 0.95)
            for r in range(3):
                e = (engine or es)[it["impl"]] if not (engine and r == 2 and it["impl"] == "parallel") else \
                    {"src/parallel.py": "changed"}
                C.write_json(os.path.join(d, f"{it['config']}_r{r}.json"),
                             dict(status="complete", config=it["config"], repeat=r, method=it["method"], N=it["N"],
                                  impl=it["impl"], threads=it["threads"], descriptive=it["descriptive"],
                                  throughput=base * (1 + 0.01 * rng.standard_normal()),
                                  timer_fractions={k: 0.1 for k in C.TIMER_KEYS}, peak_rss_mb=100.0,
                                  contention=dict(status="uncontended"), allocation=dict(flagged=False),
                                  signature=dict(engine_sources=e[it["impl"]] if isinstance(e, dict) and it["impl"]
                                                 in e else e)))
    write_runs()
    dec = BB.decide("gateway", out_root=out, spec=spec, tau_key=BB.TAU_KEY)
    assert dec["gate"]["verdict"] == "PASS" and dec["production_licensed"], dec["license_problems"]
    assert dec["engine_sources"] == es and dec["vb4_engine_match"] is None
    alt = dec["competitor_completion"]["alternative"]
    assert alt["tau_key"] == BB.TAU_KEY_ALT and set(alt["added"]) == {"abf", "fr"}
    assert [a["arm"] for a in dec["arms"]][:4] == ["abf16", "abf64", "abf512", "fr512"]
    for a in dec["arms"]:
        assert a["impl"] == "parallel" and a["threads"] in (8, 16)
        assert a["projected_run_wall_s"] == pytest.approx(spec["B"] * a["wall_per_walker_step"])
    assert os.path.exists(os.path.join(out, "bench", "gateway", "decision.json"))
    # the tau reading changes the completion arms (synthetic tau): not licensed unless --tau-key fixes it
    tau_a = lambda m, n: {s: (0.01 if (m, n) in (("abf", 8), ("fr", 64)) else 0.5) for s in range(8)}   # noqa: E731
    tau_b = lambda m, n: {s: (0.01 if (m, n) in (("abf", 32), ("fr", 64)) else 0.5) for s in range(8)}  # noqa: E731
    d1 = BB.decide("gateway", out_root=out, spec=spec, tau_source=tau_a, tau_source_alt=tau_b, write=False)
    assert not d1["production_licensed"] and any("tau reading" in p for p in d1["license_problems"])
    d2 = BB.decide("gateway", out_root=out, spec=spec, tau_source=tau_a, tau_source_alt=tau_b, write=False,
                   tau_key=BB.TAU_KEY)
    assert d2["production_licensed"] and d2["competitor_completion"]["alternative"]["selection_differs"]
    # a core arm without any benchmark data: not licensed (the arm would silently vanish from production)
    for f in os.listdir(d):
        if f.startswith("abf16_"):
            os.remove(os.path.join(d, f))
    d3 = BB.decide("gateway", out_root=out, spec=spec, tau_key=BB.TAU_KEY, write=False)
    assert d3["gate"]["verdict"] == "PASS" and not d3["production_licensed"]
    assert any("abf16" in p for p in d3["license_problems"])
    # the parallel engine's sources changed between benchmark runs: not licensed
    write_runs(engine=es)
    d4 = BB.decide("gateway", out_root=out, spec=spec, tau_key=BB.TAU_KEY, write=False)
    assert not d4["production_licensed"] and any("distinct engine source" in p for p in d4["license_problems"])
    # the benchmarked engine differs from the V-B4 counter runs' engine: not licensed
    write_runs()
    C.write_json(os.path.join(out, "equiv", "gateway", "counter", "fr512", "s8500.json"),
                 dict(status="complete", signature=dict(engine_sources={"src/parallel.py": "other"})))
    d5 = BB.decide("gateway", out_root=out, spec=spec, tau_key=BB.TAU_KEY, write=False)
    assert d5["vb4_engine_match"] is False and not d5["production_licensed"]


# ============================================================================================ production
class FakeAlloc:
    """Hands out CCD 15 (cores 120-127) through a real registry; no idle check."""

    def __init__(self, registry):
        self.registry = registry
        self.calls = 0

    def try_allocate(self, t, run_id, info=None, wait_s=0.0):
        self.calls += 1
        cores = list(range(120, 120 + t))
        tok = self.registry.acquire([OUR_CCD], dict(info or {}, run_id=run_id, threads=t, cores=cores,
                                                    cpus=A.ccd_cpus(OUR_CCD)))
        if tok is None:
            return None
        return dict(run_id=run_id, token=tok, ccds=[OUR_CCD], cores=cores, cpus=A.ccd_cpus(OUR_CCD), threads=t,
                    flagged=False, wait_s=wait_s, busy_at_launch={}, max_busy_at_launch=0.0)

    def fallback(self, *a, **k):
        return None

    def allocate(self, t, run_id, info=None):
        return self.try_allocate(t, run_id, info)

    def release(self, a):
        self.registry.release(a["ccds"], a["token"])


def _write_tel(path, cpus, cores, t0, wall, foreign_cpu=None):
    with open(path, "w") as fh:
        fh.write(json.dumps(dict(hdr=1, run_id="x", cpus=cpus, cores=cores)) + "\n")
        t = t0 - 2.0
        while t < t0 + wall + 2.0:
            b = [(100 if c in cores else 0) + (50 if c == foreign_cpu else 0) for c in cpus]
            o = [100 if c in cores else 0 for c in cpus]
            fh.write(json.dumps(dict(t=t + 1.0, tp=t, b=b, n=[100] * len(cpus), o=o, dv=0, h=len(cores),
                                     f=[3000] * len(cores))) + "\n")
            t += 1.0


class TelLauncher(R.InProcessLauncher):
    """Runs the child in-process and writes a synthetic telemetry file: contended (foreign work on an SMT sibling)
    for the replicates listed in ``contended`` {(arm, seed): n_contended_replicates}."""

    def __init__(self, out_root, engine_module, contended=None):
        super().__init__(out_root, engine_module)
        self.contended = contended or {}
        self.launched = []

    def launch(self, job, paths, cores):
        self.launched.append((job["arm"], job["seed"], job["replicate"]))
        h = super().launch(job, paths, cores)
        if h.rc == 0:
            rec = C.read_json(paths["json"])
            bad = job["replicate"] < self.contended.get((job["arm"], job["seed"]), 0)
            _write_tel(paths["tel_jsonl"], A.ccd_cpus(OUR_CCD), cores, rec["t0_epoch"], rec["wall_total_s"],
                       foreign_cpu=250 if bad else None)
        return h


def _stamp(decision, spec, engine_module):
    """decision.json carries the benchmarked engine's source hashes (here: the fake engine's)."""
    d = json.loads(json.dumps(decision))
    d.setdefault("engine_sources", {impl: C.engine_sources_sha(spec, impl, engine_module)
                                    for impl in {a["impl"] for a in d["arms"] if a.get("impl")}})
    return d


DECISION = dict(production_licensed=True, gate=dict(verdict="PASS"), arms=[
    dict(arm="abf16", method="abf", N=16, impl="parallel", threads=4, wall_per_walker_step=1e-8),
    dict(arm="abf512", method="abf", N=512, impl="parallel", threads=8, wall_per_walker_step=1e-8),
    dict(arm="fr512", method="fr", N=512, impl="parallel", threads=8, wall_per_walker_step=1e-8)])
OVER = dict(B=2048 * 100, seeds=[1, 2, 3, 4, 5, 6])


def _scheduler(tmp_path, fake_engine, launcher=None, decision=DECISION, cap=10.0, guard=None, contended=None,
               seeds=None, allocator=None):
    out = str(tmp_path / "out")
    reg = A.Registry(str(tmp_path / "reg"))
    spec = C.system_spec("gateway", overrides=OVER)
    launcher = launcher or TelLauncher(out, fake_engine["name"], contended)
    decision = _stamp(decision, spec, fake_engine["name"])
    return R.Scheduler("gateway", decision, out_root=out, spec=spec, allocator=allocator or FakeAlloc(reg),
                       launcher=launcher,
                       registry_root=str(tmp_path / "reg"), ledger=str(tmp_path / "ledger.csv"), cap_core_h=cap,
                       drift=False, monitor=False, engine_module=fake_engine["name"], spec_overrides=OVER,
                       poll_s=0.01, finalize_sleep_s=0, log=lambda *a: None, guard=guard, seeds=seeds,
                       sleep=lambda s: None, pin_self=False), out, launcher


def test_production_order_seed_major():
    arms = ["abf16", "abf64", "abf512", "fr512", "abf8"]
    order = R.production_order(arms, [8400, 8401])
    assert [s for _, s in order] == [8400] * 5 + [8401] * 5
    for s in (8400, 8401):
        perm = np.random.default_rng(20261011 + s).permutation(5)
        assert [a for a, ss in order if ss == s] == [arms[i] for i in perm]


def test_scheduler_repeats_resume_and_mismatch(tmp_path, fake_engine):
    S, out, L = _scheduler(tmp_path, fake_engine, contended={("fr512", 2): 2, ("abf16", 3): 9})
    S.run()
    d = os.path.join(out, "prod", "gateway")
    # fr512 s2: 2 contended replicates, the third (rep2) uncontended -> stops repeating
    recs = R.replicate_records(R.prod_job(S.spec, S.by_arm["fr512"], 2), out)
    assert [r["uncontended_all"] for r in recs] == [False, False, True]
    assert recs[1]["science_identical_to_first"]
    with np.load(os.path.join(d, "fr512", "s2.rep1.npz")) as z:          # a repeat keeps only its timing arrays
        assert set(z.files) == {"wall_s_at_save", "n_force_evals_at_save", "__json_keys__"}
    # abf16 s3: always contended -> first + 3 repeats, then flagged
    recs = R.replicate_records(R.prod_job(S.spec, S.by_arm["abf16"], 3), out)
    assert len(recs) == 1 + R.MAX_REPEATS and not any(r["uncontended_all"] for r in recs)
    assert recs[0]["contention_status"]["W_mid"] == "contended"
    n_launched = len(L.launched)
    assert n_launched == 18 + 2 + 3
    rows = open(str(tmp_path / "ledger.csv")).read().strip().splitlines()
    assert rows[0].startswith("utc_start,stage") and len(rows) == 1 + n_launched
    # resume: nothing left to do
    S2, _, L2 = _scheduler(tmp_path, fake_engine)
    pending, n_complete, repeats, mismatch = S2.plan()
    assert pending == [] and repeats == [] and n_complete == 18 and mismatch == []
    S2.run()
    assert L2.launched == []
    # a deleted repeat record is re-derived on resume
    os.remove(os.path.join(d, "fr512", "s2.rep2.json"))
    S3, _, L3 = _scheduler(tmp_path, fake_engine)
    S3.run()
    assert L3.launched == [("fr512", 2, 2)]
    # a complete record of a DIFFERENT job (threads changed) is never overwritten
    dec = json.loads(json.dumps(DECISION))
    dec["arms"][0]["threads"] = 2
    S4, _, L4 = _scheduler(tmp_path, fake_engine, decision=dec)
    with pytest.raises(C.ExpBError, match="signature mismatch"):
        S4.run()
    assert L4.launched == []


def test_scheduler_failed_run_is_retried(tmp_path, fake_engine, monkeypatch):
    monkeypatch.setenv("FAKE_FAIL_SEED", "2")
    S, out, L = _scheduler(tmp_path, fake_engine, seeds=[1, 2])
    S.run()
    assert not os.path.exists(os.path.join(out, "prod", "gateway", "abf16", "s2.json"))
    monkeypatch.delenv("FAKE_FAIL_SEED")
    S2, _, L2 = _scheduler(tmp_path, fake_engine, seeds=[1, 2])
    S2.run()
    assert sorted(L2.launched) == [("abf16", 2, 0), ("abf512", 2, 0), ("fr512", 2, 0)]


def test_projection_gate(tmp_path, fake_engine):
    dec = json.loads(json.dumps(DECISION))
    for a in dec["arms"]:
        a["wall_per_walker_step"] = 1e-3                      # 204.8 s per run x 4-8 threads
    S, out, L = _scheduler(tmp_path, fake_engine, decision=dec, cap=0.5)
    with pytest.raises(C.ExpBError, match="projection gate"):
        S.run()
    assert L.launched == []
    proj = S.projection(S.plan()[0])
    assert proj["total_core_h"] == pytest.approx(sum(
        a["threads"] * (OVER["B"] * 1e-3 + R.RUN_OVERHEAD_S) / 3600 for a in dec["arms"]) * 6)


def test_disk_guard_and_scheduler_pause(tmp_path, fake_engine, monkeypatch):
    U = collections.namedtuple("U", "total used free")
    seq = iter([5e9, 1.5e9, 2.5e9, 2.9e9, 3.1e9, 1.0e9, 4e9])
    logs = []
    g = C.DiskGuard(usage=lambda p: U(0, 0, next(seq)), log=logs.append, sleep=lambda s: None)
    assert g.ok() and not g.paused
    assert not g.ok() and g.paused                            # < 2 GB: pause
    assert not g.ok() and not g.ok()                          # 2.5, 2.9 GB: still paused (hysteresis)
    assert g.ok()                                             # >= 3 GB: resume
    assert g.wait() >= 0.0                                    # 1.0 -> paused, then 4.0 -> resumes
    assert any("PAUSE" in m for m in logs) and any("RESUME" in m for m in logs) and len(g.pause_events) == 2
    t = [0.0]
    g2 = C.DiskGuard(usage=lambda p: U(0, 0, 1e9), log=lambda *a: None, sleep=lambda s: t.__setitem__(0, t[0] + s),
                     clock=lambda: t[0], poll_s=600, max_pause_s=3600)
    with pytest.raises(C.DiskPauseTimeout):
        g2.wait()
    # the scheduler launches nothing while the (mocked) disk is full, then proceeds
    frees = iter([1e9] * 5 + [5e9] * 100000)
    monkeypatch.setattr(C.shutil, "disk_usage", lambda p: U(0, 0, next(frees)))
    S, out, L = _scheduler(tmp_path, fake_engine, seeds=[1], guard=C.DiskGuard(log=lambda *a: None, poll_s=0.0))
    S.run()
    ev = [json.loads(x)["event"] for x in open(S.events)]
    assert ev.count("disk_pause") >= 1 and ev.index("disk_pause") < ev.index("launch")
    assert len(L.launched) == 3


def test_scheduler_recovers_ran_records_and_finalize_errors(tmp_path, fake_engine, monkeypatch):
    """A record the child wrote ('ran') but no driver finalised (driver died) is finalised on the next start (ledger
    row from the child's process age; no telemetry -> contention unknown -> repeat), never relaunched; an exception
    while finalising one run neither kills the scheduler nor loses the record."""
    S, out, L = _scheduler(tmp_path, fake_engine, seeds=[1])
    j = S._job("abf512", 1, 0)
    R.run_job_child(j, out, fake_engine["name"])                     # the orphan's child finished; nobody finalised
    p = R.job_paths(j, out)
    assert C.read_json(p["json"])["status"] == "ran"
    pend, n_c, rep, mm = S.plan()
    assert n_c == 0 and all(x["arm"] != "abf512" for x in pend) and len(S.plan_full()["ran"]) == 1
    S.run()
    rec = C.read_json(p["json"])
    assert rec["status"] == "complete" and "endpoints" in rec and rec["contention_status"]["W_mid"] == "unknown"
    assert "recovered" in rec["finalize_note"]
    assert ("abf512", 1, 0) not in L.launched and ("abf512", 1, 1) in L.launched    # repeat for the timing only
    rows = open(str(tmp_path / "ledger.csv")).read()
    assert rows.count("abf512 s1 rep0") == 1 and "recovered" in rows
    # finalisation error on one run: logged, the others proceed, the record stays 'ran' and is recovered later
    S2, out2, L2 = _scheduler(tmp_path / "b", fake_engine, seeds=[2])
    real = R.finalize_job

    def flaky(job, *a, **k):
        if job["arm"] == "fr512":
            raise ValueError("corrupt npz")
        return real(job, *a, **k)
    monkeypatch.setattr(R, "finalize_job", flaky)
    S2.run()
    ev = [json.loads(x) for x in open(S2.events)]
    assert any(e["event"] == "finalize_error" for e in ev) and len(L2.launched) == 3
    pf = R.job_paths(S2._job("fr512", 2, 0), out2)
    assert C.read_json(pf["json"])["status"] == "ran"
    monkeypatch.setattr(R, "finalize_job", real)
    S3, _, L3 = _scheduler(tmp_path / "b", fake_engine, seeds=[2])
    S3.run()
    assert C.read_json(pf["json"])["status"] == "complete" and ("fr512", 2, 0) not in L3.launched


def test_scheduler_waits_for_live_orphan(tmp_path, fake_engine, monkeypatch):
    """A job whose key is held by a live CCD lock (a run of a driver that died, still executing) is never launched a
    second time: the scheduler waits for the lock to go, then finalises the orphan's record (or relaunches the job if
    the orphan left none)."""
    S, out, L = _scheduler(tmp_path, fake_engine, seeds=[1])
    ja, jf = S._job("abf16", 1, 0), S._job("fr512", 1, 0)
    live = {A.job_key(ja): dict(job_key=A.job_key(ja)), A.job_key(jf): dict(job_key=A.job_key(jf))}
    calls = {"n": 0}

    def fake_live():
        calls["n"] += 1
        if calls["n"] == 6:                               # the orphan of abf16 finishes (writes 'ran'); fr512 dies
            R.run_job_child(ja, out, fake_engine["name"])
            live.clear()
        return dict(live)
    monkeypatch.setattr(S.registry, "live_job_keys", fake_live)
    S.run()
    assert ("abf16", 1, 0) not in L.launched                         # finalised from the orphan's record
    assert C.read_json(R.job_paths(ja, out)["json"])["status"] == "complete"
    assert ("fr512", 1, 0) in L.launched                              # the dead orphan left nothing: run again
    ev = [json.loads(x)["event"] for x in open(S.events)]
    assert ev.count("orphan_ended") == 2


def test_scheduler_terminates_overdue_orphan(tmp_path, fake_engine):
    """An orphan still alive after the run timeout (counted from its lock's creation) is hung: terminated."""
    S, out, L = _scheduler(tmp_path, fake_engine, seeds=[1])
    S.run_timeout_s = 3600.0
    j = S._job("abf16", 1, 0)
    p = A.launch_pinned([sys.executable, "-c", "import time; time.sleep(120)"], [127])
    try:
        tok = S.registry.acquire([3], dict(run_id="orph", job_key=A.job_key(j), pids=[
            dict(pid=p.pid, start=A.proc_start_time(p.pid), role="run")]))
        info = S.registry.live_job_keys()[A.job_key(j)]
        info["created_utc"] = "2026-10-01T00:00:00Z"
        S.orphans = {A.job_key(j): j}
        S._terminate_overdue_orphan(A.job_key(j), dict(info, created_utc=time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
        assert p.poll() is None                                    # young: left alone
        S._terminate_overdue_orphan(A.job_key(j), info)
        assert p.wait(timeout=10) is not None
        S.registry.release([3], tok)
    finally:
        A.kill_process_group(p)


class SleepLauncher:
    """Real child processes that just sleep (on core 127): for driver-exit handling."""

    def __init__(self):
        self.procs = []

    def launch(self, job, paths, cores):
        os.makedirs(os.path.dirname(paths["json"]), exist_ok=True)
        p = A.launch_pinned([sys.executable, "-c", "import time; time.sleep(120)"], [127], log_path=paths["log"])
        self.procs.append(p)
        return p


def test_scheduler_terminates_children_at_exit(tmp_path, fake_engine):
    class Boom(FakeAlloc):
        def try_allocate(self, t, run_id, info=None, wait_s=0.0):
            if self.calls >= 1:
                raise KeyboardInterrupt
            return super().try_allocate(t, run_id, info, wait_s)
    reg = A.Registry(str(tmp_path / "reg"))
    L = SleepLauncher()
    S, out, _ = _scheduler(tmp_path, fake_engine, launcher=L, seeds=[1], allocator=Boom(reg))
    with pytest.raises(KeyboardInterrupt):
        S.run()
    assert len(L.procs) == 1 and L.procs[0].poll() is not None        # our child was terminated
    assert reg.held() == {}
    rows = open(str(tmp_path / "ledger.csv")).read()
    assert "terminated at driver exit" in rows
    assert any(json.loads(x)["event"] == "terminated_at_exit" for x in open(S.events))


def test_run_timeout_excludes_disk_pause(tmp_path, fake_engine):
    S, out, _ = _scheduler(tmp_path, fake_engine, seeds=[1])
    S.run_timeout_s = 5.0
    job = S._job("abf16", 1, 0)
    paths = R.job_paths(job, out)
    os.makedirs(os.path.dirname(paths["json"]), exist_ok=True)
    reg = A.Registry(str(tmp_path / "reg"))
    a = FakeAlloc(reg).try_allocate(1, "x")
    p = A.launch_pinned([sys.executable, "-c", "import time; time.sleep(120)"], [127])
    import time as _t
    C._write_marker(paths["pause"], dict(start=_t.time() - 100.0))  # paused for the last 100 s (ongoing)
    active = {"x": (job, paths, a, p, _t.time() - 60.0, None)}
    try:
        S._reap(active, collections.deque())
        assert "x" in active and p.poll() is None                     # 60 s elapsed, 100 s paused: not timed out
        os.remove(paths["pause"])
        S._reap(active, collections.deque())
        assert "x" not in active and p.poll() is not None             # 60 s of run time > 5 s: terminated
    finally:
        A.kill_process_group(p)


def test_child_disk_pause_excluded_from_cost(tmp_path, fake_engine):
    U = collections.namedtuple("U", "total used free")
    frees = iter([1e9, 1e9, 1e9, 5e9])
    t = [0.0]
    g = C.DiskGuard(usage=lambda p: U(0, 0, next(frees)), log=lambda *a: None,
                    sleep=lambda s: t.__setitem__(0, t[0] + s), clock=lambda: t[0], poll_s=600.0)
    S, out, _ = _scheduler(tmp_path, fake_engine, seeds=[1])
    job = S._job("abf16", 1, 0)
    paths = R.job_paths(job, out)
    rec = R.run_job_child(job, out, fake_engine["name"], guard=g)
    assert rec["disk_pause_s"] == pytest.approx(1800.0)
    m = json.load(open(paths["pause"]))
    assert m["end"] >= m["start"] and m["paused_s"] == pytest.approx(1800.0)
    fin = R.finalize_job(job, paths, None, 0, 1900.0, S.spec, S.scorer, out, str(tmp_path / "ledger.csv"), "t")
    assert fin["status"] == "complete" and fin["used_wall_s"] == pytest.approx(100.0)
    row = open(str(tmp_path / "ledger.csv")).read().strip().splitlines()[-1].split(",")
    assert float(row[4]) == pytest.approx(100.0) and "disk pause 1800 s excluded" in row[-1]
    assert not os.path.exists(paths["pause"])
    # finalising again (a restarted driver) never adds a second ledger row for the same attempt
    R.finalize_job(job, paths, None, 0, 1900.0, S.spec, S.scorer, out, str(tmp_path / "ledger.csv"), "t")
    assert len(open(str(tmp_path / "ledger.csv")).read().strip().splitlines()) == 2


def test_scheduler_disk_pause_timeout_stops(tmp_path, fake_engine, monkeypatch):
    U = collections.namedtuple("U", "total used free")
    t = [0.0]
    g = C.DiskGuard(usage=lambda p: U(0, 0, 1e9), log=lambda *a: None, poll_s=600.0, max_pause_s=3600.0)
    S, out, L = _scheduler(tmp_path, fake_engine, seeds=[1], guard=g)
    S.clock = lambda: t[0]
    S.sleep = lambda s: t.__setitem__(0, t[0] + s)
    r = S.run()
    assert r["stopped"] and L.launched == []
    ev = [json.loads(x) for x in open(S.events)]
    assert any(e["event"] == "stop" and "disk pause" in e["reason"] for e in ev)
    assert sum(e["event"] == "disk_pause" for e in ev) == 1          # logged at the transition only


def test_head_of_line_wait_resets_and_drift_index_reserved(tmp_path, fake_engine, monkeypatch):
    """A job inserted at the head (repeat / drift) starts its own 30-min wait; two drift jobs never share an index."""
    t = [0.0]

    class Never(FakeAlloc):
        def __init__(self, registry):
            super().__init__(registry)
            self.fallbacks = []

        def try_allocate(self, *a, **k):
            return None

        def fallback(self, th, run_id, info=None, wait_s=0.0):
            self.fallbacks.append((run_id, wait_s, t[0]))
            return FakeAlloc.try_allocate(self, th, run_id, info, wait_s)
    reg = A.Registry(str(tmp_path / "reg"))
    al = Never(reg)
    S, out, L = _scheduler(tmp_path, fake_engine, seeds=[1], allocator=al)
    S.clock = lambda: t[0]
    S.sleep = lambda s: t.__setitem__(0, t[0] + s)
    S.max_wait_s, S.poll_s = 100.0, 30.0
    real_reap = S._reap
    st = {"n": 0}

    def reap(active, queue):
        real_reap(active, queue)
        st["n"] += 1
        if t[0] >= 60.0 and not st.get("inserted") and queue:
            queue.appendleft(dict(queue[0], seed=1, arm=queue[0]["arm"], replicate=3))   # a new head mid-wait
            st["inserted"] = t[0]
    monkeypatch.setattr(S, "_reap", reap)
    S.run()
    first = al.fallbacks[0]
    assert "-r3-" in first[0] and first[1] >= 100.0                  # the new head waited its own 100 s ...
    assert first[2] >= st["inserted"] + 100.0                         # ... counted from when it became the head
    k1, k2 = S._drift_job()["replicate"], S._drift_job()["replicate"]
    assert k2 == k1 + 1


def test_allocator_reserved_ccds_and_pin_fallback(tmp_path, monkeypatch):
    reg = A.Registry(str(tmp_path / "reg"))
    busy = {c: 0.5 for c in range(256)}
    busy.update({c: 0.0 for c in A.ccd_cpus(14) + A.ccd_cpus(15)})
    al = A.Allocator(reg, reader=MockStat([busy] * 50), sleep=lambda s: None, log=lambda *a: None, reserved=[15])
    a1 = al.try_allocate(8, "r1")
    assert a1["ccds"] == [14] and al.try_allocate(8, "r2") is None   # CCD 15 (the monitor's home) never allocated
    assert al.fallback(8, "r3")["ccds"] != [15]
    assert A.ccds_of_cpus([124, 125, 252]) == [15]
    # every allowed core held: pin_outside falls back to a non-held core (logged) instead of staying put
    reg2 = A.Registry(str(tmp_path / "reg2"))
    reg2.acquire([15], dict(run_id="x"))
    msgs, pinned = [], []
    monkeypatch.setattr(A.os, "sched_setaffinity", lambda tid, cpus: pinned.append((tid, set(cpus))))  # never move
    c = A.pin_outside(reg2, allowed=[124, 125], current=124, log=msgs.append)
    assert c is not None and A.ccd_of_cpu(c) != 15 and msgs and pinned and pinned[0] == (0, {c})


# ============================================================================================ V-B4
def test_equivalence_rules():
    rng = np.random.default_rng(4)

    def arm(m, scale=1.0, n=32, s0=0):
        return {s0 + i: float(m * scale * (1 + 0.05 * rng.standard_normal())) for i in range(n)}
    data = dict(abf16=dict(counter=arm(0.05, s0=100), validated=arm(0.05)),
                abf512=dict(counter=arm(0.03, s0=100), validated=arm(0.03)),
                fr512=dict(counter=arm(0.02, s0=100), validated=arm(0.02)))
    data["fr512"]["counter"] = {s: data["abf512"]["counter"][s] * 0.66 for s in data["abf512"]["counter"]}
    data["fr512"]["validated"] = {s: data["abf512"]["validated"][s] * 0.66 for s in data["abf512"]["validated"]}
    r = EQ.evaluate(data, n_boot=2000)
    assert r["verdict"] == "PASS" and r["G_difference"]["status"] == "PASS"
    assert abs(r["G_difference"]["point"]) < 1e-12
    # noisy arm with point inside the window -> extension required (once), then FAIL if still outside
    noisy = dict(data)
    noisy["abf16"] = dict(counter={i: float(v) for i, v in enumerate(rng.lognormal(0, 0.8, 32) * 0.05)},
                          validated={i: float(v) for i, v in enumerate(rng.lognormal(0, 0.8, 32) * 0.05)})
    rn = EQ.evaluate(noisy, n_boot=2000)
    pt = rn["arms"]["abf16"]["point"]
    if 0.93 <= pt <= 1.075:
        assert rn["verdict"] == "EXTENSION REQUIRED" and rn["extension_arms"] == ["abf16"]
        assert EQ.evaluate(noisy, extension_done=True, n_boot=2000)["verdict"] == "FAIL"
    else:
        assert rn["verdict"] == "FAIL"
    # a shifted arm (point outside the window) fails outright
    shifted = dict(data)
    shifted["abf512"] = dict(counter={s: 1.3 * v for s, v in data["abf512"]["counter"].items()},
                             validated=data["abf512"]["validated"])
    rs = EQ.evaluate(shifted, n_boot=2000)
    assert rs["arms"]["abf512"]["status"] == "FAIL" and rs["verdict"] == "FAIL"
    assert EQ.classify([0.86, 1.17], 1.0, (0.85, 1.18), (0.93, 1.075)) == "PASS"
    assert EQ.classify([0.84, 1.10], 0.95, (0.85, 1.18), (0.93, 1.075)) == "EXTEND"
    assert EQ.classify([-0.12, 0.02], -0.06, (-0.1, 0.1), (-0.05, 0.05)) == "FAIL"
    few = dict(abf16=dict(counter=arm(0.05, n=4), validated=arm(0.05)))
    assert EQ.evaluate(few, n_boot=200)["verdict"] == "INCOMPLETE"


def test_equiv_jobs_seed_sets():
    gw, lt = C.system_spec("gateway"), C.system_spec("lta150")
    j = R.equiv_jobs(gw)
    assert {x["kind"] for x in j} == {"equiv_counter"} and len(j) == 3 * 32
    assert {x["arm"] for x in j} == {"abf16", "abf512", "fr512"} and all(x["grid"] == "eqb" for x in j)
    j = R.equiv_jobs(lt)
    assert sum(x["kind"] == "equiv_validated" for x in j) == 3 * 16 and sum(x["kind"] == "equiv_counter" for x in j) == 96
    assert {x["arm"] for x in j} == {"abf2", "abf512", "fr512"}
    j = R.equiv_jobs(gw, extension_arms=["abf16"])
    assert sum(1 for x in j if x["arm"] == "abf16") == 64 + 32 and sum(1 for x in j if x["arm"] == "fr512") == 32


def test_equiv_runner_and_vb4_evaluation(tmp_path, fake_engine):
    """V-B4 plumbing: counter jobs through run_equiv (in-process), records carry Ibar_F, expB_equivalence.run reads
    them against the equal-budget production per-seed values (budgets differ here: plumbing only)."""
    out = str(tmp_path / "out")
    over = dict(B=2048 * 50)
    spec = C.system_spec("gateway", overrides=over)
    r = R.run_equiv("gateway", [120], which="counter", out_root=out, registry_root=str(tmp_path / "reg"),
                    ledger=str(tmp_path / "ledger.csv"), engine_module=fake_engine["name"],
                    launcher=R.InProcessLauncher(out, fake_engine["name"]), log=lambda *a: None,
                    sleep=lambda s: None, lock=False, spec_overrides=over)
    assert r == dict(n_jobs=96, ran=96, stopped=None)
    rec = C.read_json(os.path.join(out, "equiv", "gateway", "counter", "fr512", "s8500.json"))
    assert rec["status"] == "complete" and rec["Ibar_F"] > 0 and rec["job"]["grid"] == "eqb"
    again = R.run_equiv("gateway", [120], which="counter", out_root=out, registry_root=str(tmp_path / "reg"),
                        ledger=str(tmp_path / "ledger.csv"), engine_module=fake_engine["name"],
                        launcher=R.InProcessLauncher(out, fake_engine["name"]), log=lambda *a: None,
                        sleep=lambda s: None, lock=False, spec_overrides=over)
    assert again["ran"] == 0                                   # resumable by signature
    v = EQ.run("gateway", out_root=out, write=True, spec=spec)
    assert set(v["arms"]) == {"abf16", "abf512", "fr512"}
    assert all(a["n_counter"] == 32 and a["n_validated"] == 32 for a in v["arms"].values())
    assert v["verdict"] in ("PASS", "FAIL", "EXTENSION REQUIRED") and v["G_difference"]["n_counter"] == 32
    assert os.path.exists(os.path.join(out, "equiv", "gateway", "vb4.json"))
    assert v["engine_sources"] == R.job_signature(R.equiv_jobs(spec, "counter")[0], spec,
                                                  fake_engine["name"])["engine_sources"]
    # counter records from two different engines validate no engine: INCOMPLETE
    p = os.path.join(out, "equiv", "gateway", "counter", "fr512", "s8501.json")
    r1 = C.read_json(p)
    r1["signature"]["engine_sources"] = {"src/parallel/x.py": "changed"}
    C.write_json(p, r1)
    v2 = EQ.run("gateway", out_root=out, write=False, spec=spec)
    assert v2["verdict"] == "INCOMPLETE" and "different engines" in v2["reason"]


def test_equiv_runner_recovers_ran_records(tmp_path, fake_engine):
    out = str(tmp_path / "out")
    over = dict(B=2048 * 50)
    spec = C.system_spec("gateway", overrides=over)
    j = [x for x in R.equiv_jobs(spec, "counter") if x["seed"] == 8500 and x["arm"] == "abf16"][0]
    j["spec_overrides"] = over
    R.run_job_child(j, out, fake_engine["name"])                     # finished, never finalised (runner died)
    launched = []

    class L(R.InProcessLauncher):
        def launch(self, job, paths, cores):
            launched.append((job["arm"], job["seed"]))
            return super().launch(job, paths, cores)
    r = R.run_equiv("gateway", [120], which="counter", out_root=out, registry_root=str(tmp_path / "reg"),
                    ledger=str(tmp_path / "ledger.csv"), engine_module=fake_engine["name"],
                    launcher=L(out, fake_engine["name"]), log=lambda *a: None, sleep=lambda s: None, lock=False,
                    spec_overrides=over)
    assert r["ran"] == 95 and ("abf16", 8500) not in launched
    rec = C.read_json(R.job_paths(j, out)["json"])
    assert rec["status"] == "complete" and rec["Ibar_F"] > 0 and "recovered" in rec["finalize_note"]


def test_bench_driver_attempts_orphans_and_ingestion(tmp_path, fake_engine, monkeypatch):
    """run_bench with real pinned children (fake engine): per-attempt files, a complete orphaned attempt ingested
    instead of re-run, a live orphan waited for (never run twice)."""
    out = str(tmp_path / "out")
    reg_root = str(tmp_path / "reg")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([fake_engine["dir"], os.path.join(ROOT, "src"),
                                                      os.environ.get("PYTHONPATH", "")]))
    spec = C.system_spec("gateway")
    only = ["fr512_parallel_t2", "abf64_parallel_t2"]
    kw = dict(out_root=out, registry_root=reg_root, engine_module=fake_engine["name"], repeats=1, only=only,
              ledger=str(tmp_path / "ledger.csv"), monitor=False, log=lambda *a: None, pin_self=False)
    # a complete attempt of fr512_parallel_t2 r0 left by a dead driver (its .partial)
    it = [x for x in BB.bench_plan(spec) if x["config"] == "fr512_parallel_t2"][0]
    ap = BB.attempt_paths(os.path.join(out, "bench", "gateway"), it, 0, "dead0001")
    os.makedirs(os.path.dirname(ap["partial"]), exist_ok=True)
    BB.run_one("gateway", "fr", 512, "parallel", 2, spec["bench_seeds"][0], spec["bench_walker_steps"], ap["partial"],
               engine_module=fake_engine["name"])
    # a live orphan of abf64_parallel_t2 r0: its lock (our own pid = alive) goes away after the first poll
    reg = A.Registry(reg_root)
    it64 = [x for x in BB.bench_plan(spec) if x["config"] == "abf64_parallel_t2"][0]
    key = BB.bench_job_key("gateway", it64, 0, spec["bench_seeds"][0])
    tok = reg.acquire([3], dict(run_id="orphan", job_key=key))
    polls = []

    def sleep(s):
        polls.append(s)
        reg.release([3], tok)
    r = BB.run_bench("gateway", allocator=FakeAlloc(A.Registry(reg_root)), sleep=sleep, **kw)
    assert r["runs_done"] == 2 and len(polls) == 1
    runs = os.path.join(out, "bench", "gateway", "runs")
    rec = C.read_json(os.path.join(runs, "fr512_parallel_t2_r0.json"))
    assert rec["recovered"] and rec["attempt"] == "dead0001" and rec["status"] == "complete"
    rec64 = C.read_json(os.path.join(runs, "abf64_parallel_t2_r0.json"))
    assert not rec64["recovered"] and rec64["pid"] > 0 and rec64["host"]["cpus"] == [120, 121]
    assert not [f for f in os.listdir(runs) if f.endswith(".partial")]
    att = [json.loads(x) for x in open(os.path.join(out, "bench", "gateway", "attempts.jsonl"))]
    assert [a["event"] for a in att].count("start") == 1 and any(a["event"] == "wait_orphan" for a in att)
    ledger = open(str(tmp_path / "ledger.csv")).read()
    assert "recovered orphaned attempt dead0001" in ledger
    # resume: nothing to do
    again = BB.run_bench("gateway", allocator=FakeAlloc(A.Registry(reg_root)), sleep=sleep, **kw)
    assert again["runs_done"] == 0


# ============================================================================================ analysis
def test_analysis_flagged_seeds_and_decision(tmp_path, fake_engine):
    import expB_analyze as AN
    S, out, L = _scheduler(tmp_path, fake_engine, contended={("abf16", 3): 9})
    S.run()
    bench = {"abf16_serial": dict(n=3, wall_per_walker_step=1e-7), "abf512_serial": dict(n=3, wall_per_walker_step=1e-7),
             "fr512_serial": dict(n=3, wall_per_walker_step=1.3e-7)}
    s = AN.analyze("gateway", out_root=out, spec=S.spec, decision=DECISION, bench_summary=bench, n_boot=500,
                   gehan_perm=100)
    res = s["results"]
    assert res["per_arm"]["abf16"]["flagged_seeds"]["W_mid"] == [3]
    assert res["W_mid"]["per_arm"]["abf16"]["n"] == 5                          # seed 3 excluded (primary)
    assert res["W_mid_sensitivity_incl_flagged"]["per_arm"]["abf16"]["n"] == 6  # included (sensitivity)
    assert s["decision_primary"] in ("FR advantage", "ABF advantage", "unresolved")
    assert res["W_mid"]["decision"]["contrasts"]["fr512|abf512"]["point"] < 1
    assert res["W_strict"]["decision"]["contrasts"]["fr512|abf16"]["budget_label"] in (
        "comparable", "not comparable (budget artefact)")
    assert res["W_strict"]["per_arm"]["abf16"]["n_censored"] == 5
    assert res["Ibar_F_wall"]["W_star"] > 0 and res["fr_added_wall_N512"]["common_thread_count"]["source"].startswith(
        "measured")
    assert os.path.exists(os.path.join(out, "tables", "gateway_seeds.csv"))
    assert os.path.exists(os.path.join(out, "analysis", "gateway", "boot.npz"))
    # science-only endpoints use every seed (seed 3's replicates are all contended, its science is valid)
    pa = res["per_arm"]["abf16"]
    assert pa["n_seeds_science"] == 6
    recs = {sd: R.replicate_records(R.prod_job(S.spec, S.by_arm["abf16"], sd), out)[0] for sd in range(1, 7)}
    want = np.mean([not r["endpoints"]["censored_mid"] for r in recs.values()])
    assert pa["frac_reached_mid"][0] == pytest.approx(want)
    assert pa["force_evals_mid"][0] == pytest.approx(np.median([C.num(r["endpoints"]["force_evals_mid"])
                                                                for r in recs.values()]))
    assert s["stage_status"] == "COMPLETE" and s["decision_primary"] == s["decision_on_available_seeds"]
    # a planned run without a record and an arm without an implementation -> INCOMPLETE, IUT unresolved
    os.remove(R.job_paths(R.prod_job(S.spec, S.by_arm["fr512"], 5), out)["json"])
    dec2 = json.loads(json.dumps(DECISION))
    dec2["arms"].append(dict(arm="abf8", method="abf", N=8, impl=None, threads=None, status="no benchmark data"))
    s2 = AN.analyze("gateway", out_root=out, spec=S.spec, decision=dec2, bench_summary=bench, n_boot=300,
                    gehan_perm=50, write=False)
    assert s2["stage_status"] == "INCOMPLETE" and s2["decision_primary"] == "INCOMPLETE"
    miss = {(m["arm"], m["seed"]) for m in s2["missing_runs"]}
    assert ("fr512", 5) in miss and {("abf8", sd) for sd in range(1, 7)} <= miss
    assert s2["decision_on_available_seeds"] == "unresolved" and "abf8" in s2["abf_arms"]


# ============================================================================================ real subprocess
def test_real_pinned_subprocess_child(tmp_path, fake_engine, monkeypatch):
    """One production job through the CLI child (taskset to 2 of our cores, omp_env), as the scheduler runs it."""
    out = str(tmp_path / "out")
    spec = C.system_spec("lta150", overrides=dict(B=512 * 60))
    arm = dict(arm="fr512", method="fr", N=512, impl="parallel", threads=2, wall_per_walker_step=1e-8)
    job = R.prod_job(spec, arm, 7)
    job["spec_overrides"] = dict(B=512 * 60)
    paths = R.job_paths(job, out)
    env_pp = os.pathsep.join([fake_engine["dir"], os.path.join(ROOT, "src"), os.environ.get("PYTHONPATH", "")])
    monkeypatch.setenv("PYTHONPATH", env_pp)
    h = R.ProcessLauncher(out, fake_engine["name"]).launch(job, paths, [120, 121])
    rc = h.wait()
    assert rc == 0, open(paths["log"]).read()[-2000:]
    rec = C.read_json(paths["json"])
    assert rec["status"] == "ran" and rec["signature"] == R.job_signature(job, spec, fake_engine["name"])
    assert rec["pid"] == h.pid and rec["disk_pause_s"] == 0.0 and rec["process_age_s"] > 0
    assert rec["host"]["cpus"] == [120, 121] and rec["threads"] == 2
    res = C.load_run(paths["npz"])
    assert res["M_prod"].shape[1] == 180 and np.all(np.diff(res["wall_s_at_save"]) >= 0)
