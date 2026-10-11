"""Tests of the LTA 150 K walker-parallel engine ``src/parallel/lta_parallel.py`` and the counter-mode RNG
``src/parallel/philox.py`` (final campaign, Experiment B; docs/final_campaign/PREREGISTRATION.md section 2.3).

Short versions of the preregistered validation (the formal runs at the preregistered lengths are the CLIs
``python -m parallel.lta_parallel --vb1 | --vb2 | --vb3``):

V-B1  replay mode is BITWISE the validated engine lta_ladder_numba.run_arm (diagnostics False) on every shared
      result key, threads {1, 4, 16}, N in {2 (ABF; FR also, an FR-inactive cell), 16, 64, 512}, both arms; the
      warm-up / burn-in / fr_start knobs are moved earlier IN BOTH ENGINES (cfg override) so short runs pass
      fr_start, with the production fr_rate and with fr_rate 40 (many events, cap-subset path); plus the
      production knobs (fr_start 20000) at N 16 just past 20000 steps; N 1, 3, 5 with threads > N; a save at
      step 0; timers and chunk size inert.
V-B2  counter mode is bitwise invariant over threads {1, 2, 4, 8, 16} and over the chunk size.
      Counter mode == replay mode fed with parallel.philox.lta_noise_block (the in-kernel counter is (global
      move step, slot, word)).
V-B5  counter-mode ABF and FR arms of one (seed, N) are bitwise identical before FR's first realised death and
      differ from it on.
V-B3  (i) numba Philox4x64-10 == numpy's Philox (exact) for 10^4 counters x 4 keys (+ a sequential stream and an
      independent pure-Python Philox); Box-Muller normals equal the formula evaluated with Python's math;
      (ii) counter-map injectivity for every production (N, n_steps, words) + an exhaustive block; (iii) the 9-test
      Holm battery on 10^7 normals (10^8 in the CLI), and its power against small defects.
Compiled-code freshness: every numba function of the engine is cached under a stamp that includes the sha256 of
      every dependency source; in a scratch copy, an edit to philox.py alone (or to lta_ladder_numba.py alone)
      recompiles the kernel in the next process; the once-per-process self-check raises on a mismatching
      dependency.  V-B1 fallback evidence: first-step forces are the kernel's own (evaluation and fused phases).

Run (16 cores, the allocation of this campaign):
  CUDA_VISIBLE_DEVICES="" NUMBA_CACHE_DIR=... PYTHONPATH=src taskset -c <16 cores> \\
      python -m pytest tests/test_parallel_lta.py -q
"""
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import numba  # noqa: E402
import lta_ladder_numba as E  # noqa: E402
from parallel import lta_parallel as P  # noqa: E402
from parallel import philox as PX  # noqa: E402

try:
    _NCPU = len(os.sched_getaffinity(0))
except AttributeError:  # pragma: no cover
    _NCPU = os.cpu_count()
numba.set_num_threads(min(_NCPU, 16, numba.config.NUMBA_NUM_THREADS))

SEED = P.BITWISE_SEEDS[0]
EARLY = dict(warmup_steps=200, burn_in_steps=200, fr_start_steps=300)


def cfg(stress=False, **kw):
    c = P.production_engine_cfg()
    c.update(EARLY)
    if stress:
        c["fr_rate"] = 40.0
    c.update(kw)
    return c


def validated(c, method, N, seed, n, grid):
    return E.run_arm(P.resolve(c, N, seed, n), method, grid)


def assert_bitwise(ref, res, keys=P.SHARED_KEYS):
    bad = P.compare_shared(ref, res, keys)
    assert not bad, f"keys differ: {bad}"


# =====================================================================================
# interface / contract
# =====================================================================================
def test_interface_keys_and_contract():
    N, n = 16, 3000
    c = cfg(stress=True)
    grid = P.expb_save_grid(n)
    ref = validated(c, "fr", N, SEED, n, grid)
    res = P.run_arm_parallel(c, "fr", N, SEED, n, grid, threads=4, noise="replay")
    # every validated diagnostics-off key that is not wall / provenance is returned, same dtype and shape
    prov = {"wall_s", "peak_rss_mb", "n_checkpoints_written", "resumed_from", "sessions_json", "meta_json"}
    need = set(ref) - prov
    assert need == set(P.SHARED_KEYS)
    for k in need:
        assert k in res, k
        if not isinstance(ref[k], str):
            assert np.asarray(ref[k]).dtype == np.asarray(res[k]).dtype, k
            assert np.asarray(ref[k]).shape == np.asarray(res[k]).shape, k
    assert not set(E.DIAG_ONLY_KEYS) & set(res)
    extra = ("wall_s_at_save", "n_force_evals_at_save", "wall_total_s", "warmup_s", "t_force_move_s", "t_abf_s",
             "t_sync_s", "t_fr_s", "t_save_s", "threads", "noise_mode", "engine", "meta")
    for k in extra:
        assert k in res, k
    S = grid.size
    w = np.asarray(res["wall_s_at_save"])
    assert w.dtype == np.float64 and w.shape == (S,)
    assert np.all(np.isfinite(w)) and np.all(np.diff(w) >= 0) and w[0] >= 0 and w[-1] <= res["wall_total_s"]
    nfe = np.asarray(res["n_force_evals_at_save"])
    assert nfe.dtype == np.int64 and np.array_equal(nfe, N * (grid + 1))
    assert res["n_force_evals"] == N * (n + 1)
    assert res["engine"] == "lta_parallel/1" and res["noise_mode"] == "replay" and int(res["threads"]) == 4
    meta = json.loads(res["meta"])
    assert meta["threading_layer"] == "omp"
    assert meta["numba_cache"]["mode"] == "dependency-keyed"
    assert meta["numba_cache"]["dependency_fingerprint"] == P._DEP_FINGERPRINT
    assert meta["selfcheck"]["passed"] and meta["selfcheck"]["pid"] == os.getpid()
    for f in ("src/parallel/lta_parallel.py", "src/parallel/philox.py", "src/lta_ladder_numba.py"):
        assert len(meta["sources_sha256"][f]) == 64
    assert meta["cpus"] and meta["git_commit"] and meta["compiler"]["numba"] == numba.__version__
    assert meta["counter_key_entropy"] == [SEED, N, 150, 2]
    for k in ("warmup_s", "t_force_move_s", "t_abf_s", "t_sync_s", "t_fr_s", "t_save_s", "wall_total_s"):
        assert np.isfinite(res[k]) and res[k] >= 0, k
    parts = sum(float(res[k]) for k in ("t_force_move_s", "t_sync_s", "t_abf_s", "t_fr_s", "t_save_s", "t_noise_s"))
    assert parts <= float(res["wall_total_s"]) * 1.0001 + 1e-4
    assert float(res["t_force_move_s"]) > 0.3 * float(res["wall_total_s"])


def test_errors():
    c = cfg()
    g = np.array([10])
    with pytest.raises(ValueError):
        P.run_arm_parallel(c, "fr", 1, SEED, 10, g, threads=1, noise="counter")
    with pytest.raises(ValueError):
        P.run_arm_parallel(c, "abf", 4, SEED, 10, g, threads=1, noise="mt")
    with pytest.raises(ValueError):
        P.run_arm_parallel(c, "xyz", 4, SEED, 10, g, threads=1, noise="counter")
    with pytest.raises(ValueError):
        P.run_arm_parallel(dict(c, N=8), "abf", 4, SEED, 10, g, threads=1, noise="counter")
    with pytest.raises(ValueError):
        P.run_arm_parallel(dict(c, accumulate_u=True), "abf", 4, SEED, 10, g, threads=1, noise="counter")
    with pytest.raises(ValueError):
        P.run_arm_parallel(c, "abf", 4, SEED, 10, np.array([5, 3]), threads=1, noise="counter")
    # a resolved cfg (with derived keys and diagnostics True) is accepted and equals the engine_cfg form
    rc = E.make_cfg(150.0, 4, SEED, n_steps=50, **dict(c, diagnostics=True))
    a = P.run_arm_parallel(rc, "abf", 4, SEED, 50, np.array([50]), threads=1, noise="counter")
    b = P.run_arm_parallel(c, "abf", 4, SEED, 50, np.array([50]), threads=1, noise="counter")
    assert_bitwise(a, b)


# =====================================================================================
# V-B1  replay identity with the validated engine
# =====================================================================================
VB1_SHORT = [(2, "abf", 40_000, True), (2, "fr", 40_000, True), (16, "abf", 20_000, True),
             (16, "fr", 20_000, True), (64, "abf", 10_000, True), (64, "fr", 10_000, True),
             (512, "abf", 3_000, True), (512, "fr", 3_000, True), (512, "fr", 14_000, False),
             (64, "fr", 120_000, False)]


@pytest.mark.parametrize("N,method,n,stress", VB1_SHORT)
def test_vb1_replay_bitwise(N, method, n, stress):
    c = cfg(stress=stress)
    grid = P.expb_save_grid(n)
    assert grid.size >= 20
    ref = validated(c, method, N, SEED + 1, n, grid)
    for t in (1, 4, 16):
        res = P.run_arm_parallel(c, method, N, SEED + 1, n, grid, threads=t, noise="replay")
        assert_bitwise(ref, res)
    if method == "fr" and N >= 16:
        assert int(ref["total_replacement_events"]) >= 50, int(ref["total_replacement_events"])
    if method == "fr":
        assert int(ref["n_fr_opps"]) == E._n_opps_in(1, n, EARLY["fr_start_steps"], 5)


def test_vb1_production_knobs_past_fr_start():
    """The frozen production knobs (warm-up / burn-in / fr_start 20000), N 16, a run just past fr_start."""
    N, n = 16, 21_000
    c = P.production_engine_cfg()
    grid = P.expb_save_grid(n)
    for method in ("abf", "fr"):
        ref = validated(c, method, N, SEED + 2, n, grid)
        for t in (1, 4, 16):
            assert_bitwise(ref, P.run_arm_parallel(c, method, N, SEED + 2, n, grid, threads=t, noise="replay"))
        if method == "fr":
            assert int(ref["n_fr_opps"]) == E._n_opps_in(1, n, 20_000, 5) == 201


@pytest.mark.parametrize("N,method", [(1, "abf"), (3, "fr"), (5, "fr"), (7, "abf")])
def test_vb1_small_and_odd_n_threads_above_n(N, method):
    n = 4000
    c = cfg(stress=True)
    grid = np.concatenate([[0], P.expb_save_grid(n)])       # a save at step 0 too
    ref = validated(c, method, N, SEED + 3, n, grid)
    for t in (1, 4, 16):
        assert_bitwise(ref, P.run_arm_parallel(c, method, N, SEED + 3, n, grid, threads=t, noise="replay"))


def test_vb1_timers_and_chunking_inert():
    N, n = 64, 4000
    c = cfg(stress=True)
    grid = P.expb_save_grid(n)
    ref = validated(c, "fr", N, SEED + 4, n, grid)
    a = P.run_arm_parallel(c, "fr", N, SEED + 4, n, grid, threads=4, noise="replay", timers=False, chunk_steps=7)
    assert_bitwise(ref, a)
    for k in ("t_force_move_s", "t_abf_s", "t_sync_s", "t_fr_s", "t_save_s"):
        assert math.isnan(float(a[k])), k
    assert np.all(np.isfinite(a["wall_s_at_save"]))
    b = P.run_arm_parallel(c, "fr", N, SEED + 4, n, grid, threads=16, noise="replay", chunk_steps=n + 1)
    assert_bitwise(ref, b)


def test_first_step_forces_bitwise():
    """V-B1 fallback evidence comes from the engine kernel itself: its step-0 forces (evaluation phase) and step-1
    forces (fused move + evaluation phase) equal the validated _mol_force at the same positions."""
    for N in (2, 16, 512):
        for t in (1, 16):
            r = P.first_step_forces(N, SEED, t)
            assert "_kernel" in r["source"], r
            for k in ("step0_eval_phase", "step1_fused_phase"):
                assert r[k]["bitwise"] and r[k]["max_rel_diff"] == 0.0, (k, r)
            assert r["bitwise"] and r["max_rel_diff"] == 0.0 and r["force_condition_met"], r
            assert "NOT the engine kernel" in r["helper_prange_diagnostic"]["note"]


def test_kernel_forces_are_the_kernel_state():
    """kernel_forces returns the kernel's own st['F']: F0 at q0 (whose first-step deposits are the validated
    engine's save-at-step-0 M_all) and the positions q1 of the validated engine after one step."""
    N = 16
    c = P.resolve(P.production_engine_cfg(), N, SEED, 1)
    q0, F0, q1, F1 = P.kernel_forces(c, 4)
    ref = E.run_arm(c, "abf", np.array([0, 1]))
    assert P.same_bits(q1, ref["q_final"])                       # q1 = the validated engine's position after step 0
    fl = np.clip(c_cf(c) * (F0[:, 0, 0] + F0[:, 1, 0]), -c["deposit_clip"], c["deposit_clip"])
    assert np.allclose(np.asarray(ref["M_all"])[0].sum(), fl.sum(), rtol=1e-12, atol=1e-9)
    assert not P.same_bits(F0, F1)


def c_cf(c):
    O, a, L = E.framework(c["framework_npz"])
    fp, _, _, _ = E._params(c, "abf", O, a, L, 0)
    return fp[E.F_CF]


# =====================================================================================
# V-B2  counter-mode thread / chunk invariance;  counter mapping
# =====================================================================================
@pytest.mark.parametrize("N,n", [(2, 20_000), (16, 10_000), (64, 5_000), (512, 2_000)])
def test_vb2_counter_thread_invariance(N, n):
    c = cfg(stress=True)
    grid = P.expb_save_grid(n)
    for method in ("abf", "fr"):
        base = P.run_arm_parallel(c, method, N, SEED + 5, n, grid, threads=1, noise="counter")
        for t in (2, 4, 8, 16):
            assert_bitwise(base, P.run_arm_parallel(c, method, N, SEED + 5, n, grid, threads=t, noise="counter"))
        assert_bitwise(base, P.run_arm_parallel(c, method, N, SEED + 5, n, grid, threads=3, noise="counter",
                                                chunk_steps=37))
        assert base["noise_mode"] == "counter"


def test_counter_mode_is_the_specified_philox_map():
    """Counter mode == replay mode fed with lta_noise_block (counter (global move step, slot, word, 0), key
    SeedSequence([seed, N, 150, 2])) for several chunkings: the in-kernel map is exactly the specified one."""
    N, n = 64, 3000
    c = cfg(stress=True)
    grid = P.expb_save_grid(n)
    k0, k1 = PX.lta_key(SEED + 6, N, 150.0)
    ref_key = np.random.SeedSequence([SEED + 6, N, 150, 2]).generate_state(2, np.uint64)
    assert (k0, k1) == (np.uint64(ref_key[0]), np.uint64(ref_key[1]))
    for method in ("abf", "fr"):
        cnt = P.run_arm_parallel(c, method, N, SEED + 6, n, grid, threads=16, noise="counter")
        for ch in (None, 101):
            rep = P.run_arm_parallel(c, method, N, SEED + 6, n, grid, threads=4, noise="replay", chunk_steps=ch,
                                     _noise_override=lambda s0, m: PX.lta_noise_block(k0, k1, s0, m, N))
            assert_bitwise(cnt, rep)
        val = P.run_arm_parallel(c, method, N, SEED + 6, n, grid, threads=4, noise="replay")
        assert not P.same_bits(cnt["q_final"], val["q_final"])      # a different stream than replay
    other = P.run_arm_parallel(c, "abf", N, SEED + 7, n, grid, threads=4, noise="counter")
    assert not P.same_bits(cnt["q_final"], other["q_final"])


# =====================================================================================
# V-B5  pairing
# =====================================================================================
@pytest.mark.parametrize("N,n", [(64, 6000), (512, 1500)])
def test_vb5_counter_pairing_before_first_death(N, n):
    c = cfg()                                   # production fr_rate, early fr_start (300)
    dense = np.arange(0, n + 1, dtype=np.int64)
    fr = P.run_arm_parallel(c, "fr", N, SEED + 8, n, dense, threads=16, noise="counter", timers=False)
    ab = P.run_arm_parallel(c, "abf", N, SEED + 8, n, dense, threads=16, noise="counter", timers=False)
    cd = np.asarray(fr["cum_deaths"])
    assert cd[-1] > 0, "no realised death: lengthen the run"
    d = int(dense[np.flatnonzero(cd > 0)[0]])
    assert d > EARLY["fr_start_steps"]
    pre = dense < d
    for k in ("M_all", "C_all", "M_prod", "C_prod"):
        assert P.same_bits(np.asarray(ab[k])[pre], np.asarray(fr[k])[pre]), k
    assert int(np.asarray(fr["cum_opps"])[pre][-1]) > 0          # FR opportunities happened before d, no event
    a1 = P.run_arm_parallel(c, "abf", N, SEED + 8, d - 1, np.array([d - 1]), threads=4, noise="counter")
    f1 = P.run_arm_parallel(c, "fr", N, SEED + 8, d - 1, np.array([d - 1]), threads=8, noise="counter")
    for k in ("q_final", "M_all_final", "C_all_final", "M_prod_final", "C_prod_final", "anc_final"):
        assert P.same_bits(a1[k], f1[k]), k
    a2 = P.run_arm_parallel(c, "abf", N, SEED + 8, d, np.array([d]), threads=4, noise="counter")
    f2 = P.run_arm_parallel(c, "fr", N, SEED + 8, d, np.array([d]), threads=8, noise="counter")
    assert not P.same_bits(a2["q_final"], f2["q_final"]) and not P.same_bits(a2["anc_final"], f2["anc_final"])


# =====================================================================================
# V-B3  RNG
# =====================================================================================
def test_vb3_i_philox_equals_numpy_oracle():
    r = P.philox_oracle_check(10_000)
    assert len(r["keys"]) == 4
    for k in r["keys"]:
        assert k["n_counters"] >= 10_000 and k["mismatches"] == 0, k
        assert k["sequential_mismatches"] == 0 and k["python_mismatches"] == 0, k
    assert r["passed"]


def test_vb3_i_numpy_counter_convention_is_determined():
    """numpy's Philox increments before generating: the block of counter c is Philox(counter = c - 1)'s first
    output, and Philox(counter = c)'s first output is the block of c + 1 (so the oracle is not off by one)."""
    k0, k1 = PX.lta_key(SEED, 512)
    c = np.array([123, 45, 1, 0], dtype=np.uint64)
    mine = PX.raw_block(k0, k1, c[None, :])[0]
    bg = np.random.Philox(key=np.array([int(k0), int(k1)], dtype=np.uint64), counter=c)
    nxt = PX.raw_block(k0, k1, np.array([[124, 45, 1, 0]], dtype=np.uint64))[0]
    assert np.array_equal(bg.random_raw(4), nxt)
    assert np.array_equal(PX.numpy_raw(k0, k1, c), mine)
    assert [int(x) for x in mine] == PX.python_philox(c, (int(k0), int(k1)))


def test_vb3_box_muller_normals_match_formula():
    k0, k1 = PX.lta_key(SEED, 64)
    z = PX.lta_noise_block(k0, k1, 1000, 50, 64)
    worst = 0.0
    for s in range(0, 50, 7):
        for i in range(0, 64, 5):
            o = list(PX.numpy_raw(k0, k1, np.array([1000 + s, i, 0, 0], dtype=np.uint64))) + \
                list(PX.numpy_raw(k0, k1, np.array([1000 + s, i, 1, 0], dtype=np.uint64)))
            u = [(int(x) >> 11) * 2.0 ** -53 for x in o]
            ref = []
            for a, b in ((0, 1), (2, 3), (4, 5)):
                r = math.sqrt(-2.0 * math.log(1.0 - u[a]))
                th = 2.0 * math.pi * u[b]
                ref += [r * math.cos(th), r * math.sin(th)]
            got = z[s, i].ravel()
            for g, e in zip(got, ref):
                worst = max(worst, abs(g - e) / max(abs(e), 1e-300))
    assert worst <= 4e-16, worst


def test_vb3_ii_counter_injectivity():
    r = P.counter_injectivity_check()
    assert r["passed"], r
    lta = [x for x in r["configs"] if x["system"] == "lta150"]
    assert sorted(x["N"] for x in lta) == [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    assert all(x["words"] == 2 and x["n_steps"] * x["N"] == P.LTA_B for x in lta)
    assert r["exhaustive"]["unique_output_blocks"] == r["exhaustive"]["n"] == 256 * 512 * 2
    # the checker detects a wrapping coordinate
    assert not PX.check_counter_injective(4, 2 ** 64 + 5, 2)["injective"]


def test_vb3_iii_battery_1e7():
    r = P.rng_battery(10 ** 7, seed=SEED)
    assert r["n_normals"] >= 10 ** 7
    assert set(r["pvalues"]) == set(P.VB3_TESTS) and len(r["holm"]) == 9
    assert r["passed"], r["pvalues"]
    assert abs(r["moments"]["variance"] - 1.0) < 2e-3


def test_vb3_iii_battery_has_power():
    """The battery rejects small defects at n ~ 1.2e7: variance 1.003, lag-1 step correlation 0.01, Box-Muller
    partner correlation 0.01, and accepts an iid numpy N(0, 1) sample of the same size."""
    rng = np.random.default_rng(3)
    S, N = 4000, 512
    base = rng.standard_normal((S, N, 6))
    assert P.battery_from_normals(base)["passed"]
    v = P.battery_from_normals(base * math.sqrt(1.003))
    assert v["holm"]["variance"]["rejected"]
    lag = base.copy()
    lag[1:] = math.sqrt(1 - 0.01 ** 2) * base[1:] + 0.01 * base[:-1]
    assert P.battery_from_normals(lag)["holm"]["lag1_steps"]["rejected"]
    bm = base.copy()
    bm[..., 1] = math.sqrt(1 - 0.01 ** 2) * base[..., 1] + 0.01 * base[..., 0]
    assert P.battery_from_normals(bm)["holm"]["box_muller_partners"]["rejected"]


def test_holm_procedure():
    h = P.holm(dict(a=0.001, b=0.004, c=0.02, d=0.5), 0.01)
    assert h["a"]["rejected"] and h["a"]["threshold"] == 0.01 / 4
    assert not h["b"]["rejected"]                 # 0.004 > 0.01 / 3 -> stop
    assert not h["c"]["rejected"] and not h["d"]["rejected"]


# =====================================================================================
# compiled-code freshness (numba cache keyed by every dependency; once-per-process self-check)
# =====================================================================================
def test_every_engine_function_uses_the_dependency_keyed_cache():
    from numba.core.registry import CPUDispatcher
    mine = [v for v in vars(P).values() if isinstance(v, CPUDispatcher) and v.py_func.__module__ == P.__name__]
    assert len(mine) >= 10
    jitted = {id(d) for d in P._JITTED}
    for d in mine:
        assert id(d) in jitted, d.py_func.__name__
        loc = d._cache._impl.locator
        assert isinstance(loc, P._DEP_STAMP), d.py_func.__name__
        stamp = loc.get_source_stamp()
        assert stamp[1:] == ("deps", P._DEP_FINGERPRINT)
    shas = {}
    for rel in ("src/parallel/lta_parallel.py", "src/parallel/philox.py", "src/parallel/__init__.py",
                "src/lta_ladder_numba.py"):
        with open(os.path.join(ROOT, rel), "rb") as f:
            shas[rel] = __import__("hashlib").sha256(f.read()).hexdigest()
    assert P._dep_fingerprint(shas) == P._DEP_FINGERPRINT
    assert P.cache_info()["mode"] == "dependency-keyed"


def test_selfcheck_detects_a_mismatching_dependency(monkeypatch):
    rec = P.kernel_selfcheck()
    assert rec["passed"] and rec["fr_events"] > 0 and rec["kernel_equals_validated"]
    assert rec["kernel_counter_equals_philox_module"]
    monkeypatch.setattr(P, "_SELFCHECK", rec)                    # restored after the test
    orig_nb = PX.lta_noise_block
    # philox.py "changed" (its current generator differs from the kernel's): caught
    monkeypatch.setattr(PX, "lta_noise_block", lambda k0, k1, s0, m, N: -orig_nb(k0, k1, s0, m, N))
    with pytest.raises(P.StaleKernelError, match="lta_noise_block"):
        P.kernel_selfcheck(force=True)
    assert P._SELFCHECK is rec                                   # a failed check never records a pass
    # ... and the first run_arm_parallel call of a process refuses to run
    monkeypatch.setattr(P, "_SELFCHECK", None)
    with pytest.raises(P.StaleKernelError):
        P.run_arm_parallel(cfg(), "abf", 4, SEED, 20, np.array([20]), threads=2, noise="counter")
    monkeypatch.setattr(PX, "lta_noise_block", orig_nb)
    # lta_ladder_numba "changed" (the validated engine's current result differs from the kernel's): caught
    orig_run = E.run_arm

    def shifted(c, m, g, **kw):
        r = orig_run(c, m, g, **kw)
        r["q_final"] = r["q_final"] + 1e-12
        return r
    monkeypatch.setattr(E, "run_arm", shifted)
    with pytest.raises(P.StaleKernelError, match="run_arm"):
        P.kernel_selfcheck(force=True)
    monkeypatch.setattr(E, "run_arm", orig_run)
    assert P.kernel_selfcheck(force=True)["passed"]


_CHILD = r"""
import json, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
from parallel import lta_parallel as P
from parallel import philox as PX
c = json.load(open(sys.argv[2]))["engine_cfg"]
c.update(framework_npz=sys.argv[3], warmup_steps=20, burn_in_steps=20, fr_start_steps=30, fr_rate=40.0)
N, seed, n = 8, 15408, 300
r = P.run_arm_parallel(c, "fr", N, seed, n, np.array([n]), threads=2, noise="counter")
k0, k1 = PX.lta_key(seed, N)
inj = P.run_arm_parallel(c, "fr", N, seed, n, np.array([n]), threads=2, noise="replay",
                         _noise_override=lambda s0, m: PX.lta_noise_block(k0, k1, s0, m, N))
m = json.loads(r["meta"])
print("RESULT " + json.dumps(dict(q=r["q_final"].tobytes().hex(), kernel_equals_philox=not P.compare_shared(r, inj),
                                  selfcheck=m["selfcheck"]["passed"], hits=m["numba_cache"]["kernel_cache_hits"],
                                  misses=m["numba_cache"]["kernel_cache_misses"],
                                  philox_sha=m["sources_sha256"]["src/parallel/philox.py"],
                                  fingerprint=m["numba_cache"]["dependency_fingerprint"])))
"""


def test_dependency_edit_recompiles_kernel_in_a_new_process():
    """The reviewer's scenario on a scratch copy with its own NUMBA_CACHE_DIR: populate the cache, then edit ONLY
    philox.py (Box-Muller cos <-> sin) -- the next process must run the NEW Philox (kernel == lta_noise_block),
    not the cached kernel; a comment-only edit of lta_ladder_numba.py alone also recompiles; an unchanged tree
    reuses the cache."""
    base = os.environ.get("TMPDIR") or tempfile.gettempdir()
    work = tempfile.mkdtemp(prefix="lta_cache_dep_", dir=base)
    try:
        src = os.path.join(work, "src")
        os.makedirs(os.path.join(src, "parallel"))
        for rel in ("parallel/__init__.py", "parallel/lta_parallel.py", "parallel/philox.py", "lta_ladder_numba.py"):
            shutil.copy2(os.path.join(ROOT, "src", rel), os.path.join(src, rel))
        child = os.path.join(work, "child.py")
        with open(child, "w") as f:
            f.write(_CHILD)
        env = dict(os.environ, NUMBA_CACHE_DIR=os.path.join(work, "numba_cache"), NUMBA_NUM_THREADS="4",
                   CUDA_VISIBLE_DEVICES="", PYTHONPATH=src)
        env.pop("NUMBA_CACHE_LOCATOR_CLASSES", None)
        args = [sys.executable, child, src, P.PROD_CFG_PATH, os.path.join(ROOT, "cache", "lta", "framework.npz")]

        def run():
            p = subprocess.run(args, env=env, capture_output=True, text=True, timeout=900)
            assert p.returncode == 0, p.stderr[-3000:]
            line = [x for x in p.stdout.splitlines() if x.startswith("RESULT ")][-1]
            return json.loads(line[7:])

        a = run()                                            # cold cache
        assert a["kernel_equals_philox"] and a["selfcheck"] and a["misses"] >= 1
        b = run()                                            # unchanged tree: served from the cache
        assert b["hits"] >= 1 and b["misses"] == 0 and b["q"] == a["q"] and b["kernel_equals_philox"]
        px = os.path.join(src, "parallel", "philox.py")
        with open(px) as f:
            txt = f.read()
        old = "return r * math.cos(th), r * math.sin(th)"
        assert txt.count(old) == 1
        with open(px, "w") as f:
            f.write(txt.replace(old, "return r * math.sin(th), r * math.cos(th)"))
        c = run()                                            # philox.py edited, lta_parallel.py untouched
        assert c["philox_sha"] != a["philox_sha"] and c["fingerprint"] != a["fingerprint"]
        assert c["misses"] >= 1 and c["hits"] == 0           # the kernel was recompiled, not loaded
        assert c["kernel_equals_philox"] and c["selfcheck"]  # and runs the NEW generator
        assert c["q"] != a["q"]
        with open(px, "w") as f:
            f.write(txt)                                     # back to the original philox.py
        ln = os.path.join(src, "lta_ladder_numba.py")
        with open(ln, "a") as f:
            f.write("\n# comment-only edit of the validated engine copy\n")
        d = run()                                            # lta_ladder_numba.py alone edited (no semantics)
        assert d["misses"] >= 1 and d["hits"] == 0 and d["fingerprint"] not in (a["fingerprint"], c["fingerprint"])
        assert d["q"] == a["q"] and d["kernel_equals_philox"] and d["selfcheck"]
    finally:
        shutil.rmtree(work, ignore_errors=True)
