"""Tests of the gateway walker-parallel engine (Experiment B): src/parallel/gateway_parallel.py + gw_kernel.c.

docs/final_campaign/PREREGISTRATION.md section 2.3 (short versions; the formal V-B1 / V-B2 / V-B5 runs use the CLI
``python -m parallel.gateway_parallel --vb1|--vb2|--vb5 ...`` at the preregistered lengths):

K0  build record (flags, compiler, sha256), OpenMP binding (proc_bind close, thread t on the t-th allocated cpu);
V-B1 replay mode is BITWISE src/gateway_ladder_numba.run_arm(..., diagnostics=False) on every shared result key
    (production cell N 16 / 64 / 512 and an aggressive cell with frequent FR deaths and wall reflections), both
    arms, threads {1, 4, 16}, every kernel variant, saves incl. the initial state;
V-B2 counter mode bitwise invariant over threads {1, 2, 4, 8, 16} (and over kernel variants);
V-B5 counter-mode ABF / FR arms bitwise identical at every save before FR's first gather; the preregistration's
    'before the first realised death' wording is the GATING verdict (an opportunity with kd = 0 < kc can permute
    the slots without a death, which changes the per-slot noise and the slot-order deposit: such a run is
    'FAIL_PREREG_DEATH_CRITERION_PASS_EVENT_ONLY', CLI exit 3, never PASS); the event criterion gates only with an
    amendment reference;
V-B3(i) C Philox4x64-10 raw outputs == numpy's np.random.Philox (its counter-increment convention determined here)
    for >= 1e4 counters x 4 keys, plus an independent pure-Python Philox;
BM  counter-mode normals == the Box-Muller formula evaluated with Python's math (glibc) on the C raw outputs,
    bitwise; the kernel's internal counter noise == the exported normals (counter run == replay of the exported
    normals, bitwise); thread-invariance of the exported normals; moment sanity on 1e6 normals; cross-check with
    src/parallel/philox.py (the LTA backend's numba Philox) when it is importable;
C   result contract (key set, dtypes, conventions), meta cpus = the process allocation (not libgomp's bound
    master cpu), timers inert (timers on/off, sampled/every-step and warm-up on/off bitwise), sampled timers: the
    sampled-step sequence is the documented function of the global step (thread- and segmentation-invariant),
    t_force_move + t_abf + t_sync = the exact region wall; timer and wall-stamp sanity, input refusals, CLI smoke
    (incl. the V-B5 exit status and the --overhead development check).

Run (CPU only, pinned, temp files in the scratchpad):
  CUDA_VISIBLE_DEVICES="" NUMBA_CACHE_DIR=~/.cache/numba_final_par PYTHONPATH=src taskset -c <16 cpus> \\
    python -m pytest tests/test_parallel_gateway.py -q -p no:cacheprovider --basetemp=<scratch dir>
"""
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from parallel import gateway_parallel as gp   # noqa: E402  (sets the OpenMP environment before libgomp loads)
import gateway_ladder_numba as gl             # noqa: E402

PROD = gp.production_cfg()
EASY = dict(PROD, beta=1.0, H=0.1, omega_out=1.0, omega_in=4.0, s=0.1, gamma=40.0, ramp_t=0.05)
SEED = 8601                                    # Exp B bitwise-test block 8600-8609 (never scored)
VARIANTS = sorted(gp.VARIANTS)
NEW_KEYS = {"wall_s_at_save", "n_force_evals_at_save", "wall_total_s", "warmup_s", "t_force_move_s", "t_abf_s",
            "t_sync_s", "t_fr_s", "t_save_s", "threads", "noise_mode", "engine", "meta"}


def grid(n):
    return np.unique(np.r_[0, gl.budget_save_grid(n)[0]]).astype(np.int64)


M64_ = (1 << 64) - 1


def mix64(s):
    """gw_kernel.c gw_mix64: splitmix64's output function of (s + 1) * golden."""
    z = ((s + 1) * 0x9E3779B97F4A7C15) & M64_
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64_
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64_
    return z ^ (z >> 31)


def sampled_steps(n, k):
    """The kernel's timer-sampled loop steps < n for sample_log2 k >= 1: s_0 = 0, s_(i+1) = s_i + 1 +
    (mix64(s_i) >> (63 - k))."""
    out, s = [], 0
    while s < n:
        out.append(s)
        s += 1 + (mix64(s) >> (63 - k))
    return out


def assert_bitwise(ref, par, keys, label=""):
    ok, recs = gp.compare(ref, par, keys)
    assert ok, (label, [r for r in recs if not r["equal"]])


# ------------------------------------------------------------------------------------------------------- K0
def test_k0_build_and_binding():
    gp.load_kernel()
    b = gp._BUILD_INFO
    for f in ("-O3", "-march=native", "-fopenmp", "-fno-fast-math", "-ffp-contract=off", "-fPIC", "-shared"):
        assert f in b["cflags"]
    assert os.path.exists(b["so_path"]) and b["so_path"].startswith(gp.BUILD_DIR)
    assert b["source_sha256"] == gp._sha256_file(gp.KERNEL_SRC) and b["compiler_version"].startswith("gcc")
    assert not b["so_path"].startswith(os.path.abspath(ROOT))           # never inside the repository
    cpus = sorted(os.sched_getaffinity(0))
    T = min(4, len(cpus))
    m = gp.omp_binding(T)
    assert m["team_size"] == T
    if os.environ.get("GW_PARALLEL_KEEP_OMP_ENV") != "1" and not gp._LIBGOMP_PRELOADED:
        assert m["proc_bind_close"] and m["num_places"] == len(gp._AFFINITY_AT_IMPORT)
        assert [c[0] for c in m["team_cpus"]] == gp._AFFINITY_AT_IMPORT[:T]


# ------------------------------------------------------------------------------------------------------- V-B1
VB1_CASES = [  # (label, cfg, N, method, n_steps)
    ("prod", PROD, 16, "abf", 40000), ("prod", PROD, 16, "fr", 40000),
    ("prod", PROD, 64, "abf", 20000), ("prod", PROD, 64, "fr", 20000),
    ("prod", PROD, 512, "abf", 8000), ("prod", PROD, 512, "fr", 12000),
    ("easy", EASY, 16, "fr", 20000), ("easy", EASY, 64, "fr", 16000), ("easy", EASY, 512, "fr", 6000),
]


@pytest.mark.parametrize("label,cfg,N,method,n_steps", VB1_CASES, ids=[f"{c[0]}-N{c[2]}-{c[3]}" for c in VB1_CASES])
def test_vb1_replay_bitwise(label, cfg, N, method, n_steps):
    sv = grid(n_steps)
    ref = gl.run_arm(cfg, method, N, SEED, n_steps, sv, diagnostics=False)
    keys = gp.shared_keys(ref)
    assert {"M_all", "C_all", "X_final", "Y_final", "fr_deaths_cum", "max_abs_bias", "cfg"} <= set(keys)
    assert len(sv) >= 3
    if label == "easy":
        assert ref["fr_deaths_cum"][-1] >= 50 and ref["fr_opp_death_cum"][-1] >= 10   # FR really exercised
        assert np.abs(ref["X_final"]).max() > 1.5                                    # walkers near the walls
    runs = [(gp.DEFAULT_VARIANT, T) for T in (1, 4, 16)] + [(v, 16) for v in VARIANTS if v != gp.DEFAULT_VARIANT]
    for v, T in runs:
        par = gp.run_arm_parallel(cfg, method, N, SEED, n_steps, sv, threads=T, noise="replay", variant=v)
        assert_bitwise(ref, par, keys, (label, N, method, v, T))


def test_vb1_reflection_and_long_segment_paths():
    """ABF with saves far apart (replay segments capped by the noise buffer), wall reflections, odd thread
    counts, N not a multiple of 8, and a save at step 0."""
    cfg = dict(EASY, gamma=0.0)
    N, n = 37, 70000
    sv = np.array([0, 1, 2, 3, 50000, n], dtype=np.int64)
    ref = gl.run_arm(cfg, "abf", N, SEED, n, sv, diagnostics=False)
    for T in (3, 7, 16):
        par = gp.run_arm_parallel(cfg, "abf", N, SEED, n, sv, threads=T, noise="replay", variant="replicated_dissem")
        assert_bitwise(ref, par, gp.shared_keys(ref), T)
        assert json.loads(par["meta"])["timers_extra"]["replay_max_segment_steps"] < 50000
    refr = gl.run_arm(EASY, "fr", N, SEED, 9000, sv[sv <= 9000].tolist() + [9000], diagnostics=False)
    par = gp.run_arm_parallel(EASY, "fr", N, SEED, 9000, sv[sv <= 9000].tolist() + [9000], threads=5,
                              noise="replay")
    assert_bitwise(refr, par, gp.shared_keys(refr))


# ------------------------------------------------------------------------------------------------------- V-B2
@pytest.mark.parametrize("cfg_label,N,method,n_steps", [("prod", 16, "fr", 40000), ("prod", 64, "abf", 20000),
                                                        ("prod", 512, "fr", 20000), ("prod", 512, "abf", 10000),
                                                        ("easy", 512, "fr", 8000), ("easy", 64, "fr", 16000)])
def test_vb2_counter_thread_invariance(cfg_label, N, method, n_steps):
    cfg = PROD if cfg_label == "prod" else EASY
    sv = grid(n_steps)
    ref = gp.run_arm_parallel(cfg, method, N, SEED, n_steps, sv, threads=1, noise="counter")
    keys = gp.science_keys(ref)
    assert {"M_all", "C_all", "X_final", "Y_final", "fr_deaths_cum", "n_force_evals_at_save"} <= set(keys)
    if cfg_label == "easy":
        assert ref["fr_deaths_cum"][-1] >= 50
    for T in (2, 4, 8, 16):
        par = gp.run_arm_parallel(cfg, method, N, SEED, n_steps, sv, threads=T, noise="counter")
        assert_bitwise(ref, par, keys, T)
    for v in VARIANTS:
        par = gp.run_arm_parallel(cfg, method, N, SEED, n_steps, sv, threads=16, noise="counter", variant=v)
        assert_bitwise(ref, par, keys, v)


def test_vb2_counter_differs_from_replay_and_across_seeds():
    sv = grid(4000)
    a = gp.run_arm_parallel(PROD, "abf", 64, SEED, 4000, sv, threads=4, noise="counter")
    b = gp.run_arm_parallel(PROD, "abf", 64, SEED, 4000, sv, threads=4, noise="replay")
    c = gp.run_arm_parallel(PROD, "abf", 64, SEED + 1, 4000, sv, threads=4, noise="counter")
    assert not np.array_equal(a["X_final"], b["X_final"]) and not np.array_equal(a["X_final"], c["X_final"])
    assert np.array_equal(a["C_all"].sum(1), 64 * sv)                    # one deposit per force evaluation


# ------------------------------------------------------------------------------------------------------- V-B5
@pytest.mark.parametrize("cfg_label,N,n_steps", [("easy", 512, 6000), ("easy", 64, 16000), ("prod", 512, 40000)])
def test_vb5_pairing_before_first_gather(cfg_label, N, n_steps):
    cfg = PROD if cfg_label == "prod" else EASY
    out = gp.vb5(N, 16, seed=SEED, n_steps=n_steps, cfg=cfg, save_steps=grid(n_steps))
    print("\nV-B5", cfg_label, N, json.dumps({k: out[k] for k in out if k != "fr_arm"}))
    assert out["n_saves_before_first_event"] >= 3 and out["first_save_after_first_event"] is not None
    assert out["identical_before_first_event"] and out["differs_after_first_event"]
    assert out["verdict_event_criterion"] == "PASS"
    # the preregistration's death criterion can only fail on saves between the first gather and the first death
    if not out["identical_before_first_death"]:
        assert out["n_saves_before_first_death"] > out["n_saves_before_first_event"]
    # the gating verdict is the preregistered death criterion
    assert out["gating_criterion_name"] == "death" and out["amendment"] is None
    assert out["verdict"] == ("PASS" if out["verdict_death_criterion"] == "PASS"
                              else "FAIL_PREREG_DEATH_CRITERION_PASS_EVENT_ONLY")


def test_vb5_death_criterion_fails_on_a_deathless_permutation():
    """A concrete run where the preregistered V-B5 wording fails although the engine is correct: the first FR
    opportunity with candidates (kd = 0 < kc) permutes the slots without a realised death, the arms diverge at
    the next save, and the first death comes later.  The gate reports it as
    FAIL_PREREG_DEATH_CRITERION_PASS_EVENT_ONLY (CLI exit 3), never PASS."""
    cfg = dict(EASY, h=4e-4, gamma=40.0, ramp_t=0.4)
    n = 3000
    out = gp.vb5(8, 4, seed=8600, n_steps=n, cfg=cfg, save_steps=np.arange(0, n + 1, 5, dtype=np.int64))
    assert out["identical_before_first_event"] and not out["identical_before_first_death"]
    assert out["first_save_after_first_event"] < out["first_save_after_first_death"]
    assert out["verdict"] == "FAIL_PREREG_DEATH_CRITERION_PASS_EVENT_ONLY" and gp.VB5_EXIT[out["verdict"]] == 3


def test_vb5_verdict_gating():
    """Exit-status / verdict mapping: the death criterion gates; event-only success is never PASS; the event
    criterion gates only with an amendment reference."""
    rec = lambda d, e: dict(verdict_death_criterion=d, verdict_event_criterion=e)   # noqa: E731
    assert gp.vb5_verdict(rec("PASS", "PASS"))["verdict"] == "PASS"
    v = gp.vb5_verdict(rec("FAIL", "PASS"))
    assert v["verdict"] == "FAIL_PREREG_DEATH_CRITERION_PASS_EVENT_ONLY" and gp.VB5_EXIT[v["verdict"]] == 3
    assert gp.vb5_verdict(rec("FAIL", "FAIL"))["verdict"] == "FAIL" and gp.VB5_EXIT["FAIL"] == 1
    with pytest.raises(ValueError):
        gp.vb5_verdict(rec("FAIL", "PASS"), criterion="event")
    with pytest.raises(ValueError):
        gp.vb5_verdict(rec("FAIL", "PASS"), criterion="event", amendment="  ")
    v = gp.vb5_verdict(rec("FAIL", "PASS"), criterion="event", amendment="Amendment 1 (2026-10-12)")
    assert v["verdict"] == "PASS" and v["amendment"] == "Amendment 1 (2026-10-12)"
    assert v["gating_criterion_name"] == "event"
    assert gp.vb5_verdict(rec("PASS", "FAIL"), criterion="event", amendment="A1")["verdict"] == "FAIL"


def test_vb5_permutation_without_death_exists():
    """Documents why the pairing is tied to the first GATHER: the validated engine itself has opportunities with
    candidates but no realised death (kd = 0 < kc and the pool law drops only copies / originals of cloned walkers),
    each a pure slot permutation (the configuration of tests/test_gateway_ladder_numba.py Q5)."""
    cfg = dict(EASY, h=4e-4, gamma=40.0, ramp_t=0.4)
    for noise in ("counter", "replay"):
        res = gp.run_arm_parallel(cfg, "fr", 24, 7100, 6000, [6000], threads=4, noise=noise)
        assert res["fr_opp_event_cum"][-1] > res["fr_opp_death_cum"][-1] > 0, noise


# ------------------------------------------------------------------------------------------------------- V-B3 (i)
M64 = (1 << 64) - 1


def py_philox(ctr, key):
    """Independent pure-Python Philox4x64-10 (Random123)."""
    c = [int(v) for v in ctr]
    k = [int(v) for v in key]
    for r in range(10):
        if r:
            k = [(k[0] + 0x9E3779B97F4A7C15) & M64, (k[1] + 0xBB67AE8584CAA73B) & M64]
        p0 = 0xD2E7470EE14C6C93 * c[0]
        p1 = 0xCA5A826395121157 * c[2]
        c = [((p1 >> 64) ^ c[1] ^ k[0]) & M64, p1 & M64, ((p0 >> 64) ^ c[3] ^ k[1]) & M64, p0 & M64]
    return c


def ctr_int(c):
    return sum(int(c[i]) << (64 * i) for i in range(4))


def int_ctr(v):
    v %= 1 << 256
    return np.array([(v >> (64 * i)) & M64 for i in range(4)], dtype=np.uint64)


def np_philox_block(key, ctr_value):
    """numpy's first 4 raw outputs for np.random.Philox(key, counter=ctr_value)."""
    bg = np.random.Philox(key=np.asarray(key, dtype=np.uint64), counter=int_ctr(ctr_value))
    return bg.random_raw(4)


KEYS = [gp.counter_key(8400, 512), gp.counter_key(8463, 16), np.array([0, 0], dtype=np.uint64),
        np.array([M64, M64], dtype=np.uint64)]


def test_vb3i_numpy_counter_convention():
    """numpy increments the counter BEFORE generating: Philox(counter=c) yields first the block of c + 1."""
    key = KEYS[0]
    for cv in (0, 1, 12345, (1 << 64) - 1, (7 << 64) + 3, (1 << 256) - 1):
        npb = np_philox_block(key, cv)
        c_this = gp.philox_raw(int_ctr(cv)[None, :], key)[0]
        c_next = gp.philox_raw(int_ctr(cv + 1)[None, :], key)[0]
        assert np.array_equal(npb, c_next) and not np.array_equal(npb, c_this)
        assert [int(v) for v in c_next] == py_philox(int_ctr(cv + 1), key)


@pytest.mark.parametrize("ki", range(len(KEYS)))
def test_vb3i_philox_matches_numpy(ki):
    key = KEYS[ki]
    rng = np.random.default_rng(20261011 + ki)
    # (a) 12000 consecutive counters from a random start (numpy's own increment incl. carries) in one stream
    start = sum(int(rng.integers(0, 1 << 62)) << (64 * w) for w in range(3))
    n = 12000
    bg = np.random.Philox(key=key, counter=int_ctr(start - 1))
    ref = bg.random_raw(4 * n).reshape(n, 4)
    ctrs = np.stack([int_ctr(start + i) for i in range(n)])
    assert np.array_equal(gp.philox_raw(ctrs, key), ref)
    # (b) 2000 random full-range counters, one numpy generator each, and the engine's (step, slot, 0, 0) counters
    rnd = rng.integers(0, np.iinfo(np.uint64).max, size=(2000, 4), dtype=np.uint64, endpoint=True)
    eng = np.array([[s, i, 0, 0] for s in (0, 1, 159, 160, 6_399_999) for i in (0, 1, 7, 511)], dtype=np.uint64)
    edge = np.array([[M64, 0, 0, 0], [M64, M64, 0, 0], [M64, M64, M64, 0], [M64, M64, M64, M64], [0, 0, 0, 0]],
                    dtype=np.uint64)
    allc = np.concatenate([rnd, eng, edge])
    out = gp.philox_raw(allc, key)
    for c, o in zip(allc, out):
        assert np.array_equal(np_philox_block(key, ctr_int(c) - 1), o)
    for c, o in zip(allc[:50], out[:50]):
        assert [int(v) for v in o] == py_philox(c, key)


# ------------------------------------------------------------------------------------------------------- Box-Muller
def bm_reference(o0, o1):
    ua = float(int(o0) >> 11) * 2.0 ** -53
    ub = float(int(o1) >> 11) * 2.0 ** -53
    r = math.sqrt(-2.0 * math.log(1.0 - ua))
    th = (2.0 * math.pi) * ub
    return r * math.cos(th), r * math.sin(th)


def test_box_muller_bitwise_and_thread_invariance():
    key = gp.counter_key(8400, 512)
    s0, ns, N = 1_234_560, 40, 512
    z = gp.counter_normals(key, s0, ns, N, nthreads=1)
    assert np.array_equal(z, gp.counter_normals(key, s0, ns, N, nthreads=8))
    ctr = np.array([[s0 + s, i, 0, 0] for s in range(ns) for i in range(N)], dtype=np.uint64)
    raw = gp.philox_raw(ctr, key)
    ref = np.array([bm_reference(o[0], o[1]) for o in raw]).reshape(ns, N, 2)
    assert np.array_equal(z[:, 0, :], ref[:, :, 0]) and np.array_equal(z[:, 1, :], ref[:, :, 1])
    # cross-check with the LTA backend's numba implementation of the same frozen convention (if present)
    try:
        from parallel import philox as ph
    except Exception:                                   # pragma: no cover
        ph = None
    if ph is not None and hasattr(ph, "gateway_normals2"):
        k0, k1 = np.uint64(key[0]), np.uint64(key[1])
        for s in range(3):
            for i in range(0, N, 37):
                zx, zy = ph.gateway_normals2(k0, k1, s0 + s, i)
                assert (zx, zy) == (z[s, 0, i], z[s, 1, i])


def test_box_muller_moments():
    z = gp.counter_normals(gp.counter_key(8401, 512), 0, 1000, 512, nthreads=8).ravel()   # 1.024e6 normals
    n = z.size
    m, v = z.mean(), z.var()
    sk = ((z - m) ** 3).mean() / v ** 1.5
    ku = ((z - m) ** 4).mean() / v ** 2 - 3.0
    assert abs(m) < 5.0 / math.sqrt(n) and abs(v - 1.0) < 5.0 * math.sqrt(2.0 / n)
    assert abs(sk) < 5.0 * math.sqrt(6.0 / n) and abs(ku) < 5.0 * math.sqrt(24.0 / n)
    zz = z.reshape(1000, 2, 512)
    rho = np.corrcoef(zz[:, 0, :].ravel(), zz[:, 1, :].ravel())[0, 1]         # Box-Muller partners
    assert abs(rho) < 5.0 / math.sqrt(n / 2)
    p3 = np.mean(np.abs(z) > 3.0)
    assert abs(p3 - 0.0026998) < 5.0 * math.sqrt(0.0027 / n)


def test_kernel_counter_noise_equals_exported_normals():
    """Counter mode == replay mode fed the exported counter normals (bitwise): the kernel's internal Philox /
    Box-Muller noise is exactly counter_normals (and so the numpy-verified Philox)."""
    for cfg, N, method, n in ((EASY, 64, "fr", 6000), (PROD, 512, "abf", 3000), (EASY, 13, "fr", 5000)):
        sv = grid(n)
        a = gp.run_arm_parallel(cfg, method, N, SEED, n, sv, threads=4, noise="counter")
        z = gp.counter_normals(gp.counter_key(SEED, N), 0, n, N, nthreads=4)
        b = gp.run_arm_parallel(cfg, method, N, SEED, n, sv, threads=7, noise="replay", _noise_ext=z)
        assert_bitwise(a, b, gp.science_keys(a), (N, method))
        assert method == "abf" or a["fr_deaths_cum"][-1] > 0


# ------------------------------------------------------------------------------------------------------- contract
def test_contract_timers_and_inertness():
    N, n = 512, 8000
    sv = grid(n)
    ref = gl.run_arm(EASY, "fr", N, SEED, n, sv, diagnostics=False)
    par = gp.run_arm_parallel(EASY, "fr", N, SEED, n, sv, threads=8, noise="replay")
    assert set(par) == (set(ref) - set(gp.WALL_KEYS)) | NEW_KEYS
    for k in gp.shared_keys(ref):
        if k != "cfg":
            assert np.asarray(par[k]).dtype == np.asarray(ref[k]).dtype and np.shape(par[k]) == np.shape(ref[k]), k
    assert par["engine"] == "gateway_parallel/1" and par["noise_mode"] == "replay" and par["threads"] == 8
    w = par["wall_s_at_save"]
    assert w.dtype == np.float64 and np.all(np.isfinite(w)) and np.all(np.diff(w) >= 0) and w[0] >= 0
    assert w[-1] <= par["wall_total_s"]
    assert par["n_force_evals_at_save"].dtype == np.int64 and np.array_equal(par["n_force_evals_at_save"], N * sv)
    for k in ("t_force_move_s", "t_abf_s", "t_sync_s", "t_fr_s", "t_save_s", "warmup_s", "wall_total_s"):
        assert isinstance(par[k], float) and par[k] >= 0.0, k
    assert par["t_force_move_s"] + par["t_abf_s"] + par["t_sync_s"] <= 1.05 * par["wall_total_s"]
    assert par["t_fr_s"] > 0 and par["t_force_move_s"] > 0
    m = json.loads(par["meta"])
    for k in ("sources_sha256", "validated_engine_sha256", "git_commit", "build", "numba", "numpy", "host", "cpus",
              "omp", "timers_extra", "run_signature", "run_signature_sha256"):
        assert k in m, k
    assert set(m["sources_sha256"]) == {"src/parallel/gateway_parallel.py", "src/parallel/gw_kernel.c",
                                        "src/parallel/__init__.py"}
    assert m["validated_engine_sha256"]["src/gateway_ladder_numba.py"] == gl._PROVENANCE["engine_sha256"]
    assert m["build"]["cflags"] == list(gp.CFLAGS)
    # cpus = the process allocation read before libgomp loaded (libgomp then binds the master to place 0)
    assert m["cpus"] == gp._AFFINITY_AT_IMPORT == gp.allocated_cpus()
    assert set(m["cpus_master_bound"]) <= set(m["cpus"])
    if os.environ.get("GW_PARALLEL_KEEP_OMP_ENV") != "1" and not gp._LIBGOMP_PRELOADED:
        assert m["cpus_omp_places_union"] == m["cpus"]
        if len(m["cpus"]) > 1:
            assert m["cpus_master_bound"] == m["cpus"][:1]       # what the old meta['cpus'] recorded
    assert m["timers_extra"]["n_fr_calls"] == int(par["n_fr_steps"]) and m["timers_extra"]["tsc_hz"] > 1e8
    # sampled in-kernel timers: the documented sampled-step sequence, exact region total
    te = m["timers_extra"]
    assert m["timer_sample_log2"] == gp.TIMER_SAMPLE_LOG2 and te["n_loop_steps"] == n
    assert te["n_sampled_steps"] == len(sampled_steps(n, gp.TIMER_SAMPLE_LOG2)) >= 1
    assert te["n_kernel_calls"] > 10                            # many segments: the sequence is carried across calls
    tot = par["t_force_move_s"] + par["t_abf_s"] + par["t_sync_s"]
    assert abs(tot - te["t_region_wall_s"]) <= 1e-9 * te["t_region_wall_s"]
    assert abs(te["t_exact_parts_sum_s"] - te["t_region_wall_s"]) <= 1e-9 * te["t_region_wall_s"]
    assert par["t_abf_s"] > 0 and par["t_sync_s"] >= 0
    # timers (off / every step / sampled) and warm-up are inert
    for kw in (dict(timers=False), dict(warmup=False), dict(timers=False, variant="replicated_dissem"),
               dict(_timer_sample_log2=0), dict(_timer_sample_log2=3, variant="owner_dissem")):
        q = gp.run_arm_parallel(EASY, "fr", N, SEED, n, sv, threads=8, noise="replay", **kw)
        assert_bitwise(ref, q, gp.shared_keys(ref), kw)
        qe = json.loads(q["meta"])["timers_extra"]
        if not kw.get("timers", True):
            assert math.isnan(q["t_force_move_s"]) and math.isnan(q["t_abf_s"]) and math.isnan(q["t_sync_s"])
            assert q["t_fr_s"] > 0 and qe["n_sampled_steps"] == 0
        elif "_timer_sample_log2" in kw:
            k = kw["_timer_sample_log2"]
            assert qe["n_sampled_steps"] == (n if k == 0 else len(sampled_steps(n, k)))
            qt = q["t_force_move_s"] + q["t_abf_s"] + q["t_sync_s"]
            assert abs(qt - qe["t_region_wall_s"]) <= 1e-9 * qe["t_region_wall_s"]


def test_timer_sampling_invariance():
    """The sampled-step set depends on the global step only: the same count at every thread count, variant,
    noise mode and segmentation (ABF: segments at the saves; FR: at every FR opportunity)."""
    n = 6000
    want = len(sampled_steps(n, 4))
    for method, sv in (("abf", [n]), ("abf", grid(n)), ("fr", grid(n))):
        for T, v, noise in ((1, "owner_omp", "counter"), (5, "replicated_dissem", "counter"),
                            (16, "owner_dissem", "replay")):
            r = gp.run_arm_parallel(PROD, method, 64, SEED, n, sv, threads=T, noise=noise, variant=v,
                                    _timer_sample_log2=4)
            te = json.loads(r["meta"])["timers_extra"]
            assert te["n_sampled_steps"] == want, (method, len(sv), T, v, noise)


def test_refusals():
    sv = [100]
    with pytest.raises(ValueError):
        gp.run_arm_parallel(PROD, "abf", 16, SEED, 100, sv, threads=2, noise="philox")
    with pytest.raises(ValueError):
        gp.run_arm_parallel(dict(PROD, alpha=0.5), "abf", 16, SEED, 100, sv, threads=2, noise="counter")
    with pytest.raises(ValueError):
        gp.run_arm_parallel(dict(PROD, lam=4.0), "abf", 16, SEED, 100, sv, threads=2, noise="counter")
    with pytest.raises(ValueError):
        gp.run_arm_parallel(PROD, "abf", 16, SEED, 100, sv, threads=0, noise="counter")
    with pytest.raises(ValueError):
        gp.run_arm_parallel(PROD, "fr", 1, SEED, 100, sv, threads=1, noise="counter")
    with pytest.raises(ValueError):
        gp.run_arm_parallel(PROD, "abf", 16, SEED, 100, sv, threads=2, noise="counter", variant="nope")
    with pytest.raises(ValueError):
        gp.run_arm_parallel(PROD, "abf", 16, SEED, 100, [50.5], threads=2, noise="counter")
    ok = gp.run_arm_parallel(dict(PROD, alpha=1.0, lam=1.0), "abf", 16, SEED, 100, sv, threads=2, noise="counter")
    assert ok["n_force_evals"] == 1600


def _restore_allocation():
    """subprocess preexec: libgomp bound this (parent) process's main thread to place 0, and a child inherits the
    calling thread's affinity; give the CLI child the parent's allocation again (never more than it)."""
    os.sched_setaffinity(0, gp._AFFINITY_AT_IMPORT)


def test_cli_smoke(tmp_path):
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src"), CUDA_VISIBLE_DEVICES="")
    env.pop("OMP_PLACES", None)                 # the child derives its places from its (restored) allocation
    py = sys.executable
    kw = dict(env=env, capture_output=True, text=True, timeout=300, preexec_fn=_restore_allocation)
    out1 = tmp_path / "vb1.json"
    r = subprocess.run([py, "-m", "parallel.gateway_parallel", "--vb1", "--N", "16", "--method", "fr",
                        "--threads", "1,4", "--steps", "6000", "--out", str(out1)], **kw)
    d = json.loads(out1.read_text())
    assert d["identical_all"] and d["verdict"] in ("PASS", "IDENTICAL_BUT_SHORT"), r.stderr[-2000:]
    assert d["validated"]["n_saves"] >= 20 and [x["threads"] for x in d["runs"]] == [1, 4]
    out2 = tmp_path / "vb2.json"
    r = subprocess.run([py, "-m", "parallel.gateway_parallel", "--vb2", "--N", "64", "--method", "abf",
                        "--threads", "1,2,16", "--steps", "3000", "--variant", "replicated_dissem", "--out",
                        str(out2)], **kw)
    assert r.returncode == 0, r.stderr[-2000:]
    d2 = json.loads(out2.read_text())
    assert d2["verdict"] == "PASS" and d2["kernel_variant"] == "replicated_dissem"
    r = subprocess.run([py, "-m", "parallel.gateway_parallel", "--probe", "64", "2", "2000"], **kw)
    assert r.returncode == 0, r.stderr[-2000:]
    p = json.loads(r.stdout)
    assert p["label"].startswith("development probe") and p["us_per_step"] > 0
    # V-B5: the exit status follows the preregistered (death) criterion
    out5 = tmp_path / "vb5.json"
    r = subprocess.run([py, "-m", "parallel.gateway_parallel", "--vb5", "--N", "64", "--threads", "4", "--steps",
                        "16000", "--out", str(out5)], **kw)
    d5 = json.loads(out5.read_text())
    assert d5["gating_criterion_name"] == "death" and d5["amendment"] is None
    assert r.returncode == gp.VB5_EXIT[d5["verdict"]], (r.returncode, d5["verdict"], r.stderr[-2000:])
    assert d5["verdict"] == "PASS" or r.returncode != 0
    summ = json.loads(r.stdout)
    assert {"verdict", "verdict_death_criterion", "verdict_event_criterion"} <= set(summ)
    assert d5["cpus"] == gp._AFFINITY_AT_IMPORT                 # the allocation, not libgomp's bound master cpu
    if len(d5["cpus"]) > 1 and not gp._LIBGOMP_PRELOADED:
        assert d5["cpus_master_bound"] == d5["cpus"][:1]
    # the event criterion needs an amendment reference (argparse usage error, exit 2)
    r = subprocess.run([py, "-m", "parallel.gateway_parallel", "--vb5", "--N", "64", "--threads", "4", "--steps",
                        "1000", "--vb5-criterion", "event"], **kw)
    assert r.returncode == 2 and "amendment" in r.stderr
    # timer-overhead development check (tiny; the threshold is not tested here)
    out6 = tmp_path / "oh.json"
    r = subprocess.run([py, "-m", "parallel.gateway_parallel", "--overhead", "4", "--reps", "2", "--walker-steps",
                        "2e5", "--max-overhead-pct", "1e6", "--out", str(out6)], **kw)
    assert r.returncode == 0, r.stderr[-2000:]
    d6 = json.loads(out6.read_text())
    assert d6["verdict"] == "PASS" and d6["rows"][0]["N"] == 4 and len(d6["rows"][0]["runs_on"]) == 2
