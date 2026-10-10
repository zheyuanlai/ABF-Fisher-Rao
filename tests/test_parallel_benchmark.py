"""Experiment III wall-clock benchmark harness (docs/mechanism/SCIENTIFIC_PLAN.md section 8):
scripts/mechanism/{bench_torch_engines,parallel_benchmark,analyze_parallel_benchmark}.py.

B1  the lean torch gateway driver is BITWISE gateway_core.simulate_batch (estimator 'histogram', one row, method abf /
    fr_uniform; CPU float64): bin accumulators at every save, final x / y, FR death / clone candidate counts, with FR
    events firing;
B2  the lean torch LTA driver is BITWISE core_lta.run_sampler (abf_estimator 'histogram', abf / fr_uniform): the
    reported mean force at every save, the final production / all-steps accumulators, the replacement count;
B3  the frozen knobs at N = 512 map onto the torch engines exactly (gateway fr_every 160, ramp 160000 steps, cap 40;
    LTA cap 10, deposit clip 480) and every rule mismatch is refused (cap at small N, non-integer ramp, ...);
B4  the numba timing hooks (chunk kernels split at the save steps) change no bit of either production engine, fire one
    monotone stamp per save and restore the engine functions;
B5  the save grid is the production grid (equal to the equal-budget N = 512 files' save_step and to
    run_ladder.save_grid);
B6  run_one end to end on tiny budgets (numba + torch-on-CPU, both systems, both arms, pinned to one CPU): result
    contract, deposit accounting, ABF / FR pairing before the LTA FR start, skip of a complete result, refusal to
    overwrite a different run, refusal of an unpinned timed run, the GPU guard (one allowed index only, busy /
    unreadable occupancy refused); then the analysis: summary / tables / figures with legibility PASS, NOT TESTED
    backends, equivalence NOT APPLICABLE at a shortened budget, the GPU ceiling check;
B7  analysis units: tau -> wall at the first save of the persistent window (stub scorer), censoring, paired ratios,
    Holm, the permutation median test, the equivalence rule on synthetic production data (consistent / inconsistent
    / INCOMPLETE below 8 pairs / reference without the seeds sharing randomness), its family-wise false-INCONSISTENT
    rate under H0 by Monte Carlo, W1 on the uncontended pairs;
B8  campaign job list (production seeds, alternating arm order), dry run, resource guard, contended-launch refusal;
B9  CUDA (skipped unless CUDA_VISIBLE_DEVICES names exactly ONE device at session start, CUDA is available and the
    parallel_benchmark GPU guard accepts that device): both torch drivers run on the device, deposit accounting holds,
    stamps are monotone and the FR device time is finite;
B14 telemetry helpers: SMT state (tri-state), GPU monitor (device memory, other processes, unknown occupancy), the
    matched-host-state GPU / CPU ratio of bench_step_costs; the gateway torch arms are unpaired after the first FR
    opportunity (documented behaviour of gateway_core.resample_indices);
B15 the EXPLORATORY early-window consistency check: step alignment with the production files (bitwise-prefix
    numba runs reproduce the production per-seed value exactly), INCOMPLETE below 8 pairs, overlap exclusion.

CPU:  CUDA_VISIBLE_DEVICES="" NUMBA_CACHE_DIR=~/.cache/numba_mech taskset -c 64-75 python -m pytest tests/test_parallel_benchmark.py -q
CUDA: CUDA_VISIBLE_DEVICES=2 ... python -m pytest tests/test_parallel_benchmark.py -q -k cuda
"""
import contextlib
import json
import math
import os
import subprocess
import sys

_CVD_AT_START = os.environ.get("CUDA_VISIBLE_DEVICES")          # captured before any module blanks it
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.expanduser("~/.cache/numba_mech"))
for _k in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
for _p in (os.path.join(ROOT, "src"), os.path.join(ROOT, "scripts", "mechanism"), os.path.join(ROOT, "scripts", "equal_budget")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import parallel_benchmark as PB  # noqa: E402

PROD_GW = json.load(open(os.path.join(ROOT, "configs", "equal_budget_v2", "gateway_production.json")))
PROD_L3 = json.load(open(os.path.join(ROOT, "configs", "equal_budget_v2", "lta_300K_production.json")))


@contextlib.contextmanager
def one_cpu():
    """Pin this process to ONE logical CPU of its affinity (timed runs refuse otherwise), then restore it."""
    old = os.sched_getaffinity(0)
    os.sched_setaffinity(0, {min(old)})
    try:
        yield
    finally:
        os.sched_setaffinity(0, old)


@pytest.fixture(scope="module")
def torch_cpu():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    torch.set_num_threads(1)
    return torch


# ------------------------------------------------------------------------------------------------ B1 / B2
def test_gateway_torch_bitwise_vs_simulate_batch(torch_cpu):
    torch = torch_cpu
    import gateway_core as gc
    import bench_torch_engines as BT
    cfg = dict(PROD_GW["engine_cfg"])
    cfg.update(ramp_t=100 * cfg["h"], gamma=15.0)            # FR events within a short run (same knobs both sides)
    N, n_steps, seed = 64, 3200, 8101
    kn = BT.gateway_knobs(cfg, N)
    g = gc.GatewayConfig(beta=16.0, H=0.5, omega_out=1.0, r=32.0, s=0.1, N=N, dt=cfg["h"], n_steps=n_steps,
                         save_every=400, estimator="histogram", n_bins=180, min_count=1.0, gamma=15.0, eta=0.1,
                         fr_every=kn["fr_every"], fr_burnin=0, ramp_fraction=kn["ramp_steps"] / n_steps,
                         score_clip=3.0, max_event_fraction=0.08)
    assert int(g.ramp_fraction * n_steps) == kn["ramp_steps"]
    saves = np.array([st + 1 for st in range(n_steps) if st % 400 == 0 or st == n_steps - 1])
    for meth, Mth in (("abf", gc.ABF), ("fr", gc.FR_UNIFORM)):
        rec = gc.simulate_batch(gc.BatchSpec([g], [seed], [Mth], batch_seed=seed), device=torch.device("cpu"),
                                dtype=torch.float64, store_accumulators=True, store_final_state=True)[0]
        r = BT.GatewayTorchRun(cfg, meth, N, seed, n_steps, saves, "cpu").run()
        for a, b in (("Mh_t", "M_all"), ("Ch_t", "C_all"), ("X_final", "X_final"), ("Y_final", "Y_final")):
            assert np.array_equal(rec[a], r[b]), (meth, a)
        assert r["fr_kd_cum"][-1] == rec["n_die"] and r["fr_kc_cum"][-1] == rec["n_clone"]
        assert r["n_fr_opps"] == rec["n_fr_apply"]
        if meth == "fr":
            assert rec["n_die"] > 0 and rec["n_clone"] > 0
        assert np.all(np.diff(r["wall_s_at_save"]) >= 0)
        assert np.array_equal(r["C_all"].sum(1), N * saves)


def test_lta_torch_bitwise_vs_run_sampler(torch_cpu):
    torch = torch_cpu
    from lta import core_lta as CL
    import bench_torch_engines as BT
    cfg = dict(PROD_L3["engine_cfg"])
    cfg.update(warmup_steps=50, burn_in_steps=60, fr_start_steps=70, fr_rate=20.0)
    N, n_steps, seed = 64, 600, 5
    sim = BT.lta_sim_config(cfg, 300.0, N, n_steps, seed)
    sim.save_every = 100
    system = CL.LTASystem(CL.LTAParams(temperature=300.0), torch.device("cpu"), torch.float64, root=ROOT)
    for meth in ("abf", "fr"):
        out = CL.run_sampler("abf" if meth == "abf" else "fr_uniform", system, sim, [seed], verbose=False)
        sv = np.array(out["steps"])
        r = BT.LTATorchRun(cfg, 300.0, meth, N, seed, n_steps, sv, "cpu").run()
        use_prod = r["C_prod"].sum(1) > 0
        Mr = np.where(use_prod[:, None], r["M_prod"], r["M_all"])
        Cr = np.where(use_prod[:, None], r["C_prod"], r["C_all"])
        mf = CL.histogram_mean_force(torch.as_tensor(Mr), torch.as_tensor(Cr)).numpy()
        assert np.array_equal(mf, out["mean_force"][:, 0]), meth
        assert np.array_equal(r["M_prod"][-1], out["fsum_prod"][0])
        assert np.array_equal(r["C_prod"][-1], out["u_counts"][0])
        assert np.array_equal(r["C_all"][-1], out["final_eff_counts"][0])
        assert int(r["cum_deaths"][-1]) == int(out["total_replacement_events"][0])
        assert r["n_fr_opps"] == len(out["event_counts"])
        if meth == "fr":
            assert int(out["total_replacement_events"][0]) > 0
        assert np.array_equal(r["C_all"].sum(1), N * (sv + 1))


# ------------------------------------------------------------------------------------------------ B3
def test_frozen_knobs_map_exactly_and_mismatches_refused(torch_cpu):
    import bench_torch_engines as BT
    kn = BT.gateway_knobs(PROD_GW["engine_cfg"], 512)
    assert (kn["fr_every"], kn["ramp_steps"], kn["cap"]) == (160, 160000, 40)
    assert math.isclose(kn["dt_fr"], 0.004)
    with pytest.raises(BT.EquivalenceError):
        BT.gateway_knobs(PROD_GW["engine_cfg"], 8)                   # numba cap 1, torch floor(0.64) = 0
    with pytest.raises(BT.EquivalenceError):
        BT.gateway_knobs(dict(PROD_GW["engine_cfg"], ramp_t=4.0 + 1e-6), 512)
    with pytest.raises(BT.EquivalenceError):
        BT.gateway_knobs(dict(PROD_GW["engine_cfg"], fr_interval_t=0.00401), 512)
    sim = BT.lta_sim_config(PROD_L3["engine_cfg"], 300.0, 512, 600000, 1)
    assert int(sim.max_event_fraction * 512) == 10 and sim.abf_force_clip * 8 == 480.0
    assert (sim.fr_every, sim.fr_start_steps, sim.abf_warmup_steps, sim.estimator_burn_in_steps) == (5, 20000, 20000, 20000)
    assert sim.abf_estimator == "histogram" and sim.fr_rate == 0.2 and sim.score_clip == 2.0 and sim.kde_bandwidth == 0.1
    with pytest.raises(BT.EquivalenceError):
        BT.lta_sim_config(PROD_L3["engine_cfg"], 300.0, 32, 1000, 1)  # numba cap 1, torch int(0.64) = 0
    with pytest.raises(BT.EquivalenceError):
        BT.lta_sim_config(dict(PROD_L3["engine_cfg"], deposit_clip=400.0), 300.0, 512, 1000, 1)


# ------------------------------------------------------------------------------------------------ B4
def test_numba_clock_hooks_are_bitwise_inert():
    import gateway_ladder_numba as GE
    import lta_ladder_numba as LE
    cfg = dict(PROD_GW["engine_cfg"], ramp_t=100 * PROD_GW["engine_cfg"]["h"], gamma=15.0)
    sv = PB.save_grid(PROD_GW, 20000, "gateway")
    orig = GE._advance
    for meth in ("abf", "fr"):
        a = GE.run_arm(cfg, meth, 64, 8100, 20000, sv, diagnostics=False, chunk_steps=777)
        clk = PB.SaveClock(sv)
        with PB.gateway_numba_clock(GE, clk):
            clk.start()
            b = GE.run_arm(cfg, meth, 64, 8100, 20000, sv, diagnostics=False, chunk_steps=777)
        for k in ("M_all", "C_all", "X_final", "Y_final", "fr_kd_cum", "fr_deaths_cum"):
            assert np.array_equal(a[k], b[k]), (meth, k)
        assert np.all(np.isfinite(clk.wall)) and np.all(np.diff(clk.wall) >= 0)
        if meth == "fr":
            assert a["fr_deaths_cum"][-1] > 0
    assert GE._advance is orig
    ec = dict(PROD_L3["engine_cfg"], warmup_steps=50, burn_in_steps=60, fr_start_steps=70, fr_rate=20.0, n_steps=3000,
              diagnostics=False)
    c = LE.make_cfg(300.0, 64, 30000, **ec)
    sv = PB.save_grid(PROD_L3, 3000, "lta")
    orig = LE._run_chunk
    for meth in ("abf", "fr"):
        a = LE.run_arm(c, meth, sv, chunk_steps=333)
        clk = PB.SaveClock(sv)
        with PB.lta_numba_clock(LE, clk):
            clk.start()
            b = LE.run_arm(c, meth, sv, chunk_steps=333)
        for k in ("M_all", "C_all", "M_prod", "C_prod", "q_final", "cum_deaths"):
            assert np.array_equal(a[k], b[k]), (meth, k)
        assert np.all(np.isfinite(clk.wall)) and np.all(np.diff(clk.wall) >= 0)
        if meth == "fr":
            assert a["cum_deaths"][-1] > 0
    assert LE._run_chunk is orig


# ------------------------------------------------------------------------------------------------ B5
@pytest.mark.parametrize("system,fn", [("gateway", "s8100_abf.npz"), ("lta300", "s30000_abf.npz")])
def test_save_grid_is_the_production_grid(system, fn):
    P, N, _ = PB.system_spec(system)
    n = PB.full_n_steps(P, N)
    p = os.path.join(ROOT, "results", "equal_budget_v2", PB.SYSTEMS[system]["out_dir"], "N512", fn)
    if not os.path.exists(p):
        pytest.skip("production file absent")
    with np.load(p, allow_pickle=False) as z:
        assert np.array_equal(z["save_step"], PB.save_grid(P, n, PB.SYSTEMS[system]["kind"]))
    code = ("import sys, json, numpy as np; sys.path.insert(0, 'scripts/equal_budget'); sys.path.insert(0, 'src'); "
            "import run_ladder as RL; P = json.load(open(sys.argv[1])); P['system_key'] = sys.argv[2]; "
            "print(json.dumps(RL.save_grid(P, int(sys.argv[3])).tolist()))")
    for nst in (n, 64000, 30000):
        out = subprocess.run([sys.executable, "-c", code, os.path.join(ROOT, P["config_path"]), system, str(nst)],
                             cwd=ROOT, capture_output=True, text=True, check=True).stdout
        assert np.array_equal(np.array(json.loads(out)), PB.save_grid(P, nst, PB.SYSTEMS[system]["kind"]))


# ------------------------------------------------------------------------------------------------ B6
@pytest.fixture(scope="module")
def tiny_root(tmp_path_factory, torch_cpu):
    root = str(tmp_path_factory.mktemp("bench"))
    with one_cpu():
        for m in PB.METHODS:
            PB.run_one("gateway", "cpu_numba_1core", m, 8100, out_root=root, n_steps=3200)
            PB.run_one("gateway", "cpu_numba_1core", m, 8101, out_root=root, n_steps=3200)
            PB.run_one("lta300", "cpu_numba_1core", m, 30000, out_root=root, n_steps=400)
            PB.run_one("gateway", "cpu_torch_test", m, 8100, out_root=root, n_steps=3200, warm=True)
            PB.run_one("lta300", "cpu_torch_test", m, 30000, out_root=root, n_steps=60, warm=False)
    return root


def test_run_one_contract(tiny_root):
    for system, backend, seed, n in (("gateway", "cpu_numba_1core", 8100, 3200), ("lta300", "cpu_numba_1core", 30000, 400),
                                     ("gateway", "cpu_torch_test", 8100, 3200), ("lta300", "cpu_torch_test", 30000, 60)):
        for m in PB.METHODS:
            npz, js = PB.out_paths(tiny_root, system, backend, seed, m)
            r = PB.load_run(npz)
            meta = json.load(open(js))
            assert meta["status"] == "complete" and meta["n_steps"] == n and meta["N"] == 512
            kind = PB.SYSTEMS[system]["kind"]
            sv = PB.save_grid(PB.system_spec(system)[0], n, kind)
            assert np.array_equal(r["save_step"], sv)
            exp = 512 * sv if kind == "gateway" else 512 * (sv + 1)
            assert np.array_equal(r["n_force_evals_at_save"], exp)
            assert np.allclose(r["C_all"].sum(1), exp)
            assert np.all(np.diff(r["wall_s_at_save"]) >= 0) and r["wall_s_at_save"][-1] <= r["wall_total_s"] + 1e-9
            if kind == "lta":
                assert "M_prod" in r and "C_prod" in r
            assert meta["bitwise_vs_production"]["status"].startswith("not applicable")
            assert meta["pinned"] and len(meta["affinity"]) == 1 and meta["smt_state"] in ("contended", "uncontended")
            assert set(meta["cpu_busy_fraction"]) == {str(c) for c in meta["affinity"] + meta["smt_siblings"]}
            assert meta["process_wall_s"] >= r["wall_total_s"] and math.isnan(float(r["peak_gpu_device_mb"]))
            if backend == "cpu_torch_test":
                assert np.isfinite(float(r["peak_rss_mb"])) and float(r["monitor_s"]) == 0.0
            if backend == "cpu_numba_1core":
                assert np.all(np.isnan(r["fr_time_s_at_save"]))
            elif m == "fr":
                assert np.isfinite(r["fr_time_s_at_save"][-1]) and r["fr_time_s_at_save"][-1] <= r["wall_total_s"]
    # LTA: n_steps < fr_start -> the FR arm is bitwise the ABF arm (shared IC and noise)
    a = PB.load_run(PB.out_paths(tiny_root, "lta300", "cpu_numba_1core", 30000, "abf")[0])
    f = PB.load_run(PB.out_paths(tiny_root, "lta300", "cpu_numba_1core", 30000, "fr")[0])
    for k in ("M_all", "C_all", "M_prod", "C_prod"):
        assert np.array_equal(a[k], f[k])
    a = PB.load_run(PB.out_paths(tiny_root, "lta300", "cpu_torch_test", 30000, "abf")[0])
    f = PB.load_run(PB.out_paths(tiny_root, "lta300", "cpu_torch_test", 30000, "fr")[0])
    assert np.array_equal(a["M_all"], f["M_all"])


def test_skip_refuse_and_gpu_guard(tiny_root, tmp_path, monkeypatch):
    npz, js = PB.out_paths(tiny_root, "gateway", "cpu_numba_1core", 8100, "abf")
    mt = os.path.getmtime(npz)
    PB.run_one("gateway", "cpu_numba_1core", "abf", 8100, out_root=tiny_root, n_steps=3200)     # complete: skipped
    assert os.path.getmtime(npz) == mt
    with pytest.raises(PB.BenchError):                                                          # different run
        PB.run_one("gateway", "cpu_numba_1core", "abf", 8100, out_root=tiny_root, n_steps=6400)
    if len(os.sched_getaffinity(0)) > 1:                                                        # unpinned timed run
        with pytest.raises(PB.BenchError, match="pinned"):
            PB.run_one("gateway", "cpu_numba_1core", "abf", 8102, out_root=str(tmp_path), n_steps=320)
    with one_cpu():
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
        with pytest.raises(PB.BenchError, match="exactly ONE"):
            PB.run_one("gateway", "gpu_torch", "abf", 8100, out_root=str(tmp_path), n_steps=320)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
        with pytest.raises(PB.BenchError, match="exactly ONE"):
            PB.run_one("gateway", "gpu_torch", "abf", 8100, out_root=str(tmp_path), n_steps=320)
        for idx in ("0", "1", "3"):                      # other users' GPUs: refused, never overridable
            monkeypatch.setenv("CUDA_VISIBLE_DEVICES", idx)
            with pytest.raises(PB.BenchError, match="not an allowed"):
                PB.run_one("gateway", "gpu_torch", "abf", 8100, out_root=str(tmp_path), n_steps=320,
                           allow_busy_gpu=True)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
        monkeypatch.setattr(PB, "gpu_processes", lambda: None)                  # nvidia-smi failing: unknown
        with pytest.raises(PB.BenchError, match="cannot be read"):
            PB.run_one("gateway", "gpu_torch", "abf", 8100, out_root=str(tmp_path), n_steps=320)
        assert PB.gpu_guard(allow_busy=True) == (2, None)
        busy = dict(processes={2: [dict(pid=999999, used_memory="100 MiB")], 3: []}, names={2: "H200", 3: "H200"})
        monkeypatch.setattr(PB, "gpu_processes", lambda: busy)
        with pytest.raises(PB.BenchError, match="already runs"):
            PB.run_one("gateway", "gpu_torch", "abf", 8100, out_root=str(tmp_path), n_steps=320)
        assert PB.gpu_guard(allow_busy=True)[0] == 2
        monkeypatch.setattr(PB, "gpu_processes", lambda: dict(processes={2: []}, names={2: "H200"}))
        assert PB.gpu_guard() == (2, [])
        with pytest.raises(PB.BenchError):
            PB.run_one("gateway", "cpu_numba_threads", "abf", 8100, out_root=str(tmp_path), n_steps=320)
        with pytest.raises(PB.BenchError):
            PB.run_one("gateway", "cpu_numba_1core", "abf", 8100, out_root=str(tmp_path), n_steps=10 ** 9)
    assert not os.listdir(tmp_path) or not any(f.endswith(".npz") for _, _, fs in os.walk(tmp_path) for f in fs)


def test_analysis_end_to_end(tiny_root, tmp_path):
    import analyze_parallel_benchmark as AP
    S, man = AP.analyze(tiny_root, str(tmp_path / "an"), str(tmp_path / "fig"))
    assert os.path.exists(tmp_path / "an" / "summary.json") and os.path.exists(tmp_path / "an" / "tables.md")
    js = json.load(open(tmp_path / "an" / "summary.json"))
    gw = js["systems"]["gateway"]
    assert set(gw["backends"]) == {"cpu_numba_1core", "cpu_torch_test"}
    nt = {x["backend"]: x["label"] for x in gw["not_tested"]}
    assert nt == {"gpu_torch": "NOT TESTED", "cpu_numba_threads": "NOT TESTED"}
    assert gw["effective_parallelism"].startswith("NOT TESTED")
    eq = gw["backends"]["cpu_torch_test"]["equivalence"]["status"]
    assert eq.startswith("NOT APPLICABLE")
    b = gw["backends"]["cpu_numba_1core"]
    assert b["n_seeds"] == {"abf": 2, "fr": 2} and "paired" in b and "fr_overhead" in b
    assert js["systems"]["lta150"]["backends"] == {}
    assert set(man) == {"T_time_to_accuracy", "C_cost", "X_error_vs_wall"}       # no full-budget torch run: no Q
    for tag, v in man.items():
        assert v["legibility"]["status"] == "pass", (tag, v["legibility"]["failures"][:3])
        for f in v["files"]:
            assert os.path.exists(os.path.join(AP.ROOT, f))
    assert "W1" in b and b["W1"]["n_pairs"] == 2 and b["W1"]["status"] in ("EVALUATED", "NOT EVALUABLE (no pair with "
                                                                              "both runs verified uncontended)")
    assert b["W1"]["n_pairs_uncontended"] + b["W1"]["n_pairs_excluded"] == 2
    cp = js["cost_projection"]
    assert set(cp) == {"cpu_numba_1core", "gpu_torch"} and not cp["cpu_numba_1core"]["all_measured"]
    assert cp["gpu_torch"]["rows"]["gateway/abf"]["source"].startswith("fallback")
    cc = cp["gpu_torch"]["ceiling_check"]["per_system"]
    assert cc["gateway"]["equivalence_gate"].startswith("CANNOT BE EVALUATED") and cc["lta300"]["within_ceiling"]
    tab = open(tmp_path / "an" / "tables.md").read()
    assert "W1" in tab and "host RSS max (MB)" in tab and "CANNOT BE EVALUATED" in tab


# ------------------------------------------------------------------------------------------------ B7
class _StubScorer:
    def __init__(self, curve):
        self.curve = curve

    def errors(self, res):
        return dict(e_F=self.curve)


def test_tau_maps_to_wall_at_first_persistent_save(tiny_root, monkeypatch):
    import analyze_parallel_benchmark as AP
    npz = PB.out_paths(tiny_root, "gateway", "cpu_numba_1core", 8100, "fr")[0]
    r = PB.load_run(npz)
    S = r["save_step"].size
    P, N, thr = PB.system_spec("gateway")
    curve = np.full(S, 1.0)
    curve[3] = thr / 2           # transient dip: not persistent
    curve[7:] = thr              # <= threshold from save 7 on (equality counts)
    monkeypatch.setitem(AP._SCORERS, "gateway", (_StubScorer(curve), P, N, thr))
    row, _ = AP.score_run(npz, "gateway")
    assert row["tau_wall"] == float(r["wall_s_at_save"][7]) and row["tau_t"] == float(r["save_t"][7])
    assert row["tau_fe"] == float(r["n_force_evals_at_save"][7]) and not row["tau_censored"]
    curve2 = curve.copy()
    curve2[-1] = 1.0
    monkeypatch.setitem(AP._SCORERS, "gateway", (_StubScorer(curve2), P, N, thr))
    row, _ = AP.score_run(npz, "gateway")
    assert row["tau_censored"] and math.isinf(row["tau_wall"]) and math.isinf(row["tau_fe"])


def test_statistics_units():
    import analyze_parallel_benchmark as AP
    h = AP.holm({"a": 0.001, "b": 0.03, "c": 0.04})
    assert h["a"][1] and not h["b"][1] and not h["c"][1]          # 0.001 <= 0.05/3; 0.03 > 0.05/2 stops
    assert math.isclose(h["a"][0], 0.003) and math.isclose(h["b"][0], 0.06) and math.isclose(h["c"][0], 0.06)
    h = AP.holm({"a": 0.001, "b": 0.02, "c": 0.04})
    assert all(v[1] for v in h.values())                          # 0.02 <= 0.05/2, 0.04 <= 0.05
    x = np.random.default_rng(1).normal(1.0, 0.1, 32)
    t = AP.perm_median_test(np.full(8, np.median(x)), x, nperm=2000)
    assert t["p"] == 1.0 and t["d"] == 0.0 and t["backend_median_null95"][0] < np.median(x) < t["backend_median_null95"][1]
    assert t["mds_abs"]["0.01"] > t["mds_abs"]["0.05"] > 0 and t["n"] == 8 and t["n_reference"] == 32
    t = AP.perm_median_test(np.full(8, 2.0), x, nperm=2000)
    assert 1.0 / 2001 <= t["p"] <= 5.0 / 2001          # (1 + #{|d*| >= |d|}) / 2001: almost no relabelling reaches d
    abf = {1: dict(t=1.0), 2: dict(t=2.0), 3: dict(t=math.inf), 4: dict(t=math.inf)}
    fr = {1: dict(t=0.5), 2: dict(t=1.0), 3: dict(t=3.0), 4: dict(t=math.inf)}
    rs = AP.ratio_stats(abf, fr, "t")
    assert rs["n_both_finite"] == 2 and rs["ratio_median"] == 0.5
    assert rs["rank"]["wins"] == 3 and rs["rank"]["ties"] == 1
    ov = AP.overhead_stats({1: dict(w=10.0), 2: dict(w=20.0)}, {1: dict(w=11.0), 2: dict(w=22.0)}, "w")
    assert math.isclose(ov["median"], 0.1)


def test_equivalence_rule_synthetic(monkeypatch):
    import analyze_parallel_benchmark as AP
    rng = np.random.default_rng(7)
    pa, pf = rng.normal(1.0, 0.1, 32), rng.normal(0.6, 0.06, 32)
    prod = dict(path="synthetic", N=512, Ibar_F={"abf": dict(enumerate(pa)), "fr": dict(enumerate(pf))},
                G_Ibar_F=dict(enumerate((pf - pa) / pa)), G_Ibar_F_median=float(np.median((pf - pa) / pa)),
                G_Ibar_F_ci95=[-0.45, -0.35], tau_t={})
    monkeypatch.setattr(AP, "production_summary", lambda system: (prod, "synthetic"))

    def runs(shift):
        a = rng.normal(1.0, 0.1, 8)
        f = rng.normal(0.6, 0.06, 8) * shift
        return {"abf": {s: dict(Ibar_F=float(a[s]), budget_fraction=1.0) for s in range(8)},
                "fr": {s: dict(Ibar_F=float(f[s]), budget_fraction=1.0) for s in range(8)}}
    assert AP.equivalence("gateway", "gpu_torch", runs(1.0))["status"] == "CONSISTENT"
    bad = AP.equivalence("gateway", "gpu_torch", runs(1.6))
    assert bad["status"] == "INCONSISTENT" and bad["rejected"]
    short = runs(1.0)
    short["abf"][0]["budget_fraction"] = 0.5
    assert AP.equivalence("gateway", "gpu_torch", short)["status"].startswith("NOT APPLICABLE")
    few = {m: {s: v for s, v in d.items() if s < 4} for m, d in runs(1.6).items()}        # 4 pairs < 8
    e = AP.equivalence("gateway", "gpu_torch", few)
    assert e["status"].startswith("INCOMPLETE (4 complete pairs < 8") and len(e["p_values"]) == 5
    assert set(e["min_detectable_shift"]) == {"abf", "fr", "G_Ibar_F"}
    ok = runs(1.0)
    e = AP.equivalence("gateway", "gpu_torch", ok)
    assert e["production"]["reference_excludes_seeds"] == [] and e["detail"]["abf"]["perm"]["n_reference"] == 32
    for m in ok:                                     # runs reproducing production seeds' x0: those seeds leave the reference
        for v in ok[m].values():
            v["shares_production_init"] = True
    e = AP.equivalence("gateway", "gpu_torch", ok)
    assert e["production"]["reference_excludes_seeds"] == list(range(8))
    assert e["detail"]["abf"]["perm"]["n_reference"] == 24 and e["detail"]["G_Ibar_F"]["perm"]["n_reference"] == 24


def test_equivalence_family_wise_rate_under_h0(monkeypatch):
    """Under H0 (backend and production iid from ONE law, arms correlated as in LTA) the rule declares INCONSISTENT
    at most ~5 % of the time (the first version: 8-12 %).  200 Monte Carlo replicates, fixed seeds, 8 vs 16 seeds."""
    import analyze_parallel_benchmark as AP
    rng = np.random.default_rng(20261010)
    cov = np.array([[1.0, 0.9], [0.9, 1.0]]) * 0.3 ** 2

    def draw(n):
        z = rng.multivariate_normal([0.0, 0.0], cov, n)
        return np.exp(z) * np.array([3.0, 2.4])
    n_inc = 0
    for _ in range(200):
        P_, B_ = draw(16), draw(8)
        prod = dict(path="mc", N=512, Ibar_F={"abf": dict(enumerate(P_[:, 0])), "fr": dict(enumerate(P_[:, 1]))},
                    G_Ibar_F=dict(enumerate((P_[:, 1] - P_[:, 0]) / P_[:, 0])), G_Ibar_F_median=0.0,
                    G_Ibar_F_ci95=[0.0, 0.0], tau_t={})
        monkeypatch.setattr(AP, "production_summary", lambda system, prod=prod: (prod, "mc"))
        runs = {"abf": {100 + s: dict(Ibar_F=float(B_[s, 0]), budget_fraction=1.0) for s in range(8)},
                "fr": {100 + s: dict(Ibar_F=float(B_[s, 1]), budget_fraction=1.0) for s in range(8)}}
        n_inc += AP.equivalence("lta300", "gpu_torch", runs, nperm=1000)["status"] == "INCONSISTENT"
    assert n_inc / 200 <= 0.07, n_inc


def test_w1_rule_uses_uncontended_pairs():
    import analyze_parallel_benchmark as AP
    st = {0: False, 1: False, 2: False, 3: False, 4: False, 5: True, 6: None, 7: False}
    abf = {s: dict(tau_wall=100.0, smt_contended=st[s], smt_state=str(st[s])) for s in range(8)}
    fr = {s: dict(tau_wall=50.0, smt_contended=False, smt_state="False") for s in range(8)}
    fr[5]["tau_wall"] = fr[6]["tau_wall"] = 400.0                   # the contended / unknown pairs look very different
    w = AP.w1_stats(abf, fr)
    assert w["status"] == "EVALUATED" and w["n_pairs_uncontended"] == 6 and w["n_pairs_excluded"] == 2
    assert w["primary"]["ratio_median"] == 0.5 and w["primary"]["n_both_finite"] == 6
    assert {e["seed"] for e in w["excluded"]} == {5, 6} and w["sensitivity_all_pairs"]["n_both_finite"] == 8
    assert w["agree"] is True                                        # the all-pairs median 0.5 is inside the primary CI
    abf2 = {s: dict(v, smt_contended=None) for s, v in abf.items()}
    w = AP.w1_stats(abf2, fr)
    assert w["status"].startswith("NOT EVALUABLE") and w["primary"] is None and w["n_pairs_excluded"] == 8


# ------------------------------------------------------------------------------------------------ B8
def test_campaign_job_list_and_dry_run(tmp_path):
    J = PB.job_list("cpu_numba_1core", ["gateway", "lta300"], 8)
    assert len(J) == 32
    gw = [j for j in J if j[0] == "gateway"]
    assert [j[3] for j in gw[::2]] == PROD_GW["seeds"][:8]
    assert [j[2] for j in gw[:4]] == ["abf", "fr", "fr", "abf"]          # alternating arm order
    todo = PB.campaign("gpu_torch", ["lta150"], 8, [64], out_root=str(tmp_path), dry_run=True, log=lambda *a: None)
    assert len(todo) == 16 and all(t[4] == "absent" for t in todo)


def _fake_complete(root, system, backend, seed, method, process_wall_s, us_per_step, wall_total_s):
    npz, js = PB.out_paths(str(root), system, backend, seed, method)
    os.makedirs(os.path.dirname(js), exist_ok=True)
    json.dump(dict(status="complete", system=system, backend=backend, method=method, seed=seed,
                   process_wall_s=process_wall_s, scalars=dict(us_per_step=us_per_step, wall_total_s=wall_total_s)),
              open(js, "w"))


def test_campaign_resource_guard_and_contended_refusal(tmp_path, monkeypatch):
    """The campaign never launches a job whose projected cost (with its not-yet-run partner arm) would bring the
    backend's allocated device-hours above --max-device-hours, and never launches on a contended core without
    --allow-contended; a GPU campaign refuses up front on a disallowed / unpinned device."""
    launched = []

    class FakeProc:                                       # "runs" instantly and leaves a complete result behind
        returncode = 0

        def __init__(self, cmd):
            launched.append(cmd)
            a = dict(zip(cmd[::1], cmd[1:] + [None]))
            system, method, seed = a["--system"], a["--method"], int(a["--seed"])
            us = PB.FALLBACK_US_PER_STEP[("cpu_numba_1core", system)]
            nst = PB.full_n_steps(*PB.system_spec(system)[:2])
            _fake_complete(a["--out-root"], system, "cpu_numba_1core", seed, method,
                           us * 1e-6 * nst + PB.RUN_OVERHEAD_FALLBACK_S, us, us * 1e-6 * nst)

        def poll(self):
            return 0
    monkeypatch.setattr(PB.subprocess, "Popen", FakeProc)
    monkeypatch.setattr(PB.time, "sleep", lambda s: None)
    quiet = dict(log=lambda *a: None)
    # fallback cost of a full gateway numba run: 66 us x 6.4e6 + 20 s = 442 s; a pair 0.246 h
    PB.campaign("cpu_numba_1core", ["gateway"], 8, [64], out_root=str(tmp_path), idle_check=False,
                max_device_hours=0.2, **quiet)
    assert launched == []
    PB.campaign("cpu_numba_1core", ["gateway"], 8, [64], out_root=str(tmp_path), idle_check=False,
                max_device_hours=0.25, **quiet)
    assert len(launched) == 2                              # one pair fits (0.246 h), the second job of pair 2 never starts
    assert [c[c.index("--seed") + 1] for c in launched] == ["8100", "8100"]
    launched.clear()
    rb = tmp_path / "b"
    _fake_complete(rb, "gateway", "cpu_numba_1core", 8100, "abf", 3000.0, 50.0, 2990.0)   # 0.83 h used
    ok, info = PB.budget_allows(str(rb), ("gateway", "cpu_numba_1core", "fr", 8100), False, 0.0, 1.0,
                                lambda s: 6_400_000)
    assert ok and info["source"] == "fallback (FALLBACK_US_PER_STEP)" and math.isclose(info["used_h"], 3000 / 3600)
    ok, info = PB.budget_allows(str(rb), ("gateway", "cpu_numba_1core", "abf", 8101), True, 0.0, 1.0,
                                lambda s: 6_400_000)
    assert not ok and info["source"].startswith("measured")          # 0.83 + 2 x (320 + 10) s > 1.0 h
    # contended cores: no launch without --allow-contended
    monkeypatch.setattr(PB, "idle_cpus", lambda free, sample_s=1.0, threshold=0.10: ([], {c: 1.0 for c in range(512)}))
    PB.campaign("cpu_numba_1core", ["lta300"], 1, [64, 65], out_root=str(tmp_path / "c"), max_wait_s=0.0,
                max_device_hours=100.0, **quiet)
    assert launched == []
    PB.campaign("cpu_numba_1core", ["lta300"], 1, [64, 65], out_root=str(tmp_path / "c"), max_wait_s=0.0,
                max_device_hours=100.0, allow_contended=True, **quiet)
    assert len(launched) == 2
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    with pytest.raises(PB.BenchError, match="not an allowed"):
        PB.campaign("gpu_torch", ["lta300"], 1, [64], out_root=str(tmp_path / "g"), **quiet)
    assert PB.default_max_hours("gpu_torch") == 8.0 and PB.default_max_hours("cpu_numba_1core") == 20.0


# ------------------------------------------------------------------------------------------------ B9
_CUDA_SCRIPT = r"""
import sys, json, numpy as np, torch
sys.path.insert(0, sys.argv[1] + '/src'); sys.path.insert(0, sys.argv[1] + '/scripts/mechanism')
import bench_torch_engines as BT
P = json.load(open(sys.argv[1] + '/configs/equal_budget_v2/gateway_production.json'))
L = json.load(open(sys.argv[1] + '/configs/equal_budget_v2/lta_300K_production.json'))
dev = torch.device('cuda:0')
out = {}
sv = np.array([160, 320, 480])
r = BT.GatewayTorchRun(P['engine_cfg'], 'fr', 512, 8100, 480, sv, dev).run()
out['gw'] = [bool(np.array_equal(r['C_all'].sum(1), 512 * sv)), bool(np.all(np.diff(r['wall_s_at_save']) >= 0)),
             bool(np.isfinite(r['fr_time_s_at_save'][-1]) and r['fr_time_s_at_save'][-1] <= r['wall_total_s'])]
c = dict(L['engine_cfg'], fr_start_steps=10, warmup_steps=10, burn_in_steps=10)
sv = np.array([0, 20, 40])
r = BT.LTATorchRun(c, 300.0, 'fr', 512, 30000, 40, sv, dev).run()
out['lta'] = [bool(np.array_equal(r['C_all'].sum(1), 512 * (sv + 1))), bool(np.all(np.diff(r['wall_s_at_save']) >= 0)),
              bool(np.isfinite(r['fr_time_s_at_save'][-1])), bool(np.all(np.isfinite(r['q_final'])))]
print(json.dumps(out))
"""


def _single_cuda_device():
    v = (_CVD_AT_START or "").strip()
    return v if (v and "," not in v and v.isdigit()) else None


@pytest.mark.skipif(_single_cuda_device() is None,
                    reason="CUDA test needs CUDA_VISIBLE_DEVICES set to exactly one device at session start")
def test_cuda_drivers_run():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=_single_cuda_device())
    old = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = _single_cuda_device()
    try:
        PB.gpu_guard()                                   # an allowed, idle benchmark GPU only
    except PB.BenchError as e:
        pytest.skip(f"GPU guard: {e}")
    finally:
        if old is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = old
    chk = subprocess.run([sys.executable, "-c", "import torch; print(int(torch.cuda.is_available()))"], env=env,
                         capture_output=True, text=True)
    if chk.stdout.strip() != "1":
        pytest.skip("no usable CUDA device")
    p = subprocess.run([sys.executable, "-c", _CUDA_SCRIPT, ROOT], env=env, capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stderr[-2000:]
    out = json.loads(p.stdout.strip().splitlines()[-1])
    assert all(out["gw"]) and all(out["lta"]), out


# ------------------------------------------------------------------------------------------------ B10
def test_gateway_fr_operator_torch_equals_numba_in_law(torch_cpu):
    """The torch engine's FR operator (eb.binned_density + uniform score + gateway_core.resample_indices) and the
    production numba operator (kde_density + uniform_scores + fr_resample with the PCG64 block): density and score
    equal to 1e-14 on the same walkers; the resampling law (capped candidate counts kd, kc; offspring count and
    zero-offspring probability per walker) equal within Bonferroni z < 4 over 20 000 draws, cap binding and not."""
    torch = torch_cpu
    import gateway_core as gc
    import eb_abffr_core as eb
    import gateway_ladder_numba as GE
    N = 32
    rng = np.random.default_rng(3)
    X = np.sort(rng.normal(-1.0, 0.3, N))
    X[:4] = rng.uniform(-0.2, 0.2, 4)
    dx = GE.grid_dx()
    kern, r = GE.gaussian_kernel_np(0.1, dx)
    p, hist, S = np.zeros(181), np.zeros(181), np.zeros(N)
    GE.kde_density(X, N, kern, r, dx, p, hist)
    GE.uniform_scores(X, N, p, dx, 3.0, S)
    cpu = torch.device("cpu")
    xg, dxt, _, _ = eb.build_grid(cpu, torch.float64)
    k, rr = eb.gaussian_kernel(0.1, dxt, cpu, torch.float64)
    Xt = torch.as_tensor(X)[None]
    pt = eb.binned_density(Xt, k, rr, dxt)
    qu = torch.ones((1, 181), dtype=torch.float64)
    q = qu / torch.clamp(eb.trapz(qu, dxt), min=eb.EPS)
    kl = eb.trapz(pt * (torch.log(torch.clamp(pt, min=eb.EPS)) - torch.log(torch.clamp(q, min=eb.EPS))), dxt).unsqueeze(1)
    St = torch.clamp(torch.log(torch.clamp(eb.interp1d(Xt, pt, dxt), min=eb.EPS))
                     - torch.log(torch.clamp(eb.interp1d(Xt, q, dxt), min=eb.EPS)) - kl, -3.0, 3.0)
    assert np.abs(pt[0].numpy() - p).max() < 1e-14 and np.abs(St[0].numpy() - S).max() < 1e-14
    g, dt_fr, R = 40.0, 0.004, 20000

    def z(a, b):
        return (a.mean(0) - b.mean(0)) / np.sqrt(a.var(0) / len(a) + b.var(0) / len(b) + 1e-30)
    for cap in (2, 30):
        gen = torch.Generator()
        gen.manual_seed(1)
        sel, die, clone = gc.resample_indices(torch.as_tensor(S)[None].repeat(R, 1), torch.ones(R, dtype=torch.bool),
                                              torch.zeros(R, dtype=torch.bool), torch.arange(R),
                                              torch.full((R, 1), g, dtype=torch.float64), dt_fr,
                                              torch.full((R, 1), cap), gen)
        sl = sel.numpy()
        off_t = np.zeros((R, N))
        np.add.at(off_t, (np.repeat(np.arange(R), N), sl.ravel()), 1.0)
        kd_t, kc_t = die.sum(1).numpy().astype(float), clone.sum(1).numpy().astype(float)
        off_n, kd_n, kc_n = np.zeros((R, N)), np.zeros(R), np.zeros(R)
        sel2, dc, cc, pool = (np.empty(N, np.int64), np.empty(N, np.int64), np.empty(N, np.int64),
                              np.empty(2 * N, np.int64))
        ur = np.random.default_rng(2)
        for i in range(R):
            a, b, _ = GE.fr_resample(S, N, g, dt_fr, cap, sel2, dc, cc, pool, ur.random(N + cap + max(N, cap)), 0, False)
            kd_n[i], kc_n[i] = a, b
            off_n[i] = np.bincount(sel2, minlength=N) if a + b > 0 else 1.0
        assert abs(z(kd_t[:, None], kd_n[:, None])[0]) < 4 and abs(z(kc_t[:, None], kc_n[:, None])[0]) < 4
        assert np.nanmax(np.abs(z(off_t, off_n))) < 4
        assert np.nanmax(np.abs(z((off_t == 0).astype(float), (off_n == 0).astype(float)))) < 4
        if cap == 2:
            assert np.mean(kd_t + kc_t == 2) > 0.2          # the cap binds in this case


# ------------------------------------------------------------------------------------------------ B11
def test_numba_run_one_equals_plain_run_arm(tiny_root):
    """The warm-up run and the timing hooks of run_one leave the timed numba run bitwise equal to a plain run_arm of
    the same (seed, n_steps, save grid) -- in this process, after other runs (the MT stream is re-seeded per run)."""
    import gateway_ladder_numba as GE
    import lta_ladder_numba as LE
    sv = PB.save_grid(PROD_GW, 3200, "gateway")
    for seed in (8100, 8101):
        for m in PB.METHODS:
            r = PB.load_run(PB.out_paths(tiny_root, "gateway", "cpu_numba_1core", seed, m)[0])
            a = GE.run_arm(dict(PROD_GW["engine_cfg"]), m, 512, seed, 3200, sv, diagnostics=False)
            for k in ("M_all", "C_all", "fr_kd_cum", "fr_kc_cum", "X_final", "Y_final"):
                assert np.array_equal(np.asarray(a[k]), r[k]), (seed, m, k)
    sv = PB.save_grid(PROD_L3, 400, "lta")
    for m in PB.METHODS:
        r = PB.load_run(PB.out_paths(tiny_root, "lta300", "cpu_numba_1core", 30000, m)[0])
        c = LE.make_cfg(300.0, 512, 30000, **dict(PROD_L3["engine_cfg"], n_steps=400, diagnostics=False))
        a = LE.run_arm(c, m, sv)
        for k in ("M_all", "C_all", "M_prod", "C_prod", "q_final"):
            assert np.array_equal(np.asarray(a[k]), r[k]), (m, k)


# ------------------------------------------------------------------------------------------------ B12
def test_single_backend_and_equivalence_figures_legible(tiny_root, tmp_path, monkeypatch):
    """Layouts that the tiny end-to-end analysis does not reach: one backend only (narrow error-vs-wall figure) and
    the equivalence figure (needs full-budget torch runs; synthetic data here).  Every figure passes the legibility
    check (no text cut off at the canvas edge, no overlaps, labelled axes, keys)."""
    import shutil
    import analyze_parallel_benchmark as AP
    root = tmp_path / "one"
    shutil.copytree(os.path.join(tiny_root, "gateway", "cpu_numba_1core"), root / "gateway" / "cpu_numba_1core")
    S, man = AP.analyze(str(root), str(tmp_path / "an1"), str(tmp_path / "fig1"))
    assert set(man) == {"T_time_to_accuracy", "C_cost", "X_error_vs_wall"}
    for tag, v in man.items():
        assert v["legibility"]["status"] == "pass", (tag, v["legibility"]["failures"][:3])
    rng = np.random.default_rng(11)
    pa, pf = rng.normal(0.0083, 0.0008, 32), rng.normal(0.0050, 0.0005, 32)
    prod = dict(path="synthetic", N=512, Ibar_F={"abf": dict(enumerate(pa)), "fr": dict(enumerate(pf))},
                G_Ibar_F=dict(enumerate((pf - pa) / pa)), G_Ibar_F_median=float(np.median((pf - pa) / pa)),
                G_Ibar_F_ci95=[-0.45, -0.35], tau_t={})
    monkeypatch.setattr(AP, "production_summary", lambda system: (prod, "synthetic"))
    a, f = rng.normal(0.0083, 0.0008, 8), rng.normal(0.0050, 0.0005, 8)
    runs = {"abf": {8100 + s: dict(Ibar_F=float(a[s]), budget_fraction=1.0) for s in range(8)},
            "fr": {8100 + s: dict(Ibar_F=float(f[s]), budget_fraction=1.0) for s in range(8)}}
    eq = AP.equivalence("gateway", "gpu_torch", runs)
    assert eq["status"] in ("CONSISTENT", "INCONSISTENT") and set(eq["holm"]) == {
        "R_median_abf", "R_median_fr", "R_mannwhitney_abf", "R_mannwhitney_fr", "R_median_G"}
    man2 = {}
    AP.setup_style()
    AP.fig_equivalence(["gateway"], {("gateway", "gpu_torch"): runs},
                       {"gateway": {"backends": {"gpu_torch": {"equivalence": eq}}}}, str(tmp_path / "fig2"), man2)
    assert man2["Q_equivalence"]["legibility"]["status"] == "pass", man2["Q_equivalence"]["legibility"]["failures"][:3]


# ------------------------------------------------------------------------------------------------ B13
def test_idle_core_selection(monkeypatch):
    """idle_cpus keeps a CPU only if it AND its SMT siblings were idle; choose_cpu waits, then falls back to the
    least-contended CPU with idle = False."""
    sib = {64: [192], 65: [193], 66: [194]}
    busy = {64: 0.0, 192: 0.9, 65: 0.0, 193: 0.02, 66: 0.5, 194: 0.0}
    state = {"t": 0}

    def fake_times(cpus):
        state["t"] += 1
        return {c: (int(1000 * busy[c] * state["t"]), 1000 * state["t"]) for c in cpus}
    monkeypatch.setattr(PB, "cpu_times", fake_times)
    monkeypatch.setattr(PB, "smt_siblings", lambda cpus: sorted({s for c in cpus for s in sib.get(c, [])} - set(cpus)))
    ok, bf = PB.idle_cpus([64, 65, 66], sample_s=0.0)
    assert ok == [65] and math.isclose(bf[192], 0.9)
    assert PB.choose_cpu([64, 65, 66], max_wait_s=0.0) == (65, True)
    monkeypatch.setattr(PB, "idle_cpus", lambda free, threshold=0.1: ([], {64: 0.0, 192: 0.9, 66: 0.5, 194: 0.0}))
    monkeypatch.setattr(PB.time, "sleep", lambda s: None)
    assert PB.choose_cpu([64, 66], max_wait_s=0.0, log=lambda *a: None) == (None, False)    # no contended launch
    c, idle = PB.choose_cpu([64, 66], max_wait_s=0.0, log=lambda *a: None, allow_contended=True)
    assert (c, idle) == (66, False)                                     # 0.5 + 0.0 < 0.0 + 0.9
    assert PB.choose_cpu([64, 66], idle_check=False) == (64, True)


# ------------------------------------------------------------------------------------------------ B14
def test_smt_state_gpu_monitor_and_matched_ratios(monkeypatch):
    assert PB.smt_state(list(range(4)), [], {})[0] == "unknown"                 # not pinned
    assert PB.smt_state([64], [], {"64": 1.0}) == ("uncontended", 0.0)           # no SMT sibling
    assert PB.smt_state([64], [192], {"64": 1.0, "192": 0.5}) == ("contended", 0.5)
    assert PB.smt_state([64], [192], {"64": 1.0, "192": 0.05}) == ("uncontended", 0.05)
    assert PB.smt_state([64], [192], {"64": 1.0})[0] == "unknown"               # sibling telemetry missing
    assert PB.parse_mib("624 MiB") == 624.0 and PB.parse_mib("2 GiB") == 2048.0 and math.isnan(PB.parse_mib("x"))
    me = os.getpid()
    seq = [dict(processes={2: [dict(pid=me, used_memory="600 MiB")]}, names={2: "H200"}),
           dict(processes={2: [dict(pid=me, used_memory="640 MiB"), dict(pid=7, used_memory="9 MiB")]},
                names={2: "H200"}),
           None]
    monkeypatch.setattr(PB, "gpu_processes", lambda: seq.pop(0))
    mon = PB.GPUMonitor(2, poll_s=10.0)
    mon.sample(0.0)
    mon.sample(5.0)                                       # within poll_s: skipped
    mon.sample(12.0)
    sm = mon.summary()
    assert sm["n_samples"] == 2 and sm["peak_device_mb"] == 640.0 and sm["occupancy"] == "shared"
    assert [p["pid"] for p in sm["other_processes"]] == [7]
    mon.sample(math.inf, force=True)                      # nvidia-smi failing after the run
    sm = mon.summary()
    assert sm["occupancy"] == "unknown" and sm["n_failed"] == 1 and sm["samples"][-1]["wall_s"] is None
    import bench_step_costs as BS
    g = [dict(impl="lean", device="cuda:0", system="gateway", method="abf", us_per_step=570.0, smt_state="contended"),
         dict(impl="lean", device="cuda:0", system="lta300", method="abf", us_per_step=900.0, smt_contended=False,
              smt_sibling_busy_max=None)]
    c = [dict(impl="numba", device="cpu", system="gateway", method="abf", us_per_step=65.0, smt_state="contended",
              diagnostics=False),
         dict(impl="numba", device="cpu", system="lta300", method="abf", us_per_step=600.0, smt_state="contended",
              diagnostics=False)]
    r = BS.matched_ratios(g, c)
    assert math.isclose(r["gateway/abf"]["ratio_gpu_over_cpu"], 570.0 / 65.0)
    assert "ratio_gpu_over_cpu" not in r["lta300/abf"] and r["lta300/abf"]["status"].startswith("UNMATCHED")


def test_gateway_torch_arms_unpaired_after_first_opportunity(torch_cpu):
    """gateway_core.resample_indices permutes all N slots at every FR opportunity, also without events (step 0: the
    ramp gives g = 0): after one step the FR arm holds the ABF arm's walkers in other slots, after two the shared
    normals have driven different walkers (law unaffected; documented in bench_torch_engines)."""
    import bench_torch_engines as BT
    cfg = dict(PROD_GW["engine_cfg"])
    r = {}
    for n in (1, 2):
        for m in PB.METHODS:
            r[(n, m)] = BT.GatewayTorchRun(cfg, m, 64, 8100, n, np.array([n]), "cpu").run()
    assert r[(1, "fr")]["fr_kd_cum"][-1] == 0 and r[(1, "fr")]["fr_kc_cum"][-1] == 0      # no event fired
    xa, xf = r[(1, "abf")]["X_final"], r[(1, "fr")]["X_final"]
    assert np.array_equal(np.sort(xa), np.sort(xf)) and not np.array_equal(xa, xf)
    assert not np.array_equal(np.sort(r[(2, "abf")]["X_final"]), np.sort(r[(2, "fr")]["X_final"]))


def test_early_window_consistency_plumbing(tmp_path):
    """EXPLORATORY early-window check: a shortened numba run of a production seed is a bitwise prefix of the
    production run, so its Ebar_common (mean e_F over the saves common to both grids) must EQUAL the production value
    of the same seed -- this pins the step alignment of the backend curves and the production rows.  With one pair
    the status is INCOMPLETE (no verdict)."""
    import analyze_parallel_benchmark as AP
    if not os.path.exists(PB.production_path("gateway", 8100, "abf")):
        pytest.skip("production files absent")
    root = str(tmp_path)
    rows, curves = {}, {}
    with one_cpu():
        for m in PB.METHODS:
            PB.run_one("gateway", "cpu_numba_1core", m, 8100, out_root=root, n_steps=64000)
            row, cv = AP.score_run(PB.out_paths(root, "gateway", "cpu_numba_1core", 8100, m)[0], "gateway")
            rows.setdefault(m, {})[8100] = row
            curves.setdefault(m, {})[8100] = cv
    ew = AP.early_window_consistency("gateway", "cpu_numba_1core", rows, curves, nperm=500)
    assert ew["common_steps"] == [640, 20000, 32000, 40000, 64000] and ew["reference_excludes_seeds"] == []
    ref = AP.production_early_values("gateway", np.array(ew["common_steps"]))
    for m in PB.METHODS:
        assert ew["backend_values"][m][8100] == ref[m][8100]
    assert ew["status"].startswith("EXPLORATORY INCOMPLETE (1 complete pairs < 8")
    for v in rows["abf"].values():
        v["shares_production_init"] = True
    ew = AP.early_window_consistency("gateway", "cpu_numba_1core", rows, curves, nperm=500)
    assert ew["reference_excludes_seeds"] == [8100] and ew["detail"]["abf"]["perm"]["n_reference"] == 31
    rows["abf"][8100]["budget_fraction"] = 1.0
    assert AP.early_window_consistency("gateway", "x", rows, curves)["status"].startswith("NOT APPLICABLE")


def test_gpu_campaign_loop_guarded(tmp_path, monkeypatch):
    """The sequential GPU campaign loop (guard, idle core and run mocked: nothing touches a GPU) stops at the
    device-hour budget at a pair boundary."""
    calls = []

    def fake_run(cmd):
        calls.append(cmd)
        a = dict(zip(cmd, cmd[1:] + [None]))
        us = PB.FALLBACK_US_PER_STEP[("gpu_torch", a["--system"])]
        nst = PB.full_n_steps(*PB.system_spec(a["--system"])[:2])
        _fake_complete(a["--out-root"], a["--system"], "gpu_torch", int(a["--seed"]), a["--method"],
                       us * 1e-6 * nst + PB.RUN_OVERHEAD_FALLBACK_S, us, us * 1e-6 * nst)
        return subprocess.CompletedProcess(cmd, 0)
    monkeypatch.setattr(PB, "gpu_guard", lambda allow_busy=False, pid=None: (2, []))
    monkeypatch.setattr(PB.subprocess, "run", fake_run)
    quiet = dict(log=lambda *a: None)
    # fallback LTA GPU run: 1150 us x 6e5 + 20 s = 710 s; a pair 0.394 h
    PB.campaign("gpu_torch", ["lta300"], 8, [64], out_root=str(tmp_path), idle_check=False, max_device_hours=0.39,
                **quiet)
    assert calls == []
    PB.campaign("gpu_torch", ["lta300"], 8, [64], out_root=str(tmp_path), idle_check=False, max_device_hours=0.6,
                **quiet)
    assert [(c[c.index("--seed") + 1], c[c.index("--method") + 1]) for c in calls] == [("30000", "abf"),
                                                                                       ("30000", "fr")]
    assert all(c[:3] == ["taskset", "-c", "64"] for c in calls)


def test_driver_on_save_hook_excluded_from_clock(torch_cpu):
    """Monitoring at saves (nvidia-smi on the GPU) is excluded from every stamp and from wall_total; on_start /
    on_end bracket the timed run."""
    import time as _t
    import bench_torch_engines as BT
    ev = []
    r = BT.GatewayTorchRun(dict(PROD_GW["engine_cfg"]), "fr", 64, 8100, 320, np.array([160, 320]), "cpu").run(
        on_start=lambda: ev.append("start"), on_save=lambda w: (ev.append(w), _t.sleep(0.5)),
        on_end=lambda: ev.append("end"))
    assert ev[0] == "start" and ev[-1] == "end" and len(ev) == 4
    assert r["monitor_s"] >= 1.0 and r["wall_total_s"] < r["monitor_s"] and r["wall_s_at_save"][-1] < 0.5
    c = dict(PROD_L3["engine_cfg"], fr_start_steps=5, warmup_steps=5, burn_in_steps=5)
    r = BT.LTATorchRun(c, 300.0, "fr", 64, 30000, 10, np.array([0, 10]), "cpu").run(on_save=lambda w: _t.sleep(0.5))
    assert r["monitor_s"] >= 1.0 and r["wall_s_at_save"][0] < 0.5
