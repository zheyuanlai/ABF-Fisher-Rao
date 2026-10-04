"""Equivalence of the CPU (numba) WCA port (src/wca_numba.py) with the accepted torch engine.

V0  configuration, constants, event cap law, grid, initial condition, seed-label mapping, save grid;
V1  ABF dynamics + P0 histogram estimator + accepted read-out:
    V1a BITWISE equal to the accepted pipeline (``wca_phase_jobs.execute_run`` on a float64 CPU
        engine) when both are driven by the same torch noise stream; the torch side runs in a
        subprocess with ATEN_CPU_CAPABILITY=default (scalar libm pow, plain norm); the stress
        variant makes every clamp fire (RC wall both sides, force clip, ABF clip, mean-force
        sample clip, unclipped-force mean force) and is still bitwise;
    V1e the FR arm end to end, BITWISE equal to torch's fr_uniform with the law made deterministic
        (N = 2, fr_rate 1e18), the torch global stream replayed from its event record: post-move
        scoring, copies, run-long and windowed genealogy, crossings with side inheritance;
    V1b in-process (AVX512 kernels: Sleef pow, fma norm) equal to round-off over 300 steps;
    V1c the V1b residual is the dynamics' own amplification of 1 ulp (numba vs numba);
    V1d arm independence: an arm's trajectory does not depend on the other arms; an FR arm with no
        possible event is bitwise the ABF arm;
    V1f one FR random stream per FR arm: an FR arm does not depend on the other FR arms either;
V2  FR pieces (KDE marginal, uniform target, interpolation, score, recentred clip, death
    probabilities) equal to the torch functions to round-off;
    V2e the FR rate plumbing INSIDE simulate (fr_rate, dt_eff = dt * fr_every, KDE bandwidth,
        score clip, cap) at a finite rate: the first event of an FR arm is bitwise the event the
        torch score / death-probability code prescribes for the same state and uniforms;
V3  the birth-death law: ``fr_select`` against the EXACT law (Poisson-binomial count law, per-walker
    death probability p_i E[min(1, cap / (1 + M_-i))], iid sources ~ S-, P(s1 == s2 | k >= 2) =
    sum pi^2) with powered statistics, shown to REJECT four plausible wrong laws, at N = 4 ... 1024
    including the cap-binding regime at N = 1024; ``fixed_population_birth_death_torch`` against the
    same exact law; cap 0 (no event, no RNG draw);
V4  genealogy / bookkeeping invariants of the FR arm; ess_window_steps <= 0 behaves as torch's
    max(., 1); an ABF arm reports n_unique_ancestor = N like the torch engine;
V5  the offline scorer reproduces the accepted run's l2 numbers from its saved accumulators;
    V5b documents the float32 count saturation (C = 2^24) of the boundary bin in all 32 accepted
    histogram runs; V5c the opt-in float32 accumulator emulation;
V6  the result carries its cfg and the ladder scorer reads it (times, domain, budget-axis
    integral, refusal of a mismatched sim or of the accepted reference for other physics),
    save_result / load_result round trip; the CPU guard hides CUDA before torch is imported.

Run on CPU:  CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_wca_numba.py -q
"""
import dataclasses
import glob
import json
import math
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import torch
from numba import njit

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import wca_abffr_core as core      # noqa: E402
import wca_numba as wn             # noqa: E402

CPU = torch.device("cpu")
F64 = torch.float64
CFG = wn.ACCEPTED_CFG
P = CFG["n_dim"] ** 2
CONF_RAW = os.path.join(ROOT, "results", "histogram_abf", "wca", "confirmation", "raw")
assert core.DEVICE.type == "cpu", "run with CUDA_VISIBLE_DEVICES=''"


@pytest.fixture(scope="module")
def accepted():
    return wn.accepted_setup()


def _accepted_file(arm, seed):
    f = sorted(glob.glob(os.path.join(CONF_RAW, f"confirmation__{arm}__*__seed{seed}__N1024__T120000__hist160__*.npz")))
    assert len(f) == 1, (arm, seed, f)
    return f[0]


# ------------------------------------------------------------------------------------------- V0
def test_v0_accepted_cfg_and_seed_mapping(accepted):
    sim, params = accepted
    assert wn.cfg_from_setup(sim, params) == wn.ACCEPTED_CFG
    sim_a, params_a = wn.accepted_setup(method="abf")
    assert wn.cfg_from_setup(sim_a, params_a) == wn.ACCEPTED_CFG
    assert sim.abf_estimator == "histogram" and sim.abf_n_bins == 160 and wn.EPS == core.EPS
    # seed label -> engine seed is the identity (build_sim), and the accepted files carry the label
    for sd in (3100, 3107, 3115):
        s, _ = wn.accepted_setup(seed=sd)
        assert s.seed == sd
        for arm in ("hist_abf", "hist_fr_uniform"):
            spec = json.loads(str(np.load(_accepted_file(arm, sd), allow_pickle=True)["spec_json"]))
            assert spec["seed"] == sd and spec["abf_n_bins"] == 160 and spec["n_replicas"] == 1024


def test_v0_event_cap_law():
    Ns = (1, 2, 4, 16, 64, 256, 1024)
    torch_law = [int(CFG["max_event_fraction"] * N) for N in Ns]       # fixed_population_birth_death_torch :1109
    assert [wn.event_cap(N, CFG, 0) for N in Ns] == torch_law == [0, 0, 0, 0, 1, 5, 20]
    assert [wn.event_cap(N, CFG, 1) for N in Ns] == [max(1, c) for c in torch_law]
    # the torch claim: cap < 1 -> no event AND no draw from the global stream
    sim, _ = wn.accepted_setup()
    q = torch.arange(16, dtype=F64).view(16, 1, 1)
    score = torch.linspace(-2.0, 2.0, 16, dtype=F64)
    torch.manual_seed(5)
    st = torch.get_rng_state()
    q2, _, stats = core.fixed_population_birth_death_torch(q, score, sim, fr_interval=5, fr_rate_override=1e6)
    assert stats["replacement"] == 0 and torch.equal(q2, q) and torch.equal(torch.get_rng_state(), st)


def test_v0_grid_init_saves(accepted):
    sim, params = accepted
    g_t = torch.linspace(sim.z_min, sim.z_max, sim.n_grid, dtype=F64).numpy()
    assert np.array_equal(wn.fr_grid(), g_t)
    assert np.allclose(g_t, np.linspace(sim.z_min, sim.z_max, sim.n_grid), rtol=0, atol=1e-15)
    for sd, N in ((3100, 1024), (3115, 4), (7, 1)):
        q_t = core.lattice_initial_conditions(params, N, CPU, torch.float32, seed=sd).numpy().astype(np.float64)
        q_n = wn.lattice_init(sd, N)
        assert q_n.dtype == np.float64 and np.array_equal(q_n, q_t)          # the engine dtype (float32) init
        assert np.array_equal(wn.lattice_init(sd, N, dtype="float64"),
                              core.lattice_initial_conditions(params, N, CPU, F64, seed=sd).numpy())
        z = np.array([wn.dimer_geometry(q_n[i], CFG["n_dim"] * CFG["a"], wn._r0(CFG), CFG["w"], 1)[3] for i in range(N)])
        assert np.all(np.abs(z) < 1e-6)                                       # dimer starts at r0 (z = 0)
    # the shared budget grid is the gateway port's
    gn = pytest.importorskip("gateway_numba")
    for n in (120000, 480000, 7680000, 122880000):
        a, b = wn.budget_save_grid(n), gn.budget_save_grid(n)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


# ------------------------------------------------------------------------------------------- V1
_V1A_SCRIPT = textwrap.dedent("""
    import dataclasses, sys, numpy as np, torch
    sys.path.insert(0, {src!r})
    import wca_abffr_core as core, wca_phase_jobs as jobs, wca_numba as wn
    assert torch.backends.cpu.get_cpu_capability() == "DEFAULT"
    torch.set_num_threads(1)
    import importlib.util
    sp_ = importlib.util.spec_from_file_location("r", {runner!r}); r = importlib.util.module_from_spec(sp_); sp_.loader.exec_module(r)
    c, base, fr = r.load_frozen()
    base = dict(base, abf_warmup_steps={warm}, estimator_burn_in_steps={burn})
    c = dict(c, run=dict(c["run"], n_steps={n_steps}, n_replicas={N}, save_every={save_every}))
    spec = r.make_spec("v1a", "hist_abf", "abf", {seed}, c, fr, "histogram", 160)
    params = jobs.build_params(spec)
    eng = core.WCADimerEngine(params, torch.device("cpu"), torch.float64)
    out = jobs.execute_run(spec, base, eng, cache_dir={cache!r}, store_profiles=True)
    q0 = core.lattice_initial_conditions(params, {N}, torch.device("cpu"), torch.float64, seed={seed}).numpy()
    torch.manual_seed({seed})
    noise = np.stack([torch.randn(({N}, {P}, 2), dtype=torch.float64).numpy() for _ in range({n_steps})])
    keep = ("hist_counts_t", "hist_mf_bins_t", "profile_steps", "l2_f", "l2_f_t", "integrated_l2_f", "l2_fp", "l2_fp_t",
            "l2_fp_nodes", "l2_fp_nodes_t", "bias_absmax", "bias_clip_fraction", "n_compact_to_stretched",
            "n_stretched_to_compact", "n_round_trips", "frac_compact", "frac_transition", "frac_stretched")
    np.savez({path!r}, q0=q0, noise=noise, **{{k: np.asarray(out[k]) for k in keep}})
""")


def test_v1a_abf_bitwise_vs_accepted_pipeline(tmp_path, accepted):
    """The accepted execute_run (dynamics + histogram estimator + read-out) vs numba on the same
    torch noise: every saved accumulator BITWISE equal, the read-out equal to 1e-13."""
    N, n_steps, seed, save_every, warm, burn = 8, 3000, 3100, 250, 1000, 1500
    path = str(tmp_path / "v1a.npz")
    code = _V1A_SCRIPT.format(src=os.path.join(ROOT, "src"), runner=os.path.join(ROOT, "scripts", "run_histogram_abf_wca.py"),
                              cache=os.path.join(ROOT, "cache", "phase_hp_v3"), warm=warm, burn=burn, n_steps=n_steps, N=N,
                              save_every=save_every, seed=seed, P=P, path=path)
    env = dict(os.environ, ATEN_CPU_CAPABILITY="default", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1")
    subprocess.run([sys.executable, "-c", code], check=True, env=env, cwd=ROOT)
    t = np.load(path)
    cfg = dict(CFG, abf_warmup_steps=warm, estimator_burn_in_steps=burn)
    save_at = t["profile_steps"].astype(np.int64)
    assert save_at[0] == 0 and save_at[-1] == n_steps
    res = wn.run_ladder_point(seed, N, n_steps, [("abf", False)], cfg, save_at=save_at, cap_min=0,
                              q0=t["q0"], ext_noise=t["noise"], fma_norm=False)
    C, M = res["C_rep"][0], res["M_rep"][0]
    mf = np.where(C > 0, M / np.where(C > 0, C, 1.0), 0.0)
    assert np.array_equal(C, t["hist_counts_t"])                         # every save, both estimator phases
    assert np.array_equal(mf, t["hist_mf_bins_t"])
    assert res["bias_absmax"][0] == t["bias_absmax"] and res["bias_clip_fraction"][0] == t["bias_clip_fraction"]
    assert res["n_c2s"][0, -1] == t["n_compact_to_stretched"] and res["n_s2c"][0, -1] == t["n_stretched_to_compact"]
    assert res["n_round_trips"][0, -1] == t["n_round_trips"]
    fr_ = np.stack([t["frac_compact"], t["frac_transition"], t["frac_stretched"]], axis=1)
    assert np.allclose(res["frac_regions"][0], fr_, rtol=0, atol=1e-12)
    # the read-out: the scorer on the numba accumulators == execute_run's own numbers
    sim, params = accepted
    sim_s = dataclasses.replace(sim, abf_warmup_steps=warm, estimator_burn_in_steps=burn)
    sc = wn.score_accumulators(M, C, save_at, sim=sim_s, reference=wn.load_reference(sim, params))
    for k in ("l2_f", "integrated_l2_f", "l2_fp", "l2_fp_nodes"):
        assert abs(sc[k] - float(t[k])) <= 1e-13 * abs(float(t[k])), k
    for k in ("l2_f_t", "l2_fp_t", "l2_fp_nodes_t"):
        assert np.allclose(sc[k], t[k], rtol=1e-13, atol=0), k


_V1A_STRESS_SCRIPT = textwrap.dedent("""
    import dataclasses, json, sys, numpy as np, torch
    sys.path.insert(0, {src!r})
    import wca_abffr_core as core, wca_numba as wn
    assert torch.backends.cpu.get_cpu_capability() == "DEFAULT"
    torch.set_num_threads(1)
    sim, params = wn.accepted_setup(method="abf", seed={seed})
    sim = dataclasses.replace(sim, **json.loads({sim_over!r}))
    params = dataclasses.replace(params, **json.loads({par_over!r}))
    eng = core.WCADimerEngine(params, torch.device("cpu"), torch.float64)
    q0 = core.lattice_initial_conditions(params, sim.n_replicas, torch.device("cpu"), torch.float64, seed={seed})
    d = core.run_sampler_gpu("abf", params, sim, eng, initial_q=q0, verbose=False, track_crossings=True, store_snapshots=1)
    torch.manual_seed({seed})
    noise = np.stack([torch.randn((sim.n_replicas, {P}, 2), dtype=torch.float64).numpy() for _ in range(sim.n_steps)])
    np.savez({path!r}, q0=q0.numpy(), noise=noise, steps=d["steps"], C=d["hist_counts"], mf=d["hist_mf_bins"],
             bias_absmax=d["bias_absmax"], clip=d["bias_clip_fraction"], c2s=d["n_compact_to_stretched"],
             s2c=d["n_stretched_to_compact"], rt=d["n_round_trips"], snap_z=d["snap_z"])
""")


def test_v1a_stress_every_clamp_bitwise(tmp_path):
    """Same as V1a with knobs that make every clamp fire (RC wall, physical force clip, ABF clip,
    mean-force sample clip) and the unclipped-force mean-force option: still BITWISE."""
    N, n_steps, seed = 8, 2500, 3104
    sim_over = dict(n_replicas=N, n_steps=n_steps, save_every=250, abf_warmup_steps=300, estimator_burn_in_steps=700,
                    z_min=-0.05, z_max=0.9, abf_force_clip=3.0, mean_force_sample_clip=15.0,
                    use_clipped_force_for_mean_force=False, boundary_wall_strength=30.0)
    par_over = dict(force_clip=25.0)
    path = str(tmp_path / "v1a_stress.npz")
    code = _V1A_STRESS_SCRIPT.format(src=os.path.join(ROOT, "src"), seed=seed, sim_over=json.dumps(sim_over),
                                     par_over=json.dumps(par_over), P=P, path=path)
    env = dict(os.environ, ATEN_CPU_CAPABILITY="default", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1")
    subprocess.run([sys.executable, "-c", code], check=True, env=env, cwd=ROOT)
    t = np.load(path)
    zz = t["snap_z"]
    assert (zz < sim_over["z_min"]).any() and (zz > sim_over["z_max"]).any()          # the wall acted on both sides
    assert t["clip"] > 0.01                                                           # the ABF clip acted
    cfg = dict(CFG, abf_warmup_steps=300, estimator_burn_in_steps=700, z_min=-0.05, z_max=0.9, abf_force_clip=3.0,
               mean_force_sample_clip=15.0, use_clipped_force_for_mean_force=False, boundary_wall_strength=30.0,
               force_clip=25.0)
    save_at = t["steps"].astype(np.int64)
    res = wn.run_ladder_point(seed, N, n_steps, [("abf", False)], cfg, save_at=save_at, cap_min=0,
                              q0=t["q0"], ext_noise=t["noise"], fma_norm=False)
    C, M = res["C_rep"][0], res["M_rep"][0]
    assert np.array_equal(C, t["C"])
    assert np.array_equal(np.where(C > 0, M / np.where(C > 0, C, 1.0), 0.0), t["mf"])
    assert res["bias_absmax"][0] == t["bias_absmax"] and res["bias_clip_fraction"][0] == t["clip"]
    assert (res["n_c2s"][0, -1], res["n_s2c"][0, -1], res["n_round_trips"][0, -1]) == (t["c2s"], t["s2c"], t["rt"])


_V1E_SCRIPT = textwrap.dedent("""
    import dataclasses, json, sys, numpy as np, torch
    sys.path.insert(0, {src!r})
    import wca_abffr_core as core, wca_numba as wn
    assert torch.backends.cpu.get_cpu_capability() == "DEFAULT"
    torch.set_num_threads(1)
    sim, params = wn.accepted_setup(method="fr_uniform", seed={seed})
    sim = dataclasses.replace(sim, **json.loads({sim_over!r}))
    eng = core.WCADimerEngine(params, torch.device("cpu"), torch.float64)
    q0 = core.lattice_initial_conditions(params, sim.n_replicas, torch.device("cpu"), torch.float64, seed={seed})
    d = core.run_sampler_gpu("fr_uniform", params, sim, eng, initial_q=q0, verbose=False, track_crossings=True)
    # replay the torch global stream: per step randn_like(q); at an FR opportunity with an event
    # (with death probability exactly 1 that is: death_mass and birth_mass > EPS) rand(R), then
    # multinomial(w, 1, replacement=True) (R exponential draws; never randperm: n_events <= cap = 1)
    ev = d["fr_event_counts"]
    torch.manual_seed({seed})
    R = sim.n_replicas
    noise, k = [], 0
    for st in range(sim.n_steps):
        noise.append(torch.randn((R, {P}, 2), dtype=torch.float64).numpy())
        nx = st + 1
        if nx >= sim.fr_start_steps and (nx - sim.fr_start_steps) % sim.fr_every == 0:
            if ev[k] > 0:
                torch.rand(R, dtype=torch.float64)
                torch.multinomial(torch.ones(R, dtype=torch.float64), int(ev[k]), replacement=True)
            k += 1
    assert k == len(ev)
    np.savez({path!r}, q0=q0.numpy(), noise=np.stack(noise), steps=d["steps"], C=d["hist_counts"], mf=d["hist_mf_bins"],
             ev=ev, total=d["total_replacement_events"], ess=d["ancestor_ess"], wmax=d["max_ancestor_frac"],
             nuniq=d["n_unique_ancestor"], min_ess_w=d["min_ancestor_ess_window"], c2s=d["n_compact_to_stretched"],
             s2c=d["n_stretched_to_compact"], rt=d["n_round_trips"], frac=np.stack([d["frac_compact"], d["frac_transition"], d["frac_stretched"]], 1))
""")


def test_v1e_fr_wiring_bitwise_deterministic_law(tmp_path):
    """End-to-end FR arm vs torch's fr_uniform with the law made deterministic (N = 2, fr_rate 1e18:
    the positive-score walker dies with probability exactly 1 and the other is the only source):
    post-move scoring, whole-configuration copies, genealogy, windowed ESS and side inheritance are
    BITWISE the torch engine's (the torch noise stream is replayed from its event record)."""
    N, n_steps, seed = 2, 1500, 3105
    sim_over = dict(n_replicas=N, n_steps=n_steps, save_every=100, abf_warmup_steps=200, estimator_burn_in_steps=300,
                    fr_start_steps=200, fr_every=5, fr_rate=1e18, max_event_fraction=0.5, ess_window_steps=100)
    path = str(tmp_path / "v1e.npz")
    code = _V1E_SCRIPT.format(src=os.path.join(ROOT, "src"), seed=seed, sim_over=json.dumps(sim_over), P=P, path=path)
    env = dict(os.environ, ATEN_CPU_CAPABILITY="default", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1")
    subprocess.run([sys.executable, "-c", code], check=True, env=env, cwd=ROOT)
    t = np.load(path)
    assert int(t["total"]) > 50                                                # an event at (almost) every opportunity
    cfg = dict(CFG, abf_warmup_steps=200, estimator_burn_in_steps=300, fr_start_steps=200, fr_every=5, fr_rate=1e18,
               max_event_fraction=0.5, ess_window_steps=100)
    save_at = t["steps"].astype(np.int64)
    res = wn.run_ladder_point(seed, N, n_steps, [("fr", True)], cfg, save_at=save_at, cap_min=0,
                              q0=t["q0"], ext_noise=t["noise"], fma_norm=False)
    C, M = res["C_rep"][0], res["M_rep"][0]
    assert np.array_equal(C, t["C"])
    assert np.array_equal(np.where(C > 0, M / np.where(C > 0, C, 1.0), 0.0), t["mf"])
    assert res["repl_cumulative"][0, -1] == int(t["total"]) and res["n_event_opportunities"][0, -1] == int((t["ev"] > 0).sum())
    assert np.allclose(res["ancestor_ess"][0], t["ess"], rtol=1e-14) and np.array_equal(res["max_ancestor_frac"][0], t["wmax"])
    assert np.array_equal(res["n_unique_ancestor"][0], t["nuniq"])
    assert np.isclose(res["min_ancestor_ess_window"][0], float(t["min_ess_w"]), rtol=1e-14)
    assert (res["n_c2s"][0, -1], res["n_s2c"][0, -1], res["n_round_trips"][0, -1]) == (t["c2s"], t["s2c"], t["rt"])
    assert np.allclose(res["frac_regions"][0], t["frac"], rtol=0, atol=1e-12)


def _torch_abf(N, n_steps, save_every, seed, warm, burn, q0):
    sim, params = wn.accepted_setup(method="abf", seed=seed)
    sim = dataclasses.replace(sim, n_replicas=N, n_steps=n_steps, save_every=save_every,
                              abf_warmup_steps=warm, estimator_burn_in_steps=burn)
    eng = core.WCADimerEngine(params, CPU, F64)
    diag = core.run_sampler_gpu("abf", params, sim, eng, initial_q=torch.as_tensor(q0), verbose=False)
    torch.manual_seed(seed)
    noise = np.stack([torch.randn((N, P, 2), dtype=F64).numpy() for _ in range(n_steps)])
    return diag, noise


def test_v1b_abf_roundoff_in_process():
    """Default (AVX512) torch kernels: only the vectorised Sleef pow differs (<= 1 ulp)."""
    N, n_steps, seed = 16, 300, 3101
    cfg = dict(CFG, abf_warmup_steps=100, estimator_burn_in_steps=150)
    q0 = wn.lattice_init(seed, N, cfg)
    diag, noise = _torch_abf(N, n_steps, 50, seed, 100, 150, q0)
    save_at = np.asarray(diag["steps"], dtype=np.int64)
    res = wn.run_ladder_point(seed, N, n_steps, [("abf", False)], cfg, save_at=save_at, cap_min=0, q0=q0, ext_noise=noise)
    C, M = res["C_rep"][0], res["M_rep"][0]
    mf = np.where(C > 0, M / np.where(C > 0, C, 1.0), 0.0)
    assert np.array_equal(C, diag["hist_counts"])
    mft = diag["hist_mf_bins"]
    assert np.all(np.abs(mf - mft) <= 1e-9 * np.abs(mft).max(axis=1, keepdims=True))


def test_v1c_residual_is_chaotic_amplification():
    """1 ulp in one coordinate, same noise: below 1e-9 at step 300 (the V1b horizon), O(1) later."""
    N, seed = 16, 3101
    cfg = dict(CFG, abf_warmup_steps=100, estimator_burn_in_steps=150)
    q0 = wn.lattice_init(seed, N, cfg)
    q1 = q0.copy()
    q1[3, 17, 0] = np.nextafter(q1[3, 17, 0], np.inf)
    save_at = np.array([300, 3000], dtype=np.int64)
    r0_ = wn.run_ladder_point(seed, N, 3000, [("abf", False)], cfg, save_at=save_at, noise_seed=11, q0=q0)
    r1_ = wn.run_ladder_point(seed, N, 3000, [("abf", False)], cfg, save_at=save_at, noise_seed=11, q0=q1)
    rel = [np.abs(r0_["M_rep"][0, k] - r1_["M_rep"][0, k]).max() / np.abs(r0_["M_rep"][0, k]).max() for k in range(2)]
    assert rel[0] < 1e-9 and rel[1] > 1e-6, rel


def test_v1d_arm_independence_and_fr_noop():
    N, n_steps, seed = 16, 800, 3102
    cfg = dict(CFG, abf_warmup_steps=100, estimator_burn_in_steps=150, fr_start_steps=200, fr_rate=40.0,
               max_event_fraction=0.25, ess_window_steps=100)
    save_at = np.array([0, 150, 199, 200, 400, 800], dtype=np.int64)
    one = wn.run_ladder_point(seed, N, n_steps, [("abf", False)], cfg, save_at=save_at, noise_seed=5)
    two = wn.run_ladder_point(seed, N, n_steps, [("fr", True), ("abf", False)], cfg, save_at=save_at, noise_seed=5)
    assert np.array_equal(one["M_bias"][0], two["M_bias"][1]) and np.array_equal(one["C_bias"][0], two["C_bias"][1])
    assert np.array_equal(one["q_final"][0], two["q_final"][1])
    assert two["repl_cumulative"][0, -1] > 0                                 # the FR arm did act ...
    # ... but not before fr_start: identical to the ABF arm up to the save at step 199 (FR at 200)
    k = int(np.searchsorted(save_at, 199))
    assert np.array_equal(two["M_bias"][0, :k + 1], two["M_bias"][1, :k + 1])
    assert not np.array_equal(two["M_bias"][0, -1], two["M_bias"][1, -1])
    # cap 0 (the accepted law at N < 50): the FR arm is bitwise the ABF arm
    nop = wn.run_ladder_point(seed, N, n_steps, [("fr", True), ("abf", False)], dict(cfg, max_event_fraction=0.02),
                              save_at=save_at, cap_min=0, noise_seed=5)
    assert nop["cap"] == 0 and nop["repl_cumulative"][0, -1] == 0
    assert np.array_equal(nop["M_bias"][0], nop["M_bias"][1]) and np.array_equal(nop["q_final"][0], nop["q_final"][1])
    # an FR arm alone == the same FR arm next to an ABF arm (the ABF arm consumes no FR draws)
    fr1 = wn.run_ladder_point(seed, N, n_steps, [("fr", True)], cfg, save_at=save_at, noise_seed=5)
    assert np.array_equal(fr1["q_final"][0], two["q_final"][0]) and np.array_equal(fr1["M_bias"][0], two["M_bias"][0])


def test_v1f_one_fr_stream_per_fr_arm():
    """Review finding: all FR arms used to share ONE FR stream, so with two FR arms each one's
    trajectory depended on the other's draws.  Now FR arm k draws from SeedSequence child 1 + k."""
    N, n_steps, seed = 16, 800, 3102
    cfg = dict(CFG, abf_warmup_steps=100, estimator_burn_in_steps=150, fr_start_steps=200, fr_rate=40.0,
               max_event_fraction=0.25, ess_window_steps=100)
    sv = np.array([0, 200, 400, 800], dtype=np.int64)
    one = wn.run_ladder_point(seed, N, n_steps, [("fr", True)], cfg, save_at=sv, noise_seed=5)
    two = wn.run_ladder_point(seed, N, n_steps, [("fr", True), ("fr80", True, 80)], cfg, save_at=sv, noise_seed=5)
    assert np.array_equal(one["q_final"][0], two["q_final"][0])                      # the first FR arm is untouched ...
    assert np.array_equal(one["repl_cumulative"][0], two["repl_cumulative"][0])
    assert list(two["arm_fr_stream"]) == [0, 1]
    # ... and the second FR arm does not depend on what the first one does (160 vs 80 bins there)
    alt = wn.run_ladder_point(seed, N, n_steps, [("fr_b", True, 80), ("fr80", True, 80)], cfg, save_at=sv, noise_seed=5)
    assert not np.array_equal(alt["q_final"][0], two["q_final"][0])
    assert np.array_equal(alt["q_final"][1], two["q_final"][1])
    assert np.array_equal(alt["M_bias"][1][:, :80], two["M_bias"][1][:, :80])       # (accumulators padded to the widest arm)
    assert np.array_equal(alt["repl_cumulative"][1], two["repl_cumulative"][1])
    assert two["repl_cumulative"][1, -1] > 0
    # the streams are the SeedSequence children; with one FR arm the layout is the old (noise, fr) pair
    rn, rf = wn.make_rngs(5, 2)
    ss = np.random.SeedSequence(5).spawn(3)
    for g, c in zip((rn,) + rf, ss):
        assert g.random() == np.random.Generator(np.random.PCG64(c)).random()
    rn1, rf1 = wn.make_rngs(5)
    assert len(rf1) == 1 and rf1[0].bit_generator.state == np.random.Generator(np.random.PCG64(ss[1])).bit_generator.state


# ------------------------------------------------------------------------------------------- V2
def _zs(N, rng, clumped=False):
    if clumped:
        z = np.concatenate([rng.normal(0.05, 0.03, N - N // 4), rng.normal(0.95, 0.05, N // 4)])
    else:
        z = rng.uniform(-0.25, 1.25, N)                    # the soft wall lets a few out of [z_min, z_max]
    z[: min(N, 3)] = [-0.2, 1.2, -0.21][: min(N, 3)]
    return z


@pytest.mark.parametrize("N,clumped", [(1, False), (2, False), (37, False), (37, True), (1024, False), (1024, True)])
def test_v2_fr_pieces(accepted, N, clumped):
    sim, _ = accepted
    rng = np.random.default_rng(N + 7 * clumped)
    z = _zs(N, rng, clumped)
    zt = torch.as_tensor(z, dtype=F64)
    grid_t = torch.linspace(sim.z_min, sim.z_max, sim.n_grid, dtype=F64)
    grid = wn.fr_grid()
    # marginal KDE
    p_t = core.normalize_density_on_grid_torch(core.kde_1d_torch(grid_t, zt, sim.kde_bandwidth, sim.z_min, sim.z_max), grid_t).numpy()
    p = np.empty(sim.n_grid)
    wn.kde_density(z, N, grid, sim.kde_bandwidth, sim.z_min, sim.z_max, p)
    assert np.allclose(p, p_t, rtol=1e-12, atol=0)
    # uniform target and edge interpolation
    q_t = core.fr_target_uniform_torch(grid_t).numpy()
    qd = np.empty(sim.n_grid)
    wn.uniform_target(grid, qd)
    assert np.allclose(qd, q_t, rtol=1e-14, atol=0)
    zi = np.concatenate([z, [-5.0, 5.0, grid[0], grid[-1], grid[17]]])
    it = core.interp_uniform_grid_edge(torch.as_tensor(p), grid_t, torch.as_tensor(zi, dtype=F64)).numpy()
    assert np.allclose([wn.interp_edge(p, grid, v) for v in zi], it, rtol=1e-14, atol=0)
    # score: fr_score_torch, then the birth-death's own recentred clip
    sc_t, _, _, kl_t = core.fr_score_torch(zt, grid_t, sim, torch.as_tensor(q_t))
    sc_t2 = core.recentered_clipped_score_torch(sc_t, sim.score_clip).numpy()
    S = np.empty(N)
    kl = wn.raw_scores(z, N, grid, p, qd, S)
    assert abs(kl - float(kl_t)) <= 1e-12 * max(1.0, abs(float(kl_t)))
    wn.recenter_clip(S, N, sim.score_clip)
    assert np.allclose(S, sc_t.numpy(), rtol=0, atol=1e-12)
    wn.recenter_clip(S, N, sim.score_clip)
    assert np.allclose(S, sc_t2, rtol=0, atol=1e-12)
    # death probabilities / birth weights: the expressions of fixed_population_birth_death_torch :1115-1126
    rate, dt_eff = 3.0, sim.dt * sim.fr_every
    s2 = torch.as_tensor(sc_t2)
    dw, bw = torch.clamp(s2, min=0.0), torch.clamp(-s2, min=0.0)
    dp_t = torch.where(dw > 0.0, 1.0 - torch.exp(-rate * dw * dt_eff), torch.zeros_like(dw)).numpy()
    dprob, bwt = np.empty(N), np.empty(N)
    dm, bm = wn.death_birth_weights(sc_t2, N, rate, dt_eff, dprob, bwt)
    assert np.allclose(dprob, dp_t, rtol=1e-13, atol=0) and np.array_equal(bwt, bw.numpy())
    assert np.isclose(dm, float(dw.sum()), rtol=1e-13) and np.isclose(bm, float(bw.sum()), rtol=1e-13)


def _torch_event_probs(q_pre, sim, params):
    """The death probabilities and birth weights the torch engine prescribes for state q_pre:
    reaction_coordinate -> fr_score_torch (uniform target) -> fixed_population_birth_death_torch's
    recentred clip and expressions (:1100-1126), with sim's fr_rate, fr_every, kde_bandwidth, score_clip."""
    z = core.reaction_coordinate(torch.as_tensor(q_pre), params)
    grid = torch.linspace(sim.z_min, sim.z_max, sim.n_grid, dtype=F64)
    score, _, _, _ = core.fr_score_torch(z, grid, sim, core.fr_target_uniform_torch(grid))
    s2 = core.recentered_clipped_score_torch(score, sim.score_clip)
    dw, bw = torch.clamp(s2, min=0.0), torch.clamp(-s2, min=0.0)
    dt_eff = sim.dt * max(int(sim.fr_every), 1)
    dprob = torch.where(dw > 0.0, 1.0 - torch.exp(-sim.fr_rate * dw * dt_eff), torch.zeros_like(dw))
    return dprob.numpy(), bw.numpy()


def _apply_event(q_pre, dprob, bw, cap, rng):
    N = q_pre.shape[0]
    deaths, sources, cum = np.empty(N, dtype=np.int64), np.empty(N, dtype=np.int64), np.empty(N)
    k, n_cand = wn.fr_select(dprob, bw, N, cap, rng, deaths, sources, cum)
    q = q_pre.copy()
    q[deaths[:k]] = q_pre[sources[:k]]
    return q, k, n_cand


@pytest.mark.parametrize("seed,fr_every,fr_rate,frac,cap_min,kde_bw,clip,ns", [
    (3107, 3, 150.0, 0.5, 0, 0.09, 1.5, 21),        # cap 32: never binds
    (3110, 4, 60.0, 0.5, 0, 0.05, 2.5, 24),         # cap 32: never binds, other knobs
    (3108, 3, 150.0, 0.05, 0, 0.09, 1.5, 22),       # cap 3: binds (uniform subset path)
    (3109, 4, 60.0, 0.01, 1, 0.05, 2.5, 23)])       # the ladder floor: cap = cap_min = 1
def test_v2e_fr_rate_plumbing_in_simulate(accepted, seed, fr_every, fr_rate, frac, cap_min, kde_bw, clip, ns):
    """Review finding: V1e saturates the death probability (only the score's sign matters) and V2/V3
    pass the knobs to the pieces by hand, so a wrong index into simp or a wrong dt_eff in _pack would
    pass.  Here the FR arm runs to its FIRST opportunity (n_steps = fr_start) at a finite rate with
    non-accepted kde_bandwidth / score_clip / fr_every; its final state must be bitwise the ABF arm's
    (= the FR arm's pre-event state) with the event the TORCH expressions prescribe for that state,
    drawn from the FR arm's own stream.  The same check with a wrong dt_eff or bandwidth fails."""
    sim0, params = accepted
    N, fs = 64, 120
    cfg = dict(CFG, abf_warmup_steps=50, estimator_burn_in_steps=60, fr_start_steps=fs, fr_every=fr_every,
               fr_rate=fr_rate, max_event_fraction=frac, kde_bandwidth=kde_bw, score_clip=clip)
    r = wn.run_ladder_point(seed, N, fs, [("abf", False), ("fr", True)], cfg, save_at=[fs], cap_min=cap_min, noise_seed=ns)
    q_pre = r["q_final"][0]
    sim = dataclasses.replace(sim0, fr_every=fr_every, fr_rate=fr_rate, kde_bandwidth=kde_bw, score_clip=clip,
                              max_event_fraction=frac)
    cap = max(cap_min, int(sim.max_event_fraction * N))                  # torch :1109, with the ladder floor
    assert r["cap"] == cap
    dprob, bw = _torch_event_probs(q_pre, sim, params)
    assert 0.0 < dprob[dprob > 0].min() and dprob.max() < 0.9               # a finite rate: the value of p matters
    q_exp, k, n_cand = _apply_event(q_pre, dprob, bw, cap, wn.make_rngs(ns)[1][0])
    assert k > 0 and np.array_equal(q_exp, r["q_final"][1])
    assert r["repl_cumulative"][1, -1] == k and r["n_event_opportunities"][1, -1] == 1
    assert r["n_cap_binding"][1, -1] == int(n_cand > k)
    if n_cand > k:
        return              # cap binding: the outcome is a cap-subset, too coarse to resolve a perturbed p
    # power (no cap: every candidate dies, so the outcome IS the candidate set u_i < p_i): the same
    # check with dt_eff = dt (fr_every dropped) or with the accepted bandwidth fails
    for bad in (dataclasses.replace(sim, fr_every=1), dataclasses.replace(sim, kde_bandwidth=CFG["kde_bandwidth"])):
        dpw, bww = _torch_event_probs(q_pre, bad, params)
        qw, _, _ = _apply_event(q_pre, dpw, bww, cap, wn.make_rngs(ns)[1][0])
        assert not np.array_equal(qw, r["q_final"][1])


# ------------------------------------------------------------------------------------------- V3
# The law of fixed_population_birth_death_torch (:1086-1147) on fixed death probabilities p_i and
# birth weights w_i, in closed form: candidates are independent Bernoulli(p_i), M = their number
# (Poisson-binomial); k = min(cap, M) deaths, a uniformly random cap-subset if M > cap, so walker i
# dies with probability p_i E[min(1, cap / (1 + M_-i))] (M_-i = the other candidates); the k
# sources are iid with pi_j = w_j / sum w, so copies_j | k ~ Bin(k, pi_j) and two sources of one
# event coincide with probability sum pi^2.  Compared against this exact law (not a second Monte
# Carlo), with Bernoulli / binomial variances; the mutation check proves each statistic has power.
@njit(cache=False)
def _poisson_binomial(p):
    n = p.shape[0]
    out = np.zeros(n + 1)
    out[0] = 1.0
    for i in range(n):
        pi = p[i]
        for m in range(i + 1, 0, -1):
            out[m] = out[m] * (1.0 - pi) + out[m - 1] * pi
        out[0] *= 1.0 - pi
    return out


@njit(cache=False)
def _exact_law(dprob, cap):
    """(P(k = m) for m = 0..cap, per-walker death probability e_i).  M_-i by deconvolving walker i
    out of the Poisson-binomial pmf, in the numerically stable direction for its p_i."""
    n = dprob.shape[0]
    pb = _poisson_binomial(dprob)
    pk = np.zeros(cap + 1)
    for m in range(n + 1):
        pk[m if m < cap else cap] += pb[m]
    e = np.zeros(n)
    q = np.zeros(n)
    for i in range(n):
        p = dprob[i]
        if p <= 0.0:
            continue
        if p <= 0.5:
            q[0] = pb[0] / (1.0 - p)
            for m in range(1, n):
                q[m] = (pb[m] - p * q[m - 1]) / (1.0 - p)
        else:
            q[n - 1] = pb[n] / p
            for m in range(n - 1, 0, -1):
                q[m - 1] = (pb[m] - (1.0 - p) * q[m]) / p
        s = 0.0
        for m in range(n):
            if q[m] > 0.0:
                s += q[m] * min(1.0, cap / (1.0 + m))
        e[i] = p * s
    return pk, e


@njit(cache=False)
def _mutant_select(dprob, bweight, n, cap, rng, deaths, sources, cum, mode):
    """Plausible WRONG laws: 1 the first cap candidates (no random subset), 2 sources uniform over
    the negative scores, 3 sources ~ sqrt(w), 4 one shared source for all deaths of an event."""
    nd = 0
    for i in range(n):
        if rng.random() < dprob[i]:
            deaths[nd] = i
            nd += 1
    if nd == 0:
        return 0, 0
    nc = nd
    if nd > cap:
        if mode != 1:
            for t in range(cap):
                j = t + int(rng.random() * (nd - t))
                tmp = deaths[t]
                deaths[t] = deaths[j]
                deaths[j] = tmp
        nd = cap
    tot = 0.0
    for i in range(n):
        wi = bweight[i]
        if mode == 2:
            wi = 1.0 if wi > 0.0 else 0.0
        elif mode == 3:
            wi = math.sqrt(wi)
        tot += wi
        cum[i] = tot
    for t in range(nd):
        if mode == 4 and t > 0:
            sources[t] = sources[0]
            continue
        u = rng.random() * tot
        lo = 0
        while not (cum[lo] > u):
            lo += 1
        sources[t] = lo
    return nd, nc


@njit(cache=False)
def _law_sample(dprob, bweight, n, cap, rng, reps, mode, khist, died, copies, joint):
    """reps opportunities of wn.fr_select (mode 0) or a mutant; returns the number of invariant
    violations (k > cap, candidates < k, a walker dying twice in one event)."""
    deaths = np.empty(n, dtype=np.int64)
    sources = np.empty(n, dtype=np.int64)
    cum = np.empty(n)
    mark = np.zeros(n, dtype=np.int64)
    bad = 0
    for r in range(reps):
        if mode == 0:
            k, nc = wn.fr_select(dprob, bweight, n, cap, rng, deaths, sources, cum)
        else:
            k, nc = _mutant_select(dprob, bweight, n, cap, rng, deaths, sources, cum, mode)
        if k > cap or nc < k or (nc > k and k != cap):
            bad += 1
        khist[k] += 1
        for t in range(k):
            if mark[deaths[t]] == r + 1:
                bad += 1
            mark[deaths[t]] = r + 1
            died[deaths[t]] += 1
            copies[sources[t]] += 1
        if k >= 2:
            joint[0] += 1
            if sources[0] == sources[1]:
                joint[1] += 1
    return bad


def _law_stats(dprob, bweight, cap, reps, khist, died, copies, joint):
    """Standardised discrepancies of a sample from the exact law (each ~N(0, 1) or chi2-z under it)."""
    pk, e = _exact_law(dprob, cap)
    m = np.arange(cap + 1)
    Ek = float((m * pk).sum())
    Vk = float((m * m * pk).sum()) - Ek ** 2
    pi = bweight / bweight.sum()
    st = {}
    ex = reps * pk                                              # count law: chi2, cells with expectation < 5 pooled
    use = ex >= 5
    chi = float((((khist[use] - ex[use]) ** 2) / ex[use]).sum())
    dof = int(use.sum()) - 1
    if ex[~use].sum() > 0:
        chi += float((khist[~use].sum() - ex[~use].sum()) ** 2 / ex[~use].sum())
        dof += 1
    st["k_chi2_z"] = (chi - dof) / math.sqrt(2 * dof) if dof > 0 else 0.0
    st["k_mean_z"] = float((khist @ m / reps - Ek) / math.sqrt(max(Vk, 1e-300) / reps))
    d = e > 0                                                   # deaths: Bernoulli(e_i) per opportunity
    zd = (died[d] - reps * e[d]) / np.sqrt(reps * e[d] * (1 - e[d]))
    st["death_chi2_z"] = float(((zd ** 2).sum() - d.sum()) / math.sqrt(2 * d.sum()))
    st["death_impossible"] = int(died[~d].sum())
    c = pi > 0                                                  # copies: Bin(k, pi_j) given k
    mu, var = Ek * pi[c], Ek * pi[c] * (1 - pi[c]) + Vk * pi[c] ** 2
    zc = (copies[c] - reps * mu) / np.sqrt(reps * var)
    st["copy_chi2_z"] = float(((zc ** 2).sum() - c.sum()) / math.sqrt(2 * c.sum()))
    st["copy_impossible"] = int(copies[~c].sum())
    s2, n2 = float((pi ** 2).sum()), int(joint[0])              # joint: iid sources
    st["n_k2"] = n2
    st["same_src_z"] = float((joint[1] - n2 * s2) / math.sqrt(n2 * s2 * (1 - s2))) if n2 > 0 else 0.0
    st["p_bind"] = float(1.0 - _poisson_binomial(dprob)[:cap + 1].sum())       # P(M > cap)
    return st


def _law_rejects(st, thr=5.0):
    return (st["k_chi2_z"] > thr or abs(st["k_mean_z"]) > thr or st["death_chi2_z"] > thr or st["copy_chi2_z"] > thr
            or abs(st["same_src_z"]) > thr or st["death_impossible"] > 0 or st["copy_impossible"] > 0)


def _law_inputs(N, rate, dt_eff=0.01, salt=0):
    """A valid fr_score_torch output (recentred clipped, then the birth-death's own recentring) and
    its death probabilities / birth weights."""
    rng = np.random.default_rng(100 + N + salt)
    S = np.clip(rng.normal(0.0, 1.2, N), -2.0, 2.0)
    wn.recenter_clip(S, N, CFG["score_clip"])
    S2 = S.copy()
    wn.recenter_clip(S2, N, CFG["score_clip"])
    dprob, bw = np.empty(N), np.empty(N)
    dm, bm = wn.death_birth_weights(S2, N, rate, dt_eff, dprob, bw)
    assert dm > wn.EPS and bm > wn.EPS
    return S, dprob, bw


def _law_run(dprob, bw, cap, reps, mode, seed):
    N = dprob.shape[0]
    kh, died, cop, jt = np.zeros(cap + 1, dtype=np.int64), np.zeros(N), np.zeros(N), np.zeros(2, dtype=np.int64)
    bad = _law_sample(dprob, bw, N, cap, np.random.default_rng(seed), reps, mode, kh, died, cop, jt)
    return bad, _law_stats(dprob, bw, cap, reps, kh, died, cop, jt)


@pytest.mark.parametrize("N,cap,rate,reps", [
    (4, 1, 300.0, 200000),       # ladder small-N regime (cap_min = 1), cap binds
    (8, 1, 60.0, 200000), (16, 1, 60.0, 200000), (64, 1, 30.0, 100000), (256, 5, 8.0, 50000),
    (1024, 20, 6.0, 100000),     # the accepted N and cap with the cap binding (P(M > cap) ~ 0.86)
    (1024, 20, 0.1, 100000)])    # the accepted rate (the cap never binds)
def test_v3_law_exact_with_power(N, cap, rate, reps):
    """Review finding: the old V3 (Monte Carlo vs Monte Carlo, per-walker marginals, a(1+a)
    variance, 5.5 sigma) passed every mutant at N = 1024 and could not see within-event structure.
    Here: fr_select vs the exact law, and every mutant that changes the law in this regime is rejected."""
    _, dprob, bw = _law_inputs(N, rate)
    bad, st = _law_run(dprob, bw, cap, reps, 0, 1)
    assert bad == 0 and not _law_rejects(st), st
    for mode in (1, 2, 3, 4):
        if mode == 1 and st["p_bind"] < 0.2:          # the cap (almost) never binds: same law
            continue
        if mode == 4 and st["n_k2"] < 1000:           # (almost) never two deaths: same law
            continue
        _, st_m = _law_run(dprob, bw, cap, reps, mode, 1 + mode)
        assert _law_rejects(st_m), (mode, st_m)


@pytest.mark.parametrize("N,cap,rate,reps", [(16, 1, 60.0, 20000), (64, 2, 30.0, 10000), (1024, 20, 6.0, 10000)])
def test_v3_torch_law_is_the_exact_law(accepted, N, cap, rate, reps):
    """The exact law above IS the law of fixed_population_birth_death_torch (it applies its own
    recentring to S, which _law_inputs reproduces), so V3 compares the port with the torch law."""
    sim, _ = accepted
    S, dprob, bw = _law_inputs(N, rate)
    simc = dataclasses.replace(sim, max_event_fraction=(cap + 0.5) / N)          # int(frac * N) == cap
    assert int(simc.max_event_fraction * N) == cap and abs(simc.dt * simc.fr_every - 0.01) < 1e-15
    q = torch.arange(N, dtype=F64).view(N, 1, 1)
    St = torch.as_tensor(S, dtype=F64)
    torch.manual_seed(1)
    kh, died, cop, jt = np.zeros(cap + 1, dtype=np.int64), np.zeros(N), np.zeros(N), np.zeros(2, dtype=np.int64)
    for _ in range(reps):
        qn, _a, out = core.fixed_population_birth_death_torch(q, St, simc, fr_interval=simc.fr_every, fr_rate_override=rate)
        k = int(out["replacement"])
        kh[k] += 1
        if k:
            d, b = out["death_idx"].numpy(), out["birth_src"].numpy()
            assert len(set(d.tolist())) == k and np.array_equal(qn.view(-1).numpy()[d], b.astype(float))
            died[d] += 1
            np.add.at(cop, b, 1)
            if k >= 2:
                jt[0] += 1
                jt[1] += int(b[0] == b[1])
    st = _law_stats(dprob, bw, cap, reps, kh, died, cop, jt)
    assert not _law_rejects(st), st


def test_v3_cap_zero_and_degenerate_scores():
    rng = np.random.default_rng(0)
    N = 8
    grid = wn.fr_grid()
    G = len(grid)
    args = lambda n: (np.full(n, 0.3), n, grid, CFG["kde_bandwidth"], CFG["z_min"], CFG["z_max"], CFG["score_clip"],
                      50.0, 0.01)
    work = lambda n: (np.empty(G), np.empty(G), np.empty(n), np.empty(n), np.empty(n), np.empty(n, dtype=np.int64),
                      np.empty(n, dtype=np.int64), np.empty(n))
    st = rng.bit_generator.state
    assert wn.fr_uniform_event(*args(N), 0, rng, *work(N)) == (0, 0)                  # cap 0: returns at once
    assert rng.bit_generator.state == st                                                # and draws nothing
    assert wn.fr_uniform_event(*args(N), 3, rng, *work(N)) == (0, 0)                  # all z equal: no death mass
    assert wn.fr_uniform_event(*args(1), 3, rng, *work(1)) == (0, 0)                  # N = 1


# ------------------------------------------------------------------------------------------- V4
def test_v4_genealogy_and_bookkeeping():
    N, n_steps, seed, win, burn = 32, 3000, 3103, 500, 400
    cfg = dict(CFG, abf_warmup_steps=200, estimator_burn_in_steps=burn, fr_start_steps=100, fr_rate=20.0,
               max_event_fraction=0.1, ess_window_steps=win)
    save_at = np.array(sorted(set([0, 1, 99, 100, 399, 400, 401, 1000, 1500, 2000, 2003, 2500, n_steps])), dtype=np.int64)
    r = wn.run_ladder_point(seed, N, n_steps, [("abf", False), ("fr", True)], cfg, save_at=save_at, cap_min=1, noise_seed=9)
    assert r["cap"] == 3
    # deposits: every replica deposits once per state, the production estimator from the burn-in on
    for a in (0, 1):
        assert np.array_equal(r["C_bias"][a].sum(1), N * (save_at + 1.0))
        assert np.array_equal(r["C_prod"][a].sum(1), N * np.maximum(save_at - burn + 1.0, 0.0))
        assert np.array_equal(r["M_rep"][a], np.where((save_at >= burn)[:, None], r["M_prod"][a], r["M_bias"][a]))
        assert np.allclose(r["z_hist"][a].sum(1), N) and np.allclose(r["frac_regions"][a].sum(1), 1.0)
    assert np.all(np.isnan(r["ancestor_ess"][0])) and np.isnan(r["min_ancestor_ess_window"][0])
    rep, nev = r["repl_cumulative"][1], r["n_event_opportunities"][1]
    assert rep[-1] > 0 and np.all(np.diff(rep) >= 0) and np.all(rep <= r["cap"] * nev) and np.all(rep >= nev)
    assert np.all(rep[save_at < 100] == 0) and r["n_cap_binding"][1, -1] > 0       # first FR event at step 100
    # run-long genealogy at the last save from the final labels
    cnt = np.bincount(r["ancestors_final"][1], minlength=N)
    assert np.isclose(r["ancestor_ess"][1, -1], N * N / np.sum(cnt.astype(float) ** 2), rtol=1e-14)
    assert r["max_ancestor_frac"][1, -1] == cnt.max() / N and r["n_unique_ancestor"][1, -1] == np.count_nonzero(cnt)
    cw = np.bincount(r["ancestors_window_final"][1], minlength=N)
    assert np.isclose(r["ancestor_ess_window"][1, -1], N * N / np.sum(cw.astype(float) ** 2), rtol=1e-14)
    # the window is reset at FR opportunities with step % win == 0: fresh at those saves
    for s in (1000, 1500, 2000, 2500, n_steps):
        assert r["ancestor_ess_window"][1, list(save_at).index(s)] == N
    assert r["ancestor_ess_window"][1, list(save_at).index(2003)] <= N
    m = r["min_ancestor_ess_window_t"][1]
    assert np.all(np.diff(m) <= 0) and np.all(m <= N) and m[-1] == r["min_ancestor_ess_window"][1] < N
    assert np.all(r["n_unique_ancestor"][0] == N)                                    # the ABF arm (torch convention)


def test_v4b_ess_window_nonpositive_is_window_one():
    """Review finding: torch resets the windowed ancestry with _win = max(ess_window_steps, 1)
    (core :1565, :1577), i.e. at every FR opportunity when ess_window_steps <= 0; the port used to
    never reset there.  ess_window_steps 0 and -7 must now be bitwise ess_window_steps 1."""
    N, n_steps, seed = 16, 600, 3104
    base = dict(CFG, abf_warmup_steps=50, estimator_burn_in_steps=60, fr_start_steps=100, fr_rate=40.0,
                max_event_fraction=0.25)
    sv = np.array([100, 300, 600], dtype=np.int64)
    rs = {w: wn.run_ladder_point(seed, N, n_steps, [("fr", True)], dict(base, ess_window_steps=w), save_at=sv, noise_seed=3)
          for w in (1, 0, -7, 4000)}
    assert rs[1]["repl_cumulative"][0, -1] > 0
    for w in (0, -7):
        for k in ("ancestor_ess_window", "min_ancestor_ess_window_t", "min_ancestor_ess_window", "ancestors_window_final",
                  "q_final"):
            assert np.array_equal(rs[w][k], rs[1][k]), (w, k)
    assert np.all(rs[1]["ancestor_ess_window"][0] == N)                 # reset at every opportunity (saves are opportunities)
    assert np.array_equal(rs[4000]["q_final"], rs[1]["q_final"])         # a read-off only: dynamics untouched
    assert rs[4000]["min_ancestor_ess_window"][0] < rs[1]["min_ancestor_ess_window"][0]


def test_v4c_abf_arm_n_unique_ancestor_is_N(accepted):
    """Review finding: the port reported 0 unique ancestors for an ABF arm; the torch engine reports
    N there (core :1791-1794; ancestor ESS and max lineage share NaN)."""
    sim, params = accepted
    for N in (1, 5):
        simt = dataclasses.replace(wn.accepted_setup(method="abf")[0], n_replicas=N, n_steps=40, save_every=20,
                                   abf_warmup_steps=10, estimator_burn_in_steps=10)
        q0 = wn.lattice_init(3100, N)
        d = core.run_sampler_gpu("abf", params, simt, core.WCADimerEngine(params, CPU, F64), initial_q=torch.as_tensor(q0),
                                 verbose=False)
        r = wn.run_ladder_point(3100, N, 40, [("abf", False), ("fr", True)], dict(CFG, abf_warmup_steps=10, estimator_burn_in_steps=10),
                                save_at=np.asarray(d["steps"], dtype=np.int64), cap_min=1, q0=q0, noise_seed=1)
        assert np.array_equal(r["n_unique_ancestor"][0], d["n_unique_ancestor"]) and np.all(r["n_unique_ancestor"][0] == N)
        assert np.all(np.isnan(r["ancestor_ess"][0])) and np.all(np.isnan(d["ancestor_ess"]))
        assert np.all(np.isnan(r["max_ancestor_frac"][0])) and np.all(np.isnan(d["max_ancestor_frac"]))
        assert np.all(r["n_unique_ancestor"][1] == N)                                 # an FR arm with no event yet


# ------------------------------------------------------------------------------------------- V5
@pytest.mark.parametrize("seed", [3100, 3115])
def test_v5_scorer_reproduces_accepted_numbers(accepted, seed):
    sim, params = accepted
    ref = wn.load_reference(sim, params)
    rows = {int(l.split(",")[0]): l.strip().split(",") for l in open(os.path.join(CONF_RAW, "..", "comparison.csv")).readlines()[1:]}
    head = open(os.path.join(CONF_RAW, "..", "comparison.csv")).readline().strip().split(",")
    for arm in ("hist_abf", "hist_fr_uniform"):
        d = np.load(_accepted_file(arm, seed), allow_pickle=True)
        C = d["hist_counts_t"]
        out = wn.score_accumulators(d["hist_mf_bins_t"] * C, C, d["profile_steps"], sim=sim, reference=ref)
        # e_F' (own and node-sampled) is computed from the stored float64 bins: exact
        for k in ("l2_fp_t", "l2_fp_nodes_t"):
            assert np.allclose(out[k], d[k], rtol=1e-12, atol=0), k
        assert np.isclose(out["l2_fp"], float(d["l2_fp"]), rtol=1e-12)
        # e_F: the accepted PMF was integrated in float32 on the GPU -> agreement to ~1e-6
        assert np.allclose(out["l2_f_t"], d["l2_f_t"], rtol=1e-5, atol=0)
        for k in ("l2_f", "integrated_l2_f", "l2_f_transition"):
            assert np.isclose(out[k], float(d[k]), rtol=1e-5), k
        row = dict(zip(head, rows[seed]))
        assert abs(out["integrated_l2_f"] - float(row[f"{arm}_int_l2_f"])) < 1e-4
        assert abs(out["l2_f"] - float(row[f"{arm}_final_l2_f"])) < 2e-6
        assert abs(out["l2_fp"] - float(row[f"{arm}_final_l2_fp"])) < 2e-6


def test_v5b_accepted_boundary_bin_float32_saturation():
    """The accepted histogram runs stored C, M in float32 (the engine dtype) and the boundary bin
    z >= 1.19125 (which also collects the wall tail z > 1.2) saturated: C == 2^24 exactly in every
    run.  The numba default (float64) does not saturate; ``accum_float32=True`` emulates it."""
    files = sorted(glob.glob(os.path.join(CONF_RAW, "confirmation__hist_*.npz")))
    assert len(files) == 32
    for f in files:
        d = np.load(f, allow_pickle=True)
        C = d["final_hist_counts"]
        assert C[-1] == 2.0 ** 24 and C[:-1].max() < 2.0 ** 24, f
        assert d["hist_edges_abf"][-2] > CFG["eval_z_lo"] and d["hist_edges_abf"][-2] > 1.1     # outside the window
    # review finding: the artefact is NOT neutral between the arms (the numbers the docstrings state)
    for arm, sat_lo, sat_hi, mf_lo, mf_hi, clip_lo, clip_hi in (("hist_abf", 105000, 107500, 44.1, 45.6, 0.047, 0.052),
                                                                ("hist_fr_uniform", 115000, 117500, 39.2, 40.4, 0.033, 0.037)):
        for f in sorted(glob.glob(os.path.join(CONF_RAW, f"confirmation__{arm}__*.npz"))):
            d = np.load(f, allow_pickle=True)
            C, mf, st = d["hist_counts_t"][:, -1], d["hist_mf_bins_t"][:, -1], d["profile_steps"]
            k = int(np.argmax(C >= 2.0 ** 24))
            assert C[k] == 2.0 ** 24 and sat_lo <= st[k] <= sat_hi and st[-1] == 120000, (f, st[k])
            assert 37.1 < mf[k - 1] < 37.4 and mf_lo < mf[-1] < mf_hi, (f, mf[k - 1], mf[-1])
            assert clip_lo < float(d["bias_clip_fraction"]) < clip_hi, f


def test_v5c_float32_emulation():
    from numba import njit

    @njit
    def bump(c, m, f):
        c[0] += 1.0                                       # the exact update used in simulate
        m[0] += f
    c = np.full(1, 2.0 ** 24, dtype=np.float32)
    m = np.full(1, 1.0e9, dtype=np.float32)
    bump(c, m, 37.0)
    assert c[0] == 2.0 ** 24 and m[0] == np.float32(1.0e9 + 37.0)          # count saturates, the sum does not
    # below saturation, over a horizon short enough that the float32 rounding of the bias has not
    # been amplified by the dynamics (V1c), the float32 path keeps the counts and the sums
    cfg = dict(CFG, abf_warmup_steps=20, estimator_burn_in_steps=30)
    sv = np.array([0, 10, 50, 100], dtype=np.int64)
    r64 = wn.run_ladder_point(3106, 16, 100, [("abf", False)], cfg, save_at=sv, noise_seed=4)
    r32 = wn.run_ladder_point(3106, 16, 100, [("abf", False)], cfg, save_at=sv, noise_seed=4, accum_float32=True)
    assert np.array_equal(r64["C_rep"][0], r32["C_rep"][0])
    assert np.allclose(r32["M_rep"][0], r64["M_rep"][0], rtol=1e-5, atol=1e-3)


# ------------------------------------------------------------------------------------------- V6
def test_v6_result_carries_cfg_and_ladder_scorer_reads_it(accepted, tmp_path):
    """Review findings: (a) the result did not carry its cfg and score_ladder_result silently scored
    any job on the ACCEPTED SimConfig (a dt = 0.004 job got the dt = 0.002 times); (b) the physical-time
    integrated_l2_f scales with n_steps = B / N, so the scorer now also returns the budget-axis integral."""
    sim, params = accepted
    ref = wn.load_reference(sim, params)
    cfg = dict(CFG, dt=0.004, abf_warmup_steps=50, estimator_burn_in_steps=100)
    sv = np.array([0, 200, 400], dtype=np.int64)
    r = wn.run_ladder_point(3101, 8, 400, [("abf", False), ("fr80", True, 80)], cfg, save_at=sv, noise_seed=2)
    assert r["cfg"] == cfg and np.array_equal(r["times"], sv * 0.004)
    sc = wn.score_ladder_result(r, "abf", reference=ref)
    assert np.array_equal(sc["times"], sv * 0.004)
    # identical to score_accumulators on the SimConfig built by hand for this cfg
    sim_c = dataclasses.replace(sim, dt=0.004, abf_warmup_steps=50, estimator_burn_in_steps=100)
    hand = wn.score_accumulators(r["M_rep"][0], r["C_rep"][0], sv, sim=sim_c, reference=ref)
    for k in ("l2_f", "integrated_l2_f", "l2_fp", "l2_f_t"):
        assert np.array_equal(sc[k], hand[k]), k
    s_c, p_c = wn.setup_from_cfg(cfg)
    assert wn.cfg_from_setup(s_c, p_c) == cfg and s_c.dt == 0.004
    # budget axis: u = s / n_steps; the physical-time integral is n_steps * dt times the u-integral
    assert np.array_equal(sc["u"], sv / 400.0)
    assert np.isclose(sc["integrated_l2_f"], sc["integrated_l2_f_u"] * 400 * 0.004, rtol=1e-12)
    assert np.isclose(sc["mean_l2_f_u"], sc["integrated_l2_f_u"], rtol=1e-12)          # u spans [0, 1] here
    sc80 = wn.score_ladder_result(r, "fr80", reference=ref)                            # scored on its own 80 bins
    assert sc80["hist_mf_bins_t"].shape == (3, 80)
    with pytest.raises(ValueError):                                                     # a sim that disagrees with the job
        wn.score_ladder_result(r, "abf", reference=ref, sim=sim)
    with pytest.raises(ValueError):                                                     # other physics: no default reference
        wn.score_ladder_result(dict(r, cfg=dict(cfg, h=2.5)), "abf")
    # save / load round trip (no pickle) scores identically
    path = str(tmp_path / "job.npz")
    wn.save_result(r, path)
    r2 = wn.load_result(path)
    assert r2["cfg"] == cfg and r2["arms"] == ["abf", "fr80"]
    sc2 = wn.score_ladder_result(r2, "abf", reference=ref)
    for k in ("l2_f", "integrated_l2_f", "integrated_l2_f_u", "l2_fp"):
        assert sc2[k] == sc[k], k


_GUARD_SCRIPT = textwrap.dedent("""
    import os, sys
    sys.path.insert(0, {src!r})
    import wca_numba as wn
    assert "torch" not in sys.modules                    # importing the port does not import torch
    g = wn.fr_grid()
    import wca_abffr_core as core
    print(repr(os.environ["CUDA_VISIBLE_DEVICES"]), core.DEVICE.type, len(g))
""")

_GUARD_REFUSE_SCRIPT = textwrap.dedent("""
    import sys, torch
    sys.path.insert(0, {src!r})
    import wca_abffr_core as core, wca_numba as wn
    core.DEVICE = torch.device("meta")                   # as if core had been imported with a GPU visible
    try:
        wn.fr_grid()
        print("ran")
    except RuntimeError as e:
        print("refused:", e)
""")


def test_v6b_cpu_guard():
    """Review finding: run_ladder_point had no CPU guard; a worker launched without
    CUDA_VISIBLE_DEVICES='' would have let wca_abffr_core's import-time choose_device() initialise
    CUDA.  The guard hides CUDA before the first torch import and refuses a non-CPU core.  (The
    child runs with CUDA_VISIBLE_DEVICES=-1, which hides every GPU even if the guard failed.)"""
    src = os.path.join(ROOT, "src")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="1")
    out = subprocess.run([sys.executable, "-c", _GUARD_SCRIPT.format(src=src)], check=True, env=env, cwd=ROOT,
                         capture_output=True, text=True).stdout.split()
    assert out == ["''", "cpu", "160"], out
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1")
    out = subprocess.run([sys.executable, "-c", _GUARD_REFUSE_SCRIPT.format(src=src)], check=True, env=env, cwd=ROOT,
                         capture_output=True, text=True).stdout
    assert out.startswith("refused:") and "CPU-only" in out, out
