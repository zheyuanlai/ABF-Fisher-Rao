"""Gates for the information-conversion audit.

Frozen protocol: ``docs/INFORMATION_CONVERSION_AUDIT_PREREGISTRATION.md``.

These run BEFORE the science and encode the preregistered audits: the
water-filling solver's KKT properties, the one-pulse discipline, the
zero-deposit property of a BD event, the counts/M unit consistency, the
FEC-blindness of dose selection, and an end-to-end parity gate of the new
runner against ``simulation_torch.run_batch`` with identical noise.
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

from abffr import info_conversion as ic          # noqa: E402
from abffr import metrics, reference, simulation_torch  # noqa: E402
from abffr.io_utils import RunSpec               # noqa: E402

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


# --------------------------------------------------------------------------- #
# Water-filling
# --------------------------------------------------------------------------- #
def _random_problem(seed, J=32, K=256):
    rng = np.random.default_rng(seed)
    av = np.exp(rng.normal(0, 2, J))
    av[:4] = 0.0                               # cells outside the mask
    C = rng.integers(1_000, 200_000, J).astype(float)
    M = float(K * rng.integers(400, 3_000))
    return av, C, M, K


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_water_filling_normalisation_and_floor(seed):
    av, C, M, K = _random_problem(seed)
    sol = ic.solve_finite_horizon_target(av, C, M, K)
    pi = sol["pi"]
    assert abs(pi.sum() - 1.0) < 1e-10
    assert np.all(pi >= 1.0 / K - 1e-12)


@pytest.mark.parametrize("seed", [0, 5])
def test_water_filling_kkt_stationarity_on_free_cells(seed):
    av, C, M, K = _random_problem(seed)
    sol = ic.solve_finite_horizon_target(av, C, M, K)
    pi = sol["pi"]
    free = (~sol["floor_bound"]) & (av > 0)
    assert free.sum() >= 2
    grad = av[free] * M / (C[free] + M * pi[free]) ** 2
    assert np.std(grad) / np.mean(grad) < 1e-6


@pytest.mark.parametrize("seed", [0, 7])
def test_water_filling_beats_uniform(seed):
    av, C, M, K = _random_problem(seed)
    sol = ic.solve_finite_horizon_target(av, C, M, K)
    J = av.size
    r_u = ic.predicted_finite_risk(av, C, M, np.full(J, 1.0 / J))
    assert sol["risk"] <= r_u + 1e-12


def test_water_filling_scale_invariance():
    av, C, M, K = _random_problem(11)
    a = ic.solve_finite_horizon_target(av, C, M, K)["pi"]
    b = ic.solve_finite_horizon_target(1e6 * av, C, M, K)["pi"]
    assert np.allclose(a, b, atol=1e-9)


def test_water_filling_zero_leverage_cells_sit_on_the_floor():
    av, C, M, K = _random_problem(3)
    sol = ic.solve_finite_horizon_target(av, C, M, K)
    assert np.allclose(sol["pi"][av == 0], 1.0 / K, atol=1e-12)


# --------------------------------------------------------------------------- #
# Structural: bd_standard is the only reallocation operator
# --------------------------------------------------------------------------- #
FORBIDDEN_IN_MODULE = (
    "systematic_resample", "ft_step", "bd_paired", "resample_cells",
    "multinomial", "qr_arms", "balanced_representation", "deadband",
    "persistent_mass", "score_clip", "max_event_fraction", "jitter",
    "apply_holdout",
)


def _source(path):
    with open(os.path.join(SRC, path)) as fh:
        return fh.read()


def test_runner_module_admits_only_bd_standard():
    src = _source("src/abffr/info_conversion.py")
    for tok in FORBIDDEN_IN_MODULE:
        assert tok not in src, f"forbidden operator token {tok!r} in runner"
    assert src.count("fr_v3.bd_standard(") == 1, \
        "bd_standard must be invoked at exactly one code site"


def test_scripts_never_touch_fr_operators_directly():
    for path in ("scripts/run_info_conversion.py",
                 "scripts/analyze_info_conversion.py"):
        src = _source(path)
        for tok in ("bd_standard", "ft_step", "systematic_resample",
                    "multinomial", "bd_paired"):
            assert tok not in src, f"{path} references {tok!r}"


# --------------------------------------------------------------------------- #
# Engine behaviour: units, one pulse, zero-deposit BD
# --------------------------------------------------------------------------- #
def _tiny_cfg(N=64, chunk=100):
    return {
        "simulation": {"beta": 4.0, "dt": 0.002, "n_particles": N,
                       "noise_chunk_steps": chunk},
        "abf": {"estimator": "binned_smooth",
                "observation_order": "post_propagation",
                "h": 0.05, "update_every": 10, "min_count": 1.0},
        "domain": {"x_min": -3.0, "x_max": 3.0, "y_min": -2.5, "y_max": 3.5},
        "potential": {"x_tilt": 0.1021665783},
        "kde": {"eta": 0.10},
    }


def _tiny_engine(cell="K2", N=64, G=201, device="cpu", noise_mode="chunk"):
    cfg = _tiny_cfg(N=N)
    x = np.linspace(-3.0, 3.0, G)
    y = np.linspace(-2.5, 3.5, 201)
    ref = reference.compute_reference(x, y, beta=4.0,
                                      x_tilt=cfg["potential"]["x_tilt"])
    mask = (x >= -2.5) & (x <= 2.5)
    geom = ic.build_cells(x, mask, 32, ref["Fprime_ref"])
    eng = ic.InfoConversionEngine(cfg, cell, x, ref["F_ref"],
                                  ref["Fprime_ref"], geom,
                                  torch.device(device), noise_mode=noise_mode)
    return eng, geom


def test_deposits_happen_every_step_units_of_M():
    eng, geom = _tiny_engine()
    st = eng.init_state([9000, 9001])
    n = 137
    eng.run(st, n)
    total = st.C_acc.sum(dim=1).cpu().numpy()
    assert np.allclose(total, eng.N * n), \
        "one deposit per replica per step is what makes M = K x H"
    assert np.allclose(st.cell_cnt.sum(dim=1).cpu().numpy(), eng.N * n)
    assert np.allclose(st.cell_cnt.sum(dim=1).cpu().numpy(),
                       st.C_acc.sum(dim=1).cpu().numpy()), \
        "hard-cell counts and grid counts must share units"


def _uniform_target_rows(eng, B):
    q = torch.full((B, eng.G), 1.0 / (eng.xmax - eng.xmin),
                   device=eng.device, dtype=eng.dtype)
    return q


def test_bd_event_deposits_zero_observations():
    eng, geom = _tiny_engine()
    st = eng.init_state([9000])
    eng.run(st, 200)
    before = float(st.C_acc.sum())
    gen = torch.Generator(device="cpu").manual_seed(1)
    rows = eng.pulse(st, _uniform_target_rows(eng, 1), {0: 0.5}, {0: gen})
    assert rows[0]["n_events"] > 0, "the test needs at least one event"
    assert float(st.C_acc.sum()) == before, "a BD event may not deposit"
    assert float(st.cell_cnt.sum()) == before
    eng.run(st, 1)
    assert float(st.C_acc.sum()) == before + eng.N


def test_exactly_one_pulse_is_enforced():
    eng, geom = _tiny_engine()
    st = eng.init_state([9000])
    eng.run(st, 100)
    gen = torch.Generator(device="cpu").manual_seed(1)
    eng.pulse(st, _uniform_target_rows(eng, 1), {0: 0.1}, {0: gen})
    with pytest.raises(RuntimeError, match="exactly one"):
        eng.pulse(st, _uniform_target_rows(eng, 1), {0: 0.1}, {0: gen})


def test_identical_target_gives_degenerate_dose_not_an_invented_one():
    eng, geom = _tiny_engine()
    st = eng.init_state([9000])
    eng.run(st, 100)
    q = eng.p_hat(st)                          # target == current marginal
    gen = torch.Generator(device="cpu").manual_seed(1)
    rows = eng.pulse(st, q, {0: 0.1}, {0: gen})
    assert rows[0]["degenerate"] is True or rows[0]["s90"] < 1e-12 or \
        rows[0]["n_events"] == 0


def test_fork_shares_noise_and_plain_arms_stay_identical():
    eng, geom = _tiny_engine()
    st = eng.init_state([9000])
    eng.run(st, 50)
    forked = eng.fork(st, 2)                   # two plain arms, no pulse
    eng.run(forked, 50)
    assert torch.equal(forked.X[0], forked.X[1]), \
        "arms forked from one seed must see identical noise"


def test_cell_index_torch_matches_numpy_digitize():
    edges = np.linspace(-3.0, 3.0, 33)
    rng = np.random.default_rng(0)
    pts = np.concatenate([rng.uniform(-3, 3, 4000), edges, [-3.0, 3.0]])
    want = np.clip(np.digitize(pts, edges) - 1, 0, 31)
    got = ic.cell_index_torch(
        torch.as_tensor(pts, dtype=torch.float64),
        torch.as_tensor(edges, dtype=torch.float64)).numpy()
    assert np.array_equal(want, got)


# --------------------------------------------------------------------------- #
# Parity: the new runner IS the engine's arithmetic
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cell", ["K0", "K2"])
def test_plain_abf_parity_against_run_batch(cell):
    n_steps, N, G = 400, 64, 401
    x = np.linspace(-3.0, 3.0, G)
    y = np.linspace(-2.5, 3.5, 401)
    tilt = 0.1021665783
    ref = reference.compute_reference(x, y, beta=4.0, x_tilt=tilt)
    seeds = [9100, 9101]

    cfg_engine = {
        "simulation": {"beta": 4.0, "dt": 0.002, "n_steps": n_steps,
                       "n_particles": N, "eval_every": n_steps,
                       "x_init_mode": "uniform", "y_init_mode": "uniform"},
        "domain": {"x_min": -3.0, "x_max": 3.0, "y_min": -2.5, "y_max": 3.5},
        "potential": {"x_tilt": tilt},
        "kappa": {"cell": cell},
        "abf": {"estimator": "binned_smooth",
                "observation_order": "post_propagation",
                "h": 0.05, "update_every": 10, "min_count": 1.0},
        "fr": {"enabled": False, "noise_chunk_steps": 1024},
    }
    specs = [RunSpec(method="abf_only", target_type="none", seed=s, gamma=0.0,
                     eta=0.10, burnin_fraction=0.0, fr_every=1,
                     stop_fraction=1.0) for s in seeds]
    ev = metrics.EvalConfig.from_domain(cfg_engine["domain"])
    res = simulation_torch.run_batch(
        specs, cfg=cfg_engine, x_grid=x, F_ref=ref["F_ref"],
        Fprime_ref=ref["Fprime_ref"], ev=ev, device=torch.device("cpu"),
        dtype=torch.float64, estimator="binned_smooth", base_seed=0)

    cfg_mine = _tiny_cfg(N=N)
    mask = (x >= -2.5) & (x <= 2.5)
    geom = ic.build_cells(x, mask, 32, ref["Fprime_ref"])
    eng = ic.InfoConversionEngine(cfg_mine, cell, x, ref["F_ref"],
                                  ref["Fprime_ref"], geom,
                                  torch.device("cpu"),
                                  noise_mode="sequential")
    st = eng.init_state(seeds, n_steps_hint=n_steps)
    eng.run(st, n_steps)

    for b in range(len(seeds)):
        F_eng = np.asarray(res.diags[b]["F_hat"][-1])
        Fp_eng = np.asarray(res.diags[b]["Fprime_hat"][-1])
        X_eng = np.asarray(res.diags[b]["X_snap"][-1])
        assert np.allclose(F_eng, st.F_hat[b].numpy(), atol=1e-11), \
            f"seed {seeds[b]}: F_hat diverges from run_batch"
        assert np.allclose(Fp_eng, st.Fprime_hat[b].numpy(), atol=1e-11)
        assert np.allclose(X_eng, st.X[b].numpy(), atol=1e-11), \
            f"seed {seeds[b]}: trajectory diverges from run_batch"


# --------------------------------------------------------------------------- #
# FEC-blind dose selection
# --------------------------------------------------------------------------- #
def _passing_flags():
    spec = importlib.util.spec_from_file_location(
        "aic", os.path.join(SRC, "scripts", "analyze_info_conversion.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_dose_selection_refuses_fec_columns():
    mod = _passing_flags()
    t = pd.DataFrame([{"p90": 0.05, "pass_both": True, "e_F_final": 0.1}])
    with pytest.raises(ValueError, match="blind"):
        mod.select_dose(t)


def test_dose_selection_takes_the_smallest_passing_dose():
    mod = _passing_flags()
    t = pd.DataFrame([
        {"p90": 0.02, "pass_both": False},
        {"p90": 0.05, "pass_both": True},
        {"p90": 0.10, "pass_both": True},
    ])
    assert mod.select_dose(t) == 0.05
    t["pass_both"] = False
    assert mod.select_dose(t) is None


def test_gate_flags_logic_on_synthetic_runs():
    mod = _passing_flags()
    rows = []
    for cell in ("K2", "K3"):
        for s in range(4):
            rows.append(dict(cell=cell, seed=s, arm="abf", p90=np.nan,
                             R_s=1.0, tv_future=0.5, kl_pre=np.nan,
                             kl_post=np.nan, ess_anc=np.nan, n_events=np.nan))
            rows.append(dict(cell=cell, seed=s, arm="p0.05", p90=0.05,
                             R_s=0.5, tv_future=0.2, kl_pre=1.0, kl_post=0.5,
                             ess_anc=0.95, n_events=10))
    df = pd.DataFrame(rows)
    runs = {c: df[df.cell == c] for c in ("K2", "K3")}
    flags = mod.gate_flags(mod.gate_table(runs), ("K2", "K3"))
    assert bool(flags.pass_both.iloc[0])
    assert mod.select_dose(flags) == 0.05
