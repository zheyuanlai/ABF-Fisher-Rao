"""Tests of src/gateway_family_numba.py, the mechanism-campaign engine (docs/mechanism/SCIENTIFIC_PLAN.md section 4).

F1  model: analytic gradient == central differences for every variant, lam does not enter it; the kernel's deposits
    are the analytic force and one kernel step is the closed-form EM update (every dynamics, lam 1 / 0.5 / 0.25 / 0.1);
F2  exact identities by quadrature in y: F_model - F* constant, E[f | x] = F*', Var(f | x) exact, the Y | x law;
F3  init_family: init_left's draws, bitwise init_left at alpha 1, exact conditional law at every x, z coupled across
    variants;
F4  integrator: frozen-x EM conditional variance (and the shift's conditional mean) vs the closed form, lam 1 and 0.1;
F5  BITWISE gateway_ladder_numba at (alpha, alpha 1, lam 1): both arms, N 2 / 128 / 2048, every diagnostic on, for
    production and FR-active knobs, and for the two campaigns' actual configs;
F6  pairing and randomness: FR(gamma 0) == ABF bitwise for every dynamics; FR stream separation (MT and PCG64
    positions at a checkpoint, chunk independence); the FR helpers equal gateway_ladder_numba's;
F7  bitwise checkpoint/resume for a non-default dynamics (shift, lam 0.25) at an FR step + snapshot save, off-grid
    chunk boundaries, a stale checkpoint, and in fresh processes;
F8  diagnostics inert for every dynamics (on/off bitwise; other save / trace / snapshot grids and chunk sizes);
F9  D1-D4 == an independent pure-Python reference BITWISE (every class exercised), plus partition invariants;
F10 accounting (n_force_evals = N n_steps), FR bookkeeping, run_job / signature (model parameters) / result I/O, cfg
    validation, meta.model;
F11 population-size sanity: with the ABF bias off and no FR, the per-walker one-step law does not depend on N;
F12 the driver scripts/mechanism/run_cells.py: job enumeration and budget, a tiny smoke job end to end, refusal of a
    complete result of a DIFFERENT run (signature, not status), lam1 run once as an alias of alpha1, the reuse rule;
F13 speed printout (ns per walker-step, N 512, both arms, every variant, diagnostics on).

CPU only:  CUDA_VISIBLE_DEVICES="" NUMBA_CACHE_DIR=... python -m pytest tests/test_gateway_family_numba.py -q
Slow tests: RUN_SLOW=1.
"""
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import Counter

import numpy as np
import pytest
from scipy import stats

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import gateway_ladder_numba as gl     # noqa: E402
import gateway_family_numba as gf     # noqa: E402

slow = pytest.mark.skipif(os.environ.get("RUN_SLOW") != "1", reason="slow; set RUN_SLOW=1")

CELL = dict(beta=16.0, H=0.5, omega_out=1.0, omega_in=32.0, s=0.1)
EASY = dict(beta=1.0, H=0.1, omega_out=1.0, omega_in=4.0, s=0.1)    # frequent crossings, many FR events

VARIANTS = dict(alpha1=dict(variant="alpha", alpha=1.0), alpha05=dict(variant="alpha", alpha=0.5),
                alpha0=dict(variant="alpha", alpha=0.0), shift=dict(variant="shift", kappa=1.0))
DYN7 = dict(VARIANTS, lam05=dict(variant="alpha", alpha=1.0, lam=0.5), lam025=dict(variant="alpha", alpha=1.0, lam=0.25),
            lam01=dict(variant="alpha", alpha=1.0, lam=0.1))

DYN_KEYS = ["M_all", "C_all", "X_final", "Y_final", "fr_kd_cum", "fr_kc_cum", "fr_deaths_cum", "fr_opp_cum",
            "fr_opp_event_cum", "fr_opp_death_cum", "fr_events_hist", "fr_events_hist_cum", "fr_deaths_hist",
            "fr_deaths_hist_cum", "max_abs_bias", "n_force_evals", "save_step"]
NEW_KEYS = ["fr_death_pos_hist_cum", "fr_birth_pos_hist_cum", "snap_step", "snap_t", "snap_u", "dep_age_C",
            "dep_age_M", "dep_age_M2", "dep_cross_C", "traces_y"]
DIAG_KEYS = ["hist_inst", "region_frac", "events_cum", "lineage_visited", "first_arrival_step",
             "gen_nuniq_run", "gen_ess_run", "gen_maxfam_run", "gen_nuniq_win", "gen_ess_win", "gen_maxfam_win",
             "gen_win_age_steps", "traces", "traces_anc", "traces_rebirth", "traces_step"] + NEW_KEYS
NOT_COMPARED = ("cfg", "meta", "wall_s", "peak_rss_mb")


def assert_same(a, b, keys):
    for k in keys:
        assert np.array_equal(np.asarray(a[k]), np.asarray(b[k]), equal_nan=True), k


def np_reflect(q, lo=gf.XMIN, hi=gf.XMAX):
    span = hi - lo
    qm = np.mod(q - lo, 2.0 * span)
    return np.where(qm > span, 2.0 * span - qm, qm) + lo


# ------------------------------------------------------------------------------------------------- F1
@pytest.mark.parametrize("name", list(VARIANTS))
def test_f1_gradient_matches_finite_differences(name):
    cfg = dict(VARIANTS[name])
    rng = np.random.default_rng(1)
    x = np.concatenate([np.linspace(-1.7, 1.7, 69), rng.uniform(-0.4, 0.4, 300), [0.0, 0.21, -0.21]])
    mu, var = gf.cond_y_law(cfg, x)
    y = mu + 3.0 * np.sqrt(var) * rng.uniform(-1.0, 1.0, x.size)
    gx, gy = gf.grad_V(cfg, x, y)
    d = 1e-6
    fdx = (gf.potential(cfg, x + d, y) - gf.potential(cfg, x - d, y)) / (2 * d)
    fdy = (gf.potential(cfg, x, y + d) - gf.potential(cfg, x, y - d)) / (2 * d)
    assert np.max(np.abs(gx - fdx) / (1.0 + np.abs(gx))) < 2e-7, name
    assert np.max(np.abs(gy - fdy) / (1.0 + np.abs(gy))) < 2e-7, name
    # lam does not enter the gradient (nor the potential)
    for lam in (0.5, 0.25, 0.1):
        gx2, gy2 = gf.grad_V(dict(cfg, lam=lam), x, y)
        assert np.array_equal(gx, gx2) and np.array_equal(gy, gy2)
    # alpha 1 is the original gateway potential
    if name == "alpha1":
        om = gf.omega_np(cfg, x)
        assert np.allclose(gf.potential(cfg, x, y), 0.5 * (x * x - 1.0) ** 2 + 0.5 * om * om * y * y, rtol=1e-13)


@pytest.mark.parametrize("name", list(DYN7))
def test_f1_kernel_force_and_one_em_step(name):
    """One walker per production bin, ONE step: M_all[1] holds each walker's deposited force (C = 1), which must be
    the analytic dV/dx; X/Y after the step must be the closed-form EM update with the Langevin stream's normals
    (N zx then N zy) and the bias M/(C + min_count) = f/2."""
    cfg = dict(DYN7[name], h=2.5e-5)
    c = gf.full_cfg(cfg)
    nb = 180
    delta = (gf.XMAX - gf.XMIN) / nb
    rng = np.random.default_rng(2)
    x0 = gf.XMIN + (np.arange(nb) + rng.uniform(0.1, 0.9, nb)) * delta
    mu, var = gf.cond_y_law(c, x0)
    y0 = mu + 2.0 * np.sqrt(var) * rng.normal(size=nb)
    res = gf.run_arm(cfg, "abf", nb, 7100, 1, [0, 1], x0=x0, y0=y0, trace_steps=[0, 1])
    assert np.array_equal(res["C_all"][1], np.ones(nb)) and np.all(res["M_all"][0] == 0)
    fk = res["M_all"][1]
    gx, gy = gf.grad_V(c, x0, y0)
    assert np.max(np.abs(fk - gx) / (1.0 + np.abs(gx))) < 1e-12
    gf._mt_seed(gf.default_noise_seed(7100, nb))
    Z = gf._mt_normals(2 * nb)
    zx, zy = Z[:nb], Z[nb:]
    h, beta, lam = c["h"], c["beta"], c["lam"]
    x1 = np_reflect(x0 + (-fk + fk / (1.0 + c["min_count"])) * h + math.sqrt(2.0 * h / beta) * zx)
    y1 = y0 - lam * gy * h + math.sqrt(2.0 * lam * h / beta) * zy
    assert np.max(np.abs(res["X_final"] - x1)) < 1e-13
    assert np.max(np.abs(res["Y_final"] - y1) / (1.0 + np.abs(y1))) < 1e-13
    # lam does not enter the deposited force: the same state deposits the same bits at lam 1
    r1 = gf.run_arm(dict(cfg, lam=1.0), "abf", nb, 7100, 1, [0, 1], x0=x0, y0=y0, trace_steps=[0, 1])
    assert np.array_equal(r1["M_all"], res["M_all"]) and np.array_equal(r1["X_final"], res["X_final"])
    if c["lam"] != 1.0:
        assert not np.array_equal(r1["Y_final"], res["Y_final"])


# ------------------------------------------------------------------------------------------------- F2
@pytest.mark.parametrize("name", list(VARIANTS))
def test_f2_exact_identities_by_quadrature(name):
    c = gf.full_cfg(VARIANTS[name])
    beta = c["beta"]
    x = np.concatenate([np.linspace(-1.5, 1.5, 61), [-0.21, 0.21, 0.07, -0.33]])
    y = np.linspace(-3.0, 4.0, 140001)            # dy 5e-5 resolves the narrowest fibre (sd 0.0078, alpha 1 gate)
    dy = y[1] - y[0]
    logZ, Ef, Vf, Ey, Vy = (np.empty(x.size) for _ in range(5))
    for i, xi in enumerate(x):
        V = gf.potential(c, xi, y)
        vmin = V.min()
        w = np.exp(-beta * (V - vmin))
        Z = np.trapezoid(w, dx=dy)
        logZ[i] = math.log(Z) - beta * vmin
        f, _ = gf.grad_V(c, np.full_like(y, xi), y)
        Ef[i] = np.trapezoid(f * w, dx=dy) / Z
        Vf[i] = np.trapezoid((f - Ef[i]) ** 2 * w, dx=dy) / Z
        Ey[i] = np.trapezoid(y * w, dx=dy) / Z
        Vy[i] = np.trapezoid((y - Ey[i]) ** 2 * w, dx=dy) / Z
    Fm = -logZ / beta
    diff = Fm - gf.F_star(c, x)
    assert np.ptp(diff) < 1e-10, np.ptp(diff)                               # F_model = F* + C
    dF = gf.dF_star(c, x)
    assert np.max(np.abs(Ef - dF) / (1.0 + np.abs(dF))) < 1e-10             # E[f | x] = F*'
    ve = gf.cond_force_var(c, x)
    if name == "alpha0":
        assert np.all(ve == 0.0) and np.max(Vf) < 1e-24                     # f does not depend on y
    else:
        big = ve > 1e-6
        assert big.sum() > 20 and np.max(np.abs(Vf[big] / ve[big] - 1.0)) < 1e-8
        assert np.max(np.abs(Vf[~big] - ve[~big])) < 1e-12
    if name == "shift":                                                     # the variance-matched control
        assert np.allclose(ve, gf.cond_force_var(VARIANTS["alpha1"], x), rtol=1e-14, atol=0)
    mu, vy = gf.cond_y_law(c, x)
    assert np.max(np.abs(Ey - mu)) < 1e-10 and np.max(np.abs(Vy / vy - 1.0)) < 1e-9
    # the exact max of Var(f|x) quoted in the plan (2.06 at |x| = 0.21 for alpha 1 and the shift)
    xg = np.linspace(-1.8, 1.8, 360001)
    vg = gf.cond_force_var(c, xg)
    if name in ("alpha1", "shift"):
        assert abs(vg.max() - 2.06) < 0.01 and abs(abs(xg[np.argmax(vg)]) - 0.21) < 0.01
    elif name == "alpha05":
        assert abs(vg.max() / gf.cond_force_var(VARIANTS["alpha1"], xg).max() - 0.25) < 1e-12
    # secondary reference: EM-consistent mean force -> F*' as h -> 0; the shift's is F*' exactly
    assert np.allclose(gf.em_mean_force(dict(c, h=1e-12), xg), gf.dF_star(c, xg), rtol=0, atol=1e-9)
    if name == "shift":
        assert np.array_equal(gf.em_mean_force(c, xg), gf.dF_star(c, xg))


# ------------------------------------------------------------------------------------------------- F3
def test_f3_init_family_draws_law_and_coupling():
    for seed, N in ((7100, 1), (8100, 2048), (5, 37), (8131, 128)):
        xr, yr = gl.init_left(seed, N)
        for cfg in (None, dict(variant="alpha", alpha=1.0), dict(variant="alpha", alpha=1.0, lam=0.1)):
            x, y = gf.init_family(seed, N, cfg)
            assert np.array_equal(x, xr) and np.array_equal(y, yr)
        assert np.array_equal(gf.init_left(seed, N)[1], yr)
    N = 200_000
    zref = None
    for name, cfg in DYN7.items():
        x, y = gf.init_family(8100, N, cfg)
        assert np.array_equal(x, gl.init_left(8100, N)[0])
        mu, var = gf.cond_y_law(cfg, x)
        z = (y - mu) / np.sqrt(var)
        assert abs(z.mean()) < 4.0 / math.sqrt(N) and abs(z.var() - 1.0) < 4.0 * math.sqrt(2.0 / N), name
        assert stats.kstest(z, "norm").pvalue > 1e-4, name
        if zref is None:
            zref = z
        else:
            assert np.max(np.abs(z - zref)) < 1e-11, name                    # the SAME z in every cell
    # the map is the exact conditional law at EVERY x (the start only visits the left well)
    rng = np.random.default_rng(3)
    xg = np.linspace(-1.8, 1.8, 3601)
    z = rng.normal(size=xg.size)
    for name, cfg in DYN7.items():
        mu, var = gf.cond_y_law(cfg, xg)
        assert np.max(np.abs((gf.y_from_z(cfg, xg, z) - mu) / np.sqrt(var) - z)) < 1e-11, name
    # the shift's centre crosses 4.9 fibre sds at the gate (plan section 1.1)
    m0 = gf.m_shift(VARIANTS["shift"], 0.0)
    assert abs(m0 - 1.225) < 1e-3 and abs(m0 / 0.25 - 4.9) < 0.01


# ------------------------------------------------------------------------------------------------- F4
@pytest.mark.parametrize("name,x_star,lam,prod", [("alpha1", 0.0, 1.0, 0.2), ("alpha1", 0.0, 0.1, 0.2),
                                                  ("alpha05", 0.0, 1.0, 0.2), ("alpha05", 0.0, 0.1, 0.2),
                                                  ("alpha1", -1.0, 0.1, 0.2), ("shift", 0.0, 1.0, 0.05),
                                                  ("shift", 0.0, 0.1, 0.05), ("alpha0", 0.0, 0.25, 0.2)])
def test_f4_frozen_x_em_conditional_law(name, x_star, lam, prod):
    """x frozen (test hook) at x_star: the kernel's y-integrator must reach the EM stationary law N(m, (1/(beta k)) /
    (1 - lam k h / 2)).  h is chosen so that lam k h = prod (inflation 11 % at 0.2: resolvable with 16384 walkers).
    The shift starts at y = 0, 4.9 sd off its centre, so its drift (sign and lam scaling) is tested too."""
    base = dict(VARIANTS[name], lam=lam)
    k = float(gf.stiffness(base, x_star))
    h = prod / (lam * k)
    cfg = dict(base, h=h, fr_interval_t=h)
    N = 16384
    n = int(round(30.0 / prod))
    x0 = np.full(N, float(x_star))
    mu, var = gf.cond_y_law(cfg, x0)
    rng = np.random.default_rng(4)
    y0 = np.zeros(N) if name == "shift" else mu + np.sqrt(var) * rng.normal(size=N)
    res = gf.run_arm(cfg, "abf", N, 7100, n, [n], x0=x0, y0=y0, trace_steps=[0, n], _freeze_x_test=True)
    assert np.array_equal(res["X_final"], x0)
    Y = res["Y_final"]
    infl = 1.0 / (1.0 - prod / 2.0)
    assert abs(float(gf.em_var_inflation(cfg, x_star)) - infl) < 1e-9
    v_em = var[0] * infl
    se = math.sqrt(2.0 / N)
    assert abs(Y.var() / v_em - 1.0) < 4.0 * se, (Y.var(), v_em, var[0])
    if infl - 1.0 > 6.0 * se:
        assert abs(Y.var() / var[0] - 1.0) > 3.0 * se                       # the test can tell EM from exact
    assert abs(Y.mean() - mu[0]) < 4.0 * math.sqrt(v_em / N)


# ------------------------------------------------------------------------------------------------- F5
@pytest.mark.parametrize("method", ["abf", "fr"])
@pytest.mark.parametrize("N,n_steps", [(2, 40000), (128, 6000), (2048, 1200)])
def test_f5_bitwise_original_at_alpha1_lam1(method, N, n_steps):
    h = 2.5e-5
    save = gl.prereg_save_grid(n_steps, h)
    assert np.array_equal(save, gf.prereg_save_grid(n_steps, h))
    for knobs in (dict(), dict(gamma=15.0, ramp_t=0.05)):           # production knobs, and FR-active ones
        cfg = dict(h=h, **knobs)
        a = gl.run_arm(cfg, method, N, 8100, n_steps, save, chunk_steps=777)
        b = gf.run_arm(dict(cfg, variant="alpha", alpha=1.0, kappa=None, lam=1.0), method, N, 8100, n_steps, save,
                       chunk_steps=1013)
        keys = [k for k in a if k not in NOT_COMPARED]
        assert len(keys) >= 40 and set(keys) <= set(b)
        bad = [k for k in keys if not np.array_equal(np.asarray(a[k]), np.asarray(b[k]), equal_nan=True)]
        assert not bad, bad
        assert set(b) - set(a) == set(NEW_KEYS)
        if method == "fr" and knobs:
            assert b["fr_deaths_cum"][-1] > 0, "test must exercise realised FR deaths"


def _mech_cfg(experiment, name):
    P = json.load(open(os.path.join(ROOT, "configs", "mechanism", experiment + ".json")))
    v = [v for v in P["variants"] if v["name"] == name][0]
    return dict(P["engine_cfg"], variant=v["variant"], alpha=v["alpha"], kappa=v["kappa"], lam=v["lam"]), P


def test_f5_campaign_configs_reproduce_equal_budget_gateway():
    """The reuse premise (plan section 5): the alpha1 / lam1 cells of BOTH campaign configs run the equal-budget
    gateway production config bitwise (short horizon; the full-length check of >= 4 production files is a separate
    preregistered step)."""
    eqb = json.load(open(os.path.join(ROOT, "configs", "equal_budget_v2", "gateway_production.json")))
    for exp, name in (("matched_free_energy", "alpha1"), ("conditional_relaxation", "lam1")):
        cm, P = _mech_cfg(exp, name)
        assert {k: v for k, v in P["engine_cfg"].items() if k != "system"} == \
            {k: v for k, v in eqb["engine_cfg"].items() if k != "system"}
        assert P["seeds"] == eqb["seeds"] and P["physical_checkpoints_t"] == eqb["physical_checkpoints_t"]
        assert gf.model_params(cm)["model_id"] == gf.MODEL_ORIG and gf.model_params(cm)["lam1"]
        for N, n in ((2048, 800), (128, 3000)):
            for method in ("abf", "fr"):
                a = gl.run_arm(eqb["engine_cfg"], method, N, 8100, n, gl.prereg_save_grid(n, 2.5e-5))
                b = gf.run_arm(cm, method, N, 8100, n, gf.prereg_save_grid(n, 2.5e-5))
                assert_same(a, b, [k for k in a if k not in NOT_COMPARED])


# ------------------------------------------------------------------------------------------------- F6
PAIR_KEYS = ["M_all", "C_all", "X_final", "Y_final", "max_abs_bias", "n_force_evals", "hist_inst", "region_frac",
             "events_cum", "lineage_visited", "first_arrival_step", "traces", "traces_y", "traces_anc",
             "traces_rebirth", "dep_age_C", "dep_age_M", "dep_age_M2", "dep_cross_C", "fr_death_pos_hist_cum",
             "fr_birth_pos_hist_cum"]


@pytest.mark.parametrize("name", list(DYN7))
def test_f6_fr_gamma0_equals_abf(name):
    h, N, n = 4e-4, 32, 6000
    cfg = dict(EASY, **DYN7[name], h=h, gamma=15.0)
    sv, _ = gf.budget_save_grid(n)
    abf = gf.run_arm(cfg, "abf", N, 7100, n, sv)
    fr0 = gf.run_arm(dict(cfg, gamma=0.0), "fr", N, 7100, n, sv)
    assert_same(abf, fr0, PAIR_KEYS)
    assert fr0["fr_opp_cum"][-1] == 600 and fr0["fr_deaths_cum"][-1] == 0
    assert abf["events_cum"][-1].min() > 0
    fr = gf.run_arm(cfg, "fr", N, 7100, n, sv)
    pre = fr["fr_opp_event_cum"] == 0
    assert pre.sum() >= 3 and (~pre).sum() >= 3
    assert np.array_equal(fr["M_all"][pre], abf["M_all"][pre])
    assert not np.array_equal(fr["M_all"][~pre], abf["M_all"][~pre])


def test_f6_stream_separation_and_chunk_independence(tmp_path):
    """Langevin noise: exactly 2N MT normals per step, whatever the arm; FR uniforms: PCG64(SeedSequence([seed, N,
    1])) advanced by exactly L doubles per opportunity (fixed blocks), whatever the chunking."""
    h, N, n, k = 4e-4, 24, 3000, 1234
    cfg = dict(EASY, variant="shift", kappa=1.0, lam=0.25, h=h, gamma=15.0)
    kn = gf.derived_knobs(gf.full_cfg(cfg), N)
    for method in ("abf", "fr"):
        ck = str(tmp_path / f"{method}.ck.npz")
        with pytest.raises(gf.RunInterrupted):
            gf.run_arm(cfg, method, N, 7100, n, [n], chunk_steps=k, checkpoint_path=ck, checkpoint_every_s=0.0,
                       stop_after_chunks=1)
        with np.load(ck) as z:
            assert int(z["ictr"][gf.I_STEP]) == k
            gf._mt_seed(gf.default_noise_seed(7100, N))
            gf._mt_normals(2 * N * k)
            idx, key = gf.mt_get()
            assert int(z["mt_index"]) == idx and np.array_equal(z["mt_key"], key)
            rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([7100, N, 1])))
            if method == "fr":
                rng.random(-(-k // kn["fr_every"]) * kn["fr_block"])
            assert json.loads(str(z["fr_state_json"])) == rng.bit_generator.state
    ref = gf.run_arm(cfg, "fr", N, 7100, n, gf.budget_save_grid(n)[0])
    assert ref["fr_deaths_cum"][-1] > 10
    for ch in (1, 37, 1234, 10 ** 6):
        other = gf.run_arm(cfg, "fr", N, 7100, n, gf.budget_save_grid(n)[0], chunk_steps=ch)
        assert_same(ref, other, DYN_KEYS + DIAG_KEYS)


@pytest.mark.parametrize("N", [2, 3, 37, 2048])
def test_f6_fr_helpers_equal_original(N):
    rng = np.random.default_rng(N)
    X = np.concatenate([rng.uniform(gf.XMIN, gf.XMAX, max(N - 3, 0)), [gf.XMIN, gf.XMAX, -1.79]])[:N]
    dx = gf.grid_dx()
    kern, r = gf.gaussian_kernel_np(0.1, dx)
    k2, r2 = gl.gaussian_kernel_np(0.1, dx)
    assert dx == gl.grid_dx() and r == r2 and np.array_equal(kern, k2)
    p1, h1, p2, h2 = (np.zeros(gf.N_GRID) for _ in range(4))
    assert gf.kde_density(X, N, kern, r, dx, p1, h1) == gl.kde_density(X, N, kern, r, dx, p2, h2)
    assert np.array_equal(p1, p2)
    S1, S2 = np.empty(N), np.empty(N)
    assert gf.uniform_scores(X, N, p1, dx, 3.0, S1) == gl.uniform_scores(X, N, p2, dx, 3.0, S2)
    assert np.array_equal(S1, S2)
    cap = max(1, int(math.floor(0.08 * N)))
    L = N + cap + max(N, cap)
    for g in (1.5, 40.0, 400.0):
        for rep in range(10):
            S = np.clip(rng.normal(0.0, 2.0, N), -3.0, 3.0)
            U = rng.random(L)
            outs = []
            for E in (gf, gl):
                sel, dc, cc, pool = (np.zeros(N, np.int64), np.zeros(N, np.int64), np.zeros(N, np.int64),
                                     np.zeros(2 * N, np.int64))
                kd, kc, nu = E.fr_resample(S, N, g, 0.004, cap, sel, dc, cc, pool, U, 0, False)
                outs.append((kd, kc, nu, sel.copy() if kd + kc > 0 else None))
            assert outs[0][:3] == outs[1][:3]
            assert (outs[0][3] is None) == (outs[1][3] is None)
            if outs[0][3] is not None:
                assert np.array_equal(outs[0][3], outs[1][3])


# ------------------------------------------------------------------------------------------------- F7
def _run(cfg, method, N, n_steps, save_at, trace_at, **kw):
    return gf.run_arm(cfg, method, N, 7100, n_steps, save_at, trace_steps=trace_at, **kw)


@pytest.mark.parametrize("method", ["abf", "fr"])
def test_f7_checkpoint_resume_bitwise_shift_lam025(method, tmp_path):
    h, N, n_steps = 4e-4, 16, 9000          # fr_every 10, window 4000, ramp 10000; snapshots 90 ... 2250 ... 9000
    cfg = dict(EASY, variant="shift", kappa=1.0, lam=0.25, h=h, gamma=15.0,
               clone_age_edges_t=[0.004, 0.04, 0.4], cross_age_edges_t=[0.04, 0.4, 1.2])
    save_at, _ = gf.budget_save_grid(n_steps)
    trace_at = np.arange(0, n_steps + 1, 50)
    assert 2250 in gf.snapshot_grid(n_steps) and 2250 % 10 == 0
    full = _run(cfg, method, N, n_steps, save_at, trace_at)
    if method == "fr":
        assert full["fr_deaths_cum"][-1] > 0
    ck = str(tmp_path / "ck.npz")
    # chunks of 450: stop after 5 chunks = step 2250 = an FR step AND a snapshot save
    with pytest.raises(gf.RunInterrupted):
        _run(cfg, method, N, n_steps, save_at, trace_at, chunk_steps=450, checkpoint_path=ck,
             checkpoint_every_s=0.0, stop_after_chunks=5)
    with np.load(ck) as z:
        assert int(z["ictr"][gf.I_STEP]) == 2250 and int(z["ictr"][gf.I_NPTR]) == 4
        assert {"clone_step", "cross_step", "dagC", "o_sC", "tr_y", "o_dpos"} <= set(z.files)
    gf._mt_seed(987654)
    stale = str(tmp_path / "stale.npz")
    shutil.copy(ck, stale)
    with pytest.raises(gf.RunInterrupted):
        _run(cfg, method, N, n_steps, save_at, trace_at, chunk_steps=37, checkpoint_path=ck,
             checkpoint_every_s=0.0, stop_after_chunks=3)
    gf._mt_seed(1)
    res = _run(cfg, method, N, n_steps, save_at, trace_at, chunk_steps=1001, checkpoint_path=ck, checkpoint_every_s=0.0)
    assert_same(res, full, DYN_KEYS + DIAG_KEYS)
    assert list(res["resumed_from"]) == [2250, 2361] and res["meta"]["status"] == "complete"
    ss = res["meta"]["sessions"]
    assert [x["start_step"] for x in ss] == [0, 2250, 2361] and ss[-1]["end_step"] == n_steps
    res2 = _run(cfg, method, N, n_steps, save_at, trace_at, chunk_steps=500, checkpoint_path=stale)
    assert_same(res2, full, DYN_KEYS + DIAG_KEYS)
    for bad in (dict(lam=0.5), dict(kappa=2.0), dict(variant="alpha", alpha=1.0, kappa=None)):
        with pytest.raises(ValueError):
            _run(dict(cfg, **bad), method, N, n_steps, save_at, trace_at, checkpoint_path=stale)


_SUB = r"""
import sys, numpy as np
sys.path.insert(0, {src!r})
import gateway_family_numba as gf
cfg = dict(beta=1.0, H=0.1, omega_out=1.0, omega_in=4.0, s=0.1, h=4e-4, gamma={gamma}, variant="shift", kappa=1.0,
           lam=0.25)
save_at = np.arange(0, 3001, 100)
try:
    res = gf.run_arm(cfg, {method!r}, 16, 7100, 3000, save_at, trace_steps=np.arange(0, 3001, 25), chunk_steps=160,
                     checkpoint_path={ck!r}, checkpoint_every_s=0.0, stop_after_chunks={stop})
except gf.RunInterrupted:
    sys.exit(3)
gf.save_result({out!r}, res)
"""


def _subrun(method, gamma, ck, out, stop):
    code = _SUB.format(src=os.path.join(ROOT, "src"), method=method, gamma=gamma, ck=ck, out=out, stop=stop)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", NUMBA_NUM_THREADS="1", MKL_NUM_THREADS="1")
    return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=600)


def test_f7_separate_processes(tmp_path):
    cfg = dict(EASY, h=4e-4, gamma=15.0, variant="shift", kappa=1.0, lam=0.25)
    save_at = np.arange(0, 3001, 100); trace_at = np.arange(0, 3001, 25)
    full = gf.run_arm(cfg, "fr", 16, 7100, 3000, save_at, trace_steps=trace_at)
    ck, out = str(tmp_path / "fr.ck.npz"), str(tmp_path / "fr.npz")
    p1 = _subrun("fr", 15.0, ck, out, 7)
    assert p1.returncode == 3, p1.stderr
    p2 = _subrun("fr", 15.0, ck, out, None)
    assert p2.returncode == 0, p2.stderr
    res = gf.load_result(out)
    assert_same(res, full, DYN_KEYS + DIAG_KEYS)
    assert list(res["resumed_from"]) == [1120]
    pa = _subrun("abf", 15.0, str(tmp_path / "a.ck.npz"), str(tmp_path / "abf.npz"), None)
    pf = _subrun("fr", 0.0, str(tmp_path / "f.ck.npz"), str(tmp_path / "fr0.npz"), None)
    assert pa.returncode == 0 and pf.returncode == 0, (pa.stderr, pf.stderr)
    a, f = gf.load_result(str(tmp_path / "abf.npz")), gf.load_result(str(tmp_path / "fr0.npz"))
    assert_same(a, f, PAIR_KEYS)


# ------------------------------------------------------------------------------------------------- F8
@pytest.mark.parametrize("name", list(DYN7))
def test_f8_diagnostics_inert(name):
    h, N, n_steps = 4e-4, 32, 6000
    cfg = dict(EASY, **DYN7[name], h=h, gamma=15.0)
    gA, _ = gf.budget_save_grid(n_steps)
    gB = np.unique(np.r_[np.arange(37, n_steps + 1, 37), gA[::3], n_steps])
    tA, tB = np.arange(0, n_steps + 1, 40), np.arange(0, n_steps + 1, 7)
    sA, sB = gf.snapshot_grid(n_steps), np.unique(np.r_[gf.snapshot_grid(n_steps)[::2], 1, 333, 4444])
    for method in ("abf", "fr"):
        on = gf.run_arm(cfg, method, N, 7100, n_steps, gA, trace_steps=tA, snapshot_steps=sA)
        off = gf.run_arm(cfg, method, N, 7100, n_steps, gA, trace_steps=tA, snapshot_steps=sA, diagnostics=False)
        assert_same(on, off, DYN_KEYS)
        assert not any(k in off for k in gf.DIAG_ONLY_KEYS) and all(k in on for k in gf.DIAG_ONLY_KEYS)
        assert off["meta"]["omitted_keys"] == list(gf.DIAG_ONLY_KEYS)
        other = gf.run_arm(cfg, method, N, 7100, n_steps, gB, trace_steps=tB, snapshot_steps=sB, chunk_steps=999)
        assert_same(on, other, ["X_final", "Y_final", "max_abs_bias", "n_force_evals", "fr_events_hist"])
        _, ia, ib = np.intersect1d(gA, gB, return_indices=True)
        assert len(ia) > 20
        for k in ("M_all", "C_all", "hist_inst", "region_frac", "events_cum", "gen_ess_run", "fr_deaths_cum",
                  "fr_death_pos_hist_cum", "fr_birth_pos_hist_cum"):
            assert np.array_equal(on[k][ia], other[k][ib], equal_nan=True), k
        _, ja, jb = np.intersect1d(tA, tB, return_indices=True)
        for k in ("traces", "traces_y", "traces_anc", "traces_rebirth"):
            assert np.array_equal(on[k][ja], other[k][jb]), k
        _, ka, kb = np.intersect1d(sA, sB, return_indices=True)
        assert len(ka) >= 3
        for k in ("dep_age_C", "dep_age_M", "dep_age_M2", "dep_cross_C"):
            assert np.array_equal(on[k][ka], other[k][kb]), k
        if method == "fr":
            assert on["fr_deaths_cum"][-1] > 0


# ------------------------------------------------------------------------------------------------- F9
def _py_resample(S, N, g, dt_fr, cap, U):
    it = iter(U)
    die, clo = [], []
    for i in range(N):
        u = next(it)
        if S[i] > 0.0:
            if u < 1.0 - math.exp(-g * S[i] * dt_fr):
                die.append(i)
        elif S[i] < 0.0:
            if u < 1.0 - math.exp(g * S[i] * dt_fr):
                clo.append(i)
    nd, nc = len(die), len(clo)
    if nd + nc > cap:
        kd = min(int(round(cap * nd / max(nd + nc, 1))), nd)
        kc = min(cap - kd, nc)
    else:
        kd, kc = nd, nc
    if kd + kc == 0:
        return 0, 0, None
    for lst, k in ((die, kd), (clo, kc)):
        n = len(lst)
        for t in range(k):
            j = min(t + int(next(it) * (n - t)), n - 1)
            lst[t], lst[j] = lst[j], lst[t]
    dead = set(die[:kd])
    pool = [i for i in range(N) if i not in dead]
    n_surv = len(pool)
    pool += clo[:kc]
    P = len(pool)
    if P >= N:
        for t in range(N):
            j = min(t + int(next(it) * (P - t)), P - 1)
            pool[t], pool[j] = pool[j], pool[t]
        sel = pool[:N]
    else:
        sel = pool + [pool[min(int(next(it) * n_surv), n_surv - 1)] for _ in range(N - P)]
    return kd, kc, sel


def _py_reflect(q, lo, hi):
    span = hi - lo
    qm = (q - lo) % (2.0 * span)
    if qm > span:
        qm = 2.0 * span - qm
    return qm + lo


def py_reference(cfg, method, N, seed, n_steps, save_at, trace_at, snap_at, x0=None, y0=None):
    """Independent pure-Python reference of one arm of the family engine, including D1-D4 (scalar math / Python
    floats; the model arithmetic written from SCIENTIFIC_PLAN section 1 in the kernel's operation order)."""
    c = gf.full_cfg(cfg)
    kn = gf.derived_knobs(c, N)
    h, beta, H, oout, oin, s = (float(c[k]) for k in ("h", "beta", "H", "omega_out", "omega_in", "s"))
    lam = float(c["lam"])
    nb, mc = int(c["nb"]), float(c["min_count"])
    delta = (gf.XMAX - gf.XMIN) / nb
    fe, W, cap, L, ramp = kn["fr_every"], kn["window_steps"], kn["cap"], kn["fr_block"], kn["ramp_steps"]
    dt_fr = h * fe
    ntr = min(N, 32)
    var = c["variant"]
    orig = var == "alpha" and c["alpha"] == 1.0
    if var == "alpha":
        al = c["alpha"]
    else:
        kap = c["kappa"]
        msc = math.sqrt(2.0 / beta) / kap

    def edges(e):
        out = []
        for v in e:
            r = v / h
            out.append(int(round(r)) if abs(r - round(r)) < 1e-6 else int(math.ceil(r)))
        return out
    ath, cth = edges(c["clone_age_edges_t"]), edges(c["cross_age_edges_t"])

    def klass(ref, k, th):
        if ref < 0:
            return 4
        d = k - ref
        return 0 if d < th[0] else (1 if d < th[1] else (2 if d < th[2] else 3))

    def binof(v):
        return min(max(int(math.floor((v - gf.XMIN) / delta + 1e-9)), 0), nb - 1)

    if x0 is None:
        x0, y0 = gf.init_family(seed, N, c)
    x, y = [float(v) for v in x0], [float(v) for v in y0]
    gf._mt_seed(gf.default_noise_seed(seed, N))
    Z = gf._mt_normals(2 * N * n_steps)
    U = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, N, 1]))).random(-(-n_steps // fe) * L)
    dx = gl.grid_dx()
    kern, r = gl.gaussian_kernel_np(c["eta"], dx)
    M, C = [0.0] * nb, [0.0] * nb
    lab = [-1 if v < -0.5 else (1 if v > 0.5 else 0) for v in x]
    vis = [1 if v > 0.5 else 0 for v in x]
    anc_r, anc_w = list(range(N)), list(range(N))
    wid, reborn = list(range(N)), [0] * N
    clone, cross = [-1] * N, [-1] * N
    dpos, bpos = [0] * nb, [0] * nb
    aC, aM, aM2, xC = (np.zeros((5, nb)) for _ in range(4))
    lr = rl = deaths = kdc = kcc = nopp = nopp_ev = nopp_d = 0
    first = 0 if any(v > 0.5 for v in x) else -1        # a walker starting in the right well arrived at step 0
    evh, dth = [0] * (cap + 1), [0] * (cap + 1)
    amp = math.sqrt(2.0 * h / beta)
    amp_y = math.sqrt(2.0 * lam * h / beta)
    out = {k: [] for k in ("M", "C", "fr", "dpos", "bpos", "hist", "ev", "evh", "dth")}
    trs, trys, tra, trb, snaps = [], [], [], [], []

    def save():
        out["M"].append(list(M)); out["C"].append(list(C))
        out["fr"].append([deaths, kdc, kcc, nopp, nopp_ev, nopp_d])
        out["dpos"].append(list(dpos)); out["bpos"].append(list(bpos))
        hh = [0] * nb
        for v in x:
            hh[binof(v)] += 1
        out["hist"].append(hh); out["ev"].append([lr, rl]); out["evh"].append(list(evh)); out["dth"].append(list(dth))

    def trace():
        slot = {w: i for i, w in enumerate(wid)}
        trs.append([x[slot[j]] for j in range(ntr)]); trys.append([y[slot[j]] for j in range(ntr)])
        tra.append([anc_r[slot[j]] for j in range(ntr)]); trb.append([reborn[j] for j in range(ntr)])

    def snap():
        snaps.append((aC.copy(), aM.copy(), aM2.copy(), xC.copy()))

    si = ti = ni = 0
    if si < len(save_at) and save_at[si] == 0:
        save(); si += 1
    if ti < len(trace_at) and trace_at[ti] == 0:
        trace(); ti += 1
    if ni < len(snap_at) and snap_at[ni] == 0:
        snap(); ni += 1
    kopp = 0
    for step in range(n_steps):
        if method == "fr" and step % W == 0:
            anc_w = list(range(N))
        zx = Z[2 * N * step: 2 * N * step + N]; zy = Z[2 * N * step + N: 2 * N * (step + 1)]
        fx, fy, jb = [0.0] * N, [0.0] * N, [0] * N
        for i in range(N):
            xi, yi = x[i], y[i]
            e = math.exp(-xi * xi / (2.0 * s * s))
            om = oout + (oin - oout) * e
            dom = -(oin - oout) * (xi / (s * s)) * e
            base = 4.0 * H * xi * (xi * xi - 1.0)
            if orig:
                f = base + om * dom * yi * yi
                fyy = om * om * yi
            elif var == "alpha":
                lrat = dom / om
                k2a = om ** (2.0 * al)
                f = base + ((1.0 - al) / beta) * lrat + al * k2a * lrat * yi * yi
                fyy = k2a * yi
            else:
                lrat = dom / om
                d = yi - msc * math.log(om)
                f = base + lrat * (1.0 / beta) - (kap * kap) * d * (msc * lrat)
                fyy = (kap * kap) * d
            fx[i], fy[i] = f, fyy
            j = binof(xi)
            jb[i] = j; C[j] += 1.0; M[j] += f
            a = klass(clone[i], step, ath)
            aC[a, j] += 1.0; aM[a, j] += f; aM2[a, j] += f * f
            xC[klass(cross[i], step, cth), j] += 1.0
        for i in range(N):
            j = jb[i]
            bias = M[j] / (C[j] + mc)
            x[i] = _py_reflect(x[i] + (-fx[i] + bias) * h + amp * zx[i], gf.XMIN, gf.XMAX)
            if lam == 1.0:
                y[i] = y[i] - fy[i] * h + amp * zy[i]
            else:
                y[i] = y[i] - lam * fy[i] * h + amp_y * zy[i]
            if x[i] > 0.5:
                if lab[i] == -1:
                    lr += 1; cross[i] = step + 1
                lab[i] = 1; vis[i] = 1
                if first < 0:
                    first = step + 1
            elif x[i] < -0.5:
                if lab[i] == 1:
                    rl += 1; cross[i] = step + 1
                lab[i] = -1
        if method == "fr" and step % fe == 0:
            g = c["gamma"] * (1.0 - math.exp(-step / ramp)) if ramp > 0 else c["gamma"]
            p, hb, Sx = np.zeros(gl.N_GRID), np.zeros(gl.N_GRID), np.empty(N)
            xa = np.array(x)
            gl.kde_density(xa, N, kern, r, dx, p, hb); gl.uniform_scores(xa, N, p, dx, c["score_clip"], Sx)
            kd, kc, sel = _py_resample(list(Sx), N, g, dt_fr, cap, U[kopp * L:(kopp + 1) * L])
            kopp += 1; nopp += 1; evh[kd + kc] += 1
            d = 0
            if kd + kc > 0:
                nopp_ev += 1; kdc += kd; kcc += kc
                off = Counter(sel)
                for i in range(N):                      # D1 on the pre-gather positions
                    if off[i] == 0:
                        dpos[binof(x[i])] += 1
                    elif off[i] >= 2:
                        bpos[binof(x[i])] += off[i] - 1
                d = sum(1 for i in range(N) if off[i] == 0); deaths += d; nopp_d += d > 0
                clone = [step + 1 if off[k] >= 2 else clone[k] for k in sel]
                x, y, anc_r, anc_w, lab, vis, cross = ([a[k] for k in sel] for a in (x, y, anc_r, anc_w, lab, vis,
                                                                                      cross))
                taken, new = set(), []
                for k in sel:
                    new.append(wid[k] if k not in taken else None)
                    taken.add(k)
                free = iter([wid[k] for k in range(N) if k not in taken])
                for i, w in enumerate(new):
                    if w is None:
                        new[i] = next(free); reborn[new[i]] += 1
                wid = new
            dth[d] += 1
        while ti < len(trace_at) and trace_at[ti] == step + 1:
            trace(); ti += 1
        while ni < len(snap_at) and snap_at[ni] == step + 1:
            snap(); ni += 1
        while si < len(save_at) and save_at[si] == step + 1:
            save(); si += 1
    return dict(M_all=np.array(out["M"]), C_all=np.array(out["C"]), fr=np.array(out["fr"]),
                fr_death_pos_hist_cum=np.array(out["dpos"]), fr_birth_pos_hist_cum=np.array(out["bpos"]),
                hist_inst=np.array(out["hist"]), events_cum=np.array(out["ev"]),
                fr_events_hist_cum=np.array(out["evh"]), fr_deaths_hist_cum=np.array(out["dth"]),
                first_arrival_step=np.array([first]), X_final=np.array(x), Y_final=np.array(y),
                traces=np.array(trs), traces_y=np.array(trys), traces_anc=np.array(tra), traces_rebirth=np.array(trb),
                dep_age_C=np.array([q[0] for q in snaps]), dep_age_M=np.array([q[1] for q in snaps]),
                dep_age_M2=np.array([q[2] for q in snaps]), dep_cross_C=np.array([q[3] for q in snaps]),
                clone_final=np.array(clone), cross_final=np.array(cross))


F9_EDGES = dict(clone_age_edges_t=[0.004, 0.04, 0.4], cross_age_edges_t=[0.04, 0.4, 1.2])   # 10/100/1000, 100/1000/3000 steps


@pytest.mark.parametrize("name,method,N,gamma,start", [("shift", "fr", 24, 15.0, "left"),
                                                       ("alpha05", "fr", 40, 15.0, "left"),
                                                       ("alpha0", "fr", 8, 15.0, "left"),
                                                       ("lam025", "fr", 16, 15.0, "left"),
                                                       ("alpha1", "fr", 12, 40.0, "left"),
                                                       ("alpha1", "abf", 5, 1.5, "left"),
                                                       ("shift", "fr", 30, 15.0, "spread"),
                                                       ("alpha05", "abf", 30, 1.5, "spread")])
def test_f9_diagnostics_equal_python_reference(name, method, N, gamma, start):
    """'spread' starts walkers across [-1.5, 1.5] (labels -1 / 0 / +1): a walker starting in the gate (label 0)
    that later enters a well is NOT a crossing (cross_step stays -1, as for events_cum)."""
    h, n_steps = 4e-4, 6000
    cfg = dict(EASY, **DYN7[name], h=h, gamma=gamma, ramp_t=0.4, genealogy_window_t=0.4, **F9_EDGES)
    save_at = np.unique(np.r_[0, 1, 9, 10, 11, 999, 1000, 1001, np.arange(97, n_steps + 1, 97), n_steps])
    trace_at = np.arange(0, n_steps + 1, 50)
    snap_at = np.unique(np.r_[0, 11, 250, 1001, 2999, gf.snapshot_grid(n_steps)])
    x0 = y0 = None
    if start == "spread":
        x0 = np.linspace(-1.5, 1.5, N)
        y0 = gf.y_from_z(cfg, x0, np.random.default_rng(5).normal(size=N))
        assert np.sum(np.abs(x0) <= 0.5) >= 5
    ref = py_reference(cfg, method, N, 7100, n_steps, save_at, trace_at, snap_at, x0, y0)
    res = gf.run_arm(cfg, method, N, 7100, n_steps, save_at, trace_steps=trace_at, snapshot_steps=snap_at,
                     chunk_steps=123, x0=x0, y0=y0)
    for k in ("M_all", "C_all", "X_final", "Y_final", "hist_inst", "events_cum", "fr_events_hist_cum",
              "fr_deaths_hist_cum", "first_arrival_step", "traces", "traces_y", "traces_anc", "traces_rebirth",
              "fr_death_pos_hist_cum", "fr_birth_pos_hist_cum", "dep_age_C", "dep_age_M", "dep_age_M2",
              "dep_cross_C"):
        assert np.array_equal(res[k], ref[k], equal_nan=True), k
    fr = np.stack([res[k] for k in ("fr_deaths_cum", "fr_kd_cum", "fr_kc_cum", "fr_opp_cum", "fr_opp_event_cum",
                                    "fr_opp_death_cum")], 1)
    assert np.array_equal(fr, ref["fr"])
    assert np.array_equal(res["snap_step"], snap_at)
    aC, xC = res["dep_age_C"][-1].sum(1), res["dep_cross_C"][-1].sum(1)
    assert xC[:4].min() > 0 and xC[4] > 0, "every time-since-crossing class must be exercised"
    if method == "fr":
        assert aC.min() > 0, "every clone-age class must be exercised"
        assert res["fr_deaths_cum"][-1] > 0 and res["traces_rebirth"][-1].sum() > 0
    else:
        assert np.all(res["dep_age_C"][:, :4] == 0) and res["fr_death_pos_hist_cum"].max() == 0


def test_f9_partition_invariants_and_scripted_n2():
    h, n = 4e-4, 6000
    cfg = dict(EASY, variant="shift", kappa=1.0, lam=0.25, h=h, gamma=15.0, **F9_EDGES)
    sv = np.unique(np.r_[gf.budget_save_grid(n)[0], gf.snapshot_grid(n)])
    for method in ("abf", "fr"):
        res = gf.run_arm(cfg, method, 32, 7100, n, sv)
        idx = np.searchsorted(res["save_step"], res["snap_step"])
        assert np.array_equal(res["save_step"][idx], res["snap_step"])
        C = res["C_all"][idx]
        assert np.array_equal(res["dep_age_C"].sum(1), C) and np.array_equal(res["dep_cross_C"].sum(1), C)
        assert np.allclose(res["dep_age_M"].sum(1), res["M_all"][idx], rtol=1e-12, atol=1e-9)
        assert np.all(res["dep_age_M2"] >= 0) and np.all(np.diff(res["dep_age_C"], axis=0) >= 0)
        assert np.all(np.diff(res["dep_cross_C"], axis=0) >= 0)
        # sum f^2 over classes >= (sum f)^2 / count bin by bin (Cauchy-Schwarz)
        Cs, Ms, M2s = res["dep_age_C"][-1].sum(0), res["dep_age_M"][-1].sum(0), res["dep_age_M2"][-1].sum(0)
        ok = Cs > 0
        assert np.all(M2s[ok] * Cs[ok] >= Ms[ok] ** 2 * (1 - 1e-12))
        dp, bp = res["fr_death_pos_hist_cum"], res["fr_birth_pos_hist_cum"]
        assert np.array_equal(dp.sum(1), res["fr_deaths_cum"]) and np.array_equal(bp.sum(1), res["fr_deaths_cum"])
        assert np.all(np.diff(dp, axis=0) >= 0) and np.all(np.diff(bp, axis=0) >= 0)
        if method == "fr":
            assert res["fr_deaths_cum"][-1] > 20
    # scripted N = 2 (cap 1): every realised death is ONE death and ONE birth; afterwards BOTH slots carry the
    # clone mark (the source and its copy), so the next deposit pair is in class 0 with zero age
    N = 2
    res = gf.run_arm(dict(cfg, gamma=400.0, ramp_t=0.0), "fr", N, 7100, 400, np.arange(1, 401),
                     snapshot_steps=np.arange(1, 401), trace_steps=np.arange(0, 401))
    d = np.diff(np.r_[0, res["fr_deaths_cum"]])
    assert set(np.unique(d)) <= {0, 1} and d.sum() > 5
    k = int(np.argmax(d > 0))                    # first death in the gather after completed step k+1 (save index k)
    step_done = int(res["save_step"][k])         # completed steps when the gather happened
    assert np.array_equal(res["fr_death_pos_hist_cum"][k].sum(), 1) and res["fr_birth_pos_hist_cum"][k].sum() == 1
    # both walkers sit at the source's position after the gather (traces_t index step_done)
    tx = res["traces"][step_done]
    assert tx[0] == tx[1]
    assert int(np.argmax(res["fr_birth_pos_hist_cum"][k])) == int(math.floor((tx[0] - gf.XMIN) / (3.6 / 180) + 1e-9))
    # the next step deposits both walkers with clone age 0 -> class 0 gains exactly 2 counts
    a_before, a_after = res["dep_age_C"][k].sum(1), res["dep_age_C"][k + 1].sum(1)
    assert a_after[0] - a_before[0] == 2.0 and (a_after.sum() - a_before.sum()) == 2.0


# ------------------------------------------------------------------------------------------------- F10
def test_f10_accounting_bookkeeping_io(tmp_path):
    h = 2.5e-5
    for name in ("alpha05", "shift", "lam01"):
        for method, N, n_steps in (("abf", 1, 30000), ("abf", 7, 20000), ("fr", 7, 20000), ("fr", 2, 30000)):
            cfg = dict(EASY, **DYN7[name], h=h, gamma=40.0, ramp_t=0.05)
            save_at = np.unique(np.r_[0, gf.budget_save_grid(n_steps)[0]])
            res = gf.run_arm(cfg, method, N, 7100, n_steps, save_at)
            assert res["n_force_evals"] == N * n_steps == res["C_all"][-1].sum()
            assert np.array_equal(res["C_all"].sum(1), N * save_at)
            assert res["meta"]["B"] == N * n_steps and res["meta"]["status"] == "complete"
            kn = gf.derived_knobs(gf.full_cfg(cfg), N)
            if method == "fr":
                n_opp = -(-n_steps // kn["fr_every"])
                assert res["fr_opp_cum"][-1] == n_opp == res["fr_events_hist"].sum() == res["n_fr_steps"]
                ev, dh, kk = res["fr_events_hist_cum"], res["fr_deaths_hist_cum"], np.arange(kn["cap"] + 1)
                assert np.array_equal(dh @ kk, res["fr_deaths_cum"]) and np.array_equal(ev @ kk, res["fr_kd_cum"] +
                                                                                        res["fr_kc_cum"])
                assert res["fr_deaths_cum"][-1] > 0
            p = str(tmp_path / f"{name}_{method}_{N}.npz")
            gf.save_result(p, res)
            back = gf.load_result(p)
            assert_same(back, res, DYN_KEYS + DIAG_KEYS + ["save_t", "save_u", "resumed_from", "traces_t"])
            assert back["meta"] == json.loads(json.dumps(res["meta"])) and back["cfg"] == json.loads(
                json.dumps(res["cfg"]))
            assert res["traces_y"].shape == res["traces"].shape and not np.isnan(res["traces_y"]).any()
            assert res["dep_age_C"].shape == (len(res["snap_step"]), 5, 180)
            assert np.array_equal(res["snap_step"], gf.snapshot_grid(n_steps))
    with pytest.raises(ValueError):
        gf.run_arm(dict(h=h), "fr", 1, 7100, 100, [100])


def test_f10_meta_model_and_cfg_validation():
    for name, cfg in DYN7.items():
        mp = gf.model_params(dict(cfg, h=2.5e-5))
        c = gf.full_cfg(cfg)
        if c["variant"] == "alpha":
            assert mp["stiffness_gate"] == 32.0 ** (2 * c["alpha"]) and mp["stiffness_well"] == 1.0
            assert mp["model_id"] == (gf.MODEL_ORIG if c["alpha"] == 1.0 else gf.MODEL_ALPHA)
        else:
            assert mp["stiffness_gate"] == mp["stiffness_max"] == 1.0 and mp["model_id"] == gf.MODEL_SHIFT
        lk = c["lam"] * mp["stiffness_max"] * 2.5e-5 / 2
        assert abs(mp["em_var_inflation_max"] - 1.0 / (1.0 - lk)) < 1e-15 and mp["em_stable"]
        assert abs(mp["tau_y_gate"] - 1.0 / (c["lam"] * mp["stiffness_gate"])) < 1e-15
    # alpha 1, lam 1: the plan's 1.3 % EM variance excess at the gate (V1 limit 2 %)
    assert abs(gf.model_params(dict(h=2.5e-5))["em_var_inflation_max"] - 1.0 - 0.01297) < 1e-4
    r = gf.run_arm(dict(DYN7["lam025"], h=2.5e-5), "abf", 4, 7100, 100, [100])
    m = r["meta"]
    assert m["engine"] == "gateway_family_numba/1" and m["system"] == "gateway_family"
    assert (m["variant"], m["alpha"], m["kappa"], m["lam"]) == ("alpha", 1.0, None, 0.25)
    assert m["model"]["stiffness_gate"] == 1024.0 and m["stiffness_max"] == 1024.0
    assert m["run_signature"]["model"] == dict(variant="alpha", alpha=1.0, kappa=None, lam=0.25, model_id=0)
    assert r["cfg"]["lam"] == 0.25 and r["cfg"]["kappa"] is None
    for bad in (dict(variant="alpha", kappa=1.0), dict(variant="shift", alpha=0.5), dict(lam=0.0), dict(lam=-1.0),
                dict(variant="beta"), dict(variant="shift", kappa=0.0), dict(clone_age_edges_t=[0.1, 0.01, 1.0]),
                dict(alpha=float("nan"))):
        with pytest.raises(ValueError):
            gf.full_cfg(bad)


def test_f10_run_job_signature_includes_model(tmp_path):
    h = 2.5e-5
    out = str(tmp_path / "job.npz")
    cfg = dict(variant="alpha", alpha=0.5, lam=1.0, h=h)
    r1 = gf.run_job(cfg, "fr", 4, 7100, 4000, out)
    assert gf.is_complete(out) and not os.path.exists(out + ".ckpt.npz")
    assert np.array_equal(r1["save_step"], gf.prereg_save_grid(4000, h))
    r2 = gf.run_job(cfg, "fr", 4, 7100, 4000, out)
    assert np.array_equal(r1["M_all"], r2["M_all"])
    sig = gf.run_signature(cfg, "fr", 4, 7100, 4000, gf.prereg_save_grid(4000, h))
    assert gf.is_complete(out, sig)
    for bad in (dict(alpha=0.0), dict(lam=0.5), dict(variant="shift", alpha=None, kappa=1.0), dict(alpha=1.0)):
        with pytest.raises(ValueError, match="DIFFERENT run"):
            gf.run_job(dict(cfg, **bad), "fr", 4, 7100, 4000, out)
        assert not gf.is_complete(out, gf.run_signature(dict(cfg, **bad), "fr", 4, 7100, 4000,
                                                        gf.prereg_save_grid(4000, h)))
    with pytest.raises(ValueError, match="DIFFERENT run"):
        gf.run_job(cfg, "fr", 4, 7100, 4000, out, snapshot_steps=[4000])
    # the default initial state of every cell is init_family (x and z coupled), and enters the signature.  (omega
    # is exactly 1.0 in double precision for |x| > 0.9, so a small start in the left well can coincide bitwise
    # across variants; at N 2048 ~2 % of the walkers start at x > -0.9.)
    s_a = gf.run_signature(dict(variant="alpha", alpha=0.5, h=h), "abf", 2048, 7100, 4000, [4000])
    s_b = gf.run_signature(dict(variant="alpha", alpha=0.0, h=h), "abf", 2048, 7100, 4000, [4000])
    assert s_a["x0_sha"] == s_b["x0_sha"] and s_a["y0_sha"] != s_b["y0_sha"]


# ------------------------------------------------------------------------------------------------- F11
@pytest.mark.parametrize("name", ["alpha1", "alpha05", "shift", "lam01"])
def test_f11_population_size_invariance(name):
    """ABF bias off (min_count 1e300 makes M/(C + min_count) ~ 1e-297) and no FR: every walker is an independent
    copy of the same EM chain, so the standardised one-step residuals (x_{k+1} - x_k + f h)/sqrt(2h/beta) and
    (y_{k+1} - y_k + lam fy h)/sqrt(2 lam h/beta) are N(0, 1) whatever N; compare N = 1 against N = 64."""
    h = 2.5e-5
    cfg = dict(DYN7[name], h=h, min_count=1e300)
    c = gf.full_cfg(cfg)
    out = {}
    for N, n in ((1, 40000), (64, 1250)):
        tr = np.arange(0, n + 1)
        res = gf.run_arm(cfg, "abf", N, 8100, n, [n], trace_steps=tr)
        X, Y = res["traces"], res["traces_y"]          # (n + 1, min(N, 32))
        assert res["max_abs_bias"] < 1e-250
        f, fyv = gf.grad_V(c, X[:-1], Y[:-1])
        rx = (X[1:] - X[:-1] + f * h) / math.sqrt(2 * h / c["beta"])
        ry = (Y[1:] - Y[:-1] + c["lam"] * fyv * h) / math.sqrt(2 * c["lam"] * h / c["beta"])
        assert np.all(np.abs(X) < 1.7)                # no wall reflection in the window
        out[N] = (rx.ravel(), ry.ravel())
        for r in out[N]:
            assert abs(r.mean()) < 4.0 / math.sqrt(r.size) and abs(r.var() - 1.0) < 4.0 * math.sqrt(2.0 / r.size)
    for a, b in zip(out[1], out[64]):
        assert stats.ks_2samp(a, b).pvalue > 1e-4


# ------------------------------------------------------------------------------------------------- F12
def _driver():
    spec = importlib.util.spec_from_file_location("mech_run_cells", os.path.join(ROOT, "scripts", "mechanism",
                                                                                 "run_cells.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_f12_driver_jobs_and_budget():
    rc = _driver()
    n_new = {}
    dyn_smoke = set()
    for exp in rc.EXPERIMENTS:
        P = rc.load_cfg(exp)
        assert rc.frozen_h(P) == 2.5e-5 and rc.budget(P) == 3_276_800_000
        J = rc.job_list(P, exp)
        n_new[exp] = len(J)
        assert all(j["N"] * j["n_steps"] == 3_276_800_000 for j in J)
        assert all(j["cfg"]["h"] == 2.5e-5 and j["seed"] in P["seeds"] for j in J)
        assert not any(v.get("reuse") and j["variant"] == v["name"] for v in P["variants"] for j in J)
        Jr = rc.job_list(P, exp, include_reuse=True)
        assert len(Jr) == len(P["variants"]) * len(P["N"]) * len(P["seeds"]) * 2
        for j in Jr:                                          # every cfg is a valid engine cfg
            gf.full_cfg(j["cfg"])
        Js = rc.job_list(P, exp, smoke=True, smoke_T=4.0, only_N=[2048])
        assert {j["seed"] for j in Js} == set(P["seeds"][:2]) and all(j["n_steps"] == 160_000 for j in Js)
        assert {j["variant"] for j in Js} == {v["name"] for v in P["variants"]}     # smoke never reuses
        dyn_smoke |= {(j["cfg"]["variant"], j["cfg"]["alpha"], j["cfg"]["kappa"], j["cfg"]["lam"]) for j in Js}
        P2 = dict(P, h_frozen=1.25e-5)
        assert rc.budget(P2) == 6_553_600_000 and all(j["cfg"]["h"] == 1.25e-5 for j in rc.job_list(P2, exp))
    assert n_new == dict(matched_free_energy=576, conditional_relaxation=384)    # plan section 5: 576 + 384
    assert len(dyn_smoke) == 7                                                     # V5: every dynamics smoke-run
    # T_N per N as preregistered
    P = rc.load_cfg("matched_free_energy")
    for N, T in P["T_N"].items():
        assert abs(rc.budget(P) // int(N) * 2.5e-5 - T) < 1e-9
    # Experiment II lam1 is the SAME run as Experiment I alpha1 (same key, same signature) at N 2048 / 512, nothing else
    keys = {}
    for exp in rc.EXPERIMENTS:
        for j in rc.job_list(rc.load_cfg(exp), exp, include_reuse=True):
            keys.setdefault(rc.job_key(j), []).append((exp, j["variant"], j["N"]))
    dup = [v for v in keys.values() if len(v) > 1]
    assert len(dup) == 2 * 32 * 2 and all(len(v) == 2 and v[0][:2] == ("matched_free_energy", "alpha1")
                                          and v[1][:2] == ("conditional_relaxation", "lam1") and v[0][2] in (2048, 512)
                                          for v in dup)
    j0 = rc.job_list(rc.load_cfg("matched_free_energy"), "matched_free_energy", include_reuse=True, only_N=[512],
                     only_variant=["alpha1"], seeds=[8100])[0]
    j1 = rc.job_list(rc.load_cfg("conditional_relaxation"), "conditional_relaxation", include_reuse=True,
                     only_N=[512], only_variant=["lam1"], seeds=[8100])[0]
    assert rc.job_signature(gf, j0) == rc.job_signature(gf, j1)


DRV_ENV = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", NUMBA_NUM_THREADS="1")


def _run_driver(args, timeout=900):
    argv = [sys.executable, os.path.join(ROOT, "scripts", "mechanism", "run_cells.py")] + [str(a) for a in args]
    return subprocess.run(argv, env=DRV_ENV, capture_output=True, text=True, timeout=timeout)


def _sha(path):
    import hashlib
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _ledger(path):
    import csv
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def test_f12_driver_smoke_end_to_end(tmp_path):
    """The driver as it is run in production (a script; its pool workers import it as __mp_main__): run, skip what is
    complete, separate smoke horizons, and REFUSE (FAILED row, file untouched, exit 1) a complete result of a
    DIFFERENT run at the output path instead of counting it as done."""
    base = ["--experiment", "conditional_relaxation", "--only-variant", "lam0.25", "--only-N", "2048", "--seeds", "8100",
            "--smoke", "--workers", "2", "--out-root", tmp_path]
    p = _run_driver(base + ["--smoke-T", "0.025"])
    assert p.returncode == 0, p.stdout + p.stderr
    root = tmp_path / "smoke" / "T0.025" / "conditional_relaxation"
    for m in ("abf", "fr"):
        r = gf.load_result(str(root / "lam0.25" / "N2048" / f"s8100_{m}.npz"))
        assert r["n_force_evals"] == 2048 * 1000 and r["meta"]["lam"] == 0.25 and r["meta"]["method"] == m
        assert r["meta"]["n_steps"] == 1000 and r["cfg"]["variant"] == "alpha" and r["cfg"]["alpha"] == 1.0
    rows = _ledger(root / "ledger.csv")
    assert len(rows) == 2 and all(x["status"] == "complete" and x["n_force_evals"] == str(2048 * 1000) for x in rows)
    p = _run_driver(base + ["--smoke-T", "0.025"])          # all complete: nothing to run
    assert p.returncode == 0 and "nothing to run" in p.stdout, p.stdout + p.stderr
    assert len(_ledger(root / "ledger.csv")) == 2
    # another smoke horizon: its own root, never mistaken for the first one
    p = _run_driver(base + ["--smoke-T", "0.0125", "--dry-run"])
    assert p.returncode == 0 and "0 complete, 2 to run" in p.stdout and "T0.0125" in p.stdout, p.stdout + p.stderr
    # a complete result of a DIFFERENT run (n_steps 1000 where 2000 is requested) at the output path: refused
    stale = tmp_path / "smoke" / "T0.05" / "conditional_relaxation" / "lam0.25" / "N2048"
    stale.mkdir(parents=True)
    for m in ("abf", "fr"):
        shutil.copyfile(root / "lam0.25" / "N2048" / f"s8100_{m}.npz", stale / f"s8100_{m}.npz")
    sha = {m: _sha(stale / f"s8100_{m}.npz") for m in ("abf", "fr")}
    p = _run_driver(base + ["--smoke-T", "0.05", "--dry-run"])
    assert p.returncode == 0 and "0 complete" in p.stdout and "2 STALE" in p.stdout, p.stdout + p.stderr
    p = _run_driver(base + ["--smoke-T", "0.05"])
    assert p.returncode == 1 and "done (2 failed)" in p.stdout, p.stdout + p.stderr
    rows = _ledger(stale.parent.parent / "ledger.csv")
    assert len(rows) == 2 and all(x["status"].startswith("FAILED") and "DIFFERENT run" in x["status"] for x in rows)
    assert all(_sha(stale / f"s8100_{m}.npz") == sha[m] for m in ("abf", "fr"))      # never overwritten
    # a planted complete result of another MODEL (alpha 0, lam 1) at the lam0.25 path: refused too
    q = root / "lam0.25" / "N2048"
    gf.save_result(str(q / "s8100_abf.npz"),
                   gf.run_arm(dict(h=2.5e-5, variant="alpha", alpha=0.0, lam=1.0), "abf", 2048, 8100, 10, [10]))
    p = _run_driver(base + ["--smoke-T", "0.025"])
    assert p.returncode == 1 and "1 STALE" in p.stdout, p.stdout + p.stderr
    rows = _ledger(root / "ledger.csv")
    assert len(rows) == 3 and rows[-1]["method"] == "abf" and "DIFFERENT run" in rows[-1]["status"]
    assert gf.load_result(str(q / "s8100_abf.npz"))["meta"]["n_steps"] == 10


def test_f12_driver_alias_runs_identical_job_once(tmp_path):
    """lam1 (Experiment II) == alpha1 (Experiment I): selecting lam1 pulls the alpha1 owner in, runs it ONCE and
    hard-links its result to the lam1 path; both ledgers get a 'complete' row; a re-run has nothing to do."""
    base = ["--experiment", "conditional_relaxation", "--only-variant", "lam1", "--only-N", "2048", "--seeds", "8100",
            "--smoke", "--smoke-T", "0.0125", "--workers", "2", "--out-root", tmp_path]
    p = _run_driver(base)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "pulled in" in p.stdout and "2 alias(es) to link" in p.stdout, p.stdout
    sm = tmp_path / "smoke" / "T0.0125"
    for m in ("abf", "fr"):
        a = sm / "matched_free_energy" / "alpha1" / "N2048" / f"s8100_{m}.npz"
        b = sm / "conditional_relaxation" / "lam1" / "N2048" / f"s8100_{m}.npz"
        assert os.path.samefile(a, b)
        r = gf.load_result(str(b))
        assert r["meta"]["n_steps"] == 500 and r["cfg"]["lam"] == 1.0 and r["cfg"]["alpha"] == 1.0
    ra = _ledger(sm / "matched_free_energy" / "ledger.csv")
    rb = _ledger(sm / "conditional_relaxation" / "ledger.csv")
    assert len(ra) == 2 and all(x["status"] == "complete" and x["variant"] == "alpha1" for x in ra)
    assert len(rb) == 2 and all(x["status"] == "complete" and x["variant"] == "lam1" and "alias of" in x["note"]
                                and x["n_force_evals"] == str(2048 * 500) for x in rb)
    p = _run_driver(base)
    assert p.returncode == 0 and "nothing to run" in p.stdout, p.stdout + p.stderr


def _reuse_cfg_dir(tmp_path, h_frozen=None):
    """Copies of the experiment and cell configs whose reuse cells name a gate record under tmp_path."""
    d = tmp_path / "cfg"
    shutil.copytree(os.path.join(ROOT, "configs", "mechanism", "cells"), d / "cells")
    for exp in ("matched_free_energy", "conditional_relaxation"):
        if h_frozen is None:
            shutil.copyfile(os.path.join(ROOT, "configs", "mechanism", exp + ".json"), d / (exp + ".json"))
        else:
            P = json.load(open(os.path.join(ROOT, "configs", "mechanism", exp + ".json")))
            P["h_frozen"] = h_frozen
            (d / (exp + ".json")).write_text(json.dumps(P, indent=1))
    gate = tmp_path / "gate" / "reuse_gate.json"
    for cell in (d / "cells" / "matched_free_energy" / "alpha1.json", d / "cells" / "conditional_relaxation" / "lam1.json"):
        P = json.load(open(cell))
        assert P["reuse"]
        P["reuse"]["gate_record"] = str(gate)
        cell.write_text(json.dumps(P, indent=1) + "\n")
    return d, gate


def _write_gate(gate, verdict):
    rd = os.path.join(ROOT, "results", "equal_budget_v2", "gateway")
    jobs = [dict(N=N, seed=8100, method=m, bitwise_equal=True,
                 reused_sha256=_sha(os.path.join(rd, f"N{N}", f"s8100_{m}.npz"))) for N in (2048, 128) for m in ("abf", "fr")]
    gate.parent.mkdir(parents=True, exist_ok=True)
    gate.write_text(json.dumps(dict(schema="mechanism_reuse_gate/1", verdict=verdict, engine="gateway_family_numba/1",
                                    model=dict(variant="alpha", alpha=1.0, kappa=None, lam=1.0), jobs=jobs)))


def test_f12_driver_reuse_policy(tmp_path):
    """A reuse cell is left out of a production run only when its reuse is licensed (precondition + cell config +
    gate record PASS); a pending gate needs --defer-reuse; an impossible / failed reuse refuses to start unless
    --include-reuse (which also runs lam1 once, as an alias of alpha1)."""
    d, gate = _reuse_cfg_dir(tmp_path)
    out = ["--out-root", tmp_path / "out", "--cfg-dir", d, "--dry-run"]
    p = _run_driver(out)                                                    # gate record absent: pending -> refused
    assert p.returncode == 2 and p.stdout.count("REFUSED") == 2 and "--defer-reuse" in p.stdout, p.stdout + p.stderr
    p = _run_driver(out + ["--defer-reuse"])
    assert p.returncode == 0 and p.stdout.count("WARNING reuse cell LEFT OUT") == 2, p.stdout + p.stderr
    assert "matched_free_energy: 576 jobs" in p.stdout and "conditional_relaxation: 384 jobs" in p.stdout
    _write_gate(gate, "FAIL")                                               # a gate that ran and failed: hard
    p = _run_driver(out + ["--defer-reuse"])
    assert p.returncode == 2 and "not PASS" in p.stdout and "remedy: run it (--include-reuse)" in p.stdout, p.stdout
    _write_gate(gate, "PASS")                                               # licensed: left out
    p = _run_driver(out)
    assert p.returncode == 0 and p.stdout.count("LEFT OUT (reuse licensed") == 2, p.stdout + p.stderr
    assert "matched_free_energy: 576 jobs" in p.stdout and "conditional_relaxation: 384 jobs" in p.stdout
    p = _run_driver(out + ["--include-reuse"])
    assert p.returncode == 0 and "matched_free_energy: 768 jobs" in p.stdout, p.stdout + p.stderr
    assert "conditional_relaxation: 512 jobs, 0 complete, 384 to run" in p.stdout and "128 alias(es)" in p.stdout
    # h refined to 1.25e-5: reuse impossible -> refused even with --defer-reuse and a PASS gate; --include-reuse runs it
    d2, gate2 = _reuse_cfg_dir(tmp_path / "h2", h_frozen=1.25e-5)
    _write_gate(gate2, "PASS")
    out2 = ["--out-root", tmp_path / "out", "--cfg-dir", d2, "--dry-run"]
    for extra in ([], ["--defer-reuse"]):
        p = _run_driver(out2 + extra)
        assert p.returncode == 2 and p.stdout.count("reuse impossible") == 2 and "h 1.25e-05" in p.stdout, p.stdout
    p = _run_driver(out2 + ["--include-reuse"])
    assert p.returncode == 0 and "matched_free_energy: 768 jobs" in p.stdout and "128 alias(es)" in p.stdout, p.stdout
    # selecting only non-reuse variants never consults the reuse rule
    p = _run_driver(["--out-root", tmp_path / "out", "--cfg-dir", d2, "--dry-run", "--only-variant", "alpha0", "lam0.1"])
    assert p.returncode == 0 and "REFUSED" not in p.stdout, p.stdout + p.stderr


# ------------------------------------------------------------------------------------------------- F13
def test_f13_speed_benchmark():
    h, N, n = 2.5e-5, 512, 8000
    rows = []
    sv = gf.prereg_save_grid(n, h)
    gf.run_arm(dict(h=h), "fr", 4, 1, 100, [100])
    for name, v in DYN7.items():
        for method in ("abf", "fr"):
            t0 = time.perf_counter()
            r = gf.run_arm(dict(v, h=h), method, N, 8100, n, sv)
            rows.append((name, method, 1e9 * (time.perf_counter() - t0) / (N * n)))
            assert r["n_force_evals"] == N * n
    print("\n  variant   arm   ns/walker-step (N 512, diagnostics on)")
    for name, m, ns in rows:
        print(f"  {name:<9s} {m:<4s}  {ns:7.1f}")
    assert max(ns for _, _, ns in rows) < 400.0


# ------------------------------------------------------------------------------------------------- slow
@slow
@pytest.mark.parametrize("method", ["abf", "fr"])
def test_slow_bitwise_original_production_knobs_long(method):
    """Production knobs over 10 t.u. (N 64, 400k steps): many FR events at the production rate, bitwise."""
    h, N, n = 2.5e-5, 64, 400_000
    sv = gl.prereg_save_grid(n, h)
    a = gl.run_arm(dict(h=h), method, N, 8100, n, sv)
    b = gf.run_arm(dict(h=h, variant="alpha", alpha=1.0, lam=1.0), method, N, 8100, n, sv)
    assert_same(a, b, [k for k in a if k not in NOT_COMPARED])
    if method == "fr":
        assert a["fr_deaths_cum"][-1] > 0


@slow
@pytest.mark.parametrize("name", list(DYN7))
def test_slow_production_stability_short(name):
    """Production knobs, N 2048, 1 t.u., both arms: finite state, no blow-up, and the SAVED final bias profile
    M/(C + min_count) on the visited bins below 2 max|F*'| there.  The engine's max_abs_bias (the running max of
    |Gamma| over every read, single-deposit bins included) is printed, NOT asserted against 2 max|F*'|: at lam 0.1
    / 0.25 it reaches 19.6 / 30.4 (lam 0.1, seeds 8100 / 8101) and 14.4 (lam 0.25) at this horizon -- walkers enter
    the gate flank with unrelaxed (well-width) y and deposit omega omega' y^2 into sparse bins -- a physical
    transient of those dynamics, not an instability (SCIENTIFIC_PLAN section 3 V5 does not say which Gamma it means;
    flagged in the engine report)."""
    h, N, n = 2.5e-5, 2048, 40_000
    c = gf.full_cfg(dict(DYN7[name], h=h))
    xc = gf.XMIN + (np.arange(180) + 0.5) * (gf.XMAX - gf.XMIN) / 180
    for method in ("abf", "fr"):
        r = gf.run_arm(c, method, N, 8100, n, gf.prereg_save_grid(n, h))
        assert np.all(np.isfinite(r["X_final"])) and np.all(np.isfinite(r["Y_final"]))
        assert np.all(np.isfinite(r["M_all"])) and r["max_abs_bias"] < 100.0
        vis = r["C_all"][-1] > 0
        G = r["M_all"][-1] / (r["C_all"][-1] + c["min_count"])
        lim = 2.0 * np.abs(gf.dF_star(c, xc[vis])).max()
        print(f"\n  {name} {method}: running max|Gamma| {r['max_abs_bias']:.2f}, final max|Gamma| visited "
              f"{np.abs(G[vis]).max():.2f}, 2 max|F*'| visited {lim:.2f}")
        assert np.abs(G[vis]).max() < lim
