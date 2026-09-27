"""The online histogram (P0) ABF mean-force estimator, both engines (docs/HISTOGRAM_ABF_REPLICATION.md).

Numbered as in the campaign brief: (1) counts / force sums == brute-force numpy; (2) Gamma_j is
exactly M_j / (C_j + min_count), i.e. M_j / C_j at min_count 0; (3) evaluate() is the own-bin P0
value with the documented [left, right) / boundary convention; (4) the PMF of a known
piecewise-constant force is exact; (5) empty bins are finite and zero-biased; (6) the kernel
path is bit-identical to the fixtures generated at 64567ea before this code existed;
(7) the new fields are inert for legacy runs (hashes, run ids, records); (8) the Fisher-Rao
marginal KDE / target / score / schedule / bandwidth source is untouched; (9) short histogram
runs are finite.  CPU float64 throughout, so every comparison is exact.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import eb_abffr_core as eb  # noqa: E402
import wca_abffr_core as wca  # noqa: E402
import wca_phase_jobs as jobs  # noqa: E402

CPU, F64 = torch.device("cpu"), torch.float64
FIX_EB = os.path.join(HERE, "fixtures", "eb_pre_histogram_fixture.npz")
FIX_WCA = os.path.join(HERE, "fixtures", "wca_pre_relax_fixture.npz")
FIX_IDS = os.path.join(HERE, "fixtures", "histogram_abf_head_ids.json")
HEAD_COMMIT = "64567ea"      # the commit the fixtures were generated at (before this code existed)


# ----------------------------------------------------------------------------- helpers
def _eb_grid():
    return eb.build_grid(CPU, F64)


def _eb_est(n_bins, min_count=1.0, R=2):
    x_grid, dx, eval_mask, idx0 = _eb_grid()
    return eb.HistogramABFEstimator(R, n_bins, min_count, x_grid, CPU, F64), x_grid, eval_mask, idx0


def _wca_tiny(**kw):
    params = wca.DimerWCAParams(n_dim=4, beta=1.0, h=1.0, w=2.0, a=1.5)
    sim = wca.SimConfig(n_replicas=64, n_steps=3000, save_every=1000, dt=2e-3, n_grid=48, fr_start_steps=500,
                        fr_every=25, abf_warmup_steps=200, estimator_burn_in_steps=200, fr_rate=0.5,
                        max_event_fraction=0.1, seed=7, **kw)
    engine = wca.WCADimerEngine(params, device=CPU, dtype=F64)
    return params, sim, engine


def _func_source(text, name):
    tree = ast.parse(text)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(text, node)
    raise KeyError(name)


def _head_text(relpath):
    try:
        return subprocess.check_output(["git", "show", f"{HEAD_COMMIT}:{relpath}"], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL)
    except Exception:
        pytest.skip("git history not available")


# ----------------------------------------------------------------------------- 1, 2, 3
def test_eb_histogram_counts_and_force_sums_match_numpy():
    est, x_grid, _, _ = _eb_est(45)
    rng = np.random.default_rng(0)
    X = rng.uniform(-1.8, 1.8, (2, 6000))
    f = rng.standard_normal((2, 6000))
    for c in range(6):                       # online, in chunks: no sample is ever revisited
        est.update(torch.tensor(X[:, c * 1000:(c + 1) * 1000]), torch.tensor(f[:, c * 1000:(c + 1) * 1000]))
    edges = np.linspace(-1.8, 1.8, 46)
    assert np.allclose(est.edges.numpy(), edges) and est.delta == pytest.approx(0.08)
    for r in range(2):
        C, _ = np.histogram(X[r], bins=edges)
        M, _ = np.histogram(X[r], bins=edges, weights=f[r])
        assert np.array_equal(est.C[r].numpy(), C)
        assert np.allclose(est.M[r].numpy(), M, atol=1e-12, rtol=0)
    # (2) Gamma_j == M_j / (C_j + min_count) exactly (same IEEE division)
    G = est.bin_mean_force().numpy()
    assert np.array_equal(G, est.M.numpy() / (est.C.numpy() + 1.0))
    est0, *_ = _eb_est(45, min_count=0.0)
    est0.C.copy_(est.C); est0.M.copy_(est.M)
    G0 = est0.bin_mean_force().numpy()
    C, M = est.C.numpy(), est.M.numpy()
    assert np.array_equal(G0, np.where(C > 0, M / np.maximum(C, 1.0), 0.0))


def test_wca_histogram_counts_and_force_sums_match_numpy():
    grid = torch.linspace(-0.2, 1.2, 160, dtype=F64)
    est = wca.HistogramABFEstimator(grid, 40, -0.2, 1.2)
    rng = np.random.default_rng(1)
    z = rng.uniform(-0.25, 1.25, 8000)      # a few outside the domain, as the soft wall allows
    f = rng.standard_normal(8000)
    for c in range(8):
        est.update(torch.tensor(z[c * 1000:(c + 1) * 1000]), torch.tensor(f[c * 1000:(c + 1) * 1000]))
    edges = np.linspace(-0.2, 1.2, 41)
    zc = np.clip(z, -0.2, 1.2 - 1e-12)      # outside -> boundary bins
    C, _ = np.histogram(zc, bins=edges)
    M, _ = np.histogram(zc, bins=edges, weights=f)
    assert np.array_equal(est.C.numpy(), C) and np.allclose(est.M.numpy(), M, atol=1e-12, rtol=0)
    assert est.n_updates == 8000
    G = est.bin_mean_force().numpy()
    Mt = est.M.numpy()                                                          # (torch's summation order)
    assert np.array_equal(G, np.where(C > 0, Mt / np.maximum(C, 1.0), 0.0))    # M_j / C_j exactly


def test_bin_edge_convention_left_closed_right_open_last_bin_closed():
    n = 45
    edges = torch.linspace(-1.8, 1.8, n + 1, dtype=F64)
    j = eb.histogram_bin_index(edges, -1.8, 3.6 / n, n).numpy()
    assert np.array_equal(j[:-1], np.arange(n)) and j[-1] == n - 1      # e_j -> bin j; hi -> last bin
    inside = torch.tensor([-1.8 + 0.5 * 0.08, 1.8 - 1e-12, -1.75], dtype=F64)
    assert eb.histogram_bin_index(inside, -1.8, 0.08, n).tolist() == [0, n - 1, 0]
    outside = torch.tensor([-2.5, 2.5], dtype=F64)
    assert eb.histogram_bin_index(outside, -1.8, 0.08, n).tolist() == [0, n - 1]
    # every EB grid node sits on an edge for every ladder width: it belongs to the bin to its right
    x_grid, *_ = _eb_grid()
    for nb in (45, 60, 90, 180, 360):
        est, *_ = _eb_est(nb)
        expect = np.clip(np.round((x_grid.numpy() + 1.8) / est.delta), 0, nb - 1)
        on_edge = np.abs((x_grid.numpy() + 1.8) / est.delta - np.round((x_grid.numpy() + 1.8) / est.delta)) < 1e-9
        assert np.array_equal(est.g2h.numpy()[on_edge], expect[on_edge].astype(int))


def test_evaluate_returns_the_own_bin_p0_value():
    est, x_grid, _, _ = _eb_est(60)
    rng = np.random.default_rng(2)
    est.update(torch.tensor(rng.uniform(-1.8, 1.8, (2, 3000))), torch.tensor(rng.standard_normal((2, 3000))))
    G = est.bin_mean_force().numpy()
    Xq = torch.tensor(rng.uniform(-1.8, 1.8, (2, 500)))
    val = est.evaluate(Xq).numpy()
    edges = np.linspace(-1.8, 1.8, 61)
    for r in range(2):
        jb = np.digitize(Xq[r].numpy(), edges[1:-1])          # [left, right) bins
        assert np.array_equal(val[r], G[r, jb])
    # no interpolation: two points in the same bin feel the same force
    same = torch.tensor([[-1.79, -1.741], [0.001, 0.059]])
    v = est.evaluate(same).numpy()
    assert v[0, 0] == v[0, 1] and v[1, 0] == v[1, 1]
    # WCA: edge_extrapolate False -> zero outside the domain, True -> boundary bin
    grid = torch.linspace(-0.2, 1.2, 160, dtype=F64)
    for flag in (False, True):
        w = wca.HistogramABFEstimator(grid, 40, -0.2, 1.2, edge_extrapolate=flag)
        w.update(torch.tensor(rng.uniform(-0.2, 1.2, 4000)), torch.tensor(rng.standard_normal(4000)))
        Gw = w.bin_mean_force().numpy()
        out = w.evaluate(torch.tensor([-0.3, 1.3, 0.5])).numpy()
        assert out[2] == Gw[20]
        assert (out[0] == 0.0 and out[1] == 0.0) if not flag else (out[0] == Gw[0] and out[1] == Gw[-1])


# ----------------------------------------------------------------------------- 4
def test_pmf_integration_of_a_known_piecewise_constant_force_is_exact():
    est, x_grid, eval_mask, idx0 = _eb_est(45, R=1)
    rng = np.random.default_rng(3)
    G = rng.standard_normal((1, 45))
    F = est.pmf_profile(idx0, torch.tensor(G)).numpy()[0]
    edges, delta = est.edges.numpy(), est.delta
    xg = x_grid.numpy()
    j = np.clip(np.floor((xg + 1.8) / delta + 1e-9).astype(int), 0, 44)
    F_exact = np.concatenate([[0.0], np.cumsum(G[0] * delta)])[j] + G[0, j] * (xg - edges[j])
    F_exact -= F_exact[idx0]
    assert np.allclose(F, F_exact, atol=1e-13, rtol=0)
    # bin averages of a LINEAR F' integrate to the exact quadratic at every EDGE node; at an
    # interior node the piecewise-linear F_hat is the chord, within a*Delta^2/8 of the parabola
    for nb in (45, 90):
        e, xg_, em, i0 = _eb_est(nb, R=1)
        a, b = 3.0, -0.7
        ed = e.edges.numpy()
        Gl = (a * 0.5 * (ed[1:] ** 2 - ed[:-1] ** 2) + b * (ed[1:] - ed[:-1])) / e.delta     # exact bin means
        Fl = e.pmf_profile(i0, torch.tensor(Gl[None, :])).numpy()[0]
        x = xg_.numpy()
        Fq = 0.5 * a * x ** 2 + b * x
        Fq -= Fq[i0]
        on_edge = np.abs((x + 1.8) / e.delta - np.round((x + 1.8) / e.delta)) < 1e-9
        # gauge both on an edge node (the x = 0 gauge node is interior for Delta = 0.08)
        Fl_e, Fq_e = Fl[on_edge] - Fl[on_edge][0], Fq[on_edge] - Fq[on_edge][0]
        assert np.allclose(Fl_e, Fq_e, atol=1e-12, rtol=0)
        assert np.all(np.abs((Fl - Fl[on_edge][0]) - (Fq - Fq[on_edge][0])) <= a * e.delta ** 2 / 8 + 1e-12)
        if nb == 90:
            assert (~on_edge).sum() > 0 and np.abs(Fl - Fq)[~on_edge].max() > 1e-6
    # the numpy mirror used offline (WCA) agrees with the torch integral
    grid = torch.linspace(-0.2, 1.2, 160, dtype=F64)
    w = wca.HistogramABFEstimator(grid, 40, -0.2, 1.2)
    Gw = rng.standard_normal(40)
    w.M.copy_(torch.tensor(Gw)); w.C.fill_(1.0)
    Fw = w.pmf_profile().numpy()
    Fn = wca.histogram_pmf_nodes_np(Gw, w.edges.numpy(), grid.numpy())
    Fn -= Fn[int(torch.argmin(torch.abs(grid - 0.5)))]
    assert np.allclose(Fw, Fn, atol=1e-13, rtol=0)


def test_own_fp_error_and_floor_are_the_function_space_rms():
    # P0 = bin averages of the reference -> the own e_F' equals an independent fine-grid RMS
    x_grid, dx, eval_mask, idx0 = _eb_grid()
    pars = [torch.tensor([[v]], dtype=F64) for v in (8.0, 2.5, 1.0, 25.0, 0.25)]
    for nb, tol in ((60, 5e-4), (90, 2e-3)):        # 60: window edges are bin edges; 90: a bin is cut in half
        fl = eb.p0_projection_floor(nb, x_grid, eval_mask, idx0, *pars, device=CPU, dtype=F64)
        xf = np.linspace(-1.5, 1.5, 600001)
        fp = eb.reference_mean_force(torch.tensor(xf[None, :]), *pars).numpy()[0]
        j = np.clip(np.floor((xf + 1.8) / fl["delta"] + 1e-9).astype(int), 0, nb - 1)
        rms = np.sqrt(np.mean((fl["Fp_bins"][j] - fp) ** 2))
        assert fl["floor_l2_fp"] == pytest.approx(rms, rel=tol)
        assert fl["floor_l2_f"] < 0.01 and fl["floor_l2_fp_nodes"] > fl["floor_l2_fp"]
    # the WCA numpy quadrature agrees with the torch one on the same P0 profile and reference
    est60, xg60, em60, _ = _eb_est(60, R=1)
    ref_sub = eb.reference_mean_force(est60.sub_x.unsqueeze(0), *pars)
    mask = (est60.sub_x >= -1.5) & (est60.sub_x <= 1.5)
    Gt = torch.tensor(np.random.default_rng(6).standard_normal((1, 60)))
    e_torch = float(est60.fp_error(ref_sub, mask, Gt)[0])
    xf = np.linspace(-1.8, 1.8, 36001)
    ref_np = dict(grid=xf, mean_force=eb.reference_mean_force(torch.tensor(xf[None, :]), *pars).numpy()[0])
    e_np = float(wca.histogram_fp_error_np(Gt.numpy()[0], est60.edges.numpy(), ref_np["grid"], ref_np["mean_force"], -1.5, 1.5))
    assert e_torch == pytest.approx(e_np, rel=1e-6)
    # exact integration: at Delta = dx every node is an edge and the F floor vanishes
    # (the bin averages use a 64-point midpoint rule, so "vanishes" means the 1e-8 level, not 1e-12)
    fl2 = eb.p0_projection_floor(180, x_grid, eval_mask, idx0, *pars, device=CPU, dtype=F64)
    assert fl2["floor_l2_f"] < 1e-6 and fl2["floor_l2_f"] < 1e-3 * fl["floor_l2_f"]
    # WCA numpy floor on a linear reference F' = a z + b: the exact integral is the chord of the
    # parabola, exact at bin edges; the WCA nodes sit INSIDE bins, so the F floor is the chord
    # term, bounded by a Delta^2 / 8, and the F' floor is a Delta / sqrt(12) (P0 of a slope)
    grid = np.linspace(-0.2, 1.2, 160)
    ref = dict(grid=grid, mean_force=2.0 * grid + 1.0, free_energy=grid ** 2 + grid)
    sim = wca.SimConfig(n_grid=160)
    fw = wca.histogram_p0_floor_np(40, ref, sim)
    assert 0 < fw["floor_l2_f"] <= 2.0 * fw["delta"] ** 2 / 8
    assert fw["floor_l2_fp"] == pytest.approx(2.0 * fw["delta"] / np.sqrt(12.0), rel=0.05)


# ----------------------------------------------------------------------------- 5
def test_empty_bins_are_finite_and_zero_biased():
    est, x_grid, eval_mask, idx0 = _eb_est(60)
    rng = np.random.default_rng(4)
    est.update(torch.tensor(rng.uniform(-1.8, 0.0, (2, 2000))), torch.tensor(rng.standard_normal((2, 2000))))
    G = est.bin_mean_force()
    assert torch.isfinite(G).all() and torch.all(G[:, 30:] == 0)
    assert torch.all(est.evaluate(torch.tensor([[0.5, 1.7], [0.9, 1.2]])) == 0)
    assert torch.isfinite(est.pmf_profile(idx0)).all()
    win = est.window_bins(-1.5, 1.5)
    frac = float((est.C[0][win] < eb.TRUST_MIN_COUNT).to(F64).mean())
    assert 0.4 < frac < 0.6 and float(est.C[0][win].min()) == 0.0
    w = wca.HistogramABFEstimator(torch.linspace(-0.2, 1.2, 160, dtype=F64), 40, -0.2, 1.2)
    w.update(torch.tensor(rng.uniform(0.6, 1.2, 500)), torch.tensor(rng.standard_normal(500)))
    assert torch.isfinite(w.bin_mean_force()).all() and torch.all(w.bin_mean_force()[:20] == 0)
    assert torch.isfinite(w.pmf_profile()).all() and torch.all(w.evaluate(torch.tensor([0.0, 0.3])) == 0)
    d = w.diagnostics(-0.1, 1.1)
    assert d["min_count_window"] == 0.0 and 0.5 < d["frac_untrusted_window"] < 0.7


# ----------------------------------------------------------------------------- 6, 7
def _eb_fixture_batch(**kw):
    cfg = eb.PhysConfig(N=64, n_steps=400, save_every=100, ess_window_steps=200, **kw)
    return eb.BatchSpec(configs=[cfg, cfg], seeds=[3, 4], methods=[eb.ABF, eb.FR_ESTIMATED, eb.FR_UNIFORM],
                        batch_seed=5)


def test_eb_kernel_path_is_bit_identical_to_the_head_fixture():
    fx = np.load(FIX_EB)
    for kw in ({}, {"abf_estimator": "kernel", "abf_n_bins": 0}):
        recs = eb.simulate_batch(_eb_fixture_batch(**kw), device=CPU, dtype=F64)
        for i, r in enumerate(recs):
            assert str(fx[f"{i}/method"]) == r["method"]
            for k in ("F_hat", "Fp_hat", "l2_f_t", "l2_fp_t", "ess_t", "p_hat", "cond_emp_var", "cond_count"):
                assert np.array_equal(np.asarray(r[k]), fx[f"{i}/{k}"], equal_nan=True), (i, k)
            sc = np.array([r["final_l2_f"], r["final_l2_fp"], r["int_l2_f"], r["n_die"], r["n_clone"], r["repl_fraction"]])
            assert np.array_equal(sc, fx[f"{i}/scalars"]), i
            assert r["abf_estimator"] == "kernel" and "Fp_bins" not in r      # legacy record shape (+ identity tag)


def test_wca_kernel_path_is_bit_identical_to_the_accepted_fixture():
    fx = np.load(FIX_WCA)
    keys = ("mean_force", "pmf", "p_hat", "eff_counts", "ancestor_ess", "raw_fsum", "raw_csum", "birth_hist",
            "death_hist", "fr_event_counts")
    for m in ("abf", "fr_uniform"):
        for kw in ({}, {"abf_estimator": "kernel"}):
            p, s, e = _wca_tiny(**kw)
            d = wca.run_sampler_gpu(m, p, s, e, collect_diagnostics=True, verbose=False, readout_bandwidths=(0.025, 0.0))
            for k in keys:
                assert np.array_equal(np.asarray(d[k]), fx[f"{m}/{k}"], equal_nan=True), (m, k)
            assert np.array_equal(np.asarray(d["readout_mean_force"][0.025]), fx[f"{m}/readout_0.025"])
            assert d["total_replacement_events"] == int(fx[f"{m}/total_replacement_events"])
            assert d["abf_estimator"] == "kernel" and "hist_mf_bins" not in d


def test_new_fields_are_inert_for_legacy_ids_and_configs():
    ids = json.load(open(FIX_IDS))
    cell = dict(beta=1.0, h=2.0, w=2.0, n_dim=10, a=1.5, sigma=1.0, epsilon=1.0)
    fr = dict(fr_rate=0.10, target_ema_rate=0.005, max_event_fraction=0.02, fr_every=5, fr_start_steps=20000,
              score_clip=2.0)
    sp = jobs.PhaseRunSpec(stage="confirmation", name="abf", method="abf", seed=700, n_steps=120000,
                           n_replicas=1024, save_every=2500, **cell, **fr)
    assert sp.run_id() == ids["run_id"] and sp.spec_hash() == ids["spec_hash"]
    base = jobs.effective_base(jobs.load_yaml(os.path.join(ROOT, "configs/wca_phase_diagram_production.yaml")), "production")
    sim = jobs.build_sim(sp, base)
    assert sim.config_hash() == ids["config_hash"] and sim.abf_estimator == "kernel" and sim.abf_n_bins == 0
    assert _wca_tiny()[1].config_hash() == ids["tiny_config_hash"]
    assert eb.PhysConfig().abf_estimator == "kernel" and eb.PhysConfig().abf_n_bins == 0
    # a histogram spec gets a distinct, self-describing id and its SimConfig carries the fields
    sph = jobs.PhaseRunSpec(stage="confirmation", name="hist_abf", method="abf", seed=700, n_steps=120000,
                            n_replicas=1024, save_every=2500, abf_estimator="histogram", abf_n_bins=80, **cell, **fr)
    assert "__hist80__" in sph.run_id() and sph.spec_hash() != sp.spec_hash()
    simh = jobs.build_sim(sph, base)
    assert simh.abf_estimator == "histogram" and simh.abf_n_bins == 80 and simh.config_hash() != sim.config_hash()
    with pytest.raises(ValueError):
        wca.make_abf_estimator(wca.SimConfig(abf_estimator="spline"), torch.linspace(-0.2, 1.2, 160, dtype=F64))
    with pytest.raises(AssertionError):
        eb.simulate_batch(eb.BatchSpec(configs=[eb.PhysConfig(N=8, n_steps=5, abf_estimator="histogram", abf_n_bins=0)],
                                       seeds=[1], methods=[eb.ABF]), device=CPU, dtype=F64)


# ----------------------------------------------------------------------------- 8
def test_fisher_rao_marginal_kde_target_score_and_schedule_are_untouched():
    # EB: the FR functions and the FR block of simulate_batch are textually identical to the pre-histogram commit
    cur = open(os.path.join(ROOT, "src", "eb_abffr_core.py")).read()
    head = _head_text("src/eb_abffr_core.py")
    for fn in ("gaussian_kernel", "smooth", "binned_density", "fr_target_from", "fr_resample_indices", "interp1d"):
        assert _func_source(cur, fn) == _func_source(head, fn), fn

    def fr_block(text):
        return text[text.index("# ---- Fisher-Rao birth-death ----"): text.index("X, Y = Xp, Yp")]
    assert fr_block(cur) == fr_block(head)
    assert "k_eta, r_eta = gaussian_kernel(c0.eta" in cur and eb.PhysConfig().eta == 0.10
    # WCA: KDE, score, target and birth-death are textually identical; the bandwidth default and YAML are unchanged
    curw = open(os.path.join(ROOT, "src", "wca_abffr_core.py")).read()
    headw = _head_text("src/wca_abffr_core.py")
    for fn in ("gaussian_kernel_torch", "kde_1d_torch", "fr_score_torch", "fr_target_uniform_torch",
               "fr_target_estimated_torch", "recentered_clipped_score_torch", "fixed_population_birth_death_torch",
               "_build_fr_target"):
        assert _func_source(curw, fn) == _func_source(headw, fn), fn
    assert wca.SimConfig().kde_bandwidth == 0.070
    base = jobs.effective_base(jobs.load_yaml(os.path.join(ROOT, "configs/wca_phase_diagram_production.yaml")), "production")
    assert float(base["kde_bandwidth"]) == 0.070 and int(base["fr_every"]) == 5 and int(base["fr_start_steps"]) == 20000
    # numerically: the FR marginal is the same function of the population under either estimator
    x_grid, dx, _, _ = _eb_grid()
    k_eta, r_eta = eb.gaussian_kernel(0.10, dx, CPU, F64)
    X = torch.tensor(np.random.default_rng(5).uniform(-1.5, 1.5, (1, 256)))
    assert torch.equal(eb.binned_density(X, k_eta, r_eta, dx), eb.binned_density(X.clone(), k_eta, r_eta, dx))


# ----------------------------------------------------------------------------- 9
def test_short_histogram_runs_are_finite_and_record_the_documented_diagnostics():
    for nb in (45, 360):
        recs = eb.simulate_batch(_eb_fixture_batch(abf_estimator="histogram", abf_n_bins=nb), device=CPU, dtype=F64)
        for r in recs:
            for k in ("F_hat", "Fp_hat", "l2_f_t", "l2_fp_t", "l2_fp_nodes_t", "Fp_bins", "C_bins", "M_bins", "Ch_t", "Mh_t"):
                assert np.isfinite(np.asarray(r[k])).all(), k
            assert r["abf_estimator"] == "histogram" and r["abf_n_bins"] == nb and len(r["hist_edges"]) == nb + 1
            assert r["C_bins"].sum() == 64 * 400 and np.array_equal(r["C_bins"], r["Ch_t"][-1])
            assert np.array_equal(r["Fp_bins"], r["M_bins"] / (r["C_bins"] + 1.0))
            assert 0.0 <= r["frac_untrusted_window"] <= 1.0 and r["min_count_window"] >= 0
            assert np.isfinite(r["max_abs_bias_force"]) and r["max_abs_bias_force"] > 0
        assert recs[1]["n_die"] + recs[1]["n_clone"] > 0 and recs[2]["n_die"] + recs[2]["n_clone"] > 0
        # the saved bins reproduce the saved profile exactly (single object, no read-out mismatch)
        r = recs[0]
        x_grid, dx, eval_mask, idx0 = _eb_grid()
        F = eb.HistogramABFEstimator(1, nb, 1.0, x_grid, CPU, F64).pmf_profile(idx0, torch.tensor(r["Fp_bins"][None, :])).numpy()[0]
        assert np.allclose(F - F[eval_mask.numpy()].mean(), r["F_hat"], atol=1e-12)
    for m in ("abf", "fr_uniform"):
        p, s, e = _wca_tiny(abf_estimator="histogram", abf_n_bins=24)
        d = wca.run_sampler_gpu(m, p, s, e, collect_diagnostics=True, verbose=False)
        assert np.isfinite(d["mean_force"]).all() and np.isfinite(d["pmf"]).all() and np.isfinite(d["hist_mf_bins"]).all()
        assert d["abf_estimator"] == "histogram" and d["abf_n_bins"] == 24 and d["hist_mf_bins"].shape == (4, 24)
        assert d["hist_counts"][-1].sum() == 64 * (3000 - 200 + 1)          # post-burn-in production estimator
        assert 0.0 <= d["bias_clip_fraction"] <= 1.0 and np.isfinite(d["bias_absmax"])
        assert d["hist_n_bins_window"] > 0 and 0.0 <= d["hist_frac_untrusted_window"] <= 1.0
    assert d["total_replacement_events"] > 0


# ----------------------------------------------------------------------------- gateway (Experiment C)
def test_gateway_kernel_path_is_bit_identical_to_the_accepted_fixture_and_histogram_path_is_exact():
    import gateway_core as gw
    fx = np.load(os.path.join(HERE, "fixtures", "gateway_pre_transport_fixture.npz"))
    keys = ("l2_f_t", "l2_fp_t", "F_hat", "Fp_hat", "ess_t", "P_regions", "Sf_t", "C_t", "kl_uniform_t")
    cfg = gw.GatewayConfig(beta=16.0, H=0.5, s=0.10, r=32.0, N=256, n_steps=2500, save_every=250, dt=4e-4, gamma=1.5)
    for kw in ({}, {"estimator": "kernel", "n_bins": 0}):
        spec = gw.BatchSpec(configs=[gw.GatewayConfig(**{**cfg.__dict__, **kw})], seeds=[0], methods=[gw.ABF, gw.FR_UNIFORM], batch_seed=12345)
        out = {r["method"]: r for r in gw.simulate_batch(spec, device=CPU, store_profiles=True, store_accumulators=True)}
        for m in ("abf", "fr_uniform"):
            for k in keys:
                assert np.array_equal(out[m][k], fx[f"two/{m}/{k}"]), (m, k)
            assert out[m]["abf_estimator"] == "kernel" and "Fp_bins" not in out[m]
    # histogram path: shared estimator, exact PMF from the saved bins, finite records, FR fires
    spec = gw.BatchSpec(configs=[gw.GatewayConfig(**{**cfg.__dict__, "estimator": "histogram", "n_bins": 60})], seeds=[0],
                        methods=[gw.ABF, gw.FR_UNIFORM], batch_seed=12345)
    out = {r["method"]: r for r in gw.simulate_batch(spec, device=CPU, store_profiles=True, store_accumulators=True)}
    x_grid, dx, eval_mask, idx0 = _eb_grid()
    for m in ("abf", "fr_uniform"):
        r = out[m]
        assert r["abf_estimator"] == "histogram" and r["abf_n_bins"] == 60 and len(r["hist_edges"]) == 61
        assert r["C_bins"].sum() == 256 * 2500 and np.array_equal(r["Fp_bins"], r["M_bins"] / (r["C_bins"] + 1.0))
        F = eb.HistogramABFEstimator(1, 60, 1.0, x_grid, CPU, F64).pmf_profile(idx0, torch.tensor(r["Fp_bins"][None, :])).numpy()[0]
        assert np.allclose(F - F[eval_mask.numpy()].mean(), r["F_hat"], atol=1e-12)
        for k in ("l2_f_t", "l2_fp_t", "l2_fp_nodes_t", "Ch_t", "Mh_t", "Sf_t", "C_t"):
            assert np.isfinite(np.asarray(r[k])).all(), k
        assert np.array_equal(r["Ch_t"][-1], r["C_bins"]) and r["C_t"].sum(-1)[-1] == 256 * 2500    # fine grid still fed (h_read* read-out)
    assert out["fr_uniform"]["n_die"] + out["fr_uniform"]["n_clone"] > 0
