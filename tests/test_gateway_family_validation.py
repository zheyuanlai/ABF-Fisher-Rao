"""Validation harness of the mechanism campaign (src/gateway_family_validation.py, scripts/mechanism/{run,analyze}_validation.py;
docs/mechanism/SCIENTIFIC_PLAN.md section 3).

M1  analytic gradients == central differences for the four potentials; the numba kernel == the numpy definitions
M2  exact identities by quadrature: E[f|x] = F*', Var(f|x) closed form, F_model - F* constant (every variant)
M3  EM (and MALA) at alpha = 1, lam = 1 are BITWISE src/gateway_validation.run_chain (bias on/off, ACF recorder on)
M4  the lambda update is y - lam d_yV h + sqrt(2 lam h/beta) zy, x unchanged (replayed bit for bit from the noise)
M5  the passive recorders (ACF, z histogram, crossings) change no bit of the dynamics
M6  MALA is exact for every variant and for the lam preconditioner, at a per-case h where EM's error in the same
    statistic is >= 2x the tolerance (asserted): final snapshot of walkers started in the target is an i.i.d. target
    sample (chi-square, z moments), and the time-accumulated pooled conditional variance is exact while EM's is not
M7  the exact i.i.d. sampler's moments (flat and unbiased laws) for every variant
M8  conditional-ACF estimator: synthetic OU with known tau (rho, tau_int, 1/e time); online in-chain sums == offline
M9  gate arithmetic on synthetic data built to pass, and to fail each gate; NO_DATA never reads as PASS; selection rule
M10 configs -> the seven dynamics; seeds of the run script unique and disjoint from earlier campaigns
"""
import importlib.util
import math
import os
import sys

import numpy as np
import pytest
from numba import njit

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import gateway_family_validation as GF  # noqa: E402
import gateway_validation as GV  # noqa: E402


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", "mechanism", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


AV = _load_script("analyze_validation")
RV = _load_script("run_validation")

MODELS = {"alpha1": GF.make_model("alpha", 1.0), "alpha0.5": GF.make_model("alpha", 0.5), "alpha0": GF.make_model("alpha", 0.0),
          "shift": GF.make_model("shift", kappa=1.0)}
P = dict(beta=16.0, H=0.5, omega_out=1.0, omega_in=32.0, s=0.1)


def points(m):
    xs = np.array([-1.5, -1.0, -0.6, -0.25, -0.21, -0.12, -0.1, -0.05, -0.01, 0.0, 0.01, 0.05, 0.1, 0.12, 0.21, 0.3, 1.0, 1.4])
    out = []
    for x in xs:
        mu = GF.m_centre(np.array([x]), m)[0]
        sd = math.sqrt(GF.cond_var_y(np.array([x]), m)[0])
        for k in (0.0, 0.5, -1.0, 2.0, 5.0):
            out.append((x, mu + k * sd))
    return np.array(out)


# ---------------------------------------------------------------------------------------------------------- M1
@pytest.mark.parametrize("name", list(MODELS))
def test_M1_gradient_fd_and_kernel(name):
    m = MODELS[name]
    pts = points(m)
    worst = 0.0
    eps = 1e-6
    for x, y in pts:
        gx, gy = GF.grad_V_np(np.array(x), np.array(y), m)
        fdx = (GF.V_np(np.array(x + eps), np.array(y), m) - GF.V_np(np.array(x - eps), np.array(y), m)) / (2 * eps)
        fdy = (GF.V_np(np.array(x), np.array(y + eps), m) - GF.V_np(np.array(x), np.array(y - eps), m)) / (2 * eps)
        worst = max(worst, abs(fdx - gx) / max(1.0, abs(gx)), abs(fdy - gy) / max(1.0, abs(gy)))
    assert worst < 1e-6, worst
    vc, a, k, lam = GF.numba_args(m)
    for x, y in pts:
        fx, fy, Fp, mu, om, dom = GF._eval(x, y, vc, a, k, P["H"], P["omega_out"], P["omega_in"], P["s"], P["beta"])
        gx, gy = GF.grad_V_np(np.array(x), np.array(y), m)
        assert abs(fx - gx) <= 1e-12 * max(1.0, abs(gx)) and abs(fy - gy) <= 1e-12 * max(1.0, abs(gy))
        assert abs(Fp - GF.Fp_exact(np.array(x), m)) <= 1e-13 * max(1.0, abs(Fp))
        assert abs(mu - GF.m_centre(np.array([x]), m)[0]) <= 1e-14
        sig, sdf = GF._cond_sd(om, dom, vc, a, k, P["beta"])
        assert abs(sig - math.sqrt(GF.cond_var_y(np.array([x]), m)[0])) <= 1e-14 * sig
        assert abs(sdf - math.sqrt(GF.cond_var_f(np.array([x]), m)[0])) <= 1e-12 * max(sdf, 1e-300)
        Vk = GF._V(x, y, om, vc, a, k, P["H"], P["omega_out"], P["omega_in"], P["s"], P["beta"])
        assert abs(Vk - GF.V_np(np.array(x), np.array(y), m)) <= 1e-12 * max(1.0, abs(Vk))
    # alpha = 1: the original gateway (gateway_validation) force, bit for bit
    if name == "alpha1":
        for x, y in pts:
            assert GF._eval(x, y, vc, a, k, P["H"], 1.0, 32.0, 0.1, 16.0)[:2] == GV._forces(x, y, 0.5, 1.0, 32.0, 0.1)[:2]


# ---------------------------------------------------------------------------------------------------------- M2
@pytest.mark.parametrize("name", list(MODELS))
def test_M2_exact_identities_quadrature(name):
    m = MODELS[name]
    t, w = np.polynomial.hermite.hermgauss(60)
    xs = np.array([-1.4, -1.0, -0.5, -0.21, -0.1, -0.03, 0.0, 0.02, 0.15, 0.3, 0.8, 1.2])
    logZ = []
    for x in xs:
        mu = GF.m_centre(np.array([x]), m)[0]
        sd = math.sqrt(GF.cond_var_y(np.array([x]), m)[0])
        y = mu + math.sqrt(2) * sd * t
        f, _ = GF.grad_V_np(np.full_like(y, x), y, m)
        Ef = np.sum(w * f) / math.sqrt(math.pi)
        Vf = np.sum(w * f * f) / math.sqrt(math.pi) - Ef ** 2
        assert abs(Ef - GF.Fp_exact(np.array(x), m)) < 1e-12 * max(1, abs(Ef))
        assert abs(Vf - GF.cond_var_f(np.array([x]), m)[0]) < 1e-10 * max(1e-3, Vf)
        # Z(x) = int exp(-beta V) dy on a wide trapezoid grid
        yy = np.linspace(mu - 12 * sd, mu + 12 * sd, 20001)
        logZ.append(math.log(np.trapezoid(np.exp(-m["beta"] * GF.V_np(np.full_like(yy, x), yy, m)), yy)))
    Fm = -np.array(logZ) / m["beta"]
    d = Fm - GF.F_exact(xs, m)
    assert np.ptp(d) < 1e-10, np.ptp(d)


# ---------------------------------------------------------------------------------------------------------- M3
@pytest.mark.parametrize("scheme", [GF.EM, GF.MALA])
@pytest.mark.parametrize("flat", [0, 1])
def test_M3_alpha1_lam1_bitwise_gateway_validation(scheme, flat):
    m = MODELS["alpha1"]
    g = np.random.default_rng(3)
    N, n, h = 12, 6000, 1e-4
    x0 = np.concatenate([g.uniform(-1.5, 1.5, N - 3), [-1.0, -0.21, 0.0]])
    y0 = g.standard_normal(N) / (4 * GV.omega(x0, P))
    ref = GV.run_chain(scheme, x0.copy(), y0.copy(), n, 700, h, 16.0, 0.5, 1.0, 32.0, 0.1, flat, 1440, 11, 37, 5)
    r = GF.chain(m, scheme, x0, y0, n, 700, h, flat, 11, trace_every=37, n_trace=5, n_acf=8, acf=GF.acf_plan(m, h), acf_rec=50)
    C, Mf, Sy, Sy2, nacc, nprop, nref, mdx, nf, tr, X, Y = ref
    for a, b in ((C, r["A1"][0]), (Mf, r["A1"][1]), (Sy, r["A1"][3]), (Sy2, r["A1"][4]), (tr, r["traces_x"]), (X, r["X"]), (Y, r["Y"])):
        assert np.array_equal(a, b)
    assert (nacc, nprop, nref, mdx, nf) == (r["n_acc"], r["n_prop"], r["n_reflect"], r["max_dx"], r["nonfinite"])
    # alpha family: the transverse residual IS y
    assert np.array_equal(r["A1"][7], r["A1"][3]) and np.array_equal(r["A1"][8], r["A1"][4])
    # the production-bin accumulators are the same samples (counts / sums agree with the fine ones up to bin edges)
    assert r["A2"][0].sum() == r["A1"][0].sum() == N * (n - 700)
    assert np.allclose(r["A2"][1].sum(), r["A1"][1].sum(), rtol=1e-10, atol=1e-8)


@pytest.mark.parametrize("variant,alpha,kappa,lam", [("alpha", 1.0, None, 1.0), ("alpha", 0.5, None, 1.0), ("alpha", 0.0, None, 1.0),
                                                     ("shift", None, 1.0, 1.0), ("alpha", 1.0, None, 0.1),
                                                     ("alpha", 0.5, None, 0.25), ("shift", None, 1.0, 0.5)])
def test_M3b_unbiased_em_is_the_production_engine(variant, alpha, kappa, lam):
    """The harness EM without bias == src/gateway_family_numba.run_arm ('abf', N = 1, min_count 1e300 so the ABF bias
    is ~1e-300 and below every ulp) bit for bit, for every variant and lam: same force, same y-update, same draws (at
    N = 1 the engine's 'all zx then all zy' order is the harness's 'zx, zy per walker'), same deposits.  This is what
    makes the gate a statement about the production dynamics (the analogue of tests/test_gateway_validation.py H1)."""
    GFN = pytest.importorskip("gateway_family_numba")
    h, n = 2.5e-5, 6000
    cfg = dict(variant=variant, alpha=alpha, kappa=kappa, lam=lam, h=h, min_count=1e300)
    m = GF.make_model(variant, alpha if alpha is not None else 1.0, kappa if kappa is not None else 1.0, lam)
    for x0 in (np.array([-0.2]), np.array([0.03]), np.array([-1.0])):
        y0 = GF.initial_y(x0, np.array([0.7]), m)
        res = GFN.run_arm(cfg, "abf", 1, 8100, n, np.array([n], np.int64), x0=x0.copy(), y0=y0.copy(), noise_seed=77,
                          diagnostics=False)
        r = GF.chain(m, GF.EM, x0, y0, n, 0, h, 0, 77)
        assert res["X_final"][0] == r["X"][0] and res["Y_final"][0] == r["Y"][0]
        assert np.array_equal(res["C_all"][-1], r["A2"][0]) and np.array_equal(res["M_all"][-1], r["A2"][1])


# ---------------------------------------------------------------------------------------------------------- M4
@njit(cache=True)
def _normals(seed, n):
    np.random.seed(seed)
    out = np.empty(n)
    for i in range(n):
        out[i] = np.random.standard_normal()
    return out


@pytest.mark.parametrize("name,lam", [("alpha1", 0.1), ("alpha0.5", 0.25), ("shift", 0.5), ("alpha0", 0.1)])
def test_M4_lambda_update_replayed(name, lam):
    base = MODELS[name]
    m = GF.make_model(base["variant"], base["alpha"] if base["variant"] == "alpha" else 1.0, 1.0, lam)
    n, h = 400, 2.5e-5
    x0, y0 = np.array([-0.2]), np.array([0.05])
    r = GF.chain(m, GF.EM, x0, y0, n, 0, h, 1, 5)
    z = _normals(5, 2 * n)
    x, y = x0[0], y0[0]
    vc, a, k, _ = GF.numba_args(m)
    amp, ampy = math.sqrt(2 * h / 16.0), math.sqrt(2 * lam * h / 16.0)
    for s in range(n):
        fx, fy, Fp, mu, om, dom = GF._eval(x, y, vc, a, k, 0.5, 1.0, 32.0, 0.1, 16.0)
        x, y = GF._reflect(x + (-fx + Fp) * h + amp * z[2 * s], -1.8, 1.8), y - lam * fy * h + ampy * z[2 * s + 1]
    assert x == r["X"][0] and y == r["Y"][0]


# ---------------------------------------------------------------------------------------------------------- M5
def test_M5_recorders_inert():
    m = GF.make_model("shift", kappa=1.0, lam=0.25)
    g = np.random.default_rng(2)
    x0 = np.concatenate([[-1.0, -0.21, 0.0, 0.01], g.uniform(-1.5, 1.5, 28)])
    y0 = GF.initial_y(x0, g.standard_normal(32), m)
    h = 2.5e-5
    a = GF.chain(m, GF.EM, x0, y0, 20000, 100, h, 1, 9)
    b = GF.chain(m, GF.EM, x0, y0, 20000, 100, h, 1, 9, trace_every=7, n_trace=4, n_acf=32, acf=GF.acf_plan(m, h), acf_rec=500)
    for k in ("A1", "A2", "X", "Y", "crossings", "zhist"):
        assert np.array_equal(a[k], b[k])
    assert b["acf0"][..., 0].sum() > 0


# ---------------------------------------------------------------------------------------------------------- M6
def _chi2_sf(stat, dof):
    from scipy.stats import chi2
    return chi2.sf(stat, dof)


# (name, lam, h, n_steps, |x| region of the pooled variance (None: every bin), tolerance).  Each h is chosen so that EM's
# error in the SAME statistic is several times the tolerance (measured, 4096 walkers, 2026-10-10 review fix: at the old
# common h 4e-4 EM read +0.7 % (alpha 0.5), +2.6 % (alpha 1, lam 0.1) and ~0 for alpha 0 / the shift, so the test could
# not tell MALA from EM): MALA / EM = alpha1 +0.3 / +24 %, alpha0.5 -0.4 / +12 %, alpha0 +0.3 / +5.2 %, shift -0.4 /
# +7.8 % (the shift's EM error is the coupled curved-channel error, not the frozen-x +0.4 %), alpha1 lam 0.1 +0.7 / +12 %;
# MALA's sd of the statistic 0.13-0.36 %, so every tolerance is >= 5.5 sd.  alpha 0 under the flat bias: x is free
# Brownian motion (exact for EM too), y an OU with EM inflation 1/(1 - lam h/2) -> h 0.1.
M6_CASES = [("alpha1", 1.0, 4e-4, 5000, 0.05, 0.03), ("alpha0.5", 1.0, 6e-3, 5000, 0.05, 0.025),
            ("alpha0", 1.0, 0.1, 2000, None, 0.01), ("shift", 1.0, 8e-3, 5000, 0.3, 0.02), ("alpha1", 0.1, 2e-3, 5000, 0.05, 0.03)]


def _pooled_resid_var_rel(r, m, h, region):
    ref = GF.bin_averages(m, GF.NB_PROD, h=h)
    A = r["A2"]
    sel = (np.abs(ref["centres"]) < region) if region is not None else np.ones(GF.NB_PROD, bool)
    sel &= A[0] > 0
    v = (A[8][sel] - A[7][sel] ** 2 / A[0][sel]).sum() / A[0][sel].sum()
    return v / ((ref["var_flat"][sel] * A[0][sel]).sum() / A[0][sel].sum()) - 1


@pytest.mark.parametrize("name,lam,h,n,region,tol", M6_CASES)
def test_M6_mala_exact(name, lam, h, n, region, tol):
    base = MODELS[name]
    m = GF.make_model(base["variant"], base["alpha"] if base["variant"] == "alpha" else 1.0, 1.0, lam)
    N = 4096
    g = np.random.default_rng(17)
    x0 = g.uniform(-1.8, 1.8, N)
    y0 = GF.initial_y(x0, g.standard_normal(N), m)
    r = GF.chain(m, GF.MALA, x0, y0, n, 0, h, 1, 23)
    assert r["nonfinite"] == 0 and r["n_acc"] / r["n_prop"] > 0.9
    # (a) final snapshot = i.i.d. sample of the flat-bias target
    cnt = np.bincount(GF.bin_index(r["X"], 20), minlength=20)
    stat = ((cnt - N / 20) ** 2 / (N / 20)).sum()
    assert _chi2_sf(stat, 19) > 1e-4, stat
    z = (r["Y"] - GF.m_centre(r["X"], m)) / np.sqrt(GF.cond_var_y(r["X"], m))
    assert abs(z.mean()) < 4 / math.sqrt(N) and abs((z * z).mean() - 1) < 4 * math.sqrt(2 / N)
    flank = (np.abs(r["X"]) > 0.1) & (np.abs(r["X"]) < 0.35)          # where omega'/omega (and the shifted centre) moves most
    zf = z[flank]
    assert abs(zf.mean()) < 4 / math.sqrt(len(zf)) and abs((zf * zf).mean() - 1) < 4 * math.sqrt(2 / len(zf))
    # (b) time-accumulated conditional variance (pooled residual variance in the region) is exact for MALA ...
    rel = _pooled_resid_var_rel(r, m, h, region)
    assert abs(rel) < tol, rel
    # ... and the SAME statistic rejects EM (same start, same h): the test has power against an inexact sampler
    rel_e = _pooled_resid_var_rel(GF.chain(m, GF.EM, x0, y0, n, 0, h, 1, 23), m, h, region)
    assert rel_e > 2 * tol, (rel_e, tol)


# ---------------------------------------------------------------------------------------------------------- M7
@pytest.mark.parametrize("name", list(MODELS))
@pytest.mark.parametrize("flat", [True, False])
def test_M7_exact_sampler_moments(name, flat):
    m = MODELS[name]
    n = 4_000_000
    A1, A2, zh = GF.exact_samples(n, m, 5, flat=flat)
    ref = GF.bin_averages(m, GF.NB_PROD)
    C = A2[0]
    assert C.sum() == n and A1[0].sum() == n
    pr = np.full(GF.NB_PROD, 1 / GF.NB_PROD) if flat else ref["P"] / ref["P"].sum()
    ok = pr * n > 2e4
    se_p = np.sqrt(pr * (1 - pr) / n)
    assert np.max(np.abs(C[ok] / n - pr[ok]) / se_p[ok]) < 5
    # conditional laws per bin (flat law: uniform x in the bin; the unbiased law is checked on the well bins where the
    # bin-averaged exact quantities under exp(-beta F*) are the 'var'/'Fp' entries)
    Cn = np.maximum(C, 1)
    mres = A2[7] / Cn
    vres = A2[8] / Cn - mres ** 2
    vex = ref["var_flat"] if flat else ref["var"]
    assert np.max(np.abs(mres[ok]) / np.sqrt(vex[ok] / C[ok])) < 5
    assert np.max(np.abs(vres[ok] / vex[ok] - 1) / np.sqrt(2 / C[ok])) < 5
    mf = A2[1] / Cn
    fex = ref["Fp_flat"] if flat else ref["Fp"]
    vf = A2[6] / Cn - (A2[5] / Cn) ** 2
    if flat:
        sd = np.sqrt(np.maximum(ref["varf_flat"], 1e-30) + ref["Fp2_flat"] - ref["Fp_flat"] ** 2)
        assert np.max(np.abs(mf[ok] - fex[ok]) / (sd[ok] / np.sqrt(C[ok]))) < 5
        big = ok & (ref["varf_flat"] > 0.05)
        if big.any():
            assert np.max(np.abs(vf[big] / ref["varf_flat"][big] - 1) / np.sqrt(14 / C[big])) < 5
        else:
            assert np.max(vf[ok] / np.maximum(A2[2][ok] / Cn[ok], 1e-300)) < 1e-12
    # z_y histogram in the windows ~ N(0,1)
    zb = np.linspace(-GF.ZH_LIM, GF.ZH_LIM, GF.ZH_NB + 1)
    for w in range(len(GF.ACF_WINDOWS)):
        if zh[w].sum() < 1e4:
            continue
        cdf = 0.5 * (1 + np.vectorize(math.erf)(zb / math.sqrt(2)))
        p = np.diff(cdf)
        e = p * zh[w].sum()
        k = e > 20
        stat = ((zh[w][k] - e[k]) ** 2 / e[k]).sum()
        assert _chi2_sf(stat, k.sum() - 1) > 1e-4


# ---------------------------------------------------------------------------------------------------------- M8
def _ou_traces(T, n, dt, tau, seed):
    g = np.random.default_rng(seed)
    a = math.exp(-dt / tau)
    z = np.empty((T, n))
    z[0] = g.standard_normal(n)
    e = g.standard_normal((T, n)) * math.sqrt(1 - a * a)
    for t in range(1, T):
        z[t] = a * z[t - 1] + e[t]
    return z


def test_M8_acf_estimator_synthetic_ou():
    T, n, dt, tau = 80000, 8, 0.01, 0.2
    z = _ou_traces(T, n, dt, tau, 4)
    zf = (z * z - 1) / math.sqrt(2)
    # x: each walker sits in one window or outside, switching every 5000 samples (conditioning is on the origin)
    xs = np.empty((T, n))
    choices = np.array([-1.0, -0.21, 0.0, 0.7])
    g = np.random.default_rng(1)
    for blk in range(0, T, 5000):
        xs[blk:blk + 5000] = choices[g.integers(0, 4, n)][None, :] + g.uniform(-0.019, 0.019, (1, n))
    lags = np.unique(np.concatenate([np.arange(0, 17), np.round(np.geomspace(16, 150, 40)).astype(int)]))
    acc = sum(GF.acf_offline(xs[:, i], z[:, i], zf[:, i], lags) for i in range(n))
    r = GF.acf_from_sums(acc)
    t = lags * dt
    for w in range(3):
        assert acc[w, 0, 0] > 1e4
        good = t <= 3 * tau
        assert np.max(np.abs(r["rho_y"][w][good] - np.exp(-t[good] / tau))) < 0.06
        assert np.max(np.abs(r["rho_f"][w][good] - np.exp(-2 * t[good] / tau))) < 0.06
        ti = GF.tau_integrated(t, r["rho_y"][w], 5 * tau)
        assert abs(ti / (tau * (1 - math.exp(-5))) - 1) < 0.12, ti
        assert abs(GF.tau_efold(t, r["rho_y"][w]) / tau - 1) < 0.12
        assert abs(GF.tau_integrated(t, r["rho_f"][w], 5 * tau / 2) / (tau / 2 * (1 - math.exp(-5))) - 1) < 0.15
    # exact arithmetic of the integrator on a known function (log-spaced lags)
    assert abs(GF.tau_integrated(t, np.exp(-t / tau), 5 * tau) / (tau * (1 - math.exp(-5))) - 1) < 5e-3
    # the in-chain online accumulator == the offline estimator on the same samples
    R = lags[-1] + 1
    ry, rf, rw = np.zeros((n, R)), np.zeros((n, R)), np.full((n, R), -1, np.int64)
    acc_on = np.zeros((3, len(lags), 11))
    wc = np.array(GF.ACF_WINDOWS)
    for s in range(T):
        for i in range(n):
            GF.acf_push(i, s, z[s, i], zf[s, i], GF._window(xs[s, i], wc, GF.ACF_HALF_WIDTH), ry, rf, rw, lags.astype(np.int64), acc_on)
    assert np.allclose(acc_on, acc, rtol=1e-10, atol=1e-8)


def test_M8_chain_acf_online_equals_offline():
    """In the chain: the lag sums of the run == acf_offline applied to the recorded (x, z_y, z_f) of the same samples."""
    for m in (GF.make_model("alpha", 1.0), GF.make_model("shift", kappa=1.0, lam=0.5)):
        h = 2.5e-5
        plan = GF.acf_plan(m, h)
        x0 = np.array([-1.0, -0.21, -0.2, 0.0, 0.005, -0.01, 0.21, -0.99])
        y0 = GF.initial_y(x0, np.random.default_rng(0).standard_normal(8), m)
        n, burn = 30000, 200
        n_samp0 = (n - burn - 1) // plan["stride"][0] + 1
        n_samp1 = (n - burn - 1) // plan["stride"][1] + 1
        r = GF.chain(m, GF.EM, x0, y0, n, burn, h, 1, 3, n_acf=8, acf=plan, acf_rec=max(n_samp0, n_samp1))
        assert list(r["acf_nsamples"]) == [n_samp0, n_samp1]
        for L in (0, 1):
            rec = r[f"acf_rec{L}"][:r["acf_nsamples"][L]]
            off = sum(GF.acf_offline(rec[:, i, 0], rec[:, i, 1], rec[:, i, 2], r[f"lags{L}"]) for i in range(8))
            assert np.allclose(r[f"acf{L}"], off, rtol=1e-9, atol=1e-9)
            assert off[:, 0, 0].sum() > 0
        # lag-0 second moment of z_y in visited windows is ~1 (normalised by the exact conditional law)
        s = GF.acf_from_sums(r["acf0"])
        n0 = r["acf0"][:, 0, 0]
        assert np.all(np.abs(s["m01_y"][n0 > 500, 0] - 1) < 0.5)


def test_M8_acf_plan_resolves_windows():
    for name in GF.DYNAMICS_ORDER:
        m = {d["name"]: d for d in GF.load_dynamics()}[name]
        for h in (2.5e-5, 1.25e-5):
            p = GF.acf_plan(m, h)
            st = p["stride"] * h
            assert p["stride"][0] <= p["stride"][1]
            assert p["lags1"][-1] * st[1] >= 5.0 / m["lam"] * 0.999        # coarse lags reach >= 5/lam t.u.
            fac = 2.0 if m["variant"] == "alpha" else 1.0
            for c, tau in p["tau_windows"].items():
                ok = [st[L] <= tau / fac / 4 and p[f"lags{L}"][-1] * st[L] >= 5 * tau for L in (0, 1)]
                assert any(ok), (name, h, c, tau, st)


# ---------------------------------------------------------------------------------------------------------- M9
def _exact_recs(m, groups=16, n=1 << 20, seed0=500):
    recs = []
    for g in range(groups):
        A1, A2, zh = GF.exact_samples(n, m, seed0 + g)
        recs.append(dict(A1=A1, A2=A2, zhist=zh, nonfinite=np.int64(0), meta=dict(walker_steps=float(n), wall_s=1.0)))
    return recs


def _copy(recs):
    return [dict(r, A1=r["A1"].copy(), A2=r["A2"].copy()) for r in recs]


@pytest.fixture(scope="module")
def synth():
    return {k: _exact_recs(MODELS[k]) for k in ("alpha1", "shift", "alpha0")}


H_SYN = 2.5e-6     # exact data read as an EM chain with a negligible closed-form error (0.13 % at the gate)


def test_M9_gates_pass_on_exact_data(synth):
    for k, recs in synth.items():
        m = MODELS[k]
        o = AV.analyse_flat(recs, m, H_SYN)
        g = AV.gates_em(o, m)
        assert (g["V1"], g["V2"], g["V3"], g["V3b"], g["V1_strict"]) == (AV.PASS,) * 5, (k, g)
        st, comp = AV.gate_exact_sampler(AV.analyse_flat(recs, m, None), m)
        assert st == AV.PASS, (k, comp)


def test_M9_gates_fail_when_built_to_fail(synth):
    m = MODELS["alpha1"]
    ref1 = GF.bin_averages(m, GF.NB_FINE)
    ref2 = GF.bin_averages(m, GF.NB_PROD)
    c2 = np.argsort(np.abs(ref1["centres"]))[:2]
    cen = ref2["centres"]
    W = (cen >= -1.5) & (cen <= 1.5)

    def gates(recs, h=H_SYN, mm=m):
        return AV.gates_em(AV.analyse_flat(recs, mm, h), mm)

    # V1: conditional variance +10 % in the two central fine bins
    r = _copy(synth["alpha1"])
    for x in r:
        for j in c2:
            x["A1"][8, j] += 0.10 * x["A1"][0, j] * ref1["var_flat"][j]
    g = gates(r)
    assert g["V1"] == AV.FAIL and (g["V2"], g["V3"], g["V3b"]) == (AV.PASS,) * 3
    # V1 closed form: h 1e-4 gives 5.4 % > 2 % at the gate
    assert gates(synth["alpha1"], h=1e-4)["V1"] == AV.FAIL
    # V2: a smooth 0.01 kT-scale bump in the density (all accumulators of a bin scaled together: mean force unchanged)
    r = _copy(synth["alpha1"])
    fac = np.exp(-16.0 * 0.01 * np.exp(-(cen / 0.3) ** 2))
    for x in r:
        x["A2"] *= fac[None, :]
    g = gates(r)
    assert g["V2"] == AV.FAIL and (g["V1"], g["V3"], g["V3b"]) == (AV.PASS,) * 3
    # V3: alternating +-0.03 mean-force error (its trapezoid integral is 0, so F_MF is untouched)
    r = _copy(synth["alpha1"])
    sgn = np.where(np.arange(GF.NB_PROD) % 2 == 0, 1.0, -1.0) * W
    for x in r:
        x["A2"][1] += 0.03 * sgn * x["A2"][0]
    g = gates(r)
    assert g["V3"] == AV.FAIL and (g["V1"], g["V2"], g["V3b"]) == (AV.PASS,) * 3
    # V3b: conditional force variance +12 %
    r = _copy(synth["alpha1"])
    for x in r:
        x["A2"][6] *= 1.12
    g = gates(r)
    assert g["V3b"] == AV.FAIL and (g["V1"], g["V2"], g["V3"]) == (AV.PASS,) * 3
    # V3b alpha = 0: any y-dependence of the force (variance 1e-6 x mean f^2)
    r = _copy(synth["alpha0"])
    for x in r:
        x["A2"][6] += 1e-6 * x["A2"][2]
    assert gates(r, mm=MODELS["alpha0"])["V3b"] == AV.FAIL
    # shifted fibre: conditional mean off by 0.5 sd in the gate bins; pooled |x| <= 0.3 variance +5 %
    ms = MODELS["shift"]
    r = _copy(synth["shift"])
    for x in r:
        for j in c2:
            x["A1"][7, j] += 0.5 * 0.25 * x["A1"][0, j]
    g = gates(r, mm=ms)
    assert g["V1"] == AV.FAIL and g["V1_components"]["shift_mean_3se"] == [False, False]
    r = _copy(synth["shift"])
    sel = np.abs(cen) <= 0.3 + 1e-12
    for x in r:
        x["A2"][8, sel] *= 1.05
    g = gates(r, mm=ms)
    assert g["V1"] == AV.FAIL and g["V1_components"]["shift_var_region_2pct"] is False
    # V4 on an "exact sampler" with a density bump
    r = _copy(synth["shift"])
    for x in r:
        x["A2"] *= fac[None, :]
    assert AV.gate_exact_sampler(AV.analyse_flat(r, ms, None), ms)[0] == AV.FAIL


def test_M9_no_data_never_passes(synth):
    m = MODELS["alpha1"]
    o = AV.analyse_flat(synth["alpha1"][:1], m, H_SYN)
    g = AV.gates_em(o, m)
    assert AV.PASS not in (g["V1"], g["V2"], g["V3"], g["V3b"])
    assert AV.gate_exact_sampler(AV.analyse_flat(synth["alpha1"][:1], m, None), m)[0] != AV.PASS
    assert AV.gates_em(AV.analyse_flat([], m, H_SYN), m) == dict(V1=AV.NO_DATA, V2=AV.NO_DATA, V3=AV.NO_DATA, V3b=AV.NO_DATA)
    assert AV.status(True, None) == AV.NO_DATA and AV.status(False, None) == AV.FAIL and AV.status(True, [True, True]) == AV.PASS


def test_M9_v5_and_abf_smoke_hook():
    o = dict(n_groups=16, nonfinite=0, cross_frac_min=0.9)
    unb = dict(nonfinite=0, reflections_per_walker_step=0.0)
    good = [dict(seed=8100, finite=True, max_abs_Gamma_visited=1.0, max_abs_Fp_visited=1.5),
            dict(seed=8101, finite=True, max_abs_Gamma_visited=2.0, max_abs_Fp_visited=1.5)]
    assert AV.gate_v5(o, unb, good)["V5"] == AV.PASS
    assert AV.gate_v5(o, unb, [])["V5"] == AV.PENDING and AV.gate_v5(o, unb, [])["V5_core"] == AV.PASS
    assert AV.gate_v5(o, unb, good[:1])["V5"] == AV.FAIL                          # < 2 seeds
    bad = [dict(good[0]), dict(good[1], max_abs_Gamma_visited=3.0)]                      # 3.0 >= 2 x 1.5
    assert AV.gate_v5(o, unb, bad)["V5"] == AV.FAIL
    assert AV.gate_v5(o, unb, [dict(good[0], finite=False), good[1]])["V5"] == AV.FAIL
    assert AV.gate_v5(dict(o, nonfinite=1), unb, good)["V5"] == AV.FAIL
    assert AV.gate_v5(dict(o, cross_frac_min=0.3), unb, good)["V5"] == AV.FAIL
    assert AV.gate_v5(o, dict(unb, reflections_per_walker_step=1e-5), good)["V5"] == AV.FAIL
    assert AV.gate_v5(o, None, good)["V5"] == AV.NO_DATA
    assert AV.gate_v5(dict(n_groups=0), unb, good)["V5"] == AV.NO_DATA


def test_M9_selection_rule():
    names = list(GF.DYNAMICS_ORDER)
    P_, F_, N_, Q_ = AV.PASS, AV.FAIL, AV.NO_DATA, AV.PENDING

    def tab(a, b=None):
        t = {d: {"2.5e-05": dict(full=a.get(d, P_), core=a.get(d, P_))} for d in names}
        if b is not None:
            for d in names:
                t[d]["1.25e-05"] = dict(full=b.get(d, P_), core=b.get(d, P_))
        return t
    assert AV.select_common_h(tab({}), names)["h_common"] == 2.5e-5
    s = AV.select_common_h(tab({"lam0.1": F_}), names)
    assert s["verdict"] == "REFINE_REQUIRED" and s["next_h"] == 1.25e-5 and s["failing"] == ["lam0.1"]
    s = AV.select_common_h(tab({"lam0.1": F_}, {}), names)
    assert s["verdict"] == "PASS" and s["h_common"] == 1.25e-5
    s = AV.select_common_h(tab({"shift": F_}, {"shift": F_}), names)
    assert s["verdict"] == "UNRESOLVED" and s["unresolved"] == ["shift"] and s["subset_h"] == 2.5e-5
    s = AV.select_common_h(tab({"shift": F_, "lam0.1": F_}, {"shift": F_}), names)
    assert s["unresolved"] == ["shift"] and s["subset_h"] == 1.25e-5
    assert AV.select_common_h(tab({"alpha1": Q_}), names)["verdict"] == "PENDING"
    assert AV.select_common_h(tab({"alpha1": N_}), names)["verdict"] == "INCOMPLETE"
    assert AV.select_common_h(tab({"alpha1": N_, "shift": F_}), names)["verdict"] == "REFINE_REQUIRED"


# ---------------------------------------------------------------------------------------------------------- M10
def test_M10_dynamics_and_seeds():
    dyn = GF.load_dynamics()
    got = [(d["name"], d["variant"], None if d["variant"] == "shift" else d["alpha"], d["lam"]) for d in dyn]
    assert got == [("alpha1", "alpha", 1.0, 1.0), ("alpha0.5", "alpha", 0.5, 1.0), ("alpha0", "alpha", 0.0, 1.0),
                   ("shift", "shift", None, 1.0), ("lam0.5", "alpha", 1.0, 0.5), ("lam0.25", "alpha", 1.0, 0.25),
                   ("lam0.1", "alpha", 1.0, 0.1)]
    assert dyn[3]["kappa"] == 1.0
    seeds = []
    for j in RV.jobs(list(GF.DYNAMICS_ORDER), list(RV.H_CANDIDATES), list(RV.CHAINS), 16, False):
        c, d, h, gs = j
        seeds += [(c, RV.seed_of(c, d, h, g)) for g in gs]
    vals = [s for _, s in seeds]
    assert len(vals) == len(set(vals))
    assert all(s >= 71000 for s in vals) and not any(8100 <= s <= 8131 for s in vals)
    em = sorted(s for c, s in seeds if c == "em_flat")
    assert em[0] == 71000 and em[-1] == 71000 + 1000 * 6 + 100 * 1 + 15
    # run lengths (frozen): EM 500/lam after 10 t.u.; MALA 500 t.u.
    lam01 = dyn[6]
    assert RV.settings("em_flat", lam01, False) == (5000.0, 10.0) and RV.settings("mala_flat", lam01, False) == (500.0, 10.0)


# ---------------------------------------------------------------------------------------------------------- M11
OLD = os.path.join(ROOT, "results", "equal_budget_v2", "gateway_validation")


@pytest.mark.skipif(not os.path.exists(os.path.join(OLD, "summary.json")), reason="accepted gateway validation data absent")
@pytest.mark.parametrize("tag,h", [("em_flat_h2.5e-05", 2.5e-5), ("em_flat_h0.0001", 1e-4)])
def test_M11_reproduces_frozen_gateway_gate_on_accepted_data(tag, h):
    """The new read-out applied to the ACCEPTED equal-budget gateway validation chains (alpha = 1, lam = 1; 1440-bin
    count / sum f / sum y / sum y^2, read only) reproduces the frozen gate's numbers in summary.json."""
    import glob
    import json
    old = json.load(open(os.path.join(OLD, "summary.json")))["candidates"][f"{h:g}"]
    recs = []
    for f in sorted(glob.glob(os.path.join(OLD, tag, "g*.npz"))):
        with np.load(f) as d:
            A1 = np.zeros((9, GF.NB_FINE))
            A1[0], A1[1], A1[3], A1[4] = d["C"], d["Mf"], d["Sy"], d["Sy2"]
            A1[7], A1[8] = d["Sy"], d["Sy2"]                       # alpha family: residual y - E[Y|x] = y
            A2 = A1.reshape(9, GF.NB_PROD, GF.NB_FINE // GF.NB_PROD).sum(-1)
            recs.append(dict(A1=A1, A2=A2, nonfinite=d["nonfinite"], meta=json.loads(str(d["meta_json"]))))
    assert len(recs) == 16
    o = AV.analyse_flat(recs, MODELS["alpha1"], h)
    for k in ("F_density", "F_MF", "mean_force"):
        for q in ("rms", "D", "noise", "upper95"):
            assert abs(o[k][q] - old[k][q]) <= 2e-6 + 1e-3 * abs(old[k][q]), (k, q, o[k][q], old[k][q])
    assert abs(o["mean_force"]["closed_form_EM_bias_rms"] - old["mean_force"]["closed_form_EM_bias_rms"]) < 1e-6
    # the two central fine bins (the old code listed them in argsort order, this one by x): compare as pairs
    pairs = lambda v: sorted(zip(v["measured"], v["se"], v["predicted_EM"]))
    # (5e-8: the exact bin average is a midpoint rule here, an endpoint-inclusive linspace mean there)
    assert np.allclose(pairs(o["var_centre"]), pairs(old["var_centre"]), rtol=1e-5, atol=1e-6)
    assert abs(o["var_centre"]["closed_form_peak"] - old["var_centre"]["closed_form_peak"]) < 1e-12
    g = AV.gates_em(o, MODELS["alpha1"])
    assert (g["V1"] == AV.PASS) == old["gates"]["V1"] and (g["V2"] == AV.PASS) == old["gates"]["V2"]
    assert (g["V1_strict"] == AV.PASS) == old["gates"]["V1"]          # the gateway V1 text, literally
    assert (g["V3"] == AV.PASS) == old["gates"]["V3"]


# ---------------------------------------------------------------------------------------------------------- M12
# review fixes 2026-10-10: V1 plan vs strict reading, V4 routing, multiplicity numbers, design check, sign-invariant
# z_f, the coupled-shift ACF plan and the measured 'resolved' flag
def test_M12_v1_plan_vs_strict_reading(synth):
    """V1 gates on the plan's 'measured <= 0.02 + 2 se'; the gateway config's extra |m - closed form| <= 3 se clause is
    the strict reading (V1_strict), reported but not in the per-h status."""
    m = MODELS["alpha1"]
    o = AV.analyse_flat(synth["alpha1"], m, H_SYN)
    o["var_centre"].update(measured=[0.025, 0.025], se=[0.0065, 0.0065], predicted_EM=[0.0013, 0.0013])
    g = AV.gates_em(o, m)                                   # within 2 % + 2 se, but 3.65 se above the closed form
    assert g["V1"] == AV.PASS and g["V1_strict"] == AV.FAIL and g["V1_consistency_3se"] == [False, False]
    o["var_centre"].update(measured=[0.04, 0.0])
    g = AV.gates_em(o, m)                                   # 0.04 - 2 x 0.0065 > 0.02
    assert g["V1"] == AV.FAIL and g["V1_strict"] == AV.FAIL


def test_M12_v4_routes_to_harness_fail_never_refinement():
    names = list(GF.DYNAMICS_ORDER)
    P_, F_ = AV.PASS, AV.FAIL
    tab = {d: {"2.5e-05": dict(full=P_, core=P_)} for d in names}
    v4 = {d: P_ for d in names}
    s = AV.decide(tab, names, v4)
    assert s["verdict"] == "PASS" and s["h_common"] == 2.5e-5
    s = AV.decide(tab, names, dict(v4, shift=F_))
    assert s["verdict"].startswith("HARNESS_FAIL") and s["h_common"] is None and s["h_selection_h"] == 2.5e-5
    assert s["harness"]["V4_fail"] == ["shift"]
    s = AV.decide(tab, names, {d: v for d, v in v4.items() if d != "alpha0"})
    assert s["verdict"].startswith("INCOMPLETE") and s["h_common"] is None
    tab2 = {d: {"2.5e-05": dict(full=F_ if d == "lam0.1" else P_, core=F_ if d == "lam0.1" else P_)} for d in names}
    assert AV.decide(tab2, names, v4)["verdict"] == "REFINE_REQUIRED"          # a per-h failure is still a refinement
    s = AV.decide(tab2, names, dict(v4, **{"lam0.1": F_}))
    assert s["verdict"].startswith("HARNESS_FAIL") and s["h_selection_verdict"] == "REFINE_REQUIRED"
    # the three-se-only classifier of a failing V4 component set
    assert AV.three_se_only(dict(central_var_3se=[True, False], F_density_upper=True, finite=True))
    assert not AV.three_se_only(dict(central_var_3se=[True, False], F_density_upper=False))
    assert not AV.three_se_only(dict(central_var_3se=[True, True]))


def test_M12_multiplicity_numbers():
    assert abs(AV.t_two_sided(3.0, 15) - 0.0089727) < 1e-6
    assert abs(AV.family_rate(32, 15) - 0.2506) < 1e-3 and abs(AV.family_rate(14, 15) - 0.1185) < 1e-3
    f = AV.family_summary([0.1, -3.2, 1.0, float("nan")], 15)
    assert f["n_tests"] == 3 and f["n_exceed"] == 1 and abs(f["max_abs_z"] - 3.2) < 1e-12
    assert abs(f["bonferroni_p_of_max"] - min(1.0, 3 * AV.t_two_sided(3.2, 15))) < 1e-12


def _meta(chain, d, h_, g_, m, **kw):
    """A conforming meta of the frozen design (kw overrides fields, e.g. h=..., group=...)."""
    if chain.startswith("exact"):
        flat = chain == "exact_flat"
        mt = dict(chain=chain, dynamics=m["name"], group=g_, seed=RV.seed_of(chain, d, h_, g_), smoke=False,
                  n_draws=RV.N_EXACT_FLAT if flat else RV.N_EXACT_UNB, flat=flat)
    else:
        T, burn = RV.settings(chain, m, False)
        mt = dict(chain=chain, dynamics=m["name"], group=g_, seed=RV.seed_of(chain, d, h_, g_), smoke=False, h=h_, T=T, burn=burn,
                  n_walkers=8 if chain == "em_unbiased" else RV.NW, flat_bias=0 if chain == "em_unbiased" else 1,
                  n_steps=int(round((T + burn) / h_)))
    mt.update(kw)
    return dict(meta=mt)


def test_M12_design_check():
    dyn = GF.load_dynamics()
    for d, m in enumerate(dyn):
        for chain, h in (("em_flat", 2.5e-5), ("em_flat", 1.25e-5), ("mala_flat", 1e-4), ("exact_flat", None), ("exact_unbiased", None)):
            assert AV.design_issues([_meta(chain, d, h, g, m) for g in range(16)], chain, m, d, h) == [], (m["name"], chain)
        assert AV.design_issues([_meta("em_unbiased", d, 2.5e-5, 0, m)], "em_unbiased", m, d, 2.5e-5) == []
    d, h = 6, 2.5e-5
    m = dyn[d]                                              # lam0.1: T = 5000 t.u.
    good = [_meta("em_flat", d, h, g, m) for g in range(16)]
    assert AV.design_issues(good[:2], "em_flat", m, d, h)                                   # 2 groups
    for bad in (dict(n_walkers=2), dict(T=500.0), dict(burn=1.0), dict(smoke=True), dict(seed=61000), dict(h=1.25e-5),
                dict(group=16)):
        assert AV.design_issues(good[:15] + [_meta("em_flat", d, h, 15, m, **bad)], "em_flat", m, d, h), bad
    sm = [_meta("em_flat", d, h, g, m, smoke=True) for g in range(16)]
    assert AV.design_issues(sm, "em_flat", m, d, h) and not AV.design_issues(sm, "em_flat", m, d, h, smoke=True)
    assert AV.design_issues([_meta("exact_flat", d, None, g, m, n_draws=1 << 20) for g in range(16)], "exact_flat", m, d, None)


def test_M12_partial_design_never_passes(tmp_path):
    """The partial-run scenario (2 groups, tiny runs, filed in a gate directory without --smoke): excluded by the design
    check, never PASS; with --smoke the same files are analysed and the verdict is labelled."""
    import json
    m = GF.load_dynamics()[0]
    for tg, chain in (("em_flat_h2.5e-05", "em_flat"), ("mala_flat_h0.0001", "mala_flat"), ("exact_flat", "exact_flat")):
        p = tmp_path / "alpha1" / tg
        p.mkdir(parents=True)
        for g in range(2):
            A1, A2, zh = GF.exact_samples(1 << 18, m, 900 + g)
            meta = dict(chain=chain, dynamics="alpha1", group=g, smoke=False, walker_steps=float(1 << 18), wall_s=1.0,
                        trace_every=1, T=0.01, n_walkers=2)
            np.savez(p / f"g{g:02d}.npz", meta_json=json.dumps(meta), A1=A1, A2=A2, zhist=zh, nonfinite=np.int64(0),
                     crossings=np.ones(2, np.int32), traces_x=np.zeros((1, 1)), n_acc=1.0, n_prop=1.0)
    p = tmp_path / "alpha1" / "em_unbiased_h2.5e-05"
    p.mkdir(parents=True)
    np.savez(p / "g00.npz", meta_json=json.dumps(dict(chain="em_unbiased", dynamics="alpha1", group=0, smoke=False,
                                                      walker_steps=8.0, wall_s=1.0, trace_every=1)),
             nonfinite=np.int64(0), n_reflect=0.0, max_dx=0.0, traces_x=np.zeros((1, 1)))
    S = AV.analyse_all(str(tmp_path), str(tmp_path / "none.json"), smoke=False)
    D = S["dynamics"]["alpha1"]
    assert set(D["design"]) == {"em_flat_h2.5e-05", "mala_flat_h0.0001", "exact_flat", "em_unbiased_h2.5e-05"}
    e = D["per_h"]["2.5e-05"]
    assert e["status"] != AV.PASS and e["gates"]["V1"] == AV.NO_DATA and D["V4"]["status"] == AV.NO_DATA
    assert not S["selection"]["verdict"].startswith("PASS") and S["selection"]["h_common"] is None
    S2 = AV.analyse_all(str(tmp_path), str(tmp_path / "none.json"), smoke=True)
    assert S2["dynamics"]["alpha1"]["per_h"]["2.5e-05"]["em_flat"]["n_groups"] == 2
    assert S2["selection"]["verdict"].startswith("SMOKE")


def test_M12_zf_sign_invariant():
    """The recorded z_f is (z_y^2 - 1)/sqrt 2 (alpha family) and -z_y (shift) on BOTH sides of x = 0: the plain
    normalised fluctuation flips sign with omega' at x = 0, so its ACF in the x = 0 window would count x crossings."""
    every = dict(stride=np.array([1, 1], np.int64), lags0=np.array([0, 1], np.int64), lags1=np.array([0, 1], np.int64))
    for m in (GF.make_model("alpha", 1.0), GF.make_model("alpha", 0.5, lam=0.25), GF.make_model("shift", kappa=1.0)):
        x0 = np.array([-0.03, -0.01, 0.0, 0.01, 0.02, -0.2, 0.2, 0.05])
        y0 = GF.initial_y(x0, np.random.default_rng(1).standard_normal(8), m)
        r = GF.chain(m, GF.EM, x0, y0, 3000, 0, 2.5e-5, 1, 5, n_acf=8, acf=every, acf_rec=3000)
        rec = r["acf_rec0"][:r["acf_nsamples"][0]]
        x, zy, zf = rec[..., 0], rec[..., 1], rec[..., 2]
        want = (zy * zy - 1) / math.sqrt(2) if m["variant"] == "alpha" else -zy
        ok = np.sqrt(GF.cond_var_f(x, m)) > 1e-6
        assert ok.sum() > 1000 and (x[ok] > 0).sum() > 100 and (x[ok] < 0).sum() > 100
        assert np.max(np.abs(zf[ok] - want[ok])) < 1e-6 * max(1.0, float(np.max(np.abs(want[ok]))))


def test_M12_acf_plan_coupled_shift():
    """Shifted fibre: the level-0 stride resolves the coupled relaxation 1/(kappa^2 (lam + max m'^2)) (~0.03 t.u., not
    the frozen-x 1/(lam kappa^2) = 1) and still reaches 5 frozen tau; the alpha family's plan is the frozen-x one."""
    dyn = {d["name"]: d for d in GF.load_dynamics()}
    for h in (2.5e-5, 1.25e-5):
        p = GF.acf_plan(dyn["shift"], h)
        assert abs(p["tau_couple"] - 1.0 / (1.0 + (2.0 / 16.0) * 16.2477 ** 2)) < 2e-4
        assert p["stride"][0] * h <= p["tau_couple"] / 10 and p["lags0"][-1] * p["stride"][0] * h >= 5.0
        for name in ("alpha1", "alpha0.5", "lam0.1"):
            q = GF.acf_plan(dyn[name], h)
            st1 = max(1, int(math.floor(q["tau_max"] / (20 * h))))
            assert q["tau_fast"] == q["tau_min"] and q["stride"][0] == min(st1, max(1, int(math.floor(q["tau_min"] / (10 * h)))))


def test_M12_acf_resolved_flag_from_measured_decay():
    """'resolved' comes from the MEASURED rho at the first nonzero lag: a tau = 0.04 process sampled every 0.05 t.u. (the
    old shifted-fibre stride, frozen-x tau 1) is flagged unresolved; sampled every 0.002 t.u. it is resolved."""
    m = GF.make_model("shift", kappa=1.0)
    h, tau = 2.5e-5, 0.04

    def recs(stride_steps, L, n_samp):
        lags = np.unique(np.concatenate([np.arange(0, 17), np.round(np.geomspace(16, L, 60)).astype(np.int64)]))
        out = []
        for g in range(2):
            acc = np.zeros((3, len(lags), 11))
            for i, c in enumerate(GF.ACF_WINDOWS):
                z = _ou_traces(n_samp, 1, stride_steps * h, tau, 10 * g + i)[:, 0]
                acc += GF.acf_offline(np.full(n_samp, c), z, -z, lags)
            out.append(dict(acf0=acc, acf1=acc, acf_stride=np.array([stride_steps, stride_steps]), lags0=lags, lags1=lags))
        return out
    fine = AV.analyse_acf(recs(80, 2600, 12000), m, h)["windows"]["-0.21"]
    coarse = AV.analyse_acf(recs(2000, 110, 3000), m, h)["windows"]["-0.21"]
    assert fine["reaches_5tau"] and coarse["reaches_5tau"]
    assert fine["resolved"] and fine["resolved_y"] and fine["resolved_f"] and fine["rho_y_first_lag"] > 0.9
    assert not coarse["resolved"] and not coarse["resolved_y"] and coarse["rho_y_first_lag"] < 0.5
    assert abs(fine["tau_e_y"] / tau - 1) < 0.25


# --- Amendment 1 (A3): Holm-Bonferroni over the V4 family of 3-se clauses -------------------------------------------
def test_holm_reject_known_answers():
    import importlib.util, os, sys
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, os.path.join(root, "scripts", "mechanism"))
    import analyze_validation as AV
    # classic example: p = 0.01, 0.04, 0.03, 0.005 at alpha 0.05 -> sorted 0.005 (<= .0125 rej), 0.01 (<= .0167 rej),
    # 0.03 (> .025 stop) -> rejected exactly the first two smallest
    assert AV.holm_reject([0.01, 0.04, 0.03, 0.005]) == [True, False, False, True]
    assert AV.holm_reject([]) == []
    # a single 3.2-se excursion among 32 clauses (t_15) is NOT rejected (unadjusted it would FAIL the 3-se clause)
    p_single = AV.t_two_sided(3.2, 15)
    assert p_single > 0.05 / 32
    assert AV.holm_reject([p_single] + [0.5] * 31) == [False] * 32
    # an extreme one is rejected
    p_ext = AV.t_two_sided(8.0, 15)
    assert AV.holm_reject([p_ext] + [0.5] * 31)[0] is True
