"""Gateway validation harness (src/gateway_validation.py) -- docs/equal_budget/GATEWAY_TIMESTEP_VALIDATION.md.

G1a analytic gradient == central differences of V (wells, gateway, strong confinement, |y| up to 5 sd);
G1b the PRODUCTION force (extracted from gateway_numba.simulate) == the analytic gradient;
G1c the reference F, F' of eb_abffr_core == H (x^2-1)^2 + log(omega)/beta and its derivative (up to a constant);
H1  the harness EM chain (bias off) is BITWISE the production simulate (same noise seed, min_count 1e300);
H2  MALA and the exact sampler reproduce the exact Gibbs law (flat-bias target: uniform x marginal, Var(Y|x)).
"""
import os
import sys

import numpy as np
import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import gateway_validation as GV  # noqa: E402

P = dict(beta=16.0, H=0.5, omega_out=1.0, omega_in=32.0, s=0.1)


def points():
    xs = np.array([-1.5, -1.0, -0.6, -0.25, -0.12, -0.1, -0.05, -0.01, 0.0, 0.01, 0.05, 0.1, 0.12, 0.3, 1.0, 1.4])
    out = []
    for x in xs:
        sd = 1.0 / (np.sqrt(P["beta"]) * GV.omega(x, P))
        for k in (0.0, 0.5, -1.0, 2.0, 5.0):
            out.append((x, k * sd))
    return np.array(out)


def test_G1a_gradient_fd():
    worst = 0.0
    for x, y in points():
        gx, gy = GV.grad_V_np(x, y, P)
        for eps in (1e-6,):
            fdx = (GV.V_np(x + eps, y, P) - GV.V_np(x - eps, y, P)) / (2 * eps)
            fdy = (GV.V_np(x, y + eps, P) - GV.V_np(x, y - eps, P)) / (2 * eps)
            worst = max(worst, abs(fdx - gx) / max(1.0, abs(gx)), abs(fdy - gy) / max(1.0, abs(gy)))
    assert worst < 1e-6, worst


def test_G1b_production_force():
    pts = points()
    fx, fy = GV.production_force(pts[:, 0], pts[:, 1], P)
    gx, gy = GV.grad_V_np(pts[:, 0], pts[:, 1], P)
    assert np.max(np.abs(fx - gx) / np.maximum(1.0, np.abs(gx))) < 1e-7
    assert np.max(np.abs(fy - gy) / np.maximum(1.0, np.abs(gy))) < 1e-7


def test_G1c_reference_profiles():
    import torch
    import eb_abffr_core as eb
    xg, dx, mask, idx0 = eb.build_grid(torch.device("cpu"), torch.float64)
    t = lambda v: torch.tensor([[float(v)]], dtype=torch.float64)
    F, Fp = eb.reference_profiles(xg, mask, t(P["beta"]), t(P["H"]), t(P["omega_out"]), t(P["omega_in"]), t(P["s"]))
    x = xg.numpy()
    m = mask.numpy()
    Fa = GV.F_exact(x, P)
    d = F.numpy()[0] - Fa
    assert np.ptp(d[m]) < 1e-10, np.ptp(d[m])                     # equal up to an additive constant
    assert np.max(np.abs(Fp.numpy()[0] - GV.Fp_exact(x, P))) < 1e-10
    xs = np.linspace(-1.5, 1.5, 7)
    fm = eb.reference_mean_force(torch.tensor(xs[None]), t(P["beta"]), t(P["H"]), t(P["omega_out"]), t(P["omega_in"]), t(P["s"])).numpy()[0]
    assert np.max(np.abs(fm - GV.Fp_exact(xs, P))) < 1e-10


def test_H1_harness_em_is_production():
    """Harness EM with bias off vs gateway_numba.simulate with min_count 1e300 (bias exactly 0), same
    numba seed: identical draws (zx then zy per walker in the harness; simulate draws all zx then all zy
    per step), so compare with N = 1 where the two orders coincide."""
    import gateway_numba as gn
    N, n, h = 1, 4000, 1e-4
    x0 = np.array([-1.0]); y0 = np.array([0.01])
    kern, r = gn.gaussian_kernel_np(0.1, gn.grid_dx())
    out = gn.simulate(77, N, n, h, P["beta"], P["H"], P["omega_out"], P["omega_in"], P["s"],
                      np.array([0], np.int64), np.array([180], np.int64), np.array([0], np.int64), 180, 1e300,
                      0.0, 0.0, 1, 3.0, 0, kern, int(r), gn.grid_dx(), x0.copy(), y0.copy(), np.array([n], np.int64), 1,
                      np.zeros((0, N)), False)
    C, Mf, Sy, Sy2, *_rest, X, Y = GV.run_chain(GV.EM, x0.copy(), y0.copy(), n, 0, h, P["beta"], P["H"], P["omega_out"],
                                                 P["omega_in"], P["s"], 0, 180, 77, 10**9, 1)
    assert np.array_equal(X, out[8][0]) and np.array_equal(Y, out[9][0])
    assert np.array_equal(C, out[1][0, 0])


@pytest.mark.parametrize("scheme", ["MALA", "EXACT"])
def test_H2_exact_samplers(scheme):
    """Exact samplers vs the exact law.  Bins are used only where the expected relative standard error
    of the variance is small (>= 4e4 effective samples): wells and gateway."""
    nb = 36
    if scheme == "EXACT":
        C, Mf, Sy, Sy2 = GV.exact_samples(8_000_000, P, nb, 3)
        ref = GV.bin_averages(P, nb)
        pr = ref["P"] / ref["P"].sum()
        frac = C / C.sum()
        ok = pr > 0.01
        assert np.max(np.abs(frac[ok] / pr[ok] - 1)) < 0.02
        var = Sy2 / np.maximum(C, 1) - (Sy / np.maximum(C, 1)) ** 2
        se = np.sqrt(2.0 / np.maximum(C, 1))
        assert np.max(np.abs(var[ok] / ref["var"][ok] - 1) / se[ok]) < 4.5
        return
    # MALA on the flat-bias target: 256 walkers x 200 t.u. at h 1e-4 (exact for any h)
    g = np.random.default_rng(5)
    x0 = g.uniform(-1.5, 1.5, 256)
    y0 = g.standard_normal(256) / (np.sqrt(P["beta"]) * GV.omega(x0, P))
    nb = 72
    C, Mf, Sy, Sy2, nacc, nprop, *_ = GV.run_chain(GV.MALA, x0, y0, 2_000_000, 50_000, 1e-4, P["beta"], P["H"], P["omega_out"],
                                                   P["omega_in"], P["s"], 1, nb, 11, 10**9, 1)
    ref = GV.bin_averages(P, nb, h=1e-4)
    var = Sy2 / C - (Sy / C) ** 2
    gate = np.abs(ref["centres"]) < 0.15               # omega >= 11: y decorrelates in < 0.005 t.u.
    rel = var[gate] / ref["var_flat"][gate] - 1
    # exact sampler of the FLAT law: no EM inflation (at h 1e-4 EM would give +5.1 % at the centre bin)
    assert np.max(np.abs(rel)) < 0.02, np.round(rel, 4)
    assert nacc / nprop > 0.8
