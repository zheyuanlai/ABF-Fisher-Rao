"""Engineering validation for the nonlinear-reaction-coordinate machinery.

These are the checks that the LINEAR-xi part of the package can never exercise:
G varies along the fiber, so the Fixman factor, the divergence term of the local
mean force, and the three distinct lifts are all nontrivial.
"""
import math
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest
import torch

from rcwfr.grid import DEVICE, DTYPE
from rcwfr.manifold import GraphCV, constrained_step
from rcwfr.systems.graph import (GraphSystem, build_graph, build_mfib,
                                 lag_coefficients, log_nu_exact)

def gen(seed=0):
    g = torch.Generator(device=DEVICE); g.manual_seed(seed); return g

@pytest.fixture(scope="module")
def sysG():
    return build_mfib(omega=1.0, a=0.6, k=1.4)

# ------------------------------------------------------------- geometry ----
def test_a_zero_reduces_to_the_linear_coordinate():
    """a = 0 must reproduce the frozen (xi = x) systems bit for bit."""
    from rcwfr.registry import build
    lin = build("EB")
    gr = build_graph("EB", a=0.0, k=1.4)
    assert torch.allclose(lin.F_ref, gr.F_ref, atol=1e-10)
    assert torch.allclose(lin.dF_ref, gr.dF_ref, atol=1e-10)
    assert gr.fixman_gap == 0.0

def test_gram_matrix_varies_along_the_fiber(sysG):
    G = sysG.cv.G(sysG._yq)
    # the y grid need not contain an exact zero of cos(k y), hence the loose bound
    assert float(G.min()) == pytest.approx(1.0, abs=1e-4)
    assert float(G.max()) > 1.5          # a k = 0.84 -> G_max = 1 + 0.84^2
    assert float(G.max() - G.min()) > 0.5

def test_coarea_jacobian_is_one(sysG):
    """(det G)^{-1/2} dsigma = dy for a graph, so the co-area factor cancels."""
    sqG = torch.sqrt(sysG.cv.G(sysG._yq))
    got = float(torch.trapezoid(sqG * sqG ** -1.0, x=sysG._yq))
    assert got == pytest.approx(float(sysG._yq[-1] - sysG._yq[0]), rel=1e-12)

def test_shake_projection_and_every_lift_land_on_the_constraint(sysG):
    cv = sysG.cv
    z = torch.full((4096,), 0.3, device=DEVICE, dtype=DTYPE)
    y = sysG.sample_fiber(z, gen(3))
    q = torch.stack([z - cv.s(y), y], -1)
    assert float((cv.xi(q) - z).abs().max()) < 1e-13
    for mode in ("cartesian", "minnorm"):
        qp = cv.project(cv.lift(q, torch.full_like(z, 0.05), mode), z + 0.05)
        assert float((cv.xi(qp) - (z + 0.05)).abs().max()) < 1e-12
    for mode in ("cartesian", "minnorm", "adiabatic"):
        yn = sysG.lift_fiber(z, y, 0.05, mode, n_sub=8)
        qn = torch.stack([z + 0.05 - cv.s(yn), yn], -1)
        assert float((cv.xi(qn) - (z + 0.05)).abs().max()) < 1e-12

# ----------------------------------------------------------- mean force ----
def test_local_mean_force_averages_to_dF(sysG):
    """LRS eq. 3.32 including the divergence term, against quadrature-exact F'."""
    M = 400_000
    g = gen(5)
    for z0 in (-0.9, -0.3, 0.6):
        z = torch.full((M,), z0, device=DEVICE, dtype=DTYPE)
        y = sysG.sample_fiber(z, g)
        q = torch.stack([z - sysG.cv.s(y), y], -1)
        f = sysG.cv.mean_force(q, sysG.grad_V_ambient(q), sysG.p.beta)
        i0, fz = sysG._z_index(torch.tensor([z0], device=DEVICE, dtype=DTYPE))
        ref = float(sysG.dF_ref[0, i0] + fz * (sysG.dF_ref[0, i0 + 1] - sysG.dF_ref[0, i0]))
        se = float(f.std()) / math.sqrt(M)
        assert abs(float(f.mean()) - ref) < 5.0 * se

def test_dropping_the_divergence_term_biases_the_mean_force(sysG):
    """The term that is identically zero when xi is linear is not optional here."""
    M = 400_000
    z = torch.full((M,), -0.3, device=DEVICE, dtype=DTYPE)
    y = sysG.sample_fiber(z, gen(6))
    q = torch.stack([z - sysG.cv.s(y), y], -1)
    gV = sysG.grad_V_ambient(q)
    full = sysG.cv.mean_force(q, gV, sysG.p.beta)
    nodiv = (gV[..., 0] + sysG.cv.c(y) * gV[..., 1]) / sysG.cv.G(y)
    assert abs(float(full.mean()) - float(nodiv.mean())) > 20.0 * float(full.std()) / math.sqrt(M)

def test_graph_frame_force_matches_the_lrs_force_in_mean(sysG):
    """Two different estimators, same conditional mean, very different variance."""
    M = 400_000
    z = torch.full((M,), -0.9, device=DEVICE, dtype=DTYPE)
    y = sysG.sample_fiber(z, gen(7))
    q = torch.stack([z - sysG.cv.s(y), y], -1)
    a = sysG.cv.mean_force(q, sysG.grad_V_ambient(q), sysG.p.beta)
    b = sysG.mean_force_z(z, y)
    se = math.sqrt(float(a.var()) + float(b.var())) / math.sqrt(M)
    assert abs(float(a.mean()) - float(b.mean())) < 5.0 * se

# --------------------------------------------------------- conditionals ----
def test_conditional_normalizes(sysG):
    ln = log_nu_exact(sysG, 0.3, sysG._yq)
    assert float(torch.trapezoid(torch.exp(ln), x=sysG._yq)) == pytest.approx(1.0, abs=1e-9)

def test_pit_of_exact_samples_is_uniform(sysG):
    z = (torch.rand((200_000,), device=DEVICE, dtype=DTYPE, generator=gen(8))
         * 2.4 - 1.2)
    u = sysG.pit(z, sysG.sample_fiber(z, gen(9)))
    h = torch.histc(u, bins=64, min=0.0, max=1.0)
    p = h / h.sum()
    kl = float((p * torch.log(torch.clamp(p * 64, min=1e-30))).sum()) - 63 / (2 * 200_000)
    assert abs(kl) < 5e-4

def test_fixman_free_energy_gap_is_nonzero_and_grows_with_nonlinearity():
    gaps = [build_graph("EB", a=a, k=1.4).fixman_gap for a in (0.0, 0.3, 0.6, 1.0)]
    assert gaps[0] == 0.0
    assert all(gaps[i] < gaps[i + 1] for i in range(len(gaps) - 1))

# ---------------------------------------------------------------- lifts ----
def test_adiabatic_lift_preserves_the_conditional_exactly(sysG):
    z = torch.full((300_000,), -0.9, device=DEVICE, dtype=DTYPE)
    y = sysG.sample_fiber(z, gen(10))
    yn = sysG.lift_cdf(z, y, z + 0.2)
    u = sysG.pit(z + 0.2, yn)
    h = torch.histc(u, bins=64, min=0.0, max=1.0)
    p = h / h.sum()
    kl = float((p * torch.log(torch.clamp(p * 64, min=1e-30))).sum()) - 63 / (2 * 300_000)
    assert abs(kl) < 5e-4

def test_naive_lifts_do_not_preserve_the_conditional(sysG):
    z = torch.full((300_000,), -0.9, device=DEVICE, dtype=DTYPE)
    y = sysG.sample_fiber(z, gen(11))
    for mode, lo in (("cartesian", 0.5), ("minnorm", 0.15)):
        yn = sysG.lift_fiber(z, y, 0.1, mode, n_sub=32)
        u = sysG.pit(z + 0.1, yn)
        h = torch.histc(u, bins=64, min=0.0, max=1.0)
        p = h / h.sum()
        kl = float((p * torch.log(torch.clamp(p * 64, min=1e-30))).sum())
        assert kl > lo          # both are far above the ~1e-4 histogram floor

def test_lag_coefficient_vanishes_only_for_the_adiabatic_lift(sysG):
    for z0 in (-0.9, -0.3, 0.6):
        c = {m: lag_coefficients(sysG, z0, m) for m in
             ("cartesian", "minnorm", "adiabatic")}
        assert c["adiabatic"]["C"] < 1e-6
        assert c["cartesian"]["C"] > 1.0
        assert c["minnorm"]["C"] > 1.0

def test_tau_eff_is_smaller_than_the_spectral_gap_time():
    """The relevant timescale is not 1/omega^2; using it over-estimates the lag."""
    for om in (1.0, 1.4, 2.0):
        s = build_mfib(omega=om, a=0.6, k=0.7)
        tau = lag_coefficients(s, -0.9, "cartesian")["tau_eff"]
        assert 0.0 < tau < 1.0 / om ** 2

# ------------------------------------------------- constrained dynamics ----
def test_constrained_step_stays_on_the_manifold(sysG):
    cv = sysG.cv
    z = torch.full((20_000,), 0.2, device=DEVICE, dtype=DTYPE)
    y = sysG.sample_fiber(z, gen(12))
    q = torch.stack([z - cv.s(y), y], -1)
    g = gen(13)
    for _ in range(50):
        q = constrained_step(cv, q, z, sysG.grad_V_ambient(q), 1e-4, sysG.p.beta, g)
        assert float((cv.xi(q) - z).abs().max()) < 1e-11

def test_constrained_sampler_reaches_the_right_conditional_with_fixman(sysG):
    cv, beta = sysG.cv, sysG.p.beta
    z = torch.full((100_000,), -0.6, device=DEVICE, dtype=DTYPE)
    g = gen(14)
    y = sysG.sample_fiber(z, g)
    q = torch.stack([z - cv.s(y), y], -1)
    hist = torch.zeros(64, device=DEVICE, dtype=DTYPE)
    for t in range(4000):
        q = constrained_step(cv, q, z, sysG.grad_V_ambient(q), 2e-4, beta, g, fixman=True)
        q[..., 1] = torch.clamp(q[..., 1], -sysG.p.y_max, sysG.p.y_max)
        q = cv.project(q, z)
        if t >= 2000 and t % 50 == 0:
            hist += torch.histc(sysG.pit(z, q[..., 1]), bins=64, min=0.0, max=1.0)
    p = hist / hist.sum()
    assert float((p * torch.log(torch.clamp(p * 64, min=1e-30))).sum()) < 5e-3
