"""Stage-0 SHUS validation: sign convention, gauge invariance, reweighting
consistency, and the estimator-protection invariant."""
import torch

from conftest import DEVICE, DTYPE
from abpfr.grid import Grid1D, gaussian_kernel, smooth
from abpfr.shus import ShusAccumulator
from abpfr.fisher_rao import theta_backoff, uniform_log_ratio
from abpfr.resampling import systematic_resample
from abpfr.shus import biased_marginal_estimate

G = Grid1D(xmin=-1.8, xmax=1.8, n=181, eval_lo=-1.5, eval_hi=1.5)


def make_shus(rows=1, beta=1.0):
    return ShusAccumulator(rows, G, torch.full((rows, 1), beta, dtype=DTYPE),
                           eps_bw=0.07, device=DEVICE, dtype=DTYPE)


def test_initial_bias_is_flat():
    s = make_shus()
    assert torch.allclose(s.F, torch.zeros_like(s.F))
    assert torch.allclose(s.Fp, torch.zeros_like(s.Fp))


def test_sign_convention_visited_region_loses_free_energy():
    # walkers sit only in the left basin -> F must drop there relative to the right
    s = make_shus()
    gen = torch.Generator().manual_seed(0)
    X = -1.0 + 0.05 * torch.randn((1, 4000), generator=gen, dtype=DTYPE)
    for _ in range(5):
        s.deposit(X)
    s.update(dt=1e-3, K=4000)
    xg = G.x(DEVICE, DTYPE)
    i_left = int(torch.argmin((xg + 1.0).abs()))
    i_right = int(torch.argmin((xg - 1.0).abs()))
    assert float(s.F[0, i_left]) < float(s.F[0, i_right])
    # and the bias force at x slightly right of the dip pushes AWAY from it
    xprobe = torch.tensor([[-0.8]], dtype=DTYPE)
    assert float(s.bias_force_at(xprobe)) > 0.0


def test_gauge_invariance_of_forces():
    # scaling R by any constant shifts F by a constant and leaves forces unchanged
    s1, s2 = make_shus(), make_shus()
    s2.R = s2.R * 37.5
    s2._refresh_bias()
    gen = torch.Generator().manual_seed(1)
    X = 0.5 * torch.randn((1, 2000), generator=gen, dtype=DTYPE).clamp(-1.7, 1.7)
    for s in (s1, s2):
        s.deposit(X)
        s.update(dt=1e-3, K=2000)
    assert torch.allclose(s1.Fp, s2.Fp, atol=1e-10)
    # renormalization makes even R identical after one update
    assert torch.allclose(s1.R, s2.R, atol=1e-12)


def test_reweighting_consistency_increment_proportional_to_gibbs():
    """Samples from the biased equilibrium p ~ exp(-beta(F_true - F_n)) deposited with
    weight exp(-beta F_n(xi)) must increment R proportionally to exp(-beta F_true):
    the bias cancels exactly.  This is the SHUS fixed-point property."""
    torch.manual_seed(3)
    beta = 1.0
    xg = G.x(DEVICE, DTYPE)
    F_true = (xg ** 2).unsqueeze(0)                     # a smooth "free energy"
    s = make_shus(beta=beta)
    # give the accumulator a NON-trivial current state (some earlier history)
    s.R = torch.exp(-0.7 * (xg - 0.4) ** 2).unsqueeze(0)
    s._refresh_bias()
    # draw many samples from p ~ exp(-beta(F_true - F_n)) by inverse CDF on the grid
    F_n = s.F
    dens = torch.exp(-beta * (F_true - F_n))[0]
    dens = dens / dens.sum()
    cdf = torch.cumsum(dens, 0)
    u = torch.rand(400_000, dtype=DTYPE)
    idx = torch.searchsorted(cdf, u).clamp(max=G.n - 1)
    X = (xg[idx] + (torch.rand(400_000, dtype=DTYPE) - 0.5) * G.dx).unsqueeze(0)
    s.deposit(X)
    raw_inc = smooth(s.buf, s.kernel, s.krad, G.dx) * (1e-3 / X.shape[1])
    target = torch.exp(-beta * F_true)[0]
    m = G.eval_mask(DEVICE, DTYPE)
    ratio = (raw_inc[0] / target)[m]
    rel_spread = float((ratio.max() - ratio.min()) / ratio.mean())
    assert rel_spread < 0.08, f"increment shape deviates from exp(-beta F): {rel_spread:.3f}"


def test_estimator_protection_fr_event_cannot_touch_accumulator():
    """A full FR event (score -> weights -> resample -> gather) must leave the SHUS
    accumulator and its deposit buffer bit-identical."""
    s = make_shus()
    gen = torch.Generator().manual_seed(4)
    X = -1.0 + 0.3 * torch.randn((1, 1024), generator=gen, dtype=DTYPE)
    s.deposit(X)
    s.update(dt=1e-3, K=1024)
    R_snap, buf_snap = s.R.clone(), s.buf.clone()
    p_hat = biased_marginal_estimate(X, 0.10, G)
    logr = uniform_log_ratio(X, p_hat, G)
    w, theta, essf = theta_backoff(logr, torch.tensor([0.2], dtype=DTYPE),
                                   torch.tensor([0.5], dtype=DTYPE))
    sel = systematic_resample(w, gen)
    X = torch.gather(X, 1, sel)
    assert torch.equal(s.R, R_snap)
    assert torch.equal(s.buf, buf_snap)


def test_deposition_diagnostic_separates_healthy_from_feedback():
    """The mechanism behind the smoke result, at accumulator level:
    - samples from the biased equilibrium deposit d_n ~ exp(-beta F_true)  (healthy);
    - samples from an FR-flattened (uniform) population deposit d_n ~ R_n  (feedback).
    """
    from abpfr.grid import trapz
    beta = 1.0
    xg = G.x(DEVICE, DTYPE)
    F_true = (xg ** 2).unsqueeze(0)
    rho_ref = torch.exp(-beta * F_true)
    rho_ref = rho_ref / trapz(rho_ref, G.dx).unsqueeze(1)

    def normalized_increment(sample_density):
        s = make_shus(beta=beta)
        s.R = torch.exp(-1.5 * (xg - 0.6) ** 2).unsqueeze(0)   # nontrivial current R_n
        s._refresh_bias()
        r_n = s.R / trapz(s.R, G.dx).unsqueeze(1)
        cdf = torch.cumsum(sample_density(s) / sample_density(s).sum(), 0)
        u = torch.rand(300_000, dtype=DTYPE, generator=torch.Generator().manual_seed(9))
        idx = torch.searchsorted(cdf, u).clamp(max=G.n - 1)
        X = xg[idx].unsqueeze(0)
        s.deposit(X)
        inc = smooth(s.buf, s.kernel, s.krad, G.dx)
        d_n = inc / trapz(inc, G.dx).unsqueeze(1)
        m = G.eval_mask(DEVICE, DTYPE)
        to_ref = float(((d_n - rho_ref)[:, m] ** 2).mean().sqrt())
        to_self = float(((d_n - r_n)[:, m] ** 2).mean().sqrt())
        return to_ref, to_self

    # healthy: biased equilibrium p ~ exp(-beta(F_true - F_n))
    ref_h, self_h = normalized_increment(
        lambda s: torch.exp(-beta * (F_true - s.F))[0])
    # feedback: FR has flattened the marginal to uniform
    ref_f, self_f = normalized_increment(
        lambda s: torch.ones_like(xg))
    assert ref_h < 0.05 and ref_h < self_h, (ref_h, self_h)
    assert self_f < 0.05 and self_f < ref_f, (ref_f, self_f)


def make_gain_shus(g, rows=1, beta=1.0):
    return ShusAccumulator(rows, G, torch.full((rows, 1), beta, dtype=DTYPE),
                           eps_bw=0.07, device=DEVICE, dtype=DTYPE,
                           gain=torch.full((rows,), g, dtype=DTYPE))


def test_gain_one_is_bitwise_identical_to_frozen_baseline():
    s0, s1 = make_shus(), make_gain_shus(1.0)
    gen = torch.Generator().manual_seed(6)
    X = 0.4 * torch.randn((1, 2000), generator=gen, dtype=DTYPE).clamp(-1.7, 1.7)
    for s in (s0, s1):
        s.deposit(X)
        s.update(dt=1e-3, K=2000)
    assert torch.equal(s0.R, s1.R)
    assert torch.equal(s0.F, s1.F)


def test_gain_scales_increment_linearly_and_preserves_shape():
    # g multiplies the raw increment exactly; the normalized deposit shape (the
    # object of the reweighting-consistency fixed point) is unchanged, so the
    # analytic fixed point R* = K_eps e^{-beta F} is gain-independent
    s1, sg = make_gain_shus(1.0), make_gain_shus(0.5)
    gen = torch.Generator().manual_seed(7)
    X = 0.4 * torch.randn((1, 2000), generator=gen, dtype=DTYPE).clamp(-1.7, 1.7)
    s1.deposit(X)
    sg.deposit(X)
    inc1 = s1.update(dt=1e-3, K=2000)
    incg = sg.update(dt=1e-3, K=2000)
    assert torch.allclose(incg, 0.5 * inc1, atol=1e-15)
    n1 = inc1 / inc1.sum()
    ng = incg / incg.sum()
    assert torch.allclose(n1, ng, atol=1e-15)


def test_gain_gauge_invariance_of_forces():
    # the gauge property survives g != 1: scaling R by a constant leaves forces
    # (and, after renormalization, R itself) unchanged
    s1, s2 = make_gain_shus(0.5), make_gain_shus(0.5)
    s2.R = s2.R * 37.5
    s2._refresh_bias()
    gen = torch.Generator().manual_seed(8)
    X = 0.5 * torch.randn((1, 2000), generator=gen, dtype=DTYPE).clamp(-1.7, 1.7)
    for s in (s1, s2):
        s.deposit(X)
        s.update(dt=1e-3, K=2000)
    assert torch.allclose(s1.Fp, s2.Fp, atol=1e-10)
    assert torch.allclose(s1.R, s2.R, atol=1e-12)


def test_gain_keeps_sign_convention():
    s = make_gain_shus(0.25)
    gen = torch.Generator().manual_seed(10)
    X = -1.0 + 0.05 * torch.randn((1, 4000), generator=gen, dtype=DTYPE)
    for _ in range(5):
        s.deposit(X)
    s.update(dt=1e-3, K=4000)
    xg = G.x(DEVICE, DTYPE)
    i_left = int(torch.argmin((xg + 1.0).abs()))
    i_right = int(torch.argmin((xg - 1.0).abs()))
    assert float(s.F[0, i_left]) < float(s.F[0, i_right])


def test_gain_must_be_positive():
    import pytest
    with pytest.raises(AssertionError):
        make_gain_shus(0.0)


def test_deposit_weight_is_block_frozen():
    # deposits within one block all see the same R, even after earlier deposits
    s = make_shus()
    X1 = torch.full((1, 100), -1.0, dtype=DTYPE)
    s.deposit(X1)
    buf_after_one = s.buf.clone()
    s.deposit(X1)
    assert torch.allclose(s.buf, 2.0 * buf_after_one, atol=1e-12)
