"""Restraint machinery for the Stage-2 conditional chi1 profiles."""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from alkanes.cv import _grad_phi4                                      # noqa: E402
from valine.system import (                                            # noqa: E402
    CHI1_ATOMS, N_ATOMS, PHI_ATOMS, PSI_ATOMS, angles_np, make_seed, make_system,
)
from valine.umbrella import (                                          # noqa: E402
    DihedralRestraint, count_states, dihedral_grad_analytic, mbar_1d_periodic, wrap_to_pi,
)

DT = torch.float64
QUADS = [PHI_ATOMS, PSI_ATOMS, CHI1_ATOMS]


@pytest.fixture(scope="module")
def conf():
    _, _, s = make_system()
    X0, _ = make_seed((-80.0, 80.0, 180.0), system=s)
    rng = np.random.default_rng(4)
    Xb = np.stack([X0 + 0.05 * rng.standard_normal(X0.shape) for _ in range(64)])
    return s, Xb, torch.tensor(Xb, dtype=DT)


def test_analytic_dihedral_gradient_matches_autodiff(conf):
    """The ONLY valid check on `dihedral_grad_analytic`.

    Translation invariance (``sum_a g_a == 0``) is satisfied identically by several *wrong*
    sign choices for the two middle atoms -- two such choices were tried during development
    and both passed it while being O(1) wrong.  Elementwise agreement with the validated
    `torch.func` primitive is what actually pins the formula.
    """
    _, _, q = conf
    for idx in QUADS:
        ga = dihedral_grad_analytic(q, idx)
        gr = _grad_phi4(q[:, idx, :].reshape(-1, 12)).reshape(-1, 4, 3)
        assert float((ga - gr).abs().max()) < 1e-10, idx


def test_analytic_dihedral_gradient_is_translation_invariant(conf):
    _, _, q = conf
    for idx in QUADS:
        assert float(dihedral_grad_analytic(q, idx).sum(dim=1).abs().max()) < 1e-10


def test_restraint_force_matches_finite_difference(conf):
    _, Xb, q = conf
    B = q.shape[0]
    cen = torch.tensor(np.radians(np.tile([-80.0, 80.0, 180.0], (B, 1))), dtype=DT)
    kap = torch.tensor(np.tile([100.0, 100.0, 50.0], (B, 1)), dtype=DT)
    R = DihedralRestraint(QUADS, cen, kap, N_ATOMS, dtype=DT)
    _, F, _ = R.energy_and_force(q)
    h = 1e-6
    worst = 0.0
    for a in (4, 6, 8, 10, 12, 20, 22):            # the atoms the three dihedrals touch
        for c in range(3):
            qp, qm = q.clone(), q.clone()
            qp[:, a, c] += h
            qm[:, a, c] -= h
            fd = -(R.energy_and_force(qp)[0] - R.energy_and_force(qm)[0]) / (2 * h)
            worst = max(worst, float((fd - F[:, a, c]).abs().max()))
    assert worst / float(F.abs().max()) < 1e-8


def test_restraint_angles_match_numpy_reference(conf):
    _, Xb, q = conf
    B = q.shape[0]
    cen = torch.zeros(B, 3, dtype=DT)
    R = DihedralRestraint(QUADS, cen, torch.zeros(B, 3, dtype=DT), N_ATOMS, dtype=DT)
    got = R.dihedrals(q).numpy()
    assert np.abs(wrap_to_pi(got - angles_np(Xb))).max() < 1e-12


def test_zero_kappa_gives_zero_force(conf):
    _, _, q = conf
    B = q.shape[0]
    R = DihedralRestraint(QUADS, torch.zeros(B, 3, dtype=DT), torch.zeros(B, 3, dtype=DT),
                          N_ATOMS, dtype=DT)
    U, F, _ = R.energy_and_force(q)
    assert float(U.abs().max()) == 0.0
    assert float(F.abs().max()) == 0.0


def test_restraint_pulls_towards_its_centre(conf):
    """A displaced walker must feel a force that reduces the restrained angle deviation."""
    _, _, q = conf
    B = q.shape[0]
    cen = torch.tensor(np.radians(np.tile([-80.0, 80.0, 60.0], (B, 1))), dtype=DT)
    kap = torch.tensor(np.tile([0.0, 0.0, 200.0], (B, 1)), dtype=DT)
    R = DihedralRestraint(QUADS, cen, kap, N_ATOMS, dtype=DT)
    _, F, th0 = R.energy_and_force(q)
    d0 = wrap_to_pi(th0[:, 2] - cen[:, 2]).abs()
    th1 = R.dihedrals(q + 1e-4 * F / F.abs().max())[:, 2]
    d1 = wrap_to_pi(th1 - cen[:, 2]).abs()
    assert (d1 <= d0 + 1e-12).all()


def test_mbar_recovers_a_known_periodic_profile():
    """Sample a known F(x) with umbrellas and check MBAR recovers it."""
    rng = np.random.default_rng(0)
    beta, kappa = 1.0 / 2.4943, 100.0
    # a 3-well periodic potential, in kJ/mol
    def Ftrue(x):
        return 6.0 * (1.0 - np.cos(3.0 * x)) + 1.5 * np.sin(x)

    centers = np.radians(np.arange(-180.0, 180.0, 15.0))
    S = []
    for c in centers:                                   # Metropolis in each window
        x = c
        out = []
        for i in range(60_000):
            xp = x + 0.25 * rng.standard_normal()
            dU = (Ftrue(xp) + 0.5 * kappa * wrap_to_pi(xp - c) ** 2
                  - Ftrue(x) - 0.5 * kappa * wrap_to_pi(x - c) ** 2)
            if dU <= 0 or rng.random() < np.exp(-beta * dU):
                x = xp
            if i >= 10_000 and i % 10 == 0:
                out.append(wrap_to_pi(x))
        S.append(out)
    n = min(len(s) for s in S)
    grid, F = mbar_1d_periodic(np.array([s[:n] for s in S]), centers, kappa, beta)
    ref = Ftrue(grid)
    ref -= ref.min()
    ok = np.isfinite(F)
    err = np.abs(F[ok] - ref[ok])
    assert err.max() < 1.5, f"max deviation {err.max():.2f} kJ/mol from the known profile"


# ----------------------------------------------------------------- state counting (sec.32)
def _g(n=72):
    return np.linspace(-np.pi, np.pi, n, endpoint=False)


@pytest.mark.parametrize("name,build,expected", [
    ("one well, no gap", lambda g: 6.0 * (1 - np.cos(g)), 1),
    ("well straddling +/-pi", lambda g: 6.0 * (1 + np.cos(g)), 1),
    ("flat", lambda g: np.zeros_like(g), 1),
    ("two wells, tall sampled ridge",
     lambda g: np.minimum(8.0 * 2.4943 * (1 - np.cos(2 * g)) / 2.0, 40.0), 2),
    ("three wells", lambda g: 10.0 * 2.4943 * (1 - np.cos(3 * g)) / 2.0, 3),
])
def test_count_states_without_gaps(name, build, expected):
    kT = 0.008314462618 * 300.0
    F = np.asarray(build(_g()), dtype=float)
    assert count_states(F, 1.0 / kT, kT)["n_states"] == expected, name


def test_count_states_gap_on_a_barrier_is_still_one_state():
    """An inaccessible arc sitting on a barrier must not manufacture a second state."""
    kT = 0.008314462618 * 300.0
    F = (6.0 * (1 - np.cos(_g()))).copy()
    F[:6] = np.nan
    F[-6:] = np.nan                       # the barrier region, straddling +/-pi
    r = count_states(F, 1.0 / kT, kT)
    assert r["n_states"] == 1 and r["has_gap"]


def test_count_states_two_arcs_split_by_inaccessible_regions():
    """The case the sec.32 gate exists for.

    Two populated arcs separated by *inaccessible* windows, with a small sampled max(F).
    Reporting max(F) would call this well-mixed; counting states correctly calls it two.
    """
    kT = 0.008314462618 * 300.0
    F = np.full(72, 5.0)
    F[5:30] = 0.2
    F[40:65] = 0.4
    F[:5] = np.nan
    F[30:40] = np.nan
    F[65:] = np.nan
    r = count_states(F, 1.0 / kT, kT)
    assert r["n_states"] == 2
    assert np.nanmax(F) / kT < 3.0        # a max(F)-based gate would have wrongly passed
