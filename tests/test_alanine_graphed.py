"""CUDA-graph replay of the alanine hot loop must be BITWISE identical to eager.

The wrappers in ``alanine.graphed`` exist purely to remove launch overhead; if they ever
changed a single bit of the local mean force or the physical force, the arms run with them
would no longer be the frozen engine.  Skipped without CUDA.
"""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(scope="module")
def rig():
    from alanine.cv2d import BackboneCV2D
    from alanine.forcefield import TorchFF, extract_parameters
    from alanine.system import PHI_ATOMS, PSI_ATOMS, reference_minimum
    system, X0 = reference_minimum()
    tff = TorchFF(extract_parameters(system), device="cuda", dtype=torch.float64)
    cv = BackboneCV2D(PHI_ATOMS, PSI_ATOMS, n_atoms=22)
    return tff, cv, X0


def _configs(X0, B, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.as_tensor(np.repeat(X0[None], B, 0), device="cuda", dtype=torch.float64)
    return x + 0.02 * torch.randn(B, 22, 3, generator=g, device="cuda", dtype=torch.float64)


def test_graphed_forces_bitwise(rig):
    from alanine.graphed import GraphedForces
    tff, _, X0 = rig
    gf = GraphedForces(tff, batch=512)
    for seed in (1, 2, 3):
        x = _configs(X0, 512, seed)
        assert torch.equal(gf(x), tff.forces(x))
    with pytest.raises(ValueError):
        gf(_configs(X0, 256, 4))          # static shape is enforced, never silently padded


def test_graphed_cv_bitwise_and_reentrant(rig):
    from alanine.graphed import GraphedCV
    tff, cv, X0 = rig
    beta = 1.0 / (0.0083144626 * 300.0)
    gcv = GraphedCV(cv, batch=512, beta=beta, n_atoms=22)
    prev = None
    for seed in (1, 2, 3):
        x = _configs(X0, 512, seed)
        f = tff.forces(x)
        e = cv.local_mean_force(x, f, beta)
        r = gcv.local_mean_force(x, f, beta)
        assert torch.equal(e[0], r[0]) and torch.equal(e[1], r[1]) and torch.equal(e[2], r[2])
        assert all(torch.equal(e[3][k], r[3][k]) for k in e[3])
        if prev is not None:              # outputs are clones: a replay must not mutate them
            assert not torch.equal(prev[0], r[0])
            assert torch.equal(prev[0], prev_ref)
        prev, prev_ref = r, e[0].clone()
    with pytest.raises(ValueError):
        gcv.local_mean_force(x, f, beta * 1.01)


def test_inv_ex_matches_inv():
    """The graph-safe inverse the CV now uses is the same arithmetic as torch.linalg.inv."""
    g = torch.Generator(device="cuda").manual_seed(0)
    A = torch.randn(4096, 2, 2, generator=g, device="cuda", dtype=torch.float64)
    G = A @ A.transpose(-1, -2) + 0.1 * torch.eye(2, device="cuda", dtype=torch.float64)
    assert torch.equal(torch.linalg.inv(G), torch.linalg.inv_ex(G).inverse)
