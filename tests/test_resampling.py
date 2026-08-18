import torch

from conftest import DTYPE
from abpfr.resampling import (ancestor_stats, matched_turnover_indices,
                              systematic_resample, turnover_counts)


def test_systematic_counts_are_floor_or_ceil_of_expectation():
    gen = torch.Generator().manual_seed(0)
    N = 64
    w = torch.rand((1, N), generator=gen, dtype=DTYPE)
    w = w / w.sum()
    sel = systematic_resample(w, gen)
    counts = torch.zeros(N, dtype=DTYPE)
    counts.scatter_add_(0, sel[0], torch.ones(N, dtype=DTYPE))
    expect = N * w[0]
    assert bool((counts >= torch.floor(expect) - 1e-9).all())
    assert bool((counts <= torch.ceil(expect) + 1e-9).all())


def test_systematic_resampling_unbiased():
    gen = torch.Generator().manual_seed(1)
    N, trials = 32, 4000
    w = torch.rand((1, N), generator=gen, dtype=DTYPE)
    w = w / w.sum()
    total = torch.zeros(N, dtype=DTYPE)
    for _ in range(trials):
        sel = systematic_resample(w, gen)
        total.scatter_add_(0, sel[0], torch.ones(N, dtype=DTYPE))
    mean_children = total / trials
    assert torch.allclose(mean_children, N * w[0], atol=0.05)


def test_turnover_counts():
    sel = torch.tensor([[0, 0, 2, 3], [0, 1, 2, 3], [1, 1, 1, 1]])
    t = turnover_counts(sel, 4)
    assert t.tolist() == [1, 0, 3]


def test_matched_turnover_replaces_exactly_m():
    gen = torch.Generator().manual_seed(2)
    N = 256
    m = torch.tensor([0, 17, 100])
    sel = matched_turnover_indices(m, N, gen, torch.device("cpu"), DTYPE)
    ar = torch.arange(N)
    for row in range(3):
        changed = (sel[row] != ar)
        assert int(changed.sum()) == int(m[row])
        # every replacement parent must itself be a survivor
        dead = set(ar[changed].tolist())
        parents = set(sel[row][changed].tolist())
        assert not (parents & dead)


def test_ancestor_stats():
    anc = torch.tensor([[0, 0, 0, 3], [0, 1, 2, 3]])
    ess, wmax = ancestor_stats(anc, 4)
    # row 0: counts (3,1) -> ESS = 16/10 = 1.6, wmax = 3/4
    assert abs(float(ess[0]) - 1.6) < 1e-12
    assert abs(float(wmax[0]) - 0.75) < 1e-12
    assert abs(float(ess[1]) - 4.0) < 1e-12
    assert abs(float(wmax[1]) - 0.25) < 1e-12
