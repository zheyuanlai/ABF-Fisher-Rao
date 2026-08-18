"""Exact-K resampling, matched-turnover sham, and ancestry accounting (batched).

All functions operate on (R, N) rows entirely on-device; the only randomness is drawn
from a caller-supplied torch.Generator living on the same device, so runs are
reproducible per (device, seed) and no host<->device traffic happens inside the loop.
"""
from __future__ import annotations

import torch

from .grid import EPS


def systematic_resample(w, gen):
    """Systematic (low-variance) resampling.  w: (R, N) normalized -> sel: (R, N) long.

    Slot j selects the parent whose CDF cell contains (j + U)/N, one U per row.
    E[#children of k] = N * w_k exactly; counts are floor/ceil of N*w_k.
    Equal weights give sel == arange(N): the identity, so a theta = 0 FR event is a no-op.
    """
    R, N = w.shape
    cdf = torch.cumsum(w, dim=1)
    cdf = cdf / torch.clamp(cdf[:, -1:], min=EPS)
    u0 = torch.rand((R, 1), device=w.device, dtype=w.dtype, generator=gen)
    pts = (torch.arange(N, device=w.device, dtype=w.dtype).unsqueeze(0) + u0) / N
    sel = torch.searchsorted(cdf, pts)
    return torch.clamp(sel, max=N - 1)


def turnover_counts(sel, N):
    """Number of distinct parents that left no descendant: kills == clones.  -> (R,) long."""
    R = sel.shape[0]
    hit = torch.zeros((R, N), device=sel.device, dtype=sel.dtype)
    hit.scatter_(1, sel, torch.ones_like(sel))
    return (N - hit.sum(dim=1)).long()


def matched_turnover_indices(m, N, gen, device, dtype):
    """Sham control: kill m uniformly random walkers, clone m uniformly random survivors.

    m: (R,) long, per-row turnover copied from the partner FR arm at the SAME event.
    Returns sel: (R, N) long with new = old[sel].  Matches the partner's timing and
    intensity by construction while destroying the Fisher-Rao direction.
    """
    R = m.shape[0]
    rk = torch.rand((R, N), device=device, dtype=dtype, generator=gen)
    order = rk.argsort(dim=1)                      # order[:, i] = walker with rank i
    rank = order.argsort(dim=1)                    # rank of each walker
    die = rank < m.unsqueeze(1)                    # the m lowest ranks die
    n_surv = (N - m).clamp(min=1).unsqueeze(1)
    j = (torch.rand((R, N), device=device, dtype=dtype, generator=gen)
         * n_surv.to(dtype)).long().clamp(max=N - 1)
    parent = torch.gather(order, 1, torch.clamp(m.unsqueeze(1) + j, max=N - 1))
    ar = torch.arange(N, device=device).unsqueeze(0).expand(R, N)
    return torch.where(die, parent, ar)


def surviving_ancestors(anc, N):
    """Number of ORIGINAL ancestors that still have living descendants.  -> (R,).

    Complements the windowed ESS: windowed ancestry recovering to K only means no
    recent resampling; lineages lost from the global genealogy never come back.
    """
    R = anc.shape[0]
    hit = torch.zeros((R, N), device=anc.device, dtype=torch.float64)
    hit.scatter_(1, anc, torch.ones_like(anc, dtype=torch.float64))
    return hit.sum(dim=1)


def ancestor_stats(anc, N):
    """Windowed ancestor ESS and largest lineage share w_max.  anc: (R, N) long.

    w_max is the acceptance gate ESS alone misses: a population can keep a respectable
    ESS while one lineage quietly owns a fifth of it.  (Ported from the validated
    ABF-Fisher-Rao gateway engine; see docs/PROVENANCE.md.)
    """
    R = anc.shape[0]
    counts = torch.zeros((R, N), device=anc.device, dtype=torch.float64)
    counts.scatter_add_(1, anc, torch.ones_like(anc, dtype=torch.float64))
    ess = counts.sum(dim=1) ** 2 / torch.clamp((counts * counts).sum(dim=1), min=EPS)
    wmax = counts.max(dim=1).values / float(N)
    return ess, wmax
