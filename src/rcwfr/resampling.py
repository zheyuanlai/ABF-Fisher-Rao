"""Exact-K resampling, controls, and ancestry accounting (batched, on-device).

Ported from ABP-Fisher-Rao src/abpfr/resampling.py (see docs/PROVENANCE.md).
"""
from __future__ import annotations

import torch

from .grid import EPS


def systematic_resample(w, gen):
    """Systematic (low-variance) resampling.  w:(R,N) normalized -> sel:(R,N) long.

    E[#children of k] = N w_k exactly; equal weights give the identity permutation,
    so a theta = 0 event is an exact no-op.
    """
    R, N = w.shape
    cdf = torch.cumsum(w, dim=1)
    cdf = cdf / torch.clamp(cdf[:, -1:], min=EPS)
    u0 = torch.rand((R, 1), device=w.device, dtype=w.dtype, generator=gen)
    pts = (torch.arange(N, device=w.device, dtype=w.dtype).unsqueeze(0) + u0) / N
    sel = torch.searchsorted(cdf, pts)
    return torch.clamp(sel, max=N - 1)


def turnover_counts(sel, N):
    """Number of parents that left no descendant (kills == clones).  -> (R,) long."""
    R = sel.shape[0]
    hit = torch.zeros((R, N), device=sel.device, dtype=sel.dtype)
    hit.scatter_(1, sel, torch.ones_like(sel))
    return (N - hit.sum(dim=1)).long()


def matched_turnover_indices(m, N, gen, device, dtype):
    """Sham control: kill m uniformly random walkers, clone m random survivors.

    Matches an FR event's timing and intensity while destroying its direction.
    """
    R = m.shape[0]
    rk = torch.rand((R, N), device=device, dtype=dtype, generator=gen)
    order = rk.argsort(dim=1)
    rank = order.argsort(dim=1)
    die = rank < m.unsqueeze(1)
    n_surv = (N - m).clamp(min=1).unsqueeze(1)
    j = (torch.rand((R, N), device=device, dtype=dtype, generator=gen)
         * n_surv.to(dtype)).long().clamp(max=N - 1)
    parent = torch.gather(order, 1, torch.clamp(m.unsqueeze(1) + j, max=N - 1))
    ar = torch.arange(N, device=device).unsqueeze(0).expand(R, N)
    return torch.where(die, parent, ar)


def ancestor_stats(anc, N):
    """Windowed ancestor ESS and largest lineage share.  anc:(R,N) long -> (R,),(R,)."""
    R = anc.shape[0]
    counts = torch.zeros((R, N), device=anc.device, dtype=torch.float64)
    counts.scatter_add_(1, anc, torch.ones_like(anc, dtype=torch.float64))
    ess = counts.sum(dim=1) ** 2 / torch.clamp((counts * counts).sum(dim=1), min=EPS)
    wmax = counts.max(dim=1).values / float(N)
    return ess, wmax


def surviving_ancestors(anc, N):
    R = anc.shape[0]
    hit = torch.zeros((R, N), device=anc.device, dtype=torch.float64)
    hit.scatter_(1, anc, torch.ones_like(anc, dtype=torch.float64))
    return hit.sum(dim=1)
