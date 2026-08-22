"""Row layout: R = n_cfg * n_seed rows in ONE batched run.

Row r = c * n_seed + s carries hyper-parameter configuration c and replicate s.
Initial conditions are drawn ONCE per replicate s and tiled across c, so every
configuration - and every ARM run with the same (seed, n_seed, N) - starts from
exactly the same ensemble.  Comparisons are therefore paired by construction.
"""
from __future__ import annotations

import itertools

import torch


def expand_grid(gridspec):
    """{'kappa':[a,b], 'theta':[c,d]} -> [ {kappa:a,theta:c}, ... ] (product order)."""
    if not gridspec:
        return [{}]
    keys = list(gridspec)
    return [dict(zip(keys, v)) for v in itertools.product(*[gridspec[k] for k in keys])]


def row_column(values, n_seed, device, dtype):
    """Per-config scalars -> (R, 1) column, tiled over replicates."""
    v = torch.tensor(values, device=device, dtype=dtype)
    return v.repeat_interleave(n_seed).unsqueeze(1)


def as_col(v, rows, device, dtype):
    """Accept a scalar or an (R,1)/(R,) tensor and return an (R,1) column."""
    if torch.is_tensor(v):
        return v.reshape(-1, 1).to(device=device, dtype=dtype)
    return torch.full((rows, 1), float(v), device=device, dtype=dtype)


def tile_seeds(t, n_cfg):
    """(n_seed, ...) -> (n_cfg * n_seed, ...) with replicate index fastest."""
    return t.repeat(n_cfg, *([1] * (t.dim() - 1)))
