"""Particle representation for v4-A: the only component allowed to resample.

Frozen protocol: ``docs/V4A_PREREGISTRATION.md``.

The third of the three separated objects::

    FR weights           ->  probability mass          (abffr.persistent_mass)
    physical propagation ->  statistical information   (the ABF accumulators)
    resampling           ->  particle representation   (this module)

This module converts a weighted measure ``sum_i w_i delta_{q_i}`` into an
equal-weight set of physical replicas.  It is the only place where probability
mass is allowed to change which configurations future physical computation is
spent on -- and therefore the only place in v4-A where genealogy is consumed.

It emits **indices and bookkeeping only**.  It never touches an ABF accumulator;
enforcing that is the orchestration layer's job, and the frozen rule is that
extra offspring contribute no information until they have propagated for
``L_hold`` steps.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass(frozen=True)
class ResampleResult:
    src: torch.Tensor          # new[i] = old[src[i]]
    is_clone: torch.Tensor     # which output slots are NEW clones (bool)
    n_replacements: int        # K - #{distinct parents} = excess children


def systematic_resample(weights: torch.Tensor,
                        generator: torch.Generator) -> torch.Tensor:
    """Systematic resampling to fixed population size.  Returns parent indices.

    Exactly uniform weights give the identity map, one child per parent, so a
    no-op reset is a no-op rather than a statistical coincidence.
    """
    K = weights.numel()
    u = torch.rand(1, generator=generator, device=weights.device,
                   dtype=weights.dtype)
    positions = (u + torch.arange(K, device=weights.device,
                                  dtype=weights.dtype)) / K
    cumsum = torch.cumsum(weights, dim=0)
    cumsum[-1] = 1.0                       # guard float drift at the top
    return torch.searchsorted(cumsum, positions.contiguous()).clamp_max(K - 1)


def clone_mask(src: torch.Tensor) -> torch.Tensor:
    """New clones, under the frozen continuation convention.

    For each parent with at least one offspring, the first output slot holding
    it is the *continuation* and inherits its history; further slots are new
    clones.  Without this a full resampling would classify nearly the whole
    population as newborn and put it all in hold-out.
    """
    K = src.numel()
    seen = torch.zeros(K, dtype=torch.bool, device=src.device)
    is_clone = torch.zeros(K, dtype=torch.bool, device=src.device)
    for i in range(K):
        parent = int(src[i])
        if seen[parent]:
            is_clone[i] = True
        else:
            seen[parent] = True
    return is_clone


def resample(weights: torch.Tensor, generator: torch.Generator) -> ResampleResult:
    src = systematic_resample(weights, generator)
    mask = clone_mask(src)
    K = src.numel()
    n_repl = int(K - int(torch.bincount(src, minlength=K).gt(0).sum()))
    return ResampleResult(src=src, is_clone=mask, n_replacements=n_repl)


def apply_holdout(hold: torch.Tensor, src: torch.Tensor,
                  is_clone: torch.Tensor, hold_steps: int) -> torch.Tensor:
    """Continuations inherit the remaining hold-out; new clones restart it."""
    new_hold = hold[src].clone()
    new_hold[is_clone] = int(hold_steps)
    return new_hold


def count_ancestry(ancestors: torch.Tensor, n_particles: int) -> Tuple[float, float]:
    """``(ESS_anc^count / 1, c_max)`` -- family *sizes*, not family mass.

    Always reported beside the mass version: an arm that never resamples has
    count ESS equal to K by construction while its mass may sit on one ancestor,
    so either number alone is misleading.
    """
    counts = torch.bincount(ancestors, minlength=n_particles).double()
    c = counts / float(n_particles)
    nz = c[c > 0]
    return float(1.0 / (nz * nz).sum()), float(c.max())
