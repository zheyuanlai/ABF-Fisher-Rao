"""abffr: reproducible 2D ABF + Fisher--Rao free-energy study.

Reaction coordinate ``xi(x, y) = x`` on the 2D model potential defined in
:mod:`abffr.potentials`.  See the top-level ``scripts/`` for the CLI entry
points and the README section "2D ABF-FR Fisher--Rao Ablation Study".
"""
from __future__ import annotations

from . import (  # noqa: F401
    diagnostics,
    io_utils,
    metrics,
    plotting,
    potentials,
    reference,
    simulation,
)

__all__ = [
    "potentials",
    "reference",
    "simulation",
    "metrics",
    "diagnostics",
    "plotting",
    "io_utils",
]

__version__ = "0.1.0"
