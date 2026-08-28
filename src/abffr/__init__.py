"""abffr: reproducible 2D ABF + Fisher--Rao free-energy study.

Reaction coordinate ``xi(x, y) = x`` on the 2D model potential defined in
:mod:`abffr.potentials`.  See the top-level ``scripts/`` for the CLI entry
points and the README section "2D ABF-FR Fisher--Rao Ablation Study".
"""
from __future__ import annotations

from . import (  # noqa: F401
    accel,
    clean_v2,
    diagnostics,
    io_utils,
    metrics,
    oracle_short_burst,
    information_target,
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
    # clean-v2 (docs/CLEAN_V2_PREREGISTRATION.md): the frozen algorithm and the
    # time-to-accuracy endpoint that replaced final-error selection.
    "clean_v2",
    "accel",
    "oracle_short_burst",
    "information_target",
]

__version__ = "0.1.0"
