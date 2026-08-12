"""Methane pair in explicit SPC/E water (v2 System 4, promoted by Amendment 11).

Specification: ``docs/SPEC_methane_water.md``.  Preregistered expectation: **likely null**.

Nothing in this package may run before the engine-equivalence gate of SPEC §3.2 passes.
"""
from __future__ import annotations

from . import system  # noqa: F401

__all__ = ["system"]
