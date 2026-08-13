"""NaCl ion pair in explicit CHARMM TIP3P water (v2 System 3, opened by Amendment 14).

Specification: ``docs/SPEC_nacl_water.md``.  The physical model is the published Talmazan 2025
tutorial system, extracted verbatim (``results/nacl/stage0/``); the periodic engine is the
methane package's, consumed rather than forked.

Nothing in this package may run before the engine-equivalence gate of SPEC §3.1 passes.
"""
from __future__ import annotations

from . import system  # noqa: F401

__all__ = ["system"]
