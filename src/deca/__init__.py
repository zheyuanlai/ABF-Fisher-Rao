"""Deca-alanine (Ace-(Ala)10-Nme, vacuum ff14SB): the v2 prior-art molecular benchmark.

The physical model is built by :mod:`deca.system` and evaluated by the *existing*
:class:`alanine.forcefield.TorchFF`, which is force-field agnostic -- it consumes whatever
``HarmonicBond`` / ``HarmonicAngle`` / ``PeriodicTorsion`` / ``Nonbonded`` parameters OpenMM
produces.  Nothing about the energy path is new here, so the Stage-0 parity gate is a
re-validation of a component that already passed it on a smaller molecule, not a new
implementation to trust.
"""
from .system import (  # noqa: F401
    N_RES, atom_index, backbone_dihedrals, build_helix, make_system, names_and_bonds,
    per_residue_chirality, terminal_carbonyls, topology, validate_structure, validate_thermal,
)
