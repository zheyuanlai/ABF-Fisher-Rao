"""Atomistic Ace-Ala-Nme (AMBER ff14SB, vacuum) for the 2-D Ramachandran mFR-ABF study.

Stage-0 only at present: system construction, umbrella seeding with its validation gate, and
the batched torch force field with its OpenMM parity gate.  No sampler, no production driver.
See ALANINE_SPEC.md and ALANINE_EXECUTION_DECISION.md.
"""
