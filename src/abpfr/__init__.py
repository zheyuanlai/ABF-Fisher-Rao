"""abpfr: mollified SHUS (adaptive biasing potential) + temporary marginal Fisher-Rao.

Application branch of the ABP-Fisher-Rao project.  The primary estimated object is a
potential F_t approximating the free energy F along a reaction coordinate; a temporary
Fisher-Rao birth-death step reallocates replicas toward the uniform SHUS target during a
post-discovery establishment transient, and is annealed off.

Frozen conventions live in docs/SPEC_SHUS_FR.md and are enforced by tests/.
"""
