"""C60-C60 in explicit TIP4P-Ew water -- the Amendment 16 eligibility study.

Frozen by docs/SPEC_c60_water.md.  Modules: geometry (the exact cage), system (OpenMM builder
and parity target), nonbonded/pme (batched torch engine, rectangular cell, 4-site water),
dynamics (BAOAB + M-SHAKE + virtual sites + the single xi DOF), observables (n_gap).
"""
