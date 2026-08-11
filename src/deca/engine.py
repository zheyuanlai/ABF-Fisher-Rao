"""Force-evaluation engine for deca-alanine: :class:`alanine.forcefield.TorchFF` plus an
optional ``torch.compile`` wrapper.

Why compile
-----------
The molecule is small (112 atoms, 6216 nonbonded pairs) and the batch is shallow, so the eager
force evaluation is dominated by kernel launch and autograd bookkeeping rather than arithmetic.
Measured on an H200 NVL, float64, per force evaluation:

===========  ========  ==========  ===========
batch ``B``  eager     compiled    CUDA graph
===========  ========  ==========  ===========
128          2.64 ms   1.42 ms     1.50 ms
512          5.25 ms   1.18 ms     3.66 ms
2048         13.10 ms  1.74 ms     12.54 ms
8192         48.68 ms  6.21 ms     48.05 ms
===========  ========  ==========  ===========

A CUDA graph removes launch overhead but not the cost of the unfused kernels themselves, which
is why it buys almost nothing; ``torch.compile`` fuses them and wins 4-8x.  Full BAOAB step
including the CV geometry, binned accumulation and bias force: **2.03 ms at B = 2048**, i.e.
87 000 ns/day aggregate, against 13 300 ns/day eager.  Beyond B ~ 4096 the step is genuinely
compute-bound and throughput saturates near 110 000 ns/day.

Compilation is a *performance* change and must never be a physics change.  The equivalence is
a hard gate, not an assumption: ``tests/test_deca.py`` asserts compiled-vs-eager agreement at
machine precision (measured max ``|dF| = 1.16e-10`` kJ/mol/nm against ``|F|max = 1.67e5``,
i.e. 7e-16 relative) and re-runs the OpenMM parity check through the compiled path.

Warm-up costs ~180 s once per process, which is why :func:`make_engine` is called once and the
handle reused for the whole run rather than per stage.
"""
from __future__ import annotations

import torch

from alanine.forcefield import TorchFF, extract_parameters

#: Chosen because CUDA graphs measurably do not help here, and bundling them into the compiled
#: artifact would silently pin the batch shape.
COMPILE_MODE = "max-autotune-no-cudagraphs"


class DecaEngine:
    """Batched ff14SB energy/forces for deca-alanine.

    ``forces(x)``: ``(B, 112, 3)`` nm -> ``(B, 112, 3)`` kJ/mol/nm.  ``energy(x)`` -> ``(B,)``
    kJ/mol.  ``compiled=False`` keeps the eager path, which is what the parity tests compare
    against and what a debugging run should use.
    """

    def __init__(self, system, device="cuda", dtype=torch.float64, compiled=True):
        self.params = extract_parameters(system)
        self.tff = TorchFF(self.params, device=device, dtype=dtype)
        self.masses = self.params["masses"]
        self.n_atoms = self.tff.n_atoms
        self.device, self.dtype, self.compiled = device, dtype, bool(compiled)
        self._eager_forces = self.tff.forces
        self._eager_energy = self.tff.energy
        if self.compiled:
            self.forces = torch.compile(self.tff.forces, dynamic=False, mode=COMPILE_MODE)
            self.energy = torch.compile(self.tff.energy, dynamic=False, mode=COMPILE_MODE)
        else:
            self.forces = self._eager_forces
            self.energy = self._eager_energy

    def eager_forces(self, x):
        """The uncompiled force path, for equivalence checks."""
        return self._eager_forces(x)

    def eager_energy(self, x):
        """The uncompiled energy path, for equivalence checks."""
        return self._eager_energy(x)

    def parameter_hash(self):
        from alanine.forcefield import parameter_hash
        return parameter_hash(self.params)


def make_engine(n_res=10, device="cuda", dtype=torch.float64, compiled=True,
                hydrogen_mass=None):
    """Build the vacuum system and its engine.  Returns ``(engine, system, topology)``."""
    from .system import make_system
    _, top, system = make_system(n_res=n_res, hydrogen_mass=hydrogen_mass)
    return DecaEngine(system, device=device, dtype=dtype, compiled=compiled), system, top
