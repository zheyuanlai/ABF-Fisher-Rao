"""The complete NaCl/water nonbonded model: the methane engine, fed the frozen NaCl arrays.

Nothing here computes physics -- ``methane.nonbonded.PairTerms`` and ``methane.pme
.PMEReciprocal`` do, unchanged (Amendment 14.1).  This class only routes the frozen parameters
(``nacl.system.load_site_params``) and the NaCl constants (cutoff 1.2 / switch 1.0 nm, pinned
PME) into them, mirroring ``MethaneNonbonded`` including the gated Triton fast path.

CHARMM TIP3P note: hydrogens carry LJ, so ``PairTerms.split_path_valid`` is ``False`` here and
the recorded-negative split path is unavailable -- the production path (``energy_forces``) and
the Triton path handle LJ-active hydrogens exactly.
"""
from __future__ import annotations

import torch

from methane.nonbonded import PairTerms
from methane.pme import PMEReciprocal

from . import system as nsys


class NaClNonbonded:
    """Switched LJ + PME for the published NaCl system, batched over walkers."""

    def __init__(self, box_nm, device=None, dtype=torch.float64,
                 alpha_per_nm=None, grid=None, order=None, params=None):
        p = nsys.load_site_params() if params is None else params
        self.params = p
        self.n = len(p["charge"])
        self.box_nm = float(box_nm)
        self.ion_index = torch.as_tensor(list(nsys.ION_INDEX), device=device, dtype=torch.long)
        self.mass = torch.as_tensor(p["mass"], device=device, dtype=dtype)

        alpha = nsys.PME_ALPHA_PER_NM if alpha_per_nm is None else float(alpha_per_nm)
        grid = nsys.PME_GRID if grid is None else tuple(grid)
        order = nsys.PME_SPLINE_ORDER if order is None else int(order)

        self.pair = PairTerms(p["sigma"], p["epsilon"], p["charge"], p["exclusions"],
                              box_nm, nsys.CUTOFF_NM, nsys.SWITCH_NM, alpha,
                              device=device, dtype=dtype)
        self.recip = PMEReciprocal(p["charge"], box_nm, grid, alpha, order=order,
                                   device=device, dtype=dtype)
        self.e_self = self.pair.self_energy()

    def enable_triton(self, block_i=64, block_j=64):
        """Fused Triton pair kernel (performance-only, gated exactly as the methane path:
        ``build_mol_id`` hard-asserts the equal-id mask against the frozen exclusions)."""
        from methane.triton_pair import build_mol_id
        self._mol_id = build_mol_id(self.pair)
        self._triton_blocks = (int(block_i), int(block_j))
        return self

    def energy_forces(self, x, chunk=256):
        """``(E (B,), F (B, N, 3))`` in kJ/mol and kJ/mol/nm."""
        if getattr(self, "_mol_id", None) is not None and x.is_cuda:
            from methane.triton_pair import pair_energy_forces_triton
            bi, bj = self._triton_blocks
            e_r, f_r = pair_energy_forces_triton(self.pair, x, self._mol_id,
                                                 block_i=bi, block_j=bj)
        else:
            e_r, f_r = self.pair.energy_forces(x, chunk=chunk)
        e_x, f_x = self.pair.exclusion_correction(x)
        e_k, f_k = self.recip.energy_forces(x)
        return e_r + e_x + e_k + self.e_self, f_r + f_x + f_k
