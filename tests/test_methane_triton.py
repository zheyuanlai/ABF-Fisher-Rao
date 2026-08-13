"""Acceptance gates for the fused Triton pair kernel (performance-only change).

Same admission rule as ``torch.compile`` and float32 before it: the change may only alter
speed, and the tests bound everything else.  Summation order differs from the tensor path, so
the contract is float32-reassociation-level agreement, gated here, not bitwise identity.

Requires a CUDA device (Triton compiles PTX): run with
``CUDA_VISIBLE_DEVICES=<idle gpu> python -m pytest tests/test_methane_triton.py -q``;
skipped automatically in CPU-only runs.  Measured acceptance values at first validation:
vs float64 ground truth 5.2e-6 max rel force (the compiled float32 torch path itself sits at
6.2e-6), vs the float32 torch path 9.9e-7.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

pytest.importorskip("openmm")
torch = pytest.importorskip("torch")

if not torch.cuda.is_available():                     # pragma: no cover
    pytest.skip("Triton kernel needs a CUDA device", allow_module_level=True)

import openmm as mm                                              # noqa: E402
import openmm.unit as u                                          # noqa: E402

from methane import system as msys                               # noqa: E402
from methane.nonbonded import MethaneNonbonded                   # noqa: E402
from methane.triton_pair import build_mol_id                     # noqa: E402


@pytest.fixture(scope="module")
def built():
    mod = msys.build_modeller(r0_nm=0.55, seed=20260812)
    system = msys.build_system(mod.topology)
    pos = msys.apply_constraints(
        system, mod.topology, np.asarray(mod.positions.value_in_unit(u.nanometer)))
    L = float(mod.topology.getUnitCellDimensions().x)
    dev = torch.device("cuda")
    rng = np.random.default_rng(3)
    cfgs = [pos] + [msys.apply_constraints(system, mod.topology,
                                           pos + rng.normal(0, s, pos.shape))
                    for s in (0.002, 0.01) for _ in range(2)]
    for r_nm in (0.34, 0.56, 0.89):
        c = pos.copy()
        m = 0.5 * (c[0] + c[1])
        e = (c[1] - c[0]) / np.linalg.norm(c[1] - c[0])
        c[0], c[1] = m - 0.5 * r_nm * e, m + 0.5 * r_nm * e
        cfgs.append(c)
    xb = torch.tensor(np.stack(cfgs), device=dev, dtype=torch.float32)
    return mod, system, L, dev, xb


def test_mol_id_reconstruction_equals_the_exclusion_mask(built):
    mod, system, L, dev, _ = built
    ff = MethaneNonbonded(system, mod.topology, L, device=dev, dtype=torch.float32)
    mol = build_mol_id(ff.pair).cpu().numpy()        # raises on mismatch -- that IS the gate
    assert len(np.unique(mol)) == msys.N_METHANES + msys.N_WATERS


def test_mol_id_gate_is_live(built):
    """Corrupt one exclusion pair; the assert must fire rather than mis-exclude silently."""
    mod, system, L, dev, _ = built
    ff = MethaneNonbonded(system, mod.topology, L, device=dev, dtype=torch.float32)
    ff.pair.exclusion_pairs = ff.pair.exclusion_pairs.clone()
    ff.pair.exclusion_pairs[0, 1] = 0                # degenerate pair breaks the reconstruction
    with pytest.raises(RuntimeError, match="mis-exclud"):
        build_mol_id(ff.pair)


def test_triton_matches_float64_as_well_as_the_torch_float32_path(built):
    mod, system, L, dev, xb = built
    ff64 = MethaneNonbonded(system, mod.topology, L, device=dev, dtype=torch.float64)
    ff32 = MethaneNonbonded(system, mod.topology, L, device=dev, dtype=torch.float32)
    fftr = MethaneNonbonded(system, mod.topology, L, device=dev,
                            dtype=torch.float32).enable_triton()

    E64, F64 = ff64.energy_forces(xb.double())
    E32, F32 = ff32.energy_forces(xb)
    ET, FT = fftr.energy_forces(xb)
    scale = float(F64.abs().max())

    rel_f_32 = float((F32.double() - F64).abs().max()) / scale
    rel_f_tr = float((FT.double() - F64).abs().max()) / scale
    assert rel_f_tr < 5e-5
    assert rel_f_tr < 3.0 * rel_f_32, "Triton drifts from f64 well beyond the torch f32 path"
    assert float(((ET.double() - E64) / E64.abs()).abs().max()) < 2e-4

    # against the float32 torch path: pure reassociation noise
    assert float((FT - F32).abs().max()) / scale < 1e-5
    assert float(((ET - E32) / E32.abs()).abs().max()) < 1e-5


def test_batched_equals_single(built):
    mod, system, L, dev, xb = built
    fftr = MethaneNonbonded(system, mod.topology, L, device=dev,
                            dtype=torch.float32).enable_triton()
    Eb, Fb = fftr.energy_forces(xb)
    for k in (0, 3, len(xb) - 1):
        E1, F1 = fftr.energy_forces(xb[k:k + 1])
        assert abs(float(Eb[k] - E1[0])) / abs(float(E1[0])) < 2e-6
        assert float((Fb[k] - F1[0]).abs().max() / F1[0].abs().max()) < 2e-6


def test_disabled_engine_is_untouched(built):
    """Without enable_triton the engine must still run the tensor path.

    Not bitwise: ``index_add_`` on CUDA (reciprocal spread, exclusion scatter) is
    atomic-order nondeterministic, so even two identical calls differ at float32
    reassociation level.  The contract is agreement within that noise.
    """
    mod, system, L, dev, xb = built
    ff = MethaneNonbonded(system, mod.topology, L, device=dev, dtype=torch.float32)
    assert getattr(ff, "_mol_id", None) is None
    e1, f1 = ff.energy_forces(xb[:2])
    e2, f2 = ff.pair.energy_forces(xb[:2])
    ex, fx = ff.pair.exclusion_correction(xb[:2])
    ek, fk = ff.recip.energy_forces(xb[:2])
    scale = float(f1.abs().max())
    assert float(((e1 - (e2 + ex + ek + ff.e_self)) / e1.abs()).abs().max()) < 1e-6
    assert float((f1 - (f2 + fx + fk)).abs().max()) / scale < 1e-6
