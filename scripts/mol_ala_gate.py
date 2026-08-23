"""Gate: the torch re-implementation of ff14SB must match OpenMM exactly."""
from __future__ import annotations

import json, os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rcwfr.mol import alanine as A


def main():
    import openmm as mm
    import openmm.unit as u
    system, X0 = A.reference_minimum()
    P = A.extract_parameters(system)
    dev, dt = torch.device("cuda"), torch.float64
    top = A.AlaTopology(P, dev, dt)
    rng = np.random.default_rng(0)
    X = X0[None] + 0.02 * rng.standard_normal((24, 22, 3))
    ctx = mm.Context(system, mm.VerletIntegrator(0.001 * u.picoseconds),
                     mm.Platform.getPlatformByName("Reference"))
    Eo, Fo = [], []
    for x in X:
        ctx.setPositions(x)
        st = ctx.getState(getEnergy=True, getForces=True)
        Eo.append(st.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole))
        Fo.append(st.getForces(asNumpy=True).value_in_unit(
            u.kilojoule_per_mole / u.nanometer))
    Eo, Fo = np.array(Eo), np.array(Fo)
    xt = torch.as_tensor(X, device=dev, dtype=dt)
    Et = top.energy(xt).cpu().numpy()
    Ft = (-top.grad(xt)).cpu().numpy()
    res = {"n_conf": len(X), "E_range": [float(Eo.min()), float(Eo.max())],
           "max_rel_E": float(np.max(np.abs(Et - Eo) / np.maximum(np.abs(Eo), 1))),
           "max_abs_E": float(np.max(np.abs(Et - Eo))),
           "max_rel_F": float(np.max(np.abs(Ft - Fo)) / np.max(np.abs(Fo))),
           "n_bonds": len(P["bonds"][0]), "n_angles": len(P["angles"][0]),
           "n_torsions": len(P["torsions"][0]), "n_pairs": int(top.pi.numel()),
           "total_charge": float(P["nb"][0].sum())}
    os.makedirs("results/mol/gate1", exist_ok=True)
    json.dump(res, open("results/mol/gate1/ALA_ff_parity.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
    assert res["max_rel_E"] < 1e-8 and res["max_rel_F"] < 1e-7, "ff parity FAILED"


if __name__ == "__main__":
    main()
