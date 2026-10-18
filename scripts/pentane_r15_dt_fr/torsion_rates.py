#!/usr/bin/env python
"""Supplementary numerical validation: torsional transition rates of the UNBIASED production integrator vs dt.

Exact-Gibbs start (Stage A init_gibbs groups), production EM (clip 200), no bias, T = 10 per dt.  A dihedral's
state is the last CORE it visited (trans |phi| < 30 deg, gauche |phi -+ 116.57 deg| < 30 deg; core-to-core
counting, so barrier recrossings are not transitions).  Reports transitions per dihedral per unit physical time
(trans->gauche and gauche->trans) with SE over groups.  Physical-time rates are dt-independent for an exact
integrator, so any trend with dt is a discretisation effect on the kinetics (the stationary populations are
checked separately in Stage A).

    python scripts/pentane_r15_dt_fr/torsion_rates.py --dt 5e-4 --group 0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
torch.set_default_dtype(torch.float64)
from alkanes import potentials as pot, geometry as geom   # noqa: E402

OUT = os.path.join(ROOT, "results", "pentane_r15_dt_fr", "torsion_rates")
GA = math.radians(116.57); W = math.radians(30.0)


def core(phi):
    c = torch.full_like(phi, -1, dtype=torch.long)
    c = torch.where(phi.abs() < W, torch.zeros_like(c), c)
    c = torch.where((phi - GA).abs() < W, torch.ones_like(c), c)
    return torch.where((phi + GA).abs() < W, 2 * torch.ones_like(c), c)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dt", type=float, required=True); ap.add_argument("--group", type=int, required=True)
    ap.add_argument("--T", type=float, default=10.0); a = ap.parse_args()
    torch.set_num_threads(1)
    path = os.path.join(OUT, f"rates__dt{a.dt:g}__g{a.group}.json")
    if os.path.exists(path):
        return
    os.makedirs(OUT, exist_ok=True)
    q = torch.as_tensor(np.load(os.path.join(ROOT, "results", "pentane_r15_dt_fr", "stage_a", "init_gibbs.npz"))["q"])[a.group * 1024:(a.group + 1) * 1024].clone()
    p = pot.AlkaneParams(n_atoms=5, beta=2.0, sigma=2.3, epsilon=1.0, decouple=False, force_clip=200.0)
    gen = torch.Generator().manual_seed(4242 + a.group * 7 + int(round(a.dt * 1e7)))
    n = int(round(a.T / a.dt)); ns = math.sqrt(2 * a.dt / p.beta)
    state = torch.stack([core(geom.signed_dihedral(q, 0, 1, 2, 3)), core(geom.signed_dihedral(q, 1, 2, 3, 4))], 1)
    t2g = 0; g2t = 0; g2g = 0; t_in_t = 0.0; t_in_g = 0.0
    for step in range(n):
        q = geom.remove_com(q + a.dt * pot.forces(q, p) + ns * torch.randn(q.shape, generator=gen))
        c = torch.stack([core(geom.signed_dihedral(q, 0, 1, 2, 3)), core(geom.signed_dihedral(q, 1, 2, 3, 4))], 1)
        known = state >= 0
        t_in_t += float((state == 0).sum()) * a.dt; t_in_g += float((state > 0).sum()) * a.dt
        ch = (c >= 0) & (c != state)
        t2g += int((ch & known & (state == 0)).sum()); g2t += int((ch & known & (state > 0) & (c == 0)).sum())
        g2g += int((ch & known & (state > 0) & (c > 0)).sum())
        state = torch.where(c >= 0, c, state)
    res = dict(dt=a.dt, group=a.group, T=a.T, n_walkers=1024, t2g=t2g, g2t=g2t, g2g=g2g, time_in_trans=t_in_t, time_in_gauche=t_in_g,
               k_t2g=t2g / max(t_in_t, 1e-12), k_g2t=g2t / max(t_in_g, 1e-12))
    json.dump(res, open(path, "w"), indent=1); print(res)


if __name__ == "__main__":
    main()
