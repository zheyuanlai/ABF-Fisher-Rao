#!/usr/bin/env python
"""Separate the timestep's KINETIC bias from any CONFIGURATIONAL bias, restrained and not.

Why this exists.  `VALINE_STAGE0_HANDOFF.md` R4 attributed a 6.8 K kinetic-temperature deficit
entirely to the stiff dihedral restraint, on the strength of a B = 64 measurement whose sampling
sigma was 5.79 K.  At that noise level a ~3.6 K deficit in the UNRESTRAINED runs is a 0.6 sigma
non-event and was read as "fine".  The first large-batch unrestrained exploration (B = 8604,
sampling sigma ~0.5 K) then reported 293 K -- the same deficit, with no restraint anywhere in
the system.  So the R4 attribution has to be re-measured at a batch size that can actually
resolve it.

What separates the two hypotheses.  BAOAB's kinetic temperature carries a known O(dt^2) bias
even where its CONFIGURATIONAL averages are accurate, and free energies depend only on the
configurational distribution.  So the decisive question is not "is T_kin below 300 K" but
"is the configurational distribution at the right temperature".  Equipartition on the stiff
internal coordinates answers it directly and exactly:

    <0.5 k_b (r - r_0)^2> = 0.5 kB T   ->   T_bond  = k_b <(r-r_0)^2> / kB
    <0.5 k_a (th - th_0)^2> = 0.5 kB T ->   T_angle = k_a <(th-th_0)^2> / kB

These estimators share the fastest modes in the system -- the C-H stretches -- so if the
timestep is corrupting configurational sampling, it shows up here first and largest.

Three timesteps x three restraint strengths are measured in ONE batch per timestep, with the
restraint constant varying by walker.  That removes run-to-run variation from the comparison
the conclusion rests on.

Usage
-----
    CUDA_VISIBLE_DEVICES=7 python -u scripts/check_valine_dt_bias.py \
        --out results/valine/dt_bias
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from alanine.dynamics import BAOAB                                            # noqa: E402
from alanine.forcefield import TorchFF, extract_parameters, parameter_hash    # noqa: E402
from valine.system import (CHI1_ATOMS, N_ATOMS, PHI_ATOMS, PSI_ATOMS,         # noqa: E402
                           make_seed, make_system, validate_seed)
from valine.umbrella import DihedralRestraint                                 # noqa: E402

ALLOWED_GPUS = {"4", "5", "6", "7"}
KB = 0.008314462618
PARENT = (-80.0, 80.0, 180.0)

#: (label, kappa_backbone, kappa_chi1).  0 means genuinely unrestrained.
GROUPS = (
    ("unrestrained", 0.0, 0.0),
    ("stage2_clamp", 500.0, 100.0),      # the Stage-2 chi1-profile setting R4 blamed
    ("pilot_clamp", 110.0, 110.0),       # the softer clamp the pilot reference will use
)


def enforce_gpu_policy():
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is None:
        raise SystemExit("CUDA_VISIBLE_DEVICES must be set explicitly (allowed: 4,5,6,7)")
    cvd = cvd.strip()
    if cvd not in ALLOWED_GPUS:
        raise SystemExit(f"CUDA_VISIBLE_DEVICES={cvd!r} not in {sorted(ALLOWED_GPUS)}")
    return cvd


def internal_temperatures(x, tff):
    """Equipartition temperature from bond lengths and bond angles, per configuration batch.

    Returns ``(T_bond, T_angle)`` in K, each averaged over all terms of that type.  Both are
    exact for a harmonic term at equilibrium; the point here is their dt DEPENDENCE, for which
    any static anharmonic offset cancels.
    """
    d = x[:, tff.bi[:, 0]] - x[:, tff.bi[:, 1]]
    dr = d.norm(dim=-1) - tff.b0
    t_bond = (tff.bk * dr * dr).mean() / KB

    v1 = x[:, tff.ai[:, 0]] - x[:, tff.ai[:, 1]]
    v2 = x[:, tff.ai[:, 2]] - x[:, tff.ai[:, 1]]
    th = torch.atan2(torch.linalg.cross(v1, v2, dim=-1).norm(dim=-1), (v1 * v2).sum(-1))
    dth = th - tff.a0
    t_ang = (tff.ak * dth * dth).mean() / KB
    return float(t_bond), float(t_ang)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/valine/dt_bias")
    ap.add_argument("--dts-fs", type=float, nargs="+", default=[1.0, 0.5, 0.25])
    ap.add_argument("--per-group", type=int, default=2048)
    ap.add_argument("--equil-ps", type=float, default=15.0)
    ap.add_argument("--measure-ps", type=float, default=30.0)
    ap.add_argument("--sample-every-ps", type=float, default=0.05)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=300.0)
    ap.add_argument("--seed", type=int, default=20260803)
    a = ap.parse_args()

    cvd = enforce_gpu_policy()
    device, dtype = "cuda", torch.float64
    G = len(GROUPS)
    B = G * a.per_group
    # sampling sigma of the kinetic temperature for ONE group: T sqrt(2 / (3 N A))
    sigma_T = a.temperature * math.sqrt(2.0 / (3.0 * a.per_group * N_ATOMS))
    print(f"{G} groups x {a.per_group} walkers = {B};  per-group kinetic sampling sigma "
          f"{sigma_T:.2f} K  (R4 measured at B=64, sigma 5.79 K)")

    _, _, system = make_system()
    P = extract_parameters(system)
    print(f"param_hash {parameter_hash(P)}  constraints {system.getNumConstraints()}  "
          f"CUDA_VISIBLE_DEVICES={cvd}")
    tff = TorchFF(P, device=device, dtype=dtype)

    X, e = make_seed(PARENT, system=system)
    validate_seed(system, X[None], np.radians([PARENT]), energy=[e])
    q0 = torch.as_tensor(np.repeat(X[None], B, 0), device=device, dtype=dtype).contiguous()

    cen = torch.as_tensor(np.repeat(np.radians([PARENT]), B, 0), device=device, dtype=dtype)
    kap = np.zeros((B, 3))
    for g, (_, kb, kc) in enumerate(GROUPS):
        s = slice(g * a.per_group, (g + 1) * a.per_group)
        kap[s, 0] = kap[s, 1] = kb
        kap[s, 2] = kc
    restraint = DihedralRestraint([PHI_ATOMS, PSI_ATOMS, CHI1_ATOMS], cen, kap, N_ATOMS,
                                  device=device, dtype=dtype)
    any_restraint = kap.any()

    def force(x):
        f = tff.forces(x)
        return f + restraint.energy_and_force(x)[1] if any_restraint else f

    os.makedirs(a.out, exist_ok=True)
    rows = []
    for dt_fs in a.dts_fs:
        dt = dt_fs / 1000.0
        integ = BAOAB(P["masses"], dt, a.gamma, a.temperature, force, device=device, dtype=dtype)
        gen = torch.Generator(device=device).manual_seed(int(a.seed))
        x = q0.clone()
        v = integ.maxwell(x.shape, gen, device, dtype)
        f = force(x)
        n_eq = int(a.equil_ps / dt)
        n_me = int(a.measure_ps / dt)
        stride = max(1, int(a.sample_every_ps / dt))
        t0 = time.perf_counter()
        for _ in range(n_eq):
            x, v, f = integ.step(x, v, f, gen)
        m = tff.masses.reshape(-1, 1)
        acc = {g: dict(kin=[], bond=[], ang=[]) for g in range(G)}
        n_s = 0
        for s in range(n_me):
            x, v, f = integ.step(x, v, f, gen)
            if (s + 1) % stride == 0:
                for g in range(G):
                    sl = slice(g * a.per_group, (g + 1) * a.per_group)
                    kin = float((m * v[sl] * v[sl]).sum() / (3.0 * a.per_group * N_ATOMS * KB))
                    tb, ta = internal_temperatures(x[sl], tff)
                    acc[g]["kin"].append(kin)
                    acc[g]["bond"].append(tb)
                    acc[g]["ang"].append(ta)
                n_s += 1
        wall = time.perf_counter() - t0
        if not torch.isfinite(x).all():
            raise RuntimeError(f"non-finite positions at dt={dt_fs} fs")
        print(f"\ndt = {dt_fs:.2f} fs   ({n_eq + n_me} steps, {wall / 60:.1f} min, "
              f"{n_s} samples)")
        print(f"  {'group':>14s} {'T_kin':>9s} {'dev':>7s} | {'T_bond':>9s} {'dev':>7s} | "
              f"{'T_angle':>9s} {'dev':>7s}")
        for g, (name, kb, kc) in enumerate(GROUPS):
            k = float(np.mean(acc[g]["kin"]))
            tb = float(np.mean(acc[g]["bond"]))
            ta = float(np.mean(acc[g]["ang"]))
            print(f"  {name:>14s} {k:9.2f} {k - a.temperature:+7.2f} | "
                  f"{tb:9.2f} {tb - a.temperature:+7.2f} | {ta:9.2f} {ta - a.temperature:+7.2f}")
            rows.append(dict(dt_fs=dt_fs, group=name, kappa_backbone=kb, kappa_chi1=kc,
                             T_kin=k, T_bond=tb, T_angle=ta,
                             T_kin_sem=float(np.std(acc[g]["kin"]) / math.sqrt(max(n_s, 1))),
                             T_bond_sem=float(np.std(acc[g]["bond"]) / math.sqrt(max(n_s, 1))),
                             T_angle_sem=float(np.std(acc[g]["ang"]) / math.sqrt(max(n_s, 1))),
                             n_samples=n_s, per_group=a.per_group))

    # ------------------------------------------------------------------ verdict
    def get(dt, name, key):
        for r in rows:
            if r["dt_fs"] == dt and r["group"] == name:
                return r[key]
        return float("nan")

    dt_hi = max(a.dts_fs)
    unres_kin = get(dt_hi, "unrestrained", "T_kin") - a.temperature
    unres_bond = get(dt_hi, "unrestrained", "T_bond") - a.temperature
    res_kin = get(dt_hi, "stage2_clamp", "T_kin") - a.temperature
    print(f"\nAt dt = {dt_hi} fs, per-group sampling sigma {sigma_T:.2f} K:")
    print(f"  unrestrained kinetic deficit  {unres_kin:+.2f} K "
          f"({abs(unres_kin) / sigma_T:.1f} sigma)")
    print(f"  clamped kinetic deficit       {res_kin:+.2f} K "
          f"({abs(res_kin) / sigma_T:.1f} sigma)")
    print(f"  unrestrained CONFIGURATIONAL (bond) deficit {unres_bond:+.2f} K")
    conclusion = (
        "configurational sampling is unbiased; the deficit is the known BAOAB kinetic "
        "artifact and free energies are unaffected"
        if abs(unres_bond) < 3.0 else
        "the CONFIGURATIONAL distribution is also off temperature -- this would bias free "
        "energies and is not a reporting artifact")
    print(f"  -> {conclusion}")

    meta = dict(rows=rows, sigma_T_per_group=sigma_T, temperature=a.temperature,
                gamma=a.gamma, equil_ps=a.equil_ps, measure_ps=a.measure_ps,
                per_group=a.per_group, groups=[list(g) for g in GROUPS],
                param_hash=parameter_hash(P), cuda_visible_devices=cvd,
                conclusion=conclusion,
                supersedes="VALINE_STAGE0_HANDOFF.md R4 (measured at B=64, sigma 5.79 K)")
    with open(os.path.join(a.out, "dt_bias.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"\nwrote {a.out}/dt_bias.json")


if __name__ == "__main__":
    main()
