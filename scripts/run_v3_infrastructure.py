#!/usr/bin/env python
"""v3 infrastructure stage: the K=1024 clouds for the offline benchmark.

Frozen protocol: docs/V3_PREREGISTRATION.md, Appendix A.1 --
4 seeds x 2 families x 6 fixed normalized times = 48 clouds.

The clouds are archived and hashed *before* the offline benchmark runs, so the
benchmark is demonstrably operating on a dataset frozen independently of its own
result.  Q-D (prediction P6) would otherwise be decidable by cloud selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from abffr import metrics as m, reference, simulation_torch as st  # noqa: E402
from abffr.io_utils import RunSpec  # noqa: E402

SEEDS = (0, 1, 2, 3)
NORMALIZED_TIMES = (0.15, 0.30, 0.45, 0.60, 0.75, 0.90)   # Appendix A.1, frozen
FAMILIES = {
    "plain_abf": dict(kind="flat"),
    "capped12": dict(kind="capped", c_cut=12.0),
}
DOMAIN = dict(x_min=-3.0, x_max=3.0, y_min=-2.5, y_max=3.5,
              nx_ref=801, ny_ref=801, nx_profile=401)


def _cfg(n_steps, n_particles):
    return dict(
        simulation=dict(beta=4.0, dt=0.002, n_steps=n_steps,
                        n_particles=n_particles, eval_every=500,
                        x_init_mode="uniform", y_init_mode="uniform"),
        abf=dict(h=0.05, update_every=10, min_count=1.0,
                 observation_order="post_propagation"),
        domain=DOMAIN, potential=dict(x_tilt=0.1021665783),
        fr=dict(noise_chunk_steps=1000))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="results/v3/infrastructure")
    p.add_argument("--n-steps", type=int, default=50000)
    p.add_argument("--n-particles", type=int, default=1024)
    p.add_argument("--device", default="cuda")
    args = p.parse_args(argv)

    out = pathlib.Path(args.out)
    (out / "clouds").mkdir(parents=True, exist_ok=True)
    cfg = _cfg(args.n_steps, args.n_particles)
    x_grid, ref, _ = reference.load_reference_for_run(
        dict(cfg, output_root=str(out)), require_csv=True)
    F_ref, Fp_ref = np.asarray(ref["F_ref"]), np.asarray(ref["Fprime_ref"])
    ev = m.EvalConfig.from_domain(DOMAIN)
    device = torch.device(args.device)

    want_steps = [int(round(f * args.n_steps)) for f in NORMALIZED_TIMES]
    eval_every = cfg["simulation"]["eval_every"]
    off_grid = [s for s in want_steps if s % eval_every != 0]
    if off_grid:
        raise ValueError(
            f"frozen normalized times {NORMALIZED_TIMES} require saved frames at "
            f"steps {want_steps}, but {off_grid} are not multiples of "
            f"eval_every={eval_every} for n_steps={args.n_steps}. The frozen "
            f"campaign uses n_steps=50000, where all six land on saved frames.")
    manifest = []

    for fam_name, fam_spec in FAMILIES.items():
        c = dict(cfg)
        # No FR in either infrastructure family: these runs only supply clouds.
        c["v3"] = dict(enabled=True, family=fam_spec, operator="none")
        specs = [RunSpec(method="v3_infra", target_type="none", seed=s,
                         gamma=0.0, eta=0.10, fr_every=1,
                         burnin_fraction=0.0, stop_fraction=1.0)
                 for s in SEEDS]
        res = st.run_batch(specs, cfg=c, x_grid=x_grid, F_ref=F_ref,
                           Fprime_ref=Fp_ref, ev=ev, device=device,
                           dtype=torch.float64)
        for b, spec in enumerate(specs):
            d = res.diags[b]
            for step in want_steps:
                k = d["steps"].index(step)
                name = f"{fam_name}__seed{spec.seed}__step{step}.npz"
                path = out / "clouds" / name
                np.savez_compressed(
                    path,
                    x=np.asarray(d["X_snap"][k]), y=np.asarray(d["Y_snap"][k]),
                    A=np.asarray(d["F_hat"][k]),
                    A_prime=np.asarray(d["Fprime_hat"][k]),
                    q=np.asarray(d["q_target_grid"][k]),
                    p_hat=np.asarray(d["p_hat_grid"][k]),
                    x_grid=np.asarray(x_grid), F_ref=F_ref,
                    step=step, seed=spec.seed, family=fam_name)
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                manifest.append(dict(file=name, family=fam_name,
                                     seed=int(spec.seed), step=int(step),
                                     t_over_T=step / args.n_steps,
                                     sha256=digest))
                print(f"  {name}  sha256={digest[:16]}")
        print(f"[infra] {fam_name}: {len(SEEDS)} seeds x {len(want_steps)} times "
              f"({res.runtime_seconds:.1f}s)")

    expected = len(FAMILIES) * len(SEEDS) * len(NORMALIZED_TIMES)
    assert len(manifest) == expected == 48, (len(manifest), expected)
    (out / "cloud_manifest.json").write_text(json.dumps(
        dict(n_clouds=len(manifest), seeds=list(SEEDS),
             normalized_times=list(NORMALIZED_TIMES),
             families=list(FAMILIES), n_particles=args.n_particles,
             n_steps=args.n_steps, clouds=manifest), indent=2))
    print(f"[infra] wrote {len(manifest)} clouds + manifest to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
