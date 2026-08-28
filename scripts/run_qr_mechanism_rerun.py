"""Phase 0: instrumented exact rerun of Stage 2 on K0-K3, arms A0/A6a/A6b.

Frozen protocol: ``docs/MECHANISM_CAMPAIGN_PREREGISTRATION.md``.

The original configs are loaded VERBATIM and the engine is unchanged; the only
difference from ``run_qr_campaign.py`` is that the per-snapshot ``Fprime_hat``,
``F_hat`` and ``p_hat_grid`` the engine already emits are saved instead of being
scored and discarded.  A fidelity gate compares the recomputed e_F(t) against the
archived ``profiles.csv``: if the rerun is not the old experiment, the audit of
it would be an audit of something else.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from abffr import metrics, simulation_torch                       # noqa: E402
from abffr.io_utils import RunSpec                                # noqa: E402
from run_qr_campaign import build                                 # noqa: E402

OUT = os.path.join(ROOT, "results", "qr_mechanism", "phase0")
CELLS = ("K0", "K1", "K2", "K3")
ARMS = ("A0", "A6a", "A6b")


def run_config(cell, arm, device):
    cfg = yaml.safe_load(open(os.path.join(
        ROOT, "configs", "qr_decoupling", f"stage2_{cell}_{arm}.yaml")))
    outdir = os.path.join(OUT, cell, arm)
    os.makedirs(outdir, exist_ok=True)
    if os.path.exists(os.path.join(outdir, "fidelity.json")):
        print(f"  {cell}/{arm}: exists, skipped", flush=True)
        return

    x, ref, ev = build(cfg)
    primary = ev.eval_mask(x)
    full = np.ones_like(x, dtype=bool)
    sim = cfg["simulation"]
    seeds = [int(s) for s in sim["seeds"]]
    batch = int(cfg.get("batch_size_configs", 16))
    eval_every, dt = int(sim["eval_every"]), float(sim["dt"])

    arch = pd.read_csv(os.path.join(
        ROOT, "results", "qr_decoupling", "stage2", cell, arm, "profiles.csv"))

    t0, fid = time.time(), []
    for i in range(0, len(seeds), batch):
        chunk = seeds[i:i + batch]
        specs = [RunSpec(method="abf_only", target_type="none", seed=s,
                         gamma=0.0, eta=0.10, burnin_fraction=0.0, fr_every=1,
                         stop_fraction=1.0) for s in chunk]
        res = simulation_torch.run_batch(
            specs, cfg=cfg, x_grid=x, F_ref=ref["F_ref"],
            Fprime_ref=ref["Fprime_ref"], ev=ev, device=device,
            dtype=torch.float64,
            estimator=cfg["abf"].get("estimator", "binned_smooth"), base_seed=0)
        for d, seed in zip(res.diags, chunk):
            F = np.asarray(d["F_hat"], dtype=float)
            Fp = np.asarray(d["Fprime_hat"], dtype=float)
            P = np.asarray(d["p_hat_grid"], dtype=float)
            t = np.arange(F.shape[0]) * eval_every * dt
            eF = np.array([metrics.l2_error_F(F[k], ref["F_ref"], x, primary)
                           for k in range(F.shape[0])])
            eFf = np.array([metrics.l2_error_F(F[k], ref["F_ref"], x, full)
                            for k in range(F.shape[0])])
            a = arch[arch.seed == seed].sort_values("t")
            fid.append(dict(seed=int(seed),
                            final_new=float(eF[-1]),
                            final_old=float(a.e_F.iloc[-1]),
                            max_abs_dev=float(np.max(np.abs(
                                eF - a.e_F.to_numpy()))),
                            final_rel=float(abs(eF[-1] - a.e_F.iloc[-1])
                                            / max(a.e_F.iloc[-1], 1e-300))))
            np.savez_compressed(
                os.path.join(outdir, f"seed{seed:04d}.npz"),
                t=t, Fprime_hat=Fp, F_hat=F, p_hat_grid=P,
                e_F=eF, e_F_full=eFf, x_grid=x,
                F_ref=ref["F_ref"], Fprime_ref=ref["Fprime_ref"],
                primary_mask=primary)
        print(f"  {cell}/{arm}: seeds {chunk[0]}-{chunk[-1]} "
              f"({time.time() - t0:.0f}s)", flush=True)

    med_rel = float(np.median([f["final_rel"] for f in fid]))
    gate = dict(cell=cell, arm=arm, n_seeds=len(fid),
                median_final_rel_dev=med_rel,
                max_final_rel_dev=float(max(f["final_rel"] for f in fid)),
                fidelity_pass=bool(med_rel < 0.01), per_seed=fid,
                wall_seconds=time.time() - t0)
    with open(os.path.join(outdir, "fidelity.json"), "w") as fh:
        json.dump(gate, fh, indent=2)
    print(f"  {cell}/{arm}: fidelity median rel dev {med_rel:.2e} "
          f"-> {'PASS' if gate['fidelity_pass'] else '*** FAIL ***'}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    device = torch.device(a.device)
    for cell in CELLS:
        for arm in ARMS:
            run_config(cell, arm, device)


if __name__ == "__main__":
    main()
