#!/usr/bin/env python
"""Run one q-r decoupling config: Stage 1 calibration, Stage 0.5, or Stage 2.

Frozen protocol: ``docs/QR_DECOUPLING_PREREGISTRATION.md``.

Writes, per config:

    profiles.csv     e_F(t) and e_F'(t) per seed, on BOTH the primary window and
                     the full domain -- Amendment 1 requires the full-domain
                     number beside every A6 result, because ``a`` vanishes at the
                     window edges and an arm that spends nothing where the metric
                     does not score could gain from that alone
    qr_events.csv    one row per allocation opportunity (r*, lambda, chi2, ...)
    summary.csv      per-seed final and integrated errors, genealogy, mass ESS

Runs are resumable at config granularity: an existing summary.csv is left alone
unless --force.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from abffr import metrics, reference, simulation_torch
from abffr.io_utils import RunSpec


def build(cfg):
    dom = cfg["domain"]
    nx = int(dom.get("nx_profile", 401))
    x = np.linspace(dom["x_min"], dom["x_max"], nx)
    y = np.linspace(dom["y_min"], dom["y_max"], int(dom.get("ny_ref", 801)))
    beta = float(cfg["simulation"]["beta"])
    tilt = float(cfg.get("potential", {}).get("x_tilt", 0.0))
    ref = reference.compute_reference(x, y, beta=beta, x_tilt=tilt)
    ev = metrics.EvalConfig.from_domain(dom)
    return x, ref, ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    out = cfg["output_root"]
    os.makedirs(out, exist_ok=True)
    summary_path = os.path.join(out, "summary.csv")
    if os.path.exists(summary_path) and not args.force:
        print(f"{cfg['experiment_name']}: already done ({summary_path})")
        return

    device = torch.device(args.device or cfg.get("device", "cpu"))
    x, ref, ev = build(cfg)
    primary = ev.eval_mask(x)
    full = np.ones_like(x, dtype=bool)
    sim = cfg["simulation"]
    seeds = [int(s) for s in sim["seeds"]]
    batch = int(args.batch or cfg.get("batch_size_configs", 16))
    eval_every = int(sim["eval_every"])
    dt = float(sim["dt"])

    prof_rows, ev_rows, sum_rows = [], [], []
    for i in range(0, len(seeds), batch):
        chunk = seeds[i:i + batch]
        specs = [RunSpec(method="abf_only", target_type="none", seed=s,
                         gamma=0.0, eta=0.10, burnin_fraction=0.0, fr_every=1,
                         stop_fraction=1.0) for s in chunk]
        res = simulation_torch.run_batch(
            specs, cfg=cfg, x_grid=x, F_ref=ref["F_ref"],
            Fprime_ref=ref["Fprime_ref"], ev=ev, device=device,
            dtype=torch.float64, estimator=cfg["abf"].get("estimator",
                                                          "binned_smooth"),
            base_seed=0)
        for d, seed in zip(res.diags, chunk):
            F = np.asarray(d["F_hat"], dtype=float)
            Fp = np.asarray(d["Fprime_hat"], dtype=float)
            t = np.arange(F.shape[0]) * eval_every * dt
            eF = [metrics.l2_error_F(F[k], ref["F_ref"], x, primary)
                  for k in range(F.shape[0])]
            eFf = [metrics.l2_error_F(F[k], ref["F_ref"], x, full)
                   for k in range(F.shape[0])]
            eFp = [metrics.l2_error_Fprime(Fp[k], ref["Fprime_ref"], x, primary)
                   for k in range(F.shape[0])]
            for k in range(F.shape[0]):
                prof_rows.append(dict(seed=seed, t=t[k], e_F=eF[k],
                                      e_F_full=eFf[k], e_Fprime=eFp[k]))
            anc = np.asarray(d["ancestor_ess"], dtype=float)
            sum_rows.append(dict(
                seed=seed, arm=d.get("qr_arm", "A0"),
                e_F_final=eF[-1], e_F_full_final=eFf[-1], e_Fprime_final=eFp[-1],
                e_F_integrated=metrics.integrated_l2_over_time(eF, list(t)),
                ancestor_ess_final=float(anc[-1]),
                n_replacements=int(np.asarray(d["cumulative_replacements"])[-1]),
                gamma_final=json_or_none(d.get("qr_gamma_final")),
                tau_final=json_or_none(d.get("qr_tau_final"))))
        for r in (res.qr_events or []):
            row = {k: v for k, v in r.items()
                   if not isinstance(v, list)}
            row["seed"] = chunk[int(r.get("row", 0))]
            ev_rows.append(row)
        print(f"  {cfg['experiment_name']}: seeds {chunk[0]}-{chunk[-1]} done",
              flush=True)

    pd.DataFrame(prof_rows).to_csv(os.path.join(out, "profiles.csv"), index=False)
    if ev_rows:
        pd.DataFrame(ev_rows).to_csv(os.path.join(out, "qr_events.csv"), index=False)
    pd.DataFrame(sum_rows).to_csv(summary_path, index=False)
    s = pd.DataFrame(sum_rows)
    print(f"{cfg['experiment_name']}: median e_F(T)={s.e_F_final.median():.4f} "
          f"(full {s.e_F_full_final.median():.4f})  "
          f"ancESS={s.ancestor_ess_final.median():.1f}  "
          f"repl={s.n_replacements.median():.0f}")


def json_or_none(v):
    import json
    return json.dumps(v) if v is not None else None


if __name__ == "__main__":
    main()
