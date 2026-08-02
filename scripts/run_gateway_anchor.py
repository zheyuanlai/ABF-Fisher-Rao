#!/usr/bin/env python
"""Four-arm mFR comparison at the preregistered establishment-limited anchor.

Reads the anchor from ``phase_classification.frozen.json``.  It refuses to run if that file
is missing: the whole point of the design is that the cell was chosen by a rule applied to
an ABF-only map, before any Fisher-Rao arm existed, and a hand-supplied cell would silently
discard that guarantee.

Arms (all four share initial conditions and Langevin noise inside a seed):

    abf            baseline
    sham           matched sham resampling -- same event times and the same REALISED
                   clone/delete counts as fr_oracle, identities drawn uniformly
    fr_oracle      FR target from the analytic F; non-deployable, diagnostic
    fr_estimated   FR target from the online EMA of the bias; deployable

Initialisations: ``left`` (headline) and ``one_right`` (mechanism control, in which one
walker starts across the gateway so discovery is free and any acceleration measured is
population establishment rather than first passage).

An FR-rate ladder is run as well.  The accepted toy setup's gamma = 15 is the frozen value,
but a null at one rate invites the reply "you used the wrong rate", and the repository's own
alkane study showed a rate ladder can be monotone in the wrong direction.  Reporting the
ladder answers the objection in advance rather than after review.

    CUDA_VISIBLE_DEVICES=2 python -u scripts/run_gateway_anchor.py
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import gateway_core as gw  # noqa: E402

FROZEN = os.path.join(ROOT, "results/gateway_phase/production/phase_classification.frozen.json")
OUT_ROOT = os.path.join(ROOT, "results", "gateway_anchor")
GAMMA_LADDER = (0.5, 1.5, 5.0, 15.0)     # 15.0 is the frozen accepted-toy value
ARMS = [gw.ABF, gw.SHAM, gw.FR_ORACLE, gw.FR_ESTIMATED]


def git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen", default=FROZEN)
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--dt", type=float, default=4e-4)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--n-walkers", type=int, default=2048)
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=128, help="max (config,seed) rows per batch")
    ap.add_argument("--tag", default="production")
    a = ap.parse_args()

    if not os.path.exists(a.frozen):
        raise SystemExit(
            f"{a.frozen} not found. The anchor must come from the frozen ABF-only "
            f"classification; run scripts/analyze_gateway_phase.py first.")
    frozen = json.load(open(a.frozen))
    anc = frozen["anchor"]
    if anc is None:
        raise SystemExit("the frozen classification found no establishment-limited cell")
    print(f"anchor (preregistered): beta={anc['beta']:g}  s={anc['s']:g}  r={anc['r']:g}  "
          f"[{anc['regime']}]")
    print(f"  frozen at {frozen['frozen_at']}, raw_sha256 {frozen['raw_sha256']}")
    print(f"  ABF-only baseline there: T_hit/T={anc['T_hit_frac']:.3f}, "
          f"T_est/T={anc['T_est_frac']:.3f}, below half for {anc['below_half_frac']:.3f} "
          f"of the run, final occ/target={anc['occ_over_target']:.3f}")

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if torch.cuda.is_available():
        assert torch.cuda.device_count() == 1, (
            f"exactly one GPU must be visible, saw {torch.cuda.device_count()}")

    beta, H = float(anc["beta"]), frozen["provenance"]["beta_H_kT"] / float(anc["beta"])
    rows = []
    for init in ("left", "one_right"):
        for gamma in GAMMA_LADDER:
            cfg = gw.GatewayConfig(beta=beta, H=H, s=float(anc["s"]), r=float(anc["r"]),
                                   gamma=gamma, init=init, N=a.n_walkers, dt=a.dt,
                                   n_steps=a.steps, save_every=a.save_every)
            for sd in range(a.seeds):
                rows.append((cfg, sd))

    out_dir = os.path.join(OUT_ROOT, a.tag)
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{len(rows)} (config, seed) rows x {len(ARMS)} arms = "
          f"{len(rows) * len(ARMS)} runs;  gamma ladder {GAMMA_LADDER}, "
          f"frozen value {GAMMA_LADDER[-1]:g}")
    print(f"  T = {a.steps * a.dt:g}, N = {a.n_walkers}, "
          f"device = {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'} "
          f"(CUDA_VISIBLE_DEVICES={cvd!r})")

    t_start = time.time()
    recs_all = []
    for i in range(0, len(rows), a.chunk):
        chunk = rows[i:i + a.chunk]
        spec = gw.BatchSpec(configs=[c for c, _ in chunk], seeds=[s for _, s in chunk],
                            methods=ARMS, batch_seed=20_000 + i)
        t0 = time.time()
        recs = gw.simulate_batch(spec)
        recs_all.extend(recs)
        print(f"  chunk {i // a.chunk + 1}/{-(-len(rows) // a.chunk)}: {len(chunk)} rows "
              f"x {len(ARMS)} arms in {time.time() - t0:.1f}s", flush=True)

    keys = ("t", "P_regions", "Q_regions", "l2_f_t", "l2_fp_t", "ess_t", "wmax_t",
            "x_grid", "F_hat", "Fp_hat", "F_ref", "Fp_ref")
    npz = {k: np.stack([r[k] for r in recs_all]) for k in keys}
    for k in recs_all[0]:
        if k not in keys and k != "config":
            npz[k] = np.array([r[k] for r in recs_all])
    npz["gamma"] = np.array([r["config"]["gamma"] for r in recs_all])
    npz["config_json"] = np.array([json.dumps(r["config"], sort_keys=True)
                                   for r in recs_all])
    np.savez_compressed(os.path.join(out_dir, "raw.npz"), **npz)

    prov = dict(script=os.path.basename(__file__), git_rev=git_rev(),
                host=socket.gethostname(), cuda_visible_devices=cvd,
                device=(torch.cuda.get_device_name(0) if torch.cuda.is_available()
                        else "cpu"),
                torch=torch.__version__, python=sys.version.split()[0],
                anchor=anc, frozen_source=os.path.relpath(a.frozen, ROOT),
                frozen_raw_sha256=frozen["raw_sha256"], frozen_at=frozen["frozen_at"],
                arms=[m.name for m in ARMS], gamma_ladder=list(GAMMA_LADDER),
                gamma_frozen=GAMMA_LADDER[-1], inits=["left", "one_right"],
                seeds=list(range(a.seeds)), n_walkers=a.n_walkers, dt=a.dt,
                n_steps=a.steps, save_every=a.save_every, T_total=a.steps * a.dt,
                health_gates=dict(ess_anc_over_N_min=0.30, wmax_max=0.05),
                wall_seconds=time.time() - t_start, n_runs=len(recs_all))
    with open(os.path.join(out_dir, "provenance.json"), "w") as fh:
        json.dump(prov, fh, indent=2, default=float)
    print(f"\nwrote {out_dir}/raw.npz  ({len(recs_all)} runs, "
          f"{time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
