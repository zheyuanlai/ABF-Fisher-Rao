#!/usr/bin/env python
"""Confirmatory gateway run: frozen rates, fresh seeds, five arms, no tuning of any kind.

Every parameter is read from ``CONFIRMATORY_PREREGISTRATION.json`` and asserted, so the
frozen values cannot drift through an edit to this script.  There is no rate ladder here and
no search: the calibration run chose the rates, this run measures them once on seeds that
have never been used.

Five arms, all sharing initial conditions and Langevin noise inside a seed:

    abf              baseline
    fr_estimated     practical mFR -- the primary claim
    sham_practical   its matched sham (same times, same realised counts, uniform identities)
    fr_oracle        oracle mFR -- non-deployable diagnostic
    sham_oracle      its matched sham

Each FR arm gets its **own** sham because the two arms build different targets and therefore
fire different numbers of events; a single sham shadowing the oracle is not an
intensity-matched control for the practical method.

A frozen-bias stage follows, as an endpoint that does not reuse the online estimator: each
arm's learned mean force is held fixed, a fresh population identical across arms is launched
under it with no adaptation and no birth-death, and the free energy is reconstructed from the
sampled density.

    CUDA_VISIBLE_DEVICES=2 python -u scripts/run_gateway_confirm.py
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import dataclasses

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import gateway_core as gw  # noqa: E402

PREREG = os.path.join(ROOT, "results/gateway_anchor/CONFIRMATORY_PREREGISTRATION.json")
OUT_ROOT = os.path.join(ROOT, "results", "gateway_anchor")


def git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:
        return "unknown"


def build_config(pre, init, gamma):
    s = pre["sampler"]
    c = pre["cell"]
    return gw.GatewayConfig(
        beta=c["beta"], H=c["beta_H_kT"] / c["beta"], omega_out=1.0, r=c["r"], s=c["s"],
        N=s["N"], dt=s["dt"], n_steps=s["n_steps"], save_every=s["save_every"], init=init,
        h=s["h"], min_count=s["min_count"], gamma=gamma, eta=s["eta"],
        fr_every=s["fr_every"], fr_burnin=s["fr_burnin"],
        ramp_fraction=s["ramp_fraction"], target_ema_rate=s["target_ema_rate"],
        score_clip=s["score_clip"], max_event_fraction=s["max_event_fraction"],
        ess_window_steps=s["ess_window_steps"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prereg", default=PREREG)
    ap.add_argument("--tag", default="confirmatory")
    ap.add_argument("--chunk", type=int, default=64,
                    help="max (config, seed) rows per batch")
    ap.add_argument("--skip-frozen-bias", action="store_true")
    a = ap.parse_args()

    pre = json.load(open(a.prereg))
    s, c = pre["sampler"], pre["cell"]
    assert abs(s["n_steps"] * s["dt"] - s["T_total"]) < 1e-9, "T_total disagrees with n_steps*dt"
    assert abs(c["beta"] * (c["beta_H_kT"] / c["beta"]) - c["beta_H_kT"]) < 1e-12

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if torch.cuda.is_available():
        assert torch.cuda.device_count() == 1, "pin exactly one GPU"

    # Arms, in the preregistered order.  The two FR rates ride on the METHODS rather than on
    # the config, so all five arms run in ONE batch and share initial conditions and Langevin
    # noise.  Splitting them across batches to give them different rates -- which the first
    # version did -- puts an arm and its baseline in different noise streams and silently
    # unpairs the comparison; see Amendment 1 of the preregistration.  A sham inherits its
    # partner's rate because it must fire at the partner's event times with its counts.
    rates = pre["rates"]
    ARMS = [gw.ABF,
            dataclasses.replace(gw.FR_ESTIMATED, gamma=float(rates["fr_estimated"])),
            gw.SHAM_PRACTICAL,
            dataclasses.replace(gw.FR_ORACLE, gamma=float(rates["fr_oracle"])),
            gw.SHAM_ORACLE]
    seeds = list(range(pre["seeds"]["first"],
                       pre["seeds"]["first"] + pre["seeds"]["count"]))
    assert not (set(seeds) & set(range(16))), "confirmatory seeds must not reuse calibration seeds"

    out_dir = os.path.join(OUT_ROOT, a.tag)
    os.makedirs(out_dir, exist_ok=True)
    print(f"CONFIRMATORY run -- frozen prereg {os.path.relpath(a.prereg, ROOT)}")
    print(f"  cell beta={c['beta']:g} s={c['s']:g} r={c['r']:g}; "
          f"gamma practical={rates['fr_estimated']:g}, oracle={rates['fr_oracle']:g}")
    print(f"  {len(seeds)} fresh seeds {seeds[0]}-{seeds[-1]}, inits {pre['inits']}, "
          f"N={s['N']}, T={s['T_total']:g}")
    print(f"  device = "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'} "
          f"(CUDA_VISIBLE_DEVICES={cvd!r})\n")

    t_start = time.time()
    recs_all = []
    rows = [(build_config(pre, init, rates["fr_estimated"]), sd)
            for init in pre["inits"] for sd in seeds]
    print(f"arms {[m.name for m in ARMS]}, {len(rows)} (config, seed) rows x "
          f"{len(ARMS)} arms, one batch per chunk (shared noise)")
    for i in range(0, len(rows), a.chunk):
        chunk = rows[i:i + a.chunk]
        spec = gw.BatchSpec(configs=[x for x, _ in chunk],
                            seeds=[y for _, y in chunk], methods=ARMS,
                            batch_seed=30_000 + i)
        t0 = time.time()
        recs = gw.simulate_batch(spec)
        for r in recs:
            r["group"] = "all"          # one batch: every arm shares this baseline
        recs_all.extend(recs)
        print(f"  chunk {i // a.chunk + 1}/{-(-len(rows) // a.chunk)}: "
              f"{len(chunk)} rows in {time.time() - t0:.1f}s", flush=True)

    # ------------------------------------------------------------ frozen bias
    fb = None
    if not a.skip_frozen_bias:
        print("\nfrozen-bias stage: no adaptation, no birth-death, identical fresh "
              "population for every arm", flush=True)
        f = pre["frozen_bias"]
        Fp = np.stack([r["Fp_hat"] for r in recs_all])
        cfgs = [build_config(pre, r["init"], r["gamma"]) for r in recs_all]
        # Group by (init, seed) so every arm of one seed is scored on the SAME fresh
        # population and the same Langevin noise -- the frozen-bias endpoint is a paired
        # comparison, exactly like the online one.
        group = [f"{r['init']}|{r['seed']}" for r in recs_all]
        t0 = time.time()
        fb = gw.run_frozen_bias(torch.as_tensor(Fp), cfgs, group=group,
                                n_steps=f["n_steps"],
                                burn_frac=f["burn_frac"], seed=f["seed"])
        print(f"  {len(recs_all)} rows in {time.time() - t0:.1f}s; "
              f"median reconstruction error "
              f"{np.median(fb['l2_f_kT']):.4f} kT", flush=True)
        for i, r in enumerate(recs_all):
            r["frozen_l2_f_kT"] = float(fb["l2_f_kT"][i])

    # ------------------------------------------------------------------ save
    keys = ("t", "P_regions", "Q_regions", "l2_f_t", "l2_fp_t", "ess_t", "wmax_t",
            "x_grid", "F_hat", "Fp_hat", "F_ref", "Fp_ref")
    npz = {k: np.stack([r[k] for r in recs_all]) for k in keys}
    for k in recs_all[0]:
        if k not in keys and k != "config":
            npz[k] = np.array([r[k] for r in recs_all])
    # r["gamma"] is the rate the arm ACTUALLY ran at, not the config default
    npz["config_json"] = np.array([json.dumps(r["config"], sort_keys=True)
                                   for r in recs_all])
    if fb is not None:
        npz["frozen_F_hat"] = fb["F_hat"]
        npz["frozen_p_B"] = fb["p_B"]
    np.savez_compressed(os.path.join(out_dir, "raw.npz"), **npz)

    prov = dict(script=os.path.basename(__file__), git_rev=git_rev(),
                host=socket.gethostname(), cuda_visible_devices=cvd,
                device=(torch.cuda.get_device_name(0) if torch.cuda.is_available()
                        else "cpu"),
                torch=torch.__version__, python=sys.version.split()[0],
                preregistration=pre,
                preregistration_path=os.path.relpath(a.prereg, ROOT),
                confirmatory=True, seeds=seeds,
                frozen_bias=(None if fb is None else
                             dict(n_steps=fb["n_steps"], burn_frac=fb["burn_frac"],
                                  seed=fb["seed"])),
                wall_seconds=time.time() - t_start, n_runs=len(recs_all))
    with open(os.path.join(out_dir, "provenance.json"), "w") as fh:
        json.dump(prov, fh, indent=2, default=float)
    print(f"\nwrote {out_dir}/raw.npz  ({len(recs_all)} runs, "
          f"{time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
