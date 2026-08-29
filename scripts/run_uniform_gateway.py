#!/usr/bin/env python
"""Uniform-FR campaign, Stage 1: Entropic Gateway, two arms, frozen mechanics.

Everything is inherited from the closed confirmatory-v2 protocol
(results/gateway_anchor/CONFIRMATORY_PREREGISTRATION.json); the only changes,
declared in configs/uniform_campaign/gateway_prereg.json and
docs/UNIFORM_FR_CAMPAIGN.md, are

    arms       abf, fr_uniform          (two arms, nothing else)
    target     uniform on the grid      (gamma = 1.5, inherited, not tuned)

Both arms run in ONE batch per chunk, sharing initial conditions and Langevin
noise, so the comparison is paired by construction.  Profile snapshots
(F, F', KDE marginal, KL to uniform) are stored at every save for the
convergence figures.

    CUDA_VISIBLE_DEVICES=3 python -u scripts/run_uniform_gateway.py
"""
from __future__ import annotations

import argparse
import dataclasses
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

PREREG = os.path.join(ROOT, "configs/uniform_campaign/gateway_prereg.json")
BASE_PREREG = os.path.join(ROOT, "results/gateway_anchor/CONFIRMATORY_PREREGISTRATION.json")
OUT_ROOT = os.path.join(ROOT, "results", "uniform_campaign")


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
    ap.add_argument("--tag", default="gateway")
    ap.add_argument("--chunk", type=int, default=64,
                    help="max (config, seed) rows per batch")
    ap.add_argument("--skip-frozen-bias", action="store_true")
    a = ap.parse_args()

    pre = json.load(open(a.prereg))
    s, c = pre["sampler"], pre["cell"]
    assert abs(s["n_steps"] * s["dt"] - s["T_total"]) < 1e-9, "T_total disagrees with n_steps*dt"

    # The campaign prereg must inherit the confirmatory sampler/cell verbatim: the whole
    # point is that ONLY the arm list and the target change.  Assert it, don't trust it.
    base = json.load(open(BASE_PREREG))
    assert pre["sampler"] == base["sampler"], "sampler block drifted from the frozen confirmatory prereg"
    assert pre["cell"] == base["cell"], "cell block drifted from the frozen confirmatory prereg"
    assert pre["seeds"]["first"] == base["seeds"]["first"] and \
        pre["seeds"]["count"] == base["seeds"]["count"], "seed block drifted"
    assert pre["inits"] == base["inits"], "init block drifted"
    assert pre["arms"] == ["abf", "fr_uniform"], "this campaign is two arms exactly"
    assert float(pre["rates"]["fr_uniform"]) == float(base["rates"]["fr_estimated"]), \
        "gamma must be inherited from the frozen practical rate, not tuned"

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if torch.cuda.is_available():
        assert torch.cuda.device_count() == 1, "pin exactly one GPU"

    rates = pre["rates"]
    ARMS = [gw.ABF,
            dataclasses.replace(gw.FR_UNIFORM, gamma=float(rates["fr_uniform"]))]
    seeds = list(range(pre["seeds"]["first"],
                       pre["seeds"]["first"] + pre["seeds"]["count"]))
    assert not (set(seeds) & set(range(16))), "seeds must not reuse calibration seeds"

    out_dir = os.path.join(OUT_ROOT, a.tag)
    os.makedirs(out_dir, exist_ok=True)
    print(f"UNIFORM-FR campaign, Stage 1 (gateway) -- prereg {os.path.relpath(a.prereg, ROOT)}")
    print(f"  cell beta={c['beta']:g} s={c['s']:g} r={c['r']:g}; "
          f"gamma uniform={rates['fr_uniform']:g}")
    print(f"  {len(seeds)} seeds {seeds[0]}-{seeds[-1]}, inits {pre['inits']}, "
          f"N={s['N']}, T={s['T_total']:g}")
    print(f"  device = "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'} "
          f"(CUDA_VISIBLE_DEVICES={cvd!r})\n")

    t_start = time.time()
    recs_all = []
    rows = [(build_config(pre, init, rates["fr_uniform"]), sd)
            for init in pre["inits"] for sd in seeds]
    print(f"arms {[m.name for m in ARMS]}, {len(rows)} (config, seed) rows x "
          f"{len(ARMS)} arms, one batch per chunk (shared noise)")
    for i in range(0, len(rows), a.chunk):
        chunk = rows[i:i + a.chunk]
        spec = gw.BatchSpec(configs=[x for x, _ in chunk],
                            seeds=[y for _, y in chunk], methods=ARMS,
                            batch_seed=30_000 + i)
        t0 = time.time()
        recs = gw.simulate_batch(spec, store_profiles=bool(pre.get("store_profiles", True)))
        for r in recs:
            r["group"] = "all"
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
    keys = ["t", "P_regions", "Q_regions", "l2_f_t", "l2_fp_t", "ess_t", "wmax_t",
            "x_grid", "F_hat", "Fp_hat", "F_ref", "Fp_ref"]
    if "F_prof_t" in recs_all[0]:
        keys += ["F_prof_t", "Fp_prof_t", "phat_t", "kl_uniform_t"]
    npz = {k: np.stack([r[k] for r in recs_all]) for k in keys}
    # profile snapshots are plotting diagnostics; float32 keeps the artifact under
    # git-hostable size without touching any scored quantity
    for k in ("F_prof_t", "Fp_prof_t", "phat_t"):
        if k in npz:
            npz[k] = npz[k].astype(np.float32)
    for k in recs_all[0]:
        if k not in keys and k != "config":
            npz[k] = np.array([r[k] for r in recs_all])
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
                campaign="uniform_fr", seeds=seeds,
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
