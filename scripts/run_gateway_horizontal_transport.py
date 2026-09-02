#!/usr/bin/env python
"""Transport campaign, Stage B: abf vs fr_uniform vs ot_matched vs ot_full, fresh paired seeds.

Preregistration: configs/transport_campaign/gateway_horizontal_transport_prereg.json.
alpha* is read from Stage A's alpha_selection.json and RE-DERIVED from the prereg rule on the
stored ladder table before anything runs; a mismatch refuses to start.  All four arms in ONE
batch per chunk (shared initial conditions and Langevin noise); raw accumulators, profiles,
D_cond and the OT event records saved at every save; frozen-bias stage inherited on every arm.

Prints wall time and safety counters only -- no error metric (prereg prohibition).

    CUDA_VISIBLE_DEVICES=3 python -u scripts/run_gateway_horizontal_transport.py   # GPU 3 ONLY
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gateway_core as gw  # noqa: E402
from run_gateway_bandwidth_audit import build_config  # noqa: E402
from run_gateway_horizontal_calibration import select_alpha, TAKEN  # noqa: E402

PREREG = os.path.join(ROOT, "configs/transport_campaign/gateway_horizontal_transport_prereg.json")
BASE_PREREG = os.path.join(ROOT, "results/gateway_anchor/CONFIRMATORY_PREREGISTRATION.json")
CORR_PREREG = os.path.join(ROOT, "configs/information_campaign/gateway_corrected_confirmation_prereg.json")
STEP1 = os.path.join(ROOT, "results/information_campaign/gateway_baseline_audit/analysis.json")
CAL_SEL = os.path.join(ROOT, "results/transport_campaign/gateway_horizontal/calibration/alpha_selection.json")
OUT_DIR = os.path.join(ROOT, "results", "transport_campaign", "gateway_horizontal", "production")
BATCH_SEED0 = 51_000
KEYS = ["t", "P_regions", "Q_regions", "l2_f_t", "l2_fp_t", "ess_t", "wmax_t",
        "x_grid", "F_hat", "Fp_hat", "F_ref", "Fp_ref",
        "F_prof_t", "Fp_prof_t", "phat_t", "kl_uniform_t", "Sf_t", "C_t",
        "dcond_t", "dcond_nbins_t", "dmove_mean_t", "dmove_p95_t", "dmove_max_t", "ot_absdx_t", "alpha_t"]


def git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--skip-frozen-bias", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    pre = json.load(open(PREREG))
    base = json.load(open(BASE_PREREG))
    corr = json.load(open(CORR_PREREG))
    step1 = json.load(open(STEP1))
    cal = json.load(open(CAL_SEL))
    sampler, cell = base["sampler"], base["cell"]
    assert pre["cell"] == cell and pre["inits"] == base["inits"]
    h_bias = float(corr["corrected_baseline"]["h_bias"])
    h_star = float(corr["corrected_baseline"]["h_read_star"])
    assert abs(h_bias - float(step1["h_bias_corrected"])) < 1e-12 and abs(h_bias - float(sampler["h"])) < 1e-12
    assert abs(h_star - float(step1["h_read_star"])) < 1e-12 and abs(h_star - pre["stage_B_production"]["readouts"]["primary"]) < 1e-12
    gamma = float(corr["rate"]["gamma"])
    assert abs(gamma - float(cal["gamma"])) < 1e-12 and abs(h_bias - float(cal["h_bias"])) < 1e-12

    # alpha*: re-derive from the stored ladder table by the prereg rule; refuse on mismatch
    ladder = [float(v) for v in cal["ladder_final"]]
    ratio = {float(k): float(v) for k, v in cal["ratio"].items()}
    a_star, rule = select_alpha(ladder, {al: ratio[al] for al in ladder})
    assert abs(a_star - float(cal["alpha_star"])) < 1e-12, (a_star, cal["alpha_star"])
    assert 0.0 < a_star < 1.0

    st = pre["stage_B_production"]["seeds"]
    seeds = list(range(st["first"], st["first"] + st["count"]))
    cal_seeds = set(int(s) for s in cal["seeds"])
    assert not (set(seeds) & (TAKEN | cal_seeds)), "production labels collide with taken or calibration labels"
    if torch.cuda.is_available():
        assert torch.cuda.device_count() == 1, "pin exactly one GPU"
    os.makedirs(a.out, exist_ok=True)

    ARMS = [gw.ABF, dataclasses.replace(gw.FR_UNIFORM, gamma=gamma),
            gw.horizontal_ot(a_star, name="ot_matched"), gw.horizontal_ot(1.0, name="ot_full")]
    gw.assert_no_oracle_leakage(ARMS)
    rows = [(init, sd) for init in pre["inits"] for sd in seeds]
    print("Transport campaign, Stage B: abf | fr_uniform | ot_matched | ot_full (fresh paired seeds)")
    print(f"  h_bias {h_bias:g}, h_read* {h_star:g}, gamma {gamma:g}, alpha* {a_star:g} [{rule}; ratio {ratio[a_star]:.3f}]")
    print(f"  {len(rows)} rows (seeds {seeds[0]}-{seeds[-1]} x {pre['inits']}), chunk {a.chunk}, batch seeds {BATCH_SEED0}+")
    print("  no error metric is printed here (prereg)")
    if a.dry_run:
        return
    t_start = time.time()

    recs_all = []
    for i in range(0, len(rows), a.chunk):
        chunk = rows[i:i + a.chunk]
        cfgs = [build_config(sampler, cell, init, h_bias) for init, _ in chunk]
        spec = gw.BatchSpec(configs=cfgs, seeds=[sd for _, sd in chunk], methods=ARMS, batch_seed=BATCH_SEED0 + i)
        t0 = time.time()
        recs = gw.simulate_batch(spec, store_profiles=True, store_accumulators=True, store_conditional=True)
        recs_all.extend(recs)
        fr = [r for r in recs if r["method"] == "fr_uniform"]
        print(f"  chunk {i // a.chunk + 1}: {len(chunk)} rows x {len(ARMS)} arms in {time.time() - t0:.0f}s; "
              f"FR median repl_fraction {np.median([r['repl_fraction'] for r in fr]):.4f}, "
              f"min ESS/N {np.median([r['min_ess_frac'] for r in fr]):.3f}", flush=True)
        for nm in ("ot_matched", "ot_full"):
            ot = [r for r in recs if r["method"] == nm]
            print(f"     {nm}: events {ot[0]['n_ot_apply']}, median mean|dx|/event {np.median([np.mean(r['ot_absdx_t'][1:]) for r in ot]):.2e}, "
                  f"median max D_move {np.median([np.max(r['dmove_max_t']) for r in ot]):.3e}, "
                  f"median mean D_cond {np.median([np.nanmean(r['dcond_t']) for r in ot]):.3e}", flush=True)

    fb = None
    if not a.skip_frozen_bias:
        f = base["frozen_bias"]
        Fp = np.stack([r["Fp_hat"] for r in recs_all])
        cfgs = [build_config(sampler, cell, r["init"], h_bias) for r in recs_all]
        group = [f"{r['init']}|{r['seed']}" for r in recs_all]
        t0 = time.time()
        fb = gw.run_frozen_bias(torch.as_tensor(Fp), cfgs, group=group, n_steps=f["n_steps"],
                                burn_frac=f["burn_frac"], seed=f["seed"])
        for i, r in enumerate(recs_all):
            r["frozen_l2_f_kT"] = float(fb["l2_f_kT"][i])
        print(f"  frozen-bias stage: {len(recs_all)} rows in {time.time() - t0:.0f}s", flush=True)

    npz = {k: np.stack([np.asarray(r[k]) for r in recs_all]) for k in KEYS}
    for k in recs_all[0]:
        if k not in KEYS and k != "config":
            npz[k] = np.array([r[k] for r in recs_all])
    npz["config_json"] = np.array([json.dumps(r["config"], sort_keys=True) for r in recs_all])
    npz["h_bias"] = np.array(h_bias)
    npz["h_read_star"] = np.array(h_star)
    npz["alpha_star"] = np.array(a_star)
    npz["gamma_selected"] = np.array(gamma)
    if fb is not None:
        npz["frozen_F_hat"] = fb["F_hat"]
        npz["frozen_p_B"] = fb["p_B"]
    np.savez_compressed(os.path.join(a.out, "raw.npz"), **npz)

    prov = dict(script=os.path.basename(__file__), git_rev=git_rev(), host=socket.gethostname(),
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                device=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
                prereg=os.path.relpath(PREREG, ROOT), calibration=os.path.relpath(CAL_SEL, ROOT),
                calibration_git_rev=cal.get("git_rev"), h_bias=h_bias, h_read_star=h_star, gamma=gamma,
                alpha_star=a_star, alpha_rule=rule, arms=[m.name for m in ARMS], seeds=seeds, inits=pre["inits"],
                chunk=a.chunk, batch_seed0=BATCH_SEED0, wall_seconds=time.time() - t_start, n_runs=len(recs_all))
    with open(os.path.join(a.out, "provenance.json"), "w") as fh:
        json.dump(prov, fh, indent=2, default=float)
    print(f"\ndone: {len(recs_all)} runs in {time.time() - t_start:.0f}s -> {os.path.relpath(a.out, ROOT)}/raw.npz")


if __name__ == "__main__":
    main()
