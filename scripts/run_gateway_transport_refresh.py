#!/usr/bin/env python
"""Transport-refresh campaign, Stage B2: seven arms on fresh paired seeds.

Preregistration: configs/transport_campaign/gateway_transport_refresh_prereg.json.
alpha** is read from Stage A2's alpha_selection.json and RE-DERIVED from the rule before anything
runs.  Arms: abf, fr_uniform, ot_exact, abf_refresh, fr_uniform_refresh, ot_exact_refresh,
ot_full_refresh -- all in ONE batch per chunk (shared initial conditions, Langevin noise and
refresh draws).  Prints wall time and safety counters only -- no error metric.

    CUDA_VISIBLE_DEVICES=3 python -u scripts/run_gateway_transport_refresh.py   # GPU 3 ONLY
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import socket
import sys
import time

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gateway_core as gw  # noqa: E402
from run_gateway_bandwidth_audit import build_config  # noqa: E402
from run_gateway_horizontal_calibration import TAKEN, BASE_PREREG, CORR_PREREG, STEP1, git_rev  # noqa: E402
from run_gateway_horizontal_transport import KEYS  # noqa: E402
from run_gateway_transport_refresh_calibration import select_alpha_exact  # noqa: E402

PREREG = os.path.join(ROOT, "configs/transport_campaign/gateway_transport_refresh_prereg.json")
CAL_SEL = os.path.join(ROOT, "results/transport_campaign/gateway_horizontal/calibration_refine/alpha_selection.json")
OUT_DIR = os.path.join(ROOT, "results", "transport_campaign", "gateway_horizontal", "production_refresh")
BATCH_SEED0 = 61_000
TAKEN_ALL = TAKEN | set(range(480, 488)) | set(range(500, 532))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--skip-frozen-bias", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    pre = json.load(open(PREREG))
    base = json.load(open(BASE_PREREG)); corr = json.load(open(CORR_PREREG)); step1 = json.load(open(STEP1))
    cal = json.load(open(CAL_SEL))
    sampler, cell = base["sampler"], base["cell"]
    h_bias = float(corr["corrected_baseline"]["h_bias"]); h_star = float(corr["corrected_baseline"]["h_read_star"])
    assert abs(h_bias - float(step1["h_bias_corrected"])) < 1e-12 and abs(h_star - float(step1["h_read_star"])) < 1e-12
    gamma = float(corr["rate"]["gamma"])
    assert abs(gamma - float(cal["gamma"])) < 1e-12
    ladder = [float(v) for v in cal["ladder"]]
    ratio = {float(k): float(v) for k, v in cal["ratio"].items()}
    a_star, rule = select_alpha_exact(ladder, ratio)
    assert abs(a_star - float(cal["alpha_star"])) < 1e-12, (a_star, cal["alpha_star"])
    st = pre["stage_B2_production"]["seeds"]
    seeds = list(range(st["first"], st["first"] + st["count"]))
    assert not (set(seeds) & TAKEN_ALL), "production labels collide with taken labels"
    if torch.cuda.is_available():
        assert torch.cuda.device_count() == 1, "pin exactly one GPU"
    os.makedirs(a.out, exist_ok=True)

    FR = dataclasses.replace(gw.FR_UNIFORM, gamma=gamma)
    OT = gw.horizontal_ot(a_star, name="ot_exact")
    ARMS = [gw.ABF, FR, OT, gw.with_refresh(gw.ABF), gw.with_refresh(FR), gw.with_refresh(OT),
            gw.with_refresh(gw.horizontal_ot(1.0, name="ot_full"))]
    assert [m.name for m in ARMS] == list(pre["stage_B2_production"]["arms"].keys()), "arm list drifted from the prereg"
    gw.assert_no_oracle_leakage(ARMS)
    rows = [(init, sd) for init in ["left", "one_right"] for sd in seeds]
    print("Transport-refresh campaign, Stage B2: " + " | ".join(m.name for m in ARMS))
    print(f"  h_bias {h_bias:g}, h_read* {h_star:g}, gamma {gamma:g}, alpha** {a_star:g} [{rule}; ratio {ratio[a_star]:.3f}]")
    print(f"  {len(rows)} rows (seeds {seeds[0]}-{seeds[-1]} x 2 inits), chunk {a.chunk}, batch seeds {BATCH_SEED0}+; no error metric printed (prereg)")
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
        print(f"  chunk {i // a.chunk + 1}: {len(chunk)} rows x {len(ARMS)} arms in {time.time() - t0:.0f}s", flush=True)
        for m in ARMS:
            rr = [r for r in recs if r["method"] == m.name]
            line = f"     {m.name:>20}: "
            if m.use_fr:
                line += f"repl {np.median([r['repl_fraction'] for r in rr]):.4f}, min ESS/N {np.median([r['min_ess_frac'] for r in rr]):.3f}; "
            if m.transport == "horizontal_ot":
                line += f"mean|dx|/event {np.median([np.mean(r['ot_absdx_t'][1:]) for r in rr]):.2e}, max D_move {np.median([np.max(r['dmove_max_t']) for r in rr]):.2e}; "
            if m.refresh == "oracle":
                line += f"refresh events {rr[0]['n_refresh_apply']}; "
            line += f"mean D_cond {np.median([np.nanmean(r['dcond_t']) for r in rr]):.3e}"
            print(line, flush=True)

    fb = None
    if not a.skip_frozen_bias:
        f = base["frozen_bias"]
        Fp = np.stack([r["Fp_hat"] for r in recs_all])
        cfgs = [build_config(sampler, cell, r["init"], h_bias) for r in recs_all]
        group = [f"{r['init']}|{r['seed']}" for r in recs_all]
        t0 = time.time()
        fb = gw.run_frozen_bias(torch.as_tensor(Fp), cfgs, group=group, n_steps=f["n_steps"], burn_frac=f["burn_frac"], seed=f["seed"])
        for i, r in enumerate(recs_all):
            r["frozen_l2_f_kT"] = float(fb["l2_f_kT"][i])
        print(f"  frozen-bias stage: {len(recs_all)} rows in {time.time() - t0:.0f}s", flush=True)

    npz = {k: np.stack([np.asarray(r[k]) for r in recs_all]) for k in KEYS}
    for k in recs_all[0]:
        if k not in KEYS and k != "config":
            npz[k] = np.array([r[k] for r in recs_all])
    npz["config_json"] = np.array([json.dumps(r["config"], sort_keys=True) for r in recs_all])
    npz["h_bias"] = np.array(h_bias); npz["h_read_star"] = np.array(h_star)
    npz["alpha_star"] = np.array(a_star); npz["gamma_selected"] = np.array(gamma)
    if fb is not None:
        npz["frozen_F_hat"] = fb["F_hat"]; npz["frozen_p_B"] = fb["p_B"]
    np.savez_compressed(os.path.join(a.out, "raw.npz"), **npz)
    prov = dict(script=os.path.basename(__file__), git_rev=git_rev(), host=socket.gethostname(),
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                device=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
                prereg=os.path.relpath(PREREG, ROOT), calibration=os.path.relpath(CAL_SEL, ROOT),
                calibration_git_rev=cal.get("git_rev"), h_bias=h_bias, h_read_star=h_star, gamma=gamma,
                alpha_star=a_star, alpha_rule=rule, arms=[m.name for m in ARMS], seeds=seeds, chunk=a.chunk,
                batch_seed0=BATCH_SEED0, wall_seconds=time.time() - t_start, n_runs=len(recs_all))
    with open(os.path.join(a.out, "provenance.json"), "w") as fh:
        json.dump(prov, fh, indent=2, default=float)
    print(f"\ndone: {len(recs_all)} runs in {time.time() - t_start:.0f}s -> {os.path.relpath(a.out, ROOT)}/raw.npz")


if __name__ == "__main__":
    main()
