#!/usr/bin/env python
"""Entropic gateway corrected-baseline CONFIRMATION (step 2): abf vs fr_uniform, fresh seeds.

Preregistration: configs/information_campaign/gateway_corrected_confirmation_prereg.json
(written AFTER step 1 closed; the corrected h_bias / h_read* are read from step 1's
analysis.json and asserted against the prereg).  Sampler/cell inherited verbatim from the
frozen confirmatory prereg except h.

Phase A (only if the prereg says the rate must be RE-EARNED, i.e. h_bias changed): a
safety-only ladder on fresh calibration labels -- the largest gamma whose median min
ancestor ESS/N >= 0.30 and median max lineage share <= 0.05.  No error metric is read.
Phase B: production, both arms in ONE batch per chunk (shared initial conditions and
Langevin noise), raw accumulators + profiles recorded, frozen-bias stage inherited.

Prints wall time and safety counters only -- no error metric (prereg prohibition).

    CUDA_VISIBLE_DEVICES=3 python -u scripts/run_gateway_corrected_confirmation.py   # GPU 3 ONLY
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
from run_gateway_bandwidth_audit import build_config  # noqa: E402  (same config builder)

PREREG = os.path.join(ROOT, "configs/information_campaign/gateway_corrected_confirmation_prereg.json")
BASE_PREREG = os.path.join(ROOT, "results/gateway_anchor/CONFIRMATORY_PREREGISTRATION.json")
STEP1 = os.path.join(ROOT, "results/information_campaign/gateway_baseline_audit/analysis.json")
OUT_DIR = os.path.join(ROOT, "results", "information_campaign", "gateway_corrected_confirmation")


def git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def run_batch(rows, methods, sampler, cell, h, batch_seed, **kw):
    cfgs = [build_config(sampler, cell, init, h) for init, _ in rows]
    spec = gw.BatchSpec(configs=cfgs, seeds=[sd for _, sd in rows], methods=methods, batch_seed=batch_seed)
    return gw.simulate_batch(spec, **kw)


def genealogy_ok(recs, floors):
    fr = [r for r in recs if r["method"] == "fr_uniform"]
    ess = float(np.median([r["min_ess_frac"] for r in fr]))
    wmax = float(np.median([r["max_wmax"] for r in fr]))
    return ess >= floors["ess_anc_over_N_min"] and wmax <= floors["wmax_max"], ess, wmax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--skip-frozen-bias", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    pre = json.load(open(PREREG))
    base = json.load(open(BASE_PREREG))
    step1 = json.load(open(STEP1))
    sampler, cell = base["sampler"], base["cell"]
    cb = pre["corrected_baseline"]
    h_bias = float(cb["h_bias"])
    assert abs(h_bias - float(step1["h_bias_corrected"])) < 1e-12, "prereg h_bias != step-1 corrected value"
    assert abs(float(cb["h_read_star"]) - float(step1["h_read_star"])) < 1e-12, "prereg h_read* != step-1 value"
    assert pre["cell"] == cell and pre["inits"] == base["inits"]
    assert pre["arms"] == ["abf", "fr_uniform"]
    seeds = list(range(pre["seed_first"], pre["seed_first"] + pre["n_seeds"]))
    taken = set(range(16)) | set(range(100, 132)) | set(range(300, 316))
    assert not (set(seeds) & taken), "reused labels"
    rate = pre["rate"]
    floors = pre["health_floors"]
    if torch.cuda.is_available():
        assert torch.cuda.device_count() == 1, "pin exactly one GPU"
    os.makedirs(a.out, exist_ok=True)

    print("Gateway corrected-baseline CONFIRMATION: abf vs fr_uniform (fresh paired seeds)")
    print(f"  corrected baseline: h_bias {h_bias:g} (legacy 0.07), h_read* {cb['h_read_star']}; "
          f"rate mode {rate['mode']}; seeds {seeds[0]}-{seeds[-1]} x {pre['inits']}")
    print("  no error metric is printed here (prereg)")
    if a.dry_run:
        return
    t_start = time.time()

    # ---------------- Phase A: rate (inherited or re-earned, safety only) ----------------
    if rate["mode"] == "inherited":
        assert abs(h_bias - float(sampler["h"])) < 1e-12, "inherited rate is only licensed at the legacy h_bias"
        gamma = float(rate["gamma"])
        sel = dict(mode="inherited", gamma=gamma)
    else:
        assert rate["mode"] == "ladder" and abs(h_bias - float(sampler["h"])) > 1e-12
        cal_seeds = list(range(rate["calib_seed_first"], rate["calib_seed_first"] + rate["calib_n_seeds"]))
        assert not (set(cal_seeds) & (taken | set(seeds)))
        rows = [(init, sd) for init in pre["inits"] for sd in cal_seeds]
        ladder = []
        for g in rate["ladder"]:
            arms = [gw.ABF, dataclasses.replace(gw.FR_UNIFORM, gamma=float(g))]
            t0 = time.time()
            recs = run_batch(rows, arms, sampler, cell, h_bias, batch_seed=21_000)
            ok, ess, wmax = genealogy_ok(recs, floors)
            ladder.append(dict(gamma=float(g), median_min_ess_frac=ess, median_max_wmax=wmax, passes=bool(ok)))
            print(f"  ladder gamma={g:g}: median min ESS/N {ess:.3f}, median max wmax {wmax:.4f} -> "
                  f"{'pass' if ok else 'FAIL'} ({time.time() - t0:.0f}s)", flush=True)
        passing = [x["gamma"] for x in ladder if x["passes"]]
        assert passing, "no rate passes the genealogy floors; step 2 cannot run (report this, do not lower the floors)"
        gamma = max(passing)
        sel = dict(mode="ladder", gamma=gamma, ladder=ladder, rule="largest passing gamma", floors=floors,
                   calib_seeds=cal_seeds)
        print(f"  RATE SELECTED (safety only): gamma = {gamma:g}", flush=True)
    with open(os.path.join(a.out, "rate_selection.json"), "w") as fh:
        json.dump(sel, fh, indent=2)

    # ---------------- Phase B: production ----------------
    ARMS = [gw.ABF, dataclasses.replace(gw.FR_UNIFORM, gamma=float(gamma))]
    rows = [(init, sd) for init in pre["inits"] for sd in seeds]
    recs_all = []
    for i in range(0, len(rows), a.chunk):
        chunk = rows[i:i + a.chunk]
        t0 = time.time()
        recs = run_batch(chunk, ARMS, sampler, cell, h_bias, batch_seed=41_000 + i,
                         store_profiles=True, store_accumulators=True)
        recs_all.extend(recs)
        fr = [r for r in recs if r["method"] == "fr_uniform"]
        print(f"  chunk {i // a.chunk + 1}: {len(chunk)} rows x 2 arms in {time.time() - t0:.0f}s; "
              f"FR median repl_fraction {np.median([r['repl_fraction'] for r in fr]):.4f}, "
              f"min ESS/N {np.median([r['min_ess_frac'] for r in fr]):.3f}", flush=True)

    fb = None
    if not a.skip_frozen_bias:
        f = pre["frozen_bias"]
        Fp = np.stack([r["Fp_hat"] for r in recs_all])
        cfgs = [build_config(sampler, cell, r["init"], h_bias) for r in recs_all]
        group = [f"{r['init']}|{r['seed']}" for r in recs_all]
        t0 = time.time()
        fb = gw.run_frozen_bias(torch.as_tensor(Fp), cfgs, group=group, n_steps=f["n_steps"],
                                burn_frac=f["burn_frac"], seed=f["seed"])
        for i, r in enumerate(recs_all):
            r["frozen_l2_f_kT"] = float(fb["l2_f_kT"][i])
        print(f"  frozen-bias stage: {len(recs_all)} rows in {time.time() - t0:.0f}s", flush=True)

    keys = ["t", "P_regions", "Q_regions", "l2_f_t", "l2_fp_t", "ess_t", "wmax_t",
            "x_grid", "F_hat", "Fp_hat", "F_ref", "Fp_ref",
            "F_prof_t", "Fp_prof_t", "phat_t", "kl_uniform_t", "Sf_t", "C_t"]
    npz = {k: np.stack([r[k] for r in recs_all]) for k in keys}
    for k in recs_all[0]:
        if k not in keys and k != "config":
            npz[k] = np.array([r[k] for r in recs_all])
    npz["config_json"] = np.array([json.dumps(r["config"], sort_keys=True) for r in recs_all])
    npz["h_bias"] = np.array(h_bias)
    npz["gamma_selected"] = np.array(float(gamma))
    if fb is not None:
        npz["frozen_F_hat"] = fb["F_hat"]
        npz["frozen_p_B"] = fb["p_B"]
    np.savez_compressed(os.path.join(a.out, "raw.npz"), **npz)

    prov = dict(script=os.path.basename(__file__), git_rev=git_rev(), host=socket.gethostname(),
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                device=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
                prereg=os.path.relpath(PREREG, ROOT), step1=os.path.relpath(STEP1, ROOT),
                h_bias=h_bias, h_read_star=cb["h_read_star"], rate_selection=sel, seeds=seeds,
                inits=pre["inits"], wall_seconds=time.time() - t_start, n_runs=len(recs_all))
    with open(os.path.join(a.out, "provenance.json"), "w") as fh:
        json.dump(prov, fh, indent=2, default=float)
    print(f"\ndone: {len(recs_all)} runs in {time.time() - t_start:.0f}s -> {os.path.relpath(a.out, ROOT)}/raw.npz")


if __name__ == "__main__":
    main()
