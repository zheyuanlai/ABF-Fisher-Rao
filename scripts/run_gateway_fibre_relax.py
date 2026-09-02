#!/usr/bin/env python
"""Fibre-relaxation campaign runner: Stage C1 (recovery block) and Stage C2 (confirmatory block).

Preregistration: configs/transport_campaign/gateway_fibre_relax_prereg.json.
C1: 16 rows x 24 arms (A/F/T/P x {0, 0.5, 1, 2, 5, inf}) in one batch.  C2: 64 rows x 7 arms
(A_0, F_0, T_0, A_c*, F_c*, T_c*, P_c*), c* and h_read** read from relax_C1/c_selection.json and
re-derived from the frozen rules before anything runs.  Prints wall time and safety counters only.

    CUDA_VISIBLE_DEVICES=3 python -u scripts/run_gateway_fibre_relax.py --stage C1   # GPU 3 ONLY
    CUDA_VISIBLE_DEVICES=3 python -u scripts/run_gateway_fibre_relax.py --stage C2
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
from run_gateway_horizontal_calibration import TAKEN, BASE_PREREG, CORR_PREREG, git_rev  # noqa: E402
from run_gateway_horizontal_transport import KEYS as KEYS0  # noqa: E402

PREREG = os.path.join(ROOT, "configs/transport_campaign/gateway_fibre_relax_prereg.json")
ALPHA_SEL = os.path.join(ROOT, "results/transport_campaign/gateway_horizontal/calibration_refine/alpha_selection.json")
CAMPAIGN = os.path.join(ROOT, "results", "transport_campaign", "gateway_horizontal")
C1_DIR, C2_DIR = os.path.join(CAMPAIGN, "relax_C1"), os.path.join(CAMPAIGN, "relax_C2")
BATCH_SEED = dict(C1=71_000, C2=81_000)
TAKEN_ALL = TAKEN | set(range(480, 488)) | set(range(500, 532)) | set(range(540, 572))
KEYS = KEYS0 + ["fibre_steps_t", "ot_tau_move_t"]
INF = float("inf")


def allocators(gamma, alpha):
    return dict(A=gw.ABF, F=dataclasses.replace(gw.FR_UNIFORM, gamma=gamma),
                T=gw.horizontal_ot(alpha, name="ot_exact"), P=gw.horizontal_ot(1.0, name="ot_full"))


def arm(base, c):
    return base if c == 0 else gw.with_relax(base, c)


def save(out_dir, recs_all, extra, fb):
    npz = {k: np.stack([np.asarray(r[k]) for r in recs_all]) for k in KEYS}
    for k in recs_all[0]:
        if k not in KEYS and k != "config":
            npz[k] = np.array([r[k] for r in recs_all])
    npz["config_json"] = np.array([json.dumps(r["config"], sort_keys=True) for r in recs_all])
    for k, v in extra.items():
        npz[k] = np.array(v)
    if fb is not None:
        npz["frozen_F_hat"] = fb["F_hat"]; npz["frozen_p_B"] = fb["p_B"]
    np.savez_compressed(os.path.join(out_dir, "raw.npz"), **npz)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["C1", "C2"], required=True)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--skip-frozen-bias", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    pre = json.load(open(PREREG)); base = json.load(open(BASE_PREREG)); corr = json.load(open(CORR_PREREG))
    sampler, cell = base["sampler"], base["cell"]
    h_bias = float(corr["corrected_baseline"]["h_bias"]); gamma = float(corr["rate"]["gamma"])
    alpha = float(json.load(open(ALPHA_SEL))["alpha_star"]); assert abs(alpha - 0.00325) < 1e-12
    ladder = [float(c) for c in pre["ladder_c"]]
    al = allocators(gamma, alpha)
    if a.stage == "C1":
        st = pre["stage_C1_recovery"]; out_dir = C1_DIR
        cs = [0.0] + ladder + [INF]
        ARMS = [arm(al[k], c) for k in ("A", "F", "T", "P") for c in cs]
        extra = dict(alpha_star=alpha, gamma_selected=gamma, ladder_c=ladder)
    else:
        st = pre["stage_C2_confirmatory"]; out_dir = C2_DIR
        sel = json.load(open(os.path.join(C1_DIR, "c_selection.json")))
        from analyze_gateway_fibre_relax import select_c, select_h   # frozen rules, re-derived
        c_star = select_c(sel["ladder_c"], sel["R_T_median"], sel["dI_T_vs_A_median"])
        h_star2 = select_h(sel["readout_ladder"], sel["readout_eF_T_median_per_arm"])
        assert abs(c_star - float(sel["c_star"])) < 1e-12 and abs(h_star2 - float(sel["h_read_star2"])) < 1e-12
        ARMS = [al["A"], al["F"], al["T"], arm(al["A"], c_star), arm(al["F"], c_star), arm(al["T"], c_star), arm(al["P"], c_star)]
        extra = dict(alpha_star=alpha, gamma_selected=gamma, c_star=c_star, h_read_star2=h_star2)
    gw.assert_no_oracle_leakage(ARMS)
    seeds = list(range(st["seeds"]["first"], st["seeds"]["first"] + st["seeds"]["count"]))
    assert not (set(seeds) & TAKEN_ALL | (set(range(580, 588)) if a.stage == "C2" else set()) & set(seeds))
    rows = [(init, sd) for init in ["left", "one_right"] for sd in seeds]
    if torch.cuda.is_available():
        assert torch.cuda.device_count() == 1, "pin exactly one GPU"
    os.makedirs(out_dir, exist_ok=True)
    print(f"Fibre-relaxation campaign, Stage {a.stage}: {len(ARMS)} arms: " + ", ".join(m.name for m in ARMS))
    print(f"  h_bias {h_bias:g}, gamma {gamma:g}, alpha** {alpha:g}; {len(rows)} rows (seeds {seeds[0]}-{seeds[-1]} x 2 inits); "
          f"chunk {a.chunk}; batch seeds {BATCH_SEED[a.stage]}+; no error metric printed (prereg)")
    if a.stage == "C2":
        print(f"  c* {extra['c_star']:g}, h_read** {extra['h_read_star2']:g} (re-derived from C1 by the frozen rules)")
    if a.dry_run:
        return
    t_start = time.time(); recs_all = []
    for i in range(0, len(rows), a.chunk):
        chunk = rows[i:i + a.chunk]
        cfgs = [build_config(sampler, cell, init, h_bias) for init, _ in chunk]
        spec = gw.BatchSpec(configs=cfgs, seeds=[sd for _, sd in chunk], methods=ARMS, batch_seed=BATCH_SEED[a.stage] + i)
        t0 = time.time()
        recs = gw.simulate_batch(spec, store_profiles=True, store_accumulators=True, store_conditional=True)
        recs_all.extend(recs)
        print(f"  chunk {i // a.chunk + 1}: {len(chunk)} rows x {len(ARMS)} arms in {time.time() - t0:.0f}s", flush=True)
        for m in ARMS:
            rr = [r for r in recs if r["method"] == m.name]
            line = f"     {m.name:>24}: "
            if m.use_fr:
                line += f"repl {np.median([r['repl_fraction'] for r in rr]):.4f}, min ESS/N {np.median([r['min_ess_frac'] for r in rr]):.3f}; "
            if m.transport == "horizontal_ot":
                line += f"|dx| {np.median([np.mean(r['ot_absdx_t'][1:]) for r in rr]):.2e}, tau_move {np.median([np.mean(r['ot_tau_move_t'][1:]) for r in rr]):.3f}; "
            if m.refresh == "ou":
                line += f"fibre cost ratio {np.median([r['fibre_cost_ratio'] for r in rr]):.1f}; "
            line += f"D_cond {np.median([np.nanmean(r['dcond_t']) for r in rr]):.3e}"
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
    save(out_dir, recs_all, dict(h_bias=h_bias, **extra), fb)
    prov = dict(script=os.path.basename(__file__), stage=a.stage, git_rev=git_rev(), host=socket.gethostname(),
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                device=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
                prereg=os.path.relpath(PREREG, ROOT), arms=[m.name for m in ARMS], seeds=seeds, chunk=a.chunk,
                batch_seed=BATCH_SEED[a.stage], wall_seconds=time.time() - t_start, n_runs=len(recs_all), **extra)
    with open(os.path.join(out_dir, "provenance.json"), "w") as fh:
        json.dump(prov, fh, indent=2, default=float)
    print(f"\ndone: {len(recs_all)} runs in {time.time() - t_start:.0f}s -> {os.path.relpath(out_dir, ROOT)}/raw.npz")


if __name__ == "__main__":
    main()
