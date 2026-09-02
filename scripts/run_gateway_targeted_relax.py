#!/usr/bin/env python
"""Targeted-relaxation campaign runner: D0 (estimator validation), D1 (cost ladder), D2 (confirmatory).

Preregistration: configs/transport_campaign/gateway_targeted_relax_prereg.json.
Prints wall time and safety counters only -- no error metric.

    CUDA_VISIBLE_DEVICES=3 python -u scripts/run_gateway_targeted_relax.py --stage D0   # GPU 3 ONLY
    CUDA_VISIBLE_DEVICES=3 python -u scripts/run_gateway_targeted_relax.py --stage D1
    CUDA_VISIBLE_DEVICES=3 python -u scripts/run_gateway_targeted_relax.py --stage D2
"""
from __future__ import annotations

import argparse
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
from run_gateway_horizontal_calibration import BASE_PREREG, CORR_PREREG, git_rev  # noqa: E402
from run_gateway_fibre_relax import allocators, save as save_npz, KEYS as KEYS1, TAKEN_ALL as TAKEN_PREV  # noqa: E402

PREREG = os.path.join(ROOT, "configs/transport_campaign/gateway_targeted_relax_prereg.json")
ALPHA_SEL = os.path.join(ROOT, "results/transport_campaign/gateway_horizontal/calibration_refine/alpha_selection.json")
CAMPAIGN = os.path.join(ROOT, "results", "transport_campaign", "gateway_horizontal")
DIRS = {s: os.path.join(CAMPAIGN, f"targeted_{s}") for s in ("D0", "D1", "D2")}
BATCH_SEED = dict(D0=85_000, D1=91_000, D2=101_000)
TAKEN_ALL = TAKEN_PREV | set(range(580, 588)) | set(range(800, 832)) | set(range(1200, 1208))
KEYS = KEYS1 + ["Sf2_t", "targ_flank_frac_t", "targ_active_frac_t", "targ_cmean_t", "ot_flank_dx_frac_t"]


def arms_for(stage, al, ladder, rho_star=None):
    A, F, T = al["A"], al["F"], al["T"]
    if stage == "D0":
        return [A]
    if stage == "D1":
        arms = [A, F, T]
        for rho in ladder:
            arms += [gw.with_targeted(A, rho), gw.with_targeted(F, rho), gw.with_targeted(T, rho), gw.with_targeted(T, rho, "v_dx")]
        return arms + [gw.with_relax(A, 0.5), gw.with_relax(T, 0.5)]
    return [A, F, T, gw.with_targeted(A, rho_star), gw.with_targeted(F, rho_star), gw.with_targeted(T, rho_star),
            gw.with_targeted(T, rho_star, "v_dx")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["D0", "D1", "D2"], required=True)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--skip-frozen-bias", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    pre = json.load(open(PREREG)); base = json.load(open(BASE_PREREG)); corr = json.load(open(CORR_PREREG))
    sampler, cell = base["sampler"], base["cell"]
    h_bias = float(corr["corrected_baseline"]["h_bias"]); gamma = float(corr["rate"]["gamma"])
    alpha = float(json.load(open(ALPHA_SEL))["alpha_star"]); assert abs(alpha - 0.00325) < 1e-12
    ladder = [float(r) for r in pre["stage_D1_cost_ladder"]["ladder_rho"]]
    al = allocators(gamma, alpha)
    out_dir = DIRS[a.stage]
    extra = dict(h_bias=h_bias, alpha_star=alpha, gamma_selected=gamma)
    if a.stage == "D0":
        st = pre["stage_D0_validation"]
        if os.path.exists(os.path.join(DIRS["D1"], "raw.npz")):
            print("WARNING: D1 already exists; D0 is a validation stage and must precede it")
    elif a.stage == "D1":
        st = pre["stage_D1_cost_ladder"]
        d0 = json.load(open(os.path.join(DIRS["D0"], "analysis.json")))
        assert d0["passed"], "D0 did not pass: fix the estimator before D1 (prereg)"
        extra["ladder_rho"] = ladder
    else:
        st = pre["stage_D2_confirmatory"]
        from analyze_gateway_targeted_relax import select_rho
        sel = json.load(open(os.path.join(DIRS["D1"], "rho_selection.json")))
        rho_star, gate = select_rho(sel["ladder_rho"], sel["dI_T_vs_A_median"], sel["compute_ratio_T"])
        assert abs(rho_star - float(sel["rho_star"])) < 1e-12 and gate == sel["gate_D1"]
        extra.update(rho_star=rho_star, gate_D1=gate)
    ARMS = arms_for(a.stage, al, ladder, extra.get("rho_star"))
    gw.assert_no_oracle_leakage(ARMS)
    seeds = list(range(st["seeds"]["first"], st["seeds"]["first"] + st["seeds"]["count"]))
    if a.stage != "D0":
        assert not (set(seeds) & (TAKEN_ALL - (set(range(1200, 1208)) if a.stage == "D1" else set()))), "label collision"
    rows = [(init, sd) for init in ["left", "one_right"] for sd in seeds]
    if torch.cuda.is_available():
        assert torch.cuda.device_count() == 1, "pin exactly one GPU"
    os.makedirs(out_dir, exist_ok=True)
    print(f"Targeted-relaxation campaign, Stage {a.stage}: {len(ARMS)} arms: " + ", ".join(m.name for m in ARMS))
    print(f"  h_bias {h_bias:g}, gamma {gamma:g}, alpha** {alpha:g}; {len(rows)} rows (seeds {seeds[0]}-{seeds[-1]} x 2 inits); "
          f"chunk {a.chunk}; batch seeds {BATCH_SEED[a.stage]}+; no error metric printed (prereg)")
    if a.stage == "D2":
        print(f"  rho* {extra['rho_star']:g} (gate_D1 {extra['gate_D1']}), re-derived from D1 by the frozen rule")
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
            line = f"     {m.name:>26}: "
            if m.use_fr:
                line += f"repl {np.median([r['repl_fraction'] for r in rr]):.4f}, ESS/N {np.median([r['min_ess_frac'] for r in rr]):.3f}; "
            if m.transport == "horizontal_ot":
                line += f"|dx| {np.median([np.mean(r['ot_absdx_t'][1:]) for r in rr]):.1e}, flank dx frac {np.median([np.mean(r['ot_flank_dx_frac_t'][1:]) for r in rr]):.2f}; "
            if m.refresh in ("ou", "targeted"):
                line += f"cost ratio {np.median([r['fibre_cost_ratio'] for r in rr]):.2f}; "
            if m.refresh == "targeted":
                line += (f"flank budget frac {np.median([np.mean(r['targ_flank_frac_t'][1:]) for r in rr]):.2f}, "
                         f"active {np.median([np.mean(r['targ_active_frac_t'][1:]) for r in rr]):.3f}, mean c {np.median([np.mean(r['targ_cmean_t'][1:]) for r in rr]):.2f}; ")
            line += f"D_cond {np.median([np.nanmean(r['dcond_t']) for r in rr]):.3e}"
            print(line, flush=True)
    fb = None
    if not a.skip_frozen_bias and a.stage != "D0":
        f = base["frozen_bias"]
        Fp = np.stack([r["Fp_hat"] for r in recs_all])
        cfgs = [build_config(sampler, cell, r["init"], h_bias) for r in recs_all]
        group = [f"{r['init']}|{r['seed']}" for r in recs_all]
        t0 = time.time()
        fb = gw.run_frozen_bias(torch.as_tensor(Fp), cfgs, group=group, n_steps=f["n_steps"], burn_frac=f["burn_frac"], seed=f["seed"])
        for i, r in enumerate(recs_all):
            r["frozen_l2_f_kT"] = float(fb["l2_f_kT"][i])
        print(f"  frozen-bias stage: {len(recs_all)} rows in {time.time() - t0:.0f}s", flush=True)
    keys = [k for k in KEYS if k in recs_all[0]]
    npz = {k: np.stack([np.asarray(r[k]) for r in recs_all]) for k in keys}
    for k in recs_all[0]:
        if k not in keys and k != "config":
            npz[k] = np.array([r[k] for r in recs_all])
    npz["config_json"] = np.array([json.dumps(r["config"], sort_keys=True) for r in recs_all])
    for k, v in extra.items():
        npz[k] = np.array(v)
    if fb is not None:
        npz["frozen_F_hat"] = fb["F_hat"]; npz["frozen_p_B"] = fb["p_B"]
    np.savez_compressed(os.path.join(out_dir, "raw.npz"), **npz)
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
