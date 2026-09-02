#!/usr/bin/env python
"""Transport campaign, Stage A: dose-match horizontal OT to the accepted FR arm by MARGINAL ACTION.

Preregistration: configs/transport_campaign/gateway_horizontal_transport_prereg.json.
One batch on the calibration labels (480-487 x {left, one_right}): abf, fr_uniform (gamma 1.5)
and ot_<alpha> for every alpha_max in the ladder, all sharing initial conditions and Langevin
noise.  Dose J_KL = int_4^40 KL(p_hat_t^x || U) dt from the engine's kl_uniform_t record;
alpha* = the smallest ladder point whose median J_KL is within +-10% of FR's, else the closest
in log ratio; the ladder is extended geometrically (factor 2, <= 4 points) if nothing brackets FR.

BLIND: no F, F' or frozen-bias error of any arm is computed, printed or stored here.  The
saved file keeps only the marginal / conditional / event records and the safety counters.

    CUDA_VISIBLE_DEVICES=3 python -u scripts/run_gateway_horizontal_calibration.py   # GPU 3 ONLY
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
from run_gateway_bandwidth_audit import build_config  # noqa: E402  (same config builder as the audits)

PREREG = os.path.join(ROOT, "configs/transport_campaign/gateway_horizontal_transport_prereg.json")
BASE_PREREG = os.path.join(ROOT, "results/gateway_anchor/CONFIRMATORY_PREREGISTRATION.json")
CORR_PREREG = os.path.join(ROOT, "configs/information_campaign/gateway_corrected_confirmation_prereg.json")
STEP1 = os.path.join(ROOT, "results/information_campaign/gateway_baseline_audit/analysis.json")
OUT_DIR = os.path.join(ROOT, "results", "transport_campaign", "gateway_horizontal", "calibration")
BATCH_SEED = 45_000
T_DOSE_START = 4.0          # end of the ramp (0.1 T)
MATCH_BAND = (0.9, 1.1)
MAX_EXTENSIONS = 4
# the record fields the calibration file is allowed to keep (NO profile / error field)
KEEP = ("t", "kl_uniform_t", "dcond_t", "dcond_nbins_t", "dmove_mean_t", "dmove_p95_t",
        "dmove_max_t", "ot_absdx_t", "alpha_t", "ess_t", "wmax_t", "P_regions", "Q_regions")
SCALARS = ("method", "init", "seed", "gamma", "transport", "alpha", "n_die", "n_clone",
           "n_fr_apply", "n_ot_apply", "repl_fraction", "min_ess_frac", "max_wmax",
           "final_ess", "final_wmax", "T_hit", "T_est")
TAKEN = set(range(16)) | set(range(100, 132)) | set(range(300, 316)) | set(range(400, 416))


def git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def dose(rec, t_start=T_DOSE_START):
    """J_KL = int_{t_start}^{T} KL(p_hat_t || U) dt, trapezoid over the saves with t >= t_start."""
    t, kl = np.asarray(rec["t"], float), np.asarray(rec["kl_uniform_t"], float)
    m = t >= t_start - 1e-12
    return float(np.trapezoid(kl[m], t[m]))


def select_alpha(ladder, ratio):
    """The prereg rule.  ``ratio[alpha]`` = median J_KL(ot_alpha) / median J_KL(fr)."""
    inside = [a for a in ladder if MATCH_BAND[0] <= ratio[a] <= MATCH_BAND[1]]
    if inside:
        return min(inside), "smallest alpha with ratio in [0.9, 1.1]"
    return min(ladder, key=lambda a: abs(np.log(ratio[a]))), "argmin |log ratio| (no point inside the band)"


def brackets(ladder, ratio):
    r = [ratio[a] for a in ladder]
    if all(v > MATCH_BAND[1] for v in r):
        return "stronger"      # even the strongest point flattens less than FR
    if all(v < MATCH_BAND[0] for v in r):
        return "weaker"        # even the weakest flattens more
    return "ok"


def run_ladder(rows, ladder, sampler, cell, h_bias, gamma):
    arms = [gw.ABF, dataclasses.replace(gw.FR_UNIFORM, gamma=float(gamma))] + \
           [gw.horizontal_ot(a) for a in ladder]
    cfgs = [build_config(sampler, cell, init, h_bias) for init, _ in rows]
    spec = gw.BatchSpec(configs=cfgs, seeds=[sd for _, sd in rows], methods=arms, batch_seed=BATCH_SEED)
    return gw.simulate_batch(spec, store_profiles=True, store_conditional=True), arms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    pre = json.load(open(PREREG))
    base = json.load(open(BASE_PREREG))
    corr = json.load(open(CORR_PREREG))
    step1 = json.load(open(STEP1))
    sampler, cell = base["sampler"], base["cell"]
    assert pre["cell"] == cell and pre["inits"] == base["inits"]
    h_bias = float(corr["corrected_baseline"]["h_bias"])
    assert abs(h_bias - float(step1["h_bias_corrected"])) < 1e-12 and abs(h_bias - float(sampler["h"])) < 1e-12
    gamma = float(corr["rate"]["gamma"])
    st = pre["stage_A_calibration"]
    seeds = list(range(st["seeds"]["first"], st["seeds"]["first"] + st["seeds"]["count"]))
    prod = pre["stage_B_production"]["seeds"]
    prod_seeds = set(range(prod["first"], prod["first"] + prod["count"]))
    assert not (set(seeds) & (TAKEN | prod_seeds)), "calibration labels collide with a taken or production label"
    ladder = [float(x) for x in st["ladder"]]
    rows = [(init, sd) for init in pre["inits"] for sd in seeds]
    if torch.cuda.is_available():
        assert torch.cuda.device_count() == 1, "pin exactly one GPU"
    os.makedirs(a.out, exist_ok=True)
    print("Transport campaign, Stage A: alpha ladder dose-matched to fr_uniform by J_KL (BLIND to every error metric)")
    print(f"  rows {len(rows)} (seeds {seeds[0]}-{seeds[-1]} x {pre['inits']}); h_bias {h_bias:g}; gamma {gamma:g}; "
          f"ladder {ladder}; batch_seed {BATCH_SEED}")
    if a.dry_run:
        return
    t_start = time.time()

    history = []
    for ext in range(MAX_EXTENSIONS + 1):
        t0 = time.time()
        recs, arms = run_ladder(rows, ladder, sampler, cell, h_bias, gamma)
        J = {m.name: np.array([dose(r) for r in recs if r["method"] == m.name]) for m in arms}
        med = {k: float(np.median(v)) for k, v in J.items()}
        ratio = {al: med[f"ot_{al:g}"] / med["fr_uniform"] for al in ladder}
        br = brackets(ladder, ratio)
        history.append(dict(ladder=list(ladder), median_J=med, ratio={f"{al:g}": ratio[al] for al in ladder},
                            bracket=br, wall_seconds=time.time() - t0))
        print(f"  batch {ext + 1}: {len(recs)} rows in {time.time() - t0:.0f}s; median J_KL: abf {med['abf']:.4f}, "
              f"fr_uniform {med['fr_uniform']:.4f}; " +
              ", ".join(f"ot_{al:g} {med[f'ot_{al:g}']:.4f} (x{ratio[al]:.3f})" for al in ladder), flush=True)
        if br == "ok":
            break
        new = (max(ladder) * 2.0) if br == "stronger" else (min(ladder) / 2.0)
        if new > 1.0:
            print("  ladder cannot exceed alpha = 1; stopping the extension", flush=True)
            break
        print(f"  no ladder point brackets FR ({br}); extending the ladder with alpha = {new:g} and re-running", flush=True)
        ladder = sorted(ladder + [new])

    alpha_star, rule = select_alpha(ladder, ratio)
    print(f"  ALPHA SELECTED: alpha* = {alpha_star:g}  [{rule}]  ratio {ratio[alpha_star]:.3f}", flush=True)

    # ---- descriptive mechanism numbers (allowed: marginal / conditional / event records only) ----
    t_axis = np.asarray(recs[0]["t"], float)
    desc = {}
    for m in arms:
        rr = [r for r in recs if r["method"] == m.name]
        dc = np.array([np.trapezoid(np.nan_to_num(np.asarray(r["dcond_t"], float)), t_axis) for r in rr])
        d = dict(median_J_KL=med[m.name], median_int_Dcond=float(np.median(dc)))
        if m.transport == "horizontal_ot":
            d.update(median_mean_absdx=float(np.median([np.mean(r["ot_absdx_t"][1:]) for r in rr])),
                     median_max_Dmove=float(np.median([np.max(r["dmove_max_t"]) for r in rr])),
                     median_mean_Dmove=float(np.median([np.mean(r["dmove_mean_t"][1:]) for r in rr])))
        if m.use_fr:
            d.update(median_repl_fraction=float(np.median([r["repl_fraction"] for r in rr])),
                     median_min_ess_frac=float(np.median([r["min_ess_frac"] for r in rr])),
                     median_max_wmax=float(np.median([r["max_wmax"] for r in rr])))
        desc[m.name] = d
        print(f"    {m.name:>10}: J_KL {d['median_J_KL']:.4f}  int D_cond {d['median_int_Dcond']:.4f}" +
              (f"  |dx|/event {d['median_mean_absdx']:.2e}  D_move mean {d['median_mean_Dmove']:.3e} max {d['median_max_Dmove']:.3e}"
               if m.transport == "horizontal_ot" else "") +
              (f"  repl {d['median_repl_fraction']:.4f}  min ESS/N {d['median_min_ess_frac']:.3f}" if m.use_fr else ""))

    sel = dict(prereg=os.path.relpath(PREREG, ROOT), rule=rule, alpha_star=float(alpha_star),
               ratio_at_alpha_star=float(ratio[alpha_star]), match_band=list(MATCH_BAND),
               dose="J_KL = int_4^40 KL(p_hat_t || U) dt (kl_uniform_t, eta 0.10 KDE), median over rows",
               ladder_final=list(ladder), median_J_KL=med, ratio={f"{al:g}": ratio[al] for al in ladder},
               history=history, descriptive=desc, seeds=seeds, inits=pre["inits"], n_rows=len(rows),
               batch_seed=BATCH_SEED, gamma=gamma, h_bias=h_bias, git_rev=git_rev(),
               wall_seconds=time.time() - t_start, blind="no error metric computed, printed or stored")
    with open(os.path.join(a.out, "alpha_selection.json"), "w") as fh:
        json.dump(sel, fh, indent=2, default=float)

    npz = {k: np.stack([np.asarray(r[k]) for r in recs]) for k in KEEP}
    for k in SCALARS:
        npz[k] = np.array([r[k] for r in recs])
    np.savez_compressed(os.path.join(a.out, "raw.npz"), **npz)
    prov = dict(script=os.path.basename(__file__), git_rev=git_rev(), host=socket.gethostname(),
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                device=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
                prereg=os.path.relpath(PREREG, ROOT), seeds=seeds, inits=pre["inits"], ladder=ladder,
                arms=[m.name for m in arms], batch_seed=BATCH_SEED, wall_seconds=time.time() - t_start,
                kept_fields=list(KEEP) + list(SCALARS))
    with open(os.path.join(a.out, "provenance.json"), "w") as fh:
        json.dump(prov, fh, indent=2, default=float)
    print(f"\ndone in {time.time() - t_start:.0f}s -> {os.path.relpath(a.out, ROOT)}/{{raw.npz, alpha_selection.json}}")


if __name__ == "__main__":
    main()
