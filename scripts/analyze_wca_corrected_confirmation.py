#!/usr/bin/env python
"""Analyzer for the WCA corrected-baseline CONFIRMATION -- written and committed BEFORE the data.

Prereg: configs/information_campaign/wca_corrected_confirmation_prereg.json.

Primary: paired Delta I_F (fr_uniform vs abf) at h_read* = 0.0125 (taken from step 1, not
re-derived); co-primary safety: Delta e_F(T) at h_read* + the Case IX genealogy floors.
Secondary: the same at the legacy read-out 0.025 (direct replication of -21.91%), the full
read-out ladder for both arms, time-to-accuracy, round trips.  Outcome R1-R4 per the prereg.

The read-out machinery (`readouts`) is imported from analyze_wca_bandwidth_audit.py, so the
0.0125 read-out here is bit-for-bit the one step 1 selected on.

    python scripts/analyze_wca_corrected_confirmation.py
    python scripts/analyze_wca_corrected_confirmation.py --raw-dir <audit raw> --stage bandwidth_audit \
        --names abf_hb0.025,abf_hb0.0125            # machinery smoke on step-1 data (NOT a result)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS)
from analyze_uniform_lta import BOOT_SEED, boot_median          # noqa: E402  (10k resamples)
from analyze_uniform_wca import tau, PERSIST, FRACTIONS, ESS_FLOOR, WMAX_CAP, RULE   # noqa: E402
from analyze_wca_bandwidth_audit import readouts, Z_LO, Z_HI, LEGACY_KEY        # noqa: E402

ROOT = os.path.join(SCRIPTS, "..")
PREREG = os.path.join(ROOT, "configs/information_campaign/wca_corrected_confirmation_prereg.json")
DEFAULT_RAW = os.path.join(ROOT, "results/information_campaign/wca_corrected_confirmation/confirmation/raw")
CASEIX_CI = (-26.30, -19.04)      # Case IX legacy Delta I_F CI95 (results/uniform_campaign/wca/summary.json)


def load_runs(raw_dir, stage, names):
    runs = {n: {} for n in names}
    for f in sorted(glob.glob(os.path.join(raw_dir, f"{stage}__*__*.npz"))):
        d = np.load(f, allow_pickle=True)
        name = str(d["name"])
        if name in runs:
            runs[name][int(d["seed"])] = {k: d[k] for k in d.files}
    return runs


def paired(x_arm, x_ref, seed_offset=0):
    d = 100.0 * (x_arm - x_ref) / x_ref
    lo, hi = boot_median(d, BOOT_SEED + seed_offset)
    return dict(median=float(np.median(d)), ci95=[lo, hi], wins=int((d < 0).sum()), n=int(len(d)),
                per_seed=[float(v) for v in d])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=DEFAULT_RAW)
    ap.add_argument("--stage", default="confirmation")
    ap.add_argument("--names", default="abf,fr_uniform", help="reference arm, FR arm")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    pre = json.load(open(PREREG))
    h_star = f"{pre['corrected_baseline']['h_read_star']:g}"
    ref_name, fr_name = a.names.split(",")
    smoke = (ref_name, fr_name) != ("abf", "fr_uniform")
    if smoke:
        print("*** MACHINERY SMOKE on non-confirmation arms -- the numbers below are NOT a result ***")

    runs = load_runs(a.raw_dir, a.stage, [ref_name, fr_name])
    seeds = sorted(set(runs[ref_name]) & set(runs[fr_name]))
    if not seeds:
        sys.exit(f"no complete pairs under {a.raw_dir}")
    print(f"pairs: {len(seeds)} seeds {seeds[0]}-{seeds[-1]}; arms {ref_name} vs {fr_name}")
    labels = {str(runs[m][s]["reference_label"]) for m in (ref_name, fr_name) for s in seeds}
    assert len(labels) == 1 and "v2" in next(iter(labels)), f"mixed references: {labels}"
    hb = {float(runs[m][s]["abf_bandwidth_online"]) for m in (ref_name, fr_name) for s in seeds}
    if not smoke:
        assert hb == {0.025}, f"online bandwidth is not the legacy in every run: {hb}"

    any_run = runs[ref_name][seeds[0]]
    grid = np.asarray(any_run["grid"], dtype=float)
    ref_F = np.asarray(any_run["reference_free_energy"], dtype=float)
    t = np.asarray(any_run["profile_times"], dtype=float)
    mask = (grid >= Z_LO) & (grid <= Z_HI)
    sigma = float(any_run.get("abf_smooth_sigma", 0.5))

    # self-check: the legacy read-out recomputed offline reproduces the engine's saved l2_f
    ro = {m: {s: readouts(runs[m][s], grid, mask, ref_F, sigma) for s in seeds} for m in (ref_name, fr_name)}
    dev = max(abs(float(ro[m][s][LEGACY_KEY][-1]) - float(runs[m][s]["l2_f"])) for m in ro for s in seeds)
    print(f"self-check: recomputed legacy e_F(T) vs saved l2_f, max |dev| {dev:.2e}")
    assert dev < 1e-5
    ladder = list(ro[ref_name][seeds[0]].keys())
    assert h_star in ladder, (h_star, ladder)

    # ---- every read-out, both statistics ----
    res = {}
    print(f"\n  {'read-out':>10} {'ABF e_F(T)':>10} {'d I_F':>8} {'CI95':>20} {'wins':>5} {'d e_F(T)':>9} {'CI95':>20}")
    for k, lab in enumerate(ladder):
        I = {m: np.array([np.trapezoid(ro[m][s][lab], t) for s in seeds]) for m in ro}
        fin = {m: np.array([ro[m][s][lab][-1] for s in seeds]) for m in ro}
        res[lab] = dict(d_int=paired(I[fr_name], I[ref_name], 10 * k),
                        d_fin=paired(fin[fr_name], fin[ref_name], 10 * k + 1),
                        abf_eF_T_median=float(np.median(fin[ref_name])),
                        abf_IF_median=float(np.median(I[ref_name])),
                        fr_eF_T_median=float(np.median(fin[fr_name])))
        di, df = res[lab]["d_int"], res[lab]["d_fin"]
        tag = "  <- PRIMARY (h_read*)" if lab == h_star else ("  <- legacy (replication)" if lab == LEGACY_KEY else "")
        print(f"  {lab:>10} {res[lab]['abf_eF_T_median']:10.5f} {di['median']:+8.2f} [{di['ci95'][0]:+8.2f},{di['ci95'][1]:+8.2f}]"
              f" {di['wins']:3d}/{di['n']} {df['median']:+9.2f} [{df['ci95'][0]:+8.2f},{df['ci95'][1]:+8.2f}]{tag}")

    # ---- genealogy (FR arm), round trips, events ----
    N = int(pre["n_replicas"])
    ess_min = [float(runs[fr_name][s]["min_ancestor_ess"]) / N for s in seeds]
    wmax = [float(runs[fr_name][s]["max_ancestor_frac_over_time"]) for s in seeds]
    health_ok = (min(ess_min) >= ESS_FLOOR and max(wmax) <= WMAX_CAP) if not smoke else True
    rt = {m: float(np.median([float(runs[m][s]["n_round_trips"]) for s in seeds])) for m in ro}
    ev = float(np.median([float(runs[fr_name][s]["total_replacement_events"]) for s in seeds]))
    print(f"\n  genealogy ({fr_name}): min ESS/N {min(ess_min) if not smoke else float('nan'):.3f} (floor {ESS_FLOOR}), "
          f"max wmax {max(wmax) if not smoke else float('nan'):.4f} (cap {WMAX_CAP}) -> ok={health_ok}")
    print(f"  round trips: {ref_name} {rt[ref_name]:.0f} vs {fr_name} {rt[fr_name]:.0f}; replacement events median {ev:.0f}")
    abf_corr = 100 * (res[h_star]["abf_eF_T_median"] / res[LEGACY_KEY]["abf_eF_T_median"] - 1)
    print(f"  ABF's own e_F(T): {LEGACY_KEY} -> {h_star} = {abf_corr:+.2f}% (step 1 measured about -1.7%)")

    # ---- time-to-accuracy at h_read* (median curves) ----
    curves = {m: np.median([ro[m][s][h_star] for s in seeds], axis=0) for m in ro}
    e0 = float(curves[ref_name][0])
    eps_list = {f"e0/{int(1 / f)}": e0 * f for f in FRACTIONS}
    eps_list["abf_final"] = float(curves[ref_name][-1])
    speed = {}
    for nm, eps in eps_list.items():
        ta, tu = tau(t, curves[ref_name], eps, PERSIST), tau(t, curves[fr_name], eps, PERSIST)
        speed[nm] = dict(eps=eps, tau_abf=ta, tau_fr=tu,
                         speedup=(ta / tu if np.isfinite(ta) and np.isfinite(tu) and tu > 0 else None))
        sp = f"{speed[nm]['speedup']:.2f}x" if speed[nm]["speedup"] else "censored"
        print(f"  tau[{nm}] @ h_read*: abf {ta:.0f} vs fr {tu:.0f} -> {sp}")

    # ---- verdicts (frozen rule) and outcome ----
    def verdict(lab):
        di, df = res[lab]["d_int"], res[lab]["d_fin"]
        accel = di["median"] <= RULE["median_max"] and di["ci95"][1] < RULE["ci_upper_max"]
        safe = accel and df["median"] <= RULE["final_margin"] and health_ok
        neutral = abs(di["median"]) < abs(RULE["median_max"]) and df["median"] <= RULE["final_margin"]
        return ("SAFE_ACCELERATOR" if safe else "ACCELERATION_POSITIVE" if accel
                else "NEUTRAL" if neutral else "NEGATIVE_OR_UNSAFE")
    v_star, v_leg = verdict(h_star), verdict(LEGACY_KEY)
    leg_ci = res[LEGACY_KEY]["d_int"]["ci95"]
    overlaps_caseix = not (leg_ci[1] < CASEIX_CI[0] or leg_ci[0] > CASEIX_CI[1])
    di = res[h_star]["d_int"]
    if not overlaps_caseix:
        outcome = "R3_replication_failure"
    elif v_star == "SAFE_ACCELERATOR":
        outcome = "R1_replicated"
    elif di["ci95"][1] < 0 or v_leg == "SAFE_ACCELERATOR":
        outcome = "R2_attenuated"
    else:
        outcome = "R4_null_or_unsafe"
    print(f"\n  verdict at h_read* {h_star}: {v_star};  at legacy {LEGACY_KEY}: {v_leg}"
          f"  (legacy CI {leg_ci[0]:+.2f}..{leg_ci[1]:+.2f} vs Case IX {CASEIX_CI[0]:+.2f}..{CASEIX_CI[1]:+.2f}: "
          f"{'overlap' if overlaps_caseix else 'DISJOINT'})")
    print(f"  OUTCOME: {outcome}{'   [SMOKE -- not a result]' if smoke else ''}")
    if len(seeds) < pre["n_seeds"]:
        print(f"  NOTE: {len(seeds)} of {pre['n_seeds']} preregistered seeds present -- not confirmatory")

    if smoke:
        return
    out_dir = a.out_dir or os.path.dirname(os.path.dirname(a.raw_dir))
    summary = dict(prereg=os.path.relpath(PREREG, ROOT), raw_dir=os.path.relpath(a.raw_dir, ROOT),
                   n_pairs=len(seeds), seeds=seeds, reference_label=next(iter(labels)),
                   h_read_star=h_star, per_readout=res, health=dict(min_ess_frac=min(ess_min), max_wmax=max(wmax),
                   ok=bool(health_ok), floors=dict(ess=ESS_FLOOR, wmax=WMAX_CAP)),
                   round_trips=rt, replacement_events_median=ev, abf_readout_correction_pct=abf_corr,
                   time_to_accuracy=speed, verdict_h_read_star=v_star, verdict_legacy=v_leg,
                   legacy_ci_overlaps_caseix=bool(overlaps_caseix), caseix_ci=list(CASEIX_CI), outcome=outcome)
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    with open(os.path.join(out_dir, "comparison.csv"), "w") as fh:
        fh.write("seed," + ",".join(f"d_int_{lab},d_fin_{lab}" for lab in ladder) + ",min_ess_frac_fr,wmax_fr\n")
        for i, s in enumerate(seeds):
            fh.write(f"{s}," + ",".join(f"{res[lab]['d_int']['per_seed'][i]:.3f},{res[lab]['d_fin']['per_seed'][i]:.3f}"
                                        for lab in ladder) + f",{ess_min[i]:.4f},{wmax[i]:.4f}\n")
    print(f"  wrote {os.path.relpath(out_dir, ROOT)}/summary.json, comparison.csv")


if __name__ == "__main__":
    main()
