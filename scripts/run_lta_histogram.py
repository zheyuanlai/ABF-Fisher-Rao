#!/usr/bin/env python
"""Ethane/LTA under the histogram (P0) estimator: runner for configs/lta_histogram/campaign.json.

Subcommands (GPU 3 only, one process at a time; every arm of one temperature in ONE process):

  production --temperature T   abf -> fr_uniform -> fr_sham (the sham replays fr_uniform's realised
                               per-opportunity counts); refuses to start unless per_T[T].fr_rate is frozen
  calibrate  --temperature T   SAFETY-ONLY FR-rate ladder under THIS sampler (new T only); writes
                               results/lta_histogram/calibration/fr_rate_selection_T{T}.json and freezes
                               the selected rate into campaign.json per_T[T].fr_rate (timestamped); never
                               touches a reference or an error metric
  width-ladder                 ABF-only n_grid ladder (sensitivity check of the frozen width), 300 K, R = 4
  movie      --temperature T   both movie arms at R = 1 with the snapshot record, then movie_data.npz for
                               scripts/lta_movie_render.py

    CUDA_VISIBLE_DEVICES=3 python -u scripts/run_lta_histogram.py production --temperature 300
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import socket
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from lta.core_lta import LTAParams, LTASimConfig, LTASystem, run_sampler  # noqa: E402

CAMPAIGN = os.path.join(ROOT, "configs/lta_histogram/campaign.json")
RESULTS = os.path.join(ROOT, "results/lta_histogram")
PI = math.pi
LADDER = (0.02, 0.05, 0.10, 0.20)
CAL_STEPS = 120_000
CAL_SEEDS = [1110, 1111]


def git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def now():
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_campaign():
    return json.load(open(CAMPAIGN))


def save_campaign(c):
    tmp = CAMPAIGN + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(c, fh, indent=2)
    os.replace(tmp, CAMPAIGN)


def log_exec(c, entry):
    c.setdefault("execution_log", []).append(dict(at=now(), **entry))
    save_campaign(c)


def build_sim(c, *, fr_rate, rng_seed, **over):
    s = dict(c["sampler"])
    kw = dict(n_steps=s["n_steps"], n_replicas=s["n_replicas"], dt=s["dt"], save_every=s["save_every"],
              n_grid=s["n_grid"], abf_bandwidth=s["abf_bandwidth"], kde_bandwidth=s["kde_bandwidth"],
              abf_warmup_steps=s["abf_warmup_steps"], abf_force_clip=s["abf_force_clip"],
              estimator_burn_in_steps=s["estimator_burn_in_steps"], fr_start_steps=s["fr_start_steps"],
              fr_every=s["fr_every"], score_clip=s["score_clip"], max_event_fraction=s["max_event_fraction"],
              target_ema_rate=s["target_ema_rate"], abf_estimator=s["abf_estimator"],
              fr_rate=float(fr_rate), rng_seed=int(rng_seed))
    kw.update(over)
    return LTASimConfig(**kw)


def device_and_system(temperature):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        assert torch.cuda.device_count() == 1, "pin exactly one GPU (CUDA_VISIBLE_DEVICES)"
    system = LTASystem(LTAParams(temperature=float(temperature)), device, root=ROOT)
    return device, system


def write_npz(path, out, meta):
    payload = {k: v for k, v in out.items() if isinstance(v, (np.ndarray, np.generic, int, float, str))}
    payload["meta"] = json.dumps(meta)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- production
def production(a):
    c = load_campaign()
    tkey = f"{a.temperature:g}"
    pt = c["per_T"][tkey]
    rate = pt["fr_rate"]
    assert rate is not None, f"fr_rate not frozen for T={tkey}: run `calibrate --temperature {tkey}` first"
    assert os.path.exists(os.path.join(ROOT, pt["reference"])), \
        f"reference missing for T={tkey} ({pt['reference']}); build it first (scripts/run_lta_reference.py)"
    assert c["arms"] == ["abf", "fr_uniform", "fr_sham"], "three arms exactly"
    seeds = list(range(int(pt["seeds_first"]), int(pt["seeds_first"]) + int(c["seeds_count"])))
    sim = build_sim(c, fr_rate=rate, rng_seed=pt["rng_seed"])
    device, system = device_and_system(a.temperature)
    out_dir = os.path.join(RESULTS, f"production_T{tkey}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"LTA histogram-estimator production, T = {tkey} K")
    print(f"  {len(seeds)} seed labels {seeds[0]}-{seeds[-1]}, N = {sim.n_replicas}, {sim.n_steps} steps, "
          f"estimator {sim.abf_estimator} ({sim.n_grid} bins), fr_start {sim.fr_start_steps}, fr_rate {rate}, "
          f"rng_seed {sim.rng_seed} (= kernel sweep's: noise-paired)")
    print(f"  device {device}  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')!r}  "
          f"config_hash {sim.config_hash()}\n", flush=True)
    log_exec(c, dict(stage=f"production_T{tkey}", event="start", git_rev=git_rev()))
    t0 = time.time()
    shadow = None
    for method in c["arms"]:
        path = os.path.join(out_dir, f"{method}.npz")
        if os.path.exists(path):
            print(f"  skip {method} (exists)", flush=True)
            if method == "fr_uniform":
                shadow = np.load(path, allow_pickle=True)["event_counts"]
            continue
        kw = {}
        if method == "fr_sham":
            assert shadow is not None, "fr_sham needs fr_uniform's event_counts"
            kw["shadow_counts"] = shadow
        out = run_sampler(method, system, sim, seeds=seeds, verbose=True, **kw)
        if method == "fr_uniform":
            shadow = out["event_counts"]
        out["seeds"] = np.asarray(seeds)
        out["config_hash"] = sim.config_hash()
        write_npz(path, out, dict(method=method, seeds=seeds, rng_seed=int(sim.rng_seed), fr_rate=rate,
                                  abf_estimator=sim.abf_estimator, n_grid=sim.n_grid,
                                  config_hash=sim.config_hash(), campaign=os.path.relpath(CAMPAIGN, ROOT),
                                  git_rev=git_rev(), host=socket.gethostname(),
                                  cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                                  temperature_K=float(a.temperature), shadow_of=("fr_uniform" if method == "fr_sham" else None)))
        print(f"  wrote {os.path.relpath(path, ROOT)}", flush=True)
    log_exec(load_campaign(), dict(stage=f"production_T{tkey}", event="done", minutes=round((time.time() - t0) / 60, 1)))
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {out_dir}")


# --------------------------------------------------------------------------- calibration (safety only)
def calibrate(a):
    """Largest ladder rate meeting the genealogy floors; no reference, no error metric."""
    c = load_campaign()
    tkey = f"{a.temperature:g}"
    pt = c["per_T"][tkey]
    prod_dir = os.path.join(RESULTS, f"production_T{tkey}")
    assert not os.path.exists(os.path.join(prod_dir, "fr_uniform.npz")), "production exists; the rate is frozen"
    rule = c["success_rule"]
    device, system = device_and_system(a.temperature)
    out_dir = os.path.join(RESULTS, "calibration")
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for rate in LADDER:
        sim = build_sim(c, fr_rate=rate, rng_seed=20260830 + int(a.temperature), n_steps=CAL_STEPS)
        out = run_sampler("fr_uniform", system, sim, seeds=CAL_SEEDS, verbose=False)
        ess = np.asarray(out["ancestor_ess"], dtype=float)
        wmax = np.asarray(out["max_ancestor_frac"], dtype=float)
        active = np.asarray(out["steps"]) >= sim.fr_start_steps
        N = sim.n_replicas
        ess_min = float(np.nanmin(ess[active]) / N)
        wmax_max = float(np.nanmax(wmax[active]))
        ev = float(out["total_replacement_events"].sum() / (len(CAL_SEEDS) * N))
        ok = ess_min >= rule["ess_anc_over_N_min"] and wmax_max <= rule["wmax_max"]
        rows.append(dict(rate=rate, ess_min=ess_min, wmax_max=wmax_max, events_per_replica=ev, ok=bool(ok)))
        print(f"  rate {rate:>5.2f}: min ESS/N {ess_min:.3f}  wmax {wmax_max:.4f}  events/replica {ev:.3f}  ok={ok}", flush=True)
    safe = [r for r in rows if r["ok"]]
    sel = max(safe, key=lambda r: r["rate"])["rate"] if safe else None
    result = dict(ladder=rows, selected=sel, temperature_K=float(a.temperature), estimator=c["sampler"]["abf_estimator"],
                  fr_start_steps=int(c["sampler"]["fr_start_steps"]), cal_steps=CAL_STEPS, seeds=CAL_SEEDS,
                  rule=dict(ess_min=rule["ess_anc_over_N_min"], wmax_max=rule["wmax_max"]), frozen_at=now(),
                  note="Selected on genealogy safety only under the histogram sampler; no error metric computed or read; "
                       "among safe rates the LARGEST is chosen so the mechanism is exercised.")
    with open(os.path.join(out_dir, f"fr_rate_selection_T{tkey}.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    c = load_campaign()
    c["per_T"][tkey]["fr_rate"] = sel
    c["per_T"][tkey]["fr_rate_frozen_at"] = result["frozen_at"]
    log_exec(c, dict(stage=f"calibrate_T{tkey}", event="frozen", fr_rate=sel))
    print(f"selected fr_rate = {sel} (T = {tkey} K), frozen into campaign.json at {result['frozen_at']}")


# --------------------------------------------------------------------------- width ladder (ABF only)
def width_ladder(a):
    c = load_campaign()
    w = c["width_rule"]
    seeds = list(range(1100, 1104))
    T = 300.0
    device, system = device_and_system(T)
    out_dir = os.path.join(RESULTS, "calibration", "width_ladder_T300")
    os.makedirs(out_dir, exist_ok=True)
    for n_grid in (90, 180, 360):
        path = os.path.join(out_dir, f"n{n_grid}.npz")
        if os.path.exists(path):
            print(f"  skip n_grid {n_grid} (exists)"); continue
        sim = build_sim(c, fr_rate=c["per_T"]["300"]["fr_rate"], rng_seed=20261003, n_grid=n_grid)
        out = run_sampler("abf", system, sim, seeds=seeds, verbose=True)
        out["seeds"] = np.asarray(seeds); out["config_hash"] = sim.config_hash()
        write_npz(path, out, dict(method="abf", seeds=seeds, rng_seed=20261003, n_grid=n_grid, temperature_K=T,
                                  abf_estimator=sim.abf_estimator, git_rev=git_rev(), purpose="width ladder (clause b), ABF only"))
        print(f"  wrote {os.path.relpath(path, ROOT)}", flush=True)
    log_exec(load_campaign(), dict(stage="width_ladder_T300", event="done"))


# --------------------------------------------------------------------------- movie simulate
def _interp_ref(F_ref, grid_ref, grid_eng):
    order = np.argsort(grid_ref)
    gr, fr = grid_ref[order], F_ref[order]
    gx = np.concatenate([gr - 2 * PI, gr, gr + 2 * PI])
    return np.interp(grid_eng, gx, np.concatenate([fr, fr, fr]))


def _aligned_rms(F, F_ref):
    d = F - F_ref
    d = d - d.mean(axis=-1, keepdims=True)
    return np.sqrt((d * d).mean(axis=-1))


def movie(a):
    c = load_campaign()
    mv = c["movies"]
    tkey = f"{a.temperature:g}"
    pt = c["per_T"][tkey]
    rate = pt["fr_rate"]
    assert rate is not None, f"fr_rate not frozen for T={tkey}"
    ref_path = os.path.join(ROOT, pt["reference"])
    ref = np.load(ref_path, allow_pickle=True)
    out_dir = os.path.join(ROOT, mv["out"].format(T=tkey))
    if a.smoke:      # a pipeline check never writes next to a real movie
        out_dir = os.path.join(a.smoke_dir or os.path.join(RESULTS, "_smoke"), f"movie_T{tkey}")
    raw_dir = os.path.join(out_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    seed = int(mv["seed_label"])
    rng_seed = 20261000 + int(round(a.temperature))
    snap = int(a.snap_every or mv["store_snapshots"])
    sim = build_sim(c, fr_rate=rate, rng_seed=rng_seed, n_replicas=int(mv["n_replicas"]),
                    store_snapshots=snap, snapshot_seed_index=0)
    if a.smoke:
        sim = LTASimConfig(**{**sim.__dict__, "n_steps": 3000, "n_replicas": 64, "save_every": 500,
                              "abf_warmup_steps": 300, "estimator_burn_in_steps": 300, "fr_start_steps": 300,
                              "store_snapshots": 50})
    device, system = device_and_system(a.temperature)
    print(f"LTA movie simulate T = {tkey} K: seed label {seed}, rng_seed {rng_seed}, N = {sim.n_replicas}, "
          f"{sim.n_steps} steps, snapshots every {sim.store_snapshots}, fr_rate {rate}, estimator {sim.abf_estimator}", flush=True)
    grid = None
    F_ref = U_ref = None
    data = {}
    runs = {}
    for method in mv["arms"]:
        path = os.path.join(raw_dir, f"{method}.npz")
        if os.path.exists(path) and not a.smoke:
            print(f"  loading existing {os.path.relpath(path, ROOT)}", flush=True)
            z = np.load(path, allow_pickle=True)
            runs[method] = {k: z[k] for k in z.files}
            continue
        out = run_sampler(method, system, sim, seeds=[seed], verbose=True)
        out["seeds"] = np.asarray([seed]); out["config_hash"] = sim.config_hash()
        write_npz(path, out, dict(method=method, seeds=[seed], rng_seed=rng_seed, fr_rate=rate, temperature_K=float(a.temperature),
                                  abf_estimator=sim.abf_estimator, git_rev=git_rev(), store_snapshots=sim.store_snapshots))
        z = np.load(path, allow_pickle=True)
        runs[method] = {k: z[k] for k in z.files}
    # ---- consolidate for the renderer ----
    r0 = runs[mv["arms"][0]]
    grid = np.asarray(r0["grid"], float)
    F_ref = _interp_ref(np.asarray(ref["F"], float), np.asarray(ref["grid_phi"], float), grid); F_ref -= F_ref.mean()
    U_ref = _interp_ref(np.asarray(ref["U"], float), np.asarray(ref["grid_phi"], float), grid); U_ref -= U_ref.mean()
    kT = float(ref["kT"]); a_ps = float(r0["a_pseudo"])
    fw = np.load(os.path.join(ROOT, "cache/lta/framework.npz"), allow_pickle=True)
    data.update(temperature=float(a.temperature), kT=kT, beta=1.0 / kT, a_pseudo=a_ps, box=float(r0["box"]),
                n_bins=int(sim.n_grid), seed=seed, featured=int(mv["featured"]),
                t_fr=float(sim.fr_start_steps * sim.dt), t_warm=float(sim.abf_warmup_steps * sim.dt),
                t_burn=float(sim.estimator_burn_in_steps * sim.dt), fr_rate=float(rate), git_rev=git_rev(),
                reference_path=os.path.relpath(ref_path, ROOT), sim_json=json.dumps(sim.__dict__, sort_keys=True),
                params_json=json.dumps(system.p.__dict__, sort_keys=True), methods=np.array(mv["arms"]),
                N=int(sim.n_replicas), T=float(sim.n_steps * sim.dt), dt=float(sim.dt),
                grid=grid, z_grid=grid * a_ps / (2 * PI), dphi=float(r0["dphi"]),
                F_ref=F_ref.astype(np.float32), U_ref=U_ref.astype(np.float32), mTS_ref=(F_ref - U_ref).astype(np.float32),
                dF_kT=float(ref["dF_barrier"]) / kT, dU_kT=float(ref["dU_barrier"]) / kT, mTdS_kT=float(ref["mTdS_barrier"]) / kT,
                o_pos=np.asarray(fw["o_pos"], np.float32), si_pos=np.asarray(fw["pos"])[np.asarray(fw["kind"]) == "Si"].astype(np.float32),
                snap_t=np.asarray(r0["snap_times"], float), snap_step=np.asarray(r0["snap_steps"]), save_t=np.asarray(r0["times"], float))
    N = int(sim.n_replicas)
    for m in mv["arms"]:
        r = runs[m]
        assert np.array_equal(r["snap_steps"], r0["snap_steps"])
        F = np.asarray(r["snap_pmf"], np.float64)
        eF = _aligned_rms(F, F_ref[None, :])
        save_eF = _aligned_rms(np.asarray(r["pmf"], float)[:, 0, :], F_ref[None, :])
        # the record's PMF at the saves must reproduce the saved series
        sv = np.searchsorted(r["snap_steps"], np.asarray(r["steps"]))
        assert np.array_equal(r["snap_steps"][sv], np.asarray(r["steps"]))
        dev = np.abs(eF[sv] - save_eF).max()
        assert dev < 1e-4, f"snapshot read-out does not reproduce the saved e_F series ({dev:.2e})"
        phi = np.asarray(r["snap_phi"], np.float32)
        zc = np.abs(phi) * a_ps / (2 * PI)
        ess = np.asarray(r["ancestor_ess"], float)[:, 0]; wm = np.asarray(r["max_ancestor_frac"], float)[:, 0]
        act = np.asarray(r["steps"]) >= sim.fr_start_steps
        data.update({f"{m}/snap_q": np.asarray(r["snap_q"], np.float32), f"{m}/snap_phi": phi,
                     f"{m}/snap_wid": np.asarray(r["snap_wid"], np.int32), f"{m}/snap_F": F.astype(np.float32),
                     f"{m}/snap_counts": np.asarray(r["snap_counts"], np.float32), f"{m}/snap_crossings": np.asarray(r["snap_crossings"]),
                     f"{m}/snap_eF": eF.astype(np.float32), f"{m}/snap_ev_die": np.asarray(r["snap_ev_die"], np.float32),
                     f"{m}/snap_ev_birth": np.asarray(r["snap_ev_birth"], np.float32), f"{m}/save_eF": save_eF,
                     f"{m}/snap_anc": np.asarray(r["snap_anc"], np.int32),
                     f"{m}/snap_ev_slots": np.asarray(r["snap_ev_slots"], np.int32),
                     f"{m}/snap_ev_ids": np.asarray(r["snap_ev_ids"], np.int64),
                     f"{m}/l2_f": float(save_eF[-1]), f"{m}/integrated_l2_f": float(np.trapezoid(save_eF, data["save_t"])),
                     f"{m}/total_replacement_events": float(np.asarray(r["total_replacement_events"]).sum()),
                     f"{m}/min_ess_frac": (float(np.nanmin(ess[act]) / N) if act.any() and np.isfinite(ess[act]).any() else float("nan")),
                     f"{m}/max_wmax": (float(np.nanmax(wm[act])) if act.any() and np.isfinite(wm[act]).any() else float("nan")),
                     f"{m}/n_cage_crossings": float(np.asarray(r["n_cage_crossings"]).sum()),
                     f"{m}/frac_window": (zc < sim.window_half).mean(axis=1).astype(np.float32),
                     f"{m}/frac_cage": (zc > sim.cage_min).mean(axis=1).astype(np.float32)})
        print(f"  {m:11s} e_F(T) {save_eF[-1]:.4f} kJ/mol  I_F {data[f'{m}/integrated_l2_f']:.3f}  events {int(data[f'{m}/total_replacement_events'])}  "
              f"crossings {int(data[f'{m}/n_cage_crossings'])}  min ESS/N {data[f'{m}/min_ess_frac']:.3f}", flush=True)
    same = np.abs(data["abf/snap_phi"] - data["fr_uniform/snap_phi"]).max(axis=1)
    k = int(np.argmax(same > 1e-6)) if (same > 1e-6).any() else len(same)
    data["arms_identical_until_t"] = float(data["snap_t"][min(k, len(same) - 1)])
    d_fin = 100 * (data["fr_uniform/l2_f"] / data["abf/l2_f"] - 1); d_int = 100 * (data["fr_uniform/integrated_l2_f"] / data["abf/integrated_l2_f"] - 1)
    print(f"  this seed: final e_F {d_fin:+.1f} %, integrated {d_int:+.1f} %; arms identical until t = {data['arms_identical_until_t']:.2f}")
    np.savez_compressed(os.path.join(out_dir, "movie_data.npz"), **data)
    with open(os.path.join(out_dir, "movie_summary.json"), "w") as fh:
        json.dump({k: (v if isinstance(v, (int, float, str)) else None) for k, v in data.items() if not isinstance(v, np.ndarray)}
                  | dict(d_final_pct=d_fin, d_int_pct=d_int), fh, indent=2, default=float)
    print("saved", os.path.relpath(os.path.join(out_dir, "movie_data.npz"), ROOT))
    if not a.smoke:
        log_exec(load_campaign(), dict(stage=f"movie_T{tkey}", event="simulated", d_final_pct=round(d_fin, 1), d_int_pct=round(d_int, 1)))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("production"); p.add_argument("--temperature", type=float, required=True)
    p = sub.add_parser("calibrate"); p.add_argument("--temperature", type=float, required=True)
    sub.add_parser("width-ladder")
    p = sub.add_parser("movie"); p.add_argument("--temperature", type=float, required=True)
    p.add_argument("--snap-every", type=int, default=0); p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke-dir", default=None)
    a = ap.parse_args()
    {"production": production, "calibrate": calibrate, "width-ladder": width_ladder, "movie": movie}[a.stage](a)


if __name__ == "__main__":
    main()
