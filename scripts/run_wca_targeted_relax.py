#!/usr/bin/env python
"""WCA FR + targeted solvent relaxation campaign runner: stages W0A, W0B, W1, W2.

Preregistration: configs/targeted_relax_campaign/wca_fr_targeted_relax_prereg.json.
Prints wall time and safety / accounting counters only -- never a free-energy error.

    CUDA_VISIBLE_DEVICES=3 python -u scripts/run_wca_targeted_relax.py --stage W0A   # GPU 3 ONLY
    CUDA_VISIBLE_DEVICES=3 python -u scripts/run_wca_targeted_relax.py --stage W0B [--extend]
    CUDA_VISIBLE_DEVICES=3 python -u scripts/run_wca_targeted_relax.py --stage W1 [--seeds 824-827]
    CUDA_VISIBLE_DEVICES=3 python -u scripts/run_wca_targeted_relax.py --stage W2
"""
from __future__ import annotations

import argparse
import glob
import hashlib
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
import wca_abffr_core as core      # noqa: E402
import wca_phase_jobs as jobs      # noqa: E402

PREREG = os.path.join(ROOT, "configs/targeted_relax_campaign/wca_fr_targeted_relax_prereg.json")
PHASE_CONFIG = os.path.join(ROOT, "configs/wca_phase_diagram_production.yaml")
CAMPAIGN = os.path.join(ROOT, "results", "targeted_relax_campaign", "wca")
CACHE = os.path.join(ROOT, "cache", "phase_hp_v3")
REFERENCE_NPZ = os.path.join(CACHE, "wca_ti_b1_h2_w2_n10_a1.5_g160.npz")
CELL = dict(beta=1.0, h=2.0, w=2.0, n_dim=10, a=1.5, sigma=1.0, epsilon=1.0)
N_STEPS, N_REPLICAS, SAVE_EVERY = 120_000, 1024, 2500
READOUT = (0.0125, 0.00625, 0.0)
FR_KNOBS = dict(fr_rate=0.10, target_ema_rate=0.005, max_event_fraction=0.02, fr_every=5, fr_start_steps=20000, score_clip=2.0)
LADDER_RHO = (0.25, 0.5, 1.0)
W0B = dict(n_sites=10, n_controls=2, n_rep=64, n_eq=4000, n_prod=16000, record_every=2)
TAKEN = (set(range(10)) | {20, 21, 42, 50, 51, 123} | set(range(100, 108)) | set(range(400, 416)) | {500, 501}
         | set(range(600, 616)) | set(range(700, 716)) | set(range(1000, 1016)) | set(range(2000, 2008)))


def git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def parse_seeds(s):
    if "-" in s:
        lo, hi = s.split("-"); return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in s.split(",")]


def base_config():
    cfg = jobs.load_yaml(PHASE_CONFIG)
    base = jobs.effective_base(cfg, "production")
    assert float(base["abf_bandwidth"]) == 0.025 and float(base["abf_smooth_sigma"]) == 0.5
    for k, v in CELL.items():
        assert float(cfg["system_defaults"][k]) == float(v), f"cell {k} moved"
    for block in ("fr_uniform", "fr_estimated"):
        for k, v in FR_KNOBS.items():
            assert float(cfg["methods"][block].get(k, base.get(k))) == float(v), f"FR knob {k} drifted"
    assert os.path.exists(REFERENCE_NPZ), REFERENCE_NPZ
    return base


def make_spec(stage, name, method, seed):
    return jobs.PhaseRunSpec(stage=stage, name=name, method=method, seed=int(seed), n_steps=N_STEPS,
                             n_replicas=N_REPLICAS, save_every=SAVE_EVERY, **CELL, **FR_KNOBS)


def load_tau_map():
    p = os.path.join(CAMPAIGN, "W0", "tau_map.json")
    d = json.load(open(p))
    tau = np.asarray(d["tau_grid"], dtype=np.float64)
    h = hashlib.sha256(tau.tobytes()).hexdigest()
    assert h == d["sha256"], "tau map hash mismatch"
    assert d["passed"], "W0 did not pass; W1 must not start"
    return tau, d


def run_block(stage_name, out_stage, seeds, arms, base, relax_of, sensitivity_record=False, overwrite=False, verbose=False):
    """arms: list of (name, method).  relax_of(name) -> RelaxConfig or None."""
    raw_dir = os.path.join(CAMPAIGN, out_stage, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    engines, done, t_start = {}, 0, time.time()
    total = len(seeds) * len(arms)
    for sd in seeds:
        for name, m in arms:
            sp = make_spec(stage_name, name, m, sd)
            path = jobs.run_npz_path(raw_dir, sp)
            if not overwrite and jobs.run_is_valid(path):
                print(f"  skip {sp.run_id()}", flush=True); done += 1; continue
            eng = jobs.get_engine(sp, engines)
            t0 = time.time()
            out = jobs.execute_run(sp, base, eng, cache_dir=CACHE, verbose=verbose, store_profiles=True,
                                   readout_bandwidths=READOUT, relax=relax_of(name), sensitivity_record=sensitivity_record)
            assert "v2" in str(out.get("reference_label", "")), "unexpected reference"
            assert float(out["abf_bandwidth_online"]) == 0.025
            assert not bool(out["had_nan"]), f"NaN in {sp.run_id()} -> UNSAFE"
            jobs.save_run(path, out)
            done += 1
            info = ""
            if m != "abf":
                info += f" repl={out['total_replacement_events']} essW={float(out['min_ancestor_ess_window']) / N_REPLICAS:.3f} wmax={float(out['max_ancestor_frac_over_time']):.3f}"
            if "relax_steps_total" in out:
                info += (f" inner={int(out['relax_steps_total'])} ({float(out['relax_cost_ratio']):.3f}x) active={float(np.mean(out['relax_active_frac'][8:])):.3f}"
                         f" innerwall={float(out['relax_inner_wall_seconds']):.0f}s")
            print(f"  [{done}/{total}] {name:>14s} seed{sd}: saved{info} ({time.time() - t0:.0f}s)", flush=True)
    return raw_dir, time.time() - t_start, done


def stage_w0a(a, pre, base):
    seeds = parse_seeds(a.seeds) if a.seeds else list(range(pre["seeds"]["W0A"][0], pre["seeds"]["W0A"][1] + 1))
    assert not (set(seeds) & TAKEN)
    print(f"W0-A: {len(seeds)} plain-ABF instrument runs (seeds {seeds[0]}-{seeds[-1]}); sensitivity accumulator + final configurations saved; no error metric printed")
    if a.dry_run:
        return
    raw_dir, wall, done = run_block("W0A", "W0", seeds, [("abf", "abf")], base, lambda n: None, sensitivity_record=True,
                                    overwrite=a.overwrite, verbose=a.verbose)
    prov = dict(script=os.path.basename(__file__), stage="W0A", git_rev=git_rev(), host=socket.gethostname(),
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""), prereg=os.path.relpath(PREREG, ROOT),
                seeds=seeds, wall_seconds=wall, n_runs=done)
    json.dump(prov, open(os.path.join(CAMPAIGN, "W0", "provenance_W0A.json"), "w"), indent=2, default=float)
    print(f"\ndone W0-A: {done} runs in {wall / 3600:.2f} h")


def select_sites(z_grid, vhat_med, z_lo, z_hi, n_sites, n_controls):
    """MECHANICAL: quantiles of the cumulative sensitivity mass + the argmin of v_hat in each half of the window."""
    m = (z_grid >= z_lo) & (z_grid <= z_hi)
    z, v = z_grid[m], np.clip(vhat_med[m], 0, None)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(z))])
    cum = cum / max(cum[-1], 1e-300)
    qs = (np.arange(n_sites) + 0.5) / n_sites
    sites = [float(np.interp(q, cum, z)) for q in qs]
    mid = 0.5 * (z_lo + z_hi)
    controls = []
    for sel in (z <= mid, z > mid):
        if sel.any():
            controls.append(float(z[sel][np.argmin(v[sel])]))
    return sites, controls[:n_controls]


def stage_w0b(a, pre, base):
    files = sorted(glob.glob(os.path.join(CAMPAIGN, "W0", "raw", "W0A__abf__*.npz")))
    assert len(files) >= 8, f"need the 8 W0-A runs, found {len(files)}"
    runs = [jobs.load_run(f) for f in files]
    grid = np.asarray(runs[0]["grid"], float)
    vhat_med = np.median(np.stack([np.asarray(r["final_vhat"], float) for r in runs]), axis=0)
    sites, controls = select_sites(grid, vhat_med, float(base["eval_z_lo"]), float(base["eval_z_hi"]), W0B["n_sites"], W0B["n_controls"])
    zk = np.array(sites + controls)
    kind = ["site"] * len(sites) + ["control"] * len(controls)
    n_prod = W0B["n_prod"] * (2 if a.extend else 1)
    print(f"W0-B: {len(sites)} quantile sites + {len(controls)} low-sensitivity controls (mechanical); n_eq {W0B['n_eq']}, n_prod {n_prod}"
          + (" [ONE preregistered 2x extension]" if a.extend else ""))
    print("  z_k: " + ", ".join(f"{z:.3f}{'c' if k == 'control' else ''}" for z, k in zip(zk, kind)))
    if a.dry_run:
        return
    # pooled final configurations of the 8 independent ABF runs
    params = jobs.build_params(make_spec("W0B", "abf", "abf", 0))
    engine = core.WCADimerEngine(params, core.DEVICE, core.DTYPE)
    sim = jobs.build_sim(make_spec("W0B", "abf", "abf", 0), base)
    Q = torch.as_tensor(np.concatenate([np.asarray(r["final_q"], np.float32) for r in runs]), device=engine.device, dtype=engine.dtype)
    Zq = core.reaction_coordinate(Q, params)
    out = dict(z_sites=zk, kind=np.array(kind), n_eq=W0B["n_eq"], n_prod=n_prod, record_every=W0B["record_every"], dt=sim.dt)
    t_start = time.time()
    for j, z_k in enumerate(zk):
        order = torch.argsort((Zq - float(z_k)).abs())[:W0B["n_rep"]]
        q0 = Q.index_select(0, order)
        gen = torch.Generator(device=engine.device); gen.manual_seed(int(pre["seeds"]["W0A"][0]) * 1000 + j)
        t0 = time.time()
        f = core.constrained_force_series(engine, params, sim, q0, float(z_k), W0B["n_eq"], n_prod, gen, record_every=W0B["record_every"])
        out[f"f_{j}"] = f.astype(np.float32)
        out[f"z_start_{j}"] = core.to_numpy(Zq.index_select(0, order))
        print(f"  site {j:2d} z={z_k:+.3f} ({kind[j]}): {f.shape[0]} replicas x {f.shape[1]} samples ({time.time() - t0:.0f}s)", flush=True)
    np.savez_compressed(os.path.join(CAMPAIGN, "W0", "constrained.npz"), **out)
    json.dump(dict(sites=sites, controls=controls, rule="quantiles of cumulative sensitivity mass + argmin per half-window",
                   vhat_median_final=vhat_med.tolist(), grid=grid.tolist(), extended=bool(a.extend), n_prod=n_prod,
                   git_rev=git_rev(), wall_seconds=time.time() - t_start),
              open(os.path.join(CAMPAIGN, "W0", "selection.json"), "w"), indent=2)
    print(f"\ndone W0-B in {(time.time() - t_start) / 60:.1f} min -> W0/constrained.npz, selection.json")


def stage_w1(a, pre, base):
    tau, tmap = load_tau_map()
    seeds = parse_seeds(a.seeds) if a.seeds else list(range(pre["seeds"]["W1"][0], pre["seeds"]["W1"][1] + 1))
    assert not (set(seeds) & (TAKEN | set(range(800, 808)) | set(range(900, 916))))
    arms = [("abf", "abf"), ("fr_uniform", "fr_uniform")]
    for rho in LADDER_RHO:
        arms += [(f"abf_targ{rho:g}", "abf"), (f"fr_targ{rho:g}", "fr_uniform")]

    def relax_of(name):
        if "targ" not in name:
            return None
        rho = float(name.split("targ")[1])
        return core.RelaxConfig(rho=rho, tau_grid=tuple(tau.tolist()), target="sensitivity")
    print(f"W1: {len(arms)} arms x {len(seeds)} seeds ({seeds[0]}-{seeds[-1]}); tau map {tmap['sha256'][:12]}; no error metric printed")
    if a.dry_run:
        return
    raw_dir, wall, done = run_block("W1", "W1", seeds, arms, base, relax_of, overwrite=a.overwrite, verbose=a.verbose)
    prov = dict(script=os.path.basename(__file__), stage="W1", git_rev=git_rev(), host=socket.gethostname(),
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""), prereg=os.path.relpath(PREREG, ROOT),
                seeds=seeds, arms=[n for n, _ in arms], tau_map_sha256=tmap["sha256"], wall_seconds=wall, n_runs=done)
    json.dump(prov, open(os.path.join(CAMPAIGN, "W1", f"provenance_{seeds[0]}-{seeds[-1]}.json"), "w"), indent=2, default=float)
    print(f"\ndone W1: {done} runs in {wall / 3600:.2f} h")


def stage_w2(a, pre, base):
    tau, tmap = load_tau_map()
    sel = json.load(open(os.path.join(CAMPAIGN, "W1", "rho_selection.json")))
    assert sel["licensed"], "W1 did not license W2 (STOP recorded); W2 must not run"
    rho = float(sel["rho_star"])
    seeds = parse_seeds(a.seeds) if a.seeds else list(range(pre["seeds"]["W2"][0], pre["seeds"]["W2"][1] + 1))
    assert not (set(seeds) & (TAKEN | set(range(800, 808)) | set(range(820, 828))))
    arms = [("abf", "abf"), ("fr_uniform", "fr_uniform"), (f"abf_targ{rho:g}", "abf"), (f"fr_targ{rho:g}", "fr_uniform"),
            (f"fr_rand{rho:g}", "fr_uniform")]
    if a.with_abf_rand:
        arms.append((f"abf_rand{rho:g}", "abf"))

    def relax_of(name):
        if "targ" in name:
            return core.RelaxConfig(rho=rho, tau_grid=tuple(tau.tolist()), target="sensitivity")
        if "rand" in name:
            return core.RelaxConfig(rho=rho, tau_grid=tuple(tau.tolist()), target="random")
        return None
    print(f"W2: {len(arms)} arms x {len(seeds)} seeds ({seeds[0]}-{seeds[-1]}); rho* {rho:g}; h_read** {sel['h_read_starstar']}; no error metric printed")
    if a.dry_run:
        return
    raw_dir, wall, done = run_block("W2", "W2", seeds, arms, base, relax_of, overwrite=a.overwrite, verbose=a.verbose)
    prov = dict(script=os.path.basename(__file__), stage="W2", git_rev=git_rev(), host=socket.gethostname(),
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""), prereg=os.path.relpath(PREREG, ROOT),
                seeds=seeds, arms=[n for n, _ in arms], rho_star=rho, tau_map_sha256=tmap["sha256"], wall_seconds=wall, n_runs=done)
    json.dump(prov, open(os.path.join(CAMPAIGN, "W2", f"provenance_{seeds[0]}-{seeds[-1]}.json"), "w"), indent=2, default=float)
    print(f"\ndone W2: {done} runs in {wall / 3600:.2f} h")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["W0A", "W0B", "W1", "W2"], required=True)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--extend", action="store_true", help="W0B: the ONE preregistered 2x extension of n_prod")
    ap.add_argument("--with-abf-rand", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    pre = json.load(open(PREREG))
    base = base_config()
    if torch.cuda.is_available():
        assert torch.cuda.device_count() == 1, "pin exactly one GPU"
    os.makedirs(CAMPAIGN, exist_ok=True)
    {"W0A": stage_w0a, "W0B": stage_w0b, "W1": stage_w1, "W2": stage_w2}[a.stage](a, pre, base)


if __name__ == "__main__":
    main()
