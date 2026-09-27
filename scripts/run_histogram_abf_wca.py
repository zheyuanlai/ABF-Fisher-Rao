#!/usr/bin/env python
"""Histogram (P0) ABF estimator campaign, WCA dimer (accepted corrected Case IX cell).

Frozen design: configs/histogram_abf/campaign.json (read here and asserted against the production
YAML and the accepted confirmation prereg).  Doc: docs/HISTOGRAM_ABF_REPLICATION.md.

Stages
------
  floor         deterministic P0 projection floor from the accepted TI reference (no simulation)
  calibration   histogram ABF ONLY, every ladder n_bins on the calibration seeds
  confirmation  four arms per seed in ONE process (shared lattice init and noise stream):
                kernel abf / kernel fr_uniform (with the accepted read-out bank), hist abf /
                hist fr_uniform at the FROZEN n_bins (configs/histogram_abf/selected_bins.json)

The accepted TI reference must exist; it is never recomputed here (execute_run would compute a
missing one silently, so the file is asserted first).  The runner prints wall time and FR safety
counters only -- no error metric of an FR arm before the analyzer runs.

    CUDA_VISIBLE_DEVICES=3 python -u scripts/run_histogram_abf_wca.py --stage floor,calibration
    python scripts/analyze_histogram_abf_wca.py --select
    CUDA_VISIBLE_DEVICES=3 python -u scripts/run_histogram_abf_wca.py --stage confirmation
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import wca_abffr_core as core   # noqa: E402
import wca_phase_jobs as jobs   # noqa: E402

CAMPAIGN = os.path.join(ROOT, "configs", "histogram_abf", "campaign.json")
SELECTED = os.path.join(ROOT, "configs", "histogram_abf", "selected_bins.json")
PHASE_CONFIG = os.path.join(ROOT, "configs", "wca_phase_diagram_production.yaml")
PREREG_IX = os.path.join(ROOT, "configs", "information_campaign", "wca_corrected_confirmation_prereg.json")
CACHE = os.path.join(ROOT, "cache", "phase_hp_v3")
REFERENCE_NPZ = os.path.join(CACHE, "wca_ti_b1_h2_w2_n10_a1.5_g160.npz")
OUT_ROOT = os.path.join(ROOT, "results", "histogram_abf", "wca")


def git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def load_frozen():
    c = json.load(open(CAMPAIGN))["wca"]
    cfg = jobs.load_yaml(PHASE_CONFIG)
    base = jobs.effective_base(cfg, "production")
    pre = json.load(open(PREREG_IX))
    kb = c["kernel_baseline"]
    assert float(base["abf_bandwidth"]) == kb["abf_bandwidth"] == pre["corrected_baseline"]["h_bias"]
    assert float(base["abf_smooth_sigma"]) == kb["abf_smooth_sigma"]
    assert float(base["kde_bandwidth"]) == c["fr_knobs"]["kde_bandwidth"]
    for k, v in c["cell"].items():
        assert float(cfg["system_defaults"][k]) == float(v), f"cell {k} moved"
    fr = {k: v for k, v in c["fr_knobs"].items() if k != "kde_bandwidth"}
    for block in ("fr_uniform", "fr_estimated"):
        m = cfg["methods"][block]
        for k, v in fr.items():
            assert float(m.get(k, base.get(k))) == float(v), f"FR knob {k} in {block}"
    for k, v in pre["fr_knobs"].items():
        if not k.startswith("_"):
            assert float(fr[k]) == float(v), f"FR knob {k} differs from the accepted prereg"
    for k, v in c["abf_shared"].items():
        assert base[k] == v, f"shared ABF setting {k}: YAML {base[k]} != frozen {v}"
    assert [float(base["z_min"]), float(base["z_max"])] == c["domain"]
    assert [float(base["eval_z_lo"]), float(base["eval_z_hi"])] == c["eval_window"]
    assert os.path.exists(REFERENCE_NPZ), f"accepted TI reference missing: {REFERENCE_NPZ} -- STOP, do not recompute"
    assert c["run"]["n_steps"] == pre["n_steps"] and c["run"]["n_replicas"] == pre["n_replicas"]
    assert kb["readout_bank"] == pre["corrected_baseline"]["readout_bank_bandwidths"]
    return c, base, fr


def make_spec(stage, name, method, seed, c, fr, estimator="kernel", n_bins=0):
    return jobs.PhaseRunSpec(stage=stage, name=name, method=method, seed=int(seed),
                             n_steps=int(c["run"]["n_steps"]), n_replicas=int(c["run"]["n_replicas"]),
                             save_every=int(c["run"]["save_every"]), abf_estimator=estimator, abf_n_bins=int(n_bins),
                             **c["cell"], **fr)


def execute(sp, base, engines, readout, overwrite, raw_dir, verbose=False):
    path = jobs.run_npz_path(raw_dir, sp)
    if not overwrite and jobs.run_is_valid(path):
        print(f"  skip {sp.run_id()}", flush=True)
        return None
    eng = jobs.get_engine(sp, engines)
    t0 = time.time()
    out = jobs.execute_run(sp, base, eng, cache_dir=CACHE, verbose=verbose, store_profiles=True,
                           readout_bandwidths=readout)
    assert "v2" in str(out.get("reference_label", "")), f"unexpected reference {out.get('reference_label')!r}"
    out["git_rev"] = git_rev()
    out["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out["reference_path"] = os.path.relpath(REFERENCE_NPZ, ROOT)
    jobs.save_run(path, out)
    wall = time.time() - t0
    safety = (f" repl={out['total_replacement_events']} essW={out['min_ancestor_ess_window'] / sp.n_replicas:.3f}"
              f" wmax={out['max_ancestor_frac_over_time']:.4f}" if sp.method != "abf" else "")
    print(f"  {sp.name:>16s} seed{sp.seed}: saved{safety} ({wall:.0f}s, sampler {out['runtime_seconds']:.0f}s)", flush=True)
    return dict(run_id=sp.run_id(), name=sp.name, seed=sp.seed, wall_seconds=wall,
                runtime_seconds=float(out["runtime_seconds"]), estimator=sp.abf_estimator, n_bins=sp.abf_n_bins)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="floor,calibration")
    ap.add_argument("--seeds", default=None, help="override seed list (inclusive a-b or comma list); default: campaign")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", default=OUT_ROOT)
    a = ap.parse_args()
    c, base, fr = load_frozen()
    campaign_sha = subprocess.check_output(["sha256sum", CAMPAIGN], text=True).split()[0]
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") == "":
        print("WARNING: CUDA_VISIBLE_DEVICES unset; pin exactly one GPU", flush=True)
    stages = [s.strip() for s in a.stage.split(",")]
    print(f"device={core.DEVICE} git={git_rev()} campaign_sha={campaign_sha[:12]} cell={c['cell']} fr={fr}", flush=True)
    t_all = time.time()

    def seeds_or(default):
        if a.seeds is None:
            return list(default)
        if "-" in a.seeds:
            lo, hi = a.seeds.split("-")
            return list(range(int(lo), int(hi) + 1))
        return [int(x) for x in a.seeds.split(",")]

    if "floor" in stages:
        out = os.path.join(a.out, "floor"); os.makedirs(out, exist_ok=True)
        ref = dict(np.load(REFERENCE_NPZ, allow_pickle=True))
        sim = jobs.build_sim(make_spec("floor", "abf", "abf", 0, c, fr), base)
        rows = []
        for nb in c["ladder_n_bins"]:
            f = core.histogram_p0_floor_np(nb, ref, sim)
            rows.append({k: float(f[k]) for k in ("n_bins", "delta", "floor_l2_f", "floor_l2_fp", "floor_l2_fp_nodes")})
            np.savez_compressed(os.path.join(out, f"floor_nbins{nb}.npz"), **f)
            print(f"  floor n_bins={nb:4d} Delta={f['delta']:.5f}: e_F {f['floor_l2_f']:.5f}  e_F' {f['floor_l2_fp']:.4f}"
                  f"  (nodes {f['floor_l2_fp_nodes']:.4f})", flush=True)
        json.dump(dict(rows=rows, git_rev=git_rev(), reference=os.path.relpath(REFERENCE_NPZ, ROOT),
                       reference_label=str(ref["label"]), eval_window=c["eval_window"], n_avg=64, n_sub=8,
                       quadrature="composite Gauss-Legendre, 8 nodes per bin; reference linear between its nodes"),
                  open(os.path.join(out, "p0_floor.json"), "w"), indent=2)

    if "calibration" in stages:
        cal = c["calibration"]
        seeds = seeds_or(cal["seeds"])
        raw_dir = os.path.join(a.out, "calibration", "raw"); os.makedirs(raw_dir, exist_ok=True)
        engines, runs = {}, []
        print(f"[calibration] histogram ABF only, n_bins {c['ladder_n_bins']}, seeds {seeds}", flush=True)
        for sd in seeds:
            for nb in c["ladder_n_bins"]:
                sp = make_spec("calibration", f"hist{nb}_abf", "abf", sd, c, fr, "histogram", nb)
                m = execute(sp, base, engines, None, a.overwrite, raw_dir, a.verbose)
                if m:
                    runs.append(m)
        prov = dict(stage="calibration", git_rev=git_rev(), campaign_sha=campaign_sha, host=socket.gethostname(),
                    device=str(core.DEVICE), cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                    cell=c["cell"], fr_knobs=fr, run=c["run"], ladder_n_bins=c["ladder_n_bins"], seeds=seeds,
                    reference=os.path.relpath(REFERENCE_NPZ, ROOT), runs=runs, wall_seconds=time.time() - t_all,
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        with open(os.path.join(a.out, "calibration", f"provenance_{seeds[0]}-{seeds[-1]}.json"), "w") as fh:
            json.dump(prov, fh, indent=2, default=float)

    if "confirmation" in stages:
        assert os.path.exists(SELECTED), "no frozen width: run analyze_histogram_abf_wca.py --select first"
        sel = json.load(open(SELECTED))["wca"]
        nb = int(sel["selected_n_bins"])
        conf = c["confirmation"]
        seeds = seeds_or(range(int(conf["seeds_first"]), int(conf["seeds_first"]) + int(conf["n_seeds"])))
        assert not (set(seeds) & set(c["calibration"]["seeds"])), "confirmation seeds overlap calibration"
        raw_dir = os.path.join(a.out, "confirmation", "raw"); os.makedirs(raw_dir, exist_ok=True)
        readout = tuple(float(h) for h in c["kernel_baseline"]["readout_bank"])
        arms = [("kernel_abf", "abf", "kernel", 0, readout), ("kernel_fr_uniform", "fr_uniform", "kernel", 0, readout),
                ("hist_abf", "abf", "histogram", nb, None), ("hist_fr_uniform", "fr_uniform", "histogram", nb, None)]
        engines, runs = {}, []
        print(f"[confirmation] frozen n_bins={nb} (selected {sel.get('selected_at')}); seeds {seeds[0]}-{seeds[-1]}; "
              f"kernel arms carry the read-out bank {readout}", flush=True)
        for sd in seeds:
            for name, method, est, nbins, ro in arms:
                sp = make_spec("confirmation", name, method, sd, c, fr, est, nbins)
                m = execute(sp, base, engines, ro, a.overwrite, raw_dir, a.verbose)
                if m:
                    runs.append(m)
        prov = dict(stage="confirmation", git_rev=git_rev(), campaign_sha=campaign_sha, host=socket.gethostname(),
                    device=str(core.DEVICE), cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                    cell=c["cell"], fr_knobs=fr, run=c["run"], selected=sel, seeds=seeds, arms=[x[0] for x in arms],
                    readout_bank_kernel_arms=list(readout), reference=os.path.relpath(REFERENCE_NPZ, ROOT), runs=runs,
                    wall_seconds=time.time() - t_all, timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        with open(os.path.join(a.out, "confirmation", f"provenance_{seeds[0]}-{seeds[-1]}.json"), "w") as fh:
            json.dump(prov, fh, indent=2, default=float)
    print(f"done in {(time.time() - t_all) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
