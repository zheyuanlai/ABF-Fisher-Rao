#!/usr/bin/env python
"""Histogram (P0) ABF estimator campaign, entropic bottleneck (Case II base regime).

Frozen design: configs/histogram_abf/campaign.json (read here, asserted against the engine
defaults).  Doc: docs/HISTOGRAM_ABF_REPLICATION.md.

Stages
------
  floor        deterministic P0 projection floor for every ladder width (no simulation)
  calibration  ABF ONLY, one batch per ladder width on the calibration seeds (same batch_seed and
               row order for every width -> paired on identical Langevin noise); a kernel ABF batch
               at the same batch_seed is run as context (NOT used by the selection rule)
  confirmation four arms on the confirmation seeds: one kernel batch [abf, fr_uniform] and one
               histogram batch [abf, fr_uniform] at the FROZEN width
               (configs/histogram_abf/selected_bins.json, written by analyze_histogram_abf_entropic.py
               --select), same batch_seed and row order

Every batch is saved as one npz (per-row records) + a provenance json.  A completed raw file is
never overwritten unless --overwrite.  The runner prints wall time only; no error metric of an FR
arm is printed before the analyzer runs.

    CUDA_VISIBLE_DEVICES=3 python -u scripts/run_histogram_abf_entropic.py --stage floor,calibration
    python scripts/analyze_histogram_abf_entropic.py --select          # freezes selected_bins.json
    CUDA_VISIBLE_DEVICES=3 python -u scripts/run_histogram_abf_entropic.py --stage confirmation
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import asdict, replace

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import eb_abffr_core as eb  # noqa: E402

CAMPAIGN = os.path.join(ROOT, "configs", "histogram_abf", "campaign.json")
SELECTED = os.path.join(ROOT, "configs", "histogram_abf", "selected_bins.json")
OUT_ROOT = os.path.join(ROOT, "results", "histogram_abf", "entropic_bottleneck")


def git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def load_campaign():
    c = json.load(open(CAMPAIGN))["entropic_bottleneck"]
    base = eb.PhysConfig()
    for k, v in c["physical"].items():
        assert getattr(base, k) == v, f"engine default {k}={getattr(base, k)} != frozen {v}"
    fr = c["fr_uniform"]
    for k, v in fr.items():
        assert getattr(base, k) == v, f"engine default {k}={getattr(base, k)} != frozen {v}"
    assert base.h == c["kernel_baseline"]["h"] and base.min_count == c["kernel_baseline"]["min_count"]
    assert base.abf_estimator == "kernel" and base.abf_n_bins == 0
    assert [round(3.6 / n, 6) for n in c["ladder_n_bins"]] == [round(d, 6) for d in c["ladder_delta"]]
    return c, base


def save_batch(path, recs, meta):
    """One npz per batch: row i's record keys become 'r{i}/<key>'; config/meta as json strings."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    flat = {"meta_json": np.array(json.dumps(meta, default=float)), "n_rows": np.array(len(recs))}
    for i, rec in enumerate(recs):
        for k, v in rec.items():
            if k == "config":
                flat[f"r{i}/config_json"] = np.array(json.dumps(v))
            elif v is None:
                continue
            else:
                flat[f"r{i}/{k}"] = np.asarray(v)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **flat)
    os.replace(tmp, path)


def batch_is_valid(path):
    if not os.path.exists(path):
        return False
    try:
        with np.load(path, allow_pickle=True) as d:
            n = int(d["n_rows"])
            return all(np.isfinite(float(d[f"r{i}/final_l2_f"])) for i in range(n))
    except Exception:
        return False


def run_batch(cfg, seeds, methods, batch_seed, path, label, overwrite, campaign_sha):
    if not overwrite and batch_is_valid(path):
        print(f"  skip {label} (complete)", flush=True)
        return None
    spec = eb.BatchSpec(configs=[cfg] * len(seeds), seeds=list(seeds), methods=list(methods),
                        batch_seed=int(batch_seed))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    recs = eb.simulate_batch(spec)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    R = len(recs)
    meta = dict(label=label, git_rev=git_rev(), campaign_sha=campaign_sha, timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                host=socket.gethostname(), device=str(eb.DEVICE), dtype=str(eb.DTYPE),
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                config=asdict(cfg), estimator=cfg.abf_estimator, n_bins=cfg.abf_n_bins,
                delta=(3.6 / cfg.abf_n_bins if cfg.abf_n_bins else None),
                hist_edges=(np.linspace(eb.XMIN, eb.XMAX, cfg.abf_n_bins + 1).tolist() if cfg.abf_n_bins else None),
                seeds=list(seeds), methods=[m.name for m in methods], batch_seed=int(batch_seed),
                wall_seconds=wall, wall_seconds_per_step=wall / cfg.n_steps, R=R, n_steps=cfg.n_steps,
                peak_gpu_mem_bytes=(int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None),
                reference="analytic F_ref = H(x^2-1)^2 + log(omega)/beta (eb_abffr_core.reference_profiles)",
                fr_parameters={k: getattr(cfg, k) for k in ("gamma", "eta", "fr_every", "fr_burnin", "ramp_fraction",
                                                            "score_clip", "max_event_fraction", "target_ema_rate")})
    save_batch(path, recs, meta)
    print(f"  {label}: R={R} ({len(seeds)} seeds x {len(methods)} methods) in {wall:.1f}s "
          f"({1e3 * wall / cfg.n_steps:.3f} ms/step)", flush=True)
    return meta


def stage_floor(c, base, out):
    x_grid, dx, eval_mask, idx0 = eb.build_grid(eb.DEVICE, eb.DTYPE)
    t = lambda v: torch.tensor([[v]], device=eb.DEVICE, dtype=eb.DTYPE)
    rows = []
    for nb in c["ladder_n_bins"]:
        f = eb.p0_projection_floor(nb, x_grid, eval_mask, idx0, t(base.beta), t(base.H), t(base.omega_out),
                                   t(base.omega_in), t(base.s))
        rows.append({k: f[k] for k in ("n_bins", "delta", "floor_l2_f", "floor_l2_fp", "floor_l2_fp_nodes")})
        np.savez_compressed(os.path.join(out, f"floor_nbins{nb}.npz"), **f)
        print(f"  floor n_bins={nb:4d} Delta={f['delta']:.3f}: e_F {f['floor_l2_f']:.5f}  "
              f"e_F' {f['floor_l2_fp']:.4f}  (nodes {f['floor_l2_fp_nodes']:.4f})", flush=True)
    json.dump(dict(rows=rows, git_rev=git_rev(), eval_window=[eb.EVAL_LO, eb.EVAL_HI], n_avg=64, n_sub=8,
                   quadrature="composite Gauss-Legendre, 8 nodes per bin"),
              open(os.path.join(out, "p0_floor.json"), "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="floor,calibration", help="comma list of floor,calibration,confirmation")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--out", default=OUT_ROOT)
    a = ap.parse_args()
    c, base = load_campaign()
    campaign_sha = subprocess.check_output(["sha256sum", CAMPAIGN], text=True).split()[0]
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") == "":
        print("WARNING: CUDA_VISIBLE_DEVICES unset", flush=True)
    print(f"device={eb.DEVICE} dtype={eb.DTYPE} git={git_rev()} campaign_sha={campaign_sha[:12]}", flush=True)
    stages = [s.strip() for s in a.stage.split(",")]
    t_all = time.time()

    if "floor" in stages:
        out = os.path.join(a.out, "floor"); os.makedirs(out, exist_ok=True)
        print("[floor] deterministic P0 projection floors", flush=True)
        stage_floor(c, base, out)

    if "calibration" in stages:
        cal = c["calibration"]
        out = os.path.join(a.out, "calibration", "raw"); os.makedirs(out, exist_ok=True)
        assert cal["arms"] == ["abf"]
        methods = [eb.ABF]
        print(f"[calibration] ABF only, seeds {cal['seeds']}, batch_seed {cal['batch_seed']}", flush=True)
        metas = []
        for nb in c["ladder_n_bins"]:
            cfg = replace(base, abf_estimator="histogram", abf_n_bins=int(nb))
            m = run_batch(cfg, cal["seeds"], methods, cal["batch_seed"], os.path.join(out, f"hist_nbins{nb}.npz"),
                          f"hist n_bins={nb}", a.overwrite, campaign_sha)
            if m:
                metas.append(m)
        m = run_batch(base, cal["seeds"], methods, cal["batch_seed"], os.path.join(out, "kernel.npz"),
                      "kernel h=0.07 (context only)", a.overwrite, campaign_sha)
        if m:
            metas.append(m)
        with open(os.path.join(a.out, "calibration", f"provenance_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"), "w") as fh:
            json.dump(dict(stage="calibration", git_rev=git_rev(), campaign_sha=campaign_sha, batches=metas,
                           wall_seconds=time.time() - t_all), fh, indent=2, default=float)

    if "diag_kernel_h" in stages:
        # POST-HOC DIAGNOSTIC (not part of the frozen design, added after the A1 result): kernel ABF
        # only, at other online bandwidths, on the calibration seeds / batch_seed -- is the frontier
        # stall seen at h = 0.07 a property of the kernel estimator at any bandwidth?
        cal = c["calibration"]
        out = os.path.join(a.out, "diagnostics", "kernel_h", "raw"); os.makedirs(out, exist_ok=True)
        metas = []
        for h in (0.14, 0.035, 0.0175):
            m = run_batch(replace(base, h=h), cal["seeds"], [eb.ABF], cal["batch_seed"], os.path.join(out, f"kernel_h{h:g}.npz"),
                          f"kernel h={h:g} (post-hoc diagnostic)", a.overwrite, campaign_sha)
            if m:
                metas.append(m)
        with open(os.path.join(a.out, "diagnostics", "kernel_h", "provenance.json"), "w") as fh:
            json.dump(dict(stage="diag_kernel_h", post_hoc=True, git_rev=git_rev(), batches=metas), fh, indent=2, default=float)

    if "confirmation" in stages:
        assert os.path.exists(SELECTED), "no frozen width: run analyze_histogram_abf_entropic.py --select first"
        sel = json.load(open(SELECTED))["entropic_bottleneck"]
        nb = int(sel["selected_n_bins"])
        conf = c["confirmation"]
        seeds = list(range(int(conf["seeds_first"]), int(conf["seeds_first"]) + int(conf["n_seeds"])))
        assert not (set(seeds) & set(c["calibration"]["seeds"])), "confirmation seeds overlap calibration"
        out = os.path.join(a.out, "confirmation", "raw"); os.makedirs(out, exist_ok=True)
        methods = [eb.ABF, eb.FR_UNIFORM]
        print(f"[confirmation] frozen n_bins={nb} (Delta {3.6 / nb:.3f}, selected {sel.get('selected_at')}), "
              f"seeds {seeds[0]}-{seeds[-1]}, batch_seed {conf['batch_seed']}", flush=True)
        metas = []
        for label, cfg, fn in (("kernel", base, "kernel.npz"),
                               ("histogram", replace(base, abf_estimator="histogram", abf_n_bins=nb), f"hist_nbins{nb}.npz")):
            m = run_batch(cfg, seeds, methods, conf["batch_seed"], os.path.join(out, fn), f"{label} [abf, fr_uniform]",
                          a.overwrite, campaign_sha)
            if m:
                metas.append(m)
        with open(os.path.join(a.out, "confirmation", f"provenance_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"), "w") as fh:
            json.dump(dict(stage="confirmation", git_rev=git_rev(), campaign_sha=campaign_sha, selected=sel,
                           batches=metas, wall_seconds=time.time() - t_all), fh, indent=2, default=float)
    print(f"done in {time.time() - t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
