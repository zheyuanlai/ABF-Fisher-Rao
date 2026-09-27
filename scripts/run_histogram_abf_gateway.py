#!/usr/bin/env python
"""Histogram (P0) ABF estimator campaign, entropic gateway (Experiment C; the accepted corrected-baseline
cell of docs/GATEWAY_CORRECTED_BASELINE.md).

Frozen design: the `gateway` block of configs/histogram_abf/campaign.json, asserted here against
results/gateway_anchor/CONFIRMATORY_PREREGISTRATION.json (sampler) and
configs/information_campaign/gateway_corrected_confirmation_prereg.json (rate, h_bias, h_read*).

Stages
------
  floor         deterministic P0 floor for every ladder width from the analytic reference (no simulation)
  calibration   ABF ONLY: one batch per width over (init x seed) rows, same batch_seed and row order
                (paired noise); a kernel ABF batch alongside as context (not used by the rule)
  confirmation  kernel batch [abf, fr_uniform gamma 1.5] and histogram batch at the FROZEN width
                (configs/histogram_abf/selected_bins.json, written by analyze_histogram_abf_gateway.py --select),
                same batch_seed and row order

Every batch records the fine-grid accumulators (store_accumulators) so the kernel arms can be scored at the
accepted h_read* = 0.0175 offline, exactly as in the corrected-baseline confirmation.

    CUDA_VISIBLE_DEVICES=3 python -u scripts/run_histogram_abf_gateway.py --stage floor,calibration
    python scripts/analyze_histogram_abf_gateway.py --select
    CUDA_VISIBLE_DEVICES=3 python -u scripts/run_histogram_abf_gateway.py --stage confirmation
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
import eb_abffr_core as eb      # noqa: E402
import gateway_core as gw       # noqa: E402
from run_histogram_abf_entropic import save_batch, batch_is_valid, git_rev   # noqa: E402

CAMPAIGN = os.path.join(ROOT, "configs", "histogram_abf", "campaign.json")
SELECTED = os.path.join(ROOT, "configs", "histogram_abf", "selected_bins.json")
BASE_PREREG = os.path.join(ROOT, "results", "gateway_anchor", "CONFIRMATORY_PREREGISTRATION.json")
CORR_PREREG = os.path.join(ROOT, "configs", "information_campaign", "gateway_corrected_confirmation_prereg.json")
OUT_ROOT = os.path.join(ROOT, "results", "histogram_abf", "gateway")


def load_frozen():
    c = json.load(open(CAMPAIGN))["gateway"]
    base, corr = json.load(open(BASE_PREREG)), json.load(open(CORR_PREREG))
    for k, v in c["sampler"].items():
        assert float(base["sampler"][k]) == float(v), f"sampler {k}: frozen {v} vs prereg {base['sampler'][k]}"
    for k in ("beta", "s", "r", "beta_H_kT"):
        assert float(base["cell"][k]) == float(c["cell"][k]) == float(corr["cell"][k]), k
    assert float(corr["rate"]["gamma"]) == float(c["fr_uniform"]["gamma"])
    assert float(corr["corrected_baseline"]["h_bias"]) == float(c["kernel_baseline"]["h_bias"]) == float(c["sampler"]["h"])
    assert float(corr["corrected_baseline"]["h_read_star"]) == float(c["kernel_baseline"]["h_read_star"])
    assert c["inits"] == base["inits"]
    assert [round(3.6 / n, 6) for n in c["ladder_n_bins"]] == [round(d, 6) for d in c["ladder_delta"]]
    return c


def build_config(c, init, estimator="kernel", n_bins=0):
    s, cell = c["sampler"], c["cell"]
    return gw.GatewayConfig(
        beta=cell["beta"], H=cell["beta_H_kT"] / cell["beta"], omega_out=cell["omega_out"], r=cell["r"], s=cell["s"],
        N=int(s["N"]), dt=s["dt"], n_steps=int(s["n_steps"]), save_every=int(s["save_every"]), init=init,
        h=float(s["h"]), min_count=s["min_count"], gamma=float("nan"), eta=s["eta"], fr_every=int(s["fr_every"]),
        fr_burnin=int(s["fr_burnin"]), ramp_fraction=s["ramp_fraction"], target_ema_rate=s["target_ema_rate"],
        score_clip=s["score_clip"], max_event_fraction=s["max_event_fraction"], ess_window_steps=int(s["ess_window_steps"]),
        estimator=estimator, n_bins=int(n_bins))


def run_batch(c, seeds, methods, batch_seed, estimator, n_bins, path, label, overwrite, campaign_sha):
    if not overwrite and batch_is_valid(path):
        print(f"  skip {label} (complete)", flush=True)
        return None
    rows = [(init, sd) for init in c["inits"] for sd in seeds]
    cfgs = [build_config(c, init, estimator, n_bins) for init, _ in rows]
    spec = gw.BatchSpec(configs=cfgs, seeds=[sd for _, sd in rows], methods=list(methods), batch_seed=int(batch_seed))
    if torch.cuda.is_available():
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    recs = gw.simulate_batch(spec, store_accumulators=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    c0 = cfgs[0]
    meta = dict(label=label, git_rev=git_rev(), campaign_sha=campaign_sha, timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                host=socket.gethostname(), device=str(gw.DEVICE), dtype=str(gw.DTYPE), cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                config=dataclasses.asdict(c0), estimator=estimator, n_bins=int(n_bins), delta=(3.6 / n_bins if n_bins else None),
                hist_edges=(np.linspace(eb.XMIN, eb.XMAX, int(n_bins) + 1).tolist() if n_bins else None),
                rows=[dict(init=i, seed=int(sd)) for i, sd in rows], seeds=list(seeds), inits=c["inits"],
                methods=[m.name for m in methods], gamma=[float(m.gamma) for m in methods], batch_seed=int(batch_seed),
                wall_seconds=wall, wall_seconds_per_step=wall / c0.n_steps, R=len(recs), n_steps=c0.n_steps,
                peak_gpu_mem_bytes=(int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None),
                reference="analytic F_ref = H(x^2-1)^2 + log(omega)/beta, omega_in = r omega_out (gateway_core via eb_abffr_core.reference_profiles)",
                kernel_baseline=c["kernel_baseline"], fr_parameters=dict(c["sampler"], gamma=c["fr_uniform"]["gamma"]))
    save_batch(path, recs, meta)
    print(f"  {label}: R={len(recs)} ({len(rows)} rows x {len(methods)} methods) in {wall:.1f}s ({1e3 * wall / c0.n_steps:.3f} ms/step)", flush=True)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="floor,calibration")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--out", default=OUT_ROOT)
    a = ap.parse_args()
    c = load_frozen()
    campaign_sha = subprocess.check_output(["sha256sum", CAMPAIGN], text=True).split()[0]
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") == "":
        print("WARNING: CUDA_VISIBLE_DEVICES unset", flush=True)
    if torch.cuda.is_available():
        assert torch.cuda.device_count() == 1, "pin exactly one GPU"
    stages = [s.strip() for s in a.stage.split(",")]
    print(f"device={gw.DEVICE} git={git_rev()} campaign_sha={campaign_sha[:12]} cell={c['cell']} gamma={c['fr_uniform']['gamma']}", flush=True)
    t_all = time.time()
    abf = gw.ABF
    fr = dataclasses.replace(gw.FR_UNIFORM, gamma=float(c["fr_uniform"]["gamma"]))

    if "floor" in stages:
        out = os.path.join(a.out, "floor"); os.makedirs(out, exist_ok=True)
        x_grid, dx, eval_mask, idx0 = eb.build_grid(gw.DEVICE, gw.DTYPE)
        cell = c["cell"]
        t = lambda v: torch.tensor([[v]], device=gw.DEVICE, dtype=gw.DTYPE)
        rows = []
        for nb in c["ladder_n_bins"]:
            f = eb.p0_projection_floor(nb, x_grid, eval_mask, idx0, t(cell["beta"]), t(cell["beta_H_kT"] / cell["beta"]),
                                       t(cell["omega_out"]), t(cell["omega_in"]), t(cell["s"]), device=gw.DEVICE, dtype=gw.DTYPE)
            rows.append({k: f[k] for k in ("n_bins", "delta", "floor_l2_f", "floor_l2_fp", "floor_l2_fp_nodes")})
            np.savez_compressed(os.path.join(out, f"floor_nbins{nb}.npz"), **f)
            print(f"  floor n_bins={nb:4d} Delta={f['delta']:.3f}: e_F {f['floor_l2_f']:.5f}  e_F' {f['floor_l2_fp']:.4f}  (nodes {f['floor_l2_fp_nodes']:.4f})", flush=True)
        json.dump(dict(rows=rows, git_rev=git_rev(), cell=cell, eval_window=[eb.EVAL_LO, eb.EVAL_HI], n_avg=64, n_sub=8,
                       quadrature="composite Gauss-Legendre, 8 nodes per bin"), open(os.path.join(out, "p0_floor.json"), "w"), indent=2)

    if "calibration" in stages:
        cal = c["calibration"]
        out = os.path.join(a.out, "calibration", "raw"); os.makedirs(out, exist_ok=True)
        assert cal["arms"] == ["abf"]
        print(f"[calibration] ABF only, seeds {cal['seeds']} x {c['inits']}, batch_seed {cal['batch_seed']}", flush=True)
        metas = []
        for nb in c["ladder_n_bins"]:
            m = run_batch(c, cal["seeds"], [abf], cal["batch_seed"], "histogram", nb, os.path.join(out, f"hist_nbins{nb}.npz"),
                          f"hist n_bins={nb}", a.overwrite, campaign_sha)
            if m:
                metas.append(m)
        m = run_batch(c, cal["seeds"], [abf], cal["batch_seed"], "kernel", 0, os.path.join(out, "kernel.npz"),
                      "kernel h=0.07 (context only)", a.overwrite, campaign_sha)
        if m:
            metas.append(m)
        with open(os.path.join(a.out, "calibration", f"provenance_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"), "w") as fh:
            json.dump(dict(stage="calibration", git_rev=git_rev(), campaign_sha=campaign_sha, batches=metas, wall_seconds=time.time() - t_all), fh, indent=2, default=float)

    if "confirmation" in stages:
        assert os.path.exists(SELECTED), "no frozen width: run analyze_histogram_abf_gateway.py --select first"
        sel = json.load(open(SELECTED))["gateway"]
        nb = int(sel["selected_n_bins"])
        conf = c["confirmation"]
        seeds = list(range(int(conf["seeds_first"]), int(conf["seeds_first"]) + int(conf["n_seeds"])))
        assert not (set(seeds) & set(c["calibration"]["seeds"]))
        out = os.path.join(a.out, "confirmation", "raw"); os.makedirs(out, exist_ok=True)
        print(f"[confirmation] frozen n_bins={nb} (Delta {3.6 / nb:.3f}, selected {sel.get('selected_at')}); seeds {seeds[0]}-{seeds[-1]} x {c['inits']}; batch_seed {conf['batch_seed']}", flush=True)
        metas = []
        for label, est, nbins, fn in (("kernel", "kernel", 0, "kernel.npz"), ("histogram", "histogram", nb, f"hist_nbins{nb}.npz")):
            m = run_batch(c, seeds, [abf, fr], conf["batch_seed"], est, nbins, os.path.join(out, fn), f"{label} [abf, fr_uniform]", a.overwrite, campaign_sha)
            if m:
                metas.append(m)
        with open(os.path.join(a.out, "confirmation", f"provenance_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"), "w") as fh:
            json.dump(dict(stage="confirmation", git_rev=git_rev(), campaign_sha=campaign_sha, selected=sel, batches=metas, wall_seconds=time.time() - t_all), fh, indent=2, default=float)
    print(f"done in {time.time() - t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
