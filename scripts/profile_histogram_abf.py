#!/usr/bin/env python
"""Efficiency profiling for the histogram vs kernel ABF estimators (separate from every scored run).

Timed regions per MD step, kernel vs histogram, for both engines:
    physical force evaluation | ABF accumulator update | ABF bias evaluation (mean-force lookup)
    free-energy (PMF) reconstruction | FR marginal KDE + score | total step (the real engine loop)
On CUDA every region is bracketed by torch.cuda.synchronize() -- PROFILING MODE ONLY; the
production loops are untouched.  The "total step" numbers come from the unmodified engine loops
(eb.simulate_batch / wca.run_sampler_gpu) over a short horizon, so they include Python dispatch
exactly as production does.  Peak memory via torch.cuda.max_memory_allocated.

The EB region timings build the SAME batched objects the engine uses (R = seeds x methods rows);
the WCA region timings build the same estimator objects and call the same methods with the
production population size.  No result of this script is a scored metric.

    CUDA_VISIBLE_DEVICES=3 python -u scripts/profile_histogram_abf.py --device cuda --eb-nbins 90 --wca-nbins 80
    python -u scripts/profile_histogram_abf.py --device cpu --eb-nbins 90 --wca-nbins 80 --wca-steps 300
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import replace

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import eb_abffr_core as eb      # noqa: E402
import wca_abffr_core as wca    # noqa: E402
import wca_phase_jobs as jobs   # noqa: E402

OUT = os.path.join(ROOT, "results", "histogram_abf", "profiling")


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def timed(fn, dev, n_rep, n_warm=3):
    for _ in range(n_warm):
        fn()
    _sync(dev)
    t0 = time.perf_counter()
    for _ in range(n_rep):
        fn()
    _sync(dev)
    return (time.perf_counter() - t0) / n_rep


# ------------------------------------------------------------------------------------------ EB
def profile_eb(dev, dtype, n_bins, n_rep, R=4, N=256, n_steps_total=2000):
    out = {}
    x_grid, dx, eval_mask, idx0 = eb.build_grid(dev, dtype)
    cfg = eb.PhysConfig()
    g = torch.Generator(device=dev); g.manual_seed(1)
    X = torch.rand((R, N), device=dev, dtype=dtype, generator=g) * 3.0 - 1.5
    Y = torch.randn((R, N), device=dev, dtype=dtype, generator=g) * 0.3
    beta = torch.full((R, 1), cfg.beta, device=dev, dtype=dtype)
    Hc = torch.full((R, 1), cfg.H, device=dev, dtype=dtype)
    oo = torch.full((R, 1), cfg.omega_out, device=dev, dtype=dtype)
    oi = torch.full((R, 1), cfg.omega_in, device=dev, dtype=dtype)
    sw = torch.full((R, 1), cfg.s, device=dev, dtype=dtype)

    def phys():
        om = eb.omega_of(X, oo, oi, sw); dom = eb.domega_of(X, oo, oi, sw)
        return eb.dU_of(X, Hc) + om * dom * Y * Y, om * om * Y
    fx, _ = phys()
    out["physical_force_s"] = timed(phys, dev, n_rep)
    k_eta, r_eta = eb.gaussian_kernel(cfg.eta, dx, dev, dtype)
    q_uni = torch.full((R, eb.N_GRID), 1.0 / 3.6, device=dev, dtype=dtype)

    def fr_kde_score():
        p = eb.binned_density(X, k_eta, r_eta, dx)
        logp = torch.log(torch.clamp(p, min=eb.EPS)); logq = torch.log(q_uni)
        kl = eb.trapz(p * (logp - logq), dx).unsqueeze(1)
        return torch.log(torch.clamp(eb.interp1d(X, p, dx), min=eb.EPS)) - torch.log(torch.clamp(eb.interp1d(X, q_uni, dx), min=eb.EPS)) - kl
    out["fr_kde_score_s"] = timed(fr_kde_score, dev, n_rep)

    for name in ("kernel", "histogram"):
        r = {}
        if name == "kernel":
            est = eb.KernelABFEstimator(R, cfg.h, cfg.min_count, dx, dev, dtype)
            r["update_s"] = timed(lambda: est.update(X, fx), dev, n_rep)
            Fp = est.mean_force_profile()
            r["mean_force_eval_s"] = timed(lambda: est.evaluate(X, est.mean_force_profile()), dev, n_rep)
            r["mean_force_profile_only_s"] = timed(lambda: est.mean_force_profile(), dev, n_rep)
            r["pmf_reconstruction_s"] = timed(lambda: est.pmf_profile(Fp, idx0), dev, n_rep)
            cfg_run = cfg
        else:
            est = eb.HistogramABFEstimator(R, n_bins, cfg.min_count, x_grid, dev, dtype)
            r["update_s"] = timed(lambda: est.update(X, fx), dev, n_rep)
            r["mean_force_eval_s"] = timed(lambda: est.evaluate(X), dev, n_rep)
            r["mean_force_profile_only_s"] = timed(lambda: est.bin_mean_force(), dev, n_rep)
            r["pmf_reconstruction_s"] = timed(lambda: est.pmf_profile(idx0), dev, n_rep)
            cfg_run = replace(cfg, abf_estimator="histogram", abf_n_bins=n_bins)
        r["estimator_per_step_s"] = r["update_s"] + r["mean_force_eval_s"] + (r["pmf_reconstruction_s"] if name == "kernel" else 0.0)
        # total step from the real engine loop (short horizon), abf + fr_uniform rows
        cfg_short = replace(cfg_run, n_steps=n_steps_total, save_every=400)
        spec = eb.BatchSpec(configs=[cfg_short] * (R // 2), seeds=list(range(R // 2)), methods=[eb.ABF, eb.FR_UNIFORM], batch_seed=3)
        eb.simulate_batch(spec, device=dev, dtype=dtype)      # warm-up
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        _sync(dev); t0 = time.perf_counter()
        eb.simulate_batch(spec, device=dev, dtype=dtype)
        _sync(dev)
        r["total_step_s"] = (time.perf_counter() - t0) / n_steps_total
        r["peak_mem_bytes"] = int(torch.cuda.max_memory_allocated()) if dev.type == "cuda" else None
        out[name] = r
    out["R"], out["N"], out["n_bins"] = R, N, n_bins
    out["estimator_speedup"] = out["kernel"]["estimator_per_step_s"] / out["histogram"]["estimator_per_step_s"]
    out["total_step_speedup"] = out["kernel"]["total_step_s"] / out["histogram"]["total_step_s"]
    return out


# ------------------------------------------------------------------------------------------ WCA
def profile_wca(dev, dtype, n_bins, n_rep, n_steps_total=600):
    out = {}
    c = json.load(open(os.path.join(ROOT, "configs", "histogram_abf", "campaign.json")))["wca"]
    base = jobs.effective_base(jobs.load_yaml(os.path.join(ROOT, "configs", "wca_phase_diagram_production.yaml")), "production")
    fr = {k: v for k, v in c["fr_knobs"].items() if k != "kde_bandwidth"}
    sp = jobs.PhaseRunSpec(stage="profile", name="abf", method="abf", seed=11, n_steps=n_steps_total,
                           n_replicas=int(c["run"]["n_replicas"]), save_every=int(c["run"]["save_every"]), **c["cell"], **fr)
    params = jobs.build_params(sp)
    engine = wca.WCADimerEngine(params, dev, dtype)
    sim = jobs.build_sim(sp, base)
    q = wca.lattice_initial_conditions(params, sim.n_replicas, dev, dtype, seed=11)
    for _ in range(50):     # a few plain steps so the population is not a lattice
        f = wca.clip_forces(engine.force(q), params.force_clip)
        q = wca.wrap_positions(q + sim.dt * f + np.sqrt(2 * sim.dt / params.beta) * torch.randn_like(q), params.box_length)
    grid = torch.linspace(sim.z_min, sim.z_max, sim.n_grid, device=dev, dtype=dtype)
    out["physical_force_s"] = timed(lambda: engine.force(q, compute_energy=False), dev, n_rep)
    forces = wca.clip_forces(engine.force(q), params.force_clip)
    z = wca.reaction_coordinate(q, params)
    f_local = torch.clamp(wca.local_mean_force(q, forces, params), -sim.mean_force_sample_clip, sim.mean_force_sample_clip)
    q_grid = wca.fr_target_uniform_torch(grid)
    out["fr_kde_score_s"] = timed(lambda: wca.fr_score_torch(z, grid, sim, q_grid), dev, n_rep)
    for name in ("kernel", "histogram"):
        r = {}
        simx = replace(sim, abf_estimator=name, abf_n_bins=(n_bins if name == "histogram" else 0))
        est = wca.make_abf_estimator(simx, grid)
        r["update_s"] = timed(lambda: est.update(z, f_local), dev, n_rep)
        r["mean_force_eval_s"] = timed(lambda: est.evaluate(z), dev, n_rep)
        r["pmf_reconstruction_s"] = timed(lambda: est.pmf_profile(), dev, n_rep)
        # the production loop keeps TWO estimators (bias + post-burn-in) and, on the kernel path,
        # rebuilds the PMF every step; the histogram path rebuilds it at saves only (abf / fr_uniform)
        r["estimator_per_step_s"] = 2 * r["update_s"] + r["mean_force_eval_s"] + (r["pmf_reconstruction_s"] if name == "kernel" else 0.0)
        simr = replace(simx, n_steps=n_steps_total, save_every=200, fr_start_steps=200, abf_warmup_steps=100,
                       estimator_burn_in_steps=100)
        wca.run_sampler_gpu("abf", params, simr, engine, initial_q=q.clone(), verbose=False)     # warm-up
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        tot = {}
        for m in ("abf", "fr_uniform"):
            _sync(dev); t0 = time.perf_counter()
            wca.run_sampler_gpu(m, params, simr, engine, initial_q=q.clone(), verbose=False)
            _sync(dev)
            tot[m] = (time.perf_counter() - t0) / n_steps_total
        r["total_step_s"] = tot["abf"]
        r["total_step_fr_uniform_s"] = tot["fr_uniform"]
        r["peak_mem_bytes"] = int(torch.cuda.max_memory_allocated()) if dev.type == "cuda" else None
        out[name] = r
    out["N"], out["n_grid"], out["n_bins"] = sim.n_replicas, sim.n_grid, n_bins
    out["estimator_speedup"] = out["kernel"]["estimator_per_step_s"] / out["histogram"]["estimator_per_step_s"]
    out["total_step_speedup"] = out["kernel"]["total_step_s"] / out["histogram"]["total_step_s"]
    out["total_step_speedup_fr_uniform"] = out["kernel"]["total_step_fr_uniform_s"] / out["histogram"]["total_step_fr_uniform_s"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--eb-nbins", type=int, required=True)
    ap.add_argument("--wca-nbins", type=int, required=True)
    ap.add_argument("--n-rep", type=int, default=200)
    ap.add_argument("--eb-steps", type=int, default=2000)
    ap.add_argument("--wca-steps", type=int, default=600)
    ap.add_argument("--skip-wca", action="store_true")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    dev = torch.device(a.device)
    os.makedirs(a.out, exist_ok=True)
    res = dict(device=str(dev), gpu_name=(torch.cuda.get_device_name(0) if dev.type == "cuda" else None),
               cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""), torch=torch.__version__,
               git_rev=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), n_rep=a.n_rep,
               note="synchronised region timings (profiling mode only); total step = unmodified engine loop over a short horizon")
    print(f"[profile] device {dev}", flush=True)
    res["entropic_bottleneck"] = profile_eb(dev, eb.DTYPE, a.eb_nbins, a.n_rep, n_steps_total=a.eb_steps)
    e = res["entropic_bottleneck"]
    print(f"  EB (R={e['R']}, N={e['N']}, float64): force {1e6 * e['physical_force_s']:.0f} us, FR kde+score {1e6 * e['fr_kde_score_s']:.0f} us")
    for k in ("kernel", "histogram"):
        r = e[k]
        print(f"    {k:9s}: update {1e6 * r['update_s']:.0f} us, eval {1e6 * r['mean_force_eval_s']:.0f} us, pmf {1e6 * r['pmf_reconstruction_s']:.0f} us,"
              f" estimator/step {1e6 * r['estimator_per_step_s']:.0f} us, TOTAL/step {1e6 * r['total_step_s']:.0f} us, peak mem {r['peak_mem_bytes']}")
    print(f"    estimator speedup {e['estimator_speedup']:.2f}x, total-step speedup {e['total_step_speedup']:.2f}x", flush=True)
    if not a.skip_wca:
        res["wca"] = profile_wca(dev, wca.DTYPE if dev.type == "cuda" else torch.float32, a.wca_nbins, a.n_rep, n_steps_total=a.wca_steps)
        w = res["wca"]
        print(f"  WCA (N={w['N']}, float32): force {1e6 * w['physical_force_s']:.0f} us, FR kde+score {1e6 * w['fr_kde_score_s']:.0f} us")
        for k in ("kernel", "histogram"):
            r = w[k]
            print(f"    {k:9s}: update {1e6 * r['update_s']:.0f} us (x2 estimators), eval {1e6 * r['mean_force_eval_s']:.0f} us, pmf {1e6 * r['pmf_reconstruction_s']:.0f} us,"
                  f" estimator/step {1e6 * r['estimator_per_step_s']:.0f} us, TOTAL/step abf {1e6 * r['total_step_s']:.0f} us, fr_uniform {1e6 * r['total_step_fr_uniform_s']:.0f} us, peak mem {r['peak_mem_bytes']}")
        print(f"    estimator speedup {w['estimator_speedup']:.2f}x, total-step speedup abf {w['total_step_speedup']:.2f}x, fr_uniform {w['total_step_speedup_fr_uniform']:.2f}x", flush=True)
    path = os.path.join(a.out, f"profile_{dev.type}.json")
    json.dump(res, open(path, "w"), indent=2, default=float)
    print(f"wrote {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()
