#!/usr/bin/env python
"""ABF-ONLY screen for the ethane/ZIF-8 cell, classified by the FROZEN rules
in configs/uniform_campaign/zif8_prereg.json.  No FR arm exists in this file;
the classification alone licenses (or refuses) the two-arm run.

Three clocks are measured and kept apart -- that separation is the reason this
stage exists:

    T_cover   discovery                 every scoring bin visited
    T_marg    marginal establishment    TV(p_t^xi, uniform) reaches 80% of the
                                        improvement ABF ultimately achieves
    T_gate    conditional equilibration  JS( p_t(A_gate | at the window)
                                        || p_ref(A_gate | at the window) )
                                        reaches the same relative threshold

T_marg and T_gate use a RELATIVE criterion rather than the CHA stage's
absolute TV < 0.10, which needed an amendment because it is beta-naive: a
plateau set by O(1 kJ/mol) estimator residuals is not a failure to establish.

    CUDA_VISIBLE_DEVICES=3 python -u scripts/run_zif8_screen.py --temperature 300
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from zif8.core_zif8 import (ZIF8SimConfig, ZIF8System, engine_kwargs,  # noqa: E402
                            js_divergence, run_sampler)

PREREG = os.path.join(ROOT, "configs/uniform_campaign/zif8_prereg.json")
OUT = os.path.join(ROOT, "results/uniform_campaign/zif8/screen")
REF = os.path.join(ROOT, "results/uniform_campaign/zif8/reference")


def git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:
        return "unknown"


def relative_time(t, curve, frac, hold_frac, high_is_bad=True):
    """First t at which ``curve`` has covered ``1 - frac`` of the way from its
    first value to its own late-time plateau, sustained for ``hold_frac * T``.

    Returns (time, J0, J_inf, threshold); inf if never sustained."""
    t = np.asarray(t, float)
    c = np.asarray(curve, float)
    T = t[-1]
    late = c[t >= 0.8 * T]
    J_inf = float(np.median(late[np.isfinite(late)])) if np.isfinite(late).any() else np.nan
    first = np.argmax(np.isfinite(c))
    J0 = float(c[first])
    thr = J_inf + frac * (J0 - J_inf)
    hold = max(1, int(hold_frac * len(t)))
    hit = float("inf")
    for i in range(len(t)):
        seg = c[i:i + hold]
        if np.isfinite(seg).all() and (seg <= thr if high_is_bad else seg >= thr).all():
            hit = float(t[i]); break
    return hit, J0, J_inf, float(thr)


def classify(out, sim, pre, ref_gate=None):
    t = np.asarray(out["times"], float)
    T = float(t[-1])
    r = pre["screen"]
    G = int(sim.n_grid)
    nvis = np.asarray(out["n_visited_bins"], float).mean(1)
    hold = max(1, int(r["cover_hold_frac"] * len(t)))
    covered = nvis >= G
    T_cover = float("inf")
    for i in range(len(t)):
        if covered[i:i + hold].all():
            T_cover = float(t[i]); break

    tv = np.asarray(out["tv_uniform"], float).mean(1)
    T_marg, J0m, Jim, thrm = relative_time(t, tv, r["relative_fraction"],
                                           r["hold_frac"])
    # J_gate against the umbrella reference conditional at the window
    T_gate, J0g, Jig, thrg = float("nan"), float("nan"), float("nan"), float("nan")
    js = None
    if ref_gate is not None:
        blocks = np.asarray(out["gate_hist_block"], float).sum(axis=1)   # (T, Gg)
        tot = blocks.sum(axis=-1)
        js = np.where(tot > r["gate_min_samples"],
                      js_divergence(blocks, np.broadcast_to(ref_gate, blocks.shape)),
                      np.nan)
        T_gate, J0g, Jig, thrg = relative_time(t, js, r["relative_fraction"],
                                               r["hold_frac"])

    q = 0.25 * T
    if T_cover > 0.5 * T or nvis[-1] < G:
        verdict = "discovery_limited"
    elif np.isfinite(T_gate) and T_gate > 0.5 * T:
        verdict = "conditional_limited"
    elif T_cover < q and T_marg < q and (not np.isfinite(T_gate) or T_gate < q):
        verdict = "abf_sufficient"
    elif T_cover < q and q <= T_marg <= 0.8 * T and \
            (not np.isfinite(T_gate) or T_gate < q):
        verdict = "establishment_limited"
    else:
        verdict = "intermediate"
    return dict(
        T=T, T_cover=T_cover, T_marg=T_marg, T_gate=T_gate,
        L_marg=(T_marg - T_cover if np.isfinite(T_marg) and np.isfinite(T_cover)
                else float("inf")),
        marg=dict(J0=J0m, J_inf=Jim, threshold=thrm, final=float(tv[-1])),
        gate=dict(J0=J0g, J_inf=Jig, threshold=thrg,
                  final=(float(js[-1]) if js is not None and np.isfinite(js[-1])
                         else None)),
        unvisited_bins=int(G - nvis[-1]),
        transits=int(np.asarray(out["n_crossings"]).sum()),
        frac_window_final=float(np.asarray(out["frac_window"])[-1].mean()),
        gate_mean_final=float(np.asarray(out["gate_mean"])[-1].mean()),
        verdict=verdict,
        js_series=(js.tolist() if js is not None else None),
        tv_series=tv.tolist(), times=t.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--temperature", type=float, default=300.0)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--n-replicas", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=None)
    ap.add_argument("--tag-suffix", default="")
    a = ap.parse_args()
    pre = json.load(open(PREREG))
    tag = f"T{a.temperature:g}{a.tag_suffix}"
    os.makedirs(OUT, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ek = engine_kwargs(pre)
    if a.chunk:
        ek["chunk"] = a.chunk
    system = ZIF8System(a.temperature, device, root=ROOT, **ek)
    print(f"  engine: dtype {ek['dtype']}, force kernel "
          f"{ek['force_dtype'] or ek['dtype']}, chunk {ek['chunk']}")
    s = {k: v for k, v in pre["sampler"].items() if not k.startswith("_")}
    sim = ZIF8SimConfig(**s, rng_seed=pre["screen"]["rng_seed"])
    if a.steps:
        sim.n_steps = a.steps
    if a.n_replicas:
        sim.n_replicas = a.n_replicas
    seeds = list(range(pre["screen"]["n_seeds"]))
    pool = os.path.join(ROOT, f"cache/zif8/init_pool_T{a.temperature:g}.npz")
    ref_path = os.path.join(REF, f"reference_T{a.temperature:g}.npz")
    ref_gate = None
    if os.path.exists(ref_path):
        ref_gate = np.load(ref_path)["gate_hist_window"].astype(float)
        print(f"  gate reference loaded from {os.path.basename(ref_path)}")
    else:
        print("  NOTE: no umbrella reference yet -- T_gate will not be computed "
              "and the classifier will not be able to see a conditional-limited "
              "cell.  Re-run the classification once the reference exists.")

    print(f"ZIF-8 ABF-only screen {tag}: {len(seeds)} labels x {sim.n_replicas} "
          f"replicas, {sim.n_steps} steps ({sim.n_steps*sim.dt:.0f} ps)", flush=True)
    out = run_sampler("abf", system, sim, seeds=seeds, init_pool=pool,
                      verbose=True, progress_every=10)
    cls = classify(out, sim, pre, ref_gate)
    T = cls["T"]
    print(f"  T_cover = {cls['T_cover']:.1f} ps ({cls['T_cover']/T:.2f} T)")
    print(f"  T_marg  = {cls['T_marg']:.1f} ps ({cls['T_marg']/T:.2f} T)   "
          f"TV {cls['marg']['J0']:.3f} -> {cls['marg']['J_inf']:.3f} "
          f"(threshold {cls['marg']['threshold']:.3f})")
    print(f"  T_gate  = {cls['T_gate']:.1f} ps   JS {cls['gate']['J0']} -> "
          f"{cls['gate']['J_inf']}")
    print(f"  transits {cls['transits']}, unvisited bins {cls['unvisited_bins']}, "
          f"A_gate final {cls['gate_mean_final']:.3f} A")
    print(f"  VERDICT: {cls['verdict']}")
    np.savez_compressed(os.path.join(OUT, f"screen_{tag}.npz"),
                        **{k: v for k, v in out.items()
                           if isinstance(v, (np.ndarray, np.generic, int, float, str))})
    with open(os.path.join(OUT, f"screen_{tag}.json"), "w") as fh:
        json.dump(dict(cls, tag=tag, seeds=seeds, n_replicas=sim.n_replicas,
                       n_steps=sim.n_steps, dt=sim.dt,
                       rng_seed=pre["screen"]["rng_seed"],
                       used_gate_reference=ref_gate is not None,
                       config_hash=sim.config_hash(), git_rev=git_rev(),
                       host=socket.gethostname()), fh, indent=2, default=float)
    print(f"  wrote screen_{tag}.npz/.json")


if __name__ == "__main__":
    main()
