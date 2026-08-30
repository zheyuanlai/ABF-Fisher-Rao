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


def relative_time(t, curve, frac, hold_frac, t0=0.0):
    """First t at which ``curve`` has covered ``1 - frac`` of the way from its
    first POST-WARMUP value to its own late-time plateau, sustained for a FULL
    ``hold_frac * T``.

    Returns (time, J0, J_inf, threshold, status); the time is inf when the
    criterion is never sustained.  Three things here are deliberate:

      * ``J0`` is read at ``t >= t0`` (the end of the ABF warm-up), exactly as
        the preregistration words it.  Reading it at t = 0 instead inflates the
        threshold -- the pre-warmup value is always the largest -- and biases
        T_marg SHORT, i.e. systematically away from the establishment-limited
        verdict this campaign is hunting for (measured: 33 ps vs 148 ps, 0.11 T
        vs 0.49 T, on a plausible TV curve).
      * A curve that gets WORSE (J_inf > J0) has not equilibrated at all.  The
        naive threshold then lands BETWEEN J0 and J_inf and the criterion is
        met at i = 0, reporting "equilibrated at t = 0" for a diverging curve.
        That case is returned as inf with status 'degrading'.
      * The persistence window must fit INSIDE the data.  Letting it truncate at
        the tail makes the criterion easier exactly where the evidence is
        weakest -- one noisy final sample could move the cell between the two
        extreme verdicts -- and it imputes where the preregistration says
        censoring must be reported.
    """
    t = np.asarray(t, float)
    c = np.asarray(curve, float)
    T = t[-1]
    late = c[t >= 0.8 * T]
    late = late[np.isfinite(late)]
    if late.size == 0:
        return float("inf"), float("nan"), float("nan"), float("nan"), "no_data"
    J_inf = float(np.median(late))
    post = np.nonzero((t >= t0) & np.isfinite(c))[0]
    if post.size == 0:
        return float("inf"), float("nan"), J_inf, float("nan"), "no_data"
    J0 = float(c[post[0]])
    if not (J0 > J_inf):
        return float("inf"), J0, J_inf, float("nan"), "degrading"
    thr = J_inf + frac * (J0 - J_inf)
    hold_t = hold_frac * T
    for i in range(post[0], len(t)):
        if t[i] + hold_t > T + 1e-12:          # window would run past the data
            break
        seg = c[(t >= t[i]) & (t <= t[i] + hold_t)]
        if np.isfinite(seg).all() and (seg <= thr).all():
            return float(t[i]), J0, J_inf, float(thr), "ok"
    return float("inf"), J0, J_inf, float(thr), "never_sustained"


def gate_js_series(blocks_xa, ref_xa, min_per_cell):
    """J_gate(t) as a CONDITIONAL divergence, resolved in xi.

    ``blocks_xa`` is (T, n_xi, n_gate) and ``ref_xa`` is (n_xi, n_gate): the
    gate histogram split by which xi sub-bin of the band the sample came from.

    Comparing p(A_gate | |xi| < band) directly would NOT be a conditional
    comparison at all.  That density is a mixture over xi with each ensemble's
    own p(xi | band) as the weights, and <A_gate | xi> varies across the band
    by construction -- the guest opens the ring where it sits in it.  The
    reference's weights come from equally-spaced umbrella windows, plain ABF's
    from its residual marginal, and the FR arm's from a marginal FR has
    DELIBERATELY FLATTENED.  So a pure marginal reshuffle inside the band moves
    the divergence as much as a real 0.25-0.5 sd gate displacement, and the
    stage's falsifiable signature would fire (or fail) for the wrong reason.

    Conditioning on the xi sub-bin and averaging with FIXED weights removes the
    confound: the marginal is exactly what is divided out.
    """
    T, nx, _ = blocks_xa.shape
    out = np.full(T, np.nan)
    ref_n = ref_xa.sum(axis=-1)
    for i in range(T):
        n = blocks_xa[i].sum(axis=-1)
        ok = (n >= min_per_cell) & (ref_n >= min_per_cell)
        if not ok.any():
            continue
        out[i] = float(np.mean(js_divergence(blocks_xa[i][ok], ref_xa[ok])))
    return out


def classify(out, sim, pre, ref_gate_xa=None):
    t = np.asarray(out["times"], float)
    T = float(t[-1])
    r = pre["screen"]
    G = int(sim.n_grid)
    t_warm = sim.abf_warmup_steps * sim.dt
    # coverage is monotone (csum only accumulates), so a "sustained" clause on
    # it is a no-op; the first fully-covered save IS the coverage time
    nvis = np.asarray(out["n_visited_bins"], float).mean(1)
    covered = np.nonzero(nvis >= G)[0]
    T_cover = float(t[covered[0]]) if covered.size else float("inf")

    tv = np.asarray(out["tv_uniform"], float).mean(1)
    T_marg, J0m, Jim, thrm, st_m = relative_time(
        t, tv, r["relative_fraction"], r["hold_frac"], t0=t_warm)
    T_gate, J0g, Jig, thrg, st_g = (float("nan"),) * 4 + ("no_reference",)
    js = None
    if ref_gate_xa is not None:
        blocks = np.asarray(out["gate_hist_block"], float).sum(axis=1)  # (T,nx,na)
        js = gate_js_series(blocks, ref_gate_xa, r["gate_min_samples"])
        T_gate, J0g, Jig, thrg, st_g = relative_time(
            t, js, r["relative_fraction"], r["hold_frac"], t0=t_warm)

    # A gate that NEVER equilibrates must not read as a gate that equilibrated
    # early.  np.isfinite(inf) is False, so testing "not isfinite" as evidence
    # of health routes the strongest possible conditional-limitation signal
    # straight into the neutrality control.
    gate_known = st_g not in ("no_reference", "no_data")
    gate_bad = gate_known and (T_gate > 0.5 * T)          # inf > 0.5T is True
    gate_fast = gate_known and (T_gate < 0.25 * T)
    q = 0.25 * T
    if T_cover > 0.5 * T or nvis[-1] < G:
        verdict = "discovery_limited"
    elif gate_bad:
        verdict = "conditional_limited"
    elif not gate_known:
        verdict = "unclassified_no_gate_reference"
    elif T_cover < q and T_marg < q and gate_fast:
        verdict = "abf_sufficient"
    elif T_cover < q and q <= T_marg <= 0.8 * T and gate_fast:
        verdict = "establishment_limited"
    else:
        verdict = "intermediate"
    return dict(
        T=T, t_warmup=t_warm, T_cover=T_cover, T_marg=T_marg, T_gate=T_gate,
        marg_status=st_m, gate_status=st_g,
        L_marg=(T_marg - T_cover if np.isfinite(T_marg) and np.isfinite(T_cover)
                else float("inf")),
        marg=dict(J0=J0m, J_inf=Jim, threshold=thrm, final=float(tv[-1])),
        gate=dict(J0=J0g, J_inf=Jig, threshold=thrg,
                  final=(float(js[-1]) if js is not None and np.isfinite(js[-1])
                         else None)),
        unvisited_bins=float(G - nvis[-1]),
        transit_events=int(np.asarray(out["cross_gate_samples"]).size),
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
        _r = np.load(ref_path)
        assert bool(_r["accepted"]), \
            "the umbrella reference on disk FAILED its acceptance gates"
        ref_gate = _r["gate_hist_window_xi"].astype(float)
        print(f"  gate reference loaded from {os.path.basename(ref_path)} "
              f"(accepted, {ref_gate.shape[0]} xi sub-bins)")
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
    print(f"  transit events {cls['transit_events']}, unvisited bins "
          f"{cls['unvisited_bins']:.2f}, "
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
