"""IO-ABF transfer campaign runner.

Frozen protocol: ``docs/IO_ABF_OVERNIGHT_PREREGISTRATION.md``.

Four phases per system, in this order and no other::

    probe          A0 only, dense cadence   -> rule R-OBS fixes obs_every
    calibration    A0 only, 16 seeds        -> freezes eps1, eps2 and R_Gamma
    pilot          8 paired seeds, 3 arms   -> implementation check only
    confirmatory   32 fresh paired seeds    -> the endpoint

The phases are separate invocations on purpose: the thresholds have to be
written to disk and re-read, so that a later phase *cannot* see a candidate
result while choosing them.  A phase refuses to start if the artifact it depends
on is missing.

Usage
-----
  CUDA_VISIBLE_DEVICES=1 python -u scripts/run_io_abf_campaign.py \
      --system eb_beta8 --phase probe
  ... --phase calibration ; ... --phase pilot ; ... --phase confirmatory
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import eb_abffr_core as eb                                       # noqa: E402
import gateway_core as gw                                        # noqa: E402
from abffr import io_abf                                         # noqa: E402

OUT_ROOT = os.path.join(ROOT, "results", "io_abf_overnight")

# --------------------------------------------------------------------------- #
# seed blocks -- disjoint by construction, and disjoint from the accepted
# gateway campaign's 0-15 (calibration) / 100-131 (confirmatory)
# --------------------------------------------------------------------------- #
SEEDS = {
    "probe": [900, 901],
    "calibration": list(range(16)),
    "pilot": list(range(200, 208)),
    "confirmatory": list(range(300, 332)),
}

#: R-OBS probe cadence and depth.  The probe samples *every step*, because it
#: cannot resolve a correlation time shorter than its own interval: the first
#: attempt on the entropic bottleneck sampled every 5 steps and came back with
#: "lag 1", which says only that tau is somewhere below 5 steps and is not a
#: measurement.  A rule that returns its own floor has not been applied.
PROBE_OBS_EVERY = 1
PROBE_CAPACITY = 4000


# --------------------------------------------------------------------------- #
# the systems
# --------------------------------------------------------------------------- #
def eb_config(beta):
    """The accepted entropic-bottleneck setting, with beta as the only change."""
    return replace(eb.PhysConfig(), beta=float(beta))


def gateway_config(init="left"):
    """The accepted gateway cell, read from its own frozen preregistration."""
    with open(os.path.join(ROOT, "results", "gateway_anchor",
                           "CONFIRMATORY_PREREGISTRATION.json")) as fh:
        pre = json.load(fh)
    c, s = pre["cell"], pre["sampler"]
    return gw.GatewayConfig(
        beta=c["beta"], H=c["beta_H_kT"] / c["beta"], omega_out=1.0,
        r=c["r"], s=c["s"], N=s["N"], dt=s["dt"], n_steps=s["n_steps"],
        save_every=s["save_every"], init=init, h=s["h"],
        min_count=s["min_count"], gamma=pre["rates"]["fr_estimated"],
        eta=s["eta"], fr_every=s["fr_every"], fr_burnin=s["fr_burnin"],
        ramp_fraction=s["ramp_fraction"],
        target_ema_rate=s["target_ema_rate"], score_clip=s["score_clip"],
        max_event_fraction=s["max_event_fraction"],
        ess_window_steps=s["ess_window_steps"])


SYSTEMS = {
    # role: 'control' where the repository already classifies the cell as
    # ABF-sufficient, 'candidate' where it classifies it as establishment-limited
    "eb_beta4":  dict(engine="eb", cfg=lambda: eb_config(4.0),  role="control"),
    "eb_beta8":  dict(engine="eb", cfg=lambda: eb_config(8.0),  role="candidate"),
    "gateway":   dict(engine="gw", cfg=lambda: gateway_config("left"),
                      role="candidate"),
}


def engine_of(system):
    return eb if SYSTEMS[system]["engine"] == "eb" else gw


def io_methods(system):
    e = engine_of(system)
    return list(e.IO_METHODS), list(e.IO_ARMS)


def sysdir(system, *parts):
    return os.path.join(OUT_ROOT, system, *parts)


# --------------------------------------------------------------------------- #
# io config assembly -- structural, plus the one R-OBS value
# --------------------------------------------------------------------------- #
def build_io_cfg(cfg, obs_every_override=None):
    cad = io_abf.cadence_for_run(cfg.n_steps,
                                 obs_every_override=obs_every_override)
    return io_abf.IOConfig(n_cells=io_abf.cells_for_walkers(cfg.N), **cad)


def read_obs_every(system):
    path = sysdir(system, "probe", "r_obs.json")
    if not os.path.exists(path):
        raise SystemExit(
            f"[{system}] no R-OBS probe on disk. Run --phase probe first: the "
            f"observation cadence is a measurement design decision and must be "
            f"fixed before any candidate arm runs.")
    with open(path) as fh:
        return int(json.load(fh)["obs_every"])


# --------------------------------------------------------------------------- #
# running
# --------------------------------------------------------------------------- #
def run_chunk(system, cfg, seeds, methods, arms, io_cfg, device, keep_series=False,
              progress=None):
    e = engine_of(system)
    spec = e.BatchSpec(configs=[cfg] * len(seeds), seeds=list(seeds),
                       methods=methods)
    io = eb.IOSpec(arms=arms, cfg=io_cfg, keep_series=keep_series)
    return e.simulate_batch(spec, device=device, dtype=torch.float64, io=io,
                            progress=progress)


def save_records(outdir, recs, tag):
    os.makedirs(outdir, exist_ok=True)
    for rec in recs:
        arm = rec.get("io_arm", "A0")
        path = os.path.join(outdir, f"{arm}__seed{int(rec['seed']):04d}.npz")
        flat = {}
        for k, v in rec.items():
            if k == "config":
                for ck, cv in v.items():
                    flat[f"cfg__{ck}"] = cv
            elif v is None:
                flat[f"none__{k}"] = True
            elif isinstance(v, (str, bool)):
                flat[k] = np.array(v)
            else:
                flat[k] = v
        np.savez_compressed(path, **flat)
    print(f"  wrote {len(recs)} records to {os.path.relpath(outdir, ROOT)}")


def provenance(extra=None):
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain", "src", "scripts", "tests"],
            cwd=ROOT, text=True).strip())
    except Exception:                                        # pragma: no cover
        commit, dirty = "unknown", True
    out = dict(commit=commit, tree_dirty=dirty, host=socket.gethostname(),
               cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
               torch=torch.__version__, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                          time.gmtime()))
    if extra:
        out.update(extra)
    return out


# --------------------------------------------------------------------------- #
# phase: probe (rule R-OBS)
# --------------------------------------------------------------------------- #
def phase_probe(system, device):
    cfg = SYSTEMS[system]["cfg"]()
    methods, _ = io_methods(system)
    io_cfg = io_abf.IOConfig(n_cells=io_abf.cells_for_walkers(cfg.N),
                             obs_every=PROBE_OBS_EVERY,
                             opportunity_every=max(1, cfg.n_steps // 4),
                             history_capacity=PROBE_CAPACITY)
    print(f"[{system}] R-OBS probe: A0 only, {len(SEEDS['probe'])} seeds, "
          f"dense obs_every={PROBE_OBS_EVERY}")
    t0 = time.time()
    recs = run_chunk(system, cfg, SEEDS["probe"], [methods[0]], ["A0"], io_cfg,
                     device, keep_series=True)
    series = np.stack([r["io_series"] for r in recs])
    out = io_abf.probe_obs_every(series, dense_obs_every=PROBE_OBS_EVERY)
    structural = io_abf.cadence_for_run(cfg.n_steps)["obs_every"]
    out.update(system=system, structural_obs_every=structural,
               dt=float(cfg.n_steps and cfg.dt), n_steps=int(cfg.n_steps),
               obs_interval_time=float(out["obs_every"]) * cfg.dt,
               wall_seconds=time.time() - t0, **provenance())
    d = sysdir(system, "probe")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "r_obs.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  lag median {out['lag_median']:.1f} dense samples over "
          f"{out['n_cells_resolved']} cells")
    print(f"  R-OBS obs_every = {out['obs_every']} steps "
          f"({out['obs_interval_time']:.4g} time units); "
          f"structural rule would have said {structural}")
    return out


# --------------------------------------------------------------------------- #
# phase: calibration (A0 only) -- freezes eps1, eps2
# --------------------------------------------------------------------------- #
def phase_calibration(system, device, chunk=16):
    cfg = SYSTEMS[system]["cfg"]()
    methods, _ = io_methods(system)
    io_cfg = build_io_cfg(cfg, read_obs_every(system))
    seeds = SEEDS["calibration"]
    print(f"[{system}] calibration: A0 only, {len(seeds)} seeds, "
          f"obs_every={io_cfg.obs_every}, opp_every={io_cfg.opportunity_every}")
    t0, recs = time.time(), []
    for i in range(0, len(seeds), chunk):
        recs += run_chunk(system, cfg, seeds[i:i + chunk], [methods[0]], ["A0"],
                          io_cfg, device)
        print(f"  ... {min(i + chunk, len(seeds))}/{len(seeds)} seeds "
              f"({time.time() - t0:.0f}s)")
    save_records(sysdir(system, "calibration"), recs, "calibration")

    t = recs[0]["t"]
    T = float(t[-1])
    curves = np.stack([r["l2_f_t"] for r in recs])
    curves_full = np.stack([r["l2_f_full_t"] for r in recs])

    def median_at(frac):
        k = int(np.argmin(np.abs(t - frac * T)))
        return float(np.median(curves[:, k])), float(t[k])

    eps1, t1 = median_at(0.40)
    eps2, t2 = median_at(0.60)

    gam = np.stack([r["io_gamma"] for r in recs])
    sig = np.stack([r["io_sigma2"] for r in recs])
    tau = np.stack([r["io_tau"] for r in recs])
    a_cell = recs[0]["io_a_cell"]
    scored = a_cell > 0
    valid_tau = float(np.mean(np.isfinite(tau[:, scored]) & (tau[:, scored] > 0)))

    def spread(v, mask=scored):
        w = v[:, mask]
        w = w[np.isfinite(w) & (w > 0)]
        if w.size < 4:
            return dict(q10=float("nan"), q90=float("nan"), ratio=float("nan"))
        q10, q90 = float(np.quantile(w, 0.1)), float(np.quantile(w, 0.9))
        return dict(q10=q10, q90=q90, ratio=q90 / max(q10, 1e-300))

    thr = dict(
        system=system, role=SYSTEMS[system]["role"], T_total=T,
        eps1=eps1, eps2=eps2, t_eps1=t1, t_eps2=t2,
        n_calibration_seeds=len(seeds), n_frames=int(t.size),
        frame_dt=float(t[1] - t[0]),
        final_l2_f_median=float(np.median(curves[:, -1])),
        final_l2_f_full_median=float(np.median(curves_full[:, -1])),
        obs_every=int(io_cfg.obs_every),
        opportunity_every=int(io_cfg.opportunity_every),
        n_cells=int(io_cfg.n_cells), n_scored_cells=int(scored.sum()),
        valid_tau_fraction=valid_tau,
        gamma_unresolved=bool(valid_tau < 0.80),
        R_gamma=spread(gam), R_sigma2=spread(sig), R_tau=spread(tau),
        a_cell=a_cell.tolist(),
        wall_seconds=time.time() - t0, **provenance())
    d = sysdir(system, "calibration")
    with open(os.path.join(d, "thresholds.json"), "w") as fh:
        json.dump(thr, fh, indent=2, default=str)
    print(f"  eps1 = {eps1:.5f} at t={t1:.3g};  eps2 = {eps2:.5f} at t={t2:.3g}")
    print(f"  R_Gamma = {thr['R_gamma']['ratio']:.2f}  "
          f"R_sigma2 = {thr['R_sigma2']['ratio']:.2f}  "
          f"R_tau = {thr['R_tau']['ratio']:.2f}")
    print(f"  valid-tau fraction = {valid_tau:.3f}"
          + ("   *** GAMMA UNRESOLVED ***" if thr["gamma_unresolved"] else "  (gate >= 0.80 PASS)"))
    return thr


# --------------------------------------------------------------------------- #
# phases: pilot and confirmatory
# --------------------------------------------------------------------------- #
def phase_arms(system, device, which, chunk):
    thr_path = sysdir(system, "calibration", "thresholds.json")
    if not os.path.exists(thr_path):
        raise SystemExit(f"[{system}] no frozen thresholds; run --phase calibration")
    cfg = SYSTEMS[system]["cfg"]()
    methods, arms = io_methods(system)
    io_cfg = build_io_cfg(cfg, read_obs_every(system))
    seeds = SEEDS[which]
    print(f"[{system}] {which}: {len(seeds)} paired seeds x {len(arms)} arms "
          f"= {len(seeds) * len(arms)} runs")
    t0, recs = time.time(), []
    for i in range(0, len(seeds), chunk):
        recs += run_chunk(system, cfg, seeds[i:i + chunk], methods, arms,
                          io_cfg, device)
        print(f"  ... {min(i + chunk, len(seeds))}/{len(seeds)} seeds "
              f"({time.time() - t0:.0f}s)", flush=True)
    save_records(sysdir(system, which), recs, which)
    with open(sysdir(system, which, "run_meta.json"), "w") as fh:
        json.dump(dict(system=system, phase=which, seeds=seeds, arms=arms,
                       config=asdict(cfg), io_cfg=asdict(io_cfg),
                       wall_seconds=time.time() - t0, **provenance()),
                  fh, indent=2, default=str)
    nan = [r for r in recs if not np.isfinite(r["final_l2_f"])]
    print(f"  done in {time.time() - t0:.0f}s; non-finite finals: {len(nan)}")
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=sorted(SYSTEMS))
    ap.add_argument("--phase", required=True,
                    choices=["probe", "calibration", "pilot", "confirmatory"])
    ap.add_argument("--chunk", type=int, default=0,
                    help="seeds per batched call; 0 picks a default per engine")
    a = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device visible")
    device = torch.device("cuda")
    # The gates are cheap and they are the reason anything else may run.
    io_abf.assert_no_reference_leakage()
    io_abf.assert_no_birth_death()

    chunk = a.chunk or (16 if SYSTEMS[a.system]["engine"] == "eb" else 8)
    if a.phase == "probe":
        phase_probe(a.system, device)
    elif a.phase == "calibration":
        phase_calibration(a.system, device, chunk=chunk)
    else:
        phase_arms(a.system, device, a.phase, chunk=chunk)


if __name__ == "__main__":
    main()
