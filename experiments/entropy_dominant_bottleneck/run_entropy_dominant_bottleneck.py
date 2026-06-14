#!/usr/bin/env python3
"""Run the ENTROPY-DOMINANT bottleneck ABF + Fisher-Rao stress test.

Sweeps the entropic share phi at fixed total barrier B0 (energetic <-> entropic
trade-off) and asks whether the FR gain over ABF grows as the barrier becomes
more entropic.  See experiments/entropy_dominant_bottleneck/configs/*.yaml and
src/edb_abffr_core.py for the model.

Single-GPU batched execution: every (config, seed, method) of a sweep runs
together in wide `simulate_batch` calls (matched seeds within a (config,seed)
row), chunked only to bound GPU memory and to flush results to disk between
calls (so long jobs are recoverable / resumable).

Modes
-----
  --smoke-test : analytic sanity checks + a tiny end-to-end run, then exit.
  --pilot      : reduced seeds + steps (config `pilot:` block), full pipeline.
  --production : full sweep + FR-rate sweep (config defaults).

Examples
--------
  python experiments/entropy_dominant_bottleneck/run_entropy_dominant_bottleneck.py \
    --config experiments/entropy_dominant_bottleneck/configs/entropy_dominant_default.yaml \
    --smoke-test --device cuda:0
  CUDA_VISIBLE_DEVICES=3 python experiments/.../run_entropy_dominant_bottleneck.py \
    --config .../entropy_dominant_default.yaml --production --device cuda:0
  # resume the most recent (unfinished) production sweep:
  CUDA_VISIBLE_DEVICES=3 python ... --production --resume
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import socket
import sys
import time
from dataclasses import asdict

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import edb_abffr_core as edb  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

STRUCT_KEYS = ("m", "N", "dt", "n_steps", "save_every", "fr_every", "fr_burnin",
               "ramp_steps", "h", "eta", "min_count", "ess_window_steps", "x_init_std")


# -----------------------------------------------------------------------------
# config resolution
# -----------------------------------------------------------------------------
def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_config(raw, mode, overrides):
    """Flatten YAML + apply the mode override block + CLI overrides -> dict."""
    r = dict(raw)
    block = {}
    if mode == "pilot":
        block = raw.get("pilot", {}) or {}
    elif mode == "smoke":
        block = raw.get("smoke", {}) or {}
    r.update(block)
    # CLI overrides (only non-None)
    for k, v in overrides.items():
        if v is not None:
            r[k] = v
    return r


def seeds_list(spec):
    if isinstance(spec, int):
        return list(range(spec))
    return [int(s) for s in spec]


def build_phys(rc, phi, gamma):
    """Construct a PhysConfig for entropic share phi and birth-death rate gamma."""
    fr = rc.get("fr", {})
    abf = rc.get("abf", {})
    n_steps = int(rc["n_steps"])
    burnin = int(round(float(fr.get("burnin_fraction", 0.0)) * n_steps))
    return edb.make_config(
        beta=float(rc["beta"]), m=int(rc["m"]), B0=float(rc["B0"]),
        omega_out=float(rc["omega_out"]), phi=float(phi), s=float(rc["s"]),
        gamma=float(gamma), N=int(rc["n_walkers"]), dt=float(rc["dt"]),
        n_steps=n_steps, save_every=int(rc["record_every"]),
        h=float(abf.get("h", 0.07)), min_count=float(abf.get("min_count", 1.0)),
        eta=float(fr.get("eta", 0.10)), fr_every=int(fr.get("fr_every", 10)),
        fr_burnin=burnin, ramp_steps=int(fr.get("ramp_steps", 2000)),
        target_ema_rate=float(fr.get("target_ema_rate", 0.005)),
        score_clip=float(fr.get("score_clip", 3.0)),
        max_event_fraction=float(fr.get("max_event_fraction", 0.08)),
        ess_window_steps=int(fr.get("ess_window_steps", 4000)),
        x_init_std=float(rc.get("x_init_std", 0.1)),
    )


# -----------------------------------------------------------------------------
# raw-run IO (idempotent npz per (method, runkey))
# -----------------------------------------------------------------------------
def run_path(raw_dir, sweep, method_name, runkey):
    return os.path.join(raw_dir, sweep, f"{method_name}__{runkey}.npz")


def save_record(path, rec):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    flat = {}
    for k, v in rec.items():
        if k == "config":
            for ck, cv in v.items():
                flat[f"cfg__{ck}"] = cv
        elif v is None:
            flat[f"none__{k}"] = True
        else:
            flat[k] = v
    np.savez_compressed(path, **flat)


def valid_existing(path):
    if not os.path.exists(path):
        return False
    try:
        with np.load(path, allow_pickle=True) as d:
            return "final_l2_f" in d and np.isfinite(float(d["final_l2_f"]))
    except Exception:
        return False


def struct_sig(cfg):
    return tuple(getattr(cfg, k) for k in STRUCT_KEYS)


# -----------------------------------------------------------------------------
# one sweep group (main or rate)
# -----------------------------------------------------------------------------
def run_group(sweep, rows, methods, raw_dir, device, max_rows_per_call,
              overwrite, batch_seed_base):
    """rows = list of (runkey, cfg, seed).  All `methods` for a row share init+noise.

    Returns (n_runs_written, elapsed_seconds).
    """
    pending = []
    for runkey, cfg, sd in rows:
        need = [m for m in methods
                if overwrite or not valid_existing(run_path(raw_dir, sweep, m.name, runkey))]
        if need:
            pending.append((runkey, cfg, sd, need))
    n_done = len(rows) - len(pending)
    print(f"[{sweep}] {len(rows)} rows x {len(methods)} methods; "
          f"{n_done} rows already complete, {len(pending)} to run", flush=True)
    if not pending:
        return 0, 0.0

    groups = {}
    for runkey, cfg, sd, need in pending:
        key = (struct_sig(cfg), tuple(m.name for m in need))
        groups.setdefault(key, []).append((runkey, cfg, sd, need))

    n_written = 0
    call_idx = 0
    t0_all = time.time()
    total_calls = sum((len(g) + max_rows_per_call - 1) // max_rows_per_call
                      for g in groups.values())
    for (_sig, mnames), grp in groups.items():
        method_objs = [edb.METHOD_REGISTRY[n] for n in mnames]
        M = len(method_objs)
        for c0 in range(0, len(grp), max_rows_per_call):
            chunk = grp[c0:c0 + max_rows_per_call]
            configs = [cfg for (_, cfg, _, _) in chunk]
            seeds = [sd for (_, _, sd, _) in chunk]
            spec = edb.BatchSpec(configs=configs, seeds=seeds, methods=method_objs,
                                 batch_seed=batch_seed_base + call_idx)
            t0 = time.time()
            recs = edb.simulate_batch(spec, device=device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            for bi, (runkey, _, _, _) in enumerate(chunk):
                for mi, m in enumerate(method_objs):
                    save_record(run_path(raw_dir, sweep, m.name, runkey), recs[bi * M + mi])
                    n_written += 1
            dt_call = time.time() - t0
            done = call_idx + 1
            elapsed = time.time() - t0_all
            eta = elapsed / done * (total_calls - done)
            print(f"  [{sweep}] call {done}/{total_calls}: R={len(chunk)*M} "
                  f"({len(chunk)} rows x {M} methods) in {dt_call:.0f}s "
                  f"| elapsed {elapsed:.0f}s eta {eta:.0f}s", flush=True)
            call_idx += 1
    return n_written, time.time() - t0_all


# -----------------------------------------------------------------------------
# sweep builders
# -----------------------------------------------------------------------------
def main_rows(rc, seeds):
    rows = []
    gamma = float(rc.get("fr", {}).get("gamma", 15.0))
    for phi in rc["phis"]:
        cfg = build_phys(rc, float(phi), gamma)
        for sd in seeds:
            rows.append((f"phi{float(phi):g}_seed{sd}", cfg, sd))
    return rows


def rate_rows(rc, seeds):
    rows = []
    for phi in rc.get("rate_sweep_phis", []):
        for ga in rc.get("rate_sweep_gammas", []):
            cfg = build_phys(rc, float(phi), float(ga))
            for sd in seeds:
                rows.append((f"phi{float(phi):g}_gamma{float(ga):g}_seed{sd}", cfg, sd))
    return rows


# -----------------------------------------------------------------------------
# smoke test
# -----------------------------------------------------------------------------
def smoke_test(rc, device):
    print("=== analytic sanity checks ===", flush=True)
    edb.sanity_checks(verbose=True)
    print("sanity checks PASSED\n", flush=True)

    print("=== tiny end-to-end batched run ===", flush=True)
    seeds = seeds_list(rc.get("seeds", 2))
    cfgs, sds = [], []
    for phi in (0.0, 0.90):
        cfgs.append(build_phys(rc, phi, float(rc.get("fr", {}).get("gamma", 15.0))))
        sds.append(seeds[0])
    spec = edb.BatchSpec(configs=cfgs, seeds=sds,
                         methods=[edb.ABF, edb.FR_ESTIMATED], batch_seed=0)
    t0 = time.time()
    recs = edb.simulate_batch(spec, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"ran {len(recs)} runs ({rc['n_steps']} steps) in {time.time()-t0:.1f}s", flush=True)
    for r in recs:
        c = r["config"]
        assert np.isfinite(r["final_l2_f"]), "NaN in L2(F)!"
        print(f"  phi={c['phi']:.2f} H={c['H']:.3f} oin={c['omega_in']:7.3f} "
              f"{r['method']:13s} L2(F)={r['final_l2_f']:.4f} "
              f"L2(F')={r['final_l2_fp']:.4f} ESS={r['final_ess']:.0f} "
              f"cross={r['final_cross']:.0f} occ={r['final_occ']:.3f} "
              f"replfrac={r['repl_fraction']:.4f}")
    # conditional sanity on the strongly-entropic run
    r = recs[-1]
    print("  conditional Var(Y_j|X) vs analytic (phi=0.90, fr_estimated):")
    for c_, ev, rv in zip(r["cond_centers"], r["cond_emp_var"], r["cond_ref_var"]):
        print(f"    x={c_:+.1f}  emp={ev:.5f}  ref={rv:.5f}")
    print("\nsmoke OK", flush=True)


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def pick_out_dir(rc, mode, args):
    root = os.path.join(REPO_ROOT, rc["output_root"])
    os.makedirs(root, exist_ok=True)
    if args.out_dir:
        d = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(root, args.out_dir)
        os.makedirs(d, exist_ok=True)
        return d
    if args.resume:
        cands = sorted(d for d in os.listdir(root)
                       if d.startswith("sweep_") and os.path.isdir(os.path.join(root, d)))
        if not cands:
            raise SystemExit("--resume: no existing sweep_* directory found")
        d = os.path.join(root, cands[-1])
        print(f"resuming {d}", flush=True)
        return d
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = {"production": "sweep", "pilot": "pilot", "smoke": "smoke"}[mode]
    d = os.path.join(root, f"{prefix}_{stamp}")
    os.makedirs(d, exist_ok=True)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default=None, help="e.g. cuda:0 (overrides config)")
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--production", action="store_true")
    ap.add_argument("--no-rate-sweep", action="store_true")
    ap.add_argument("--only-rate-sweep", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="continue the most recent sweep_* dir (idempotent skip)")
    ap.add_argument("--out-dir", default=None, help="explicit output dir (resume-friendly)")
    ap.add_argument("--overwrite", action="store_true")
    # quick overrides
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--n-steps", type=int, default=None)
    ap.add_argument("--max-rows-per-call", type=int, default=None)
    args = ap.parse_args()

    mode = ("smoke" if args.smoke_test else "pilot" if args.pilot
            else "production")
    if not (args.smoke_test or args.pilot or args.production):
        print("no mode flag given; defaulting to --production", flush=True)

    raw = load_yaml(args.config)
    overrides = {"device": args.device, "seeds": args.seeds, "n_steps": args.n_steps,
                 "max_rows_per_call": args.max_rows_per_call}
    rc = resolve_config(raw, mode, overrides)

    dev_str = rc.get("device", "cuda:0")
    device = torch.device(dev_str if torch.cuda.is_available() else "cpu")
    edb.DEVICE = device
    edb.configure_grid(rc["x_min"], rc["x_max"], rc["n_grid"], rc["eval_x_min"], rc["eval_x_max"])
    print(f"device={device} dtype={edb.DTYPE} mode={mode} "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','?')}", flush=True)

    if mode == "smoke":
        smoke_test(rc, device)
        return

    seeds = seeds_list(rc["seeds"])
    methods = [edb.METHOD_REGISTRY[m] for m in rc["methods"]]
    max_rows = int(rc.get("max_rows_per_call", 64))

    out_dir = pick_out_dir(rc, mode, args)
    raw_dir = os.path.join(out_dir, "raw")
    print(f"output dir: {out_dir}", flush=True)

    # dump resolved config
    with open(os.path.join(out_dir, "config_resolved.yaml"), "w") as f:
        yaml.safe_dump(rc, f, sort_keys=False)
    # pointer to latest sweep for the analyzer
    with open(os.path.join(REPO_ROOT, rc["output_root"], "latest_sweep.txt"), "w") as f:
        f.write(out_dir + "\n")

    t_start = time.time()
    summary = {"main_runs": 0, "rate_runs": 0}

    if not args.only_rate_sweep:
        rows = main_rows(rc, seeds)
        n, el = run_group("main", rows, methods, raw_dir, device, max_rows,
                          args.overwrite, batch_seed_base=10000)
        summary["main_runs"] = n
        summary["main_seconds"] = el

    if not args.no_rate_sweep and rc.get("rate_sweep_phis"):
        rs_methods = [edb.ABF, edb.FR_ESTIMATED]
        rows = rate_rows(rc, seeds)
        n, el = run_group("rate", rows, rs_methods, raw_dir, device, max_rows,
                          args.overwrite, batch_seed_base=20000)
        summary["rate_runs"] = n
        summary["rate_seconds"] = el

    total = time.time() - t_start
    meta = {
        "mode": mode,
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "hostname": socket.gethostname(),
        "torch": torch.__version__,
        "gpu_name": (torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"),
        "config_path": os.path.abspath(args.config),
        "n_seeds": len(seeds),
        "seeds": seeds,
        "phis": rc["phis"],
        "methods": rc["methods"],
        "rate_sweep_phis": rc.get("rate_sweep_phis", []),
        "rate_sweep_gammas": rc.get("rate_sweep_gammas", []),
        "n_steps": int(rc["n_steps"]),
        "n_walkers": int(rc["n_walkers"]),
        "dt": float(rc["dt"]),
        "beta": float(rc["beta"]), "m": int(rc["m"]), "B0": float(rc["B0"]),
        "total_runtime_seconds": total,
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "completed": True,
        **summary,
    }
    # merge with any prior metadata (resume)
    meta_path = os.path.join(out_dir, "run_metadata.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                prev = json.load(f)
            meta["prior_runtime_seconds"] = (prev.get("total_runtime_seconds", 0.0)
                                             + prev.get("prior_runtime_seconds", 0.0))
        except Exception:
            pass
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nDONE mode={mode} in {total:.0f}s "
          f"(main={summary.get('main_runs',0)} rate={summary.get('rate_runs',0)} runs)",
          flush=True)
    print(f"results in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
