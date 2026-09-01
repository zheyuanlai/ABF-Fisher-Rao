#!/usr/bin/env python
"""ZIF-8 300 K corrected-baseline experiment: safety calibration and production.

    python scripts/run_zif8_corrected.py --stage calibrate --rate 0.05
    python scripts/run_zif8_corrected.py --stage produce --only abf
"""
from __future__ import annotations
import argparse, json, os, socket, subprocess, sys, time
import numpy as np, torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from zif8.core_zif8 import ZIF8SimConfig, ZIF8System, engine_kwargs, run_sampler  # noqa

PRE_LEGACY = os.path.join(ROOT, "configs/uniform_campaign/zif8_prereg.json")
PRE_CORR = os.path.join(ROOT, "configs/information_campaign/corrected_baseline_prereg.json")
OUT = os.path.join(ROOT, "results/information_campaign/corrected")


def git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["calibrate", "produce"])
    ap.add_argument("--rate", type=float, default=None)
    ap.add_argument("--only", default=None, choices=["abf", "fr_uniform"])
    a = ap.parse_args()
    legacy = json.load(open(PRE_LEGACY)); corr = json.load(open(PRE_CORR))
    s = {k: v for k, v in legacy["sampler"].items() if not k.startswith("_")}
    s["abf_bandwidth_A"] = corr["corrected_baseline"]["h_bias_A"]      # THE change
    os.makedirs(OUT, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    system = ZIF8System(300.0, dev, root=ROOT, **engine_kwargs(legacy))
    pool = os.path.join(ROOT, "cache/zif8/init_pool_T300.npz")

    if a.stage == "calibrate":
        c = corr["fr_rate_rule"]; rule = legacy["success_rule"]
        sim = ZIF8SimConfig(**s, rng_seed=c["rng_seed"], fr_rate=float(a.rate))
        sim.n_replicas = c["n_replicas"]
        seeds = list(range(c["seed_first"], c["seed_first"] + c["n_seeds"]))
        t0 = time.time()
        out = run_sampler("fr_uniform", system, sim, seeds=seeds,
                          init_pool=pool, verbose=False)
        ess = np.asarray(out["ancestor_ess"], float)
        wmax = np.asarray(out["max_ancestor_frac"], float)
        act = np.asarray(out["steps"]) >= sim.fr_start_steps
        N = sim.n_replicas
        ev = float(out["total_replacement_events"].sum() / (len(seeds) * N))
        n_ops = (sim.n_steps - sim.fr_start_steps) / max(sim.fr_every, 1)
        r = dict(rate=float(a.rate), ess_min=float(np.nanmin(ess[act]) / N),
                 wmax_max=float(np.nanmax(wmax[act])), events_per_replica=ev,
                 event_fraction=ev / max(n_ops, 1), h_bias=s["abf_bandwidth_A"],
                 minutes=(time.time() - t0) / 60)
        r["ok"] = bool(r["ess_min"] >= rule["ess_anc_over_N_min"]
                       and r["wmax_max"] <= rule["wmax_max"]
                       and r["event_fraction"] <= s["max_event_fraction"])
        with open(os.path.join(OUT, f"cal_rate_{a.rate:g}.json"), "w") as fh:
            json.dump(r, fh, indent=2)
        print(f"  rate {a.rate:>6.3f}: min ESS/N {r['ess_min']:.3f}  "
              f"wmax {r['wmax_max']:.4f}  events/replica {ev:6.2f}  ok={r['ok']}  "
              f"[{r['minutes']:.1f} min]", flush=True)
        return

    sel = json.load(open(os.path.join(OUT, "fr_rate_selection.json")))
    rate = sel["selected"]
    assert rate is not None, "no safe FR rate under the corrected baseline"
    assert abs(sel["h_bias"] - s["abf_bandwidth_A"]) < 1e-12, \
        "the calibration was run at a different h_bias than production"
    p = corr["production"]
    seeds = list(range(p["seed_first"], p["seed_first"] + p["n_seeds"]))
    sim = ZIF8SimConfig(**s, rng_seed=int(p["rng_seed"]), fr_rate=float(rate))
    print(f"corrected production: h_bias={s['abf_bandwidth_A']} A, fr_rate={rate}, "
          f"seeds {seeds[0]}-{seeds[-1]}, N={sim.n_replicas}, {sim.n_steps} steps",
          flush=True)
    for method in (("abf", "fr_uniform") if a.only is None else (a.only,)):
        path = os.path.join(OUT, f"{method}.npz")
        if os.path.exists(path):
            print(f"  skip {method}"); continue
        out = run_sampler(method, system, sim, seeds=seeds, init_pool=pool,
                          verbose=True, progress_every=10)
        payload = {k: v for k, v in out.items()
                   if isinstance(v, (np.ndarray, np.generic, int, float, str))}
        payload["meta"] = json.dumps(dict(
            method=method, h_bias_A=s["abf_bandwidth_A"],
            h_read_A=corr["corrected_baseline"]["h_read_A"], fr_rate=rate,
            seeds=seeds, rng_seed=int(p["rng_seed"]), n_replicas=sim.n_replicas,
            n_steps=sim.n_steps, dt=sim.dt, fr_start_steps=sim.fr_start_steps,
            git_rev=git_rev(), host=socket.gethostname()))
        tmp = path + ".tmp.npz"; np.savez_compressed(tmp, **payload); os.replace(tmp, path)
        print(f"  wrote {path}", flush=True)


if __name__ == "__main__":
    main()
