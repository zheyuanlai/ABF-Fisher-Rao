#!/usr/bin/env python3
"""Freeze laplacian_scale per cell from the fresh-seed pilot with the PREDECLARED rule (docs/LRS_LAPLACIAN_BASELINE.md,
"Pilot calibration"): reference-free, no free-energy error is read.

    python scripts/laplacian_baselines/pilot_decide.py            # -> results/laplacian_baselines/pilot/decision.json + .md
    python scripts/laplacian_baselines/pilot_decide.py --apply    # also freeze the scales into the production job files

Per pilot run (FR arm and Laplacian arm at every ladder scale, same 8 fresh seeds): realised replacements R (deaths),
min windowed ESS/K, max windowed largest-family share, and (LTA / ERI) the R2 controller's stop reason.  Rule:
  safety   median over seeds of min windowed ESS/K >= 0.15 AND of max family share <= 0.10 (the frozen R2 thresholds)
  activity among the scales passing safety, the one whose median R is closest in log-ratio to the FR arm's median R on
           the same seeds (dose matching on realised selection activity, since score units differ)
  fallback no scale passes -> the smallest ladder scale, flagged NO_SAFE_SCALE; FR itself fails safety -> activity only,
           flagged FR_UNSAFE.  Ties -> the smaller scale.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
PILOT = os.path.join(ROOT, "results", "laplacian_baselines", "pilot")
ESS_MIN, MAXFAM_MAX = 0.15, 0.10
CELLS = {  # cell -> (engine, pilot dir, N)
    "R2C1": ("eri", "R2C1_ite300_K1024_R2", 1024), "R2C2": ("eri", "R2C2_ite150_K256_R2", 256),
    "R2C3": ("lta", "R2C3_lta150_K1024_R2", 1024), "R2C4": ("lta", "R2C4_lta300_K1024_R2", 1024),
    "R2H80": ("lta", "R2H80_lta80_K1024_R2", 1024), "R2H350": ("lta", "R2H350_lta350_K1024_R2", 1024),
    "gateway512": ("gateway", "gateway/N512", 512), "gateway2048": ("gateway", "gateway/N2048", 2048),
}


def _stats(engine, path, N):
    """(replacements, min windowed ESS/K, max windowed family share, stop reason or None, t_stop or None)."""
    if engine == "eri":
        z = np.load(path, allow_pickle=False)
        ess = np.asarray(z["sv_u"][:, 2], float)
        mx = np.asarray(z["sv_u"][:, 3], float)
        repl = int(z["sv_i"][-1, 2])
        reason = str(z["fr_controller_reason"]) if "fr_controller_reason" in z.files else None
        tstop = float(z["fr_controller_t_stop"]) if "fr_controller_t_stop" in z.files else None
    elif engine == "lta":
        import lta_ladder_numba as L
        r = L.load_result(path)
        ess = np.asarray(r["gen_ess_win"], float)
        mx = np.asarray(r["gen_max_frac_win"], float)
        repl = int(np.asarray(r["cum_deaths"])[-1])
        reason = str(r["fr_controller_reason"]) if "fr_controller_reason" in r else None
        tstop = float(r["fr_controller_t_stop"]) if "fr_controller_t_stop" in r else None
    else:
        import gateway_ladder_numba as G
        r = G.load_result(path)
        ess = np.asarray(r["gen_ess_win"], float)
        mx = np.asarray(r["gen_maxfam_win"], float)
        repl = int(np.asarray(r["fr_deaths_cum"])[-1])
        reason = tstop = None
    if np.nanmax(ess) > 1.5:          # absolute ESS -> ESS/K
        ess = ess / N
    ess = ess[np.isfinite(ess)]
    mx = mx[np.isfinite(mx)]
    return dict(repl=repl, ess_min=float(ess.min()) if ess.size else float("nan"),
                maxfam_max=float(mx.max()) if mx.size else float("nan"), reason=reason, t_stop=tstop)


def _collect(cell):
    engine, d, N = CELLS[cell]
    base = os.path.join(PILOT, d)
    arms = {}
    for f in sorted(glob.glob(os.path.join(base, "*.npz"))):
        if f.endswith(".ckpt.npz") or f.endswith(".tmp.npz"):
            continue
        name = os.path.basename(f)
        m = re.match(r"(?:fr_N\d+_s(\d+)|laplacian_sc([\d.e-]+)_N\d+_s(\d+)|s(\d+)_fr|s(\d+)_laplacian_sc([\d.e-]+))\.npz$", name)
        if not m:
            continue
        if m.group(1) or m.group(4):
            arm, seed = "fr", int(m.group(1) or m.group(4))
        else:
            arm = f"lap{float(m.group(2) or m.group(6)):g}"
            seed = int(m.group(3) or m.group(5))
        try:
            arms.setdefault(arm, {})[seed] = _stats(engine, f, N)
        except Exception as e:  # incomplete file
            print(f"  {cell} {name}: unreadable ({e})", file=sys.stderr)
    return arms


def _med(v):
    return float(np.median(v)) if len(v) else float("nan")


def decide(cell):
    arms = _collect(cell)
    if "fr" not in arms:
        return dict(cell=cell, status="NO_FR_PILOT", scale=None, arms={})
    summary = {}
    for arm, runs in arms.items():
        reps = [r["repl"] for r in runs.values()]
        summary[arm] = dict(n=len(runs), repl_median=_med(reps), repl_per_replica_median=_med(reps) / CELLS[cell][2],
                            ess_min_median=_med([r["ess_min"] for r in runs.values()]),
                            maxfam_max_median=_med([r["maxfam_max"] for r in runs.values()]),
                            n_safety_stops=sum(1 for r in runs.values() if r["reason"] and "safety" in r["reason"].lower()),
                            n_established=sum(1 for r in runs.values() if r["reason"] and "establ" in r["reason"].lower()),
                            t_stop_median=_med([r["t_stop"] for r in runs.values() if r["t_stop"] is not None and r["t_stop"] >= 0]),
                            per_seed={int(s): r for s, r in sorted(runs.items())})
        summary[arm]["safe"] = bool(summary[arm]["ess_min_median"] >= ESS_MIN and summary[arm]["maxfam_max_median"] <= MAXFAM_MAX)
    fr = summary["fr"]
    laps = {float(a[3:]): s for a, s in summary.items() if a.startswith("lap")}
    flags = []
    if not fr["safe"]:
        flags.append("FR_UNSAFE")
    passing = sorted(sc for sc, s in laps.items() if s["safe"])
    if not passing:
        flags.append("NO_SAFE_SCALE")
        passing = [min(laps)] if laps else []
    if not passing:
        return dict(cell=cell, status="NO_LAPLACIAN_PILOT", scale=None, arms=summary)
    target = max(fr["repl_median"], 1.0)

    def dist(sc):
        return abs(math.log(max(laps[sc]["repl_median"], 1.0) / target))
    best = min(passing, key=lambda sc: (round(dist(sc), 12), sc))
    return dict(cell=cell, status="OK" if not flags else "+".join(flags), scale=best, fr_repl_median=fr["repl_median"],
                scales_passing_safety=[sc for sc in sorted(laps) if laps[sc]["safe"]],
                log_ratio_to_fr={f"{sc:g}": math.log(max(laps[sc]["repl_median"], 1.0) / target) for sc in sorted(laps)},
                arms=summary)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", nargs="*", default=list(CELLS))
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--out", default="results/laplacian_baselines/pilot/decision.json")
    a = ap.parse_args(argv)
    dec = dict(rule=dict(ess_min=ESS_MIN, maxfam_max=MAXFAM_MAX,
                         text="safety: median min windowed ESS/K >= 0.15 and median max family share <= 0.10; activity: "
                              "closest log-ratio of median replacements to the FR arm on the same fresh seeds"),
               cells={c: decide(c) for c in a.cells})
    out = os.path.join(ROOT, a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(dec, open(out, "w"), indent=1, allow_nan=True)
    lines = ["| cell | arm | n | repl median | repl/K | min ESS/K (med) | max family (med) | safety stops | established | t_stop med | safe |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for c, d in dec["cells"].items():
        for arm, s in d.get("arms", {}).items():
            lines.append(f"| {c} | {arm} | {s['n']} | {s['repl_median']:.0f} | {s['repl_per_replica_median']:.3f} | {s['ess_min_median']:.3f} | "
                         f"{s['maxfam_max_median']:.3f} | {s['n_safety_stops']} | {s['n_established']} | {s['t_stop_median']:.2f} | {'yes' if s['safe'] else 'no'} |")
        lines.append(f"| **{c}** | **decision** | | | | | | | | | **scale {d.get('scale')} ({d['status']})** |")
    md = "\n".join(lines) + "\n"
    open(out[:-5] + ".md", "w").write(md)
    print(md)
    for c, d in dec["cells"].items():
        print(f"{c}: scale {d.get('scale')} status {d['status']} passing {d.get('scales_passing_safety')}")
    if a.apply:
        if any(d.get("scale") is None for d in dec["cells"].values()):
            raise SystemExit("some cell has no decision; not applying")
        import subprocess
        subprocess.check_call([sys.executable, os.path.join(ROOT, "scripts/laplacian_baselines/make_jobs.py"), "apply",
                               "--decision", a.out])


if __name__ == "__main__":
    main()
