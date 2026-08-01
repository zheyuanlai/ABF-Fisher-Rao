"""Analysis and preregistered decision for the alanine ABF vs oracle-mFR study.

Writes the required artifacts under ``<root>/analysis/`` and a decision JSON.  The go/no-go
criteria are evaluated exactly as preregistered; the classification is not softened after seeing
results.

Usage:
  python scripts/analyze_alanine.py --root results/alanine_oracle/pilot --stage N2048 \
      --window 20 100 --kind pilot
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from alanine.basins import from_reference                       # noqa: E402
from alanine.metrics_ala import (basin_summary, build_masks, cost_summary,  # noqa: E402
                                 genealogy_summary, paired_bootstrap,
                                 smooth_reference, time_series)

PILOT = dict(fes_med=-0.10, seed_wins=3, of=4, grad=-0.05, ess=0.30, wmax=0.05,
             event=0.05, clip=1e-4, frozen_retention=2.0 / 3.0)
PROD = dict(fes_med=-0.15, grad=-0.10, seed_wins=8, of=10, ess=0.30, wmax=0.05,
            event=0.05, clip=1e-4, frozen_retention=2.0 / 3.0)


def load_runs(root, stage):
    out = {}
    for f in sorted(glob.glob(os.path.join(root, stage, "raw", "*.npz"))):
        d = np.load(f, allow_pickle=True)
        meta = json.loads(str(d["meta"]))
        out[meta["method"]] = (d, meta)
    return out


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(str(r.get(k, "")) for k in keys) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--window", nargs=2, type=float, default=[20.0, 100.0])
    ap.add_argument("--kind", choices=("pilot", "production"), default="pilot")
    ap.add_argument("--reference", default="results/alanine/reference/reference.npz")
    ap.add_argument("--frozen", default=None, help="frozen-bias metrics json, if available")
    a = ap.parse_args()

    ana = os.path.join(a.root, "analysis")
    os.makedirs(ana, exist_ok=True)
    bm, ref_meta = from_reference(a.reference)
    F_ref = np.load(a.reference, allow_pickle=True)["F"]
    kT = ref_meta["kT_kJ"]
    pack = build_masks(F_ref, kT)
    runs = load_runs(a.root, a.stage)
    if set(runs) != {"abf", "fr_oracle"}:
        raise SystemExit(f"expected both arms, found {sorted(runs)}")
    n_grid = int(ref_meta["n_grid"])
    F_sm = smooth_reference(F_ref, 0.08, n_grid)

    ts_rows, seed_rows, gen_rows, bas_rows, cost_rows = [], [], [], [], []
    for m, (d, meta) in runs.items():
        out = {k: d[k] for k in d.files if k != "meta"}
        out["method"] = m
        out["n_replicas"] = meta["n_replicas"]
        out["n_steps"] = meta["n_steps"]
        t, ps = time_series(out, pack, F_sm, n_grid, tuple(a.window))
        ts_rows += t
        seed_rows += ps
        gen_rows += genealogy_summary(out, tuple(a.window),
                                      fr_start_steps=int(meta.get("fr_start_steps", 20000)),
                                      fr_every=int(meta.get("fr_every", 500)))
        bas_rows += basin_summary(out, bm.names, tuple(a.window), 0.001)
        cost_rows.append(cost_summary(out, meta) | {"run_id": meta["run_id"]})

    write_csv(os.path.join(ana, "time_series_metrics.csv"), ts_rows)
    write_csv(os.path.join(ana, "paired_seed_metrics.csv"), seed_rows)
    write_csv(os.path.join(ana, "genealogy_metrics.csv"), gen_rows)
    write_csv(os.path.join(ana, "basin_metrics.csv"), bas_rows)
    write_csv(os.path.join(ana, "cost_metrics.csv"), cost_rows)
    json.dump(dict(reference=a.reference, param_hash=ref_meta["param_hash"],
                   kT_kJ=kT, n_grid=n_grid, basins=bm.summary(),
                   dG_systematic_kT=0.25), open(os.path.join(ana, "reference_provenance.json"), "w"),
              indent=2, default=float)

    # ------------------------------------------------------------------ paired statistics
    crit = PILOT if a.kind == "pilot" else PROD
    def by(method, key):
        return np.array([r[key] for r in sorted([x for x in seed_rows if x["method"] == method],
                                                key=lambda z: z["seed"])], dtype=float)

    stats, signs = {}, {}
    for wname in ("equilibrium", "uniform8", "uniform10"):
        k = f"int_eF_km_{wname}"
        A, Bv = by("abf", k), by("fr_oracle", k)
        stats[wname] = paired_bootstrap(A, Bv)
        signs[wname] = float(np.sign(stats[wname]["median"]))
    kg = "final_egradF_equilibrium"
    gstat = paired_bootstrap(by("abf", kg), by("fr_oracle", kg))

    fr_gen = [r for r in gen_rows if r["method"] == "fr_oracle"]
    ess_min = min(r["ess_age_min"] for r in fr_gen)
    wmax_max = max(r["wmax_max"] for r in fr_gen)
    ev_max = max(r["event_fraction"] for r in fr_gen)
    ev_cum_max = max(r["event_fraction_cumulative"] for r in fr_gen)
    clip_max = max(r["clip_fraction"] for r in cost_rows)

    frozen_ret = None
    if a.frozen and os.path.exists(a.frozen):
        frozen_ret = json.load(open(a.frozen)).get("retention")

    prim = stats["equilibrium"]
    checks = {
        "fes_median_improvement": (prim["median"] <= crit["fes_med"], prim["median"]),
        "seed_wins": (prim["win_rate"] * prim["n"] >= crit["seed_wins"],
                      f"{int(prim['win_rate'] * prim['n'])}/{prim['n']}"),
        "grad_improvement": (gstat["median"] <= crit["grad"], gstat["median"]),
        "sign_consistent_all_weightings": (len({s for s in signs.values()}) == 1, signs),
        "ci_upper_below_zero": (prim["hi"] < 0, prim["hi"]),
        "ess_age_ge_0.30N": (ess_min >= crit["ess"], ess_min),
        "wmax_le_0.05": (wmax_max <= crit["wmax"], wmax_max),
        "event_fraction_lt_0.05": (ev_max < crit["event"], ev_max),
        "clip_fraction_lt_1e-4": (clip_max < crit["clip"], clip_max),
    }
    if frozen_ret is not None:
        checks["frozen_retention_ge_2_3"] = (frozen_ret >= crit["frozen_retention"], frozen_ret)

    passed = all(v[0] for v in checks.values())
    if passed:
        cls = "SAFE PILOT PASS" if a.kind == "pilot" else "POSITIVE"
    elif (not checks["ess_age_ge_0.30N"][0] or not checks["wmax_le_0.05"][0]
          or (frozen_ret is not None and frozen_ret < crit["frozen_retention"])):
        cls = "FALSE IMPROVEMENT" if prim["median"] < 0 else "GENEALOGICAL FAILURE"
    elif prim["lo"] > -0.10 and prim["hi"] < 0.10:
        cls = "EQUIVALENT"
    elif prim["lo"] > 0:
        cls = "HARMFUL"
    else:
        cls = "INCONCLUSIVE"

    decision = dict(kind=a.kind, stage=a.stage, window_ps=a.window,
                    n_paired_seeds=int(prim["n"]),
                    primary_kernel_matched_integrated_FES=stats,
                    endpoint_mean_force=gstat, signs=signs,
                    genealogy=dict(ess_age_min=ess_min, wmax_max=wmax_max,
                                   event_fraction_per_opportunity_max=ev_max,
                                   event_fraction_cumulative_max=ev_cum_max),
                    clip_fraction_max=clip_max, frozen_retention=frozen_ret,
                    checks={k: [bool(v[0]), v[1]] for k, v in checks.items()},
                    all_pass=bool(passed), classification=cls,
                    dG_systematic_kT=0.25,
                    note=("dG(C7ax) effects smaller than the 0.25 kT reference systematic are "
                          "not resolvable and must not be claimed."))
    name = "pilot_decision.json" if a.kind == "pilot" else "production_decision.json"
    json.dump(decision, open(os.path.join(ana, name), "w"), indent=2, default=float)

    print("=" * 78)
    print(f"{a.kind.upper()}  stage={a.stage}  window={a.window} ps  seeds={int(prim['n'])}")
    print("=" * 78)
    for k, (ok, val) in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {k:34s} {val}")
    print(f"\n  primary (kernel-matched integrated FES, equilibrium weighting):")
    print(f"     median {prim['median']:+.4f}  CI [{prim['lo']:+.4f}, {prim['hi']:+.4f}]  "
          f"wins {prim['win_rate']:.2f}")
    print(f"  endpoint mean force: median {gstat['median']:+.4f} "
          f"CI [{gstat['lo']:+.4f}, {gstat['hi']:+.4f}]")
    print(f"\n  CLASSIFICATION: {cls}")
    print(f"  wrote {os.path.join(ana, name)}")


if __name__ == "__main__":
    main()
