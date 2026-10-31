#!/usr/bin/env python3
"""Compare ONLY published LRS Laplacian selection against archived ABF and FR positives.

Uses the unchanged, validated reference scorers in scripts/equal_budget/eqb_metrics.py.
This script is analysis-only: it never runs dynamics or overwrites old results.

  python scripts/laplacian_baselines/analyze.py --case all
  python scripts/laplacian_baselines/analyze.py --case R2C3
  python scripts/laplacian_baselines/analyze.py --case gateway512
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/equal_budget"))
sys.path.insert(0, str(ROOT / "src"))
import eqb_metrics as M  # noqa: E402

# The only comparison arms are archival ABF, archival Fisher-Rao, and NEW LRS Laplacian.
ITE = {
    "R2C1": (300, 1024, "R2C1_ite300_K1024_R2", 1400000, "results/overnight/ite/reference"),
    "R2C2": (150, 256, "R2C2_ite150_K256_R2", 1410000, "results/overnight/ite150/reference"),
}
LTA = {
    "R2C3": (150, 1024, "R2C3_lta150_K1024_R2", 1420000),
    "R2C4": (300, 1024, "R2C4_lta300_K1024_R2", 1430000),
    "R2H80": (80, 1024, "R2H80_lta80_K1024_R2", 1450000),
    "R2H350": (350, 1024, "R2H350_lta350_K1024_R2", 1460000),
}
GATEWAY = {"gateway512": 512, "gateway2048": 2048}


def _paths(case, seed):
    if case in ITE:
        t, n, tag, _, _ = ITE[case]
        old = ROOT / "results/r2_campaign/raw" / tag
        new = ROOT / "results/laplacian_baselines/raw" / tag
        return {"abf": old / f"abf_N{n}_s{seed}.npz",
                "fr": old / f"fr_N{n}_s{seed}.npz",
                "laplacian": new / f"laplacian_N{n}_s{seed}.npz"}
    if case in LTA:
        t, n, tag, _ = LTA[case]
        old = ROOT / "results/r2_campaign/raw" / tag
        new = ROOT / "results/laplacian_baselines/raw" / tag
        return {
            "abf": old / f"abf_N{n}_s{seed}.npz",
            "fr": old / f"fr_N{n}_s{seed}.npz",
            "laplacian": new / f"laplacian_N{n}_s{seed}.npz",
        }
    n = GATEWAY[case]
    old = ROOT / "results/equal_budget_v2/gateway" / f"N{n}"
    new = ROOT / "results/laplacian_baselines/raw/gateway" / f"N{n}"
    return {arm: (new if arm == "laplacian" else old) / f"s{seed}_{arm}.npz"
            for arm in ("abf", "fr", "laplacian")}


def _scorer(case):
    if case in ITE:
        from eri import metrics as mt
        _, _, _, _, refbase = ITE[case]
        mt.set_reference_base(str(ROOT / refbase))
        refs = mt.reference_for_grid(180, "ite_itq3")
        return "ite", refs
    if case in LTA:
        return "lta150", M.LTAScorer(float(LTA[case][0]))
    p = json.loads((ROOT / "configs/equal_budget_v2/gateway_production.json").read_text())
    return "gateway", M.GatewayScorer(p["engine_cfg"])


def _read(case, seed, arm, path, system, scorer):
    if not path.is_file():
        raise FileNotFoundError(f"{case}, seed {seed}, method {arm}: {path}")
    if system == "ite":
        from eri import metrics as mt
        res = mt.load_run(str(path))
        cfg = res["cfg"]
        if res["method"] != arm:
            raise ValueError(f"{path} contains method {res['method']}, not {arm}")
        if int(cfg["seed"]) != int(seed):
            raise ValueError(f"{path} contains seed {cfg['seed']} != {seed}")
        errs = mt.errors(res, *scorer)
        e = np.asarray(errs["eF"], dtype=float)
        g = np.asarray(errs["eG"], dtype=float)
        steps = np.asarray(res["save_steps"], dtype=np.int64)
        idx, _ = M.uniform_indices(steps, int(cfg["n_steps"]))
        return {"step": steps, "t": np.asarray(res["times"], dtype=float),
                "nfe": int(res["n_force_evals"]), "I_F": float(e[idx].mean()),
                "e_F": float(e[-1]), "I_Fp": float(g[idx].mean()), "e_Fp": float(g[-1]),
                "repl": int(res["sv_i"][-1, 2]),
                "essw_min": float(np.nanmin(res["sv_u"][:, 2])) if arm != "abf" else None}
    res = M.load_run(str(path), system)
    if res["meta"]["method"] != arm:
        raise ValueError(f"{path} contains method {res['meta']['method']}, not {arm}")
    if int(res["meta"]["seed"]) != int(seed):
        raise ValueError(f"{path} contains seed {res['meta']['seed']} != {seed}")
    errs = scorer.errors(res)
    e = np.asarray(errs["e_F"], dtype=float)
    g = np.asarray(errs["e_Fp"], dtype=float)
    n_steps = int(res["meta"]["n_steps"])
    idx, _ = M.uniform_indices(res["save_step"], n_steps)
    return {
        "step": res["save_step"], "t": res["save_t"], "nfe": int(res["n_force_evals"]),
        "I_F": float(e[idx].mean()), "e_F": float(e[-1]),
        "I_Fp": float(g[idx].mean()), "e_Fp": float(g[-1]),
        "repl": int(np.asarray(res.get("cum_deaths", res.get("fr_deaths_cum", [0])))[-1]),
        "essw_min": float(np.nanmin(res["gen_ess_win"])) if arm != "abf" else None,
    }


def analyze(case, nboot):
    if case in ITE:
        seeds = list(range(ITE[case][3], ITE[case][3] + 16))
    elif case in LTA:
        seeds = list(range(LTA[case][3], LTA[case][3] + 16))
    else:
        p = json.loads((ROOT / "configs/equal_budget_v2/gateway_production.json").read_text())
        seeds = list(p["seeds"])
    system, scorer = _scorer(case)
    out = {arm: {} for arm in ("abf", "fr", "laplacian")}
    for seed in seeds:
        paths = _paths(case, seed)
        rows = {arm: _read(case, seed, arm, paths[arm], system, scorer)
                for arm in out}
        for arm in out:
            out[arm][seed] = rows[arm]
        for key in ("step", "t"):
            if not np.array_equal(rows["abf"][key], rows["fr"][key]) or not np.array_equal(rows["fr"][key], rows["laplacian"][key]):
                raise ValueError(f"{case}, seed {seed}: save grids differ at {key}")
        if len({rows[a]["nfe"] for a in rows}) != 1:
            raise ValueError(f"{case}, seed {seed}: unequal force evaluations")
    stats = {}
    for metric in ("I_F", "e_F", "I_Fp", "e_Fp"):
        v = {a: {k: r[metric] for k, r in out[a].items()} for a in out}
        stats[metric] = {
            "fr_vs_abf": M.paired_contrast(v["abf"], v["fr"], n_boot=nboot),
            "laplacian_vs_abf": M.paired_contrast(v["abf"], v["laplacian"], n_boot=nboot),
            "laplacian_vs_fr": M.paired_contrast(v["fr"], v["laplacian"], n_boot=nboot),
        }
    activity = {a: {"repl_median": float(np.median([r["repl"] for r in out[a].values()])),
                    "essw_min_median": (None if a == "abf" else
                                        float(np.median([r["essw_min"] for r in out[a].values()])))}
                for a in out}
    return {"case": case, "n_seeds": len(seeds), "seeds": seeds, "units": "LTA/ITE kJ/mol, gateway reduced energy",
            "metric_convention": "eqb_metrics reference/scorer; I_F = mean e_F at 200 uniform budget fractions",
            "contrast_convention": "(second - first)/first; negative favours second method",
            "activity": activity, "metrics": stats}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", choices=["all", *ITE, *LTA, *GATEWAY], default="all")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--out", default="results/laplacian_baselines/analysis")
    args = ap.parse_args(argv)
    cases = [*ITE, *LTA, *GATEWAY] if args.case == "all" else [args.case]
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    for case in cases:
        res = analyze(case, args.n_boot)
        (out / f"{case}.json").write_text(json.dumps(res, indent=2, allow_nan=False) + "\n")
        print(f"{case}: {res['n_seeds']} seeds")
        for metric in ("I_F", "e_F"):
            print(" ", metric, "laplacian vs FR:",
                  res["metrics"][metric]["laplacian_vs_fr"]["G_median"],
                  res["metrics"][metric]["laplacian_vs_fr"]["G_ci95"])
    print(f"Saved analysis to {out}")


if __name__ == "__main__":
    main()
