#!/usr/bin/env python3
"""Job files of the LRS Laplacian baseline on NSCC (docs/LRS_LAPLACIAN_BASELINE.md, section "NSCC execution").

    python scripts/laplacian_baselines/make_jobs.py rerun    # ABF + FR arms of the 8 positive cells, archived seeds,
                                                             #   archived configs, NEW output paths (the archived raw
                                                             #   .npz files are not on NSCC)
    python scripts/laplacian_baselines/make_jobs.py pilot    # FR + Laplacian scale ladder on FRESH pilot seeds
    python scripts/laplacian_baselines/make_jobs.py apply --decision results/laplacian_baselines/pilot/decision.json
                                                             # freeze the chosen scale into the production job files

Every job dict carries "engine" in {"gateway", "lta", "eri"} and "out" (repo-relative .npz); scripts/laplacian_baselines/
run_all.py runs any list of them.  LTA / ERI entries keep the exact dict layout of configs/adaptive_fr/jobs_R2*.json
(run through the UNCHANGED run_lta_jobs.job / run_jobs.job functions); gateway entries are
{"engine": "gateway", "N", "seed", "method", "scale" (optional override), "out"} and use the frozen
configs/laplacian_baselines/gateway.json engine_cfg, budget and save grid.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from datetime import datetime, timezone

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
CFG = os.path.join(ROOT, "configs", "laplacian_baselines")
RAW = "results/laplacian_baselines/raw"
PILOT = "results/laplacian_baselines/pilot"

# cell -> (engine, archived job file, result tag, N, pilot seed base)
CELLS = {
    "R2C1": ("eri", "configs/adaptive_fr/jobs_R2C1.json", "R2C1_ite300_K1024_R2", 1024, 1_951_000),
    "R2C2": ("eri", "configs/adaptive_fr/jobs_R2C2.json", "R2C2_ite150_K256_R2", 256, 1_952_000),
    "R2C3": ("lta", "configs/adaptive_fr/jobs_R2C3.json", "R2C3_lta150_K1024_R2", 1024, 1_953_000),
    "R2C4": ("lta", "configs/adaptive_fr/jobs_R2C4.json", "R2C4_lta300_K1024_R2", 1024, 1_954_000),
    "R2H80": ("lta", "configs/adaptive_fr/jobs_R2H80.json", "R2H80_lta80_K1024_R2", 1024, 1_955_000),
    "R2H350": ("lta", "configs/adaptive_fr/jobs_R2H350.json", "R2H350_lta350_K1024_R2", 1024, 1_956_000),
}
GATEWAY_N = (512, 2048)
GATEWAY_PILOT_SEEDS = list(range(8300, 8308))      # 8100-8131 archived production, 8200-8231 confirm_best: both avoided
N_PILOT = 8
SCALES = (0.003, 0.01, 0.03, 0.1, 0.3)


def _load(rel):
    with open(os.path.join(ROOT, rel)) as f:
        return json.load(f)


def _dump(name, obj):
    path = os.path.join(CFG, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
        f.write("\n")
    print(f"wrote {os.path.relpath(path, ROOT)}: {len(obj.get('jobs', []))} jobs")


def _set_seed(engine, job, seed):
    if engine == "lta":
        job["seed"] = int(seed)
    else:
        job["cfg"]["seed"] = int(seed)


def rerun():
    for cell, (engine, src, tag, N, _) in CELLS.items():
        spec = _load(src)
        jobs = []
        for j in spec["jobs"]:
            j = copy.deepcopy(j)
            assert j["out"].startswith("results/r2_campaign/raw/"), j["out"]
            j["out"] = RAW + "/" + j["out"][len("results/r2_campaign/raw/"):]
            j["engine"] = engine
            jobs.append(j)
        _dump(f"rerun_{cell}.json", dict(
            stage=f"LB_RERUN_{cell}", source_config=src,
            purpose=("NSCC re-run of the archived ABF and FR arms (same seeds, same configs, same engines); the "
                     "archived raw .npz files are gitignored and absent on NSCC. Pairs with the Laplacian arm at "
                     "identical seeds and force evaluations."), jobs=jobs))
    jobs = []
    for N in GATEWAY_N:
        for seed in _load("configs/laplacian_baselines/gateway.json")["seeds"]:
            for method in ("abf", "fr"):
                jobs.append(dict(engine="gateway", N=N, seed=int(seed), method=method,
                                 out=f"{RAW}/gateway/N{N}/s{seed}_{method}.npz"))
    _dump("rerun_gateway.json", dict(stage="LB_RERUN_GATEWAY", source_config="configs/equal_budget_v2/gateway_production.json",
                                     purpose="NSCC re-run of the archived equal-budget gateway ABF and FR arms at N = 512 and 2048.",
                                     jobs=jobs))


def pilot():
    for cell, (engine, src, tag, N, s0) in CELLS.items():
        spec = _load(src)
        fr_template = [j for j in spec["jobs"] if j["method"] == "fr"][0]
        jobs = []
        for i in range(N_PILOT):
            seed = s0 + i
            j = copy.deepcopy(fr_template)
            _set_seed(engine, j, seed)
            j["engine"] = engine
            j["out"] = f"{PILOT}/{tag}/fr_N{N}_s{seed}.npz"
            jobs.append(j)
            for sc in SCALES:
                j = copy.deepcopy(fr_template)
                _set_seed(engine, j, seed)
                j["engine"] = engine
                j["method"] = "laplacian"
                j["cfg"]["laplacian_scale"] = sc
                j["out"] = f"{PILOT}/{tag}/laplacian_sc{sc:g}_N{N}_s{seed}.npz"
                jobs.append(j)
        _dump(f"pilot_{cell}.json", dict(
            stage=f"LB_PILOT_{cell}", source_config=src, pilot_seeds=[s0 + i for i in range(N_PILOT)], scales=list(SCALES),
            purpose=("laplacian_scale calibration on FRESH seeds (never used for any production cell): FR arm plus the "
                     "Laplacian arm at each ladder scale, archived FR config (rate, R2 controller) otherwise unchanged. "
                     "Decision rule: docs/LRS_LAPLACIAN_BASELINE.md, 'Pilot calibration'."), jobs=jobs))
    jobs = []
    for N in GATEWAY_N:
        for seed in GATEWAY_PILOT_SEEDS:
            jobs.append(dict(engine="gateway", N=N, seed=seed, method="fr", out=f"{PILOT}/gateway/N{N}/s{seed}_fr.npz"))
            for sc in SCALES:
                jobs.append(dict(engine="gateway", N=N, seed=seed, method="laplacian", scale=sc,
                                 out=f"{PILOT}/gateway/N{N}/s{seed}_laplacian_sc{sc:g}.npz"))
    _dump("pilot_gateway.json", dict(stage="LB_PILOT_GATEWAY", pilot_seeds=GATEWAY_PILOT_SEEDS, scales=list(SCALES),
                                     purpose="laplacian_scale calibration of the gateway cells N = 512 and 2048 on fresh seeds.",
                                     jobs=jobs))


def apply(decision_path):
    dec = _load(decision_path)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%MZ")
    for cell, (engine, src, tag, N, _) in CELLS.items():
        sc = float(dec["cells"][cell]["scale"])
        name = f"jobs_{cell}.json"
        spec = _load(f"configs/laplacian_baselines/{name}")
        spec["laplacian_scale"] = sc
        spec["calibration_status"] = (f"FROZEN {stamp} by scripts/laplacian_baselines/pilot_decide.py from the fresh-seed "
                                      f"pilot ({decision_path}); rule in docs/LRS_LAPLACIAN_BASELINE.md")
        for j in spec["jobs"]:
            j["cfg"]["laplacian_scale"] = sc
            j["engine"] = engine
        _dump(name, spec)
    g = _load("configs/laplacian_baselines/gateway.json")
    g["laplacian_scale_by_N"] = {str(N): float(dec["cells"][f"gateway{N}"]["scale"]) for N in GATEWAY_N}
    g["calibration_status"] = (f"FROZEN {stamp} per N by scripts/laplacian_baselines/pilot_decide.py from the fresh-seed pilot "
                               f"({decision_path}); engine_cfg.laplacian_scale is the pre-pilot placeholder, "
                               f"laplacian_scale_by_N is used")
    _dump("gateway.json", g)
    jobs = [dict(engine="gateway", N=N, seed=int(seed), method="laplacian", scale=g["laplacian_scale_by_N"][str(N)],
                 out=f"{RAW}/gateway/N{N}/s{seed}_laplacian.npz") for N in GATEWAY_N for seed in g["seeds"]]
    _dump("prod_gateway.json", dict(stage="LB_PROD_GATEWAY", purpose="Laplacian arm, frozen per-N scale, archived seeds.", jobs=jobs))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("rerun", "pilot", "apply"))
    ap.add_argument("--decision", default="results/laplacian_baselines/pilot/decision.json")
    a = ap.parse_args(argv)
    if a.what == "rerun":
        rerun()
    elif a.what == "pilot":
        pilot()
    else:
        apply(a.decision)


if __name__ == "__main__":
    main()
