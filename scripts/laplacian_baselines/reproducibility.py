#!/usr/bin/env python3
"""NSCC re-runs of the archived ABF / FR arms vs the archive's tracked per-seed analyses.

    python scripts/laplacian_baselines/reproducibility.py [--cells R2C3 R2C4 ...] [--ite300-refbase ...] [--ite150-refbase ...]

LTA / ITE cells: the UNCHANGED scripts/adaptive_fr/analyze_online.py is run on the re-run directory with the archive's
cell arguments (scripts/r2_campaign/analyze_all.sh) -> results/laplacian_baselines/analysis/rerun_<cell>_r2.json, then
compared field by field with results/r2_campaign/analysis/<cell>.json (S_geo median / CI / per seed, FR replacements and
stop times per seed, e_F(T) medians, establishment, translocations).  Gateway: eqb_metrics.run_metrics on every re-run
file vs results/equal_budget_v2/gateway/analysis/run_metrics/N<N>/s<seed>_<method>.json (Ibar_F, final_e_F, Ibar_Fp,
final_e_Fp).  Output: results/laplacian_baselines/analysis/reproducibility.{json,md}.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "equal_budget"))
AN = os.path.join(ROOT, "results", "laplacian_baselines", "analysis")
RAW = "results/laplacian_baselines/raw"
R2 = {  # cell -> (engine, tag, analyze_online args without --dir/--out)
    "R2C1": ("eri", "R2C1_ite300_K1024_R2", "--K 1024 --refbase {ite300} --variant ite_itq3 --ladder kT --expect-seeds 1400000:16"),
    "R2C2": ("eri", "R2C2_ite150_K256_R2", "--K 256 --refbase {ite150} --variant ite_itq3 --ladder abf --expect-seeds 1410000:16"),
    "R2C3": ("lta", "R2C3_lta150_K1024_R2", "--K 1024 --T-K 150 --ladder abf --expect-seeds 1420000:16"),
    "R2C4": ("lta", "R2C4_lta300_K1024_R2", "--K 1024 --T-K 300 --ladder abf --expect-seeds 1430000:16"),
    "R2H80": ("lta", "R2H80_lta80_K1024_R2", "--K 1024 --T-K 80 --ladder abf --expect-seeds 1450000:16"),
    "R2H350": ("lta", "R2H350_lta350_K1024_R2", "--K 1024 --T-K 350 --ladder abf --expect-seeds 1460000:16"),
}


def rel(a, b):
    a, b = float(a), float(b)
    return (a - b) / abs(b) if b else float("nan")


def r2_cell(cell, ite300, ite150):
    engine, tag, args = R2[cell]
    args = args.format(ite300=ite300, ite150=ite150)
    out = os.path.join(AN, f"rerun_{cell}_r2.json")
    cmd = [sys.executable, os.path.join(ROOT, "scripts/adaptive_fr/analyze_online.py"), "--block", f"RERUN_{cell}", "--engine", engine,
           "--dir", os.path.join(RAW, tag), *args.split(), "--out", out]
    env = dict(os.environ, ABFFR_DATA_ROOT=ROOT, PYTHONPATH=os.path.join(ROOT, "src"))
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        return dict(cell=cell, status="analyzer failed", stderr=r.stderr[-1500:])
    new = json.load(open(out))
    old = json.load(open(os.path.join(ROOT, "results/r2_campaign/analysis", f"{cell}.json")))
    rep = dict(cell=cell, status="ok", archived_analysis=f"results/r2_campaign/analysis/{cell}.json", rerun_analysis=os.path.relpath(out, ROOT))
    rep["S_geo_median"] = dict(rerun=new["primary"]["median"], archive=old["primary"]["median"], rel=rel(new["primary"]["median"], old["primary"]["median"]))
    rep["S_geo_ci95"] = dict(rerun=new["primary"]["ci95"], archive=old["primary"]["ci95"])
    ps_new, ps_old = np.asarray(new["primary"]["per_seed"], float), np.asarray(old["primary"]["per_seed"], float)
    rep["S_geo_per_seed_max_abs_diff"] = float(np.nanmax(np.abs(ps_new - ps_old)))
    rep["S_geo_per_seed_identical"] = bool(np.array_equal(ps_new, ps_old))
    for k in ("replacements", "stop_t", "t1"):
        a, b = np.asarray(new["fr_schedule"][k], float), np.asarray(old["fr_schedule"][k], float)
        rep[f"fr_{k}"] = dict(identical=bool(np.array_equal(a, b)), max_abs_diff=float(np.nanmax(np.abs(a - b))),
                              rerun_median=float(np.median(a)), archive_median=float(np.median(b)))
    rep["fr_reasons"] = dict(rerun=new["fr_schedule"]["reason"], archive=old["fr_schedule"]["reason"])
    for k in ("eF_T", "Ibar_F_full", "eG_T"):
        n, o = new["secondary"][k], old["secondary"][k]
        rep[k] = dict(abf_median=dict(rerun=n["abf_median"], archive=o["abf_median"], rel=rel(n["abf_median"], o["abf_median"])),
                      fr_median=dict(rerun=n["fr_median"], archive=o["fr_median"], rel=rel(n["fr_median"], o["fr_median"])),
                      per_seed_rel_max_abs_diff=float(np.nanmax(np.abs(np.asarray(n["per_seed"], float) - np.asarray(o["per_seed"], float)))))
    rep["establishment"] = dict(rerun=new["establishment"], archive=old["establishment"])
    rep["translocations"] = dict(rerun=new["translocations"], archive=old["translocations"])
    rep["paired_bitwise_before_fr_start"] = dict(rerun=sum(new["paired_bitwise_before_fr_start"]), archive=sum(old["paired_bitwise_before_fr_start"]))
    rep["force_evals_per_run"] = dict(rerun=new["force_evals_per_run"], archive=old["force_evals_per_run"])
    rep["wall_seconds_median"] = dict(rerun=new["wall_seconds_median"], archive=old["wall_seconds_median"])
    return rep


def gateway():
    import eqb_metrics as M
    P = json.load(open(os.path.join(ROOT, "configs/equal_budget_v2/gateway_production.json")))
    sc = M.GatewayScorer(P["engine_cfg"])
    rows = []
    for N in (512, 2048):
        for seed in P["seeds"]:
            for method in ("abf", "fr"):
                path = os.path.join(ROOT, RAW, "gateway", f"N{N}", f"s{seed}_{method}.npz")
                arch = os.path.join(ROOT, "results/equal_budget_v2/gateway/analysis/run_metrics", f"N{N}", f"s{seed}_{method}.json")
                if not (os.path.exists(path) and os.path.exists(arch)):
                    continue
                res = M.load_run(path, "gateway")
                _, s = M.run_metrics(res, "gateway", P, sc, path)
                a = json.load(open(arch))["scalars"]
                rows.append(dict(N=N, seed=seed, method=method, **{k: dict(rerun=float(s[k]), archive=float(a[k]), rel=rel(s[k], a[k]))
                                                                for k in ("Ibar_F", "final_e_F", "Ibar_Fp", "final_e_Fp")},
                                 repl_rerun=int(np.asarray(res["fr_deaths_cum"])[-1])))
    summ = {}
    for k in ("Ibar_F", "final_e_F", "Ibar_Fp", "final_e_Fp"):
        r = np.array([x[k]["rel"] for x in rows], float)
        summ[k] = dict(n=len(r), median_rel=float(np.median(r)) if r.size else None, max_abs_rel=float(np.max(np.abs(r))) if r.size else None,
                       n_identical=int(np.sum(r == 0.0)))
    med = {}
    for N in (512, 2048):
        for method in ("abf", "fr"):
            sel = [x for x in rows if x["N"] == N and x["method"] == method]
            if sel:
                med[f"N{N}_{method}"] = {k: dict(rerun=float(np.median([x[k]["rerun"] for x in sel])), archive=float(np.median([x[k]["archive"] for x in sel])))
                                        for k in ("Ibar_F", "final_e_F")}
    return dict(status="ok" if rows else "no re-run files", n_runs=len(rows), per_metric=summ, medians=med, runs=rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", nargs="*", default=[*R2, "gateway"])
    ap.add_argument("--ite300-refbase", default="results/overnight/ite/reference")
    ap.add_argument("--ite150-refbase", default="results/overnight/ite150/reference")
    a = ap.parse_args(argv)
    os.makedirs(AN, exist_ok=True)
    out = {}
    for c in a.cells:
        print(f"== {c} ==", flush=True)
        out[c] = gateway() if c == "gateway" else r2_cell(c, a.ite300_refbase, a.ite150_refbase)
        if c == "gateway":
            print(json.dumps({k: v for k, v in out[c].items() if k != "runs"}, indent=1, default=float))
        else:
            print(json.dumps({k: v for k, v in out[c].items() if k not in ("fr_reasons",)}, indent=1, default=float)[:3000])
    json.dump(out, open(os.path.join(AN, "reproducibility.json"), "w"), indent=1, default=float)
    lines = ["| cell | quantity | re-run | archive | rel. diff |", "|---|---|---:|---:|---:|"]
    for c, r in out.items():
        if r.get("status") != "ok":
            lines.append(f"| {c} | status | {r.get('status')} | | |")
            continue
        if c == "gateway":
            for k, v in r["medians"].items():
                for m in ("Ibar_F", "final_e_F"):
                    lines.append(f"| gateway {k} | median {m} | {v[m]['rerun']:.5g} | {v[m]['archive']:.5g} | {rel(v[m]['rerun'], v[m]['archive']):+.3%} |")
            for m, v in r["per_metric"].items():
                lines.append(f"| gateway all {r['n_runs']} runs | {m}: median rel / max abs rel / identical | {v['median_rel']:+.2%} | {v['max_abs_rel']:.2%} | {v['n_identical']} |")
        else:
            lines.append(f"| {c} | S_geo median | {r['S_geo_median']['rerun']:.3f} | {r['S_geo_median']['archive']:.3f} | {r['S_geo_median']['rel']:+.2%} |")
            lines.append(f"| {c} | S_geo per seed max abs diff (identical) | {r['S_geo_per_seed_max_abs_diff']:.4f} ({r['S_geo_per_seed_identical']}) | | |")
            lines.append(f"| {c} | FR replacements median (identical) | {r['fr_replacements']['rerun_median']:.0f} ({r['fr_replacements']['identical']}) | {r['fr_replacements']['archive_median']:.0f} | |")
            lines.append(f"| {c} | FR stop t median (identical) | {r['fr_stop_t']['rerun_median']:.2f} ({r['fr_stop_t']['identical']}) | {r['fr_stop_t']['archive_median']:.2f} | |")
            for k in ("eF_T", "Ibar_F_full"):
                lines.append(f"| {c} | {k} ABF median | {r[k]['abf_median']['rerun']:.4g} | {r[k]['abf_median']['archive']:.4g} | {r[k]['abf_median']['rel']:+.2%} |")
                lines.append(f"| {c} | {k} FR median | {r[k]['fr_median']['rerun']:.4g} | {r[k]['fr_median']['archive']:.4g} | {r[k]['fr_median']['rel']:+.2%} |")
    open(os.path.join(AN, "reproducibility.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
