"""Aggregate closure run npz -> per-run metric CSV + per-(cell,method,hyperparams)
summary with seed statistics. Pure post-hoc over src/closure_metrics.py; safe to
run any time (skips nothing, reads whatever npz exist).

Usage:
  python scripts/aggregate_closure.py --raw results/opes_closure/wca/raw \
         --out results/opes_closure/wca/metrics [--stage tune_r1]
Outputs:
  <out>/per_run.csv        one row per run (full 61-col schema)
  <out>/per_config.csv     seed-averaged per hyperparameter cell (mean/std/n/CI)
"""
from __future__ import annotations
import argparse, csv, glob, json, math, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import closure_metrics as cm  # noqa: E402

# metric columns summarized across seeds (mean/std/sem/ci95) in per_config
SUMMARY_METRICS = ["l2_f_common", "l2_fp_common", "l2_f_native", "l2_fp_native",
                   "integrated_l2_f", "normalized_anytime_l2_f", "tau_abs", "tau_rel",
                   "marginal_kl", "marginal_tv", "covered_fraction", "n_round_trips",
                   "first_passage_step", "opes_neff_frac_min", "runtime_seconds"]
# keys that define a hyperparameter configuration (a "cell" for successive halving)
CONFIG_KEYS = ["study", "stage", "name", "beta", "h", "n_dim", "a",
               "opes_barrier", "opes_pace", "opes_sigma", "opes_gamma"]


def load_rows(raw_dir, stage=None):
    rows = []
    for f in sorted(glob.glob(os.path.join(raw_dir, "*.npz"))):
        try:
            d = np.load(f, allow_pickle=True)
        except Exception:
            continue
        if stage is not None and str(cm._get(d, "stage", "")) != stage:
            d.close(); continue
        rows.append(cm.compute_metrics(d)); d.close()
    return rows


def write_per_run(rows, path):
    schema = cm.metrics_schema()
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=schema, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in schema})
    return len(rows)


def _config_key(r):
    return tuple(str(r.get(k)) for k in CONFIG_KEYS)


def summarize(rows):
    """Group by hyperparameter config; compute seed mean/std/sem/ci95 per metric."""
    groups = {}
    for r in rows:
        groups.setdefault(_config_key(r), []).append(r)
    out = []
    for key, rs in sorted(groups.items()):
        summary = dict(zip(CONFIG_KEYS, key))
        summary["n_seeds"] = len(rs)
        summary["n_nan"] = sum(1 for r in rs if bool(r.get("had_nan")))
        for m in SUMMARY_METRICS:
            vals = np.array([r.get(m, np.nan) for r in rs], dtype=float)
            vals = vals[np.isfinite(vals)]
            n = vals.size
            mean = float(np.mean(vals)) if n else float("nan")
            std = float(np.std(vals, ddof=1)) if n > 1 else float("nan")
            sem = std / math.sqrt(n) if (n > 1 and np.isfinite(std)) else float("nan")
            ci = 1.96 * sem if np.isfinite(sem) else float("nan")
            summary[f"{m}_mean"] = mean
            summary[f"{m}_std"] = std
            summary[f"{m}_sem"] = sem
            summary[f"{m}_ci95"] = ci
        out.append(summary)
    return out


def write_per_config(summ, path):
    if not summ:
        open(path, "w").close(); return 0
    cols = list(summ[0].keys())
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for s in summ:
            w.writerow(s)
    return len(summ)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", default=None)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    rows = load_rows(args.raw, args.stage)
    n1 = write_per_run(rows, os.path.join(args.out, "per_run.csv"))
    summ = summarize(rows)
    n2 = write_per_config(summ, os.path.join(args.out, "per_config.csv"))
    meta = dict(raw=os.path.abspath(args.raw), stage=args.stage, n_runs=n1,
                n_configs=n2, config_keys=CONFIG_KEYS, summary_metrics=SUMMARY_METRICS)
    with open(os.path.join(args.out, "aggregate_manifest.json"), "w") as fh:
        json.dump(meta, fh, indent=2, default=str)
    print(f"[aggregate] {n1} runs -> {n2} configs; stage={args.stage} -> {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
