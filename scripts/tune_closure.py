"""Successive-halving orchestrator for the OPES closure hyperparameter search.

Given a COMPLETED round's per_config.csv (from aggregate_closure.py), rank configs
WITHIN each physics cell by the primary metric, keep the top fraction (halving),
and emit the next round's YAML at a larger step budget. The per-cell ranking is
also written to JSON so production can freeze the winning (barrier,pace,sigma) per
cell -- successive halving is per-cell because the optimal bias schedule is
physics-dependent (the whole point of the closure).

Primary ranking metric: l2_f_common_mean (common mean-force estimator, the paper's
primary estimator), NaN-runs pushed to the bottom, ties broken by tau_abs_mean then
integrated_l2_f_mean. Design choice documented in the closure report.

Usage:
  # after round r1 npz are aggregated to <metrics>/per_config.csv:
  python scripts/tune_closure.py rank   --config configs/opes_closure/wca_closure.yaml \
         --round tune_r1 --metrics results/opes_closure/wca/metrics
  python scripts/tune_closure.py emit    --config configs/opes_closure/wca_closure.yaml \
         --from-round tune_r1 --to-round tune_r2 --keep-frac 0.5 --n-steps 80000 \
         --metrics results/opes_closure/wca/metrics
"""
from __future__ import annotations
import argparse, csv, json, math, os, sys
import yaml

PRIMARY = "l2_f_common_mean"
TIES = ["tau_abs_mean", "integrated_l2_f_mean"]
CELL_KEYS = ("beta", "h")
HP_KEYS = ("opes_barrier", "opes_pace", "opes_sigma")


def _f(x, big=1e9):
    try:
        v = float(x)
        return v if math.isfinite(v) else big
    except Exception:
        return big


def load_per_config(metrics_dir, stage):
    path = os.path.join(metrics_dir, "per_config.csv")
    rows = [r for r in csv.DictReader(open(path)) if r.get("stage") == stage
            and r.get("name") == "opes"]
    return rows


def rank_within_cells(rows):
    cells = {}
    for r in rows:
        cells.setdefault(tuple(r[k] for k in CELL_KEYS), []).append(r)
    ranked = {}
    for cell, rs in cells.items():
        rs.sort(key=lambda r: (_f(r.get(PRIMARY)), *[_f(r.get(t)) for t in TIES]))
        ranked[cell] = rs
    return ranked


def cmd_rank(args):
    rows = load_per_config(args.metrics, args.round)
    ranked = rank_within_cells(rows)
    out = {}
    print(f"=== round {args.round}: per-cell ranking by {PRIMARY} ===")
    for cell, rs in ranked.items():
        beta, h = cell
        print(f"\ncell beta={beta} h={h}  ({len(rs)} configs)")
        print(f"  {'rank':>4s} {'barr':>5s}{'pace':>6s}{'sig':>6s} {'l2F':>8s}{'±ci':>7s} {'tau':>7s} {'nan':>4s}")
        cell_out = []
        for i, r in enumerate(rs):
            cell_out.append({k: r.get(k) for k in (*HP_KEYS, PRIMARY, "l2_f_common_ci95",
                                                   "tau_abs_mean", "n_seeds", "n_nan")})
            if i < 8:
                print(f"  {i+1:>4d} {r['opes_barrier']:>5s}{r['opes_pace']:>6s}{r['opes_sigma']:>6s} "
                      f"{_f(r.get(PRIMARY)):>8.4f}{_f(r.get('l2_f_common_ci95')):>7.4f} "
                      f"{_f(r.get('tau_abs_mean')):>7.0f} {r.get('n_nan','?'):>4s}")
        out[f"beta{beta}_h{h}"] = cell_out
    path = os.path.join(args.metrics, f"ranking_{args.round}.json")
    json.dump(out, open(path, "w"), indent=2, default=str)
    print(f"\n[rank] wrote {path}")
    return 0


def cmd_emit(args):
    rows = load_per_config(args.metrics, args.from_round)
    ranked = rank_within_cells(rows)
    survivors = set()
    per_cell = {}
    survivors_cfg = {}
    for cell, rs in ranked.items():
        keep = max(1, math.ceil(len(rs) * args.keep_frac))
        winners = rs[:keep]
        per_cell[f"beta{cell[0]}_h{cell[1]}"] = [{k: w[k] for k in HP_KEYS} for w in winners]
        # normalized cell key (float :g) so the runner can match physics beta/h
        ckey = f"b{float(cell[0]):g}_h{float(cell[1]):g}"
        survivors_cfg[ckey] = [[float(w["opes_barrier"]), int(float(w["opes_pace"])),
                                float(w["opes_sigma"])] for w in winners]
        for w in winners:
            survivors.add((float(w["opes_barrier"]), int(float(w["opes_pace"])), float(w["opes_sigma"])))
    # Explicit per-cell survivor tuples drive the next round (true successive halving:
    # only survivors advance to higher budget). Union axes are kept for reference/back-compat.
    barriers = sorted({b for b, _, _ in survivors})
    paces = sorted({p for _, p, _ in survivors})
    sigmas = sorted({s for _, _, s in survivors})
    cfg = yaml.safe_load(open(args.config))
    src = cfg["stages"][args.from_round]
    cfg["stages"][args.to_round] = {
        "n_steps": args.n_steps, "n_replicas": src.get("n_replicas", 1024),
        "seeds": src.get("seeds", [20, 21]) if not args.seeds else [int(x) for x in args.seeds.split(",")],
        "methods": ["opes"], "barriers": barriers, "paces": paces, "sigmas": sigmas,
        "survivors": survivors_cfg,
        "cells": src["cells"],
    }
    yaml.safe_dump(cfg, open(args.config, "w"), sort_keys=False)
    meta = dict(from_round=args.from_round, to_round=args.to_round, keep_frac=args.keep_frac,
                n_steps=args.n_steps, union_grid=dict(barriers=barriers, paces=paces, sigmas=sigmas),
                n_survivor_tuples=len(survivors), per_cell_survivors=per_cell)
    mpath = os.path.join(args.metrics, f"emit_{args.from_round}_to_{args.to_round}.json")
    json.dump(meta, open(mpath, "w"), indent=2, default=str)
    per_cell_counts = {c: len(t) for c, t in survivors_cfg.items()}
    total_cfg = sum(per_cell_counts.values())
    print(f"[emit] {args.to_round}: {total_cfg} survivor configs @ {args.n_steps} steps "
          f"(per-cell {per_cell_counts}; keep_frac={args.keep_frac}). "
          f"Union axes {len(barriers)}x{len(paces)}x{len(sigmas)} kept for reference. Wrote {mpath}")
    print(f"[emit] updated {args.config} with stage '{args.to_round}'")
    return 0


def cmd_freeze(args):
    """Freeze the top-1 config per tuned cell into a production stage. Scope is
    restricted to the cells actually tuned in --from-round (no untuned-cell configs
    are ever fabricated). All methods run the SAME frozen (barrier,pace,sigma) per
    cell; method knobs (gamma etc.) still differ via _opes_knobs."""
    rows = load_per_config(args.metrics, args.from_round)
    ranked = rank_within_cells(rows)
    survivors_cfg, winners_tbl = {}, {}
    tuned_cells = []
    for cell, rs in ranked.items():
        w = rs[0]
        ckey = f"b{float(cell[0]):g}_h{float(cell[1]):g}"
        survivors_cfg[ckey] = [[float(w["opes_barrier"]), int(float(w["opes_pace"])),
                                float(w["opes_sigma"])]]
        winners_tbl[ckey] = {k: w.get(k) for k in (*HP_KEYS, PRIMARY, "l2_f_common_ci95")}
        tuned_cells.append({"beta": float(cell[0]), "h": float(cell[1])})
    cfg = yaml.safe_load(open(args.config))
    src = cfg["stages"][args.from_round]
    methods = args.methods.split(",") if args.methods else ["opes", "opes_fixedg", "opes_flat"]
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else list(range(10))
    cfg["stages"][args.to_stage] = {
        "n_steps": args.n_steps, "n_replicas": src.get("n_replicas", 1024),
        "seeds": seeds, "methods": methods,
        "survivors": survivors_cfg, "cells": tuned_cells,
    }
    yaml.safe_dump(cfg, open(args.config, "w"), sort_keys=False)
    meta = dict(from_round=args.from_round, to_stage=args.to_stage, n_steps=args.n_steps,
                methods=methods, seeds=seeds, tuned_cells=tuned_cells, winners=winners_tbl)
    mpath = os.path.join(args.metrics, f"freeze_{args.from_round}_to_{args.to_stage}.json")
    json.dump(meta, open(mpath, "w"), indent=2, default=str)
    n = len(tuned_cells) * len(methods) * len(seeds)
    print(f"[freeze] {args.to_stage}: {len(tuned_cells)} tuned cells x {len(methods)} methods "
          f"x {len(seeds)} seeds = {n} runs @ {args.n_steps} steps.")
    for ck, w in winners_tbl.items():
        print(f"  {ck}: barrier={w['opes_barrier']} pace={w['opes_pace']} sigma={w['opes_sigma']} "
              f"(L2={float(w[PRIMARY]):.4f})")
    print(f"[freeze] wrote {mpath}; updated {args.config} with stage '{args.to_stage}'")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rank"); r.add_argument("--config", required=True)
    r.add_argument("--round", required=True); r.add_argument("--metrics", required=True)
    r.set_defaults(func=cmd_rank)
    e = sub.add_parser("emit"); e.add_argument("--config", required=True)
    e.add_argument("--from-round", required=True); e.add_argument("--to-round", required=True)
    e.add_argument("--keep-frac", type=float, default=0.5); e.add_argument("--n-steps", type=int, required=True)
    e.add_argument("--metrics", required=True); e.add_argument("--seeds", default=None)
    e.set_defaults(func=cmd_emit)
    f = sub.add_parser("freeze"); f.add_argument("--config", required=True)
    f.add_argument("--from-round", required=True); f.add_argument("--to-stage", required=True)
    f.add_argument("--n-steps", type=int, default=120000); f.add_argument("--metrics", required=True)
    f.add_argument("--methods", default=None); f.add_argument("--seeds", default=None)
    f.set_defaults(func=cmd_freeze)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
