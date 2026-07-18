#!/usr/bin/env python3
"""OPES_METAD production on the two toy systems (metastability + entropic bottleneck).

Mirrors the WCA methodology: tune the barrier on DISJOINT seeds {20,21}, then run
production on seeds 0-4 (matching each baseline's 5-seed design) at the selected
barrier. Writes per-seed results and a summary CSV per toy under
results/opes_toys/summaries/. Mean-force reconstruction estimator (fair vs ABF/mFR).

Usage:
  CUDA_VISIBLE_DEVICES=4 python scripts/run_opes_toys.py --toy meta
  CUDA_VISIBLE_DEVICES=5 python scripts/run_opes_toys.py --toy eb
"""
from __future__ import annotations
import argparse, csv, os, sys, statistics as st
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

TUNE_SEEDS = [20, 21]
PROD_SEEDS = [0, 1, 2, 3, 4]
BARRIERS = [3.0, 4.0, 6.0, 8.0]
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "opes_toys", "summaries")


def _median(xs):
    xs = [x for x in xs if x == x]
    return st.median(xs) if xs else float("nan")


def run_meta(seed, barrier):
    import opes_meta as om
    r = om.run_opes_meta(beta=4.0, n_steps=100000, n_particles=1000, seed=seed,
                         barrier=barrier, pace=100, sigma=0.12, estimator="meanforce")
    return r["l2_f"], r["l2_fp"], r["opes_neff_frac"]


def run_eb(seed, barrier):
    import eb_abffr_core as eb, opes_core as oc, opes_eb as oe
    cfg = eb.PhysConfig(beta=8.0, H=2.5, omega_out=1.0, omega_in=25.0, s=0.25,
                        N=256, dt=0.001, n_steps=40000, h=0.07, min_count=1.0)
    oc_cfg = oc.OPESConfig(z_min=eb.XMIN, z_max=eb.XMAX, n_grid=eb.N_GRID, beta=8.0,
                           barrier=barrier, pace=100, sigma=0.05, gamma=float("inf"),
                           gamma_from_barrier=True, bias_force_clip=200.0,
                           warmup_steps=4000, fill_edges=True)
    r = oe.run_opes_eb(cfg, seed=seed, opes_cfg=oc_cfg, estimator="meanforce")
    return r["l2_f"], r["l2_fp"], r["opes_neff_frac"]


RUNNERS = {"meta": run_meta, "eb": run_eb}
# baseline medians (from existing studies) for the summary row
BASELINE = {
    "meta": dict(abf_l2_f=0.0137, abf_l2_fp=0.0718, mfr_l2_f=0.0134, mfr_l2_fp=0.0750,
                 beta=4.0, n_steps=100000, n_seeds=5, label="Metastability (2D, xi=x)"),
    "eb": dict(abf_l2_f=0.2098, abf_l2_fp=1.8270, mfr_l2_f=0.0954, mfr_l2_fp=1.0910,
               beta=8.0, n_steps=40000, n_seeds=5, label="Entropic bottleneck"),
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--toy", required=True, choices=["meta", "eb"])
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)
    run = RUNNERS[args.toy]
    os.makedirs(OUTDIR, exist_ok=True)

    # ---- tune barrier on disjoint seeds ----
    print(f"[tune] {args.toy}: barriers {BARRIERS} on seeds {TUNE_SEEDS}")
    tune_rows = []
    best_barrier, best_med = None, float("inf")
    for b in BARRIERS:
        vals = [run(s, b)[0] for s in TUNE_SEEDS]
        med = _median(vals)
        tune_rows.append(dict(toy=args.toy, barrier=b, l2_f_median=round(med, 5),
                              seeds=";".join(map(str, TUNE_SEEDS))))
        print(f"  barrier={b}: L2F_med={med:.4f}")
        if med == med and med < best_med:
            best_med, best_barrier = med, b
    print(f"[tune] selected barrier={best_barrier} (L2F_med={best_med:.4f})")

    # ---- production on held-out seeds ----
    print(f"[prod] {args.toy}: barrier={best_barrier} on seeds {PROD_SEEDS}")
    prod = []
    for s in PROD_SEEDS:
        l2f, l2fp, neff = run(s, best_barrier)
        prod.append(dict(seed=s, l2_f=l2f, l2_fp=l2fp, neff=neff))
        print(f"  seed{s}: L2F={l2f:.4f} L2Fp={l2fp:.4f} neff={neff:.3f}")
    med_f = _median([p["l2_f"] for p in prod])
    med_fp = _median([p["l2_fp"] for p in prod])
    med_neff = _median([p["neff"] for p in prod])

    # ---- write per-seed + summary ----
    with open(os.path.join(OUTDIR, f"opes_{args.toy}_runs.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["seed", "l2_f", "l2_fp", "neff"])
        w.writeheader(); [w.writerow(p) for p in prod]
    with open(os.path.join(OUTDIR, f"opes_{args.toy}_tuning.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["toy", "barrier", "l2_f_median", "seeds"])
        w.writeheader(); [w.writerow(r) for r in tune_rows]
    bl = BASELINE[args.toy]
    summ = dict(toy=args.toy, label=bl["label"], beta=bl["beta"], n_steps=bl["n_steps"],
                n_seeds=len(PROD_SEEDS), opes_best_barrier=best_barrier,
                opes_l2_f=round(med_f, 5), opes_l2_fp=round(med_fp, 5),
                opes_neff=round(med_neff, 4),
                abf_l2_f=bl["abf_l2_f"], abf_l2_fp=bl["abf_l2_fp"],
                mfr_l2_f=bl["mfr_l2_f"], mfr_l2_fp=bl["mfr_l2_fp"],
                opes_vs_abf_pct=round(100 * (bl["abf_l2_f"] - med_f) / bl["abf_l2_f"], 1),
                opes_vs_mfr_pct=round(100 * (bl["mfr_l2_f"] - med_f) / bl["mfr_l2_f"], 1))
    with open(os.path.join(OUTDIR, f"opes_{args.toy}_summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ.keys()))
        w.writeheader(); w.writerow(summ)
    print(f"[done] {args.toy}: OPES L2F={med_f:.4f} vs ABF {bl['abf_l2_f']} vs mFR {bl['mfr_l2_f']}"
          f"  (vs ABF {summ['opes_vs_abf_pct']:+.1f}%, vs mFR {summ['opes_vs_mfr_pct']:+.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
