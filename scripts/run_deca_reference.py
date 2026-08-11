"""Build the deca-alanine umbrella + MBAR reference.

v2 preregistration §6.3, as revised by **Amendment 1** (bracketed windows, structural
screening) and **Amendment 2** (interleaved builds, checkpoint-and-stop).

All three builds advance together in one batch. §4.5 acceptance is a statement about the spread
*between* independent builds, so it cannot be evaluated until every build has reached the same
amount of sampling -- sequential builds would forbid stopping early by construction.

    python scripts/run_deca_reference.py --out results/deca/reference

Every checkpoint is written, so the §4.5 convergence-versus-compute trace falls out of the run.
No mFR arm exists at this point and none may.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from deca.engine import make_engine                                        # noqa: E402
from deca.labels import N_GATE_A_STATES, conditional_tv                    # noqa: E402
from deca.umbrella import (UmbrellaConfig, mbar_weights, pmf_from_weights,  # noqa: E402
                           run_umbrella, window_centers)

# --- Amendment 2, frozen -------------------------------------------------------------------
STOP_RATIO = 0.5          # reference must be 2x better than the §4.5 minimum to stop early
STOP_FLOOR_NS = 2.0       # never stop on a single lucky checkpoint
EFFECT_SIZE_PCT = 10.0    # the effect the reference must be able to resolve


def analyse_builds(xi, y, keep, cfg, n_builds, per_build):
    """Per-build MBAR + PMF, then the §4.5 between-build statistic.  Returns a dict."""
    cen = window_centers(cfg)
    Fs, extras = [], []
    for b in range(n_builds):
        sl = slice(b * per_build, (b + 1) * per_build)
        xa, w, info, aux = mbar_weights(xi[:, sl], cen, cfg.n_rep, cfg.k_umbrella, cfg.beta,
                                        keep=keep[sl], aux={"y": y[:, sl].astype(np.float64)})
        grid, dz, p, F, counts = pmf_from_weights(xa, w, cfg)
        Fs.append(F)
        extras.append(dict(xi=xa, w=w, y=aux["y"].astype(int), counts=counts,
                           stride=info["stride"], n_mbar=info["n_mbar_samples"]))
    grid, dz, _, _, _ = pmf_from_weights(extras[0]["xi"], extras[0]["w"], cfg)
    mask = (grid >= cfg.R_lo) & (grid <= cfg.R_hi)

    A = np.stack([F - F[mask].mean() for F in Fs])
    n = len(Fs)
    M = np.full((n, n), np.nan)
    for a in range(n):
        for b in range(n):
            if a != b:
                d = A[a] - A[b]
                M[a, b] = float(np.sqrt((d[mask] ** 2).sum() * dz))
    Fm = A.mean(0)
    Fm = Fm - Fm[mask].mean()
    span = float(Fm[mask].max() - Fm[mask].min())
    tolerable = EFFECT_SIZE_PCT / 100.0 * span
    ratio = float(np.nanmax(M) / tolerable) if tolerable > 0 else float("inf")
    return dict(grid=grid, dz=dz, mask=mask, F_builds=A, F_consensus=Fm,
                pairwise_l2=M, span=span, tolerable=tolerable, ratio=ratio,
                extras=extras)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-builds", type=int, default=3)
    ap.add_argument("--out", default="results/deca/reference")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-prod-steps", type=int, default=None)
    ap.add_argument("--checkpoint-every", type=int, default=1_000_000)   # 1 ns
    ap.add_argument("--no-stop", action="store_true",
                    help="collect every checkpoint but never stop early")
    # smoke-only overrides; production uses the frozen UmbrellaConfig defaults
    ap.add_argument("--n-windows", type=int, default=None)
    ap.add_argument("--n-rep", type=int, default=None)
    ap.add_argument("--n-pull-steps", type=int, default=None)
    ap.add_argument("--n-equil-steps", type=int, default=None)
    ap.add_argument("--sample-every", type=int, default=None)
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("no CUDA device visible; set CUDA_VISIBLE_DEVICES to one idle GPU")

    over = {k: v for k, v in dict(
        n_prod_steps=args.n_prod_steps, n_windows=args.n_windows, n_rep=args.n_rep,
        n_pull_steps=args.n_pull_steps, n_equil_steps=args.n_equil_steps,
        sample_every=args.sample_every).items() if v is not None}
    cfg = UmbrellaConfig(**over)
    if over:
        print(f"!! NON-DEFAULT CONFIG (smoke): {over}", flush=True)
    per_build = cfg.n_windows * cfg.n_rep
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "raw"), exist_ok=True)

    engine, system, top = make_engine(10, device=args.device, dtype=torch.float64, compiled=True)
    with open(os.path.join(args.out, "provenance.json"), "w") as fh:
        json.dump(dict(config=cfg.__dict__, config_hash=cfg.config_hash(),
                       parameter_hash=engine.parameter_hash(), n_builds=args.n_builds,
                       n_atoms=int(engine.n_atoms), torch=torch.__version__,
                       batch=args.n_builds * per_build,
                       device=torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu",
                       stop_ratio=STOP_RATIO, stop_floor_ns=STOP_FLOOR_NS,
                       prereg="docs/V2_PREREGISTRATION.md §6.3 + Amendments 1, 2"), fh, indent=2)

    trace = []
    t0 = time.perf_counter()

    ckpt_F = []

    def on_checkpoint(step, snap):
        ns = step * cfg.dt / 1000.0
        r = analyse_builds(snap["xi"], snap["y"], snap["keep"], cfg, args.n_builds, per_build)
        # Retain the consensus F itself, not only the between-build spread.  Agreement BETWEEN
        # builds is statistical reproducibility; it cannot see a systematic error that moves all
        # three together (e.g. an equilibration that is uniformly too short).  Drift of the
        # consensus ACROSS checkpoints can.  Saving F per checkpoint is what makes that
        # measurable after the fact.
        ckpt_F.append(dict(ns=float(ns), F=r["F_consensus"].copy()))
        drift = float("nan")
        if len(ckpt_F) > 1:
            d = ckpt_F[-1]["F"] - ckpt_F[-2]["F"]
            m = r["mask"]
            drift = float(np.sqrt((d[m] ** 2).sum() * r["dz"]))
        rec = dict(step=int(step), ns_per_replica=float(ns), ratio=r["ratio"],
                   pairwise_l2_max=float(np.nanmax(r["pairwise_l2"])),
                   consensus_drift_l2=drift,
                   F_span_kJ=r["span"], resolvable_kJ=r["tolerable"],
                   elapsed_hours=(time.perf_counter() - t0) / 3600.0)
        trace.append(rec)
        np.savez_compressed(os.path.join(args.out, "checkpoint_pmfs.npz"),
                            grid=r["grid"], ns=np.array([c["ns"] for c in ckpt_F]),
                            F=np.stack([c["F"] for c in ckpt_F]))
        print(f"    [checkpoint {ns:.2f} ns/replica]  max pairwise L2 "
              f"{rec['pairwise_l2_max']:.4f} kJ/mol   span {r['span']:.1f}   "
              f"ratio {r['ratio']:.4f}   drift {drift:.4f}  "
              f"(stop at <= {STOP_RATIO} and >= {STOP_FLOOR_NS} ns)", flush=True)
        with open(os.path.join(args.out, "convergence_trace.json"), "w") as fh:
            json.dump(trace, fh, indent=2)
        if args.no_stop:
            return False
        return bool(r["ratio"] <= STOP_RATIO and ns >= STOP_FLOOR_NS)

    print(f"batch = {args.n_builds * per_build} states "
          f"({args.n_builds} builds x {cfg.n_windows} windows x {cfg.n_rep} replicas)", flush=True)
    out = run_umbrella(engine, cfg, build_index=0, n_builds=args.n_builds,
                       device=args.device, dtype=torch.float64,
                       progress_every=200_000, checkpoint_every=args.checkpoint_every,
                       on_checkpoint=on_checkpoint)

    # ---- final analysis on whatever was collected ----
    r = analyse_builds(out["xi"], out["y"], out["keep"], cfg, args.n_builds, per_build)
    ns_final = out["n_sample"] * cfg.sample_every * cfg.dt / 1000.0

    xi_all = np.concatenate([e["xi"] for e in r["extras"]])
    w_all = np.concatenate([e["w"] / e["w"].sum() for e in r["extras"]])
    y_all = np.concatenate([e["y"] for e in r["extras"]])
    edges = np.linspace(cfg.R_lo, cfg.R_hi, 41)
    tv, occ, p_cond = conditional_tv(xi_all, y_all, w_all, edges, min_count=1e-3 * w_all.sum())
    with np.errstate(invalid="ignore"):
        tv_max = float(np.nanmax(tv)) if np.isfinite(tv).any() else float("nan")

    summary = dict(
        n_builds=args.n_builds, ns_per_replica=float(ns_final),
        aggregate_ns=float(ns_final * per_build * args.n_builds),
        stopped_early_at_step=int(out["stopped_early_at"]),
        runtime_hours=float(out["runtime_seconds"]) / 3600.0,
        pairwise_l2_max=float(np.nanmax(r["pairwise_l2"])),
        F_span_kJ=r["span"], resolvable_effect_kJ=r["tolerable"], ratio=r["ratio"],
        reference_accepted=bool(r["ratio"] < 1.0),
        n_excluded_replicas=int(out["n_excluded"]),
        n_fail_cis_pull=int(out["n_fail_cis_pull"]),
        n_fail_chirality_pull=int(out["n_fail_chirality_pull"]),
        pull_error_max_nm=float(out["pull_error_nm"].max()),
        empty_bins_in_domain=int((r["extras"][0]["counts"][r["mask"]] == 0).sum()),
        F_argmin_nm=float(r["grid"][r["mask"]][r["F_consensus"][r["mask"]].argmin()]),
        gate_a_max_pairwise_tv=tv_max,
        gate_a_labels_occupied=int((occ > 0).sum()),
        gate_a_pass=bool(np.isfinite(tv_max) and tv_max >= 0.30),
        config_hash=out["config_hash"], convergence_trace=trace)

    np.savez_compressed(
        os.path.join(args.out, "raw", f"deca_umbrella__{out['config_hash']}.npz"),
        grid=r["grid"], dz=r["dz"], F_consensus=r["F_consensus"], F_builds=r["F_builds"],
        pairwise_l2=r["pairwise_l2"], centers=window_centers(cfg), keep=out["keep"],
        gate_a_tv=tv, gate_a_occupancy=occ, gate_a_p_cond=p_cond, gate_a_edges=edges,
        xi_all=xi_all.astype(np.float32), weights=w_all, y_all=y_all.astype(np.int8),
        n_hbonds=out["n_hbonds"], alpha_frac=out["alpha_frac"], rg=out["rg"],
        ca_rmsd_helix=out["ca_rmsd_helix"])
    with open(os.path.join(args.out, "reference_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n{'='*78}")
    for k in ("ns_per_replica", "aggregate_ns", "runtime_hours", "pairwise_l2_max",
              "F_span_kJ", "resolvable_effect_kJ", "ratio", "reference_accepted",
              "n_excluded_replicas", "empty_bins_in_domain", "F_argmin_nm",
              "gate_a_labels_occupied", "gate_a_max_pairwise_tv", "gate_a_pass"):
        print(f"  {k:26s} {summary[k]}")
    print(f"{'='*78}\n-> {args.out}")


if __name__ == "__main__":
    main()
